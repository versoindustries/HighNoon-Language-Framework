# highnoon/services/hpo_meta_cache.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Meta-Learning Cache for Cross-Sweep Knowledge Transfer.

This module provides persistent storage and retrieval of hyperparameter
optimization results across sweeps. It enables warm-starting new sweeps
with knowledge from prior successful configurations.

Key Features:
1. SQLite-backed persistent storage
2. Similarity-based prior sweep retrieval
3. Warm-start population initialization
4. Best config recommendation based on history

References:
- Feurer et al., "Initializing Bayesian Hyperparameter Optimization via
  Meta-Learning" (AAAI 2015)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SweepRecord:
    """Record of a completed HPO sweep.

    Attributes:
        sweep_id: Unique sweep identifier
        task_hash: Hash of task characteristics for similarity
        best_config: Best configuration found
        best_loss: Best loss achieved
        all_configs: All configurations tried (optional)
        all_losses: Corresponding losses (optional)
        metadata: Additional sweep metadata
    """

    sweep_id: str
    task_hash: str
    best_config: dict[str, Any]
    best_loss: float
    all_configs: list[dict[str, Any]] | None = None
    all_losses: list[float] | None = None
    metadata: dict[str, Any] | None = None


class MetaLearningCache:
    """Persistent cache for cross-sweep knowledge transfer.

    Stores sweep results in SQLite and enables warm-starting new sweeps
    based on similar prior tasks.

    Attributes:
        cache_path: Path to SQLite database
        max_entries: Maximum number of sweeps to store (LRU eviction)
    """

    def __init__(
        self,
        cache_path: Path | str | None = None,
        max_entries: int = 100,
    ):
        """Initialize meta-learning cache.

        Args:
            cache_path: Path to SQLite database (default: artifacts/hpo_meta_cache.db)
            max_entries: Maximum sweeps to store before LRU eviction
        """
        if cache_path is None:
            cache_path = Path("artifacts/hpo_meta_cache.db")
        self.cache_path = Path(cache_path)
        self.max_entries = max_entries

        # Ensure directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        logger.info(f"[MetaCache] Initialized at {self.cache_path}")

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()

            # Main sweep results table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sweep_results (
                    sweep_id TEXT PRIMARY KEY,
                    task_hash TEXT NOT NULL,
                    best_config TEXT NOT NULL,
                    best_loss REAL NOT NULL,
                    all_configs TEXT,
                    all_losses TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Index for task similarity lookup
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_hash
                ON sweep_results(task_hash)
            """
            )

            conn.commit()

    @staticmethod
    def compute_task_hash(
        vocab_size: int | None = None,
        context_length: int | None = None,
        param_budget: int | None = None,
        optimizer_choices: list[str] | None = None,
        search_space_keys: list[str] | None = None,
    ) -> str:
        """Compute hash representing task characteristics.

        Similar tasks will have matching or similar hashes, enabling
        knowledge transfer between sweeps.

        Args:
            vocab_size: Vocabulary size
            context_length: Context/sequence length
            param_budget: Parameter budget
            optimizer_choices: Available optimizers
            search_space_keys: Keys in search space

        Returns:
            Hash string for similarity matching
        """
        # Bucket continuous values for fuzzy matching
        vocab_bucket = vocab_size // 10000 if vocab_size else 0
        context_bucket = context_length // 1000 if context_length else 0
        budget_bucket = param_budget // 100_000_000 if param_budget else 0  # 100M buckets

        parts = [
            f"v{vocab_bucket}",
            f"c{context_bucket}",
            f"b{budget_bucket}",
        ]

        if optimizer_choices:
            parts.append(f"o{','.join(sorted(optimizer_choices))}")

        if search_space_keys:
            # Use subset of keys for hash
            key_hash = hashlib.md5(",".join(sorted(search_space_keys)).encode()).hexdigest()[:8]
            parts.append(f"k{key_hash}")

        full_str = "|".join(parts)
        return hashlib.md5(full_str.encode()).hexdigest()[:16]

    def save_sweep_results(
        self,
        sweep_id: str,
        task_hash: str,
        best_config: dict[str, Any],
        best_loss: float,
        all_configs: list[dict[str, Any]] | None = None,
        all_losses: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save sweep results to cache.

        Args:
            sweep_id: Unique sweep identifier
            task_hash: Task characteristics hash
            best_config: Best configuration found
            best_loss: Best loss achieved
            all_configs: All configurations tried
            all_losses: Corresponding losses
            metadata: Additional metadata
        """
        # Clean config of internal keys
        clean_best = {k: v for k, v in best_config.items() if not k.startswith("_")}
        clean_all = None
        if all_configs:
            clean_all = [{k: v for k, v in c.items() if not k.startswith("_")} for c in all_configs]

        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()

            # Insert or update
            cursor.execute(
                """
                INSERT OR REPLACE INTO sweep_results
                (sweep_id, task_hash, best_config, best_loss, all_configs, all_losses, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sweep_id,
                    task_hash,
                    json.dumps(clean_best),
                    best_loss,
                    json.dumps(clean_all) if clean_all else None,
                    json.dumps(all_losses) if all_losses else None,
                    json.dumps(metadata) if metadata else None,
                ),
            )

            # LRU eviction if over limit
            cursor.execute("SELECT COUNT(*) FROM sweep_results")
            count = cursor.fetchone()[0]

            if count > self.max_entries:
                cursor.execute(
                    """
                    DELETE FROM sweep_results
                    WHERE sweep_id IN (
                        SELECT sweep_id FROM sweep_results
                        ORDER BY accessed_at ASC
                        LIMIT ?
                    )
                    """,
                    (count - self.max_entries,),
                )

            conn.commit()

        logger.info(
            f"[MetaCache] Saved sweep {sweep_id} (task_hash={task_hash}, loss={best_loss:.6f})"
        )

    def load_similar_sweeps(
        self,
        task_hash: str,
        limit: int = 5,
    ) -> list[SweepRecord]:
        """Load sweep results with matching task hash.

        Args:
            task_hash: Task hash to match
            limit: Maximum number of results

        Returns:
            List of matching SweepRecords sorted by best_loss
        """
        with sqlite3.connect(self.cache_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Exact hash match first
            cursor.execute(
                """
                SELECT * FROM sweep_results
                WHERE task_hash = ?
                ORDER BY best_loss ASC
                LIMIT ?
                """,
                (task_hash, limit),
            )

            rows = cursor.fetchall()

            # Update access time
            if rows:
                sweep_ids = [row["sweep_id"] for row in rows]
                cursor.execute(
                    f"""
                    UPDATE sweep_results
                    SET accessed_at = CURRENT_TIMESTAMP
                    WHERE sweep_id IN ({','.join('?' * len(sweep_ids))})
                    """,
                    sweep_ids,
                )
                conn.commit()

        results = []
        for row in rows:
            record = SweepRecord(
                sweep_id=row["sweep_id"],
                task_hash=row["task_hash"],
                best_config=json.loads(row["best_config"]),
                best_loss=row["best_loss"],
                all_configs=json.loads(row["all_configs"]) if row["all_configs"] else None,
                all_losses=json.loads(row["all_losses"]) if row["all_losses"] else None,
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            results.append(record)

        logger.info(f"[MetaCache] Found {len(results)} similar sweeps for task_hash={task_hash}")
        return results

    def get_warm_start_configs(
        self,
        task_hash: str,
        n_configs: int = 4,
    ) -> list[dict[str, Any]]:
        """Get configs for warm-starting a new sweep.

        Returns best configs from similar prior sweeps.

        Args:
            task_hash: Task hash to match
            n_configs: Number of configs to return

        Returns:
            List of warm-start configurations
        """
        similar = self.load_similar_sweeps(task_hash, limit=n_configs * 2)

        configs = []
        for record in similar:
            # Add best config
            if len(configs) < n_configs:
                configs.append(record.best_config)

            # Add top configs from all_configs if available
            if record.all_configs and record.all_losses:
                sorted_pairs = sorted(
                    zip(record.all_configs, record.all_losses),
                    key=lambda x: x[1],
                )
                for config, _ in sorted_pairs[:2]:
                    if len(configs) < n_configs:
                        configs.append(config)

        logger.info(
            f"[MetaCache] Providing {len(configs)} warm-start configs for task_hash={task_hash}"
        )
        return configs

    def warm_start_surrogate(
        self,
        surrogate: Any,  # TPESurrogate
        task_hash: str,
        n_prior_observations: int = 20,
    ) -> int:
        """Warm-start a TPE surrogate with prior observations.

        Args:
            surrogate: TPESurrogate instance to warm-start
            task_hash: Task hash to match
            n_prior_observations: Maximum observations to transfer

        Returns:
            Number of observations transferred
        """
        similar = self.load_similar_sweeps(task_hash, limit=10)

        transferred = 0
        for record in similar:
            if record.all_configs and record.all_losses:
                for config, loss in zip(record.all_configs, record.all_losses):
                    if transferred >= n_prior_observations:
                        break
                    surrogate.add_observation(config, loss)
                    transferred += 1

        if transferred > 0:
            logger.info(f"[MetaCache] Warm-started surrogate with {transferred} prior observations")

        return transferred

    def get_best_config_ever(self) -> tuple[dict[str, Any], float] | None:
        """Get the best config across all sweeps.

        Returns:
            Tuple of (config, loss) or None if cache empty
        """
        with sqlite3.connect(self.cache_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT best_config, best_loss FROM sweep_results
                ORDER BY best_loss ASC
                LIMIT 1
                """
            )

            row = cursor.fetchone()

        if row:
            return json.loads(row["best_config"]), row["best_loss"]
        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM sweep_results")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT task_hash) FROM sweep_results")
            unique_tasks = cursor.fetchone()[0]

            cursor.execute("SELECT MIN(best_loss) FROM sweep_results")
            best_ever = cursor.fetchone()[0]

        return {
            "cache_path": str(self.cache_path),
            "total_sweeps": total,
            "unique_tasks": unique_tasks,
            "best_loss_ever": best_ever,
            "max_entries": self.max_entries,
        }


__all__ = [
    "SweepRecord",
    "MetaLearningCache",
]
