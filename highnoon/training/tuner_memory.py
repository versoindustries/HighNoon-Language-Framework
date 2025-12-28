# highnoon/training/tuner_memory.py
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

"""Cross-Trial Memory System for the Unified Smart Tuner.

This module provides persistent storage and retrieval of tuning trajectories
from previous HPO trials. It enables the Unified Smart Tuner to learn from
past experiments and make better initialization decisions for new trials.

Key Features:
    - Persists tuning trajectories across trials
    - Finds similar architectures and suggests initial configurations
    - Predicts optimal LR schedules based on historical data
    - Supports memory cleanup and compaction

Example:
    >>> memory = TunerMemory(Path("artifacts/tuner_memory"))
    >>> memory.record_trial(TrialRecord(...))
    >>> suggested = memory.suggest_initial_config({"embedding_dim": 512})

Reference:
    Smart_Tuner_Upgrade.md - Cross-Trial Memory System specification
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# TRIAL RECORD
# =============================================================================


@dataclass
class TrialRecord:
    """Record of a single HPO trial for learning.

    Attributes:
        trial_id: Unique identifier for the trial.
        architecture_config: Model architecture configuration.
        hyperparameters: Training hyperparameters used.
        final_loss: Final validation loss achieved.
        best_epoch: Epoch with best validation loss.
        tuner_trajectory: List of tuning states over time.
        timestamp: Unix timestamp when trial was recorded.
        total_steps: Total training steps completed.
        converged: Whether training converged.
        tags: Optional metadata tags.
    """

    trial_id: str
    architecture_config: dict
    hyperparameters: dict
    final_loss: float
    best_epoch: int
    tuner_trajectory: list[dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    total_steps: int = 0
    converged: bool = False
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrialRecord":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# TUNER MEMORY
# =============================================================================


class TunerMemory:
    """Cross-trial memory for learning from previous HPO sweeps.

    This class persists tuning trajectories across trials and uses them
    to initialize future trials more intelligently. It stores trial records
    in a JSON-lines file and provides methods for querying similar trials.

    Attributes:
        memory_path: Directory path for memory storage.

    Example:
        >>> memory = TunerMemory(Path("artifacts/tuner_memory"))
        >>> memory.record_trial(
        ...     trial_id="trial_001",
        ...     architecture_config={"embedding_dim": 512},
        ...     hyperparameters={"lr": 3e-4},
        ...     final_loss=0.5,
        ...     best_epoch=10,
        ...     tuner_trajectory=[{"step": 0, "learning_rate": 3e-4}],
        ... )
        >>> suggested = memory.suggest_initial_config({"embedding_dim": 512})
    """

    def __init__(self, memory_path: Path):
        """Initialize the tuner memory.

        Args:
            memory_path: Directory path for storing memory files.
        """
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)

        self._trials_file = self.memory_path / "trials.jsonl"
        self._index_file = self.memory_path / "index.json"
        self._trials: list[TrialRecord] = []
        self._index: dict[str, Any] = {}

        self._load_history()

    def _load_history(self) -> None:
        """Load trial history from disk."""
        self._trials.clear()
        self._index.clear()

        # Load trials from JSONL file
        if self._trials_file.exists():
            try:
                with open(self._trials_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self._trials.append(TrialRecord.from_dict(data))
                logger.info(
                    "[TunerMemory] Loaded %d trials from %s",
                    len(self._trials),
                    self._trials_file,
                )
            except Exception as e:
                logger.warning("[TunerMemory] Failed to load trials: %s", e)

        # Load index
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning("[TunerMemory] Failed to load index: %s", e)
                self._rebuild_index()
        else:
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the trial index."""
        self._index = {
            "trial_count": len(self._trials),
            "best_trial_id": None,
            "best_loss": float("inf"),
            "architecture_hashes": {},
        }

        for trial in self._trials:
            # Track best trial
            if trial.final_loss < self._index["best_loss"]:
                self._index["best_loss"] = trial.final_loss
                self._index["best_trial_id"] = trial.trial_id

            # Build architecture hash index
            arch_hash = self._hash_architecture(trial.architecture_config)
            if arch_hash not in self._index["architecture_hashes"]:
                self._index["architecture_hashes"][arch_hash] = []
            self._index["architecture_hashes"][arch_hash].append(trial.trial_id)

        self._persist_index()

    def _hash_architecture(self, config: dict) -> str:
        """Create a hash key for architecture similarity matching.

        Uses key architectural dimensions to create a coarse hash.
        """
        key_dims = [
            config.get("embedding_dim", 512),
            config.get("num_reasoning_blocks", 8),
            config.get("num_moe_experts", 8),
            config.get("vocab_size", 32000),
        ]
        return "_".join(str(d) for d in key_dims)

    def _persist(self) -> None:
        """Persist trials to disk."""
        try:
            with open(self._trials_file, "w") as f:
                for trial in self._trials:
                    f.write(json.dumps(trial.to_dict()) + "\n")
        except Exception as e:
            logger.error("[TunerMemory] Failed to persist trials: %s", e)

    def _persist_index(self) -> None:
        """Persist index to disk."""
        try:
            with open(self._index_file, "w") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error("[TunerMemory] Failed to persist index: %s", e)

    def record_trial(
        self,
        trial_id: str,
        architecture_config: dict,
        hyperparameters: dict,
        final_loss: float,
        best_epoch: int,
        tuner_trajectory: list[dict] | None = None,
        total_steps: int = 0,
        converged: bool = False,
        tags: list[str] | None = None,
    ) -> None:
        """Record a completed trial.

        Args:
            trial_id: Unique identifier for the trial.
            architecture_config: Model architecture configuration.
            hyperparameters: Training hyperparameters.
            final_loss: Final validation loss.
            best_epoch: Epoch with best validation loss.
            tuner_trajectory: Optional list of tuning states over time.
            total_steps: Total training steps completed.
            converged: Whether training converged.
            tags: Optional metadata tags.
        """
        record = TrialRecord(
            trial_id=trial_id,
            architecture_config=architecture_config,
            hyperparameters=hyperparameters,
            final_loss=final_loss,
            best_epoch=best_epoch,
            tuner_trajectory=tuner_trajectory or [],
            total_steps=total_steps,
            converged=converged,
            tags=tags or [],
        )

        self._trials.append(record)

        # Update index
        if final_loss < self._index.get("best_loss", float("inf")):
            self._index["best_loss"] = final_loss
            self._index["best_trial_id"] = trial_id

        arch_hash = self._hash_architecture(architecture_config)
        if arch_hash not in self._index.get("architecture_hashes", {}):
            self._index.setdefault("architecture_hashes", {})[arch_hash] = []
        self._index["architecture_hashes"][arch_hash].append(trial_id)
        self._index["trial_count"] = len(self._trials)

        self._persist()
        self._persist_index()

        logger.info(
            "[TunerMemory] Recorded trial %s: loss=%.4f, best_epoch=%d",
            trial_id,
            final_loss,
            best_epoch,
        )

    def _find_similar_trials(
        self, architecture_config: dict, max_results: int = 10
    ) -> list[TrialRecord]:
        """Find trials with similar architecture.

        Args:
            architecture_config: Target architecture configuration.
            max_results: Maximum number of results to return.

        Returns:
            List of similar TrialRecords sorted by final_loss.
        """
        arch_hash = self._hash_architecture(architecture_config)

        # First try exact hash match
        exact_ids = self._index.get("architecture_hashes", {}).get(arch_hash, [])
        exact_trials = [t for t in self._trials if t.trial_id in exact_ids]

        if len(exact_trials) >= max_results:
            return sorted(exact_trials, key=lambda t: t.final_loss)[:max_results]

        # If not enough exact matches, find similar by distance
        target_dims = np.array(
            [
                architecture_config.get("embedding_dim", 512),
                architecture_config.get("num_reasoning_blocks", 8) * 50,
                architecture_config.get("num_moe_experts", 8) * 50,
                architecture_config.get("vocab_size", 32000) / 1000,
            ]
        )

        scored_trials = []
        for trial in self._trials:
            trial_dims = np.array(
                [
                    trial.architecture_config.get("embedding_dim", 512),
                    trial.architecture_config.get("num_reasoning_blocks", 8) * 50,
                    trial.architecture_config.get("num_moe_experts", 8) * 50,
                    trial.architecture_config.get("vocab_size", 32000) / 1000,
                ]
            )
            distance = np.linalg.norm(target_dims - trial_dims)
            scored_trials.append((distance, trial))

        scored_trials.sort(key=lambda x: x[0])
        return [t for _, t in scored_trials[:max_results]]

    def suggest_initial_config(self, architecture_config: dict) -> dict[str, Any]:
        """Suggest initial tuner configuration based on similar past trials.

        Finds trials with similar architecture and uses their successful
        tuning trajectories to initialize this trial.

        Args:
            architecture_config: Current model architecture configuration.

        Returns:
            Dictionary with suggested initial parameters.
        """
        similar_trials = self._find_similar_trials(architecture_config)

        if not similar_trials:
            logger.info(
                "[TunerMemory] No similar trials found, using defaults"
            )
            return {}

        # Find the best trial among similar ones
        best_trial = min(similar_trials, key=lambda t: t.final_loss)

        # Extract initial parameters from best trajectory
        trajectory = best_trial.tuner_trajectory
        if not trajectory:
            logger.info(
                "[TunerMemory] Best similar trial has no trajectory, using hyperparameters"
            )
            return {
                "initial_lr": best_trial.hyperparameters.get("learning_rate"),
                "exploration_factor": 0.8,  # Start with less exploration
            }

        initial_state = trajectory[0]
        suggested = {
            "initial_lr": initial_state.get("learning_rate"),
            "galore_rank": initial_state.get("galore_rank"),
            "exploration_factor": 0.8,  # Start with less exploration
        }

        # If trial converged, use its final LR as guidance
        if best_trial.converged and trajectory:
            final_state = trajectory[-1]
            suggested["target_lr"] = final_state.get("learning_rate")

        logger.info(
            "[TunerMemory] Suggested config from trial %s (loss=%.4f): %s",
            best_trial.trial_id,
            best_trial.final_loss,
            suggested,
        )

        return {k: v for k, v in suggested.items() if v is not None}

    def predict_lr_schedule(
        self, architecture_config: dict, total_steps: int
    ) -> list[float] | None:
        """Predict optimal LR schedule based on similar past trials.

        Args:
            architecture_config: Target architecture configuration.
            total_steps: Total number of training steps.

        Returns:
            List of predicted learning rates, or None if not enough data.
        """
        similar_trials = self._find_similar_trials(architecture_config)

        # Need at least 3 trials for meaningful prediction
        if len(similar_trials) < 3:
            return None

        # Collect LR trajectories from converged trials
        converged_trials = [t for t in similar_trials if t.converged]
        if len(converged_trials) < 2:
            return None

        # Aggregate LR trajectories
        lr_samples: dict[int, list[float]] = {}
        for trial in converged_trials:
            for entry in trial.tuner_trajectory:
                step = entry.get("step", 0)
                lr = entry.get("learning_rate")
                if lr is not None:
                    # Normalize step to [0, 1] range
                    normalized_step = int(step / max(1, trial.total_steps) * 100)
                    if normalized_step not in lr_samples:
                        lr_samples[normalized_step] = []
                    lr_samples[normalized_step].append(lr)

        if not lr_samples:
            return None

        # Build predicted schedule
        predicted = []
        for i in range(100):
            if i in lr_samples:
                # Use weighted median (more weight to better trials)
                predicted.append(np.median(lr_samples[i]))
            elif predicted:
                # Interpolate
                predicted.append(predicted[-1])
            else:
                # Use first available
                first_key = min(lr_samples.keys())
                predicted.append(np.median(lr_samples[first_key]))

        # Scale to actual step count
        step_size = max(1, total_steps // 100)
        full_schedule = []
        for i in range(total_steps):
            idx = min(i // step_size, 99)
            full_schedule.append(predicted[idx])

        return full_schedule

    def get_statistics(self) -> dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory statistics.
        """
        if not self._trials:
            return {
                "trial_count": 0,
                "best_loss": None,
                "best_trial_id": None,
                "unique_architectures": 0,
            }

        losses = [t.final_loss for t in self._trials]
        return {
            "trial_count": len(self._trials),
            "best_loss": self._index.get("best_loss"),
            "best_trial_id": self._index.get("best_trial_id"),
            "unique_architectures": len(
                self._index.get("architecture_hashes", {})
            ),
            "mean_loss": np.mean(losses),
            "std_loss": np.std(losses),
            "converged_count": sum(1 for t in self._trials if t.converged),
        }

    def clear(self) -> None:
        """Clear all memory (for fresh starts)."""
        self._trials.clear()
        self._index.clear()

        if self._trials_file.exists():
            self._trials_file.unlink()
        if self._index_file.exists():
            self._index_file.unlink()

        logger.info("[TunerMemory] Memory cleared")

    def compact(self, keep_top_n: int = 100) -> int:
        """Compact memory by keeping only the top N trials.

        Args:
            keep_top_n: Number of best trials to keep.

        Returns:
            Number of trials removed.
        """
        if len(self._trials) <= keep_top_n:
            return 0

        original_count = len(self._trials)
        self._trials.sort(key=lambda t: t.final_loss)
        self._trials = self._trials[:keep_top_n]

        self._rebuild_index()
        self._persist()

        removed = original_count - len(self._trials)
        logger.info(
            "[TunerMemory] Compacted memory: removed %d trials, kept %d",
            removed,
            len(self._trials),
        )
        return removed


__all__ = [
    "TunerMemory",
    "TrialRecord",
]
