"""HPO Metrics Collection and Reporting System.

This module provides a file-based IPC system for collecting metrics from
distributed HPO trials and aggregating them for the orchestrator and web UI.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .hpo_utils import convert_numpy_types

logger = logging.getLogger(__name__)

# Default artifacts directory for HPO trials
DEFAULT_HPO_ROOT = Path("artifacts/hpo_trials")

# Default lambda for efficiency penalty (higher = favor smaller models more)
DEFAULT_LAMBDA_EFFICIENCY = 0.1


def compute_efficiency_score(
    loss: float,
    param_count: int,
    param_budget: int,
    lambda_efficiency: float = DEFAULT_LAMBDA_EFFICIENCY,
) -> float:
    """Compute efficiency-penalized loss score.

    The efficiency score rewards smaller models that achieve similar accuracy.
    Formula: loss × (1 + λ × (params / budget))

    Args:
        loss: Best loss achieved by the trial
        param_count: Estimated parameter count of the model
        param_budget: User's parameter budget constraint
        lambda_efficiency: Penalty weight (higher = favor smaller models)

    Returns:
        Efficiency-penalized score (lower is better)

    Example:
        >>> # A 500M model using 50% of 1B budget gets 5% bonus
        >>> compute_efficiency_score(0.1, 500_000_000, 1_000_000_000, 0.1)
        0.105
        >>> # A 1B model using 100% of budget gets no bonus
        >>> compute_efficiency_score(0.1, 1_000_000_000, 1_000_000_000, 0.1)
        0.11
    """
    if param_budget <= 0:
        return loss  # No penalty if no budget set

    norm_params = param_count / param_budget
    return loss * (1 + lambda_efficiency * norm_params)


# Default weights for composite score components
DEFAULT_ALPHA_LOSS = 0.5  # Training loss weight
DEFAULT_BETA_PERPLEXITY = 0.3  # Perplexity weight
DEFAULT_GAMMA_CALIBRATION = 0.2  # Calibration (ECE) weight


def compute_composite_score(
    loss: float,
    perplexity: float | None = None,
    ece: float | None = None,
    param_count: int | None = None,
    param_budget: int | None = None,
    alpha_loss: float = DEFAULT_ALPHA_LOSS,
    beta_perplexity: float = DEFAULT_BETA_PERPLEXITY,
    gamma_calibration: float = DEFAULT_GAMMA_CALIBRATION,
    lambda_efficiency: float = DEFAULT_LAMBDA_EFFICIENCY,
) -> float:
    """Compute multi-objective composite score for trial ranking.

    The composite score combines:
    1. Training loss (lower is better)
    2. Perplexity on validation set (lower is better)
    3. Calibration quality via ECE (lower is better)
    4. Efficiency penalty for model size

    Formula: α×loss + β×norm_ppl + γ×ece + efficiency_multiplier

    Args:
        loss: Best training loss achieved
        perplexity: Validation set perplexity (None = skip)
        ece: Expected Calibration Error (None = skip)
        param_count: Model parameter count
        param_budget: User's parameter budget
        alpha_loss: Weight for loss component
        beta_perplexity: Weight for perplexity component
        gamma_calibration: Weight for calibration component
        lambda_efficiency: Weight for efficiency penalty

    Returns:
        Composite score (lower is better)

    Example:
        >>> compute_composite_score(0.5, perplexity=50.0, ece=0.05)
        0.4597...
    """
    score = alpha_loss * loss

    if perplexity is not None:
        # Normalize perplexity (log scale, capped at 1000)
        norm_ppl = min(math.log(perplexity + 1) / math.log(1001), 1.0)
        score += beta_perplexity * norm_ppl

    if ece is not None:
        # ECE is already 0-1 scale (0 = perfect calibration)
        score += gamma_calibration * ece

    # Efficiency penalty as multiplier
    if param_count and param_budget and param_budget > 0:
        norm_params = param_count / param_budget
        score *= 1 + lambda_efficiency * norm_params

    return score


@dataclass
class TrialMetrics:
    """Metrics snapshot for a single trial at a specific step."""

    trial_id: str
    step: int
    timestamp: float
    loss: float | None = None
    gradient_norm: float | None = None
    learning_rate: float | None = None
    memory_mb: float | None = None
    peak_memory_mb: float | None = None  # Peak RSS memory during trial
    wall_time_seconds: float | None = None

    # Quality metrics (evaluated at trial end)
    perplexity: float | None = None
    mean_confidence: float | None = None
    expected_calibration_error: float | None = None

    # Additional custom metrics
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = asdict(self)
        result["timestamp_iso"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return result


@dataclass
class TrialStatus:
    """Current status of an HPO trial."""

    trial_id: str
    status: str  # "pending", "running", "completed", "failed", "stopped"
    start_time: float | None = None
    end_time: float | None = None
    best_loss: float | None = None
    best_step: int | None = None
    total_steps: int = 0
    error_message: str | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    # Efficiency tracking (Phase: HPO Efficiency-Aware Optimization)
    param_count: int | None = None  # Estimated parameter count from config
    efficiency_score: float | None = None  # loss × (1 + λ × norm_params)

    # Multi-objective quality metrics
    perplexity: float | None = None
    mean_confidence: float | None = None
    expected_calibration_error: float | None = None
    composite_score: float | None = None  # Combined multi-objective score

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = asdict(self)
        if self.start_time:
            result["start_time_iso"] = datetime.fromtimestamp(self.start_time).isoformat()
        if self.end_time:
            result["end_time_iso"] = datetime.fromtimestamp(self.end_time).isoformat()
        return result


class HPOMetricsCollector:
    """Collects and persists metrics from HPO trials using file-based IPC."""

    def __init__(self, hpo_root: Path | None = None):
        """Initialize the metrics collector.

        Args:
            hpo_root: Root directory for HPO trial artifacts. Defaults to artifacts/hpo_trials
        """
        self.hpo_root = Path(hpo_root or DEFAULT_HPO_ROOT).resolve()
        self.hpo_root.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

        logger.info(f"[HPO Metrics] Initialized collector at {self.hpo_root}")

    def get_trial_dir(self, trial_id: str) -> Path:
        """Get the directory for a specific trial."""
        trial_dir = self.hpo_root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        return trial_dir

    def report_metrics(self, metrics: TrialMetrics) -> None:
        """Report metrics for a trial (append to JSONL file).

        Args:
            metrics: TrialMetrics object to persist
        """
        trial_dir = self.get_trial_dir(metrics.trial_id)
        metrics_file = trial_dir / "metrics.jsonl"

        # Atomic append using file locking
        with self._write_lock:
            try:
                with metrics_file.open("a", encoding="utf-8") as f:
                    json.dump(metrics.to_dict(), f, ensure_ascii=False)
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
            except Exception as exc:
                logger.error(f"[HPO Metrics] Failed to write metrics for {metrics.trial_id}: {exc}")

    def update_trial_status(self, status: TrialStatus) -> None:
        """Update the status file for a trial (overwrites).

        Args:
            status: TrialStatus object to persist
        """
        trial_dir = self.get_trial_dir(status.trial_id)
        status_file = trial_dir / "status.json"

        with self._write_lock:
            try:
                # Write to temporary file first, then atomic rename
                temp_file = status_file.with_suffix(".tmp")
                with temp_file.open("w", encoding="utf-8") as f:
                    # Convert numpy types to Python native types for JSON serialization
                    status_dict = convert_numpy_types(status.to_dict())
                    json.dump(status_dict, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())

                temp_file.replace(status_file)  # Atomic on POSIX
            except Exception as exc:
                logger.error(f"[HPO Metrics] Failed to update status for {status.trial_id}: {exc}")

    def load_trial_status(self, trial_id: str) -> TrialStatus | None:
        """Load the current status of a trial.

        Args:
            trial_id: ID of the trial

        Returns:
            TrialStatus object or None if not found
        """
        trial_dir = self.get_trial_dir(trial_id)
        status_file = trial_dir / "status.json"

        if not status_file.exists():
            return None

        try:
            with status_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            return TrialStatus(
                trial_id=data["trial_id"],
                status=data["status"],
                start_time=data.get("start_time"),
                end_time=data.get("end_time"),
                best_loss=data.get("best_loss"),
                best_step=data.get("best_step"),
                total_steps=data.get("total_steps", 0),
                error_message=data.get("error_message"),
                hyperparameters=data.get("hyperparameters", {}),
                param_count=data.get("param_count"),
                efficiency_score=data.get("efficiency_score"),
                # Multi-objective quality metrics
                perplexity=data.get("perplexity"),
                mean_confidence=data.get("mean_confidence"),
                expected_calibration_error=data.get("expected_calibration_error"),
                composite_score=data.get("composite_score"),
            )
        except Exception as exc:
            logger.error(f"[HPO Metrics] Failed to load status for {trial_id}: {exc}")
            return None

    def load_trial_metrics(self, trial_id: str, last_n: int | None = None) -> list[TrialMetrics]:
        """Load metrics history for a trial.

        Args:
            trial_id: ID of the trial
            last_n: If specified, return only the last N entries

        Returns:
            List of TrialMetrics objects
        """
        trial_dir = self.get_trial_dir(trial_id)
        metrics_file = trial_dir / "metrics.jsonl"

        if not metrics_file.exists():
            return []

        metrics_list = []
        try:
            with metrics_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    metrics_list.append(
                        TrialMetrics(
                            trial_id=data["trial_id"],
                            step=data["step"],
                            timestamp=data["timestamp"],
                            loss=data.get("loss"),
                            gradient_norm=data.get("gradient_norm"),
                            learning_rate=data.get("learning_rate"),
                            memory_mb=data.get("memory_mb"),
                            peak_memory_mb=data.get("peak_memory_mb"),
                            wall_time_seconds=data.get("wall_time_seconds"),
                            extras=data.get("extras", {}),
                        )
                    )
        except Exception as exc:
            logger.error(f"[HPO Metrics] Failed to load metrics for {trial_id}: {exc}")
            return []

        if last_n is not None and len(metrics_list) > last_n:
            return metrics_list[-last_n:]

        return metrics_list

    def list_trials(self) -> list[str]:
        """List all trial IDs that have reported metrics.

        Returns:
            List of trial IDs (directory names)
        """
        if not self.hpo_root.exists():
            return []

        trials = []
        for item in self.hpo_root.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                trials.append(item.name)

        return sorted(trials)

    def get_best_trial(
        self,
        use_efficiency: bool = True,
        use_composite: bool = True,
    ) -> tuple[str, float] | None:
        """Find the trial with the best (lowest) loss or composite score.

        Args:
            use_efficiency: If True and efficiency_score is available,
                rank by efficiency_score instead of loss. This rewards
                smaller models that achieve similar accuracy.
            use_composite: If True and composite_score is available,
                rank by composite_score (multi-objective). Takes precedence.

        Returns:
            Tuple of (trial_id, best_score) or None if no trials found
        """
        trials = self.list_trials()
        if not trials:
            return None

        best_trial_id = None
        best_score = float("inf")

        for trial_id in trials:
            status = self.load_trial_status(trial_id)
            if status:
                # Priority: composite > efficiency > loss
                if use_composite and status.composite_score is not None:
                    score = status.composite_score
                elif use_efficiency and status.efficiency_score is not None:
                    score = status.efficiency_score
                elif status.best_loss is not None:
                    score = status.best_loss
                else:
                    continue

                if score < best_score:
                    best_score = score
                    best_trial_id = trial_id

        if best_trial_id is None:
            return None

        return best_trial_id, best_score

    def export_summary(self) -> dict[str, Any]:
        """Export a summary of all trials.

        Returns:
            Dictionary with trial summaries and best trial info
        """
        trials = self.list_trials()
        summaries = []

        for trial_id in trials:
            status = self.load_trial_status(trial_id)
            if status:
                summaries.append(status.to_dict())

        # Calculate aggregate memory stats
        total_peak_memory = 0.0
        max_peak_memory = 0.0
        for trial_id in trials:
            metrics = self.load_trial_metrics(trial_id, last_n=1)
            if metrics and metrics[-1].peak_memory_mb:
                peak = metrics[-1].peak_memory_mb
                total_peak_memory += peak
                max_peak_memory = max(max_peak_memory, peak)

        return {
            "total_trials": len(trials),
            "trials": summaries,
            "best_trial": (
                {
                    "trial_id": self.get_best_trial()[0],
                    "best_loss": self.get_best_trial()[1],
                }
                if self.get_best_trial()
                else None
            ),
            "memory_stats": {
                "avg_peak_memory_mb": total_peak_memory / len(trials) if trials else 0,
                "max_peak_memory_mb": max_peak_memory,
            },
            "timestamp": time.time(),
        }

    def cleanup_trial(self, trial_id: str) -> None:
        """Remove all artifacts for a trial.

        Args:
            trial_id: ID of the trial to clean up
        """
        trial_dir = self.get_trial_dir(trial_id)
        if trial_dir.exists():
            import shutil

            shutil.rmtree(trial_dir)
            logger.info(f"[HPO Metrics] Cleaned up trial {trial_id}")
