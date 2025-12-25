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


# =============================================================================
# SMART PARAMETER BUDGET TUNER
# =============================================================================


@dataclass
class SkippedTrialRecord:
    """Record of a trial that was skipped due to budget constraints."""

    trial_id: str
    config: dict[str, Any]
    estimated_params: int
    param_budget: int
    reason: str
    timestamp: float = field(default_factory=time.time)
    architecture_signature: str = ""  # For clustering similar configs

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "trial_id": self.trial_id,
            "config": self.config,
            "estimated_params": self.estimated_params,
            "param_budget": self.param_budget,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "architecture_signature": self.architecture_signature,
            "budget_ratio": (
                self.estimated_params / self.param_budget if self.param_budget > 0 else 0
            ),
        }


class OversizedConfigTracker:
    """Tracks configurations that exceeded parameter budget for adaptive sampling.

    Enterprise Features:
    - Persists skipped configs to disk for cross-session learning
    - Computes architecture signatures for clustering similar failures
    - Provides statistics on common over-budget patterns
    - Enables adaptive sampling bias to avoid similar configurations

    Example:
        >>> tracker = OversizedConfigTracker(param_budget=1_000_000_000)
        >>> tracker.record_oversized({"hidden_dim": 4096, "num_blocks": 24}, 5_000_000_000)
        >>> tracker.should_skip({"hidden_dim": 4096, "num_blocks": 20})  # Likely True
    """

    def __init__(
        self,
        param_budget: int = 1_000_000_000,
        hpo_root: Path | None = None,
        max_history: int = 500,
    ):
        """Initialize the oversized config tracker.

        Args:
            param_budget: Maximum allowed parameter count
            hpo_root: Root directory for persistence
            max_history: Maximum number of records to keep
        """
        self.param_budget = param_budget
        self.hpo_root = Path(hpo_root or DEFAULT_HPO_ROOT).resolve()
        self.max_history = max_history
        self._skipped_records: list[SkippedTrialRecord] = []
        self._architecture_failures: dict[str, int] = {}  # signature -> failure count
        self._lock = threading.Lock()

        # Load existing history
        self._load_history()
        logger.info(
            f"[SmartTuner] Initialized with budget={param_budget / 1e9:.1f}B, "
            f"history={len(self._skipped_records)} records"
        )

    def _get_architecture_signature(self, config: dict[str, Any]) -> str:
        """Compute a signature for the architecture portion of a config.

        This groups similar architectures together for pattern detection.
        """
        # Key architecture parameters that most affect param count
        key_params = [
            ("hidden_dim", config.get("hidden_dim", config.get("embedding_dim", 512))),
            ("blocks", config.get("num_reasoning_blocks", 8)),
            ("experts", config.get("num_moe_experts", 8)),
            ("mamba", config.get("mamba_state_dim", 64)),
        ]
        return "_".join(f"{k}{v}" for k, v in key_params)

    def _persistence_file(self) -> Path:
        """Get the path to the persistence file."""
        self.hpo_root.mkdir(parents=True, exist_ok=True)
        return self.hpo_root / "oversized_configs.jsonl"

    def _load_history(self) -> None:
        """Load skipped config history from disk."""
        persistence_file = self._persistence_file()
        if not persistence_file.exists():
            return

        try:
            with persistence_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    record = SkippedTrialRecord(
                        trial_id=data.get("trial_id", "unknown"),
                        config=data.get("config", {}),
                        estimated_params=data.get("estimated_params", 0),
                        param_budget=data.get("param_budget", self.param_budget),
                        reason=data.get("reason", "exceeded_budget"),
                        timestamp=data.get("timestamp", 0),
                        architecture_signature=data.get("architecture_signature", ""),
                    )
                    self._skipped_records.append(record)

                    # Track architecture failure counts
                    sig = record.architecture_signature or self._get_architecture_signature(
                        record.config
                    )
                    self._architecture_failures[sig] = self._architecture_failures.get(sig, 0) + 1

            # Trim to max_history
            if len(self._skipped_records) > self.max_history:
                self._skipped_records = self._skipped_records[-self.max_history :]

            logger.debug(
                f"[SmartTuner] Loaded {len(self._skipped_records)} oversized config records"
            )

        except Exception as e:
            logger.warning(f"[SmartTuner] Failed to load history: {e}")

    def record_oversized(
        self,
        trial_id: str,
        config: dict[str, Any],
        estimated_params: int,
        reason: str = "exceeded_budget",
    ) -> SkippedTrialRecord:
        """Record a configuration that exceeded the parameter budget.

        Args:
            trial_id: The trial ID that was skipped
            config: The hyperparameter configuration
            estimated_params: Estimated parameter count
            reason: Reason for skipping

        Returns:
            The created SkippedTrialRecord
        """
        signature = self._get_architecture_signature(config)

        record = SkippedTrialRecord(
            trial_id=trial_id,
            config=config,
            estimated_params=estimated_params,
            param_budget=self.param_budget,
            reason=reason,
            timestamp=time.time(),
            architecture_signature=signature,
        )

        with self._lock:
            self._skipped_records.append(record)
            self._architecture_failures[signature] = (
                self._architecture_failures.get(signature, 0) + 1
            )

            # Trim to max_history
            if len(self._skipped_records) > self.max_history:
                self._skipped_records = self._skipped_records[-self.max_history :]

            # Persist to disk
            try:
                with self._persistence_file().open("a", encoding="utf-8") as f:
                    json.dump(record.to_dict(), f, ensure_ascii=False)
                    f.write("\n")
            except Exception as e:
                logger.warning(f"[SmartTuner] Failed to persist oversized record: {e}")

        # Log the skip
        logger.warning(
            f"[SmartTuner] SKIPPED trial {trial_id}: {reason} "
            f"({estimated_params / 1e6:.1f}M > {self.param_budget / 1e6:.1f}M budget, "
            f"signature={signature})"
        )

        return record

    def get_failure_count(self, config: dict[str, Any]) -> int:
        """Get the number of times a similar architecture has failed.

        Args:
            config: Configuration to check

        Returns:
            Number of previous failures with similar architecture
        """
        signature = self._get_architecture_signature(config)
        return self._architecture_failures.get(signature, 0)

    def should_avoid_architecture(
        self,
        config: dict[str, Any],
        threshold: int = 2,
    ) -> tuple[bool, str]:
        """Check if an architecture should be avoided based on failure history.

        Args:
            config: Configuration to check
            threshold: Number of failures before avoiding

        Returns:
            Tuple of (should_avoid, reason)
        """
        signature = self._get_architecture_signature(config)
        failure_count = self._architecture_failures.get(signature, 0)

        if failure_count >= threshold:
            return True, f"Architecture {signature} failed {failure_count} times"

        return False, ""

    def get_safe_architecture_bounds(self) -> dict[str, int]:
        """Compute safe bounds for architecture parameters based on failure history.

        Analyzes failed configs to determine maximum safe values for each parameter.

        Returns:
            Dictionary with recommended maximum values for architecture params
        """
        if not self._skipped_records:
            return {}

        # Find minimum values that still exceeded budget
        min_failing = {
            "hidden_dim": float("inf"),
            "num_reasoning_blocks": float("inf"),
            "num_moe_experts": float("inf"),
            "embedding_dim": float("inf"),
        }

        for record in self._skipped_records:
            config = record.config
            for key in min_failing:
                if key in config and config[key] < min_failing[key]:
                    min_failing[key] = config[key]

        # Return one step below minimum failing values
        safe_bounds = {}
        step_down = {
            "hidden_dim": [256, 512, 768, 1024, 2048, 4096],
            "embedding_dim": [256, 512, 768, 1024, 2048, 4096],
            "num_reasoning_blocks": [4, 6, 8, 12, 16, 20, 24],
            "num_moe_experts": [4, 6, 8, 12],
        }

        for key, min_val in min_failing.items():
            if min_val != float("inf") and key in step_down:
                # Find the largest value below the minimum failing value
                options = [v for v in step_down[key] if v < min_val]
                if options:
                    safe_bounds[f"max_{key}"] = max(options)

        return safe_bounds

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about oversized configurations.

        Returns:
            Dictionary with failure statistics
        """
        if not self._skipped_records:
            return {"total_skipped": 0}

        # Compute average overage
        total_overage = sum(r.estimated_params - r.param_budget for r in self._skipped_records)
        avg_overage = total_overage / len(self._skipped_records)

        # Most common failing architectures
        sorted_failures = sorted(self._architecture_failures.items(), key=lambda x: -x[1])[:5]

        return {
            "total_skipped": len(self._skipped_records),
            "avg_overage_params": avg_overage,
            "avg_overage_pct": (
                avg_overage / self.param_budget * 100 if self.param_budget > 0 else 0
            ),
            "unique_architectures_failed": len(self._architecture_failures),
            "top_failing_architectures": sorted_failures,
            "safe_bounds": self.get_safe_architecture_bounds(),
        }

    def get_skipped_records(self, last_n: int | None = None) -> list[SkippedTrialRecord]:
        """Get skipped trial records.

        Args:
            last_n: Return only the last N records

        Returns:
            List of SkippedTrialRecord objects
        """
        if last_n is not None:
            return self._skipped_records[-last_n:]
        return self._skipped_records.copy()


class BudgetAwareHPOSampler:
    """Smart HPO sampler with automatic parameter budget enforcement.

    Enterprise Features:
    - Automatically estimates parameter count before trial execution
    - Skips trials that exceed budget (no wasted compute)
    - Tracks failure patterns for adaptive sampling
    - Provides WebUI-compatible skip statistics

    Example:
        >>> sampler = BudgetAwareHPOSampler(param_budget=1_000_000_000)
        >>> result = sampler.sample_with_budget_check(trial_id=0, base_sampler=my_sampler)
        >>> if result["skipped"]:
        ...     print(f"Skipped: {result['reason']}")
    """

    def __init__(
        self,
        param_budget: int,
        estimate_params_fn: Any | None = None,
        hpo_root: Path | None = None,
        max_resample_attempts: int = 10,
        skip_on_repeated_failure: bool = True,
    ):
        """Initialize budget-aware sampler.

        Args:
            param_budget: Maximum allowed parameter count
            estimate_params_fn: Function to estimate params from config
            hpo_root: Root directory for persistence
            max_resample_attempts: Max attempts before skipping
            skip_on_repeated_failure: Skip if architecture failed before
        """
        self.param_budget = param_budget
        self.hpo_root = Path(hpo_root or DEFAULT_HPO_ROOT).resolve()
        self.max_resample_attempts = max_resample_attempts
        self.skip_on_repeated_failure = skip_on_repeated_failure

        # Lazy import estimate_params if not provided
        if estimate_params_fn is None:
            try:
                from highnoon.services.hpo_manager import estimate_model_params

                self.estimate_params = estimate_model_params
            except ImportError:
                logger.warning("[SmartTuner] estimate_model_params not available")
                self.estimate_params = lambda x: 0
        else:
            self.estimate_params = estimate_params_fn

        # Initialize tracker
        self.tracker = OversizedConfigTracker(
            param_budget=param_budget,
            hpo_root=hpo_root,
        )

        # Statistics
        self._total_samples = 0
        self._skipped_samples = 0
        self._resampled_count = 0

        logger.info(
            f"[SmartTuner] BudgetAwareHPOSampler initialized: "
            f"budget={param_budget / 1e9:.1f}B, max_attempts={max_resample_attempts}"
        )

    def sample_with_budget_check(
        self,
        trial_id: int | str,
        sample_fn: Any,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Sample a configuration with automatic budget checking.

        Args:
            trial_id: Trial identifier
            sample_fn: Function to call for sampling (e.g., search_space.sample)
            sample_kwargs: Additional kwargs for sample_fn

        Returns:
            Dictionary with:
            - config: The sampled configuration (or None if skipped)
            - skipped: True if trial was skipped
            - reason: Reason for skipping (if applicable)
            - estimated_params: Estimated parameter count
            - attempts: Number of sampling attempts
        """
        trial_id_str = str(trial_id)
        sample_kwargs = sample_kwargs or {}
        self._total_samples += 1

        # Check if we should avoid based on history
        if self.skip_on_repeated_failure:
            # Pre-check: sample once to see the architecture pattern
            test_config = sample_fn(trial_id, **sample_kwargs)
            should_avoid, avoid_reason = self.tracker.should_avoid_architecture(test_config)

            if should_avoid:
                # Don't even try - this architecture pattern fails repeatedly
                estimated = self.estimate_params(test_config)
                self._skipped_samples += 1

                self.tracker.record_oversized(
                    trial_id=trial_id_str,
                    config=test_config,
                    estimated_params=estimated,
                    reason=f"pattern_skip: {avoid_reason}",
                )

                return {
                    "config": None,
                    "skipped": True,
                    "reason": avoid_reason,
                    "estimated_params": estimated,
                    "attempts": 0,
                    "pattern_skip": True,
                }

        # Try sampling with budget check
        attempts = 0
        last_config = None
        last_estimated = 0

        while attempts < self.max_resample_attempts:
            # Sample configuration
            config = sample_fn(trial_id + attempts * 1000, **sample_kwargs)
            last_config = config
            attempts += 1

            # Estimate parameters
            estimated_params = self.estimate_params(config)
            last_estimated = estimated_params

            # Check budget
            if estimated_params <= self.param_budget:
                # Good config - within budget
                logger.debug(
                    f"[SmartTuner] Trial {trial_id_str}: config OK "
                    f"({estimated_params / 1e6:.1f}M <= {self.param_budget / 1e6:.1f}M) "
                    f"after {attempts} attempt(s)"
                )

                if attempts > 1:
                    self._resampled_count += 1

                return {
                    "config": config,
                    "skipped": False,
                    "reason": None,
                    "estimated_params": estimated_params,
                    "attempts": attempts,
                    "pattern_skip": False,
                }

            # Log the excess
            logger.info(
                f"[SmartTuner] Trial {trial_id_str} attempt {attempts}: "
                f"config exceeds budget ({estimated_params / 1e6:.1f}M > "
                f"{self.param_budget / 1e6:.1f}M), resampling..."
            )

        # Exceeded max attempts - skip this trial
        self._skipped_samples += 1

        self.tracker.record_oversized(
            trial_id=trial_id_str,
            config=last_config,
            estimated_params=last_estimated,
            reason=f"exceeded_after_{attempts}_attempts",
        )

        return {
            "config": None,
            "skipped": True,
            "reason": f"Exceeded budget after {attempts} attempts "
            f"({last_estimated / 1e6:.1f}M > {self.param_budget / 1e6:.1f}M)",
            "estimated_params": last_estimated,
            "attempts": attempts,
            "pattern_skip": False,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get sampling statistics.

        Returns:
            Dictionary with sampling and skip statistics
        """
        tracker_stats = self.tracker.get_statistics()

        return {
            "total_samples": self._total_samples,
            "skipped_samples": self._skipped_samples,
            "resampled_count": self._resampled_count,
            "skip_rate": (
                self._skipped_samples / self._total_samples if self._total_samples > 0 else 0
            ),
            "param_budget": self.param_budget,
            **tracker_stats,
        }

    def get_skipped_for_webui(self) -> list[dict[str, Any]]:
        """Get skipped trials formatted for WebUI display.

        Returns:
            List of skipped trial records as dictionaries
        """
        return [r.to_dict() for r in self.tracker.get_skipped_records()]


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
