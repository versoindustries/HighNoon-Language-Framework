"""Enterprise-grade HPO Sweep Executor with multi-trial orchestration.

This module provides the central orchestrator for running HPO sweeps with:
- Multi-trial execution (the critical fix for single-trial bug)
- Fault tolerance with configurable retry strategies
- Checkpoint/resume support for crash recovery
- Integration with multiple scheduler backends
- Multi-objective optimization with Pareto tracking

Copyright 2025 Verso Industries
"""

from __future__ import annotations

import asyncio
import gc
import gzip
import json
import logging
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Phase 201.6: Sweep compression config
from highnoon.config import SWEEP_COMPRESS_CHECKPOINTS, SWEEP_MAX_IN_MEMORY_TRIALS
from highnoon.services.hpo_utils import next_power_of_2, snap_to_multiple

# Note: Using synchronous file I/O for checkpoints (small files)

if TYPE_CHECKING:
    from highnoon.services.hpo_schedulers import HPOSchedulerBase

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class FaultToleranceConfig:
    """Fault tolerance configuration for HPO sweeps.

    Attributes:
        max_retries_per_trial: Maximum retry attempts for failed trials
        retry_on_oom: Whether to retry on OOM with reduced config
        retry_on_nan: Whether to retry on NaN loss with adjusted LR
        checkpoint_frequency: Steps between state checkpoints
        memory_scale_factor: Factor to reduce memory on OOM (0.75 = 25% smaller)
        lr_scale_factor: Factor to reduce LR on NaN (0.5 = halve)
    """

    max_retries_per_trial: int = 3
    retry_on_oom: bool = True
    retry_on_nan: bool = True
    checkpoint_frequency: int = 10
    memory_scale_factor: float = 0.75
    lr_scale_factor: float = 0.5


@dataclass
class SweepConfig:
    """Complete HPO sweep configuration.

    Attributes:
        max_trials: Maximum number of trials to run
        max_parallel: Maximum concurrent trials (default 1 for sequential)
        time_budget_hours: Optional time limit for entire sweep
        objective: Optimization objective ("loss", "perplexity", "composite")
        direction: Optimization direction ("minimize" or "maximize")
        epochs_per_trial: Training epochs per trial
        steps_per_epoch: Steps per epoch
        min_epochs: Minimum epochs before early stopping
        max_epochs: Maximum epochs per trial
        search_strategy: Scheduler type ("random", "bayesian", "hyperband", etc.)
        use_optuna: Whether to use Optuna for sampling
        fault_tolerance: Fault tolerance configuration
        alpha_loss: Weight for loss in composite score
        beta_perplexity: Weight for perplexity in composite score
        gamma_calibration: Weight for calibration (ECE) in composite score
        lambda_efficiency: Weight for efficiency in composite score
        param_budget: Maximum parameter count
        evaluate_quality: Whether to evaluate quality metrics at trial end
        quality_eval_samples: Number of samples for quality evaluation
    """

    max_trials: int = 50
    max_parallel: int = 1
    time_budget_hours: float | None = None
    objective: str = "composite"
    direction: str = "minimize"
    epochs_per_trial: int = 3  # 3 epochs × 500 steps = 1500 steps (HPO_MAX_TRIAL_STEPS limit)
    steps_per_epoch: int = 500  # Per-epoch step count for multi-epoch HPO trials
    min_epochs: int = 1
    max_epochs: int = 100
    search_strategy: str = "bayesian"
    use_optuna: bool = True
    hyperband_eta: int = 3
    pbt_population_size: int = 8
    pbt_perturbation_interval: int = 10
    fault_tolerance: FaultToleranceConfig = field(default_factory=FaultToleranceConfig)
    alpha_loss: float = 0.4
    beta_perplexity: float = 0.3
    gamma_calibration: float = 0.1
    lambda_efficiency: float = 0.2
    param_budget: int = 1_000_000_000
    evaluate_quality: bool = True
    quality_eval_samples: int = 50
    sweep_id: str = ""

    # Convergence-based stopping (runs until convergence detected)
    auto_stop_on_convergence: bool = True

    # PBT (Population-Based Training) configuration
    enable_pbt: bool = False  # Enable cross-trial weight transfer
    pbt_checkpoint_dir: str = "pbt_checkpoints"  # Shared checkpoint directory
    pbt_exploit_threshold: float = 1.2  # Exploit if 20% worse than best
    pbt_explore_perturbation: float = 0.2  # ±20% perturbation on explore

    # Additional model config passthrough
    model_config: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    """Result from a completed HPO trial.

    Attributes:
        trial_id: Unique trial identifier
        loss: Final training loss
        perplexity: Optional perplexity score
        mean_confidence: Optional mean confidence score
        ece: Optional expected calibration error
        param_count: Model parameter count
        hyperparams: Hyperparameters used
        epochs_completed: Number of epochs completed
        wall_time_seconds: Total wall clock time
        memory_peak_mb: Peak memory usage in MB
        error: Error message if trial failed
        composite_score: Weighted composite score
        is_on_pareto_frontier: Whether trial is Pareto-optimal
    """

    trial_id: str
    loss: float
    perplexity: float | None = None
    mean_confidence: float | None = None
    ece: float | None = None
    param_count: int = 0
    hyperparams: dict[str, Any] = field(default_factory=dict)
    epochs_completed: int = 0
    wall_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    throughput_tokens_per_sec: float = 0.0  # Training/generation throughput
    error: str | None = None
    composite_score: float = 0.0
    is_on_pareto_frontier: bool = False

    def _sanitize_float(self, value: float | None) -> float | None:
        """Sanitize float for JSON serialization.

        JSON spec doesn't allow NaN or Inf values, so we convert them to None.

        Args:
            value: Float value to sanitize.

        Returns:
            Sanitized value safe for JSON serialization.
        """
        if value is None:
            return None
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Field names match frontend HPOTrialInfo type in types.ts.
        NaN and Inf float values are converted to None for JSON compliance.
        """
        sanitized_loss = self._sanitize_float(self.loss)
        return {
            "trial_id": self.trial_id,
            "loss": sanitized_loss,
            "perplexity": self._sanitize_float(self.perplexity),
            "mean_confidence": self._sanitize_float(self.mean_confidence),
            "expected_calibration_error": self._sanitize_float(
                self.ece
            ),  # Frontend expects this name
            "param_count": self.param_count,
            "hyperparams": self.hyperparams,
            "epochs_completed": self.epochs_completed,
            "wall_time_seconds": self._sanitize_float(self.wall_time_seconds),
            "memory_peak_mb": self._sanitize_float(self.memory_peak_mb),
            "memory_mb": self._sanitize_float(
                self.memory_peak_mb
            ),  # Alias for frontend compatibility
            "throughput_tokens_per_sec": self._sanitize_float(self.throughput_tokens_per_sec),
            "error": self.error,
            "composite_score": self._sanitize_float(self.composite_score),
            "is_on_pareto_frontier": self.is_on_pareto_frontier,
            # Frontend table expects these fields
            "status": "failed" if self.error else "completed",
            "learning_rate": self._sanitize_float(self.hyperparams.get("learning_rate", 0)),
            "duration_seconds": self._sanitize_float(self.wall_time_seconds),
            "best_loss": sanitized_loss,
            "step": self.epochs_completed,
            "batch_size": self.hyperparams.get("batch_size", 0),
            "optimizer": self.hyperparams.get("optimizer", "unknown"),
        }


@dataclass
class SweepResult:
    """Result from a completed HPO sweep.

    Attributes:
        sweep_id: Unique sweep identifier
        completed_trials: List of all trial results
        best_trial: Best trial by objective
        best_trial_id: ID of best trial
        best_loss: Best loss achieved
        best_hyperparams: Hyperparameters of best trial
        total_wall_time_seconds: Total sweep duration
        state: Final sweep state ("completed", "cancelled", "failed")
        pareto_frontier: Trials on Pareto frontier
    """

    sweep_id: str
    completed_trials: list[TrialResult] = field(default_factory=list)
    best_trial: TrialResult | None = None
    best_trial_id: str | None = None
    best_loss: float | None = None
    best_hyperparams: dict[str, Any] | None = None
    total_wall_time_seconds: float = 0.0
    state: str = "completed"
    pareto_frontier: list[TrialResult] = field(default_factory=list)


# =============================================================================
# Retry Strategy
# =============================================================================


class RetryStrategy:
    """Configurable retry strategy for failed trials.

    Modifies hyperparameters based on failure type to increase
    likelihood of success on retry.
    """

    def __init__(
        self,
        max_retries: int = 3,
        scale_factors: dict[str, float] | None = None,
    ):
        """Initialize retry strategy.

        Args:
            max_retries: Maximum retry attempts per trial
            scale_factors: Dictionary of parameter scale factors for retries
        """
        self.max_retries = max_retries
        self.scale_factors = scale_factors or {
            "learning_rate": 0.5,
            "batch_size": 0.5,
            "num_moe_experts": 0.75,
            "hidden_dim": 0.75,
        }

    def modify_config_for_retry(
        self,
        config: dict[str, Any],
        failure_type: str,
        retry_count: int,
        guidance: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Modify hyperparameters based on failure type.

        Args:
            config: Original hyperparameter configuration
            failure_type: Type of failure (nan_loss, oom, swap_spike, etc.)
            retry_count: Current retry attempt number
            guidance: Optional fANOVA importance guidance

        Returns:
            Modified configuration with adjusted hyperparameters
        """
        modified = config.copy()

        if failure_type in ["nan_loss", "nan_gradient", "gradient_explosion"]:
            # Reduce learning rate
            if "learning_rate" in modified:
                modified["learning_rate"] = (
                    modified["learning_rate"] * self.scale_factors["learning_rate"]
                )
            # Increase gradient clipping
            modified["max_grad_norm"] = max(0.1, modified.get("max_grad_norm", 1.0) * 0.5)

        elif failure_type in ["oom", "swap_spike", "memory_critical"]:
            # Reduce memory footprint
            if "batch_size" in modified:
                modified["batch_size"] = max(
                    1, int(modified["batch_size"] * self.scale_factors["batch_size"])
                )
            if retry_count > 1:
                if "num_moe_experts" in modified:
                    modified["num_moe_experts"] = max(
                        2,
                        int(modified["num_moe_experts"] * self.scale_factors["num_moe_experts"]),
                    )
                if "hidden_dim" in modified:
                    modified["hidden_dim"] = snap_to_multiple(
                        modified["hidden_dim"] * self.scale_factors["hidden_dim"],
                        8,
                        min_val=128,
                    )

        elif failure_type == "crash":
            # Trial crashed before producing a loss - reduce model complexity
            logger.warning("[HPO] Trial crashed (no loss produced), reducing complexity")
            if "batch_size" in modified and retry_count == 1:
                modified["batch_size"] = max(1, modified["batch_size"] // 2)
            if retry_count > 1 and "num_reasoning_blocks" in modified:
                modified["num_reasoning_blocks"] = max(2, modified["num_reasoning_blocks"] - 2)
            if retry_count > 2 and "num_moe_experts" in modified:
                modified["num_moe_experts"] = max(2, modified["num_moe_experts"] // 2)

        elif failure_type == "budget_exceeded":
            # Model exceeds parameter budget - aggressively reduce architecture
            # Phase 201.7: Primary driver is often vocab_size + HQE, not blocks/dim
            logger.warning(
                f"[HPO] Budget exceeded on retry {retry_count}, "
                f"applying multi-stage reduction strategy"
            )

            # Stage 1: Reduce vocab_size (main driver when HQE is enabled)
            # HQE embedding params = vocab_size * hd_dim, so reducing vocab has huge impact
            current_vocab = modified.get("vocab_size")
            if current_vocab and current_vocab > 8000:
                new_vocab = max(8000, int(current_vocab * 0.5))  # Halve vocab each retry
                logger.info(f"  vocab_size: {current_vocab} → {new_vocab}")
                modified["vocab_size"] = new_vocab
            elif current_vocab is None:
                # If vocab_size is None (auto-detected), set explicit small value
                # This ensures the trial runner doesn't auto-expand to 128K
                budget = modified.get("param_budget", 100_000_000)
                embedding_dim = modified.get("hidden_dim", 256)
                use_hqe = modified.get("use_hyperdimensional_embedding", True)
                hd_dim = modified.get("hd_dim")

                # Import compute function (available in hpo_manager)
                from highnoon.services.hpo_manager import compute_max_vocab_for_budget

                # Use conservative model overhead on retries
                new_vocab = compute_max_vocab_for_budget(
                    param_budget=budget,
                    embedding_dim=embedding_dim,
                    use_hqe=use_hqe,
                    hd_dim=hd_dim,
                    model_overhead_fraction=0.6
                    + 0.1 * retry_count,  # More overhead on later retries
                )
                logger.info(f"  vocab_size: None → {new_vocab} (budget-aware)")
                modified["vocab_size"] = new_vocab

            # Stage 2: Reduce hd_dim if specified (reduces HQE embedding params)
            if "hd_dim" in modified and modified["hd_dim"]:
                new_hd_dim = max(256, int(modified["hd_dim"] * 0.5))
                logger.info(f"  hd_dim: {modified['hd_dim']} → {new_hd_dim}")
                modified["hd_dim"] = new_hd_dim

            # Stage 3: Reduce architecture params (blocks, experts, dim)
            # CRITICAL: Floor values must be LOW to avoid increasing minimal configs
            # Previous max(4, ...) was too high for configs starting at 2 blocks/3 experts
            reduction_factor = 0.6**retry_count  # 0.6, 0.36, 0.216...
            if "num_reasoning_blocks" in modified:
                new_blocks = max(2, int(modified["num_reasoning_blocks"] * reduction_factor))
                if new_blocks != modified["num_reasoning_blocks"]:
                    logger.info(f"  blocks: {modified['num_reasoning_blocks']} → {new_blocks}")
                    modified["num_reasoning_blocks"] = new_blocks
            if "num_moe_experts" in modified:
                new_experts = max(2, int(modified["num_moe_experts"] * reduction_factor))
                if new_experts != modified["num_moe_experts"]:
                    logger.info(f"  experts: {modified['num_moe_experts']} → {new_experts}")
                    modified["num_moe_experts"] = new_experts
            if "hidden_dim" in modified:
                # Use guidance to decide how much to reduce
                dim_reduction = 0.7
                if guidance and "hidden_dim" in guidance:
                    dim_reduction = 0.8 if guidance["hidden_dim"] > 0.1 else 0.6

                new_dim = snap_to_multiple(
                    modified["hidden_dim"] * (dim_reduction**retry_count), 8, min_val=128
                )
                if new_dim != modified["hidden_dim"]:
                    logger.info(f"  hidden_dim: {modified['hidden_dim']} → {new_dim}")
                    modified["hidden_dim"] = new_dim
                    modified["embedding_dim"] = new_dim

            # Ensure any other architectural params remain valid
            if "latent_kv_dim" in modified:
                modified["latent_kv_dim"] = snap_to_multiple(modified["latent_kv_dim"], 8)
            if "num_heads" in modified:
                modified["num_heads"] = next_power_of_2(modified["num_heads"])

        logger.info(
            f"[HPO] Modified config for retry {retry_count}: "
            f"failure_type={failure_type}, changes applied"
        )

        return modified

    def classify_failure(self, error_message: str | None, loss: float | None) -> str:
        """Classify the type of failure from error message and loss.

        Properly distinguishes between:
        - budget_exceeded: Explicitly reported budget violations
        - crash: Segfaults (exit -11) or other abrupt terminations
        - oom: Out of memory errors
        - nan_loss: NaN from gradient issues (not budget)
        - gradient_explosion: Inf loss from exploding gradients

        Args:
            error_message: Error message from trial
            loss: Final loss value (may be None if trial crashed)

        Returns:
            Failure type string
        """
        if error_message:
            error_lower = error_message.lower()

            # Check for EXPLICIT budget exceeded indicators
            # Must contain specific budget keywords, not just any error
            if "budget_exceeded" in error_lower or "exceeds parameter budget" in error_lower:
                return "budget_exceeded"

            # Segfault detection: exit code -11, SIGSEGV, or signal references
            if any(
                x in error_lower
                for x in ["exit code -11", "sigsegv", "segmentation fault", "signal 11"]
            ):
                return "crash"

            # CUDA/XLA errors are typically crashes, not budget issues
            if any(x in error_lower for x in ["cuda", "xla", "stream_executor", "tensorflow"]):
                return "crash"

            if "resource exhausted" in error_lower or "oom" in error_lower:
                return "oom"
            if "swap" in error_lower:
                return "swap_spike"
            if "memory" in error_lower:
                return "memory_critical"

        # S15: Improved classification for non-finite loss
        # Key fix: nan loss WITHOUT error should be classified as crash (likely segfault)
        # not budget_exceeded, since budget issues produce explicit messages
        if loss is not None:
            if math.isnan(loss):
                # If there's an error message, it's numerical instability (nan_loss)
                # If no error message, trial likely crashed before producing output
                return "nan_loss" if error_message else "crash"
            if math.isinf(loss):
                return "gradient_explosion"

        return "unknown" if error_message else "crash"


# =============================================================================
# Sweep Checkpointer
# =============================================================================


class SweepCheckpointer:
    """Persist sweep state for fault recovery.

    Saves checkpoint files that allow resuming interrupted sweeps.
    """

    def __init__(self, sweep_id: str, storage_path: Path):
        """Initialize checkpointer.

        Args:
            sweep_id: Unique sweep identifier
            storage_path: Directory for checkpoint files
        """
        self.sweep_id = sweep_id
        self.storage_path = Path(storage_path)
        self.checkpoint_file = self.storage_path / f"{sweep_id}_checkpoint.json"

    async def save_checkpoint(self, executor: SweepExecutor) -> None:
        """Save current sweep state to disk.

        Args:
            executor: SweepExecutor instance to checkpoint
        """
        self.storage_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "sweep_id": self.sweep_id,
            "timestamp": datetime.utcnow().isoformat(),
            "completed_trials": [t.to_dict() for t in executor.completed_trials],
            "pending_trial_count": executor.pending_trials.qsize(),
            "best_trial": (executor.best_trial.to_dict() if executor.best_trial else None),
            "config": {
                "max_trials": executor.config.max_trials,
                "max_parallel": executor.config.max_parallel,
                "objective": executor.config.objective,
            },
            # Phase 201.6: Track evicted trials count
            "evicted_trials_count": getattr(executor, "_evicted_trials_count", 0),
        }

        # Phase 201.6: Use gzip compression for checkpoint files
        if SWEEP_COMPRESS_CHECKPOINTS:
            checkpoint_path = self.storage_path / f"{self.sweep_id}_checkpoint.json.gz"
            with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2)
        else:
            with open(self.checkpoint_file, "w") as f:
                f.write(json.dumps(checkpoint, indent=2))

        logger.info(f"[HPO] Checkpoint saved: {len(executor.completed_trials)} trials completed")

    async def restore_checkpoint(self) -> dict[str, Any] | None:
        """Restore sweep state from checkpoint.

        Returns:
            Checkpoint data dictionary or None if no checkpoint exists
        """
        # Phase 201.6: Try gzip first, then plain JSON
        gzip_file = self.storage_path / f"{self.sweep_id}_checkpoint.json.gz"
        if gzip_file.exists():
            with gzip.open(gzip_file, "rt", encoding="utf-8") as f:
                return json.load(f)

        if not self.checkpoint_file.exists():
            return None

        with open(self.checkpoint_file) as f:
            return json.loads(f.read())


# =============================================================================
# Multi-Objective Scorer
# =============================================================================


class MultiObjectiveScorer:
    """Multi-objective scoring with Pareto frontier tracking.

    Computes weighted composite scores and tracks Pareto-optimal trials.
    Includes memory efficiency as a scoring component for enterprise-grade optimization.
    """

    def __init__(
        self,
        alpha_loss: float = 0.35,
        beta_perplexity: float = 0.25,
        gamma_calibration: float = 0.1,
        lambda_efficiency: float = 0.15,
        mu_memory: float = 0.15,  # NEW: Memory efficiency weight
        param_budget: int = 1_000_000_000,
        memory_budget_mb: float = 16_000.0,  # 16GB default memory budget
    ):
        """Initialize scorer.

        Args:
            alpha_loss: Weight for loss component
            beta_perplexity: Weight for perplexity component
            gamma_calibration: Weight for calibration (ECE) component
            lambda_efficiency: Weight for parameter efficiency component
            mu_memory: Weight for memory efficiency component
            param_budget: Parameter budget for efficiency calculation
            memory_budget_mb: Memory budget in MB for memory efficiency
        """
        self.weights = {
            "loss": alpha_loss,
            "perplexity": beta_perplexity,
            "ece": gamma_calibration,
            "efficiency": lambda_efficiency,
            "memory": mu_memory,
            "confidence": 0.1,  # NEW: Confidence weight (fixed for now, can be made configurable)
        }
        self.param_budget = param_budget
        self.memory_budget_mb = memory_budget_mb
        self.pareto_frontier: list[TrialResult] = []

    def compute_score(self, result: TrialResult) -> float:
        """Compute weighted composite score including memory efficiency.

        The composite score considers:
        - Loss: Primary training objective
        - Perplexity: Model quality metric
        - ECE (Expected Calibration Error): Confidence calibration
        - Parameter efficiency: Model size relative to budget
        - Memory efficiency: Peak memory usage relative to budget

        Args:
            result: Trial result to score

        Returns:
            Composite score (lower is better)
        """
        # Normalize metrics to [0, 1] range
        loss_norm = min(result.loss / 10.0, 1.0) if not math.isinf(result.loss) else 1.0
        ppl_norm = min(math.log(result.perplexity + 1) / 10.0, 1.0) if result.perplexity else 0.5
        ece = result.ece if result.ece is not None else 0.1
        efficiency = result.param_count / self.param_budget if result.param_count > 0 else 0.5

        # Memory efficiency: lower memory is better
        memory_norm = (
            min(result.memory_peak_mb / self.memory_budget_mb, 1.0)
            if result.memory_peak_mb > 0
            else 0.5  # Default if not tracked
        )

        # Confidence: higher is better, so we use (1 - conf) to minimize
        conf_norm = 1.0 - (result.mean_confidence or 0.5)

        return (
            self.weights["loss"] * loss_norm
            + self.weights["perplexity"] * ppl_norm
            + self.weights["ece"] * ece
            + self.weights["efficiency"] * efficiency
            + self.weights["memory"] * memory_norm
            + self.weights["confidence"] * conf_norm
        )

    def update_pareto_frontier(self, result: TrialResult) -> bool:
        """Update Pareto frontier with new result.

        Args:
            result: Trial result to consider

        Returns:
            True if result is on the frontier
        """
        # Check if any existing point dominates new result
        dominated = False
        for existing in self.pareto_frontier:
            if self._dominates(existing, result):
                dominated = True
                break

        if dominated:
            return False

        # Remove any points dominated by new result
        self.pareto_frontier = [p for p in self.pareto_frontier if not self._dominates(result, p)]
        self.pareto_frontier.append(result)
        result.is_on_pareto_frontier = True
        return True

    def _dominates(self, a: TrialResult, b: TrialResult) -> bool:
        """Check if result a Pareto-dominates result b.

        Considers all objectives: loss, ece, param_count, perplexity, and memory.
        Result a dominates b if a is no worse in all objectives and better in at least one.

        Args:
            a: First result
            b: Second result

        Returns:
            True if a dominates b
        """
        better_in_any = False

        # Compare objectives where lower is better
        for obj in ["loss", "ece", "param_count", "memory_peak_mb"]:
            val_a = getattr(a, obj, None)
            val_b = getattr(b, obj, None)
            if val_a is None or val_b is None:
                continue
            if val_a > val_b:
                return False
            if val_a < val_b:
                better_in_any = True

        # For perplexity, lower is better
        if a.perplexity is not None and b.perplexity is not None:
            if a.perplexity > b.perplexity:
                return False
            if a.perplexity < b.perplexity:
                better_in_any = True

        return better_in_any


# =============================================================================
# Sweep Executor
# =============================================================================


class SweepExecutor:
    """Enterprise-grade HPO sweep executor with multi-trial support.

    This is the core orchestrator that fixes the single-trial bug in the
    original implementation. It manages the complete lifecycle of an HPO sweep:

    1. Initializes scheduler and trial queue
    2. Spawns trials up to max_parallel limit
    3. Processes results and asks scheduler for more trials
    4. Handles failures with retry logic
    5. Checkpoints state for crash recovery
    6. Tracks best trial and Pareto frontier

    Example:
        config = SweepConfig(max_trials=10, max_parallel=2)
        scheduler = create_scheduler("hyperband", config)
        executor = SweepExecutor("sweep_123", config, scheduler)
        result = await executor.run()
        print(f"Best trial: {result.best_trial_id} with loss {result.best_loss}")
    """

    def __init__(
        self,
        sweep_id: str,
        config: SweepConfig,
        scheduler: HPOSchedulerBase,
        max_parallel: int = 1,
        storage_path: str | None = None,
        log_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        """Initialize the sweep executor.

        Args:
            sweep_id: Unique identifier for this sweep
            config: Sweep configuration
            scheduler: HPO scheduler instance
            max_parallel: Maximum concurrent trials
            storage_path: Directory for checkpoints and artifacts
            log_callback: Optional callback for log entries
        """
        self.sweep_id = sweep_id
        self.config = config
        self.scheduler = scheduler
        self.max_parallel = max_parallel
        self.storage_path = Path(storage_path or f"artifacts/hpo_trials/{sweep_id}")
        self.log_callback = log_callback

        # Trial tracking
        self.pending_trials: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.running_trials: dict[str, asyncio.Task[TrialResult]] = {}
        self.running_trial_configs: dict[str, dict[str, Any]] = {}  # Track configs for API
        self.completed_trials: list[TrialResult] = []
        self.best_trial: TrialResult | None = None
        self._trial_counter = 0

        # State management
        self._shutdown = asyncio.Event()
        self._lock = asyncio.Lock()
        self._start_time: float = 0.0

        # Retry and scoring
        self.retry_strategy = RetryStrategy(
            max_retries=config.fault_tolerance.max_retries_per_trial
        )
        self.scorer = MultiObjectiveScorer(
            alpha_loss=config.alpha_loss,
            beta_perplexity=config.beta_perplexity,
            gamma_calibration=config.gamma_calibration,
            lambda_efficiency=config.lambda_efficiency,
            param_budget=config.param_budget,
        )
        self.checkpointer = SweepCheckpointer(sweep_id, self.storage_path)

        # Phase 201.6: LRU eviction tracking
        self._max_in_memory_trials = SWEEP_MAX_IN_MEMORY_TRIALS
        self._evicted_trials_count = 0
        self._pareto_protected_ids: set[str] = set()  # Never evict Pareto-optimal trials

    def _log(self, level: str, message: str, **extra: Any) -> None:
        """Log a message and optionally call the log callback.

        Args:
            level: Log level (INFO, WARNING, ERROR)
            message: Log message
            **extra: Additional fields for log entry
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **extra,
        }

        if level == "ERROR":
            logger.error(f"[HPO] {message}")
        elif level == "WARNING":
            logger.warning(f"[HPO] {message}")
        else:
            logger.info(f"[HPO] {message}")

        if self.log_callback:
            self.log_callback(log_entry)

    async def run(self) -> SweepResult:
        """Execute the full HPO sweep with multi-trial support.

        This is the main entry point that orchestrates the entire sweep:
        1. Check for existing checkpoint and resume if found
        2. Get initial trials from scheduler
        3. Loop until max_trials reached or stopped
        4. Save checkpoints periodically
        5. Return final results

        Returns:
            SweepResult with all trials and best configuration
        """
        self._start_time = time.time()
        self._log("INFO", f"Starting sweep {self.sweep_id} with {self.config.max_trials} trials")

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Check for existing checkpoint
        checkpoint = await self.checkpointer.restore_checkpoint()
        if checkpoint:
            self._restore_from_checkpoint(checkpoint)
            self._log(
                "INFO",
                f"Resumed from checkpoint with {len(self.completed_trials)} completed trials",
            )

        # Get initial trials from scheduler
        initial_trial_count = min(
            self.max_parallel,
            self.config.max_trials - len(self.completed_trials),
        )
        if initial_trial_count > 0:
            await self._queue_next_trials(initial_trial_count)

        # Main orchestration loop
        while not self._shutdown.is_set():
            # Check time budget
            if self.config.time_budget_hours:
                elapsed_hours = (time.time() - self._start_time) / 3600
                if elapsed_hours >= self.config.time_budget_hours:
                    self._log("WARNING", "Time budget exceeded, stopping sweep")
                    break

            # Start pending trials up to max_parallel
            while len(self.running_trials) < self.max_parallel and not self.pending_trials.empty():
                trial_config = await self.pending_trials.get()
                trial_id = trial_config["trial_id"]
                task = asyncio.create_task(self._run_trial(trial_config))
                self.running_trials[trial_id] = task
                self.running_trial_configs[trial_id] = trial_config  # Track config for API

            # Wait for any trial to complete
            if self.running_trials:
                done, _ = await asyncio.wait(
                    self.running_trials.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    result = await task
                    await self._process_trial_result(result)

            # Check if sweep is complete
            if len(self.completed_trials) >= self.config.max_trials:
                self._log("INFO", f"Completed all {self.config.max_trials} trials")
                break

            # Queue more trials if needed
            if (
                self.pending_trials.empty()
                and len(self.running_trials) == 0
                and len(self.completed_trials) < self.config.max_trials
            ):
                remaining = self.config.max_trials - len(self.completed_trials)
                to_queue = min(self.max_parallel, remaining)
                queued = await self._queue_next_trials(to_queue)
                if queued == 0:
                    self._log("INFO", "Scheduler has no more trials to run")
                    break

        # Final cleanup
        return self._build_sweep_result()

    async def _queue_next_trials(self, n_trials: int) -> int:
        """Queue the next batch of trials from the scheduler.

        Args:
            n_trials: Number of trials to queue

        Returns:
            Number of trials actually queued
        """
        try:
            trial_configs = self.scheduler.get_next_trials(n_trials)

            for trial_config in trial_configs:
                # Merge scheduler config with model config
                full_config = {
                    **self.config.model_config,
                    **trial_config.hyperparams,
                    "trial_id": trial_config.trial_id,
                    "budget": trial_config.budget,
                    "sweep_id": self.sweep_id,
                }
                await self.pending_trials.put(full_config)

            return len(trial_configs)

        except Exception as e:
            self._log("ERROR", f"Failed to get trials from scheduler: {e}")
            return 0

    async def _run_trial(self, config: dict[str, Any]) -> TrialResult:
        """Execute a single trial with retry support.

        Args:
            config: Trial configuration with hyperparameters

        Returns:
            TrialResult with training results
        """
        trial_id = config.get("trial_id", f"trial_{self._trial_counter}")
        self._trial_counter += 1
        retries = 0
        max_retries = self.config.fault_tolerance.max_retries_per_trial
        current_config = config.copy()

        while retries <= max_retries:
            try:
                self._log("INFO", f"Starting trial {trial_id} (attempt {retries + 1})")
                result = await self._execute_trial_subprocess(trial_id, current_config)

                # DEBUG: Log result details for tracing
                self._log(
                    "INFO",
                    f"[DEBUG] Trial {trial_id} result: loss={result.loss}, "
                    f"error={result.error[:100] if result.error else None}",
                )

                # Check for budget exceeded - retry with smaller architecture
                # Use proper classification instead of loose string matching
                # This prevents segfaults/CUDA errors from being misclassified as budget_exceeded
                failure_type = self.retry_strategy.classify_failure(result.error, result.loss)
                if failure_type == "budget_exceeded":
                    self._log(
                        "INFO",
                        f"[DEBUG] Trial {trial_id} detected as budget_exceeded, "
                        f"retries={retries}/{max_retries}",
                    )
                    if retries < max_retries:
                        # Retry with reduced architecture
                        failure_type = "budget_exceeded"
                        old_blocks = current_config.get("num_reasoning_blocks")
                        old_experts = current_config.get("num_moe_experts")
                        old_dim = current_config.get("hidden_dim")

                        guidance = (
                            self.scheduler.get_reduction_guidance()
                            if hasattr(self.scheduler, "get_reduction_guidance")
                            else None
                        )
                        current_config = self.retry_strategy.modify_config_for_retry(
                            current_config, failure_type, retries + 1, guidance=guidance
                        )
                        retries += 1

                        self._log(
                            "WARNING",
                            f"Trial {trial_id} exceeded budget, retrying with smaller config: "
                            f"blocks={old_blocks}→{current_config.get('num_reasoning_blocks')}, "
                            f"experts={old_experts}→{current_config.get('num_moe_experts')}, "
                            f"dim={old_dim}→{current_config.get('hidden_dim')}",
                        )
                        continue
                    else:
                        # Exhausted retries - mark as budget exceeded
                        self._log(
                            "WARNING",
                            f"Trial {trial_id} exceeded budget after {max_retries} attempts",
                        )
                        result.error = "BUDGET_EXCEEDED: " + (result.error or "")
                        return result

                if result.loss == float("inf") or math.isnan(result.loss):
                    failure_type = self.retry_strategy.classify_failure(result.error, result.loss)
                    self._log(
                        "INFO",
                        f"[DEBUG] Trial {trial_id} classified as: {failure_type} "
                        f"(loss={result.loss}, error={result.error[:50] if result.error else None})",
                    )
                    if retries < max_retries:
                        current_config = self.retry_strategy.modify_config_for_retry(
                            current_config, failure_type, retries + 1
                        )
                        retries += 1
                        self._log(
                            "WARNING",
                            f"Trial {trial_id} failed ({failure_type}), retrying...",
                        )
                        continue

                return result

            except Exception as e:
                self._log("ERROR", f"Trial {trial_id} exception: {e}")
                retries += 1
                if retries > max_retries:
                    return TrialResult(
                        trial_id=trial_id,
                        loss=float("inf"),
                        error=str(e),
                        hyperparams=current_config,
                    )

        return TrialResult(
            trial_id=trial_id,
            loss=float("inf"),
            error="Max retries exceeded",
            hyperparams=config,
        )

    async def _execute_trial_subprocess(self, trial_id: str, config: dict[str, Any]) -> TrialResult:
        """Execute trial as a subprocess.

        Args:
            trial_id: Unique trial identifier
            config: Trial configuration

        Returns:
            TrialResult from trial execution
        """
        start_time = time.time()

        # Write config file for trial runner
        trial_dir = self.storage_path / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_file = trial_dir / "config.json"

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        # VERBOSE: Print trial start info for debugging
        print(f"\n[HPO DEBUG] Starting trial {trial_id}")
        print(f"[HPO DEBUG] Config file: {config_file}")
        print(
            f"[HPO DEBUG] Key hyperparams: lr={config.get('learning_rate')}, "
            f"optimizer={config.get('optimizer')}, batch_size={config.get('batch_size')}"
        )
        print(f"[HPO DEBUG] Dataset: {config.get('hf_dataset_name', 'sample_texts')}")

        # Run trial subprocess with complete environment
        trial_runner_module = "highnoon.services.hpo_trial_runner"
        env = {
            **dict(__import__("os").environ.items()),
            "HPO_SWEEP_ID": self.sweep_id,
            "HPO_TRIAL_ID": trial_id,
            "HPO_TRIAL_DIR": str(trial_dir),  # Explicit trial directory
            "HPO_ROOT": str(self.storage_path),  # HPO artifacts root
            "HPO_API_HOST": "127.0.0.1:8000",
        }

        # Determine project root
        project_root = Path(__file__).parent.parent.parent

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            trial_runner_module,
            "--trial_id",
            trial_id,
            "--config",
            str(config_file),
            "--epochs",
            str(self.config.epochs_per_trial),
            "--steps_per_epoch",
            str(self.config.steps_per_epoch),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(project_root),
        )

        # Capture output
        stdout, stderr = await process.communicate()
        wall_time = time.time() - start_time

        # Parse result from status file
        status_file = trial_dir / "status.json"
        loss = float("inf")
        perplexity = None
        mean_confidence = None
        ece = None
        param_count = 0
        epochs_completed = 0
        error = None

        if status_file.exists():
            try:
                with open(status_file) as f:
                    status = json.load(f)
                    # HPOReporter writes 'best_loss', not 'loss'
                    loss = status.get("best_loss") or status.get("loss", float("inf"))
                    if loss is None:
                        loss = float("inf")
                    perplexity = status.get("perplexity")
                    mean_confidence = status.get("mean_confidence")
                    # HPOReporter writes 'expected_calibration_error', not 'ece'
                    ece = status.get("expected_calibration_error") or status.get("ece")
                    param_count = status.get("param_count", 0)
                    epochs_completed = status.get("epochs_completed", 0)
                    # CRITICAL: Read error from status.json (includes BUDGET_EXCEEDED)
                    error = status.get("error")

                    # DEBUG: Log parsed status for visibility
                    logger.info(
                        f"[HPO DEBUG] Parsed status.json for {trial_id}: "
                        f"loss={loss}, error={error[:80] if error else None}, "
                        f"param_count={param_count}"
                    )
            except Exception as e:
                error = f"Failed to read status: {e}"
                logger.error(f"[HPO DEBUG] Failed to parse status.json: {e}")

        # Read memory from status.json first (authoritative), fallback to metrics.jsonl
        memory_peak_mb = 0.0
        if status_file.exists():
            try:
                with open(status_file) as f:
                    status_data = json.load(f)
                    memory_peak_mb = status_data.get("memory_peak_mb", 0.0) or 0.0
            except Exception:
                pass

        # Fallback: read peak memory from metrics.jsonl if not in status
        if memory_peak_mb == 0.0:
            metrics_file = trial_dir / "metrics.jsonl"
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        for line in f:
                            if line.strip():
                                try:
                                    entry = json.loads(line)
                                    if "peak_memory_mb" in entry and entry["peak_memory_mb"]:
                                        memory_peak_mb = max(
                                            memory_peak_mb, entry["peak_memory_mb"]
                                        )
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.debug(f"[HPO] Could not read metrics.jsonl for memory: {e}")

        if process.returncode != 0 and not error:
            # Capture full error - don't truncate for debugging
            error = stderr.decode() if stderr else "Unknown error"

        # VERBOSE LOGGING: Print trial subprocess output to console for debugging
        print(f"\n{'='*80}")

        # Translate negative exit codes to signal names for better debugging
        exit_code_str = str(process.returncode)
        if process.returncode < 0:
            import signal

            sig_num = -process.returncode
            sig_name = (
                signal.Signals(sig_num).name
                if sig_num in signal.Signals._value2member_map_
                else f"SIGNAL_{sig_num}"
            )
            exit_code_str = f"{process.returncode} ({sig_name})"
            print(f"[HPO DEBUG] ⚠️  CRASH DETECTED: {sig_name} (signal {sig_num})")
            if sig_num == 11:  # SIGSEGV
                print("[HPO DEBUG] SIGSEGV typically indicates:")
                print("  - Null pointer dereference in C++ op")
                print("  - Out-of-bounds memory access")
                print("  - CUDA memory allocation failure in native code")
                print("  - Check stderr below for faulthandler stack trace")

        print(
            f"[HPO DEBUG] Trial {trial_id} completed in {wall_time:.1f}s (exit code: {exit_code_str})"
        )
        print(
            f"[HPO DEBUG] Config: learning_rate={config.get('learning_rate')}, "
            f"optimizer={config.get('optimizer')}, max_grad_norm={config.get('max_grad_norm')}"
        )
        print(
            f"[HPO DEBUG] Model: hidden_dim={config.get('hidden_dim')}, "
            f"blocks={config.get('num_reasoning_blocks')}, experts={config.get('num_moe_experts')}"
        )
        print(f"[HPO DEBUG] Loss: {loss}")
        if error:
            # Print full error, not truncated
            print("[HPO DEBUG] Error:")
            for line in error.split("\n"):
                if line.strip():
                    print(f"  [ERROR] {line}")

        if stdout:
            stdout_text = stdout.decode()
            print(f"\n[HPO DEBUG] === STDOUT ({len(stdout_text)} chars) ===")
            # Print all stdout lines
            for line in stdout_text.split("\n"):
                if line.strip():
                    print(f"  | {line}")

        if stderr:
            stderr_text = stderr.decode()
            print(f"\n[HPO DEBUG] === STDERR ({len(stderr_text)} chars) ===")
            # Print ALL stderr lines for debugging, not just last 50
            for line in stderr_text.split("\n"):
                if line.strip():
                    print(f"  ! {line}")

        print(f"{'='*80}\n")

        # Log trial output for debugging (for the WebUI logs)
        # Send more lines to WebUI for better visibility
        if stdout:
            for line in stdout.decode().split("\n")[-30:]:  # Last 30 lines (was 10)
                if line.strip():
                    self._log("INFO", line.strip(), trial_id=trial_id)

        # Also send errors to WebUI logs (so they appear in TrainingConsole)
        if error and process.returncode != 0:
            # Send full error to WebUI, split into lines for readability
            self._log(
                "ERROR",
                f"Trial {trial_id} failed (exit code {process.returncode})",
                trial_id=trial_id,
            )
            for line in error.split("\n")[-50:]:  # Last 50 lines of error
                if line.strip():
                    self._log("ERROR", line.strip(), trial_id=trial_id)

        result = TrialResult(
            trial_id=trial_id,
            loss=loss,
            perplexity=perplexity,
            mean_confidence=mean_confidence,
            ece=ece,
            param_count=param_count,
            hyperparams=config,
            epochs_completed=epochs_completed,
            wall_time_seconds=wall_time,
            memory_peak_mb=memory_peak_mb,
            error=error,
        )

        # Compute composite score
        result.composite_score = self.scorer.compute_score(result)

        # Cleanup memory
        gc.collect()

        return result

    async def _process_trial_result(self, result: TrialResult) -> None:
        """Process completed trial and update state.

        Args:
            result: Completed trial result
        """
        async with self._lock:
            # Remove from running (both task and config)
            self.running_trials.pop(result.trial_id, None)
            self.running_trial_configs.pop(result.trial_id, None)

            # Check for budget exceeded - don't count as a completed trial
            if result.error and result.error.startswith("BUDGET_EXCEEDED:"):
                self._log(
                    "INFO",
                    f"Trial {result.trial_id} skipped (budget exceeded), "
                    f"param_count={result.param_count}, notifying scheduler",
                )
                # CRITICAL: Inform scheduler of budget violation so c-TPE can learn
                # This enables the constrained acquisition to penalize oversized configs
                if hasattr(self.scheduler, "report_budget_violation"):
                    try:
                        self.scheduler.report_budget_violation(
                            config=result.hyperparams,
                            estimated_params=result.param_count,
                        )
                        self._log("INFO", "Scheduler notified of budget violation")
                    except Exception as e:
                        self._log("WARNING", f"Failed to report budget violation: {e}")
                # Request a replacement trial from the scheduler
                try:
                    queued = await self._queue_next_trials(1)
                    if queued > 0:
                        self._log("INFO", "Replacement trial queued")
                except Exception as e:
                    self._log("WARNING", f"Failed to queue replacement trial: {e}")
                return  # Don't count this trial

            # Add to completed
            self.completed_trials.append(result)

            # Report to scheduler for adaptive sampling
            # CRITICAL: Must call report_result(), NOT report_intermediate()!
            # - report_intermediate(): Only updates amplitude during trial execution
            # - report_result(): Updates trial history, triggers population evolution,
            #   enables convergence tracking, and quantum tunneling mechanisms
            try:
                from highnoon.services.hpo_schedulers import TrialResult as SchedulerTrialResult

                scheduler_result = SchedulerTrialResult(
                    trial_id=result.trial_id,
                    loss=result.loss,
                    step=result.epochs_completed,
                    memory_mb=result.memory_peak_mb,
                    wall_time_seconds=result.wall_time_seconds,
                    is_complete=True,
                )
                # Use report_result to trigger QAHPO population evolution
                self.scheduler.report_result(scheduler_result)
            except Exception as e:
                self._log("WARNING", f"Failed to report to scheduler: {e}")

            # Update Pareto frontier
            self.scorer.update_pareto_frontier(result)

            # Phase 201.6: Track Pareto-optimal trials for LRU protection
            if result.is_on_pareto_frontier:
                self._pareto_protected_ids.add(result.trial_id)

            # Phase 201.6: LRU eviction - keep memory bounded for large sweeps
            self._evict_old_trials_if_needed()

            # Update best trial
            if result.loss < (self.best_trial.loss if self.best_trial else float("inf")):
                self.best_trial = result
                self._log(
                    "INFO",
                    f"New best: {result.trial_id} with loss {result.loss:.6f}",
                )

            # Log progress
            self._log(
                "INFO",
                f"Trial {result.trial_id} completed: loss={result.loss:.6f}, "
                f"progress={len(self.completed_trials)}/{self.config.max_trials}",
            )

            # Periodic checkpoint
            if len(self.completed_trials) % self.config.fault_tolerance.checkpoint_frequency == 0:
                await self.checkpointer.save_checkpoint(self)

    def _evict_old_trials_if_needed(self) -> None:
        """Phase 201.6: Evict oldest non-essential trials if over memory limit.

        Protects:
        - Best trial (self.best_trial)
        - Pareto frontier trials (self._pareto_protected_ids)
        - Most recent N trials (where N = _max_in_memory_trials)

        Evicts oldest trials that don't meet protection criteria.
        """
        if len(self.completed_trials) <= self._max_in_memory_trials:
            return

        # How many to evict
        num_to_evict = len(self.completed_trials) - self._max_in_memory_trials

        # Build eviction candidates (oldest first, excluding protected)
        best_id = self.best_trial.trial_id if self.best_trial else None
        evictable = []

        for trial in self.completed_trials:
            if trial.trial_id == best_id:
                continue  # Protect best
            if trial.trial_id in self._pareto_protected_ids:
                continue  # Protect Pareto-optimal
            evictable.append(trial)

        # Evict oldest first (they're at the front of the list)
        to_evict = evictable[:num_to_evict]
        if to_evict:
            evict_ids = {t.trial_id for t in to_evict}
            self.completed_trials = [
                t for t in self.completed_trials if t.trial_id not in evict_ids
            ]
            self._evicted_trials_count += len(to_evict)
            logger.debug(
                f"[HPO] Evicted {len(to_evict)} old trials, "
                f"keeping {len(self.completed_trials)} in memory"
            )

    def _restore_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore state from checkpoint.

        Args:
            checkpoint: Checkpoint data dictionary
        """
        for trial_data in checkpoint.get("completed_trials", []):
            result = TrialResult(
                trial_id=trial_data["trial_id"],
                loss=trial_data.get("loss", float("inf")) or float("inf"),
                perplexity=trial_data.get("perplexity"),
                mean_confidence=trial_data.get("mean_confidence"),
                ece=trial_data.get("ece"),
                param_count=trial_data.get("param_count", 0),
                hyperparams=trial_data.get("hyperparams", {}),
                epochs_completed=trial_data.get("epochs_completed", 0),
                wall_time_seconds=trial_data.get("wall_time_seconds", 0),
                error=trial_data.get("error"),
                composite_score=trial_data.get("composite_score", 0),
            )
            self.completed_trials.append(result)
            self.scorer.update_pareto_frontier(result)

        if checkpoint.get("best_trial"):
            best_data = checkpoint["best_trial"]
            self.best_trial = TrialResult(
                trial_id=best_data["trial_id"],
                loss=best_data.get("loss", float("inf")) or float("inf"),
                perplexity=best_data.get("perplexity"),
                mean_confidence=best_data.get("mean_confidence"),
                ece=best_data.get("ece"),
                param_count=best_data.get("param_count", 0),
                hyperparams=best_data.get("hyperparams", {}),
            )

    def _build_sweep_result(self) -> SweepResult:
        """Build final sweep result.

        Returns:
            SweepResult summarizing the sweep
        """
        total_time = time.time() - self._start_time

        return SweepResult(
            sweep_id=self.sweep_id,
            completed_trials=self.completed_trials,
            best_trial=self.best_trial,
            best_trial_id=self.best_trial.trial_id if self.best_trial else None,
            best_loss=self.best_trial.loss if self.best_trial else None,
            best_hyperparams=(self.best_trial.hyperparams if self.best_trial else None),
            total_wall_time_seconds=total_time,
            state="completed",
            pareto_frontier=self.scorer.pareto_frontier,
        )

    async def cancel(self) -> None:
        """Cancel the running sweep."""
        self._log("WARNING", "Sweep cancellation requested")
        self._shutdown.set()

        # Cancel running trials
        for trial_id, task in self.running_trials.items():
            task.cancel()
            self._log("INFO", f"Cancelled trial {trial_id}")

        # Save checkpoint
        await self.checkpointer.save_checkpoint(self)
