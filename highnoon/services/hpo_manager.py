"""HPO Trial Manager for orchestrating hyperparameter optimization.

This module manages the lifecycle of HPO trials, including spawning,
monitoring, and coordinating with the C++ HPO orchestrator.

Phase 5 Update: Integrated grouped parameter search for multi-stage HPO.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from highnoon.services.hpo_metrics import HPOMetricsCollector, OversizedConfigTracker, TrialStatus
from highnoon.services.hpo_utils import convert_numpy_types

# Phase 5: Import grouped parameter sampler (TPE-style forest sampling)
try:
    from highnoon.services.hpo_grouped_sampling import ParameterGroupSampler

    GROUPED_SAMPLING_AVAILABLE = True
except ImportError as e:
    GROUPED_SAMPLING_AVAILABLE = False
    # Only log at debug level - not critical for operation
    logging.getLogger(__name__).debug(f"[HPO Manager] Grouped sampling not available: {e}")

logger = logging.getLogger(__name__)


def load_hpo_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load HPO configuration from JSON file.

    Args:
        config_path: Path to config file. Defaults to artifacts/hpo_trials/hpo_config.json

    Returns:
        Dictionary with configuration values
    """
    if config_path is None:
        config_path = Path("artifacts/hpo_trials/hpo_config.json")

    if not config_path.exists():
        logger.warning(f"[HPO Manager] Config file not found: {config_path}, using defaults")
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        logger.info(f"[HPO Manager] Loaded config from: {config_path}")
        return config
    except Exception as exc:
        logger.error(f"[HPO Manager] Failed to load config: {exc}")
        return {}


def estimate_model_params(config: dict[str, Any]) -> int:
    """Estimate total model parameters from architecture configuration.

    Uses the HSMN 6-block pattern (SpatialBlock, TimeCrystal, LatentReasoning,
    SpatialBlock, WLAM, MoE) to compute approximate parameter count.

    Args:
        config: Architecture configuration containing:
            - vocab_size: Vocabulary size (default 32000)
            - embedding_dim: Embedding dimension (default 512)
            - num_reasoning_blocks: Number of reasoning blocks (default 8)
            - num_moe_experts: Number of MoE experts (default 8)
            - mamba_state_dim: Mamba state dimension (default 64)

    Returns:
        Estimated total parameter count.

    Example:
        >>> config = {"vocab_size": 32000, "embedding_dim": 512,
        ...           "num_reasoning_blocks": 8, "num_moe_experts": 8}
        >>> params = estimate_model_params(config)
        >>> print(f"{params / 1e6:.1f}M parameters")
    """
    vocab_size = config.get("vocab_size", 32000)
    embedding_dim = config.get("embedding_dim", config.get("hidden_dim", 512))
    num_blocks = config.get("num_reasoning_blocks", 8)
    num_experts = config.get("num_moe_experts", 8)
    mamba_state = config.get("mamba_state_dim", 64)

    # Embedding: vocab × embedding_dim
    embed_params = vocab_size * embedding_dim

    # Per-block parameter estimates based on HSMN architecture
    ff_dim = embedding_dim * 4  # ff_expansion=4
    d_inner = embedding_dim * 2  # expand_factor=2

    # SpatialBlock (Mamba-2): input proj + conv + SSM params + output proj
    spatial_params = 4 * embedding_dim * d_inner + mamba_state * embedding_dim * 2

    # TimeCrystalSequenceBlock: HNN weights (W1, W2, W3) + output proj + VQC
    timecrystal_params = 8 * embedding_dim * embedding_dim

    # LatentReasoningBlock: FFN layers with expansion
    latent_params = 3 * embedding_dim * ff_dim

    # WLAMBlock: wavelet transforms + attention
    wlam_params = 4 * embedding_dim * embedding_dim

    # MoELayer: num_experts * (2 * d_model * ff_dim) + router
    moe_params = num_experts * (2 * embedding_dim * ff_dim) + embedding_dim * num_experts

    # Blocks per 6-block pattern (2 spatial blocks per pattern)
    blocks_per_pattern = 6
    num_patterns = (num_blocks + blocks_per_pattern - 1) // blocks_per_pattern
    params_per_pattern = (
        spatial_params * 2 + timecrystal_params + latent_params + wlam_params + moe_params
    )

    # Output layer (LM head): embedding_dim × vocab
    output_params = embedding_dim * vocab_size

    total = embed_params + params_per_pattern * num_patterns + output_params
    return int(total)


@dataclass
class HyperparameterConfig:
    """Configuration for a single hyperparameter."""

    name: str
    value: Any
    type: str  # "float", "int", "categorical"
    min_value: Any | None = None
    max_value: Any | None = None
    choices: list[Any] | None = None


@dataclass
class HPOSearchSpace:
    """Defines the search space for hyperparameter optimization.

    Lite Edition Limits (enforced by C++ binaries):
    - vocab_size: max 65536
    - context_window: max 5000000
    - embedding_dim: max 4096
    - num_reasoning_blocks: max 24
    - num_moe_experts: max 12

    Memory-Aware Shrinking:
    When memory_failure_count > 0, the sampler automatically reduces
    architecture sizes to prevent repeated OOM/swap failures.
    """

    # Training hyperparameters
    learning_rate: tuple[float, float] = (1e-5, 1e-2)
    batch_size: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    optimizer: list[str] = field(default_factory=lambda: ["sophiag", "adam"])
    warmup_steps: tuple[int, int] = (0, 1000)
    weight_decay: tuple[float, float] = (0.0, 0.1)

    # Model architecture params (Lite edition limits)
    num_reasoning_blocks: list[int] = field(default_factory=lambda: [4, 6, 8, 12, 16, 24])
    hidden_dim: list[int] = field(default_factory=lambda: [256, 512, 768, 1024])
    embedding_dim: list[int] = field(default_factory=lambda: [256, 512, 768, 1024, 2048, 4096])
    num_moe_experts: list[int] = field(default_factory=lambda: [4, 6, 8, 12])

    # HSMN-specific architecture params
    mamba_state_dim: list[int] = field(default_factory=lambda: [32, 64, 128])
    moe_top_k: list[int] = field(default_factory=lambda: [1, 2, 4])
    superposition_dim: list[int] = field(default_factory=lambda: [1, 2])  # Max 2 for Lite
    tt_rank_middle: list[int] = field(default_factory=lambda: [8, 16, 32])
    hamiltonian_hidden_dim: list[int] = field(default_factory=lambda: [128, 256, 512])

    # Tokenizer/context config (user-set, not tuned)
    vocab_size: int | None = None
    context_window: int | None = None
    position_embedding: str | None = None

    # Parameter budget constraint (user-set) - configs exceeding this are rejected
    param_budget: int | None = None
    lambda_da: tuple[float, float] | None = None  # Domain adaptation weight
    unfreeze_schedule: list[str] | None = None  # Unfreezing strategy
    unfreeze_interval: tuple[int, int] | None = None  # Steps between unfreezing

    # Memory-aware shrinking: count of consecutive memory failures
    # When > 0, sample() reduces architecture sizes to prevent OOM
    memory_failure_count: int = 0

    # Smart tuner: track skipped trials for WebUI and adaptive sampling
    _budget_tracker: OversizedConfigTracker | None = None

    def record_memory_failure(self) -> None:
        """Record a memory-related trial failure (OOM, swap spike, etc.).

        Increments memory_failure_count which causes subsequent samples to use
        smaller architectures to prevent repeated failures.
        """
        self.memory_failure_count += 1
        logger.warning(
            f"[HPO Manager] Memory failure recorded. "
            f"Shrink level now: {self.memory_failure_count}"
        )

    def record_trial_success(self) -> None:
        """Record a successful trial completion.

        Resets memory_failure_count since the current config fits in memory.
        """
        if self.memory_failure_count > 0:
            logger.info(
                f"[HPO Manager] Trial succeeded. Resetting memory shrink level "
                f"from {self.memory_failure_count} to 0"
            )
            self.memory_failure_count = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> HPOSearchSpace:
        """Create HPOSearchSpace from config dictionary.

        Phase 5 Update: Stores original config for grouped format detection.

        Args:
            config: Configuration dictionary with 'search_space' section

        Returns:
            HPOSearchSpace instance
        """
        search_space = config.get("search_space", {})

        # Phase 5: Check if config uses grouped format
        group_keys = ["architecture_groups", "training_groups", "control_groups"]
        is_grouped = any(key in config for key in group_keys)

        if is_grouped:
            # For grouped configs, create instance with defaults and store full config
            instance = cls()
            instance._grouped_config = config
            instance._is_grouped = True
            return instance

        # Legacy flat format handling
        learning_rate = search_space.get("learning_rate", {})
        lr_range = (learning_rate.get("min", 1e-5), learning_rate.get("max", 1e-2))

        batch_size = search_space.get("batch_size", {})
        batch_choices = batch_size.get("choices", [16, 32, 64, 128])

        optimizer = search_space.get("optimizer", {})
        opt_choices = optimizer.get("choices", ["sophiag", "adam"])

        warmup = search_space.get("warmup_steps", {})
        warmup_range = (warmup.get("min", 0), warmup.get("max", 1000))

        weight_decay = search_space.get("weight_decay", {})
        wd_range = (weight_decay.get("min", 0.0), weight_decay.get("max", 0.1))

        reasoning_blocks = search_space.get("num_reasoning_blocks", {})
        rb_choices = reasoning_blocks.get("choices", [4, 6, 8])

        hidden_dim = search_space.get("hidden_dim", {})
        hd_choices = hidden_dim.get("choices", [256, 512, 768])

        # Phase 3: Transfer learning / fine-tuning params
        lambda_da = search_space.get("lambda_da", {})
        lambda_da_range = None
        if lambda_da:
            lambda_da_range = (lambda_da.get("min", 0.001), lambda_da.get("max", 0.1))

        unfreeze_sched = search_space.get("unfreeze_schedule", {})
        unfreeze_sched_choices = unfreeze_sched.get("choices")

        unfreeze_int = search_space.get("unfreeze_interval", {})
        unfreeze_int_range = None
        if unfreeze_int:
            unfreeze_int_range = (unfreeze_int.get("min", 100), unfreeze_int.get("max", 1000))

        instance = cls(
            learning_rate=lr_range,
            batch_size=batch_choices,
            optimizer=opt_choices,
            warmup_steps=warmup_range,
            weight_decay=wd_range,
            num_reasoning_blocks=rb_choices,
            hidden_dim=hd_choices,
            lambda_da=lambda_da_range,
            unfreeze_schedule=unfreeze_sched_choices,
            unfreeze_interval=unfreeze_int_range,
        )
        instance._grouped_config = None
        instance._is_grouped = False
        return instance

    def sample(self, trial_id: int, stage: int = 1) -> dict[str, Any]:
        """Sample a configuration from the search space.

        Phase 5 Update: Uses grouped parameter search when available,
        supporting multi-stage sampling (COARSE → REFINE → FINE).

        Args:
            trial_id: Integer trial ID for deterministic sampling
            stage: Sampling stage (1=COARSE, 2=REFINE, 3=FINE)

        Returns:
            Dictionary of hyperparameter values
        """
        # Phase 5: Use grouped sampling if available and config is grouped
        if GROUPED_SAMPLING_AVAILABLE and getattr(self, "_is_grouped", False):
            sampler = ParameterGroupSampler(seed=trial_id)
            config = sampler.sample_grouped_configuration(self._grouped_config, trial_id, stage)

            # Apply any fixed parameters from the config
            if "fixed_parameters" in self._grouped_config:
                for key, value in self._grouped_config["fixed_parameters"].items():
                    if key != "comment":
                        config[key] = value

            logger.info(
                f"[HPO Manager] Sampled Stage {stage} config for trial {trial_id}: "
                f"keys={sorted(config.keys())}"
            )
            return config

        # Legacy flat sampling for backward compatibility
        import random

        random.seed(trial_id)  # Deterministic for reproducibility

        # Memory-aware shrinking: reduce architecture when previous trials failed due to OOM
        # Each memory failure decreases max allowed sizes to avoid repeated OOM
        memory_shrink_level = getattr(self, "memory_failure_count", 0)

        if memory_shrink_level >= 3:
            # Severe memory pressure: use minimum viable config
            max_blocks = 4
            max_experts = 4
            max_dim = 256
            max_batch = 8
            logger.warning(
                f"[HPO Manager] Memory shrink level {memory_shrink_level}: "
                "Using minimum viable config (4 blocks, 4 experts, 256 dim)"
            )
        elif memory_shrink_level >= 2:
            # High memory pressure: significantly reduced
            max_blocks = 6
            max_experts = 6
            max_dim = 512
            max_batch = 16
            logger.warning(
                f"[HPO Manager] Memory shrink level {memory_shrink_level}: "
                "Using reduced config (6 blocks, 6 experts, 512 dim)"
            )
        elif memory_shrink_level >= 1:
            # Moderate memory pressure: slightly reduced
            max_blocks = 8
            max_experts = 8
            max_dim = 768
            max_batch = 32
            logger.info(
                f"[HPO Manager] Memory shrink level {memory_shrink_level}: "
                "Using conservative config (8 blocks, 8 experts, 768 dim)"
            )
        else:
            # No memory pressure: use full search space
            max_blocks = 24
            max_experts = 12
            max_dim = 4096
            max_batch = 128

        # Filter search space by memory constraints
        filtered_blocks = [b for b in self.num_reasoning_blocks if b <= max_blocks] or [4]
        filtered_experts = [e for e in self.num_moe_experts if e <= max_experts] or [4]
        filtered_dims = [d for d in self.embedding_dim if d <= max_dim] or [256]
        filtered_batch = [b for b in self.batch_size if b <= max_batch] or [8]

        config = {
            # Training hyperparameters
            "learning_rate": random.uniform(*self.learning_rate),
            "batch_size": random.choice(filtered_batch),
            "optimizer": random.choice(self.optimizer),
            "warmup_steps": random.randint(*self.warmup_steps),
            "weight_decay": random.uniform(*self.weight_decay),
            # Model architecture params (memory-aware)
            "num_reasoning_blocks": random.choice(filtered_blocks),
            "hidden_dim": random.choice(filtered_dims),
            "embedding_dim": random.choice(filtered_dims),
            "num_moe_experts": random.choice(filtered_experts),
            # HSMN-specific params
            "mamba_state_dim": random.choice(self.mamba_state_dim),
            "moe_top_k": random.choice(self.moe_top_k),
            "superposition_dim": random.choice(self.superposition_dim),
            "tt_rank_middle": random.choice(self.tt_rank_middle),
            "hamiltonian_hidden_dim": random.choice(self.hamiltonian_hidden_dim),
        }

        # Add user-set tokenizer/context config if provided
        if self.vocab_size is not None:
            config["vocab_size"] = self.vocab_size
        if self.context_window is not None:
            config["context_window"] = self.context_window
        if self.position_embedding is not None:
            config["position_embedding"] = self.position_embedding

        # Phase 3: Add transfer learning params if configured
        if self.lambda_da is not None:
            config["lambda_da"] = random.uniform(*self.lambda_da)
        if self.unfreeze_schedule is not None:
            config["unfreeze_schedule"] = random.choice(self.unfreeze_schedule)
        if self.unfreeze_interval is not None:
            config["unfreeze_interval"] = random.randint(*self.unfreeze_interval)

        # Parameter budget constraint check with smart tuner integration
        if self.param_budget is not None:
            # Initialize budget tracker if not already done
            if self._budget_tracker is None:
                self._budget_tracker = OversizedConfigTracker(
                    param_budget=self.param_budget,
                )

            # Check initial estimate
            initial_estimated = estimate_model_params(config)
            max_attempts = 50  # Allow many attempts to find a fitting config
            attempt = 0

            # Log if initial config exceeds budget
            if initial_estimated > self.param_budget:
                logger.info(
                    f"[SmartTuner] Trial {trial_id}: initial config exceeds budget "
                    f"({initial_estimated / 1e6:.1f}M > {self.param_budget / 1e6:.1f}M), resampling..."
                )

            estimated_params = initial_estimated

            while estimated_params > self.param_budget and attempt < max_attempts:
                # Re-sample with progressively smaller architecture
                attempt += 1
                random.seed(trial_id + attempt * 1000)

                # Log each resample attempt
                logger.debug(
                    f"[SmartTuner] Trial {trial_id} attempt {attempt}: "
                    f"adjusting architecture to fit budget"
                )

                # Progressively tighten constraints based on attempt number
                if attempt < 15:
                    # Early attempts: bias toward medium configs
                    max_blocks = 12
                    max_experts = 8
                    max_dim = 1024
                elif attempt < 30:
                    # Middle attempts: smaller configs
                    max_blocks = 8
                    max_experts = 6
                    max_dim = 768
                else:
                    # Late attempts: minimum viable configs
                    max_blocks = 4
                    max_experts = 4
                    max_dim = 512

                smaller_blocks = [b for b in self.num_reasoning_blocks if b <= max_blocks]
                smaller_experts = [e for e in self.num_moe_experts if e <= max_experts]
                smaller_dims = [d for d in self.embedding_dim if d <= max_dim]

                if smaller_blocks:
                    config["num_reasoning_blocks"] = random.choice(smaller_blocks)
                if smaller_experts:
                    config["num_moe_experts"] = random.choice(smaller_experts)
                if smaller_dims:
                    config["embedding_dim"] = random.choice(smaller_dims)
                    config["hidden_dim"] = config["embedding_dim"]

                estimated_params = estimate_model_params(config)

            if estimated_params > self.param_budget:
                # Last resort: use absolute minimum config
                config["num_reasoning_blocks"] = min(self.num_reasoning_blocks)
                config["num_moe_experts"] = min(self.num_moe_experts)
                config["embedding_dim"] = min(self.embedding_dim)
                config["hidden_dim"] = config["embedding_dim"]
                estimated_params = estimate_model_params(config)

                if estimated_params > self.param_budget:
                    # Record this as a skipped configuration
                    self._budget_tracker.record_oversized(
                        trial_id=f"trial_{trial_id}",
                        config=config.copy(),
                        estimated_params=estimated_params,
                        reason=f"minimum_config_exceeds_budget_after_{max_attempts}_attempts",
                    )
                    logger.warning(
                        f"[SmartTuner] Trial {trial_id}: SKIPPED - even minimum config exceeds param_budget "
                        f"({estimated_params / 1e6:.1f}M > {self.param_budget / 1e6:.1f}M). "
                        f"Consider increasing budget or reducing vocab_size."
                    )
                else:
                    logger.info(
                        f"[SmartTuner] Trial {trial_id}: using minimum config to fit budget "
                        f"({estimated_params / 1e6:.1f}M <= {self.param_budget / 1e6:.1f}M) "
                        f"after {attempt} attempts"
                    )
            else:
                if attempt > 0:
                    logger.info(
                        f"[SmartTuner] Trial {trial_id}: config adjusted to fit budget "
                        f"({estimated_params / 1e6:.1f}M <= {self.param_budget / 1e6:.1f}M) "
                        f"after {attempt} attempts"
                    )
                else:
                    logger.debug(
                        f"[SmartTuner] Trial {trial_id}: config within budget "
                        f"({estimated_params / 1e6:.1f}M <= {self.param_budget / 1e6:.1f}M)"
                    )

        return config

    def get_skipped_trials(self) -> list[dict[str, Any]]:
        """Get list of trials that were skipped due to budget constraints.

        Returns:
            List of skipped trial records formatted for WebUI
        """
        if self._budget_tracker is None:
            return []
        return [r.to_dict() for r in self._budget_tracker.get_skipped_records()]

    def get_budget_statistics(self) -> dict[str, Any]:
        """Get statistics about parameter budget enforcement.

        Returns:
            Dictionary with budget enforcement statistics
        """
        if self._budget_tracker is None:
            return {"enabled": False}

        stats = self._budget_tracker.get_statistics()
        stats["enabled"] = True
        stats["param_budget"] = self.param_budget
        return stats


class HPOTrialManager:
    """Manages the lifecycle of HPO trials."""

    def __init__(
        self,
        search_space: HPOSearchSpace | None = None,
        metrics_collector: HPOMetricsCollector | None = None,
        max_trials: int = 10,
        hpo_binary: Path | None = None,
        config_path: Path | None = None,
    ):
        """Initialize the HPO trial manager.

        Args:
            search_space: Search space configuration
            metrics_collector: Metrics collector instance
            max_trials: Maximum number of trials to run
            hpo_binary: Path to the compiled HPO binary (hpo_main)
            config_path: Path to HPO config JSON file
        """
        # Load config from file if not provided
        self.config = load_hpo_config(config_path)

        # Use search space from config if not explicitly provided
        if search_space is None and self.config:
            self.search_space = HPOSearchSpace.from_config(self.config)
        else:
            self.search_space = search_space or HPOSearchSpace()

        self.metrics_collector = metrics_collector or HPOMetricsCollector()

        # Get max_trials from config if available
        hpo_runner = self.config.get("hpo_runner", {})
        trial_settings = hpo_runner.get("trial_settings", {})
        config_max_trials = trial_settings.get("max_trials", max_trials)
        self.max_trials = max_trials if max_trials != 10 else config_max_trials

        # Find the HPO binary
        if hpo_binary is None:
            # Try standard locations (in order of preference)
            import platform

            arch = platform.machine()
            if arch in ("x86_64", "AMD64"):
                arch = "x86_64"
            elif arch in ("aarch64", "arm64"):
                arch = "arm64"

            candidates = [
                # HighNoon native binary locations
                Path(__file__).parent.parent / "_native" / "bin" / arch / "_hpo_main.so",
                Path(__file__).parent.parent / "_native" / "bin" / arch / "hpo_main",
                Path(__file__).parent.parent / "_native" / "ops" / "_hpo_main.so",
                # Legacy HSMN locations
                Path("src/ops/_hpo_main.so"),
                Path("src/ops/_hpo_main.x86_64.so"),
                Path("_hpo_main.so"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    self.hpo_binary = candidate.resolve()
                    break
            else:
                # Don't fail - HPO can work without the binary in many cases
                self.hpo_binary = None
                logger.warning(
                    "[HPO Manager] HPO binary not found. Some features may be unavailable. "
                    "Build with: cd highnoon/_native && mkdir -p build && cd build && "
                    "cmake .. -DCMAKE_BUILD_TYPE=Release && make hpo_main"
                )
        else:
            self.hpo_binary = Path(hpo_binary).resolve()

        if self.hpo_binary:
            logger.info(f"[HPO Manager] Using binary: {self.hpo_binary}")
        logger.info(f"[HPO Manager] Max trials: {max_trials}")

    def write_trial_config(self, trial_id: str, config: dict[str, Any]) -> Path:
        """Write trial configuration to disk for the C++ orchestrator.

        Phase 5 Update: Converts numpy types to Python native types for JSON serialization.

        Args:
            trial_id: Trial identifier
            config: Hyperparameter configuration

        Returns:
            Path to the written config file
        """
        trial_dir = self.metrics_collector.get_trial_dir(trial_id)
        config_file = trial_dir / "config.json"

        # Convert numpy types to Python native types
        config_clean = convert_numpy_types(config)

        with config_file.open("w", encoding="utf-8") as f:
            json.dump(config_clean, f, indent=2)
            f.write("\n")

        logger.info(f"[HPO Manager] Wrote config for trial {trial_id}: {config_file}")
        return config_file

    def initialize_trial(self, trial_id: str, config: dict[str, Any]) -> None:
        """Initialize a trial with pending status.

        Args:
            trial_id: Trial identifier
            config: Hyperparameter configuration
        """
        # Compute estimated parameter count for efficiency scoring
        param_count = estimate_model_params(config)

        status = TrialStatus(
            trial_id=trial_id,
            status="pending",
            hyperparameters=config,
            param_count=param_count,
        )
        self.metrics_collector.update_trial_status(status)
        logger.info(
            f"[HPO Manager] Initialized trial {trial_id} (~{param_count / 1e6:.1f}M params)"
        )

    def start_trial(self, trial_id: str) -> None:
        """Mark a trial as running.

        Args:
            trial_id: Trial identifier
        """
        status = self.metrics_collector.load_trial_status(trial_id)
        if status:
            status.status = "running"
            status.start_time = time.time()
            self.metrics_collector.update_trial_status(status)
        logger.info(f"[HPO Manager] Started trial {trial_id}")

    def complete_trial(
        self, trial_id: str, success: bool = True, error_message: str | None = None
    ) -> None:
        """Mark a trial as completed or failed.

        Args:
            trial_id: Trial identifier
            success: Whether the trial completed successfully
            error_message: Error message if failed
        """
        status = self.metrics_collector.load_trial_status(trial_id)
        if status:
            status.status = "completed" if success else "failed"
            status.end_time = time.time()
            if error_message:
                status.error_message = error_message
            self.metrics_collector.update_trial_status(status)

        # Memory-aware shrinking: track memory failures to shrink future configs
        if not success and error_message:
            # Detect memory-related failures (swap spike, OOM, memory critical)
            memory_keywords = ["memory", "swap", "oom", "resourceexhausted"]
            is_memory_failure = any(kw in error_message.lower() for kw in memory_keywords)
            if is_memory_failure:
                self.search_space.record_memory_failure()
                logger.info(
                    f"[HPO Manager] Memory failure detected in trial {trial_id}, "
                    f"next trials will use smaller configs"
                )
        elif success:
            # Reset memory failure count on successful trial
            self.search_space.record_trial_success()

        logger.info(
            f"[HPO Manager] Trial {trial_id} {'completed' if success else 'failed'}"
            + (f": {error_message}" if error_message else "")
        )

    def generate_trials(self) -> list[tuple[str, dict[str, Any], int]]:
        """Generate trial configurations for the HPO sweep.

        Phase 5 Update: Returns (trial_id, config, stage) tuples for multi-stage HPO.

        Returns:
            List of (trial_id, config, stage) tuples
        """
        trials = []

        # Phase 5: Check if using grouped search with multi-stage settings
        is_grouped = getattr(self.search_space, "_is_grouped", False)
        if is_grouped and GROUPED_SAMPLING_AVAILABLE:
            # Multi-stage trial generation
            hpo_runner = self.config.get("hpo_runner", {})
            trial_settings = hpo_runner.get("trial_settings", {})
            max_stage1 = trial_settings.get("max_trials_stage1", 15)
            max_stage2 = trial_settings.get("max_trials_stage2", 25)
            max_stage3 = trial_settings.get("max_trials_stage3", 30)

            trial_idx = 0
            # Stage 1: COARSE
            for i in range(max_stage1):
                trial_id = f"trial_{trial_idx:04d}"
                config = self.search_space.sample(trial_idx, stage=1)
                trials.append((trial_id, config, 1))
                trial_idx += 1

            # Stage 2: REFINE
            for i in range(max_stage2):
                trial_id = f"trial_{trial_idx:04d}"
                config = self.search_space.sample(trial_idx, stage=2)
                trials.append((trial_id, config, 2))
                trial_idx += 1

            # Stage 3: FINE
            for i in range(max_stage3):
                trial_id = f"trial_{trial_idx:04d}"
                config = self.search_space.sample(trial_idx, stage=3)
                trials.append((trial_id, config, 3))
                trial_idx += 1

            logger.info(
                f"[HPO Manager] Generated {len(trials)} multi-stage trial configs: "
                f"Stage1={max_stage1}, Stage2={max_stage2}, Stage3={max_stage3}"
            )
        else:
            # Legacy flat search: all trials are stage 1
            for i in range(self.max_trials):
                trial_id = f"trial_{i:04d}"
                config = self.search_space.sample(i, stage=1)
                trials.append((trial_id, config, 1))

            logger.info(f"[HPO Manager] Generated {len(trials)} trial configurations")

        return trials

    def prepare_sweep(self) -> Path:
        """Prepare a full HPO sweep by writing all trial configs.

        Returns:
            Path to the sweep configuration file
        """
        trials = self.generate_trials()

        # Write individual trial configs
        for trial_entry in trials:
            if len(trial_entry) == 3:
                trial_id, config, stage = trial_entry
            else:
                # Backward compatibility
                trial_id, config = trial_entry

            self.write_trial_config(trial_id, config)
            self.initialize_trial(trial_id, config)

        # Write master sweep config
        is_grouped = getattr(self.search_space, "_is_grouped", False)

        if is_grouped and GROUPED_SAMPLING_AVAILABLE:
            # Phase 5: Multi-stage sweep config with stage metadata
            sweep_config = {
                "max_trials": len(trials),
                "multi_stage": True,
                "search_space_type": "grouped",
                "grouped_config": self.search_space._grouped_config,
                "trials": [
                    {
                        "trial_id": tid,
                        "config": cfg,
                        "stage": stage,
                        "stage_name": ["COARSE", "REFINE", "FINE"][stage - 1],
                    }
                    for tid, cfg, stage in trials
                ],
                "created_at": time.time(),
            }
        else:
            # Legacy flat sweep config
            sweep_config = {
                "max_trials": self.max_trials,
                "multi_stage": False,
                "search_space_type": "flat",
                "search_space": {
                    "learning_rate": self.search_space.learning_rate,
                    "batch_size": self.search_space.batch_size,
                    "optimizer": self.search_space.optimizer,
                    "warmup_steps": self.search_space.warmup_steps,
                    "weight_decay": self.search_space.weight_decay,
                    "num_reasoning_blocks": self.search_space.num_reasoning_blocks,
                    "hidden_dim": self.search_space.hidden_dim,
                },
                "trials": [
                    {"trial_id": tid, "config": cfg, "stage": stage} for tid, cfg, stage in trials
                ],
                "created_at": time.time(),
            }

        sweep_file = self.metrics_collector.hpo_root / "sweep_config.json"
        with sweep_file.open("w", encoding="utf-8") as f:
            # Convert numpy types to Python native types for JSON serialization
            sweep_config_clean = convert_numpy_types(sweep_config)
            json.dump(sweep_config_clean, f, indent=2)
            f.write("\n")

        logger.info(f"[HPO Manager] Prepared sweep config: {sweep_file}")
        return sweep_file

    def get_trial_env(self, trial_id: str) -> dict[str, str]:
        """Get environment variables for a trial.

        Args:
            trial_id: Trial identifier

        Returns:
            Dictionary of environment variables
        """
        env = os.environ.copy()
        env["HPO_TRIAL_ID"] = trial_id
        env["HPO_TRIAL_DIR"] = str(self.metrics_collector.get_trial_dir(trial_id))
        return env

    def monitor_trials(self) -> dict[str, Any]:
        """Monitor all active trials and return their status.

        Returns:
            Dictionary with trial statuses and progress
        """
        trials = self.metrics_collector.list_trials()

        running_count = 0
        completed_count = 0
        failed_count = 0

        trial_statuses = []
        for trial_id in trials:
            status = self.metrics_collector.load_trial_status(trial_id)
            if status:
                trial_statuses.append(status.to_dict())

                if status.status == "running":
                    running_count += 1
                elif status.status == "completed":
                    completed_count += 1
                elif status.status == "failed":
                    failed_count += 1

        best = self.metrics_collector.get_best_trial()

        return {
            "total_trials": len(trials),
            "running": running_count,
            "completed": completed_count,
            "failed": failed_count,
            "trials": trial_statuses,
            "best_trial": (
                {
                    "trial_id": best[0],
                    "best_loss": best[1],
                }
                if best
                else None
            ),
        }
