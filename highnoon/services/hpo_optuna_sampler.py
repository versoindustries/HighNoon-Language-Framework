"""Optuna-based HPO Sampler for HighNoon Language Framework.

This module provides an Optuna-powered hyperparameter sampler that integrates
with the HSMN architecture search space while respecting Lite edition limits.

Features:
- TPE (Tree-structured Parzen Estimator) sampling
- CMA-ES for continuous hyperparameters
- Hyperband pruner integration
- Parameter budget constraint enforcement

Copyright 2025 Verso Industries
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lite Edition Limits
LITE_LIMITS = {
    "max_params": 20_000_000_000,  # 20B
    "max_vocab_size": 65536,
    "max_context_window": 5_000_000,
    "max_embedding_dim": 4096,
    "max_reasoning_blocks": 24,
    "max_moe_experts": 12,
    "max_superposition_dim": 2,
}

# HSMN default search space bounds
DEFAULT_SEARCH_SPACE = {
    # Training hyperparameters
    "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
    "batch_size": {"type": "categorical", "choices": [8, 16, 32, 64, 128]},
    "weight_decay": {"type": "uniform", "low": 0.0, "high": 0.1},
    "warmup_steps": {"type": "int", "low": 0, "high": 2000},
    # Architecture hyperparameters
    "num_reasoning_blocks": {"type": "categorical", "choices": [4, 6, 8, 12, 16, 20, 24]},
    "embedding_dim": {"type": "categorical", "choices": [256, 512, 768, 1024, 2048, 4096]},
    "num_moe_experts": {"type": "categorical", "choices": [4, 6, 8, 12]},
    "mamba_state_dim": {"type": "categorical", "choices": [32, 64, 128]},
    "moe_top_k": {"type": "categorical", "choices": [1, 2, 4]},
    "superposition_dim": {"type": "categorical", "choices": [1, 2]},
    # Optimizer selection
    "optimizer": {"type": "categorical", "choices": ["sophiag", "adamw", "adam"]},
}

# Check if optuna is available
try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner
    from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("[Optuna] optuna package not installed. Install with: pip install optuna>=3.5.0")


def estimate_model_params(config: dict[str, Any]) -> int:
    """Estimate total model parameters from architecture configuration.

    Uses the HSMN 6-block pattern to compute approximate parameter count.

    Args:
        config: Architecture configuration

    Returns:
        Estimated total parameter count
    """
    vocab_size = config.get("vocab_size", 32000)
    embedding_dim = config.get("embedding_dim", 512)
    num_blocks = config.get("num_reasoning_blocks", 8)
    num_experts = config.get("num_moe_experts", 8)
    mamba_state = config.get("mamba_state_dim", 64)

    # Embedding: vocab × embedding_dim
    embed_params = vocab_size * embedding_dim

    # Per-block estimates (HSMN 6-block pattern)
    ff_dim = embedding_dim * 4
    d_inner = embedding_dim * 2

    spatial_params = 4 * embedding_dim * d_inner + mamba_state * embedding_dim * 2
    timecrystal_params = 8 * embedding_dim * embedding_dim
    latent_params = 3 * embedding_dim * ff_dim
    wlam_params = 4 * embedding_dim * embedding_dim
    moe_params = num_experts * (2 * embedding_dim * ff_dim) + embedding_dim * num_experts

    blocks_per_pattern = 6
    num_patterns = (num_blocks + blocks_per_pattern - 1) // blocks_per_pattern
    params_per_pattern = (
        spatial_params * 2 + timecrystal_params + latent_params + wlam_params + moe_params
    )

    output_params = embedding_dim * vocab_size

    return int(embed_params + params_per_pattern * num_patterns + output_params)


class OptunaHPOSampler:
    """Optuna-based hyperparameter sampler for HSMN models.

    Provides advanced sampling strategies (TPE, CMA-ES) with:
    - Lite edition limit enforcement
    - Parameter budget pruning
    - Hyperband integration

    Example:
        >>> sampler = OptunaHPOSampler(param_budget=1_000_000_000)
        >>> config = sampler.sample(trial_id=0)
        >>> print(f"Sampled config: {config}")
    """

    def __init__(
        self,
        search_space: dict[str, dict[str, Any]] | None = None,
        sampler_type: str = "tpe",
        param_budget: int | None = None,
        vocab_size: int = 32000,
        context_window: int = 4096,
        study_name: str = "hsmn_hpo",
        storage: str | None = None,
        seed: int | None = None,
    ):
        """Initialize Optuna sampler.

        Args:
            search_space: Custom search space definition
            sampler_type: Sampler type ('tpe', 'cmaes', 'random')
            param_budget: Maximum parameter count (None = no limit)
            vocab_size: Fixed vocabulary size
            context_window: Fixed context window
            study_name: Optuna study name
            storage: Database URL for persistent storage
            seed: Random seed for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install with: pip install optuna>=3.5.0")

        self.search_space = search_space or DEFAULT_SEARCH_SPACE.copy()
        self.param_budget = param_budget or LITE_LIMITS["max_params"]
        self.vocab_size = min(vocab_size, LITE_LIMITS["max_vocab_size"])
        self.context_window = min(context_window, LITE_LIMITS["max_context_window"])
        self.seed = seed

        # Create sampler
        if sampler_type == "cmaes":
            self.sampler = CmaEsSampler(seed=seed)
        elif sampler_type == "random":
            self.sampler = RandomSampler(seed=seed)
        else:  # Default: TPE
            self.sampler = TPESampler(seed=seed, multivariate=True, constant_liar=True)

        # Create pruner (Hyperband-based)
        self.pruner = HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)

        # Create or load study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=self.sampler,
            pruner=self.pruner,
            direction="minimize",
            load_if_exists=True,
        )

        logger.info(
            f"[Optuna] Initialized: sampler={sampler_type}, param_budget={self.param_budget / 1e9:.1f}B"
        )

    def _suggest_param(self, trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
        """Suggest a parameter value based on spec.

        Args:
            trial: Optuna trial
            name: Parameter name
            spec: Parameter specification

        Returns:
            Suggested value
        """
        param_type = spec.get("type", "uniform")

        if param_type == "loguniform":
            return trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif param_type == "uniform":
            return trial.suggest_float(name, spec["low"], spec["high"])
        elif param_type == "int":
            return trial.suggest_int(name, spec["low"], spec["high"])
        elif param_type == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def _enforce_lite_limits(self, config: dict[str, Any]) -> dict[str, Any]:
        """Enforce Lite edition limits on configuration.

        Args:
            config: Sampled configuration

        Returns:
            Configuration with limits enforced
        """
        config["embedding_dim"] = min(
            config.get("embedding_dim", 512), LITE_LIMITS["max_embedding_dim"]
        )
        config["num_reasoning_blocks"] = min(
            config.get("num_reasoning_blocks", 8), LITE_LIMITS["max_reasoning_blocks"]
        )
        config["num_moe_experts"] = min(
            config.get("num_moe_experts", 8), LITE_LIMITS["max_moe_experts"]
        )
        config["superposition_dim"] = min(
            config.get("superposition_dim", 2), LITE_LIMITS["max_superposition_dim"]
        )
        return config

    def sample(self, trial_id: int) -> dict[str, Any]:
        """Sample a configuration from the search space.

        Args:
            trial_id: Trial identifier (for reproducibility)

        Returns:
            Sampled hyperparameter configuration
        """

        def objective(trial: optuna.Trial) -> float:
            """Dummy objective for sampling (actual training happens externally)."""
            config = {}

            # Sample all hyperparameters
            for name, spec in self.search_space.items():
                config[name] = self._suggest_param(trial, name, spec)

            # Add fixed parameters
            config["vocab_size"] = self.vocab_size
            config["context_window"] = self.context_window

            # Enforce Lite limits
            config = self._enforce_lite_limits(config)

            # Check parameter budget
            estimated_params = estimate_model_params(config)
            if estimated_params > self.param_budget:
                # Prune trials that exceed budget
                raise optuna.TrialPruned(
                    f"Estimated params {estimated_params / 1e9:.2f}B > budget {self.param_budget / 1e9:.2f}B"
                )

            # Store config for retrieval
            trial.set_user_attr("config", config)
            trial.set_user_attr("estimated_params", estimated_params)

            # Return dummy value (actual loss comes from training)
            return float("inf")

        # Run a single trial to sample
        try:
            trial = self.study.ask()  # Get next trial
            config = {}

            for name, spec in self.search_space.items():
                config[name] = self._suggest_param(trial, name, spec)

            config["vocab_size"] = self.vocab_size
            config["context_window"] = self.context_window
            config = self._enforce_lite_limits(config)

            # Check budget
            estimated_params = estimate_model_params(config)
            if estimated_params > self.param_budget:
                logger.warning(
                    f"[Optuna] Trial {trial_id} exceeds budget ({estimated_params / 1e9:.2f}B), "
                    "re-sampling with smaller config"
                )
                # Force smaller architecture
                config["num_reasoning_blocks"] = min(config["num_reasoning_blocks"], 8)
                config["num_moe_experts"] = min(config["num_moe_experts"], 6)
                config["embedding_dim"] = min(config["embedding_dim"], 768)

            logger.debug(
                f"[Optuna] Sampled trial {trial_id}: {list(config.keys())}, "
                f"~{estimate_model_params(config) / 1e6:.1f}M params"
            )

            return config

        except Exception as e:
            logger.error(f"[Optuna] Sampling failed: {e}")
            # Return default config on failure
            return {
                "learning_rate": 1e-4,
                "batch_size": 32,
                "weight_decay": 0.01,
                "warmup_steps": 100,
                "num_reasoning_blocks": 8,
                "embedding_dim": 512,
                "num_moe_experts": 8,
                "mamba_state_dim": 64,
                "moe_top_k": 2,
                "superposition_dim": 2,
                "optimizer": "sophiag",
                "vocab_size": self.vocab_size,
                "context_window": self.context_window,
            }

    def report(self, trial_id: int, step: int, loss: float) -> bool:
        """Report intermediate result and check for pruning.

        Args:
            trial_id: Trial identifier
            step: Current step/epoch
            loss: Current loss

        Returns:
            True if trial should be pruned, False otherwise
        """
        try:
            trial = self.study.trials[trial_id]
            trial.report(loss, step)
            return trial.should_prune()
        except (IndexError, KeyError):
            return False

    def complete(self, trial_id: int, final_loss: float, hyperparams: dict[str, Any]) -> None:
        """Mark a trial as complete.

        Args:
            trial_id: Trial identifier
            final_loss: Final loss value
            hyperparams: Hyperparameters used
        """
        try:
            trial = self.study.trials[trial_id]
            self.study.tell(trial, final_loss)
            logger.info(f"[Optuna] Trial {trial_id} completed with loss {final_loss:.6f}")
        except (IndexError, KeyError):
            logger.warning(f"[Optuna] Could not complete trial {trial_id}")

    def get_best_params(self) -> dict[str, Any] | None:
        """Get the best hyperparameters found so far.

        Returns:
            Best parameters or None if no trials completed
        """
        try:
            return self.study.best_params
        except ValueError:
            return None

    def get_best_trial(self) -> tuple[int, float] | None:
        """Get the best trial ID and loss.

        Returns:
            Tuple of (trial_number, best_loss) or None
        """
        try:
            best = self.study.best_trial
            return (best.number, best.value)
        except ValueError:
            return None


class OptunaPruner:
    """Custom Optuna pruner with parameter budget awareness.

    This pruner combines Hyperband-style resource allocation with
    parameter budget constraints specific to the HSMN architecture.
    """

    def __init__(
        self,
        param_budget: int = LITE_LIMITS["max_params"],
        min_resource: int = 1,
        max_resource: int = 100,
        reduction_factor: int = 3,
    ):
        """Initialize pruner.

        Args:
            param_budget: Maximum parameter count
            min_resource: Minimum resource allocation
            max_resource: Maximum resource allocation
            reduction_factor: Hyperband reduction factor
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed")

        self.param_budget = param_budget
        self.hyperband = HyperbandPruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor,
        )

    def prune(self, study: optuna.Study, trial: optuna.FrozenTrial) -> bool:
        """Determine if trial should be pruned.

        Args:
            study: Optuna study
            trial: Trial to evaluate

        Returns:
            True if should prune, False otherwise
        """
        # Check parameter budget first
        estimated_params = trial.user_attrs.get("estimated_params", 0)
        if estimated_params > self.param_budget:
            logger.info(f"[Optuna] Pruning trial {trial.number}: exceeds param budget")
            return True

        # Then check Hyperband pruning
        return self.hyperband.prune(study, trial)


def create_optuna_sampler(
    sampler_type: str = "tpe",
    param_budget: int | None = None,
    vocab_size: int = 32000,
    context_window: int = 4096,
    **kwargs: Any,
) -> OptunaHPOSampler:
    """Factory function to create Optuna sampler.

    Args:
        sampler_type: 'tpe', 'cmaes', or 'random'
        param_budget: Maximum parameters (20B default for Lite)
        vocab_size: Fixed vocabulary size
        context_window: Fixed context window
        **kwargs: Additional sampler parameters

    Returns:
        Configured OptunaHPOSampler

    Example:
        >>> sampler = create_optuna_sampler('tpe', param_budget=1_000_000_000)
        >>> config = sampler.sample(trial_id=0)
    """
    return OptunaHPOSampler(
        sampler_type=sampler_type,
        param_budget=param_budget,
        vocab_size=vocab_size,
        context_window=context_window,
        **kwargs,
    )


__all__ = [
    "OptunaHPOSampler",
    "OptunaPruner",
    "create_optuna_sampler",
    "estimate_model_params",
    "OPTUNA_AVAILABLE",
    "LITE_LIMITS",
    "DEFAULT_SEARCH_SPACE",
]
