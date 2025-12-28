"""Scheduler Factory for HPO.

Creates scheduler instances based on strategy string configuration.
Supports integration with existing schedulers and Optuna backend.

Copyright 2025 Verso Industries
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from highnoon.services.hpo_schedulers import HPOSchedulerBase

logger = logging.getLogger(__name__)


def sample_search_space(search_space: dict[str, Any] | None, trial_id: int) -> dict[str, Any]:
    """Sample hyperparameters from search space.

    Simple random sampling for use when no specific sampler is provided.

    Args:
        search_space: Search space definition
        trial_id: Trial ID for reproducibility

    Returns:
        Dictionary of sampled hyperparameters
    """
    import random

    if not search_space:
        return {}

    random.seed(trial_id)
    config: dict[str, Any] = {}

    for name, spec in search_space.items():
        spec_type = spec.get("type", "uniform")

        if spec_type == "uniform":
            config[name] = random.uniform(spec.get("low", 0), spec.get("high", 1))
        elif spec_type == "loguniform":
            import math

            log_low = math.log(spec.get("low", 1e-5))
            log_high = math.log(spec.get("high", 1e-1))
            config[name] = math.exp(random.uniform(log_low, log_high))
        elif spec_type == "int":
            config[name] = random.randint(spec.get("low", 1), spec.get("high", 10))
        elif spec_type == "categorical":
            config[name] = random.choice(spec.get("choices", [None]))
        else:
            # Default to first value or None
            config[name] = spec.get("default")

    return config


def create_scheduler(
    strategy: str,
    config: Any,  # SweepConfig
    search_space: dict[str, Any] | None = None,
) -> HPOSchedulerBase:
    """Create scheduler based on strategy string.

    As of Phase 11, all strategies default to QAHPO (Quantum Adaptive HPO).
    Legacy strategy names are supported for backward compatibility but
    route to QAHPO.

    Args:
        strategy: Scheduler strategy name. Recommended:
            - "quantum" / "qahpo": Quantum Adaptive HPO (default)
            Legacy (all route to QAHPO):
            - "random", "bayesian", "hyperband", "successive_halving", "pbt"
        config: SweepConfig with scheduler parameters
        search_space: Optional custom search space

    Returns:
        Configured QAHPO scheduler instance
    """
    from highnoon.services.hpo_manager import HPOSearchSpace
    from highnoon.services.quantum_hpo_scheduler import QAHPOConfig, QuantumAdaptiveHPOScheduler

    # Get model_config from sweep config (contains optimizer info for LR clamping)
    model_config = getattr(config, "model_config", None) or {}

    # Extract user-defined constraints from config
    param_budget = getattr(config, "param_budget", None) or model_config.get("param_budget")
    vocab_size = model_config.get("vocab_size")
    context_window = model_config.get("sequence_length") or model_config.get("context_window")

    # Extract optimizer from model_config to set in search space
    # This ensures HPOSearchSpace.sample() uses the user-selected optimizer(s)
    user_optimizer = model_config.get("optimizer")
    optimizer_list = [user_optimizer] if user_optimizer else None

    # Create HPOSearchSpace with user's budget constraint for proper enforcement
    # This ensures sampled configs respect param_budget, vocab_size, context_window, and optimizer
    hpo_search_space = HPOSearchSpace(
        vocab_size=vocab_size,
        context_window=context_window,
        param_budget=param_budget,
    )

    # Override optimizer list if user specified one
    if optimizer_list:
        hpo_search_space.optimizer = optimizer_list
        logger.info(f"[HPO] Using user-specified optimizer: {optimizer_list}")

    if param_budget:
        logger.info(
            f"[HPO] Scheduler using HPOSearchSpace with param_budget={param_budget / 1e6:.0f}M"
        )
    if context_window:
        logger.info(f"[HPO] Scheduler using context_window={context_window}")

    # Budget-aware sampler that uses HPOSearchSpace.sample() for proper constraint enforcement
    def budget_aware_sampler(trial_id: int) -> dict[str, Any]:
        # Start with model_config as base (includes optimizer, hidden_dim, dataset info, etc.)
        base_config = dict(model_config)

        # Preserve critical user-set parameters that should NOT be overwritten by sampling
        preserved_params = {
            k: v
            for k, v in base_config.items()
            if k in ("optimizer", "hf_dataset_name", "curriculum_id", "sweep_id") and v is not None
        }

        # Use HPOSearchSpace.sample() which enforces param_budget constraints
        # This will resample if initial config exceeds budget
        sampled = hpo_search_space.sample(trial_id)

        # Merge: sampled hyperparams override base config
        base_config.update(sampled)

        # Restore preserved user-set parameters after sampling (they take priority)
        base_config.update(preserved_params)

        # Overlay any custom search space params (these take highest priority)
        if search_space:
            custom_sampled = sample_search_space(search_space, trial_id)
            base_config.update(custom_sampled)

        return base_config

    max_budget = getattr(config, "max_epochs", 100)
    min_budget = getattr(config, "min_epochs", 10)
    population_size = getattr(config, "population_size", 8)

    # All strategies now route to QAHPO
    strategy_lower = strategy.lower()

    # Map legacy names to QAHPO with appropriate messages
    if strategy_lower in ("quantum", "qahpo", "adaptive"):
        logger.info("[HPO] Creating Quantum Adaptive HPO scheduler (QAHPO)")
    elif strategy_lower == "pbt":
        logger.info("[HPO] PBT → Using Quantum Adaptive HPO (QAHPO, quantum-enhanced PBT)")
    elif strategy_lower == "bayesian":
        logger.info("[HPO] Bayesian → Using Quantum Adaptive HPO (QAHPO with amplitude selection)")
    elif strategy_lower == "hyperband":
        logger.info("[HPO] Hyperband → Using Quantum Adaptive HPO (QAHPO with adaptive budgets)")
    elif strategy_lower == "successive_halving":
        logger.info("[HPO] Successive Halving → Using Quantum Adaptive HPO (QAHPO)")
    else:
        logger.info(f"[HPO] Strategy '{strategy}' → Using Quantum Adaptive HPO (QAHPO)")

    # Configure QAHPO
    qahpo_config = QAHPOConfig(
        population_size=population_size,
        initial_temperature=2.0,
        final_temperature=0.1,
        tunneling_probability=0.15,
        mutation_strength=0.3,
        crossover_rate=0.4,
        elite_fraction=0.25,
    )

    return QuantumAdaptiveHPOScheduler(
        max_budget=max_budget,
        min_budget=min_budget,
        search_space_sampler=budget_aware_sampler,
        config=qahpo_config,
    )
