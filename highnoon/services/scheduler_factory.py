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

    Args:
        strategy: Scheduler strategy name. One of:
            - "random": Simple random sampling
            - "bayesian": TPE via Optuna
            - "hyperband": Adaptive resource allocation
            - "successive_halving": Single-bracket early stopping
            - "pbt": Population-based training (not yet implemented)
        config: SweepConfig with scheduler parameters
        search_space: Optional custom search space

    Returns:
        Configured HPO scheduler instance

    Raises:
        ValueError: If unknown strategy specified
    """
    from highnoon.services.hpo_schedulers import HyperbandScheduler, SuccessiveHalvingScheduler

    # Default search space sampler
    def default_sampler(trial_id: int) -> dict[str, Any]:
        return sample_search_space(search_space, trial_id)

    max_budget = getattr(config, "max_epochs", 100)
    min_budget = getattr(config, "min_epochs", 1)
    eta = getattr(config, "hyperband_eta", 3)

    if strategy == "hyperband":
        logger.info(
            f"[HPO] Creating Hyperband scheduler (eta={eta}, "
            f"min_budget={min_budget}, max_budget={max_budget})"
        )
        return HyperbandScheduler(
            max_budget=max_budget,
            min_budget=min_budget,
            eta=eta,
            search_space_sampler=default_sampler,
        )

    elif strategy == "successive_halving":
        logger.info(
            f"[HPO] Creating Successive Halving scheduler "
            f"(min_budget={min_budget}, max_budget={max_budget})"
        )
        return SuccessiveHalvingScheduler(
            max_budget=max_budget,
            min_budget=min_budget,
            reduction_factor=eta,
            search_space_sampler=default_sampler,
        )

    elif strategy == "bayesian":
        # Try to use Optuna-based scheduler
        try:
            from highnoon.services.hpo_optuna_sampler import OPTUNA_AVAILABLE

            if OPTUNA_AVAILABLE:
                logger.info("[HPO] Creating Optuna Bayesian scheduler (TPE)")
                return _create_optuna_scheduler(config, search_space, default_sampler)
            else:
                logger.warning("[HPO] Optuna not available, falling back to random scheduler")
                return _create_random_scheduler(max_budget, default_sampler)
        except ImportError:
            logger.warning("[HPO] Optuna sampler not available, falling back to random scheduler")
            return _create_random_scheduler(max_budget, default_sampler)

    elif strategy == "pbt":
        logger.warning("[HPO] PBT scheduler not yet implemented, falling back to Hyperband")
        return HyperbandScheduler(
            max_budget=max_budget,
            min_budget=min_budget,
            eta=eta,
            search_space_sampler=default_sampler,
        )

    else:  # random or unknown
        if strategy != "random":
            logger.warning(f"[HPO] Unknown strategy '{strategy}', using random")
        return _create_random_scheduler(max_budget, default_sampler)


def _create_random_scheduler(
    max_budget: int,
    sampler: Callable[[int], dict[str, Any]],
) -> HPOSchedulerBase:
    """Create a simple random sampling scheduler.

    Args:
        max_budget: Maximum epochs per trial
        sampler: Hyperparameter sampling function

    Returns:
        Random scheduler instance
    """
    from highnoon.services.hpo_schedulers import HPOSchedulerBase, TrialConfig, TrialResult

    class RandomScheduler(HPOSchedulerBase):
        """Simple random sampling scheduler."""

        def __init__(
            self,
            max_budget: int,
            search_space_sampler: Callable[[int], dict[str, Any]],
        ):
            super().__init__(
                max_budget=max_budget,
                min_budget=1,
                search_space_sampler=search_space_sampler,
            )
            self._trial_counter = 0

        def get_trial_budget(self, trial_id: str) -> int:
            """Return max budget for all trials."""
            return self.max_budget

        def should_stop_trial(self, trial_id: str, current_loss: float) -> bool:
            """Never stop early for random scheduler."""
            return False

        def report_intermediate(self, result: TrialResult) -> None:
            """Record intermediate result (no-op for random)."""
            if result.trial_id not in self.results:
                self.results[result.trial_id] = []
            self.results[result.trial_id].append(result)

        def get_next_trials(self, n_trials: int) -> list[TrialConfig]:
            """Generate random trial configurations."""
            configs = []
            for _ in range(n_trials):
                trial_id = f"trial_{self._trial_counter}"
                hyperparams = self.search_space_sampler(self._trial_counter)
                configs.append(
                    TrialConfig(
                        trial_id=trial_id,
                        hyperparams=hyperparams,
                        budget=self.max_budget,
                    )
                )
                self._trial_counter += 1
            return configs

    logger.info(f"[HPO] Creating Random scheduler (max_budget={max_budget})")
    return RandomScheduler(max_budget=max_budget, search_space_sampler=sampler)


def _create_optuna_scheduler(
    config: Any,
    search_space: dict[str, Any] | None,
    default_sampler: Callable[[int], dict[str, Any]],
) -> HPOSchedulerBase:
    """Create Optuna-based Bayesian scheduler.

    Args:
        config: SweepConfig instance
        search_space: Custom search space
        default_sampler: Fallback sampler function

    Returns:
        Optuna-integrated scheduler
    """
    from highnoon.services.hpo_optuna_sampler import OptunaHPOSampler
    from highnoon.services.hpo_schedulers import HPOSchedulerBase, TrialConfig

    # Get parameters from config
    sweep_id = getattr(config, "sweep_id", "hpo_sweep")
    vocab_size = (
        config.model_config.get("vocab_size", 32000) if hasattr(config, "model_config") else 32000
    )
    param_budget = getattr(config, "param_budget", None)

    # Create Optuna sampler
    optuna_sampler = OptunaHPOSampler(
        search_space=search_space,
        sampler_type="tpe",
        param_budget=param_budget,
        vocab_size=vocab_size,
        study_name=f"hpo_{sweep_id}",
    )

    class OptunaBayesianScheduler(HPOSchedulerBase):
        """Optuna-powered Bayesian optimization scheduler."""

        def __init__(
            self,
            optuna_sampler: OptunaHPOSampler,
            max_budget: int,
        ):
            super().__init__(max_budget=max_budget)
            self.optuna_sampler = optuna_sampler
            self._trial_counter = 0

        def get_trial_budget(self, trial_id: str) -> int:
            """Return max budget for all trials."""
            return self.max_budget

        def should_stop_trial(self, trial_id: str, current_loss: float) -> bool:
            """Check if Optuna recommends pruning."""
            try:
                trial_num = int(trial_id.split("_")[-1])
                return self.optuna_sampler.report(trial_num, 0, current_loss)
            except (ValueError, AttributeError):
                return False

        def get_next_trials(self, n_trials: int) -> list[TrialConfig]:
            """Generate trials using Optuna TPE sampling."""
            configs = []
            for _ in range(n_trials):
                # Sample from Optuna
                hyperparams = self.optuna_sampler.sample(self._trial_counter)
                trial_id = f"trial_{self._trial_counter}"
                configs.append(
                    TrialConfig(
                        trial_id=trial_id,
                        hyperparams=hyperparams,
                        budget=self.max_budget,
                    )
                )
                self._trial_counter += 1
            return configs

        def report_intermediate(self, result: Any) -> None:
            """Report result to Optuna for adaptive sampling."""
            if result.trial_id not in self.results:
                self.results[result.trial_id] = []
            self.results[result.trial_id].append(result)
            try:
                trial_num = int(result.trial_id.split("_")[-1])
                if result.is_complete:
                    self.optuna_sampler.complete(trial_num, result.loss, {})
                else:
                    self.optuna_sampler.report(trial_num, result.step, result.loss)
            except (ValueError, AttributeError):
                pass

    max_budget = getattr(config, "max_epochs", 100)
    return OptunaBayesianScheduler(optuna_sampler, max_budget)
