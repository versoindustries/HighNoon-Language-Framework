"""HPO Schedulers for HighNoon Language Framework.

This module implements enterprise-grade hyperparameter optimization schedulers:
- Hyperband: Adaptive resource allocation with successive halving
- Successive Halving: Single-bracket early stopping
- Population-Based Training (PBT): Evolutionary hyperparameter optimization

All schedulers respect Lite edition limits (20B parameters max).

Copyright 2025 Verso Industries
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lite Edition Limits
LITE_MAX_PARAMS = 20_000_000_000  # 20B


@dataclass
class TrialConfig:
    """Configuration for a single HPO trial."""

    trial_id: str
    hyperparams: dict[str, Any]
    budget: int  # Resource budget (epochs or steps)
    bracket: int = 0  # Hyperband bracket index
    rung: int = 0  # Current rung within bracket
    stage: int = 1  # HPO stage (1=COARSE, 2=REFINE, 3=FINE)


@dataclass
class TrialResult:
    """Result from a completed or intermediate trial report."""

    trial_id: str
    loss: float
    step: int
    memory_mb: float = 0.0
    wall_time_seconds: float = 0.0
    is_complete: bool = False


class HPOSchedulerBase(ABC):
    """Abstract base class for HPO schedulers.

    All schedulers must implement:
    - get_trial_budget(): Resource allocation for a trial
    - should_stop_trial(): Early stopping decision
    - report_intermediate(): Handle intermediate results
    - get_next_trials(): Generate next batch of trial configs
    """

    def __init__(
        self,
        max_budget: int = 100,
        min_budget: int = 1,
        search_space_sampler: Callable[[int], dict[str, Any]] | None = None,
    ):
        """Initialize scheduler.

        Args:
            max_budget: Maximum resource budget per trial (epochs)
            min_budget: Minimum resource budget per trial
            search_space_sampler: Function to sample hyperparameters given trial_id
        """
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.search_space_sampler = search_space_sampler or (lambda _: {})
        self.trials: dict[str, TrialConfig] = {}
        self.results: dict[str, list[TrialResult]] = {}

    @abstractmethod
    def get_trial_budget(self, trial_id: str) -> int:
        """Get the resource budget for a specific trial.

        Args:
            trial_id: Trial identifier

        Returns:
            Budget in epochs/steps
        """
        pass

    @abstractmethod
    def should_stop_trial(self, trial_id: str, current_loss: float) -> bool:
        """Determine if a trial should be stopped early.

        Args:
            trial_id: Trial identifier
            current_loss: Current loss value

        Returns:
            True if trial should stop, False to continue
        """
        pass

    @abstractmethod
    def report_intermediate(self, result: TrialResult) -> None:
        """Report intermediate results during training.

        Args:
            result: Intermediate trial result
        """
        pass

    @abstractmethod
    def get_next_trials(self, n_trials: int) -> list[TrialConfig]:
        """Generate the next batch of trial configurations.

        Args:
            n_trials: Number of trials to generate

        Returns:
            List of trial configurations
        """
        pass

    def get_best_trial(self) -> tuple[str, float] | None:
        """Get the best trial so far.

        Returns:
            Tuple of (trial_id, best_loss) or None if no results
        """
        best_trial = None
        best_loss = float("inf")

        for trial_id, results in self.results.items():
            for result in results:
                if result.loss < best_loss:
                    best_loss = result.loss
                    best_trial = trial_id

        if best_trial:
            return (best_trial, best_loss)
        return None


class HyperbandScheduler(HPOSchedulerBase):
    """Hyperband scheduler with adaptive resource allocation.

    Hyperband allocates resources adaptively using successive halving
    across multiple brackets. Each bracket starts with a different
    number of configurations and resource budget ratio.

    Reference: Li et al., "Hyperband: A Novel Bandit-Based Approach
    to Hyperparameter Optimization" (JMLR 2018)
    """

    def __init__(
        self,
        max_budget: int = 100,
        min_budget: int = 1,
        eta: int = 3,
        search_space_sampler: Callable[[int], dict[str, Any]] | None = None,
    ):
        """Initialize Hyperband scheduler.

        Args:
            max_budget: Maximum resource budget (R in paper)
            min_budget: Minimum resource budget (r in paper)
            eta: Reduction factor (default 3)
            search_space_sampler: Hyperparameter sampling function
        """
        super().__init__(max_budget, min_budget, search_space_sampler)
        self.eta = eta

        # Calculate s_max and B based on eta and budgets
        self.s_max = int(math.floor(math.log(max_budget / min_budget, eta)))
        self.B = (self.s_max + 1) * max_budget  # Total budget per bracket

        # Track brackets and rungs
        self.brackets: list[dict[str, Any]] = []
        self.current_bracket = 0
        self.trial_counter = 0

        logger.info(
            f"[Hyperband] Initialized: eta={eta}, s_max={self.s_max}, "
            f"max_budget={max_budget}, min_budget={min_budget}"
        )

    def _init_bracket(self, s: int) -> dict[str, Any]:
        """Initialize a Hyperband bracket.

        Args:
            s: Bracket index (0 to s_max)

        Returns:
            Bracket configuration
        """
        # Number of configurations in first rung
        n = int(math.ceil(self.B / self.max_budget * (self.eta**s) / (s + 1)))
        # Initial budget per configuration
        r = self.max_budget * (self.eta ** (-s))

        bracket = {
            "s": s,
            "n_configs": n,
            "initial_budget": max(self.min_budget, int(r)),
            "rungs": [],
            "trials": {},
        }

        # Pre-compute rung configurations
        for i in range(s + 1):
            n_i = int(math.floor(n * (self.eta ** (-i))))
            r_i = int(r * (self.eta**i))
            bracket["rungs"].append({"n_configs": max(1, n_i), "budget": min(r_i, self.max_budget)})

        return bracket

    def get_trial_budget(self, trial_id: str) -> int:
        """Get budget for a trial based on its bracket and rung."""
        if trial_id not in self.trials:
            return self.max_budget

        trial = self.trials[trial_id]
        bracket = self.brackets[trial.bracket]
        return bracket["rungs"][trial.rung]["budget"]

    def should_stop_trial(self, trial_id: str, current_loss: float) -> bool:
        """Determine if trial should be stopped (not promoted to next rung)."""
        if trial_id not in self.trials:
            return False

        trial = self.trials[trial_id]
        bracket = self.brackets[trial.bracket]

        # If at max rung, don't stop
        if trial.rung >= len(bracket["rungs"]) - 1:
            return False

        # Check if we have enough results at current rung to decide
        current_rung_budget = bracket["rungs"][trial.rung]["budget"]
        trial_results = self.results.get(trial_id, [])

        completed_budget = sum(1 for r in trial_results if r.step >= current_rung_budget)
        if completed_budget == 0:
            return False

        # Compare against other trials at same rung
        rung_results = []
        for tid, tcfg in self.trials.items():
            if tcfg.bracket == trial.bracket and tcfg.rung == trial.rung:
                if tid in self.results and self.results[tid]:
                    best = min(r.loss for r in self.results[tid])
                    rung_results.append((tid, best))

        if not rung_results:
            return False

        # Sort by loss and check if in top 1/eta
        rung_results.sort(key=lambda x: x[1])
        n_promote = max(1, len(rung_results) // self.eta)

        for _i, (tid, _) in enumerate(rung_results[:n_promote]):
            if tid == trial_id:
                return False  # Trial should continue

        return True  # Trial should stop

    def report_intermediate(self, result: TrialResult) -> None:
        """Record intermediate result and potentially promote trial."""
        if result.trial_id not in self.results:
            self.results[result.trial_id] = []
        self.results[result.trial_id].append(result)

        # Check for rung completion and promotion
        if result.trial_id in self.trials:
            trial = self.trials[result.trial_id]
            bracket = self.brackets[trial.bracket]
            current_budget = bracket["rungs"][trial.rung]["budget"]

            if result.step >= current_budget and not self.should_stop_trial(
                result.trial_id, result.loss
            ):
                # Promote to next rung
                if trial.rung < len(bracket["rungs"]) - 1:
                    trial.rung += 1
                    trial.budget = bracket["rungs"][trial.rung]["budget"]
                    logger.info(
                        f"[Hyperband] Promoted {result.trial_id} to rung {trial.rung}, "
                        f"new budget={trial.budget}"
                    )

    def get_next_trials(self, n_trials: int) -> list[TrialConfig]:
        """Generate next batch of trial configs using Hyperband logic."""
        trials = []

        # Initialize brackets if needed
        while len(self.brackets) <= self.s_max:
            self.brackets.append(self._init_bracket(len(self.brackets)))

        # Generate trials for each bracket
        for b_idx, bracket in enumerate(self.brackets):
            if len(bracket["trials"]) >= bracket["n_configs"]:
                continue  # Bracket is full

            remaining = bracket["n_configs"] - len(bracket["trials"])
            for _ in range(min(remaining, n_trials - len(trials))):
                trial_id = f"hb_b{b_idx}_t{self.trial_counter}"
                self.trial_counter += 1

                hyperparams = self.search_space_sampler(self.trial_counter)
                config = TrialConfig(
                    trial_id=trial_id,
                    hyperparams=hyperparams,
                    budget=bracket["initial_budget"],
                    bracket=b_idx,
                    rung=0,
                )
                self.trials[trial_id] = config
                bracket["trials"][trial_id] = config
                trials.append(config)

                if len(trials) >= n_trials:
                    break

            if len(trials) >= n_trials:
                break

        logger.info(f"[Hyperband] Generated {len(trials)} trial configs")
        return trials


class SuccessiveHalvingScheduler(HPOSchedulerBase):
    """Successive Halving scheduler (single-bracket Hyperband).

    Simpler than full Hyperband, runs a single bracket with
    successive halving elimination based on intermediate performance.

    Good for scenarios where the maximum budget is known in advance.
    """

    def __init__(
        self,
        max_budget: int = 100,
        min_budget: int = 1,
        eta: int = 3,
        n_configs: int = 27,
        search_space_sampler: Callable[[int], dict[str, Any]] | None = None,
    ):
        """Initialize Successive Halving scheduler.

        Args:
            max_budget: Maximum resource budget per trial
            min_budget: Minimum resource budget per trial
            eta: Reduction factor (default 3)
            n_configs: Initial number of configurations
            search_space_sampler: Hyperparameter sampling function
        """
        super().__init__(max_budget, min_budget, search_space_sampler)
        self.eta = eta
        self.n_configs = n_configs

        # Calculate number of rungs
        self.n_rungs = int(math.ceil(math.log(n_configs, eta))) + 1
        self.current_rung = 0
        self.rung_trials: list[list[str]] = [[] for _ in range(self.n_rungs)]
        self.trial_counter = 0

        # Compute budgets per rung
        self.rung_budgets = []
        budget = min_budget
        for _ in range(self.n_rungs):
            self.rung_budgets.append(min(budget, max_budget))
            budget *= eta

        logger.info(
            f"[SuccessiveHalving] Initialized: eta={eta}, n_configs={n_configs}, "
            f"n_rungs={self.n_rungs}, budgets={self.rung_budgets}"
        )

    def get_trial_budget(self, trial_id: str) -> int:
        """Get budget for trial based on its rung."""
        if trial_id not in self.trials:
            return self.rung_budgets[0]
        return self.rung_budgets[self.trials[trial_id].rung]

    def should_stop_trial(self, trial_id: str, current_loss: float) -> bool:
        """Check if trial should be stopped at current rung."""
        if trial_id not in self.trials:
            return False

        trial = self.trials[trial_id]

        # Collect results for current rung
        rung_results = []
        for tid in self.rung_trials[trial.rung]:
            if tid in self.results and self.results[tid]:
                best_loss = min(r.loss for r in self.results[tid])
                rung_results.append((tid, best_loss))

        if len(rung_results) < len(self.rung_trials[trial.rung]):
            return False  # Wait for all trials

        # Keep top 1/eta configurations
        rung_results.sort(key=lambda x: x[1])
        n_keep = max(1, len(rung_results) // self.eta)
        survivors = {tid for tid, _ in rung_results[:n_keep]}

        return trial_id not in survivors

    def report_intermediate(self, result: TrialResult) -> None:
        """Record result and check for rung completion."""
        if result.trial_id not in self.results:
            self.results[result.trial_id] = []
        self.results[result.trial_id].append(result)

    def get_next_trials(self, n_trials: int) -> list[TrialConfig]:
        """Generate trials for the current rung."""
        trials = []

        # First rung: generate initial configurations
        if self.current_rung == 0 and not self.rung_trials[0]:
            for _i in range(min(self.n_configs, n_trials)):
                trial_id = f"sha_r0_t{self.trial_counter}"
                self.trial_counter += 1

                hyperparams = self.search_space_sampler(self.trial_counter)
                config = TrialConfig(
                    trial_id=trial_id,
                    hyperparams=hyperparams,
                    budget=self.rung_budgets[0],
                    rung=0,
                )
                self.trials[trial_id] = config
                self.rung_trials[0].append(trial_id)
                trials.append(config)

        # Subsequent rungs: promote survivors
        elif self.current_rung > 0:
            prev_rung = self.current_rung - 1
            if len(self.rung_trials[prev_rung]) > 0:
                # Get survivors from previous rung
                rung_results = []
                for tid in self.rung_trials[prev_rung]:
                    if tid in self.results and self.results[tid]:
                        best_loss = min(r.loss for r in self.results[tid])
                        rung_results.append((tid, best_loss))

                rung_results.sort(key=lambda x: x[1])
                n_keep = max(1, len(rung_results) // self.eta)

                for tid, _ in rung_results[:n_keep]:
                    if tid not in [t.trial_id for t in trials]:
                        trial = self.trials[tid]
                        trial.rung = self.current_rung
                        trial.budget = self.rung_budgets[self.current_rung]
                        self.rung_trials[self.current_rung].append(tid)
                        trials.append(trial)

        logger.info(
            f"[SuccessiveHalving] Generated {len(trials)} trials for rung {self.current_rung}"
        )
        return trials

    def advance_rung(self) -> bool:
        """Advance to the next rung if current rung is complete.

        Returns:
            True if advanced, False if no more rungs
        """
        if self.current_rung >= self.n_rungs - 1:
            return False
        self.current_rung += 1
        logger.info(f"[SuccessiveHalving] Advanced to rung {self.current_rung}")
        return True


@dataclass
class PBTMember:
    """A member of the PBT population."""

    trial_id: str
    hyperparams: dict[str, Any]
    score: float = float("inf")
    checkpoint_path: Path | None = None
    generation: int = 0


class PopulationBasedTraining(HPOSchedulerBase):
    """Population-Based Training scheduler.

    PBT maintains a population of models trained in parallel.
    Periodically, underperforming members exploit better members
    by copying their weights and exploring new hyperparameters.

    Reference: Jaderberg et al., "Population Based Training of
    Neural Networks" (arXiv 2017)

    Note: This is a single-node implementation suitable for the
    Lite edition CPU-focused training.
    """

    def __init__(
        self,
        max_budget: int = 100,
        min_budget: int = 1,
        population_size: int = 8,
        exploit_fraction: float = 0.2,
        perturb_factor: float = 0.2,
        search_space_sampler: Callable[[int], dict[str, Any]] | None = None,
        checkpoint_dir: Path | None = None,
    ):
        """Initialize PBT scheduler.

        Args:
            max_budget: Maximum epochs per trial
            min_budget: Minimum epochs per trial
            population_size: Number of members in population
            exploit_fraction: Bottom fraction to exploit (default 20%)
            perturb_factor: Factor for hyperparameter perturbation (±20%)
            search_space_sampler: Hyperparameter sampling function
            checkpoint_dir: Directory to store checkpoints
        """
        super().__init__(max_budget, min_budget, search_space_sampler)
        self.population_size = population_size
        self.exploit_fraction = exploit_fraction
        self.perturb_factor = perturb_factor
        self.checkpoint_dir = checkpoint_dir or Path("artifacts/pbt_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.population: list[PBTMember] = []
        self.generation = 0
        self.trial_counter = 0
        self.eval_interval = 10  # Epochs between exploit/explore

        logger.info(
            f"[PBT] Initialized: population={population_size}, "
            f"exploit_fraction={exploit_fraction}, perturb_factor={perturb_factor}"
        )

    def _perturb_hyperparams(self, hyperparams: dict[str, Any]) -> dict[str, Any]:
        """Perturb hyperparameters for exploration.

        Args:
            hyperparams: Original hyperparameters

        Returns:
            Perturbed hyperparameters
        """
        perturbed = {}
        for key, value in hyperparams.items():
            if isinstance(value, (int, float)):
                # Randomly multiply by 0.8 or 1.2
                factor = random.choice([1 - self.perturb_factor, 1 + self.perturb_factor])
                new_value = value * factor
                if isinstance(value, int):
                    new_value = max(1, int(round(new_value)))
                perturbed[key] = new_value
            else:
                perturbed[key] = value
        return perturbed

    def get_trial_budget(self, trial_id: str) -> int:
        """PBT trials run for the full budget."""
        return self.max_budget

    def should_stop_trial(self, trial_id: str, current_loss: float) -> bool:
        """PBT trials don't stop early - they exploit/explore instead."""
        return False

    def report_intermediate(self, result: TrialResult) -> None:
        """Report result and potentially trigger exploit/explore."""
        if result.trial_id not in self.results:
            self.results[result.trial_id] = []
        self.results[result.trial_id].append(result)

        # Update member score
        for member in self.population:
            if member.trial_id == result.trial_id:
                member.score = result.loss
                break

        # Check if it's time for exploit/explore
        if result.step > 0 and result.step % self.eval_interval == 0:
            self._exploit_explore()

    def _exploit_explore(self) -> None:
        """Perform exploit/explore step for underperforming members."""
        if len(self.population) < 2:
            return

        # Sort population by score (lower is better)
        sorted_pop = sorted(self.population, key=lambda m: m.score)
        n_exploit = max(1, int(len(sorted_pop) * self.exploit_fraction))

        # Bottom performers exploit top performers
        bottom = sorted_pop[-n_exploit:]
        top = sorted_pop[:n_exploit]

        for bottom_member in bottom:
            # Select a random top performer
            donor = random.choice(top)
            logger.info(
                f"[PBT] Member {bottom_member.trial_id} exploiting {donor.trial_id} "
                f"(score {bottom_member.score:.4f} -> {donor.score:.4f})"
            )

            # Copy checkpoint (in real impl, this would copy TF weights)
            bottom_member.checkpoint_path = donor.checkpoint_path

            # Explore: perturb hyperparameters
            bottom_member.hyperparams = self._perturb_hyperparams(donor.hyperparams)
            bottom_member.generation += 1

    def get_next_trials(self, n_trials: int) -> list[TrialConfig]:
        """Generate initial population or continue existing."""
        trials = []

        # Initialize population if empty
        if not self.population:
            for i in range(min(self.population_size, n_trials)):
                trial_id = f"pbt_g{self.generation}_m{i}"
                self.trial_counter += 1

                hyperparams = self.search_space_sampler(self.trial_counter)
                member = PBTMember(trial_id=trial_id, hyperparams=hyperparams, generation=0)
                self.population.append(member)

                config = TrialConfig(
                    trial_id=trial_id, hyperparams=hyperparams, budget=self.max_budget
                )
                self.trials[trial_id] = config
                trials.append(config)

        logger.info(f"[PBT] Generated {len(trials)} trial configs")
        return trials


def create_scheduler(
    name: str,
    search_space_sampler: Callable[[int], dict[str, Any]] | None = None,
    **kwargs: Any,
) -> HPOSchedulerBase:
    """Factory function to create HPO schedulers.

    Args:
        name: Scheduler name ('hyperband', 'successive_halving', 'pbt', 'random')
        search_space_sampler: Function to sample hyperparameters
        **kwargs: Additional scheduler-specific parameters

    Returns:
        Configured scheduler instance

    Raises:
        ValueError: If scheduler name is unknown

    Example:
        >>> scheduler = create_scheduler('hyperband', max_budget=50, eta=3)
        >>> trials = scheduler.get_next_trials(10)
    """
    name = name.lower().replace("-", "_").replace(" ", "_")

    if name == "hyperband":
        return HyperbandScheduler(search_space_sampler=search_space_sampler, **kwargs)
    elif name in ("successive_halving", "sha", "successivehalving"):
        return SuccessiveHalvingScheduler(search_space_sampler=search_space_sampler, **kwargs)
    elif name in ("pbt", "population_based_training"):
        return PopulationBasedTraining(search_space_sampler=search_space_sampler, **kwargs)
    elif name == "random":
        # Random is just Successive Halving with 1 rung (no elimination)
        return SuccessiveHalvingScheduler(
            search_space_sampler=search_space_sampler,
            eta=1000,  # Effectively no elimination
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown scheduler: {name}. " f"Available: hyperband, successive_halving, pbt, random"
        )


# Export scheduler types for type hints
__all__ = [
    "HPOSchedulerBase",
    "HyperbandScheduler",
    "SuccessiveHalvingScheduler",
    "PopulationBasedTraining",
    "TrialConfig",
    "TrialResult",
    "PBTMember",
    "create_scheduler",
]
