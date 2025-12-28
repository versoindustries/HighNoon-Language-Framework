# highnoon/services/hpo_multi_fidelity.py
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

"""Multi-Fidelity HPO Scheduling (BOHB-style Hyperband).

This module implements Hyperband-style successive halving for multi-fidelity
hyperparameter optimization. Trials start with low resource budgets and only
promising configurations are promoted to higher fidelity levels.

Key Components:
1. FidelityBracket: Single bracket with specific resource allocation
2. HyperbandSchedule: Full Hyperband schedule generator
3. SuccessiveHalvingRound: Manages promotion within a bracket

References:
- Li et al., "Hyperband: A Novel Bandit-Based Approach to Hyperparameter
  Optimization" (JMLR 2018)
- Falkner et al., "BOHB: Robust and Efficient Hyperparameter Optimization
  at Scale" (ICML 2018)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FidelityLevel:
    """Single fidelity level in a Hyperband bracket.

    Attributes:
        epochs: Number of training epochs at this level
        n_configs: Number of configurations to run at this level
        survival_rate: Fraction of configs promoted to next level
    """

    epochs: int
    n_configs: int
    survival_rate: float = 0.5

    @property
    def promoted_count(self) -> int:
        """Number of configs promoted to next level."""
        return max(1, int(self.n_configs * self.survival_rate))


@dataclass
class FidelityBracket:
    """A complete Hyperband bracket with multiple fidelity levels.

    A bracket starts many configs at low fidelity and progressively
    eliminates underperformers at each rung.

    Attributes:
        bracket_id: Identifier for this bracket
        levels: List of fidelity levels from low to high
        current_level: Index of current fidelity level
        configs_at_level: Configs running at each level
        results_at_level: Results collected at each level
    """

    bracket_id: int
    levels: list[FidelityLevel] = field(default_factory=list)
    current_level: int = 0
    configs_at_level: dict[int, list[dict]] = field(default_factory=dict)
    results_at_level: dict[int, list[tuple[dict, float]]] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if all levels are complete."""
        return self.current_level >= len(self.levels)

    @property
    def current_epochs(self) -> int:
        """Get epochs for current fidelity level."""
        if self.current_level >= len(self.levels):
            return self.levels[-1].epochs if self.levels else 0
        return self.levels[self.current_level].epochs

    @property
    def configs_needed(self) -> int:
        """Number of configs needed at current level."""
        if self.current_level >= len(self.levels):
            return 0
        level = self.levels[self.current_level]
        current_results = len(self.results_at_level.get(self.current_level, []))
        return level.n_configs - current_results

    def add_result(self, config: dict[str, Any], loss: float) -> None:
        """Add a trial result at current level.

        Args:
            config: Configuration that was run
            loss: Achieved loss value
        """
        if self.current_level not in self.results_at_level:
            self.results_at_level[self.current_level] = []
        self.results_at_level[self.current_level].append((config, loss))

    def should_advance(self) -> bool:
        """Check if current level has enough results to advance."""
        if self.current_level >= len(self.levels):
            return False

        level = self.levels[self.current_level]
        results = self.results_at_level.get(self.current_level, [])
        return len(results) >= level.n_configs

    def advance_level(self) -> list[dict[str, Any]]:
        """Advance to next fidelity level, returning promoted configs.

        Returns:
            List of configs promoted to next level
        """
        if not self.should_advance():
            return []

        level = self.levels[self.current_level]
        results = self.results_at_level.get(self.current_level, [])

        # Sort by loss and promote top performers
        sorted_results = sorted(results, key=lambda x: x[1])
        n_promote = level.promoted_count

        promoted_configs = [config for config, _ in sorted_results[:n_promote]]

        logger.info(
            f"[Hyperband] Bracket {self.bracket_id}: Level {self.current_level} → "
            f"{self.current_level + 1}, promoting {n_promote}/{len(results)} configs"
        )

        self.current_level += 1
        if self.current_level < len(self.levels):
            self.configs_at_level[self.current_level] = promoted_configs

        return promoted_configs

    def get_statistics(self) -> dict[str, Any]:
        """Get bracket statistics."""
        return {
            "bracket_id": self.bracket_id,
            "n_levels": len(self.levels),
            "current_level": self.current_level,
            "is_complete": self.is_complete,
            "configs_needed": self.configs_needed,
            "total_results": sum(len(r) for r in self.results_at_level.values()),
        }


class HyperbandSchedule:
    """Hyperband Schedule Generator.

    Generates Hyperband brackets with different exploration/exploitation
    tradeoffs. Early brackets favor exploration (more configs, less budget),
    later brackets favor exploitation (fewer configs, more budget).

    Attributes:
        max_epochs: Maximum epochs for full training
        min_epochs: Minimum epochs for first rung
        eta: Reduction factor (typically 3)
        brackets: List of generated brackets
    """

    def __init__(
        self,
        max_epochs: int = 50,
        min_epochs: int = 1,
        eta: int = 3,
        max_configs_per_bracket: int = 27,
    ):
        """Initialize Hyperband schedule.

        Args:
            max_epochs: Maximum epochs for any trial
            min_epochs: Minimum epochs at first rung
            eta: Halving rate (default 3 = keep top 1/3)
            max_configs_per_bracket: Maximum configs in first rung
        """
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.eta = eta
        self.max_configs_per_bracket = max_configs_per_bracket

        self.brackets: list[FidelityBracket] = []
        self._next_bracket_id = 0

        # Compute s_max (number of brackets)
        self.s_max = int(math.log(max_epochs / min_epochs, eta)) + 1

        logger.info(
            f"[Hyperband] Initialized with R={max_epochs}, r={min_epochs}, "
            f"η={eta}, s_max={self.s_max}"
        )

    def generate_bracket(self, s: int | None = None) -> FidelityBracket:
        """Generate a new Hyperband bracket.

        Args:
            s: Bracket index (0 to s_max-1). If None, cycles through brackets.

        Returns:
            New FidelityBracket ready for trials
        """
        if s is None:
            s = self._next_bracket_id % self.s_max
            self._next_bracket_id += 1
        else:
            s = s % self.s_max

        # Hyperband formulas
        n_configs = int(
            min(
                self.max_configs_per_bracket,
                math.ceil((self.s_max + 1) * (self.eta**s) / (s + 1)),
            )
        )
        r = self.max_epochs / (self.eta**s)  # Initial budget

        levels = []
        for i in range(s + 1):
            # Number of configs at this rung
            n_i = max(1, int(n_configs / (self.eta**i)))
            # Epochs at this rung
            r_i = int(r * (self.eta**i))
            r_i = max(self.min_epochs, min(self.max_epochs, r_i))

            survival_rate = 1.0 / self.eta if i < s else 1.0

            levels.append(
                FidelityLevel(
                    epochs=r_i,
                    n_configs=n_i,
                    survival_rate=survival_rate,
                )
            )

        bracket = FidelityBracket(
            bracket_id=len(self.brackets),
            levels=levels,
        )
        self.brackets.append(bracket)

        logger.info(
            f"[Hyperband] Generated bracket {bracket.bracket_id} (s={s}): "
            f"{[(l.epochs, l.n_configs) for l in levels]}"
        )

        return bracket

    def get_active_bracket(self) -> FidelityBracket | None:
        """Get the current active bracket that needs trials.

        Returns:
            Active bracket or None if all complete
        """
        # Find incomplete bracket
        for bracket in self.brackets:
            if not bracket.is_complete and bracket.configs_needed > 0:
                return bracket

        # All brackets complete, generate new one
        if len(self.brackets) < self.s_max * 2:  # Limit total brackets
            return self.generate_bracket()

        return None

    def get_trial_budget(self, bracket: FidelityBracket | None = None) -> int:
        """Get epochs for the next trial.

        Args:
            bracket: Specific bracket, or use active bracket

        Returns:
            Number of epochs to train
        """
        if bracket is None:
            bracket = self.get_active_bracket()

        if bracket is None:
            return self.max_epochs

        return bracket.current_epochs

    def report_result(
        self,
        config: dict[str, Any],
        loss: float,
        bracket_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Report trial result and get promoted configs if level complete.

        Args:
            config: Configuration that was trained
            loss: Achieved loss value
            bracket_id: Bracket to report to (uses active if None)

        Returns:
            List of configs to run at next level (empty if not advancing)
        """
        # Find target bracket
        if bracket_id is not None:
            bracket = next(
                (b for b in self.brackets if b.bracket_id == bracket_id),
                None,
            )
        else:
            bracket = self.get_active_bracket()

        if bracket is None:
            logger.warning("[Hyperband] No active bracket for result")
            return []

        bracket.add_result(config, loss)

        # Check if level is complete and should advance
        if bracket.should_advance():
            return bracket.advance_level()

        return []

    def get_statistics(self) -> dict[str, Any]:
        """Get Hyperband schedule statistics."""
        return {
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "eta": self.eta,
            "s_max": self.s_max,
            "total_brackets": len(self.brackets),
            "completed_brackets": sum(1 for b in self.brackets if b.is_complete),
            "brackets": [b.get_statistics() for b in self.brackets],
        }


@dataclass
class DEHBConfig:
    """Configuration for DEHB (Differential Evolution + Hyperband).

    Combines Hyperband multi-fidelity with DE mutation operators.

    Attributes:
        hyperband: Hyperband schedule configuration
        de_F: DE scaling factor (typical: 0.5-0.9)
        de_CR: DE crossover rate (typical: 0.7-0.9)
        mutation_strategy: DE mutation strategy
    """

    max_epochs: int = 50
    min_epochs: int = 1
    eta: int = 3
    de_F: float = 0.8
    de_CR: float = 0.9
    mutation_strategy: str = "rand_1"  # "rand_1", "best_1", "current_to_best_1"


__all__ = [
    "FidelityLevel",
    "FidelityBracket",
    "HyperbandSchedule",
    "DEHBConfig",
]
