# highnoon/services/hpo_theoretical.py
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

"""Theoretical Foundations for QAHPO.

Enterprise Enhancement Phase 7: Theoretical components including:
    - Regret Tracking: Simple and cumulative regret bounds
    - Information-Theoretic Acquisition: MES, GIBBON, info-gain

References:
    - Srinivas et al., "Gaussian Process Optimization with Bandit Assumptions" (JMLR 2012)
    - Wang & Jegelka, "Max-value Entropy Search for Efficient Bayesian Optimization" (ICML 2017)

Example:
    >>> tracker = RegretTracker(optimal_loss=0.01)
    >>> tracker.record(0.5)
    >>> tracker.record(0.3)
    >>> print(tracker.cumulative_regret)  # 0.78
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# REGRET TRACKING
# =============================================================================
@dataclass
class RegretBound:
    """Theoretical regret bound.

    Attributes:
        bound_type: Type of bound (simple, cumulative, high_prob).
        value: Bound value.
        confidence: Confidence level for probabilistic bounds.
        n_samples: Number of samples used.
    """

    bound_type: str
    value: float
    confidence: float = 0.95
    n_samples: int = 0


class RegretTracker:
    """Track and bound optimization regret.

    Regret measures the gap between achieved performance and the
    global optimum. Two types:
    - Simple Regret: Gap between best found and optimal
    - Cumulative Regret: Sum of gaps across all evaluations

    Attributes:
        optimal_loss: Known or estimated optimal loss.
        mode: Optimization mode ("min" or "max").
        observations: List of observed losses.
        best_observed: Best loss found.
    """

    def __init__(
        self,
        optimal_loss: float = 0.0,
        mode: str = "min",
        kernel_bandwidth: float = 1.0,
    ) -> None:
        """Initialize regret tracker.

        Args:
            optimal_loss: Known or estimated optimal.
            mode: Optimization mode.
            kernel_bandwidth: RBF kernel bandwidth for bounds.
        """
        self.optimal_loss = optimal_loss
        self.mode = mode
        self.kernel_bandwidth = kernel_bandwidth

        self.observations: list[float] = []
        self.best_observed: float = float("inf") if mode == "min" else float("-inf")
        self._simple_regrets: list[float] = []
        self._cumulative_regret: float = 0.0

    def record(self, loss: float) -> tuple[float, float]:
        """Record an observation and compute regrets.

        Args:
            loss: Observed loss value.

        Returns:
            Tuple of (simple_regret, instantaneous_regret).
        """
        self.observations.append(loss)

        # Update best
        if self.mode == "min":
            self.best_observed = min(self.best_observed, loss)
            instantaneous_regret = max(0, loss - self.optimal_loss)
            simple_regret = max(0, self.best_observed - self.optimal_loss)
        else:
            self.best_observed = max(self.best_observed, loss)
            instantaneous_regret = max(0, self.optimal_loss - loss)
            simple_regret = max(0, self.optimal_loss - self.best_observed)

        self._simple_regrets.append(simple_regret)
        self._cumulative_regret += instantaneous_regret

        return simple_regret, instantaneous_regret

    @property
    def simple_regret(self) -> float:
        """Current simple regret."""
        if not self._simple_regrets:
            return float("inf")
        return self._simple_regrets[-1]

    @property
    def cumulative_regret(self) -> float:
        """Cumulative regret."""
        return self._cumulative_regret

    def get_simple_regret_bound(self, delta: float = 0.05) -> RegretBound:
        """Compute high-probability simple regret bound.

        Uses GP-UCB style bounds assuming RBF kernel.

        Args:
            delta: Failure probability.

        Returns:
            Regret bound with confidence 1-delta.
        """
        n = len(self.observations)
        if n == 0:
            return RegretBound("simple", float("inf"), 1 - delta, 0)

        # GP-UCB bound: O(sqrt(gamma_T * T * log(T/delta)))
        # gamma_T is information gain, bounded by O(log(T)^d) for RBF
        d = 10  # Assume 10-dimensional
        gamma_t = math.log(n + 1) ** d
        beta_t = 2 * math.log(n * math.pi**2 / (6 * delta))

        bound = math.sqrt(gamma_t * beta_t / n)

        return RegretBound(
            bound_type="simple_gp_ucb",
            value=min(bound, self.simple_regret),
            confidence=1 - delta,
            n_samples=n,
        )

    def get_cumulative_regret_bound(self, delta: float = 0.05) -> RegretBound:
        """Compute high-probability cumulative regret bound.

        Args:
            delta: Failure probability.

        Returns:
            Cumulative regret bound.
        """
        n = len(self.observations)
        if n == 0:
            return RegretBound("cumulative", float("inf"), 1 - delta, 0)

        # Bound: O(sqrt(T * gamma_T * log(T/delta)))
        d = 10
        gamma_t = math.log(n + 1) ** d
        beta_t = 2 * math.log(n * math.pi**2 / (6 * delta))

        bound = math.sqrt(n * gamma_t * beta_t)

        return RegretBound(
            bound_type="cumulative_gp_ucb",
            value=min(bound, self._cumulative_regret),
            confidence=1 - delta,
            n_samples=n,
        )

    def estimate_optimal(self, quantile: float = 0.01) -> float:
        """Estimate optimal loss from observations.

        Args:
            quantile: Quantile for estimation.

        Returns:
            Estimated optimal loss.
        """
        if not self.observations:
            return 0.0

        return float(np.quantile(self.observations, quantile))

    def convergence_rate(self, window: int = 10) -> float:
        """Estimate convergence rate.

        Args:
            window: Window for rate estimation.

        Returns:
            Estimated convergence rate.
        """
        if len(self._simple_regrets) < window:
            return 0.0

        early = np.mean(self._simple_regrets[:window])
        late = np.mean(self._simple_regrets[-window:])

        if early < 1e-10:
            return 0.0

        return float((early - late) / early)

    def get_statistics(self) -> dict[str, Any]:
        """Get tracker statistics."""
        return {
            "type": "RegretTracker",
            "n_observations": len(self.observations),
            "simple_regret": self.simple_regret,
            "cumulative_regret": self._cumulative_regret,
            "best_observed": self.best_observed,
            "optimal_loss": self.optimal_loss,
            "convergence_rate": self.convergence_rate(),
        }


# =============================================================================
# INFORMATION-THEORETIC ACQUISITION
# =============================================================================
class InformationGainAcquisition:
    """Information-Theoretic Acquisition Functions.

    Implements acquisition functions that maximize information gain
    about the global optimum.

    Types:
        - MES: Max-value Entropy Search
        - GIBBON: General-purpose Information-Based Bayesian Optimization

    Attributes:
        num_samples: Monte Carlo samples for estimation.
        num_fantasies: Fantasy samples for MES.
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_fantasies: int = 10,
        seed: int = 42,
    ) -> None:
        """Initialize information-gain acquisition.

        Args:
            num_samples: Monte Carlo samples.
            num_fantasies: Fantasy samples for MES.
            seed: Random seed.
        """
        self.num_samples = num_samples
        self.num_fantasies = num_fantasies
        self._rng = np.random.default_rng(seed)

        # Cached samples of y*
        self._y_star_samples: np.ndarray | None = None
        self._observations: list[tuple[dict[str, Any], float]] = []

    def add_observation(self, config: dict[str, Any], loss: float) -> None:
        """Add observation.

        Args:
            config: Configuration.
            loss: Observed loss.
        """
        self._observations.append((config, loss))
        self._y_star_samples = None  # Invalidate cache

    def _sample_y_star(self) -> np.ndarray:
        """Sample from p(y*) using observed data.

        Returns:
            Samples of the minimum value.
        """
        if len(self._observations) < 3:
            return self._rng.uniform(0, 1, self.num_samples)

        losses = np.array([l for _, l in self._observations])

        # Use Gumbel distribution to model minimum
        # p(y*) ≈ Gumbel(min(y), scale)
        location = losses.min()
        scale = max(0.01, losses.std())

        samples = location - scale * np.log(
            -np.log(self._rng.uniform(0.001, 0.999, self.num_samples))
        )

        return np.clip(samples, 0, losses.max())

    def mes(
        self,
        mean: float,
        std: float,
        y_star_samples: np.ndarray | None = None,
    ) -> float:
        """Max-value Entropy Search acquisition.

        Maximizes I(y*; f(x)) = H(y*) - E[H(y* | f(x))]

        Args:
            mean: Predicted mean at point.
            std: Predicted std at point.
            y_star_samples: Samples of y*.

        Returns:
            MES acquisition value.
        """
        if y_star_samples is None:
            if self._y_star_samples is None:
                self._y_star_samples = self._sample_y_star()
            y_star_samples = self._y_star_samples

        if std < 1e-10:
            return 0.0

        # Compute gamma = (y* - mean) / std
        gamma = (y_star_samples - mean) / std

        # MES = E[gamma * phi(gamma) / (2 * Phi(gamma)) - log(Phi(gamma))]
        # where phi is normal PDF, Phi is normal CDF
        phi = np.exp(-0.5 * gamma**2) / np.sqrt(2 * np.pi)
        Phi = 0.5 * (1 + np.vectorize(math.erf)(gamma / np.sqrt(2)))

        # Avoid division by zero
        Phi = np.clip(Phi, 1e-10, 1 - 1e-10)

        acquisition = gamma * phi / (2 * Phi) - np.log(Phi)

        return float(np.mean(acquisition))

    def gibbon(
        self,
        mean: float,
        std: float,
        current_best: float,
    ) -> float:
        """General-purpose Information-Based Bayesian OptN.

        Simplified GIBBON using local information gain.

        Args:
            mean: Predicted mean.
            std: Predicted std.
            current_best: Current best observation.

        Returns:
            GIBBON acquisition value.
        """
        if std < 1e-10:
            return 0.0

        # Improvement
        improvement = current_best - mean

        # Information gain approximation
        z = improvement / std
        phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        Phi = 0.5 * (1 + math.erf(z / np.sqrt(2)))

        # Expected information gain
        if Phi > 1e-10:
            info_gain = np.log(Phi) + (z * phi) / (2 * Phi)
        else:
            info_gain = 0.0

        return float(-info_gain)  # Negate for maximization

    def thompson_sampling(
        self,
        mean: float,
        std: float,
    ) -> float:
        """Thompson Sampling acquisition.

        Sample from posterior and use as acquisition.

        Args:
            mean: Predicted mean.
            std: Predicted std.

        Returns:
            Thompson sample (acquisition value).
        """
        return float(self._rng.normal(mean, std))

    def knowledge_gradient(
        self,
        mean: float,
        std: float,
        current_best: float,
    ) -> float:
        """Knowledge Gradient acquisition.

        Expected improvement in optimal value after observation.

        Args:
            mean: Predicted mean.
            std: Predicted std.
            current_best: Current best observation.

        Returns:
            Knowledge gradient value.
        """
        if std < 1e-10:
            return 0.0

        z = (current_best - mean) / std
        phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        Phi = 0.5 * (1 + math.erf(z / np.sqrt(2)))

        # KG = std * (z * Phi + phi)
        kg = std * (z * Phi + phi)

        return float(kg)

    def get_statistics(self) -> dict[str, Any]:
        """Get acquisition statistics."""
        return {
            "type": "InformationGainAcquisition",
            "num_samples": self.num_samples,
            "num_fantasies": self.num_fantasies,
            "num_observations": len(self._observations),
        }


# =============================================================================
# A/B TESTING FRAMEWORK FOR HPO STRATEGIES
# =============================================================================
@dataclass
class StrategyArm:
    """Arm in the HPO strategy A/B test.

    Attributes:
        name: Strategy name.
        cumulative_regret: Total regret for this strategy.
        num_pulls: Number of times selected.
        best_loss: Best loss achieved.
        mean_time_to_best: Average time to find best.
    """

    name: str
    cumulative_regret: float = 0.0
    num_pulls: int = 0
    best_loss: float = float("inf")
    mean_time_to_best: float = float("inf")


class HPOExperimentFramework:
    """A/B Testing Framework for HPO Strategies.

    Compares multiple HPO strategies using multi-armed bandit
    allocation with Thompson Sampling.

    Attributes:
        arms: Dictionary of strategy arms.
        allocation: Allocation method (thompson, ucb, uniform).
    """

    def __init__(
        self,
        strategy_names: list[str],
        allocation: str = "thompson",
        seed: int = 42,
    ) -> None:
        """Initialize experiment framework.

        Args:
            strategy_names: Names of strategies to compare.
            allocation: Allocation method.
            seed: Random seed.
        """
        self.arms = {name: StrategyArm(name=name) for name in strategy_names}
        self.allocation = allocation
        self._rng = np.random.default_rng(seed)

        # Beta distribution parameters for Thompson Sampling
        self._alpha = dict.fromkeys(strategy_names, 1.0)
        self._beta = dict.fromkeys(strategy_names, 1.0)

    def select_strategy(self) -> str:
        """Select next strategy to evaluate.

        Returns:
            Name of selected strategy.
        """
        if self.allocation == "uniform":
            return self._rng.choice(list(self.arms.keys()))

        elif self.allocation == "ucb":
            total_pulls = sum(arm.num_pulls for arm in self.arms.values())
            ucb_scores = {}

            for name, arm in self.arms.items():
                if arm.num_pulls == 0:
                    ucb_scores[name] = float("inf")
                else:
                    mean_reward = -arm.cumulative_regret / arm.num_pulls
                    exploration = math.sqrt(2 * math.log(total_pulls + 1) / arm.num_pulls)
                    ucb_scores[name] = mean_reward + exploration

            return max(ucb_scores, key=ucb_scores.get)

        else:  # Thompson Sampling
            samples = {
                name: self._rng.beta(self._alpha[name], self._beta[name]) for name in self.arms
            }
            return max(samples, key=samples.get)

    def record_result(
        self,
        strategy_name: str,
        regret: float,
        best_loss: float,
        time_to_best: float,
    ) -> None:
        """Record result for a strategy.

        Args:
            strategy_name: Strategy that was evaluated.
            regret: Regret incurred.
            best_loss: Best loss achieved.
            time_to_best: Time to find best.
        """
        if strategy_name not in self.arms:
            return

        arm = self.arms[strategy_name]
        arm.cumulative_regret += regret
        arm.num_pulls += 1
        arm.best_loss = min(arm.best_loss, best_loss)

        # Update mean time to best
        if arm.mean_time_to_best == float("inf"):
            arm.mean_time_to_best = time_to_best
        else:
            arm.mean_time_to_best = (
                arm.mean_time_to_best * (arm.num_pulls - 1) + time_to_best
            ) / arm.num_pulls

        # Update Beta parameters (reward = 1 - normalized_regret)
        normalized_regret = min(1, regret / 10)  # Normalize
        self._alpha[strategy_name] += 1 - normalized_regret
        self._beta[strategy_name] += normalized_regret

    def get_best_strategy(self) -> str:
        """Get the best performing strategy.

        Returns:
            Name of best strategy.
        """
        # Rank by cumulative regret per pull
        scores = {}
        for name, arm in self.arms.items():
            if arm.num_pulls > 0:
                scores[name] = arm.cumulative_regret / arm.num_pulls
            else:
                scores[name] = float("inf")

        return min(scores, key=scores.get)

    def get_statistics(self) -> dict[str, Any]:
        """Get experiment statistics."""
        return {
            "type": "HPOExperimentFramework",
            "allocation": self.allocation,
            "num_strategies": len(self.arms),
            "best_strategy": self.get_best_strategy(),
            "arms": {
                name: {
                    "num_pulls": arm.num_pulls,
                    "cumulative_regret": arm.cumulative_regret,
                    "best_loss": arm.best_loss if arm.best_loss < float("inf") else None,
                    "mean_time_to_best": (
                        arm.mean_time_to_best if arm.mean_time_to_best < float("inf") else None
                    ),
                }
                for name, arm in self.arms.items()
            },
        }


__all__ = [
    "RegretTracker",
    "RegretBound",
    "InformationGainAcquisition",
    "HPOExperimentFramework",
    "StrategyArm",
]
