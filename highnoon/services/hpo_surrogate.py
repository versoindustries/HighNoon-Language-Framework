# highnoon/services/hpo_surrogate.py
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

"""TPE Surrogate Model and Constrained Acquisition for QAHPO.

This module implements Tree-Parzen Estimator (TPE) based surrogate modeling
with constraint-aware acquisition functions for budget-aware hyperparameter
optimization.

Key Components:
1. KernelDensityEstimator: Per-parameter KDE for good/bad distributions
2. TPESurrogate: Full surrogate model with EI approximation
3. ConstrainedAcquisition: c-TPE style constraint handling

References:
- Bergstra et al., "Algorithms for Hyper-Parameter Optimization" (NeurIPS 2011)
- Watanabe, "c-TPE: Tree-Structured Parzen Estimator with Inequality Constraints" (IJCAI 2023)
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParameterKDE:
    """Kernel Density Estimator for a single parameter.

    Supports continuous (Gaussian KDE) and categorical (frequency) parameters.
    Uses Scott's rule for bandwidth selection.

    Attributes:
        name: Parameter name
        param_type: "continuous" or "categorical"
        values: Observed values
        bandwidth: KDE bandwidth (for continuous)
        frequencies: Category frequencies (for categorical)
    """

    name: str
    param_type: str = "continuous"
    values: list = field(default_factory=list)
    bandwidth: float = 1.0
    frequencies: dict = field(default_factory=dict)
    min_val: float | None = None
    max_val: float | None = None

    def fit(self, values: list[Any]) -> None:
        """Fit the KDE to observed values.

        Args:
            values: List of observed parameter values
        """
        if not values:
            return

        self.values = list(values)

        if isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
            self.param_type = "continuous"
            arr = np.array(values, dtype=float)
            # Scott's rule for bandwidth
            n = len(arr)
            std = np.std(arr) + 1e-8
            self.bandwidth = std * (n ** (-1 / 5))
            self.min_val = float(np.min(arr))
            self.max_val = float(np.max(arr))
        else:
            self.param_type = "categorical"
            counts = Counter(values)
            total = sum(counts.values())
            self.frequencies = {k: v / total for k, v in counts.items()}

    def score(self, value: Any) -> float:
        """Compute probability density at a value.

        Args:
            value: Parameter value to score

        Returns:
            Probability density (or probability for categorical)
        """
        if not self.values:
            return 1.0  # Uniform prior

        if self.param_type == "continuous":
            # Gaussian KDE
            arr = np.array(self.values, dtype=float)
            diff = (float(value) - arr) / (self.bandwidth + 1e-8)
            kernel = np.exp(-0.5 * diff**2) / (self.bandwidth * np.sqrt(2 * np.pi))
            return float(np.mean(kernel)) + 1e-12
        else:
            # Categorical frequency
            return self.frequencies.get(value, 1e-6)

    def sample(self, rng: np.random.Generator) -> Any:
        """Sample from the KDE.

        Args:
            rng: NumPy random generator

        Returns:
            Sampled value
        """
        if not self.values:
            return None

        if self.param_type == "continuous":
            # Sample from KDE: pick a data point and add noise
            idx = rng.choice(len(self.values))
            base = self.values[idx]
            noise = rng.normal(0, self.bandwidth)
            sampled = base + noise
            # Clamp to observed range
            if self.min_val is not None and self.max_val is not None:
                sampled = np.clip(sampled, self.min_val, self.max_val)
            return float(sampled)
        else:
            # Sample from categorical
            items = list(self.frequencies.keys())
            probs = list(self.frequencies.values())
            return rng.choice(items, p=probs)


class TPESurrogate:
    """Tree-Parzen Estimator Surrogate Model.

    Builds separate KDE models for "good" (gamma quantile) and "bad"
    configurations, then uses the ratio l(x)/g(x) as an Expected
    Improvement approximation.

    Attributes:
        gamma: Fraction of observations considered "good" (default 0.25)
        good_kdes: KDEs for good configurations
        bad_kdes: KDEs for bad configurations
        param_names: List of parameter names being modeled
    """

    def __init__(self, gamma: float = 0.25, min_observations: int = 10):
        """Initialize TPE surrogate.

        Args:
            gamma: Good/bad partition ratio (top gamma are "good")
            min_observations: Minimum observations before fitting
        """
        self.gamma = gamma
        self.min_observations = min_observations
        self.good_kdes: dict[str, ParameterKDE] = {}
        self.bad_kdes: dict[str, ParameterKDE] = {}
        self.param_names: list[str] = []
        self.observations: list[tuple[dict, float]] = []
        self._is_fitted = False

    def add_observation(self, config: dict[str, Any], loss: float) -> None:
        """Add an observation to the surrogate.

        Args:
            config: Hyperparameter configuration
            loss: Achieved loss value
        """
        # Filter out internal keys
        clean_config = {k: v for k, v in config.items() if not k.startswith("_")}
        self.observations.append((clean_config, loss))

        # Update param names
        for key in clean_config:
            if key not in self.param_names:
                self.param_names.append(key)

    def fit(self) -> bool:
        """Fit the surrogate model on accumulated observations.

        Returns:
            True if fitting succeeded, False if not enough data
        """
        if len(self.observations) < self.min_observations:
            logger.debug(
                f"[TPE] Not enough observations ({len(self.observations)} < {self.min_observations})"
            )
            return False

        # Sort by loss
        sorted_obs = sorted(self.observations, key=lambda x: x[1])
        n_good = max(1, int(len(sorted_obs) * self.gamma))

        good_configs = [config for config, _ in sorted_obs[:n_good]]
        bad_configs = [config for config, _ in sorted_obs[n_good:]]

        # Fit per-parameter KDEs
        for param_name in self.param_names:
            # Good KDE
            good_values = [c.get(param_name) for c in good_configs if param_name in c]
            good_values = [v for v in good_values if v is not None]
            good_kde = ParameterKDE(name=param_name)
            if good_values:
                good_kde.fit(good_values)
            self.good_kdes[param_name] = good_kde

            # Bad KDE
            bad_values = [c.get(param_name) for c in bad_configs if param_name in c]
            bad_values = [v for v in bad_values if v is not None]
            bad_kde = ParameterKDE(name=param_name)
            if bad_values:
                bad_kde.fit(bad_values)
            self.bad_kdes[param_name] = bad_kde

        self._is_fitted = True
        logger.info(
            f"[TPE] Fitted surrogate on {len(self.observations)} observations "
            f"({n_good} good, {len(bad_configs)} bad)"
        )
        return True

    def acquisition(self, config: dict[str, Any]) -> float:
        """Compute acquisition value for a configuration.

        Uses Expected Improvement approximation: l(x) / g(x)
        where l(x) is good density and g(x) is bad density.

        Args:
            config: Configuration to evaluate

        Returns:
            Acquisition value (higher is better)
        """
        if not self._is_fitted:
            return 1.0  # Uniform before fitting

        l_score = 1.0  # Good density
        g_score = 1.0  # Bad density

        for param_name in self.param_names:
            if param_name not in config:
                continue

            value = config[param_name]

            if param_name in self.good_kdes:
                l_score *= self.good_kdes[param_name].score(value)
            if param_name in self.bad_kdes:
                g_score *= self.bad_kdes[param_name].score(value)

        # EI approximation: l(x) / g(x)
        # Add small epsilon to avoid division by zero
        return l_score / (g_score + 1e-12)

    def sample_from_good(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """Sample a configuration biased toward good region.

        Args:
            rng: NumPy random generator

        Returns:
            Sampled configuration
        """
        if rng is None:
            rng = np.random.default_rng()

        config = {}
        for param_name in self.param_names:
            if param_name in self.good_kdes:
                kde = self.good_kdes[param_name]
                value = kde.sample(rng)
                if value is not None:
                    config[param_name] = value

        return config

    def get_statistics(self) -> dict[str, Any]:
        """Get surrogate model statistics.

        Returns:
            Dictionary with model statistics
        """
        if not self._is_fitted:
            return {"fitted": False, "observations": len(self.observations)}

        # Best observed
        best_obs = min(self.observations, key=lambda x: x[1])

        return {
            "fitted": True,
            "observations": len(self.observations),
            "n_params": len(self.param_names),
            "gamma": self.gamma,
            "best_loss": best_obs[1],
            "param_names": self.param_names[:10],  # First 10
        }


class ConstrainedAcquisition:
    """Constrained TPE Acquisition Function (c-TPE).

    Modifies the acquisition function to consider constraint satisfaction:
    acquisition(x) = TPE_acquisition(x) × P(feasible | x)

    Supports parameter budget constraints and learns which regions
    violate constraints.

    Attributes:
        tpe: Underlying TPE surrogate
        param_budget: Maximum parameter count
        budget_penalty_lambda: Penalty strength for budget violations
    """

    def __init__(
        self,
        tpe_surrogate: TPESurrogate | None = None,
        param_budget: int | None = None,
        budget_penalty_lambda: float = 1.0,
        constraint_temperature: float = 0.1,
    ):
        """Initialize constrained acquisition.

        Args:
            tpe_surrogate: TPE surrogate model (creates new if None)
            param_budget: Parameter count budget constraint
            budget_penalty_lambda: Penalty weight for constraint violations
            constraint_temperature: Sigmoid temperature for soft constraints
        """
        self.tpe = tpe_surrogate or TPESurrogate()
        self.param_budget = param_budget
        self.budget_penalty_lambda = budget_penalty_lambda
        self.constraint_temperature = constraint_temperature

        # Track constraint violations for learning
        self.violation_history: list[tuple[dict, float, bool]] = []

    def set_param_budget(self, budget: int | None) -> None:
        """Set the parameter budget constraint.

        Args:
            budget: Maximum allowed parameter count, or None to disable
        """
        self.param_budget = budget

    def compute_constraint_probability(
        self,
        estimated_params: int,
    ) -> float:
        """Compute probability that config satisfies constraint.

        Uses sigmoid function for soft constraint handling:
        P(feasible) = sigmoid(-(estimated - budget) / (temperature × budget))

        Args:
            estimated_params: Estimated parameter count for config

        Returns:
            Probability of feasibility (0 to 1)
        """
        if self.param_budget is None:
            return 1.0  # No constraint

        if estimated_params <= self.param_budget:
            return 1.0  # Definitely feasible

        # Soft penalty for near-budget configs
        # Sigmoid centered at budget, steepness controlled by temperature
        normalized_excess = (estimated_params - self.param_budget) / (
            self.constraint_temperature * self.param_budget + 1e-8
        )
        return 1.0 / (1.0 + math.exp(normalized_excess))

    def acquisition(
        self,
        config: dict[str, Any],
        estimated_params: int | None = None,
    ) -> float:
        """Compute constrained acquisition value.

        Args:
            config: Configuration to evaluate
            estimated_params: Pre-computed param estimate (optional)

        Returns:
            Constrained acquisition value (higher is better)
        """
        # Base TPE acquisition
        base_acq = self.tpe.acquisition(config)

        # Constraint probability
        if estimated_params is not None and self.param_budget is not None:
            constraint_prob = self.compute_constraint_probability(estimated_params)
        else:
            constraint_prob = 1.0

        # Combined acquisition
        return base_acq * constraint_prob

    def record_constraint_violation(
        self,
        config: dict[str, Any],
        estimated_params: int,
        violated: bool,
    ) -> None:
        """Record a constraint check result for learning.

        Args:
            config: Configuration that was checked
            estimated_params: Estimated parameter count
            violated: Whether constraint was violated
        """
        self.violation_history.append((config, estimated_params, violated))

        # Log trends
        if len(self.violation_history) % 10 == 0:
            recent = self.violation_history[-10:]
            violation_rate = sum(1 for _, _, v in recent if v) / len(recent)
            logger.info(
                f"[c-TPE] Recent constraint violation rate: {violation_rate:.1%} "
                f"({len(self.violation_history)} total checks)"
            )

    def get_feasibility_statistics(self) -> dict[str, Any]:
        """Get constraint handling statistics.

        Returns:
            Dictionary with feasibility stats
        """
        if not self.violation_history:
            return {"total_checks": 0}

        violations = [v for _, _, v in self.violation_history]
        violation_rate = sum(violations) / len(violations)

        return {
            "total_checks": len(self.violation_history),
            "violation_count": sum(violations),
            "violation_rate": violation_rate,
            "param_budget": self.param_budget,
        }


__all__ = [
    "ParameterKDE",
    "TPESurrogate",
    "ConstrainedAcquisition",
]
