# highnoon/services/hpo_fanova_advanced.py
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

"""Advanced fANOVA Enhancements for QAHPO.

Enterprise Enhancement Phase 4 (extended): Advanced importance analysis:
    - Graph fANOVA: Conditional importance via DAG structure
    - Causal Importance: Interventional vs observational analysis
    - Streaming fANOVA: O(1) incremental updates

References:
    - Hutter et al., "An Efficient Approach for Assessing Hyperparameter Importance" (ICML 2014)
    - Pearl, "Causality: Models, Reasoning and Inference" (2009)

Example:
    >>> graph_fanova = GraphfANOVA(dependency_graph)
    >>> graph_fanova.fit(configs, losses)
    >>> conditional_importance = graph_fanova.get_conditional_importance("lr", given="optimizer")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DependencyEdge:
    """Edge in the parameter dependency graph.

    Attributes:
        parent: Parent parameter name.
        child: Child parameter name.
        condition: Optional condition for activation.
    """

    parent: str
    child: str
    condition: Any = None


class GraphfANOVA:
    """fANOVA with Conditional Dependency Graph.

    Extends standard fANOVA to handle hierarchical configurations
    where some parameters are only active given parent values.

    Attributes:
        dependency_graph: DAG of parameter dependencies.
        param_names: List of parameter names.
    """

    def __init__(
        self,
        dependency_edges: list[DependencyEdge] | None = None,
        param_names: list[str] | None = None,
    ) -> None:
        """Initialize Graph fANOVA.

        Args:
            dependency_edges: Edges defining parameter dependencies.
            param_names: Parameter names.
        """
        self.dependency_edges = dependency_edges or []
        self.param_names = param_names or []

        # Build adjacency lists
        self._parents: dict[str, list[str]] = {}
        self._children: dict[str, list[str]] = {}

        for edge in self.dependency_edges:
            if edge.child not in self._parents:
                self._parents[edge.child] = []
            self._parents[edge.child].append(edge.parent)

            if edge.parent not in self._children:
                self._children[edge.parent] = []
            self._children[edge.parent].append(edge.child)

        # Fitted data
        self._configs: list[dict[str, Any]] = []
        self._losses: list[float] = []
        self._importance_cache: dict[str, float] = {}

    def fit(self, configs: list[dict[str, Any]], losses: list[float]) -> None:
        """Fit Graph fANOVA on trial data.

        Args:
            configs: List of configurations.
            losses: Corresponding losses.
        """
        self._configs = configs
        self._losses = np.array(losses)
        self._importance_cache = {}

        # Discover param names if not provided
        if not self.param_names and configs:
            self.param_names = list(configs[0].keys())

        logger.info(
            f"[GraphfANOVA] Fitted on {len(configs)} configs, {len(self.param_names)} params"
        )

    def get_unconditional_importance(self, param: str) -> float:
        """Get unconditional importance (marginal variance).

        Args:
            param: Parameter name.

        Returns:
            Importance score [0, 1].
        """
        if not self._configs or param not in self.param_names:
            return 0.0

        # Group by parameter value
        groups: dict[Any, list[float]] = {}
        for config, loss in zip(self._configs, self._losses):
            val = config.get(param)
            if val not in groups:
                groups[val] = []
            groups[val].append(loss)

        if len(groups) < 2:
            return 0.0

        # Compute explained variance
        total_var = np.var(self._losses)
        if total_var < 1e-10:
            return 0.0

        group_means = [np.mean(g) for g in groups.values()]
        group_sizes = [len(g) for g in groups.values()]
        between_var = np.average(
            [(m - np.mean(self._losses)) ** 2 for m in group_means],
            weights=group_sizes,
        )

        return float(between_var / total_var)

    def get_conditional_importance(
        self,
        param: str,
        given: str | None = None,
        given_value: Any = None,
    ) -> float:
        """Get conditional importance given parent value.

        Args:
            param: Parameter to evaluate.
            given: Conditioning parameter.
            given_value: Value of conditioning parameter.

        Returns:
            Conditional importance score.
        """
        if not given:
            return self.get_unconditional_importance(param)

        # Filter to configs where given=given_value
        filtered_configs = []
        filtered_losses = []

        for config, loss in zip(self._configs, self._losses):
            if config.get(given) == given_value:
                filtered_configs.append(config)
                filtered_losses.append(loss)

        if len(filtered_configs) < 5:
            return 0.0

        # Compute importance on filtered set
        groups: dict[Any, list[float]] = {}
        for config, loss in zip(filtered_configs, filtered_losses):
            val = config.get(param)
            if val not in groups:
                groups[val] = []
            groups[val].append(loss)

        if len(groups) < 2:
            return 0.0

        total_var = np.var(filtered_losses)
        if total_var < 1e-10:
            return 0.0

        group_means = [np.mean(g) for g in groups.values()]
        group_sizes = [len(g) for g in groups.values()]
        between_var = np.average(
            [(m - np.mean(filtered_losses)) ** 2 for m in group_means],
            weights=group_sizes,
        )

        return float(between_var / total_var)

    def get_all_importance(self) -> dict[str, float]:
        """Get importance scores for all parameters.

        Returns:
            Dictionary of param -> importance.
        """
        if self._importance_cache:
            return self._importance_cache

        for param in self.param_names:
            self._importance_cache[param] = self.get_unconditional_importance(param)

        return self._importance_cache

    def get_topological_order(self) -> list[str]:
        """Get topological ordering of parameters.

        Returns:
            Parameters sorted by dependency.
        """
        # Kahn's algorithm
        in_degree = {p: len(self._parents.get(p, [])) for p in self.param_names}
        queue = [p for p in self.param_names if in_degree.get(p, 0) == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in self._children.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

    def get_statistics(self) -> dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "type": "GraphfANOVA",
            "num_params": len(self.param_names),
            "num_edges": len(self.dependency_edges),
            "num_samples": len(self._configs),
        }


class CausalImportanceAnalyzer:
    """Causal fANOVA using Interventional Analysis.

    Distinguishes between correlation and causation by comparing:
    - Observational: P(loss | param=x)
    - Interventional: P(loss | do(param=x))

    Uses adjustment formula to estimate causal effects.

    Attributes:
        confounders: Known confounding parameters.
        adjustment_set: Parameters to adjust for.
    """

    def __init__(
        self,
        confounders: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize causal analyzer.

        Args:
            confounders: Known confounding parameters.
            seed: Random seed.
        """
        self.confounders = confounders or []
        self.adjustment_set: list[str] = []
        self._rng = np.random.default_rng(seed)

        self._configs: list[dict[str, Any]] = []
        self._losses: np.ndarray = np.array([])

    def fit(self, configs: list[dict[str, Any]], losses: list[float]) -> None:
        """Fit causal analyzer.

        Args:
            configs: Trial configurations.
            losses: Corresponding losses.
        """
        self._configs = configs
        self._losses = np.array(losses)

    def observational_effect(self, param: str, value: Any) -> float:
        """Compute observational effect E[loss | param=value].

        Args:
            param: Parameter name.
            value: Parameter value.

        Returns:
            Expected loss given observation.
        """
        losses = [l for c, l in zip(self._configs, self._losses) if c.get(param) == value]
        if not losses:
            return np.mean(self._losses)
        return float(np.mean(losses))

    def interventional_effect(
        self,
        param: str,
        value: Any,
        adjust_for: list[str] | None = None,
    ) -> float:
        """Compute interventional effect E[loss | do(param=value)].

        Uses backdoor adjustment formula.

        Args:
            param: Parameter to intervene on.
            value: Value to set.
            adjust_for: Parameters to adjust for.

        Returns:
            Expected loss under intervention.
        """
        adjust_for = adjust_for or self.confounders

        if not adjust_for:
            return self.observational_effect(param, value)

        # Backdoor adjustment: E[Y | do(X=x)] = Σ_z E[Y | X=x, Z=z] * P(Z=z)
        # Group by adjustment set values
        adjustment_groups: dict[tuple, list[float]] = {}
        adjustment_counts: dict[tuple, int] = {}

        for config, loss in zip(self._configs, self._losses):
            if config.get(param) != value:
                continue

            z_vals = tuple(config.get(z) for z in adjust_for)
            if z_vals not in adjustment_groups:
                adjustment_groups[z_vals] = []
                adjustment_counts[z_vals] = 0
            adjustment_groups[z_vals].append(loss)

        # Count total for P(Z=z)
        z_marginal: dict[tuple, int] = {}
        for config in self._configs:
            z_vals = tuple(config.get(z) for z in adjust_for)
            z_marginal[z_vals] = z_marginal.get(z_vals, 0) + 1

        # Compute adjusted expectation
        total_count = len(self._configs)
        adjusted_effect = 0.0

        for z_vals, losses in adjustment_groups.items():
            if z_vals in z_marginal:
                p_z = z_marginal[z_vals] / total_count
                e_y_given_xz = np.mean(losses)
                adjusted_effect += e_y_given_xz * p_z

        return float(adjusted_effect) if adjusted_effect > 0 else np.mean(self._losses)

    def causal_importance(self, param: str) -> float:
        """Compute causal importance for a parameter.

        Measures the variance of E[loss | do(param=x)] across values.

        Args:
            param: Parameter name.

        Returns:
            Causal importance score.
        """
        # Get unique values
        values = {c.get(param) for c in self._configs}

        if len(values) < 2:
            return 0.0

        # Compute interventional effects for each value
        effects = [self.interventional_effect(param, v) for v in values]

        # Importance = variance of effects normalized
        total_var = np.var(self._losses)
        if total_var < 1e-10:
            return 0.0

        return float(np.var(effects) / total_var)

    def get_statistics(self) -> dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "type": "CausalImportanceAnalyzer",
            "num_confounders": len(self.confounders),
            "num_samples": len(self._configs),
        }


class StreamingfANOVA:
    """Online fANOVA with O(1) Incremental Updates.

    Uses running statistics to update importance estimates
    without refitting on all data.

    Attributes:
        param_names: Parameter names.
        window_size: Rolling window size (0 = all data).
    """

    def __init__(
        self,
        param_names: list[str] | None = None,
        window_size: int = 0,
    ) -> None:
        """Initialize streaming fANOVA.

        Args:
            param_names: Parameter names.
            window_size: Size of rolling window (0 = unbounded).
        """
        self.param_names = param_names or []
        self.window_size = window_size

        # Running statistics
        self._n: int = 0
        self._mean_loss: float = 0.0
        self._m2_total: float = 0.0  # Sum of squared deviations

        # Per-parameter group statistics
        self._group_stats: dict[str, dict[Any, dict[str, float]]] = {}

        # Rolling window buffer
        self._buffer: list[tuple[dict[str, Any], float]] = []

    def update(self, config: dict[str, Any], loss: float) -> dict[str, float]:
        """Update with a new observation.

        Args:
            config: New configuration.
            loss: Observed loss.

        Returns:
            Updated importance scores.
        """
        # Handle rolling window
        if self.window_size > 0 and len(self._buffer) >= self.window_size:
            old_config, old_loss = self._buffer.pop(0)
            self._remove_observation(old_config, old_loss)

        self._buffer.append((config, loss))

        # Update overall statistics (Welford's algorithm)
        self._n += 1
        delta = loss - self._mean_loss
        self._mean_loss += delta / self._n
        delta2 = loss - self._mean_loss
        self._m2_total += delta * delta2

        # Update per-parameter statistics
        if not self.param_names and config:
            self.param_names = list(config.keys())

        for param in self.param_names:
            val = config.get(param)
            if param not in self._group_stats:
                self._group_stats[param] = {}

            if val not in self._group_stats[param]:
                self._group_stats[param][val] = {"n": 0, "mean": 0.0, "m2": 0.0}

            group = self._group_stats[param][val]
            n = group["n"] + 1
            delta = loss - group["mean"]
            group["mean"] += delta / n
            group["m2"] += delta * (loss - group["mean"])
            group["n"] = n

        return self.get_importance()

    def _remove_observation(self, config: dict[str, Any], loss: float) -> None:
        """Remove old observation from statistics.

        Args:
            config: Configuration to remove.
            loss: Loss to remove.
        """
        if self._n <= 1:
            self._n = 0
            self._mean_loss = 0.0
            self._m2_total = 0.0
            return

        # Reverse Welford's update
        old_mean = self._mean_loss
        self._mean_loss = (self._mean_loss * self._n - loss) / (self._n - 1)
        delta = loss - self._mean_loss
        delta2 = loss - old_mean
        self._m2_total -= delta * delta2
        self._n -= 1

        # Update group stats
        for param in self.param_names:
            val = config.get(param)
            if param in self._group_stats and val in self._group_stats[param]:
                group = self._group_stats[param][val]
                if group["n"] <= 1:
                    del self._group_stats[param][val]
                else:
                    n = group["n"]
                    old_group_mean = group["mean"]
                    group["mean"] = (group["mean"] * n - loss) / (n - 1)
                    delta = loss - group["mean"]
                    delta2 = loss - old_group_mean
                    group["m2"] -= delta * delta2
                    group["n"] = n - 1

    def get_importance(self) -> dict[str, float]:
        """Get current importance estimates.

        Returns:
            Dictionary of param -> importance.
        """
        if self._n < 2:
            return dict.fromkeys(self.param_names, 0.0)

        total_var = self._m2_total / (self._n - 1)
        if total_var < 1e-10:
            return dict.fromkeys(self.param_names, 0.0)

        importance = {}
        for param in self.param_names:
            if param not in self._group_stats:
                importance[param] = 0.0
                continue

            groups = self._group_stats[param]
            if len(groups) < 2:
                importance[param] = 0.0
                continue

            # Between-group variance
            group_means = [g["mean"] for g in groups.values()]
            group_sizes = [g["n"] for g in groups.values()]

            between_var = np.average(
                [(m - self._mean_loss) ** 2 for m in group_means],
                weights=group_sizes,
            )

            importance[param] = float(between_var / total_var)

        return importance

    def get_statistics(self) -> dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "type": "StreamingfANOVA",
            "num_observations": self._n,
            "window_size": self.window_size,
            "num_params": len(self.param_names),
            "mean_loss": self._mean_loss,
        }


__all__ = [
    "GraphfANOVA",
    "CausalImportanceAnalyzer",
    "StreamingfANOVA",
    "DependencyEdge",
]
