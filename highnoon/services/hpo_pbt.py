# highnoon/services/hpo_pbt.py
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

"""Population-Based Training Enhancements for QAHPO.

Enterprise Enhancement Phase 5: Advanced PBT features including:
    - PBT with Backtracking (checkpoint tree)
    - Neural PBT (hypernetwork-based schedule generation)
    - Multi-Objective PBT (Pareto optimization)

References:
    - Jaderberg et al., "Population Based Training" (2017)
    - Parker-Holder et al., "Provably Efficient Online Hyperparameter Optimization" (2020)

Example:
    >>> pbt = PBTBacktrackScheduler(population_size=8)
    >>> pbt.initialize_population(search_space)
    >>> for step in range(max_steps):
    ...     configs = pbt.get_configs()
    ...     results = train_step(configs)
    ...     pbt.report_results(results)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CheckpointNode:
    """Node in the checkpoint tree for PBT with backtracking.

    Attributes:
        node_id: Unique node identifier.
        parent_id: Parent node ID (None for root).
        config: Hyperparameter configuration at this node.
        step: Training step when checkpoint was created.
        loss: Loss at this checkpoint.
        visits: Number of times this node was visited.
        children: List of child node IDs.
    """

    node_id: str
    parent_id: str | None
    config: dict[str, Any]
    step: int
    loss: float
    visits: int = 0
    children: list[str] = field(default_factory=list)

    def ucb_score(self, total_visits: int, exploration_weight: float = 1.41) -> float:
        """Compute UCB score for node selection.

        Args:
            total_visits: Total visits across all nodes.
            exploration_weight: Exploration vs exploitation tradeoff.

        Returns:
            UCB score (higher = more promising).
        """
        if self.visits == 0:
            return float("inf")

        exploitation = -self.loss  # Lower loss = higher exploitation value
        exploration = exploration_weight * math.sqrt(math.log(total_visits + 1) / self.visits)
        return exploitation + exploration


class PBTBacktrackScheduler:
    """Population-Based Training with Backtracking.

    Maintains a tree of checkpoints instead of just the current population,
    allowing backtracking to previous promising configurations when the
    current path stagnates.

    Attributes:
        population_size: Number of parallel workers.
        exploit_threshold: Threshold for exploitation (top fraction).
        explore_rate: Probability of exploring new mutations.
        backtrack_threshold: Steps without improvement before backtracking.
        max_tree_depth: Maximum depth of checkpoint tree.
    """

    def __init__(
        self,
        population_size: int = 8,
        exploit_threshold: float = 0.2,
        explore_rate: float = 0.4,
        backtrack_threshold: int = 10,
        max_tree_depth: int = 50,
        seed: int = 42,
    ) -> None:
        """Initialize PBT with backtracking.

        Args:
            population_size: Number of parallel workers.
            exploit_threshold: Top fraction for exploitation.
            explore_rate: Mutation probability.
            backtrack_threshold: Steps before considering backtrack.
            max_tree_depth: Maximum checkpoint tree depth.
            seed: Random seed.
        """
        self.population_size = population_size
        self.exploit_threshold = exploit_threshold
        self.explore_rate = explore_rate
        self.backtrack_threshold = backtrack_threshold
        self.max_tree_depth = max_tree_depth

        self._rng = np.random.default_rng(seed)
        self._checkpoint_tree: dict[str, CheckpointNode] = {}
        self._current_nodes: list[str] = []  # Node IDs for current population
        self._node_counter: int = 0
        self._total_visits: int = 0
        self._best_loss: float = float("inf")
        self._best_config: dict[str, Any] | None = None
        self._steps_without_improvement: int = 0

    def _create_node(
        self,
        config: dict[str, Any],
        step: int,
        loss: float,
        parent_id: str | None = None,
    ) -> str:
        """Create a new checkpoint node.

        Args:
            config: Configuration at this checkpoint.
            step: Training step.
            loss: Loss value.
            parent_id: Parent node ID.

        Returns:
            New node ID.
        """
        node_id = f"node_{self._node_counter}"
        self._node_counter += 1

        node = CheckpointNode(
            node_id=node_id,
            parent_id=parent_id,
            config=config.copy(),
            step=step,
            loss=loss,
        )

        self._checkpoint_tree[node_id] = node

        # Update parent's children
        if parent_id and parent_id in self._checkpoint_tree:
            self._checkpoint_tree[parent_id].children.append(node_id)

        return node_id

    def initialize_population(
        self,
        configs: list[dict[str, Any]],
        initial_losses: list[float] | None = None,
    ) -> None:
        """Initialize the population with starting configurations.

        Args:
            configs: Initial configurations for each worker.
            initial_losses: Optional initial losses.
        """
        if len(configs) < self.population_size:
            # Pad with copies
            while len(configs) < self.population_size:
                configs.append(configs[len(configs) % len(configs)].copy())

        # Create root nodes for each worker
        self._current_nodes = []
        for i, config in enumerate(configs[: self.population_size]):
            loss = initial_losses[i] if initial_losses else float("inf")
            node_id = self._create_node(config, step=0, loss=loss, parent_id=None)
            self._current_nodes.append(node_id)

        logger.info(
            f"[PBT-Backtrack] Initialized population with {len(self._current_nodes)} workers"
        )

    def get_configs(self) -> list[dict[str, Any]]:
        """Get current configurations for all workers.

        Returns:
            List of configurations.
        """
        return [self._checkpoint_tree[node_id].config.copy() for node_id in self._current_nodes]

    def report_results(
        self,
        losses: list[float],
        step: int,
    ) -> None:
        """Report training results and update population.

        Args:
            losses: Loss values for each worker.
            step: Current training step.
        """
        if len(losses) != len(self._current_nodes):
            logger.warning("[PBT-Backtrack] Mismatch between losses and population size")
            return

        # Update current nodes with new losses
        for node_id, loss in zip(self._current_nodes, losses):
            self._checkpoint_tree[node_id].loss = loss
            self._checkpoint_tree[node_id].visits += 1
            self._total_visits += 1

        # Check for improvement
        min_loss = min(losses)
        if min_loss < self._best_loss:
            self._best_loss = min_loss
            best_idx = losses.index(min_loss)
            self._best_config = self._checkpoint_tree[self._current_nodes[best_idx]].config.copy()
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1

        # Sort workers by loss
        indexed_losses = list(enumerate(losses))
        indexed_losses.sort(key=lambda x: x[1])

        # Determine exploit boundary
        num_exploit = max(1, int(len(losses) * self.exploit_threshold))
        top_indices = [idx for idx, _ in indexed_losses[:num_exploit]]
        bottom_indices = [idx for idx, _ in indexed_losses[-num_exploit:]]

        # Perform exploit/explore
        new_nodes = list(self._current_nodes)

        for bottom_idx in bottom_indices:
            if self._rng.random() < self.explore_rate:
                # Pick a top performer to exploit
                top_idx = self._rng.choice(top_indices)
                top_node = self._checkpoint_tree[self._current_nodes[top_idx]]

                # Should we backtrack?
                if self._steps_without_improvement >= self.backtrack_threshold:
                    # Use UCB to select from tree
                    ancestor = self._select_backtrack_node(top_node.node_id)
                    if ancestor:
                        base_config = self._checkpoint_tree[ancestor].config.copy()
                        logger.info(f"[PBT-Backtrack] Backtracking to {ancestor}")
                    else:
                        base_config = top_node.config.copy()
                else:
                    base_config = top_node.config.copy()

                # Mutate configuration
                mutated_config = self._perturb(base_config)

                # Create new node in tree
                parent_id = self._current_nodes[top_idx]
                new_node_id = self._create_node(
                    mutated_config, step=step, loss=float("inf"), parent_id=parent_id
                )
                new_nodes[bottom_idx] = new_node_id

        self._current_nodes = new_nodes

    def _select_backtrack_node(self, current_node_id: str) -> str | None:
        """Select a node to backtrack to using UCB.

        Args:
            current_node_id: Current node ID.

        Returns:
            Selected ancestor node ID or None.
        """
        # Get ancestors
        ancestors = []
        node_id = current_node_id
        while node_id:
            node = self._checkpoint_tree.get(node_id)
            if node and node.parent_id:
                ancestors.append(node.parent_id)
                node_id = node.parent_id
            else:
                break

        if not ancestors:
            return None

        # Select using UCB
        scores = [
            (anc, self._checkpoint_tree[anc].ucb_score(self._total_visits))
            for anc in ancestors
            if anc in self._checkpoint_tree
        ]

        if not scores:
            return None

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def _perturb(self, config: dict[str, Any]) -> dict[str, Any]:
        """Perturb a configuration.

        Args:
            config: Configuration to perturb.

        Returns:
            Mutated configuration.
        """
        mutated = config.copy()

        for key, val in mutated.items():
            if key.startswith("_"):
                continue

            if isinstance(val, float):
                # Multiply by 0.8 or 1.2
                factor = self._rng.choice([0.8, 1.2])
                mutated[key] = val * factor
            elif isinstance(val, int):
                # Add -1, 0, or +1
                delta = self._rng.choice([-1, 0, 1])
                mutated[key] = max(1, val + delta)

        return mutated

    def get_best(self) -> tuple[dict[str, Any] | None, float]:
        """Get the best configuration found.

        Returns:
            Tuple of (best_config, best_loss).
        """
        return self._best_config, self._best_loss

    def get_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with scheduler state.
        """
        return {
            "type": "PBTBacktrack",
            "population_size": self.population_size,
            "total_nodes": len(self._checkpoint_tree),
            "total_visits": self._total_visits,
            "best_loss": self._best_loss if self._best_loss < float("inf") else None,
            "steps_without_improvement": self._steps_without_improvement,
            "tree_depth": self._get_max_depth(),
        }

    def _get_max_depth(self) -> int:
        """Get maximum depth of checkpoint tree."""
        max_depth = 0
        for node_id, _node in self._checkpoint_tree.items():
            depth = 0
            current = node_id
            while current:
                depth += 1
                parent = self._checkpoint_tree.get(current)
                current = parent.parent_id if parent else None
            max_depth = max(max_depth, depth)
        return max_depth


@dataclass
class ParetoPoint:
    """Point in multi-objective Pareto front.

    Attributes:
        config: Configuration.
        objectives: Dictionary of objective values (all minimized).
        dominated: Whether this point is dominated.
        crowding_distance: NSGA-II crowding distance.
    """

    config: dict[str, Any]
    objectives: dict[str, float]
    dominated: bool = False
    crowding_distance: float = 0.0


class MultiObjectivePBT:
    """Multi-Objective Population-Based Training.

    Optimizes multiple objectives (e.g., loss, latency, memory) using
    Pareto dominance and NSGA-II style crowding distance selection.

    Attributes:
        population_size: Number of parallel workers.
        objective_names: Names of objectives to optimize.
    """

    def __init__(
        self,
        population_size: int = 16,
        objective_names: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize multi-objective PBT.

        Args:
            population_size: Number of parallel workers.
            objective_names: Names of objectives (default: ["loss"]).
            seed: Random seed.
        """
        self.population_size = population_size
        self.objective_names = objective_names or ["loss"]
        self._rng = np.random.default_rng(seed)

        self._population: list[ParetoPoint] = []
        self._pareto_front: list[ParetoPoint] = []

    def initialize(self, configs: list[dict[str, Any]]) -> None:
        """Initialize population.

        Args:
            configs: Initial configurations.
        """
        self._population = []
        for config in configs[: self.population_size]:
            point = ParetoPoint(
                config=config.copy(),
                objectives={name: float("inf") for name in self.objective_names},
            )
            self._population.append(point)

        logger.info(
            f"[MO-PBT] Initialized with {len(self._population)} workers, "
            f"objectives: {self.objective_names}"
        )

    def get_configs(self) -> list[dict[str, Any]]:
        """Get current configurations."""
        return [p.config.copy() for p in self._population]

    def report_results(self, objectives: list[dict[str, float]]) -> None:
        """Report multi-objective results.

        Args:
            objectives: List of objective dictionaries for each worker.
        """
        if len(objectives) != len(self._population):
            return

        # Update objectives
        for point, obj in zip(self._population, objectives):
            point.objectives = obj.copy()

        # Compute Pareto dominance
        self._compute_pareto_front()

        # Compute crowding distance
        self._compute_crowding_distance()

        # Selection and mutation
        self._evolve()

    def _dominates(self, a: ParetoPoint, b: ParetoPoint) -> bool:
        """Check if point a dominates point b (all objectives minimized)."""
        all_leq = all(
            a.objectives.get(name, float("inf")) <= b.objectives.get(name, float("inf"))
            for name in self.objective_names
        )
        any_lt = any(
            a.objectives.get(name, float("inf")) < b.objectives.get(name, float("inf"))
            for name in self.objective_names
        )
        return all_leq and any_lt

    def _compute_pareto_front(self) -> None:
        """Compute Pareto front from population."""
        for i, point in enumerate(self._population):
            point.dominated = False
            for j, other in enumerate(self._population):
                if i != j and self._dominates(other, point):
                    point.dominated = True
                    break

        self._pareto_front = [p for p in self._population if not p.dominated]

    def _compute_crowding_distance(self) -> None:
        """Compute NSGA-II crowding distance for Pareto front."""
        front = self._pareto_front
        if len(front) <= 2:
            for p in front:
                p.crowding_distance = float("inf")
            return

        # Initialize
        for p in front:
            p.crowding_distance = 0.0

        # For each objective
        for obj_name in self.objective_names:
            sorted_front = sorted(front, key=lambda p: p.objectives.get(obj_name, 0))

            # Boundary points get infinite distance
            sorted_front[0].crowding_distance = float("inf")
            sorted_front[-1].crowding_distance = float("inf")

            # Get objective range
            obj_min = sorted_front[0].objectives.get(obj_name, 0)
            obj_max = sorted_front[-1].objectives.get(obj_name, 0)
            obj_range = obj_max - obj_min if obj_max > obj_min else 1.0

            # Compute intermediate distances
            for i in range(1, len(sorted_front) - 1):
                prev_obj = sorted_front[i - 1].objectives.get(obj_name, 0)
                next_obj = sorted_front[i + 1].objectives.get(obj_name, 0)
                sorted_front[i].crowding_distance += (next_obj - prev_obj) / obj_range

    def _evolve(self) -> None:
        """Evolve population using Pareto ranking and crowding."""
        # Sort by (rank, -crowding_distance)
        ranked = sorted(
            self._population,
            key=lambda p: (p.dominated, -p.crowding_distance),
        )

        # Keep top half as parents
        num_parents = self.population_size // 2
        parents = ranked[:num_parents]

        # Generate offspring
        offspring = []
        for _ in range(self.population_size - num_parents):
            parent = self._rng.choice(parents)
            child_config = self._mutate(parent.config)
            child = ParetoPoint(
                config=child_config,
                objectives={name: float("inf") for name in self.objective_names},
            )
            offspring.append(child)

        self._population = parents + offspring

    def _mutate(self, config: dict[str, Any]) -> dict[str, Any]:
        """Mutate a configuration."""
        mutated = config.copy()
        for key, val in mutated.items():
            if key.startswith("_"):
                continue
            if isinstance(val, float) and self._rng.random() < 0.3:
                mutated[key] = val * self._rng.uniform(0.8, 1.2)
            elif isinstance(val, int) and self._rng.random() < 0.3:
                mutated[key] = max(1, val + self._rng.choice([-1, 0, 1]))
        return mutated

    def get_pareto_front(self) -> list[dict[str, Any]]:
        """Get current Pareto front configurations."""
        return [{"config": p.config, "objectives": p.objectives} for p in self._pareto_front]

    def get_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "type": "MultiObjectivePBT",
            "population_size": len(self._population),
            "pareto_front_size": len(self._pareto_front),
            "objectives": self.objective_names,
        }


__all__ = [
    "PBTBacktrackScheduler",
    "MultiObjectivePBT",
    "CheckpointNode",
    "ParetoPoint",
]
