# highnoon/training/fisher_layer_grouper.py
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

"""Fisher Information Based Layer Grouping for Adaptive Tuning.

This module provides layer grouping based on Quantum Fisher Information (QFIM)
characteristics. Layers with similar Fisher profiles respond similarly to
optimization changes, enabling:

1. Group-specific learning rates (LARS/LAMB style)
2. Proportional GaLore rank allocation
3. Earlier group-level barren plateau detection
4. Reduced HPO dimensionality (tune per-group, not per-layer)

Key Concepts:
    - Fisher Information measures sensitivity to parameter changes
    - High Fisher = near optimum, needs lower LR
    - Low Fisher = far from optimum, can tolerate higher LR
    - Groups are formed via k-means on log-Fisher values

Example:
    >>> grouper = FisherLayerGrouper(model)
    >>> grouper.update_fisher_estimates(gradients, variables)
    >>> groups = grouper.regroup_layers()
    >>> lr_scales = grouper.get_group_lr_scales()

Reference:
    Smart_Tuner_Upgrade.md - Section 6.3: Quantum Fisher Information Layer Grouping
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf

from highnoon.config import (
    FISHER_TN_RANK_MAX,
    FISHER_TN_RANK_MIN,
    FISHER_TN_RANK_SCALE,
    USE_FISHER_TN_RANKS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class FisherLayerGrouperConfig:
    """Configuration for Fisher-based layer grouping.

    Attributes:
        num_groups: Number of layer groups to create.
        regroup_frequency: Recompute groupings every N training steps.
        use_quantum_fisher: Use QFIM for quantum layers (vs classical).
        ema_decay: EMA decay factor for Fisher estimate updates.
        adaptive_lr_per_group: Apply group-specific learning rates.
        adaptive_galore_per_group: Apply group-specific GaLore ranks.
        min_samples_for_regroup: Minimum gradient samples before regrouping.
    """

    num_groups: int = 4
    regroup_frequency: int = 100
    use_quantum_fisher: bool = True
    ema_decay: float = 0.95
    adaptive_lr_per_group: bool = True
    adaptive_galore_per_group: bool = True
    min_samples_for_regroup: int = 10


# =============================================================================
# GROUP STATISTICS
# =============================================================================


@dataclass
class GroupStatistics:
    """Statistics for a layer group.

    Attributes:
        group_id: Identifier for this group.
        variable_names: Names of variables in this group.
        mean_fisher: Mean Fisher Information value.
        std_fisher: Standard deviation of Fisher values.
        mean_weight_norm: Mean weight norm for the group.
        lr_scale: Computed learning rate scale.
        galore_rank: Allocated GaLore rank.
        tt_rank: Allocated Tensor-Train rank for TTDense layers.
        tucker_ranks: Allocated Tucker ranks [R1, R2] for TuckerLayer.
        tensor_ring_rank: Allocated Tensor-Ring bond dimension.
    """

    group_id: int
    variable_names: list[str] = field(default_factory=list)
    mean_fisher: float = 0.0
    std_fisher: float = 0.0
    mean_weight_norm: float = 0.0
    lr_scale: float = 1.0
    galore_rank: int = 32
    # Phase 201.11: TN rank allocation
    tt_rank: int = 8  # Tensor-Train rank
    tucker_ranks: list[int] = field(default_factory=lambda: [8, 8])  # Tucker [R1, R2]
    tensor_ring_rank: int = 8  # TensorRing bond dimension


# =============================================================================
# FISHER LAYER GROUPER
# =============================================================================


class FisherLayerGrouper:
    """Groups layers by Quantum Fisher Information similarity.

    Layers with similar Fisher Information profiles respond similarly to
    learning rate and optimization changes. By grouping them, we can:

    1. Apply group-specific learning rates (LARS/LAMB style)
    2. Allocate GaLore rank budget proportionally
    3. Detect group-level barren plateaus earlier
    4. Reduce HPO dimensionality (tune per-group, not per-layer)

    The grouper uses k-means clustering on log-transformed Fisher values
    to form natural layer groups.

    Attributes:
        model: The Keras model being analyzed.
        config: Grouper configuration.

    Example:
        >>> grouper = FisherLayerGrouper(model, FisherLayerGrouperConfig(num_groups=4))
        >>> for step, (grads, vars) in enumerate(training_loop):
        ...     grouper.update_fisher_estimates(grads, vars)
        ...     if step % 100 == 0:
        ...         groups = grouper.regroup_layers()
        ...         lr_scales = grouper.get_group_lr_scales()
    """

    def __init__(
        self,
        model: tf.keras.Model,
        config: FisherLayerGrouperConfig | None = None,
    ):
        """Initialize Fisher Layer Grouper.

        Args:
            model: The Keras model to analyze.
            config: Grouper configuration.
        """
        self.model = model
        self.config = config or FisherLayerGrouperConfig()

        # Fisher estimates per variable
        self._fisher_estimates: dict[str, tf.Variable] = {}

        # Weight norms per variable (for LARS/LAMB)
        self._weight_norms: dict[str, float] = {}

        # Current groupings: variable_name -> group_id
        self._layer_groups: dict[str, int] = {}

        # Group-level statistics
        self._group_stats: dict[int, GroupStatistics] = {}

        # Sample counter for regrouping
        self._sample_count = 0
        self._last_regroup_step = 0

        # QNG integration (optional)
        self._qng: Any = None
        if self.config.use_quantum_fisher:
            self._init_qng()

        # Quantum layer name patterns
        self._quantum_layer_patterns = [
            "vqc",
            "quantum",
            "qmamba",
            "time_crystal",
            "zne",
            "qng",
        ]

        # Phase 201.S1: VQC gradient variance tracking for GaLore synergy
        self._vqc_gradient_variance: dict[str, float] = {}
        self._vqc_variance_boost: float = 1.5  # Fisher boost for high-variance VQC layers

        logger.info(
            "[FisherGrouper] Initialized: %d groups, use_qfim=%s, regroup_freq=%d",
            self.config.num_groups,
            self.config.use_quantum_fisher,
            self.config.regroup_frequency,
        )

    def _init_qng(self) -> None:
        """Initialize Quantum Natural Gradient for QFIM estimation."""
        try:
            from highnoon.training.quantum_gradient import QuantumNaturalGradient

            self._qng = QuantumNaturalGradient(enabled=True)
            logger.debug("[FisherGrouper] QNG integration enabled for QFIM")
        except ImportError:
            logger.warning(
                "[FisherGrouper] QuantumNaturalGradient not available, "
                "using classical Fisher estimation"
            )
            self._qng = None

    def _is_quantum_layer(self, variable_name: str) -> bool:
        """Check if a variable belongs to a quantum-enhanced layer.

        Args:
            variable_name: Name of the weight variable.

        Returns:
            True if this is a quantum layer variable.
        """
        name_lower = variable_name.lower()
        return any(pattern in name_lower for pattern in self._quantum_layer_patterns)

    def update_fisher_estimates(
        self,
        gradients: list[tf.Tensor],
        variables: list[tf.Variable],
    ) -> None:
        """Update Fisher Information estimates from gradients.

        Uses the gradient outer product approximation:
            F_ii ≈ E[(∂L/∂θ_i)²]

        For quantum layers with QNG enabled, uses the Quantum Fisher Information:
            F_ij = Re[⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩]

        Args:
            gradients: List of gradient tensors.
            variables: Corresponding trainable variables.
        """
        for grad, var in zip(gradients, variables):
            if grad is None:
                continue

            var_name = var.name

            # Compute squared gradient (Fisher diagonal approximation)
            fisher_diag = tf.reduce_mean(tf.square(grad))

            # For quantum layers, optionally use QFIM from QNG
            if (
                self._is_quantum_layer(var_name)
                and self.config.use_quantum_fisher
                and self._qng is not None
            ):
                # Check if QNG has QFIM state for this variable
                if hasattr(self._qng, "_qfim_states") and var_name in self._qng._qfim_states:
                    qfim_state = self._qng._qfim_states[var_name]
                    if hasattr(qfim_state, "qfim_diagonal"):
                        fisher_diag = tf.reduce_mean(qfim_state.qfim_diagonal)

            # EMA update
            fisher_val = float(
                fisher_diag.numpy() if hasattr(fisher_diag, "numpy") else fisher_diag
            )

            if var_name not in self._fisher_estimates:
                self._fisher_estimates[var_name] = tf.Variable(
                    fisher_val,
                    trainable=False,
                    name=f"fisher/{var_name.replace(':', '_')}",
                )
            else:
                current = float(self._fisher_estimates[var_name].numpy())
                updated = self.config.ema_decay * current + (1 - self.config.ema_decay) * fisher_val
                self._fisher_estimates[var_name].assign(updated)

            # Update weight norm
            self._weight_norms[var_name] = float(
                tf.norm(var).numpy() if hasattr(tf.norm(var), "numpy") else tf.norm(var)
            )

        self._sample_count += 1

    def register_vqc_variance(self, variable_name: str, variance: float) -> None:
        """Phase 201.S1: Register VQC gradient variance for GaLore synergy.

        High variance in VQC gradients indicates that the quantum layer is
        actively learning and needs more gradient capacity. This variance
        boosts the Fisher estimate for rank allocation.

        Args:
            variable_name: Name of the VQC-related variable.
            variance: Computed gradient variance from GaLore compressor.
        """
        # EMA update
        alpha = 0.1
        if variable_name in self._vqc_gradient_variance:
            old_var = self._vqc_gradient_variance[variable_name]
            variance = alpha * variance + (1 - alpha) * old_var
        self._vqc_gradient_variance[variable_name] = variance

        # Boost Fisher estimate for high-variance VQC layers
        if variable_name in self._fisher_estimates and variance > 0.1:
            current = float(self._fisher_estimates[variable_name].numpy())
            boosted = current * self._vqc_variance_boost
            self._fisher_estimates[variable_name].assign(boosted)
            logger.debug(
                "[FisherGrouper] VQC variance boost for %s: variance=%.4f, fisher %.4f -> %.4f",
                variable_name,
                variance,
                current,
                boosted,
            )

    def get_vqc_rank_allocation(self) -> dict[str, int]:
        """Phase 201.S1: Get VQC-aware rank allocations for GaLore.

        Returns rank allocations that are boosted for high-variance VQC layers.
        This provides the feedback loop from VQC gradients to GaLore rank.

        Returns:
            Dict mapping variable names to recommended GaLore ranks.
        """
        allocations = {}
        base_rank = 32

        for var_name, variance in self._vqc_gradient_variance.items():
            if variance > 0.1:  # High variance threshold
                # Linear interpolation: variance 0.1 -> base_rank, variance 1.0 -> 2x base_rank
                boost_factor = 1.0 + min(0.9, (variance - 0.1) / 0.9)
                allocations[var_name] = int(base_rank * boost_factor)
            else:
                allocations[var_name] = base_rank

        return allocations

    def regroup_layers(self, step: int = 0) -> dict[str, int]:
        """Recompute layer groupings using k-means on Fisher values.

        Groups are ordered by mean Fisher (highest Fisher = group 0).
        This allows applying different strategies to different groups:
        - Group 0 (high Fisher): Near optimum, lower LR
        - Group N (low Fisher): Far from optimum, higher LR

        Args:
            step: Current training step (for frequency check).

        Returns:
            Dictionary mapping variable names to group IDs (0 to num_groups-1).
        """
        # Check if we should regroup
        if step > 0 and step - self._last_regroup_step < self.config.regroup_frequency:
            return self._layer_groups

        if self._sample_count < self.config.min_samples_for_regroup:
            logger.debug(
                "[FisherGrouper] Not enough samples for regrouping (%d < %d)",
                self._sample_count,
                self.config.min_samples_for_regroup,
            )
            return self._layer_groups

        if not self._fisher_estimates:
            return {}

        self._last_regroup_step = step

        # Extract Fisher values
        var_names = list(self._fisher_estimates.keys())
        fisher_values = np.array(
            [float(self._fisher_estimates[name].numpy()) for name in var_names]
        )

        # Handle edge cases
        if len(var_names) < self.config.num_groups:
            # Fewer variables than groups - assign each to its own group
            self._layer_groups = {name: i for i, name in enumerate(var_names)}
            self._update_group_stats(fisher_values, list(range(len(var_names))))
            return self._layer_groups

        # Log-transform for better clustering (Fisher spans orders of magnitude)
        # Add small epsilon to avoid log(0)
        log_fisher = np.log1p(np.maximum(fisher_values, 1e-12))

        # K-means clustering
        labels = self._kmeans_1d(log_fisher, self.config.num_groups)

        # Order groups by mean Fisher (highest Fisher = group 0)
        group_means = [
            np.mean(fisher_values[labels == g]) if np.sum(labels == g) > 0 else 0.0
            for g in range(self.config.num_groups)
        ]
        group_order = np.argsort(group_means)[::-1]  # Descending
        label_map = {old: new for new, old in enumerate(group_order)}

        # Update groupings
        self._layer_groups = {
            var_names[i]: label_map.get(labels[i], 0) for i in range(len(var_names))
        }

        # Update group statistics
        remapped_labels = np.array([label_map.get(l, 0) for l in labels])
        self._update_group_stats(fisher_values, remapped_labels, var_names)

        logger.info(
            "[FisherGrouper] Regrouped %d variables into %d groups at step %d",
            len(var_names),
            self.config.num_groups,
            step,
        )

        return self._layer_groups

    def _kmeans_1d(self, data: np.ndarray, k: int, max_iters: int = 100) -> np.ndarray:
        """Simple 1D k-means clustering.

        Args:
            data: 1D array of values to cluster.
            k: Number of clusters.
            max_iters: Maximum iterations.

        Returns:
            Array of cluster labels.
        """
        n = len(data)
        if n == 0:
            return np.array([], dtype=np.int32)

        # Initialize centroids using k-means++ style
        centroids = [data[np.random.randint(n)]]
        for _ in range(k - 1):
            dists = np.array([min(abs(x - c) for c in centroids) for x in data])
            dists_sum = dists.sum()
            if dists_sum > 1e-10:
                probs = dists / dists_sum
            else:
                # All points equidistant, use uniform distribution
                probs = np.ones(n) / n
            # Ensure probabilities sum to 1
            probs = probs / probs.sum()
            centroids.append(data[np.random.choice(n, p=probs)])
        centroids = np.array(centroids)

        labels = np.zeros(n, dtype=np.int32)

        for _ in range(max_iters):
            # Assign points to nearest centroid
            old_labels = labels.copy()
            for i, x in enumerate(data):
                dists = np.abs(x - centroids)
                labels[i] = np.argmin(dists)

            # Check convergence
            if np.array_equal(labels, old_labels):
                break

            # Update centroids
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    centroids[j] = data[mask].mean()

        return labels

    def _update_group_stats(
        self,
        fisher_values: np.ndarray,
        labels: np.ndarray,
        var_names: list[str] | None = None,
    ) -> None:
        """Update group statistics after regrouping.

        Args:
            fisher_values: Array of Fisher values.
            labels: Array of group labels.
            var_names: Optional list of variable names.
        """
        self._group_stats.clear()

        # Ensure labels is a numpy array for proper boolean indexing
        labels_arr = np.array(labels) if not isinstance(labels, np.ndarray) else labels

        for g in range(self.config.num_groups):
            mask = labels_arr == g
            if not np.any(mask):
                continue

            group_fisher = fisher_values[mask]
            group_names = [var_names[i] for i in np.where(mask)[0]] if var_names else []

            # Compute weight norms for this group
            group_weight_norms = [self._weight_norms.get(name, 1.0) for name in group_names]

            self._group_stats[g] = GroupStatistics(
                group_id=g,
                variable_names=group_names,
                mean_fisher=float(np.mean(group_fisher)),
                std_fisher=float(np.std(group_fisher)) if len(group_fisher) > 1 else 0.0,
                mean_weight_norm=float(np.mean(group_weight_norms)) if group_weight_norms else 1.0,
            )

        # Compute LR scales, GaLore ranks, and TN ranks
        self._compute_lr_scales()
        self._compute_galore_ranks()
        if USE_FISHER_TN_RANKS:
            self._compute_tn_ranks()

    def _compute_lr_scales(self) -> None:
        """Compute LARS/LAMB style learning rate scales per group.

        Groups with higher Fisher Information get lower LR (near optimum).
        Groups with lower Fisher Information get higher LR (far from optimum).
        """
        if not self._group_stats:
            return

        max_fisher = max(s.mean_fisher for s in self._group_stats.values())

        for group_id, stats in self._group_stats.items():
            if max_fisher > 1e-10:
                # Inverse relationship: high Fisher -> low LR scale
                relative_fisher = stats.mean_fisher / max_fisher

                # LAMB-style trust ratio
                trust_ratio = 1.0 / (1.0 + relative_fisher)

                # Clamp to reasonable range
                stats.lr_scale = float(np.clip(trust_ratio, 0.1, 3.0))
            else:
                stats.lr_scale = 1.0

    def _compute_galore_ranks(self, total_budget: int = 256) -> None:
        """Compute GaLore rank allocation across groups.

        Allocates rank proportionally to Fisher Information.
        Higher Fisher groups get more rank to preserve gradient info.

        Args:
            total_budget: Total rank budget to allocate.
        """
        if not self._group_stats:
            return

        # Try to use native quantum GaLore ops
        try:
            from highnoon._native.ops.quantum_galore_ops import (
                allocate_block_ranks,
                compute_block_influence,
            )

            # Compute influence scores from group Fisher values
            fisher_values = tf.constant(
                [self._group_stats[g].mean_fisher for g in sorted(self._group_stats.keys())],
                dtype=tf.float32,
            )

            weight_norms = tf.constant(
                [self._group_stats[g].mean_weight_norm for g in sorted(self._group_stats.keys())],
                dtype=tf.float32,
            )

            influence = compute_block_influence(fisher_values, weight_norms)

            ranks = allocate_block_ranks(
                influence,
                total_rank_budget=total_budget,
                min_rank_per_block=8,
                critical_block_ids=tf.constant([0]),  # Highest Fisher group is critical
            )

            for i, g in enumerate(sorted(self._group_stats.keys())):
                self._group_stats[g].galore_rank = int(ranks[i])

            logger.debug("[FisherGrouper] Used quantum GaLore ops for rank allocation")
            return

        except (ImportError, RuntimeError, Exception) as e:
            logger.debug(
                "[FisherGrouper] Quantum GaLore ops unavailable (%s), using fallback",
                e,
            )

        # Fallback: proportional allocation based on Fisher
        total_fisher = sum(s.mean_fisher for s in self._group_stats.values())

        if total_fisher < 1e-10:
            # Uniform allocation
            per_group = total_budget // len(self._group_stats)
            for stats in self._group_stats.values():
                stats.galore_rank = max(8, per_group)
        else:
            # Proportional allocation
            for stats in self._group_stats.values():
                proportion = stats.mean_fisher / total_fisher
                stats.galore_rank = max(8, int(total_budget * proportion))

    def _compute_tn_ranks(self) -> None:
        """Compute Tensor Network rank allocation across groups.

        Phase 201.11: Fisher-based TN Rank Allocation.

        Allocates TT, Tucker, and TensorRing ranks proportionally to Fisher
        Information. Higher importance groups get larger ranks to preserve
        model capacity at compression boundaries.

        Formula:
            layer_importance = fisher_value / total_fisher
            layer_rank = min_rank + floor(importance * (max_rank - min_rank) * scale)
        """
        if not self._group_stats:
            return

        total_fisher = sum(s.mean_fisher for s in self._group_stats.values())

        if total_fisher < 1e-10:
            # Uniform allocation at minimum rank
            for stats in self._group_stats.values():
                stats.tt_rank = FISHER_TN_RANK_MIN
                stats.tucker_ranks = [FISHER_TN_RANK_MIN, FISHER_TN_RANK_MIN]
                stats.tensor_ring_rank = FISHER_TN_RANK_MIN
            logger.debug("[FisherGrouper] TN ranks set to minimum (no Fisher signal)")
            return

        rank_range = FISHER_TN_RANK_MAX - FISHER_TN_RANK_MIN

        for stats in self._group_stats.values():
            # Compute importance proportion (0 to 1)
            proportion = stats.mean_fisher / total_fisher

            # Apply scale factor and compute rank
            scaled_proportion = min(1.0, proportion * FISHER_TN_RANK_SCALE * len(self._group_stats))
            rank_delta = int(np.floor(scaled_proportion * rank_range))

            # Compute final ranks
            base_rank = FISHER_TN_RANK_MIN + rank_delta

            # TT rank (single value)
            stats.tt_rank = base_rank

            # Tucker ranks (slightly asymmetric for rectangular matrices)
            # Higher rank for first mode to prioritize output dimension
            stats.tucker_ranks = [
                min(FISHER_TN_RANK_MAX, base_rank + 2),
                base_rank,
            ]

            # TensorRing rank (use base rank)
            stats.tensor_ring_rank = base_rank

        logger.debug(
            "[FisherGrouper] TN ranks computed: TT=%s, Tucker=%s, TR=%s",
            [s.tt_rank for s in self._group_stats.values()],
            [s.tucker_ranks for s in self._group_stats.values()],
            [s.tensor_ring_rank for s in self._group_stats.values()],
        )

    def get_group_lr_scales(self) -> dict[int, float]:
        """Get learning rate scales per group.

        Returns:
            Dict mapping group ID to LR scale factor.
        """
        return {g: s.lr_scale for g, s in self._group_stats.items()}

    def get_group_galore_ranks(self) -> dict[int, int]:
        """Get GaLore ranks per group.

        Returns:
            Dict mapping group ID to GaLore rank.
        """
        return {g: s.galore_rank for g, s in self._group_stats.items()}

    def get_tt_ranks(self) -> dict[int, int]:
        """Get Tensor-Train ranks per group.

        Phase 201.11: Fisher-based TN rank allocation.

        Returns:
            Dict mapping group ID to TT rank.
        """
        return {g: s.tt_rank for g, s in self._group_stats.items()}

    def get_tucker_ranks(self) -> dict[int, list[int]]:
        """Get Tucker ranks per group.

        Phase 201.11: Fisher-based TN rank allocation.

        Returns:
            Dict mapping group ID to Tucker ranks [R1, R2].
        """
        return {g: s.tucker_ranks for g, s in self._group_stats.items()}

    def get_tensor_ring_ranks(self) -> dict[int, int]:
        """Get TensorRing ranks per group.

        Phase 201.11: Fisher-based TN rank allocation.

        Returns:
            Dict mapping group ID to TensorRing bond dimension.
        """
        return {g: s.tensor_ring_rank for g, s in self._group_stats.items()}

    def get_variable_tn_ranks(self, variable_name: str) -> dict[str, Any]:
        """Get TN ranks for a specific variable's group.

        Phase 201.11: Fisher-based TN rank allocation.

        Args:
            variable_name: Name of the variable.

        Returns:
            Dict with tt_rank, tucker_ranks, tensor_ring_rank for this variable's group.
        """
        group_id = self.get_variable_group(variable_name)
        if group_id in self._group_stats:
            stats = self._group_stats[group_id]
            return {
                "tt_rank": stats.tt_rank,
                "tucker_ranks": stats.tucker_ranks,
                "tensor_ring_rank": stats.tensor_ring_rank,
            }
        return {
            "tt_rank": FISHER_TN_RANK_MIN,
            "tucker_ranks": [FISHER_TN_RANK_MIN, FISHER_TN_RANK_MIN],
            "tensor_ring_rank": FISHER_TN_RANK_MIN,
        }

    def get_variable_group(self, variable_name: str) -> int:
        """Get group ID for a specific variable.

        Args:
            variable_name: Name of the variable.

        Returns:
            Group ID (defaults to 0 if not grouped).
        """
        return self._layer_groups.get(variable_name, 0)

    def get_variable_lr_scale(self, variable_name: str) -> float:
        """Get LR scale for a specific variable.

        Args:
            variable_name: Name of the variable.

        Returns:
            LR scale factor for this variable's group.
        """
        group_id = self.get_variable_group(variable_name)
        if group_id in self._group_stats:
            return self._group_stats[group_id].lr_scale
        return 1.0

    def get_fisher_estimates(self) -> dict[str, float]:
        """Get current Fisher estimates for all variables.

        Returns:
            Dict mapping variable names to Fisher values.
        """
        return {name: float(var.numpy()) for name, var in self._fisher_estimates.items()}

    def get_statistics(self) -> dict[str, Any]:
        """Get grouper statistics for logging/monitoring.

        Returns:
            Dictionary with current state and group statistics.
        """
        stats = {
            "num_groups": self.config.num_groups,
            "num_variables": len(self._fisher_estimates),
            "sample_count": self._sample_count,
            "last_regroup_step": self._last_regroup_step,
            "using_qfim": self.config.use_quantum_fisher and self._qng is not None,
        }

        # Add group-level stats
        stats["groups"] = {}
        for g, gs in self._group_stats.items():
            stats["groups"][g] = {
                "num_variables": len(gs.variable_names),
                "mean_fisher": gs.mean_fisher,
                "std_fisher": gs.std_fisher,
                "lr_scale": gs.lr_scale,
                "galore_rank": gs.galore_rank,
            }

        return stats

    def reset(self) -> None:
        """Reset all grouper state."""
        self._fisher_estimates.clear()
        self._weight_norms.clear()
        self._layer_groups.clear()
        self._group_stats.clear()
        self._sample_count = 0
        self._last_regroup_step = 0
        logger.info("[FisherGrouper] State reset")


__all__ = [
    "FisherLayerGrouper",
    "FisherLayerGrouperConfig",
    "GroupStatistics",
]
