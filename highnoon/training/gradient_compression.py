# highnoon/training/gradient_compression.py
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

"""Tensor-GaLore: Gradient Low-Rank Projection for Memory-Efficient Training.

This module implements Tensor-GaLore, a technique for reducing optimizer memory
by projecting high-rank gradients to low-rank subspaces using Tucker decomposition.
This can reduce optimizer state memory by 50-75% while maintaining training quality.

Key concepts:
- Project gradient tensor G to low-rank subspace via Tucker decomposition
- Store optimizer states (momentum, variance) only for low-rank representation
- Periodically update projection matrices to track gradient distribution

Mathematical Framework:
    For weight tensor W of shape (d1, d2, ..., dn):
    1. Compute gradient G = ∂L/∂W
    2. Decompose: G ≈ core × U1 × U2 × ... × Un (Tucker decomposition)
    3. Optimize in low-rank space: m̂, v̂ have shape of core tensor
    4. Reconstruct update: ΔW = core_update × U1 × U2 × ... × Un

Reference:
    "Tensor-GaLore: Memory-Efficient Training via Gradient Low-Rank Projection"
    (NeurIPS 2024 OPT Workshop)

Example:
    >>> compressor = TensorGaLoreCompressor(rank=32)
    >>> compressed_grad = compressor.compress(gradient)
    >>> # Optimizer updates work on compressed gradient
    >>> update = optimizer.apply(compressed_grad)
    >>> full_update = compressor.decompress(update)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf

from highnoon import config

logger = logging.getLogger(__name__)


@dataclass
class ProjectionState:
    """Stores projection matrices for a variable.

    Attributes:
        variable_name: Name of the weight variable.
        shape: Original shape of the weight tensor.
        factor_matrices: List of Tucker factor matrices (U_i for each mode).
        last_update_step: Step at which projection was last updated.
        compression_ratio: Actual compression ratio achieved.
    """

    variable_name: str
    shape: tuple[int, ...]
    factor_matrices: list[tf.Tensor]
    last_update_step: int = 0
    compression_ratio: float = 1.0

    def needs_update(self, current_step: int, update_gap: int) -> bool:
        """Check if projection matrices need to be updated."""
        return (current_step - self.last_update_step) >= update_gap


@dataclass
class TensorGaLoreCompressor:
    """Compresses gradients using Tucker decomposition for memory efficiency.

    Implements Tensor-GaLore algorithm for gradient low-rank projection.
    Works with both 2D (matrix) and higher-dimensional tensor weights.

    Attributes:
        rank: Target rank for each mode of the decomposition.
        update_proj_gap: Steps between projection matrix updates.
        scale: Scaling factor for gradient after decompression.
        enabled: Whether compression is active.
    """

    rank: int = field(default_factory=lambda: config.GALORE_RANK)
    update_proj_gap: int = field(default_factory=lambda: config.GALORE_UPDATE_PROJ_GAP)
    scale: float = field(default_factory=lambda: config.GALORE_SCALE)
    enabled: bool = field(default_factory=lambda: config.USE_TENSOR_GALORE)

    # Internal state
    _projections: dict[str, ProjectionState] = field(default_factory=dict)
    _step_counter: int = 0

    def __post_init__(self) -> None:
        """Initialize internal state after dataclass construction."""
        if not hasattr(self, "_projections") or self._projections is None:
            self._projections = {}
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0

    def _initialize_projection(
        self,
        variable_name: str,
        gradient: tf.Tensor,
    ) -> ProjectionState:
        """Initialize projection matrices for a new variable.

        Uses SVD to find initial low-rank subspace for each mode.

        Args:
            variable_name: Name of the weight variable.
            gradient: Gradient tensor to analyze.

        Returns:
            Initialized ProjectionState with factor matrices.
        """
        shape = gradient.shape.as_list()
        ndim = len(shape)

        factor_matrices = []

        if ndim == 2:
            # For 2D matrices, use standard SVD projection
            # tf.linalg.svd returns (s, u, v) where A = u @ diag(s) @ v.T
            d1, d2 = shape
            rank = min(self.rank, d1, d2)

            # Compute SVD: G = U @ S @ V^T
            s, u, v = tf.linalg.svd(gradient)

            if d1 >= d2:
                # Tall/square matrix: project rows using left singular vectors
                # G_c = U_r^T @ G (shape: rank x d2)
                factor_matrices.append(u[:, :rank])  # U: d1 x rank
                factor_matrices.append(None)  # No column projection needed
            else:
                # Wide matrix: project columns using right singular vectors
                # G_c = G @ V_r (shape: d1 x rank)
                # v from SVD has shape [d2, min(d1,d2)] - columns are right singular vectors
                factor_matrices.append(None)  # No row projection needed
                factor_matrices.append(v[:, :rank])  # V_r: d2 x rank

        else:
            # For higher-order tensors, use HOSVD-style mode-wise SVD
            for mode in range(ndim):
                mode_dim = shape[mode]
                mode_rank = min(self.rank, mode_dim)

                # Unfold tensor along this mode
                unfolded = self._tensor_unfold(gradient, mode)

                # SVD of unfolded matrix
                _, u, _ = tf.linalg.svd(unfolded)
                factor_matrices.append(u[:, :mode_rank])

        # Compute compression ratio
        original_size = np.prod(shape)
        compressed_size = self._compute_core_size(shape, self.rank)
        for _i, fm in enumerate(factor_matrices):
            if fm is not None:
                compressed_size += np.prod(fm.shape)
        compression_ratio = original_size / max(compressed_size, 1)

        state = ProjectionState(
            variable_name=variable_name,
            shape=tuple(shape),
            factor_matrices=factor_matrices,
            last_update_step=self._step_counter,
            compression_ratio=compression_ratio,
        )

        logger.info(
            "[GALORE] Initialized projection for '%s': shape=%s, rank=%d, " "compression=%.2fx",
            variable_name,
            shape,
            self.rank,
            compression_ratio,
        )

        return state

    def _tensor_unfold(self, tensor: tf.Tensor, mode: int) -> tf.Tensor:
        """Unfold tensor along specified mode (matricize).

        Args:
            tensor: Input tensor of shape (d0, d1, ..., dn-1).
            mode: Mode along which to unfold (0 to n-1).

        Returns:
            Unfolded matrix of shape (d_mode, prod(other dims)).
        """
        ndim = len(tensor.shape)
        perm = [mode] + [i for i in range(ndim) if i != mode]
        permuted = tf.transpose(tensor, perm)

        mode_dim = tensor.shape[mode]
        np.prod([tensor.shape[i].numpy() for i in range(ndim) if i != mode])

        return tf.reshape(permuted, [mode_dim, -1])

    def _compute_core_size(self, shape: tuple[int, ...], rank: int) -> int:
        """Compute size of Tucker core tensor."""
        core_dims = [min(rank, d) for d in shape]
        return int(np.prod(core_dims))

    def compress(
        self,
        gradient: tf.Tensor,
        variable: tf.Variable,
    ) -> tuple[tf.Tensor, str]:
        """Compress gradient to low-rank representation.

        Args:
            gradient: Full gradient tensor.
            variable: Associated weight variable.

        Returns:
            Tuple of (compressed gradient, variable name).
        """
        if not self.enabled:
            return gradient, variable.name

        var_name = variable.name
        shape = gradient.shape.as_list()
        ndim = len(shape)

        # Initialize or update projection if needed
        if var_name not in self._projections:
            self._projections[var_name] = self._initialize_projection(var_name, gradient)
        elif self._projections[var_name].needs_update(self._step_counter, self.update_proj_gap):
            self._projections[var_name] = self._initialize_projection(var_name, gradient)

        state = self._projections[var_name]

        if ndim == 2:
            # 2D case: project to low-rank subspace
            u, v = state.factor_matrices[0], state.factor_matrices[1]
            if u is not None:
                # Tall/square matrix: G_c = U^T @ G (shape: rank x d2)
                compressed = tf.matmul(u, gradient, transpose_a=True)
            else:
                # Wide matrix: G_c = G @ V_r (shape: d1 x rank)
                # v is V_r (d2 x rank), so G @ v -> (d1 x d2) @ (d2 x rank) = (d1 x rank)
                compressed = tf.matmul(gradient, v)
        else:
            # Higher-order: mode-wise projection (Tucker core)
            compressed = gradient
            for mode in range(ndim):
                u = state.factor_matrices[mode]
                # Contract tensor with U^T along mode
                compressed = self._mode_product(compressed, u, mode, transpose_u=True)

        return compressed, var_name

    def decompress(
        self,
        compressed_update: tf.Tensor,
        variable_name: str,
    ) -> tf.Tensor:
        """Decompress low-rank update to full tensor.

        Args:
            compressed_update: Update in low-rank space.
            variable_name: Name of the associated variable.

        Returns:
            Full-rank update tensor.
        """
        if not self.enabled or variable_name not in self._projections:
            return compressed_update

        state = self._projections[variable_name]
        shape = state.shape
        ndim = len(shape)

        if ndim == 2:
            u, v = state.factor_matrices[0], state.factor_matrices[1]
            if u is not None:
                # Tall/square matrix: ΔW = U @ G_c (shape: d1 x d2)
                full_update = tf.matmul(u, compressed_update)
            else:
                # Wide matrix: ΔW = G_c @ V_r^T (shape: d1 x d2)
                # G_c is (d1 x rank), v is V_r (d2 x rank)
                # G_c @ v^T -> (d1 x rank) @ (rank x d2) = (d1 x d2)
                full_update = tf.matmul(compressed_update, v, transpose_b=True)
        else:
            # Higher-order: mode-wise reconstruction
            full_update = compressed_update
            for mode in range(ndim):
                u = state.factor_matrices[mode]
                # Contract with U along mode (no transpose)
                full_update = self._mode_product(full_update, u, mode, transpose_u=False)

        return full_update * self.scale

    def _mode_product(
        self,
        tensor: tf.Tensor,
        matrix: tf.Tensor,
        mode: int,
        transpose_u: bool = False,
    ) -> tf.Tensor:
        """Compute mode-n product of tensor with matrix.

        Args:
            tensor: Input tensor.
            matrix: Matrix to multiply (shape k x I_n or I_n x k).
            mode: Mode along which to multiply.
            transpose_u: Whether to transpose matrix before multiplication.

        Returns:
            Result tensor with shape modified along mode.
        """
        ndim = len(tensor.shape)

        # Unfold, multiply, refold
        unfolded = self._tensor_unfold(tensor, mode)

        if transpose_u:
            result = tf.matmul(matrix, unfolded, transpose_a=True)
        else:
            result = tf.matmul(matrix, unfolded)

        # Compute new shape
        new_shape = list(tensor.shape.as_list())
        new_shape[mode] = result.shape[0]

        # Refold
        perm_back = list(range(1, mode + 1)) + [0] + list(range(mode + 1, ndim))
        result = tf.reshape(
            result, [new_shape[mode]] + [new_shape[i] for i in range(ndim) if i != mode]
        )
        result = tf.transpose(result, perm_back)

        return result

    def step(self) -> None:
        """Increment step counter. Call once per training step."""
        self._step_counter += 1

    def get_statistics(self) -> dict[str, Any]:
        """Get compression statistics for all tracked variables."""
        stats = {
            "enabled": self.enabled,
            "rank": self.rank,
            "step_counter": self._step_counter,
            "num_variables": len(self._projections),
            "variables": {},
        }

        total_original = 0
        total_compressed = 0

        for name, state in self._projections.items():
            original_size = np.prod(state.shape)
            total_original += original_size
            total_compressed += original_size / max(state.compression_ratio, 1)

            stats["variables"][name] = {
                "shape": state.shape,
                "compression_ratio": state.compression_ratio,
                "last_update_step": state.last_update_step,
            }

        if total_original > 0:
            stats["overall_compression_ratio"] = total_original / max(total_compressed, 1)
        else:
            stats["overall_compression_ratio"] = 1.0

        return stats

    def reset(self) -> None:
        """Reset all compression state."""
        self._projections.clear()
        self._step_counter = 0
        logger.info("[GALORE] Compressor state reset")


class GaLoreOptimizerWrapper:
    """Wrapper that adds GaLore compression to any optimizer.

    This wrapper intercepts gradients, compresses them, applies optimizer
    updates in the low-rank space, then decompresses the updates.

    Example:
        >>> base_optimizer = tf.keras.optimizers.Adam()
        >>> galore_optimizer = GaLoreOptimizerWrapper(base_optimizer, rank=32)
        >>> galore_optimizer.apply_gradients(zip(grads, vars))
    """

    def __init__(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        rank: int = 32,
        update_gap: int = 200,
        scale: float = 0.25,
    ):
        """Initialize GaLore wrapper.

        Args:
            optimizer: Base optimizer to wrap.
            rank: Projection rank for GaLore.
            update_gap: Steps between projection updates.
            scale: Gradient scaling factor.
        """
        self.optimizer = optimizer
        self.compressor = TensorGaLoreCompressor(
            rank=rank,
            update_proj_gap=update_gap,
            scale=scale,
            enabled=True,
        )

        # Store compressed optimizer states
        self._compressed_states: dict[str, dict[str, tf.Variable]] = {}

    @property
    def learning_rate(self):
        """Passthrough to base optimizer learning rate."""
        return self.optimizer.learning_rate

    def apply_gradients(
        self,
        grads_and_vars: list[tuple[tf.Tensor, tf.Variable]],
    ) -> None:
        """Apply gradients with GaLore compression.

        Args:
            grads_and_vars: List of (gradient, variable) pairs.
        """
        for grad, var in grads_and_vars:
            if grad is None:
                continue

            # Compress gradient
            compressed_grad, var_name = self.compressor.compress(grad, var)

            # Initialize compressed optimizer state if needed
            if var_name not in self._compressed_states:
                self._initialize_compressed_state(var_name, compressed_grad)

            # Apply optimizer update in compressed space
            compressed_state = self._compressed_states[var_name]
            compressed_update = self._apply_adam_step(
                compressed_grad,
                compressed_state["m"],
                compressed_state["v"],
            )

            # Decompress and apply to variable
            full_update = self.compressor.decompress(compressed_update, var_name)
            var.assign_sub(full_update)

        self.compressor.step()

    def _initialize_compressed_state(
        self,
        var_name: str,
        compressed_grad: tf.Tensor,
    ) -> None:
        """Initialize optimizer state for compressed space."""
        self._compressed_states[var_name] = {
            "m": tf.Variable(
                tf.zeros_like(compressed_grad),
                trainable=False,
                name=f"galore_m/{var_name}",
            ),
            "v": tf.Variable(
                tf.zeros_like(compressed_grad),
                trainable=False,
                name=f"galore_v/{var_name}",
            ),
        }

    def _apply_adam_step(
        self,
        grad: tf.Tensor,
        m: tf.Variable,
        v: tf.Variable,
    ) -> tf.Tensor:
        """Apply Adam update step.

        Args:
            grad: Gradient tensor.
            m: First moment estimate.
            v: Second moment estimate.

        Returns:
            Update tensor.
        """
        # Get Adam hyperparameters
        lr = float(self.optimizer.learning_rate)
        beta_1 = float(getattr(self.optimizer, "beta_1", 0.9))
        beta_2 = float(getattr(self.optimizer, "beta_2", 0.999))
        epsilon = float(getattr(self.optimizer, "epsilon", 1e-7))

        # Update biased first moment estimate
        m.assign(beta_1 * m + (1 - beta_1) * grad)

        # Update biased second moment estimate
        v.assign(beta_2 * v + (1 - beta_2) * tf.square(grad))

        # Compute update (no bias correction for simplicity)
        update = lr * m / (tf.sqrt(v) + epsilon)

        return update


__all__ = [
    "TensorGaLoreCompressor",
    "GaLoreOptimizerWrapper",
    "ProjectionState",
]
