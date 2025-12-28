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
from highnoon.config import (
    GALORE_TT_MANIFOLD_PROJECTION,
    GALORE_TT_PROJECTION_EPS,
    GALORE_VQC_AWARE,
    GALORE_VQC_VARIANCE_BOOST,
)

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
        # Phase 130.2: VQC gradient variance tracking for adaptive rank
        self._vqc_gradient_variance: dict[str, float] = {}
        # S20: Barren plateau awareness - skip compression or boost rank during recovery
        self._barren_plateau_active: bool = False
        self._bp_rank_boost: float = 1.5  # Multiply rank by this during BP recovery
        # Phase 201.12: TT layer detection patterns
        self._tt_layer_patterns: list[str] = [
            "tt_core",
            "tt_dense",
            "ttlayer",
            "tensor_train",
            "superposition_tt",
            "tt_ffn",
            "tt_lm_head",
            "tt_embedding",
        ]
        # Track TT layers detected for statistics
        self._tt_layers_detected: set[str] = set()

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

            # Phase 130.2: Boost rank for high-variance VQC layers
            if GALORE_VQC_AWARE and variable_name in self._vqc_gradient_variance:
                vqc_var = self._vqc_gradient_variance[variable_name]
                if vqc_var > 0.1:  # High variance threshold
                    rank = min(int(rank * GALORE_VQC_VARIANCE_BOOST), d1, d2)
                    logger.debug(
                        "[GALORE] Boosted rank for VQC layer %s: variance=%.4f, new_rank=%d",
                        variable_name,
                        vqc_var,
                        rank,
                    )

            # S20: Boost rank during barren plateau recovery to preserve gradient info
            if self._barren_plateau_active:
                boosted_rank = min(int(rank * self._bp_rank_boost), d1, d2)
                if boosted_rank > rank:
                    logger.debug(
                        "[GALORE-S20] BP boost: %d → %d for %s",
                        rank,
                        boosted_rank,
                        variable_name,
                    )
                    rank = boosted_rank

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
            # Higher-order tensors (3D+): Skip projection - will pass through unchanged
            # HOSVD mode-product has dimension tracking issues that need fixing
            # TODO: Implement proper HOSVD with correct dimension tracking
            logger.debug(
                "[GALORE] Skipping projection init for %dD tensor '%s'", ndim, variable_name
            )
            # Return minimal state - compression/decompression will skip this
            return ProjectionState(
                variable_name=variable_name,
                shape=tuple(shape),
                factor_matrices=[],  # Empty - signals skip
                last_update_step=self._step_counter,
                compression_ratio=1.0,  # No compression
            )

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
        # Use -1 for dynamic reshape to infer the product of other dimensions
        return tf.reshape(permuted, [mode_dim, -1])

    def _compute_core_size(self, shape: tuple[int, ...], rank: int) -> int:
        """Compute size of Tucker core tensor."""
        core_dims = [min(rank, d) for d in shape]
        return int(np.prod(core_dims))

    def _is_tt_layer(self, variable_name: str) -> bool:
        """Check if a variable belongs to a TT-decomposed layer.

        Phase 201.12: TT layer detection for manifold projection.

        Args:
            variable_name: Name of the weight variable.

        Returns:
            True if this is a TT layer variable.
        """
        name_lower = variable_name.lower()
        return any(pattern in name_lower for pattern in self._tt_layer_patterns)

    def _project_to_tt_tangent(
        self,
        gradient: tf.Tensor,
        variable: tf.Variable,
    ) -> tf.Tensor:
        """Project gradient to TT tangent space for TT-decomposed layers.

        Phase 201.12: GaLore TT-Manifold Projection.

        For TT cores G_k, projects gradient dG_k to the tangent space of the
        TT manifold to ensure updates stay compatible with TT structure:

            dG_k_projected = dG_k - G_k @ (G_k^T @ G_k)^{-1} @ (G_k^T @ dG_k)

        This removes components that would take the TT core out of the manifold,
        improving convergence by respecting the TT geometry.

        Args:
            gradient: Gradient tensor for a TT core.
            variable: Associated TT core weight variable.

        Returns:
            Projected gradient in TT tangent space.
        """
        if not GALORE_TT_MANIFOLD_PROJECTION:
            return gradient

        # Get the current TT core value
        G = variable
        shape = gradient.shape.as_list()
        ndim = len(shape)

        if ndim == 2:
            # 2D TT core: [r_in, d × r_out] or similar
            # Project to tangent space of low-rank manifold

            # Compute G^T @ G (normal equations matrix)
            GtG = tf.matmul(G, G, transpose_a=True)

            # Regularize for numerical stability
            reg = GALORE_TT_PROJECTION_EPS * tf.eye(tf.shape(GtG)[0], dtype=GtG.dtype)
            GtG_reg = GtG + reg

            # Compute G^T @ dG
            GtdG = tf.matmul(G, gradient, transpose_a=True)

            # Solve (G^T @ G)^{-1} @ (G^T @ dG)
            try:
                solve_term = tf.linalg.solve(GtG_reg, GtdG)
            except Exception:
                # Fallback if solve fails (singular matrix)
                logger.debug(
                    "[GALORE-TT] Solve failed for %s, using regularized pinv",
                    variable.name,
                )
                solve_term = tf.linalg.lstsq(GtG_reg, GtdG, l2_regularizer=GALORE_TT_PROJECTION_EPS)

            # Compute projection: dG - G @ solve_term
            correction = tf.matmul(G, solve_term)
            projected = gradient - correction

            logger.debug(
                "[GALORE-TT] Projected %s: correction_norm=%.4e",
                variable.name,
                float(tf.norm(correction).numpy()),
            )
            return projected

        elif ndim == 3:
            # 3D TT core: [r_in, d, r_out]
            # Flatten to 2D for projection, then reshape back
            r_in, d, r_out = shape
            flat_shape = [r_in, d * r_out]

            G_flat = tf.reshape(G, flat_shape)
            grad_flat = tf.reshape(gradient, flat_shape)

            # Compute projection in flattened space
            GtG = tf.matmul(G_flat, G_flat, transpose_a=True)
            reg = GALORE_TT_PROJECTION_EPS * tf.eye(r_in, dtype=GtG.dtype)
            GtG_reg = GtG + reg

            GtdG = tf.matmul(G_flat, grad_flat, transpose_a=True)

            try:
                solve_term = tf.linalg.solve(GtG_reg, GtdG)
            except Exception:
                solve_term = tf.linalg.lstsq(GtG_reg, GtdG, l2_regularizer=GALORE_TT_PROJECTION_EPS)

            correction = tf.matmul(G_flat, solve_term)
            projected_flat = grad_flat - correction

            return tf.reshape(projected_flat, shape)

        else:
            # Higher-order cores: skip projection
            return gradient

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
            Tuple of (compressed gradient, variable identifier for decompress).
        """
        if not self.enabled:
            return gradient, str(id(variable))

        # Use id(variable) as unique key to avoid name collisions
        # Multiple layers can have variables named 'kernel', 'bias', etc.
        var_id = str(id(variable))
        var_name = variable.name  # For logging only
        shape = gradient.shape.as_list()
        ndim = len(shape)

        # Initialize or update projection if needed
        needs_reinit = (
            var_id not in self._projections
            or self._projections[var_id].needs_update(self._step_counter, self.update_proj_gap)
            or tuple(shape) != self._projections[var_id].shape  # Shape mismatch
        )
        if needs_reinit:
            self._projections[var_id] = self._initialize_projection(var_name, gradient)

        state = self._projections[var_id]

        # Phase 201.12: Apply TT tangent projection for TT layers before compression
        if self._is_tt_layer(var_name) and GALORE_TT_MANIFOLD_PROJECTION:
            self._tt_layers_detected.add(var_name)
            gradient = self._project_to_tt_tangent(gradient, variable)

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
            # Higher-order tensors (3D+): Skip compression for now
            # The HOSVD mode-product implementation has shape tracking issues
            # that require a more substantial fix. 2D covers most parameters.
            # TODO: Fix HOSVD dimension tracking across mode products
            logger.debug(
                "[GALORE] Skipping %dD tensor '%s' - HOSVD not yet supported", ndim, var_name
            )
            return gradient, var_id  # Return var_id for consistent decompress lookup

        return compressed, var_id  # Return var_id for consistent decompress lookup

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

        # Skip decompression if compression was skipped (empty factor_matrices or 3D+)
        if not state.factor_matrices or ndim != 2:
            return compressed_update * self.scale

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
            # Higher-order tensors (3D+): Skip decompression since compression was skipped
            # The compressed_update is already the original gradient
            # TODO: Implement proper HOSVD reconstruction when compression is fixed
            return compressed_update * self.scale

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
            # Phase 201.12: TT layer statistics
            "tt_manifold_projection": GALORE_TT_MANIFOLD_PROJECTION,
            "tt_layers_detected": len(self._tt_layers_detected),
            "tt_layer_names": list(self._tt_layers_detected)[:10],  # First 10 for brevity
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
        self._vqc_gradient_variance.clear()
        logger.info("[GALORE] Compressor state reset")

    def register_vqc_gradient(
        self,
        variable_name: str,
        gradient: tf.Tensor,
    ) -> None:
        """Register VQC layer gradient for variance tracking (Phase 130.2).

        Tracks the variance of gradients from VQC layers to inform rank allocation.
        Layers with higher gradient variance get boosted rank budgets.

        Args:
            variable_name: Name of the VQC-related variable.
            gradient: Gradient tensor from VQC layer.
        """
        if not GALORE_VQC_AWARE:
            return

        # Compute gradient variance
        variance = float(tf.math.reduce_variance(gradient).numpy())

        # Exponential moving average with previous variance
        alpha = 0.1
        if variable_name in self._vqc_gradient_variance:
            old_var = self._vqc_gradient_variance[variable_name]
            variance = alpha * variance + (1 - alpha) * old_var

        self._vqc_gradient_variance[variable_name] = variance

    def set_barren_plateau_mode(self, active: bool, rank_boost: float = 1.5) -> None:
        """Set barren plateau recovery mode (S20: GaLore ↔ BP synergy).

        When barren plateau is detected, GaLore adjusts behavior to preserve
        gradient information during the recovery phase:
        - Increases rank to capture more gradient directions
        - Prevents over-compression that could lose escape gradient signal

        Args:
            active: Whether barren plateau recovery is currently active.
            rank_boost: Rank multiplier during BP recovery (default 1.5x).
        """
        old_active = self._barren_plateau_active
        self._barren_plateau_active = active
        self._bp_rank_boost = rank_boost

        if active and not old_active:
            logger.info(
                "[GALORE-S20] Entering barren plateau mode: rank boosted by %.1fx",
                rank_boost,
            )
        elif not active and old_active:
            logger.info("[GALORE-S20] Exiting barren plateau mode: rank restored")


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


@dataclass
class QuantumGaLoreCompressor:
    """Phase 91: Quantum-enhanced GaLore with dynamic rank selection.

    Extends standard Tensor-GaLore with:
    - Entropy-based effective rank computation from gradient eigenvalue spectrum
    - Block-wise rank budget allocation using Taylor expansion influence scores
    - Quantum random feature projection for stable subspace updates

    Attributes:
        max_rank: Maximum projection rank per variable.
        min_rank: Minimum projection rank.
        update_proj_gap: Steps between projection matrix updates.
        scale: Scaling factor for gradient after decompression.
        enabled: Whether compression is active.
        use_block_allocation: Enable block-wise rank allocation.
        total_rank_budget: Total rank budget for block allocation.

    Example:
        >>> compressor = QuantumGaLoreCompressor(max_rank=64, min_rank=8)
        >>> compressed, var_name = compressor.compress(gradient, variable)
        >>> update = optimizer.apply(compressed)
        >>> full_update = compressor.decompress(update, var_name)
    """

    max_rank: int = field(default_factory=lambda: config.GALORE_RANK)
    min_rank: int = field(default=4)
    update_proj_gap: int = field(default_factory=lambda: config.GALORE_UPDATE_PROJ_GAP)
    scale: float = field(default_factory=lambda: config.GALORE_SCALE)
    enabled: bool = field(default_factory=lambda: config.USE_TENSOR_GALORE)
    use_block_allocation: bool = field(default=False)
    total_rank_budget: int = field(default=256)

    # Internal state
    _projections: dict[str, dict[str, Any]] = field(default_factory=dict)
    _step_counter: int = 0
    _block_ranks: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize quantum random features and internal state."""
        if not hasattr(self, "_projections") or self._projections is None:
            self._projections = {}
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0
        if not hasattr(self, "_block_ranks"):
            self._block_ranks = {}

        # Phase 130.2: VQC gradient variance tracking
        self._vqc_gradient_variance: dict[str, float] = {}

        # Try to import native ops
        self._native_available = False
        try:
            from highnoon._native.ops.quantum_galore_ops import is_native_available

            self._native_available = is_native_available()
        except ImportError:
            pass

        logger.info(
            "[QUANTUM_GALORE] Initialized: max_rank=%d, min_rank=%d, native=%s",
            self.max_rank,
            self.min_rank,
            self._native_available,
        )

    def _compute_effective_rank(self, gradient: tf.Tensor) -> int:
        """Compute effective rank from gradient eigenvalue spectrum.

        Uses Shannon entropy: effective_rank = exp(-Σ p_i log(p_i))

        Args:
            gradient: Gradient tensor to analyze.

        Returns:
            Effective rank clamped to [min_rank, max_rank].
        """
        # Get singular values via SVD (eigenvalues of G^T G)
        s = tf.linalg.svd(gradient, compute_uv=False)

        if self._native_available:
            from highnoon._native.ops.quantum_galore_ops import compute_effective_rank

            rank = compute_effective_rank(s, self.max_rank, self.min_rank)
            return int(rank.numpy())

        # Vectorized fallback (fast) - only entropy computation
        s = tf.maximum(s, 0.0)
        total = tf.reduce_sum(s) + 1e-8
        p = s / total + 1e-8
        entropy = -tf.reduce_sum(p * tf.math.log(p))
        effective_rank = tf.exp(entropy)

        rank = int(tf.round(effective_rank).numpy())
        base_rank = max(self.min_rank, min(self.max_rank, rank))

        # Phase 130.2: Boost rank for high-variance VQC layers
        # Note: This will be applied at compress time using variable name
        return base_rank

    def _init_quantum_features(
        self,
        var_name: str,
        rows: int,
        cols: int,
        rank: int,
    ) -> dict[str, tf.Tensor]:
        """Initialize quantum random feature parameters for a variable.

        Args:
            var_name: Variable name for seeding.
            rows: Number of rows in gradient.
            cols: Number of columns in gradient.
            rank: Target projection rank.

        Returns:
            Dict with rotation_matrix and bias tensors.
        """
        # Deterministic seed from variable name
        seed = hash(var_name) % (2**31)
        tf.random.set_seed(seed)

        project_rows = rows >= cols
        dim = rows if project_rows else cols

        scale = 1.0 / np.sqrt(dim)

        rotation_matrix = tf.Variable(
            tf.random.normal([rank, dim], stddev=scale),
            trainable=False,
            name=f"qgalore_rotation/{var_name}",
        )

        bias = tf.Variable(
            tf.random.uniform([rank], 0, 2 * np.pi),
            trainable=False,
            name=f"qgalore_bias/{var_name}",
        )

        return {
            "rotation_matrix": rotation_matrix,
            "bias": bias,
            "project_rows": project_rows,
        }

    def compress(
        self,
        gradient: tf.Tensor,
        variable: tf.Variable,
    ) -> tuple[tf.Tensor, str]:
        """Compress gradient using quantum random projection with dynamic rank.

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

        if len(shape) != 2:
            # Only handle 2D for now, pass through others
            return gradient, var_name

        rows, cols = shape

        # Initialize or update projection if needed
        needs_update = (
            var_name not in self._projections
            or (self._step_counter - self._projections[var_name]["last_update"])
            >= self.update_proj_gap
        )

        if needs_update:
            # Compute dynamic effective rank
            effective_rank = self._compute_effective_rank(gradient)

            # Use block-allocated rank if available
            if self.use_block_allocation and var_name in self._block_ranks:
                effective_rank = self._block_ranks[var_name]

            # Phase 130.2: Boost rank for high-variance VQC layers
            if GALORE_VQC_AWARE and var_name in self._vqc_gradient_variance:
                vqc_var = self._vqc_gradient_variance[var_name]
                if vqc_var > 0.1:  # High variance threshold
                    effective_rank = min(
                        int(effective_rank * GALORE_VQC_VARIANCE_BOOST), rows, cols
                    )
                    logger.debug(
                        "[QUANTUM_GALORE] VQC boost for %s: var=%.4f, rank=%d",
                        var_name,
                        vqc_var,
                        effective_rank,
                    )

            # Initialize quantum features
            features = self._init_quantum_features(var_name, rows, cols, effective_rank)

            self._projections[var_name] = {
                "rank": effective_rank,
                "shape": (rows, cols),
                "rotation_matrix": features["rotation_matrix"],
                "bias": features["bias"],
                "project_rows": features["project_rows"],
                "last_update": self._step_counter,
            }

            # Compute compression ratio
            original_size = rows * cols
            compressed_size = effective_rank * (cols if features["project_rows"] else rows)
            ratio = original_size / max(compressed_size, 1)

            logger.debug(
                "[QUANTUM_GALORE] %s: rank=%d (eff), shape=%s, compression=%.2fx",
                var_name,
                effective_rank,
                shape,
                ratio,
            )

        proj = self._projections[var_name]

        # Apply quantum random projection via C++ or SVD fallback
        if self._native_available:
            from highnoon._native.ops.quantum_galore_ops import quantum_galore_project

            s = tf.linalg.svd(gradient, compute_uv=False)
            compressed, _ = quantum_galore_project(
                gradient,
                s,
                proj["rotation_matrix"],
                proj["bias"],
                max_rank=self.max_rank,
                min_rank=self.min_rank,
            )
            return compressed, var_name

        # Vectorized SVD fallback (fast)
        rank = proj["rank"]
        if proj["project_rows"]:
            _, u, _ = tf.linalg.svd(gradient)
            compressed = tf.matmul(u[:, :rank], gradient, transpose_a=True)
        else:
            _, _, v = tf.linalg.svd(gradient)
            compressed = tf.matmul(gradient, v[:, :rank])

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

        proj = self._projections[variable_name]
        rows, cols = proj["shape"]

        if self._native_available:
            from highnoon._native.ops.quantum_galore_ops import quantum_galore_deproject

            full_update = quantum_galore_deproject(
                compressed_update,
                proj["rotation_matrix"],
                proj["bias"],
                (rows, cols),
                row_projection=proj["project_rows"],
            )
            return full_update * self.scale

        # Vectorized fallback using cosine features
        rank = proj["rank"]
        rotation = proj["rotation_matrix"]

        if proj["project_rows"]:
            full_update = tf.matmul(
                tf.cos(rotation[:, :rows]),
                compressed_update,
                transpose_a=True,
            ) / np.sqrt(rank)
        else:
            full_update = tf.matmul(
                compressed_update,
                tf.cos(rotation[:, :cols]),
            ) / np.sqrt(rank)

        return full_update * self.scale

    def allocate_block_ranks(
        self,
        variables: list[tf.Variable],
        gradients: list[tf.Tensor],
        critical_indices: list[int] | None = None,
    ) -> None:
        """Allocate rank budget across blocks based on influence scores.

        Call this before training loop to enable block-wise allocation.

        Args:
            variables: List of weight variables.
            gradients: List of gradients (can be from a sample batch).
            critical_indices: Indices of critical blocks (first/last layers).
        """
        if not self.use_block_allocation:
            return

        num_blocks = len(variables)
        if num_blocks == 0:
            return

        # Compute gradient and weight norms
        grad_norms = []
        weight_norms = []

        for var, grad in zip(variables, gradients):
            if grad is not None:
                grad_norms.append(tf.norm(grad).numpy())
                weight_norms.append(tf.norm(var).numpy())
            else:
                grad_norms.append(0.0)
                weight_norms.append(1.0)

        grad_norms = tf.constant(grad_norms, dtype=tf.float32)
        weight_norms = tf.constant(weight_norms, dtype=tf.float32)

        if self._native_available:
            from highnoon._native.ops.quantum_galore_ops import (
                allocate_block_ranks,
                compute_block_influence,
            )

            influence = compute_block_influence(grad_norms, weight_norms)
            if critical_indices is None:
                critical_indices = [0, num_blocks - 1]
            ranks = allocate_block_ranks(
                influence,
                self.total_rank_budget,
                self.min_rank,
                critical_indices,
            )
            for i, var in enumerate(variables):
                self._block_ranks[var.name] = int(ranks[i].numpy())
            logger.info(
                "[QUANTUM_GALORE] Block allocation: %d blocks, budget=%d",
                num_blocks,
                self.total_rank_budget,
            )
            return

        # Vectorized fallback
        w_norm = weight_norms + 1e-8
        influence = (grad_norms**2) / (w_norm**2)
        total_influence = tf.reduce_sum(influence) + 1e-8
        influence = influence / total_influence

        for i, var in enumerate(variables):
            rank = max(
                self.min_rank,
                int(influence[i].numpy() * self.total_rank_budget),
            )
            self._block_ranks[var.name] = min(rank, self.max_rank)

    def step(self) -> None:
        """Increment step counter. Call once per training step."""
        self._step_counter += 1

    def get_statistics(self) -> dict[str, Any]:
        """Get compression statistics for all tracked variables."""
        stats = {
            "enabled": self.enabled,
            "max_rank": self.max_rank,
            "min_rank": self.min_rank,
            "step_counter": self._step_counter,
            "num_variables": len(self._projections),
            "native_available": self._native_available,
            "variables": {},
        }

        total_original = 0
        total_compressed = 0

        for name, proj in self._projections.items():
            rows, cols = proj["shape"]
            rank = proj["rank"]
            original_size = rows * cols

            if proj["project_rows"]:
                compressed_size = rank * cols
            else:
                compressed_size = rows * rank

            total_original += original_size
            total_compressed += compressed_size

            stats["variables"][name] = {
                "shape": proj["shape"],
                "rank": rank,
                "compression_ratio": original_size / max(compressed_size, 1),
            }

        if total_original > 0:
            stats["overall_compression_ratio"] = total_original / max(total_compressed, 1)
        else:
            stats["overall_compression_ratio"] = 1.0

        return stats

    def reset(self) -> None:
        """Reset all compression state."""
        self._projections.clear()
        self._block_ranks.clear()
        self._step_counter = 0
        self._vqc_gradient_variance.clear()
        logger.info("[QUANTUM_GALORE] Compressor state reset")

    def register_vqc_gradient(
        self,
        variable_name: str,
        gradient: tf.Tensor,
    ) -> None:
        """Register VQC layer gradient for variance tracking (Phase 130.2).

        Tracks the variance of gradients from VQC layers to inform rank allocation.
        Layers with higher gradient variance get boosted rank budgets.

        Args:
            variable_name: Name of the VQC-related variable.
            gradient: Gradient tensor from VQC layer.
        """
        if not GALORE_VQC_AWARE:
            return

        # Compute gradient variance
        variance = float(tf.math.reduce_variance(gradient).numpy())

        # Exponential moving average with previous variance
        alpha = 0.1
        if variable_name in self._vqc_gradient_variance:
            old_var = self._vqc_gradient_variance[variable_name]
            variance = alpha * variance + (1 - alpha) * old_var

        self._vqc_gradient_variance[variable_name] = variance


__all__ = [
    "TensorGaLoreCompressor",
    "GaLoreOptimizerWrapper",
    "ProjectionState",
    "QuantumGaLoreCompressor",
]
