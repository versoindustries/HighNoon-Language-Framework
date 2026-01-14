# highnoon/_native/ops/intrinsic_plasticity_ops.py
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

"""Python wrappers for Intrinsic Plasticity C++ operations (Phase 71).

Provides Stiefel manifold plasticity preservation for unitary layers,
ensuring orthogonality constraints are maintained during training.

NO PYTHON FALLBACKS: RuntimeError raised if native ops unavailable.

Ops:
    - cayley_parameterization: Convert skew-symmetric to unitary
    - enforce_unitary_constraint: Project to orthonormal
    - project_gradient_tangent: Project gradient to tangent space
    - retract_to_manifold: Retract updates to unitary manifold
    - compute_plasticity_metric: Track plasticity from weight trajectory
    - measure_layer_plasticity: Instant plasticity from gradients
"""

import logging

import tensorflow as tf

from highnoon import config
from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

_module = None
_available = False


def _load_ops():
    """Load ops from consolidated binary."""
    global _module, _available
    if _module is not None:
        return _available

    try:
        lib_path = resolve_op_library(__file__, "_highnoon_core.so")
        if lib_path is None:
            raise RuntimeError("Could not find _highnoon_core.so")
        _module = tf.load_op_library(lib_path)
        _available = True
        logger.info(f"Intrinsic plasticity ops loaded from {lib_path}")
    except Exception as e:
        _available = False
        logger.warning(f"Failed to load intrinsic plasticity ops: {e}")
        raise RuntimeError(
            "Intrinsic plasticity native ops not available. Run ./build_secure.sh to compile."
        ) from e
    return _available


def ops_available() -> bool:
    """Check if native ops are available."""
    try:
        _load_ops()
        return _available
    except RuntimeError:
        return False


# =============================================================================
# Phase 71: Stiefel Manifold Operations
# =============================================================================


def cayley_parameterization(
    skew_params: tf.Tensor,
    dim: int,
) -> tf.Tensor:
    """Cayley parameterization of unitary matrix.

    Converts skew-symmetric parameters to orthogonal/unitary matrix:
    W = (I - A)(I + A)^{-1}

    This ensures weights remain on the unitary manifold during
    unconstrained optimization.

    Args:
        skew_params: Upper-triangular skew-symmetric params [dim*(dim-1)/2].
        dim: Target dimension for output matrix.

    Returns:
        Orthogonal matrix [dim, dim].

    Raises:
        RuntimeError: If native op not available.

    Example:
        >>> num_params = dim * (dim - 1) // 2
        >>> skew = tf.Variable(tf.random.normal([num_params], stddev=0.01))
        >>> W = cayley_parameterization(skew, dim=64)
        >>> # W^T @ W ≈ I
    """
    _load_ops()
    return _module.cayley_parameterization(skew_params, dim=dim)


def enforce_unitary_constraint(
    weights: tf.Tensor,
) -> tf.Tensor:
    """Enforce unitary constraint on weight matrix.

    Projects weights to nearest orthonormal matrix via
    Gram-Schmidt orthogonalization.

    Args:
        weights: Input weight matrix [rows, cols].

    Returns:
        Orthonormalized matrix [rows, cols].

    Raises:
        RuntimeError: If native op not available.
    """
    if not config.USE_INTRINSIC_PLASTICITY:
        return weights

    _load_ops()
    return _module.enforce_unitary_constraint(weights)


def project_gradient_tangent(
    gradient: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    """Project gradient to tangent space of unitary manifold.

    For W ∈ O(n): ∇_tang = ∇ - W * sym(W^T * ∇)

    This ensures gradient updates stay on the manifold,
    preventing drift from unitarity.

    Args:
        gradient: Euclidean gradient [dim, dim].
        weights: Current unitary weights [dim, dim].

    Returns:
        Projected tangent gradient [dim, dim].

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    return _module.project_gradient_tangent(gradient, weights)


def retract_to_manifold(
    weights: tf.Tensor,
    direction: tf.Tensor,
    step_size: float = 1.0,
) -> tf.Tensor:
    """Retract updated parameters back to unitary manifold.

    Uses QR-based retraction: W_new = qr(W + step * direction).Q

    Args:
        weights: Current weights [dim, dim].
        direction: Update direction [dim, dim].
        step_size: Step size multiplier.

    Returns:
        Retracted weights on manifold [dim, dim].

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    return _module.retract_to_manifold(weights, direction, step_size=step_size)


def compute_plasticity_metric(
    weight_trajectory: tf.Tensor,
) -> tf.Tensor:
    """Compute plasticity metric from weight trajectory.

    Measures capacity to learn new information by tracking
    weight changes over training snapshots.

    Args:
        weight_trajectory: Weight snapshots [num_snapshots, num_params].

    Returns:
        Plasticity score in [0, 1] (scalar).

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    return _module.compute_plasticity_metric(weight_trajectory)


def measure_layer_plasticity(
    gradients: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    """Measure layer plasticity using relative gradient norm.

    Quick instantaneous measure of plasticity useful for
    dynamic learning rate adaptation.

    Args:
        gradients: Current gradients [num_params].
        weights: Current weights [num_params].

    Returns:
        Relative gradient norm (scalar).

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    return _module.measure_layer_plasticity(gradients, weights)


__all__ = [
    "cayley_parameterization",
    "enforce_unitary_constraint",
    "project_gradient_tangent",
    "retract_to_manifold",
    "compute_plasticity_metric",
    "measure_layer_plasticity",
    "ops_available",
]
