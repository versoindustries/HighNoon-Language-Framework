# highnoon/_native/ops/entropy_regularization_ops.py
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

"""Python wrappers for Entropy Regularization C++ operations (Phase 45).

Provides Von Neumann entropy regularization for representation diversity
in quantum-enhanced neural networks.

NO PYTHON FALLBACKS: RuntimeError raised if native ops unavailable.

Ops:
    - von_neumann_entropy_loss: Compute entropy-based regularization loss
    - compute_activation_covariance: Compute covariance matrix
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
        logger.info(f"Entropy regularization ops loaded from {lib_path}")
    except Exception as e:
        _available = False
        logger.warning(f"Failed to load entropy regularization ops: {e}")
        raise RuntimeError(
            "Entropy regularization native ops not available. " "Run ./build_secure.sh to compile."
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
# Phase 45: Von Neumann Entropy Regularization
# =============================================================================


def von_neumann_entropy_loss(
    activations: tf.Tensor,
    entropy_weight: float | None = None,
    spectral_weight: float | None = None,
    target_entropy: float | None = None,
    spectral_flatness_target: float = 0.8,
    power_iter_steps: int = 10,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute Von Neumann entropy regularization loss.

    Uses eigenvalue decomposition of activation covariance to compute:
    1. Von Neumann entropy: S = -Σ λ log λ
    2. Spectral flatness measure

    Regularization encourages diverse, high-entropy representations
    while maintaining appropriate spectral properties.

    Args:
        activations: Activation tensor [batch, dim].
        entropy_weight: Weight for entropy term (default from config).
        spectral_weight: Weight for spectral flatness (default from config).
        target_entropy: Target entropy value (default from config).
        spectral_flatness_target: Target spectral flatness (default 0.8).
        power_iter_steps: Power iteration steps for eigenvalues.

    Returns:
        Tuple of (loss scalar, entropy scalar, flatness scalar).

    Raises:
        RuntimeError: If native op not available.

    Example:
        >>> activations = model.hidden_states  # [batch, hidden_dim]
        >>> loss, entropy, flatness = von_neumann_entropy_loss(activations)
        >>> total_loss += loss * config.ENTROPY_REG_WEIGHT
    """
    if not config.USE_ENTROPY_REGULARIZATION:
        return tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)

    _load_ops()
    entropy_weight = entropy_weight if entropy_weight is not None else config.ENTROPY_REG_WEIGHT
    spectral_weight = spectral_weight if spectral_weight is not None else config.SPECTRAL_REG_WEIGHT
    target_entropy = target_entropy if target_entropy is not None else config.TARGET_ENTROPY

    return _module.von_neumann_entropy_loss(
        activations,
        entropy_weight=entropy_weight,
        spectral_weight=spectral_weight,
        target_entropy=target_entropy,
        spectral_flatness_target=spectral_flatness_target,
        power_iter_steps=power_iter_steps,
    )


def compute_activation_covariance(
    activations: tf.Tensor,
) -> tf.Tensor:
    """Compute covariance matrix from activations.

    Useful for analyzing representation structure and
    debugging entropy regularization.

    Args:
        activations: Activation tensor [batch, dim].

    Returns:
        Covariance matrix [dim, dim].

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    return _module.compute_activation_covariance(activations)


__all__ = [
    "von_neumann_entropy_loss",
    "compute_activation_covariance",
    "ops_available",
]
