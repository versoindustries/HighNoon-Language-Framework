# highnoon/_native/ops/cayley_transform.py
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

"""Python wrapper for CayleyDense C++ operations with gradient support.

Provides Python interface to SIMD-optimized Cayley transform ops.
Includes gradient registration for training support.
NO PYTHON FALLBACK - requires native library.
"""

from __future__ import annotations

import logging

import tensorflow as tf

from highnoon._native.ops.lib_loader import get_highnoon_core_path

logger = logging.getLogger(__name__)

# Load native ops - REQUIRED
_lib = None
_ops_available = False

try:
    _lib_path = get_highnoon_core_path()
    _lib = tf.load_op_library(_lib_path)
    if hasattr(_lib, "CayleyDenseForward"):
        _ops_available = True
        logger.info("CayleyDense C++ ops loaded successfully from %s", _lib_path)
    else:
        raise ImportError("CayleyDenseForward op not found in library")
except Exception as e:
    logger.error("Failed to load CayleyDense C++ ops: %s", e)
    _ops_available = False


def cayley_dense_ops_available() -> bool:
    """Check if CayleyDense C++ ops are available."""
    return _ops_available


# =============================================================================
# Gradient Registration for CayleyDenseForward
# =============================================================================

if _ops_available:

    @tf.RegisterGradient("CayleyDenseForward")
    def _cayley_dense_forward_grad(op, grad_output):
        """Gradient function for CayleyDenseForward.

        Uses the C++ backward pass for gradient computation.

        Args:
            op: The forward operation.
            grad_output: Gradient w.r.t. output [batch, output_dim].

        Returns:
            Gradients for each input: (grad_input, grad_skew_params, grad_proj_weight, grad_bias).
        """
        # Get inputs from the forward op
        inputs = op.inputs[0]  # input tensor
        skew_params = op.inputs[1]  # skew_params
        proj_weight = op.inputs[2]  # proj_weight
        op.inputs[3]  # bias

        # Get attributes
        input_dim = op.get_attr("input_dim")
        output_dim = op.get_attr("output_dim")
        use_bias = op.get_attr("use_bias")

        # Call C++ backward pass
        grad_input, grad_skew_params, grad_proj_weight, grad_bias = _lib.CayleyDenseBackward(
            grad_output=tf.cast(grad_output, tf.float32),
            input=tf.cast(inputs, tf.float32),
            skew_params=tf.cast(skew_params, tf.float32),
            proj_weight=tf.cast(proj_weight, tf.float32),
            input_dim=input_dim,
            output_dim=output_dim,
            use_bias=use_bias,
        )

        # Return gradients for each input (including None for attributes)
        # Order must match inputs: input, skew_params, proj_weight, bias
        return grad_input, grad_skew_params, grad_proj_weight, grad_bias


def cayley_dense_forward(
    inputs: tf.Tensor,
    skew_params: tf.Tensor,
    proj_weight: tf.Tensor | None,
    bias: tf.Tensor | None,
    input_dim: int,
    output_dim: int,
    training: bool = False,
) -> tf.Tensor:
    """Forward pass for CayleyDense using native C++ ops.

    Args:
        inputs: Input tensor [batch, input_dim].
        skew_params: Skew-symmetric parameters [n*(n-1)/2].
        proj_weight: Projection weight for rectangular case [input_dim, output_dim].
        bias: Bias tensor [output_dim] or None.
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        training: Whether in training mode.

    Returns:
        Output tensor [batch, output_dim].

    Raises:
        RuntimeError: If C++ ops are not available.
    """
    if not _ops_available:
        raise RuntimeError(
            "CayleyDense C++ ops not available. "
            "Build the native library: cd highnoon/_native && ./build_secure.sh --debug --lite"
        )

    # Ensure tensors are the right type
    inputs = tf.cast(inputs, tf.float32)
    skew_params = tf.cast(skew_params, tf.float32)

    # Handle optional tensors
    if proj_weight is None:
        proj_weight = tf.zeros([0], dtype=tf.float32)
    else:
        proj_weight = tf.cast(proj_weight, tf.float32)

    if bias is None:
        bias = tf.zeros([0], dtype=tf.float32)
    else:
        bias = tf.cast(bias, tf.float32)

    return _lib.CayleyDenseForward(
        input=inputs,
        skew_params=skew_params,
        proj_weight=proj_weight,
        bias=bias,
        input_dim=input_dim,
        output_dim=output_dim,
        use_bias=(bias.shape[0] > 0),
        training=training,
    )


def cayley_dense_backward(
    grad_output: tf.Tensor,
    inputs: tf.Tensor,
    skew_params: tf.Tensor,
    proj_weight: tf.Tensor | None,
    input_dim: int,
    output_dim: int,
    use_bias: bool = True,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Backward pass for CayleyDense using native C++ ops.

    Args:
        grad_output: Gradient w.r.t. output [batch, output_dim].
        inputs: Original inputs [batch, input_dim].
        skew_params: Skew-symmetric parameters.
        proj_weight: Projection weight or None for square case.
        input_dim: Input dimension.
        output_dim: Output dimension.
        use_bias: Whether bias was used.

    Returns:
        Tuple of (grad_input, grad_skew_params, grad_proj_weight, grad_bias).

    Raises:
        RuntimeError: If C++ ops are not available.
    """
    if not _ops_available:
        raise RuntimeError(
            "CayleyDense C++ ops not available. "
            "Build the native library: cd highnoon/_native && ./build_secure.sh --debug --lite"
        )

    if proj_weight is None:
        proj_weight = tf.zeros([0], dtype=tf.float32)

    return _lib.CayleyDenseBackward(
        grad_output=tf.cast(grad_output, tf.float32),
        input=tf.cast(inputs, tf.float32),
        skew_params=tf.cast(skew_params, tf.float32),
        proj_weight=tf.cast(proj_weight, tf.float32),
        input_dim=input_dim,
        output_dim=output_dim,
        use_bias=use_bias,
    )


__all__ = [
    "cayley_dense_ops_available",
    "cayley_dense_forward",
    "cayley_dense_backward",
]
