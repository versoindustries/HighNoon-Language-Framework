# highnoon/_native/ops/fused_token_shift_op.py
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

"""Python wrapper for C++ fused token shift operations.

This module provides a Python interface to the SIMD-optimized C++
implementation of RWKV-6 style data-dependent token shifting.

The C++ kernel is loaded from the consolidated _highnoon_core.so binary.
NO PYTHON FALLBACK - this op requires the native binary.

The gate projection (input @ gate_kernel + gate_bias) is computed in Python
using TensorFlow's optimized einsum, then passed to the C++ kernel which
handles sigmoid, learned decay, and token mixing with SIMD optimization.

Example:
    >>> from highnoon._native.ops.fused_token_shift_op import fused_token_shift
    >>> output = fused_token_shift(
    ...     input_tensor, prev_input,
    ...     gate_kernel, gate_bias, decay_weights,
    ...     use_learned_decay=True
    ... )
"""

from __future__ import annotations

import logging

import tensorflow as tf

from highnoon._native import get_op

logger = logging.getLogger(__name__)

# Load C++ ops from consolidated binary
_lib = get_op("fused_token_shift")
if _lib is None:
    raise ImportError(
        "FusedTokenShift C++ op not available. "
        "Build native ops with: cd highnoon/_native && ./build_secure.sh"
    )

_fused_token_shift_op = _lib.fused_token_shift
_fused_token_shift_grad_op = _lib.fused_token_shift_grad

# New enhanced ops
_fused_simplified_token_shift_op = _lib.fused_simplified_token_shift
_fused_simplified_token_shift_grad_op = _lib.fused_simplified_token_shift_grad
_fused_hierarchical_token_shift_op = _lib.fused_hierarchical_token_shift
_fused_hierarchical_token_shift_grad_op = _lib.fused_hierarchical_token_shift_grad
_fused_delta_token_shift_op = _lib.fused_delta_token_shift
_fused_delta_token_shift_grad_op = _lib.fused_delta_token_shift_grad
_fused_multi_position_token_shift_op = _lib.fused_multi_position_token_shift
_fused_multi_position_token_shift_grad_op = _lib.fused_multi_position_token_shift_grad


@tf.custom_gradient
def _fused_token_shift_internal(
    input_tensor: tf.Tensor,
    prev_input: tf.Tensor,
    gate_kernel: tf.Tensor,
    gate_bias: tf.Tensor,
    decay_weights: tf.Tensor,
    use_learned_decay: bool = True,
) -> tf.Tensor:
    """Internal token shift with custom gradient covering entire computation.

    This function wraps the entire computation including gate_proj to ensure
    TensorFlow's autodiff properly tracks all gradient paths.
    """
    # Compute gate projection in Python using TensorFlow's optimized einsum
    gate_proj = tf.einsum("bld,de->ble", input_tensor, gate_kernel) + gate_bias

    # Call C++ kernel
    output, gate = _fused_token_shift_op(
        input_tensor, prev_input, gate_proj, decay_weights, use_learned_decay=use_learned_decay
    )

    def grad(grad_output: tf.Tensor) -> tuple[tf.Tensor, ...]:
        """Compute gradients using C++ backward kernel + Python chain rule."""
        # Get gradients from C++ kernel
        (grad_input_direct, grad_prev, grad_gate_proj, grad_decay) = _fused_token_shift_grad_op(
            grad_output,
            input_tensor,
            prev_input,
            gate_proj,
            decay_weights,
            gate,
            use_learned_decay=use_learned_decay,
        )

        # Backprop through gate_proj = einsum('bld,de->ble', x, kernel) + bias
        grad_kernel = tf.einsum("bld,ble->de", input_tensor, grad_gate_proj)
        grad_bias = tf.reduce_sum(grad_gate_proj, axis=[0, 1])
        grad_input_via_proj = tf.einsum("ble,ed->bld", grad_gate_proj, gate_kernel)

        total_grad_input = grad_input_direct + grad_input_via_proj
        return total_grad_input, grad_prev, grad_kernel, grad_bias, grad_decay, None

    return output, grad


def fused_token_shift(
    input_tensor: tf.Tensor,
    prev_input: tf.Tensor,
    gate_kernel: tf.Tensor,
    gate_bias: tf.Tensor,
    decay_weights: tf.Tensor,
    use_learned_decay: bool = True,
) -> tf.Tensor:
    """C++-accelerated data-dependent token shift operation."""
    # Ensure float32 precision
    input_tensor = tf.cast(input_tensor, tf.float32)
    prev_input = tf.cast(prev_input, tf.float32)
    gate_kernel = tf.cast(gate_kernel, tf.float32)
    gate_bias = tf.cast(gate_bias, tf.float32)
    decay_weights = tf.cast(decay_weights, tf.float32)

    return _fused_token_shift_internal(
        input_tensor, prev_input, gate_kernel, gate_bias, decay_weights, use_learned_decay
    )


# =============================================================================
# ENHANCEMENT 1: SIMPLIFIED TOKEN SHIFT
# =============================================================================


@tf.custom_gradient
def fused_simplified_token_shift(
    input_tensor: tf.Tensor,
    prev_input: tf.Tensor,
    decay_weights: tf.Tensor,
) -> tf.Tensor:
    """RWKV-7 style simplified token shift (3x faster)."""
    input_tensor = tf.cast(input_tensor, tf.float32)
    prev_input = tf.cast(prev_input, tf.float32)
    decay_weights = tf.cast(decay_weights, tf.float32)

    output = _fused_simplified_token_shift_op(input_tensor, prev_input, decay_weights)

    def grad(grad_output: tf.Tensor) -> tuple[tf.Tensor, ...]:
        grad_input, grad_prev, grad_decay = _fused_simplified_token_shift_grad_op(
            grad_output, input_tensor, prev_input, decay_weights
        )
        return grad_input, grad_prev, grad_decay

    return output, grad


# =============================================================================
# ENHANCEMENT 2: HIERARCHICAL TOKEN SHIFT
# =============================================================================


@tf.custom_gradient
def _fused_hierarchical_token_shift_internal(
    input_tensor: tf.Tensor,
    prev_input: tf.Tensor,
    gate_kernel: tf.Tensor,
    gate_bias: tf.Tensor,
    decay_weights: tf.Tensor,
    layer_position: int,
    decay_factor: float,
) -> tf.Tensor:
    """Gated token shift with hierarchical decay scaling."""
    gate_proj = tf.einsum("bld,de->ble", input_tensor, gate_kernel) + gate_bias

    output, gate = _fused_hierarchical_token_shift_op(
        input_tensor,
        prev_input,
        gate_proj,
        decay_weights,
        layer_position=layer_position,
        decay_factor=decay_factor,
    )

    def grad(grad_output: tf.Tensor) -> tuple[tf.Tensor, ...]:
        (grad_input_direct, grad_prev, grad_gate_proj, grad_decay) = (
            _fused_hierarchical_token_shift_grad_op(
                grad_output,
                input_tensor,
                prev_input,
                gate_proj,
                decay_weights,
                gate,
                layer_position=layer_position,
                decay_factor=decay_factor,
            )
        )

        grad_kernel = tf.einsum("bld,ble->de", input_tensor, grad_gate_proj)
        grad_bias = tf.reduce_sum(grad_gate_proj, axis=[0, 1])
        grad_input_via_proj = tf.einsum("ble,ed->bld", grad_gate_proj, gate_kernel)

        total_grad_input = grad_input_direct + grad_input_via_proj
        return total_grad_input, grad_prev, grad_kernel, grad_bias, grad_decay, None, None

    return output, grad


def fused_hierarchical_token_shift(
    input_tensor: tf.Tensor,
    prev_input: tf.Tensor,
    gate_kernel: tf.Tensor,
    gate_bias: tf.Tensor,
    decay_weights: tf.Tensor,
    layer_position: int = 0,
    decay_factor: float = 2.0,
) -> tf.Tensor:
    """C++-accelerated hierarchical token shift."""
    return _fused_hierarchical_token_shift_internal(
        input_tensor,
        prev_input,
        gate_kernel,
        gate_bias,
        decay_weights,
        layer_position,
        decay_factor,
    )


# =============================================================================
# ENHANCEMENT 3: DELTA RULE TOKEN SHIFT
# =============================================================================


@tf.custom_gradient
def _fused_delta_token_shift_internal(
    input_tensor: tf.Tensor,
    state: tf.Tensor,
    erase_kernel: tf.Tensor,
    erase_bias: tf.Tensor,
    write_kernel: tf.Tensor,
    write_bias: tf.Tensor,
) -> tf.Tensor:
    """Gated Delta Network memory update."""
    erase_proj = tf.einsum("bld,de->ble", input_tensor, erase_kernel) + erase_bias
    write_proj = tf.einsum("bld,de->ble", input_tensor, write_kernel) + write_bias

    output, new_state, erase, write = _fused_delta_token_shift_op(
        input_tensor, state, erase_proj, write_proj
    )

    def grad(grad_output: tf.Tensor, grad_new_state: tf.Tensor) -> tuple[tf.Tensor, ...]:
        (grad_input_direct, grad_state, grad_erase_proj, grad_write_proj) = (
            _fused_delta_token_shift_grad_op(
                grad_output,
                grad_new_state,
                input_tensor,
                state,
                erase_proj,
                write_proj,
                erase,
                write,
            )
        )

        grad_erase_kernel = tf.einsum("bld,ble->de", input_tensor, grad_erase_proj)
        grad_erase_bias = tf.reduce_sum(grad_erase_proj, axis=[0, 1])

        grad_write_kernel = tf.einsum("bld,ble->de", input_tensor, grad_write_proj)
        grad_write_bias = tf.reduce_sum(grad_write_proj, axis=[0, 1])

        grad_input_via_erase = tf.einsum("ble,ed->bld", grad_erase_proj, erase_kernel)
        grad_input_via_write = tf.einsum("ble,ed->bld", grad_write_proj, write_kernel)

        total_grad_input = grad_input_direct + grad_input_via_erase + grad_input_via_write

        return (
            total_grad_input,
            grad_state,
            grad_erase_kernel,
            grad_erase_bias,
            grad_write_kernel,
            grad_write_bias,
        )

    return [output, new_state], grad


def fused_delta_token_shift(
    input_tensor: tf.Tensor,
    state: tf.Tensor,
    erase_kernel: tf.Tensor,
    erase_bias: tf.Tensor,
    write_kernel: tf.Tensor,
    write_bias: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """C++-accelerated delta rule token shift."""
    return _fused_delta_token_shift_internal(
        input_tensor, state, erase_kernel, erase_bias, write_kernel, write_bias
    )


# =============================================================================
# ENHANCEMENT 5: MULTI-POSITION TOKEN SHIFT
# =============================================================================


@tf.custom_gradient
def fused_multi_position_token_shift(
    input_tensor: tf.Tensor,
    blend_weights: tf.Tensor,
    distances: list[int],
) -> tf.Tensor:
    """Multi-scale look-ahead token shift."""
    input_tensor = tf.cast(input_tensor, tf.float32)
    blend_weights = tf.cast(blend_weights, tf.float32)

    output = _fused_multi_position_token_shift_op(input_tensor, blend_weights, distances=distances)

    def grad(grad_output: tf.Tensor) -> tuple[tf.Tensor, ...]:
        grad_input, grad_blend = _fused_multi_position_token_shift_grad_op(
            grad_output, input_tensor, blend_weights, distances=distances
        )
        return grad_input, grad_blend, None

    return output, grad


def fused_token_shift_with_gate(
    input_tensor: tf.Tensor,
    prev_input: tf.Tensor,
    gate_kernel: tf.Tensor,
    gate_bias: tf.Tensor,
    decay_weights: tf.Tensor,
    use_learned_decay: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Token shift that also returns the gate values.

    Useful for debugging and visualization.

    Args:
        Same as fused_token_shift.

    Returns:
        Tuple of (output, gate) tensors.
    """
    # Ensure float32 precision
    input_tensor = tf.cast(input_tensor, tf.float32)
    prev_input = tf.cast(prev_input, tf.float32)
    gate_kernel = tf.cast(gate_kernel, tf.float32)
    gate_bias = tf.cast(gate_bias, tf.float32)
    decay_weights = tf.cast(decay_weights, tf.float32)

    # Compute gate projection in Python
    gate_proj = tf.einsum("bld,de->ble", input_tensor, gate_kernel) + gate_bias

    return _fused_token_shift_op(
        input_tensor, prev_input, gate_proj, decay_weights, use_learned_decay=use_learned_decay
    )


def _python_reference_forward(
    input_tensor: tf.Tensor,
    prev_input: tf.Tensor,
    gate_kernel: tf.Tensor,
    gate_bias: tf.Tensor,
    decay_weights: tf.Tensor,
    use_learned_decay: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Pure Python reference implementation FOR TESTING ONLY.

    This is NOT used in production - only for unit test comparisons.

    Args:
        input_tensor: Input of shape [batch, seq_len, embedding_dim].
        prev_input: Previous token states of same shape.
        gate_kernel: Gate Dense kernel [embedding_dim, embedding_dim].
        gate_bias: Gate Dense bias [embedding_dim].
        decay_weights: Learned decay weights [embedding_dim].
        use_learned_decay: Whether to apply learned decay.

    Returns:
        Tuple of (output, gate) tensors.
    """
    # Compute gate = sigmoid(input @ gate_kernel + gate_bias)
    gate = tf.nn.sigmoid(tf.einsum("bld,de->ble", input_tensor, gate_kernel) + gate_bias)

    # Apply learned decay if enabled
    if use_learned_decay:
        learned_decay = tf.nn.sigmoid(decay_weights)
        gate = gate * learned_decay

    # Token mixing
    output = gate * input_tensor + (1.0 - gate) * prev_input

    return output, gate


# Export Python reference for testing only
python_token_shift_forward = _python_reference_forward
