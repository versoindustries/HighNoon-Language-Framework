# highnoon/_native/ops/fused_hnn_step/__init__.py
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

"""Fused HNN Step Operation - Python wrapper for C++ kernel.

This module provides the fused_hnn_step operation for single-step
Hamiltonian Neural Network forward pass. Uses native C++ ops exclusively -
requires compiled binaries to be available.

The operation computes a single step of the HNN update via symplectic
integration (Leapfrog integrator), used during inference with state caching.
"""

from __future__ import annotations

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

# Load native op
_native_op = None
_load_error: str | None = None

try:
    from highnoon._native import get_op, is_native_available

    if is_native_available():
        try:
            _op_module = get_op("fused_hnn_step")
            if _op_module is not None:
                # TF converts CamelCase op names to snake_case
                # FusedHNNStep -> fused_hnn_step
                if hasattr(_op_module, "fused_hnn_step"):
                    _native_op = _op_module.fused_hnn_step
                    logger.debug("Native fused_hnn_step op loaded successfully")
                else:
                    # Some TF versions use the original name
                    _load_error = (
                        f"Op module loaded but 'fused_hnn_step' attribute not found. "
                        f"Available: {dir(_op_module)}"
                    )
            else:
                _load_error = "get_op returned None for fused_hnn_step"
        except Exception as e:
            _load_error = f"Could not load native fused_hnn_step: {e}"
    else:
        _load_error = "Native ops not available"
except ImportError as e:
    _load_error = f"Import error: {e}"


# --- Gradient Registration ---
# Register the gradient so TensorFlow knows how to call the C++ grad kernel
from tensorflow.python.framework import ops as _ops  # noqa: E402


@_ops.RegisterGradient("FusedHNNStep")
def _fused_hnn_step_grad(op: tf.Operation, *grads):
    """Gradient for FusedHNNStep using C++ kernel.

    Connects the forward FusedHNNStep op with its FusedHNNStepGrad backward kernel.

    Args:
        op: The forward operation.
        grads: Gradients for outputs (grad_q_next, grad_p_next, grad_output).

    Returns:
        Tuple of gradients for all inputs.

    Raises:
        RuntimeError: If C++ grad op is not available.
    """
    # grads contains gradients for all outputs:
    # (grad_q_next, grad_p_next, grad_output)
    grad_q_next = grads[0]
    grad_p_next = grads[1]
    grad_output = grads[2]

    # Access grad op from module
    if _op_module is None or not hasattr(_op_module, "fused_hnn_step_grad"):
        raise RuntimeError(
            "FusedHNNStepGrad C++ op not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh --debug --lite"
        )

    # Get inputs from forward op
    q_t = op.inputs[0]
    p_t = op.inputs[1]
    x_t = op.inputs[2]
    w1 = op.inputs[3]
    b1 = op.inputs[4]
    w2 = op.inputs[5]
    b2 = op.inputs[6]
    w3 = op.inputs[7]
    b3 = op.inputs[8]
    w_out = op.inputs[9]
    b_out = op.inputs[10]
    evolution_time_param = op.inputs[11]

    # Call the C++ gradient kernel
    grad_results = _op_module.fused_hnn_step_grad(
        grad_q_next=grad_q_next,
        grad_p_next=grad_p_next,
        grad_output=grad_output,
        q_t=q_t,
        p_t=p_t,
        x_t=x_t,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        w3=w3,
        b3=b3,
        w_out=w_out,
        b_out=b_out,
        evolution_time_param=evolution_time_param,
    )
    return tuple(grad_results)


def fused_hnn_step(
    q_t: tf.Tensor,
    p_t: tf.Tensor,
    x_t: tf.Tensor,
    w1: tf.Tensor,
    b1: tf.Tensor,
    w2: tf.Tensor,
    b2: tf.Tensor,
    w3: tf.Tensor,
    b3: tf.Tensor,
    w_out: tf.Tensor,
    b_out: tf.Tensor,
    evolution_time_param: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Fused HNN step operation via C++ kernel.

    Computes a single step of the Hamiltonian Neural Network update using
    symplectic integration (Leapfrog). Uses native C++ implementation with
    SIMD vectorization (AVX2/AVX512/NEON) for optimal CPU performance.

    This is the single-step version used during cached inference, where we
    process one token at a time with state carried forward.

    The Hamiltonian is computed as:
        H(q, p, x) = W3^T · sin(W2^T · sin(W1^T · [q; p; x] + b1) + b2) + b3

    The symplectic update follows the Leapfrog integrator:
        1. p_half = p_t - (ε/2) · ∂H/∂q
        2. q_next = q_t + ε · ∂H/∂p
        3. p_next = p_half - (ε/2) · ∂H/∂q

    Args:
        q_t: Position state tensor [batch, state_dim].
        p_t: Momentum state tensor [batch, state_dim].
        x_t: Input tensor [batch, input_dim].
        w1: First layer weights [d_in, hidden_dim].
        b1: First layer bias [hidden_dim].
        w2: Second layer weights [hidden_dim, hidden_dim].
        b2: Second layer bias [hidden_dim].
        w3: Third layer weights [hidden_dim, 1].
        b3: Third layer bias (scalar).
        w_out: Output projection weights [2*state_dim, output_dim].
        b_out: Output projection bias [output_dim].
        evolution_time_param: Evolution time step ε (scalar).

    Returns:
        Tuple of:
            - q_next: Updated position state [batch, state_dim]
            - p_next: Updated momentum state [batch, state_dim]
            - output: Output projection [batch, output_dim]
            - h_initial: Initial Hamiltonian (scalar per batch, for energy monitoring)
            - h_final: Final Hamiltonian (scalar per batch, for energy monitoring)

    Raises:
        RuntimeError: If native C++ op is not available.

    Example:
        >>> q_next, p_next, output, h_init, h_final = fused_hnn_step(
        ...     q_t=q_state, p_t=p_state, x_t=input_embedding,
        ...     w1=W1, b1=b1, w2=W2, b2=b2, w3=W3, b3=b3,
        ...     w_out=W_out, b_out=b_out,
        ...     evolution_time_param=dt,
        ... )
    """
    if _native_op is None:
        raise RuntimeError(
            f"Native fused_hnn_step op not available. Error: {_load_error}. "
            "Rebuild native ops with: ./build_secure.sh --debug --lite"
        )

    # Compute initial Hamiltonian for energy monitoring BEFORE symplectic update
    # H(q, p, x) = W3^T · sin(W2^T · sin(W1^T · [q; p; x] + b1) + b2) + b3
    concat_initial = tf.concat([q_t, p_t, x_t], axis=-1)
    layer1_initial = tf.sin(tf.matmul(concat_initial, w1) + b1)
    layer2_initial = tf.sin(tf.matmul(layer1_initial, w2) + b2)
    h_initial = tf.squeeze(tf.matmul(layer2_initial, w3) + b3, axis=-1)

    # Call C++ op for symplectic integration (returns q_next, p_next, output)
    q_next, p_next, output = _native_op(
        q_t,
        p_t,
        x_t,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        w_out,
        b_out,
        evolution_time_param,
    )

    # Compute final Hamiltonian AFTER symplectic update for energy drift tracking
    concat_final = tf.concat([q_next, p_next, x_t], axis=-1)
    layer1_final = tf.sin(tf.matmul(concat_final, w1) + b1)
    layer2_final = tf.sin(tf.matmul(layer1_final, w2) + b2)
    h_final = tf.squeeze(tf.matmul(layer2_final, w3) + b3, axis=-1)

    return q_next, p_next, output, h_initial, h_final


def fused_hnn_step_available() -> bool:
    """Check if the native fused_hnn_step op is available.

    Returns:
        True if the C++ op is loaded and ready to use.
    """
    return _native_op is not None


def get_load_error() -> str | None:
    """Get the error message if native op failed to load.

    Returns:
        Error message string, or None if op loaded successfully.
    """
    return _load_error


__all__ = ["fused_hnn_step", "fused_hnn_step_available", "get_load_error"]
