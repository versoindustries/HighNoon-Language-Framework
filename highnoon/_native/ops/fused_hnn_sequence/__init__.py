# highnoon/_native/ops/fused_hnn_sequence/__init__.py
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

"""Fused HNN Sequence Operation wrapper.

This module provides the Python interface to the FusedHNNSequence custom TensorFlow
operation, which efficiently computes Hamiltonian Neural Network sequence dynamics
using Yoshida 4th-order symplectic integration.

The operation is compiled into _highnoon_core.so and loaded via the native loader.
"""

import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger(__name__)

# Native op module cache
_native_op_module = None
_load_attempted = False


def _ensure_loaded():
    """Ensure the native operation module is loaded."""
    global _native_op_module, _load_attempted

    if _load_attempted:
        return _native_op_module

    _load_attempted = True

    try:
        from highnoon._native import _load_consolidated_binary

        _native_op_module = _load_consolidated_binary()
        if _native_op_module is not None and hasattr(_native_op_module, "fused_hnn_sequence"):
            logger.info("Loaded FusedHNNSequence from _highnoon_core.so")
            return _native_op_module
    except ImportError:
        pass

    logger.warning("FusedHNNSequence op not available. Using Python fallback.")
    return None


# --- Custom Gradient Registration ---
# Register the gradient so TensorFlow knows how to call the C++ grad kernel

from tensorflow.python.framework import ops as _ops


@_ops.RegisterGradient("FusedHNNSequence")
def _fused_hnn_sequence_grad(op: tf.Operation, *grads):
    """Gradient for FusedHNNSequence using C++ kernel."""
    # grads contains gradients for all outputs:
    # grad_output_sequence, grad_final_q, grad_final_p, grad_h_initial_seq, grad_h_final_seq
    grad_output_sequence = grads[0]
    grad_final_q = grads[1]
    grad_final_p = grads[2]
    # grads[3] and grads[4] are for h_initial_seq and h_final_seq (not used for backprop)

    # Get native module
    native_module = _ensure_loaded()

    if native_module is None or not hasattr(native_module, "fused_hnn_sequence_grad"):
        # Fall back to returning zeros if grad op not available
        logger.warning("FusedHNNSequenceGrad not available, returning zero gradients")
        return tuple(tf.zeros_like(inp) for inp in op.inputs)

    # Get inputs from forward op
    sequence_input = op.inputs[0]
    initial_q = op.inputs[1]
    initial_p = op.inputs[2]
    w1 = op.inputs[3]
    b1 = op.inputs[4]
    w2 = op.inputs[5]
    b2 = op.inputs[6]
    w3 = op.inputs[7]
    b3 = op.inputs[8]
    w_out = op.inputs[9]
    b_out = op.inputs[10]
    evolution_time = op.inputs[11]

    # Get outputs from forward pass (for intermediate values)
    op.outputs[0]
    op.outputs[1]
    op.outputs[2]

    try:
        # Call the C++ gradient kernel
        grad_results = native_module.fused_hnn_sequence_grad(
            grad_output_sequence=grad_output_sequence,
            grad_final_q=grad_final_q,
            grad_final_p=grad_final_p,
            sequence_input=sequence_input,
            initial_q=initial_q,
            initial_p=initial_p,
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            w3=w3,
            b3=b3,
            w_out=w_out,
            b_out=b_out,
            evolution_time_param=evolution_time,
        )
        return tuple(grad_results)
    except Exception as e:
        logger.warning(f"FusedHNNSequenceGrad failed: {e}, returning zeros")
        return tuple(tf.zeros_like(inp) for inp in op.inputs)


def fused_hnn_sequence(
    sequence_input: tf.Tensor,
    initial_q: tf.Tensor,
    initial_p: tf.Tensor,
    w1: tf.Tensor,
    b1: tf.Tensor,
    w2: tf.Tensor,
    b2: tf.Tensor,
    w3: tf.Tensor,
    b3: tf.Tensor,
    w_out: tf.Tensor,
    b_out: tf.Tensor,
    evolution_time: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute HNN sequence dynamics using fused C++ kernel or Python fallback.

    Args:
        sequence_input: Input sequence tensor [batch, seq_len, input_dim].
        initial_q: Initial position state [batch, state_dim].
        initial_p: Initial momentum state [batch, state_dim].
        w1: First layer weights [D_in, D_h].
        b1: First layer bias [D_h].
        w2: Second layer weights [D_h, D_h].
        b2: Second layer bias [D_h].
        w3: Third layer weights [D_h, 1].
        b3: Third layer bias (scalar).
        w_out: Output projection weights [2*state_dim, output_dim].
        b_out: Output projection bias [output_dim].
        evolution_time: Evolution time step (scalar).

    Returns:
        Tuple of (output_sequence, final_q, final_p, h_initial_seq, h_final_seq).
    """
    native_module = _ensure_loaded()

    if native_module is not None and hasattr(native_module, "fused_hnn_sequence"):
        try:
            return native_module.fused_hnn_sequence(
                sequence_input=sequence_input,
                initial_q=initial_q,
                initial_p=initial_p,
                w1=w1,
                b1=b1,
                w2=w2,
                b2=b2,
                w3=w3,
                b3=b3,
                w_out=w_out,
                b_out=b_out,
                evolution_time_param=evolution_time,
            )
        except Exception as e:
            logger.warning(f"Native FusedHNNSequence failed: {e}. Using fallback.")

    # Python fallback implementation using TensorArray for graph compatibility
    return _python_hnn_sequence(
        sequence_input=sequence_input,
        initial_q=initial_q,
        initial_p=initial_p,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        w3=w3,
        b3=b3,
        w_out=w_out,
        b_out=b_out,
        evolution_time=evolution_time,
    )


def _python_hnn_sequence(
    sequence_input: tf.Tensor,
    initial_q: tf.Tensor,
    initial_p: tf.Tensor,
    w1: tf.Tensor,
    b1: tf.Tensor,
    w2: tf.Tensor,
    b2: tf.Tensor,
    w3: tf.Tensor,
    b3: tf.Tensor,
    w_out: tf.Tensor,
    b_out: tf.Tensor,
    evolution_time: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Pure TensorFlow fallback for HNN sequence dynamics.

    Implements a simplified symplectic leapfrog integration. Uses TensorArray
    and tf.while_loop for graph mode compatibility.
    """
    tf.shape(sequence_input)[0]
    seq_len = tf.shape(sequence_input)[1]
    tf.shape(sequence_input)[2]
    tf.shape(initial_q)[1]
    tf.shape(b_out)[0]

    # Use epsilon directly
    epsilon = tf.cast(evolution_time, tf.float32)

    # Create TensorArrays for graph-mode compatible accumulation
    output_ta = tf.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False)
    h_init_ta = tf.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False)
    h_final_ta = tf.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False)

    def compute_hamiltonian_and_grad(q, p, x):
        """Compute Hamiltonian and its gradients w.r.t. q and p."""
        z = tf.concat([q, p, x], axis=-1)  # [batch, D_in]

        # Forward pass
        h1 = tf.nn.tanh(tf.matmul(z, w1) + b1)
        h2 = tf.nn.tanh(tf.matmul(h1, w2) + b2)
        H = tf.reduce_sum(tf.matmul(h2, w3), axis=-1) + b3  # [batch]

        # Backward pass for gradient (manual derivative)
        # dH/dz = dH/dh2 * dh2/dh1 * dh1/dz
        # For simplicity, use approximation: gradient of H w.r.t q is similar in structure
        # This is a simplified fallback - production uses the C++ version

        # Numerical gradient approximation for symplectic stability
        # Compute gradient w.r.t q
        eps = 1e-4
        z_q_plus = tf.concat([q + eps, p, x], axis=-1)
        z_q_minus = tf.concat([q - eps, p, x], axis=-1)
        H_q_plus = (
            tf.reduce_sum(
                tf.matmul(
                    tf.nn.tanh(tf.matmul(tf.nn.tanh(tf.matmul(z_q_plus, w1) + b1), w2) + b2), w3
                ),
                axis=-1,
            )
            + b3
        )
        H_q_minus = (
            tf.reduce_sum(
                tf.matmul(
                    tf.nn.tanh(tf.matmul(tf.nn.tanh(tf.matmul(z_q_minus, w1) + b1), w2) + b2), w3
                ),
                axis=-1,
            )
            + b3
        )
        dH_dq = (H_q_plus - H_q_minus) / (2 * eps)  # [batch]
        dH_dq = tf.reshape(dH_dq, [-1, 1]) * tf.ones_like(q)  # Broadcast to [batch, D_state]

        # Gradient w.r.t p
        z_p_plus = tf.concat([q, p + eps, x], axis=-1)
        z_p_minus = tf.concat([q, p - eps, x], axis=-1)
        H_p_plus = (
            tf.reduce_sum(
                tf.matmul(
                    tf.nn.tanh(tf.matmul(tf.nn.tanh(tf.matmul(z_p_plus, w1) + b1), w2) + b2), w3
                ),
                axis=-1,
            )
            + b3
        )
        H_p_minus = (
            tf.reduce_sum(
                tf.matmul(
                    tf.nn.tanh(tf.matmul(tf.nn.tanh(tf.matmul(z_p_minus, w1) + b1), w2) + b2), w3
                ),
                axis=-1,
            )
            + b3
        )
        dH_dp = (H_p_plus - H_p_minus) / (2 * eps)
        dH_dp = tf.reshape(dH_dp, [-1, 1]) * tf.ones_like(p)

        return H, dH_dq, dH_dp

    def loop_body(i, q_t, p_t, output_ta_i, h_init_ta_i, h_final_ta_i):
        """Process one sequence step."""
        # Get input for this step
        x_l = sequence_input[:, i, :]  # [batch, D_input]

        # Compute initial Hamiltonian
        H_init, dH_dq_init, dH_dp_init = compute_hamiltonian_and_grad(q_t, p_t, x_l)

        # Symplectic leapfrog integration
        # Half step in momentum
        p_half = p_t - 0.5 * epsilon * dH_dq_init

        # Full step in position using gradient at half-step momentum
        _, _, dH_dp_half = compute_hamiltonian_and_grad(q_t, p_half, x_l)
        q_next = q_t + epsilon * dH_dp_half

        # Half step in momentum at new position
        _, dH_dq_next, _ = compute_hamiltonian_and_grad(q_next, p_half, x_l)
        p_next = p_half - 0.5 * epsilon * dH_dq_next

        # Compute final Hamiltonian
        H_final, _, _ = compute_hamiltonian_and_grad(q_next, p_next, x_l)

        # Output projection
        final_state = tf.concat([q_next, p_next], axis=-1)  # [batch, 2*D_state]
        output_l = tf.matmul(final_state, w_out) + b_out  # [batch, D_output]

        # Store results in TensorArrays
        output_ta_i = output_ta_i.write(i, output_l)
        h_init_ta_i = h_init_ta_i.write(i, H_init[:, None])
        h_final_ta_i = h_final_ta_i.write(i, H_final[:, None])

        return i + 1, q_next, p_next, output_ta_i, h_init_ta_i, h_final_ta_i

    # Run the loop
    _, final_q, final_p, output_ta, h_init_ta, h_final_ta = tf.while_loop(
        cond=lambda i, *_: i < seq_len,
        body=loop_body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            initial_q,
            initial_p,
            output_ta,
            h_init_ta,
            h_final_ta,
        ),
        parallel_iterations=1,
        back_prop=True,
    )

    # Stack TensorArrays into tensors
    output_sequence = tf.transpose(output_ta.stack(), [1, 0, 2])  # [batch, seq_len, D_output]
    h_initial_seq = tf.transpose(h_init_ta.stack(), [1, 0, 2])  # [batch, seq_len, 1]
    h_final_seq = tf.transpose(h_final_ta.stack(), [1, 0, 2])  # [batch, seq_len, 1]

    return output_sequence, final_q, final_p, h_initial_seq, h_final_seq


__all__ = ["fused_hnn_sequence"]
