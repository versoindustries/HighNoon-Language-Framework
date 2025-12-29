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

from tensorflow.python.framework import ops as _ops  # noqa: E402


@_ops.RegisterGradient("FusedHNNSequence")
def _fused_hnn_sequence_grad(op: tf.Operation, *grads):
    """Gradient for FusedHNNSequence using C++ kernel.

    Raises:
        RuntimeError: If C++ grad op is not available.
    """
    # grads contains gradients for all outputs:
    # grad_output_sequence, grad_final_q, grad_final_p, grad_h_initial_seq, grad_h_final_seq
    grad_output_sequence = grads[0]
    grad_final_q = grads[1]
    grad_final_p = grads[2]
    # grads[3] and grads[4] are for h_initial_seq and h_final_seq (not used for backprop)

    # Get native module
    native_module = _ensure_loaded()

    if native_module is None or not hasattr(native_module, "fused_hnn_sequence_grad"):
        raise RuntimeError(
            "FusedHNNSequenceGrad C++ op not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

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
    """Compute HNN sequence dynamics using fused C++ kernel.

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

    Raises:
        RuntimeError: If C++ op is not available.
    """
    native_module = _ensure_loaded()

    if native_module is None or not hasattr(native_module, "fused_hnn_sequence"):
        raise RuntimeError(
            "FusedHNNSequence C++ op not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

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


__all__ = ["fused_hnn_sequence"]
