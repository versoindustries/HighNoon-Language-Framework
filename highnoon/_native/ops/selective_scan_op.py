# highnoon/_native/ops/selective_scan_op.py
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

"""Python wrapper for the Selective Scan (Mamba) custom C++ operation.

Provides a high-performance selective scan operation used in Mamba-style
sequence models for efficient linear-time sequence modeling.
"""

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

# Default cache limit for selective scan (affects internal C++ buffer sizing)
# NOTE: Per-instance seq_len is computed in selective_scan() to handle longer sequences
SELECTIVE_SCAN_CACHE_MAX_SEQ_LEN = 8192

# --- Load the Custom Operator ---
_selective_scan_module = None
selective_scan_op = None
selective_scan_grad_op = None


def _load_selective_scan_op():
    """Load the selective scan operation from the consolidated library."""
    global _selective_scan_module, selective_scan_op, selective_scan_grad_op

    # Check if already registered in TensorFlow
    if "SelectiveScan" in tf.raw_ops.__dict__:
        selective_scan_op = tf.raw_ops.SelectiveScan
        selective_scan_grad_op = tf.raw_ops.SelectiveScanGrad
        return

    # First try to load from the consolidated _highnoon_core.so library
    import os

    from highnoon._native import resolve_op_library

    consolidated_lib_path = resolve_op_library(__file__, "_highnoon_core.so")
    if os.path.exists(consolidated_lib_path):
        try:
            _selective_scan_module = tf.load_op_library(consolidated_lib_path)
            selective_scan_op = _selective_scan_module.selective_scan
            selective_scan_grad_op = _selective_scan_module.selective_scan_grad
            logger.info("Loaded SelectiveScan from consolidated _highnoon_core.so")
            return
        except (tf.errors.NotFoundError, OSError, AttributeError) as e:
            logger.debug(f"Could not load SelectiveScan from consolidated library: {e}")

    # Fallback: try individual .so file (legacy path)
    try:
        _op_lib_path = resolve_op_library(__file__, "_selective_scan_op.so")
        if os.path.exists(_op_lib_path):
            _selective_scan_module = tf.load_op_library(_op_lib_path)
            selective_scan_op = _selective_scan_module.selective_scan
            selective_scan_grad_op = _selective_scan_module.selective_scan_grad
            logger.info("Loaded SelectiveScan from individual .so file")
            return
    except (tf.errors.NotFoundError, OSError) as e:
        logger.error(
            f"Could not load C++ _selective_scan_op: {e}. "
            "NO PYTHON FALLBACK IS PROVIDED - native op is required."
        )
        selective_scan_op = None
        selective_scan_grad_op = None


_load_selective_scan_op()


def selective_scan_available() -> bool:
    """Check if the selective scan operation is available."""
    return selective_scan_op is not None


@tf.custom_gradient
def selective_scan(
    u: tf.Tensor,
    delta: tf.Tensor,
    a_log: tf.Tensor,
    b: tf.Tensor,
    c: tf.Tensor,
    d: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Python wrapper for the custom SelectiveScan operator.

    Implements the selective scan mechanism from Mamba for efficient
    linear-time sequence modeling with content-based reasoning.

    Args:
        u: Input tensor of shape [batch, seq_len, d_inner].
        delta: Discretization parameter of shape [batch, seq_len, d_inner].
        a_log: Log of A matrix for state space model.
        b: B matrix for state space model.
        c: C matrix for state space model.
        d: D matrix (skip connection).

    Returns:
        Tuple of (output, hidden_states) tensors.

    Raises:
        NotImplementedError: If the C++ operation is not available.
    """
    if selective_scan_op is None:
        raise NotImplementedError(
            "The C++ selective_scan operator could not be loaded. "
            "Please compile it and place '_selective_scan_op.so' in the bin directory."
        )

    # Determine cache limit
    cache_limit = int(SELECTIVE_SCAN_CACHE_MAX_SEQ_LEN)
    seq_len_static = u.shape[1]
    if seq_len_static is not None:
        try:
            cache_limit = max(cache_limit, int(seq_len_static))
        except (TypeError, ValueError):
            pass

    output, hidden_states = selective_scan_op(
        u,
        delta,
        a_log,
        b,
        c,
        d,
        max_seq_len_for_caching=cache_limit,
    )

    def grad_fn(
        dy: tf.Tensor,
        d_hidden_states: tf.Tensor | None,
        variables: list[tf.Variable] | None = None,
    ) -> tuple[tuple[tf.Tensor, ...], list[tf.Tensor | None]]:
        """Gradient function for selective scan."""
        if selective_scan_grad_op is None:
            raise NotImplementedError(
                "The C++ SelectiveScanGrad operator could not be loaded. "
                "NO PYTHON FALLBACK IS PROVIDED - native op is required."
            )

        # CRITICAL FIX: Pass the forward-pass cached `hidden_states` to the C++ grad op,
        # NOT `d_hidden_states` (the gradient w.r.t. hidden_states output).
        # The C++ backward kernel needs the actual hidden state values h[t] to compute
        # gradients for A_log, B, and delta. Passing the gradient instead results in
        # empty tensor (since hidden_states output is typically unused) → zero gradients.

        # Call the high-performance backward kernel with FORWARD hidden_states
        input_grads = selective_scan_grad_op(dy, u, delta, a_log, b, c, d, hidden_states)

        # GRADIENT FIX: Map C++ gradient outputs to tf.Variables by name pattern
        # Instead of returning [None] * len(variables) which zeros out all gradients
        if variables is not None and len(variables) > 0:
            # input_grads is a tuple: (grad_u, grad_delta, grad_a_log, grad_b, grad_c, grad_d)
            grad_map = {
                "a_log": input_grads[2] if len(input_grads) > 2 else None,
                "b_proj": input_grads[3] if len(input_grads) > 3 else None,
                "c_proj": input_grads[4] if len(input_grads) > 4 else None,
                "d": input_grads[5] if len(input_grads) > 5 else None,
            }
            variable_grads_list = []
            for v in variables:
                name = v.name.lower()
                found_grad = None
                for key, grad in grad_map.items():
                    if key in name:
                        found_grad = grad
                        break
                variable_grads_list.append(found_grad)
        else:
            variable_grads_list = []
        return input_grads, variable_grads_list

    return (output, hidden_states), grad_fn


__all__ = [
    "selective_scan",
    "selective_scan_available",
    "SELECTIVE_SCAN_CACHE_MAX_SEQ_LEN",
]
