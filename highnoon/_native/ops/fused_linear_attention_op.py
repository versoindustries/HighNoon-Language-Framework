# highnoon/_native/ops/fused_linear_attention_op.py
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

"""Python wrapper for the Fused Linear Attention custom C++ operation.

Provides a high-performance O(L) linear attention operation with SIMD
optimization for CPU-bound execution.

Phase 1 C++ Migration: Flash Linear Attention.

Note: This op loads from the consolidated `_highnoon_core.so` binary
built by `./build_secure.sh`. For development testing only, individual
ops can be built with `./build_ops.sh fused_linear_attention`.
"""

import logging

import tensorflow as tf

from highnoon._native import get_op

logger = logging.getLogger(__name__)

# --- Load the Custom Operator from Consolidated Binary ---
_fused_linear_attention_module = None
fused_linear_attention_op = None
fused_linear_attention_grad_op = None


def _load_fused_linear_attention_op():
    """Load the fused linear attention operation from consolidated binary."""
    global _fused_linear_attention_module, fused_linear_attention_op, fused_linear_attention_grad_op

    # Check if already registered in tf.raw_ops (from previous load)
    if "FusedLinearAttention" in tf.raw_ops.__dict__:
        fused_linear_attention_op = tf.raw_ops.FusedLinearAttention
        fused_linear_attention_grad_op = tf.raw_ops.FusedLinearAttentionGrad
        return

    # Load from consolidated binary via _native loader
    try:
        _fused_linear_attention_module = get_op("fused_linear_attention")
        if _fused_linear_attention_module is not None:
            fused_linear_attention_op = _fused_linear_attention_module.fused_linear_attention
            fused_linear_attention_grad_op = (
                _fused_linear_attention_module.fused_linear_attention_grad
            )
            logger.info("Successfully loaded FusedLinearAttention from _highnoon_core.so")
        else:
            logger.error(
                "FusedLinearAttention op not available. " "Please compile with: ./build_secure.sh"
            )
            fused_linear_attention_op = None
            fused_linear_attention_grad_op = None
    except (AttributeError, Exception) as e:
        logger.error(
            f"Could not load FusedLinearAttention from consolidated binary. "
            f"Please rebuild with: ./build_secure.sh. Error: {e}"
        )
        fused_linear_attention_op = None
        fused_linear_attention_grad_op = None


_load_fused_linear_attention_op()


def fused_linear_attention_available() -> bool:
    """Check if the fused linear attention operation is available."""
    return fused_linear_attention_op is not None


@tf.custom_gradient
def fused_linear_attention(
    q: tf.Tensor,
    k: tf.Tensor,
    v: tf.Tensor,
    eps: float = 1e-6,
) -> tf.Tensor:
    """Python wrapper for the custom FusedLinearAttention operator.

    Implements O(L) linear attention: O = φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K)^T @ 1)

    Feature map must be applied before calling this function.

    Args:
        q: Query tensor of shape [batch, heads, seq_len, head_dim] (feature map applied).
        k: Key tensor of shape [batch, heads, seq_len, head_dim] (feature map applied).
        v: Value tensor of shape [batch, heads, seq_len, head_dim].
        eps: Epsilon for numerical stability.

    Returns:
        Output tensor of shape [batch, heads, seq_len, head_dim].

    Raises:
        NotImplementedError: If the C++ operation is not available.
    """
    if fused_linear_attention_op is None:
        raise NotImplementedError(
            "The C++ fused_linear_attention operator could not be loaded. "
            "Please compile with: ./build_secure.sh"
        )

    output, kv_cache, k_sum_cache = fused_linear_attention_op(q, k, v, eps=eps)

    def grad_fn(dy: tf.Tensor):
        """Gradient function for fused linear attention."""
        if fused_linear_attention_grad_op is None:
            raise NotImplementedError(
                "The C++ fused_linear_attention_grad operator could not be loaded."
            )

        grad_q, grad_k, grad_v = fused_linear_attention_grad_op(
            dy, q, k, v, kv_cache, k_sum_cache, eps=eps
        )

        # Return gradients for q, k, v only (eps is not differentiable)
        return grad_q, grad_k, grad_v

    return output, grad_fn


__all__ = [
    "fused_linear_attention",
    "fused_linear_attention_available",
]
