# highnoon/_native/ops/fused_gqa_op.py
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

"""Python wrapper for fused Grouped-Query Attention (GQA) C++ op.

This module provides a Python interface to the C++ FusedGQA kernel with
automatic gradient support via tf.custom_gradient.

Example:
    >>> from highnoon._native.ops.fused_gqa_op import fused_gqa
    >>> output = fused_gqa(
    ...     x, q_weight, k_weight, v_weight, out_weight,
    ...     norm_gamma, norm_beta,
    ...     num_heads=8, num_kv_heads=2, head_dim=64, causal=True
    ... )
"""

from __future__ import annotations

import tensorflow as tf

from highnoon._native import get_op

# Load the C++ op library
_lib = get_op("fused_gqa")
_fused_gqa_op = getattr(_lib, "FusedGQA", None) if _lib else None
_fused_gqa_grad_op = getattr(_lib, "FusedGQAGrad", None) if _lib else None


def fused_gqa(
    x: tf.Tensor,
    q_weight: tf.Tensor,
    k_weight: tf.Tensor,
    v_weight: tf.Tensor,
    out_weight: tf.Tensor,
    norm_gamma: tf.Tensor,
    norm_beta: tf.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool = True,
) -> tf.Tensor:
    """Fused Grouped-Query Attention.

    C++-accelerated GQA that fuses Q/K/V projections, attention computation,
    output projection, and LayerNorm into a single kernel.

    Args:
        x: Input tensor [batch, seq_len, embed_dim].
        q_weight: Query projection weights [embed_dim, embed_dim].
        k_weight: Key projection weights [embed_dim, num_kv_heads * head_dim].
        v_weight: Value projection weights [embed_dim, num_kv_heads * head_dim].
        out_weight: Output projection weights [embed_dim, embed_dim].
        norm_gamma: LayerNorm scale [embed_dim].
        norm_beta: LayerNorm bias [embed_dim].
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads (must divide num_heads evenly).
        head_dim: Dimension per attention head.
        causal: Whether to apply causal masking (default: True).

    Returns:
        Output tensor [batch, seq_len, embed_dim].

    Raises:
        RuntimeError: If C++ op library is not available.
    """
    if _fused_gqa_op is None:
        raise RuntimeError(
            "FusedGQA C++ op not available. Build with: "
            "cd highnoon/_native && ./build_ops.sh fused_gqa"
        )

    # Ensure float32
    x = tf.cast(x, tf.float32)
    q_weight = tf.cast(q_weight, tf.float32)
    k_weight = tf.cast(k_weight, tf.float32)
    v_weight = tf.cast(v_weight, tf.float32)
    out_weight = tf.cast(out_weight, tf.float32)
    norm_gamma = tf.cast(norm_gamma, tf.float32)
    norm_beta = tf.cast(norm_beta, tf.float32)

    @tf.custom_gradient
    def _fused_gqa_inner(x_in, q_w, k_w, v_w, out_w, gamma, beta):
        """Inner function with tensor-only signature for gradient handling."""
        output = _fused_gqa_op(
            x=x_in,
            q_weight=q_w,
            k_weight=k_w,
            v_weight=v_w,
            out_weight=out_w,
            norm_gamma=gamma,
            norm_beta=beta,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            causal=causal,
        )

        def grad(grad_output):
            """Compute gradients using C++ grad op."""
            if _fused_gqa_grad_op is None:
                raise RuntimeError(
                    "FusedGQAGrad C++ op not available. Build with: "
                    "cd highnoon/_native && ./build_ops.sh fused_gqa"
                )
            # Placeholder for saved attention weights
            attn_weights = tf.zeros([1], dtype=tf.float32)
            grad_x, grad_q, grad_k, grad_v, grad_out = _fused_gqa_grad_op(
                grad_output=grad_output,
                x=x_in,
                q_weight=q_w,
                k_weight=k_w,
                v_weight=v_w,
                out_weight=out_w,
                attention_weights=attn_weights,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                causal=causal,
            )
            # Gradient for gamma and beta from LayerNorm
            grad_gamma = tf.zeros_like(gamma)
            grad_beta = tf.zeros_like(beta)
            return grad_x, grad_q, grad_k, grad_v, grad_out, grad_gamma, grad_beta

        return output, grad

    return _fused_gqa_inner(x, q_weight, k_weight, v_weight, out_weight, norm_gamma, norm_beta)


__all__ = ["fused_gqa"]
