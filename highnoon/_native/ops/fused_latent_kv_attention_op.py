# highnoon/_native/ops/fused_latent_kv_attention_op.py
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

"""Python wrapper for Fused Latent KV Attention C++ op.

Phase 18.1: Implements MLA-style KV compression for efficient attention
with 4-8x memory reduction and O(n) linear attention option.
"""

from __future__ import annotations

import tensorflow as tf

from highnoon._native import get_op

# Load the C++ op library
_lib = get_op("fused_latent_kv_attention")
_fused_latent_kv_attention_op = getattr(_lib, "FusedLatentKVAttention", None) if _lib else None
_fused_latent_kv_attention_grad_op = (
    getattr(_lib, "FusedLatentKVAttentionGrad", None) if _lib else None
)


def fused_latent_kv_attention_available() -> bool:
    """Check if the C++ op is available."""
    return _fused_latent_kv_attention_op is not None


def fused_latent_kv_attention(
    x: tf.Tensor,
    q_proj: tf.Tensor,
    k_compress: tf.Tensor,
    v_compress: tf.Tensor,
    k_expand: tf.Tensor,
    v_expand: tf.Tensor,
    out_proj: tf.Tensor,
    latent_dim: int,
    num_heads: int,
    head_dim: int,
    use_linear_attention: bool = True,
    use_float64: bool = False,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Fused Latent KV Attention with KV compression.

    Phase 18.1: C++-accelerated MLA-style attention that compresses K/V to a
    latent space, reducing memory from O(n*d) to O(n*d_latent).

    Args:
        x: Input tensor [batch, seq_len, embed_dim].
        q_proj: Query projection [embed_dim, num_heads * head_dim].
        k_compress: Key compression matrix [embed_dim, latent_dim].
        v_compress: Value compression matrix [embed_dim, latent_dim].
        k_expand: Key expansion matrix [latent_dim, num_heads * head_dim].
        v_expand: Value expansion matrix [latent_dim, num_heads * head_dim].
        out_proj: Output projection [num_heads * head_dim, embed_dim].
        latent_dim: Size of the latent KV space.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        use_linear_attention: Use O(n) linear attention (default: True).
        use_float64: Use float64 precision for quantum mode (default: False).

    Returns:
        Tuple of (output, latent_k, latent_v):
        - output: Transformed output [batch, seq_len, embed_dim]
        - latent_k: Compressed keys [batch, seq_len, latent_dim]
        - latent_v: Compressed values [batch, seq_len, latent_dim]

    Raises:
        RuntimeError: If C++ op is not available and fallback fails.
    """
    if _fused_latent_kv_attention_op is None:
        return _latent_kv_attention_fallback(
            x,
            q_proj,
            k_compress,
            v_compress,
            k_expand,
            v_expand,
            out_proj,
            latent_dim,
            num_heads,
            head_dim,
            use_linear_attention,
            use_float64,
        )

    # Select dtype
    dtype = tf.float64 if use_float64 else tf.float32
    x = tf.cast(x, dtype)
    q_proj = tf.cast(q_proj, dtype)
    k_compress = tf.cast(k_compress, dtype)
    v_compress = tf.cast(v_compress, dtype)
    k_expand = tf.cast(k_expand, dtype)
    v_expand = tf.cast(v_expand, dtype)
    out_proj = tf.cast(out_proj, dtype)

    @tf.custom_gradient
    def _inner(x_in, qp, kc, vc, ke, ve, op):
        output, latent_k, latent_v = _fused_latent_kv_attention_op(
            x=x_in,
            q_proj=qp,
            k_compress=kc,
            v_compress=vc,
            k_expand=ke,
            v_expand=ve,
            out_proj=op,
            latent_dim=latent_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            use_linear_attention=use_linear_attention,
        )

        def grad(grad_output, grad_latent_k, grad_latent_v):
            if _fused_latent_kv_attention_grad_op is not None:
                grads = _fused_latent_kv_attention_grad_op(
                    grad_output=grad_output,
                    x=x_in,
                    q_proj=qp,
                    k_compress=kc,
                    v_compress=vc,
                    k_expand=ke,
                    v_expand=ve,
                    out_proj=op,
                    latent_dim=latent_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                )
                return grads
            else:
                return tuple(tf.zeros_like(t) for t in [x_in, qp, kc, vc, ke, ve, op])

        return (output, latent_k, latent_v), grad

    return _inner(x, q_proj, k_compress, v_compress, k_expand, v_expand, out_proj)


def _latent_kv_attention_fallback(
    x: tf.Tensor,
    q_proj: tf.Tensor,
    k_compress: tf.Tensor,
    v_compress: tf.Tensor,
    k_expand: tf.Tensor,
    v_expand: tf.Tensor,
    out_proj: tf.Tensor,
    latent_dim: int,
    num_heads: int,
    head_dim: int,
    use_linear_attention: bool,
    use_float64: bool,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Pure TensorFlow fallback for latent KV attention."""
    dtype = tf.float64 if use_float64 else tf.float32
    x = tf.cast(x, dtype)

    batch = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    embed_dim = tf.shape(x)[2]
    proj_dim = num_heads * head_dim

    # Reshape for batch matmul: [batch * seq_len, embed_dim]
    x_flat = tf.reshape(x, [-1, embed_dim])

    # Q = x @ q_proj
    Q = tf.matmul(x_flat, q_proj)
    Q = tf.reshape(Q, [batch, seq_len, proj_dim])

    # Compress K and V to latent space
    latent_k = tf.matmul(x_flat, k_compress)
    latent_k = tf.reshape(latent_k, [batch, seq_len, latent_dim])

    latent_v = tf.matmul(x_flat, v_compress)
    latent_v = tf.reshape(latent_v, [batch, seq_len, latent_dim])

    # Expand K and V for attention
    latent_k_flat = tf.reshape(latent_k, [-1, latent_dim])
    latent_v_flat = tf.reshape(latent_v, [-1, latent_dim])

    K = tf.matmul(latent_k_flat, k_expand)
    K = tf.reshape(K, [batch, seq_len, proj_dim])

    V = tf.matmul(latent_v_flat, v_expand)
    V = tf.reshape(V, [batch, seq_len, proj_dim])

    # Attention
    scale = tf.cast(1.0 / tf.sqrt(tf.cast(head_dim, dtype)), dtype)

    if use_linear_attention:
        # O(n) linear attention with ELU+1 feature map
        Q_feat = tf.where(Q > 0, Q * scale + 1.0, tf.exp(Q * scale))
        K_feat = tf.where(K > 0, K + 1.0, tf.exp(K))

        # K^T @ V: [batch, proj_dim, proj_dim]
        KV = tf.einsum("bsd,bsD->bdD", K_feat, V)

        # K.sum(0): [batch, proj_dim]
        K_sum = tf.reduce_sum(K_feat, axis=1, keepdims=True)

        # Q @ KV: [batch, seq_len, proj_dim]
        numerator = tf.einsum("bsd,bdD->bsD", Q_feat, KV)

        # Q @ K_sum: [batch, seq_len, 1]
        denominator = tf.einsum("bsd,bnd->bsn", Q_feat, K_sum)

        attn_out = numerator / (denominator + 1e-10)
    else:
        # O(n²) standard attention
        attn_scores = tf.einsum("bsd,btd->bst", Q, K) * scale
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_out = tf.einsum("bst,btd->bsd", attn_weights, V)

    # Output projection
    attn_out_flat = tf.reshape(attn_out, [-1, proj_dim])
    output = tf.matmul(attn_out_flat, out_proj)
    output = tf.reshape(output, [batch, seq_len, embed_dim])

    return output, latent_k, latent_v


def get_cache_reduction_ratio(embed_dim: int, latent_dim: int) -> float:
    """Calculate the KV cache memory reduction ratio.

    Args:
        embed_dim: Original embedding dimension.
        latent_dim: Latent KV dimension.

    Returns:
        Reduction ratio (e.g., 4.0 means 4x memory reduction).
    """
    return float(embed_dim) / float(latent_dim)


__all__ = [
    "fused_latent_kv_attention",
    "fused_latent_kv_attention_available",
    "get_cache_reduction_ratio",
]
