# highnoon/_native/ops/fused_local_attention_op.py
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

"""Python wrapper for FusedLocalAttention C++ kernel.

This module provides the Python interface for the Griffin-style local
attention C++ kernel with SIMD optimizations. The kernel implements
windowed local attention O(L·w²) with optional causal masking.

Usage:
    from highnoon._native.ops.fused_local_attention_op import fused_local_attention

    output = fused_local_attention(
        input_tensor,
        proj_q_kernel, proj_q_bias,
        proj_k_kernel, proj_k_bias,
        proj_v_kernel, proj_v_bias,
        out_proj_kernel, out_proj_bias,
        num_heads=8,
        window_size=128,
        causal=True
    )
"""

from __future__ import annotations

import logging

import tensorflow as tf

from highnoon._native import get_op

logger = logging.getLogger(__name__)

# Load the C++ library
_lib = get_op("fused_local_attention")
_fused_local_attention_op = _lib.fused_local_attention if _lib else None
_fused_local_attention_grad_op = _lib.fused_local_attention_grad if _lib else None


def fused_local_attention(
    input_tensor: tf.Tensor,
    proj_q_kernel: tf.Tensor,
    proj_q_bias: tf.Tensor,
    proj_k_kernel: tf.Tensor,
    proj_k_bias: tf.Tensor,
    proj_v_kernel: tf.Tensor,
    proj_v_bias: tf.Tensor,
    out_proj_kernel: tf.Tensor,
    out_proj_bias: tf.Tensor,
    num_heads: int = 8,
    window_size: int = 128,
    causal: bool = True,
    # Enhancement 1: Multi-Scale Window Attention
    use_multiscale: bool = False,
    window_min: int = 64,
    window_max: int = 512,
    # Enhancement 2: Sigmoid Attention
    use_sigmoid_attention: bool = False,
    sigmoid_temperature: float = 1.0,
    # Enhancement 3: ALiBi Position Bias
    use_alibi: bool = False,
    alibi_slope_base: float = 8.0,
    # Enhancement 4: Block-Sparse Optimization
    use_block_sparse: bool = False,
    block_size: int = 32,
    sparsity_ratio: float = 0.5,
    # Enhancement 5: Quantum Kernel Approximation
    use_quantum_kernel: bool = False,
    quantum_rank: int = 16,
) -> tf.Tensor:
    """C++-accelerated local attention operation with enhancements.

    Implements Griffin-style windowed local attention with O(L·w²) complexity.
    Positions can only attend to other positions within window_size distance.
    If causal=True, positions can only attend to earlier positions.

    Enhancements:
        - Multi-Scale Windows (MSWA): Per-head variable window sizes
        - Sigmoid Attention: Replace softmax to eliminate attention sink
        - ALiBi Position Bias: Linear position encoding for extrapolation
        - Block-Sparse: Dynamic block pruning for speedup
        - Quantum Kernel: O(n) fidelity-based attention (research)

    Args:
        input_tensor: Input tensor [batch, seq_len, embedding_dim].
        proj_q_kernel: Query projection weights [embedding_dim, embedding_dim].
        proj_q_bias: Query projection bias [embedding_dim].
        proj_k_kernel: Key projection weights [embedding_dim, embedding_dim].
        proj_k_bias: Key projection bias [embedding_dim].
        proj_v_kernel: Value projection weights [embedding_dim, embedding_dim].
        proj_v_bias: Value projection bias [embedding_dim].
        out_proj_kernel: Output projection weights [embedding_dim, embedding_dim].
        out_proj_bias: Output projection bias [embedding_dim].
        num_heads: Number of attention heads.
        window_size: Size of local attention window (used if use_multiscale=False).
        causal: Whether to apply causal masking.
        use_multiscale: Enable per-head variable window sizes.
        window_min: Minimum window size for multi-scale (head 0).
        window_max: Maximum window size for multi-scale (head N-1).
        use_sigmoid_attention: Replace softmax with sigmoid.
        sigmoid_temperature: Temperature scaling for sigmoid.
        use_alibi: Enable ALiBi position bias.
        alibi_slope_base: Base for ALiBi slope computation.
        use_block_sparse: Enable block-sparse pruning (reserved).
        block_size: Block size for sparse attention.
        sparsity_ratio: Ratio of blocks to keep (0.5 = 50%).
        use_quantum_kernel: Enable quantum-inspired attention (reserved).
        quantum_rank: Low-rank approximation rank for O(n).

    Returns:
        Output tensor [batch, seq_len, embedding_dim].

    Raises:
        RuntimeError: If C++ kernel is not available.
    """
    if _fused_local_attention_op is None:
        raise RuntimeError(
            "FusedLocalAttention C++ kernel not available. "
            "Build the native library with: cd highnoon/_native && ./build_ops.sh fused_local_attention"
        )

    # Ensure float32 for C++ kernel
    input_tensor = tf.cast(input_tensor, tf.float32)
    proj_q_kernel = tf.cast(proj_q_kernel, tf.float32)
    proj_q_bias = tf.cast(proj_q_bias, tf.float32)
    proj_k_kernel = tf.cast(proj_k_kernel, tf.float32)
    proj_k_bias = tf.cast(proj_k_bias, tf.float32)
    proj_v_kernel = tf.cast(proj_v_kernel, tf.float32)
    proj_v_bias = tf.cast(proj_v_bias, tf.float32)
    out_proj_kernel = tf.cast(out_proj_kernel, tf.float32)
    out_proj_bias = tf.cast(out_proj_bias, tf.float32)

    # Inner function for clean gradient handling with @tf.custom_gradient
    @tf.custom_gradient
    def _fused_local_attention_inner(inp, q_k, q_b, k_k, k_b, v_k, v_b, o_k, o_b):
        output, attn_cache, q_cache, k_cache, v_cache = _fused_local_attention_op(
            inp,
            q_k,
            q_b,
            k_k,
            k_b,
            v_k,
            v_b,
            o_k,
            o_b,
            num_heads=num_heads,
            window_size=window_size,
            causal=causal,
            # Enhancement 1
            use_multiscale=use_multiscale,
            window_min=window_min,
            window_max=window_max,
            # Enhancement 2
            use_sigmoid_attention=use_sigmoid_attention,
            sigmoid_temperature=sigmoid_temperature,
            # Enhancement 3
            use_alibi=use_alibi,
            alibi_slope_base=alibi_slope_base,
            # Enhancement 4
            use_block_sparse=use_block_sparse,
            block_size=block_size,
            sparsity_ratio=sparsity_ratio,
            # Enhancement 5
            use_quantum_kernel=use_quantum_kernel,
            quantum_rank=quantum_rank,
        )

        def grad(grad_output):
            grads = _fused_local_attention_grad_op(
                grad_output,
                inp,
                q_k,
                q_b,
                k_k,
                k_b,
                v_k,
                v_b,
                o_k,
                o_b,
                attn_cache,
                q_cache,
                k_cache,
                v_cache,
                num_heads=num_heads,
                window_size=window_size,
                causal=causal,
                # Enhancement 1
                use_multiscale=use_multiscale,
                window_min=window_min,
                window_max=window_max,
                # Enhancement 2
                use_sigmoid_attention=use_sigmoid_attention,
                sigmoid_temperature=sigmoid_temperature,
                # Enhancement 3
                use_alibi=use_alibi,
                alibi_slope_base=alibi_slope_base,
                # Enhancement 4
                use_block_sparse=use_block_sparse,
                block_size=block_size,
                sparsity_ratio=sparsity_ratio,
                # Enhancement 5
                use_quantum_kernel=use_quantum_kernel,
                quantum_rank=quantum_rank,
            )
            # Returns: (grad_input, grad_q_kernel, grad_q_bias, grad_k_kernel, grad_k_bias,
            #           grad_v_kernel, grad_v_bias, grad_out_kernel, grad_out_bias)
            return grads

        return output, grad

    return _fused_local_attention_inner(
        input_tensor,
        proj_q_kernel,
        proj_q_bias,
        proj_k_kernel,
        proj_k_bias,
        proj_v_kernel,
        proj_v_bias,
        out_proj_kernel,
        out_proj_bias,
    )


def fused_local_attention_layer(
    layer,
    x: tf.Tensor,
) -> tf.Tensor:
    """Helper to apply fused local attention using a LocalAttentionBlock layer's weights.

    This function extracts weights from an existing LocalAttentionBlock instance
    and applies the fused C++ kernel with all enhancement parameters.

    Args:
        layer: LocalAttentionBlock instance with weights.
        x: Input tensor [batch, seq_len, embedding_dim].

    Returns:
        Output tensor [batch, seq_len, embedding_dim].
    """
    return fused_local_attention(
        x,
        layer.proj_q.kernel,
        layer.proj_q.bias,
        layer.proj_k.kernel,
        layer.proj_k.bias,
        layer.proj_v.kernel,
        layer.proj_v.bias,
        layer.out_proj.kernel,
        layer.out_proj.bias,
        num_heads=layer.num_heads,
        window_size=layer.window_size,
        causal=layer.causal,
        # Enhancement 1: Multi-Scale Window Attention
        use_multiscale=getattr(layer, "use_multiscale", False),
        window_min=getattr(layer, "window_min", 64),
        window_max=getattr(layer, "window_max", 512),
        # Enhancement 2: Sigmoid Attention
        use_sigmoid_attention=getattr(layer, "use_sigmoid_attention", False),
        sigmoid_temperature=getattr(layer, "sigmoid_temperature", 1.0),
        # Enhancement 3: ALiBi Position Bias
        use_alibi=getattr(layer, "use_alibi", False),
        alibi_slope_base=getattr(layer, "alibi_slope_base", 8.0),
        # Enhancement 4: Block-Sparse Optimization
        use_block_sparse=getattr(layer, "use_block_sparse", False),
        block_size=getattr(layer, "block_size", 32),
        sparsity_ratio=getattr(layer, "sparsity_ratio", 0.5),
        # Enhancement 5: Quantum Kernel Approximation
        use_quantum_kernel=getattr(layer, "use_quantum_kernel", False),
        quantum_rank=getattr(layer, "quantum_rank", 16),
    )


__all__ = ["fused_local_attention", "fused_local_attention_layer"]
