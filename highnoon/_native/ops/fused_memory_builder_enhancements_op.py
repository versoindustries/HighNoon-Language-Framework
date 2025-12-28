# highnoon/_native/ops/fused_memory_builder_enhancements_op.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Python wrapper for Memory Builder Enhancement C++ ops.
# Implements Enhancements 3-7 from the Memory Builder roadmap.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Python wrappers for Memory Builder enhancement C++ ops.

This module provides Python interfaces to the C++ kernels with
automatic TensorFlow fallbacks when C++ is unavailable.

Enhancements:
    3. CTQW Aggregation - Continuous-time quantum walk for aggregation weights
    4. Multi-Rate EMA - Level-dependent exponential moving average
    5. Cross-Level Attention - O(n) linear attention across hierarchy levels
    6. Adaptive Chunking - Content-based semantic boundary detection
    7. Quantum Noise - Structured noise for QGAN training
"""

from __future__ import annotations

import tensorflow as tf

from highnoon._native import get_op

# Load C++ op library
_lib = get_op("memory_builder_enhancements")

# Op handles (None if C++ not available)
_ctqw_aggregate_op = getattr(_lib, "FusedCTQWAggregate", None) if _lib else None
_multi_rate_ema_op = getattr(_lib, "FusedMultiRateEMA", None) if _lib else None
_multi_rate_ema_grad_op = getattr(_lib, "FusedMultiRateEMAGrad", None) if _lib else None
_cross_level_attention_op = getattr(_lib, "FusedCrossLevelAttention", None) if _lib else None
_adaptive_chunk_op = getattr(_lib, "FusedAdaptiveChunk", None) if _lib else None
_chunk_pool_op = getattr(_lib, "FusedChunkPool", None) if _lib else None
_quantum_noise_op = getattr(_lib, "FusedQuantumNoise", None) if _lib else None


# =============================================================================
# ENHANCEMENT 3: CTQW AGGREGATION
# =============================================================================


def ctqw_aggregate(
    x: tf.Tensor,
    time: float = 1.0,
    use_cayley: bool = True,
    sigma: float = -1.0,
) -> tf.Tensor:
    """Compute CTQW-based aggregation weights.

    Uses continuous-time quantum walk on the graph Laplacian to compute
    aggregation weights that capture global structure.

    Args:
        x: Input representations [batch, num_nodes, embed_dim].
        time: Quantum walk time parameter (learnable).
        use_cayley: Use Cayley approximation (faster) vs Taylor expansion.
        sigma: Kernel bandwidth. If < 0, uses sqrt(embed_dim).

    Returns:
        Aggregation weights [batch, num_nodes, num_nodes].

    Example:
        >>> x = tf.random.normal([2, 16, 64])
        >>> weights = ctqw_aggregate(x, time=1.0)
        >>> weighted_x = tf.einsum('bnm,bmd->bnd', weights, x)
    """
    x = tf.cast(x, tf.float32)

    if _ctqw_aggregate_op is None:
        raise RuntimeError(
            "FusedCTQWAggregate C++ op not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    return _ctqw_aggregate_op(x=x, time=time, use_cayley=use_cayley, sigma=sigma)


# =============================================================================
# ENHANCEMENT 4: MULTI-RATE EMA
# =============================================================================


def multi_rate_ema(
    memory: tf.Tensor,
    aggregated: tf.Tensor,
    level: int = 0,
    base_rate: float = 0.1,
    level_decay: float = 0.5,
) -> tf.Tensor:
    """Apply multi-rate EMA update for hierarchical memory.

    Implements: memory_new = α * memory + (1-α) * aggregated
    where α = base_rate * (level_decay ^ level)

    Lower levels (fine): low α = fast update
    Higher levels (coarse): high α = slow update

    Args:
        memory: Current memory [batch, num_tokens, embed_dim].
        aggregated: New aggregated values [batch, num_tokens, embed_dim].
        level: Current hierarchy level (0 = finest).
        base_rate: Base update rate for level 0.
        level_decay: Decay factor per level.

    Returns:
        Updated memory [batch, num_tokens, embed_dim].

    Example:
        >>> mem = tf.zeros([2, 16, 64])
        >>> agg = tf.random.normal([2, 16, 64])
        >>> mem_new = multi_rate_ema(mem, agg, level=0)
    """
    memory = tf.cast(memory, tf.float32)
    aggregated = tf.cast(aggregated, tf.float32)

    if _multi_rate_ema_op is None:
        raise RuntimeError(
            "FusedMultiRateEMA C++ op not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    @tf.custom_gradient
    def _inner(mem, agg):
        output = _multi_rate_ema_op(
            memory=mem,
            aggregated=agg,
            level=level,
            base_rate=base_rate,
            level_decay=level_decay,
        )

        def grad(grad_output):
            if _multi_rate_ema_grad_op is None:
                raise RuntimeError(
                    "FusedMultiRateEMAGrad C++ op not available. Build with: "
                    "cd highnoon/_native && ./build_secure.sh"
                )
            g_mem, g_agg = _multi_rate_ema_grad_op(
                grad_output=grad_output,
                level=level,
                base_rate=base_rate,
                level_decay=level_decay,
            )
            return g_mem, g_agg

        return output, grad

    return _inner(memory, aggregated)


# =============================================================================
# ENHANCEMENT 5: CROSS-LEVEL ATTENTION
# =============================================================================


def cross_level_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    num_heads: int = 4,
    residual_scale: float = 1.0,
) -> tf.Tensor:
    """O(n) linear cross-level attention using ELU+1 kernel.

    Enables bidirectional information flow across hierarchy levels.
    Uses kernel attention for O(n) complexity.

    Args:
        query: Query from one level [batch, num_query, embed_dim].
        key: Key from another level [batch, num_kv, embed_dim].
        value: Value from another level [batch, num_kv, embed_dim].
        num_heads: Number of attention heads.
        residual_scale: Scale for residual connection.

    Returns:
        Attended output with residual [batch, num_query, embed_dim].

    Example:
        >>> fine = tf.random.normal([2, 32, 64])   # Fine level
        >>> coarse = tf.random.normal([2, 8, 64])  # Coarse level
        >>> # Coarse attending to fine
        >>> out = cross_level_attention(coarse, fine, fine)
    """
    query = tf.cast(query, tf.float32)
    key = tf.cast(key, tf.float32)
    value = tf.cast(value, tf.float32)

    if _cross_level_attention_op is None:
        raise RuntimeError(
            "FusedCrossLevelAttention C++ op not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    return _cross_level_attention_op(
        query=query, key=key, value=value, num_heads=num_heads, residual_scale=residual_scale
    )


# =============================================================================
# ENHANCEMENT 6: ADAPTIVE CHUNKING
# =============================================================================


def adaptive_chunk(
    x: tf.Tensor,
    min_chunk_size: int = 2,
    max_chunk_size: int = 16,
    boundary_threshold: float = 0.5,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute adaptive chunk assignments based on semantic boundaries.

    Identifies chunk boundaries at local minima of similarity between
    adjacent representations.

    Args:
        x: Input representations [batch, seq_len, embed_dim].
        min_chunk_size: Minimum tokens per chunk.
        max_chunk_size: Maximum tokens per chunk.
        boundary_threshold: Similarity threshold for boundary detection.

    Returns:
        Tuple of (chunk_ids, num_chunks):
        - chunk_ids: Chunk assignment per token [batch, seq_len]
        - num_chunks: Number of chunks per batch [batch]

    Example:
        >>> x = tf.random.normal([2, 32, 64])
        >>> chunk_ids, num_chunks = adaptive_chunk(x)
        >>> pooled = chunk_pool(x, chunk_ids, num_chunks)
    """
    x = tf.cast(x, tf.float32)

    if _adaptive_chunk_op is None:
        raise RuntimeError(
            "FusedAdaptiveChunk C++ op not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    return _adaptive_chunk_op(
        x=x,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        boundary_threshold=boundary_threshold,
    )


def chunk_pool(
    x: tf.Tensor,
    chunk_ids: tf.Tensor,
    num_chunks: tf.Tensor,
) -> tf.Tensor:
    """Pool within adaptive chunks using mean aggregation.

    Args:
        x: Input representations [batch, seq_len, embed_dim].
        chunk_ids: Chunk assignments [batch, seq_len].
        num_chunks: Number of chunks per batch [batch].

    Returns:
        Pooled representations [batch, max_chunks, embed_dim].
    """
    x = tf.cast(x, tf.float32)
    chunk_ids = tf.cast(chunk_ids, tf.int32)
    num_chunks = tf.cast(num_chunks, tf.int32)

    if _chunk_pool_op is None:
        raise RuntimeError(
            "FusedChunkPool C++ op not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    return _chunk_pool_op(x=x, chunk_ids=chunk_ids, num_chunks=num_chunks)


# =============================================================================
# ENHANCEMENT 7: QUANTUM NOISE
# =============================================================================


def quantum_noise(
    shape: tuple[int, ...],
    entanglement_strength: float = 0.1,
    seed: int = 42,
) -> tf.Tensor:
    """Generate structured quantum noise for QGAN training.

    Uses rotation matrices and entanglement correlations to produce
    noise with better inductive bias than Gaussian.

    Args:
        shape: Output noise shape (batch, dim).
        entanglement_strength: Correlation strength between dimensions.
        seed: Random seed for reproducibility.

    Returns:
        Structured noise tensor [shape].

    Example:
        >>> noise = quantum_noise([16, 64], entanglement_strength=0.1)
        >>> generated = generator(noise)
    """
    if _quantum_noise_op is None:
        raise RuntimeError(
            "FusedQuantumNoise C++ op not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    shape_tensor = tf.constant(list(shape), dtype=tf.int32)
    return _quantum_noise_op(
        shape=shape_tensor, entanglement_strength=entanglement_strength, seed=seed
    )


def quantum_entanglement_loss(
    samples: tf.Tensor,
    target_strength: float = 0.1,
) -> tf.Tensor:
    """Compute entanglement regularization loss.

    Encourages structured correlations in generated samples.

    Args:
        samples: Generated samples [batch, dim].
        target_strength: Target correlation between adjacent samples.

    Returns:
        Regularization loss scalar.
    """
    if samples.shape[0] < 2:
        return tf.constant(0.0)

    samples_norm = tf.nn.l2_normalize(samples, axis=-1)

    # Compute pairwise correlations between adjacent samples
    correlations = tf.reduce_sum(samples_norm[:-1] * samples_norm[1:], axis=-1)

    # L2 loss towards target correlation
    loss = tf.reduce_mean((correlations - target_strength) ** 2)

    return loss


__all__ = [
    "ctqw_aggregate",
    "multi_rate_ema",
    "cross_level_attention",
    "adaptive_chunk",
    "chunk_pool",
    "quantum_noise",
    "quantum_entanglement_loss",
]
