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

    if _ctqw_aggregate_op is not None:
        return _ctqw_aggregate_op(x=x, time=time, use_cayley=use_cayley, sigma=sigma)

    # TensorFlow fallback
    return _ctqw_aggregate_tf(x, time, use_cayley, sigma)


def _ctqw_aggregate_tf(
    x: tf.Tensor,
    time: float,
    use_cayley: bool,
    sigma: float,
) -> tf.Tensor:
    """TensorFlow fallback for CTQW aggregation."""
    batch = tf.shape(x)[0]
    num_nodes = tf.shape(x)[1]
    embed_dim = tf.shape(x)[2]

    if sigma < 0:
        sigma = tf.sqrt(tf.cast(embed_dim, tf.float32))

    # Compute pairwise squared distances
    # x_i - x_j squared norm
    x_sq = tf.reduce_sum(x**2, axis=-1, keepdims=True)  # [B, N, 1]
    dist2 = x_sq + tf.transpose(x_sq, [0, 2, 1]) - 2 * tf.matmul(x, x, transpose_b=True)
    dist2 = tf.maximum(dist2, 0.0)  # Numerical stability

    # RBF kernel for adjacency
    adjacency = tf.exp(-dist2 / (sigma**2 + 1e-8))
    adjacency = adjacency * (1.0 - tf.eye(num_nodes, dtype=tf.float32))

    # Laplacian: L = D - A
    degree = tf.reduce_sum(adjacency, axis=-1, keepdims=True)
    laplacian = tf.linalg.diag(tf.squeeze(degree, -1)) - adjacency

    # Cayley approximation: (I - itL/2)(I + itL/2)^-1
    eye = tf.eye(num_nodes, batch_shape=[batch], dtype=tf.float32)
    alpha = time * 0.5

    I_plus = eye + alpha * laplacian
    I_minus = eye - alpha * laplacian

    # Solve via matrix inverse
    I_plus_inv = tf.linalg.inv(I_plus + 1e-6 * eye)
    evolution = tf.matmul(I_minus, I_plus_inv)

    # Probability amplitudes (squared)
    weights = evolution**2

    # Normalize rows
    weights = weights / (tf.reduce_sum(weights, axis=-1, keepdims=True) + 1e-8)

    return weights


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

    if _multi_rate_ema_op is not None:

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
                if _multi_rate_ema_grad_op is not None:
                    g_mem, g_agg = _multi_rate_ema_grad_op(
                        grad_output=grad_output,
                        level=level,
                        base_rate=base_rate,
                        level_decay=level_decay,
                    )
                    return g_mem, g_agg
                else:
                    alpha = base_rate * (level_decay**level)
                    return alpha * grad_output, (1.0 - alpha) * grad_output

            return output, grad

        return _inner(memory, aggregated)

    # TensorFlow fallback
    alpha = base_rate * (level_decay**level)
    return alpha * memory + (1.0 - alpha) * aggregated


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

    if _cross_level_attention_op is not None:
        return _cross_level_attention_op(
            query=query, key=key, value=value, num_heads=num_heads, residual_scale=residual_scale
        )

    # TensorFlow fallback with linear attention
    return _cross_level_attention_tf(query, key, value, num_heads, residual_scale)


def _cross_level_attention_tf(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    num_heads: int,
    residual_scale: float,
) -> tf.Tensor:
    """TensorFlow fallback for cross-level attention."""

    # ELU+1 kernel
    def elu_plus_1(x):
        return tf.nn.elu(x) + 1.0

    q_elu = elu_plus_1(query)  # [B, Nq, D]
    k_elu = elu_plus_1(key)  # [B, Nk, D]

    # KV summary: S[d1, d2] = sum_j k[j, d1] * v[j, d2]
    kv_sum = tf.einsum("bkd,bke->bde", k_elu, value)  # [B, D, D]
    k_sum = tf.reduce_sum(k_elu, axis=1, keepdims=True)  # [B, 1, D]

    # Output = (Q @ S) / (Q @ K_sum)
    numerator = tf.einsum("bqd,bde->bqe", q_elu, kv_sum)  # [B, Nq, D]
    denominator = tf.reduce_sum(q_elu * k_sum, axis=-1, keepdims=True)  # [B, Nq, 1]

    attended = numerator / (denominator + 1e-6)

    # Residual connection
    return query + residual_scale * attended


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

    if _adaptive_chunk_op is not None:
        return _adaptive_chunk_op(
            x=x,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            boundary_threshold=boundary_threshold,
        )

    # TensorFlow fallback
    return _adaptive_chunk_tf(x, min_chunk_size, max_chunk_size, boundary_threshold)


def _adaptive_chunk_tf(
    x: tf.Tensor,
    min_chunk_size: int,
    max_chunk_size: int,
    boundary_threshold: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """TensorFlow fallback for adaptive chunking."""
    batch = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]

    # Compute cosine similarity between adjacent tokens
    x_norm = tf.nn.l2_normalize(x, axis=-1)
    x_curr = x_norm[:, :-1]  # [B, S-1, D]
    x_next = x_norm[:, 1:]  # [B, S-1, D]
    tf.reduce_sum(x_curr * x_next, axis=-1)  # [B, S-1]

    # Simple chunking: evenly divide (for TF fallback simplicity)
    # Real implementation uses min-max constraints and boundary detection
    avg_chunk = (min_chunk_size + max_chunk_size) // 2
    num_chunks = tf.maximum(1, seq_len // avg_chunk)

    # Create chunk IDs
    indices = tf.range(seq_len)
    chunk_ids = tf.minimum(indices // avg_chunk, num_chunks - 1)
    chunk_ids = tf.broadcast_to(chunk_ids, [batch, seq_len])

    num_chunks_out = tf.fill([batch], tf.cast(num_chunks, tf.int32))

    return tf.cast(chunk_ids, tf.int32), num_chunks_out


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

    if _chunk_pool_op is not None:
        return _chunk_pool_op(x=x, chunk_ids=chunk_ids, num_chunks=num_chunks)

    # TensorFlow fallback
    batch = tf.shape(x)[0]
    tf.shape(x)[2]
    max_chunks = tf.reduce_max(num_chunks)

    # Use segment_mean per batch
    outputs = []
    for b in tf.range(batch):
        x_b = x[b]  # [S, D]
        ids_b = chunk_ids[b]  # [S]
        n_chunks = num_chunks[b]

        # Segment mean
        pooled_b = tf.math.unsorted_segment_mean(x_b, ids_b, n_chunks)

        # Pad to max_chunks
        padding = [[0, max_chunks - n_chunks], [0, 0]]
        pooled_b = tf.pad(pooled_b, padding)
        outputs.append(pooled_b)

    return tf.stack(outputs, axis=0)


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
    if _quantum_noise_op is not None:
        shape_tensor = tf.constant(list(shape), dtype=tf.int32)
        return _quantum_noise_op(
            shape=shape_tensor, entanglement_strength=entanglement_strength, seed=seed
        )

    # TensorFlow fallback
    return _quantum_noise_tf(shape, entanglement_strength, seed)


def _quantum_noise_tf(
    shape: tuple[int, ...],
    entanglement_strength: float,
    seed: int,
) -> tf.Tensor:
    """TensorFlow fallback for quantum noise."""
    tf.random.set_seed(seed)

    # Base Gaussian noise
    noise = tf.random.normal(shape, dtype=tf.float32)

    if len(shape) < 2:
        return noise

    batch = shape[0]
    dim = shape[1] if len(shape) > 1 else 1

    # Apply rotation to pairs of dimensions
    for d in range(0, dim - 1, 2):
        theta = tf.random.uniform([batch, 1]) * entanglement_strength * 2 * 3.14159
        cos_t = tf.cos(theta)
        sin_t = tf.sin(theta)

        x = noise[:, d : d + 1]
        y = noise[:, d + 1 : d + 2]

        # Rotation
        x_new = cos_t * x - sin_t * y
        y_new = sin_t * x + cos_t * y

        # Update (using concat for simplicity)
        if d == 0:
            rotated = tf.concat([x_new, y_new], axis=1)
        else:
            rotated = tf.concat([rotated, x_new, y_new], axis=1)

    # Handle odd dimension
    if dim % 2 == 1:
        rotated = tf.concat([rotated, noise[:, -1:]], axis=1)

    # Apply entanglement between adjacent samples
    if entanglement_strength > 0:
        for i in range(1, batch):
            rotated_i = rotated[i : i + 1]
            prev = rotated[i - 1 : i]
            blended = (1 - entanglement_strength) * rotated_i + entanglement_strength * prev
            rotated = tf.concat([rotated[:i], blended, rotated[i + 1 :]], axis=0)

    return rotated


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
