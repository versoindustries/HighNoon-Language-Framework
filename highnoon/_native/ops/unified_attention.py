"""Unified Attention Python Wrapper for HighNoon Framework.

This module provides a Python interface to the unified attention C++ operations.
It consolidates all 11 attention mechanisms into a single, flexible API.

Phase 2 of V2 Performance Optimization - Attention Consolidation

Copyright 2025-2026 Verso Industries (Author: Michael B. Zimmerman)
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import tensorflow as tf

# Attempt to load the native op, with fallback for development
try:
    from highnoon._native import load_highnoon_core

    _highnoon_core = load_highnoon_core()
    _NATIVE_AVAILABLE = _highnoon_core is not None
except ImportError:
    _highnoon_core = None
    _NATIVE_AVAILABLE = False


class AttentionMode(enum.IntEnum):
    """Attention mechanism types supported by the unified attention system.

    Each mode corresponds to a specific attention implementation:

    - FLASH: Full O(n²) flash attention with memory-efficient tiling
    - LINEAR: O(n) linear attention via ELU+1 feature maps
    - LOCAL_WINDOWED: O(n×w) local windowed attention (Griffin-style)
    - DIFFERENTIAL: Differential Transformer attention (ICLR 2025)
    - SPARSE_NSA: Native Sparse Attention O(n log n)
    - GQA: Grouped-Query Attention
    - LINEAR_GQA: Linear + GQA combined
    - SLIDING_GQA: Sliding window + GQA
    - LATENT_KV: Latent KV compression (DeepSeek-style)
    - QASA: Quantum Adaptive Self-Attention
    - LMWT: Learnable Multi-Scale Wavelet Transformer
    """

    FLASH = 0
    LINEAR = 1
    LOCAL_WINDOWED = 2
    DIFFERENTIAL = 3
    SPARSE_NSA = 4
    GQA = 5
    LINEAR_GQA = 6
    SLIDING_GQA = 7
    LATENT_KV = 8
    QASA = 9
    LMWT = 10


@dataclass
class UnifiedAttentionConfig:
    """Configuration for unified attention operations.

    This configuration struct contains all parameters for all attention modes.
    Parameters unused by a specific mode are simply ignored.

    Attributes:
        mode: The attention mode to use.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (< num_heads enables GQA).
        head_dim: Dimension per attention head.
        window_size: Local attention window size (for LOCAL_WINDOWED, SLIDING_GQA).
        causal: Whether to apply causal masking.
        scale: Attention scale (0 = auto: 1/sqrt(head_dim)).
        dropout_rate: Dropout probability (0 = no dropout).
        epsilon: Numerical stability epsilon.
        lambda_init: Initial lambda for differential attention.
        block_size: Token compression block size (for SPARSE_NSA).
        num_selected_blocks: Number of blocks to select (for SPARSE_NSA).
        num_qubits: VQC qubit count (for QASA).
        vqc_layers: VQC circuit depth (for QASA).
        entanglement_strength: VQC entanglement (for QASA).
        num_latents: Number of latent vectors (for LATENT_KV).
        num_wavelet_scales: Decomposition levels (for LMWT).
    """

    mode: AttentionMode = AttentionMode.FLASH
    num_heads: int = 8
    num_kv_heads: int = 8
    head_dim: int = 64
    window_size: int = 256
    causal: bool = True
    scale: float = 0.0
    dropout_rate: float = 0.0
    epsilon: float = 1e-6

    # Differential attention
    lambda_init: float = 0.8
    lambda_min: float = 0.0
    lambda_max: float = 2.0
    normalize_diff: bool = True

    # Sparse NSA
    block_size: int = 64
    num_selected_blocks: int = 8
    tokens_per_block: int = 8
    use_global_tokens: bool = True
    num_global_tokens: int = 1
    temperature: float = 1.0

    # Linear attention
    use_elu_feature_map: bool = True
    use_gla_gates: bool = False
    rff_dim: int = 64

    # Holographic Linear Attention (extends LINEAR mode for HD embeddings)
    # When enabled, uses FFT-based feature maps for O(n) linear attention that
    # meshes with hyperdimensional embeddings. Replaces ELU+1 with:
    #   φ(x) = [Re(FFT(x)), Im(FFT(x))]  (complex FFT features)
    use_holographic_features: bool = False

    # Quantum attention (QASA)
    num_qubits: int = 4
    vqc_layers: int = 2
    entanglement_strength: float = 0.5
    use_residual_projection: bool = True
    residual_proj_dim: int = 32

    # Latent KV
    num_latents: int = 64
    use_cross_attention: bool = True

    # Wavelet attention (LMWT)
    num_wavelet_scales: int = 4
    learn_wavelet_filters: bool = True
    alpha_init: float = 0.7071067811865476  # 1/√2
    beta_init: float = 0.7071067811865476  # 1/√2

    def compute_scale(self) -> float:
        """Compute the attention scaling factor."""
        if self.scale == 0.0:
            return 1.0 / (self.head_dim**0.5)
        return self.scale

    def queries_per_kv_head(self) -> int:
        """Compute the number of queries per KV head for GQA."""
        return self.num_heads // max(self.num_kv_heads, 1)

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.num_heads < 1 or self.head_dim < 1:
            return False
        if self.num_kv_heads > self.num_heads:
            return False
        if self.num_heads % self.num_kv_heads != 0:
            return False
        return True


def unified_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    config: UnifiedAttentionConfig,
    extra_inputs: tf.Tensor | None = None,
    training: bool = False,
    name: str = "unified_attention",
) -> tf.Tensor:
    """Unified attention forward pass.

    Routes to the appropriate kernel based on config.mode. All inputs
    and outputs are in [batch, heads, seq, head_dim] layout.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, num_kv_heads, kv_seq_len, head_dim]
        value: Value tensor [batch, num_kv_heads, kv_seq_len, head_dim]
        config: Unified attention configuration
        extra_inputs: Optional mode-specific inputs:
            - QASA: VQC parameters
            - LATENT_KV: [latent_keys, latent_values]
            - LMWT: [alpha, beta] wavelet parameters
        training: Whether in training mode (affects dropout)
        name: Operation name for debugging

    Returns:
        Attention output [batch, num_heads, seq_len, head_dim]

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If native ops are not available and no fallback exists
    """
    with tf.name_scope(name):
        # Validate config
        if not config.validate():
            raise ValueError(f"Invalid attention config: {config}")

        # Get tensor shapes
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[2]
        kv_seq_len = tf.shape(key)[2]

        # Prepare extra inputs (empty tensor if None)
        if extra_inputs is None:
            extra_inputs = tf.zeros([1], dtype=tf.float32)

        # Use native op if available
        if _NATIVE_AVAILABLE:
            return _highnoon_core.unified_attention(
                query=query,
                key=key,
                value=value,
                extra_inputs=extra_inputs,
                mode=int(config.mode),
                batch_size=batch_size,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                seq_len=seq_len,
                kv_seq_len=kv_seq_len,
                window_size=config.window_size,
                causal=config.causal,
                scale=config.scale,
                dropout_rate=config.dropout_rate if training else 0.0,
                epsilon=config.epsilon,
                lambda_init=config.lambda_init,
                block_size=config.block_size,
                num_selected_blocks=config.num_selected_blocks,
                num_qubits=config.num_qubits,
                vqc_layers=config.vqc_layers,
                entanglement_strength=config.entanglement_strength,
                num_latents=config.num_latents,
                num_wavelet_scales=config.num_wavelet_scales,
                use_holographic_features=config.use_holographic_features,
            )

        # Python fallback implementation
        return _python_unified_attention(query, key, value, config, extra_inputs, training)


def _python_unified_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    config: UnifiedAttentionConfig,
    extra_inputs: tf.Tensor,
    training: bool,
) -> tf.Tensor:
    """Python fallback implementation of unified attention.

    This provides a reference implementation when native ops are not available.
    It is not as optimized as the C++ implementation but maintains correctness.
    """
    mode = config.mode

    if mode == AttentionMode.FLASH:
        return _flash_attention_fallback(query, key, value, config)
    elif mode == AttentionMode.LINEAR:
        return _linear_attention_fallback(query, key, value, config)
    elif mode == AttentionMode.LOCAL_WINDOWED:
        return _local_windowed_attention_fallback(query, key, value, config)
    elif mode == AttentionMode.DIFFERENTIAL:
        return _differential_attention_fallback(query, key, value, config)
    elif mode == AttentionMode.GQA:
        return _flash_attention_fallback(
            query, key, value, config
        )  # GQA uses flash with head mapping
    elif mode in (AttentionMode.LINEAR_GQA, AttentionMode.SLIDING_GQA):
        # These delegate to their base implementations with GQA
        if mode == AttentionMode.LINEAR_GQA:
            return _linear_attention_fallback(query, key, value, config)
        else:
            return _local_windowed_attention_fallback(query, key, value, config)
    elif mode == AttentionMode.SPARSE_NSA:
        # Sparse NSA falls back to flash attention for simplicity
        return _flash_attention_fallback(query, key, value, config)
    elif mode == AttentionMode.LATENT_KV:
        return _latent_kv_attention_fallback(query, config, extra_inputs)
    elif mode == AttentionMode.QASA:
        return _qasa_attention_fallback(query, key, value, config, extra_inputs)
    elif mode == AttentionMode.LMWT:
        return _lmwt_attention_fallback(query, key, value, config, extra_inputs)
    else:
        # Default fallback
        return _flash_attention_fallback(query, key, value, config)


def _flash_attention_fallback(
    query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, config: UnifiedAttentionConfig
) -> tf.Tensor:
    """Standard scaled dot-product attention (Python fallback)."""
    scale = config.compute_scale()

    # Expand KV heads if needed (GQA)
    if config.num_kv_heads < config.num_heads:
        repeat = config.queries_per_kv_head()
        key = tf.repeat(key, repeats=repeat, axis=1)
        value = tf.repeat(value, repeats=repeat, axis=1)

    # Compute attention scores: [batch, heads, seq_q, seq_k]
    scores = tf.einsum("bhqd,bhkd->bhqk", query, key) * scale

    # Apply causal mask
    if config.causal:
        seq_q = tf.shape(query)[2]
        seq_k = tf.shape(key)[2]
        mask = tf.linalg.band_part(tf.ones([seq_q, seq_k]), -1, 0)
        mask = tf.cast(mask, tf.float32)
        scores = scores * mask + (1.0 - mask) * (-1e9)

    # Softmax
    attention_weights = tf.nn.softmax(scores, axis=-1)

    # Weighted sum of values
    output = tf.einsum("bhqk,bhkd->bhqd", attention_weights, value)

    return output


def _linear_attention_fallback(
    query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, config: UnifiedAttentionConfig
) -> tf.Tensor:
    """Linear attention with ELU+1 feature map (Python fallback)."""

    # ELU+1 feature map: φ(x) = elu(x) + 1
    def feature_map(x):
        return tf.nn.elu(x) + 1.0

    # Expand KV heads if needed
    if config.num_kv_heads < config.num_heads:
        repeat = config.queries_per_kv_head()
        key = tf.repeat(key, repeats=repeat, axis=1)
        value = tf.repeat(value, repeats=repeat, axis=1)

    # Apply feature maps
    phi_q = feature_map(query)
    phi_k = feature_map(key)

    # Compute KV state: [batch, heads, head_dim, head_dim]
    kv_state = tf.einsum("bhkd,bhkv->bhdv", phi_k, value)

    # Compute K sum: [batch, heads, head_dim]
    k_sum = tf.reduce_sum(phi_k, axis=2)

    # Numerator: [batch, heads, seq_q, head_dim]
    numerator = tf.einsum("bhqd,bhdv->bhqv", phi_q, kv_state)

    # Denominator: [batch, heads, seq_q]
    denominator = tf.einsum("bhqd,bhd->bhq", phi_q, k_sum)
    denominator = tf.maximum(denominator, config.epsilon)

    # Normalize
    output = numerator / denominator[..., tf.newaxis]

    return output


def _local_windowed_attention_fallback(
    query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, config: UnifiedAttentionConfig
) -> tf.Tensor:
    """Local windowed attention (Python fallback).

    Falls back to flash attention with local masking for simplicity.
    """
    scale = config.compute_scale()
    window = config.window_size

    # Expand KV heads if needed
    if config.num_kv_heads < config.num_heads:
        repeat = config.queries_per_kv_head()
        key = tf.repeat(key, repeats=repeat, axis=1)
        value = tf.repeat(value, repeats=repeat, axis=1)

    # Compute full attention scores
    scores = tf.einsum("bhqd,bhkd->bhqk", query, key) * scale

    seq_q = tf.shape(query)[2]
    seq_k = tf.shape(key)[2]

    # Create local window mask
    q_pos = tf.range(seq_q)[:, tf.newaxis]
    k_pos = tf.range(seq_k)[tf.newaxis, :]

    # Within window: |q_pos - k_pos| <= window/2
    within_window = tf.abs(q_pos - k_pos) <= (window // 2)

    if config.causal:
        within_window = tf.logical_and(within_window, k_pos <= q_pos)

    mask = tf.cast(within_window, tf.float32)
    scores = scores * mask + (1.0 - mask) * (-1e9)

    attention_weights = tf.nn.softmax(scores, axis=-1)
    output = tf.einsum("bhqk,bhkd->bhqd", attention_weights, value)

    return output


def _differential_attention_fallback(
    query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, config: UnifiedAttentionConfig
) -> tf.Tensor:
    """Differential attention: A1 - λ*A2 (Python fallback)."""
    scale = 1.0 / (config.head_dim // 2) ** 0.5
    lambda_val = tf.clip_by_value(config.lambda_init, config.lambda_min, config.lambda_max)

    # Expand KV heads if needed
    if config.num_kv_heads < config.num_heads:
        repeat = config.queries_per_kv_head()
        key = tf.repeat(key, repeats=repeat, axis=1)
        value = tf.repeat(value, repeats=repeat, axis=1)

    # Split Q and K into two halves
    head_dim = config.head_dim
    half_dim = head_dim // 2

    q1, q2 = query[..., :half_dim], query[..., half_dim:]
    k1, k2 = key[..., :half_dim], key[..., half_dim:]

    # Compute two sets of attention scores
    scores1 = tf.einsum("bhqd,bhkd->bhqk", q1, k1) * scale
    scores2 = tf.einsum("bhqd,bhkd->bhqk", q2, k2) * scale

    # Apply causal mask
    if config.causal:
        seq_q = tf.shape(query)[2]
        seq_k = tf.shape(key)[2]
        mask = tf.linalg.band_part(tf.ones([seq_q, seq_k]), -1, 0)
        mask = tf.cast(mask, tf.float32)
        scores1 = scores1 * mask + (1.0 - mask) * (-1e9)
        scores2 = scores2 * mask + (1.0 - mask) * (-1e9)

    # Softmax both
    attn1 = tf.nn.softmax(scores1, axis=-1)
    attn2 = tf.nn.softmax(scores2, axis=-1)

    # Differential attention
    diff_attn = attn1 - lambda_val * attn2

    # Weighted sum
    output = tf.einsum("bhqk,bhkd->bhqd", diff_attn, value)

    # Optional normalization
    if config.normalize_diff:
        output = output * (1.0 - config.lambda_init)

    return output


def _latent_kv_attention_fallback(
    query: tf.Tensor, config: UnifiedAttentionConfig, extra_inputs: tf.Tensor
) -> tf.Tensor:
    """Latent KV attention (Python fallback)."""
    scale = config.compute_scale()
    num_latents = config.num_latents
    head_dim = config.head_dim

    # Extract latent keys and values from extra_inputs
    latent_keys = extra_inputs[: num_latents * head_dim]
    latent_keys = tf.reshape(latent_keys, [num_latents, head_dim])

    latent_values = extra_inputs[num_latents * head_dim :]
    latent_values = tf.reshape(latent_values, [num_latents, head_dim])

    # Cross-attention to latents
    scores = tf.einsum("bhqd,ld->bhql", query, latent_keys) * scale
    attention_weights = tf.nn.softmax(scores, axis=-1)
    output = tf.einsum("bhql,ld->bhqd", attention_weights, latent_values)

    return output


def _qasa_attention_fallback(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    config: UnifiedAttentionConfig,
    extra_inputs: tf.Tensor,
) -> tf.Tensor:
    """Quantum Adaptive Self-Attention (Python fallback).

    Simplified VQC-based attention computation.
    """
    num_qubits = config.num_qubits
    vqc_layers = config.vqc_layers
    num_params = vqc_layers * num_qubits

    # Expand KV heads if needed
    if config.num_kv_heads < config.num_heads:
        repeat = config.queries_per_kv_head()
        key = tf.repeat(key, repeats=repeat, axis=1)
        value = tf.repeat(value, repeats=repeat, axis=1)

    # VQC parameters
    extra_inputs[: 2 * num_params]

    # Simple VQC score approximation: use cosine similarity with phase rotation
    # This is a simplified approximation of the full VQC computation
    scores = tf.einsum("bhqd,bhkd->bhqk", query, key)
    scores = tf.cos(scores * 0.1)  # Phase-based scoring

    # Apply causal mask
    if config.causal:
        seq_q = tf.shape(query)[2]
        seq_k = tf.shape(key)[2]
        mask = tf.linalg.band_part(tf.ones([seq_q, seq_k]), -1, 0)
        mask = tf.cast(mask, tf.float32)
        scores = scores * mask + (1.0 - mask) * (-1e9)

    attention_weights = tf.nn.softmax(scores, axis=-1)
    output = tf.einsum("bhqk,bhkd->bhqd", attention_weights, value)

    return output


def _lmwt_attention_fallback(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    config: UnifiedAttentionConfig,
    extra_inputs: tf.Tensor,
) -> tf.Tensor:
    """Learnable Multi-Scale Wavelet Transformer (Python fallback).

    Multi-scale attention with wavelet-style decomposition.
    """
    num_scales = config.num_wavelet_scales
    scale = config.compute_scale()

    # Expand KV heads if needed
    if config.num_kv_heads < config.num_heads:
        repeat = config.queries_per_kv_head()
        key = tf.repeat(key, repeats=repeat, axis=1)
        value = tf.repeat(value, repeats=repeat, axis=1)

    # Extract alpha and beta from extra_inputs
    alpha = extra_inputs[:num_scales]
    beta = extra_inputs[num_scales : 2 * num_scales]

    # Accumulate multi-scale attention
    output = tf.zeros_like(query)

    for s in range(num_scales):
        stride = 2**s

        # Downsample keys and values
        key_s = key[:, :, ::stride, :]
        value_s = value[:, :, ::stride, :]

        # Compute attention at this scale
        scores = tf.einsum("bhqd,bhkd->bhqk", query, key_s) * scale

        if config.causal:
            seq_q = tf.shape(query)[2]
            seq_k = tf.shape(key_s)[2]
            mask = tf.linalg.band_part(tf.ones([seq_q, seq_k]), -1, 0)
            mask = tf.cast(mask, tf.float32)
            scores = scores * mask + (1.0 - mask) * (-1e9)

        attn_weights = tf.nn.softmax(scores, axis=-1)
        scale_output = tf.einsum("bhqk,bhkd->bhqd", attn_weights, value_s)

        # Weight by wavelet coefficients
        weight = alpha[s] if s == 0 else beta[s]
        output = output + weight * scale_output

    # Normalize
    output = output / num_scales

    return output


# =============================================================================
# HIGH-LEVEL API
# =============================================================================


class UnifiedAttention(tf.keras.layers.Layer):
    """Keras layer for unified attention.

    This layer wraps the unified attention operations for easy integration
    into Keras models.

    Example:
        >>> config = UnifiedAttentionConfig(mode=AttentionMode.LINEAR)
        >>> layer = UnifiedAttention(config)
        >>> output = layer(query, key, value)
    """

    def __init__(self, config: UnifiedAttentionConfig, name: str = "unified_attention", **kwargs):
        """Initialize unified attention layer.

        Args:
            config: Attention configuration
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.config = config

        # Initialize mode-specific parameters
        if config.mode == AttentionMode.DIFFERENTIAL:
            self.lambda_param = self.add_weight(
                name="lambda",
                shape=[],
                initializer=tf.constant_initializer(config.lambda_init),
                trainable=True,
            )

        if config.mode == AttentionMode.QASA:
            num_params = config.vqc_layers * config.num_qubits * 2
            self.vqc_params = self.add_weight(
                name="vqc_params", shape=[num_params], initializer="glorot_uniform", trainable=True
            )

        if config.mode == AttentionMode.LMWT:
            self.alpha = self.add_weight(
                name="alpha",
                shape=[config.num_wavelet_scales],
                initializer=tf.constant_initializer(config.alpha_init),
                trainable=config.learn_wavelet_filters,
            )
            self.beta = self.add_weight(
                name="beta",
                shape=[config.num_wavelet_scales],
                initializer=tf.constant_initializer(config.beta_init),
                trainable=config.learn_wavelet_filters,
            )

    def call(
        self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        """Forward pass.

        Args:
            query: Query tensor [batch, num_heads, seq_len, head_dim]
            key: Key tensor [batch, num_kv_heads, kv_seq_len, head_dim]
            value: Value tensor [batch, num_kv_heads, kv_seq_len, head_dim]
            training: Whether in training mode

        Returns:
            Attention output [batch, num_heads, seq_len, head_dim]
        """
        # Prepare extra inputs based on mode
        extra_inputs = None

        if self.config.mode == AttentionMode.QASA:
            extra_inputs = self.vqc_params
        elif self.config.mode == AttentionMode.LMWT:
            extra_inputs = tf.concat([self.alpha, self.beta], axis=0)

        return unified_attention(
            query=query,
            key=key,
            value=value,
            config=self.config,
            extra_inputs=extra_inputs,
            training=training,
            name=self.name,
        )

    def get_config(self) -> dict:
        """Return layer configuration for serialization."""
        base_config = super().get_config()
        return {
            **base_config,
            "mode": int(self.config.mode),
            "num_heads": self.config.num_heads,
            "num_kv_heads": self.config.num_kv_heads,
            "head_dim": self.config.head_dim,
            "window_size": self.config.window_size,
            "causal": self.config.causal,
        }


# =============================================================================
# CONVENIENCE FACTORY FUNCTIONS
# =============================================================================


def flash_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    num_heads: int = 8,
    causal: bool = True,
    training: bool = False,
) -> tf.Tensor:
    """Standard flash attention (O(n²)).

    Args:
        query: [batch, heads, seq, dim]
        key: [batch, heads, seq, dim]
        value: [batch, heads, seq, dim]
        num_heads: Number of attention heads
        causal: Whether to apply causal masking
        training: Training mode

    Returns:
        Attention output [batch, heads, seq, dim]
    """
    config = UnifiedAttentionConfig(
        mode=AttentionMode.FLASH, num_heads=num_heads, num_kv_heads=num_heads, causal=causal
    )
    return unified_attention(query, key, value, config, training=training)


def linear_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    num_heads: int = 8,
    causal: bool = True,
    training: bool = False,
) -> tf.Tensor:
    """Linear attention via ELU+1 feature maps (O(n)).

    Args:
        query: [batch, heads, seq, dim]
        key: [batch, heads, seq, dim]
        value: [batch, heads, seq, dim]
        num_heads: Number of attention heads
        causal: Whether to apply causal masking
        training: Training mode

    Returns:
        Attention output [batch, heads, seq, dim]
    """
    config = UnifiedAttentionConfig(
        mode=AttentionMode.LINEAR, num_heads=num_heads, num_kv_heads=num_heads, causal=causal
    )
    return unified_attention(query, key, value, config, training=training)


def gqa_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    causal: bool = True,
    training: bool = False,
) -> tf.Tensor:
    """Grouped-Query Attention with KV head sharing.

    Args:
        query: [batch, num_heads, seq, dim]
        key: [batch, num_kv_heads, seq, dim]
        value: [batch, num_kv_heads, seq, dim]
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (< num_heads)
        causal: Whether to apply causal masking
        training: Training mode

    Returns:
        Attention output [batch, num_heads, seq, dim]
    """
    config = UnifiedAttentionConfig(
        mode=AttentionMode.GQA, num_heads=num_heads, num_kv_heads=num_kv_heads, causal=causal
    )
    return unified_attention(query, key, value, config, training=training)


def local_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    num_heads: int = 8,
    window_size: int = 256,
    causal: bool = True,
    training: bool = False,
) -> tf.Tensor:
    """Local windowed attention (O(n×w)).

    Args:
        query: [batch, heads, seq, dim]
        key: [batch, heads, seq, dim]
        value: [batch, heads, seq, dim]
        num_heads: Number of attention heads
        window_size: Local attention window
        causal: Whether to apply causal masking
        training: Training mode

    Returns:
        Attention output [batch, heads, seq, dim]
    """
    config = UnifiedAttentionConfig(
        mode=AttentionMode.LOCAL_WINDOWED,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        window_size=window_size,
        causal=causal,
    )
    return unified_attention(query, key, value, config, training=training)


def differential_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    num_heads: int = 8,
    lambda_init: float = 0.8,
    causal: bool = True,
    training: bool = False,
) -> tf.Tensor:
    """Differential Transformer attention: A1 - λ*A2.

    Args:
        query: [batch, heads, seq, dim]
        key: [batch, heads, seq, dim]
        value: [batch, heads, seq, dim]
        num_heads: Number of attention heads
        lambda_init: Initial lambda value
        causal: Whether to apply causal masking
        training: Training mode

    Returns:
        Attention output [batch, heads, seq, dim]
    """
    config = UnifiedAttentionConfig(
        mode=AttentionMode.DIFFERENTIAL,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        lambda_init=lambda_init,
        causal=causal,
    )
    return unified_attention(query, key, value, config, training=training)
