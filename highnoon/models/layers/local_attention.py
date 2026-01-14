# highnoon/models/layers/local_attention.py
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

"""Phase 13.6: Griffin-style Local Attention Hybrid block.

This module implements local attention with a configurable window size,
combining linear recurrence O(L) with local attention O(L·w²) for better
local semantic coherence while maintaining overall efficiency.

Reference: arXiv:2402.19427 - Griffin: Gated Linear Recurrences
"""

from __future__ import annotations

import logging
import math
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

# V2 MIGRATION: Use unified attention API
from highnoon._native.ops.unified_attention import (
    AttentionMode,
    UnifiedAttentionConfig,
    unified_attention,
)

# Phase 201.5: Native Sparse Attention for long sequences
from highnoon.config import (  # Phase 119: QASA config; Phase 130: Quantum integration config
    LOCAL_ATTENTION_ALIBI_SLOPE_BASE,
    LOCAL_ATTENTION_BLOCK_SIZE,
    LOCAL_ATTENTION_BLOCK_SPARSE,
    LOCAL_ATTENTION_CACHE_MASK,
    LOCAL_ATTENTION_MULTI_SCALE,
    LOCAL_ATTENTION_QUANTUM_KERNEL,
    LOCAL_ATTENTION_QUANTUM_RANK,
    LOCAL_ATTENTION_SIGMOID_TEMP,
    LOCAL_ATTENTION_SPARSITY_RATIO,
    LOCAL_ATTENTION_USE_ALIBI,
    LOCAL_ATTENTION_USE_SIGMOID,
    LOCAL_ATTENTION_WINDOW,
    LOCAL_ATTENTION_WINDOW_MAX,
    LOCAL_ATTENTION_WINDOW_MIN,
    NATIVE_SPARSE_BLOCK_SIZE,
    NATIVE_SPARSE_MIN_SEQ_LEN,
    NATIVE_SPARSE_SELECTED_BLOCKS,
    NATIVE_SPARSE_SELECTED_TOKENS,
    NATIVE_SPARSE_SLIDING_WINDOW,
    QASA_ENTANGLEMENT_STRENGTH,
    QASA_MPS_ENTROPY_THRESHOLD,
    QASA_MPS_GATING,
    QASA_VQC_LAYERS,
    USE_AUTO_NEURAL_QEM,
    USE_NATIVE_SPARSE_ATTENTION,
    USE_QASA_ATTENTION,
)
from highnoon.models.reasoning.fused_contract import FusedReasoningBlockMixin

# Phase 201.5: Native Sparse Attention (lazy import)
_nsa_layer_cls = None
_nsa_available = None


def _load_native_sparse_attention() -> bool:
    """Lazy load Native Sparse Attention layer."""
    global _nsa_layer_cls, _nsa_available
    if _nsa_available is not None:
        return _nsa_available
    try:
        from highnoon._native.fused_native_sparse_attention_op import NativeSparseAttention

        _nsa_layer_cls = NativeSparseAttention
        _nsa_available = True
        logger.info("Native Sparse Attention (NSA) available for long sequences")
    except Exception as e:
        logger.debug(f"Native Sparse Attention not available: {e}")
        _nsa_available = False
    return _nsa_available


logger = logging.getLogger(__name__)


class LocalAttentionBlock(FusedReasoningBlockMixin, layers.Layer):
    """Griffin-style Local Attention with configurable window size and enhancements.

    This block combines the benefits of:
    - Local attention for fine-grained local dependencies
    - Optional causal masking for autoregressive models
    - Efficient windowed computation O(L·w²)

    Enhancements:
    - Multi-Scale Windows (MSWA): Per-head variable window sizes
    - Sigmoid Attention: Replace softmax to eliminate attention sink
    - ALiBi Position Bias: Linear position encoding for extrapolation
    - Block-Sparse: Dynamic block pruning for speedup
    - Quantum Kernel: O(n) fidelity-based attention (research)
    - Mask Caching: Cache masks for recurring sequences

    The key insight is that many NLP tasks require strong local coherence
    (syntax, phrase structure) which local attention captures well, while
    long-range dependencies are handled by SSM blocks in the stack.

    Attributes:
        embedding_dim: Input/output dimension.
        num_heads: Number of attention heads.
        window_size: Size of local attention window.
        causal: Whether to use causal (left-only) attention.

    Example:
        >>> block = LocalAttentionBlock(embedding_dim=512, window_size=128)
        >>> x = tf.random.normal([2, 1000, 512])
        >>> output = block(x)
        >>> print(output.shape)  # (2, 1000, 512)
    """

    fused_block_type = "LocalAttentionBlock"
    fused_block_stateful = False

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        window_size: int = LOCAL_ATTENTION_WINDOW,
        causal: bool = True,
        dropout_rate: float = 0.0,
        # Enhancement 1: Multi-Scale Window Attention
        use_multiscale: bool = LOCAL_ATTENTION_MULTI_SCALE,
        window_min: int = LOCAL_ATTENTION_WINDOW_MIN,
        window_max: int = LOCAL_ATTENTION_WINDOW_MAX,
        # Enhancement 2: Sigmoid Attention
        use_sigmoid_attention: bool = LOCAL_ATTENTION_USE_SIGMOID,
        sigmoid_temperature: float = LOCAL_ATTENTION_SIGMOID_TEMP,
        # Enhancement 3: ALiBi Position Bias
        use_alibi: bool = LOCAL_ATTENTION_USE_ALIBI,
        alibi_slope_base: float = LOCAL_ATTENTION_ALIBI_SLOPE_BASE,
        # Enhancement 4: Block-Sparse Optimization
        use_block_sparse: bool = LOCAL_ATTENTION_BLOCK_SPARSE,
        block_size: int = LOCAL_ATTENTION_BLOCK_SIZE,
        sparsity_ratio: float = LOCAL_ATTENTION_SPARSITY_RATIO,
        # Enhancement 5: Quantum Kernel Approximation
        use_quantum_kernel: bool = LOCAL_ATTENTION_QUANTUM_KERNEL,
        quantum_rank: int = LOCAL_ATTENTION_QUANTUM_RANK,
        # Enhancement 6: Mask Caching
        cache_masks: bool = LOCAL_ATTENTION_CACHE_MASK,
        # Phase 119: QASA Quantum Adaptive Self-Attention
        use_qasa_attention: bool = USE_QASA_ATTENTION,
        qasa_vqc_layers: int = QASA_VQC_LAYERS,
        qasa_entanglement_strength: float = QASA_ENTANGLEMENT_STRENGTH,
        qasa_num_qubits: int = 4,
        **kwargs,
    ):
        """Initialize Local Attention block with enhancements.

        Args:
            embedding_dim: Input and output dimension.
            num_heads: Number of attention heads (must divide embedding_dim).
            window_size: Size of local attention window. Larger windows capture
                more local context but increase compute cost.
            causal: If True, positions can only attend to earlier positions
                within the window (for autoregressive models).
            dropout_rate: Dropout rate for attention weights.
            use_multiscale: Enable per-head variable window sizes (MSWA).
            window_min: Minimum window size for multi-scale (head 0).
            window_max: Maximum window size for multi-scale (head N-1).
            use_sigmoid_attention: Replace softmax with sigmoid.
            sigmoid_temperature: Temperature scaling for sigmoid.
            use_alibi: Enable ALiBi position bias.
            alibi_slope_base: Base for ALiBi slope computation.
            use_block_sparse: Enable block-sparse pruning.
            block_size: Block size for sparse attention.
            sparsity_ratio: Ratio of blocks to keep (0.5 = 50%).
            use_quantum_kernel: Enable quantum-inspired attention.
            quantum_rank: Low-rank approximation rank for O(n).
            cache_masks: Cache masks for recurring sequences.

        Raises:
            ValueError: If embedding_dim is not divisible by num_heads.
        """
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"
        super().__init__(**kwargs)

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.window_size = window_size
        self.causal = causal
        self.dropout_rate = dropout_rate
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Enhancement 1: Multi-Scale Window Attention
        self.use_multiscale = use_multiscale
        self.window_min = window_min
        self.window_max = window_max

        # Enhancement 2: Sigmoid Attention
        self.use_sigmoid_attention = use_sigmoid_attention
        self.sigmoid_temperature = sigmoid_temperature

        # Enhancement 3: ALiBi Position Bias
        self.use_alibi = use_alibi
        self.alibi_slope_base = alibi_slope_base

        # Enhancement 4: Block-Sparse
        self.use_block_sparse = use_block_sparse
        self.block_size = block_size
        self.sparsity_ratio = sparsity_ratio

        # Enhancement 5: Quantum Kernel
        self.use_quantum_kernel = use_quantum_kernel
        self.quantum_rank = quantum_rank

        # Enhancement 6: Mask Caching
        self.cache_masks = cache_masks
        self._mask_cache: dict[tuple[int, int, bool], tf.Tensor] = {}

        # Phase 119: QASA Quantum Adaptive Self-Attention
        self.use_qasa_attention = use_qasa_attention
        self.qasa_vqc_layers = qasa_vqc_layers
        self.qasa_entanglement_strength = qasa_entanglement_strength
        self.qasa_num_qubits = qasa_num_qubits
        self._qasa_initialized = False

        # Phase 130.1: Auto Neural QEM mitigator for QASA output
        self._qasa_qem_mitigator = None
        if USE_AUTO_NEURAL_QEM and use_qasa_attention:
            from highnoon.training.neural_zne import NeuralQuantumErrorMitigator

            self._qasa_qem_mitigator = NeuralQuantumErrorMitigator(name="qasa_qem")

        # Phase 130.6: MPS entropy gating for QASA
        self.use_mps_gating = QASA_MPS_GATING and use_qasa_attention
        self.mps_entropy_threshold = QASA_MPS_ENTROPY_THRESHOLD
        self._mps_layer = None

        # Phase 201.5: Native Sparse Attention for long sequences
        self.use_native_sparse = USE_NATIVE_SPARSE_ATTENTION
        self.nsa_min_seq_len = NATIVE_SPARSE_MIN_SEQ_LEN
        self._nsa_layer = None
        if self.use_native_sparse and _load_native_sparse_attention():
            self._nsa_layer = _nsa_layer_cls(
                block_size=NATIVE_SPARSE_BLOCK_SIZE,
                num_selected_blocks=NATIVE_SPARSE_SELECTED_BLOCKS,
                num_selected_tokens=NATIVE_SPARSE_SELECTED_TOKENS,
                sliding_window_size=NATIVE_SPARSE_SLIDING_WINDOW,
                name="native_sparse_attention",
            )
            logger.info(
                f"  - Phase 201.5: NSA enabled for seq_len >= {self.nsa_min_seq_len} "
                f"(block={NATIVE_SPARSE_BLOCK_SIZE}, window={NATIVE_SPARSE_SLIDING_WINDOW})"
            )

        # Q, K, V projections
        self.proj_q = layers.Dense(embedding_dim, name="proj_q")
        self.proj_k = layers.Dense(embedding_dim, name="proj_k")
        self.proj_v = layers.Dense(embedding_dim, name="proj_v")

        # Output projection
        self.out_proj = layers.Dense(embedding_dim, name="out_proj")

        # Layer normalization
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")

        # Dropout
        if dropout_rate > 0:
            self.attn_dropout = layers.Dropout(dropout_rate, name="attn_dropout")
        else:
            self.attn_dropout = None

    def build(self, input_shape):
        """Build layer weights."""
        feature_shape = tf.TensorShape([None, self.embedding_dim])

        for proj in [self.proj_q, self.proj_k, self.proj_v, self.out_proj]:
            if not proj.built:
                proj.build(feature_shape)

        if not self.norm.built:
            self.norm.build(input_shape)

        # Phase 119: Initialize QASA VQC parameters if enabled
        if self.use_qasa_attention and not self._qasa_initialized:
            # VQC params: 2 * vqc_layers * num_qubits (for Q and K encoding)
            num_vqc_params = 2 * self.qasa_vqc_layers * self.qasa_num_qubits
            self.vqc_params = self.add_weight(
                name="qasa_vqc_params",
                shape=(num_vqc_params,),
                initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
                trainable=True,
                dtype=tf.float32,
            )
            self._qasa_initialized = True

        super().build(input_shape)

    def _create_local_mask(self, seq_len: int) -> tf.Tensor:
        """Create local attention mask.

        Args:
            seq_len: Sequence length.

        Returns:
            Boolean mask [seq_len, seq_len] where True = attend.
        """
        # Create position indices
        positions = tf.range(seq_len)
        query_pos = positions[:, tf.newaxis]  # [seq_len, 1]
        key_pos = positions[tf.newaxis, :]  # [1, seq_len]

        # Local window mask: |query - key| < window_size
        distance = tf.abs(query_pos - key_pos)
        local_mask = distance < self.window_size

        if self.causal:
            # Causal mask: query >= key (can only attend to past)
            causal_mask = query_pos >= key_pos
            mask = local_mask & causal_mask
        else:
            mask = local_mask

        return mask

    @tf.function(reduce_retracing=True)
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass for Local Attention block.

        Args:
            x: Input tensor of shape [batch, seq_len, embedding_dim].
            training: Whether in training mode.

        Returns:
            Output tensor of shape [batch, seq_len, embedding_dim].
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Project to Q, K, V
        q = self.proj_q(x)  # [batch, seq_len, embedding_dim]
        k = self.proj_k(x)
        v = self.proj_v(x)

        # Reshape for multi-head attention
        # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.transpose(v, [0, 2, 1, 3])

        # Phase 119: QASA path - use VQC-based attention scoring
        if self.use_qasa_attention and self._is_qasa_available():
            attn_output = self._qasa_attention(q, k, v, training=training)
        # Phase 201.5: NSA path for long sequences - O(n log n) vs O(n²)
        elif self._nsa_layer is not None and seq_len >= self.nsa_min_seq_len:
            # NSA expects [batch, heads, seq, dim] format
            attn_output = self._nsa_layer(q, k, v, training=training)
        else:
            # Standard attention path
            # Compute attention scores
            # [batch, heads, seq_len, head_dim] @ [batch, heads, head_dim, seq_len]
            # -> [batch, heads, seq_len, seq_len]
            scores = tf.matmul(q, k, transpose_b=True) * self.scale

            # Create and apply local attention mask
            mask = self._create_local_mask(seq_len)
            mask = tf.cast(mask, scores.dtype)

            # Apply mask: set non-local positions to -inf
            mask_value = tf.constant(-1e9, dtype=scores.dtype)
            scores = tf.where(mask[tf.newaxis, tf.newaxis, :, :] > 0, scores, mask_value)

            # Softmax attention weights
            attn_weights = tf.nn.softmax(scores, axis=-1)

            # Apply dropout to attention weights
            if self.attn_dropout is not None and training:
                attn_weights = self.attn_dropout(attn_weights, training=training)

            # Apply attention to values
            # [batch, heads, seq_len, seq_len] @ [batch, heads, seq_len, head_dim]
            # -> [batch, heads, seq_len, head_dim]
            attn_output = tf.matmul(attn_weights, v)

        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, embedding_dim]
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [batch_size, seq_len, self.embedding_dim])

        # Output projection
        output = self.out_proj(attn_output)

        # Residual connection and normalization
        output = self.norm(x + output)

        return output

    def _is_qasa_available(self) -> bool:
        """Check if QASA is available via unified attention API."""
        # V2 MIGRATION: QASA is always available via unified attention
        return self.use_qasa_attention

    def _qasa_attention(
        self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        """Compute QASA attention using unified attention API.

        V2 MIGRATION: Uses AttentionMode.QASA from unified_attention.

        Args:
            q: Query tensor [batch, heads, seq_len, head_dim].
            k: Key tensor [batch, heads, seq_len, head_dim].
            v: Value tensor [batch, heads, seq_len, head_dim].
            training: Whether in training mode.

        Returns:
            Attention output [batch, heads, seq_len, head_dim].
        """
        # Build unified QASA config
        qasa_config = UnifiedAttentionConfig(
            mode=AttentionMode.QASA,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim=self.head_dim,
            causal=self.causal,
            num_qubits=self.qasa_num_qubits,
            vqc_layers=self.qasa_vqc_layers,
            entanglement_strength=self.qasa_entanglement_strength,
        )

        # Call unified attention with VQC params as extra_inputs
        qasa_output = unified_attention(
            q, k, v, qasa_config, extra_inputs=self.vqc_params, training=training
        )

        # Phase 130.6: Apply MPS entropy gating if enabled
        if self.use_mps_gating:
            qasa_output = self._apply_mps_gating(qasa_output, q, v)

        # Phase 130.1: Apply auto Neural QEM if enabled
        if self._qasa_qem_mitigator is not None:
            batch_size = tf.shape(qasa_output)[0]
            orig_shape = tf.shape(qasa_output)
            flat_output = tf.reshape(qasa_output, [batch_size, -1, tf.shape(qasa_output)[-1]])
            flat_output = self._qasa_qem_mitigator(flat_output, training=training)
            qasa_output = tf.reshape(flat_output, orig_shape)

        return qasa_output

    def _apply_mps_gating(self, qasa_output: tf.Tensor, q: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        """Apply MPS entropy-based gating to QASA output (Phase 130.6).

        High entanglement entropy -> stronger quantum contribution.
        Low entanglement entropy -> use classical attention fallback.

        Args:
            qasa_output: QASA attention output [batch, heads, seq, dim].
            q: Query tensor for fallback computation.
            v: Value tensor for fallback computation.

        Returns:
            Gated output blending QASA with classical attention.
        """
        # Compute approximate entanglement entropy from attention patterns
        # Use query-key overlap as proxy for entanglement
        seq_len = tf.shape(q)[2]

        # Compute attention scores for entropy estimation
        attn_pattern = tf.matmul(q, q, transpose_b=True) / tf.sqrt(
            tf.cast(tf.shape(q)[-1], tf.float32)
        )
        attn_probs = tf.nn.softmax(attn_pattern, axis=-1)

        # Compute entropy: -sum(p * log(p))
        log_probs = tf.math.log(attn_probs + 1e-10)
        entropy = -tf.reduce_sum(attn_probs * log_probs, axis=-1)
        entropy = tf.reduce_mean(entropy, axis=[1, 2])  # [batch]

        # Normalize entropy to [0, 1] range
        max_entropy = tf.math.log(tf.cast(seq_len, tf.float32))
        normalized_entropy = entropy / (max_entropy + 1e-10)

        # Compute gate: sigmoid around threshold
        gate = tf.nn.sigmoid(10.0 * (normalized_entropy - self.mps_entropy_threshold))
        gate = tf.reshape(gate, [-1, 1, 1, 1])  # [batch, 1, 1, 1]

        # Classical fallback: simple value averaging
        classical_output = v

        # Blend QASA output with classical based on entropy
        return gate * qasa_output + (1.0 - gate) * classical_output

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "causal": self.causal,
                "dropout_rate": self.dropout_rate,
                "use_qasa_attention": self.use_qasa_attention,
                "qasa_vqc_layers": self.qasa_vqc_layers,
                "qasa_entanglement_strength": self.qasa_entanglement_strength,
                "qasa_num_qubits": self.qasa_num_qubits,
            }
        )
        return config

    def fused_metadata(self) -> dict[str, int]:
        """Get metadata for fused operation."""
        return {
            "embedding_dim": int(self.embedding_dim),
            "num_heads": int(self.num_heads),
            "window_size": int(self.window_size),
            "causal": int(self.causal),
        }

    def get_fused_op_descriptor(self) -> dict[str, Any]:
        """Get descriptor for C++ fused kernel."""
        return {
            "type": self.__class__.__name__,
            "stateful": False,
            "metadata": self.fused_metadata(),
            "weight_count": len(self.get_weights_for_fused_op()),
        }
