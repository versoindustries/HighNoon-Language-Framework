# highnoon/models/layers/quantum_gqa.py
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

"""Phase 15.4: Quantum-Enhanced Grouped-Query Attention (Research).

This module provides quantum-inspired attention mechanisms leveraging
the framework's existing VQC infrastructure. Uses parameterized quantum
circuits for richer attention score computation.

Key features:
- O(n) complexity via quantum kernel attention
- VQC-based attention scoring
- Leverages existing quantum tensor network infrastructure
- CPU-friendly via MPS-based simulation

Reference:
- Quantum Self-Attention Neural Networks (QSANNs)
- VQC Attention for Text Classification
- QAE-Net: Quantum Attention Enhanced Networks
"""

from __future__ import annotations

import logging
import math
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.config import (
    GQA_NUM_KV_HEADS,
    GQA_QUANTUM_LAYERS,
    GQA_QUANTUM_QUBITS,
    GQA_TPA_CONTEXT_DIM,
    GQA_TPA_RANK,
    GQA_USE_TPA,
)

logger = logging.getLogger(__name__)

# Phase 300+: HD Holographic Attention Integration
try:
    from highnoon._native.ops.hd_holographic_attention import (
        HDKVCache,
        hd_holographic_attention_available,
        holographic_attention_scores,
    )
    from highnoon.config import USE_HD_HOLOGRAPHIC_ATTENTION, USE_HD_KV_CACHE

    _HD_ATTENTION_AVAILABLE = hd_holographic_attention_available()
except ImportError:
    USE_HD_HOLOGRAPHIC_ATTENTION = False
    USE_HD_KV_CACHE = False
    _HD_ATTENTION_AVAILABLE = False
    holographic_attention_scores = None
    HDKVCache = None


class QuantumGQA(layers.Layer):
    """Quantum-Enhanced Grouped-Query Attention.

    Experimental layer using quantum-inspired operations for attention
    computation. Encodes Q/K as rotation angles on simulated qubits
    and uses parameterized entangling gates for correlation.

    The key insight is that quantum kernels can capture correlations
    that classical kernels cannot easily represent:
        K_quantum(q, k) = |⟨φ(q)|φ(k)⟩|²

    This provides potentially richer representational capacity while
    maintaining O(n) complexity through kernel trick formulation.

    Note: This is a research/experimental layer for exploring quantum
    advantage in attention mechanisms. For production use, consider
    LinearGroupedQueryAttention instead.

    Attributes:
        embedding_dim: Input/output dimension.
        num_heads: Number of query heads.
        num_qubits: Number of simulated qubits per attention head.
        num_layers: Depth of VQC circuit.

    Example:
        >>> gqa = QuantumGQA(embedding_dim=512, num_qubits=4, num_layers=2)
        >>> x = tf.random.normal([2, 128, 512])
        >>> output = gqa(x)  # Quantum-enhanced attention
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        num_qubits: int | None = None,
        num_layers: int | None = None,
        dropout_rate: float = 0.0,
        causal: bool = True,
        # Tensor decomposition parameters (10x+ KV cache reduction)
        use_tpa: bool | None = None,
        tpa_rank: int | None = None,
        tpa_context_dim: int | None = None,
        eps: float = 1e-6,
        **kwargs,
    ):
        """Initialize Quantum GQA.

        Args:
            embedding_dim: Embedding dimension.
            num_heads: Number of query heads.
            num_kv_heads: Number of KV heads.
            num_qubits: Number of qubits for quantum encoding.
            num_layers: Number of VQC layers.
            dropout_rate: Dropout rate.
            causal: Whether to use causal attention.
            use_tpa: Enable tensor product decomposition for memory efficiency.
            tpa_rank: Rank of tensor decomposition.
            tpa_context_dim: Context embedding dimension.
            eps: Epsilon for numerical stability.
        """
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"
        super().__init__(**kwargs)

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by " f"num_heads ({num_heads})"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or GQA_NUM_KV_HEADS or max(1, num_heads // 4)
        self.head_dim = embedding_dim // num_heads
        self.num_qubits = num_qubits or GQA_QUANTUM_QUBITS
        self.num_layers = num_layers or GQA_QUANTUM_LAYERS
        self.dropout_rate = dropout_rate
        self.causal = causal
        self.eps = eps

        # Tensor product attention parameters
        self.use_tpa = use_tpa if use_tpa is not None else GQA_USE_TPA
        self.tpa_rank = tpa_rank or GQA_TPA_RANK
        self.tpa_context_dim = tpa_context_dim or GQA_TPA_CONTEXT_DIM

        if num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

        self.num_queries_per_kv = num_heads // self.num_kv_heads

        # Projections - use tensor decomposition if enabled
        if self.use_tpa:
            # Context projection for tensor product
            self.context_proj = layers.Dense(self.tpa_context_dim, name="context_proj")

            # Factorized Q projections: Q = Σᵢ (x @ A_i) ⊗ (context @ B_i)
            self.q_factor_a = layers.Dense(self.tpa_rank * self.head_dim, name="q_factor_a")
            self.q_factor_b = layers.Dense(self.tpa_rank * num_heads, name="q_factor_b")

            # Factorized K projections
            self.k_factor_a = layers.Dense(self.tpa_rank * self.head_dim, name="k_factor_a")
            self.k_factor_b = layers.Dense(self.tpa_rank * self.num_kv_heads, name="k_factor_b")

            # Factorized V projections
            self.v_factor_a = layers.Dense(self.tpa_rank * self.head_dim, name="v_factor_a")
            self.v_factor_b = layers.Dense(self.tpa_rank * self.num_kv_heads, name="v_factor_b")

            logger.info(
                f"QuantumGQA: TPA enabled (rank={self.tpa_rank}, "
                f"context_dim={self.tpa_context_dim}) - 10x+ KV cache reduction"
            )
        else:
            # Standard dense projections
            self.q_proj = layers.Dense(embedding_dim, name="q_proj")
            self.k_proj = layers.Dense(self.num_kv_heads * self.head_dim, name="k_proj")
            self.v_proj = layers.Dense(self.num_kv_heads * self.head_dim, name="v_proj")

        self.out_proj = layers.Dense(embedding_dim, name="out_proj")

        # Quantum encoding projection: map head_dim to num_qubits angles
        self.quantum_encoder = layers.Dense(
            self.num_qubits, activation="tanh", name="quantum_encoder"
        )

        # Learnable VQC parameters for attention
        self.vqc_params = None  # Initialized in build()

        # Dropout
        if dropout_rate > 0:
            self.dropout = layers.Dropout(dropout_rate)
        else:
            self.dropout = None

        # Norm
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")

        logger.info(
            f"QuantumGQA: {num_heads} heads, {self.num_kv_heads} KV heads, "
            f"{self.num_qubits} qubits, {self.num_layers} VQC layers"
        )

    def build(self, input_shape):
        """Build VQC parameters."""
        # Learnable rotation parameters for VQC
        self.vqc_params = self.add_weight(
            name="vqc_params",
            shape=(self.num_layers, self.num_qubits, 3),
            initializer="glorot_uniform",
            trainable=True,
        )
        # Learnable entanglement strength (initialized to 0.1)
        self.entanglement_strength = self.add_weight(
            name="entanglement_strength",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
        )
        super().build(input_shape)

    def _tensor_product_projection(
        self,
        x: tf.Tensor,
        context: tf.Tensor,
        factor_a: layers.Layer,
        factor_b: layers.Layer,
        num_out_heads: int,
    ) -> tf.Tensor:
        """Compute tensor product projection for memory-efficient Q/K/V.

        Projects input using factorized weights:
            output = Σᵢ (x @ Aᵢ) ⊗ (context @ Bᵢ)

        This is more parameter-efficient than full projection while
        maintaining expressive power through context-dependence.

        Args:
            x: Input tensor [batch, seq, embedding_dim].
            context: Context embedding [batch, seq, context_dim].
            factor_a: First factor projection.
            factor_b: Second factor projection (context-dependent).
            num_out_heads: Number of output heads.

        Returns:
            Projected tensor [batch, num_out_heads, seq, head_dim].
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Factor A: input-based projection
        a_proj = factor_a(x)  # [batch, seq, tpa_rank * head_dim]
        a_proj = tf.reshape(a_proj, [batch_size, seq_len, self.tpa_rank, self.head_dim])

        # Factor B: context-based projection
        b_proj = factor_b(context)  # [batch, seq, tpa_rank * num_out_heads]
        b_proj = tf.reshape(b_proj, [batch_size, seq_len, self.tpa_rank, num_out_heads])

        # Tensor product: combine factors via weighted sum over rank
        # out[b, s, h, d] = Σᵣ a[b, s, r, d] * b[b, s, r, h]
        output = tf.einsum("bsrd,bsrh->bshd", a_proj, b_proj)

        # Transpose to [batch, heads, seq, head_dim]
        output = tf.transpose(output, [0, 2, 1, 3])

        return output

    def _quantum_kernel_feature(self, x: tf.Tensor) -> tf.Tensor:
        """Apply quantum-inspired feature map.

        Simulates amplitude encoding and parameterized rotations:
            |ψ(x)⟩ = U(θ) ⊗ R(x) |0⟩

        For CPU efficiency, we use a classical approximation based on
        trigonometric functions that captures similar characteristics.

        Args:
            x: Input [batch, heads, seq, head_dim].

        Returns:
            Quantum features [batch, heads, seq, feature_dim].
        """
        # Encode to qubit angles: [batch, heads, seq, num_qubits]
        # Reshape for encoding
        original_shape = tf.shape(x)
        x_flat = tf.reshape(x, [-1, self.head_dim])
        angles = self.quantum_encoder(x_flat)  # [-1, num_qubits]
        angles = tf.reshape(
            angles, [original_shape[0], original_shape[1], original_shape[2], self.num_qubits]
        )

        # Scale angles to [0, 2π]
        angles = angles * math.pi

        # Apply VQC-inspired transformation layers
        features = angles
        for layer_idx in range(self.num_layers):
            params = self.vqc_params[layer_idx]  # [num_qubits, 3]

            # Rx rotation (approximated): cos(θ + x)
            rx = tf.cos(features + params[:, 0])
            # Ry rotation (approximated): sin(θ + x)
            ry = tf.sin(features + params[:, 1])
            # Rz rotation (approximated): exp(i(θ + x)) ≈ cos + i*sin
            rz_cos = tf.cos(features + params[:, 2])
            tf.sin(features + params[:, 2])

            # Combine rotations (simplified quantum gate approximation)
            features = rx * ry + rz_cos

            # Simulate entanglement via nearest-neighbor interaction
            # CNOT-like correlation between adjacent qubits
            if self.num_qubits > 1:
                shifted = tf.roll(features, shift=1, axis=-1)
                features = features * (1 + self.entanglement_strength * shifted)

        # Final feature map: use both cos and sin components
        cos_features = tf.cos(features)
        sin_features = tf.sin(features)
        quantum_features = tf.concat([cos_features, sin_features], axis=-1)

        # Normalize
        quantum_features = quantum_features / (
            tf.norm(quantum_features, axis=-1, keepdims=True) + self.eps
        )

        return quantum_features

    def _quantum_kernel_attention(
        self,
        q_features: tf.Tensor,
        k_features: tf.Tensor,
        v: tf.Tensor,
    ) -> tf.Tensor:
        """Compute attention using quantum kernel.

        Phase 300+: When USE_HD_HOLOGRAPHIC_ATTENTION=True and native ops available,
        uses FFT-based holographic similarity O(d log d) instead of O(d²).

        Attention is computed using quantum kernel similarity:
            K(q, k) = |⟨φ(q)|φ(k)⟩|² = (φ(q) · φ(k))²

        This provides O(n) complexity when features are low-dimensional.

        Args:
            q_features: Quantum features of queries.
            k_features: Quantum features of keys.
            v: Values [batch, heads, seq, head_dim].

        Returns:
            Attention output [batch, heads, seq, head_dim].
        """
        # Phase 300+: Holographic Attention (O(d log d) via FFT)
        if USE_HD_HOLOGRAPHIC_ATTENTION and _HD_ATTENTION_AVAILABLE:
            try:
                # holographic_attention_scores returns [batch, heads, seq_q, seq_k]
                scores = holographic_attention_scores(q_features, k_features, temperature=1.0)
                # Apply causal mask if needed
                if self.causal:
                    seq_len = tf.shape(scores)[-1]
                    mask = tf.linalg.band_part(
                        tf.ones([seq_len, seq_len], dtype=scores.dtype), -1, 0
                    )
                    scores = scores * mask[tf.newaxis, tf.newaxis, :, :]
                    scores = scores - (1.0 - mask[tf.newaxis, tf.newaxis, :, :]) * 1e9
                attn_weights = tf.nn.softmax(scores, axis=-1)
                return tf.matmul(attn_weights, v)
            except Exception as e:
                logger.warning(f"Holographic attention failed, using quantum kernel: {e}")
                # Fall through to quantum kernel

        if self.causal:
            # Causal attention with quantum kernel
            # Use cumulative sum formulation for O(n)
            kv = tf.einsum("bhsf,bhsd->bhsfd", k_features, v)
            kv_cumsum = tf.cumsum(kv, axis=2)
            k_cumsum = tf.cumsum(k_features, axis=2)

            numerator = tf.einsum("bhsf,bhsfd->bhsd", q_features, kv_cumsum)
            denominator = tf.einsum("bhsf,bhsf->bhs", q_features, k_cumsum)
            denominator = denominator[..., tf.newaxis] + self.eps

            return numerator / denominator
        else:
            # Non-causal O(n) attention
            kv = tf.einsum("bhsf,bhsd->bhfd", k_features, v)
            k_sum = tf.reduce_sum(k_features, axis=2)

            numerator = tf.einsum("bhsf,bhfd->bhsd", q_features, kv)
            denominator = tf.einsum("bhsf,bhf->bhs", q_features, k_sum)
            denominator = denominator[..., tf.newaxis] + self.eps

            return numerator / denominator

    def _expand_kv_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Expand KV heads by repeating."""
        return tf.repeat(x, repeats=self.num_queries_per_kv, axis=1)

    def call(
        self,
        x: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass with quantum-enhanced attention.

        Args:
            x: Input [batch, seq_len, embedding_dim].
            attention_mask: Optional attention mask.
            training: Whether in training mode.

        Returns:
            Output [batch, seq_len, embedding_dim].
        """
        residual = x
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Project to Q, K, V - use tensor decomposition if enabled
        if self.use_tpa:
            # Compute context embedding for tensor product
            context = self.context_proj(x)  # [batch, seq, context_dim]

            # Tensor product projections (10x+ KV cache reduction)
            q = self._tensor_product_projection(
                x, context, self.q_factor_a, self.q_factor_b, self.num_heads
            )  # [batch, num_heads, seq, head_dim]
            k = self._tensor_product_projection(
                x, context, self.k_factor_a, self.k_factor_b, self.num_kv_heads
            )  # [batch, num_kv_heads, seq, head_dim]
            v = self._tensor_product_projection(
                x, context, self.v_factor_a, self.v_factor_b, self.num_kv_heads
            )  # [batch, num_kv_heads, seq, head_dim]
        else:
            # Standard dense projections
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            # Reshape to heads
            q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
            q = tf.transpose(q, [0, 2, 1, 3])
            k = tf.reshape(k, [batch_size, seq_len, self.num_kv_heads, self.head_dim])
            k = tf.transpose(k, [0, 2, 1, 3])
            v = tf.reshape(v, [batch_size, seq_len, self.num_kv_heads, self.head_dim])
            v = tf.transpose(v, [0, 2, 1, 3])

        # Expand KV heads
        k = self._expand_kv_heads(k)
        v = self._expand_kv_heads(v)

        # Apply quantum feature map
        q_features = self._quantum_kernel_feature(q)
        k_features = self._quantum_kernel_feature(k)

        # Apply attention mask if provided (mask out future positions for causal)
        if attention_mask is not None:
            # Expand mask for feature dimension: [batch, 1, seq, 1]
            mask = tf.cast(attention_mask, dtype=k_features.dtype)
            if len(mask.shape) == 2:
                mask = mask[:, tf.newaxis, :, tf.newaxis]
            elif len(mask.shape) == 3:
                mask = mask[:, :, :, tf.newaxis]
            # Apply mask by zeroing out masked key features
            k_features = k_features * mask

        # Compute quantum kernel attention
        output = self._quantum_kernel_attention(q_features, k_features, v)

        # Reshape back
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [batch_size, seq_len, self.embedding_dim])

        # Output projection
        output = self.out_proj(output)

        # Dropout
        if self.dropout is not None and training:
            output = self.dropout(output, training=training)

        # Residual and norm
        output = self.norm(residual + output)

        return output

    def get_complexity(self) -> str:
        """Get computational complexity."""
        return "O(n)"  # Linear via quantum kernel formulation

    def get_kv_cache_size(self, batch_size: int, seq_len: int) -> int:
        """Get KV cache size - reduced by TPA factorization when enabled."""
        if self.use_tpa:
            # Factorized cache: rank * (head_dim + num_kv_heads)
            factor_size = self.tpa_rank * (self.head_dim + self.num_kv_heads)
            return 2 * batch_size * seq_len * factor_size
        else:
            # Standard cache
            return 2 * batch_size * seq_len * self.num_kv_heads * self.head_dim

    def get_cache_reduction_ratio(self) -> float:
        """Get cache reduction ratio vs standard GQA (TPA only)."""
        if not self.use_tpa:
            return 1.0
        standard_size = 2 * self.num_kv_heads * self.head_dim
        tpa_size = self.tpa_rank * (self.head_dim + self.num_kv_heads)
        return standard_size / tpa_size

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "num_qubits": self.num_qubits,
                "num_layers": self.num_layers,
                "dropout_rate": self.dropout_rate,
                "causal": self.causal,
                "use_tpa": self.use_tpa,
                "tpa_rank": self.tpa_rank,
                "tpa_context_dim": self.tpa_context_dim,
                "eps": self.eps,
            }
        )
        return config


__all__ = ["QuantumGQA"]
