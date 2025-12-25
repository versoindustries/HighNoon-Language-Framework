# highnoon/models/layers/wlam.py
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

"""Wavelet-Enhanced Linear Attention Mechanism (WLAM) Layer.

Enhanced features (per WLAM roadmap):
- Multi-level hierarchical DWT decomposition (1-5 levels)
- Lifting scheme with learnable predict/update wavelets
- Frequency-adaptive processing gating
- Wavelet scattering transform for translation-invariant features
- Cross-frequency linear attention between frequency bands

NOTE: All computation is delegated to the C++ fused_wlam kernel.
Python layers exist only to provide trainable weights.
"""

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.models.reasoning.fused_contract import FusedReasoningBlockMixin


class WLAMBlock(FusedReasoningBlockMixin, layers.Layer):
    """Wavelet-Enhanced Linear Attention Mechanism.

    This block implements a multi-resolution analysis of the input sequence
    using a learnable 1D Discrete Wavelet Transform (DWT).

    Enhanced features:
    - Multi-level hierarchical decomposition (1-5 DWT levels)
    - Frequency-adaptive processing with learned gating
    - Wavelet scattering for translation-invariant features
    - Cross-frequency attention between bands

    NOTE: All computation is handled by the C++ FusedWLAM kernel.
    Python layers (Conv1D, Dense) exist only to provide trainable kernels.
    Biases are disabled since the C++ op only uses kernel weights.
    """

    fused_block_type = "WLAMBlock"
    fused_block_stateful = False

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        wavelet_kernel_size: int = 4,
        # Enhanced features
        num_levels: int = 1,
        use_lifting: bool = False,  # Lifting scheme implemented in C++ fused op
        use_adaptive: bool = False,
        scattering_layers: int = 0,
        scattering_pool: int = 4,
        use_cross_attn: bool = False,
        **kwargs,
    ):
        """Initialize WLAMBlock.

        Args:
            embedding_dim: Dimension of token embeddings.
            num_heads: Number of attention heads for low-freq processing.
            wavelet_kernel_size: Size of wavelet filter kernels.
            num_levels: Number of DWT decomposition levels (1-5).
            use_lifting: Use lifting scheme DWT (C++ fused op only).
            use_adaptive: Enable frequency-adaptive processing gating.
            scattering_layers: Number of scattering layers (0=disabled).
            scattering_pool: Scattering average pooling size.
            use_cross_attn: Enable cross-frequency attention.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.wavelet_kernel_size = wavelet_kernel_size

        # Enhanced features
        self.num_levels = min(max(num_levels, 1), 5)  # Clamp to 1-5
        self.use_lifting = use_lifting  # Passed to C++ fused op
        self.use_adaptive = use_adaptive
        self.scattering_layers = scattering_layers
        self.scattering_pool = scattering_pool
        self.use_cross_attn = use_cross_attn

        # ===== Weight-Only Layers (use_bias=False) =====
        # The C++ fused_wlam op only uses kernel weights, not biases.
        # These layers exist solely to provide trainable kernels.

        # 1. DWT Analysis Filters (low-pass and high-pass)
        self.h_filter = layers.Conv1D(
            embedding_dim,
            wavelet_kernel_size,
            padding="same",
            use_bias=False,
            name="dwt_h_filter",
        )
        self.g_filter = layers.Conv1D(
            embedding_dim,
            wavelet_kernel_size,
            padding="same",
            use_bias=False,
            name="dwt_g_filter",
        )

        # 2. IWT Synthesis Filters (low-pass and high-pass)
        self.h_synth_filter = layers.Conv1D(
            embedding_dim,
            wavelet_kernel_size,
            padding="same",
            use_bias=False,
            name="iwt_h_filter",
        )
        self.g_synth_filter = layers.Conv1D(
            embedding_dim,
            wavelet_kernel_size,
            padding="same",
            use_bias=False,
            name="iwt_g_filter",
        )

        # 3. LayerNorm (gamma/beta used by C++)
        self.norm = layers.LayerNormalization(epsilon=1e-5)

        # 4. Scattering filter (optional, if enabled)
        if self.scattering_layers > 0:
            self.scatter_filter = layers.Conv1D(
                embedding_dim,
                wavelet_kernel_size,
                padding="same",
                use_bias=False,
                name="scatter_filter",
            )
            self.scatter_weight = self.add_weight(
                name="scatter_weight", shape=(1,), initializer="zeros", trainable=True
            )

        # 5. Cross-frequency attention (optional, if enabled)
        if self.use_cross_attn:
            self.cross_attn_q = layers.Dense(embedding_dim, use_bias=False, name="cross_attn_q")
            self.cross_attn_k = layers.Dense(embedding_dim, use_bias=False, name="cross_attn_k")
            self.cross_attn_v = layers.Dense(embedding_dim, use_bias=False, name="cross_attn_v")
            self.cross_attn_o = layers.Dense(embedding_dim, use_bias=False, name="cross_attn_o")

    def build(self, input_shape):
        """Explicitly builds the weights for the sub-layers.

        All layers use use_bias=False since the C++ fused_wlam op
        only requires kernel weights, not biases.
        """
        # Conv1D layers for DWT analysis
        if not self.h_filter.built:
            self.h_filter.build(input_shape)
        if not self.g_filter.built:
            self.g_filter.build(input_shape)

        # Conv1D layers for IWT synthesis
        synth_shape = tf.TensorShape([input_shape[0], None, input_shape[2]])
        if not self.h_synth_filter.built:
            self.h_synth_filter.build(synth_shape)
        if not self.g_synth_filter.built:
            self.g_synth_filter.build(synth_shape)

        # Normalization layer
        if not self.norm.built:
            self.norm.build(input_shape)

        # Build enhanced feature layers (optional)
        if self.scattering_layers > 0:
            if not self.scatter_filter.built:
                self.scatter_filter.build(input_shape)

        if self.use_cross_attn:
            # Cross-attention uses input embedding dim
            for attn_layer in [
                self.cross_attn_q,
                self.cross_attn_k,
                self.cross_attn_v,
                self.cross_attn_o,
            ]:
                if not attn_layer.built:
                    attn_layer.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass for the WLAM block using C++ fused kernel.

        Uses FusedWLAM C++ op for all wavelet processing:
        - DWT decomposition (conv or lifting scheme)
        - Frequency-specific processing (low-freq attention, high-freq conv)
        - IWT reconstruction
        - Optional: cross-frequency attention, wavelet scattering
        """
        from highnoon._native.ops.fused_wlam_op import fused_wlam

        # Get filter weights from Conv1D layers
        # Conv1D kernel shape: [kernel_size, in_channels, out_channels]
        # C++ expects: [kernel_size, embed_dim] - need to extract diagonal
        h_kernel = self.h_filter.kernel  # [K, D, D]
        g_kernel = self.g_filter.kernel  # [K, D, D]
        h_synth_kernel = self.h_synth_filter.kernel  # [K, D, D]
        g_synth_kernel = self.g_synth_filter.kernel  # [K, D, D]

        # Reduce to [K, D] by taking diagonal of last two dims
        h_filter = tf.reduce_mean(h_kernel, axis=1)  # [K, D]
        g_filter = tf.reduce_mean(g_kernel, axis=1)  # [K, D]
        h_synth = tf.reduce_mean(h_synth_kernel, axis=1)  # [K, D]
        g_synth = tf.reduce_mean(g_synth_kernel, axis=1)  # [K, D]

        # Get norm parameters
        norm_gamma = self.norm.gamma
        norm_beta = self.norm.beta

        # Optional cross-attention weights
        cross_q = self.cross_attn_q.kernel if self.use_cross_attn else None
        cross_k = self.cross_attn_k.kernel if self.use_cross_attn else None
        cross_v = self.cross_attn_v.kernel if self.use_cross_attn else None
        cross_o = self.cross_attn_o.kernel if self.use_cross_attn else None

        # Optional scattering
        scatter_filter = (
            tf.reduce_mean(self.scatter_filter.kernel, axis=1)
            if self.scattering_layers > 0
            else None
        )

        return fused_wlam(
            x=inputs,
            h_filter=h_filter,
            g_filter=g_filter,
            h_synth=h_synth,
            g_synth=g_synth,
            norm_gamma=norm_gamma,
            norm_beta=norm_beta,
            kernel_size=self.wavelet_kernel_size,
            num_heads=self.num_heads,
            cross_attn_q=cross_q,
            cross_attn_k=cross_k,
            cross_attn_v=cross_v,
            cross_attn_o=cross_o,
            scatter_filter=scatter_filter,
            num_levels=self.num_levels,
            use_lifting=self.use_lifting,
            use_adaptive=self.use_adaptive,
            scattering_layers=self.scattering_layers,
            scattering_pool=self.scattering_pool,
            use_cross_attn=self.use_cross_attn,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "wavelet_kernel_size": self.wavelet_kernel_size,
                "num_levels": self.num_levels,
                "use_lifting": self.use_lifting,
                "use_adaptive": self.use_adaptive,
                "scattering_layers": self.scattering_layers,
                "scattering_pool": self.scattering_pool,
                "use_cross_attn": self.use_cross_attn,
            }
        )
        return config

    def fused_metadata(self):
        return {
            "embedding_dim": int(self.embedding_dim),
            "num_heads": int(self.num_heads),
            "wavelet_kernel_size": int(self.wavelet_kernel_size),
            "num_levels": int(self.num_levels),
            "use_lifting": bool(self.use_lifting),
            "use_adaptive": bool(self.use_adaptive),
            "scattering_layers": int(self.scattering_layers),
            "scattering_pool": int(self.scattering_pool),
            "use_cross_attn": bool(self.use_cross_attn),
        }

    def get_fused_op_descriptor(self) -> dict[str, Any]:
        """Returns descriptor for C++ fused kernel (WLAMBlock uses Conv1D, no TT)."""
        return {
            "type": self.__class__.__name__,
            "stateful": False,
            "metadata": self.fused_metadata(),
            "weight_count": len(self.get_weights_for_fused_op()),
        }
