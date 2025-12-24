# src/models/monarch.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from tensorflow.keras import layers

from highnoon._native.ops.fused_norm_proj_acct_op import fused_norm_proj_act
from highnoon.config import TT_LAYER_CONFIGS  # Import centralized TT configs

# MODIFIED: Import TTLayer for systematic footprint reduction
from highnoon.models.tensor_layers import TTLayer  # Import TTLayer


class MonarchMixer(layers.Layer):
    """
    A hardware-efficient block for unified sequence and channel mixing,
    based on the Monarch matrix principles. This layer serves as a
    sub-quadratic drop-in replacement for both attention and MLP layers.
    """

    def __init__(self, d_model, ff_expansion=2, dropout=0.1, lrf_rank: int = None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.ff_expansion = ff_expansion
        self.dropout = dropout
        self.lrf_rank = lrf_rank

        self.ffn_dim = int(self.d_model * self.ff_expansion)

        # The first FFN block (LayerNorm -> Dense -> Activation) is replaced
        # by weights for the fused kernel.
        # MODIFIED: The second FFN block is replaced with TTLayer, using the centralized config.
        self.ffn2 = TTLayer(**TT_LAYER_CONFIGS["monarch_ffn2"], name="ffn2_tt")
        self.ffn_dropout = layers.Dropout(self.dropout)

        # The norm1 is now part of the fused kernel.
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

    def build(self, input_shape):
        """Creates the weights for the fused kernel."""
        # Weights for the fused LayerNorm -> Dense -> GELU operation (ffn1).
        self.ffn1_gamma = self.add_weight(
            name="ffn1_gamma", shape=(self.d_model,), initializer="ones"
        )
        self.ffn1_beta = self.add_weight(
            name="ffn1_beta", shape=(self.d_model,), initializer="zeros"
        )
        self.ffn1_proj_weights = self.add_weight(
            name="ffn1_proj_weights",
            shape=(self.d_model, self.ffn_dim),
            initializer="glorot_uniform",
        )
        self.ffn1_proj_bias = self.add_weight(
            name="ffn1_proj_bias", shape=(self.ffn_dim,), initializer="zeros"
        )
        super().build(input_shape)

    def call(self, x, training=False):
        """
        Forward pass for the Monarch Mixer.
        Applies a simple feed-forward network as a placeholder for the
        more complex Monarch matrix operations.
        """
        # First residual connection, now using the fused kernel for the first part of the FFN.
        ffn1_output = fused_norm_proj_act(
            x,
            self.ffn1_gamma,
            self.ffn1_beta,
            self.ffn1_proj_weights,
            self.ffn1_proj_bias,
            activation="gelu",
        )
        ffn2_output = self.ffn2(ffn1_output)
        x = x + self.ffn_dropout(ffn2_output, training=training)

        # Second residual connection (identity in this simplified version)
        x = x + self.norm2(x)

        return x
