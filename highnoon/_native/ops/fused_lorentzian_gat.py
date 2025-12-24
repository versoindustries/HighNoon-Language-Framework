# highnoon/_native/ops/fused_lorentzian_gat.py
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

"""Python wrapper for the Fused Lorentzian Graph Attention custom C++ operation.

Provides graph attention with Lorentzian (hyperbolic) geometry for modeling
hierarchical relationships in language structures.
"""

import logging

import tensorflow as tf

from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

# --- Load the Custom Operator ---
_lorentzian_gat_module = None
fused_lorentzian_gat_op = None
fused_lorentzian_gat_grad_op = None

try:
    _op_lib_path = resolve_op_library(__file__, "_fused_lorentzian_gat_op.so")
    _lorentzian_gat_module = tf.load_op_library(_op_lib_path)

    if hasattr(_lorentzian_gat_module, "fused_lorentzian_gat"):
        fused_lorentzian_gat_op = _lorentzian_gat_module.fused_lorentzian_gat
        fused_lorentzian_gat_grad_op = getattr(
            _lorentzian_gat_module, "fused_lorentzian_gat_grad", None
        )
        logger.info("Successfully loaded custom C++ FusedLorentzianGat operator.")
    else:
        raise AttributeError("fused_lorentzian_gat op not found in library")
except (tf.errors.NotFoundError, OSError, AttributeError) as e:
    logger.warning(f"Could not load the custom C++ FusedLorentzianGat op: {e}")
    fused_lorentzian_gat_op = None


def fused_lorentzian_gat_available() -> bool:
    """Check if the Fused Lorentzian GAT operation is available."""
    return fused_lorentzian_gat_op is not None


class LorentzianGATLayer(tf.keras.layers.Layer):
    """Keras layer wrapper for the Lorentzian Graph Attention operation.

    This layer applies graph attention with Lorentzian (hyperbolic) inner products,
    which is useful for modeling hierarchical structures in language.

    Args:
        feature_dim: Dimension of node features.
        num_heads: Number of attention heads.
        name: Layer name.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 1,
        name: str = "lorentzian_gat",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.feature_dim = feature_dim
        self.num_heads = num_heads

    def build(self, input_shape):
        # Transformation weights
        self.transform_weights = self.add_weight(
            name="transform_weights",
            shape=(self.feature_dim, self.feature_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.transform_bias = self.add_weight(
            name="transform_bias",
            shape=(self.feature_dim,),
            initializer="zeros",
            trainable=True,
        )

        # Activation weights
        self.activation_weights = self.add_weight(
            name="activation_weights",
            shape=(self.feature_dim, self.feature_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.activation_bias = self.add_weight(
            name="activation_bias",
            shape=(self.feature_dim,),
            initializer="zeros",
            trainable=True,
        )

        # Output weights
        self.output_weights = self.add_weight(
            name="output_weights",
            shape=(self.feature_dim, self.feature_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.output_bias = self.add_weight(
            name="output_bias",
            shape=(self.feature_dim,),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(
        self,
        node_features: tf.Tensor,
        adj_indices: tf.Tensor,
        adj_values: tf.Tensor,
        adj_dense_shape: tf.Tensor,
        attention_weights: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply Lorentzian Graph Attention.

        Args:
            node_features: [batch, num_nodes, feature_dim]
            adj_indices: [num_edges, 3] - (batch_idx, src, dst)
            adj_values: [num_edges]
            adj_dense_shape: [3] - (batch_size, num_nodes, num_nodes)
            attention_weights: [batch, num_nodes, num_heads]
            training: Whether in training mode.

        Returns:
            Output features [batch, num_nodes, feature_dim]
        """
        if fused_lorentzian_gat_op is None:
            raise NotImplementedError("The C++ FusedLorentzianGat operator could not be loaded.")

        return fused_lorentzian_gat_op(
            node_features=node_features,
            adj_indices=adj_indices,
            adj_values=adj_values,
            adj_dense_shape=adj_dense_shape,
            attention_weights=attention_weights,
            lor_transform_weights=self.transform_weights,
            lor_transform_bias=self.transform_bias,
            lor_activation_weights=self.activation_weights,
            lor_activation_bias=self.activation_bias,
            lor_output_weights=self.output_weights,
            lor_output_bias=self.output_bias,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "feature_dim": self.feature_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


__all__ = [
    "fused_lorentzian_gat_available",
    "LorentzianGATLayer",
]
