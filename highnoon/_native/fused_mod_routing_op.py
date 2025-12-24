# highnoon/_native/fused_mod_routing_op.py
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

"""Python wrapper for Mixture-of-Depths (MoD) routing operations.

Implements dynamic per-token layer skipping from Google DeepMind (2024).
Reduces compute by ~50% by only processing "important" tokens through layers.
"""


import tensorflow as tf

from highnoon._native import _ops


class MoDRouter(tf.keras.layers.Layer):
    """Mixture-of-Depths router layer.

    Routes tokens to either process through a layer or skip it based on
    learned importance scores. Maintains a capacity budget to ensure
    consistent compute cost.

    Args:
        capacity_factor: Fraction of tokens to process (0.0-1.0).
        use_auxiliary_loss: Whether to add load balancing loss.
        aux_loss_weight: Weight of auxiliary loss.

    Example:
        >>> router = MoDRouter(capacity_factor=0.5)
        >>> hidden = tf.random.normal([2, 128, 512])
        >>> probs, mask, aux_loss = router(hidden)
        >>> # ~64 tokens selected per sequence
    """

    def __init__(
        self,
        capacity_factor: float = 0.5,
        use_auxiliary_loss: bool = True,
        aux_loss_weight: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.capacity_factor = capacity_factor
        self.use_auxiliary_loss = use_auxiliary_loss
        self.aux_loss_weight = aux_loss_weight

    def build(self, input_shape):
        hidden_dim = input_shape[-1]

        self.router_weight = self.add_weight(
            name="router_weight",
            shape=(hidden_dim,),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.router_bias = self.add_weight(
            name="router_bias",
            shape=(),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(
        self,
        hidden: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute routing decisions.

        Args:
            hidden: Input hidden states [batch, seq_len, hidden_dim]
            training: Whether in training mode

        Returns:
            router_probs: Per-token probabilities [batch, seq_len]
            selected_mask: Binary selection mask [batch, seq_len]
            auxiliary_loss: Load balancing loss [batch]
        """
        return _ops.mo_d_route(
            hidden=hidden,
            router_weight=self.router_weight,
            router_bias=self.router_bias,
            capacity_factor=self.capacity_factor,
            use_auxiliary_loss=self.use_auxiliary_loss,
            aux_loss_weight=self.aux_loss_weight,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "capacity_factor": self.capacity_factor,
                "use_auxiliary_loss": self.use_auxiliary_loss,
                "aux_loss_weight": self.aux_loss_weight,
            }
        )
        return config


class MoDBlock(tf.keras.layers.Layer):
    """Mixture-of-Depths block wrapper.

    Wraps any layer with MoD routing, allowing dynamic depth based on
    token importance.

    Args:
        inner_layer: The layer to wrap (e.g., attention, FFN).
        capacity_factor: Fraction of tokens to process.
        use_auxiliary_loss: Whether to add load balancing loss.

    Example:
        >>> ffn = tf.keras.layers.Dense(2048, activation='gelu')
        >>> mod_ffn = MoDBlock(ffn, capacity_factor=0.5)
        >>> output = mod_ffn(hidden)  # Only 50% of tokens processed by FFN
    """

    def __init__(
        self,
        inner_layer: tf.keras.layers.Layer,
        capacity_factor: float = 0.5,
        use_auxiliary_loss: bool = True,
        aux_loss_weight: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inner_layer = inner_layer
        self.capacity_factor = capacity_factor
        self.use_auxiliary_loss = use_auxiliary_loss
        self.aux_loss_weight = aux_loss_weight
        self.router = None

    def build(self, input_shape):
        input_shape[-1]
        seq_len = input_shape[1]

        self.router = MoDRouter(
            capacity_factor=self.capacity_factor,
            use_auxiliary_loss=self.use_auxiliary_loss,
            aux_loss_weight=self.aux_loss_weight,
            name=f"{self.name}_router",
        )

        # Pre-compute capacity
        if seq_len is not None:
            self._capacity = max(1, int(seq_len * self.capacity_factor))
        else:
            self._capacity = None

        super().build(input_shape)

    def call(
        self,
        hidden: tf.Tensor,
        training: bool = False,
        return_aux_loss: bool = False,
    ) -> tf.Tensor:
        """Apply inner layer with MoD routing.

        Args:
            hidden: Input [batch, seq_len, hidden_dim]
            training: Whether in training mode
            return_aux_loss: Whether to return auxiliary loss

        Returns:
            output: Result with MoD routing applied
            aux_loss: (if return_aux_loss) Load balancing loss
        """
        tf.shape(hidden)[0]
        seq_len = tf.shape(hidden)[1]
        hidden.shape[-1]

        # Get routing decisions
        router_probs, selected_mask, aux_loss = self.router(hidden, training=training)

        # Compute capacity dynamically if needed
        capacity = self._capacity or tf.maximum(
            1, tf.cast(tf.cast(seq_len, tf.float32) * self.capacity_factor, tf.int32)
        )

        # Gather selected tokens
        gathered, gather_indices, num_selected = _ops.mo_d_gather(
            hidden=hidden,
            selected_mask=selected_mask,
            capacity=capacity,
        )

        # Apply inner layer to gathered tokens
        processed = self.inner_layer(gathered, training=training)

        # Scatter back to original positions
        output = _ops.mo_d_scatter(
            original=hidden,
            processed=processed,
            gather_indices=gather_indices,
            router_probs=router_probs,
            num_selected=num_selected,
        )

        if return_aux_loss:
            return output, tf.reduce_mean(aux_loss)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "capacity_factor": self.capacity_factor,
                "use_auxiliary_loss": self.use_auxiliary_loss,
                "aux_loss_weight": self.aux_loss_weight,
            }
        )
        return config


def wrap_with_mod(layer: tf.keras.layers.Layer, capacity_factor: float = 0.5, **kwargs) -> MoDBlock:
    """Convenience function to wrap a layer with MoD routing.

    Args:
        layer: Layer to wrap
        capacity_factor: Fraction of tokens to process
        **kwargs: Additional MoDBlock arguments

    Returns:
        MoDBlock wrapping the input layer
    """
    return MoDBlock(layer, capacity_factor=capacity_factor, **kwargs)
