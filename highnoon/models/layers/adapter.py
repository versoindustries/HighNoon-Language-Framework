# src/models/layers/adapter.py
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

import tensorflow as tf
from tensorflow.keras import layers


class AdapterLayer(layers.Layer):
    """
    A simple adapter layer that projects an input down to a smaller dimension
    and then projects it back up, adding the result to the original input.
    This is a form of lightweight fine-tuning.
    """

    def __init__(self, input_dim, adapter_dim=64, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim
        self.activation = tf.keras.activations.get(activation)

        self.down_project = layers.Dense(
            self.adapter_dim, activation=self.activation, dtype="float32"
        )
        self.up_project = layers.Dense(self.input_dim, dtype="float32")

    def build(self, input_shape):
        """Explicitly builds the weights for the sub-layers."""
        self.down_project.build(input_shape)
        # The input shape for the up-projection is the output of the down-projection
        up_project_input_shape = tf.TensorShape([input_shape[0], self.adapter_dim])
        self.up_project.build(up_project_input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass of the adapter layer."""
        down_projected = self.down_project(inputs)
        up_projected = self.up_project(down_projected)
        return inputs + up_projected

    def get_config(self):
        """Enables serialization of the layer."""
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "adapter_dim": self.adapter_dim,
                "activation": tf.keras.activations.serialize(self.activation),
            }
        )
        return config
