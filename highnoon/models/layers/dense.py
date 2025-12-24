# src/models/layers/dense.py
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


class LowRankDense(layers.Layer):
    """
    A Dense layer implemented with low-rank factorization (W = W1 * W2)
    to reduce the number of trainable parameters. This is a more flexible
    implementation that can be used as a drop-in for a standard Dense layer.
    """

    def __init__(
        self,
        output_dim: int,
        rank: int | None = None,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.rank = rank
        self.activation_fn = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.W1, self.W2, self.W, self.b = None, None, None, None

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # If no rank is provided, or if rank is too large, default to a standard Dense layer.
        if self.rank is None or self.rank >= min(input_dim, self.output_dim):
            self.W = self.add_weight(
                shape=(input_dim, self.output_dim),
                initializer=self.kernel_initializer,
                trainable=True,
                name="W",
                dtype=tf.float32,
            )
        else:  # Use low-rank factorization
            self.W1 = self.add_weight(
                shape=(input_dim, self.rank),
                initializer=self.kernel_initializer,
                trainable=True,
                name="W1",
                dtype=tf.float32,
            )
            self.W2 = self.add_weight(
                shape=(self.rank, self.output_dim),
                initializer=self.kernel_initializer,
                trainable=True,
                name="W2",
                dtype=tf.float32,
            )

        self.b = self.add_weight(
            shape=(self.output_dim,),
            initializer=self.bias_initializer,
            trainable=True,
            name="b",
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, inputs):
        """Forward pass of the low-rank dense layer."""
        if self.W is not None:  # Standard dense operation
            x = tf.matmul(inputs, self.W)
        else:  # Low-rank operation
            x = tf.matmul(tf.matmul(inputs, self.W1), self.W2)

        x = x + self.b
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x

    def get_config(self):
        """Enables serialization of the layer."""
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "rank": self.rank,
                "activation": tf.keras.activations.serialize(self.activation_fn),
                "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            }
        )
        return config
