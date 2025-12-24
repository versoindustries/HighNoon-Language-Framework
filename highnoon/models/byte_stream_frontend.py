# highnoon/models/byte_stream_frontend.py
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

"""Phase 10.5: Byte-Stream Frontend for token-free processing.

This module implements the ByteStreamFrontend layer, which provides an
alternative to semantic tokenization by processing raw UTF-8 bytes through
a Mamba-based spatial block before the wavelet transform stage.

Inspired by MambaByte, this frontend enables the model to learn its own
internal representation from raw bytes rather than relying on learned
tokenization vocabularies.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf


class ByteStreamFrontend(tf.keras.layers.Layer):
    """Frontend layer for byte-level text processing with strided pooling.

    This layer converts raw byte IDs (0-255) into embeddings and processes
    them through a spatial block (Mamba-style SSM) before outputting to
    the main model pipeline. Strided pooling reduces sequence length to
    prevent explosion from byte-level tokenization.

    Architecture:
        Byte IDs → Embedding(256) → SpatialBlock → StridedPool → Projection → Output

    Example:
        >>> frontend = ByteStreamFrontend(
        ...     embedding_dim=512,
        ...     byte_embedding_dim=128,
        ...     stride_factor=4,
        ... )
        >>> byte_ids = tf.constant([[72, 101, 108, 108, 111, 33, 33, 33]])  # 8 bytes
        >>> embeddings = frontend(byte_ids)
        >>> embeddings.shape
        TensorShape([1, 2, 512])  # 8 / 4 = 2

    Attributes:
        embedding_dim: Output embedding dimension (matches model dim).
        byte_embedding_dim: Internal byte embedding dimension.
        state_dim: State dimension for spatial processing.
        conv_dim: Convolution dimension for spatial block.
        stride_factor: Pooling stride to reduce sequence length.
    """

    def __init__(
        self,
        embedding_dim: int,
        byte_embedding_dim: int = 128,
        state_dim: int = 16,
        conv_dim: int = 4,
        expand_factor: int = 2,
        dropout_rate: float = 0.1,
        stride_factor: int = 4,
        name: str = "byte_stream_frontend",
        **kwargs,
    ):
        """Initialize the ByteStreamFrontend.

        Args:
            embedding_dim: Output embedding dimension (model hidden size).
            byte_embedding_dim: Byte embedding dimension before projection.
            state_dim: State dimension for spatial block.
            conv_dim: Convolution kernel size for spatial block.
            expand_factor: Expansion factor for spatial block.
            dropout_rate: Dropout rate for regularization.
            stride_factor: Pooling stride to reduce byte sequence length.
                Output length = input_length / stride_factor.
            name: Layer name.
            **kwargs: Additional layer arguments.
        """
        super().__init__(name=name, **kwargs)
        self.embedding_dim = embedding_dim
        self.byte_embedding_dim = byte_embedding_dim
        self.state_dim = state_dim
        self.conv_dim = conv_dim
        self.expand_factor = expand_factor
        self.dropout_rate = dropout_rate
        self.stride_factor = stride_factor

        # Layers built in build()
        self.byte_embedding: tf.keras.layers.Embedding | None = None
        self.byte_mamba: tf.keras.layers.Layer | None = None  # SpatialBlock
        self.output_projection: tf.keras.layers.Dense | None = None
        self.output_norm: tf.keras.layers.LayerNormalization | None = None
        self.dropout: tf.keras.layers.Dropout | None = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer weights.

        Args:
            input_shape: Shape of input tensor [batch, seq_len].
        """
        # Byte embedding table (256 tokens for all byte values)
        self.byte_embedding = tf.keras.layers.Embedding(
            input_dim=256,
            output_dim=self.byte_embedding_dim,
            embeddings_initializer="glorot_uniform",
            name=f"{self.name}_byte_embedding",
        )

        # Import SpatialBlock here to avoid circular imports
        from .spatial.mamba import SpatialBlock

        # Spatial processing block for byte-level patterns
        self.byte_mamba = SpatialBlock(
            embedding_dim=self.byte_embedding_dim,
            state_dim=self.state_dim,
            conv_dim=self.conv_dim,
            expand_factor=self.expand_factor,
            name=f"{self.name}_byte_mamba",
        )

        # Project to model embedding dimension
        self.output_projection = tf.keras.layers.Dense(
            self.embedding_dim,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            name=f"{self.name}_output_projection",
        )

        self.output_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{self.name}_output_norm",
        )

        self.dropout = tf.keras.layers.Dropout(
            self.dropout_rate,
            name=f"{self.name}_dropout",
        )

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass through byte-stream frontend.

        Args:
            inputs: Byte IDs tensor [batch, seq_len] with values 0-255.
            training: Whether in training mode.

        Returns:
            Embeddings tensor [batch, seq_len // stride_factor, embedding_dim].
        """
        # Embed bytes
        x = self.byte_embedding(inputs)  # [batch, seq_len, byte_embedding_dim]

        # Process through spatial block (O(L) linear-time)
        x = self.byte_mamba(x, training=training)

        # Strided pooling to reduce sequence length (Phase 11.3)
        # This prevents sequence length explosion from byte-level processing
        if self.stride_factor > 1:
            x = x[:, :: self.stride_factor, :]  # [batch, seq_len // stride, dim]

        # Project to model dimension
        x = self.output_projection(x)  # [batch, pooled_len, embedding_dim]
        x = self.output_norm(x)
        x = self.dropout(x, training=training)

        return x

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "byte_embedding_dim": self.byte_embedding_dim,
                "state_dim": self.state_dim,
                "conv_dim": self.conv_dim,
                "expand_factor": self.expand_factor,
                "dropout_rate": self.dropout_rate,
                "stride_factor": self.stride_factor,
            }
        )
        return config


__all__ = ["ByteStreamFrontend"]
