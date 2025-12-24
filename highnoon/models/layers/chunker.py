# src/models/layers/chunker.py
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

import logging

import tensorflow as tf
from tensorflow.keras import layers

# Configure logger for this module
log = logging.getLogger(__name__)


class StreamingChunker(layers.Layer):
    """
    Processes sequences into fixed-size, overlapping chunks.
    This version has been simplified to remove JIT-specific workarounds.
    """

    def __init__(
        self,
        chunk_size: int = 128,
        stride: int = 64,
        min_chunks: int = 2,
        tokenizer: object = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if tokenizer is None or not hasattr(tokenizer, "pad_token_id"):
            raise ValueError("A tokenizer with a valid `pad_token_id` must be provided.")
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id cannot be None. Please ensure it is set.")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("stride must be a positive integer.")
        if chunk_size <= stride:
            log.warning(
                f"chunk_size ({chunk_size}) should ideally be greater than stride ({stride}) for overlap."
            )

        self.chunk_size = chunk_size
        self.stride = stride
        self.min_chunks = min_chunks
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer

        log.info(f"StreamingChunker initialized with pad_token_id: {self.pad_token_id}")

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.bool),
        ]
    )
    def call(self, inputs: tf.Tensor, is_target: bool = False) -> tf.Tensor:
        seq_length = tf.shape(inputs)[1]
        tf.debugging.assert_type(inputs, tf.int32, "Inputs must be int32 token IDs.")

        min_length_needed = self.chunk_size + (self.min_chunks - 1) * self.stride
        padding_needed = tf.maximum(0, min_length_needed - seq_length)
        padded_inputs = tf.pad(
            inputs,
            [[0, 0], [0, padding_needed]],
            constant_values=tf.cast(self.pad_token_id, inputs.dtype),
        )

        chunks = tf.signal.frame(
            padded_inputs,
            frame_length=self.chunk_size,
            frame_step=self.stride,
            pad_end=True,
            pad_value=self.pad_token_id,
            axis=1,
        )

        tf.debugging.assert_rank(chunks, 3, "Final output must be a 3D tensor.")
        return chunks

    def compute_num_chunks(self, seq_length: tf.Tensor) -> tf.Tensor:
        """
        Computes the number of chunks that will be generated for a given sequence length.
        This logic mirrors the calculation performed by `tf.signal.frame`.
        """
        seq_length = tf.cast(seq_length, dtype=tf.int32)
        chunk_size = tf.cast(self.chunk_size, dtype=tf.int32)
        stride = tf.cast(self.stride, dtype=tf.int32)
        min_chunks = tf.cast(self.min_chunks, dtype=tf.int32)

        # First, calculate the length after padding to meet min_chunks requirement.
        min_length_needed = chunk_size + (min_chunks - 1) * stride
        padded_length = tf.maximum(seq_length, min_length_needed)

        # Then, calculate the number of frames `tf.signal.frame` will produce.
        # The formula is: (padded_length - chunk_size) // stride + 1
        # We add a small epsilon for floating point safety if we were not using integers.
        # Since we are using integers, direct calculation is fine.
        num_chunks = (padded_length - chunk_size) // stride + 1
        return num_chunks

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "chunk_size": self.chunk_size,
                "stride": self.stride,
                "min_chunks": self.min_chunks,
                "tokenizer": None,
            }
        )
        return config
