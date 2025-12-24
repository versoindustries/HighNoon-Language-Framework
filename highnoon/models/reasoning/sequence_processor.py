# highnoon/models/reasoning/sequence_processor.py
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

"""Sequence Processor for HighNoon Reasoning Module.

This module provides utilities for processing sequences through
reasoning blocks, including training and inference modes.
"""

from typing import Any

import tensorflow as tf


class SequenceProcessor:
    """Utilities for sequence processing through reasoning blocks.

    Provides static methods for:
    - Processing sequences for training
    - Getting initial states for stateful blocks
    - Running autoregressive inference
    """

    @staticmethod
    def process_sequence_for_training(
        sequence: tf.Tensor,
        blocks: list[tf.keras.layers.Layer],
        training: bool = True,
    ) -> tuple[tf.Tensor, dict[str, Any], dict[str, Any]]:
        """Process a sequence through reasoning blocks for training.

        Args:
            sequence: Input sequence [batch, seq_len, embedding_dim].
            blocks: List of reasoning block layers.
            training: Whether in training mode.

        Returns:
            Tuple of (output_sequence, moe_info, aux_metrics).
        """
        output = sequence
        moe_info = {}
        aux_metrics = {}

        for _i, block in enumerate(blocks):
            output = block(output, training=training)

        return output, moe_info, aux_metrics

    @staticmethod
    def get_initial_states(
        blocks: list[tf.keras.layers.Layer],
        batch_size: int,
    ) -> list[tf.Tensor | None]:
        """Get initial states for stateful blocks.

        Args:
            blocks: List of reasoning block layers.
            batch_size: Batch size for state tensors.

        Returns:
            List of initial state tensors (None for stateless blocks).
        """
        states = []
        for _block in blocks:
            # Most blocks are stateless
            states.append(None)
        return states

    @staticmethod
    def run_inference(
        blocks: list[tf.keras.layers.Layer],
        token_embedding_layer: tf.keras.layers.Layer,
        final_norm_layer: tf.keras.layers.Layer,
        final_dense_layer: tf.keras.layers.Layer,
        start_tokens: tf.Tensor,
        stacked_memory: tf.Tensor,
        max_gen_len: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> tf.Tensor:
        """Run autoregressive inference.

        Args:
            blocks: List of reasoning block layers.
            token_embedding_layer: Token embedding layer.
            final_norm_layer: Final normalization layer.
            final_dense_layer: Final output projection layer.
            start_tokens: Starting tokens [batch, 1].
            stacked_memory: Memory context [batch, mem_len, dim].
            max_gen_len: Maximum generation length.
            eos_token_id: End-of-sequence token ID.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.

        Returns:
            Generated token IDs [batch, gen_len].
        """
        generated = start_tokens

        for _ in range(max_gen_len):
            # Embed current tokens
            token_embeddings = token_embedding_layer(generated)

            # Concatenate with memory
            full_sequence = tf.concat([stacked_memory, token_embeddings], axis=1)

            # Process through blocks
            for block in blocks:
                full_sequence = block(full_sequence, training=False)

            # Get last position output
            last_output = full_sequence[:, -1:, :]

            # Normalize and project
            normalized = final_norm_layer(last_output)
            logits = final_dense_layer(normalized)

            # Apply temperature
            logits = logits[:, 0, :] / temperature

            # Top-k filtering
            if top_k is not None and top_k > 0:
                top_k_vals, _ = tf.math.top_k(logits, k=top_k)
                threshold = top_k_vals[:, -1:]
                logits = tf.where(
                    logits < threshold,
                    tf.fill(tf.shape(logits), float("-inf")),
                    logits,
                )

            # Sample
            probs = tf.nn.softmax(logits)
            next_token = tf.random.categorical(tf.math.log(probs + 1e-10), num_samples=1)
            next_token = tf.cast(next_token, tf.int32)

            # Append to generated
            generated = tf.concat([generated, next_token], axis=1)

            # Check for EOS
            if tf.reduce_all(next_token == eos_token_id):
                break

        return generated


# Convenience functions for direct import
def process_sequence_for_training(
    sequence: tf.Tensor,
    initial_states: list[tf.Tensor | None],
    blocks: list[tf.keras.layers.Layer],
    training: bool = True,
) -> tuple[tf.Tensor, dict[str, Any], dict[str, Any]]:
    """Process sequence for training (wrapper for compatibility)."""
    return SequenceProcessor.process_sequence_for_training(sequence, blocks, training)


def get_initial_states(
    blocks: list[tf.keras.layers.Layer],
    batch_size: int,
) -> list[tf.Tensor | None]:
    """Get initial states (wrapper for compatibility)."""
    return SequenceProcessor.get_initial_states(blocks, batch_size)


def run_inference(
    reasoning_blocks: list[tf.keras.layers.Layer],
    token_embedding_layer: tf.keras.layers.Layer,
    final_norm_layer: tf.keras.layers.Layer,
    final_dense_layer: tf.keras.layers.Layer,
    start_tokens: tf.Tensor,
    stacked_memory: tf.Tensor,
    max_gen_len: tf.Tensor,
    eos_token_id: tf.Tensor,
) -> tf.Tensor:
    """Run inference (wrapper for compatibility)."""
    return SequenceProcessor.run_inference(
        reasoning_blocks,
        token_embedding_layer,
        final_norm_layer,
        final_dense_layer,
        start_tokens,
        stacked_memory,
        int(max_gen_len),
        int(eos_token_id),
    )
