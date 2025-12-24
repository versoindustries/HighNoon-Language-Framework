# src/models/custom_attention.py
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

import os

import tensorflow as tf
from tensorflow.keras import layers

# --- Import DEBUG_MODE flag ---
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"


class CustomBigBirdAttention(layers.Layer):
    """
    A self-contained implementation of BigBird sparse attention.

    This layer avoids the dependency on tf-models-official and resolves the
    internal mask-creation error by handling all sparse masking logic internally.
    It combines window, global, and random attention patterns to efficiently
    process long sequences.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        block_size=64,
        num_rand_blocks=3,
        window_size=3,
        global_tokens=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.block_size = block_size
        self.num_rand_blocks = num_rand_blocks
        self.window_size = window_size
        self.global_tokens = global_tokens
        self.head_dim = key_dim // num_heads
        if key_dim % num_heads != 0:
            raise ValueError(f"key_dim ({key_dim}) must be divisible by num_heads ({num_heads})")

    def build(self, input_shape):
        self.query_proj = layers.Dense(self.key_dim, name="query")
        self.key_proj = layers.Dense(self.key_dim, name="key")
        self.value_proj = layers.Dense(self.key_dim, name="value")
        self.output_proj = layers.Dense(input_shape[-1], name="output")
        super().build(input_shape)

    def call(self, query, value, key, attention_mask=None, training=None):
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[1]

        # 1. Project inputs to Q, K, V
        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        # Reshape for multi-head attention
        q = tf.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = tf.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        scaling_factor = tf.cast(self.head_dim, q.dtype) ** -0.5
        q *= scaling_factor

        # 2. Generate the block-sparse attention mask and gather indices
        blocked_mask, attn_indices = self._create_sparse_attention_mask(seq_len)
        num_sparse_neighbors = tf.shape(attn_indices)[1]

        # Reshape Q, K, V to merge batch and head dimensions: (B, H, N, D_h) -> (B*H, N, D_h)
        q_reshaped = tf.reshape(q, (batch_size * self.num_heads, seq_len, self.head_dim))
        k_reshaped = tf.reshape(k, (batch_size * self.num_heads, seq_len, self.head_dim))
        v_reshaped = tf.reshape(v, (batch_size * self.num_heads, seq_len, self.head_dim))

        # Tile attention indices for the merged batch dimension
        attn_indices_tiled = tf.tile(
            attn_indices[tf.newaxis, :, :], [batch_size * self.num_heads, 1, 1]
        )

        # 3. Compute sparse attention
        # Gather the key and value vectors corresponding to the sparse indices using batch_dims=1
        k_gathered = tf.gather(k_reshaped, attn_indices_tiled, batch_dims=1)
        v_gathered = tf.gather(v_reshaped, attn_indices_tiled, batch_dims=1)

        # Compute sparse dot-product: (B*H, N, D_h) x (B*H, N, M, D_h) -> (B*H, N, M)
        attn_scores_reshaped = tf.einsum("bnd,bnmd->bnm", q_reshaped, k_gathered)

        # Reshape scores back to (B, H, N, M) to apply masks
        attn_scores = tf.reshape(
            attn_scores_reshaped, (batch_size, self.num_heads, seq_len, num_sparse_neighbors)
        )

        # Apply the sparse mask (add -1e9 to positions that are not attended to)
        attn_scores += blocked_mask

        # Apply padding mask if provided
        if attention_mask is not None:
            mask_expanded = attention_mask[:, tf.newaxis, tf.newaxis, :]  # (B, 1, 1, N)
            attn_scores += (1.0 - tf.cast(mask_expanded, attn_scores.dtype)) * -1e9

        attn_probs = tf.nn.softmax(attn_scores, axis=-1)

        # Reshape probs to (B*H, N, M) for einsum with v_gathered
        attn_probs_reshaped = tf.reshape(
            attn_probs, (batch_size * self.num_heads, seq_len, num_sparse_neighbors)
        )

        # Compute weighted sum: (B*H, N, M) x (B*H, N, M, D_h) -> (B*H, N, D_h)
        context_vector_reshaped = tf.einsum("bnm,bnmd->bnd", attn_probs_reshaped, v_gathered)

        # Reshape context back to (B, H, N, D_h)
        context_vector = tf.reshape(
            context_vector_reshaped, (batch_size, self.num_heads, seq_len, self.head_dim)
        )

        # 4. Reshape and project output
        context_vector = tf.transpose(context_vector, [0, 2, 1, 3])
        context_vector = tf.reshape(context_vector, (batch_size, seq_len, self.key_dim))

        output = self.output_proj(context_vector)
        # MODIFIED: Return the attention indices along with the output and scores.
        # This makes the learned graph structure accessible to other layers.
        return output, attn_scores, attn_indices

    @tf.function
    def _create_sparse_attention_mask(self, seq_len):
        """
        Creates the block-sparse attention mask and gather indices.
        This is the heart of the BigBird implementation.
        FIX: Rewritten to be pure TensorFlow, removing tf.py_function to prevent
        graph mode errors. This resolves the `InvalidArgumentError` about tensor rank.
        """

        # --- START: FIX ---
        # Handle short sequences by creating a dense mask, bypassing the complex blocking logic
        # which fails when seq_len < block_size. Use tf.cond to handle symbolic seq_len.
        def dense_mask_fn():
            full_indices = tf.range(seq_len, dtype=tf.int32)
            gather_indices = tf.tile(full_indices[tf.newaxis, :], [seq_len, 1])
            attention_mask = tf.zeros((1, 1, seq_len, seq_len), dtype=tf.float32)
            return attention_mask, gather_indices

        def sparse_mask_fn():
            num_blocks = (seq_len + self.block_size - 1) // self.block_size

            # --- Window Attention Indices ---
            r = tf.range(num_blocks)
            window_indices = tf.stack(tf.meshgrid(r, r, indexing="ij"), axis=-1)
            window_mask = (
                tf.abs(window_indices[..., 0] - window_indices[..., 1]) <= self.window_size
            )
            window_indices = tf.boolean_mask(window_indices, window_mask)

            # --- Global Attention Indices ---
            global_indices_to = tf.stack(
                [
                    tf.tile(tf.range(num_blocks)[:, tf.newaxis], [1, self.global_tokens]),
                    tf.tile(tf.range(self.global_tokens)[tf.newaxis, :], [num_blocks, 1]),
                ],
                axis=-1,
            )
            global_indices_from = tf.stack(
                [
                    tf.tile(tf.range(self.global_tokens)[:, tf.newaxis], [1, num_blocks]),
                    tf.tile(tf.range(num_blocks)[tf.newaxis, :], [self.global_tokens, 1]),
                ],
                axis=-1,
            )
            global_indices = tf.concat(
                [tf.reshape(global_indices_to, [-1, 2]), tf.reshape(global_indices_from, [-1, 2])],
                axis=0,
            )

            # --- Random Attention Indices ---
            num_random_pairs = self.num_rand_blocks * num_blocks
            rand_indices_1 = tf.random.uniform(
                shape=(num_random_pairs,), maxval=num_blocks, dtype=tf.int32
            )
            rand_indices_2 = tf.random.uniform(
                shape=(num_random_pairs,), maxval=num_blocks, dtype=tf.int32
            )
            random_indices = tf.stack([rand_indices_1, rand_indices_2], axis=-1)

            # --- Combine, Sort, and Uniq ---
            all_indices = tf.concat([window_indices, global_indices, random_indices], axis=0)
            # Unique op requires a 1D tensor, so we flatten and then reconstruct.
            flat_indices = all_indices[:, 0] * num_blocks + all_indices[:, 1]
            unique_flat_indices = tf.unique(flat_indices).y
            unique_indices = tf.stack(
                [unique_flat_indices // num_blocks, unique_flat_indices % num_blocks], axis=1
            )

            # --- Create Gather Indices and Mask without materialising (seq_len x seq_len) ---
            flat_rows = tf.cast(unique_indices[:, 0], tf.int32)
            flat_cols = tf.cast(unique_indices[:, 1], tf.int32)

            sort_keys = flat_rows * num_blocks + flat_cols
            sort_order = tf.argsort(sort_keys, stable=True)
            sorted_rows = tf.gather(flat_rows, sort_order)
            sorted_cols = tf.gather(flat_cols, sort_order)

            block_neighbors = tf.RaggedTensor.from_value_rowids(
                sorted_cols,
                value_rowids=sorted_rows,
                nrows=num_blocks,
            )

            block_indices = tf.range(num_blocks, dtype=tf.int32)
            block_starts = block_indices * self.block_size
            block_ends = tf.minimum(block_starts + self.block_size, seq_len)

            neighbor_flat = block_neighbors.flat_values
            neighbor_starts = tf.gather(block_starts, neighbor_flat)
            neighbor_limits = tf.gather(block_ends, neighbor_flat)
            neighbor_token_ranges = tf.ragged.range(neighbor_starts, neighbor_limits)

            neighbor_to_block = tf.cast(
                tf.ragged.row_splits_to_segment_ids(block_neighbors.row_splits),
                tf.int32,
            )
            neighbor_token_lengths = tf.cast(neighbor_token_ranges.row_lengths(), tf.int32)
            block_token_lengths = tf.math.unsorted_segment_sum(
                neighbor_token_lengths,
                neighbor_to_block,
                num_segments=num_blocks,
            )
            block_token_row_splits = tf.cast(
                tf.concat([[0], tf.cumsum(block_token_lengths)], axis=0),
                tf.int64,
            )

            block_token_indices = tf.RaggedTensor.from_row_splits(
                values=neighbor_token_ranges.values,
                row_splits=block_token_row_splits,
            )

            block_token_dense = block_token_indices.to_tensor(default_value=-1)
            block_token_mask = tf.where(
                block_token_dense >= 0,
                tf.zeros_like(block_token_dense, dtype=tf.float32),
                tf.fill(tf.shape(block_token_dense), tf.constant(-1e9, dtype=tf.float32)),
            )

            row_block_ids = tf.range(seq_len, dtype=tf.int32) // self.block_size
            row_block_ids = tf.minimum(row_block_ids, num_blocks - 1)

            padded_indices = tf.gather(block_token_dense, row_block_ids)
            final_attention_mask = tf.gather(block_token_mask, row_block_ids)

            safe_padded_indices = tf.maximum(0, padded_indices)

            return final_attention_mask[tf.newaxis, tf.newaxis, :, :], safe_padded_indices

        return tf.cond(seq_len <= self.block_size, true_fn=dense_mask_fn, false_fn=sparse_mask_fn)
        # --- END: FIX ---

    def _get_random_block_indices(self, num_blocks):
        """Generates pairs of random block indices to attend to."""
        rand_indices_1 = tf.random.uniform(
            shape=(self.num_rand_blocks * num_blocks,), maxval=num_blocks, dtype=tf.int32
        )
        rand_indices_2 = tf.random.uniform(
            shape=(self.num_rand_blocks * num_blocks,), maxval=num_blocks, dtype=tf.int32
        )
        rand_pairs = tf.stack([rand_indices_1, rand_indices_2], axis=1)
        return [(int(p[0]), int(p[1])) for p in rand_pairs.numpy()]
