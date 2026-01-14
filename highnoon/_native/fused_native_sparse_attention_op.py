# highnoon/_native/fused_native_sparse_attention_op.py
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

"""Python wrapper for Native Sparse Attention (NSA) operation.

Implements DeepSeek's NSA mechanism (ACL 2025 Best Paper) achieving:
- 9x faster forward computation
- 6x faster backward computation
- 11.6x memory reduction for 64K sequences

Complexity: O(n log n) instead of O(n²)
"""

import tensorflow as tf

from highnoon._native import _ops


def native_sparse_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    block_size: int = 64,
    num_selected_blocks: int = 8,
    num_selected_tokens: int = 32,
    sliding_window_size: int = 128,
    num_global_tokens: int = 1,
    temperature: float = 1.0,
    name: str | None = None,
) -> tf.Tensor:
    """Native Sparse Attention operation.

    Combines three attention patterns for efficient long-context:
    1. Block-level compressed attention (global context)
    2. Fine-grained token selection (important details)
    3. Sliding window (local context preservation)

    Args:
        query: Query tensor [batch, num_heads, seq_q, head_dim]
        key: Key tensor [batch, num_heads, seq_k, head_dim]
        value: Value tensor [batch, num_heads, seq_k, head_dim]
        block_size: Size of compression blocks (default 64)
        num_selected_blocks: Blocks to attend after compression (default 8)
        num_selected_tokens: Tokens per block for fine-grained (default 32)
        sliding_window_size: Local attention window (default 128)
        num_global_tokens: Global tokens like CLS (default 1)
        temperature: Attention temperature (default 1.0)
        name: Optional operation name

    Returns:
        Output tensor [batch, num_heads, seq_q, head_dim]

    Example:
        >>> batch, heads, seq, dim = 2, 8, 4096, 64
        >>> Q = tf.random.normal([batch, heads, seq, dim])
        >>> K = tf.random.normal([batch, heads, seq, dim])
        >>> V = tf.random.normal([batch, heads, seq, dim])
        >>> output = native_sparse_attention(Q, K, V)
        >>> # Efficient attention with O(n log n) complexity
    """
    with tf.name_scope(name or "NativeSparseAttention"):
        query = tf.cast(query, tf.float32)
        key = tf.cast(key, tf.float32)
        value = tf.cast(value, tf.float32)

        return _ops.native_sparse_attention(
            query=query,
            key=key,
            value=value,
            block_size=block_size,
            num_selected_blocks=num_selected_blocks,
            num_selected_tokens=num_selected_tokens,
            sliding_window_size=sliding_window_size,
            num_global_tokens=num_global_tokens,
            temperature=temperature,
        )


@tf.RegisterGradient("NativeSparseAttention")
def _native_sparse_attention_grad(op, grad_output):
    """Gradient for NativeSparseAttention."""
    query = op.inputs[0]
    key = op.inputs[1]
    value = op.inputs[2]

    grad_query, grad_key, grad_value = _ops.native_sparse_attention_grad(
        grad_output=grad_output,
        query=query,
        key=key,
        value=value,
        block_size=op.get_attr("block_size"),
        num_selected_blocks=op.get_attr("num_selected_blocks"),
        num_selected_tokens=op.get_attr("num_selected_tokens"),
        sliding_window_size=op.get_attr("sliding_window_size"),
        num_global_tokens=op.get_attr("num_global_tokens"),
        temperature=op.get_attr("temperature"),
    )

    return [grad_query, grad_key, grad_value]


class NativeSparseAttention(tf.keras.layers.Layer):
    """Keras layer for Native Sparse Attention.

    Implements DeepSeek's NSA with configurable sparsity patterns.
    Optimal for sequences >4K tokens.

    Args:
        block_size: Token compression block size
        num_selected_blocks: Blocks to attend after compression
        num_selected_tokens: Tokens per block for fine-grained attention
        sliding_window_size: Local attention window
        num_global_tokens: Always-attend tokens (CLS, etc.)
        temperature: Attention temperature

    Example:
        >>> layer = NativeSparseAttention(
        ...     block_size=64,
        ...     num_selected_blocks=8,
        ...     sliding_window_size=128
        ... )
        >>> Q = tf.random.normal([2, 8, 4096, 64])
        >>> K = tf.random.normal([2, 8, 4096, 64])
        >>> V = tf.random.normal([2, 8, 4096, 64])
        >>> output = layer(Q, K, V)
    """

    def __init__(
        self,
        block_size: int = 64,
        num_selected_blocks: int = 8,
        num_selected_tokens: int = 32,
        sliding_window_size: int = 128,
        num_global_tokens: int = 1,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.num_selected_blocks = num_selected_blocks
        self.num_selected_tokens = num_selected_tokens
        self.sliding_window_size = sliding_window_size
        self.num_global_tokens = num_global_tokens
        self.temperature = temperature

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply NSA.

        Args:
            query: [batch, num_heads, seq_q, head_dim]
            key: [batch, num_heads, seq_k, head_dim]
            value: [batch, num_heads, seq_k, head_dim]
            training: Whether in training mode

        Returns:
            Output [batch, num_heads, seq_q, head_dim]
        """
        return native_sparse_attention(
            query=query,
            key=key,
            value=value,
            block_size=self.block_size,
            num_selected_blocks=self.num_selected_blocks,
            num_selected_tokens=self.num_selected_tokens,
            sliding_window_size=self.sliding_window_size,
            num_global_tokens=self.num_global_tokens,
            temperature=self.temperature,
        )

    def compute_output_shape(self, input_shape):
        """Returns query shape as output shape."""
        return input_shape[0]

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update(
            {
                "block_size": self.block_size,
                "num_selected_blocks": self.num_selected_blocks,
                "num_selected_tokens": self.num_selected_tokens,
                "sliding_window_size": self.sliding_window_size,
                "num_global_tokens": self.num_global_tokens,
                "temperature": self.temperature,
            }
        )
        return config


def estimate_sparsity(
    seq_len: int,
    block_size: int = 64,
    num_selected_blocks: int = 8,
    num_selected_tokens: int = 32,
    sliding_window_size: int = 128,
) -> float:
    """Estimate sparsity ratio of NSA vs full attention.

    Args:
        seq_len: Sequence length
        block_size: Compression block size
        num_selected_blocks: Selected blocks
        num_selected_tokens: Tokens per block
        sliding_window_size: Local window size

    Returns:
        Sparsity ratio (attended tokens / total possible)
    """
    # Average tokens attended per query
    avg_local = min(sliding_window_size, seq_len)
    avg_selected = num_selected_blocks * num_selected_tokens
    avg_attended = avg_local + avg_selected

    # Full attention tokens
    full = seq_len

    return avg_attended / full
