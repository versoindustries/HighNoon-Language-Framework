# highnoon/_native/fused_differential_attention_op.py
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

"""Python wrapper for Fused Differential Attention operation.

Implements the ICLR 2025 Differential Transformer attention mechanism:
    DiffAttn(Q, K, V) = (softmax(Q₁K₁ᵀ) - λ·softmax(Q₂K₂ᵀ)) · V

This reduces attention noise by ~65% and improves long-context handling.
"""


import tensorflow as tf

# Load the custom op library
from highnoon._native import _ops


def fused_differential_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    lambda_params: tf.Tensor,
    mask: tf.Tensor | None = None,
    normalize_output: bool = True,
    lambda_init: float = 0.8,
    name: str | None = None,
) -> tf.Tensor:
    """Fused Differential Attention operation.

    Computes attention as the difference of two softmax attention maps,
    which cancels noise and amplifies relevant patterns.

    Args:
        query: Query tensor [batch, num_heads, seq_q, head_dim].
            head_dim must be even as Q is split into Q1, Q2.
        key: Key tensor [batch, num_heads, seq_k, head_dim].
        value: Value tensor [batch, num_heads, seq_k, head_dim].
        lambda_params: Per-head differential scaling factors [num_heads].
            Learnable parameter, typically initialized to 0.8.
        mask: Optional attention mask [batch, seq_q, seq_k].
            Positive values mean attend, negative means masked.
        normalize_output: Whether to apply groupnorm-style output normalization.
        lambda_init: Initial lambda value used for output scaling.
        name: Optional operation name.

    Returns:
        Output tensor [batch, num_heads, seq_q, head_dim].

    Examples:
        >>> batch, heads, seq, dim = 2, 8, 128, 64
        >>> Q = tf.random.normal([batch, heads, seq, dim])
        >>> K = tf.random.normal([batch, heads, seq, dim])
        >>> V = tf.random.normal([batch, heads, seq, dim])
        >>> lambdas = tf.Variable(tf.ones([heads]) * 0.8)
        >>> output = fused_differential_attention(Q, K, V, lambdas)
        >>> assert output.shape == [batch, heads, seq, dim]
    """
    with tf.name_scope(name or "FusedDifferentialAttention"):
        # Ensure float32
        query = tf.cast(query, tf.float32)
        key = tf.cast(key, tf.float32)
        value = tf.cast(value, tf.float32)
        lambda_params = tf.cast(lambda_params, tf.float32)

        # Handle optional mask
        if mask is None:
            mask = tf.constant([], dtype=tf.float32)
        else:
            mask = tf.cast(mask, tf.float32)

        return _ops.fused_differential_attention(
            query=query,
            key=key,
            value=value,
            lambda_params=lambda_params,
            mask=mask,
            normalize_output=normalize_output,
            lambda_init=lambda_init,
        )


@tf.RegisterGradient("FusedDifferentialAttention")
def _fused_differential_attention_grad(op, grad_output):
    """Gradient for FusedDifferentialAttention."""
    query = op.inputs[0]
    key = op.inputs[1]
    value = op.inputs[2]
    lambda_params = op.inputs[3]
    mask = op.inputs[4]

    normalize_output = op.get_attr("normalize_output")
    lambda_init = op.get_attr("lambda_init")

    grad_query, grad_key, grad_value, grad_lambda = _ops.fused_differential_attention_grad(
        grad_output=grad_output,
        query=query,
        key=key,
        value=value,
        lambda_params=lambda_params,
        mask=mask,
        normalize_output=normalize_output,
        lambda_init=lambda_init,
    )

    # Mask gradient is not computed (non-differentiable)
    return [grad_query, grad_key, grad_value, grad_lambda, None]


class DifferentialAttention(tf.keras.layers.Layer):
    """Keras layer for Differential Attention.

    This layer wraps the fused differential attention operation with
    learnable lambda parameters per head.

    Args:
        num_heads: Number of attention heads.
        head_dim: Dimension per head (must be even).
        lambda_init: Initial value for learnable lambda parameters.
        normalize_output: Whether to normalize output.

    Example:
        >>> layer = DifferentialAttention(num_heads=8, head_dim=64)
        >>> Q = tf.random.normal([2, 8, 128, 64])
        >>> K = tf.random.normal([2, 8, 128, 64])
        >>> V = tf.random.normal([2, 8, 128, 64])
        >>> output = layer(Q, K, V)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        lambda_init: float = 0.8,
        normalize_output: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lambda_init = lambda_init
        self.normalize_output = normalize_output

    def build(self, input_shape):
        """Create the learnable lambda parameters."""
        self.lambda_params = self.add_weight(
            name="lambda_params",
            shape=(self.num_heads,),
            initializer=tf.keras.initializers.Constant(self.lambda_init),
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply differential attention.

        Args:
            query: [batch, num_heads, seq_q, head_dim]
            key: [batch, num_heads, seq_k, head_dim]
            value: [batch, num_heads, seq_k, head_dim]
            mask: Optional [batch, seq_q, seq_k]
            training: Whether in training mode (unused).

        Returns:
            Output tensor [batch, num_heads, seq_q, head_dim].
        """
        return fused_differential_attention(
            query=query,
            key=key,
            value=value,
            lambda_params=self.lambda_params,
            mask=mask,
            normalize_output=self.normalize_output,
            lambda_init=self.lambda_init,
        )

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "lambda_init": self.lambda_init,
                "normalize_output": self.normalize_output,
            }
        )
        return config
