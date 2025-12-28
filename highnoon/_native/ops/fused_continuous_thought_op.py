# highnoon/_native/ops/fused_continuous_thought_op.py
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

"""Python wrapper for fused Continuous Thought Block C++ op.

This module provides a Python interface to the C++ FusedContinuousThought
kernel with automatic gradient support via tf.custom_gradient.

The ContinuousThoughtBlock implements COCONUT-style continuous thought
reasoning in latent space without generating intermediate tokens.
"""

from __future__ import annotations

import logging
import tensorflow as tf

from highnoon._native import get_op

logger = logging.getLogger(__name__)

# Load the C++ op library
_lib = get_op("fused_continuous_thought")
_fused_continuous_thought_op = getattr(_lib, "FusedContinuousThought", None) if _lib else None
_fused_continuous_thought_grad_op = (
    getattr(_lib, "FusedContinuousThoughtGrad", None) if _lib else None
)



def fused_continuous_thought_available() -> bool:
    """Check if the fused continuous thought op is available."""
    return _fused_continuous_thought_op is not None


def fused_continuous_thought(
    x: tf.Tensor,
    input_norm_gamma: tf.Tensor,
    input_norm_beta: tf.Tensor,
    aggregator_weight: tf.Tensor,
    aggregator_bias: tf.Tensor,
    projector_norm_gamma: tf.Tensor,
    projector_norm_beta: tf.Tensor,
    projector_dense1_weight: tf.Tensor,
    projector_dense1_bias: tf.Tensor,
    projector_dense2_weight: tf.Tensor,
    projector_dense2_bias: tf.Tensor,
    broadcast_weight: tf.Tensor,
    broadcast_bias: tf.Tensor,
    gate_weight: tf.Tensor,
    gate_bias: tf.Tensor,
    output_norm_gamma: tf.Tensor,
    output_norm_beta: tf.Tensor,
    num_thought_steps: int = 4,
    use_gating: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Fused Continuous Thought Block.

    C++-accelerated COCONUT-style continuous thought reasoning that fuses:
    - Input layer normalization
    - Mean pooling for thought extraction
    - Aggregator projection
    - Iterative thought refinement (LayerNorm + GELU MLP + residual)
    - Broadcast projection back to sequence
    - Gated residual connection
    - Output layer normalization

    Args:
        x: Input tensor [batch, seq_len, embed_dim].
        input_norm_gamma: Input LayerNorm scale [embed_dim].
        input_norm_beta: Input LayerNorm bias [embed_dim].
        aggregator_weight: Thought aggregator weight [embed_dim, embed_dim].
        aggregator_bias: Aggregator bias [embed_dim].
        projector_norm_gamma: Projector LayerNorm scale [embed_dim].
        projector_norm_beta: Projector LayerNorm bias [embed_dim].
        projector_dense1_weight: Up projection weight [embed_dim, hidden_dim].
        projector_dense1_bias: Up projection bias [hidden_dim].
        projector_dense2_weight: Down projection weight [hidden_dim, embed_dim].
        projector_dense2_bias: Down projection bias [embed_dim].
        broadcast_weight: Broadcast projection weight [embed_dim, embed_dim].
        broadcast_bias: Broadcast projection bias [embed_dim].
        gate_weight: Gate projection weight [embed_dim, embed_dim].
        gate_bias: Gate projection bias [embed_dim].
        output_norm_gamma: Output LayerNorm scale [embed_dim].
        output_norm_beta: Output LayerNorm bias [embed_dim].
        num_thought_steps: Number of thought refinement iterations.
        use_gating: Whether to use gated residual connection.

    Returns:
        Tuple of (output, thought_state):
        - output: Enhanced hidden states [batch, seq_len, embed_dim]
        - thought_state: Final thought state [batch, embed_dim]

    Raises:
        RuntimeError: If C++ op library is not available.
    """
    if _fused_continuous_thought_op is None:
        raise RuntimeError(
            "FusedContinuousThought C++ op not available. Build with: "
            "cd highnoon/_native && ./build_ops.sh fused_continuous_thought"
        )

    # Ensure float32
    x = tf.cast(x, tf.float32)
    input_norm_gamma = tf.cast(input_norm_gamma, tf.float32)
    input_norm_beta = tf.cast(input_norm_beta, tf.float32)
    aggregator_weight = tf.cast(aggregator_weight, tf.float32)
    aggregator_bias = tf.cast(aggregator_bias, tf.float32)
    projector_norm_gamma = tf.cast(projector_norm_gamma, tf.float32)
    projector_norm_beta = tf.cast(projector_norm_beta, tf.float32)
    projector_dense1_weight = tf.cast(projector_dense1_weight, tf.float32)
    projector_dense1_bias = tf.cast(projector_dense1_bias, tf.float32)
    projector_dense2_weight = tf.cast(projector_dense2_weight, tf.float32)
    projector_dense2_bias = tf.cast(projector_dense2_bias, tf.float32)
    broadcast_weight = tf.cast(broadcast_weight, tf.float32)
    broadcast_bias = tf.cast(broadcast_bias, tf.float32)
    gate_weight = tf.cast(gate_weight, tf.float32)
    gate_bias = tf.cast(gate_bias, tf.float32)
    output_norm_gamma = tf.cast(output_norm_gamma, tf.float32)
    output_norm_beta = tf.cast(output_norm_beta, tf.float32)

    @tf.custom_gradient
    def _fused_continuous_thought_inner(
        x_in,
        ing,
        inb,
        agg_w,
        agg_b,
        png,
        pnb,
        pd1_w,
        pd1_b,
        pd2_w,
        pd2_b,
        bc_w,
        bc_b,
        g_w,
        g_b,
        ong,
        onb,
    ):
        """Inner function with tensor-only signature."""
        output, thought_state = _fused_continuous_thought_op(
            x=x_in,
            input_norm_gamma=ing,
            input_norm_beta=inb,
            aggregator_weight=agg_w,
            aggregator_bias=agg_b,
            projector_norm_gamma=png,
            projector_norm_beta=pnb,
            projector_dense1_weight=pd1_w,
            projector_dense1_bias=pd1_b,
            projector_dense2_weight=pd2_w,
            projector_dense2_bias=pd2_b,
            broadcast_weight=bc_w,
            broadcast_bias=bc_b,
            gate_weight=g_w,
            gate_bias=g_b,
            output_norm_gamma=ong,
            output_norm_beta=onb,
            num_thought_steps=num_thought_steps,
            use_gating=use_gating,
        )

        def grad(grad_output, grad_thought_state):
            """Compute gradients using C++ grad op."""
            if _fused_continuous_thought_grad_op is None:
                raise RuntimeError(
                    "FusedContinuousThoughtGrad C++ op not available. Build with: "
                    "cd highnoon/_native && ./build_ops.sh fused_continuous_thought"
                )

            grads = _fused_continuous_thought_grad_op(
                grad_output=grad_output,
                grad_thought_state=grad_thought_state,
                x=x_in,
                input_norm_gamma=ing,
                aggregator_weight=agg_w,
                projector_dense1_weight=pd1_w,
                projector_dense2_weight=pd2_w,
                broadcast_weight=bc_w,
                gate_weight=g_w,
                num_thought_steps=num_thought_steps,
                use_gating=use_gating,
            )
            return grads

        return (output, thought_state), grad

    return _fused_continuous_thought_inner(
        x,
        input_norm_gamma,
        input_norm_beta,
        aggregator_weight,
        aggregator_bias,
        projector_norm_gamma,
        projector_norm_beta,
        projector_dense1_weight,
        projector_dense1_bias,
        projector_dense2_weight,
        projector_dense2_bias,
        broadcast_weight,
        broadcast_bias,
        gate_weight,
        gate_bias,
        output_norm_gamma,
        output_norm_beta,
    )


__all__ = ["fused_continuous_thought", "fused_continuous_thought_available"]
