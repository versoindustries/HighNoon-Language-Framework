# highnoon/_native/ops/fused_latent_reasoning_op.py
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

"""Python wrapper for fused Latent Reasoning Block C++ op.

This module provides a Python interface to the C++ FusedLatentReasoning kernel
with automatic gradient support via tf.custom_gradient.
"""

from __future__ import annotations

import tensorflow as tf

from highnoon import config as hn_config
from highnoon._native import get_op

# Load the C++ op library
_lib = get_op("fused_latent_reasoning")
_fused_latent_reasoning_op = getattr(_lib, "FusedLatentReasoning", None) if _lib else None
_fused_latent_reasoning_grad_op = getattr(_lib, "FusedLatentReasoningGrad", None) if _lib else None


def fused_latent_reasoning(
    x: tf.Tensor,
    thought_norm_gamma: tf.Tensor,
    thought_norm_beta: tf.Tensor,
    thought_up_weight: tf.Tensor,
    thought_up_bias: tf.Tensor,
    thought_down_weight: tf.Tensor,
    thought_down_bias: tf.Tensor,
    output_norm_gamma: tf.Tensor,
    output_norm_beta: tf.Tensor,
    num_thought_steps: int = 4,
    use_entropy_guidance: bool = True,
    uncertainty_threshold: float = 0.5,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Fused Latent Reasoning Block.

    C++-accelerated latent reasoning that fuses multi-step thought refinement,
    LayerNorm, GELU FFN, and entropy-guided masking.

    Args:
        x: Input tensor [batch, seq_len, embed_dim].
        thought_norm_gamma: LayerNorm scale [embed_dim].
        thought_norm_beta: LayerNorm bias [embed_dim].
        thought_up_weight: Up projection [embed_dim, d_inner].
        thought_up_bias: Up projection bias [d_inner].
        thought_down_weight: Down projection [d_inner, embed_dim].
        thought_down_bias: Down projection bias [embed_dim].
        output_norm_gamma: Output LayerNorm scale [embed_dim].
        output_norm_beta: Output LayerNorm bias [embed_dim].
        num_thought_steps: Number of refinement iterations.
        use_entropy_guidance: Enable uncertainty-based masking.
        uncertainty_threshold: Threshold for uncertainty masking.

    Returns:
        Tuple of (output, halt_prob):
        - output: Refined hidden states [batch, seq_len, embed_dim]
        - halt_prob: Final halting probability [batch, 1]

    Raises:
        RuntimeError: If C++ op library is not available.
    """
    if _fused_latent_reasoning_op is None:
        raise RuntimeError(
            "FusedLatentReasoning C++ op not available. Build with: "
            "cd highnoon/_native && ./build_ops.sh fused_latent_reasoning"
        )

    # Ensure float32
    x = tf.cast(x, tf.float32)
    thought_norm_gamma = tf.cast(thought_norm_gamma, tf.float32)
    thought_norm_beta = tf.cast(thought_norm_beta, tf.float32)
    thought_up_weight = tf.cast(thought_up_weight, tf.float32)
    thought_up_bias = tf.cast(thought_up_bias, tf.float32)
    thought_down_weight = tf.cast(thought_down_weight, tf.float32)
    thought_down_bias = tf.cast(thought_down_bias, tf.float32)
    output_norm_gamma = tf.cast(output_norm_gamma, tf.float32)
    output_norm_beta = tf.cast(output_norm_beta, tf.float32)

    @tf.custom_gradient
    def _fused_latent_reasoning_inner(x_in, tng, tnb, tuw, tub, tdw, tdb, ong, onb):
        """Inner function with tensor-only signature."""
        streaming_chunk_size = (
            hn_config.STREAMING_CHUNK_SIZE if getattr(hn_config, "STREAMING_ENABLED", True) else 0
        )
        output, halt_prob = _fused_latent_reasoning_op(
            x=x_in,
            thought_norm_gamma=tng,
            thought_norm_beta=tnb,
            thought_up_weight=tuw,
            thought_up_bias=tub,
            thought_down_weight=tdw,
            thought_down_bias=tdb,
            output_norm_gamma=ong,
            output_norm_beta=onb,
            num_thought_steps=num_thought_steps,
            use_entropy_guidance=use_entropy_guidance,
            uncertainty_threshold=uncertainty_threshold,
            streaming_chunk_size=streaming_chunk_size,
        )

        def grad(grad_output, grad_halt_prob):
            """Compute gradients."""
            if _fused_latent_reasoning_grad_op is not None:
                grads = _fused_latent_reasoning_grad_op(
                    grad_output=grad_output,
                    x=x_in,
                    thought_norm_gamma=tng,
                    thought_up_weight=tuw,
                    thought_down_weight=tdw,
                    num_thought_steps=num_thought_steps,
                    streaming_chunk_size=streaming_chunk_size,
                )
                grad_x, grad_gamma, grad_beta, grad_uw, grad_ub, grad_dw, grad_db = grads
                # Output norms get zero gradients for now
                return (
                    grad_x,
                    grad_gamma,
                    grad_beta,
                    grad_uw,
                    grad_ub,
                    grad_dw,
                    grad_db,
                    tf.zeros_like(ong),
                    tf.zeros_like(onb),
                )
            else:
                return (
                    tf.zeros_like(x_in),
                    tf.zeros_like(tng),
                    tf.zeros_like(tnb),
                    tf.zeros_like(tuw),
                    tf.zeros_like(tub),
                    tf.zeros_like(tdw),
                    tf.zeros_like(tdb),
                    tf.zeros_like(ong),
                    tf.zeros_like(onb),
                )

        return (output, halt_prob), grad

    return _fused_latent_reasoning_inner(
        x,
        thought_norm_gamma,
        thought_norm_beta,
        thought_up_weight,
        thought_up_bias,
        thought_down_weight,
        thought_down_bias,
        output_norm_gamma,
        output_norm_beta,
    )


__all__ = ["fused_latent_reasoning"]
