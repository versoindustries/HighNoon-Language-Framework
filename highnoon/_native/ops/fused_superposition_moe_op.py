# highnoon/_native/ops/fused_superposition_moe_op.py
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

"""Python wrapper for the Unified HD-SuperposedExpert C++ operator (v2.0).

This module provides the Python interface to the holographic routing-based
superposition MoE kernel. NO PYTHON FALLBACK is provided.

Key changes from v1.0:
- Removed context input (self-routing only)
- Replaced Q/K/V/O attention collapse with path_bases/path_weights
- Added optional HD projection layer support
- Returns routing_weights for visualization
"""

import logging
from typing import NamedTuple

import tensorflow as tf
from tensorflow.python.framework import ops

# --- Setup ---
logger = logging.getLogger(__name__)

# --- Load the Custom C++ Operator via consolidated binary ---
_fused_superposition_moe_module = None
fused_superposition_moe_op = None
fused_superposition_moe_grad_op = None

try:
    # Try consolidated binary first via parent module
    from highnoon._native import get_op

    _fused_superposition_moe_module = get_op("fused_superposition_moe")

    if _fused_superposition_moe_module is not None:
        # Try TensorFlow's snake_case naming convention
        fused_superposition_moe_op = getattr(
            _fused_superposition_moe_module, "fused_superposition_moe", None
        )
        fused_superposition_moe_grad_op = getattr(
            _fused_superposition_moe_module, "fused_superposition_moe_grad", None
        )

        if fused_superposition_moe_op is None:
            # Fallback to CamelCase
            fused_superposition_moe_op = getattr(
                _fused_superposition_moe_module, "FusedSuperpositionMoe", None
            )
            fused_superposition_moe_grad_op = getattr(
                _fused_superposition_moe_module, "FusedSuperpositionMoeGrad", None
            )

        if fused_superposition_moe_op is not None:
            logger.debug("Successfully loaded C++ UnifiedHDSuperposedExpert v2.0.")
        else:
            logger.warning(
                "C++ superposition MoE op loaded but symbols not found. Available: %s",
                [a for a in dir(_fused_superposition_moe_module) if not a.startswith("_")][:10],
            )
    else:
        logger.warning("Consolidated binary not available for superposition MoE op.")
except Exception as e:
    logger.warning(f"Could not load FusedSuperpositionMoe op: {e}")


class SuperpositionMoeOutput(NamedTuple):
    """Output from fused_superposition_moe."""

    output: tf.Tensor  # [batch, d_model]
    routing_weights: tf.Tensor  # [batch, K]


# --- Custom Gradient Definition ---
@ops.RegisterGradient("FusedSuperpositionMoe")
def _fused_superposition_moe_grad(
    op: tf.Operation, grad_output: tf.Tensor, grad_routing_weights: tf.Tensor
) -> tuple[tf.Tensor, ...]:
    """
    Defines the gradient for the UnifiedHDSuperposedExpert operator (v2.0).

    Args:
        op: The forward op.
        grad_output: Gradient w.r.t. output tensor.
        grad_routing_weights: Gradient w.r.t. routing_weights (usually unused).

    Returns:
        Gradients for all 7 inputs.
    """
    # Unpack inputs: tokens, ffn1_cores, ffn2_cores, path_bases, path_weights,
    #                hd_input_proj, hd_output_proj
    (
        tokens,
        ffn1_cores,
        ffn2_cores,
        path_bases,
        path_weights,
        hd_input_proj,
        hd_output_proj,
    ) = op.inputs

    # Get the routing_weights output for reuse in backward
    routing_weights = op.outputs[1]

    input_dims = op.get_attr("input_dims")
    output_dims_ffn1 = op.get_attr("output_dims_ffn1")
    output_dims_ffn2 = op.get_attr("output_dims_ffn2")
    tt_ranks = op.get_attr("tt_ranks")
    superposition_dim = op.get_attr("superposition_dim")
    micro_batch_size = op.get_attr("micro_batch_size")
    hd_dim = op.get_attr("hd_dim")
    use_hd_projection = op.get_attr("use_hd_projection")
    routing_temperature = op.get_attr("routing_temperature")

    def call_custom_grad_op():
        """Calls the custom C++ gradient kernel."""
        grad_results = fused_superposition_moe_grad_op(
            grad_output=grad_output,
            tokens=tokens,
            ffn1_cores=ffn1_cores,
            ffn2_cores=ffn2_cores,
            path_bases=path_bases,
            path_weights=path_weights,
            hd_input_proj=hd_input_proj,
            hd_output_proj=hd_output_proj,
            routing_weights=routing_weights,
            input_dims=input_dims,
            output_dims_ffn1=output_dims_ffn1,
            output_dims_ffn2=output_dims_ffn2,
            tt_ranks=tt_ranks,
            superposition_dim=superposition_dim,
            micro_batch_size=micro_batch_size,
            hd_dim=hd_dim,
            use_hd_projection=use_hd_projection,
            routing_temperature=routing_temperature,
        )
        return tuple(grad_results)

    def return_zero_grads():
        """The safeguard: returns zero gradients with correct shapes for all 7 inputs."""
        return (
            tf.zeros_like(tokens),
            tf.zeros_like(ffn1_cores),
            tf.zeros_like(ffn2_cores),
            tf.zeros_like(path_bases),
            tf.zeros_like(path_weights),
            tf.zeros_like(hd_input_proj),
            tf.zeros_like(hd_output_proj),
        )

    is_grad_zero = tf.equal(tf.reduce_sum(tf.abs(grad_output)), 0.0)

    grads = tf.cond(is_grad_zero, true_fn=return_zero_grads, false_fn=call_custom_grad_op)

    return grads


# --- Python Wrapper Function ---
def fused_superposition_moe(
    tokens: tf.Tensor,
    ffn1_cores: tf.Tensor,
    ffn2_cores: tf.Tensor,
    path_bases: tf.Tensor,
    path_weights: tf.Tensor,
    hd_input_proj: tf.Tensor,
    hd_output_proj: tf.Tensor,
    input_dims: list[int],
    output_dims_ffn1: list[int],
    output_dims_ffn2: list[int],
    tt_ranks: list[int],
    superposition_dim: int,
    micro_batch_size: int = 32,
    hd_dim: int = 4096,
    use_hd_projection: bool = False,
    routing_temperature: float = 1.0,
) -> SuperpositionMoeOutput:
    """
    Python wrapper for the Unified HD-SuperposedExpert C++ operator (v2.0).

    Uses holographic circular correlation routing instead of attention-based collapse.

    Args:
        tokens: Input tokens [batch, d_model].
        ffn1_cores: TT cores for first FFN layer (flattened).
        ffn2_cores: TT cores for second FFN layer (flattened).
        path_bases: Holographic routing bases [K, d_model].
        path_weights: Transformation weights per path [K, d_model].
        hd_input_proj: HD projection matrix in [d_model, hd_dim] (or empty).
        hd_output_proj: HD projection matrix out [hd_dim, d_model] (or empty).
        input_dims: TT input dimensions.
        output_dims_ffn1: TT output dimensions for FFN1.
        output_dims_ffn2: TT output dimensions for FFN2.
        tt_ranks: TT ranks.
        superposition_dim: Number of superposition paths K.
        micro_batch_size: Micro-batch size for memory efficiency.
        hd_dim: HD projection dimension.
        use_hd_projection: Whether to use HD projection.
        routing_temperature: Softmax temperature for path routing.

    Returns:
        SuperpositionMoeOutput with output tensor and routing_weights.

    Raises:
        RuntimeError: If the C++ operator could not be loaded.
    """
    if fused_superposition_moe_op is None:
        raise RuntimeError(
            "The UnifiedHDSuperposedExpert C++ operator could not be loaded. "
            "Please ensure the .so file is compiled. NO PYTHON FALLBACK PROVIDED."
        )

    output, routing_weights = fused_superposition_moe_op(
        tokens=tokens,
        ffn1_cores=ffn1_cores,
        ffn2_cores=ffn2_cores,
        path_bases=path_bases,
        path_weights=path_weights,
        hd_input_proj=hd_input_proj,
        hd_output_proj=hd_output_proj,
        input_dims=input_dims,
        output_dims_ffn1=output_dims_ffn1,
        output_dims_ffn2=output_dims_ffn2,
        tt_ranks=tt_ranks,
        superposition_dim=superposition_dim,
        micro_batch_size=micro_batch_size,
        hd_dim=hd_dim,
        use_hd_projection=use_hd_projection,
        routing_temperature=routing_temperature,
    )
    return SuperpositionMoeOutput(output=output, routing_weights=routing_weights)
