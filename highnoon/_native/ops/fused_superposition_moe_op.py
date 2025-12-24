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

"""Python wrapper for the FusedSuperpositionMoe C++ operator.

NO PYTHON FALLBACK: This module requires the compiled .so to function.
"""

import logging

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
            logger.debug("Successfully loaded C++ FusedSuperpositionMoe from consolidated binary.")
        else:
            logger.warning(
                "C++ superposition MoE op loaded but symbols not found. Available: %s",
                [a for a in dir(_fused_superposition_moe_module) if not a.startswith("_")][:10],
            )
    else:
        logger.warning("Consolidated binary not available for superposition MoE op.")
except Exception as e:
    logger.warning(f"Could not load FusedSuperpositionMoe op: {e}")


# --- Custom Gradient Definition ---
@ops.RegisterGradient("FusedSuperpositionMoe")
def _fused_superposition_moe_grad(
    op: tf.Operation, grad_output: tf.Tensor
) -> tuple[tf.Tensor, ...]:
    """
    Defines the gradient for the FusedSuperpositionMoe operator.
    """
    # Unpack all 12 inputs, including the bias tensors.
    (
        tokens,
        context,
        ffn1_cores,
        ffn2_cores,
        collapse_q_weights,
        collapse_k_weights,
        collapse_v_weights,
        collapse_o_weights,
        collapse_q_bias,
        collapse_k_bias,
        collapse_v_bias,
        collapse_o_bias,
    ) = op.inputs

    input_dims = op.get_attr("input_dims")
    output_dims_ffn1 = op.get_attr("output_dims_ffn1")
    output_dims_ffn2 = op.get_attr("output_dims_ffn2")
    tt_ranks = op.get_attr("tt_ranks")
    superposition_dim = op.get_attr("superposition_dim")
    micro_batch_size = op.get_attr("micro_batch_size")

    def call_custom_grad_op():
        """Calls the custom C++ gradient kernel and ensures a tuple is returned."""
        grad_results = fused_superposition_moe_grad_op(
            grad_output=grad_output,
            tokens=tokens,
            context=context,
            ffn1_cores=ffn1_cores,
            ffn2_cores=ffn2_cores,
            collapse_q_weights=collapse_q_weights,
            collapse_k_weights=collapse_k_weights,
            collapse_v_weights=collapse_v_weights,
            collapse_o_weights=collapse_o_weights,
            collapse_q_bias=collapse_q_bias,
            collapse_k_bias=collapse_k_bias,
            collapse_v_bias=collapse_v_bias,
            collapse_o_bias=collapse_o_bias,
            input_dims=input_dims,
            output_dims_ffn1=output_dims_ffn1,
            output_dims_ffn2=output_dims_ffn2,
            tt_ranks=tt_ranks,
            superposition_dim=superposition_dim,
            micro_batch_size=micro_batch_size,
        )
        return tuple(grad_results)

    def return_zero_grads():
        """The safeguard: returns zero gradients with correct shapes for all 12 inputs."""
        return (
            tf.zeros_like(tokens),
            tf.zeros_like(context),
            tf.zeros_like(ffn1_cores),
            tf.zeros_like(ffn2_cores),
            tf.zeros_like(collapse_q_weights),
            tf.zeros_like(collapse_k_weights),
            tf.zeros_like(collapse_v_weights),
            tf.zeros_like(collapse_o_weights),
            tf.zeros_like(collapse_q_bias),
            tf.zeros_like(collapse_k_bias),
            tf.zeros_like(collapse_v_bias),
            tf.zeros_like(collapse_o_bias),
        )

    is_grad_zero = tf.equal(tf.reduce_sum(tf.abs(grad_output)), 0.0)

    grads = tf.cond(is_grad_zero, true_fn=return_zero_grads, false_fn=call_custom_grad_op)

    # Unpack all 12 gradient results.
    (
        grad_tokens,
        grad_context,
        grad_ffn1_cores,
        grad_ffn2_cores,
        grad_collapse_q_weights,
        grad_collapse_k_weights,
        grad_collapse_v_weights,
        grad_collapse_o_weights,
        grad_collapse_q_bias,
        grad_collapse_k_bias,
        grad_collapse_v_bias,
        grad_collapse_o_bias,
    ) = grads

    return (
        grad_tokens,
        grad_context,
        grad_ffn1_cores,
        grad_ffn2_cores,
        grad_collapse_q_weights,
        grad_collapse_k_weights,
        grad_collapse_v_weights,
        grad_collapse_o_weights,
        grad_collapse_q_bias,
        grad_collapse_k_bias,
        grad_collapse_v_bias,
        grad_collapse_o_bias,
    )


# --- Python Wrapper Function ---
def fused_superposition_moe(
    tokens: tf.Tensor,
    context: tf.Tensor,
    ffn1_cores: tf.Tensor,
    ffn2_cores: tf.Tensor,
    collapse_q_weights: tf.Tensor,
    collapse_k_weights: tf.Tensor,
    collapse_v_weights: tf.Tensor,
    collapse_o_weights: tf.Tensor,
    collapse_q_bias: tf.Tensor,
    collapse_k_bias: tf.Tensor,
    collapse_v_bias: tf.Tensor,
    collapse_o_bias: tf.Tensor,
    input_dims: list,
    output_dims_ffn1: list,
    output_dims_ffn2: list,
    tt_ranks: list,
    superposition_dim: int,
    micro_batch_size: int = 32,
) -> tf.Tensor:
    """
    Python wrapper for the FusedSuperpositionMoe custom C++ operator.

    Raises:
        RuntimeError: If the C++ operator could not be loaded.
    """
    if fused_superposition_moe_op is None:
        raise RuntimeError(
            "The FusedSuperpositionMoe custom operator could not be loaded. "
            "Please ensure the .so file is compiled and in the correct location. "
            "NO PYTHON FALLBACK IS PROVIDED."
        )

    output = fused_superposition_moe_op(
        tokens=tokens,
        context=context,
        ffn1_cores=ffn1_cores,
        ffn2_cores=ffn2_cores,
        collapse_q_weights=collapse_q_weights,
        collapse_k_weights=collapse_k_weights,
        collapse_v_weights=collapse_v_weights,
        collapse_o_weights=collapse_o_weights,
        collapse_q_bias=collapse_q_bias,
        collapse_k_bias=collapse_k_bias,
        collapse_v_bias=collapse_v_bias,
        collapse_o_bias=collapse_o_bias,
        input_dims=input_dims,
        output_dims_ffn1=output_dims_ffn1,
        output_dims_ffn2=output_dims_ffn2,
        tt_ranks=tt_ranks,
        superposition_dim=superposition_dim,
        micro_batch_size=micro_batch_size,
    )
    return output
