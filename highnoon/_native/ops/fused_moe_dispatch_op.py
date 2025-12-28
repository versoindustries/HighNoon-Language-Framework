# highnoon/_native/ops/fused_moe_dispatch_op.py
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

"""Python wrapper for the fused MoE dispatch C++ operator.

NO PYTHON FALLBACK: This module requires the compiled .so to function.
"""

import logging
import sys

import tensorflow as tf

# --- Setup ---
logger = logging.getLogger(__name__)

# --- Load Custom C++ Operator via consolidated binary ---
_fused_moe_dispatch_module = None
fused_moe_dispatch_op = None
fused_moe_dispatch_grad_op = None

try:
    # Try consolidated binary first via parent module
    from highnoon._native import get_op

    _fused_moe_dispatch_module = get_op("fused_moe_dispatch")

    if _fused_moe_dispatch_module is not None:
        # TensorFlow exposes ops with various naming conventions
        fused_moe_dispatch_op = getattr(_fused_moe_dispatch_module, "fused_mo_e_dispatch", None)
        fused_moe_dispatch_grad_op = getattr(
            _fused_moe_dispatch_module, "fused_mo_e_dispatch_grad", None
        )

        if fused_moe_dispatch_op is None:
            # Try alternate names
            fused_moe_dispatch_op = getattr(_fused_moe_dispatch_module, "fused_moe_dispatch", None)
            fused_moe_dispatch_grad_op = getattr(
                _fused_moe_dispatch_module, "fused_moe_dispatch_grad", None
            )

        if fused_moe_dispatch_op is None:
            # Last fallback to CamelCase names
            fused_moe_dispatch_op = getattr(_fused_moe_dispatch_module, "FusedMoEDispatch", None)
            fused_moe_dispatch_grad_op = getattr(
                _fused_moe_dispatch_module, "FusedMoEDispatchGrad", None
            )

        if fused_moe_dispatch_op is not None:
            logger.debug("Successfully loaded C++ MoE dispatch operator from consolidated binary.")
        else:
            logger.warning(
                "C++ MoE op loaded but symbols not found. Available: %s",
                [a for a in dir(_fused_moe_dispatch_module) if not a.startswith("_")][:10],
            )
    else:
        logger.warning("Consolidated binary not available for MoE dispatch op.")
except Exception as e:
    logger.warning(f"Could not load fused MoE dispatch op: {e}")


@tf.custom_gradient
def fused_moe_dispatch(
    tokens: tf.Tensor,
    router_logits: tf.Tensor,
    expert_capacity: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    """
    Python wrapper for the fused MoE dispatch custom operator.

    This function implements expert choice routing, where each expert selects its
    top-k tokens. It's designed to be a high-performance replacement for equivalent
    Python logic.

    Args:
        tokens: A [num_tokens, d_model] tensor of input token embeddings.
        router_logits: A [num_tokens, num_experts] tensor of scores for assigning
                       each token to each expert.
        expert_capacity: A scalar int32 tf.Tensor specifying the maximum number
                         of tokens each expert can process.

    Returns:
        A tuple containing:
        - dispatched_tokens: A [num_dispatched, d_model] tensor of the selected tokens.
        - dispatched_gates: A [num_dispatched] tensor of the router scores for the
                            selected tokens.
        - dispatch_metadata: A [num_dispatched] int32 tensor mapping each dispatched
                             token back to its original index in the input.
        - expert_boundaries: A [num_experts + 1] int32 tensor marking the start and
                             end indices for each expert's tokens in the dispatched tensors.
        - expert_indices: A [num_dispatched] int32 tensor indicating which expert
                          each dispatched token was assigned to.

    Raises:
        NotImplementedError: If the C++ operator could not be loaded.
    """
    if fused_moe_dispatch_op is None:
        raise NotImplementedError(
            "The C++ FusedMoEDispatch operator could not be loaded. "
            "Please check the build process and ensure the '.so' file is present and valid. "
            "NO PYTHON FALLBACK IS PROVIDED."
        )

    (
        dispatched_tokens,
        dispatched_gates,
        dispatch_metadata,
        expert_boundaries,
        expert_indices,
    ) = fused_moe_dispatch_op(
        tokens=tokens,
        router_logits=router_logits,
        expert_capacity=expert_capacity,
    )

    def grad_fn(
        grad_dispatched_tokens,
        grad_dispatched_gates,
        grad_dispatch_metadata,
        grad_expert_boundaries,
        grad_expert_indices,
        variables=None,
    ):
        """
        Gradient function that calls the custom C++ backward kernel.
        """
        if fused_moe_dispatch_grad_op is None:
            raise NotImplementedError("The C++ FusedMoEDispatchGrad operator could not be loaded.")
        if variables is not None:
            num_vars = len(variables)
            tf.print(
                "[DEBUG] FusedMoEDispatch grad_fn is being watched for",
                num_vars,
                "variables.",
                output_stream=sys.stderr,
            )

        grad_tokens, grad_router_logits = fused_moe_dispatch_grad_op(
            grad_dispatched_tokens=grad_dispatched_tokens,
            grad_dispatched_gates=grad_dispatched_gates,
            dispatch_metadata=dispatch_metadata,
            expert_indices=expert_indices,
            tokens=tokens,
            router_logits=router_logits,
        )

        input_grads = (
            grad_tokens,
            grad_router_logits,
            None,
        )  # Grad for tokens, router_logits, expert_capacity
        variable_grads_list = [None] * len(variables) if variables is not None else []
        return input_grads, variable_grads_list

    return (
        dispatched_tokens,
        dispatched_gates,
        dispatch_metadata,
        expert_boundaries,
        expert_indices,
    ), grad_fn


# --- V2 Dispatch with Routing Bias and Sigmoid Support ---
fused_moe_dispatch_v2_op = None
fused_moe_dispatch_v2_grad_op = None

try:
    if _fused_moe_dispatch_module is not None:
        fused_moe_dispatch_v2_op = getattr(
            _fused_moe_dispatch_module, "fused_mo_e_dispatch_v2", None
        )
        if fused_moe_dispatch_v2_op is None:
            fused_moe_dispatch_v2_op = getattr(
                _fused_moe_dispatch_module, "fused_moe_dispatch_v2", None
            )
        if fused_moe_dispatch_v2_op is None:
            fused_moe_dispatch_v2_op = getattr(
                _fused_moe_dispatch_module, "FusedMoEDispatchV2", None
            )
        if fused_moe_dispatch_v2_op is not None:
            logger.debug("Successfully loaded C++ MoE dispatch V2 operator.")
except Exception as e:
    logger.warning(f"Could not load fused MoE dispatch V2 op: {e}")


def fused_moe_dispatch_v2(
    tokens: tf.Tensor,
    router_logits: tf.Tensor,
    expert_capacity: tf.Tensor,
    routing_bias: tf.Tensor,
    use_sigmoid_routing: bool = False,
    apply_bias_before_topk: bool = True,
) -> tuple[tf.Tensor, ...]:
    """
    Enhanced MoE dispatch with routing bias and sigmoid gating support.

    Args:
        tokens: [num_tokens, d_model] tensor of input token embeddings.
        router_logits: [num_tokens, num_experts] tensor of router scores.
        expert_capacity: Scalar int32 specifying max tokens per expert.
        routing_bias: [num_experts] tensor of EMA-updated load-balancing bias.
        use_sigmoid_routing: Use GLM-4.5 style sigmoid gating instead of softmax.
        apply_bias_before_topk: Apply routing bias before top-k selection.

    Returns:
        Tuple of:
        - dispatched_tokens: [num_dispatched, d_model]
        - dispatched_gates: [num_dispatched]
        - dispatch_metadata: [num_dispatched] int32
        - expert_boundaries: [num_experts + 1] int32
        - expert_indices: [num_dispatched] int32
        - expert_loads: [num_experts] float32 (tokens per expert)
    """

    # Define internal custom gradient function that only takes tensors
    # The boolean flags are captured from the outer scope (closure)
    @tf.custom_gradient
    def _fused_moe_dispatch_v2_internal(tokens_in, logits_in, capacity_in, bias_in):
        if fused_moe_dispatch_v2_op is None:
            raise NotImplementedError(
                "The C++ FusedMoEDispatchV2 operator could not be loaded. "
                "Please check the build process and ensure the '.so' file is present and valid. "
                "NO FALLBACK IS PROVIDED - V2 dispatch is required."
            )

        # Call C++ Op with captured attributes
        (d_tok, d_gate, d_meta, ex_bound, ex_idx, ex_loads) = fused_moe_dispatch_v2_op(
            tokens=tokens_in,
            router_logits=logits_in,
            expert_capacity=capacity_in,
            routing_bias=bias_in,
            use_sigmoid_routing=use_sigmoid_routing,
            apply_bias_before_topk=apply_bias_before_topk,
        )

        def grad_fn_v2(
            grad_d_tok,
            grad_d_gate,
            grad_d_meta,
            grad_ex_bound,
            grad_ex_idx,
            grad_ex_load,
            variables=None,
        ):
            if fused_moe_dispatch_grad_op is None:
                raise NotImplementedError("C++ FusedMoEDispatchGrad op missing")

            grad_tok, grad_logits = fused_moe_dispatch_grad_op(
                grad_dispatched_tokens=grad_d_tok,
                grad_dispatched_gates=grad_d_gate,
                dispatch_metadata=d_meta,
                expert_indices=ex_idx,
                tokens=tokens_in,
                router_logits=logits_in,
            )
            # Grads for: tokens, logits, capacity, bias
            return (grad_tok, grad_logits, None, None), (
                [None] * len(variables) if variables else []
            )

        return (d_tok, d_gate, d_meta, ex_bound, ex_idx, ex_loads), grad_fn_v2

    # Call the internal function
    return _fused_moe_dispatch_v2_internal(tokens, router_logits, expert_capacity, routing_bias)
