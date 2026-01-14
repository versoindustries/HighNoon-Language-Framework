# highnoon/_native/ops/fused_collapse_op.py
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

"""Python wrapper for the Fused Collapse custom C++ operation.

Provides a high-performance collapse mechanism for superposition states using
multi-head cross-attention with Gumbel-Softmax sampling.

Phase 16: Contextual Gating Collapse Enhancement.

This op loads from the consolidated `_highnoon_core.so` binary built by
`./build_secure.sh`. For development testing only, individual ops can be
built with `./build_ops.sh fused_collapse`.
"""

import logging

import tensorflow as tf

from highnoon._native import get_op

logger = logging.getLogger(__name__)

# --- Load the Custom Operator from Consolidated Binary ---
_fused_collapse_module = None
fused_collapse_op = None
fused_collapse_grad_op = None


def _load_fused_collapse_op():
    """Load the fused collapse operation from consolidated binary."""
    global _fused_collapse_module, fused_collapse_op, fused_collapse_grad_op

    # Check if already registered in tf.raw_ops (from previous load)
    if "FusedCollapse" in dir(tf.raw_ops):
        fused_collapse_op = tf.raw_ops.FusedCollapse
        fused_collapse_grad_op = tf.raw_ops.FusedCollapseGrad
        return

    # Load from consolidated binary via _native loader
    try:
        _fused_collapse_module = get_op("fused_collapse")
        if _fused_collapse_module is not None:
            # Try lowercase first (TensorFlow snake_case)
            fused_collapse_op = getattr(_fused_collapse_module, "fused_collapse", None)
            fused_collapse_grad_op = getattr(_fused_collapse_module, "fused_collapse_grad", None)

            # Try PascalCase if snake_case not found
            if fused_collapse_op is None:
                fused_collapse_op = getattr(_fused_collapse_module, "FusedCollapse", None)
                fused_collapse_grad_op = getattr(_fused_collapse_module, "FusedCollapseGrad", None)

            if fused_collapse_op is not None:
                logger.info("Successfully loaded FusedCollapse from _highnoon_core.so")
            else:
                logger.warning(
                    "FusedCollapse op not found in module. Available: %s",
                    [a for a in dir(_fused_collapse_module) if not a.startswith("_")][:10],
                )
        else:
            logger.error("FusedCollapse op not available. Please compile with: ./build_secure.sh")
    except (AttributeError, Exception) as e:
        logger.error(
            f"Could not load FusedCollapse from consolidated binary. "
            f"Please rebuild with: ./build_secure.sh. Error: {e}"
        )
        fused_collapse_op = None
        fused_collapse_grad_op = None


_load_fused_collapse_op()


def fused_collapse_available() -> bool:
    """Check if the fused collapse operation is available."""
    return fused_collapse_op is not None


@tf.custom_gradient
def fused_collapse(
    context: tf.Tensor,
    superposed: tf.Tensor,
    q_weights: tf.Tensor,
    k_weights: tf.Tensor,
    v_weights: tf.Tensor,
    o_weights: tf.Tensor,
    q_bias: tf.Tensor,
    k_bias: tf.Tensor,
    v_bias: tf.Tensor,
    o_bias: tf.Tensor,
    num_heads: int = 4,
    temperature: float = 1.0,
    training: bool = True,
    use_kernel_attention: bool = False,
    feature_map: int = 0,
) -> tuple[tf.Tensor, callable]:
    """Python wrapper for the custom FusedCollapse operator.

    Implements multi-head cross-attention collapse for superposition states
    with Gumbel-Softmax unified training/inference.

    Args:
        context: Context tensor [batch, d_in] for query generation.
        superposed: Superposed states [batch, superposition_dim, d_out].
        q_weights: Query projection weights [d_in, d_out].
        k_weights: Key projection weights [d_out, d_out].
        v_weights: Value projection weights [d_out, d_out].
        o_weights: Output projection weights [d_out, d_out].
        q_bias: Query bias [d_out].
        k_bias: Key bias [d_out].
        v_bias: Value bias [d_out].
        o_bias: Output bias [d_out].
        num_heads: Number of attention heads.
        temperature: Gumbel-Softmax temperature (lower = sharper).
        training: Whether in training mode.
        use_kernel_attention: Use kernel attention feature map.
        feature_map: Feature map type (0=softmax, 1=elu+1, 2=relu²).

    Returns:
        Tuple of (output tensor [batch, d_out], gradient function).

    Raises:
        NotImplementedError: If the C++ operation is not available.
    """
    if fused_collapse_op is None:
        raise NotImplementedError(
            "The C++ fused_collapse operator could not be loaded. "
            "Please compile with: ./build_secure.sh"
        )

    output, attention_cache = fused_collapse_op(
        context=context,
        superposed=superposed,
        q_weights=q_weights,
        k_weights=k_weights,
        v_weights=v_weights,
        o_weights=o_weights,
        q_bias=q_bias,
        k_bias=k_bias,
        v_bias=v_bias,
        o_bias=o_bias,
        num_heads=num_heads,
        temperature=temperature,
        training=training,
        use_kernel_attention=use_kernel_attention,
        feature_map=feature_map,
    )

    def grad_fn(dy: tf.Tensor):
        """Gradient function for fused collapse."""
        if fused_collapse_grad_op is None:
            raise NotImplementedError("The C++ fused_collapse_grad operator could not be loaded.")

        (
            grad_context,
            grad_superposed,
            grad_q_weights,
            grad_k_weights,
            grad_v_weights,
            grad_o_weights,
            grad_q_bias,
            grad_k_bias,
            grad_v_bias,
            grad_o_bias,
        ) = fused_collapse_grad_op(
            grad_output=dy,
            context=context,
            superposed=superposed,
            attention_cache=attention_cache,
            q_weights=q_weights,
            k_weights=k_weights,
            v_weights=v_weights,
            o_weights=o_weights,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            num_heads=num_heads,
        )

        # Return gradients for all inputs in order
        # (context, superposed, q_weights, k_weights, v_weights, o_weights,
        #  q_bias, k_bias, v_bias, o_bias)
        # Non-differentiable params (num_heads, temperature, etc.) get None
        return (
            grad_context,
            grad_superposed,
            grad_q_weights,
            grad_k_weights,
            grad_v_weights,
            grad_o_weights,
            grad_q_bias,
            grad_k_bias,
            grad_v_bias,
            grad_o_bias,
        )

    return output, grad_fn


def gumbel_softmax_tf(
    logits: tf.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
) -> tf.Tensor:
    """TensorFlow implementation of Gumbel-Softmax for fallback.

    Args:
        logits: Input logits [batch, num_classes].
        temperature: Softmax temperature (lower = sharper).
        hard: If True, return one-hot with straight-through gradient.

    Returns:
        Soft or hard samples from Gumbel-Softmax distribution.
    """
    # Generate Gumbel noise: -log(-log(U))
    u = tf.random.uniform(tf.shape(logits), minval=1e-9, maxval=1.0 - 1e-9)
    gumbel_noise = -tf.math.log(-tf.math.log(u))

    # Soft Gumbel-Softmax
    y_soft = tf.nn.softmax((logits + gumbel_noise) / temperature)

    if hard:
        # Straight-through estimator
        y_hard = tf.one_hot(tf.argmax(y_soft, axis=-1), tf.shape(logits)[-1])
        return y_hard - tf.stop_gradient(y_soft) + y_soft

    return y_soft


__all__ = [
    "fused_collapse",
    "fused_collapse_available",
    "gumbel_softmax_tf",
]
