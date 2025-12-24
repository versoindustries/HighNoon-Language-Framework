# highnoon/_native/ops/mla_collapse_op.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Python wrapper for the Multi-Head Latent Attention Collapse C++ operator.
#
# NO PYTHON FALLBACK: This module requires the compiled .so to function.

"""Multi-Head Latent Attention (MLA) Collapse operator.

Implements DeepSeek-V2 style latent compression for efficient superposition collapse:
1. Compress K/V to lower-dimensional latent space
2. Perform attention in latent space (faster)
3. Expand result back to full dimension

This reduces memory and compute for the collapse mechanism while maintaining quality.
"""

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

# --- Load Custom C++ Operator via consolidated binary ---
_mla_collapse_module = None
mla_collapse_op = None
mla_collapse_grad_op = None

try:
    from highnoon._native import get_op

    _mla_collapse_module = get_op("mla_collapse")

    if _mla_collapse_module is not None:
        mla_collapse_op = getattr(_mla_collapse_module, "mla_collapse", None)
        mla_collapse_grad_op = getattr(_mla_collapse_module, "mla_collapse_grad", None)

        if mla_collapse_op is None:
            # Try CamelCase
            mla_collapse_op = getattr(_mla_collapse_module, "MLACollapse", None)
            mla_collapse_grad_op = getattr(_mla_collapse_module, "MLACollapseGrad", None)

        if mla_collapse_op is not None:
            logger.debug("Successfully loaded C++ MLA collapse operator from consolidated binary.")
        else:
            logger.warning(
                "C++ MLA op loaded but symbols not found. Available: %s",
                [a for a in dir(_mla_collapse_module) if not a.startswith("_")][:10],
            )
except Exception as e:
    logger.warning(f"Could not load MLA collapse op: {e}")


@tf.custom_gradient
def mla_collapse(
    query: tf.Tensor,
    key_stack: tf.Tensor,
    value_stack: tf.Tensor,
    kv_compress: tf.Tensor,
    kv_expand: tf.Tensor,
    latent_dim: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Multi-Head Latent Attention collapse for superposition paths.

    Performs efficient attention in a compressed latent space rather than
    full d_model space, reducing compute complexity.

    Args:
        query: [batch, d_model] - Query vectors (typically from context).
        key_stack: [batch, S, d_model] - Key vectors from S superposition paths.
        value_stack: [batch, S, d_model] - Value vectors from S superposition paths.
        kv_compress: [d_model, latent_dim] - Compression projection matrix.
        kv_expand: [latent_dim, d_model] - Expansion projection matrix.
        latent_dim: Dimension of the latent space (typically d_model // 4).

    Returns:
        Tuple of:
        - collapsed: [batch, d_model] - Collapsed output.
        - attention_probs: [batch, S] - Attention probabilities over paths.

    Raises:
        NotImplementedError: If C++ operator not available.

    Example:
        >>> batch, d_model, S, latent_dim = 2, 256, 4, 64
        >>> query = tf.random.normal([batch, d_model])
        >>> key_stack = tf.random.normal([batch, S, d_model])
        >>> value_stack = tf.random.normal([batch, S, d_model])
        >>> kv_compress = tf.random.normal([d_model, latent_dim])
        >>> kv_expand = tf.random.normal([latent_dim, d_model])
        >>> collapsed, probs = mla_collapse(
        ...     query, key_stack, value_stack, kv_compress, kv_expand, latent_dim
        ... )
    """
    if mla_collapse_op is None:
        raise NotImplementedError(
            "The C++ MLACollapse operator could not be loaded. "
            "Please ensure it has been compiled correctly. NO PYTHON FALLBACK."
        )

    collapsed, attention_probs = mla_collapse_op(
        query=query,
        key_stack=key_stack,
        value_stack=value_stack,
        kv_compress=kv_compress,
        kv_expand=kv_expand,
        latent_dim=latent_dim,
    )

    def grad_fn(grad_collapsed, grad_attention_probs, variables=None):
        """Gradient function calling the C++ backward kernel."""
        if mla_collapse_grad_op is None:
            raise NotImplementedError("The C++ MLACollapseGrad operator could not be loaded.")

        (
            grad_query,
            grad_key_stack,
            grad_value_stack,
            grad_kv_compress,
            grad_kv_expand,
        ) = mla_collapse_grad_op(
            grad_collapsed=grad_collapsed,
            query=query,
            key_stack=key_stack,
            value_stack=value_stack,
            kv_compress=kv_compress,
            kv_expand=kv_expand,
            attention_probs=attention_probs,
            latent_dim=latent_dim,
        )

        input_grads = (
            grad_query,
            grad_key_stack,
            grad_value_stack,
            grad_kv_compress,
            grad_kv_expand,
        )
        variable_grads = [None] * len(variables) if variables is not None else []
        return input_grads, variable_grads

    return (collapsed, attention_probs), grad_fn
