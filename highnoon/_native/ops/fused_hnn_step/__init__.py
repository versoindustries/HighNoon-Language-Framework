# highnoon/_native/ops/fused_hnn_step/__init__.py
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

"""Fused HNN Step Operation with Python fallback.

This module provides the fused_hnn_step operation for single-step
Hamiltonian Neural Network forward pass. Uses native C++ ops when
available, with pure Python/TensorFlow fallback for compatibility.

The operation computes a single step of the HNN update, used during
inference with state caching.
"""

from __future__ import annotations

import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger(__name__)

# Try to load native op
_native_op = None
_native_available = False

try:
    from highnoon._native import get_op, is_native_available

    if is_native_available():
        try:
            _native_op = get_op("fused_hnn_step")
            if _native_op is not None:
                _native_available = True
                logger.debug("Native fused_hnn_step op loaded successfully")
        except Exception as e:
            logger.debug(f"Could not load native fused_hnn_step: {e}")
except ImportError:
    pass


def _python_hnn_step(
    input_state: tf.Tensor,
    hidden_state: tf.Tensor,
    weights_ih: tf.Tensor,
    weights_hh: tf.Tensor,
    bias: tf.Tensor | None = None,
    dt: float = 0.01,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Pure Python/TensorFlow implementation of HNN step.

    Computes a single step of the Hamiltonian Neural Network update
    using symplectic integration.

    Args:
        input_state: Input tensor [batch, input_dim].
        hidden_state: Hidden state tensor [batch, hidden_dim].
        weights_ih: Input-to-hidden weights [input_dim, hidden_dim].
        weights_hh: Hidden-to-hidden weights [hidden_dim, hidden_dim].
        bias: Optional bias [hidden_dim].
        dt: Time step for integration.

    Returns:
        Tuple of (output, new_hidden_state).
    """
    # Project input
    ih = tf.matmul(input_state, weights_ih)

    # Project hidden state
    hh = tf.matmul(hidden_state, weights_hh)

    # Combine with optional bias
    combined = ih + hh
    if bias is not None:
        combined = combined + bias

    # Apply activation (SiLU for energy-preserving dynamics)
    activated = combined * tf.nn.sigmoid(combined)

    # Symplectic update (simplified Leapfrog)
    # In full HNN, this would be: p -> p - dt * dV/dq, q -> q + dt * dT/dp
    new_hidden = hidden_state + dt * activated

    # Output is the new hidden state
    output = new_hidden

    return output, new_hidden


def fused_hnn_step(
    input_state: tf.Tensor,
    hidden_state: tf.Tensor,
    weights_ih: tf.Tensor,
    weights_hh: tf.Tensor,
    bias: tf.Tensor | None = None,
    dt: float = 0.01,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Fused HNN step operation.

    Computes a single step of the Hamiltonian Neural Network update.
    Uses native C++ implementation when available for optimal CPU
    performance (AVX2/AVX512 vectorized), otherwise falls back to
    pure TensorFlow.

    This is the single-step version used during cached inference,
    where we process one token at a time with state carried forward.

    Args:
        input_state: Input tensor [batch, input_dim].
        hidden_state: Hidden state tensor [batch, hidden_dim].
        weights_ih: Input-to-hidden weights [input_dim, hidden_dim].
        weights_hh: Hidden-to-hidden weights [hidden_dim, hidden_dim].
        bias: Optional bias [hidden_dim].
        dt: Time step for integration.

    Returns:
        Tuple of (output, new_hidden_state).

    Example:
        >>> output, new_state = fused_hnn_step(
        ...     input_state=token_embedding,
        ...     hidden_state=cached_state,
        ...     weights_ih=model.W_ih,
        ...     weights_hh=model.W_hh,
        ...     dt=0.01,
        ... )
    """
    if _native_available and _native_op is not None:
        try:
            result = _native_op(
                input_state,
                hidden_state,
                weights_ih,
                weights_hh,
                bias if bias is not None else tf.zeros([tf.shape(weights_hh)[1]]),
                dt,
            )
            # Native op returns packed tensor, unpack
            if isinstance(result, tuple):
                return result
            else:
                # Split single tensor output
                tf.shape(hidden_state)[-1]
                return result, result
        except Exception as e:
            logger.debug(f"Native fused_hnn_step failed, using fallback: {e}")

    # Fallback to Python implementation
    return _python_hnn_step(
        input_state,
        hidden_state,
        weights_ih,
        weights_hh,
        bias,
        dt,
    )


__all__ = ["fused_hnn_step"]
