# highnoon/_native/ops/optimizers.py
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

"""Python wrapper for native optimizer operations.

Provides access to high-performance C++ optimizer implementations
including SophiaG, Lion, and optimized AdamW variants.
"""

import logging

import tensorflow as tf

from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

# --- Load the Custom Operators ---
_optimizers_module = None
_sophia_update_op = None
_lion_update_op = None

try:
    # lib_loader.resolve_op_library now returns _highnoon_core.so path
    _op_lib_path = resolve_op_library(__file__, "_optimizers.so")
    _optimizers_module = tf.load_op_library(_op_lib_path)
    # Only set ops if they exist in the library
    if hasattr(_optimizers_module, "sophia_update"):
        _sophia_update_op = _optimizers_module.sophia_update
    if hasattr(_optimizers_module, "lion_update"):
        _lion_update_op = _optimizers_module.lion_update
    if _sophia_update_op or _lion_update_op:
        logger.info("Successfully loaded native optimizer operations.")
    else:
        logger.debug("Native optimizer ops not found in loaded library. " "Using Python fallbacks.")
except (tf.errors.NotFoundError, OSError, AttributeError) as e:
    logger.warning(f"Could not load native optimizer ops. Using Python fallbacks. Error: {e}")


def native_optimizers_available() -> bool:
    """Check if native optimizer operations are available."""
    return _optimizers_module is not None


def sophia_update_available() -> bool:
    """Check if native SophiaG update operation is available."""
    return _sophia_update_op is not None


def lion_update_available() -> bool:
    """Check if native Lion update operation is available."""
    return _lion_update_op is not None


def sophia_update(
    param: tf.Tensor,
    grad: tf.Tensor,
    exp_avg: tf.Tensor,
    hessian: tf.Tensor,
    learning_rate: float,
    beta1: float = 0.965,
    beta2: float = 0.99,
    rho: float = 0.04,
    weight_decay: float = 0.0,
) -> tf.Tensor:
    """Apply SophiaG optimizer update using native C++ operation.

    SophiaG is a second-order optimizer that uses Hessian diagonal
    approximation for adaptive learning rates.

    Args:
        param: Parameter tensor to update.
        grad: Gradient tensor.
        exp_avg: Exponential moving average of gradients.
        hessian: Hessian diagonal estimate.
        learning_rate: Learning rate.
        beta1: Momentum coefficient (default: 0.965).
        beta2: Hessian EMA coefficient (default: 0.99).
        rho: Clipping threshold (default: 0.04).
        weight_decay: Weight decay coefficient (default: 0.0).

    Returns:
        Updated parameter tensor.

    Raises:
        NotImplementedError: If native operation is not available.
    """
    if _sophia_update_op is None:
        raise NotImplementedError(
            "Native SophiaG update operation is unavailable. "
            "Use the Python SophiaG optimizer instead."
        )

    return _sophia_update_op(
        param,
        grad,
        exp_avg,
        hessian,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        rho=rho,
        weight_decay=weight_decay,
    )


def lion_update(
    param: tf.Tensor,
    grad: tf.Tensor,
    exp_avg: tf.Tensor,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
) -> tf.Tensor:
    """Apply Lion optimizer update using native C++ operation.

    Lion is a memory-efficient optimizer using only sign-based updates.

    Args:
        param: Parameter tensor to update.
        grad: Gradient tensor.
        exp_avg: Exponential moving average of gradients.
        learning_rate: Learning rate.
        beta1: Momentum coefficient for update (default: 0.9).
        beta2: Momentum coefficient for EMA (default: 0.99).
        weight_decay: Weight decay coefficient (default: 0.0).

    Returns:
        Updated parameter tensor.

    Raises:
        NotImplementedError: If native operation is not available.
    """
    if _lion_update_op is None:
        raise NotImplementedError(
            "Native Lion update operation is unavailable. " "Use the Python Lion optimizer instead."
        )

    return _lion_update_op(
        param,
        grad,
        exp_avg,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
    )


__all__ = [
    "native_optimizers_available",
    "sophia_update_available",
    "lion_update_available",
    "sophia_update",
    "lion_update",
]
