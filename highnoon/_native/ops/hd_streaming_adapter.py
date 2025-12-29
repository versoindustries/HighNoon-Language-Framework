# highnoon/_native/ops/hd_streaming_adapter.py
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

"""HD Streaming Adapter - Python bindings for C++ ops.

Phase 200+: Quantum-Enhanced HD Streaming Mode

This module provides the Python interface to the HD streaming projection ops.
All computation is performed by the C++ kernel - no Python fallbacks.

Usage:
    >>> from highnoon._native.ops.hd_streaming_adapter import HDStreamingAdapter
    >>> adapter = HDStreamingAdapter(hidden_dim=256, hd_dim=1024)
    >>> output = adapter(hd_bundles)  # (batch, 1, hidden_dim)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import tensorflow as tf

logger = logging.getLogger(__name__)

# --- Load the Custom Operators ---
_hd_streaming_module = None
hd_streaming_project_op = None
hd_streaming_project_grad_op = None


def _load_hd_streaming_ops():
    """Load the HD streaming operations from the consolidated library."""
    global _hd_streaming_module, hd_streaming_project_op, hd_streaming_project_grad_op

    # Check if already registered in TensorFlow
    if "HDStreamingProject" in tf.raw_ops.__dict__:
        hd_streaming_project_op = tf.raw_ops.HDStreamingProject
        hd_streaming_project_grad_op = tf.raw_ops.HDStreamingProjectGrad
        logger.debug("HDStreamingProject already registered in TensorFlow")
        return

    # Load from the consolidated _highnoon_core.so library
    from highnoon._native import resolve_op_library

    consolidated_lib_path = resolve_op_library(__file__, "_highnoon_core.so")
    if os.path.exists(consolidated_lib_path):
        try:
            _hd_streaming_module = tf.load_op_library(consolidated_lib_path)
            hd_streaming_project_op = _hd_streaming_module.hd_streaming_project
            hd_streaming_project_grad_op = _hd_streaming_module.hd_streaming_project_grad
            logger.info("Loaded HDStreamingProject from consolidated _highnoon_core.so")
            return
        except (tf.errors.NotFoundError, OSError, AttributeError) as e:
            logger.error(f"Could not load HDStreamingProject from consolidated library: {e}")

    logger.error("HD Streaming ops not found. Run build_secure.sh to compile native ops.")


# Load ops on module import
_load_hd_streaming_ops()


# Register gradient for the HDStreamingProject op
@tf.RegisterGradient("HDStreamingProject")
def _hd_streaming_project_grad_fn(op, grad):
    """Gradient function for HDStreamingProject op.

    This connects the forward op to the backward op for automatic differentiation.
    """
    if hd_streaming_project_grad_op is None:
        raise NotImplementedError("HDStreamingProjectGrad op not loaded. Cannot compute gradients.")

    # Get forward pass inputs
    hd_bundles = op.inputs[0]
    projection_weights = op.inputs[1]
    # projection_bias = op.inputs[2]  # Not needed for grad

    # Get attrs
    hd_dim = op.get_attr("hd_dim")
    hidden_dim = op.get_attr("hidden_dim")

    # Flatten sequence dim if present: (batch, 1, hidden_dim) -> (batch, hidden_dim)
    if len(grad.shape) == 3:
        grad = tf.squeeze(grad, axis=1)

    # Call the gradient op
    grad_bundles, grad_weights, grad_bias = hd_streaming_project_grad_op(
        grad,
        hd_bundles,
        projection_weights,
        hd_dim=hd_dim,
        hidden_dim=hidden_dim,
    )

    # Return gradients for each input: (hd_bundles, projection_weights, projection_bias)
    return grad_bundles, grad_weights, grad_bias


def hd_streaming_available() -> bool:
    """Check if the HD streaming operations are available."""
    return hd_streaming_project_op is not None


def hd_streaming_project(
    hd_bundles: tf.Tensor,
    projection_weights: tf.Tensor,
    projection_bias: tf.Tensor,
    hd_dim: int,
    hidden_dim: int,
) -> tf.Tensor:
    """Project HD bundles to model hidden dimension.

    Pure C++ operation - no Python fallback.

    Args:
        hd_bundles: Input HD bundles [batch, hd_dim], float32
        projection_weights: Projection matrix [hd_dim, hidden_dim], float32
        projection_bias: Bias vector [hidden_dim], float32
        hd_dim: HD bundle dimension
        hidden_dim: Model hidden dimension

    Returns:
        Projected output [batch, 1, hidden_dim], float32
    """
    if hd_streaming_project_op is None:
        raise NotImplementedError(
            "The C++ HDStreamingProject operator could not be loaded. "
            "Please compile it: ./highnoon/_native/build_secure.sh"
        )

    return hd_streaming_project_op(
        hd_bundles,
        projection_weights,
        projection_bias,
        hd_dim=hd_dim,
        hidden_dim=hidden_dim,
    )


def hd_streaming_project_grad(
    grad_output: tf.Tensor,
    hd_bundles: tf.Tensor,
    projection_weights: tf.Tensor,
    hd_dim: int,
    hidden_dim: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute gradients for HD streaming projection.

    Pure C++ operation - no Python fallback.

    Args:
        grad_output: Upstream gradient [batch, hidden_dim], float32
        hd_bundles: Forward pass input [batch, hd_dim], float32
        projection_weights: Forward pass weights [hd_dim, hidden_dim], float32
        hd_dim: HD bundle dimension
        hidden_dim: Model hidden dimension

    Returns:
        Tuple of (grad_bundles, grad_weights, grad_bias)
    """
    if hd_streaming_project_grad_op is None:
        raise NotImplementedError(
            "The C++ HDStreamingProjectGrad operator could not be loaded. "
            "Please compile it: ./highnoon/_native/build_secure.sh"
        )

    return hd_streaming_project_grad_op(
        grad_output,
        hd_bundles,
        projection_weights,
        hd_dim=hd_dim,
        hidden_dim=hidden_dim,
    )


class HDStreamingAdapter(tf.keras.layers.Layer):
    """Keras layer adapting HD bundles to sequence format for ReasoningModule.

    Transforms: (batch, hd_dim) -> (batch, 1, hidden_dim)

    This layer projects holographic bundles from HolographicCorpus to the model's
    hidden dimension, adding a sequence dimension for compatibility with the
    standard HSMN architecture.

    All computation is performed by C++ ops - no Python fallbacks.

    Args:
        hidden_dim: Model hidden dimension (output)
        hd_dim: HD bundle dimension (input)
        use_bias: Whether to include learnable bias
        kernel_initializer: Initializer for projection weights
        bias_initializer: Initializer for bias

    Example:
        >>> adapter = HDStreamingAdapter(hidden_dim=256, hd_dim=1024)
        >>> hd_bundles = tf.random.normal([32, 1024])  # batch of HD bundles
        >>> output = adapter(hd_bundles)  # (32, 1, 256)
    """

    def __init__(
        self,
        hidden_dim: int,
        hd_dim: int = 1024,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.hd_dim = hd_dim
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def build(self, input_shape: tf.TensorShape) -> None:
        # Validate input shape
        if input_shape[-1] != self.hd_dim:
            raise ValueError(f"Input shape {input_shape} incompatible with hd_dim={self.hd_dim}")

        # Create projection weights [hd_dim, hidden_dim]
        self.projection_weights = self.add_weight(
            name="projection_weights",
            shape=(self.hd_dim, self.hidden_dim),
            initializer=self.kernel_initializer,
            trainable=True,
        )

        # Create bias if enabled
        if self.use_bias:
            self.projection_bias = self.add_weight(
                name="projection_bias",
                shape=(self.hidden_dim,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        else:
            # Create zero bias for C++ op (which always expects bias tensor)
            self.projection_bias = tf.zeros([self.hidden_dim], dtype=tf.float32)

        super().build(input_shape)

    def call(self, hd_bundles: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Project HD bundles to sequence format.

        Args:
            hd_bundles: Input HD bundles [batch, hd_dim], float32
            training: Whether in training mode (unused, for API compatibility)

        Returns:
            Projected output [batch, 1, hidden_dim], float32
        """
        # Use the C++ op for forward pass
        output = hd_streaming_project(
            hd_bundles,
            self.projection_weights,
            self.projection_bias,
            self.hd_dim,
            self.hidden_dim,
        )
        return output

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "hd_dim": self.hd_dim,
                "use_bias": self.use_bias,
                "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            }
        )
        return config


__all__ = [
    "hd_streaming_project",
    "hd_streaming_project_grad",
    "hd_streaming_available",
    "HDStreamingAdapter",
]
