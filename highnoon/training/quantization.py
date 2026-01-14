# highnoon/training/quantization.py
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

"""Phase 13.4: Quantization-Aware Training (QAT).

This module provides infrastructure for training quantized models that
maintain accuracy while enabling faster inference with lower precision.

Supports:
- INT8 quantization (symmetric and asymmetric)
- INT4 quantization (for extreme compression)
- Mixed-precision training
- Quantization-aware fine-tuning

Example:
    >>> from highnoon.training.quantization import QuantizationConfig, apply_qat
    >>>
    >>> model = HSMN(vocab_size=32000, embedding_dim=512)
    >>> quantized_model = apply_qat(model, QuantizationConfig(bits=8))
    >>> quantized_model.fit(dataset, epochs=5)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.config import QUANTIZATION_BITS

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for quantization-aware training.

    Attributes:
        bits: Quantization bit width (4, 8, or 16).
        symmetric: Use symmetric quantization range.
        per_channel: Quantize per output channel vs per tensor.
        exclude_layers: Layer names to exclude from quantization.
        ema_decay: Decay for exponential moving average of ranges.
    """

    bits: int = QUANTIZATION_BITS
    symmetric: bool = True
    per_channel: bool = True
    exclude_layers: list[str] = field(default_factory=list)
    ema_decay: float = 0.999

    def __post_init__(self):
        if self.bits not in [4, 8, 16]:
            raise ValueError(f"Unsupported bit width: {self.bits}. Use 4, 8, or 16.")


class FakeQuantize(layers.Layer):
    """Fake quantization layer for QAT.

    Simulates quantization during training by:
    1. Computing min/max ranges from activations
    2. Quantizing to discrete levels
    3. Dequantizing back to float (for gradient flow)

    The layer learns optimal quantization ranges during training.
    """

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        ema_decay: float = 0.999,
        **kwargs,
    ):
        """Initialize fake quantization layer.

        Args:
            bits: Number of quantization bits.
            symmetric: Use symmetric range around zero.
            per_channel: Quantize per channel (for weights).
            ema_decay: EMA decay for range updates.
        """
        super().__init__(**kwargs)
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.ema_decay = ema_decay

        # Compute number of quantization levels
        self.num_levels = 2**bits
        if symmetric:
            self.qmin = -(2 ** (bits - 1))
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**bits - 1

    def build(self, input_shape):
        # Learnable or EMA-tracked quantization ranges
        if self.per_channel:
            num_channels = input_shape[-1]
            range_shape = [num_channels]
        else:
            num_channels = 1
            range_shape = [1]

        self.min_val = self.add_weight(
            name="min_val",
            shape=range_shape,
            initializer=tf.initializers.Constant(-1.0),
            trainable=False,
        )
        self.max_val = self.add_weight(
            name="max_val",
            shape=range_shape,
            initializer=tf.initializers.Constant(1.0),
            trainable=False,
        )
        self.initialized = self.add_weight(
            name="initialized",
            shape=[],
            initializer=tf.initializers.Constant(0.0),
            trainable=False,
        )

        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply fake quantization.

        Args:
            x: Input tensor.
            training: Whether in training mode.

        Returns:
            Fake-quantized tensor (same dtype as input).
        """
        if training:
            # Update range statistics
            self._update_ranges(x)

        # Get quantization parameters
        scale, zero_point = self._compute_scale_zp()

        # Fake quantize: quantize then dequantize
        # Use straight-through estimator for gradients
        x_quant = self._fake_quantize(x, scale, zero_point)

        return x_quant

    def _update_ranges(self, x: tf.Tensor) -> None:
        """Update min/max ranges from activation statistics."""
        if self.per_channel:
            # Compute per-channel min/max
            axes = list(range(len(x.shape) - 1))
            batch_min = tf.reduce_min(x, axis=axes)
            batch_max = tf.reduce_max(x, axis=axes)
        else:
            batch_min = tf.reduce_min(x)
            batch_max = tf.reduce_max(x)

        # Initialize on first batch
        is_initialized = tf.greater(self.initialized, 0.5)

        new_min = tf.where(
            is_initialized,
            self.ema_decay * self.min_val + (1 - self.ema_decay) * batch_min,
            batch_min,
        )
        new_max = tf.where(
            is_initialized,
            self.ema_decay * self.max_val + (1 - self.ema_decay) * batch_max,
            batch_max,
        )

        self.min_val.assign(new_min)
        self.max_val.assign(new_max)
        self.initialized.assign(1.0)

    def _compute_scale_zp(self) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute quantization scale and zero point."""
        if self.symmetric:
            # Symmetric: zero_point = 0, scale = max(|min|, |max|) / qmax
            abs_max = tf.maximum(tf.abs(self.min_val), tf.abs(self.max_val))
            scale = abs_max / self.qmax
            zero_point = tf.zeros_like(scale)
        else:
            # Asymmetric: use full range
            scale = (self.max_val - self.min_val) / (self.qmax - self.qmin)
            zero_point = self.qmin - self.min_val / scale
            zero_point = tf.round(zero_point)

        # Avoid division by zero
        scale = tf.maximum(scale, 1e-8)

        return scale, zero_point

    def _fake_quantize(
        self,
        x: tf.Tensor,
        scale: tf.Tensor,
        zero_point: tf.Tensor,
    ) -> tf.Tensor:
        """Perform fake quantization with straight-through estimator."""
        # Quantize
        x_int = tf.round(x / scale + zero_point)
        x_int = tf.clip_by_value(x_int, self.qmin, self.qmax)

        # Dequantize
        x_quant = (x_int - zero_point) * scale

        # Straight-through estimator: use quantized value in forward,
        # but pass gradients through as if identity
        return x + tf.stop_gradient(x_quant - x)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "bits": self.bits,
                "symmetric": self.symmetric,
                "per_channel": self.per_channel,
                "ema_decay": self.ema_decay,
            }
        )
        return config


class QuantizedDense(layers.Layer):
    """Dense layer with quantized weights and activations.

    Applies fake quantization to both weights and activations during
    training for quantization-aware training.
    """

    def __init__(
        self,
        units: int,
        config: QuantizationConfig | None = None,
        activation: str | Callable | None = None,
        use_bias: bool = True,
        **kwargs,
    ):
        """Initialize quantized dense layer.

        Args:
            units: Number of output units.
            config: Quantization configuration.
            activation: Activation function.
            use_bias: Whether to include bias.
        """
        super().__init__(**kwargs)
        self.units = units
        self.config = config or QuantizationConfig()
        self.activation_fn = tf.keras.activations.get(activation)
        self.use_bias = use_bias

        # Fake quantization layers
        self.weight_quantize = FakeQuantize(
            bits=self.config.bits,
            symmetric=self.config.symmetric,
            per_channel=self.config.per_channel,
            name="weight_quantize",
        )
        self.activation_quantize = FakeQuantize(
            bits=self.config.bits,
            symmetric=self.config.symmetric,
            per_channel=False,
            name="activation_quantize",
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_dim, self.units],
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.units],
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None

        # Build quantization layers
        self.weight_quantize.build([input_dim, self.units])
        self.activation_quantize.build([None, self.units])

        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass with quantization."""
        # Quantize weights
        kernel_q = self.weight_quantize(self.kernel, training=training)

        # Matrix multiply
        output = tf.matmul(x, kernel_q)

        if self.use_bias:
            output = output + self.bias

        if self.activation_fn is not None:
            output = self.activation_fn(output)

        # Quantize activations
        output = self.activation_quantize(output, training=training)

        return output

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "use_bias": self.use_bias,
            }
        )
        return config


def apply_qat(
    model: tf.keras.Model,
    config: QuantizationConfig | None = None,
) -> tf.keras.Model:
    """Apply quantization-aware training to a model.

    Wraps model layers with fake quantization for QAT.

    Args:
        model: Keras model to quantize.
        config: Quantization configuration.

    Returns:
        Model with QAT applied.

    Note:
        This is a simplified implementation. For production use,
        consider TensorFlow Model Optimization Toolkit.
    """
    config = config or QuantizationConfig()

    logger.info(
        f"Applying QAT with {config.bits}-bit quantization "
        f"(symmetric={config.symmetric}, per_channel={config.per_channel})"
    )

    # Clone the model to avoid modifying the original
    try:
        import tensorflow_model_optimization as tfmot

        # Use TF-MOT if available
        quantize_model = tfmot.quantization.keras.quantize_model
        qat_model = quantize_model(model)
        logger.info("Applied QAT using TensorFlow Model Optimization Toolkit")
        return qat_model
    except ImportError:
        logger.warning(
            "TensorFlow Model Optimization not installed. "
            "Using simplified QAT wrapper. "
            "Install with: pip install tensorflow-model-optimization"
        )

    # Simplified fallback: wrap the model
    return QuantizedModelWrapper(model, config)


class QuantizedModelWrapper(tf.keras.Model):
    """Wrapper that adds quantization to model inputs/outputs.

    This is a simplified QAT implementation that quantizes
    the model's inputs and outputs.
    """

    def __init__(self, model: tf.keras.Model, config: QuantizationConfig):
        super().__init__()
        self.wrapped_model = model
        self.config = config

        # Input/output quantization
        self.input_quantize = FakeQuantize(
            bits=config.bits,
            symmetric=config.symmetric,
            name="input_quantize",
        )
        self.output_quantize = FakeQuantize(
            bits=config.bits,
            symmetric=config.symmetric,
            name="output_quantize",
        )

    def call(self, inputs, training=False, **kwargs):
        # Quantize inputs (for embedding lookup, this is optional)
        # For language models, we typically skip input quantization

        # Forward through wrapped model
        outputs = self.wrapped_model(inputs, training=training, **kwargs)

        # Quantize outputs (logits)
        if isinstance(outputs, tf.Tensor):
            outputs = self.output_quantize(outputs, training=training)
        elif hasattr(outputs, "logits"):
            outputs.logits = self.output_quantize(outputs.logits, training=training)

        return outputs

    def get_config(self):
        return {
            "config": {
                "bits": self.config.bits,
                "symmetric": self.config.symmetric,
                "per_channel": self.config.per_channel,
            }
        }


def export_quantized_model(
    model: tf.keras.Model,
    output_path: str,
    config: QuantizationConfig | None = None,
) -> None:
    """Export a quantized model for inference.

    Converts the QAT model to actual quantized format for deployment.

    Args:
        model: QAT-trained model.
        output_path: Path to save the quantized model.
        config: Quantization configuration.
    """
    config = config or QuantizationConfig()

    try:
        # Use TF-MOT for conversion
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if config.bits == 8:
            converter.target_spec.supported_types = [tf.int8]

        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        logger.info(f"Exported quantized model to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export quantized model: {e}")
        raise


__all__ = [
    "QuantizationConfig",
    "FakeQuantize",
    "QuantizedDense",
    "QuantizedModelWrapper",
    "apply_qat",
    "export_quantized_model",
]
