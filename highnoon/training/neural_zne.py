# highnoon/training/neural_zne.py
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

"""Neural Zero-Noise Extrapolation (Neural ZNE) for Quantum Error Mitigation.

This module implements a neural network-based approach to Zero-Noise
Extrapolation (ZNE), a quantum error mitigation technique. Instead of
polynomial extrapolation, we use a small MLP to learn the noise-to-noiseless
mapping, which can better capture non-polynomial error behavior.

Key Concepts:
- ZNE runs circuits at multiple noise levels and extrapolates to zero noise
- Neural ZNE uses an MLP to learn: f(noisy_output) → noiseless_output
- The network is trained online during the training loop

Mathematical Framework:
    Classical ZNE: O_clean = extrapolate({(λ_i, O_λi)}) as λ → 0
    Neural ZNE: O_clean = MLP(O_noisy; θ) where MLP is trained to minimize
                ||MLP(O_noisy) - O_true||²

Reference:
    - "Quantum Error Mitigation using Neural Networks" (2022)
    - "Zero-noise extrapolation for quantum-gate error mitigation" (2020)

Example:
    >>> zne = NeuralZNE(hidden_dim=128)
    >>> clean_output = zne.mitigate(noisy_output)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf

from highnoon import config

logger = logging.getLogger(__name__)


class ZNEMitigator(tf.keras.layers.Layer):
    """Small MLP for noise-to-clean mapping.

    Architecture: input → Dense(hidden) → GELU → Dense(hidden) → GELU → output
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        name: str = "zne_mitigator",
        **kwargs,
    ):
        """Initialize ZNE mitigator network.

        Args:
            hidden_dim: Hidden layer dimension.
            name: Layer name.
        """
        super().__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim

        self.dense1 = tf.keras.layers.Dense(
            hidden_dim,
            activation="gelu",
            name=f"{name}_dense1",
        )
        self.dense2 = tf.keras.layers.Dense(
            hidden_dim,
            activation="gelu",
            name=f"{name}_dense2",
        )
        self.output_layer = None  # Built dynamically

    def build(self, input_shape):
        """Build the output layer based on input shape."""
        output_dim = input_shape[-1]
        self.output_layer = tf.keras.layers.Dense(
            output_dim,
            name=f"{self.name}_output",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through the mitigator.

        Args:
            inputs: Noisy input tensor.
            training: Whether in training mode.

        Returns:
            Mitigated (denoised) output.
        """
        x = self.dense1(inputs)
        x = self.dense2(x)

        # Residual connection for stability
        output = self.output_layer(x) + inputs
        return output


@dataclass
class NeuralZNE:
    """Neural Zero-Noise Extrapolation for quantum layer error mitigation.

    Maintains a small MLP that learns to map noisy outputs to clean outputs.
    The network is trained online using samples collected during training.

    Attributes:
        hidden_dim: Hidden dimension for the mitigator MLP.
        learning_rate: Learning rate for mitigator training.
        buffer_size: Size of the training sample buffer.
        train_every_n_steps: Train mitigator every N steps.
        enabled: Whether Neural ZNE is active.
    """

    hidden_dim: int = field(default_factory=lambda: config.NEURAL_ZNE_HIDDEN_DIM)
    learning_rate: float = 1e-4
    buffer_size: int = field(default_factory=lambda: config.NEURAL_ZNE_TRAIN_SAMPLES)
    train_every_n_steps: int = 100
    enabled: bool = field(default_factory=lambda: config.USE_NEURAL_ZNE)

    # Internal state
    _mitigators: dict[str, ZNEMitigator] = field(default_factory=dict)
    _optimizers: dict[str, tf.keras.optimizers.Optimizer] = field(default_factory=dict)
    _sample_buffers: dict[str, list[tuple[tf.Tensor, tf.Tensor]]] = field(default_factory=dict)
    _step_counter: int = 0

    def __post_init__(self) -> None:
        """Initialize internal state."""
        if not hasattr(self, "_mitigators") or self._mitigators is None:
            self._mitigators = {}
        if not hasattr(self, "_optimizers") or self._optimizers is None:
            self._optimizers = {}
        if not hasattr(self, "_sample_buffers") or self._sample_buffers is None:
            self._sample_buffers = {}
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0

    def _get_or_create_mitigator(self, layer_name: str) -> ZNEMitigator:
        """Get or create a mitigator for a specific layer."""
        if layer_name not in self._mitigators:
            mitigator = ZNEMitigator(
                hidden_dim=self.hidden_dim,
                name=f"zne_{layer_name.replace('/', '_').replace(':', '_')}",
            )
            self._mitigators[layer_name] = mitigator
            self._optimizers[layer_name] = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            )
            self._sample_buffers[layer_name] = []
            logger.debug("[ZNE] Created mitigator for '%s'", layer_name)

        return self._mitigators[layer_name]

    def add_training_sample(
        self,
        layer_name: str,
        noisy_output: tf.Tensor,
        clean_output: tf.Tensor,
    ) -> None:
        """Add a training sample to the buffer.

        Args:
            layer_name: Name of the quantum layer.
            noisy_output: Output with noise/errors.
            clean_output: Ground truth output (or lower-noise estimate).
        """
        if not self.enabled:
            return

        self._get_or_create_mitigator(layer_name)

        buffer = self._sample_buffers[layer_name]
        buffer.append((noisy_output, clean_output))

        # Keep buffer bounded
        if len(buffer) > self.buffer_size:
            buffer.pop(0)

    def mitigate(
        self,
        layer_name: str,
        noisy_output: tf.Tensor,
    ) -> tf.Tensor:
        """Apply neural ZNE mitigation to a noisy output.

        Args:
            layer_name: Name of the quantum layer.
            noisy_output: Noisy output tensor.

        Returns:
            Mitigated output tensor.
        """
        if not self.enabled:
            return noisy_output

        mitigator = self._get_or_create_mitigator(layer_name)

        # Ensure mitigator is built
        if mitigator.output_layer is None:
            # Build with a dummy call
            _ = mitigator(noisy_output[:1] if len(noisy_output.shape) > 1 else noisy_output)

        return mitigator(noisy_output, training=False)

    def train_step(self, layer_name: str | None = None) -> dict[str, float]:
        """Train the mitigator networks on buffered samples.

        Args:
            layer_name: Specific layer to train, or None for all.

        Returns:
            Dictionary of layer names to training losses.
        """
        if not self.enabled:
            return {}

        losses = {}
        layers_to_train = [layer_name] if layer_name else list(self._mitigators.keys())

        for name in layers_to_train:
            if name not in self._sample_buffers:
                continue

            buffer = self._sample_buffers[name]
            if len(buffer) < 10:  # Need minimum samples
                continue

            # Get batch from buffer
            batch_size = min(32, len(buffer))
            indices = np.random.choice(len(buffer), batch_size, replace=False)
            noisy_batch = tf.stack([buffer[i][0] for i in indices])
            clean_batch = tf.stack([buffer[i][1] for i in indices])

            mitigator = self._mitigators[name]
            optimizer = self._optimizers[name]

            with tf.GradientTape() as tape:
                predictions = mitigator(noisy_batch, training=True)
                loss = tf.reduce_mean(tf.square(predictions - clean_batch))

            grads = tape.gradient(loss, mitigator.trainable_variables)
            if grads and any(g is not None for g in grads):
                optimizer.apply_gradients(zip(grads, mitigator.trainable_variables))

            losses[name] = float(loss.numpy())

        self._step_counter += 1
        return losses

    def step(self) -> dict[str, float]:
        """Perform one step of ZNE (potentially training mitigators).

        Returns:
            Training losses if training occurred, else empty dict.
        """
        self._step_counter += 1

        if self._step_counter % self.train_every_n_steps == 0:
            return self.train_step()

        return {}

    def get_statistics(self) -> dict[str, Any]:
        """Get ZNE statistics."""
        stats = {
            "enabled": self.enabled,
            "hidden_dim": self.hidden_dim,
            "step_counter": self._step_counter,
            "num_mitigators": len(self._mitigators),
            "layers": {},
        }

        for name in self._mitigators:
            buffer_size = len(self._sample_buffers.get(name, []))
            num_params = (
                sum(np.prod(v.shape) for v in self._mitigators[name].trainable_variables)
                if self._mitigators[name].trainable_variables
                else 0
            )

            stats["layers"][name] = {
                "buffer_size": buffer_size,
                "num_params": int(num_params),
            }

        return stats

    def reset(self) -> None:
        """Reset all ZNE state."""
        self._mitigators.clear()
        self._optimizers.clear()
        self._sample_buffers.clear()
        self._step_counter = 0
        logger.info("[ZNE] State reset")


__all__ = [
    "NeuralZNE",
    "ZNEMitigator",
]
