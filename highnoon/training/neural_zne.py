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

try:
    from highnoon._native.ops.alphaqubit_ops import (
        alphaqubit_correct,
        create_alphaqubit_correct_weights,
    )
    from highnoon._native.ops.alphaqubit_ops import ops_available as alphaqubit_ops_available

    _ALPHAQUBIT_NATIVE_AVAILABLE = alphaqubit_ops_available()
except ImportError:
    _ALPHAQUBIT_NATIVE_AVAILABLE = False
    alphaqubit_correct = None
    create_alphaqubit_correct_weights = None

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

    Phase 130.8 Enhancement (S8): Q-SSM Gate Statistics Integration
    When NEURAL_ZNE_USE_QSSM_STATS is enabled, gate statistics from Q-SSM
    layers are used to calibrate the noise model, improving mitigation
    accuracy for quantum-classical hybrid layers.

    Attributes:
        hidden_dim: Hidden dimension for the mitigator MLP.
        learning_rate: Learning rate for mitigator training.
        buffer_size: Size of the training sample buffer.
        train_every_n_steps: Train mitigator every N steps.
        enabled: Whether Neural ZNE is active.
        use_qssm_stats: Whether to use Q-SSM gate statistics for calibration.
    """

    hidden_dim: int = field(default_factory=lambda: config.NEURAL_ZNE_HIDDEN_DIM)
    learning_rate: float = 1e-4
    buffer_size: int = field(default_factory=lambda: config.NEURAL_ZNE_TRAIN_SAMPLES)
    train_every_n_steps: int = 100
    enabled: bool = field(default_factory=lambda: config.USE_NEURAL_ZNE)
    use_qssm_stats: bool = field(
        default_factory=lambda: getattr(config, "NEURAL_ZNE_USE_QSSM_STATS", True)
    )

    # Internal state
    _mitigators: dict[str, ZNEMitigator] = field(default_factory=dict)
    _optimizers: dict[str, tf.keras.optimizers.Optimizer] = field(default_factory=dict)
    _sample_buffers: dict[str, list[tuple[tf.Tensor, tf.Tensor]]] = field(default_factory=dict)
    _step_counter: int = 0
    # S8: Q-SSM gate statistics for calibration
    _qssm_gate_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    _noise_scale_adjustments: dict[str, float] = field(default_factory=dict)

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
        if not hasattr(self, "_qssm_gate_stats") or self._qssm_gate_stats is None:
            self._qssm_gate_stats = {}
        if not hasattr(self, "_noise_scale_adjustments") or self._noise_scale_adjustments is None:
            self._noise_scale_adjustments = {}

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
        self._qssm_gate_stats.clear()
        self._noise_scale_adjustments.clear()
        logger.info("[ZNE] State reset")

    def receive_qssm_gate_stats(
        self,
        layer_name: str,
        gate_mean: float,
        gate_std: float,
        selectivity: float,
    ) -> None:
        """S8: Receive Q-SSM gate statistics for noise calibration.

        Q-SSM layers report their gate statistics which are used to
        dynamically adjust the noise model for more accurate mitigation.

        Higher selectivity (sharper gating) indicates more deterministic
        behavior and thus lower effective noise - we reduce the correction
        strength. Lower selectivity indicates more stochastic behavior
        requiring stronger correction.

        Args:
            layer_name: Name of the Q-SSM layer.
            gate_mean: Mean of gate values (0-1 range).
            gate_std: Standard deviation of gate values.
            selectivity: Gate selectivity score (higher = sharper gating).
        """
        if not self.use_qssm_stats or not self.enabled:
            return

        # Store statistics
        self._qssm_gate_stats[layer_name] = {
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "selectivity": selectivity,
        }

        # Compute noise scale adjustment based on gate statistics
        # High selectivity -> low noise -> scale < 1.0 (less correction)
        # Low selectivity -> high noise -> scale > 1.0 (more correction)
        # Gate std contributes to uncertainty -> higher std = more noise
        base_scale = 1.0 - 0.3 * selectivity + 0.5 * gate_std
        self._noise_scale_adjustments[layer_name] = max(0.5, min(1.5, base_scale))

        logger.debug(
            "[ZNE-S8] Q-SSM stats for '%s': mean=%.3f, std=%.3f, "
            "selectivity=%.3f -> noise_scale=%.3f",
            layer_name,
            gate_mean,
            gate_std,
            selectivity,
            self._noise_scale_adjustments[layer_name],
        )

    def get_noise_scale(self, layer_name: str) -> float:
        """Get the noise scale adjustment for a layer.

        Returns a scale factor based on Q-SSM gate statistics if available,
        otherwise returns 1.0 (no adjustment).

        Args:
            layer_name: Name of the layer.

        Returns:
            Noise scale factor (1.0 = default, <1 = less noise, >1 = more noise).
        """
        return self._noise_scale_adjustments.get(layer_name, 1.0)


# =============================================================================
# Phase 129: ML-Enhanced Quantum Error Mitigation (Neural QEM)
# =============================================================================


class NeuralQuantumErrorMitigator(tf.keras.layers.Layer):
    """ML-enhanced ZNE for quantum layer outputs.

    This layer implements the Phase 129 ML-QEM enhancement, using neural
    networks to achieve near noise-free results with lower overhead than
    traditional ZNE polynomial extrapolation.

    Architecture:
        - Error correction network: Dense -> GELU -> Dense -> GELU -> Output
        - Noise predictor: Dense layer predicting noise scale distribution
        - Residual connection for gradient stability

    Mathematical Framework:
        Classical ZNE: O_clean = extrapolate({(λ_i, O_λi)}) as λ → 0
        ML-QEM: O_clean = ErrorModel(O_noisy) where ErrorModel learns
                layer-specific noise correction patterns

    Complexity: O(n · d) per mitigation pass, O(d) memory for network weights.

    Reference:
        - ML-QEM (2025): "Machine Learning for Quantum Error Mitigation"

    Attributes:
        output_dim: Output dimensionality (set during build).
        noise_levels: Tuple of noise scales for extrapolation.
        hidden_dim: Hidden dimension for error correction network.

    Example:
        >>> mitigator = NeuralQuantumErrorMitigator(noise_levels=[1.0, 1.5, 2.0])
        >>> corrected_output = mitigator(quantum_layer_output)
    """

    def __init__(
        self,
        noise_levels: tuple[float, ...] | list[float] | None = None,
        hidden_dim: int | None = None,
        name: str = "neural_qem",
        **kwargs,
    ):
        """Initialize NeuralQuantumErrorMitigator.

        Args:
            noise_levels: Noise levels for extrapolation. Defaults to
                config.NEURAL_QEM_NOISE_LEVELS.
            hidden_dim: Hidden dimension for error model. Defaults to
                config.NEURAL_QEM_HIDDEN_DIM.
            name: Layer name.
            **kwargs: Additional layer arguments.

        Raises:
            ValueError: If noise_levels is empty.
        """
        super().__init__(name=name, **kwargs)

        # Import config values with fallback
        self.noise_levels = tuple(
            noise_levels if noise_levels is not None else config.NEURAL_QEM_NOISE_LEVELS
        )
        self.hidden_dim = hidden_dim if hidden_dim is not None else config.NEURAL_QEM_HIDDEN_DIM

        if len(self.noise_levels) == 0:
            raise ValueError("noise_levels must not be empty")

        self.output_dim: int | None = None

        # Error corrector network (built dynamically based on input shape)
        self.dense1: tf.keras.layers.Dense | None = None
        self.dense2: tf.keras.layers.Dense | None = None
        self.output_layer: tf.keras.layers.Dense | None = None

        # Noise scale predictor
        self.noise_predictor: tf.keras.layers.Dense | None = None

    def build(self, input_shape) -> None:
        """Build the error correction network based on input shape.

        Args:
            input_shape: Shape of input tensor (batch, ..., features).
        """
        self.output_dim = int(input_shape[-1])

        # Error correction network
        self.dense1 = tf.keras.layers.Dense(
            self.output_dim * 2,
            activation="gelu",
            name=f"{self.name}_dense1",
            dtype=tf.float32,
        )
        self.dense2 = tf.keras.layers.Dense(
            self.output_dim * 2,
            activation="gelu",
            name=f"{self.name}_dense2",
            dtype=tf.float32,
        )
        self.output_layer = tf.keras.layers.Dense(
            self.output_dim,
            name=f"{self.name}_output",
            dtype=tf.float32,
        )

        # Noise scale predictor - predicts probability distribution over noise levels
        self.noise_predictor = tf.keras.layers.Dense(
            len(self.noise_levels),
            name=f"{self.name}_noise_predictor",
            dtype=tf.float32,
        )

        super().build(input_shape)

    def extrapolate_to_zero_noise(self, noisy_outputs: tf.Tensor) -> tf.Tensor:
        """Neural ZNE: learn the extrapolation to zero noise.

        Combines noise level prediction with learned correction to estimate
        the zero-noise output from noisy quantum layer outputs.

        Args:
            noisy_outputs: Noisy output tensor from quantum layer.

        Returns:
            Corrected output tensor with learned zero-noise extrapolation.
        """
        # Predict which noise level each output corresponds to
        noise_logits = self.noise_predictor(noisy_outputs)
        noise_weights = tf.nn.softmax(noise_logits, axis=-1)

        # Apply learned error correction
        x = self.dense1(noisy_outputs)
        x = self.dense2(x)
        correction = self.output_layer(x)

        # Residual connection weighted by noise confidence
        # Higher noise -> stronger correction
        noise_scale = tf.reduce_sum(
            noise_weights * tf.constant(self.noise_levels, dtype=tf.float32),
            axis=-1,
            keepdims=True,
        )
        # Normalize scale: scale=1.0 -> no correction, scale>1.0 -> add correction
        correction_weight = tf.clip_by_value((noise_scale - 1.0) / 2.0, 0.0, 1.0)

        corrected = noisy_outputs + correction * correction_weight

        return corrected

    def _apply_inference_correction(self, quantum_output: tf.Tensor) -> tf.Tensor:
        """Apply mild correction during inference.

        Args:
            quantum_output: Output tensor from quantum layer.

        Returns:
            Corrected output tensor (less aggressive than training).
        """
        x = self.dense1(quantum_output)
        x = self.dense2(x)
        correction = self.output_layer(x)
        # Apply a mild correction during inference
        # (less aggressive than training to avoid overcorrection)
        return quantum_output + correction * 0.5

    def call(
        self,
        quantum_output: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Mitigate errors in quantum layer outputs.

        During training, applies the full extrapolation pipeline to learn
        layer-specific error patterns. During inference, applies the
        learned correction directly.

        Uses tf.cond() for graph-mode compatibility when training is a
        symbolic tensor rather than a Python boolean.

        Args:
            quantum_output: Output tensor from quantum layer.
            training: Whether in training mode (can be tf.Tensor).

        Returns:
            Mitigated output tensor.
        """
        if not config.USE_NEURAL_QEM:
            return quantum_output

        # Convert training to tensor for graph-mode compatibility
        training_tensor = tf.cast(training, tf.bool)

        # Use tf.cond for graph-safe branching (training can be symbolic)
        return tf.cond(
            training_tensor,
            # During training: learn error model via extrapolation
            true_fn=lambda: self.extrapolate_to_zero_noise(quantum_output),
            # During inference: apply learned correction directly
            false_fn=lambda: self._apply_inference_correction(quantum_output),
        )

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config_dict = super().get_config()
        config_dict.update(
            {
                "noise_levels": self.noise_levels,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config_dict


class QuantumErrorMitigationWrapper(tf.keras.layers.Layer):
    """Wrapper layer that applies error mitigation to any quantum layer output.

    This wrapper can be applied to outputs from VQC layers, quantum attention,
    Q-SSM gates, or any other quantum-enhanced layer to automatically apply
    error mitigation.

    Supports two mitigation backends:
    1. VQEM C++ op (preferred when USE_VQEM=True and native ops available)
    2. NeuralQuantumErrorMitigator Python fallback

    Compatible with USE_QUANTUM_FIDELITY_LOSS for training.

    Attributes:
        inner_layer: The wrapped quantum layer (optional).
        mitigator: The NeuralQuantumErrorMitigator instance (fallback).
        use_vqem: Whether VQEM C++ ops are being used.

    Example:
        >>> vqc_layer = HybridVQCLayer(num_qubits=2)
        >>> wrapped = QuantumErrorMitigationWrapper(inner_layer=vqc_layer)
        >>> output = wrapped(inputs)  # Automatically mitigated
    """

    def __init__(
        self,
        inner_layer: tf.keras.layers.Layer | None = None,
        noise_levels: tuple[float, ...] | list[float] | None = None,
        hidden_dim: int | None = None,
        name: str = "qem_wrapper",
        **kwargs,
    ):
        """Initialize QuantumErrorMitigationWrapper.

        Args:
            inner_layer: Optional quantum layer to wrap. If None, this wrapper
                should be called directly on quantum layer outputs.
            noise_levels: Noise levels for the mitigator.
            hidden_dim: Hidden dimension for the mitigator.
            name: Layer name.
            **kwargs: Additional layer arguments.
        """
        super().__init__(name=name, **kwargs)
        self.inner_layer = inner_layer

        # Check if VQEM C++ ops are available
        self._use_vqem = False
        self._vqem_params: tf.Variable | None = None

        if config.USE_VQEM:
            try:
                from highnoon._native.ops.vqem_ops import create_vqem_params
                from highnoon._native.ops.vqem_ops import ops_available as vqem_ops_available
                from highnoon._native.ops.vqem_ops import vqem_forward as _vqem_forward

                if vqem_ops_available():
                    self._use_vqem = True
                    self._vqem_forward = _vqem_forward
                    self._vqem_params = create_vqem_params()
                    logger.info(
                        "[QEM] VQEM C++ ops available (inference only - Python fallback for training)"
                    )
            except (ImportError, RuntimeError) as e:
                logger.info(f"[QEM] VQEM C++ ops not available, using Python fallback: {e}")

        # Always create Python mitigator for training (VQEM C++ lacks gradient support)
        self.mitigator = NeuralQuantumErrorMitigator(
            noise_levels=noise_levels,
            hidden_dim=hidden_dim,
            name=f"{name}_mitigator",
        )

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        """Apply wrapped layer and error mitigation.

        Args:
            inputs: Input tensor.
            training: Whether in training mode.
            **kwargs: Additional arguments passed to inner layer.

        Returns:
            Mitigated output tensor.
        """
        if self.inner_layer is not None:
            quantum_output = self.inner_layer(inputs, training=training, **kwargs)
        else:
            quantum_output = inputs

        # Use Python mitigator during training (C++ ops lack gradient support)
        # Use VQEM C++ for inference only
        if training or not self._use_vqem or self._vqem_params is None:
            return self.mitigator(quantum_output, training=training)

        # Inference with VQEM C++ op
        return self._vqem_forward(quantum_output, self._vqem_params)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config_dict = super().get_config()
        config_dict.update(
            {
                "use_vqem": self._use_vqem,
            }
        )
        if self.mitigator is not None:
            config_dict.update(
                {
                    "noise_levels": self.mitigator.noise_levels,
                    "hidden_dim": self.mitigator.hidden_dim,
                }
            )
        if self.inner_layer is not None:
            config_dict["inner_layer"] = tf.keras.layers.serialize(self.inner_layer)
        return config_dict


__all__ = [
    "NeuralZNE",
    "ZNEMitigator",
    "NeuralQuantumErrorMitigator",
    "QuantumErrorMitigationWrapper",
    "UnifiedAlphaQubitDecoder",
]


# =============================================================================
# S11: Unified AlphaQubit Decoder - All Quantum Layer Outputs
# =============================================================================


class UnifiedAlphaQubitDecoder(tf.keras.layers.Layer):
    """S11: Unified quantum error correction for all quantum layer outputs.

    This layer implements AlphaQubit-style neural syndrome decoding that can
    be applied uniformly to outputs from VQC, QASA, QMamba, Q-SSM, and other
    quantum-enhanced layers. It combines:

    1. Neural QEM: Layer-specific error pattern learning
    2. Syndrome Decoder: AlphaQubit-inspired transformer for systematic errors
    3. Residual Correction: Gradient-stable output refinement

    Research Basis:
        Phase 61 (AlphaQubit-2) + Phase 129 (ML-QEM) + Phase 130 S11 Synergy

    Key Features:
        - Universal: Works with any quantum layer output
        - Layer-Aware: Learns layer-specific error patterns
        - Syndrome-Based: Detects and corrects systematic quantum errors
        - Differentiable: Full gradient flow for end-to-end training

    The decoder architecture uses self-attention to identify error syndromes:
        - Input: Quantum layer output [batch, ..., dim]
        - Syndrome Detection: MHA over feature dimension
        - Error Correction: Dense + residual

    Complexity: O(n · d²) for attention, O(n · d) for correction

    Args:
        num_attention_layers: Number of syndrome decoder attention layers.
        attention_heads: Number of attention heads for syndrome detection.
        hidden_dim: Hidden dimension for correction network.
        enabled_layers: Layer type names to apply decoding to.
        **kwargs: Additional layer arguments.

    Example:
        >>> decoder = UnifiedAlphaQubitDecoder(num_attention_layers=2)
        >>> corrected = decoder(vqc_output, layer_type="vqc")
    """

    def __init__(
        self,
        num_attention_layers: int | None = None,
        attention_heads: int = 4,
        hidden_dim: int | None = None,
        enabled_layers: tuple[str, ...] | None = None,
        name: str = "unified_alphaqubit",
        **kwargs,
    ):
        """Initialize UnifiedAlphaQubitDecoder.

        Args:
            num_attention_layers: Attention layers for syndrome detection.
                Defaults to config.ALPHAQUBIT_NUM_LAYERS.
            attention_heads: Number of attention heads.
            hidden_dim: Hidden dimension. Defaults to config.NEURAL_QEM_HIDDEN_DIM.
            enabled_layers: Layer types to decode. Defaults to
                config.ALPHAQUBIT_ENABLED_LAYERS.
            name: Layer name.
            **kwargs: Additional layer arguments.
        """
        super().__init__(name=name, **kwargs)

        # Configuration from config module with fallbacks
        self.num_attention_layers = (
            num_attention_layers
            if num_attention_layers is not None
            else getattr(config, "ALPHAQUBIT_NUM_LAYERS", 2)
        )
        self.attention_heads = attention_heads
        self.hidden_dim = (
            hidden_dim if hidden_dim is not None else getattr(config, "NEURAL_QEM_HIDDEN_DIM", 64)
        )
        self.enabled_layers = (
            enabled_layers
            if enabled_layers is not None
            else getattr(
                config,
                "ALPHAQUBIT_ENABLED_LAYERS",
                ("vqc", "qmamba", "qssm", "qasa"),
            )
        )

        # Check if enabled globally
        self._enabled = getattr(config, "USE_UNIFIED_ALPHAQUBIT", True)

        # Use C++ ops if available
        self._use_native_ops = _ALPHAQUBIT_NATIVE_AVAILABLE and self._enabled
        if self._use_native_ops:
            logger.info("[AlphaQubit] S11 using C++ native ops")
        else:
            logger.info("[AlphaQubit] S11 using Python fallback")

        # Syndrome attention layers (built dynamically)
        self.syndrome_attention_layers: list[tf.keras.layers.MultiHeadAttention] = []
        self.syndrome_ffn_layers: list[tf.keras.layers.Dense] = []
        self.layer_norms: list[tf.keras.layers.LayerNormalization] = []

        # Error correction output
        self.correction_dense1: tf.keras.layers.Dense | None = None
        self.correction_dense2: tf.keras.layers.Dense | None = None
        self.output_gate: tf.keras.layers.Dense | None = None

        # Layer type embedding for layer-specific decoding
        self.layer_type_embedding: dict[str, tf.Variable] = {}

        # Built flag
        self._built_for_dim: int | None = None

        # C++ op weights (created in build)
        self._native_weights: dict[str, tf.Variable] | None = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build decoder layers based on input shape.

        Args:
            input_shape: Shape of input tensor [batch, ..., features].
        """
        feature_dim = int(input_shape[-1])
        self._built_for_dim = feature_dim

        # Syndrome attention layers
        for i in range(self.num_attention_layers):
            attn = tf.keras.layers.MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=max(16, feature_dim // self.attention_heads),
                name=f"{self.name}_syndrome_attn_{i}",
            )
            self.syndrome_attention_layers.append(attn)

            ffn = tf.keras.layers.Dense(
                self.hidden_dim,
                activation="gelu",
                name=f"{self.name}_syndrome_ffn_{i}",
            )
            self.syndrome_ffn_layers.append(ffn)

            ln = tf.keras.layers.LayerNormalization(
                epsilon=1e-6,
                name=f"{self.name}_ln_{i}",
            )
            self.layer_norms.append(ln)

        # Error correction layers
        self.correction_dense1 = tf.keras.layers.Dense(
            self.hidden_dim,
            activation="gelu",
            name=f"{self.name}_corr_dense1",
        )
        self.correction_dense2 = tf.keras.layers.Dense(
            feature_dim,
            name=f"{self.name}_corr_dense2",
        )

        # Output gate for residual mixing
        self.output_gate = tf.keras.layers.Dense(
            feature_dim,
            activation="sigmoid",
            name=f"{self.name}_output_gate",
        )

        # Layer type embeddings (learnable bias for each layer type)
        for layer_type in self.enabled_layers:
            self.layer_type_embedding[layer_type] = self.add_weight(
                name=f"{self.name}_embed_{layer_type}",
                shape=(feature_dim,),
                initializer="zeros",
                trainable=True,
            )

        # Create C++ op weights if using native implementation
        if self._use_native_ops and create_alphaqubit_correct_weights is not None:
            self._native_weights = create_alphaqubit_correct_weights(
                feature_dim=feature_dim,
                hidden_dim=self.hidden_dim,
                num_attn_layers=self.num_attention_layers,
            )
            # Register weights as layer weights for training
            for name, weight in self._native_weights.items():
                setattr(self, f"_native_{name}", weight)

        super().build(input_shape)

    def _apply_syndrome_detection(
        self,
        x: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply syndrome-based error detection via attention.

        Uses self-attention to identify correlated error patterns
        in the quantum layer output, similar to AlphaQubit's syndrome
        detection mechanism.

        Args:
            x: Input tensor [batch, features] or [batch, seq, features].
            training: Whether in training mode.

        Returns:
            Syndrome-enhanced tensor with error pattern information.
        """
        # Ensure 3D for attention: [batch, seq=1, features]
        original_shape = tf.shape(x)
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)  # [batch, 1, features]

        # Apply syndrome attention layers
        for i in range(self.num_attention_layers):
            # Self-attention for syndrome detection
            attn_out = self.syndrome_attention_layers[i](query=x, value=x, key=x, training=training)
            x = self.layer_norms[i](x + attn_out)

            # FFN for syndrome processing
            ffn_out = self.syndrome_ffn_layers[i](x)
            # Project back to feature dim
            ffn_out = (
                tf.keras.layers.Dense(self._built_for_dim, name=f"proj_{i}")(ffn_out)
                if ffn_out.shape[-1] != self._built_for_dim
                else ffn_out
            )
            x = x + ffn_out

        # Restore original shape if needed
        if len(original_shape) == 2:
            x = tf.squeeze(x, axis=1)

        return x

    def call(
        self,
        quantum_output: tf.Tensor,
        layer_type: str = "vqc",
        training: bool = False,
    ) -> tf.Tensor:
        """Apply unified AlphaQubit decoding to quantum layer output.

        Args:
            quantum_output: Output tensor from quantum layer.
            layer_type: Type of quantum layer ("vqc", "qmamba", "qssm", "qasa").
            training: Whether in training mode.

        Returns:
            Error-corrected quantum output.
        """
        if not self._enabled:
            return quantum_output

        # Skip if layer type not enabled
        if layer_type not in self.enabled_layers:
            return quantum_output

        # Add layer-type-specific bias
        if layer_type in self.layer_type_embedding and self._built_for_dim is not None:
            quantum_output = quantum_output + self.layer_type_embedding[layer_type]

        # Use C++ native ops if available
        if (
            self._use_native_ops
            and self._native_weights is not None
            and alphaqubit_correct is not None
        ):
            return alphaqubit_correct(
                quantum_output,
                self._native_weights["qkv_weights"],
                self._native_weights["proj_weights"],
                self._native_weights["corr_w1"],
                self._native_weights["corr_w2"],
                self._native_weights["gate_w"],
                self._native_weights["gate_b"],
                feature_dim=self._built_for_dim,
                hidden_dim=self.hidden_dim,
                num_attn_layers=self.num_attention_layers,
                num_heads=self.attention_heads,
            )

        # Python fallback: Detect error syndromes
        syndrome_features = self._apply_syndrome_detection(quantum_output, training)

        # Compute correction
        correction = self.correction_dense1(syndrome_features)
        correction = self.correction_dense2(correction)

        # Gated residual: blend original and corrected
        gate = self.output_gate(syndrome_features)
        corrected = gate * correction + (1 - gate) * quantum_output

        return corrected

    def get_config(self) -> dict:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config_dict = super().get_config()
        config_dict.update(
            {
                "num_attention_layers": self.num_attention_layers,
                "attention_heads": self.attention_heads,
                "hidden_dim": self.hidden_dim,
                "enabled_layers": self.enabled_layers,
            }
        )
        return config_dict


def create_unified_alphaqubit_decoder(
    **kwargs,
) -> UnifiedAlphaQubitDecoder | None:
    """Factory function for UnifiedAlphaQubitDecoder.

    Returns None if USE_UNIFIED_ALPHAQUBIT is False.

    Args:
        **kwargs: Arguments for UnifiedAlphaQubitDecoder.

    Returns:
        UnifiedAlphaQubitDecoder instance or None if disabled.
    """
    if not getattr(config, "USE_UNIFIED_ALPHAQUBIT", True):
        logger.debug("[AlphaQubit] S11 disabled via config")
        return None

    return UnifiedAlphaQubitDecoder(**kwargs)
