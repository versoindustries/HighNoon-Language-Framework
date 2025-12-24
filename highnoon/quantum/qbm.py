# src/quantum/qbm.py
#
# Quantum Boltzmann Machine with simulated quantum annealing for MoE routing.
# Phase 7.2: Stochastic sampling with Metropolis-Hastings annealing.

from __future__ import annotations

import logging
import math

import tensorflow as tf

from highnoon._native.ops.qbm_sample import (
    compute_expert_entropy,
    compute_exploration_ratio,
    qbm_sample,
)
from highnoon.models.utils.control_vars import ControlVarMixin

from .layers import QuantumEnergyLayer

logger = logging.getLogger(__name__)


class QuantumBoltzmannMachine(ControlVarMixin, tf.keras.layers.Layer):
    """
    Quantum Boltzmann Machine (QBM) with simulated quantum annealing for MoE routing.

    Phase 7.2 Implementation:
    - Uses qbm_sample_op for stochastic expert selection via annealing
    - Metropolis-Hastings sampling with temperature schedule
    - Exploration: H(t) = (1-s(t)) * H_0 + s(t) * H_f
    - REINFORCE gradients for policy learning

    This addresses expert collapse by:
    - High initial temperature promotes exploration across all experts
    - Annealing allows escape from local minima
    - Stochastic sampling prevents deterministic collapse to single expert
    """

    def __init__(
        self,
        num_visible: int,
        num_experts: int,
        hidden_dim: int = 16,
        initial_inverse_temp: float = 1.0,
        energy_scale: float = 3.0,
        num_qubits: int | None = None,
        # Phase 7.2: Annealing parameters
        use_annealing: bool = True,
        temperature_init: float = 1.0,
        temperature_final: float = 0.1,
        num_annealing_steps: int = 100,
        **kwargs,
    ):
        super().__init__(dtype="float32", **kwargs)
        if num_visible <= 0:
            raise ValueError("num_visible must be positive.")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")

        self.num_visible = num_visible
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self._requested_qubits = num_qubits

        # Phase 7.2: Annealing configuration
        self.use_annealing = use_annealing
        self.temperature_init = temperature_init
        self.temperature_final = temperature_final
        self.num_annealing_steps = num_annealing_steps

        self.feature_encoder = tf.keras.layers.Dense(
            hidden_dim,
            activation=tf.nn.tanh,
            use_bias=True,
            dtype="float32",
            name=f"{self.name or 'qbm'}_feature_encoder",
        )
        self.energy_layer = QuantumEnergyLayer(
            energy_scale=energy_scale,
            num_qubits=self._resolve_num_qubits(),
            name=f"{self.name or 'qbm'}_energy",
        )
        self.inverse_temperature = self.add_weight(
            name="inverse_temperature",
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_inverse_temp),
            trainable=True,
            dtype=tf.float32,
        )
        self.register_control_var("inverse_temperature", self.inverse_temperature)
        self._avg_energy_metric = None
        self._beta_metric = None
        # Phase 7.2: Additional metrics for annealing
        self._entropy_metric = None
        self._exploration_ratio_metric = None
        self._annealing_counter = None

    def _resolve_num_qubits(self) -> int:
        if self._requested_qubits is not None:
            return max(1, int(self._requested_qubits))
        required = max(2, self.num_experts)
        return max(1, int(math.ceil(math.log2(required))))

    def build(self, input_shape):
        visible_dim = int(input_shape[-1])
        if visible_dim != self.num_visible:
            raise ValueError(f"Expected input dimension {self.num_visible}, got {visible_dim}.")
        if not self.feature_encoder.built:
            self.feature_encoder.build(tf.TensorShape([None, visible_dim]))
        if self._avg_energy_metric is None:
            metric_prefix = self.name or "qbm"
            self._avg_energy_metric = tf.keras.metrics.Mean(name=f"{metric_prefix}_avg_energy")
            self._beta_metric = tf.keras.metrics.Mean(name=f"{metric_prefix}_beta")
            # Phase 7.2: Metrics for monitoring exploration
            self._entropy_metric = tf.keras.metrics.Mean(name=f"{metric_prefix}_expert_entropy")
            self._exploration_ratio_metric = tf.keras.metrics.Mean(
                name=f"{metric_prefix}_exploration_ratio"
            )
        # Phase 7.2: Counter for deterministic seed generation
        if self._annealing_counter is None:
            self._annealing_counter = self.add_weight(
                name="annealing_counter",
                shape=(),
                initializer=tf.keras.initializers.Constant(0),
                trainable=False,
                dtype=tf.int32,
            )
        super().build(input_shape)

    @property
    def metrics(self):
        base_metrics = super().metrics
        trackers = []
        if self._avg_energy_metric is not None:
            trackers.append(self._avg_energy_metric)
        if self._beta_metric is not None:
            trackers.append(self._beta_metric)
        # Phase 7.2: Include annealing metrics
        if self._entropy_metric is not None:
            trackers.append(self._entropy_metric)
        if self._exploration_ratio_metric is not None:
            trackers.append(self._exploration_ratio_metric)
        return base_metrics + trackers

    def call(self, inputs: tf.Tensor, training: bool = False):
        """
        Forward pass with optional quantum annealing sampling.

        Phase 7.2:
        - Training mode: Use qbm_sample_op for stochastic expert selection
        - Inference mode: Use deterministic softmax (faster)

        Returns:
            probabilities: [batch, num_experts] - routing probabilities
            energies: [batch, num_experts] - expert affinity energies
            logits: [batch, num_experts] - router logits for gradient computation
        """
        training_tensor = tf.constant(training, dtype=tf.bool)
        x = tf.reshape(inputs, [-1, self.num_visible])
        encoded = self.feature_encoder(x)
        batch_size = tf.shape(encoded)[0]

        # [B, H] -> [B, 1, H]
        encoded_expanded = tf.expand_dims(encoded, axis=1)
        # [B, E, H]
        encoded_tiled = tf.tile(encoded_expanded, [1, self.num_experts, 1])

        # [E, E] -> [1, E, E] -> [B, E, E]
        expert_indicators = tf.one_hot(
            tf.range(self.num_experts), self.num_experts, dtype=tf.float32
        )
        indicators_tiled = tf.tile(tf.expand_dims(expert_indicators, axis=0), [batch_size, 1, 1])

        # [B, E, H+E] -> reshape to [B*E, H+E] for the dense layer inside energy_layer
        combined_features = tf.concat([encoded_tiled, indicators_tiled], axis=-1)
        flat_features = tf.reshape(combined_features, [-1, self.hidden_dim + self.num_experts])

        # [B*E, 1] -> [B, E]
        flat_energies = self.energy_layer(flat_features, training=training_tensor)
        energies = tf.reshape(flat_energies, [batch_size, self.num_experts])

        beta = tf.nn.softplus(self.inverse_temperature)
        logits = -beta * energies

        # Phase 7.2: Stochastic routing via annealing during training
        if training and self.use_annealing:
            # Increment counter for deterministic but varying seed
            self._annealing_counter.assign_add(1)
            seed = tf.cast(self._annealing_counter, tf.int32)

            # Use qbm_sample_op for stochastic expert selection
            expert_assignments, sample_log_probs, annealing_energies = qbm_sample(
                energy_matrix=energies,
                temperature_init=self.temperature_init,
                temperature_final=self.temperature_final,
                num_annealing_steps=self.num_annealing_steps,
                seed=seed,
            )

            # Convert sampled assignments to one-hot probabilities for routing
            # This maintains interface compatibility with downstream MoE code
            probabilities = tf.one_hot(expert_assignments, self.num_experts, dtype=tf.float32)

            # Update exploration metrics
            entropy = compute_expert_entropy(expert_assignments, self.num_experts)
            exploration_ratio = compute_exploration_ratio(expert_assignments, self.num_experts)

            if self._entropy_metric is not None:
                self._entropy_metric.update_state(entropy)
            if self._exploration_ratio_metric is not None:
                self._exploration_ratio_metric.update_state(exploration_ratio)
        else:
            # Inference mode: deterministic softmax routing (faster)
            probabilities = tf.nn.softmax(logits, axis=-1)

        # Standard metrics
        if self._avg_energy_metric is not None:
            self._avg_energy_metric.update_state(tf.reduce_mean(energies))
        if self._beta_metric is not None:
            self._beta_metric.update_state(beta)

        return probabilities, energies, logits
