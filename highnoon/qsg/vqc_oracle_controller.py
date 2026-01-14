# highnoon/qsg/vqc_oracle_controller.py
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

"""Phase A3: VQC-Enhanced Adaptive Control for QSG.

Implements closed-loop VQC-based control of QSG generation parameters.
Dynamically adjusts beta, Grover iterations, and coherence range based
on context entropy and Hopfield energy.

The controller uses a lightweight VQC circuit (4-8 qubits, 2-4 layers)
to map entropy/energy inputs to optimal parameter settings.
"""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.config import QSG_COHERENCE_RANGE, QSG_GROVER_ITERATIONS, QSG_HOPFIELD_BETA

logger = logging.getLogger(__name__)


class VQCOracleController(layers.Layer):
    """VQC-based adaptive controller for QSG generation parameters.

    Maps context entropy and Hopfield energy to optimal QSG parameters:
    - beta: Hopfield inverse temperature (sharpness)
    - grover_iterations: Amplitude amplification steps
    - coherence_range: Position coherence window

    Uses a small learnable MLP that mimics VQC behavior (rotation + entanglement).
    This allows training with standard backprop while maintaining the VQC structure.

    Attributes:
        beta_range: (min, max) for output beta.
        grover_range: (min, max) for grover iterations (rounded to int).
        coherence_range: (min, max) for coherence window (-1 = all).
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        beta_range: tuple[float, float] = (0.5, 4.0),
        grover_range: tuple[int, int] = (1, 5),
        coherence_range_limits: tuple[int, int] = (16, 256),
        name: str = "vqc_oracle_controller",
        **kwargs,
    ):
        """Initialize VQCOracleController.

        Args:
            hidden_dim: Hidden layer dimension for controller MLP.
            beta_range: (min, max) for output beta parameter.
            grover_range: (min, max) for Grover iterations.
            coherence_range_limits: (min, max) for coherence range.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.
        """
        super().__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.beta_range = beta_range
        self.grover_range = grover_range
        self.coherence_range_limits = coherence_range_limits

        # VQC-like MLP: input -> rotation-like hidden -> entanglement-like mixing -> output
        self.input_proj = layers.Dense(hidden_dim, activation="tanh", name=f"{name}_input")
        self.mixing = layers.Dense(hidden_dim, activation="tanh", name=f"{name}_mix")

        # Output heads for each parameter (sigmoid scaled to range)
        self.beta_head = layers.Dense(1, activation="sigmoid", name=f"{name}_beta")
        self.grover_head = layers.Dense(1, activation="sigmoid", name=f"{name}_grover")
        self.coherence_head = layers.Dense(1, activation="sigmoid", name=f"{name}_coherence")

    def call(
        self,
        entropy: tf.Tensor,
        energy: tf.Tensor | None = None,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Compute optimal QSG parameters from context statistics.

        Args:
            entropy: Context entropy [batch] or [batch, seq-1].
            energy: Optional Hopfield energy [batch] or [batch, seq].
            training: Whether in training mode.

        Returns:
            Dict with keys:
                - 'beta': Hopfield inverse temperature [batch]
                - 'grover_iterations': Grover iterations [batch] (float, round for use)
                - 'coherence_range': Coherence window [batch] (float, round for use)
        """
        # Aggregate entropy to batch-level
        if len(entropy.shape) > 1:
            entropy = tf.reduce_mean(entropy, axis=-1)  # [batch]

        # Prepare input features
        if energy is not None:
            if len(energy.shape) > 1:
                energy = tf.reduce_mean(energy, axis=-1)  # [batch]
            features = tf.stack([entropy, energy], axis=-1)  # [batch, 2]
        else:
            features = tf.expand_dims(entropy, axis=-1)  # [batch, 1]

        # VQC-like forward pass
        h = self.input_proj(features)
        h = self.mixing(h)

        # Compute parameters with range scaling
        beta_raw = self.beta_head(h)  # [batch, 1]
        grover_raw = self.grover_head(h)  # [batch, 1]
        coherence_raw = self.coherence_head(h)  # [batch, 1]

        # Scale to ranges
        beta_min, beta_max = self.beta_range
        beta = beta_min + (beta_max - beta_min) * tf.squeeze(beta_raw, axis=-1)

        grover_min, grover_max = self.grover_range
        grover_float = grover_min + (grover_max - grover_min) * tf.squeeze(grover_raw, axis=-1)

        coh_min, coh_max = self.coherence_range_limits
        coherence_float = coh_min + (coh_max - coh_min) * tf.squeeze(coherence_raw, axis=-1)

        return {
            "beta": beta,
            "grover_iterations": grover_float,
            "coherence_range": coherence_float,
        }

    def get_discrete_params(
        self,
        entropy: tf.Tensor,
        energy: tf.Tensor | None = None,
    ) -> tuple[float, int, int]:
        """Get discrete parameter values (for inference).

        Args:
            entropy: Context entropy tensor.
            energy: Optional Hopfield energy tensor.

        Returns:
            Tuple of (beta, grover_iterations, coherence_range) as Python scalars.
        """
        params = self.call(entropy, energy, training=False)

        beta = float(tf.reduce_mean(params["beta"]).numpy())
        grover = int(tf.round(tf.reduce_mean(params["grover_iterations"])).numpy())
        coherence = int(tf.round(tf.reduce_mean(params["coherence_range"])).numpy())

        return beta, grover, coherence

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "beta_range": self.beta_range,
                "grover_range": self.grover_range,
                "coherence_range_limits": self.coherence_range_limits,
            }
        )
        return config


def compute_context_entropy(hidden_states: tf.Tensor) -> tf.Tensor:
    """Compute approximate entropy of context representations.

    Uses spectral entropy of hidden state covariance as a proxy
    for information density.

    Args:
        hidden_states: Context hidden states [batch, seq, dim].

    Returns:
        Entropy estimate [batch] in range [0, 1].
    """
    # Compute covariance along sequence dimension
    # hidden: [B, L, D]
    mean = tf.reduce_mean(hidden_states, axis=1, keepdims=True)
    centered = hidden_states - mean

    # Simplified: use variance as entropy proxy
    var = tf.reduce_mean(centered**2, axis=[1, 2])  # [batch]

    # Normalize to [0, 1] using sigmoid
    entropy = tf.nn.sigmoid(var - 1.0)  # 1.0 is a typical variance

    return entropy


def adaptive_qsg_parameters(
    context: tf.Tensor,
    controller: VQCOracleController | None = None,
    energy: tf.Tensor | None = None,
) -> tuple[float, int, int]:
    """Get adaptive QSG parameters for given context.

    Convenience function that computes entropy and runs controller.

    Args:
        context: Context hidden states [batch, seq, dim].
        controller: Optional VQCOracleController (uses defaults if None).
        energy: Optional Hopfield energy from vocabulary projection.

    Returns:
        Tuple of (beta, grover_iterations, coherence_range).
    """
    if controller is None:
        # Return defaults
        return QSG_HOPFIELD_BETA, QSG_GROVER_ITERATIONS, QSG_COHERENCE_RANGE

    entropy = compute_context_entropy(context)
    return controller.get_discrete_params(entropy, energy)
