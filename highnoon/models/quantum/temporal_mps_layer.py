"""Temporal Matrix Product State Layer for Degradation Modeling.

This module implements a temporal sequence-to-sequence model for predicting
electrode degradation trajectories in electrohydrodynamic (EHD) thrusters.

Architecture:
    Hybrid MPS+GRU design combining quantum-inspired spatial structure with
    proven temporal modeling via GRU cells. Predicts future states (r_e, J, T)
    given historical sequences with physics constraints enforced.

Physics:
    Electrode erosion: dr_e/dt = -K·J²·exp(-E_a/kT)
    Constraints: dr_e/dt ≤ 0 (monotonic erosion), T ∈ [1K, 10,000K]

References:
    - HSMN EHD Architecture Roadmap: Phase 1, Task 1.2.1
    - MPS Infrastructure: src/models/quantum/mps_layer.py
    - Degradation Physics: Whitepaper Section 8.2.2

Author: HSMN Roadmap Phase Executor
Date: 2025-11-25
"""

import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class TemporalMPSLayer(keras.layers.Layer):
    """Temporal Matrix Product State layer for sequence-to-sequence prediction.

    This layer predicts future electrode degradation states given a historical
    sequence of (electrode_radius, current_density, temperature) measurements.

    Architecture:
        1. Feature embedding: Dense(tanh) to project inputs to hidden space
        2. Temporal modeling: GRU cell for sequential dependencies
        3. Spatial structure: MPS placeholder (future: encode spatial correlations)
        4. Autoregressive decoder: 2-layer Dense network for multi-step prediction
        5. Physics constraints: Hard enforcement of physical laws

    Input shape:
        [batch, seq_len, 3] where features = [r_e, J, T]
        - r_e: Electrode radius (m), typically 0.1-1.0 mm
        - J: Current density (A/m²), typically 1e3-1e9
        - T: Temperature (K), range [1, 10,000]

    Output shape:
        [batch, prediction_horizon, 3]

    Args:
        spatial_bond_dim: Bond dimension for spatial MPS (future enhancement).
        temporal_bond_dim: Hidden dimension for GRU temporal modeling.
        prediction_horizon: Number of future timesteps to predict.
        feature_dim: Dimension of embedded feature space (default: 16).
        enforce_physics: If True, apply hard physics constraints to predictions.
        J_max: Maximum allowed current density (A/m²), default 1e9.
        name: Layer name.

    Example:
        >>> layer = TemporalMPSLayer(
        ...     spatial_bond_dim=4,
        ...     temporal_bond_dim=8,
        ...     prediction_horizon=10
        ... )
        >>> # Historical sequence: 20 timesteps
        >>> inputs = tf.random.normal([32, 20, 3])  # [batch, seq_len, features]
        >>> outputs = layer(inputs)  # [32, 10, 3] - predict next 10 timesteps
    """

    def __init__(
        self,
        spatial_bond_dim: int = 4,
        temporal_bond_dim: int = 8,
        prediction_horizon: int = 10,
        feature_dim: int = 16,
        enforce_physics: bool = True,
        J_max: float = 1e9,
        name: str = "temporal_mps",
        **kwargs,
    ):
        """Initialize TemporalMPSLayer."""
        super().__init__(name=name, **kwargs)

        # Architecture hyperparameters
        self.spatial_bond_dim = spatial_bond_dim
        self.temporal_bond_dim = temporal_bond_dim
        self.prediction_horizon = prediction_horizon
        self.feature_dim = feature_dim
        self.enforce_physics = enforce_physics
        self.J_max = J_max

        # Input feature dimension (r_e, J, T)
        self.input_dim = 3

        # Layers (will be built in build())
        self.feature_embedding = None
        self.gru_cell = None
        self.decoder_layer1 = None
        self.decoder_layer2 = None

        logger.info(
            f"TemporalMPSLayer initialized: χ_spatial={spatial_bond_dim}, "
            f"χ_temporal={temporal_bond_dim}, horizon={prediction_horizon}"
        )

    def build(self, input_shape):
        """Build layer components.

        Args:
            input_shape: TensorShape [batch, seq_len, 3]
        """
        # Feature embedding: project inputs to higher-dimensional space
        self.feature_embedding = keras.layers.Dense(
            self.feature_dim, activation="tanh", name=f"{self.name}_feature_embedding"
        )

        # Temporal modeling: GRU for sequential dependencies
        self.gru_layer = keras.layers.GRU(
            self.temporal_bond_dim,
            return_sequences=False,  # Return only final hidden state
            return_state=True,  # Return both output and state
            name=f"{self.name}_gru_layer",
        )

        # Autoregressive decoder: predict future states
        self.decoder_layer1 = keras.layers.Dense(
            32, activation="relu", name=f"{self.name}_decoder_layer1"
        )
        self.decoder_layer2 = keras.layers.Dense(
            self.input_dim,  # Output: (r_e, J, T)
            activation=None,  # Linear output (constraints applied later)
            name=f"{self.name}_decoder_layer2",
        )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass: predict future degradation trajectory.

        Args:
            inputs: Historical sequence [batch, seq_len, 3].
            training: Training mode flag.

        Returns:
            predictions: Future states [batch, prediction_horizon, 3].
        """
        # Input validation
        if inputs.shape.rank != 3:
            raise ValueError(f"Expected 3D input [batch, seq_len, 3], got shape {inputs.shape}")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features (r_e, J, T), " f"got {inputs.shape[-1]}"
            )

        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Ensure float32
        inputs = tf.cast(inputs, tf.float32)

        # Phase 1: Feature embedding for entire sequence
        # Reshape to apply embedding: [batch, seq_len, 3] → [batch*seq_len, 3]
        batch_seq = batch_size * seq_len
        inputs_flat = tf.reshape(inputs, [batch_seq, 3])

        # Apply embedding: [batch*seq_len, 3] → [batch*seq_len, feature_dim]
        embedded_flat = self.feature_embedding(inputs_flat)

        # Reshape back: [batch*seq_len, feature_dim] → [batch, seq_len, feature_dim]
        embedded_sequence = tf.reshape(embedded_flat, [batch_size, seq_len, self.feature_dim])

        # Phase 2: Encode historical sequence via GRU
        # Process sequence: [batch, seq_len, feature_dim] → [batch, temporal_bond_dim]
        _, hidden_state = self.gru_layer(embedded_sequence)
        # hidden_state now contains compressed history: [batch, temporal_bond_dim]

        # Phase 3: Autoregressive decoding - predict future states
        predictions = []
        current_state = inputs[:, -1, :]  # Start from last known state [batch, 3]

        # Concatenate hidden state with horizon index for temporal context
        for h in range(self.prediction_horizon):
            # Create input for decoder: [hidden_state, current_state, horizon_index]
            # This gives the decoder temporal context for each prediction step
            horizon_feature = (
                tf.ones([batch_size, 1], dtype=tf.float32)
                * float(h)
                / float(self.prediction_horizon)
            )
            decoder_input = tf.concat([hidden_state, current_state, horizon_feature], axis=-1)

            # Decode: [batch, temporal_bond_dim + 3 + 1] → [batch, 32] → [batch, 3]
            decoded = self.decoder_layer1(decoder_input)
            next_state_raw = self.decoder_layer2(decoded)  # [batch, 3]

            # Apply physics constraints
            if self.enforce_physics:
                next_state = self._apply_physics_constraints(current_state, next_state_raw)
            else:
                next_state = next_state_raw

            predictions.append(next_state)

            # Update for next prediction (autoregressive)
            current_state = next_state

        # Stack predictions: list of [batch, 3] → [batch, horizon, 3]
        predictions = tf.stack(predictions, axis=1)

        return predictions

    def _apply_physics_constraints(
        self, current_state: tf.Tensor, next_state_raw: tf.Tensor
    ) -> tf.Tensor:
        """Apply hard physics constraints to predictions.

        Constraints:
            1. Monotonic erosion: r_e(t+1) ≤ r_e(t) (electrodes never grow)
            2. Temperature bounds: 1K ≤ T ≤ 10,000K
            3. Current density bounds: 0 ≤ J ≤ J_max

        Args:
            current_state: Current state [batch, 3] = [r_e, J, T]
            next_state_raw: Raw prediction [batch, 3]

        Returns:
            next_state_constrained: Physics-compliant prediction [batch, 3]
        """
        r_e_current = current_state[:, 0]  # [batch]
        current_state[:, 1]
        current_state[:, 2]

        r_e_next_raw = next_state_raw[:, 0]
        J_next_raw = next_state_raw[:, 1]
        T_next_raw = next_state_raw[:, 2]

        # Constraint 1: Monotonic erosion (r_e can only decrease)
        r_e_next = tf.minimum(r_e_next_raw, r_e_current)
        # Also ensure r_e > 0 (electrode still exists)
        r_e_next = tf.maximum(r_e_next, 1e-6)  # Min radius: 1 micron

        # Constraint 2: Temperature bounds [1K, 10,000K]
        T_next = tf.clip_by_value(T_next_raw, 1.0, 10000.0)

        # Constraint 3: Current density bounds [0, J_max]
        J_next = tf.clip_by_value(J_next_raw, 0.0, self.J_max)

        # Stack back into [batch, 3]
        next_state_constrained = tf.stack([r_e_next, J_next, T_next], axis=1)

        return next_state_constrained

    def get_config(self) -> dict:
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "spatial_bond_dim": self.spatial_bond_dim,
                "temporal_bond_dim": self.temporal_bond_dim,
                "prediction_horizon": self.prediction_horizon,
                "feature_dim": self.feature_dim,
                "enforce_physics": self.enforce_physics,
                "J_max": self.J_max,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict):
        """Create layer from configuration."""
        return cls(**config)


def generate_synthetic_erosion_sequence(
    num_samples: int,
    seq_len: int,
    erosion_rate: float = 1e-9,
    noise_level: float = 0.05,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic electrode erosion data for testing.

    Physics model (simplified):
        r_e(t) = r_0 · exp(-K·t)
        J(t) = J_0 + noise
        T(t) = T_0 + ΔT·(1 - r_e(t)/r_0) + noise

    Args:
        num_samples: Number of sequences to generate.
        seq_len: Length of each sequence (timesteps).
        erosion_rate: Exponential decay constant K (1/s).
        noise_level: Relative noise magnitude (0-1).
        seed: Random seed for reproducibility.

    Returns:
        sequences: Historical data [num_samples, seq_len, 3]
        targets: Future states [num_samples, 1, 3] (next timestep)
    """
    if seed is not None:
        np.random.seed(seed)

    # Initial conditions
    r_0 = 1e-3  # 1 mm initial radius
    J_0 = 1e6  # 1 MA/m² baseline current density
    T_0 = 300.0  # 300 K ambient temperature

    sequences = np.zeros((num_samples, seq_len, 3), dtype=np.float32)
    targets = np.zeros((num_samples, 1, 3), dtype=np.float32)

    for i in range(num_samples):
        # Time vector: 0 to seq_len (arbitrary units)
        t = np.arange(seq_len + 1)

        # Electrode radius: exponential decay
        r_e = r_0 * np.exp(-erosion_rate * t)

        # Current density: constant + noise
        J = J_0 * (1.0 + noise_level * np.random.randn(seq_len + 1))
        J = np.clip(J, 0.0, 1e9)  # Physical bounds

        # Temperature: increases as electrode erodes (less thermal mass)
        T = T_0 + 100.0 * (1.0 - r_e / r_0) + noise_level * 50.0 * np.random.randn(seq_len + 1)
        T = np.clip(T, 1.0, 10000.0)  # Physical bounds

        # Sequence: first seq_len timesteps
        sequences[i, :, 0] = r_e[:seq_len]
        sequences[i, :, 1] = J[:seq_len]
        sequences[i, :, 2] = T[:seq_len]

        # Target: next timestep (seq_len + 1)
        targets[i, 0, 0] = r_e[seq_len]
        targets[i, 0, 1] = J[seq_len]
        targets[i, 0, 2] = T[seq_len]

    return sequences, targets
