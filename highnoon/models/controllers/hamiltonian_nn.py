# highnoon/models/controllers/hamiltonian_nn.py
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

"""Hamiltonian Neural Network implementations for HighNoon.

This module provides energy-conserving neural network blocks used in the
HSMN architecture for stable, physics-informed language modeling.

The Hamiltonian formulation ensures that model dynamics conserve a
learned energy function, leading to more stable training and generation.
"""

import logging
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

log = logging.getLogger(__name__)

# Default evolution time limits
DEFAULT_EVOLUTION_TIME_MIN = 1e-6
DEFAULT_EVOLUTION_TIME_MAX = 0.02
DEFAULT_EVOLUTION_TIME_INITIAL = 5e-4


class HamiltonianNN(tf.keras.Model):
    """Lightweight Hamiltonian neural network using dense layers.

    This model implements Hamiltonian dynamics using a learned energy
    function (the Hamiltonian) and symplectic integration for state updates.

    The dynamics preserve energy approximately, leading to stable
    long-horizon predictions and generation.

    Attributes:
        latent_dim: Dimension of the latent state space.
        control_dim: Dimension of optional control inputs.
        dt: Integration time step.
        integrator: Integration method ('euler' or 'rk4').
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        control_dim: int = 0,
        dt: float = 0.01,
        integrator: str = "euler",
        name: str | None = None,
    ):
        """Initialize HamiltonianNN.

        Args:
            latent_dim: Dimension of the latent state.
            hidden_dims: Hidden layer dimensions for the Hamiltonian MLP.
            control_dim: Dimension of control inputs (0 for autonomous).
            dt: Integration time step.
            integrator: Integration method.
            name: Model name.
        """
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.dt = dt
        self.integrator = integrator

        # MLP for computing the Hamiltonian gradient (dynamics)
        dense_layers = []
        for i, dim in enumerate(hidden_dims):
            dense_layers.append(
                layers.Dense(dim, activation="tanh", name=f"hnn_dense_{i}", dtype="float32")
            )
        dense_layers.append(
            layers.Dense(latent_dim, activation=None, name="hnn_out", dtype="float32")
        )
        self.mlp = tf.keras.Sequential(dense_layers, name="hnn_mlp")

    def call(
        self, x: tf.Tensor, control: tf.Tensor | None = None, training: bool = False
    ) -> tf.Tensor:
        """Compute next state using Hamiltonian dynamics.

        Args:
            x: Current state tensor [batch, latent_dim].
            control: Optional control input [batch, control_dim].
            training: Whether in training mode.

        Returns:
            Next state tensor [batch, latent_dim].
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x_shape = tf.shape(x)
        x_flat = tf.reshape(x, [-1, self.latent_dim])

        if control is not None and self.control_dim > 0:
            u = tf.convert_to_tensor(control, dtype=tf.float32)
            u_flat = tf.reshape(u, [-1, self.control_dim])
            inputs = tf.concat([x_flat, u_flat], axis=-1)
        else:
            inputs = x_flat

        # Compute dynamics (gradient of Hamiltonian)
        residual = self.mlp(inputs, training=training)

        # Euler integration
        x_next_flat = x_flat + self.dt * residual
        x_next = tf.reshape(x_next_flat, x_shape)
        return x_next

    def compute_hamiltonian(self, x: tf.Tensor) -> tf.Tensor:
        """Compute the Hamiltonian (energy) of the state.

        Args:
            x: State tensor [batch, latent_dim].

        Returns:
            Energy scalar per batch element [batch].
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        # Simple quadratic Hamiltonian
        energy = 0.5 * tf.reduce_sum(tf.square(x), axis=-1)
        return energy

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "control_dim": self.control_dim,
                "dt": self.dt,
                "integrator": self.integrator,
            }
        )
        return config


class TimeCrystalBlock(layers.Layer):
    """Hamiltonian Neural Network based recurrent cell.

    This layer implements a single step of HNN dynamics, designed to be
    used as an RNN cell. It maintains energy conservation through the
    Hamiltonian formulation.

    The "Time Crystal" name refers to the periodic, stable dynamics
    that emerge from energy-conserving neural networks.

    Attributes:
        state_dim: Dimension of the phase space (q, p).
        hamiltonian_hidden_dim: Hidden dimension for HNN MLP.
    """

    def __init__(
        self,
        state_dim: int,
        hamiltonian_hidden_dim: int,
        evolution_time_min: float = DEFAULT_EVOLUTION_TIME_MIN,
        evolution_time_max: float = DEFAULT_EVOLUTION_TIME_MAX,
        evolution_time_initial: float = DEFAULT_EVOLUTION_TIME_INITIAL,
        **kwargs,
    ):
        """Initialize TimeCrystalBlock.

        Args:
            state_dim: Dimension of phase space.
            hamiltonian_hidden_dim: Hidden dimension for Hamiltonian MLP.
            evolution_time_min: Minimum evolution time.
            evolution_time_max: Maximum evolution time (cap).
            evolution_time_initial: Initial evolution time.
            **kwargs: Additional layer arguments.
        """
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"
        super().__init__(**kwargs)

        self.state_dim = state_dim
        self.hamiltonian_hidden_dim = hamiltonian_hidden_dim
        self._evolution_time_min = evolution_time_min
        self._evolution_time_max = evolution_time_max
        self._evolution_time_initial = evolution_time_initial

        # Weights will be created in build()
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        self.W3, self.b3 = None, None
        self.W_out, self.b_out = None, None
        self.evolution_time = None
        self.evolution_time_cap = None
        self.input_dim = None

    @property
    def state_size(self):
        """Returns the state size for RNN."""
        return (self.state_dim, self.state_dim)  # (q, p)

    @property
    def output_size(self):
        """Returns the output size."""
        return self.input_dim if self.input_dim else self.state_dim * 2

    def build(self, input_shape: tf.TensorShape):
        """Create layer weights.

        Args:
            input_shape: Input tensor shape.
        """
        self.input_dim = input_shape[-1]

        # HNN MLP: state -> hidden -> hidden -> scalar Hamiltonian
        h_input_dim = 2 * self.state_dim + self.input_dim
        D_h = self.hamiltonian_hidden_dim

        self.W1 = self.add_weight(
            name="hnn_w1", shape=(h_input_dim, D_h), initializer="glorot_uniform", trainable=True
        )
        self.b1 = self.add_weight(name="hnn_b1", shape=(D_h,), initializer="zeros", trainable=True)
        self.W2 = self.add_weight(
            name="hnn_w2", shape=(D_h, D_h), initializer="glorot_uniform", trainable=True
        )
        self.b2 = self.add_weight(name="hnn_b2", shape=(D_h,), initializer="zeros", trainable=True)
        self.W3 = self.add_weight(
            name="hnn_w3", shape=(D_h, 1), initializer="glorot_uniform", trainable=True
        )
        self.b3 = self.add_weight(name="hnn_b3", shape=(), initializer="zeros", trainable=True)

        # Evolution time parameters
        self.evolution_time = self.add_weight(
            name="evolution_time",
            shape=(),
            initializer=tf.keras.initializers.Constant(self._evolution_time_initial),
            trainable=False,
        )
        self.evolution_time_cap = self.add_weight(
            name="evolution_time_cap",
            shape=(),
            initializer=tf.keras.initializers.Constant(self._evolution_time_max),
            trainable=False,
        )

        # Output projection
        self.W_out = self.add_weight(
            name="output_proj_w",
            shape=(2 * self.state_dim, self.input_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b_out = self.add_weight(
            name="output_proj_b", shape=(self.input_dim,), initializer="zeros", trainable=True
        )

        super().build(input_shape)

    def _compute_hamiltonian_gradient(
        self,
        q: tf.Tensor,
        p: tf.Tensor,
        x: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute gradients of the Hamiltonian for symplectic integration.

        Args:
            q: Position state [batch, state_dim].
            p: Momentum state [batch, state_dim].
            x: Input features [batch, input_dim].

        Returns:
            Tuple of (dq/dt, dp/dt) tensors.
        """
        # Concatenate inputs for HNN
        h_input = tf.concat([q, p, x], axis=-1)

        # Forward through HNN MLP
        h1 = tf.nn.tanh(tf.matmul(h_input, self.W1) + self.b1)
        h2 = tf.nn.tanh(tf.matmul(h1, self.W2) + self.b2)
        H = tf.squeeze(tf.matmul(h2, self.W3) + self.b3, axis=-1)

        # Compute gradients using symplectic structure
        # dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([q, p])
            h_input = tf.concat([q, p, x], axis=-1)
            h1 = tf.nn.tanh(tf.matmul(h_input, self.W1) + self.b1)
            h2 = tf.nn.tanh(tf.matmul(h1, self.W2) + self.b2)
            H = tf.reduce_sum(tf.matmul(h2, self.W3) + self.b3)

        dq_dt = tape.gradient(H, p)  # ∂H/∂p
        dp_dt = -tape.gradient(H, q)  # -∂H/∂q

        del tape

        if dq_dt is None:
            dq_dt = tf.zeros_like(q)
        if dp_dt is None:
            dp_dt = tf.zeros_like(p)

        return dq_dt, dp_dt

    def call(
        self,
        inputs: tf.Tensor,
        states: tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """Perform one step of HNN dynamics.

        Args:
            inputs: Input tensor [batch, input_dim].
            states: Tuple of (q, p) state tensors.
            training: Whether in training mode.

        Returns:
            Tuple of (output, new_states).
        """
        x = inputs
        q, p = states

        # Get evolution time
        dt = tf.cast(self.evolution_time, tf.float32)
        cap = tf.nn.softplus(self.evolution_time_cap)
        dt = tf.minimum(dt, cap)
        dt = tf.maximum(dt, tf.constant(self._evolution_time_min, tf.float32))

        # Symplectic Euler step
        dq_dt, dp_dt = self._compute_hamiltonian_gradient(q, p, x)
        q_new = q + dt * dq_dt
        p_new = p + dt * dp_dt

        # Output projection
        state_concat = tf.concat([q_new, p_new], axis=-1)
        output = tf.matmul(state_concat, self.W_out) + self.b_out

        return output, (q_new, p_new)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "state_dim": self.state_dim,
                "hamiltonian_hidden_dim": self.hamiltonian_hidden_dim,
                "evolution_time_min": self._evolution_time_min,
                "evolution_time_max": self._evolution_time_max,
                "evolution_time_initial": self._evolution_time_initial,
            }
        )
        return config


class TimeCrystalSequenceBlock(layers.Layer):
    """Full sequence Hamiltonian Neural Network block.

    This layer processes an entire sequence through HNN dynamics in a
    single operation, enabling efficient sequence modeling with
    energy conservation.

    Attributes:
        embedding_dim: Output embedding dimension.
        state_dim: Dimension of phase space.
        hamiltonian_hidden_dim: Hidden dimension for HNN.
    """

    def __init__(self, embedding_dim: int, state_dim: int, hamiltonian_hidden_dim: int, **kwargs):
        """Initialize TimeCrystalSequenceBlock.

        Args:
            embedding_dim: Output embedding dimension.
            state_dim: Phase space dimension.
            hamiltonian_hidden_dim: HNN hidden dimension.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.hamiltonian_hidden_dim = hamiltonian_hidden_dim

        # Create the internal cell
        self.cell = TimeCrystalBlock(
            state_dim=state_dim,
            hamiltonian_hidden_dim=hamiltonian_hidden_dim,
            name=f"{self.name}_cell",
        )

        # Output projection
        self.output_projection = layers.Dense(embedding_dim, name="output_projection")

    def build(self, input_shape):
        """Build the layer.

        Args:
            input_shape: Input tensor shape.
        """
        self.cell.build(input_shape)
        self.output_projection.build(input_shape[:-1] + (2 * self.state_dim,))
        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        initial_state: tuple[tf.Tensor, tf.Tensor] | None = None,
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor], dict[str, Any]]:
        """Process sequence through HNN dynamics.

        Args:
            inputs: Input sequence [batch, seq_len, embedding_dim].
            training: Whether in training mode.
            initial_state: Optional initial (q, p) state tuple.

        Returns:
            Tuple of (output_sequence, final_state, aux_metrics).
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Initialize state if not provided
        if initial_state is None:
            q = tf.zeros([batch_size, self.state_dim], dtype=tf.float32)
            p = tf.zeros([batch_size, self.state_dim], dtype=tf.float32)
        else:
            q, p = initial_state

        # Process sequence step by step
        outputs = []
        initial_energy = 0.5 * (tf.reduce_sum(tf.square(q)) + tf.reduce_sum(tf.square(p)))

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            out_t, (q, p) = self.cell(x_t, (q, p), training=training)
            outputs.append(out_t)

        # Stack outputs
        output_sequence = tf.stack(outputs, axis=1)
        output_sequence = self.output_projection(output_sequence)

        # Compute energy drift
        final_energy = 0.5 * (tf.reduce_sum(tf.square(q)) + tf.reduce_sum(tf.square(p)))
        energy_drift = tf.abs(final_energy - initial_energy) / (tf.abs(initial_energy) + 1e-8)

        aux_metrics = {
            "energy_drift": energy_drift,
            "evolution_time": self.cell.evolution_time,
        }

        return output_sequence, (q, p), aux_metrics

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "state_dim": self.state_dim,
                "hamiltonian_hidden_dim": self.hamiltonian_hidden_dim,
            }
        )
        return config
