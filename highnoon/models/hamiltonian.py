# src/models/hamiltonian.py
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

import logging
import math
import sys

import tensorflow as tf
from tensorflow.keras import layers

from highnoon import config
from highnoon._native.ops.fused_hnn_sequence import fused_hnn_sequence

# NEW: Import the fused C++ operations
from highnoon._native.ops.fused_hnn_step import fused_hnn_step
from highnoon.config import DEBUG_MODE, QUANTUM_ZNE_SCALES
from highnoon.models.reasoning.fused_contract import FusedReasoningBlockMixin

# Lorentzian/hyperbolic geometry ops for hierarchical structure
try:
    from highnoon._native.ops.lorentzian_feature_transform import (
        lorentzian_feature_transform,
        lorentzian_feature_transform_available,
    )
except ImportError:
    lorentzian_feature_transform = None

    def lorentzian_feature_transform_available():
        return False


# --- START: NEW IMPORT FOR ROBUST DISCOVERY ---
from highnoon.models.utils.control_vars import ControlVarMixin

# --- END: NEW IMPORT ---
from highnoon.quantum.layers import EvolutionTimeVQCLayer
from highnoon.runtime.control_limits import get_evolution_time_limits

log = logging.getLogger(__name__)


class HamiltonianNN(tf.keras.Model):
    """Lightweight Hamiltonian neural network using dense layers (Python wrapper).

    This mirrors the expected interface for the PhysicsInformedTwin tests and
    defers heavy lifting to fused C++ ops where available via the surrounding
    call logic. For the unit tests here, a compact MLP + Euler step is sufficient
    to validate shapes and energy bookkeeping.
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
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.dt = dt
        self.integrator = integrator

        # Simple MLP for residual dynamics
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
        """Compute next state using a simple Euler step."""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x_shape = tf.shape(x)
        x_flat = tf.reshape(x, [-1, self.latent_dim])

        if control is not None and self.control_dim > 0:
            u = tf.convert_to_tensor(control, dtype=tf.float32)
            u_flat = tf.reshape(u, [-1, self.control_dim])
            inputs = tf.concat([x_flat, u_flat], axis=-1)
        else:
            inputs = x_flat

        residual = self.mlp(inputs, training=training)
        x_next_flat = x_flat + self.dt * residual
        x_next = tf.reshape(x_next_flat, x_shape)
        return x_next

    def compute_hamiltonian(self, x: tf.Tensor) -> tf.Tensor:
        """Simple quadratic Hamiltonian for energy tracking."""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        energy = 0.5 * tf.reduce_sum(tf.square(x), axis=-1)
        return energy

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "latent_dim": self.latent_dim,
                "control_dim": self.control_dim,
                "dt": self.dt,
                "integrator": self.integrator,
            }
        )
        return base


class TimeCrystalBlock(ControlVarMixin, layers.Layer):
    """
    A Hamiltonian Neural Network (HNN) based recurrent cell.

    This class serves as the single-step Keras RNN Cell, adhering to the standard
    `call(inputs, states)` signature for inference and manual loop unrolling.
    """

    def __init__(self, state_dim: int, hamiltonian_hidden_dim: int, **kwargs):
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"
        super().__init__(**kwargs)

        self.state_dim = state_dim
        self.hamiltonian_hidden_dim = hamiltonian_hidden_dim
        self._control_limits = get_evolution_time_limits()

        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        self.W3, self.b3 = None, None
        self.W_out, self.b_out = None, None
        self.input_dim = None
        self.evolution_time_gain = None
        self.evolution_time_shift = None
        self.evolution_time_cap = None
        self.evolution_time_generator = EvolutionTimeVQCLayer(
            num_layers=config.QUANTUM_VQC_LAYERS,
            num_qubits=config.QUANTUM_VQC_QUBITS,
            zne_scales=QUANTUM_ZNE_SCALES,
            entanglement=config.QUANTUM_ENTANGLEMENT,
            shots=config.QUANTUM_VQC_SHOTS,
            enable_sampling_during_training=config.QUANTUM_ENABLE_SAMPLING,
            backend_preference=config.QUANTUM_BACKEND,
            name=f"{self.name}_evolution_vqc",
        )
        self.evolution_time_bias = None
        self._generator_input_dim = None

        self._state_size = None
        self._output_size = None
        self._evolution_metric = None

    @property
    def state_size(self):
        """Returns the padded state size required by the hybrid structure."""
        if self._state_size is None:
            # The actual state shape is handled by the `initial_state` argument
            return (tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None]))
        return self._state_size

    @property
    def output_size(self):
        """Returns the output size (full embedding dimension)."""
        if self._output_size is None and self.input_dim is not None:
            return self.input_dim
        raise AttributeError("output_size is not available until the layer is built.")

    def build(self, input_shape: tf.TensorShape):
        """
        Creates the weights for the layer.
        """
        self.input_dim = input_shape[-1]
        self._output_size = self.input_dim

        h_input_dim = 2 * self.state_dim + (self.input_dim or 0)
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
        self.b3 = self.add_weight(
            name="hnn_b3", shape=(), initializer="zeros", trainable=True
        )  # Shape must be scalar ()

        self.evolution_time_bias = self.add_weight(
            name="evolution_time",
            shape=(),
            initializer=tf.keras.initializers.Constant(5e-4),
            trainable=False,
        )
        if self.evolution_time_gain is None:
            self.evolution_time_gain = self.add_weight(
                name="evolution_time_gain",
                shape=(),
                initializer=tf.keras.initializers.Constant(0.1),
                trainable=False,
            )
        if self.evolution_time_shift is None:
            self.evolution_time_shift = self.add_weight(
                name="evolution_time_shift",
                shape=(),
                initializer=tf.keras.initializers.Constant(7.0),
                trainable=False,
            )
        if self.evolution_time_cap is None:
            self.evolution_time_cap = self.add_weight(
                name="evolution_time_cap",
                shape=(),
                initializer=tf.keras.initializers.Constant(0.02),
                trainable=False,
            )

        self.W_out = self.add_weight(
            name="output_proj_w",
            shape=(2 * self.state_dim, self.input_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b_out = self.add_weight(
            name="output_proj_b", shape=(self.input_dim,), initializer="zeros", trainable=True
        )

        self._generator_input_dim = (self.input_dim or 0) + 2 * self.state_dim
        if not self.evolution_time_generator.built:
            self.evolution_time_generator.build(tf.TensorShape([None, self._generator_input_dim]))

        log.info(f"[{self.name}] Registering control variables...")
        self.register_control_var("evolution_time", self.evolution_time_bias)
        self.register_control_var("evolution_time_gain", self.evolution_time_gain)
        self.register_control_var("evolution_time_shift", self.evolution_time_shift)
        self.register_control_var("evolution_time_cap", self.evolution_time_cap)
        log.info(f"[{self.name}] Finished registering control variables.")

        if self._evolution_metric is None:
            metric_prefix = self.name or "timecrystal_block"
            self._evolution_metric = tf.keras.metrics.Mean(name=f"{metric_prefix}_evolution_time")

        if not hasattr(self, "last_stable_evolution_time"):
            self.last_stable_evolution_time = self.add_weight(
                name="last_stable_evolution_time",
                shape=(),
                initializer=tf.keras.initializers.Constant(self._control_limits.initial_value),
                trainable=False,
            )

        super().build(input_shape)

    @property
    def metrics(self):
        base_metrics = super().metrics
        if self._evolution_metric is not None:
            return base_metrics + [self._evolution_metric]
        return base_metrics

    @tf.function(reduce_retracing=True, experimental_relax_shapes=True)
    def call(
        self,
        inputs: tf.Tensor,
        states: tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
        return_aux_metrics: bool = False,
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """
        Performs a single step of the recurrent dynamics using the fused HNN step op.
        This is used during inference or manual loop unrolling (e.g., in KalmanBlock).
        """
        x = inputs
        h_padded, aux_padded = states[0], states[1]

        batch_size = tf.shape(x)[0]

        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] --- TimeCrystalBlock.call START (1-step) ---",
                output_stream=sys.stderr,
            )
            tf.print(
                f"[{self.name}] Input x shape (1-step):", tf.shape(x), output_stream=sys.stderr
            )
            tf.print(
                f"[{self.name}] Input h_padded shape:", tf.shape(h_padded), output_stream=sys.stderr
            )
            tf.print(
                f"[{self.name}] Static h_padded shape (for debugging None):",
                h_padded.shape,
                output_stream=sys.stderr,
            )

        x_t_squeezed = tf.cond(tf.equal(tf.rank(x), 3), lambda: tf.squeeze(x, axis=1), lambda: x)

        mamba_state_dim = tf.cast(tf.shape(h_padded)[-1], tf.int32)
        total_state_dim = tf.constant(2 * self.state_dim, dtype=tf.int32)
        num_slices = tf.math.floordiv(total_state_dim, mamba_state_dim)

        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] Calculated mamba_state_dim (dynamic):",
                mamba_state_dim,
                output_stream=sys.stderr,
            )
            tf.print(
                f"[{self.name}] Calculated num_slices (dynamic):",
                num_slices,
                output_stream=sys.stderr,
            )

        state_as_slices = h_padded[:, :num_slices, :]
        unpacked_state = tf.reshape(state_as_slices, [batch_size, total_state_dim])
        q_t, p_t = tf.split(unpacked_state, 2, axis=-1)

        control_features = tf.concat([x_t_squeezed, q_t, p_t], axis=-1)
        evolution_samples = self.evolution_time_generator(control_features, training=training)
        evolution_samples = tf.squeeze(evolution_samples, axis=-1)
        soft_gain = tf.nn.softplus(self.evolution_time_gain)
        soft_shift = tf.nn.softplus(self.evolution_time_shift)
        adjusted_times = tf.nn.softplus(soft_gain * evolution_samples - soft_shift)
        raw_evolution_time = tf.reduce_mean(adjusted_times) + tf.cast(
            self.evolution_time_bias, tf.float32
        )
        cap = tf.nn.softplus(self.evolution_time_cap)
        cap = tf.maximum(cap, tf.constant(1e-6, dtype=tf.float32))
        evolution_time_scalar = cap * tf.sigmoid(raw_evolution_time / cap)
        evolution_time_scalar = tf.where(
            tf.math.is_finite(evolution_time_scalar),
            evolution_time_scalar,
            cap * tf.sigmoid(tf.cast(self.evolution_time_bias, tf.float32) / cap),
        )

        min_time = tf.constant(self._control_limits.min_value, dtype=tf.float32)
        previous_time = tf.maximum(tf.cast(self.last_stable_evolution_time, tf.float32), min_time)
        candidate_time = tf.maximum(evolution_time_scalar, min_time)

        guard_terms = []
        if self._control_limits.hardware_relative_step > 0.0:
            guard_terms.append(
                previous_time
                * tf.constant(self._control_limits.hardware_relative_step, dtype=tf.float32)
            )
        if self._control_limits.hardware_absolute_step > 0.0:
            guard_terms.append(
                tf.constant(self._control_limits.hardware_absolute_step, dtype=tf.float32)
            )
        if guard_terms:
            max_step = guard_terms[0]
            for term in guard_terms[1:]:
                max_step = tf.maximum(max_step, term)
            candidate_time = tf.minimum(candidate_time, previous_time + max_step)

        max_guard = self._control_limits.max_value
        if max_guard is not None and math.isfinite(max_guard):
            candidate_time = tf.minimum(candidate_time, tf.constant(max_guard, dtype=tf.float32))

        safe_time = tf.where(tf.math.is_finite(candidate_time), candidate_time, previous_time)

        padding_len = tf.shape(h_padded)[1] - num_slices
        padding_shape = tf.stack([batch_size, padding_len, mamba_state_dim])
        static_batch_dim = h_padded.shape[0]
        static_seq_len_dim = h_padded.shape[1]
        static_state_dim = h_padded.shape[-1]

        def _integrate(time_scalar: tf.Tensor):
            q_next_local, p_next_local, output_proj_local, h_initial_local, h_final_local = (
                fused_hnn_step(
                    q_t,
                    p_t,
                    x_t_squeezed,
                    self.W1,
                    self.b1,
                    self.W2,
                    self.b2,
                    self.W3,
                    self.b3,
                    self.W_out,
                    self.b_out,
                    time_scalar,
                )
            )
            new_packed_state_local = tf.concat([q_next_local, p_next_local], axis=-1)
            new_packed_state_sliced_local = tf.reshape(
                new_packed_state_local, [batch_size, num_slices, mamba_state_dim]
            )
            new_h_padded_local = tf.concat(
                [new_packed_state_sliced_local, tf.zeros(padding_shape, dtype=tf.float32)],
                axis=1,
            )
            new_h_padded_local.set_shape([static_batch_dim, static_seq_len_dim, static_state_dim])
            output_local = tf.expand_dims(output_proj_local, axis=1)
            output_local.set_shape([None, 1, self.input_dim])
            energy_drift_local = tf.reduce_mean(tf.abs(h_final_local - h_initial_local))
            return output_local, new_h_padded_local, energy_drift_local

        candidate_output, candidate_h_padded, candidate_drift = _integrate(safe_time)
        fallback_output, fallback_h_padded, fallback_drift = _integrate(previous_time)

        finite_candidate = tf.reduce_all(tf.math.is_finite(candidate_output))
        finite_candidate = tf.logical_and(
            finite_candidate, tf.reduce_all(tf.math.is_finite(candidate_h_padded))
        )
        finite_candidate = tf.logical_and(
            finite_candidate, tf.reduce_all(tf.math.is_finite(candidate_drift))
        )

        output = tf.where(finite_candidate, candidate_output, fallback_output)
        new_h_padded = tf.where(finite_candidate, candidate_h_padded, fallback_h_padded)
        selected_drift = tf.where(finite_candidate, candidate_drift, fallback_drift)
        selected_time = tf.where(finite_candidate, safe_time, previous_time)
        selected_cap = tf.where(finite_candidate, cap, tf.maximum(cap, min_time))
        selected_raw = tf.where(finite_candidate, raw_evolution_time, previous_time)

        new_state = (new_h_padded, aux_padded)
        self.last_stable_evolution_time.assign(selected_time)

        aux_metrics = {}
        if return_aux_metrics:
            aux_metrics["energy_drift"] = selected_drift
            aux_metrics["evolution_time_raw"] = selected_raw
            aux_metrics["evolution_time_cap"] = selected_cap
            aux_metrics["evolution_time"] = selected_time
        if self._evolution_metric is not None:
            self._evolution_metric.update_state(selected_time)

        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] Output tensor shape:", tf.shape(output), output_stream=sys.stderr
            )
            tf.print(
                f"[{self.name}] Output h_padded shape:",
                tf.shape(new_h_padded),
                output_stream=sys.stderr,
            )
            tf.print(
                f"[{self.name}] Energy Drift:",
                aux_metrics.get("energy_drift", "N/A"),
                output_stream=sys.stderr,
            )
            tf.print(f"[{self.name}] --- TimeCrystalBlock.call END ---", output_stream=sys.stderr)

        # MODIFIED: Return as a tuple instead of a list for stability.
        # The original `list(new_state)` could cause issues with tf.function tracing.
        return output, new_state

    def get_config(self) -> dict:
        # The base config from the superclass should be retrieved first.
        config = super().get_config()
        config.update(
            {
                "state_dim": self.state_dim,
                "hamiltonian_hidden_dim": self.hamiltonian_hidden_dim,
            }
        )
        return config


class TimeCrystalSequenceBlock(FusedReasoningBlockMixin, ControlVarMixin, layers.Layer):
    """
    A full sequence layer implementation for the Time Crystal Block.
    This block handles the full input sequence in a single, unrolled C++ operation.
    It replaces the problematic `tf.keras.layers.RNN(TimeCrystalBlock)` wrapper.
    """

    fused_block_type = "TimeCrystalSequenceBlock"
    fused_block_stateful = True

    def __init__(
        self,
        embedding_dim: int,
        state_dim: int,
        hamiltonian_hidden_dim: int,
        d_inner: int | None = None,
        use_lorentzian: bool = False,
        lorentzian_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.hamiltonian_hidden_dim = hamiltonian_hidden_dim

        # Lorentzian/hyperbolic geometry enhancement
        self.use_lorentzian = use_lorentzian and lorentzian_feature_transform_available()
        self.lorentzian_dim = lorentzian_dim or config.LORENTZIAN_HYPERBOLIC_DIM
        self.lorentzian_spatial_dim = self.lorentzian_dim - 1

        self.cell = TimeCrystalBlock(
            state_dim=state_dim,
            hamiltonian_hidden_dim=hamiltonian_hidden_dim,
            name=f"{self.name}_cell",
        )
        self.output_projection = layers.Dense(self.embedding_dim, name="output_projection")
        self.d_inner = d_inner or (2 * state_dim)
        self.conv_dim = 2
        self._sequence_evolution_metric = None

    def build(self, input_shape):
        self.cell.build(input_shape)
        # The projection layer's input will be the output of the HNN evolution,
        # which has a dimension of 2 * state_dim.
        self.output_projection.build(input_shape[:-1] + (2 * self.state_dim,))

        # Build Lorentzian weights if enabled
        if self.use_lorentzian:
            # Boost vector controls hyperbolic "velocity"
            self.lorentzian_boost = self.add_weight(
                name="lorentzian_boost",
                shape=(self.lorentzian_spatial_dim,),
                initializer="zeros",
                trainable=True,
            )
            # Rotation parameter matrix (skew-symmetric part used)
            self.lorentzian_rotation = self.add_weight(
                name="lorentzian_rotation",
                shape=(self.lorentzian_spatial_dim, self.lorentzian_spatial_dim),
                initializer="glorot_uniform",
                trainable=True,
            )
            # Projection to/from hyperbolic dimension
            self.lorentzian_proj_in = self.add_weight(
                name="lorentzian_proj_in",
                shape=(self.embedding_dim, self.lorentzian_dim),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.lorentzian_proj_out = self.add_weight(
                name="lorentzian_proj_out",
                shape=(self.lorentzian_dim, self.embedding_dim),
                initializer="glorot_uniform",
                trainable=True,
            )
            log.info(f"[{self.name}] Lorentzian enhancement enabled (D_hyp={self.lorentzian_dim})")

        super().build(input_shape)
        # --- START: DEFINITIVE FIX for Variable Discovery ---
        log.info(f"[{self.name}] Registering sequence control variables...")
        self.register_control_var("evolution_time", self.cell.evolution_time_bias)
        self.register_control_var("vqc_thetas", self.cell.evolution_time_generator.theta)
        # --- END: DEFINITIVE FIX ---

        if self._sequence_evolution_metric is None:
            metric_prefix = self.name or "timecrystal_sequence"
            self._sequence_evolution_metric = tf.keras.metrics.Mean(
                name=f"{metric_prefix}_evolution_time"
            )

    @property
    def metrics(self):
        base_metrics = super().metrics
        if self._sequence_evolution_metric is not None:
            return base_metrics + [self._sequence_evolution_metric]
        return base_metrics

    @tf.function(reduce_retracing=True, experimental_relax_shapes=True)
    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        initial_state: tuple[tf.Tensor, tf.Tensor] | None = None,
        return_aux_metrics: bool = False,
    ):
        """
        Forward pass using the fused C++ sequence op for full sequence traversal.
        """
        x = inputs
        batch_size = tf.shape(x)[0]
        tf.shape(x)[1]

        # Auto-generate initial state if not provided
        # This enables the block to work in the ReasoningModule without explicit state management
        if initial_state is None:
            # Create zero-initialized state with appropriate shape
            # h_padded: [batch, num_slices, mamba_state_dim]
            # Using a reasonable default for mamba_state_dim (64) and num_slices (8)
            mamba_state_dim = 64  # Default from config
            num_slices = max(8, (2 * self.state_dim + mamba_state_dim - 1) // mamba_state_dim)
            h_padded = tf.zeros([batch_size, num_slices, mamba_state_dim], dtype=tf.float32)
            aux_padded = tf.zeros([batch_size, num_slices, mamba_state_dim], dtype=tf.float32)
            initial_state = (h_padded, aux_padded)

        h_padded, aux_padded = initial_state[0], initial_state[1]

        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] --- TimeCrystalSequenceBlock.call START (Sequence) ---",
                output_stream=sys.stderr,
            )
            tf.print(
                f"[{self.name}] Input x shape (Sequence):", tf.shape(x), output_stream=sys.stderr
            )
            tf.print(
                f"[{self.name}] Input h_padded shape:", tf.shape(h_padded), output_stream=sys.stderr
            )
            tf.print(
                f"[{self.name}] Static h_padded shape (for debugging None):",
                h_padded.shape,
                output_stream=sys.stderr,
            )

        total_state_dim = 2 * self.state_dim
        mamba_state_dim = tf.cast(tf.shape(h_padded)[-1], tf.int32)
        # Ensure enough slices to reconstruct the state even when the dimensions are not divisible.
        num_slices = tf.cast(
            tf.math.ceil(
                tf.cast(total_state_dim, tf.float32) / tf.cast(mamba_state_dim, tf.float32)
            ),
            tf.int32,
        )

        state_as_slices = h_padded[:, :num_slices, :]
        flat_state = tf.reshape(state_as_slices, [batch_size, -1])
        current_len = tf.shape(flat_state)[1]
        needed = tf.maximum(0, total_state_dim - current_len)
        padded_state = tf.pad(flat_state, [[0, 0], [0, needed]])
        unpacked_state = padded_state[:, :total_state_dim]
        initial_q, initial_p = tf.split(unpacked_state, 2, axis=-1)

        # --- START: DEFINITIVE FIX for tf.function retracing ---
        # Convert the Python boolean `training` to a `tf.Tensor` before passing
        # it to the VQC layer to prevent unnecessary retracing.
        training_tensor = tf.constant(training, dtype=tf.bool)
        sequence_mean = tf.reduce_mean(x, axis=1)
        control_features = tf.concat([sequence_mean, initial_q, initial_p], axis=-1)
        evolution_samples = self.cell.evolution_time_generator(
            control_features, training=training_tensor
        )
        # --- END: DEFINITIVE FIX ---
        evolution_samples = tf.squeeze(evolution_samples, axis=-1)
        seq_soft_gain = tf.nn.softplus(self.cell.evolution_time_gain)
        seq_soft_shift = tf.nn.softplus(self.cell.evolution_time_shift)
        seq_adjusted_times = tf.nn.softplus(seq_soft_gain * evolution_samples - seq_soft_shift)
        raw_seq_time = tf.reduce_mean(seq_adjusted_times) + tf.cast(
            self.cell.evolution_time_bias, tf.float32
        )
        seq_cap = tf.nn.softplus(self.cell.evolution_time_cap)
        seq_cap = tf.maximum(seq_cap, tf.constant(1e-6, dtype=tf.float32))
        evolution_time_scalar = seq_cap * tf.sigmoid(raw_seq_time / seq_cap)
        evolution_time_scalar = tf.where(
            tf.math.is_finite(evolution_time_scalar),
            evolution_time_scalar,
            seq_cap * tf.sigmoid(tf.cast(self.cell.evolution_time_bias, tf.float32) / seq_cap),
        )

        limits = self.cell._control_limits
        min_time = tf.constant(limits.min_value, dtype=tf.float32)
        previous_time = tf.maximum(
            tf.cast(self.cell.last_stable_evolution_time, tf.float32), min_time
        )
        candidate_time = tf.maximum(evolution_time_scalar, min_time)

        guard_terms = []
        if limits.hardware_relative_step > 0.0:
            guard_terms.append(
                previous_time * tf.constant(limits.hardware_relative_step, dtype=tf.float32)
            )
        if limits.hardware_absolute_step > 0.0:
            guard_terms.append(tf.constant(limits.hardware_absolute_step, dtype=tf.float32))
        if guard_terms:
            max_step = guard_terms[0]
            for term in guard_terms[1:]:
                max_step = tf.maximum(max_step, term)
            candidate_time = tf.minimum(candidate_time, previous_time + max_step)

        if limits.max_value is not None and math.isfinite(limits.max_value):
            candidate_time = tf.minimum(
                candidate_time, tf.constant(limits.max_value, dtype=tf.float32)
            )

        safe_time = tf.where(tf.math.is_finite(candidate_time), candidate_time, previous_time)

        padding_len = tf.shape(h_padded)[1] - num_slices
        padding_shape = tf.stack([batch_size, padding_len, mamba_state_dim])

        def _integrate_sequence(time_scalar: tf.Tensor):
            (
                seq_output,
                final_q_raw_local,
                final_p_raw_local,
                h_initial_seq_local,
                h_final_seq_local,
            ) = fused_hnn_sequence(
                x,
                initial_q,
                initial_p,
                self.cell.W1,
                self.cell.b1,
                self.cell.W2,
                self.cell.b2,
                self.cell.W3,
                self.cell.b3,
                self.cell.W_out,
                self.cell.b_out,
                time_scalar,
            )
            final_packed_state_local = tf.concat([final_q_raw_local, final_p_raw_local], axis=-1)
            final_len_local = tf.shape(final_packed_state_local)[1]
            target_len_local = num_slices * mamba_state_dim
            tail_needed_local = tf.maximum(0, target_len_local - final_len_local)
            padded_final_local = tf.pad(final_packed_state_local, [[0, 0], [0, tail_needed_local]])
            trimmed_final_local = padded_final_local[:, :target_len_local]
            final_packed_state_sliced_local = tf.reshape(
                trimmed_final_local, [batch_size, num_slices, mamba_state_dim]
            )
            final_h_padded_local = tf.concat(
                [final_packed_state_sliced_local, tf.zeros(padding_shape, dtype=tf.float32)],
                axis=1,
            )
            final_h_padded_local.set_shape(h_padded.shape)
            energy_drift_local = tf.reduce_mean(
                tf.abs(h_final_seq_local - h_initial_seq_local), name="energy_drift_calc"
            )
            return seq_output, final_h_padded_local, energy_drift_local

        candidate_output, candidate_h_padded, candidate_drift = _integrate_sequence(safe_time)
        fallback_output, fallback_h_padded, fallback_drift = _integrate_sequence(previous_time)

        finite_candidate = tf.reduce_all(tf.math.is_finite(candidate_output))
        finite_candidate = tf.logical_and(
            finite_candidate, tf.reduce_all(tf.math.is_finite(candidate_h_padded))
        )
        finite_candidate = tf.logical_and(
            finite_candidate, tf.reduce_all(tf.math.is_finite(candidate_drift))
        )

        output_sequence = tf.where(finite_candidate, candidate_output, fallback_output)
        output_sequence = self.output_projection(output_sequence)

        # Apply Lorentzian transform for hierarchical structure enhancement
        if self.use_lorentzian:
            # Project to hyperbolic dimension
            hyp_features = tf.einsum("bsd,dh->bsh", output_sequence, self.lorentzian_proj_in)
            # Apply Lorentzian transform (Lie algebra matrix exponential)
            hyp_transformed = lorentzian_feature_transform(
                hyp_features,
                self.lorentzian_boost,
                self.lorentzian_rotation,
            )
            # Project back to embedding dimension
            output_sequence = tf.einsum("bsh,hd->bsd", hyp_transformed, self.lorentzian_proj_out)

        final_h_padded = tf.where(finite_candidate, candidate_h_padded, fallback_h_padded)
        selected_drift = tf.where(finite_candidate, candidate_drift, fallback_drift)
        selected_time = tf.where(finite_candidate, safe_time, previous_time)
        selected_cap = tf.where(finite_candidate, seq_cap, tf.maximum(seq_cap, min_time))
        selected_raw = tf.where(finite_candidate, raw_seq_time, previous_time)

        self.cell.last_stable_evolution_time.assign(selected_time)

        final_state = (final_h_padded, aux_padded)
        aux_metrics = {}
        if return_aux_metrics:
            aux_metrics["energy_drift"] = selected_drift
            aux_metrics["evolution_time_raw"] = selected_raw
            aux_metrics["evolution_time_cap"] = selected_cap
            aux_metrics["evolution_time"] = selected_time
        if self._sequence_evolution_metric is not None:
            self._sequence_evolution_metric.update_state(selected_time)

        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] Output sequence shape:",
                tf.shape(output_sequence),
                output_stream=sys.stderr,
            )
            tf.print(
                f"[{self.name}] Final h_padded shape:",
                tf.shape(final_h_padded),
                output_stream=sys.stderr,
            )
            tf.print(
                f"[{self.name}] Energy Drift (Sequence Avg):",
                aux_metrics.get("energy_drift", "N/A"),
                output_stream=sys.stderr,
            )
            tf.print(
                f"[{self.name}] --- TimeCrystalSequenceBlock.call END ---", output_stream=sys.stderr
            )

        # --- END: DEFINITIVE FIX ---
        return output_sequence, final_state, aux_metrics

    def get_config(self) -> dict:
        base_config = super().get_config()
        base_config.update(
            {
                "embedding_dim": self.embedding_dim,
                "state_dim": self.state_dim,
                "hamiltonian_hidden_dim": self.hamiltonian_hidden_dim,
                "d_inner": self.d_inner,
                "use_lorentzian": self.use_lorentzian,
                "lorentzian_dim": self.lorentzian_dim,
            }
        )
        return base_config

    def get_weights_for_fused_op(self):
        """
        Returns only the classical weights consumed by the fused C++ operator,
        excluding the quantum generator parameters. This now passes the weights
        for the sequence-level output projection, aligning with the C++ operator's
        expectation for the FusedReasoningStack.
        """
        return [
            self.cell.W1,
            self.cell.b1,
            self.cell.W2,
            self.cell.b2,
            self.cell.W3,
            self.cell.b3,
            self.cell.evolution_time_bias,
            self.output_projection.kernel,
            self.output_projection.bias,
        ]

    def fused_metadata(self) -> dict[str, int]:
        metadata = super().fused_metadata()
        metadata.update(
            {
                "embedding_dim": int(self.embedding_dim),
                "state_dim": int(self.state_dim),
                "hamiltonian_hidden_dim": int(self.hamiltonian_hidden_dim),
                "d_inner": int(self.d_inner),
            }
        )
        return metadata
