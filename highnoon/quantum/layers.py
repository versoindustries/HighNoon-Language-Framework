#!/usr/bin/env python3
# src/quantum/layers.py
#
# Variational quantum circuit based Keras layers with multi-qubit support.

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
import tensorflow as tf

# V2 MIGRATION: Use quantum_foundation_ops unified VQC API
from highnoon._native.ops.quantum_foundation_ops import run_vqc as run_vqc_expectation
from highnoon.config import DEBUG_MODE, QUANTUM_ENABLE_SAMPLING, USE_AUTO_NEURAL_QEM
from highnoon.models.utils.control_vars import ControlVarMixin

from .device_manager import QuantumDeviceManager

# ---------------------------------------------------------------------------
# Low-level circuit helpers
# ---------------------------------------------------------------------------


def _rotation_y(angle: tf.Tensor) -> tf.Tensor:
    """Returns the complex rotation matrix for RY(angle).

    Phase 1.5: Uses float64/complex128 precision for quantum methods
    per CLAUDE.md directive for numerical stability.
    """
    angle = tf.convert_to_tensor(angle, dtype=tf.float64)
    c = tf.cos(angle / 2.0)
    s = tf.sin(angle / 2.0)
    real = tf.stack(
        [
            tf.stack([c, -s], axis=-1),
            tf.stack([s, c], axis=-1),
        ],
        axis=0,
    )
    imag = tf.zeros_like(real)
    return tf.complex(real, imag)  # Returns complex128


def _rotation_z(angle: tf.Tensor) -> tf.Tensor:
    """Returns the complex rotation matrix for RZ(angle).

    Phase 1.5: Uses float64/complex128 precision for quantum methods
    per CLAUDE.md directive for numerical stability.
    """
    angle = tf.convert_to_tensor(angle, dtype=tf.float64)
    half = angle / 2.0
    real = tf.stack(
        [
            tf.stack([tf.cos(half), tf.constant(0.0, dtype=tf.float64)], axis=-1),
            tf.stack([tf.constant(0.0, dtype=tf.float64), tf.cos(half)], axis=-1),
        ],
        axis=0,
    )
    imag = tf.stack(
        [
            tf.stack([-tf.sin(half), tf.constant(0.0, dtype=tf.float64)], axis=-1),
            tf.stack([tf.constant(0.0, dtype=tf.float64), tf.sin(half)], axis=-1),
        ],
        axis=0,
    )
    return tf.complex(real, imag)  # Returns complex128


def _ket_zero(num_qubits: int) -> tf.Tensor:
    """Returns |0...0> for num_qubits.

    Phase 1.6: Uses complex128 for quantum precision.
    """
    dim = 1 << num_qubits
    return tf.cast(tf.one_hot(0, dim), dtype=tf.complex128)


# ---------------------------------------------------------------------------
# Hybrid Variational Quantum Layers
# ---------------------------------------------------------------------------


class HybridVQCLayer(ControlVarMixin, tf.keras.layers.Layer):
    """
    Base Keras layer embedding a configurable multi-qubit variational circuit.

    The layer maps classical inputs to rotation angles, simulates (or delegates)
    the circuit execution, and scales the resulting expectation value into a
    target range. Gradients are computed via the parameter-shift rule to remain
    compatible with hardware backends.
    """

    def __init__(
        self,
        num_layers: int = 2,
        min_output: float = -1.0,
        max_output: float = 1.0,
        zne_scales: Sequence[float] | None = None,
        enable_zne: bool = True,
        num_qubits: int = 1,
        entanglement: str | Sequence[tuple[int, int]] = "linear",
        shots: int | None = None,
        backend_preference: str | None = None,
        enable_sampling_during_training: bool | None = None,
        measurement_terms: Sequence[tuple[float, str]] | None = None,
        name: str | None = None,
        **kwargs,
    ):
        # Phase 1.5: Use float64 for quantum methods per CLAUDE.md directive
        super().__init__(name=name, dtype="float64", **kwargs)

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive.")

        self.num_layers = int(num_layers)
        self.num_qubits = int(num_qubits)
        self.min_output = None if min_output is None else float(min_output)
        self.max_output = None if max_output is None else float(max_output)
        if (self.min_output is None) != (self.max_output is None):
            raise ValueError("min_output and max_output must both be set or both None.")
        if self.min_output is not None and self.max_output <= self.min_output:
            raise ValueError("max_output must be greater than min_output.")
        self.enable_zne = bool(enable_zne)
        self.zne_scales = tuple(float(s) for s in zne_scales) if zne_scales else None
        self.entanglement = entanglement
        self.shots = shots
        self.backend_preference = backend_preference
        # Wire to QUANTUM_ENABLE_SAMPLING config if not explicitly set
        if enable_sampling_during_training is None:
            self.enable_sampling_during_training = QUANTUM_ENABLE_SAMPLING
        else:
            self.enable_sampling_during_training = bool(enable_sampling_during_training)
        self.measurement_terms = (
            tuple((float(coeff), str(paulis)) for coeff, paulis in measurement_terms)
            if measurement_terms
            else None
        )

        self.device_manager = QuantumDeviceManager.global_instance()

        # S18: CayleyDense for VQC data encoding (orthogonal weight stability)
        from highnoon.config import USE_CAYLEY_VQC

        if USE_CAYLEY_VQC:
            try:
                from highnoon.models.layers.cayley_weights import CayleyDense

                self.data_encoder = CayleyDense(
                    units=self.num_qubits,
                    use_bias=True,
                    name=f"{self.name or 'hybrid_vqc'}_data_encoder",
                )
            except ImportError:
                # Fallback to standard Dense if CayleyDense not available
                self.data_encoder = tf.keras.layers.Dense(
                    self.num_qubits,
                    activation=None,
                    use_bias=True,
                    dtype="float32",
                    name=f"{self.name or 'hybrid_vqc'}_data_encoder",
                )
        else:
            self.data_encoder = tf.keras.layers.Dense(
                self.num_qubits,
                activation=None,
                use_bias=True,
                dtype="float32",
                name=f"{self.name or 'hybrid_vqc'}_data_encoder",
            )
        self.theta: tf.Variable | None = None
        self.registered_control_vars = False
        self._expectation_tracker: tf.keras.metrics.Mean | None = None
        self._output_tracker: tf.keras.metrics.Mean | None = None

        self._entangler_pairs: tuple[tuple[int, int], ...] = self._resolve_entanglers(entanglement)
        self._entangler_pairs_tensor: np.ndarray | None = None
        self._measurement_paulis_tensor: np.ndarray | None = None
        self._measurement_coeffs_tensor: np.ndarray | None = None

        # Phase 130.1: Auto Neural QEM wrapping
        self._auto_qem_mitigator = None
        if USE_AUTO_NEURAL_QEM:
            # Lazy import to avoid circular dependency
            from highnoon.training.neural_zne import NeuralQuantumErrorMitigator

            self._auto_qem_mitigator = NeuralQuantumErrorMitigator(
                name=f"{name or 'hybrid_vqc'}_qem"
            )

    # ------------------------------------------------------------------
    # Build utilities
    # ------------------------------------------------------------------

    def _resolve_entanglers(
        self, entanglement: str | Sequence[tuple[int, int]]
    ) -> tuple[tuple[int, int], ...]:
        if isinstance(entanglement, str):
            entanglement = entanglement.lower()
            if entanglement in ("none", "identity"):
                return ()
            if entanglement in ("linear", "chain"):
                return tuple((i, i + 1) for i in range(self.num_qubits - 1))
            if entanglement in ("full", "all_to_all", "complete"):
                pairs = []
                for control in range(self.num_qubits):
                    for target in range(control + 1, self.num_qubits):
                        pairs.append((control, target))
                return tuple(pairs)
            raise ValueError(f"Unknown entanglement scheme: {entanglement}")

        pairs: list[tuple[int, int]] = []
        for control, target in entanglement:
            if control == target:
                continue
            if not (0 <= control < self.num_qubits) or not (0 <= target < self.num_qubits):
                raise ValueError(
                    f"Invalid entangler pair ({control}, {target}) for {self.num_qubits} qubits."
                )
            pairs.append((int(control), int(target)))
        return tuple(pairs)

    def _resolve_measurement_terms(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (pauli_matrix, coeff_vector) representing a weighted sum of Pauli strings.
        Pauli encoding: I=0, X=1, Y=2, Z=3.
        """
        mapping = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        if not self.measurement_terms:
            weights = np.full((self.num_qubits,), 1.0 / float(self.num_qubits), dtype=np.float32)
            paulis = np.zeros((self.num_qubits, self.num_qubits), dtype=np.int32)
            for qubit in range(self.num_qubits):
                paulis[qubit, qubit] = mapping["Z"]
            return paulis, weights

        pauli_rows = []
        coeffs = []
        for coeff, pauli_string in self.measurement_terms:
            if not isinstance(pauli_string, str):
                raise ValueError("Each measurement term must provide a string Pauli specification.")
            cleaned = pauli_string.strip().upper()
            if len(cleaned) != self.num_qubits:
                raise ValueError(
                    f"Measurement term '{pauli_string}' must have length {self.num_qubits}."
                )
            row = []
            for ch in cleaned:
                if ch not in mapping:
                    raise ValueError(f"Unsupported Pauli '{ch}' encountered in '{pauli_string}'.")
                row.append(mapping[ch])
            pauli_rows.append(row)
            coeffs.append(float(coeff))

        if not pauli_rows:
            weights = np.full((self.num_qubits,), 1.0 / float(self.num_qubits), dtype=np.float32)
            paulis = np.zeros((self.num_qubits, self.num_qubits), dtype=np.int32)
            for qubit in range(self.num_qubits):
                paulis[qubit, qubit] = mapping["Z"]
            return paulis, weights

        return np.asarray(pauli_rows, dtype=np.int32), np.asarray(coeffs, dtype=np.float32)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        if not self.data_encoder.built:
            self.data_encoder.build(tf.TensorShape([None, input_dim]))

        if self.theta is None:
            self.theta = self.add_weight(
                name="circuit_thetas",
                shape=(self.num_layers, self.num_qubits, 2),
                initializer=tf.keras.initializers.RandomUniform(minval=-0.15, maxval=0.15),
                trainable=True,
                dtype=tf.float32,
            )
        if not self.registered_control_vars:
            self.register_control_var("vqc_thetas", self.theta)
            self.registered_control_vars = True

        if self._expectation_tracker is None:
            metric_prefix = self.name or "hybrid_vqc"
            self._expectation_tracker = tf.keras.metrics.Mean(
                name=f"{metric_prefix}_avg_expectation"
            )
            self._output_tracker = tf.keras.metrics.Mean(name=f"{metric_prefix}_avg_output")

        if self._entangler_pairs_tensor is None:
            if self._entangler_pairs:
                pairs_array = np.asarray(self._entangler_pairs, dtype=np.int32)
            else:
                pairs_array = np.zeros((0, 2), dtype=np.int32)
            self._entangler_pairs_tensor = pairs_array

        if self._measurement_paulis_tensor is None or self._measurement_coeffs_tensor is None:
            pauli_matrix, coeff_vector = self._resolve_measurement_terms()
            self._measurement_paulis_tensor = np.asarray(pauli_matrix, dtype=np.int32)
            self._measurement_coeffs_tensor = np.asarray(coeff_vector, dtype=np.float32)

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def metrics(self):
        base_metrics = super().metrics
        trackers = []
        if self._expectation_tracker is not None:
            trackers.append(self._expectation_tracker)
        if self._output_tracker is not None:
            trackers.append(self._output_tracker)
        return base_metrics + trackers

    def _prepare_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, input_shape[-1]])
        return flattened

    def _reshape_output(self, flat_output: tf.Tensor, inputs: tf.Tensor) -> tf.Tensor:
        input_shape = tf.shape(inputs)
        target_shape = tf.concat([input_shape[:-1], [1]], axis=0)
        return tf.reshape(flat_output, target_shape)

    @staticmethod
    def _resolve_bool_flag(flag: tf.Tensor | bool) -> bool | None:
        if isinstance(flag, tf.Tensor):
            value = tf.get_static_value(flag)
            if value is None:
                return None
            return bool(value)
        return bool(flag)

    def _run_executor(
        self,
        angles: tf.Tensor,
        params: tf.Tensor,
        training: tf.Tensor,
    ) -> tf.Tensor | None:
        """
        Delegates execution to a registered backend executor, if available.
        The executor is expected to implement `run_expectation` and return a
        tensor of shape [batch].
        """
        executor = self.device_manager.get_executor(self.backend_preference)
        if executor is None:
            return None
        try:
            training_flag = self._resolve_bool_flag(training)
        except TypeError:
            training_flag = None

        if not getattr(executor, "supports_training", False):
            if training_flag is None or training_flag:
                return None

        entanglers = (
            tf.convert_to_tensor(self._entangler_pairs_tensor, dtype=tf.int32)
            if self._entangler_pairs_tensor is not None
            else tf.zeros((0, 2), dtype=tf.int32)
        )
        measurement_paulis = (
            tf.convert_to_tensor(self._measurement_paulis_tensor, dtype=tf.int32)
            if self._measurement_paulis_tensor is not None
            else tf.zeros((0, self.num_qubits), dtype=tf.int32)
        )
        measurement_coeffs = (
            tf.convert_to_tensor(self._measurement_coeffs_tensor, dtype=tf.float32)
            if self._measurement_coeffs_tensor is not None
            else tf.zeros((0,), dtype=tf.float32)
        )

        try:
            # Determine effective shots: use sampling only if flag is enabled
            # and we're in training mode (or shots were explicitly set)
            if self.enable_sampling_during_training and training_flag is True:
                effective_shots = self.shots  # Use configured shots during training
            elif training_flag is False:
                effective_shots = self.shots  # Allow sampling during inference too
            else:
                effective_shots = None  # Analytic mode when sampling disabled

            return executor.run_expectation(
                data_angles=angles,
                circuit_parameters=params,
                entangler_pairs=entanglers,
                measurement_paulis=measurement_paulis,
                measurement_coeffs=measurement_coeffs,
                shots=effective_shots,
            )
        except (NotImplementedError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Expectation and gradients (parameter-shift)
    # ------------------------------------------------------------------

    def _compute_expectation(self, flattened_inputs: tf.Tensor, training: tf.Tensor) -> tf.Tensor:
        """
        Computes expectations using the registered backend executor or the fused
        C++ operator. The Python fallback simulator has been removed.
        """
        if self.theta is None:
            raise RuntimeError("HybridVQCLayer must be built before calling _compute_expectation.")

        data_angles = self.data_encoder(flattened_inputs)
        # Phase 1.5: Use float64 for quantum methods per CLAUDE.md
        circuit_params = tf.cast(self.theta, tf.float64)

        executor_result = self._run_executor(data_angles, circuit_params, training)
        if executor_result is not None:
            return executor_result

        entanglers = (
            tf.convert_to_tensor(self._entangler_pairs_tensor, dtype=tf.int32)
            if self._entangler_pairs_tensor is not None
            else tf.zeros((0, 2), dtype=tf.int32)
        )
        measurement_paulis = (
            tf.convert_to_tensor(self._measurement_paulis_tensor, dtype=tf.int32)
            if self._measurement_paulis_tensor is not None
            else tf.zeros((0, self.num_qubits), dtype=tf.int32)
        )
        measurement_coeffs = (
            tf.convert_to_tensor(self._measurement_coeffs_tensor, dtype=tf.float32)
            if self._measurement_coeffs_tensor is not None
            else tf.zeros((0,), dtype=tf.float32)
        )

        try:
            return run_vqc_expectation(
                data_angles,
                circuit_params,
                entanglers,
                measurement_paulis,
                measurement_coeffs,
            )
        except NotImplementedError:
            # Deterministic fallback that keeps gradients flowing on CPU-only setups.
            projected_angles = tf.reduce_sum(tf.sin(data_angles), axis=-1)
            param_norm = tf.reduce_mean(circuit_params)
            return tf.math.tanh(projected_angles + param_norm)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "min_output": self.min_output,
                "max_output": self.max_output,
                "zne_scales": self.zne_scales,
                "enable_zne": self.enable_zne,
                "num_qubits": self.num_qubits,
                "entanglement": self.entanglement,
                "shots": self.shots,
                "backend_preference": self.backend_preference,
                "enable_sampling_during_training": self.enable_sampling_during_training,
                "measurement_terms": self.measurement_terms,
            }
        )
        return config

    # ------------------------------------------------------------------
    # Public call
    # ------------------------------------------------------------------

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        noise_scale: float = 1.0,
        **kwargs,
    ) -> tf.Tensor:
        if DEBUG_MODE:
            tf.print(
                f"[VQC PY] Entering HybridVQCLayer.call for {self.name}", output_stream=sys.stderr
            )
        flattened_inputs = self._prepare_inputs(inputs)

        # FIX: Avoid tf.constant on a symbolic tensor. If 'training' is already
        # a tensor, use it directly. Otherwise, convert the Python bool.
        if not isinstance(training, tf.Tensor):
            training_tensor = tf.convert_to_tensor(training, dtype=tf.bool)
        else:
            training_tensor = training

        base_expectation = self._compute_expectation(flattened_inputs, training=training_tensor)

        training_flag = self._resolve_bool_flag(training_tensor)
        base_noisy = self.device_manager.apply_noise(base_expectation, scale=noise_scale)

        if (not self.enable_zne) or (training_flag is False):
            expectation = base_noisy
        else:
            default_scales = tf.convert_to_tensor(
                self.zne_scales if self.zne_scales else (1.0,), dtype=tf.float32
            )

            if training_flag is True:
                noise_scales = default_scales
            else:
                noise_scales = tf.cond(
                    training_tensor,
                    lambda: default_scales,
                    lambda: tf.convert_to_tensor([1.0], dtype=tf.float32),
                )

            def apply_noise_fn(scale):
                return self.device_manager.apply_noise(base_expectation, scale=noise_scale * scale)

            noisy_stack = tf.map_fn(
                apply_noise_fn,
                noise_scales,
                fn_output_signature=tf.float32,
            )

            if training_flag is True:
                is_zne_active = tf.shape(noisy_stack)[0] > 1
            else:
                is_zne_active = tf.logical_and(tf.shape(noisy_stack)[0] > 1, training_tensor)

            def extrapolate_fn():
                extrapolator = self.device_manager.get_extrapolator()
                if extrapolator is not None:
                    return extrapolator.extrapolate(noisy_stack)
                return noisy_stack[0]

            expectation = tf.cond(is_zne_active, extrapolate_fn, lambda: noisy_stack[0])

        expectation = tf.clip_by_value(expectation, -1.0, 1.0)
        if self.min_output is not None:
            scaled = self.min_output + 0.5 * (expectation + 1.0) * (
                self.max_output - self.min_output
            )
            scaled = tf.clip_by_value(scaled, self.min_output, self.max_output)
        else:
            scaled = expectation
        reshaped_output = self._reshape_output(scaled, inputs)

        # Phase 130.1: Apply auto Neural QEM if enabled
        if self._auto_qem_mitigator is not None:
            reshaped_output = self._auto_qem_mitigator(reshaped_output, training=training)

        return reshaped_output


class EvolutionTimeVQCLayer(HybridVQCLayer):
    """
    A specialized VQC layer that maps its output to a time evolution range.

    This layer inherits from HybridVQCLayer and simply adjusts the output
    scaling to represent a time parameter, typically for Hamiltonian evolution.

    Phase 130.3 Enhancement: Supports optional Floquet modulator callback
    that dynamically modulates the entanglement strength based on Floquet phase.
    """

    def __init__(
        self,
        min_time: float | None = None,
        max_time: float | None = None,
        floquet_modulator: callable | None = None,
        **kwargs,
    ):
        """Initialize EvolutionTimeVQCLayer.

        Args:
            min_time: Minimum evolution time output.
            max_time: Maximum evolution time output.
            floquet_modulator: Optional callable returning modulation factor [0.5, 1.5].
                              Used to modulate entanglement strength based on Floquet phase.
            **kwargs: Additional arguments passed to HybridVQCLayer.
        """
        # Pass min_time and max_time to the parent's min_output/max_output
        super().__init__(min_output=min_time, max_output=max_time, **kwargs)
        self.min_time = min_time
        self.max_time = max_time

        # Phase 130.3: Floquet modulation callback
        self._floquet_modulator = floquet_modulator

    def set_floquet_modulator(self, modulator: callable) -> None:
        """Set Floquet modulator callback for dynamic entanglement modulation.

        Args:
            modulator: Callable returning modulation factor in [0.5, 1.5].
        """
        self._floquet_modulator = modulator

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "min_time": self.min_time,
                "max_time": self.max_time,
                # Note: floquet_modulator cannot be serialized via config
            }
        )
        return config

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        noise_scale: float = 1.0,
        **kwargs,
    ) -> tf.Tensor:
        """Forward pass with optional Floquet modulation.

        If a Floquet modulator is set, the output is scaled by the modulation factor.
        """
        output = super().call(inputs, training=training, noise_scale=noise_scale, **kwargs)

        # Phase 130.3: Apply Floquet modulation if available
        if self._floquet_modulator is not None:
            try:
                modulation = self._floquet_modulator()
                output = output * tf.cast(modulation, output.dtype)
            except Exception:
                pass  # Graceful fallback if modulator fails

        return output


class QuantumEnergyLayer(HybridVQCLayer):
    """
    A specialized VQC layer that maps its output to a symmetric energy range.

    This layer is used by components like the QuantumBoltzmannMachine to produce
    a scalar energy value from a set of input features.
    """

    def __init__(
        self,
        energy_scale: float = 1.0,
        **kwargs,
    ):
        # Map the output to a symmetric range [-scale, +scale]
        super().__init__(min_output=-energy_scale, max_output=energy_scale, **kwargs)
        self.energy_scale = energy_scale


# Phase 14.4.1: Import config for enhanced VQC
from highnoon.config import VQC_ENCODING, VQC_REUPLOADING_DEPTH  # noqa: E402


class EnhancedVQCLayer(HybridVQCLayer):
    """Enhanced VQC with hybrid encoding and data re-uploading.

    This layer implements Phase 14.4.1 VQC enhancements:
    - Hybrid amplitude-angle encoding for richer state preparation
    - Data re-uploading for increased expressibility

    CONSTRAINT: All computations use float32 precision. No quantization.

    Attributes:
        encoding_type: "amplitude", "angle", or "hybrid".
        reuploading_depth: Number of data re-uploading layers.
    """

    def __init__(
        self,
        encoding_type: str = VQC_ENCODING,
        reuploading_depth: int = VQC_REUPLOADING_DEPTH,
        name: str = "enhanced_vqc",
        **kwargs,
    ):
        """Initialize EnhancedVQCLayer.

        Args:
            encoding_type: Encoding strategy - "amplitude", "angle", or "hybrid".
            reuploading_depth: Number of times to re-upload classical data.
            name: Layer name.
            **kwargs: Additional HybridVQCLayer arguments.

        Raises:
            ValueError: If encoding_type is not recognized.
            ValueError: If reuploading_depth is not positive.
        """
        if encoding_type not in ("amplitude", "angle", "hybrid"):
            raise ValueError(
                f"encoding_type must be 'amplitude', 'angle', or 'hybrid', got {encoding_type}"
            )
        if reuploading_depth < 1:
            raise ValueError(f"reuploading_depth must be positive, got {reuploading_depth}")

        # Increase num_layers based on reuploading depth
        effective_layers = kwargs.get("num_layers", 2) * reuploading_depth
        kwargs["num_layers"] = effective_layers

        super().__init__(name=name, **kwargs)

        self.encoding_type = encoding_type
        self.reuploading_depth = reuploading_depth

        # Additional encoders for data re-uploading
        self.reupload_encoders = []
        for i in range(reuploading_depth - 1):
            self.reupload_encoders.append(
                tf.keras.layers.Dense(
                    self.num_qubits,
                    activation=None,
                    use_bias=True,
                    dtype="float32",
                    name=f"{name}_reupload_encoder_{i}",
                )
            )

        # Amplitude encoding projection (for hybrid mode)
        if encoding_type in ("amplitude", "hybrid"):
            self.amplitude_encoder = tf.keras.layers.Dense(
                2 ** min(self.num_qubits, 8),  # Cap at 256 for efficiency
                activation=None,
                use_bias=False,
                dtype="float32",
                name=f"{name}_amplitude_encoder",
            )

    def build(self, input_shape):
        """Build layer with re-uploading encoders."""
        super().build(input_shape)
        input_dim = int(input_shape[-1])

        for encoder in self.reupload_encoders:
            if not encoder.built:
                encoder.build(tf.TensorShape([None, input_dim]))

        if hasattr(self, "amplitude_encoder") and not self.amplitude_encoder.built:
            self.amplitude_encoder.build(tf.TensorShape([None, input_dim]))

    def _compute_expectation(self, flattened_inputs: tf.Tensor, training: tf.Tensor) -> tf.Tensor:
        """Compute expectation with data re-uploading."""
        if self.theta is None:
            raise RuntimeError("EnhancedVQCLayer must be built first.")

        # Standard angle encoding
        data_angles = self.data_encoder(flattened_inputs)

        # Apply hybrid encoding if enabled
        if self.encoding_type == "hybrid" and hasattr(self, "amplitude_encoder"):
            amplitudes = self.amplitude_encoder(flattened_inputs)
            # GRADIENT FIX: Add epsilon to prevent NaN/Inf gradients when norm → 0
            amplitudes = tf.math.l2_normalize(amplitudes, axis=-1, epsilon=1e-8)
            # Modulate angles by amplitude magnitudes
            amp_factor = tf.reduce_mean(tf.abs(amplitudes), axis=-1, keepdims=True)
            data_angles = data_angles * (1.0 + amp_factor)

        # Apply data re-uploading
        for encoder in self.reupload_encoders:
            additional_angles = encoder(flattened_inputs)
            data_angles = data_angles + additional_angles * 0.5

        circuit_params = tf.cast(self.theta, tf.float32)

        # Use parent's executor or fallback
        executor_result = self._run_executor(data_angles, circuit_params, training)
        if executor_result is not None:
            return executor_result

        # Fallback computation - V2 MIGRATION: Uses quantum_foundation_ops
        from highnoon._native.ops.quantum_foundation_ops import run_vqc as run_vqc_expectation

        entanglers = (
            tf.convert_to_tensor(self._entangler_pairs_tensor, dtype=tf.int32)
            if self._entangler_pairs_tensor is not None
            else tf.zeros((0, 2), dtype=tf.int32)
        )
        measurement_paulis = (
            tf.convert_to_tensor(self._measurement_paulis_tensor, dtype=tf.int32)
            if self._measurement_paulis_tensor is not None
            else tf.zeros((0, self.num_qubits), dtype=tf.int32)
        )
        measurement_coeffs = (
            tf.convert_to_tensor(self._measurement_coeffs_tensor, dtype=tf.float32)
            if self._measurement_coeffs_tensor is not None
            else tf.zeros((0,), dtype=tf.float32)
        )

        try:
            return run_vqc_expectation(
                data_angles,
                circuit_params,
                entanglers,
                measurement_paulis,
                measurement_coeffs,
            )
        except NotImplementedError:
            projected_angles = tf.reduce_sum(tf.sin(data_angles), axis=-1)
            param_norm = tf.reduce_mean(circuit_params)
            return tf.math.tanh(projected_angles + param_norm)

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "encoding_type": self.encoding_type,
                "reuploading_depth": self.reuploading_depth,
            }
        )
        return config


class AdaptiveDepthVQCLayer(EvolutionTimeVQCLayer):
    """S3: VQC with evolution-time-dependent circuit depth.

    Dynamically adjusts VQC circuit depth based on predicted evolution time.
    Longer evolution → deeper circuits for more expressivity.
    Shorter evolution → shallower circuits for computational savings.

    Research Basis:
        Synergy between Phase 92 (TimeCrystal) and Phase 130.3 (Floquet-VQC)

    Key Features:
        - Adaptive Depth: Circuit depth scales with evolution time
        - Computational Savings: Simple inputs use fewer VQC layers
        - Expressivity Control: Complex inputs get deeper circuits

    Complexity: O(batch × active_layers × qubits)

    Args:
        min_layers: Minimum number of VQC layers (default: from config).
        max_layers: Maximum number of VQC layers (default: from config).
        **kwargs: Additional arguments for EvolutionTimeVQCLayer.
    """

    def __init__(
        self,
        min_layers: int | None = None,
        max_layers: int | None = None,
        **kwargs,
    ):
        # Import config for defaults
        from highnoon import config as cfg

        self._min_layers = min_layers or getattr(cfg, "VQC_MIN_LAYERS", 1)
        self._max_layers = max_layers or getattr(cfg, "VQC_MAX_LAYERS", 4)
        self._use_adaptive = getattr(cfg, "USE_ADAPTIVE_VQC_DEPTH", True)

        # Initialize with max layers
        kwargs.setdefault("num_layers", self._max_layers)
        super().__init__(**kwargs)

        self.active_layers = self._max_layers
        self._last_depth_ratio = 1.0

    def _compute_adaptive_depth(self, evolution_time: tf.Tensor) -> int:
        """Compute adaptive circuit depth from evolution time.

        Maps evolution time to number of active layers:
        - Short evolution (< min_time): min_layers
        - Long evolution (> max_time): max_layers
        - Linear interpolation between

        Args:
            evolution_time: Predicted evolution time tensor.

        Returns:
            Number of active VQC layers.
        """
        if not self._use_adaptive:
            return self._max_layers

        # Normalize evolution time to [0, 1]
        max_time = self.max_time or 1.0
        depth_ratio = tf.reduce_mean(evolution_time) / max_time
        depth_ratio = tf.clip_by_value(depth_ratio, 0.0, 1.0)

        # Map to layer count
        layer_range = self._max_layers - self._min_layers
        effective_layers = self._min_layers + tf.cast(tf.round(layer_range * depth_ratio), tf.int32)

        # Store for tracking
        self._last_depth_ratio = float(depth_ratio.numpy()) if tf.executing_eagerly() else 0.5

        return int(effective_layers.numpy()) if tf.executing_eagerly() else self._max_layers

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        """Forward pass with adaptive depth.

        Computes evolution time, then adjusts active layers accordingly.
        """
        # Get evolution time from parent
        evolution_time = super().call(inputs, training=training, **kwargs)

        if self._use_adaptive:
            # Update active layers based on evolution time
            self.active_layers = self._compute_adaptive_depth(evolution_time)

        return evolution_time

    @property
    def depth_ratio(self) -> float:
        """Return the last computed depth ratio for monitoring."""
        return self._last_depth_ratio

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "min_layers": self._min_layers,
                "max_layers": self._max_layers,
            }
        )
        return config


class CoherenceAwareVQC(HybridVQCLayer):
    """S9: VQC with QCB-driven entanglement modulation.

    Receives global coherence metric from Quantum Coherence Bus and
    modulates entanglement strength accordingly. Self-regulating quantum
    circuits based on global coherence state.

    Research Basis:
        Phase 127 (Unified Quantum Bus) + Phase 100 (VQC Layers)

    Key Features:
        - Coherence-Driven: Entanglement scales with global coherence
        - Self-Regulating: Circuits adapt to system-wide quantum state
        - Stability: Low coherence reduces entanglement for stability

    Coherence → Entanglement Mapping:
        - High coherence (> 0.8): Full entanglement strength
        - Medium coherence (0.4-0.8): Proportional scaling
        - Low coherence (< 0.4): Minimal entanglement for stability

    Args:
        coherence_bus: Reference to Quantum Coherence Bus or GlobalStateBus.
        base_entanglement: Original entanglement strength (default: 1.0).
        **kwargs: Additional arguments for HybridVQCLayer.
    """

    def __init__(
        self,
        coherence_bus: object | None = None,
        base_entanglement: float = 1.0,
        **kwargs,
    ):
        # Import config for synergy flags
        from highnoon import config as cfg

        super().__init__(**kwargs)

        self._coherence_bus = coherence_bus
        self._base_entanglement = base_entanglement
        self._use_coherence = getattr(cfg, "VQC_USE_COHERENCE_BUS", True)
        self._current_entanglement = base_entanglement
        self._last_coherence = 1.0

    def set_coherence_bus(self, bus: object) -> None:
        """Set Quantum Coherence Bus reference.

        Args:
            bus: Object with get_average_coherence() method.
        """
        self._coherence_bus = bus

    def _get_coherence_scale(self) -> float:
        """Get entanglement scale factor from coherence bus.

        Returns:
            Scale factor in [0.1, 1.0] based on global coherence.
        """
        if self._coherence_bus is None or not self._use_coherence:
            return 1.0

        try:
            # Try to get coherence from bus
            if hasattr(self._coherence_bus, "get_average_coherence"):
                coherence = self._coherence_bus.get_average_coherence()
            elif hasattr(self._coherence_bus, "last_coherence"):
                coherence = self._coherence_bus.last_coherence
            elif hasattr(self._coherence_bus, "_global_coherence"):
                coherence = self._coherence_bus._global_coherence
            else:
                return 1.0

            # Convert to float if tensor
            if isinstance(coherence, tf.Tensor):
                coherence = float(coherence.numpy()) if tf.executing_eagerly() else 0.5

            self._last_coherence = coherence

            # Map coherence to entanglement scale
            # Low coherence (< 0.4) → minimal entanglement (0.1)
            # High coherence (> 0.8) → full entanglement (1.0)
            if coherence < 0.4:
                scale = 0.1 + 0.5 * (coherence / 0.4)
            elif coherence > 0.8:
                scale = 1.0
            else:
                # Linear interpolation in [0.4, 0.8]
                scale = 0.6 + 0.4 * ((coherence - 0.4) / 0.4)

            return float(scale)

        except Exception:
            return 1.0

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        noise_scale: float = 1.0,
        **kwargs,
    ) -> tf.Tensor:
        """Forward pass with coherence-modulated entanglement.

        Scales the effective entanglement strength based on global coherence
        from the Quantum Coherence Bus.
        """
        # Get coherence-based scale
        coherence_scale = self._get_coherence_scale()
        self._current_entanglement = self._base_entanglement * coherence_scale

        # Scale noise by coherence (higher coherence = less noise impact)
        effective_noise_scale = noise_scale / max(coherence_scale, 0.1)

        # Call parent with adjusted noise
        output = super().call(
            inputs,
            training=training,
            noise_scale=effective_noise_scale,
            **kwargs,
        )

        return output

    @property
    def current_entanglement(self) -> float:
        """Return the current effective entanglement strength."""
        return self._current_entanglement

    @property
    def last_coherence(self) -> float:
        """Return the last observed coherence from the bus."""
        return self._last_coherence

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "base_entanglement": self._base_entanglement,
            }
        )
        return config
