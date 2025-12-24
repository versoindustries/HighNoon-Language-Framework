# src/quantum/executors/qiskit_executor.py
#
# Optional Qiskit-backed executor that can be registered with the
# QuantumDeviceManager. It provides high-fidelity state evolution with optional
# noise models using Aer density-matrix simulation. The executor is designed for
# inference and evaluation scenarios; parameter-shift gradients are not exposed.

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector

    try:
        from qiskit_aer import AerSimulator  # type: ignore
    except ImportError:  # pragma: no cover - Aer not installed
        AerSimulator = None  # type: ignore

    _HAS_QISKIT = True
except ImportError:  # pragma: no cover - Qiskit not installed
    QuantumCircuit = None  # type: ignore
    transpile = None  # type: ignore
    DensityMatrix = None  # type: ignore
    SparsePauliOp = None  # type: ignore
    Statevector = None  # type: ignore
    AerSimulator = None  # type: ignore
    _HAS_QISKIT = False


def is_qiskit_available() -> bool:
    """Returns True when Qiskit (and its quantum_info primitives) can be imported."""
    return _HAS_QISKIT


class QiskitVQCExecutor:
    """
    Executes HSMN variational quantum circuits using Qiskit.

    The executor mirrors the signature expected by :class:`HybridVQCLayer` and
    produces TensorFlow tensors so it can operate within eager or graph mode
    through `tf.py_function`. Gradients are not provided; callers should rely on
    the TensorFlow custom op for training and use this executor for evaluation,
    calibration, or hardware-aligned studies.
    """

    supports_training: bool = False  # Parameter-shift gradients are not exposed.

    _PAULI_SYMBOLS = ("I", "X", "Y", "Z")

    def __init__(
        self,
        num_qubits: int,
        *,
        noise_model: object | None = None,
        default_shots: int | None = None,
    ) -> None:
        if not _HAS_QISKIT:
            raise ImportError("Qiskit is not available in this environment.")

        self.num_qubits = int(num_qubits)
        self._noise_model = noise_model
        self._default_shots = default_shots

        # Cache compiled observables keyed by the measurement tensor signature.
        self._observable_cache: dict[tuple[tuple[int, ...], tuple[float, ...]], SparsePauliOp] = {}

        if AerSimulator is None and noise_model is not None:
            logger.warning(
                "Qiskit noise model provided but qiskit-aer is unavailable; "
                "falling back to noiseless statevector simulation."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_noise_model(self, noise_model: object | None) -> None:
        """Injects or clears the active Qiskit noise model."""
        self._noise_model = noise_model
        if noise_model is not None and AerSimulator is None:
            logger.warning(
                "Qiskit noise model assigned but qiskit-aer is unavailable; "
                "noise will not be applied."
            )

    def run_expectation(
        self,
        *,
        data_angles: tf.Tensor,
        circuit_parameters: tf.Tensor,
        entangler_pairs: tf.Tensor,
        measurement_paulis: tf.Tensor,
        measurement_coeffs: tf.Tensor,
        shots: int | None = None,
    ) -> tf.Tensor:
        """
        Evaluates expectation values for the provided batch of inputs.

        The returned tensor has shape [batch] and resides on CPU memory.
        """

        if not tf.executing_eagerly():
            # Wrap the numpy heavy lifting in a py_function to keep graph traces valid.
            return self._run_via_py_function(
                data_angles,
                circuit_parameters,
                entangler_pairs,
                measurement_paulis,
                measurement_coeffs,
                shots,
            )

        # Eager path (mostly used for debugging/inference).
        values = self._evaluate_numpy(
            np.asarray(data_angles.numpy(), dtype=np.float32),
            np.asarray(circuit_parameters.numpy(), dtype=np.float32),
            np.asarray(entangler_pairs.numpy(), dtype=np.int32),
            np.asarray(measurement_paulis.numpy(), dtype=np.int32),
            np.asarray(measurement_coeffs.numpy(), dtype=np.float32),
            shots=shots,
        )
        return tf.convert_to_tensor(values, dtype=tf.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_via_py_function(
        self,
        data_angles: tf.Tensor,
        circuit_parameters: tf.Tensor,
        entangler_pairs: tf.Tensor,
        measurement_paulis: tf.Tensor,
        measurement_coeffs: tf.Tensor,
        shots: int | None,
    ) -> tf.Tensor:
        def _wrapper(angles, params, entanglers, paulis, coeffs):
            return self._evaluate_numpy(
                np.asarray(angles, dtype=np.float32),
                np.asarray(params, dtype=np.float32),
                np.asarray(entanglers, dtype=np.int32),
                np.asarray(paulis, dtype=np.int32),
                np.asarray(coeffs, dtype=np.float32),
                shots=shots,
            )

        result = tf.py_function(
            func=_wrapper,
            inp=[
                data_angles,
                circuit_parameters,
                entangler_pairs,
                measurement_paulis,
                measurement_coeffs,
            ],
            Tout=tf.float32,
        )
        batch_dim = data_angles.shape[0]
        if batch_dim is not None:
            result.set_shape([int(batch_dim)])
        else:
            result.set_shape([None])
        return result

    def _evaluate_numpy(
        self,
        angles: np.ndarray,
        params: np.ndarray,
        entanglers: np.ndarray,
        measurement_paulis: np.ndarray,
        measurement_coeffs: np.ndarray,
        *,
        shots: int | None,
    ) -> np.ndarray:
        if angles.ndim != 2:
            raise ValueError(
                f"data_angles must have shape [batch, qubits], received {angles.shape}."
            )

        batch_size, num_qubits = angles.shape
        if num_qubits != self.num_qubits:
            raise ValueError(
                f"Executor configured for {self.num_qubits} qubits but received angles for {num_qubits}."
            )
        if params.ndim != 3:
            raise ValueError(
                f"circuit_parameters must have shape [layers, qubits, 2], received {params.shape}."
            )

        num_layers = params.shape[0]
        entangler_pairs: list[tuple[int, int]] = (
            [(int(pair[0]), int(pair[1])) for pair in entanglers.reshape(-1, 2)]
            if entanglers.size
            else []
        )

        observable = self._resolve_observable(measurement_paulis, measurement_coeffs, num_qubits)
        density_objects = self._simulate_circuits(
            angles, params, entangler_pairs, num_layers, shots
        )

        expectations = [
            float(np.real(rho.expectation_value(observable))) for rho in density_objects
        ]
        return np.asarray(expectations, dtype=np.float32)

    def _simulate_circuits(
        self,
        angles: np.ndarray,
        params: np.ndarray,
        entangler_pairs: Sequence[tuple[int, int]],
        num_layers: int,
        shots: int | None,
    ) -> list[DensityMatrix]:
        circuits: list[QuantumCircuit] = []
        for batch_row in angles:
            qc = QuantumCircuit(self.num_qubits)
            for qubit, theta in enumerate(batch_row):
                qc.ry(float(theta), qubit)
            for layer in range(num_layers):
                for qubit in range(self.num_qubits):
                    qc.rz(float(params[layer, qubit, 0]), qubit)
                for qubit in range(self.num_qubits):
                    qc.ry(float(params[layer, qubit, 1]), qubit)
                for control, target in entangler_pairs:
                    qc.cz(control, target)
            circuits.append(qc)

        if self._noise_model is not None and AerSimulator is not None:
            simulator = AerSimulator(method="density_matrix", noise_model=self._noise_model)
            compiled = transpile(circuits, simulator)
            result = simulator.run(compiled, shots=shots or self._default_shots).result()
            density_matrices = [
                DensityMatrix(result.data(i)["density_matrix"]) for i in range(len(circuits))
            ]
        else:
            density_matrices = [
                DensityMatrix(Statevector.from_instruction(circuit)) for circuit in circuits
            ]
        return density_matrices

    def _resolve_observable(
        self,
        measurement_paulis: np.ndarray,
        measurement_coeffs: np.ndarray,
        num_qubits: int,
    ) -> SparsePauliOp:
        if measurement_paulis.size == 0 or measurement_coeffs.size == 0:
            weights = [1.0 / float(num_qubits)] * num_qubits
            pauli_labels = []
            for qubit in range(num_qubits):
                label = ["I"] * num_qubits
                label[qubit] = "Z"
                pauli_labels.append("".join(label))
            pauli_tuples = tuple(pauli_labels)
            coeff_tuple = tuple(weights)
        else:
            pauli_rows = [
                tuple(int(v) for v in row) for row in measurement_paulis.reshape(-1, num_qubits)
            ]
            coeff_tuple = tuple(float(c) for c in measurement_coeffs.reshape(-1))
            pauli_labels = ["".join(self._PAULI_SYMBOLS[val] for val in row) for row in pauli_rows]
            pauli_tuples = tuple(pauli_labels)

        cache_key = (pauli_tuples, coeff_tuple)
        if cache_key in self._observable_cache:
            return self._observable_cache[cache_key]

        terms = list(zip(pauli_tuples, coeff_tuple))
        observable = SparsePauliOp.from_list(terms)
        self._observable_cache[cache_key] = observable
        return observable
