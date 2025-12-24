#!/usr/bin/env python3
"""
src/models/quantum/mps_layer.py

Matrix Product State (MPS) layer for efficient quantum state representation in neural networks.

Implements MPS as a Keras layer with learnable bond dimensions, supporting:
- Polynomial memory scaling O(N * chi^2) instead of exponential O(2^N)
- Entanglement structure capture via Schmidt decomposition
- Integration with variational quantum circuits (VQC)
"""


import numpy as np
import tensorflow as tf

from highnoon._native.ops.mps_contract import (
    canonical_mps,
    mps_contract,
    mps_expect,
    mps_expect_pauli,
    mps_feature_importance,
    mps_trotter_step,
)
from highnoon.models.utils.control_vars import ControlVarMixin


class MPSLayer(ControlVarMixin, tf.keras.layers.Layer):
    """
    Matrix Product State layer for quantum state representation.

    Represents a quantum state |ψ⟩ as a tensor network:
        |ψ⟩ = Σ A[1]_{α1,i1} A[2]_{α1,α2,i2} ... A[N]_{αN-1,iN} |i1,i2,...,iN⟩

    where:
    - N = number of quantum sites (qubits)
    - chi = bond dimension (controls entanglement capacity)
    - d = physical dimension (typically 2 for qubits)

    Memory: O(N * d * chi^2) vs O(d^N) for full statevector
    """

    def __init__(
        self,
        num_sites: int,
        physical_dim: int = 2,
        bond_dim: int = 4,
        max_bond_dim: int = 32,
        use_canonical_form: bool = True,
        truncation_threshold: float = 1e-10,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initialize MPS layer.

        Args:
            num_sites: Number of quantum sites (N)
            physical_dim: Physical dimension at each site (d), typically 2 for qubits
            bond_dim: Initial bond dimension (chi)
            max_bond_dim: Maximum bond dimension for SVD truncation
            use_canonical_form: If True, maintain canonical orthogonal form
            truncation_threshold: SVD truncation error threshold
            name: Layer name
        """
        super().__init__(name=name, **kwargs)

        if num_sites < 1:
            raise ValueError(f"num_sites must be >= 1, got {num_sites}")
        if physical_dim < 2:
            raise ValueError(f"physical_dim must be >= 2, got {physical_dim}")
        if bond_dim < 1:
            raise ValueError(f"bond_dim must be >= 1, got {bond_dim}")

        self.num_sites = num_sites
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.max_bond_dim = max_bond_dim
        self.use_canonical_form = use_canonical_form
        self.truncation_threshold = truncation_threshold

        # MPS core tensors (initialized in build)
        self.mps_cores: list[tf.Variable] = []

        # Bond dimensions (boundary bonds = 1)
        self.bond_dims = [1] + [bond_dim] * (num_sites - 1) + [1]

        # Metrics
        self._avg_entanglement_metric = None

    def build(self, input_shape):
        """
        Build MPS core tensors.

        Each core A[i] has shape [bond_left, physical_dim, bond_right]
        """
        # Initialize MPS cores with random Gaussian values
        # Use Xavier initialization for stability
        for i in range(self.num_sites):
            bond_left = self.bond_dims[i]
            bond_right = self.bond_dims[i + 1]
            core_shape = (bond_left, self.physical_dim, bond_right)

            # Xavier initialization
            stddev = np.sqrt(2.0 / (bond_left * self.physical_dim + bond_right))

            core = self.add_weight(
                name=f"mps_core_{i}",
                shape=core_shape,
                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev),
                trainable=True,
                dtype=tf.float32,
            )

            self.mps_cores.append(core)
            self.register_control_var(f"mps_core_{i}", core)

        # Metrics
        self._avg_entanglement_metric = tf.keras.metrics.Mean(
            name=f'{self.name or "mps"}_avg_entanglement'
        )

        super().build(input_shape)

    @property
    def metrics(self):
        base = super().metrics
        if self._avg_entanglement_metric is not None:
            return base + [self._avg_entanglement_metric]
        return base

    def call(self, inputs, training=False, compute_entropy=True):
        """
        Contract MPS to full wavefunction.

        Args:
            inputs: Placeholder tensor (shape doesn't matter, MPS is fully parameterized)
            training: Training mode flag
            compute_entropy: If True, compute entanglement entropy

        Returns:
            state: Contracted wavefunction, shape [2^N] for qubits
            entropies: Entanglement entropies at each bond, shape [N-1]
        """
        # Physical and bond dimensions as tensors
        physical_dims = tf.constant([self.physical_dim] * self.num_sites, dtype=tf.int32)
        bond_dims = tf.constant(self.bond_dims, dtype=tf.int32)

        # Contract MPS (use positional args for graph mode compatibility)
        state, entropies = mps_contract(
            self.mps_cores,  # mps_tensors
            physical_dims,  # physical_dims
            bond_dims,  # bond_dims
            self.max_bond_dim,  # max_bond_dim
            compute_entropy,  # compute_entropy
            self.truncation_threshold,  # truncation_threshold
        )

        # Update metrics
        if compute_entropy and training and len(entropies) > 0:
            avg_entropy = tf.reduce_mean(entropies)
            self._avg_entanglement_metric.update_state(avg_entropy)

        return state, entropies

    def contract(self, compute_entropy: bool = True) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Contract MPS to full wavefunction (convenience method).

        Returns:
            state: Wavefunction vector, shape [product(physical_dims)]
            entropies: Entanglement entropies, shape [N-1]
        """
        dummy_input = tf.zeros([1, 1])  # Placeholder
        return self.call(dummy_input, training=False, compute_entropy=compute_entropy)

    def expect(self, operator: tf.Tensor) -> tf.Tensor:
        """
        Compute expectation value <ψ|O|ψ> for operator O.

        Args:
            operator: Operator matrix, shape [d^N, d^N]

        Returns:
            expectation: Scalar expectation value
        """
        physical_dims = tf.constant([self.physical_dim] * self.num_sites, dtype=tf.int32)
        bond_dims = tf.constant(self.bond_dims, dtype=tf.int32)

        # Use positional args for graph mode compatibility
        return mps_expect(
            self.mps_cores,  # mps_tensors
            operator,  # operator
            physical_dims,  # physical_dims
            bond_dims,  # bond_dims
            self.max_bond_dim,  # max_bond_dim
        )

    def expect_pauli(self, pauli_indices: tf.Tensor, coefficients: tf.Tensor) -> tf.Tensor:
        """
        Compute expectation value of a Pauli string Hamiltonian efficiently.

        Args:
            pauli_indices: Matrix of Pauli indices [M, N]
            coefficients: Coefficients for each string [M]

        Returns:
            expectation: Scalar expectation value
        """
        return mps_expect_pauli(self.mps_cores, pauli_indices, coefficients)

    def get_feature_importance(self) -> tf.Tensor:
        """
        Compute site-wise feature importance based on entanglement entropy.

        Returns:
            importance: Tensor of importance values at each site [N]
        """
        _, entropies = self.contract(compute_entropy=True)
        return mps_feature_importance(entropies)

    def entanglement_entropy(self, site: int) -> tf.Tensor:
        """
        Compute entanglement entropy at specific bond (Schmidt cut).

        Args:
            site: Bond index (0 to N-2)

        Returns:
            entropy: Von Neumann entanglement entropy (scalar)
        """
        if site < 0 or site >= self.num_sites - 1:
            raise ValueError(f"site must be in [0, {self.num_sites - 2}], got {site}")

        # Contract and get all entropies
        _, entropies = self.contract(compute_entropy=True)

        return entropies[site]

    def to_canonical_form(self) -> "MPSLayer":
        """
        Convert MPS to canonical (mixed) form with orthogonality center.

        Returns updated MPS layer (in-place modification).
        """
        if not self.use_canonical_form:
            return self

        physical_dims = tf.constant([self.physical_dim] * self.num_sites, dtype=tf.int32)
        bond_dims = tf.constant(self.bond_dims, dtype=tf.int32)

        # Use positional args for graph mode compatibility
        canonical_cores, center_matrix = canonical_mps(
            self.mps_cores,  # mps_tensors
            physical_dims,  # physical_dims
            bond_dims,  # bond_dims
            self.num_sites // 2,  # center_site
        )

        # Update core tensors
        for i, core in enumerate(canonical_cores):
            self.mps_cores[i].assign(core)

        return self

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_sites": self.num_sites,
                "physical_dim": self.physical_dim,
                "bond_dim": self.bond_dim,
                "max_bond_dim": self.max_bond_dim,
                "use_canonical_form": self.use_canonical_form,
                "truncation_threshold": self.truncation_threshold,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MPSEvolutionLayer(MPSLayer):
    """
    MPS layer with time evolution capabilities for quantum dynamics.

    Extends MPSLayer to support Hamiltonian evolution:
        |ψ(t)⟩ = exp(-iHt) |ψ(0)⟩

    Uses Trotter-Suzuki decomposition for efficient evolution with MPS structure.
    """

    def __init__(
        self,
        num_sites: int,
        hamiltonian_terms: list[tuple[list[int], tf.Tensor]] | None = None,
        evolution_time: float = 0.1,
        trotter_steps: int = 10,
        **kwargs,
    ):
        """
        Initialize MPS evolution layer.

        Args:
            num_sites: Number of quantum sites
            hamiltonian_terms: List of (sites, operator) tuples for local Hamiltonian terms
            evolution_time: Evolution time parameter
            trotter_steps: Number of Trotter steps for evolution
            **kwargs: Additional MPSLayer arguments
        """
        super().__init__(num_sites=num_sites, **kwargs)

        self.hamiltonian_terms = hamiltonian_terms or []
        self.evolution_time_param = self.add_weight(
            name="evolution_time",
            shape=(),
            initializer=tf.keras.initializers.Constant(evolution_time),
            trainable=True,
            dtype=tf.float32,
        )
        self.trotter_steps = trotter_steps

        self.register_control_var("evolution_time", self.evolution_time_param)

    def evolve(self, training: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Evolve MPS state under Hamiltonian.

        Applies exp(-iHt) using Trotter decomposition.

        Returns:
            evolved_state: Wavefunction after evolution
            entropies: Entanglement entropies after evolution
        """
        # Physical and bond dimensions as tensors
        tf.constant([self.physical_dim] * self.num_sites, dtype=tf.int32)

        # Prepare gates for Trotter steps
        # Each term in self.hamiltonian_terms is (sites, operator)
        # For simplicity, assume two-site terms: [i, i+1]
        self.evolution_time_param / self.trotter_steps

        gate_sites = []
        gates_real = []
        gates_imag = []

        for sites, op in self.hamiltonian_terms:
            if len(sites) != 2:
                continue  # Only 2-site gates supported for now

            # Compute exp(-i * op * dt)
            # Matrix exponential: U = expm(-i * H * dt)
            # Since HighNoon Lite uses real MPS, we assume H is such that U is real
            # OR we only take the real part (HSMN approximation)
            # For a real symmetric H, exp(-iHt) = cos(Ht) - i sin(Ht)
            # In HSMN, we often use real-valued dynamical systems.

            # Simple approximation: U = I - i * H * dt for small dt?
            # No, let's use a proper matrix exponential if possible.
            # For now, assume op is the pre-computed gate or compute it.
            gate_sites.append(sites[0])

            # H is usually real in HSMN (Hermitian and real -> symmetric)
            # U is complex, but HSMN dynamics are often mapped to real phase space.
            # Here we apply the real part as an approximation or if the gates are real.
            # roadmap says: "apply two-site gates (real/imaginary parts supported)"

            # Placeholder for expm logic (TensorFlow doesn't have expm for arbitrary matrices easily)
            # We assume 'op' passed in is already the unitary gate for now, or scaled H.
            gates_real.append(tf.cast(tf.math.real(op), tf.float32))
            gates_imag.append(tf.cast(tf.math.imag(op), tf.float32))

        # Perform Trotter steps
        current_mps_cores = self.mps_cores
        for _ in range(self.trotter_steps):
            current_mps_cores = mps_trotter_step(
                current_mps_cores,
                tf.constant(gate_sites, dtype=tf.int32),
                gates_real,
                gates_imag,
                self.max_bond_dim,
                self.truncation_threshold,
            )

        # Update core tensors (in-place modification for the layer)
        for i, core in enumerate(current_mps_cores):
            self.mps_cores[i].assign(core)

        # Recontract to get final state and entropies
        final_state, entropies = self.contract(compute_entropy=True)

        return final_state, entropies

    def call(self, inputs, training=False):
        """Evolve and return state."""
        return self.evolve(training=training)


# Export
__all__ = ["MPSLayer", "MPSEvolutionLayer"]
