# highnoon/_native/ops/fused_unified_quantum_block_op.py
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

"""Python wrappers for Unified Quantum Block operations (Phases 19-24).

This module provides TensorFlow bindings for the quantum-enhanced kernels:
- Holographic Bind/Unbind (Phase 19.1)
- Port-Hamiltonian Step (Phase 19.2)
- Thermodynamic Routing (Phase 19.4)
- Orthogonalize Keys (Phase 22.2)
- QSVT Activation (Phase 24.1)
- Quantum Reservoir (Phase 24.3)

All operations use float32/float64 precision with no quantization.
"""

from __future__ import annotations

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

# =============================================================================
# Load Native Ops Library
# =============================================================================

_lib = None
_ops_available = False


def _ensure_lib_loaded():
    """Ensure the native ops library is loaded."""
    global _lib, _ops_available
    if _lib is not None:
        return _ops_available

    try:
        from highnoon._native.ops.lib_loader import get_highnoon_core_path

        lib_path = get_highnoon_core_path()
        _lib = tf.load_op_library(lib_path)
        _ops_available = True
        logger.debug("Unified quantum ops loaded from %s", lib_path)
    except Exception as e:
        logger.warning("Failed to load unified quantum ops: %s", e)
        _ops_available = False

    return _ops_available


def unified_quantum_ops_available() -> bool:
    """Check if unified quantum C++ ops are available."""
    return _ensure_lib_loaded()


# =============================================================================
# Phase 19.1: Holographic Associative Memory
# =============================================================================


def holographic_bind(a: tf.Tensor, b: tf.Tensor, name: str = "holographic_bind") -> tf.Tensor:
    """Holographic binding via circular convolution (FFT-based).

    Binds two vectors using circular convolution:
    bind(a, b) = ifft(fft(a) * fft(b))

    Args:
        a: First tensor to bind [B, D] where D must be power of 2.
        b: Second tensor to bind [B, D].
        name: Op name.

    Returns:
        Bound tensor [B, D].
    """
    if not _ensure_lib_loaded():
        raise RuntimeError(
            "Unified quantum C++ ops not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    return _lib.high_noon_holographic_bind(a, b, name=name)


@tf.RegisterGradient("HighNoonHolographicBind")
def _holographic_bind_grad(op, grad):
    """Gradient for holographic bind is unbind."""
    a, b = op.inputs
    # ∂L/∂a = unbind(grad, b), ∂L/∂b = unbind(grad, a)
    grad_a = holographic_unbind(grad, b)
    grad_b = holographic_unbind(grad, a)
    return grad_a, grad_b


def holographic_unbind(
    composite: tf.Tensor, key: tf.Tensor, name: str = "holographic_unbind"
) -> tf.Tensor:
    """Holographic unbinding (inverse of bind).

    unbind(c, a) = ifft(fft(c) * conj(fft(a)))

    Args:
        composite: Composite vector [B, D].
        key: Key vector to unbind [B, D].
        name: Op name.

    Returns:
        Unbound tensor [B, D].
    """
    if not _ensure_lib_loaded():
        raise RuntimeError(
            "Unified quantum C++ ops not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    # Use the bind op with conjugated key for unbinding
    # unbind(c, k) = bind(c, reverse(k))
    key_reversed = tf.reverse(key, axis=[-1])
    return _lib.high_noon_holographic_bind(composite, key_reversed, name=name)


# =============================================================================
# Phase 19.2: Port-Hamiltonian Systems
# =============================================================================


def port_hamiltonian_step(
    state: tf.Tensor,
    j_matrix: tf.Tensor,
    r_matrix: tf.Tensor,
    grad_h: tf.Tensor,
    external_input: tf.Tensor | None = None,
    dt: float = 0.01,
    name: str = "port_hamiltonian_step",
) -> tf.Tensor:
    """Port-Hamiltonian integration step with dissipation.

    Implements: ẋ = [J(x) - R(x)]∇H(x) + g(x)u

    Args:
        state: Current state [B, D].
        j_matrix: Interconnection (skew-symmetric) [D, D].
        r_matrix: Dissipation (positive semi-definite) [D, D].
        grad_h: Gradient of Hamiltonian [B, D].
        external_input: External port input [B, D] or None.
        dt: Integration timestep.
        name: Op name.

    Returns:
        Next state [B, D].
    """
    if not _ensure_lib_loaded():
        raise RuntimeError(
            "Unified quantum C++ ops not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    if external_input is None:
        external_input = tf.zeros_like(state)

    return _lib.high_noon_port_hamiltonian_step(
        state, j_matrix, r_matrix, grad_h, external_input, dt=dt, name=name
    )


# =============================================================================
# Phase 19.4: Thermodynamic Entropic Routing
# =============================================================================


def thermodynamic_route(
    logits: tf.Tensor, temperature: float = 1.0, name: str = "thermodynamic_route"
) -> tuple[tf.Tensor, tf.Tensor]:
    """Boltzmann-distributed routing with temperature.

    P(expert) ∝ exp(logits / T)

    Args:
        logits: Expert routing logits [B, N, E].
        temperature: Current temperature for annealing.
        name: Op name.

    Returns:
        Tuple of (routing_weights [B, N, E], entropy [B]).
    """
    if not _ensure_lib_loaded():
        raise RuntimeError(
            "Unified quantum C++ ops not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    return _lib.high_noon_thermodynamic_route(logits, temperature=temperature, name=name)


# =============================================================================
# Phase 22.2: Orthogonalized Keys
# =============================================================================


def orthogonalize_keys(
    keys: tf.Tensor, name: str = "orthogonalize_keys"
) -> tuple[tf.Tensor, tf.Tensor]:
    """Gram-Schmidt orthogonalization for attention keys.

    Prevents attention collapse by ensuring key distinctness.

    Args:
        keys: Key matrix [B, H, L, D].
        name: Op name.

    Returns:
        Tuple of (orthogonalized_keys [B, H, L, D], penalty [B, H]).
    """
    if not _ensure_lib_loaded():
        raise RuntimeError(
            "Unified quantum C++ ops not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    return _lib.high_noon_orthogonalize_keys(keys, name=name)


# =============================================================================
# Phase 24.1: QSVT Activation
# =============================================================================


def qsvt_activation(
    x: tf.Tensor, coefficients: tf.Tensor, degree: int = 8, name: str = "qsvt_activation"
) -> tf.Tensor:
    """QSVT-inspired activation via Chebyshev polynomial approximation.

    σ(x) = Σ_i c_i T_i(x) where T_i are Chebyshev polynomials.

    Args:
        x: Input tensor [B, L, D].
        coefficients: Chebyshev coefficients [degree + 1].
        degree: Polynomial degree.
        name: Op name.

    Returns:
        Activated output [B, L, D].
    """
    if not _ensure_lib_loaded():
        raise RuntimeError(
            "Unified quantum C++ ops not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    return _lib.high_noon_qsvt_activation(x, coefficients, degree=degree, name=name)


def get_gelu_chebyshev_coefficients(degree: int = 8) -> tf.Tensor:
    """Get Chebyshev coefficients approximating GELU activation.

    GELU(x) = x * Φ(x) where Φ is the CDF of standard normal.

    Args:
        degree: Polynomial degree.

    Returns:
        Chebyshev coefficients [degree + 1].
    """
    # Pre-computed coefficients for GELU approximation on [-1, 1]
    # These approximate GELU scaled to the Chebyshev domain
    if degree >= 8:
        return tf.constant(
            [
                0.5,  # c_0
                0.5,  # c_1
                0.0,  # c_2
                -0.044,  # c_3
                0.0,  # c_4
                0.0028,  # c_5
                0.0,  # c_6
                -0.0002,  # c_7
                0.0,  # c_8
            ],
            dtype=tf.float32,
        )
    else:
        # Lower degree approximation
        coeffs = [0.5, 0.5, 0.0, -0.044]
        while len(coeffs) <= degree:
            coeffs.append(0.0)
        return tf.constant(coeffs[: degree + 1], dtype=tf.float32)


# =============================================================================
# Phase 24.3: Quantum Reservoir
# =============================================================================


def quantum_reservoir(
    x: tf.Tensor,
    reservoir_state: tf.Tensor,
    reservoir_weights: tf.Tensor,
    readout_weights: tf.Tensor,
    reservoir_dim: int = 64,
    evolution_steps: int = 4,
    name: str = "quantum_reservoir",
) -> tuple[tf.Tensor, tf.Tensor]:
    """Quantum reservoir with fixed dynamics and trainable readout.

    Args:
        x: Input sequence [B, L, D_in].
        reservoir_state: Initial reservoir state [B, R].
        reservoir_weights: Fixed reservoir coupling [R, R + D_in].
        readout_weights: Trainable readout [D_out, R].
        reservoir_dim: Reservoir dimension R.
        evolution_steps: Quantum dynamics steps per input.
        name: Op name.

    Returns:
        Tuple of (output [B, L, D_out], new_reservoir_state [B, R]).
    """
    if not _ensure_lib_loaded():
        raise RuntimeError(
            "Unified quantum C++ ops not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    return _lib.high_noon_quantum_reservoir(
        x,
        reservoir_state,
        reservoir_weights,
        readout_weights,
        reservoir_dim=reservoir_dim,
        evolution_steps=evolution_steps,
        name=name,
    )


# =============================================================================
# Phase 7: Entanglement Preservation Loss
# =============================================================================


def entanglement_loss(
    bond_entropies: tf.Tensor,
    min_entropy: float = 0.1,
    weight: float = 0.01,
    name: str = "entanglement_loss",
) -> tf.Tensor:
    """Entanglement Preservation Loss (Phase 7).

    Computes loss based on bond entropies.

    Args:
        bond_entropies: Tensor of bond entropies [B*L].
        min_entropy: Minimum entropy threshold.
        weight: Loss weight.
        name: Op name.

    Returns:
        Loss value [1].
    """
    if not _ensure_lib_loaded():
        raise RuntimeError(
            "Unified quantum C++ ops not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    return _lib.high_noon_entanglement_loss(
        bond_entropies, min_entropy=min_entropy, weight=weight, name=name
    )


# =============================================================================
# Phase 6: Quantum Memory Replay
# =============================================================================


def quantum_memory_replay(
    inputs: tf.Tensor,
    weights: list[tf.Tensor],
    checkpoint_interval: int = 1,
    name: str = "quantum_memory_replay",
) -> tuple[tf.Tensor, tf.Tensor]:
    """Quantum Memory Replay (Phase 6).

    Recomputes gradients using logarithmic checkpointing and unitary reconstruction.

    Args:
        inputs: Input tensor [B, L, D].
        weights: List of weights for the unitary transformation.
        checkpoint_interval: Interval for checkpoints.
        name: Op name.

    Returns:
        Tuple of (output [B, L, D], gradients [same shape as weights? or separate tensor?]).
        Actually, the Op returns (output, replay_buffer).
    """
    if not _ensure_lib_loaded():
        raise RuntimeError(
            "Unified quantum C++ ops not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    # Note: Variable number of weight inputs not fully supported by this simple wrapper
    # unless we pack them. The Op definition expects input "weights" as a list.
    return _lib.high_noon_quantum_memory_replay(
        inputs, weights, checkpoint_interval=checkpoint_interval, name=name
    )


# =============================================================================
# Phase 25: Quantum Holographic Persistent Memory
# =============================================================================


def quantum_holographic_memory(
    query: tf.Tensor,
    keys: tf.Tensor,
    values: tf.Tensor,
    capacity: int = 1000,
    decay: float = 0.99,
    crystallize_threshold: float = 0.8,
    name: str = "quantum_holographic_memory",
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Quantum Holographic Memory (Phase 25).

    Associative retrieval using holographic binding and Modern Hopfield energy.

    Args:
        query: Query tensor [B, D].
        keys: Key memory [B, M, D] or [M, D].
        values: Value memory [B, M, D] or [M, D].
        capacity: Memory capacity.
        decay: Decay rate.
        crystallize_threshold: Crystallization threshold.
        name: Op name.

    Returns:
        Tuple of (retrieved [B, D], new_keys, new_values).
    """
    # Handle both batched and unbatched memory
    if len(keys.shape) == 2:
        keys = tf.expand_dims(keys, 0)
        values = tf.expand_dims(values, 0)
        batch_size = tf.shape(query)[0]
        keys = tf.tile(keys, [batch_size, 1, 1])
        values = tf.tile(values, [batch_size, 1, 1])

    if not _ensure_lib_loaded():
        raise RuntimeError(
            "Unified quantum C++ ops not available. Build with: "
            "cd highnoon/_native && ./build_secure.sh"
        )

    # Use C++ op - QuantumHolographicMemory takes (inputs, memory_bank, beta)
    # We concatenate keys as memory_bank for the Hopfield-style lookup
    # The C++ op performs Modern Hopfield attention with configurable beta
    memory_bank = keys[:, :, :]  # [B, M, D] - use keys as memory patterns

    # Call C++ op - TensorFlow auto-converts QuantumHolographicMemory to quantum_holographic_memory
    retrieved = _lib.quantum_holographic_memory(
        query,  # inputs [B, D]
        memory_bank,  # memory_bank [B, M, D] or [M, D]
        beta=1.0,  # Inverse temperature for Hopfield attention
        name=name,
    )

    # Return retrieved along with unchanged keys/values
    return retrieved, keys, values


# =============================================================================
# Unified Quantum Block Layer (Keras)
# =============================================================================


class UnifiedQuantumEnhancements(tf.keras.layers.Layer):
    """Unified quantum enhancements layer for the reasoning stack.

    Applies selected quantum-inspired enhancements to input:
    - Holographic memory binding
    - Port-Hamiltonian dynamics
    - Thermodynamic MoE routing
    - Orthogonalized attention keys
    - QSVT activations
    - Quantum reservoir dynamics

    All operations use float32 precision with no quantization.
    """

    def __init__(
        self,
        embedding_dim: int,
        use_holographic_memory: bool = True,
        use_port_hamiltonian: bool = True,
        use_thermodynamic_routing: bool = True,
        use_orthogonal_keys: bool = True,
        use_qsvt_activations: bool = True,
        use_quantum_reservoir: bool = False,
        holographic_dim: int = 512,
        qsvt_degree: int = 8,
        reservoir_dim: int = 64,
        **kwargs,
    ):
        """Initialize unified quantum enhancements.

        Args:
            embedding_dim: Model embedding dimension.
            use_holographic_memory: Enable holographic associative memory.
            use_port_hamiltonian: Enable Port-Hamiltonian dynamics.
            use_thermodynamic_routing: Enable thermodynamic MoE routing.
            use_orthogonal_keys: Enable orthogonalized attention keys.
            use_qsvt_activations: Enable QSVT-based activations.
            use_quantum_reservoir: Enable quantum reservoir memory.
            holographic_dim: Dimension for holographic binding (power of 2).
            qsvt_degree: Chebyshev polynomial degree for QSVT.
            reservoir_dim: Quantum reservoir hidden dimension.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.use_holographic_memory = use_holographic_memory
        self.use_port_hamiltonian = use_port_hamiltonian
        self.use_thermodynamic_routing = use_thermodynamic_routing
        self.use_orthogonal_keys = use_orthogonal_keys
        self.use_qsvt_activations = use_qsvt_activations
        self.use_quantum_reservoir = use_quantum_reservoir
        self.holographic_dim = holographic_dim
        self.qsvt_degree = qsvt_degree
        self.reservoir_dim = reservoir_dim

    def build(self, input_shape):
        """Build layer weights."""
        if self.use_port_hamiltonian:
            # Interconnection matrix (skew-symmetric)
            self.j_upper = self.add_weight(
                name="j_upper",
                shape=(self.embedding_dim, self.embedding_dim),
                initializer="glorot_uniform",
                trainable=True,
            )
            # Dissipation matrix (will be made positive semi-definite)
            self.r_diag = self.add_weight(
                name="r_diag",
                shape=(self.embedding_dim,),
                initializer=tf.keras.initializers.Constant(0.01),
                trainable=True,
            )
            # Hamiltonian gradient projection
            self.h_proj = self.add_weight(
                name="h_proj",
                shape=(self.embedding_dim, self.embedding_dim),
                initializer="glorot_uniform",
                trainable=True,
            )

        if self.use_qsvt_activations:
            self.qsvt_coefficients = self.add_weight(
                name="qsvt_coefficients",
                shape=(self.qsvt_degree + 1,),
                initializer=tf.keras.initializers.Constant(
                    get_gelu_chebyshev_coefficients(self.qsvt_degree).numpy()
                ),
                trainable=True,
            )

        if self.use_quantum_reservoir:
            self.reservoir_weights = self.add_weight(
                name="reservoir_weights",
                shape=(self.reservoir_dim, self.reservoir_dim + self.embedding_dim),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                trainable=False,  # Fixed reservoir
            )
            self.readout_weights = self.add_weight(
                name="readout_weights",
                shape=(self.embedding_dim, self.reservoir_dim),
                initializer="glorot_uniform",
                trainable=True,
            )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False, return_aux: bool = False):
        """Apply quantum enhancements.

        Args:
            inputs: Input tensor [B, L, D].
            training: Whether in training mode.
            return_aux: Whether to return auxiliary outputs.

        Returns:
            Enhanced output [B, L, D], optionally with auxiliary dict.
        """
        x = inputs
        aux = {}

        # Port-Hamiltonian dynamics
        if self.use_port_hamiltonian:
            # Make J skew-symmetric
            j_matrix = self.j_upper - tf.transpose(self.j_upper)
            # Make R positive semi-definite (diagonal)
            r_matrix = tf.linalg.diag(tf.nn.softplus(self.r_diag))
            # Compute gradient of Hamiltonian
            grad_h = tf.einsum("ij,blj->bli", self.h_proj, x)
            # Average over sequence for dynamics
            x_mean = tf.reduce_mean(x, axis=1)
            grad_h_mean = tf.reduce_mean(grad_h, axis=1)
            x_evolved = port_hamiltonian_step(x_mean, j_matrix, r_matrix, grad_h_mean, dt=0.01)
            # Residual connection
            x = x + tf.expand_dims(x_evolved - x_mean, axis=1)

        # QSVT activation
        if self.use_qsvt_activations:
            # Normalize to [-1, 1] range for Chebyshev
            x_norm = tf.nn.l2_normalize(x, axis=-1)
            x_activated = qsvt_activation(x_norm, self.qsvt_coefficients, degree=self.qsvt_degree)
            x = x + 0.1 * x_activated  # Residual with small scaling

        if return_aux:
            return x, aux
        return x

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "use_holographic_memory": self.use_holographic_memory,
                "use_port_hamiltonian": self.use_port_hamiltonian,
                "use_thermodynamic_routing": self.use_thermodynamic_routing,
                "use_orthogonal_keys": self.use_orthogonal_keys,
                "use_qsvt_activations": self.use_qsvt_activations,
                "use_quantum_reservoir": self.use_quantum_reservoir,
                "holographic_dim": self.holographic_dim,
                "qsvt_degree": self.qsvt_degree,
                "reservoir_dim": self.reservoir_dim,
            }
        )
        return config
