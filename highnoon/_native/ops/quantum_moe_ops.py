# highnoon/_native/ops/quantum_moe_ops.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Python wrapper for Quantum-Inspired MoE Operations.
# Uses float64 precision for quantum layers.
#
# NO PYTHON FALLBACK: This module requires the compiled .so to function.

"""Quantum-Inspired MoE Operations.

Implements quantum computing concepts for enhanced routing and expert dynamics:

A. Quantum Interference Routing (QIR)
   - Complex-valued logits with phase relationships
   - Born rule probability: P(expert) = |ψ|²

B. Hamiltonian Expert Dynamics
   - Unitary expert transformations preserving norm
   - Matrix exponential via Padé approximation

C. Entangled MPO Router
   - Tensor network factorized routing weights
   - O(D log E) complexity vs O(DE) dense

D. Born Rule Sampling
   - Probabilistic top-K selection based on |ψ|²
   - Temperature-controlled phase sampling
"""

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

# --- Load Custom C++ Operators via consolidated binary ---
_quantum_moe_module = None
quantum_interference_routing_op = None
hamiltonian_expert_dynamics_op = None
entangled_mpo_router_op = None
born_rule_sampling_op = None

try:
    from highnoon._native import get_op

    _quantum_moe_module = get_op("quantum_moe_ops")

    if _quantum_moe_module is not None:
        # QIR
        quantum_interference_routing_op = getattr(
            _quantum_moe_module, "quantum_interference_routing", None
        )
        if quantum_interference_routing_op is None:
            quantum_interference_routing_op = getattr(
                _quantum_moe_module, "QuantumInterferenceRouting", None
            )

        # Hamiltonian
        hamiltonian_expert_dynamics_op = getattr(
            _quantum_moe_module, "hamiltonian_expert_dynamics", None
        )
        if hamiltonian_expert_dynamics_op is None:
            hamiltonian_expert_dynamics_op = getattr(
                _quantum_moe_module, "HamiltonianExpertDynamics", None
            )

        # MPO Router
        entangled_mpo_router_op = getattr(_quantum_moe_module, "entangled_mpo_router", None)
        if entangled_mpo_router_op is None:
            entangled_mpo_router_op = getattr(_quantum_moe_module, "EntangledMPORouter", None)

        # Born Rule
        born_rule_sampling_op = getattr(_quantum_moe_module, "born_rule_sampling", None)
        if born_rule_sampling_op is None:
            born_rule_sampling_op = getattr(_quantum_moe_module, "BornRuleSampling", None)

        loaded_ops = sum(
            [
                quantum_interference_routing_op is not None,
                hamiltonian_expert_dynamics_op is not None,
                entangled_mpo_router_op is not None,
                born_rule_sampling_op is not None,
            ]
        )
        logger.debug(f"Loaded {loaded_ops}/4 quantum MoE operators from consolidated binary.")

except Exception as e:
    logger.warning(f"Could not load quantum MoE ops: {e}")


def quantum_interference_routing(
    tokens: tf.Tensor,
    w_real: tf.Tensor,
    w_imag: tf.Tensor,
    phase_bias: tf.Tensor,
    temperature: float = 1.0,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Quantum Interference Routing with complex-valued logits.

    Computes router probabilities using Born rule: P = |ψ|²
    where ψ = tokens @ (W_real + i*W_imag) with phase bias.

    Args:
        tokens: [batch, d_model] float64 input tokens.
        w_real: [d_model, num_experts] float64 real weight matrix.
        w_imag: [d_model, num_experts] float64 imaginary weight matrix.
        phase_bias: [num_experts] float64 learnable phase bias (radians).
        temperature: Temperature for probability scaling.

    Returns:
        Tuple of:
        - router_probs: [batch, num_experts] Born rule probabilities.
        - phase_angles: [batch, num_experts] Phase information.
        - amplitudes: [batch, num_experts] |ψ| values for gradient.

    Raises:
        NotImplementedError: If C++ operator not available.
    """
    if quantum_interference_routing_op is None:
        raise NotImplementedError(
            "The C++ QuantumInterferenceRouting operator could not be loaded. NO PYTHON FALLBACK."
        )

    # Ensure float64 for quantum precision
    tokens = tf.cast(tokens, tf.float64)
    w_real = tf.cast(w_real, tf.float64)
    w_imag = tf.cast(w_imag, tf.float64)
    phase_bias = tf.cast(phase_bias, tf.float64)

    return quantum_interference_routing_op(
        tokens=tokens,
        w_real=w_real,
        w_imag=w_imag,
        phase_bias=phase_bias,
        temperature=temperature,
    )


def hamiltonian_expert_dynamics(
    expert_output: tf.Tensor,
    hamiltonian: tf.Tensor,
    dt: float = 0.1,
    pade_order: int = 4,
) -> tf.Tensor:
    """
    Apply Hamiltonian dynamics to expert output.

    Performs unitary transformation U = exp(-i*H*dt) that preserves
    the norm of the expert output, improving training stability.

    Args:
        expert_output: [batch, d_model] float64 expert outputs.
        hamiltonian: [d_model, d_model] float64 Hermitian matrix.
        dt: Evolution time step (smaller = more accurate).
        pade_order: Padé approximation order (1-4).

    Returns:
        evolved_output: [batch, d_model] unitarily evolved outputs.

    Raises:
        NotImplementedError: If C++ operator not available.
    """
    if hamiltonian_expert_dynamics_op is None:
        raise NotImplementedError(
            "The C++ HamiltonianExpertDynamics operator could not be loaded. NO PYTHON FALLBACK."
        )

    expert_output = tf.cast(expert_output, tf.float64)
    hamiltonian = tf.cast(hamiltonian, tf.float64)

    return hamiltonian_expert_dynamics_op(
        expert_output=expert_output,
        hamiltonian=hamiltonian,
        dt=dt,
        pade_order=pade_order,
    )


def entangled_mpo_router(
    tokens: tf.Tensor,
    mpo_cores: tf.Tensor,
    num_cores: int,
    core_dims: list[int],
) -> tf.Tensor:
    """
    Route using Matrix Product Operator factorization.

    Uses tensor network factorization for parameter-efficient routing
    with O(D log E) complexity instead of O(DE) for dense routing.

    Args:
        tokens: [batch, d_model] float64 input tokens.
        mpo_cores: Flattened MPO cores tensor.
        num_cores: Number of MPO cores.
        core_dims: List of core dimensions [r0, d0, D0, r1, ...].

    Returns:
        router_logits: [batch, num_experts] routing logits.

    Raises:
        NotImplementedError: If C++ operator not available.
    """
    if entangled_mpo_router_op is None:
        raise NotImplementedError(
            "The C++ EntangledMPORouter operator could not be loaded. NO PYTHON FALLBACK."
        )

    tokens = tf.cast(tokens, tf.float64)
    mpo_cores = tf.cast(mpo_cores, tf.float64)

    return entangled_mpo_router_op(
        tokens=tokens,
        mpo_cores=mpo_cores,
        num_cores=num_cores,
        core_dims=core_dims,
    )


def born_rule_sampling(
    amplitudes: tf.Tensor,
    phases: tf.Tensor,
    k: int,
    temperature: float = 1.0,
    seed: int = 0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Sample top-K experts using Born rule probabilities.

    Performs probabilistic selection based on |ψ|² distribution,
    which provides quantum-inspired exploration during training.

    Args:
        amplitudes: [batch, num_experts] float64 |ψ| values.
        phases: [batch, num_experts] float64 phase angles.
        k: Number of experts to sample.
        temperature: Probability temperature (higher = more uniform).
        seed: Random seed (0 = non-deterministic).

    Returns:
        Tuple of:
        - selected_indices: [batch, k] int32 selected expert indices.
        - selected_probs: [batch, k] float64 selection probabilities.

    Raises:
        NotImplementedError: If C++ operator not available.
    """
    if born_rule_sampling_op is None:
        raise NotImplementedError(
            "The C++ BornRuleSampling operator could not be loaded. NO PYTHON FALLBACK."
        )

    amplitudes = tf.cast(amplitudes, tf.float64)
    phases = tf.cast(phases, tf.float64)

    return born_rule_sampling_op(
        amplitudes=amplitudes,
        phases=phases,
        k=k,
        temperature=temperature,
        seed=seed,
    )
