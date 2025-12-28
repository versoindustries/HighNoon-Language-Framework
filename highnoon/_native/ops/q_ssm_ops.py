# highnoon/_native/ops/q_ssm_ops.py
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

"""Phase 120: Q-SSM Quantum-Optimized Selective State Space Gating Python Wrapper.

This module provides Python wrappers for the C++ Q-SSM TensorFlow operations.
Q-SSM integrates variational quantum circuits as adaptive gating mechanisms for
SSM memory updates, stabilizing training and enhancing long-term dependencies.

Key Features:
    - VQC Gating: RY-RX ansatz regulates memory updates adaptively
    - Quantum Stabilization: Prevents optimization instabilities
    - Born Rule Interpretation: Gate values via quantum measurement

Research Basis:
    "Q-SSM: Quantum-Optimized Selective State Space Model" (arXiv 2025)

Complexity: O(N × D × vqc_layers)
"""

import logging
from typing import Tuple

import tensorflow as tf

from highnoon._native.ops.lib_loader import get_highnoon_core_path

logger = logging.getLogger(__name__)

# Load the native library
_ops = None
_ops_load_error = None

try:
    _lib_path = get_highnoon_core_path()
    _ops = tf.load_op_library(_lib_path)
except Exception as e:
    _ops_load_error = str(e)
    logger.debug(f"Failed to load Q-SSM ops: {e}")


# =============================================================================
# OP WRAPPERS
# =============================================================================


def qssm_forward(
    input_seq: tf.Tensor,
    state: tf.Tensor,
    vqc_params: tf.Tensor,
    state_dim: int = 16,
    input_dim: int = 64,
    vqc_qubits: int = 4,
    vqc_layers: int = 2,
    use_born_rule: bool = True,
    measurement_temp: float = 1.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Q-SSM forward pass with VQC-based selective gating.

    Implements the core Q-SSM equation:
        S_t = σ_VQC(x_t) ⊙ S_{t-1} + (1 - σ_VQC(x_t)) ⊙ Update(x_t)

    The VQC-based gate adaptively controls memory retention vs. update,
    stabilizing training and improving long-term dependency modeling.

    Args:
        input_seq: Input sequence [batch, seq_len, input_dim].
        state: Initial SSM state [batch, state_dim].
        vqc_params: VQC rotation parameters [vqc_layers, vqc_qubits, 2].
        state_dim: SSM state dimension (default: 16).
        input_dim: Input feature dimension (default: 64).
        vqc_qubits: Number of virtual qubits (default: 4).
        vqc_layers: Number of VQC rotation layers (default: 2).
        use_born_rule: Use Born rule gate (True) or sigmoid (False).
        measurement_temp: Temperature for soft measurement (default: 1.0).

    Returns:
        Tuple of (output, final_state):
        - output: Output sequence [batch, seq_len, input_dim].
        - final_state: Updated state [batch, state_dim].

    Raises:
        RuntimeError: If Q-SSM C++ operators are not available.

    Example:
        >>> batch, seq, dim = 2, 32, 64
        >>> x = tf.random.normal([batch, seq, dim])
        >>> state = tf.zeros([batch, 16])
        >>> vqc_params = tf.random.normal([2, 4, 2])
        >>> output, final_state = qssm_forward(x, state, vqc_params)
    """
    if _ops is None:
        raise RuntimeError(
            "Q-SSM C++ operators not available. " "Rebuild with: ./build_secure.sh --debug --lite"
        )

    return _ops.qssm_forward(
        input=input_seq,
        state=state,
        vqc_params=vqc_params,
        state_dim=state_dim,
        input_dim=input_dim,
        vqc_qubits=vqc_qubits,
        vqc_layers=vqc_layers,
        use_born_rule=use_born_rule,
        measurement_temp=measurement_temp,
    )


def vqc_gate_expectation(
    encoded_input: tf.Tensor,
    rotation_params: tf.Tensor,
    vqc_qubits: int = 4,
    vqc_layers: int = 2,
    use_born_rule: bool = True,
    measurement_temp: float = 1.0,
) -> tf.Tensor:
    """Compute VQC expectation values for gating.

    Simulates RY-RX VQC with CNOT entanglement (ring topology) and
    measures ⟨Z⟩ expectation on the first qubit.

    Args:
        encoded_input: Encoded input features [batch, vqc_qubits].
        rotation_params: VQC rotation parameters [vqc_layers, vqc_qubits, 2].
        vqc_qubits: Number of virtual qubits (default: 4).
        vqc_layers: Number of VQC rotation layers (default: 2).
        use_born_rule: Use Born rule mapping (True) or sigmoid (False).
        measurement_temp: Temperature for soft measurement (default: 1.0).

    Returns:
        Gate values in [0, 1] range [batch].

    Raises:
        RuntimeError: If Q-SSM C++ operators are not available.
    """
    if _ops is None:
        raise RuntimeError(
            "Q-SSM C++ operators not available. " "Rebuild with: ./build_secure.sh --debug --lite"
        )

    return _ops.vqc_gate_expectation(
        encoded_input=encoded_input,
        rotation_params=rotation_params,
        vqc_qubits=vqc_qubits,
        vqc_layers=vqc_layers,
        use_born_rule=use_born_rule,
        measurement_temp=measurement_temp,
    )


def qssm_compute_gates(
    input_seq: tf.Tensor,
    vqc_params: tf.Tensor,
    input_dim: int = 64,
    vqc_qubits: int = 4,
    vqc_layers: int = 2,
    use_born_rule: bool = True,
    measurement_temp: float = 1.0,
) -> tf.Tensor:
    """Compute Q-SSM gate values for full sequence.

    Utility function for monitoring and visualization of gating behavior.

    Args:
        input_seq: Input sequence [batch, seq_len, input_dim].
        vqc_params: VQC rotation parameters [vqc_layers, vqc_qubits, 2].
        input_dim: Input feature dimension (default: 64).
        vqc_qubits: Number of virtual qubits (default: 4).
        vqc_layers: Number of VQC rotation layers (default: 2).
        use_born_rule: Use Born rule mapping (True) or sigmoid (False).
        measurement_temp: Temperature for soft measurement (default: 1.0).

    Returns:
        Gate values for each position [batch, seq_len].

    Raises:
        RuntimeError: If Q-SSM C++ operators are not available.
    """
    if _ops is None:
        raise RuntimeError(
            "Q-SSM C++ operators not available. " "Rebuild with: ./build_secure.sh --debug --lite"
        )

    return _ops.qssm_compute_gates(
        input=input_seq,
        vqc_params=vqc_params,
        input_dim=input_dim,
        vqc_qubits=vqc_qubits,
        vqc_layers=vqc_layers,
        use_born_rule=use_born_rule,
        measurement_temp=measurement_temp,
    )


# =============================================================================
# AVAILABILITY CHECK
# =============================================================================


def is_qssm_available() -> bool:
    """Check if Q-SSM C++ operators are available.

    Returns:
        True if ops are loaded and available, False otherwise.
    """
    if _ops is None:
        return False

    try:
        return hasattr(_ops, "qssm_forward")
    except Exception:
        return False


__all__ = [
    "qssm_forward",
    "vqc_gate_expectation",
    "qssm_compute_gates",
    "is_qssm_available",
]
