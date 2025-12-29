# highnoon/_native/ops/quantum_teleport_bus_ops.py
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

"""Python wrappers for Quantum Teleport Bus C++ operations (Phase 44).

Provides quantum state teleportation via Bell measurements for
cross-block communication in the HSMN architecture.

NO PYTHON FALLBACKS: RuntimeError raised if native ops unavailable.

Ops:
    - quantum_teleport_state: Teleport state with error correction
    - bell_measurement: Perform Bell measurement on state pairs
"""

import logging

import tensorflow as tf

from highnoon import config
from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

_module = None
_available = False


def _load_ops():
    """Load ops from consolidated binary."""
    global _module, _available
    if _module is not None:
        return _available

    try:
        lib_path = resolve_op_library(__file__, "_highnoon_core.so")
        if lib_path is None:
            raise RuntimeError("Could not find _highnoon_core.so")
        _module = tf.load_op_library(lib_path)
        _available = True
        logger.info(f"Quantum Teleport Bus ops loaded from {lib_path}")
    except Exception as e:
        _available = False
        logger.warning(f"Failed to load Quantum Teleport Bus ops: {e}")
        raise RuntimeError(
            "Quantum Teleport Bus native ops not available. " "Run ./build_secure.sh to compile."
        ) from e
    return _available


def ops_available() -> bool:
    """Check if native ops are available."""
    try:
        _load_ops()
        return _available
    except RuntimeError:
        return False


# =============================================================================
# Phase 44: Quantum Teleport Bus
# =============================================================================


def quantum_teleport_state(
    input_state: tf.Tensor,
    entanglement_dim: int | None = None,
    fidelity_threshold: float | None = None,
    use_error_correction: bool | None = None,
    seed: int = 42,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Quantum state teleportation for cross-block communication.

    Implements quantum teleportation protocol:
    1. Create Bell pair shared between source and destination
    2. Bell measurement at source produces 2 classical bits
    3. Apply Pauli corrections at destination

    The teleported state is essentially identical to the input with
    high fidelity when error correction is enabled.

    Args:
        input_state: State to teleport [batch, dim].
        entanglement_dim: Dimension of entanglement pairs (default from config).
        fidelity_threshold: Minimum fidelity for success (default from config).
        use_error_correction: Enable error correction (default from config).
        seed: Random seed for Bell measurements.

    Returns:
        Tuple of (teleported_state [batch, dim], fidelity [batch]).

    Raises:
        RuntimeError: If native op not available.

    Example:
        >>> state = tf.random.normal([4, 64])
        >>> teleported, fidelity = quantum_teleport_state(state)
        >>> print(f"Mean fidelity: {tf.reduce_mean(fidelity):.4f}")
    """
    if not config.USE_QUANTUM_TELEPORT_BUS:
        # Pass through if disabled, with perfect fidelity
        batch_size = tf.shape(input_state)[0]
        return input_state, tf.ones([batch_size], dtype=tf.float32)

    _load_ops()
    entanglement_dim = entanglement_dim or config.TELEPORT_ENTANGLEMENT_DIM
    fidelity_threshold = fidelity_threshold or config.TELEPORT_FIDELITY_THRESHOLD
    use_error_correction = (
        use_error_correction if use_error_correction is not None else config.TELEPORT_USE_CORRECTION
    )

    return _module.quantum_teleport_state(
        input_state,
        entanglement_dim=entanglement_dim,
        fidelity_threshold=fidelity_threshold,
        use_error_correction=use_error_correction,
        seed=seed,
    )


def bell_measurement(
    state_a: tf.Tensor,
    state_b: tf.Tensor,
    seed: int = 42,
) -> tf.Tensor:
    """Perform Bell measurement on two states.

    The Bell measurement projects the two-qubit state onto one of
    four Bell basis states and returns the 2-bit classical outcome.

    Args:
        state_a: First state vector [batch, dim].
        state_b: Second state vector [batch, dim].
        seed: Random seed for measurement.

    Returns:
        Classical bits representing measurement outcome [batch, 2].

        Outcomes:
        - [0, 0]: |Φ+⟩ state detected
        - [0, 1]: |Φ-⟩ state detected
        - [1, 0]: |Ψ+⟩ state detected
        - [1, 1]: |Ψ-⟩ state detected

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    return _module.bell_measurement(state_a, state_b, seed=seed)


__all__ = [
    "quantum_teleport_state",
    "bell_measurement",
    "ops_available",
]
