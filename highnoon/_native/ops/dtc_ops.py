# highnoon/_native/ops/dtc_ops.py
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

"""Phase 38/128: Discrete Time Crystal (DTC) Operations Python Wrapper.

This module provides Python wrappers for the C++ DTC TensorFlow operations
that implement Floquet-engineered time crystal dynamics for state protection.

DTC dynamics provide:
- Discrete time-translation symmetry breaking
- Many-Body Localization (MBL) for error mitigation
- Natural stability against perturbations through periodic driving

Research Basis: "Phase Transitions in DTCs" (Nature Physics 2024)

Complexity: O(n · d) per Floquet period
Memory: O(d²) for Floquet Hamiltonian
"""

import logging

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
    logger.warning("DTC C++ ops not available: %s", e)

# =============================================================================
# OP WRAPPERS
# =============================================================================


def dtc_stabilized_evolution(
    hidden_state: tf.Tensor,
    h_evolution: tf.Tensor,
    floquet_period: int = 4,
    coupling_j: float = 1.0,
    disorder_w: float = 0.5,
    pi_pulse_error: float = 0.01,
    use_prethermal: bool = True,
    num_cycles: int = 1,
    seed: int = 42,
) -> tf.Tensor:
    """Apply DTC-stabilized evolution to hidden states.

    Implements the Floquet operator:
      U_F = exp(-i H T/2) · R_x(π + ε) · exp(-i H T/2)

    This creates period-doubling characteristic of discrete time crystals,
    providing natural error suppression through many-body localization.

    Args:
        hidden_state: Input hidden state [batch, seq, state_dim].
        h_evolution: Effective Hamiltonian [state_dim, state_dim].
        floquet_period: Floquet driving period T (default: 4).
        coupling_j: Heisenberg coupling strength J (default: 1.0).
        disorder_w: MBL disorder strength W (default: 0.5).
        pi_pulse_error: π-pulse imperfection ε (default: 0.01).
        use_prethermal: Enable prethermal DTC regime (default: True).
        num_cycles: Number of Floquet cycles per step (default: 1).
        seed: Random seed for disorder (default: 42).

    Returns:
        Stabilized hidden state [batch, seq, state_dim].
    """
    if _ops is None:
        raise RuntimeError(
            "DTC C++ operators not available. " "Rebuild with: ./build_secure.sh --debug --lite"
        )

    return _ops.dtc_stabilized_evolution(
        hidden_state=hidden_state,
        h_evolution=h_evolution,
        floquet_period=floquet_period,
        coupling_j=coupling_j,
        disorder_w=disorder_w,
        pi_pulse_error=pi_pulse_error,
        use_prethermal=use_prethermal,
        num_cycles=num_cycles,
        seed=seed,
    )


def dtc_order_parameter(
    state: tf.Tensor,
    floquet_period: int = 4,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute DTC order parameter from hidden state magnetization.

    The DTC phase is characterized by period-doubled oscillations:
      M(t) = ⟨Σ_i σ_i^z(t)⟩

    Strong peak at ω/2 (half the drive frequency) indicates DTC order.

    Args:
        state: Hidden state [batch, seq, state_dim].
        floquet_period: Floquet driving period (default: 4).

    Returns:
        Tuple of (magnetization, dtc_order):
        - magnetization: Magnetization time series [batch, seq].
        - dtc_order: DTC phase order strength [batch] (0 to 1, higher = more DTC).
    """
    if _ops is None:
        raise RuntimeError(
            "DTC C++ operators not available. " "Rebuild with: ./build_secure.sh --debug --lite"
        )

    return _ops.dtc_order_parameter(
        state=state,
        floquet_period=floquet_period,
    )


def apply_pi_pulse(
    state: tf.Tensor,
    error: float = 0.01,
) -> tf.Tensor:
    """Apply π-pulse rotation for DTC dynamics.

    The π-pulse creates the period-doubling characteristic of DTCs:
      R_x(π + ε) ≈ cos(π + ε) · I + sin(π + ε) · X

    Args:
        state: Input state [batch, seq, state_dim].
        error: π-pulse imperfection ε (default: 0.01).

    Returns:
        Rotated state [batch, seq, state_dim].
    """
    if _ops is None:
        raise RuntimeError(
            "DTC C++ operators not available. " "Rebuild with: ./build_secure.sh --debug --lite"
        )

    return _ops.apply_pi_pulse(
        state=state,
        error=error,
    )


# =============================================================================
# AVAILABILITY CHECK
# =============================================================================


def is_dtc_available() -> bool:
    """Check if DTC C++ operators are available.

    Returns:
        True if ops are loaded and available, False otherwise.
    """
    if _ops is None:
        return False

    try:
        # Check that the op is registered
        return hasattr(_ops, "dtc_stabilized_evolution")
    except Exception:
        return False


def get_dtc_load_error() -> str | None:
    """Get the error message if DTC ops failed to load.

    Returns:
        Error message string, or None if ops loaded successfully.
    """
    return _ops_load_error


__all__ = [
    "dtc_stabilized_evolution",
    "dtc_order_parameter",
    "apply_pi_pulse",
    "is_dtc_available",
    "get_dtc_load_error",
]
