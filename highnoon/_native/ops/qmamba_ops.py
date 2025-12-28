# highnoon/_native/ops/qmamba_ops.py
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

"""Phase 102: QMamba - Quantum Selective State Space Model Python Wrapper.

This module provides Python wrappers for the C++ QMamba TensorFlow operations.
QMamba extends Mamba SSM with quantum-enhanced state transitions via:
- Quantum State Superposition: K parallel state paths exist simultaneously
- Entanglement-Aware Updates: VQC encodes inter-position correlations
- Amplitude-Weighted Selection: Born rule for selective scanning

Research Basis: QMamba (Koelle et al., ICAART 2025), Q-SSM (arXiv 2025)

Complexity: O(n · K · state_dim) where K is num_superposition_states
"""

import logging
from typing import Optional

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
    logger.warning("QMamba C++ ops not available: %s", e)

# =============================================================================
# OP WRAPPERS
# =============================================================================


def qmamba_selective_scan(
    x: tf.Tensor,
    a_log: tf.Tensor,
    b: tf.Tensor,
    c: tf.Tensor,
    dt: tf.Tensor,
    rotation_angles: tf.Tensor,
    num_superposition_states: int = 4,
    entanglement_depth: int = 2,
    entanglement_strength: float = 0.5,
    use_amplitude_selection: bool = True,
    gumbel_temperature: float = 1.0,
    seed: int = 42,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Full QMamba selective scan with quantum superposition.

    Extends standard Mamba scan with K parallel state paths:
      1. Initialize K superposition states
      2. Apply entanglement layers for correlations
      3. Run parallel SSM scans on each path
      4. Collapse to single output via Born rule/Gumbel

    Args:
        x: Input sequence [batch, seq_len, d_inner].
        a_log: Log of decay rates [d_inner, state_dim].
        b: B projections [batch, seq_len, state_dim].
        c: C projections [batch, seq_len, state_dim].
        dt: Delta timesteps [batch, seq_len, d_inner].
        rotation_angles: VQC angles [entanglement_depth, K].
        num_superposition_states: K parallel quantum state paths (default: 4).
        entanglement_depth: VQC entanglement layers (default: 2).
        entanglement_strength: α ∈ [0,1] for quantum mixing (default: 0.5).
        use_amplitude_selection: Born rule vs softmax (default: True).
        gumbel_temperature: Temperature for Gumbel-softmax collapse (default: 1.0).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Tuple of (output, superposition_states):
        - output: Collapsed output [batch, seq_len, d_inner].
        - superposition_states: Final superposed states [batch, K, d_inner, state_dim].
    """
    if _ops is None:
        raise RuntimeError(
            "QMamba C++ operators not available. " "Rebuild with: ./build_secure.sh --debug --lite"
        )

    return _ops.qmamba_selective_scan(
        x=x,
        a_log=a_log,
        b=b,
        c=c,
        dt=dt,
        rotation_angles=rotation_angles,
        num_superposition_states=num_superposition_states,
        entanglement_depth=entanglement_depth,
        entanglement_strength=entanglement_strength,
        use_amplitude_selection=use_amplitude_selection,
        gumbel_temperature=gumbel_temperature,
        seed=seed,
    )


def qmamba_entangle(
    states: tf.Tensor,
    rotation_angles: tf.Tensor,
    entanglement_depth: int = 2,
    entanglement_strength: float = 0.5,
) -> tf.Tensor:
    """Apply VQC-inspired entanglement layers to superposition states.

    Uses parameterized rotations to create correlations between paths:
      RY(θ) rotation followed by CNOT-like correlation

    Args:
        states: Input states [batch, K, state_dim].
        rotation_angles: VQC angles [entanglement_depth, K].
        entanglement_depth: Number of entanglement layers (default: 2).
        entanglement_strength: Mixing strength α ∈ [0,1] (default: 0.5).

    Returns:
        Entangled states [batch, K, state_dim].
    """
    if _ops is None:
        raise RuntimeError(
            "QMamba C++ operators not available. " "Rebuild with: ./build_secure.sh --debug --lite"
        )

    return _ops.qmamba_entangle(
        states=states,
        rotation_angles=rotation_angles,
        entanglement_depth=entanglement_depth,
        entanglement_strength=entanglement_strength,
    )


def qmamba_collapse(
    h_super: tf.Tensor,
    path_logits: tf.Tensor,
    use_born_rule: bool = True,
    gumbel_temperature: float = 1.0,
    seed: int = 42,
) -> tf.Tensor:
    """Collapse superposition states via Born rule or Gumbel-Softmax.

    Born rule collapse:
      prob_k = |ψ_k|² / Σ|ψ_k|²
      h_out = Σ prob_k * h_k

    Gumbel-Softmax collapse (differentiable):
      logits_k = log(|ψ_k|²) + Gumbel_noise
      weights_k = softmax(logits_k / τ)

    Args:
        h_super: Superposition states [batch, K, state_dim].
        path_logits: Path selection logits [batch, K].
        use_born_rule: Use Born rule (True) or Gumbel-Softmax (False).
        gumbel_temperature: Temperature τ for Gumbel-Softmax (default: 1.0).
        seed: Random seed for Gumbel noise (default: 42).

    Returns:
        Collapsed state [batch, state_dim].
    """
    if _ops is None:
        raise RuntimeError(
            "QMamba C++ operators not available. " "Rebuild with: ./build_secure.sh --debug --lite"
        )

    return _ops.qmamba_collapse(
        h_super=h_super,
        path_logits=path_logits,
        use_born_rule=use_born_rule,
        gumbel_temperature=gumbel_temperature,
        seed=seed,
    )


# =============================================================================
# AVAILABILITY CHECK
# =============================================================================


def is_qmamba_available() -> bool:
    """Check if QMamba C++ operators are available.

    Returns:
        True if ops are loaded and available, False otherwise.
    """
    if _ops is None:
        return False

    try:
        # Check that the op is registered
        return hasattr(_ops, "qmamba_selective_scan")
    except Exception:
        return False


__all__ = [
    "qmamba_selective_scan",
    "qmamba_entangle",
    "qmamba_collapse",
    "is_qmamba_available",
]
