# highnoon/_native/ops/qasa_ops.py
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

"""Phase 119: QASA - Quantum Adaptive Self-Attention Python Wrapper.

This module provides Python wrappers for the C++ QASA TensorFlow operations.
QASA enhances classical attention with VQC-powered scoring:

    A(q,k) = ⟨0|U†(q)V(k)|0⟩

where U and V are parameterized quantum circuits encoding query and key vectors.

Benefits:
- Non-classical correlations captured via entanglement
- Richer attention patterns beyond dot-product similarity

Complexity: O(N² × P) where P = VQC circuit parameters

Research Reference: arXiv:2501.xxxxx - Quantum Adaptive Self-Attention for Transformers
"""

import logging

import tensorflow as tf

from highnoon._native.ops.lib_loader import get_highnoon_core_path

logger = logging.getLogger(__name__)

# Load the native library
_ops = None
_ops_load_error = None
_gradient_registered = False

try:
    _lib_path = get_highnoon_core_path()
    _ops = tf.load_op_library(_lib_path)

    # Register gradient for QASAAttention op
    @tf.RegisterGradient("QASAAttention")
    def _qasa_attention_grad(op, grad_output):
        """Gradient for QASAAttention using C++ finite difference kernel."""
        queries = op.inputs[0]
        keys = op.inputs[1]
        values = op.inputs[2]
        vqc_params = op.inputs[3]

        # Get attributes from the forward op
        num_qubits = op.get_attr("num_qubits")
        vqc_layers = op.get_attr("vqc_layers")
        entanglement_strength = op.get_attr("entanglement_strength")

        # Compute gradient w.r.t. vqc_params using C++ grad op
        grad_vqc_params = _ops.qasa_attention_grad(
            queries=queries,
            keys=keys,
            values=values,
            vqc_params=vqc_params,
            grad_output=grad_output,
            num_qubits=num_qubits,
            vqc_layers=vqc_layers,
            entanglement_strength=entanglement_strength,
        )

        # Return gradients: [grad_queries, grad_keys, grad_values, grad_vqc_params]
        # For now, we only compute gradients w.r.t. vqc_params
        # Q, K, V gradients can be added later if needed
        return [None, None, None, grad_vqc_params]

    _gradient_registered = True

except Exception as e:
    _ops_load_error = str(e)
    logger.warning("QASA C++ ops not available: %s", e)


# =============================================================================
# OP WRAPPERS
# =============================================================================


def run_qasa_attention(
    queries: tf.Tensor,
    keys: tf.Tensor,
    values: tf.Tensor,
    vqc_params: tf.Tensor,
    num_qubits: int = 4,
    vqc_layers: int = 2,
    entanglement_strength: float = 0.5,
) -> tf.Tensor:
    """Compute QASA attention using VQC-based scoring.

    Replaces standard dot-product attention with quantum circuit overlap:
        A(q,k) = ⟨0|U†(q)V(k)|0⟩

    where U encodes query vectors and V encodes key vectors into parameterized
    quantum circuits. The overlap computes attention scores capturing
    non-classical correlations.

    Args:
        queries: Query tensor [batch, heads, seq_len, head_dim].
        keys: Key tensor [batch, heads, seq_len, head_dim].
        values: Value tensor [batch, heads, seq_len, head_dim].
        vqc_params: VQC parameters [2 * vqc_layers * num_qubits].
            First half for query encoding, second half for key encoding.
        num_qubits: Number of qubits in VQC (default: 4).
        vqc_layers: Number of VQC rotation layers (default: 2).
        entanglement_strength: α ∈ [0,1] for entanglement contribution (default: 0.5).

    Returns:
        Output tensor [batch, heads, seq_len, head_dim] after QASA attention.

    Raises:
        RuntimeError: If QASA C++ operators are not available.

    Example:
        >>> queries = tf.random.normal([2, 8, 16, 64])
        >>> keys = tf.random.normal([2, 8, 16, 64])
        >>> values = tf.random.normal([2, 8, 16, 64])
        >>> vqc_params = tf.Variable(tf.random.uniform([32], -0.1, 0.1))
        >>> output = run_qasa_attention(queries, keys, values, vqc_params)
        >>> print(output.shape)  # (2, 8, 16, 64)
    """
    if _ops is None:
        raise RuntimeError(
            "QASA C++ operators not available. "
            f"Error: {_ops_load_error}. "
            "Rebuild with: ./build_secure.sh --debug --lite"
        )

    return _ops.qasa_attention(
        queries=queries,
        keys=keys,
        values=values,
        vqc_params=vqc_params,
        num_qubits=num_qubits,
        vqc_layers=vqc_layers,
        entanglement_strength=entanglement_strength,
    )


# =============================================================================
# AVAILABILITY CHECK
# =============================================================================


def is_qasa_available() -> bool:
    """Check if QASA C++ operators are available.

    Returns:
        True if ops are loaded and available, False otherwise.
    """
    if _ops is None:
        return False

    try:
        return hasattr(_ops, "qasa_attention")
    except Exception:
        return False


__all__ = [
    "run_qasa_attention",
    "is_qasa_available",
]
