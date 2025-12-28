# highnoon/_native/ops/quantum_dropout_ops.py
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

"""Python wrappers for Quantum Measurement Dropout C++ operations (Phase 47).

Provides quantum-inspired dropout mechanisms that simulate measurement
collapse for improved regularization in quantum-enhanced neural networks.

NO PYTHON FALLBACKS: RuntimeError raised if native ops unavailable.

Ops:
    - quantum_measurement_dropout: Hard measurement dropout
    - soft_quantum_dropout: Learnable soft dropout
    - entangling_dropout: Correlated dropout across dimensions
"""

import logging
from typing import Tuple

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
        logger.info(f"Quantum Dropout ops loaded from {lib_path}")
    except Exception as e:
        _available = False
        logger.warning(f"Failed to load Quantum Dropout ops: {e}")
        raise RuntimeError(
            "Quantum Dropout native ops not available. " "Run ./build_secure.sh to compile."
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
# Phase 47: Quantum Measurement Dropout
# =============================================================================


def quantum_measurement_dropout(
    input_tensor: tf.Tensor,
    drop_rate: float | None = None,
    training: bool = True,
    seed: int = 42,
) -> tf.Tensor:
    """Quantum measurement dropout with state collapse simulation.

    Randomly measures selected positions in the activation tensor,
    collapsing their "quantum state" to a classical value. This creates
    an ensemble effect through varying effective circuit depths.

    Unlike standard dropout, this preserves quantum correlations
    between non-dropped positions.

    Args:
        input_tensor: Input activations [batch, seq, dim].
        drop_rate: Measurement probability (default from config).
        training: If True, apply dropout. If False, pass through.
        seed: Random seed for reproducible measurements.

    Returns:
        Dropped activations [batch, seq, dim].

    Raises:
        RuntimeError: If native op not available.

    Example:
        >>> x = tf.random.normal([4, 32, 64])
        >>> y = quantum_measurement_dropout(x, drop_rate=0.1, training=True)
    """
    if not config.USE_QUANTUM_MEASUREMENT_DROPOUT or not training:
        return input_tensor

    _load_ops()
    drop_rate = drop_rate if drop_rate is not None else config.QMD_DROP_RATE

    return _module.quantum_measurement_dropout(
        input_tensor,
        drop_rate=drop_rate,
        seed=seed,
        training=training,
    )


def soft_quantum_dropout(
    input_tensor: tf.Tensor,
    softening_params: tf.Tensor,
    drop_rate: float | None = None,
    temperature: float | None = None,
    training: bool = True,
    seed: int = 42,
) -> tf.Tensor:
    """Soft quantum dropout with learnable softening parameters.

    Uses soft measurement operator: M_soft = (1-σ)·I + σ·|0⟩⟨0|
    where σ controls the measurement strength per dimension.

    This allows gradients to flow through the dropout operation,
    enabling end-to-end training of the softening parameters.

    Args:
        input_tensor: Input activations [batch, seq, dim].
        softening_params: Learnable softening parameters [dim].
        drop_rate: Base measurement probability (default from config).
        temperature: Softening temperature (default from config).
        training: If True, apply dropout. If False, pass through.
        seed: Random seed.

    Returns:
        Softly dropped activations [batch, seq, dim].

    Raises:
        RuntimeError: If native op not available.
    """
    if not config.USE_QUANTUM_MEASUREMENT_DROPOUT or not training:
        return input_tensor

    _load_ops()
    drop_rate = drop_rate if drop_rate is not None else config.QMD_DROP_RATE
    temperature = temperature if temperature is not None else config.QMD_SOFTENING_TEMP

    return _module.soft_quantum_dropout(
        input_tensor,
        softening_params,
        drop_rate=drop_rate,
        temperature=temperature,
        seed=seed,
        training=training,
    )


def soft_quantum_dropout_grad(
    grad_output: tf.Tensor,
    input_tensor: tf.Tensor,
    softening_params: tf.Tensor,
    drop_rate: float | None = None,
    temperature: float | None = None,
    seed: int = 42,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Gradient computation for soft quantum dropout.

    Args:
        grad_output: Gradient from output [batch, seq, dim].
        input_tensor: Original input [batch, seq, dim].
        softening_params: Learnable parameters [dim].
        drop_rate: Base drop rate (default from config).
        temperature: Softening temperature (default from config).
        seed: Random seed (must match forward pass).

    Returns:
        Tuple of (grad_input [batch, seq, dim], grad_params [dim]).

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    drop_rate = drop_rate if drop_rate is not None else config.QMD_DROP_RATE
    temperature = temperature if temperature is not None else config.QMD_SOFTENING_TEMP

    return _module.soft_quantum_dropout_grad(
        grad_output,
        input_tensor,
        softening_params,
        drop_rate=drop_rate,
        temperature=temperature,
        seed=seed,
    )


# Register gradient for SoftQuantumDropout
@tf.RegisterGradient("SoftQuantumDropout")
def _soft_quantum_dropout_grad(op, grad):
    """Gradient registration for SoftQuantumDropout."""
    input_tensor = op.inputs[0]
    softening_params = op.inputs[1]
    drop_rate = op.get_attr("drop_rate")
    temperature = op.get_attr("temperature")
    seed = op.get_attr("seed")

    _load_ops()
    grad_input, grad_params = _module.soft_quantum_dropout_grad(
        grad,
        input_tensor,
        softening_params,
        drop_rate=drop_rate,
        temperature=temperature,
        seed=seed,
    )
    return [grad_input, grad_params]


def entangling_dropout(
    input_tensor: tf.Tensor,
    drop_rate: float | None = None,
    training: bool = True,
    seed: int = 42,
) -> tf.Tensor:
    """Entangling gate dropout with correlated mask.

    Randomly skips entangling operations between feature dimensions,
    creating varying effective circuit depths. Unlike standard dropout,
    the mask is correlated across sequence positions.

    Args:
        input_tensor: Input after local gates [batch, seq, dim].
        drop_rate: Entangling dropout rate (default from config).
        training: If True, apply dropout. If False, pass through.
        seed: Random seed.

    Returns:
        Output with entangling dropout [batch, seq, dim].

    Raises:
        RuntimeError: If native op not available.
    """
    if not config.USE_QUANTUM_MEASUREMENT_DROPOUT or not training:
        return input_tensor

    _load_ops()
    drop_rate = drop_rate if drop_rate is not None else config.QMD_DROP_RATE

    return _module.entangling_dropout(
        input_tensor,
        drop_rate=drop_rate,
        seed=seed,
        training=training,
    )


__all__ = [
    "quantum_measurement_dropout",
    "soft_quantum_dropout",
    "soft_quantum_dropout_grad",
    "entangling_dropout",
    "ops_available",
]
