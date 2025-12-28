# highnoon/_native/ops/vqem_ops.py
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

"""Python wrappers for VQEM Error Mitigation C++ operations (Phase 62).

Variational Quantum Error Mitigation (VQEM) uses learnable parameters
to correct errors in quantum-enhanced activations.

NO PYTHON FALLBACKS: RuntimeError raised if native ops unavailable.

Ops:
    - vqem_forward: Apply error mitigation to noisy states
    - vqem_train_step: Update mitigation parameters
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
        logger.info(f"VQEM ops loaded from {lib_path}")
    except Exception as e:
        _available = False
        logger.warning(f"Failed to load VQEM ops: {e}")
        raise RuntimeError(
            "VQEM native ops not available. " "Run ./build_secure.sh to compile."
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
# Phase 62: VQEM Error Mitigation
# =============================================================================


def vqem_forward(
    input_state: tf.Tensor,
    mitigation_params: tf.Tensor,
) -> tf.Tensor:
    """Apply VQEM error mitigation to quantum states.

    VQEM learns to correct systematic errors in quantum-enhanced
    activations by applying a learned parameterized correction circuit.

    Args:
        input_state: Noisy quantum state [batch, dim].
        mitigation_params: Learned mitigation parameters [num_params].
            The number of parameters determines the complexity of the
            error mitigation circuit.

    Returns:
        Mitigated quantum state [batch, dim].

    Raises:
        RuntimeError: If native op not available.

    Example:
        >>> noisy_state = vqc_layer(x)  # VQC output with noise
        >>> params = tf.Variable(tf.zeros([config.VQEM_NUM_PARAMS]))
        >>> clean_state = vqem_forward(noisy_state, params)
    """
    if not config.USE_VQEM:
        return input_state

    _load_ops()
    return _module.vqem_forward(input_state, mitigation_params)


def vqem_train_step(
    mitigation_params: tf.Tensor,
    noisy_output: tf.Tensor,
    ideal_output: tf.Tensor,
    learning_rate: float = 0.01,
) -> tf.Tensor:
    """Train VQEM mitigation parameters.

    Updates the mitigation parameters to minimize the difference
    between the mitigated noisy output and the ideal output.

    Args:
        mitigation_params: Current mitigation parameters [num_params].
        noisy_output: Output from noisy quantum circuit [batch, dim].
        ideal_output: Target clean output [batch, dim].
        learning_rate: Learning rate for parameter update.

    Returns:
        Updated mitigation parameters [num_params].

    Raises:
        RuntimeError: If native op not available.

    Example:
        >>> # During training with access to ideal outputs
        >>> noisy = vqc_layer(x)
        >>> ideal = ideal_vqc_layer(x)  # From simulator or reference
        >>> params = vqem_train_step(params, noisy, ideal, lr=0.01)
    """
    _load_ops()
    return _module.vqem_train_step(
        mitigation_params,
        noisy_output,
        ideal_output,
        learning_rate=learning_rate,
    )


def create_vqem_params(num_params: int | None = None) -> tf.Variable:
    """Create initialized VQEM mitigation parameters.

    Args:
        num_params: Number of parameters (default from config).

    Returns:
        Trainable VQEM parameter variable.
    """
    num_params = num_params or config.VQEM_NUM_PARAMS
    return tf.Variable(
        tf.random.uniform([num_params], -0.1, 0.1),
        trainable=True,
        name="vqem_mitigation_params",
    )


__all__ = [
    "vqem_forward",
    "vqem_train_step",
    "create_vqem_params",
    "ops_available",
]
