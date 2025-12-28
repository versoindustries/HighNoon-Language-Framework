# highnoon/_native/ops/quantum_advanced_ops.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
"""Python wrappers for Quantum Advanced C++ operations (Phases 73, 79, 80, 84).

Provides NQS decoder, QCOT reasoning, waveform attention, and coherence metrics.

NO PYTHON FALLBACKS: RuntimeError raised if native ops unavailable.
"""

import logging

import tensorflow as tf

from highnoon import config
from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

_module = None
_available = False


def _load_ops():
    global _module, _available
    if _module is not None:
        return _available
    try:
        lib_path = resolve_op_library(__file__, "_highnoon_core.so")
        _module = tf.load_op_library(lib_path)
        _available = True
    except Exception as e:
        _available = False
        raise RuntimeError(f"Quantum advanced ops unavailable: {e}") from e
    return _available


def ops_available() -> bool:
    try:
        _load_ops()
        return _available
    except RuntimeError:
        return False


def nqs_decoder(
    visible: tf.Tensor,
    weights: tf.Tensor,
    bias: tf.Tensor,
) -> tf.Tensor:
    """Phase 73: Neural Quantum State decoder.

    Args:
        visible: Visible layer tensor [batch, v_dim].
        weights: Weight matrix [v_dim, h_dim].
        bias: Bias vector [h_dim].

    Returns:
        Hidden layer tensor [batch, h_dim].
    """
    if not config.HAMILTONIAN_ENABLE_NQS:
        return tf.zeros([tf.shape(visible)[0], bias.shape[0]])
    _load_ops()
    return _module.nqs_decoder(visible, weights, bias)


def qcot_reason(
    thought: tf.Tensor,
    reasoning_weights: tf.Tensor,
    steps: int | None = None,
) -> tf.Tensor:
    """Phase 79: Quantum chain-of-thought reasoning.

    Args:
        thought: Current thought tensor [batch, dim].
        reasoning_weights: Reasoning weights [dim, dim].
        steps: Number of reasoning steps (default from config).

    Returns:
        Next thought tensor [batch, dim].
    """
    if not config.USE_QCOT_REASONING:
        return thought
    _load_ops()
    steps = steps or config.QCOT_REASONING_STEPS
    return _module.qcot_reason(thought, reasoning_weights, steps=steps)


def waveform_attention(
    input_tensor: tf.Tensor,
) -> tf.Tensor:
    """Phase 80: Waveform-based attention pooling.

    Args:
        input_tensor: Input tensor [batch, seq, dim].

    Returns:
        Pooled output tensor [batch, dim].
    """
    if not config.USE_WAVEFORM_ATTENTION:
        return tf.reduce_mean(input_tensor, axis=1)
    _load_ops()
    return _module.waveform_attention(input_tensor)


def compute_coherence(
    state: tf.Tensor,
) -> tf.Tensor:
    """Phase 84: Compute coherence metric for training monitoring.

    Args:
        state: State tensor [batch, dim] or flat.

    Returns:
        Coherence metric (scalar).
    """
    _load_ops()
    return _module.compute_coherence(state)


__all__ = ["nqs_decoder", "qcot_reason", "waveform_attention", "compute_coherence", "ops_available"]
