# highnoon/_native/ops/quantum_crystallization_ops.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
"""Python wrappers for Quantum Crystallization C++ operations (Phase 65/83).

Provides long-term quantum memory storage with orthogonal gradient projection.

NO PYTHON FALLBACKS: RuntimeError raised if native ops unavailable.
"""

import logging
import tensorflow as tf
from highnoon._native.ops.lib_loader import resolve_op_library
from highnoon import config

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
        raise RuntimeError(f"Quantum crystallization ops unavailable: {e}") from e
    return _available


def ops_available() -> bool:
    try:
        _load_ops()
        return _available
    except RuntimeError:
        return False


def crystallize_memory(
    knowledge: tf.Tensor,
    importance: tf.Tensor,
    threshold: float | None = None,
) -> tf.Tensor:
    """Crystallize important knowledge for long-term persistence.

    Args:
        knowledge: Knowledge tensor [batch, dim].
        importance: Importance scores [batch, dim].
        threshold: Crystallization threshold (default from config).

    Returns:
        Crystallized memory tensor [batch, dim].
    """
    if not config.USE_QUANTUM_CRYSTALLIZATION:
        return knowledge
    _load_ops()
    threshold = threshold or config.CRYSTALLIZATION_THRESHOLD
    return _module.crystallize_memory(knowledge, importance, threshold=threshold)


def retrieve_from_crystal(
    crystal: tf.Tensor,
    query: tf.Tensor,
) -> tf.Tensor:
    """Retrieve from crystallized memory.

    Args:
        crystal: Crystallized memory [batch, dim].
        query: Query tensor [batch, dim].

    Returns:
        Retrieved memory [batch, dim].
    """
    _load_ops()
    return _module.retrieve_from_crystal(crystal, query)


__all__ = ["crystallize_memory", "retrieve_from_crystal", "ops_available"]
