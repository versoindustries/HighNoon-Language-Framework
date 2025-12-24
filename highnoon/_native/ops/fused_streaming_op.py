# highnoon/_native/ops/fused_streaming_op.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
"""Python wrapper for fused Streaming C++ ops."""

from __future__ import annotations

import tensorflow as tf

from highnoon._native import get_op

_lib = get_op("fused_streaming")
_compress_op = getattr(_lib, "FusedStreamingCompress", None) if _lib else None
_update_op = getattr(_lib, "FusedStreamingUpdate", None) if _lib else None


def fused_streaming_compress(state: tf.Tensor, target_dim: int) -> tf.Tensor:
    """Compress streaming state to target dimension."""
    if _compress_op is None:
        # Fallback: simple truncation
        return state[..., :target_dim]
    return _compress_op(state=tf.cast(state, tf.float32), target_dim=target_dim)


def fused_streaming_update(
    old_state: tf.Tensor, new_state: tf.Tensor, alpha: float = 0.9
) -> tf.Tensor:
    """Update streaming state with EMA."""
    if _update_op is None:
        return alpha * old_state + (1 - alpha) * new_state
    return _update_op(
        old_state=tf.cast(old_state, tf.float32),
        new_state=tf.cast(new_state, tf.float32),
        alpha=alpha,
    )


__all__ = ["fused_streaming_compress", "fused_streaming_update"]
