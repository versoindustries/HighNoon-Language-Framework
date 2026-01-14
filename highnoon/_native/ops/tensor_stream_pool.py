"""
highnoon/_native/ops/tensor_stream_pool.py
Python wrapper for TensorStreamPool zero-copy inter-kernel streaming.

This module provides a Pythonic interface to the TensorStreamPool C++ ops
for testing, benchmarking, and telemetry access.

Part of TensorStreamPool C++ Enhancement Roadmap Phase 0.
"""

import logging
from typing import Any

import tensorflow as tf

from highnoon._native import get_op
from highnoon.config import TENSOR_STREAM_DEBUG

logger = logging.getLogger(__name__)

# Lazy loading of the ops module
_ops_module = None


def _get_ops():
    """Lazy load the tensor stream pool ops."""
    global _ops_module
    if _ops_module is None:
        _ops_module = get_op("highnoon_core")
        if _ops_module is None:
            raise ImportError(
                "TensorStreamPool ops not available. Build with: ./build_secure.sh --debug --lite"
            )
    return _ops_module


# =============================================================================
# Core API Functions
# =============================================================================


def tensor_stream_acquire(size_bytes: int, producer_hint: str = "") -> int:
    """Acquire a buffer from the TensorStreamPool.

    Returns a pointer (as int64) to an aligned buffer suitable for
    zero-copy streaming between kernels.

    Args:
        size_bytes: Number of bytes to allocate.
        producer_hint: Optional name of producing kernel (for debugging).

    Returns:
        Buffer pointer as int64, or 0 on failure.

    Example:
        >>> ptr = tensor_stream_acquire(4096, "my_kernel")
        >>> assert ptr > 0  # Valid pointer
        >>> tensor_stream_release(ptr)
    """
    ops = _get_ops()
    result = ops.tensor_stream_acquire(
        tf.constant(size_bytes, dtype=tf.int64), producer_hint=producer_hint
    )
    ptr = int(result.numpy())

    if TENSOR_STREAM_DEBUG:
        logger.debug(
            "[TensorStreamPool] ACQUIRE: size=%d bytes, producer='%s', ptr=0x%x",
            size_bytes,
            producer_hint,
            ptr,
        )

    return ptr


def tensor_stream_handoff(buffer_ptr: int, consumer_hint: str = "") -> None:
    """Mark buffer as ready for handoff to consumer.

    Signals that the producer has finished writing and the buffer
    is ready for zero-copy consumption.

    Args:
        buffer_ptr: Pointer from tensor_stream_acquire.
        consumer_hint: Optional name of expected consumer kernel.
    """
    if TENSOR_STREAM_DEBUG:
        logger.debug(
            "[TensorStreamPool] HANDOFF: ptr=0x%x, consumer='%s'", buffer_ptr, consumer_hint
        )

    ops = _get_ops()
    ops.tensor_stream_handoff(tf.constant(buffer_ptr, dtype=tf.int64), consumer_hint=consumer_hint)


def tensor_stream_release(buffer_ptr: int) -> None:
    """Release buffer back to pool for reuse.

    The buffer remains allocated but is marked available for future
    acquire calls of compatible size.

    Args:
        buffer_ptr: Pointer from tensor_stream_acquire.
    """
    if TENSOR_STREAM_DEBUG:
        logger.debug("[TensorStreamPool] RELEASE: ptr=0x%x", buffer_ptr)

    ops = _get_ops()
    ops.tensor_stream_release(tf.constant(buffer_ptr, dtype=tf.int64))


def tensor_stream_get_stats() -> dict[str, Any]:
    """Get streaming statistics from TensorStreamPool.

    Returns:
        Dictionary with telemetry data:
        - total_allocated_bytes: Total bytes allocated by pool
        - num_buffers: Number of buffer entries
        - acquire_count: Total acquire() calls
        - reuse_count: acquire() that reused existing buffer
        - zero_copy_handoffs: Successful zero-copy handoffs
        - release_count: Total release() calls
        - peak_usage_bytes: Peak memory in use simultaneously
        - current_usage_bytes: Current memory in use
    """
    ops = _get_ops()
    results = ops.tensor_stream_get_stats()

    # Unpack 8 outputs
    return {
        "total_allocated_bytes": int(results[0].numpy()),
        "num_buffers": int(results[1].numpy()),
        "acquire_count": int(results[2].numpy()),
        "reuse_count": int(results[3].numpy()),
        "zero_copy_handoffs": int(results[4].numpy()),
        "release_count": int(results[5].numpy()),
        "peak_usage_bytes": int(results[6].numpy()),
        "current_usage_bytes": int(results[7].numpy()),
    }


def tensor_stream_clear() -> None:
    """Clear all buffers from the pool, freeing memory.

    WARNING: Invalidates all previously acquired pointers!
    """
    ops = _get_ops()
    ops.tensor_stream_clear()


# =============================================================================
# High-Level Utilities
# =============================================================================


class StreamingBuffer:
    """Context manager for safe buffer lifecycle.

    Ensures proper acquire/release even if exceptions occur.

    Example:
        >>> with StreamingBuffer(4096) as buf:
        ...     # Use buf.ptr for computation
        ...     buf.handoff("consumer")
        # Buffer automatically released on exit
    """

    def __init__(self, size_bytes: int, producer_hint: str = ""):
        """Initialize streaming buffer.

        Args:
            size_bytes: Number of bytes to allocate.
            producer_hint: Optional producer name.
        """
        self.size_bytes = size_bytes
        self.producer_hint = producer_hint
        self.ptr: int = 0
        self._handed_off = False

    def __enter__(self) -> "StreamingBuffer":
        """Acquire buffer."""
        self.ptr = tensor_stream_acquire(self.size_bytes, self.producer_hint)
        if self.ptr == 0:
            raise MemoryError(f"Failed to acquire {self.size_bytes} bytes")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release buffer."""
        if self.ptr > 0:
            tensor_stream_release(self.ptr)
            self.ptr = 0
        return False

    def handoff(self, consumer_hint: str = "") -> None:
        """Mark buffer ready for consumer."""
        if self.ptr > 0 and not self._handed_off:
            tensor_stream_handoff(self.ptr, consumer_hint)
            self._handed_off = True


def print_stats() -> None:
    """Print formatted streaming statistics."""
    stats = tensor_stream_get_stats()

    print("\n=== TensorStreamPool Statistics ===")
    print(f"Buffers:           {stats['num_buffers']}")
    print(f"Total Allocated:   {stats['total_allocated_bytes'] / (1024**2):.2f} MB")
    print(f"Current Usage:     {stats['current_usage_bytes'] / (1024**2):.2f} MB")
    print(f"Peak Usage:        {stats['peak_usage_bytes'] / (1024**2):.2f} MB")
    print(f"Acquire Count:     {stats['acquire_count']}")
    print(f"Reuse Count:       {stats['reuse_count']}")
    if stats["acquire_count"] > 0:
        reuse_rate = stats["reuse_count"] / stats["acquire_count"] * 100
        print(f"Reuse Rate:        {reuse_rate:.1f}%")
    print(f"Zero-Copy Handoffs: {stats['zero_copy_handoffs']}")
    print(f"Release Count:     {stats['release_count']}")
    print()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core API
    "tensor_stream_acquire",
    "tensor_stream_handoff",
    "tensor_stream_release",
    "tensor_stream_get_stats",
    "tensor_stream_clear",
    # Utilities
    "StreamingBuffer",
    "print_stats",
]
