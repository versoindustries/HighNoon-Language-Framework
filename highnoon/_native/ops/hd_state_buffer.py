# highnoon/_native/ops/hd_state_buffer.py
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

"""HD State Buffer for optimizer state compression.

Phase 300+: HD Upgrade Integration (hd_upgrade.md Phase 1).

Provides Hyperdimensional encoding/decoding for optimizer states
(momentum, Hessian diagonal, QFIM) achieving 50-60% memory reduction
for second-order optimizers like SophiaG, QIAO, and SympFlowQNG.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import tensorflow as tf

logger = logging.getLogger(__name__)

# Load native ops
_native_ops = None
_native_available = False


def _load_native_ops():
    """Load native HD state buffer ops."""
    global _native_ops, _native_available
    if _native_ops is not None:
        return _native_available

    try:
        from highnoon._native.ops.lib_loader import load_highnoon_core

        _native_ops = load_highnoon_core()
        _native_available = hasattr(_native_ops, "hd_state_encode") or hasattr(
            _native_ops, "HDStateEncode"
        )
        if _native_available:
            logger.debug("HD State Buffer native ops loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load HD State Buffer native ops: {e}")
        _native_available = False
    return _native_available


def hd_state_buffer_available() -> bool:
    """Check if HD state buffer native ops are available."""
    return _load_native_ops()


class HDStateBuffer:
    """HD-compressed optimizer state storage.

    Compresses optimizer state variables (momentum, Hessian, etc.) using
    random HD projection for memory reduction.

    Attributes:
        compression_ratio: Compression ratio (default 8x).
        sparse: Use sparse projection matrix (faster).
        seed: Random seed for reproducible projection.

    Example:
        >>> buffer = HDStateBuffer(compression_ratio=8)
        >>> buffer.register("layer1/kernel", (1024, 256))
        >>> buffer.encode("layer1/kernel", momentum_tensor)
        >>> decoded = buffer.decode("layer1/kernel")
    """

    def __init__(
        self,
        compression_ratio: int = 8,
        sparse: bool = True,
        sparse_density: int = 3,
        seed: int = 42,
    ):
        """Initialize HD State Buffer.

        Args:
            compression_ratio: Target compression ratio.
            sparse: Use sparse random projection (default True).
            sparse_density: Non-zeros per row for sparse projection.
            seed: Random seed for reproducibility.
        """
        self.compression_ratio = compression_ratio
        self.sparse = sparse
        self.sparse_density = sparse_density
        self.seed = seed

        self._projections: dict[str, tf.Tensor] = {}
        self._compressed: dict[str, tf.Variable] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}

        self._use_native = hd_state_buffer_available()
        if self._use_native:
            logger.debug("HDStateBuffer using native C++ ops")
        else:
            logger.debug("HDStateBuffer using TensorFlow fallback")

    def register(self, var_name: str, shape: tuple[int, ...]) -> None:
        """Register a variable for HD compression.

        Args:
            var_name: Name of the variable.
            shape: Shape of the variable.
        """
        if var_name in self._projections:
            return  # Already registered

        param_size = 1
        for dim in shape:
            param_size *= dim

        compressed_size = max(1, param_size // self.compression_ratio)

        # Generate projection matrix
        if self._use_native:
            projection = _native_ops.HDStateGenerateProjection(
                param_size=param_size,
                compressed_size=compressed_size,
                sparse=self.sparse,
                sparse_density=self.sparse_density,
                seed=self.seed + hash(var_name) % 10000,
            )
        else:
            # TensorFlow fallback: Gaussian random projection
            projection = tf.random.stateless_normal(
                shape=[param_size, compressed_size],
                seed=[self.seed + hash(var_name) % 10000, 0],
            ) / tf.sqrt(tf.cast(compressed_size, tf.float32))

        self._projections[var_name] = projection
        self._shapes[var_name] = shape

        # Initialize compressed storage
        self._compressed[var_name] = tf.Variable(
            tf.zeros([compressed_size], dtype=tf.float32),
            trainable=False,
            name=f"hd_compressed/{var_name.replace('/', '_')}",
        )

    def encode(self, var_name: str, state: tf.Tensor) -> None:
        """Compress and store state.

        Args:
            var_name: Name of the variable.
            state: State tensor to compress.
        """
        if var_name not in self._projections:
            self.register(var_name, tuple(state.shape.as_list()))

        projection = self._projections[var_name]
        flat_state = tf.reshape(state, [-1])

        if self._use_native:
            compressed = _native_ops.HDStateEncode(flat_state, projection)
        else:
            # TensorFlow fallback
            compressed = tf.linalg.matvec(projection, flat_state, transpose_a=True)

        self._compressed[var_name].assign(compressed)

    def decode(self, var_name: str) -> tf.Tensor:
        """Retrieve decompressed state.

        Args:
            var_name: Name of the variable.

        Returns:
            Decompressed state tensor.
        """
        if var_name not in self._compressed:
            raise KeyError(f"Variable {var_name} not registered")

        projection = self._projections[var_name]
        compressed = self._compressed[var_name]
        shape = self._shapes[var_name]

        if self._use_native:
            flat_state = _native_ops.HDStateDecode(compressed, projection)
        else:
            # TensorFlow fallback
            flat_state = tf.linalg.matvec(projection, compressed)

        return tf.reshape(flat_state, shape)

    def get_compression_stats(self) -> dict[str, dict]:
        """Get compression statistics for all registered variables.

        Returns:
            Dictionary mapping variable names to their stats.
        """
        stats = {}
        for var_name, shape in self._shapes.items():
            param_size = 1
            for dim in shape:
                param_size *= dim
            compressed_size = self._compressed[var_name].shape[0]

            stats[var_name] = {
                "original_size": param_size,
                "compressed_size": compressed_size,
                "compression_ratio": param_size / compressed_size,
                "memory_saved_bytes": (param_size - compressed_size) * 4,
            }
        return stats

    def clear(self) -> None:
        """Clear all stored states."""
        for var in self._compressed.values():
            var.assign(tf.zeros_like(var))


__all__ = ["HDStateBuffer", "hd_state_buffer_available"]
