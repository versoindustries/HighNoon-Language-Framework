# highnoon/_native/ops/hd_gradient_compression.py
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

"""HD-native gradient compression via frequency-domain filtering.

Phase 300+: GaLore integrated into Hyperdimensional Architecture.

This module replaces traditional SVD-based GaLore with frequency-domain
gradient compression that operates natively in HD space. Key insight:
HD computing uses FFT for holographic binding, so gradient compression
should also operate in frequency domain.

Compression method:
    1. FFT(gradient) to frequency domain
    2. Keep top-K frequency components by magnitude
    3. Zero out remaining frequencies
    4. IFFT back to spatial domain

This achieves similar memory savings to GaLore's low-rank projection
but is more coherent with HD architecture.

Example:
    >>> compressor = HDGradientCompressor(bandwidth=256)
    >>> compressed, mask = compressor.compress(gradient, variable)
    >>> # ... optimizer update on compressed gradient ...
    >>> full_grad = compressor.decompress(compressed, mask)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf

logger = logging.getLogger(__name__)

# Attempt to load native ops
_NATIVE_AVAILABLE = False
_hd_gradient_fft_compress = None
_hd_gradient_fft_decompress = None

try:
    from highnoon._native import get_op

    _hd_gc_module = get_op("hd_gradient_compression")
    if _hd_gc_module is not None:
        _hd_gradient_fft_compress = getattr(_hd_gc_module, "HDGradientFFTCompress", None)
        _hd_gradient_fft_decompress = getattr(_hd_gc_module, "HDGradientFFTDecompress", None)
        if _hd_gradient_fft_compress is not None:
            _NATIVE_AVAILABLE = True
            logger.info("Loaded hd_gradient_compression C++ FFT ops")
except ImportError:
    logger.debug("hd_gradient_compression C++ ops not available, using TF fallback")
except Exception as e:
    logger.warning("Failed to load hd_gradient_compression C++ ops: %s", e)


# Module-level cache for gradient compressor functions to prevent retracing
# Each unique bandwidth value gets its own cached function
_GRADIENT_COMPRESSOR_CACHE: dict[int, callable] = {}


def hd_gradient_compression_available() -> bool:
    """Check if native HD gradient compression ops are available."""
    return _NATIVE_AVAILABLE


# =============================================================================
# Pure TensorFlow Implementation
# =============================================================================


@tf.autograph.experimental.do_not_convert
def frequency_topk_mask(
    gradient: tf.Tensor,
    bandwidth: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Create frequency-domain top-K mask for gradient compression.

    Computes FFT of gradient and identifies top-K frequency components
    by magnitude. Returns a boolean mask for the kept frequencies.

    Note: This function is NOT decorated with @tf.function to avoid retracing
    issues. The C++ ops are already graph-compatible. The outer calling context
    (HDGradientCompressor) handles graph mode appropriately.

    Args:
        gradient: Input gradient tensor [*shape] (any shape, FFT on last dim)
        bandwidth: Number of frequency components to keep (Python int)

    Returns:
        Tuple of (compressed_fft, mask_indices):
        - compressed_fft: Complex tensor of kept frequencies [*shape[:-1], bandwidth]
        - mask_indices: Int32 indices of kept frequencies [*shape[:-1], bandwidth]
    """
    if not _NATIVE_AVAILABLE or _hd_gradient_fft_compress is None:
        raise RuntimeError(
            "HD gradient compression native ops not available. "
            "Rebuild native extensions with: ./build_secure.sh --debug --lite"
        )

    # Use C++ op
    # Flatten if needed
    is_tensor = hasattr(gradient, "shape")
    if is_tensor:
        rank_tensor = len(gradient.shape)
        if rank_tensor > 2:
            # Flatten to [batch, dim] or [dim]
            flat_dim = gradient.shape[-1]
            flat_grad = tf.reshape(gradient, [-1, flat_dim])
        else:
            flat_grad = gradient
    else:
        flat_grad = gradient

    # Ensure bandwidth is a Python int for the C++ op
    bw = int(bandwidth) if not isinstance(bandwidth, int) else bandwidth

    compressed_real, compressed_imag, mask_indices = _hd_gradient_fft_compress(
        gradient=flat_grad, bandwidth=bw, preserve_dc=True
    )

    # Combine real/imag into complex
    compressed_fft = tf.complex(compressed_real, compressed_imag)

    return compressed_fft, mask_indices


@tf.autograph.experimental.do_not_convert
def frequency_reconstruct(
    compressed_fft: tf.Tensor,
    mask_indices: tf.Tensor,
    original_dim: int | tf.Tensor,
) -> tf.Tensor:
    """Reconstruct gradient from compressed frequency representation.

    Note: This function is NOT decorated with @tf.function to avoid retracing
    issues. The C++ ops are already graph-compatible.

    Args:
        compressed_fft: Compressed FFT coefficients [batch, bandwidth]
        mask_indices: Indices of kept frequencies [batch, bandwidth]
        original_dim: Original dimension size (Python int or scalar tensor)

    Returns:
        Reconstructed gradient [batch, original_dim]
    """
    if not _NATIVE_AVAILABLE or _hd_gradient_fft_decompress is None:
        raise RuntimeError(
            "HD gradient compression native ops not available. "
            "Rebuild native extensions with: ./build_secure.sh --debug --lite"
        )

    # Convert original_dim to Python int if it's a tensor
    if isinstance(original_dim, tf.Tensor):
        # In graph mode, we need to handle this carefully
        # Use tf.py_function or ensure static shape is available
        if original_dim.shape.rank == 0:
            # Scalar tensor - try to get static value
            try:
                original_dim = int(original_dim.numpy())
            except (AttributeError, RuntimeError):
                # In graph mode, pass through as-is (C++ op handles it)
                pass
    else:
        original_dim = int(original_dim)

    # Extract components
    comp_real = tf.math.real(compressed_fft)
    comp_imag = tf.math.imag(compressed_fft)

    reconstructed = _hd_gradient_fft_decompress(
        compressed_real=comp_real,
        compressed_imag=comp_imag,
        indices=mask_indices,
        original_dim=original_dim,
        scale=1.0,
    )
    return reconstructed


# =============================================================================
# HDGradientCompressor Class
# =============================================================================


@dataclass
class HDGradientCompressor:
    """Frequency-domain gradient compressor for HD embeddings.

    Compresses gradients by keeping only the top-K frequency components
    after FFT. This is the HD-native equivalent of GaLore's low-rank
    projection.

    Attributes:
        bandwidth: Number of frequency components to keep per dimension.
        enabled: Whether compression is active.
        scale: Gradient scaling factor after decompression.

    Example:
        >>> compressor = HDGradientCompressor(bandwidth=256)
        >>> compressed, meta = compressor.compress(grad, var)
        >>> # Apply optimizer to compressed gradient
        >>> full_update = compressor.decompress(compressed, meta)
    """

    bandwidth: int = 256
    enabled: bool = True
    scale: float = 1.0

    # Internal state
    _compression_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    _step_counter: int = 0
    _stats: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize compressor state."""
        if not hasattr(self, "_compression_cache") or self._compression_cache is None:
            self._compression_cache = {}
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0
        if not hasattr(self, "_stats") or self._stats is None:
            self._stats = {"total_compressed": 0, "total_original": 0}
        else:
            self._stats.setdefault("total_compressed", 0)
            self._stats.setdefault("total_original", 0)

        logger.info(
            "[HD_GRADIENT] Initialized: bandwidth=%d, enabled=%s, scale=%.2f",
            self.bandwidth,
            self.enabled,
            self.scale,
        )

    def compress(
        self,
        gradient: tf.Tensor,
        variable: tf.Variable,
    ) -> tuple[tf.Tensor, dict[str, Any]]:
        """Compress gradient using frequency-domain filtering.

        Args:
            gradient: Full gradient tensor.
            variable: Associated weight variable.

        Returns:
            Tuple of (compressed_gradient, compression_metadata).
            Metadata includes indices and original shape for decompression.
        """
        if not self.enabled:
            return gradient, {"passthrough": True}

        var_name = variable.name
        original_shape = gradient.shape.as_list()
        original_dim = original_shape[-1] if original_shape else 1
        if original_dim is None:
            logger.warning(
                "[HD_GRADIENT] Skipping FFT compression for %s: dynamic last dimension",
                var_name,
            )
            return gradient, {"passthrough": True}

        effective_bandwidth = min(self.bandwidth, int(original_dim))
        if effective_bandwidth < 1:
            return gradient, {"passthrough": True}

        # Flatten to 2D for processing: [batch, features]
        flat_grad = tf.reshape(gradient, [-1, original_dim])

        # Compute frequency compression
        compressed_fft, mask_indices = frequency_topk_mask(flat_grad, effective_bandwidth)

        # Track stats
        if tf.executing_eagerly():
            original_size = int(tf.size(flat_grad).numpy())
            compressed_size = int(tf.size(compressed_fft).numpy())
            self._stats.setdefault("total_compressed", 0)
            self._stats.setdefault("total_original", 0)
            self._stats["total_compressed"] += compressed_size
            self._stats["total_original"] += original_size

        # Store metadata for decompression
        metadata = {
            "passthrough": False,
            "var_name": var_name,
            "original_shape": original_shape,
            "original_dim": original_dim,
            "mask_indices": mask_indices,
            "flat_shape": flat_grad.shape.as_list(),
            "bandwidth": effective_bandwidth,
        }

        logger.debug(
            "[HD_GRADIENT] Compressed %s: %d -> %d (%.1fx)",
            var_name,
            original_size,
            compressed_size,
            original_size / max(compressed_size, 1),
        )

        return compressed_fft, metadata

    def decompress(
        self,
        compressed: tf.Tensor,
        metadata: dict[str, Any],
    ) -> tf.Tensor:
        """Decompress gradient back to original shape.

        Args:
            compressed: Compressed gradient (FFT coefficients).
            metadata: Compression metadata from compress().

        Returns:
            Reconstructed full gradient tensor.
        """
        if metadata.get("passthrough", False):
            return compressed

        original_shape = metadata["original_shape"]
        original_dim = metadata["original_dim"]
        mask_indices = metadata["mask_indices"]

        # Reconstruct from frequency domain
        reconstructed_flat = frequency_reconstruct(
            compressed,
            mask_indices,
            original_dim,
        )

        # Reshape to original
        reconstructed = tf.reshape(reconstructed_flat, original_shape)

        # Apply scale
        if self.scale != 1.0:
            reconstructed = reconstructed * self.scale

        return reconstructed

    def step(self) -> None:
        """Increment step counter (for potential adaptive bandwidth)."""
        self._step_counter += 1

    def reset(self) -> None:
        """Reset compressor state."""
        self._compression_cache.clear()
        self._step_counter = 0
        self._stats = {"total_compressed": 0, "total_original": 0}

    def get_statistics(self) -> dict[str, Any]:
        """Get compression statistics.

        Returns:
            Dict with compression metrics.
        """
        total_orig = self._stats.get("total_original", 0)
        total_comp = self._stats.get("total_compressed", 0)

        return {
            "bandwidth": self.bandwidth,
            "step_counter": self._step_counter,
            "total_original": total_orig,
            "total_compressed": total_comp,
            "overall_compression_ratio": (
                total_orig / max(total_comp, 1) if total_comp > 0 else 1.0
            ),
            "enabled": self.enabled,
        }


# =============================================================================
# Custom Gradient Modifier Layer
# =============================================================================


@tf.autograph.experimental.do_not_convert
def _get_gradient_compressor(bandwidth: int):
    """Get or create a cached gradient compression function for a specific bandwidth.

    This function caches the created compression functions to prevent tf.function
    retracing on every call. Each unique bandwidth value gets its own cached function.

    The @tf.autograph.experimental.do_not_convert decorator prevents AutoGraph
    from attempting to transform this function, silencing "could not get source code"
    warnings that occur with dynamically loaded C++ ops.

    Args:
        bandwidth: Number of frequency components to keep.

    Returns:
        A cached function decorated with @tf.custom_gradient that compresses gradients.
    """
    # Check cache first to avoid creating new functions
    if bandwidth in _GRADIENT_COMPRESSOR_CACHE:
        return _GRADIENT_COMPRESSOR_CACHE[bandwidth]

    # Create new function for this bandwidth value
    # Capture bandwidth as a closure variable (constant for this function)
    bw = bandwidth  # Capture in closure

    @tf.autograph.experimental.do_not_convert
    @tf.custom_gradient
    def compress_gradients(x):
        @tf.autograph.experimental.do_not_convert
        def grad(dy):
            # Compress the incoming gradient
            # Get the last dimension size as a Python int when possible
            dy_shape = dy.shape
            if dy_shape.rank is not None and dy_shape[-1] is not None:
                last_dim = int(dy_shape[-1])
            else:
                # Fallback for dynamic shapes - use tf.shape
                last_dim = tf.shape(dy)[-1]

            flat_dy = tf.reshape(dy, [-1, last_dim])
            compressed, indices = frequency_topk_mask(flat_dy, bw)
            reconstructed = frequency_reconstruct(compressed, indices, last_dim)
            return tf.reshape(reconstructed, tf.shape(dy))

        return x, grad

    # Cache the function
    _GRADIENT_COMPRESSOR_CACHE[bandwidth] = compress_gradients
    logger.debug("[HD_GRADIENT] Cached compressor function for bandwidth=%d", bandwidth)

    return compress_gradients


class HDGradientCompressionLayer(tf.keras.layers.Layer):
    """Layer wrapper that applies HD gradient compression during training.

    Wraps any layer and compresses gradients flowing through it during
    backpropagation. Forward pass is unaffected.

    Args:
        wrapped_layer: The layer to wrap.
        bandwidth: Frequency bandwidth for compression.

    Example:
        >>> dense = tf.keras.layers.Dense(256)
        >>> compressed_dense = HDGradientCompressionLayer(dense, bandwidth=128)
    """

    def __init__(
        self,
        wrapped_layer: tf.keras.layers.Layer,
        bandwidth: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wrapped_layer = wrapped_layer
        self.bandwidth = bandwidth
        self._compressor = HDGradientCompressor(bandwidth=bandwidth)

    def call(self, inputs, training=None):
        """Forward pass with gradient compression hook."""
        output = self.wrapped_layer(inputs, training=training)

        if training:
            # Apply gradient compression via custom gradient (cached to prevent retracing)
            compress_fn = _get_gradient_compressor(self.bandwidth)
            output = compress_fn(output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bandwidth": self.bandwidth,
                "wrapped_layer": tf.keras.layers.serialize(self.wrapped_layer),
            }
        )
        return config


__all__ = [
    "HDGradientCompressor",
    "HDGradientCompressionLayer",
    "frequency_topk_mask",
    "frequency_reconstruct",
    "hd_gradient_compression_available",
]
