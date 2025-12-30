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

"""HD Fisher Compression Python Wrapper.

VQC-HD Integration Enhancement #1: Compress layer-wise Fisher information
using holographic bundling before encoding into VQC qubit angles.

Example:
    >>> from highnoon._native.ops.hd_fisher_compression import hd_fisher_compress
    >>> fisher = tf.constant([0.1, 0.5, 0.3, 0.8])  # 4 layers
    >>> pos_keys = tf.random.uniform([4, 4096])
    >>> proj = tf.random.uniform([4096, 64])
    >>> compressed = hd_fisher_compress(fisher, pos_keys, proj)
    >>> print(compressed.shape)  # (64,)
"""

from __future__ import annotations

import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger(__name__)

# Module-level flags
_native_available = False
_lib = None


def _load_native_ops() -> bool:
    """Load native C++ operations."""
    global _native_available, _lib

    try:
        import tensorflow as tf

        from highnoon._native.ops.lib_loader import get_consolidated_library

        lib_path = get_consolidated_library()
        if lib_path is not None:
            _lib = tf.load_op_library(lib_path)
            if _lib is not None:
                _native_available = True
                logger.debug("[HDFisherCompression] Native C++ ops available")
                return True
    except Exception as e:
        logger.debug(f"[HDFisherCompression] Native ops not available: {e}")

    _native_available = False
    return False


# Attempt to load on module import
_load_native_ops()


def hd_fisher_compress(
    fisher_values: tf.Tensor,
    pos_keys: tf.Tensor,
    proj_weights: tf.Tensor,
    hd_dim: int = 4096,
    out_dim: int = 64,
    normalize: bool = True,
    scale: float = 1.0,
    use_native: bool = True,
) -> tf.Tensor:
    """Compress layer-wise Fisher info using holographic bundling.

    Compresses variable-length layer Fisher information into a fixed-size
    HD vector suitable for VQC qubit encoding.

    Args:
        fisher_values: Fisher information per layer [num_layers] or [batch, num_layers].
        pos_keys: Position binding keys [num_layers, hd_dim].
        proj_weights: Projection from HD space [hd_dim, out_dim].
        hd_dim: Hyperdimensional space dimension.
        out_dim: Output dimension for VQC encoding.
        normalize: Whether to normalize the bundle to unit sphere.
        scale: Scaling factor for Fisher values.
        use_native: Whether to use native C++ ops if available.

    Returns:
        Compressed representation [out_dim] or [batch, out_dim].
    """
    if use_native and _native_available:
        return _hd_fisher_compress_native(
            fisher_values, pos_keys, proj_weights, hd_dim, out_dim, normalize, scale
        )
    else:
        return _hd_fisher_compress_tf(
            fisher_values, pos_keys, proj_weights, hd_dim, out_dim, normalize, scale
        )


@tf.custom_gradient
def _hd_fisher_compress_native(
    fisher_values: tf.Tensor,
    pos_keys: tf.Tensor,
    proj_weights: tf.Tensor,
    hd_dim: int,
    out_dim: int,
    normalize: bool,
    scale: float,
) -> tf.Tensor:
    """Native C++ implementation with custom gradient."""

    output = _lib.hd_fisher_compress(
        fisher_values,
        pos_keys,
        proj_weights,
        hd_dim=hd_dim,
        out_dim=out_dim,
        normalize=normalize,
        scale=scale,
    )

    def grad_fn(grad_output):
        grad_fisher, grad_keys, grad_proj = _lib.hd_fisher_compress_grad(
            grad_output,
            fisher_values,
            pos_keys,
            proj_weights,
            hd_dim=hd_dim,
            out_dim=out_dim,
            normalize=normalize,
            scale=scale,
        )
        return grad_fisher, grad_keys, grad_proj, None, None, None, None

    return output, grad_fn


def _hd_fisher_compress_tf(
    fisher_values: tf.Tensor,
    pos_keys: tf.Tensor,
    proj_weights: tf.Tensor,
    hd_dim: int,
    out_dim: int,
    normalize: bool,
    scale: float,
) -> tf.Tensor:
    """Pure TensorFlow fallback implementation.

    Uses FFT-based circular convolution for holographic binding.
    """
    # Handle batched input
    is_batched = len(fisher_values.shape) == 2
    if not is_batched:
        fisher_values = tf.expand_dims(fisher_values, 0)

    batch_size = tf.shape(fisher_values)[0]
    num_layers = tf.shape(fisher_values)[1]

    # Scale Fisher values
    fisher_scaled = fisher_values * scale

    # Initialize bundle accumulator
    bundle = tf.zeros([batch_size, hd_dim], dtype=tf.float32)

    # Holographic binding for each layer
    for layer_idx in tf.range(num_layers):
        # Get Fisher value for this layer: [batch]
        fisher_val = fisher_scaled[:, layer_idx]

        # Scale the position key by Fisher value: [batch, hd_dim]
        key = pos_keys[layer_idx]  # [hd_dim]
        scaled_vec = tf.ones([batch_size, hd_dim]) * tf.expand_dims(fisher_val, -1)

        # Holographic bind via FFT circular convolution
        # bind(a, b) = IFFT(FFT(a) * FFT(b))
        scaled_complex = tf.cast(scaled_vec, tf.complex64)
        key_complex = tf.cast(key, tf.complex64)

        scaled_freq = tf.signal.fft(scaled_complex)
        key_freq = tf.signal.fft(key_complex)

        # Element-wise multiplication in frequency domain
        bound_freq = scaled_freq * tf.expand_dims(key_freq, 0)

        # Inverse FFT
        bound = tf.math.real(tf.signal.ifft(bound_freq))

        # Accumulate into bundle
        bundle = bundle + bound

    # Normalize bundle to unit sphere
    if normalize:
        norm = tf.sqrt(tf.reduce_sum(bundle**2, axis=-1, keepdims=True) + 1e-8)
        bundle = bundle / norm

    # Project to output dimension
    output = tf.matmul(bundle, proj_weights)

    # Remove batch dimension if input was unbatched
    if not is_batched:
        output = tf.squeeze(output, 0)

    return output


def create_position_keys(
    num_layers: int,
    hd_dim: int = 4096,
    seed: int | None = None,
) -> tf.Variable:
    """Create learnable position keys for holographic binding.

    Args:
        num_layers: Maximum number of layers to support.
        hd_dim: Hyperdimensional space dimension.
        seed: Random seed for reproducibility.

    Returns:
        TensorFlow Variable containing position keys [num_layers, hd_dim].
    """
    if seed is not None:
        tf.random.set_seed(seed)

    # Initialize with random bipolar vectors
    keys = tf.random.uniform([num_layers, hd_dim], -1.0, 1.0)

    # Normalize each key to unit norm
    keys = keys / tf.sqrt(tf.reduce_sum(keys**2, axis=-1, keepdims=True) + 1e-8)

    return tf.Variable(keys, trainable=True, name="hd_fisher_pos_keys")


def create_projection_weights(
    hd_dim: int = 4096,
    out_dim: int = 64,
) -> tf.Variable:
    """Create learnable projection weights.

    Args:
        hd_dim: Input dimension (HD space).
        out_dim: Output dimension (VQC encoding).

    Returns:
        TensorFlow Variable containing projection weights [hd_dim, out_dim].
    """
    # Xavier/Glorot initialization
    stddev = tf.sqrt(2.0 / (hd_dim + out_dim))
    weights = tf.random.truncated_normal([hd_dim, out_dim], stddev=stddev)

    return tf.Variable(weights, trainable=True, name="hd_fisher_proj_weights")


def is_native_available() -> bool:
    """Check if native C++ ops are available."""
    return _native_available
