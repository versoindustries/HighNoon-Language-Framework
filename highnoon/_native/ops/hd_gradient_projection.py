# highnoon/_native/ops/hd_gradient_projection.py
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

"""HD Gradient Projection for gradient compression.

Phase 300+: HD Upgrade Integration (hd_upgrade.md Phase 2).

Replaces Tucker decomposition (periodic O(d³) SVD updates) with
HD random projection (fixed projection matrix, no SVD needed).
"""

from __future__ import annotations

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

_native_ops = None
_native_available = False


def _load_native_ops():
    """Load native HD gradient projection ops."""
    global _native_ops, _native_available
    if _native_ops is not None:
        return _native_available

    try:
        from highnoon._native import load_highnoon_core

        _native_ops = load_highnoon_core()
        _native_available = hasattr(_native_ops, "hd_gradient_project") or hasattr(
            _native_ops, "HDGradientProject"
        )
        if _native_available:
            logger.debug("HD Gradient Projection native ops loaded")
    except Exception as e:
        logger.warning(f"Failed to load HD Gradient Projection ops: {e}")
        _native_available = False
    return _native_available


def hd_gradient_projection_available() -> bool:
    """Check if HD gradient projection native ops are available."""
    return _load_native_ops()


class HDGradientCompressor:
    """HD-based gradient compressor.

    Uses Subsampled Randomized Hadamard Transform (SRHT) for
    fast gradient compression without periodic SVD updates.

    Attributes:
        rank: Target compressed rank.
        seed: Random seed for reproducibility.
    """

    def __init__(self, rank: int = 128, seed: int = 314159):
        """Initialize HD gradient compressor.

        Args:
            rank: Target compressed rank.
            seed: Random seed.
        """
        self.rank = rank
        self.seed = seed

        self._projections: dict[str, tuple[tf.Tensor, tf.Tensor]] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}

        self._use_native = hd_gradient_projection_available()

    def _var_key(self, variable: tf.Variable) -> str:
        """Create a unique key for projection state to avoid name collisions."""
        return f"{variable.name}::{id(variable)}"

    def _get_or_create_projection(
        self,
        var_key: str,
        param_size: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Get or create projection parameters for a variable."""
        if var_key in self._projections:
            return self._projections[var_key]

        if self._use_native:
            signs, indices = _native_ops.HDGradientGenerateProjection(
                param_size=param_size,
                rank=self.rank,
                seed=self.seed + hash(var_key) % 10000,
            )
        else:
            # TensorFlow fallback: random Gaussian projection
            padded = 1
            while padded < param_size:
                padded *= 2

            # Random signs
            signs = tf.random.stateless_uniform(
                [padded],
                seed=[self.seed + hash(var_key) % 10000, 0],
                minval=0,
                maxval=2,
                dtype=tf.int32,
            )
            signs = tf.cast(signs * 2 - 1, tf.float32)

            # Random indices
            indices = tf.random.stateless_uniform(
                [self.rank],
                seed=[self.seed + hash(var_key) % 10000 + 1, 0],
                minval=0,
                maxval=padded,
                dtype=tf.int32,
            )

        self._projections[var_key] = (signs, indices)
        return signs, indices

    def compress(
        self,
        gradient: tf.Tensor,
        variable: tf.Variable,
    ) -> tuple[tf.Tensor, str]:
        """Compress gradient to low-rank representation.

        Args:
            gradient: Full gradient tensor.
            variable: Associated weight variable.

        Returns:
            Tuple of (compressed gradient, variable identifier).
        """
        var_key = self._var_key(variable)
        shape = gradient.shape
        self._shapes[var_key] = tuple(shape.as_list())

        flat_grad = tf.reshape(gradient, [-1])
        param_size = flat_grad.shape[0]

        signs, indices = self._get_or_create_projection(var_key, param_size)

        if self._use_native:
            compressed = _native_ops.HDGradientProject(
                gradient=flat_grad,
                signs=signs,
                indices=indices,
            )
        else:
            compressed = self._tf_srht_project(flat_grad, signs, indices)

        return compressed, var_key

    def decompress(
        self,
        compressed: tf.Tensor,
        var_key: str,
    ) -> tf.Tensor:
        """Decompress low-rank gradient back to full tensor.

        Args:
            compressed: Compressed gradient.
            var_name: Variable identifier.

        Returns:
            Full-rank gradient tensor.
        """
        if var_key not in self._projections:
            raise KeyError(f"Variable {var_key} not registered")

        shape = self._shapes[var_key]
        param_size = 1
        for dim in shape:
            param_size *= dim

        signs, indices = self._projections[var_key]

        if self._use_native:
            flat_grad = _native_ops.HDGradientReconstruct(
                compressed=compressed,
                signs=signs,
                indices=indices,
                param_size=param_size,
            )
        else:
            flat_grad = self._tf_srht_reconstruct(compressed, signs, indices, param_size)

        return tf.reshape(flat_grad, shape)

    def _tf_srht_project(
        self,
        x: tf.Tensor,
        signs: tf.Tensor,
        indices: tf.Tensor,
    ) -> tf.Tensor:
        """TensorFlow fallback for SRHT projection."""
        # Pad to power of 2
        dim = x.shape[0]
        padded_dim = signs.shape[0]

        padded = tf.concat([x, tf.zeros([padded_dim - dim])], axis=0)

        # Apply random signs
        signed = padded * signs

        # Walsh-Hadamard via FFT approximation
        # (True WHT not available in TF, use FFT as proxy)
        # Phase 1.5: Use complex128 for quantum precision per GRADIENT_CONNECTIVITY_ROADMAP
        complex_x = tf.cast(signed, tf.complex128)
        fft_x = tf.signal.fft(complex_x)

        # Subsample, cast back to float32
        sampled = tf.gather(tf.cast(tf.math.real(fft_x), tf.float32), indices)
        scale = 1.0 / tf.sqrt(tf.cast(self.rank, tf.float32))

        return sampled * scale

    def _tf_srht_reconstruct(
        self,
        compressed: tf.Tensor,
        signs: tf.Tensor,
        indices: tf.Tensor,
        param_size: int,
    ) -> tf.Tensor:
        """TensorFlow fallback for SRHT reconstruction."""
        padded_dim = signs.shape[0]

        # Expand to padded dimension
        expanded = tf.scatter_nd(
            indices[:, None],
            compressed,
            [padded_dim],
        )

        scale = 1.0 / tf.sqrt(tf.cast(self.rank, tf.float32))
        expanded = expanded * scale

        # Inverse via FFT
        # Phase 1.5: Use complex128 for quantum precision per GRADIENT_CONNECTIVITY_ROADMAP
        complex_x = tf.cast(expanded, tf.complex128)
        ifft_x = tf.signal.ifft(complex_x) * tf.cast(padded_dim, tf.complex128)

        # Apply signs and truncate, cast back to float32
        result = tf.cast(tf.math.real(ifft_x), tf.float32) * signs
        return result[:param_size]

    def get_compression_ratio(self, var_name: str) -> float:
        """Get compression ratio for a variable."""
        if var_name not in self._shapes:
            return 1.0

        param_size = 1
        for dim in self._shapes[var_name]:
            param_size *= dim

        return param_size / self.rank


__all__ = ["HDGradientCompressor", "hd_gradient_projection_available"]
