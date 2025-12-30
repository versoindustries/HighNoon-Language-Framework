# highnoon/_native/ops/hd_holographic_attention.py
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

"""HD Holographic Attention ops.

Phase 300+: HD Upgrade Integration (hd_upgrade.md Phase 3).

FFT-based holographic attention with O(d log d) similarity computation.
Also includes HD KV cache compression for inference.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import tensorflow as tf

logger = logging.getLogger(__name__)

_native_ops = None
_native_available = False


def _load_native_ops():
    """Load native holographic attention ops."""
    global _native_ops, _native_available
    if _native_ops is not None:
        return _native_available

    try:
        from highnoon._native.ops.lib_loader import get_highnoon_core_path

        lib_path = get_highnoon_core_path()
        _native_ops = tf.load_op_library(lib_path)
        _native_available = hasattr(_native_ops, "holographic_attention_scores") or hasattr(
            _native_ops, "HolographicAttentionScores"
        )
        if _native_available:
            logger.debug("Holographic Attention native ops loaded from %s", lib_path)
    except Exception as e:
        logger.warning(f"Failed to load Holographic Attention ops: {e}")
        _native_available = False
    return _native_available


def hd_holographic_attention_available() -> bool:
    """Check if holographic attention native ops are available."""
    return _load_native_ops()


def holographic_bind(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Holographic bind: x ⊛ y = IFFT(FFT(x) * FFT(y)).

    Binds two vectors in a position-preserving way.

    Args:
        x: First tensor [..., dim].
        y: Second tensor [..., dim].

    Returns:
        Bound tensor [..., dim].
    """
    if _load_native_ops():
        return _native_ops.HolographicBind(x, y)

    # TensorFlow fallback
    x_complex = tf.cast(x, tf.complex64)
    y_complex = tf.cast(y, tf.complex64)

    x_fft = tf.signal.fft(x_complex)
    y_fft = tf.signal.fft(y_complex)

    bound_fft = x_fft * y_fft
    bound = tf.signal.ifft(bound_fft)

    return tf.math.real(bound)


def holographic_unbind(bundle: tf.Tensor, key: tf.Tensor, epsilon: float = 1e-8) -> tf.Tensor:
    """Holographic unbind: retrieval via complex division.

    unbind(bundle, key) = IFFT(FFT(bundle) / FFT(key))

    Args:
        bundle: Bundled tensor [..., dim].
        key: Key to unbind with [..., dim].
        epsilon: Numerical stability.

    Returns:
        Unbound tensor [..., dim].
    """
    b_complex = tf.cast(bundle, tf.complex64)
    k_complex = tf.cast(key, tf.complex64)

    b_fft = tf.signal.fft(b_complex)
    k_fft = tf.signal.fft(k_complex)

    # Complex division with stability
    denom = tf.abs(k_fft) ** 2 + epsilon
    result_fft = b_fft * tf.math.conj(k_fft) / tf.cast(denom, tf.complex64)

    result = tf.signal.ifft(result_fft)
    return tf.math.real(result)


def holographic_similarity(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Holographic similarity via circular correlation.

    Args:
        x: First tensor [..., dim].
        y: Second tensor [..., dim].

    Returns:
        Similarity scores [...].
    """
    if _load_native_ops():
        return _native_ops.HolographicSimilarity(x, y)

    # TensorFlow fallback
    x_complex = tf.cast(x, tf.complex64)
    y_complex = tf.cast(y, tf.complex64)

    x_fft = tf.signal.fft(x_complex)
    y_fft = tf.signal.fft(y_complex)

    # Correlation = x * conj(y) in frequency domain
    corr_fft = x_fft * tf.math.conj(y_fft)
    corr = tf.signal.ifft(corr_fft)

    # Return max correlation (position 0 for aligned vectors)
    return tf.math.real(corr[..., 0])


def holographic_attention_scores(
    queries: tf.Tensor,
    keys: tf.Tensor,
    temperature: float = 1.0,
) -> tf.Tensor:
    """Compute holographic attention scores.

    Args:
        queries: Query tensor [batch, heads, seq_q, head_dim].
        keys: Key tensor [batch, heads, seq_k, head_dim].
        temperature: Softmax temperature.

    Returns:
        Attention scores [batch, heads, seq_q, seq_k].
    """
    if _load_native_ops():
        return _native_ops.HolographicAttentionScores(queries, keys, temperature=temperature)

    # TensorFlow fallback - compute pairwise holographic similarity
    # This is O(n² × d log d) instead of native O(n × d log d + n²)
    batch = tf.shape(queries)[0]
    heads = tf.shape(queries)[1]
    seq_q = tf.shape(queries)[2]
    seq_k = tf.shape(keys)[3]
    head_dim = queries.shape[-1]

    q_complex = tf.cast(queries, tf.complex64)
    k_complex = tf.cast(keys, tf.complex64)

    q_fft = tf.signal.fft(q_complex)  # [B, H, Sq, D]
    k_fft = tf.signal.fft(k_complex)  # [B, H, Sk, D]

    # Compute correlation: sum over dim of q_fft * conj(k_fft)
    # For position-0 correlation, this is just the sum
    k_fft_conj = tf.math.conj(k_fft)

    # [B, H, Sq, D] @ [B, H, D, Sk] -> [B, H, Sq, Sk]
    scores = tf.einsum(
        "bhqd,bhkd->bhqk",
        tf.math.real(q_fft * k_fft_conj[:, :, None, :, :]),
        tf.ones_like(keys)[..., 0:1],
    )

    # Simplified: use standard attention with FFT features
    q_feat = tf.math.real(q_fft)
    k_feat = tf.math.real(k_fft)
    scores = tf.einsum("bhqd,bhkd->bhqk", q_feat, k_feat)

    return scores / (temperature * tf.sqrt(tf.cast(head_dim, tf.float32)))


def generate_position_keys(
    max_seq: int,
    head_dim: int,
    base_freq: float = 10000.0,
) -> tf.Tensor:
    """Generate Floquet-inspired position keys.

    Args:
        max_seq: Maximum sequence length.
        head_dim: Head dimension.
        base_freq: Base frequency.

    Returns:
        Position keys [max_seq, head_dim].
    """
    if _load_native_ops():
        return _native_ops.GeneratePositionKeys(
            max_seq=max_seq, head_dim=head_dim, base_freq=base_freq
        )

    # TensorFlow fallback
    positions = tf.range(max_seq, dtype=tf.float32)
    dims = tf.range(head_dim, dtype=tf.float32)

    freqs = 1.0 / tf.pow(base_freq, 2.0 * (dims // 2) / tf.cast(head_dim, tf.float32))

    angles = positions[:, None] * freqs[None, :]

    # Interleave sin and cos
    pos_keys = tf.where(
        tf.range(head_dim) % 2 == 0,
        tf.sin(angles),
        tf.cos(angles),
    )

    return pos_keys


class HDKVCache:
    """HD-compressed KV cache for inference.

    Compresses K/V via holographic bundling for 8-16x memory reduction.
    """

    def __init__(
        self,
        compression_ratio: int = 8,
        head_dim: int = 64,
        max_seq: int = 8192,
    ):
        """Initialize HD KV cache.

        Args:
            compression_ratio: Tokens per HD bundle.
            head_dim: Head dimension.
            max_seq: Maximum sequence length.
        """
        self.compression_ratio = compression_ratio
        self.head_dim = head_dim
        self.max_seq = max_seq

        self._pos_keys = generate_position_keys(max_seq, head_dim)
        self._cache: tf.Variable | None = None
        self._current_len = 0

    def reset(self, batch_size: int, num_heads: int) -> None:
        """Reset cache for new sequence."""
        num_bundles = self.max_seq // self.compression_ratio + 1
        self._cache = tf.Variable(
            tf.zeros([batch_size, num_heads, num_bundles, self.head_dim]),
            trainable=False,
        )
        self._current_len = 0

    def append(self, kv: tf.Tensor) -> None:
        """Append new K/V to cache.

        Args:
            kv: New K/V vector [batch, heads, head_dim].
        """
        pos = self._current_len
        pos_key = self._pos_keys[pos]

        # Bind with position key
        bound = holographic_bind(kv, pos_key)

        # Add to appropriate bundle
        bundle_idx = pos // self.compression_ratio

        current = self._cache[:, :, bundle_idx, :]
        self._cache[:, :, bundle_idx, :].assign(current + bound)

        self._current_len += 1

    def get(self, positions: tf.Tensor | None = None) -> tf.Tensor:
        """Retrieve K/V for specified positions.

        Args:
            positions: Positions to retrieve [num_positions], or None for all.

        Returns:
            Retrieved K/V [batch, heads, num_positions, head_dim].
        """
        if positions is None:
            positions = tf.range(self._current_len)

        results = []
        for pos in positions:
            bundle_idx = pos // self.compression_ratio
            bundle = self._cache[:, :, bundle_idx, :]
            pos_key = self._pos_keys[pos]

            retrieved = holographic_unbind(bundle, pos_key)
            results.append(retrieved)

        return tf.stack(results, axis=2)


__all__ = [
    "holographic_bind",
    "holographic_unbind",
    "holographic_similarity",
    "holographic_attention_scores",
    "generate_position_keys",
    "HDKVCache",
    "hd_holographic_attention_available",
]
