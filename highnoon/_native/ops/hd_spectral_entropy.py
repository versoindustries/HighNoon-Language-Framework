# highnoon/_native/ops/hd_spectral_entropy.py
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

"""HD Spectral Entropy for QULS.

Phase 300+: HD Upgrade Integration (hd_upgrade.md Phase 2).

FFT-based spectral entropy computation for QULS, replacing O(d²)
power iteration with O(d log d) FFT-based spectral analysis.
"""

from __future__ import annotations

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

_native_ops = None
_native_available = False


def _load_native_ops():
    """Load native HD spectral entropy ops."""
    global _native_ops, _native_available
    if _native_ops is not None:
        return _native_available

    try:
        from highnoon._native.ops.lib_loader import load_highnoon_core

        _native_ops = load_highnoon_core()
        _native_available = hasattr(_native_ops, "hd_spectral_entropy") or hasattr(
            _native_ops, "HDSpectralEntropy"
        )
        if _native_available:
            logger.debug("HD Spectral Entropy native ops loaded")
    except Exception as e:
        logger.warning(f"Failed to load HD Spectral Entropy native ops: {e}")
        _native_available = False
    return _native_available


def hd_spectral_entropy_available() -> bool:
    """Check if HD spectral entropy native ops are available."""
    return _load_native_ops()


def hd_spectral_entropy(
    hidden_states: tf.Tensor,
    epsilon: float = 1e-8,
) -> tf.Tensor:
    """Compute HD spectral entropy from hidden states.

    Uses FFT-based O(d log d) spectral analysis instead of O(d²)
    eigenvalue computation.

    Args:
        hidden_states: Input tensor [batch, dim] or [batch, seq, dim].
        epsilon: Numerical stability constant.

    Returns:
        Spectral entropy values [batch].
    """
    if _load_native_ops():
        return _native_ops.HDSpectralEntropy(hidden_states, epsilon=epsilon)

    # TensorFlow fallback implementation
    return _tf_spectral_entropy(hidden_states, epsilon)


def _tf_spectral_entropy(
    hidden_states: tf.Tensor,
    epsilon: float = 1e-8,
) -> tf.Tensor:
    """TensorFlow fallback for spectral entropy computation."""
    rank = len(hidden_states.shape)

    if rank == 3:
        # [batch, seq, dim] -> average over seq
        batch_size = tf.shape(hidden_states)[0]
        seq_len = tf.shape(hidden_states)[1]
        dim = hidden_states.shape[2] or tf.shape(hidden_states)[2]

        # Flatten to [batch * seq, dim]
        flat = tf.reshape(hidden_states, [-1, dim])
        entropies = _compute_single_entropy(flat, epsilon)
        # Reshape and average
        entropies = tf.reshape(entropies, [batch_size, seq_len])
        return tf.reduce_mean(entropies, axis=1)
    else:
        return _compute_single_entropy(hidden_states, epsilon)


def _compute_single_entropy(x: tf.Tensor, epsilon: float) -> tf.Tensor:
    """Compute spectral entropy for 2D tensor [batch, dim]."""
    # Cast to complex for FFT
    x_complex = tf.cast(x, tf.complex64)

    # FFT along last dimension
    x_fft = tf.signal.fft(x_complex)

    # Power spectrum: |FFT(x)|²
    power = tf.abs(x_fft) ** 2

    # Normalize to probability
    total = tf.reduce_sum(power, axis=-1, keepdims=True) + epsilon
    p = power / total

    # Entropy: -Σ p log p
    log_p = tf.math.log(p + epsilon)
    entropy = -tf.reduce_sum(p * log_p, axis=-1)

    # Normalize by max entropy
    dim = tf.cast(tf.shape(x)[-1], tf.float32)
    max_entropy = tf.math.log(dim)

    return entropy / (max_entropy + epsilon)


def hd_spectral_flatness(
    hidden_states: tf.Tensor,
    epsilon: float = 1e-8,
) -> tf.Tensor:
    """Compute spectral flatness (Wiener entropy).

    Spectral flatness = geometric_mean(power) / arithmetic_mean(power).
    Ranges from 0 (pure tone) to 1 (white noise).

    Args:
        hidden_states: Input tensor [batch, dim].
        epsilon: Numerical stability constant.

    Returns:
        Spectral flatness values [batch].
    """
    if _load_native_ops():
        return _native_ops.HDSpectralFlatness(hidden_states, epsilon=epsilon)

    # TensorFlow fallback
    x_complex = tf.cast(hidden_states, tf.complex64)
    x_fft = tf.signal.fft(x_complex)
    power = tf.abs(x_fft) ** 2 + epsilon

    # Geometric mean: exp(mean(log(power)))
    log_power = tf.math.log(power)
    geometric_mean = tf.exp(tf.reduce_mean(log_power, axis=-1))

    # Arithmetic mean
    arithmetic_mean = tf.reduce_mean(power, axis=-1)

    return geometric_mean / arithmetic_mean


@tf.custom_gradient
def hd_spectral_entropy_with_grad(
    hidden_states: tf.Tensor,
    epsilon: float = 1e-8,
) -> tf.Tensor:
    """Spectral entropy with gradient support.

    Args:
        hidden_states: Input tensor [batch, dim].
        epsilon: Numerical stability constant.

    Returns:
        Spectral entropy values [batch].
    """
    entropy = hd_spectral_entropy(hidden_states, epsilon)

    def grad(upstream):
        if _load_native_ops():
            return _native_ops.HDSpectralEntropyGrad(hidden_states, upstream, epsilon=epsilon), None

        # TensorFlow gradient via automatic differentiation
        with tf.GradientTape() as tape:
            tape.watch(hidden_states)
            e = _tf_spectral_entropy(hidden_states, epsilon)
        return tape.gradient(e, hidden_states) * upstream[:, None], None

    return entropy, grad


__all__ = [
    "hd_spectral_entropy",
    "hd_spectral_entropy_with_grad",
    "hd_spectral_flatness",
    "hd_spectral_entropy_available",
]
