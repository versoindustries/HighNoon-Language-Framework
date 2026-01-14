# highnoon/qsg/hd_compression.py
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

"""Phase A5: HD Compression Pipeline for QSG.

Implements Hyperdimensional (HD) vocabulary encoding to achieve memory reduction
via holographic bundling. This allows large vocabularies (e.g. 256k) to be
represented by a much smaller set of physical embeddings (e.g. 10k) combined
with unitary keys.

Key Components:
    - HDVocabularyEncoder: Holographic dual-path vocabulary embedding layer
    - TF-native holographic operations (bind/unbind) via FFT

The encoder maps a token ID `t` to a pair `(base_idx, key_idx)`:
    Embedding(t) = Bind( BaseEmbedding(base_idx), Key(key_idx) )

This provides a "Virtual Vocabulary" of size Base * Keys.
"""

from __future__ import annotations

import logging
import math

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.config import HD_ACTIVE_VOCAB_SIZE, HD_EMBEDDING_DIM, HQE_CTQW_STEPS

logger = logging.getLogger(__name__)


# =============================================================================
# TF-Native Holographic Operations
# =============================================================================


def holographic_bind(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Holographic bind: x ⊛ y = IFFT(FFT(x) * FFT(y)).

    Uses circular convolution implemented via FFT.
    Input and output are real-valued.

    Phase 1.5: Uses complex128 for quantum precision per GRADIENT_CONNECTIVITY_ROADMAP.

    Args:
        x: First vector [..., dim].
        y: Second vector [..., dim].

    Returns:
        Bound vector [..., dim].
    """
    # Phase 1.5: Cast to complex128 for quantum precision
    x_c = tf.cast(x, tf.complex128)
    y_c = tf.cast(y, tf.complex128)

    # FFT
    x_fft = tf.signal.fft(x_c)
    y_fft = tf.signal.fft(y_c)

    # Element-wise multiply in frequency domain
    bound_fft = x_fft * y_fft

    # IFFT and take real part, cast back to float32 for downstream compatibility
    bound = tf.cast(tf.math.real(tf.signal.ifft(bound_fft)), tf.float32)

    return bound


def holographic_unbind(bundle: tf.Tensor, key: tf.Tensor, epsilon: float = 1e-8) -> tf.Tensor:
    """Holographic unbind: bundle ⊘ key.

    Inverse of binding. Since keys are unitary (in freq domain),
    Unbind(B, K) ≈ Bind(B, Conj(K)).

    Phase 1.5: Uses complex128 for quantum precision per GRADIENT_CONNECTIVITY_ROADMAP.

    Args:
        bundle: Bundled vector [..., dim].
        key: Key vector [..., dim].
        epsilon: Numerical stability.

    Returns:
        Unbound vector [..., dim].
    """
    # Phase 1.5: Cast to complex128 for quantum precision
    b_c = tf.cast(bundle, tf.complex128)
    k_c = tf.cast(key, tf.complex128)

    b_fft = tf.signal.fft(b_c)
    k_fft = tf.signal.fft(k_c)

    # Complex division: B / K = B * conj(K) / |K|^2
    # Ensure denominator is stable
    denom = tf.math.abs(k_fft) ** 2 + epsilon
    result_fft = b_fft * tf.math.conj(k_fft) / tf.cast(denom, tf.complex128)

    # Cast back to float32 for downstream compatibility
    return tf.cast(tf.math.real(tf.signal.ifft(result_fft)), tf.float32)


def create_unitary_keys(num_keys: int, dim: int, seed: int = 42) -> tf.Tensor:
    """Create unitary keys for holographic binding.

    Keys have flat magnitude spectrum in frequency domain to ensure
    length preservation and invertibility.

    Phase 1.5: Uses complex128 for quantum precision per GRADIENT_CONNECTIVITY_ROADMAP.

    Args:
        num_keys: Number of keys to generate.
        dim: Vector dimension.
        seed: Random seed.

    Returns:
        Keys tensor [num_keys, dim].
    """
    # Generate random phases
    rng = tf.random.Generator.from_seed(seed)
    # Start with Gaussian noise
    raw = rng.normal(shape=[num_keys, dim])

    # Phase 1.5: FFT with complex128 precision
    raw_c = tf.cast(raw, tf.complex128)
    raw_fft = tf.signal.fft(raw_c)

    # Normalize magnitude to 1.0 (Unitary in freq domain)
    magnitude = tf.math.abs(raw_fft) + 1e-8
    phases = raw_fft / tf.cast(magnitude, tf.complex128)

    # IFFT back to time domain, cast to float32 for storage
    keys = tf.cast(tf.math.real(tf.signal.ifft(phases)), tf.float32)

    # Scale to maintain L2 norm of 1.0 roughly?
    # Actually for circular convolution, preserving 1.0 magnitude in freq
    # means E[norm] is preserved.
    return keys


# =============================================================================
# HD Vocabulary Encoder
# =============================================================================


class HDVocabularyEncoder(layers.Layer):
    """Holographic Dual-Path Vocabulary Encoder.

    Compresses vocabulary layer by representing tokens as a combination of
    a 'base' embedding and a 'position' key.

    Virtual Vocab Size = Active Size * Num Keys

    Example:
        Active = 10,000, Keys = 26
        Virtual = 260,000 (Covering 256k vocab)

    Parameters: 260k * 512 = 133M floats -> (10k + 26) * 512 = 5M floats
    Compression: ~25x reduction.
    """

    def __init__(
        self,
        vocab_size: int,
        active_vocab_size: int = HD_ACTIVE_VOCAB_SIZE,  # e.g. 10000
        embedding_dim: int = HD_EMBEDDING_DIM,  # e.g. 4096 (or 512 for test)
        ctqw_steps: int = HQE_CTQW_STEPS,
        name: str = "hd_vocab_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.active_vocab_size = active_vocab_size
        self.embedding_dim = embedding_dim
        self.ctqw_steps = ctqw_steps

        # Calculate number of keys needed
        # num_keys = ceil(vocab_size / active_vocab_size)
        self.num_keys = math.ceil(vocab_size / active_vocab_size)

        # Base embeddings (Learnable)
        self.base_embeddings = self.add_weight(
            name="base_embeddings",
            shape=[active_vocab_size, embedding_dim],
            initializer="glorot_uniform",
            trainable=True,
        )

        # Unitary Keys (Fixed or Learnable? Usually Fixed for stability)
        # We'll use fixed keys for true holographic properties
        # They act as the "address" space.
        initial_keys = create_unitary_keys(self.num_keys, embedding_dim)
        self.keys = tf.Variable(
            initial_value=initial_keys,
            name="holographic_keys",
            trainable=False,  # Fixed basis
            dtype=tf.float32,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Embed token IDs using holographic encoding.

        Args:
            inputs: Token IDs [batch, seq].

        Returns:
            Embeddings [batch, seq, dim].
        """
        # Map token_id to (base_idx, key_idx)
        # Base = token % active
        # Key = token // active

        base_indices = inputs % self.active_vocab_size
        key_indices = inputs // self.active_vocab_size

        # Gather
        bases = tf.gather(self.base_embeddings, base_indices)  # [B, L, D]
        keys = tf.gather(self.keys, key_indices)  # [B, L, D]

        # Bind: Embedding = Base ⊛ Key
        embeddings = holographic_bind(bases, keys)

        # Optional: Apply CTQW for spreading/mixing
        # (Not implemented in this basics step, but reserved in config)

        return embeddings

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "active_vocab_size": self.active_vocab_size,
                "embedding_dim": self.embedding_dim,
                "ctqw_steps": self.ctqw_steps,
            }
        )
        return config
