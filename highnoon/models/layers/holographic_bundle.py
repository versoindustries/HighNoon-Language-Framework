# highnoon/models/layers/holographic_bundle.py
# Copyright 2025-2026 Verso Industries (Author: Michael B. Zimmerman)
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

"""Holographic Token Bundling Layer for Memory-Efficient HD Computing.

Phase 2 of Memory Architecture Roadmap: HD Token Bundling.

This module compresses sequences of tokens into bundled HD vectors using
FFT-domain circular correlation. Instead of storing per-token HD embeddings,
we bundle groups of tokens into single holographic representations.

Memory Impact (32K seq, batch=4, hd_dim=4096):
    - Per-token HD: 32K × 4096 × 4 = 537 MB per sample
    - Bundled (128 tokens): 256 × 4096 × 4 = 4.2 MB per sample
    - Savings: 128×

Key Features:
    - FFT-domain circular correlation for binding
    - Position-aware bundling with decay weighting
    - Configurable bundle size (QAHPO tunable: 32-256)
    - Optional overlap for smooth boundaries
    - Unbundling for retrieval (inverse correlation)

References:
    - Plate (2003): Holographic Reduced Representations
    - Kanerva (2009): Hyperdimensional Computing
    - MEMORY_ANALYSIS_sweep_15c266c6.md: Phase 2 specification
"""

from __future__ import annotations

import logging
import math
from typing import Any

import tensorflow as tf

from highnoon import config

logger = logging.getLogger(__name__)


class HolographicBundle(tf.keras.layers.Layer):
    """Compress token sequences into bundled HD vectors via circular correlation.

    Bundles groups of `bundle_size` tokens into single holographic vectors.
    Uses FFT-based circular correlation for position-aware binding, enabling
    approximate retrieval via unbinding.

    Memory Formula:
        Output shape: [batch, seq_len // bundle_size, hd_dim]
        Memory: batch × (seq_len / bundle_size) × hd_dim × 4 bytes

    Attributes:
        hd_dim: Hyperdimensional space dimension.
        bundle_size: Number of tokens per bundle (default from config.HD_BUNDLE_SIZE).
        overlap: Number of overlapping tokens between bundles.
        use_position_weights: Apply position-based decay within bundles.
        decay_rate: Exponential decay rate for position weighting.
    """

    def __init__(
        self,
        hd_dim: int = 4096,
        bundle_size: int | None = None,
        overlap: int | None = None,
        use_position_weights: bool = True,
        decay_rate: float = 0.99,
        **kwargs: Any,
    ) -> None:
        """Initialize HolographicBundle layer.

        Args:
            hd_dim: HD vector dimension.
            bundle_size: Tokens per bundle (default: config.HD_BUNDLE_SIZE).
            overlap: Overlapping tokens (default: config.HD_BUNDLE_OVERLAP).
            use_position_weights: Apply decay weighting by position.
            decay_rate: Decay factor for position weights.
        """
        super().__init__(**kwargs)

        self.hd_dim = hd_dim
        self.bundle_size = bundle_size or getattr(config, "HD_BUNDLE_SIZE", 128)
        self.overlap = overlap or getattr(config, "HD_BUNDLE_OVERLAP", 0)
        self.use_position_weights = use_position_weights
        self.decay_rate = decay_rate

        if self.overlap >= self.bundle_size:
            raise ValueError(f"overlap ({self.overlap}) must be < bundle_size ({self.bundle_size})")

        # Effective stride between bundles
        self.stride = self.bundle_size - self.overlap

        logger.info(
            "[HolographicBundle] hd_dim=%d, bundle_size=%d, overlap=%d, stride=%d",
            self.hd_dim,
            self.bundle_size,
            self.overlap,
            self.stride,
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer weights.

        Creates position-binding vectors for within-bundle composition.
        These are cyclic shifts of a base vector to enable circular correlation.
        """
        # Base position vector for within-bundle binding
        # Uses orthogonal init for quasi-orthogonal position vectors
        self.bundle_position_base = self.add_weight(
            name="bundle_position_base",
            shape=(self.hd_dim,),
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1.0 / math.sqrt(self.hd_dim)
            ),
            trainable=True,
        )

        # Optional learnable decay weights per bundle position
        if self.use_position_weights:
            # Initialize with exponential decay: [1, decay, decay^2, ...]
            init_weights = [self.decay_rate**i for i in range(self.bundle_size)]
            self.position_weights = self.add_weight(
                name="position_weights",
                shape=(self.bundle_size,),
                initializer=tf.constant_initializer(init_weights),
                trainable=True,
            )
        else:
            self.position_weights = None

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool | None = None,
    ) -> tf.Tensor:
        """Bundle input HD vectors into compressed representations.

        Args:
            inputs: HD vectors [batch, seq_len, hd_dim].
            training: Training mode flag.

        Returns:
            Bundled HD vectors [batch, num_bundles, hd_dim].
        """
        tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Compute number of bundles
        # num_bundles = ceil((seq_len - bundle_size) / stride) + 1
        num_bundles = (seq_len - self.bundle_size) // self.stride + 1
        num_bundles = tf.maximum(num_bundles, 1)

        # Generate within-bundle position vectors via cyclic permutation
        # pos_vectors[i] = roll(base, shift=i) for i in [0, bundle_size)
        shifts = tf.range(self.bundle_size)
        # Vectorized cyclic shift
        indices = tf.range(self.hd_dim)[None, :] - shifts[:, None]
        indices = tf.math.floormod(indices, self.hd_dim)
        pos_vectors = tf.gather(self.bundle_position_base, indices)  # [bundle_size, hd_dim]

        # Apply position weights if enabled
        if self.position_weights is not None:
            weights = tf.nn.softmax(self.position_weights)  # Normalize
            pos_vectors = pos_vectors * weights[:, None]

        # Process bundles using strided extraction
        def bundle_tokens(bundle_idx: tf.Tensor) -> tf.Tensor:
            """Bundle tokens for a single bundle index."""
            start_idx = bundle_idx * self.stride
            end_idx = start_idx + self.bundle_size

            # Extract tokens for this bundle: [batch, bundle_size, hd_dim]
            bundle_tokens_slice = inputs[:, start_idx:end_idx, :]

            # Handle partial bundles at sequence end
            actual_len = tf.shape(bundle_tokens_slice)[1]
            pad_len = self.bundle_size - actual_len

            # Pad if necessary (for partial bundles)
            bundle_tokens_padded = tf.cond(
                pad_len > 0,
                lambda: tf.pad(
                    bundle_tokens_slice,
                    [[0, 0], [0, pad_len], [0, 0]],
                    mode="CONSTANT",
                ),
                lambda: bundle_tokens_slice,
            )

            # Bind tokens with positions via FFT circular correlation
            # bundle = sum_i (token[i] ⊛ pos[i])
            bound = self._circular_correlation_batch(
                bundle_tokens_padded, pos_vectors
            )  # [batch, bundle_size, hd_dim]

            # Sum to create bundle: [batch, hd_dim]
            bundle = tf.reduce_sum(bound, axis=1)

            # Normalize for stability (add epsilon to prevent NaN gradients)
            bundle = tf.nn.l2_normalize(bundle, axis=-1, epsilon=1e-8)

            return bundle

        # Map over bundle indices
        bundles = tf.map_fn(
            bundle_tokens,
            tf.range(num_bundles),
            fn_output_signature=tf.TensorSpec([None, self.hd_dim], tf.float32),
        )  # [num_bundles, batch, hd_dim]

        # Transpose to [batch, num_bundles, hd_dim]
        bundles = tf.transpose(bundles, [1, 0, 2])

        return bundles

    def _circular_correlation_batch(
        self,
        tokens: tf.Tensor,
        positions: tf.Tensor,
    ) -> tf.Tensor:
        """Circular correlation for batch binding.

        Computes: output[b, i] = IFFT(FFT(tokens[b,i]) * conj(FFT(positions[i])))

        Args:
            tokens: Token HD vectors [batch, bundle_size, hd_dim].
            positions: Position HD vectors [bundle_size, hd_dim].

        Returns:
            Bound vectors [batch, bundle_size, hd_dim].
        """
        # Expand positions for batch broadcasting
        positions = tf.expand_dims(positions, 0)  # [1, bundle_size, hd_dim]

        # Phase 1.5: Cast to complex128 for quantum precision per GRADIENT_CONNECTIVITY_ROADMAP
        tokens_c = tf.cast(tokens, tf.complex128)
        pos_c = tf.cast(positions, tf.complex128)

        # FFT along HD dimension
        tokens_fft = tf.signal.fft(tokens_c)
        pos_fft = tf.signal.fft(pos_c)

        # Correlation = element-wise product with conjugate
        # Uses conjugate for correlation (vs. convolution)
        correlated_fft = tokens_fft * tf.math.conj(pos_fft)

        # Inverse FFT, cast back to float32 for downstream compatibility
        result = tf.signal.ifft(correlated_fft)

        return tf.cast(tf.math.real(result), tf.float32)

    def unbundle(
        self,
        bundles: tf.Tensor,
        query_position: int,
    ) -> tf.Tensor:
        """Retrieve approximate token representation from bundle.

        Uses inverse circular correlation (binding with conjugate) to
        extract token at specified position within bundle.

        Args:
            bundles: Bundled HD vectors [batch, num_bundles, hd_dim].
            query_position: Position within bundle to retrieve (0 to bundle_size-1).

        Returns:
            Retrieved HD vectors [batch, num_bundles, hd_dim].
        """
        if query_position < 0 or query_position >= self.bundle_size:
            raise ValueError(
                f"query_position must be in [0, {self.bundle_size}), got {query_position}"
            )

        # Get position vector for query position
        shift = query_position
        indices = tf.range(self.hd_dim) - shift
        indices = tf.math.floormod(indices, self.hd_dim)
        query_pos_vector = tf.gather(self.bundle_position_base, indices)  # [hd_dim]

        # Phase 1.5: Unbind via circular convolution with complex128 precision
        bundles_c = tf.cast(bundles, tf.complex128)
        pos_c = tf.cast(query_pos_vector, tf.complex128)

        bundles_fft = tf.signal.fft(bundles_c)
        pos_fft = tf.signal.fft(pos_c)

        # Convolution (not conjugate) for unbinding
        unbound_fft = bundles_fft * pos_fft

        result = tf.signal.ifft(unbound_fft)

        return tf.cast(tf.math.real(result), tf.float32)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape for Keras.

        Args:
            input_shape: Input shape [batch, seq_len, hd_dim].

        Returns:
            Output shape [batch, num_bundles, hd_dim].
        """
        if input_shape[1] is not None:
            num_bundles = max(1, (input_shape[1] - self.bundle_size) // self.stride + 1)
        else:
            num_bundles = None

        return tf.TensorShape([input_shape[0], num_bundles, self.hd_dim])

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization."""
        base_config = super().get_config()
        base_config.update(
            {
                "hd_dim": self.hd_dim,
                "bundle_size": self.bundle_size,
                "overlap": self.overlap,
                "use_position_weights": self.use_position_weights,
                "decay_rate": self.decay_rate,
            }
        )
        return base_config


class HolographicUnbundle(tf.keras.layers.Layer):
    """Expand bundled HD vectors back to per-token representations.

    Inverse of HolographicBundle: reconstructs approximate per-token
    HD vectors from bundled representations using learned expansion.

    This is useful for:
        - Attention mechanisms that need per-token queries
        - Output layers that produce per-token predictions
        - Debugging/analysis of bundle contents

    Note: Expansion is approximate due to information compression.
    """

    def __init__(
        self,
        hd_dim: int = 4096,
        bundle_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize HolographicUnbundle layer.

        Args:
            hd_dim: HD vector dimension.
            bundle_size: Tokens per bundle (must match bundling layer).
        """
        super().__init__(**kwargs)
        self.hd_dim = hd_dim
        self.bundle_size = bundle_size or getattr(config, "HD_BUNDLE_SIZE", 128)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer weights for unbundling."""
        # Learned expansion weights: map bundle -> per-token
        self.expand_weights = self.add_weight(
            name="expand_weights",
            shape=(self.bundle_size, self.hd_dim, self.hd_dim),
            initializer=tf.keras.initializers.Orthogonal(),
            trainable=True,
        )

        super().build(input_shape)

    def call(
        self,
        bundles: tf.Tensor,
        training: bool | None = None,
    ) -> tf.Tensor:
        """Expand bundles to per-token representations.

        Args:
            bundles: Bundled HD vectors [batch, num_bundles, hd_dim].
            training: Training mode flag.

        Returns:
            Expanded HD vectors [batch, num_bundles * bundle_size, hd_dim].
        """
        batch_size = tf.shape(bundles)[0]
        num_bundles = tf.shape(bundles)[1]

        # Expand each bundle to bundle_size tokens
        # output[b, n, i, :] = bundles[b, n, :] @ expand_weights[i]
        expanded = tf.einsum("bnh,ihj->bnij", bundles, self.expand_weights)
        # Shape: [batch, num_bundles, bundle_size, hd_dim]

        # Reshape to sequential: [batch, num_bundles * bundle_size, hd_dim]
        output = tf.reshape(
            expanded,
            [batch_size, num_bundles * self.bundle_size, self.hd_dim],
        )

        return output

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape."""
        if input_shape[1] is not None:
            seq_len = input_shape[1] * self.bundle_size
        else:
            seq_len = None
        return tf.TensorShape([input_shape[0], seq_len, self.hd_dim])

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        base_config = super().get_config()
        base_config.update(
            {
                "hd_dim": self.hd_dim,
                "bundle_size": self.bundle_size,
            }
        )
        return base_config


__all__ = ["HolographicBundle", "HolographicUnbundle"]
