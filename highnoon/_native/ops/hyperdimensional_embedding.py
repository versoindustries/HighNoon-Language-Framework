# highnoon/_native/ops/hyperdimensional_embedding.py
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

"""Python wrapper for the Hyperdimensional Embedding C++ operations.

This module provides TensorFlow ops for hyperdimensional embedding using
holographic bundling and CTQW spreading. Falls back to pure TensorFlow
implementation if C++ ops are unavailable.

Phase 48: Hyperdimensional Quantum Embeddings (HQE)
- FFT-based circular convolution for holographic binding
- CTQW semantic spreading for diffusion
- O(N log N) complexity for binding
- 50-100x compression via holographic representation
"""

from __future__ import annotations

import logging
from typing import Tuple

import tensorflow as tf

logger = logging.getLogger(__name__)

# Attempt to load C++ ops
_HDE_NATIVE_AVAILABLE = False
_hde_holographic_bundle = None
_hde_ctqw_spread = None

try:
    from highnoon._native import get_op

    _hde_module = get_op("hyperdimensional_embedding")
    if _hde_module is not None:
        _hde_holographic_bundle = _hde_module.holographic_bundle
        _hde_ctqw_spread = _hde_module.ctqw_spread
        _HDE_NATIVE_AVAILABLE = True
        logger.info("Loaded hyperdimensional_embedding C++ ops")
except ImportError:
    logger.debug("hyperdimensional_embedding C++ ops not available, using TF fallback")
except Exception as e:
    logger.warning("Failed to load hyperdimensional_embedding C++ ops: %s", e)


def hyperdimensional_embedding_available() -> bool:
    """Check if native HDE ops are available."""
    return _HDE_NATIVE_AVAILABLE


# =============================================================================
# Pure TensorFlow Fallback Implementations
# =============================================================================


@tf.function
def circular_convolution_tf(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Circular convolution via FFT (TensorFlow implementation).

    Computes a ⊛ b = IFFT(FFT(a) * FFT(b))

    Args:
        a: First tensor [batch, hd_dim] or [hd_dim]
        b: Second tensor [batch, hd_dim] or [hd_dim]

    Returns:
        Circular convolution result with same shape.
    """
    # Cast to complex for FFT
    a_complex = tf.cast(a, tf.complex64)
    b_complex = tf.cast(b, tf.complex64)

    # FFT along last dimension
    a_fft = tf.signal.fft(a_complex)
    b_fft = tf.signal.fft(b_complex)

    # Element-wise multiplication in frequency domain
    product_fft = a_fft * b_fft

    # Inverse FFT
    result = tf.signal.ifft(product_fft)

    # Return real part
    return tf.math.real(result)


@tf.function
def holographic_bundle_tf(
    token_ids: tf.Tensor,
    base_vectors: tf.Tensor,
    position_keys: tf.Tensor,
    hd_dim: int,
    model_dim: int,
) -> tf.Tensor:
    """Holographic bundling with FFT-based binding (TensorFlow implementation).

    For each sequence position:
    1. Look up token embedding from base_vectors
    2. Bind with position key via circular convolution
    3. Sum into holographic bundle
    4. Project to model dimension

    Args:
        token_ids: Token IDs [batch, seq_len]
        base_vectors: Base embeddings [vocab_size, hd_dim]
        position_keys: Position encodings [max_seq_len, hd_dim]
        hd_dim: Hyperdimensional space dimension
        model_dim: Output model dimension

    Returns:
        Embeddings [batch, model_dim]
    """
    batch_size = tf.shape(token_ids)[0]
    seq_len = tf.shape(token_ids)[1]

    # Gather token embeddings: [batch, seq, hd_dim]
    token_embeds = tf.gather(base_vectors, token_ids)

    # Gather position keys: [seq, hd_dim] -> broadcast to [1, seq, hd_dim]
    pos_keys = position_keys[:seq_len]  # [seq, hd_dim]
    pos_keys = tf.expand_dims(pos_keys, 0)  # [1, seq, hd_dim]

    # Bind tokens with positions via circular convolution
    # Process each sequence position
    def bind_position(idx):
        tok = token_embeds[:, idx, :]  # [batch, hd_dim]
        pos = pos_keys[0, idx, :]  # [hd_dim]
        pos = tf.expand_dims(pos, 0)  # [1, hd_dim]
        pos = tf.tile(pos, [batch_size, 1])  # [batch, hd_dim]
        return circular_convolution_tf(tok, pos)  # [batch, hd_dim]

    # Stack bound embeddings
    bound_embeds = tf.map_fn(
        bind_position,
        tf.range(seq_len),
        fn_output_signature=tf.TensorSpec([None, hd_dim], tf.float32),
    )  # [seq, batch, hd_dim]

    # Transpose to [batch, seq, hd_dim]
    bound_embeds = tf.transpose(bound_embeds, [1, 0, 2])

    # Sum bundle: [batch, hd_dim]
    bundle = tf.reduce_sum(bound_embeds, axis=1)

    # Project to model dimension via strided averaging
    stride = hd_dim // model_dim
    projected = tf.reshape(bundle, [batch_size, model_dim, stride])
    output = tf.reduce_mean(projected, axis=-1)

    return output


@tf.function
def ctqw_spread_tf(
    embeddings: tf.Tensor,
    steps: int = 3,
) -> tf.Tensor:
    """CTQW semantic spreading (TensorFlow implementation).

    Applies diffusion-style spreading for semantic smoothing.
    Each step: x[i] = 0.5*x[i] + 0.25*(x[i-1] + x[i+1])

    Args:
        embeddings: Input embeddings [batch, dim]
        steps: Number of diffusion steps

    Returns:
        Spread embeddings with same shape.
    """
    result = embeddings

    for _ in range(steps):
        # Circular shift neighbors
        prev = tf.roll(result, shift=1, axis=-1)
        next_ = tf.roll(result, shift=-1, axis=-1)

        # Diffusion step
        result = 0.5 * result + 0.25 * (prev + next_)

    return result


# =============================================================================
# Public API (uses C++ if available, falls back to TF)
# =============================================================================


def holographic_bundle(
    token_ids: tf.Tensor,
    base_vectors: tf.Tensor,
    position_keys: tf.Tensor,
    hd_dim: int,
    model_dim: int,
) -> tf.Tensor:
    """Compute holographic bundle embedding.

    Uses C++ implementation if available, otherwise TensorFlow.

    Args:
        token_ids: Token IDs [batch, seq_len]
        base_vectors: Base embeddings [vocab_size, hd_dim]
        position_keys: Position encodings [max_seq_len, hd_dim]
        hd_dim: Hyperdimensional space dimension
        model_dim: Output model dimension

    Returns:
        Holographic embeddings [batch, model_dim]
    """
    if _HDE_NATIVE_AVAILABLE and _hde_holographic_bundle is not None:
        return _hde_holographic_bundle(
            token_ids,
            base_vectors,
            position_keys,
            hd_dim=hd_dim,
            model_dim=model_dim,
        )
    return holographic_bundle_tf(
        token_ids,
        base_vectors,
        position_keys,
        hd_dim,
        model_dim,
    )


def ctqw_spread(
    embeddings: tf.Tensor,
    steps: int = 3,
) -> tf.Tensor:
    """Apply CTQW semantic spreading.

    Uses C++ implementation if available, otherwise TensorFlow.
    When using C++ op, gradient flows through TensorFlow fallback.

    Args:
        embeddings: Input embeddings [batch, dim]
        steps: Number of diffusion steps

    Returns:
        Spread embeddings with same shape.
    """
    if _HDE_NATIVE_AVAILABLE and _hde_ctqw_spread is not None:
        # Wrap C++ op with custom gradient using TF fallback for backprop
        @tf.custom_gradient
        def ctqw_with_grad(x):
            # Forward: use C++ op
            output = _hde_ctqw_spread(x, steps=steps)

            def grad(dy):
                # Backward: use TF implementation which has defined gradients
                # The TF fallback is differentiable
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    tf_output = ctqw_spread_tf(x, steps)
                return tape.gradient(tf_output, x, output_gradients=dy)

            return output, grad

        return ctqw_with_grad(embeddings)
    return ctqw_spread_tf(embeddings, steps)



__all__ = [
    "holographic_bundle",
    "ctqw_spread",
    "hyperdimensional_embedding_available",
    # TF fallbacks exposed for testing
    "circular_convolution_tf",
    "holographic_bundle_tf",
    "ctqw_spread_tf",
]
