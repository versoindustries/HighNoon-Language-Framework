# highnoon/qsg/hopfield_vocab.py
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

"""Phase A2: Modern Hopfield Network Vocabulary Projection for QSG.

Provides energy-based vocabulary scoring and semantic oracle for
Quantum Superposition Generation (QSG).

Key Components:
    - HopfieldVocabularyProjector: Energy-based vocab→context scoring
    - hopfield_semantic_oracle: Oracle scores for Grover amplification

Advantages over cosine similarity:
    1. Exponential storage capacity (Modern Hopfield)
    2. Energy-based OOD detection
    3. Differentiable for end-to-end training

Reference: "Hopfield Networks is All You Need" (Ramsauer et al., 2020)
"""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.config import QSG_HOPFIELD_BETA

logger = logging.getLogger(__name__)


class HopfieldVocabularyProjector(layers.Layer):
    """Modern Hopfield Network for vocabulary projection.

    Uses vocabulary embeddings as stored patterns and computes energy-based
    retrieval scores for context-to-vocabulary mapping.

    Energy function:
        E(s, ξ) = -β⁻¹ log(Σᵢ exp(β s^T ξᵢ)) + ½||s||²

    The retrieval operation is equivalent to softmax attention:
        p(v|s) = softmax(β * s^T * V)  where V is vocabulary embeddings

    This provides:
        1. Exponential storage capacity vs classical Hopfield
        2. Natural probability distribution over vocabulary
        3. Energy values for OOD detection
        4. Full differentiability for training

    Attributes:
        beta: Inverse temperature (higher = sharper retrieval).
        energy_threshold: Threshold for OOD detection.
    """

    def __init__(
        self,
        beta: float = QSG_HOPFIELD_BETA,
        energy_threshold: float = 0.0,
        learnable_beta: bool = False,
        name: str = "hopfield_vocab_projector",
        **kwargs,
    ):
        """Initialize HopfieldVocabularyProjector.

        Args:
            beta: Inverse temperature for Hopfield energy (default from config).
            energy_threshold: Energy threshold for OOD detection.
            learnable_beta: If True, beta is a trainable parameter.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.
        """
        super().__init__(name=name, **kwargs)
        self._initial_beta = beta
        self.energy_threshold = energy_threshold
        self.learnable_beta = learnable_beta

        # Will be set in build()
        self.beta = None
        self._last_energy = None

    def build(self, input_shape):
        """Build layer weights."""
        if self.learnable_beta:
            # Learnable inverse temperature (log-parameterized for stability)
            self.log_beta = self.add_weight(
                name="log_beta",
                shape=(),
                initializer=tf.keras.initializers.Constant(tf.math.log(self._initial_beta)),
                trainable=True,
            )
        else:
            self.log_beta = None
        super().build(input_shape)

    @property
    def effective_beta(self) -> tf.Tensor:
        """Get effective beta value (supports learnable beta)."""
        if self.learnable_beta and self.log_beta is not None:
            return tf.exp(self.log_beta)
        return tf.constant(self._initial_beta, dtype=tf.float32)

    def compute_energy(
        self,
        context: tf.Tensor,
        vocab_embeddings: tf.Tensor,
    ) -> tf.Tensor:
        """Compute Hopfield energy for context-vocabulary pairs.

        Lower energy indicates better match (more likely token).

        Args:
            context: Context representations [batch, seq_len, dim].
            vocab_embeddings: Vocabulary embeddings [vocab_size, dim].

        Returns:
            Energy tensor [batch, seq_len] (scalar energy per position).
        """
        beta = self.effective_beta

        # Compute similarities: [batch, seq_len, vocab_size]
        # Normalize for numerical stability
        # GRADIENT FIX: Add epsilon to prevent NaN/Inf gradients when norm → 0
        context_norm = tf.nn.l2_normalize(context, axis=-1, epsilon=1e-8)
        vocab_norm = tf.nn.l2_normalize(vocab_embeddings, axis=-1, epsilon=1e-8)

        similarities = tf.matmul(context_norm, vocab_norm, transpose_b=True)
        scaled_sim = beta * similarities

        # Log-sum-exp for energy computation
        # E = -β⁻¹ log(Σᵢ exp(β s^T ξᵢ))
        lse = tf.reduce_logsumexp(scaled_sim, axis=-1)  # [batch, seq_len]
        energy = -lse / beta

        # Add regularization term: ½||s||²
        context_sq_norm = tf.reduce_sum(context_norm**2, axis=-1) * 0.5
        energy = energy + context_sq_norm

        self._last_energy = energy
        return energy

    def call(
        self,
        context: tf.Tensor,
        vocab_embeddings: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute vocabulary logits via Hopfield retrieval.

        Args:
            context: Context hidden states [batch, seq_len, dim].
            vocab_embeddings: Vocabulary embeddings [vocab_size, dim].
            training: Whether in training mode.

        Returns:
            Tuple of:
                - logits: Vocabulary logits [batch, seq_len, vocab_size]
                - energy: Position energy [batch, seq_len]
        """
        beta = self.effective_beta

        # Normalize inputs
        # GRADIENT FIX: Add epsilon to prevent NaN/Inf gradients when norm → 0
        context_norm = tf.nn.l2_normalize(context, axis=-1, epsilon=1e-8)
        vocab_norm = tf.nn.l2_normalize(vocab_embeddings, axis=-1, epsilon=1e-8)

        # Compute scaled similarities (these become logits)
        # Hopfield update: p(v|s) = softmax(β * s^T * V)
        similarities = tf.matmul(context_norm, vocab_norm, transpose_b=True)
        logits = beta * similarities  # [batch, seq_len, vocab_size]

        # Compute energy for OOD detection
        energy = self.compute_energy(context, vocab_embeddings)

        return logits, energy

    def get_ood_mask(self, energy: tf.Tensor) -> tf.Tensor:
        """Get out-of-distribution mask from energy values.

        High energy indicates the context doesn't match any vocabulary
        pattern well, suggesting OOD tokens.

        Args:
            energy: Energy tensor [batch, seq_len].

        Returns:
            OOD mask [batch, seq_len] where True = OOD.
        """
        return energy > self.energy_threshold

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "beta": self._initial_beta,
                "energy_threshold": self.energy_threshold,
                "learnable_beta": self.learnable_beta,
            }
        )
        return config


def hopfield_semantic_oracle(
    context: tf.Tensor,
    vocab_embeddings: tf.Tensor,
    beta: float = QSG_HOPFIELD_BETA,
) -> tf.Tensor:
    """Compute semantic oracle scores for Grover amplification.

    Uses Hopfield energy to determine semantic consistency between
    context and vocabulary tokens. Maps energy to [0, 1] oracle scores.

    Higher oracle score = more semantically consistent token.

    Args:
        context: Context hidden states [batch, seq_len, dim].
        vocab_embeddings: Vocabulary embeddings [vocab_size, dim].
        beta: Inverse temperature for Hopfield energy.

    Returns:
        Oracle scores [batch, seq_len, vocab_size] in range [0, 1].
    """
    # Normalize inputs
    # GRADIENT FIX: Add epsilon to prevent NaN/Inf gradients when norm → 0
    context_norm = tf.nn.l2_normalize(context, axis=-1, epsilon=1e-8)
    vocab_norm = tf.nn.l2_normalize(vocab_embeddings, axis=-1, epsilon=1e-8)

    # Compute scaled similarities
    similarities = tf.matmul(context_norm, vocab_norm, transpose_b=True)
    scaled_sim = beta * similarities

    # Apply softmax to get probability-like scores
    # This is the Hopfield retrieval pattern
    oracle_scores = tf.nn.softmax(scaled_sim, axis=-1)

    return oracle_scores


def hopfield_vocabulary_energy(
    context: tf.Tensor,
    vocab_embeddings: tf.Tensor,
    beta: float = QSG_HOPFIELD_BETA,
) -> tf.Tensor:
    """Compute per-token Hopfield energy for vocabulary.

    Lower energy = better match. Can be used for:
    - OOD detection
    - Confidence estimation
    - Curriculum learning

    Args:
        context: Context hidden states [batch, seq_len, dim] or [batch, dim].
        vocab_embeddings: Vocabulary embeddings [vocab_size, dim].
        beta: Inverse temperature.

    Returns:
        Per-token energy [batch, seq_len, vocab_size] or [batch, vocab_size].
    """
    # Handle both 2D and 3D context
    squeeze_output = False
    if len(context.shape) == 2:
        context = tf.expand_dims(context, axis=1)  # [batch, 1, dim]
        squeeze_output = True

    # Normalize
    # GRADIENT FIX: Add epsilon to prevent NaN/Inf gradients when norm → 0
    context_norm = tf.nn.l2_normalize(context, axis=-1, epsilon=1e-8)
    vocab_norm = tf.nn.l2_normalize(vocab_embeddings, axis=-1, epsilon=1e-8)

    # Similarities: [batch, seq_len, vocab_size]
    similarities = tf.matmul(context_norm, vocab_norm, transpose_b=True)

    # Per-token energy: E_v = -β * s^T * ξ_v + ½||s||² + ½||ξ_v||²
    # Since normalized: ||s||² = ||ξ||² = 1
    energy = -beta * similarities + 1.0  # Add regularization terms (both = 1 when normalized)

    if squeeze_output:
        energy = tf.squeeze(energy, axis=1)

    return energy
