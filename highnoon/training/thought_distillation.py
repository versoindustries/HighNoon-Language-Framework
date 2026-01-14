# highnoon/training/thought_distillation.py
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

"""Phase 87: Thought Distillation for COCONUT Continuous Latent Reasoning.

This module provides tools to distill Chain-of-Thought (CoT) traces into
continuous latent thought representations. The goal is to transfer the
reasoning capability from explicit CoT traces into implicit latent-space
reasoning that COCONUT can leverage.

Key Components:
    - ThoughtDistillationLoss: KL divergence between CoT teacher and student
    - CoTToLatentConverter: Convert discrete CoT tokens to continuous vectors
    - ThoughtDistillationCallback: Progressive distillation during training

Research Reference:
    - "Training LLMs to Reason in a Continuous Latent Space" (Meta COCONUT, 2024)
    - Phase 87 COCONUT specification

Example:
    >>> from highnoon.training.thought_distillation import (
    ...     ThoughtDistillationLoss,
    ...     ThoughtDistillationCallback,
    ... )
    >>> loss_fn = ThoughtDistillationLoss(temperature=2.0)
    >>> callback = ThoughtDistillationCallback(distillation_weight=0.5)
"""

from __future__ import annotations

import logging
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.config import COCONUT_CRYSTALLIZE_THRESHOLD

logger = logging.getLogger(__name__)


class CoTToLatentConverter(layers.Layer):
    """Convert discrete Chain-of-Thought tokens to continuous latent vectors.

    Takes a sequence of token IDs representing a CoT trace and converts them
    to a sequence of continuous thought vectors suitable for distillation
    into COCONUT's latent reasoning space.

    Attributes:
        embedding_dim: Dimension of output latent vectors.
        hidden_dim: Internal hidden dimension for transformation.
        num_layers: Number of transformer layers for contextualization.

    Example:
        >>> converter = CoTToLatentConverter(embedding_dim=512)
        >>> cot_tokens = tf.constant([[101, 2003, 1037, 3231, 102]])  # "this is a test"
        >>> latent_thoughts = converter(cot_tokens, token_embeddings)
        >>> # latent_thoughts: [1, 5, 512]
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        name: str = "cot_to_latent",
        **kwargs: Any,
    ) -> None:
        """Initialize CoTToLatentConverter.

        Args:
            embedding_dim: Output embedding dimension.
            hidden_dim: Hidden layer dimension. Defaults to 4x embedding_dim.
            num_layers: Number of transformer layers.
            dropout_rate: Dropout rate for regularization.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.
        """
        super().__init__(name=name, **kwargs)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim or (4 * embedding_dim)
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Input projection to embedding_dim
        self.input_projection = layers.Dense(
            embedding_dim,
            name=f"{name}_input_proj",
        )

        # Transformer-like layers for contextualization
        self.attention_layers = []
        self.ffn_layers = []
        self.norm_layers = []

        for i in range(num_layers):
            # Multi-head self-attention
            self.attention_layers.append(
                layers.MultiHeadAttention(
                    num_heads=8,
                    key_dim=embedding_dim // 8,
                    dropout=dropout_rate,
                    name=f"{name}_attn_{i}",
                )
            )
            # Feed-forward network
            self.ffn_layers.append(
                tf.keras.Sequential(
                    [
                        layers.Dense(self.hidden_dim, activation="gelu"),
                        layers.Dropout(dropout_rate),
                        layers.Dense(embedding_dim),
                        layers.Dropout(dropout_rate),
                    ],
                    name=f"{name}_ffn_{i}",
                )
            )
            # Layer norms (pre-norm architecture)
            self.norm_layers.append(
                [
                    layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_{i}_a"),
                    layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_{i}_b"),
                ]
            )

        # Output projection
        self.output_projection = layers.Dense(
            embedding_dim,
            name=f"{name}_output_proj",
        )
        self.output_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{name}_output_norm",
        )

    def call(
        self,
        cot_token_ids: tf.Tensor,
        token_embeddings: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Convert CoT tokens to latent thought vectors.

        Args:
            cot_token_ids: Token IDs for CoT trace [batch, seq_len].
            token_embeddings: Token embedding matrix [vocab_size, embed_dim].
            attention_mask: Optional attention mask [batch, seq_len].
            training: Whether in training mode.

        Returns:
            Continuous thought vectors [batch, seq_len, embedding_dim].
        """
        # Get embeddings for CoT tokens
        x = tf.nn.embedding_lookup(token_embeddings, cot_token_ids)

        # Project to internal dimension
        x = self.input_projection(x)

        # Apply transformer layers
        for i in range(self.num_layers):
            # Pre-norm attention
            normed = self.norm_layers[i][0](x)
            attn_out = self.attention_layers[i](
                normed,
                normed,
                attention_mask=attention_mask,
                training=training,
            )
            x = x + attn_out

            # Pre-norm FFN
            normed = self.norm_layers[i][1](x)
            ffn_out = self.ffn_layers[i](normed, training=training)
            x = x + ffn_out

        # Final projection and normalization
        x = self.output_projection(x)
        x = self.output_norm(x)

        return x

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class ThoughtDistillationLoss(tf.keras.losses.Loss):
    """Distillation loss for transferring CoT reasoning to continuous thoughts.

    Computes a combined loss that includes:
    1. KL divergence between CoT teacher and COCONUT student distributions
    2. MSE between teacher and student hidden representations
    3. Optional consistency regularization

    Attributes:
        temperature: Softmax temperature for softer distributions.
        alpha: Weight for hard label cross-entropy (1-alpha for soft labels).
        use_cosine_similarity: Whether to use cosine similarity for hidden match.

    Example:
        >>> loss_fn = ThoughtDistillationLoss(temperature=2.0, alpha=0.5)
        >>> loss = loss_fn(teacher_logits, student_logits, hidden_teacher, hidden_student)
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        use_cosine_similarity: bool = True,
        name: str = "thought_distillation_loss",
    ) -> None:
        """Initialize ThoughtDistillationLoss.

        Args:
            temperature: Temperature for softmax softening. Higher = softer.
            alpha: Weight for hard label CE vs soft label KL divergence.
                   0 = only soft, 1 = only hard.
            use_cosine_similarity: Use cosine sim instead of MSE for hidden.
            name: Loss name.
        """
        super().__init__(name=name)
        self.temperature = temperature
        self.alpha = alpha
        self.use_cosine_similarity = use_cosine_similarity

    def call(
        self,
        teacher_logits: tf.Tensor,
        student_logits: tf.Tensor,
        teacher_hidden: tf.Tensor | None = None,
        student_hidden: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """Compute distillation loss.

        Args:
            teacher_logits: Teacher (CoT) model logits [batch, seq, vocab].
            student_logits: Student (COCONUT) model logits [batch, seq, vocab].
            teacher_hidden: Optional teacher hidden states [batch, seq, dim].
            student_hidden: Optional student hidden states [batch, seq, dim].
            labels: Optional hard labels for cross-entropy [batch, seq].

        Returns:
            Combined distillation loss scalar.
        """
        # Soft target loss (KL divergence)
        teacher_soft = tf.nn.softmax(teacher_logits / self.temperature, axis=-1)
        student_soft = tf.nn.log_softmax(student_logits / self.temperature, axis=-1)

        # KL divergence: D_KL(teacher || student)
        kl_loss = tf.reduce_sum(
            teacher_soft * (tf.math.log(teacher_soft + 1e-8) - student_soft), axis=-1
        )
        kl_loss = tf.reduce_mean(kl_loss) * (self.temperature**2)

        total_loss = kl_loss

        # Hard label loss if labels provided
        if labels is not None and self.alpha > 0:
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
                labels,
                student_logits,
                from_logits=True,
            )
            ce_loss = tf.reduce_mean(ce_loss)
            total_loss = (1 - self.alpha) * kl_loss + self.alpha * ce_loss

        # Hidden state matching loss
        if teacher_hidden is not None and student_hidden is not None:
            if self.use_cosine_similarity:
                # Cosine similarity loss: 1 - cos_sim
                # GRADIENT FIX: Add epsilon to prevent NaN/Inf gradients when norm → 0
                teacher_norm = tf.nn.l2_normalize(teacher_hidden, axis=-1, epsilon=1e-8)
                student_norm = tf.nn.l2_normalize(student_hidden, axis=-1, epsilon=1e-8)
                cos_sim = tf.reduce_sum(teacher_norm * student_norm, axis=-1)
                hidden_loss = tf.reduce_mean(1 - cos_sim)
            else:
                # MSE loss
                hidden_loss = tf.reduce_mean((teacher_hidden - student_hidden) ** 2)

            total_loss = total_loss + 0.5 * hidden_loss

        return total_loss

    def get_config(self) -> dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config.update(
            {
                "temperature": self.temperature,
                "alpha": self.alpha,
                "use_cosine_similarity": self.use_cosine_similarity,
            }
        )
        return config


class ThoughtDistillationCallback(tf.keras.callbacks.Callback):
    """Callback for progressive thought distillation during training.

    Manages the distillation process by:
    1. Gradually increasing distillation weight during warmup
    2. Tracking distillation metrics
    3. Optionally crystallizing high-confidence thought patterns

    Attributes:
        distillation_weight: Weight for distillation loss vs main task loss.
        warmup_steps: Steps over which to ramp up distillation weight.
        crystallize_threshold: Confidence threshold for crystallization.
        max_crystals: Maximum crystallized reasoning patterns to store.

    Example:
        >>> callback = ThoughtDistillationCallback(
        ...     distillation_weight=0.5,
        ...     warmup_steps=1000,
        ... )
        >>> model.fit(data, callbacks=[callback])
    """

    def __init__(
        self,
        distillation_weight: float = 0.5,
        warmup_steps: int = 1000,
        crystallize_threshold: float = COCONUT_CRYSTALLIZE_THRESHOLD,
        max_crystals: int = 64,
        log_frequency: int = 100,
    ) -> None:
        """Initialize ThoughtDistillationCallback.

        Args:
            distillation_weight: Target weight for distillation loss.
            warmup_steps: Steps to ramp up distillation weight from 0.
            crystallize_threshold: Confidence for crystallizing thought patterns.
            max_crystals: Max stored crystal patterns.
            log_frequency: Logging frequency in steps.
        """
        super().__init__()
        self.distillation_weight = distillation_weight
        self.warmup_steps = warmup_steps
        self.crystallize_threshold = crystallize_threshold
        self.max_crystals = max_crystals
        self.log_frequency = log_frequency

        self._current_step = 0
        self._current_weight = 0.0
        self._distillation_losses = []

    @property
    def current_weight(self) -> float:
        """Get current distillation weight based on warmup schedule."""
        if self._current_step >= self.warmup_steps:
            return self.distillation_weight

        # Linear warmup
        return self.distillation_weight * (self._current_step / self.warmup_steps)

    def on_train_batch_begin(self, batch: int, logs: dict | None = None) -> None:
        """Update distillation weight at batch start."""
        self._current_weight = self.current_weight

        # Set weight in model if it has the attribute
        if hasattr(self.model, "distillation_weight"):
            self.model.distillation_weight = self._current_weight

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        """Track distillation metrics and update step count."""
        self._current_step += 1

        # Track distillation loss if present
        if logs and "distillation_loss" in logs:
            self._distillation_losses.append(logs["distillation_loss"])

        # Log periodically
        if self._current_step % self.log_frequency == 0:
            avg_distill = (
                sum(self._distillation_losses[-self.log_frequency :])
                / min(len(self._distillation_losses), self.log_frequency)
                if self._distillation_losses
                else 0.0
            )
            logger.info(
                f"Step {self._current_step}: distillation_weight={self._current_weight:.3f}, "
                f"avg_distillation_loss={avg_distill:.4f}"
            )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Log epoch-level distillation statistics."""
        if self._distillation_losses:
            avg_loss = sum(self._distillation_losses) / len(self._distillation_losses)
            logger.info(
                f"Epoch {epoch}: avg_distillation_loss={avg_loss:.4f}, "
                f"total_steps={self._current_step}, "
                f"final_weight={self._current_weight:.3f}"
            )
            self._distillation_losses = []

    def get_config(self) -> dict[str, Any]:
        """Get callback configuration."""
        return {
            "distillation_weight": self.distillation_weight,
            "warmup_steps": self.warmup_steps,
            "crystallize_threshold": self.crystallize_threshold,
            "max_crystals": self.max_crystals,
            "log_frequency": self.log_frequency,
        }


def create_distillation_loss(
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> ThoughtDistillationLoss:
    """Factory function for creating distillation loss.

    Args:
        temperature: Softmax temperature for distillation.
        alpha: Weight for hard labels (1-alpha for soft labels).

    Returns:
        Configured ThoughtDistillationLoss instance.
    """
    return ThoughtDistillationLoss(temperature=temperature, alpha=alpha)


def create_distillation_callback(
    distillation_weight: float = 0.5,
    warmup_steps: int = 1000,
) -> ThoughtDistillationCallback:
    """Factory function for creating distillation callback.

    Args:
        distillation_weight: Target distillation loss weight.
        warmup_steps: Steps to ramp up distillation.

    Returns:
        Configured ThoughtDistillationCallback instance.
    """
    return ThoughtDistillationCallback(
        distillation_weight=distillation_weight,
        warmup_steps=warmup_steps,
    )


__all__ = [
    "CoTToLatentConverter",
    "ThoughtDistillationLoss",
    "ThoughtDistillationCallback",
    "create_distillation_loss",
    "create_distillation_callback",
]
