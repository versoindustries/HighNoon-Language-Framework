# highnoon/learning/hd_learner.py
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

"""Hyperdimensional Learner for Model Inference Update.

Handles HD-space operations for continual learning:
    - Holographic bundling of content embeddings
    - Memory consolidation via Modern Hopfield
    - Gradient computation for HD bundles
    - Integration with model's HD embedding layers

This module bridges the ContentIndexer's output with the CrystallizedOptimizer,
computing gradients that can be safely applied without catastrophic forgetting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf

from highnoon import config
from highnoon.learning.content_indexer import ContentChunk

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# MEMORY BANK
# =============================================================================


@dataclass
class MemoryEntry:
    """A single entry in the HD memory bank.

    Attributes:
        embedding: HD vector representation.
        source_id: Original content source identifier.
        confidence: Retrieval confidence (0-1).
        access_count: Number of times retrieved.
        crystallized: Whether this memory is protected.
    """

    embedding: np.ndarray
    source_id: str
    confidence: float = 0.0
    access_count: int = 0
    crystallized: bool = False


@dataclass
class LearningBatch:
    """A batch of content ready for gradient computation.

    Attributes:
        embeddings: HD embeddings tensor [batch, hd_dim].
        targets: Target outputs for loss computation.
        weights: Sample weights for importance weighting.
        metadata: Additional batch metadata.
    """

    embeddings: tf.Tensor
    targets: tf.Tensor | None = None
    weights: tf.Tensor | None = None
    metadata: dict = field(default_factory=dict)


# =============================================================================
# HD LEARNER
# =============================================================================


class HDLearner:
    """Hyperdimensional learning engine for MIU.

    This class manages:
        1. HD memory bank for content storage
        2. Modern Hopfield associative memory for consolidation
        3. Gradient computation for HD bundle updates
        4. Integration with model's forward pass

    The HDLearner uses the existing infrastructure from:
        - hyperdimensional_embedding_op.h (HolographicBundle, CTQWSpread)
        - quantum_holographic_memory.h (ModernHopfieldRetrieve)
        - unified_memory_system_op.h (memory primitives)

    Attributes:
        hd_dim: Hyperdimensional embedding dimension.
        memory_slots: Maximum number of memory entries.
        hopfield_beta: Modern Hopfield inverse temperature.

    Example:
        >>> learner = HDLearner(model)
        >>> learner.add_embeddings(chunks)
        >>> learner.consolidate_memory()
        >>> gradients = learner.compute_gradients(batch)
    """

    def __init__(
        self,
        model: tf.keras.Model | None = None,
        hd_dim: int | None = None,
        memory_slots: int | None = None,
        hopfield_beta: float = 1.0,
    ) -> None:
        """Initialize HDLearner.

        Args:
            model: The model to learn into (optional, for gradient computation).
            hd_dim: HD embedding dimension (defaults to config.MIU_HD_DIM).
            memory_slots: Max memory entries (defaults to config.MIU_MEMORY_SLOTS).
            hopfield_beta: Modern Hopfield inverse temperature.
        """
        self.model = model
        self.hd_dim = hd_dim or config.MIU_HD_DIM
        self.memory_slots = memory_slots or config.MIU_MEMORY_SLOTS
        self.hopfield_beta = hopfield_beta

        # Memory bank
        self._memory: list[MemoryEntry] = []
        self._memory_matrix: tf.Variable | None = None

        # Loss function for gradient computation
        self._loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

        logger.info(
            f"HDLearner initialized: hd_dim={self.hd_dim}, "
            f"memory_slots={self.memory_slots}, beta={self.hopfield_beta}"
        )

    @property
    def memory_size(self) -> int:
        """Current number of entries in memory."""
        return len(self._memory)

    @property
    def memory_matrix(self) -> tf.Tensor:
        """Get memory bank as tensor [num_entries, hd_dim]."""
        if self._memory_matrix is None or len(self._memory) == 0:
            return tf.zeros([0, self.hd_dim], dtype=tf.float32)
        return self._memory_matrix

    def add_chunk(self, chunk: ContentChunk) -> None:
        """Add a content chunk to memory.

        Args:
            chunk: Content chunk with embedding.
        """
        if chunk.embedding is None:
            logger.warning(f"Chunk {chunk.chunk_idx} has no embedding, skipping")
            return

        entry = MemoryEntry(
            embedding=chunk.embedding,
            source_id=chunk.source_id,
        )
        self._memory.append(entry)

        # Evict if over capacity (FIFO for non-crystallized)
        if len(self._memory) > self.memory_slots:
            self._evict_oldest()

        # Invalidate cached matrix
        self._memory_matrix = None

    def add_embeddings(self, chunks: list[ContentChunk]) -> int:
        """Add multiple content chunks to memory.

        Args:
            chunks: List of content chunks with embeddings.

        Returns:
            Number of chunks successfully added.
        """
        added = 0
        for chunk in chunks:
            if chunk.embedding is not None:
                self.add_chunk(chunk)
                added += 1
        logger.info(f"Added {added} chunks to HD memory")
        return added

    def _evict_oldest(self) -> None:
        """Evict oldest non-crystallized entry."""
        for i, entry in enumerate(self._memory):
            if not entry.crystallized:
                self._memory.pop(i)
                return
        # All crystallized, remove oldest anyway
        self._memory.pop(0)

    def _build_memory_matrix(self) -> tf.Variable:
        """Build memory matrix from entries."""
        if len(self._memory) == 0:
            return tf.Variable(
                tf.zeros([1, self.hd_dim], dtype=tf.float32),
                trainable=False,
            )

        embeddings = np.stack([e.embedding for e in self._memory])
        return tf.Variable(
            tf.cast(embeddings, tf.float32),
            trainable=False,
            name="hd_memory_matrix",
        )

    def query(self, query_embedding: tf.Tensor, top_k: int = 5) -> list[MemoryEntry]:
        """Query memory using Modern Hopfield retrieval.

        Args:
            query_embedding: Query vector [hd_dim] or [batch, hd_dim].
            top_k: Number of entries to retrieve.

        Returns:
            List of top-k memory entries by similarity.
        """
        if len(self._memory) == 0:
            return []

        if self._memory_matrix is None:
            self._memory_matrix = self._build_memory_matrix()

        # Ensure query is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = tf.expand_dims(query_embedding, 0)

        # Compute similarities (Modern Hopfield style)
        # attention = softmax(beta * query @ memory^T)
        similarities = tf.matmul(
            query_embedding,
            self._memory_matrix,
            transpose_b=True,
        )  # [batch, num_entries]

        # Apply temperature scaling
        attention = tf.nn.softmax(self.hopfield_beta * similarities, axis=-1)

        # Get top-k indices
        top_k = min(top_k, len(self._memory))
        _, top_indices = tf.math.top_k(attention[0], k=top_k)

        # Update access counts and return entries
        results = []
        for idx in top_indices.numpy():
            entry = self._memory[idx]
            entry.access_count += 1
            entry.confidence = float(attention[0, idx].numpy())
            results.append(entry)

        return results

    def consolidate_memory(self, threshold: float | None = None) -> int:
        """Consolidate memory using Hopfield energy minimization.

        High-confidence retrievals become "core memories" that are
        candidates for crystallization.

        Args:
            threshold: Confidence threshold for consolidation
                (defaults to config.MIU_CONSOLIDATION_THRESHOLD).

        Returns:
            Number of memories consolidated.
        """
        threshold = threshold or config.MIU_CONSOLIDATION_THRESHOLD

        consolidated = 0
        for entry in self._memory:
            if entry.confidence >= threshold and not entry.crystallized:
                # Mark as ready for crystallization
                entry.crystallized = True
                consolidated += 1

        if consolidated > 0:
            logger.info(f"Consolidated {consolidated} memories for crystallization")

        return consolidated

    def create_learning_batch(
        self,
        batch_size: int | None = None,
    ) -> LearningBatch:
        """Create a batch for gradient computation.

        Samples from memory to create a training batch.

        Args:
            batch_size: Batch size (defaults to config.MIU_BATCH_SIZE).

        Returns:
            LearningBatch ready for gradient computation.
        """
        batch_size = batch_size or config.MIU_BATCH_SIZE

        if len(self._memory) == 0:
            return LearningBatch(embeddings=tf.zeros([0, self.hd_dim], dtype=tf.float32))

        # Sample with importance weighting (higher confidence = more important)
        weights = np.array([e.confidence + 0.1 for e in self._memory])
        weights = weights / weights.sum()

        indices = np.random.choice(
            len(self._memory),
            size=min(batch_size, len(self._memory)),
            replace=False,
            p=weights,
        )

        embeddings = np.stack([self._memory[i].embedding for i in indices])
        sample_weights = np.array([self._memory[i].confidence for i in indices])

        return LearningBatch(
            embeddings=tf.cast(embeddings, tf.float32),
            weights=tf.cast(sample_weights, tf.float32),
            metadata={"indices": indices.tolist()},
        )

    def compute_gradients(
        self,
        batch: LearningBatch,
        tape: tf.GradientTape | None = None,
    ) -> list[tuple[tf.Tensor, tf.Variable]]:
        """Compute gradients for HD bundle learning.

        Uses the model's forward pass to compute gradients that
        will improve prediction on the batch content.

        Args:
            batch: Learning batch with embeddings.
            tape: Optional gradient tape (creates new one if None).

        Returns:
            List of (gradient, variable) tuples.
        """
        if self.model is None:
            raise ValueError("Model must be provided for gradient computation")

        if batch.embeddings.shape[0] == 0:
            return []

        # Create tape if not provided
        own_tape = tape is None
        if own_tape:
            tape = tf.GradientTape()

        with tape:
            # Forward pass through model with HD embeddings as input
            # This assumes the model can accept pre-embedded inputs
            outputs = self._forward_with_embeddings(batch.embeddings)

            # Compute reconstruction loss
            # The goal is to make the model "remember" this content
            loss = self._compute_memory_loss(batch, outputs)

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Filter out None gradients
        grads_and_vars = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]

        logger.debug(
            f"Computed gradients for {len(grads_and_vars)} variables, loss={float(loss):.4f}"
        )

        return grads_and_vars

    def _forward_with_embeddings(self, embeddings: tf.Tensor) -> tf.Tensor:
        """Forward pass using pre-computed embeddings.

        Args:
            embeddings: HD embeddings [batch, hd_dim].

        Returns:
            Model outputs.
        """
        # Try to find the model's embedding layer and bypass it
        # This is model-specific and may need adaptation
        if hasattr(self.model, "reasoning_module"):
            # HighNoon-specific: inject into reasoning module
            return self.model.reasoning_module(embeddings, training=True)
        else:
            # Generic: assume model accepts embeddings directly
            return self.model(embeddings, training=True)

    def _compute_memory_loss(self, batch: LearningBatch, outputs: tf.Tensor) -> tf.Tensor:
        """Compute loss for memory retention.

        Uses contrastive loss to encourage the model to distinguish
        learned content from random content.

        Args:
            batch: Learning batch.
            outputs: Model outputs.

        Returns:
            Scalar loss value.
        """
        # Normalize outputs for similarity computation
        outputs_norm = tf.nn.l2_normalize(outputs, axis=-1, epsilon=1e-6)
        inputs_norm = tf.nn.l2_normalize(batch.embeddings, axis=-1, epsilon=1e-6)

        # Contrastive loss: maximize similarity with input embeddings
        similarity = tf.reduce_sum(outputs_norm * inputs_norm, axis=-1)
        loss = -tf.reduce_mean(similarity)  # Negative because we maximize

        # Apply sample weights if available
        if batch.weights is not None:
            weights = batch.weights / tf.reduce_sum(batch.weights)
            loss = -tf.reduce_sum(similarity * weights)

        return loss

    def clear_memory(self) -> None:
        """Clear all memory entries."""
        self._memory.clear()
        self._memory_matrix = None
        logger.info("HD memory cleared")

    def get_crystallized_embeddings(self) -> list[np.ndarray]:
        """Get embeddings of crystallized memories.

        Returns:
            List of crystallized HD embeddings.
        """
        return [e.embedding for e in self._memory if e.crystallized]


__all__ = ["HDLearner", "MemoryEntry", "LearningBatch"]
