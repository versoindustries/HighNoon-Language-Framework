# highnoon/layers/factored_vocab_embedding.py
# Copyright 2026 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Factored vocabulary embedding layer for QSG optimization.
# Implements Phase 1.2 of the QSG Enterprise Optimization Roadmap.

"""Factored vocabulary embedding layer for efficient QSG oracle scoring.

This module provides a trainable factored embedding representation where
the vocabulary embeddings are stored as E ≈ U × V^T, reducing:
- Memory: V×d → V×r + r×d (e.g., 30.7MB → 8MB for 60K vocab, 128 dim, 32 rank)
- Compute: O(K×d) → O(d×r) + O(K×r) for oracle scoring

The factored representation is trained from scratch as a first-class
architectural feature, providing lossless precision compared to SVD retrofitting.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class FactoredVocabEmbedding(layers.Layer):
    """Factored vocabulary embedding with trainable U and V factors.

    Represents vocabulary embeddings as U @ V^T where:
    - U: [vocab_size, rank] - per-token reduced representation
    - V: [rank, dim] - shared projection basis (transposed for efficiency)

    Benefits:
    - Reduced memory footprint (~4x for rank=32, dim=128)
    - Faster oracle scoring via low-rank dot products
    - Full gradient flow through both factors
    - Native checkpoint format (no conversion needed)

    Usage:
        embed = FactoredVocabEmbedding(vocab_size=60000, dim=128, rank=32)
        token_embeddings = embed(token_ids)  # [batch, seq_len, dim]

        # For QSG oracle scoring (fast path):
        projected_ctx = embed.project_context(context)  # [batch, seq_len, rank]
        scores = embed.score_candidates(projected_ctx, candidates)  # [batch, seq_len, K]

    Attributes:
        vocab_size: Size of vocabulary.
        dim: Full embedding dimension.
        rank: Factorization rank (reduced dimension).
        U: Per-token factor [vocab_size, rank].
        V: Projection basis [rank, dim].
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        rank: int = 32,
        u_initializer: str = "glorot_uniform",
        v_initializer: str = "orthogonal",
        u_regularizer: tf.keras.regularizers.Regularizer | None = None,
        v_regularizer: tf.keras.regularizers.Regularizer | None = None,
        **kwargs,
    ):
        """Initialize factored vocabulary embedding.

        Args:
            vocab_size: Number of tokens in vocabulary.
            dim: Full embedding dimension (output dimension).
            rank: Factorization rank (latent dimension). Should be < dim.
            u_initializer: Initializer for U factor.
            v_initializer: Initializer for V factor ('orthogonal' recommended).
            u_regularizer: Optional regularizer for U.
            v_regularizer: Optional regularizer for V.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dim = dim
        self.rank = rank
        self.u_initializer = tf.keras.initializers.get(u_initializer)
        self.v_initializer = tf.keras.initializers.get(v_initializer)
        self.u_regularizer = u_regularizer
        self.v_regularizer = v_regularizer

    def build(self, input_shape):
        """Build the factored embedding weights.

        Creates:
        - U: [vocab_size, rank] with u_initializer
        - V: [rank, dim] with v_initializer (orthogonal recommended)
        - U_norms: [vocab_size] precomputed L2 norms (non-trainable)
        """
        self.U = self.add_weight(
            name="U",
            shape=(self.vocab_size, self.rank),
            initializer=self.u_initializer,
            regularizer=self.u_regularizer,
            trainable=True,
            dtype=self.dtype,
        )

        self.V = self.add_weight(
            name="V",
            shape=(self.rank, self.dim),
            initializer=self.v_initializer,
            regularizer=self.v_regularizer,
            trainable=True,
            dtype=self.dtype,
        )

        # Precomputed norms for fast cosine similarity (updated during call)
        self.U_norms = self.add_weight(
            name="U_norms",
            shape=(self.vocab_size,),
            initializer="zeros",
            trainable=False,
            dtype=self.dtype,
        )

        super().build(input_shape)

    def call(self, token_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Look up embeddings for given token IDs.

        Computes: embeddings = U[token_ids] @ V

        Args:
            token_ids: Integer token IDs [..., sequence_length].
            training: Whether in training mode.

        Returns:
            Token embeddings [..., sequence_length, dim].
        """
        # Gather U rows for token IDs: [..., seq_len, rank]
        u_rows = tf.gather(self.U, token_ids)

        # Multiply by V to get full embeddings: [..., seq_len, dim]
        embeddings = tf.matmul(u_rows, self.V)

        # Update norms during training (for oracle scoring)
        if training:
            self._update_norms()

        return embeddings

    def project_context(self, context: tf.Tensor) -> tf.Tensor:
        """Project context embeddings to low-rank space for fast oracle scoring.

        Computes: projected = context @ V^T

        This is shared across all candidates at each position, amortizing
        the O(d × r) cost.

        Args:
            context: Context embeddings [batch, seq_len, dim].

        Returns:
            Projected embeddings [batch, seq_len, rank].
        """
        # V is [rank, dim], so V^T is [dim, rank]
        return tf.matmul(context, self.V, transpose_b=True)

    def score_candidates(
        self, projected_context: tf.Tensor, candidates: tf.Tensor, normalize: bool = True
    ) -> tf.Tensor:
        """Score candidate tokens using projected context.

        Computes cosine similarity between projected context and U rows
        for specified candidates only.

        Args:
            projected_context: Projected embeddings [batch, seq_len, rank].
            candidates: Candidate token indices [batch, seq_len, num_candidates].
            normalize: Whether to compute cosine similarity (True) or dot product (False).

        Returns:
            Similarity scores [batch, seq_len, num_candidates] in [0, 1].
        """
        tf.shape(projected_context)[0]
        tf.shape(projected_context)[1]
        tf.shape(candidates)[2]

        # Gather U rows for candidates: [batch, seq_len, K, rank]
        u_candidates = tf.gather(self.U, candidates)

        # Expand projected context for broadcasting: [batch, seq_len, 1, rank]
        projected_expanded = tf.expand_dims(projected_context, axis=2)

        # Dot product: [batch, seq_len, K]
        dot = tf.reduce_sum(projected_expanded * u_candidates, axis=-1)

        if normalize:
            # Compute norms for cosine similarity
            proj_norm = tf.norm(projected_context, axis=-1, keepdims=True)  # [B, L, 1]

            # Gather precomputed U norms: [batch, seq_len, K]
            u_norms = tf.gather(self.U_norms, candidates)

            # Cosine similarity
            cos_sim = dot / (proj_norm * u_norms + 1e-8)

            # Map from [-1, 1] to [0, 1] for oracle compatibility
            return (cos_sim + 1.0) * 0.5
        else:
            return dot

    def get_full_embeddings(self, token_ids: tf.Tensor | None = None) -> tf.Tensor:
        """Reconstruct full embeddings for specified tokens or all vocab.

        Args:
            token_ids: Optional token IDs. If None, returns all vocab embeddings.

        Returns:
            Full embeddings [num_tokens, dim] or [vocab_size, dim].
        """
        if token_ids is None:
            # All vocabulary: [vocab_size, dim]
            return tf.matmul(self.U, self.V)
        else:
            # Specific tokens
            u_rows = tf.gather(self.U, token_ids)
            return tf.matmul(u_rows, self.V)

    def _update_norms(self):
        """Update precomputed U norms for cosine similarity."""
        norms = tf.norm(self.U, axis=-1)
        self.U_norms.assign(norms)

    def get_config(self) -> dict:
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "dim": self.dim,
                "rank": self.rank,
                "u_initializer": tf.keras.initializers.serialize(self.u_initializer),
                "v_initializer": tf.keras.initializers.serialize(self.v_initializer),
                "u_regularizer": (
                    tf.keras.regularizers.serialize(self.u_regularizer)
                    if self.u_regularizer
                    else None
                ),
                "v_regularizer": (
                    tf.keras.regularizers.serialize(self.v_regularizer)
                    if self.v_regularizer
                    else None
                ),
            }
        )
        return config

    @classmethod
    def from_full_embeddings(
        cls, full_embeddings: np.ndarray, rank: int = 32, **kwargs
    ) -> FactoredVocabEmbedding:
        """Create factored embedding from full embeddings via truncated SVD.

        This is for retrofitting existing models. For new training,
        use the standard constructor with random initialization.

        Args:
            full_embeddings: Full vocabulary embeddings [vocab_size, dim].
            rank: Target factorization rank.
            **kwargs: Additional layer arguments.

        Returns:
            Initialized FactoredVocabEmbedding layer.
        """
        vocab_size, dim = full_embeddings.shape

        # Truncated SVD: E ≈ U @ S @ V^T, where we absorb sqrt(S) into both
        from scipy.linalg import svd

        U_full, S, Vt = svd(full_embeddings, full_matrices=False)

        # Truncate to rank
        sqrt_S = np.sqrt(S[:rank])
        U_init = U_full[:, :rank] * sqrt_S[np.newaxis, :]  # [V, r]
        V_init = Vt[:rank, :] * sqrt_S[:, np.newaxis]  # [r, d]

        # Create layer
        layer = cls(vocab_size=vocab_size, dim=dim, rank=rank, **kwargs)

        # Build and set weights
        layer.build(None)
        layer.U.assign(U_init.astype(np.float32))
        layer.V.assign(V_init.astype(np.float32))
        layer._update_norms()

        return layer


class FactoredVocabWithPQ(FactoredVocabEmbedding):
    """Factored vocabulary embedding with Product Quantization for fast top-K.

    Extends FactoredVocabEmbedding with an integrated PQ index for
    approximate nearest neighbor search during QSG candidate selection.

    The PQ index is built on the U factor (low-rank representation),
    enabling O(M × K_pq) + O(V × M) search instead of O(V × r) brute force.

    Typical workflow:
    1. Training: Use standard embed() and score_candidates()
    2. Inference: Use select_topk_candidates() for fast approximate selection,
                  then score_candidates() for exact scoring of candidates
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        rank: int = 32,
        pq_num_subvectors: int = 8,
        pq_num_centroids: int = 256,
        pq_rerank_factor: int = 4,
        **kwargs,
    ):
        """Initialize factored embedding with PQ index.

        Args:
            vocab_size: Number of tokens in vocabulary.
            dim: Full embedding dimension.
            rank: Factorization rank.
            pq_num_subvectors: Number of PQ subvector spaces (M).
            pq_num_centroids: Centroids per subspace (K_pq, typically 256).
            pq_rerank_factor: Re-rank top (factor × K) candidates with exact distances.
            **kwargs: Additional layer arguments.
        """
        super().__init__(vocab_size, dim, rank, **kwargs)
        self.pq_num_subvectors = pq_num_subvectors
        self.pq_num_centroids = pq_num_centroids
        self.pq_rerank_factor = pq_rerank_factor

        # PQ index will be built after training
        self._pq_centroids = None
        self._pq_codes = None
        self._pq_built = False

    def build_pq_index(self, num_kmeans_iters: int = 25):
        """Build PQ index on current U weights.

        Should be called after training, before inference.

        Args:
            num_kmeans_iters: K-means iterations for training centroids.
        """
        U_np = self.U.numpy()

        # Compute subvector dimension
        subvec_dim = (self.rank + self.pq_num_subvectors - 1) // self.pq_num_subvectors

        # Train centroids and encode
        centroids = []
        codes = np.zeros((self.vocab_size, self.pq_num_subvectors), dtype=np.uint8)

        for m in range(self.pq_num_subvectors):
            start = m * subvec_dim
            end = min(start + subvec_dim, self.rank)

            if start >= self.rank:
                # Pad with zeros
                centroids.append(np.zeros((self.pq_num_centroids, subvec_dim), dtype=np.float32))
                continue

            subvecs = U_np[:, start:end]
            actual_dim = subvecs.shape[1]

            # Pad to subvec_dim if needed
            if actual_dim < subvec_dim:
                subvecs = np.pad(subvecs, ((0, 0), (0, subvec_dim - actual_dim)))

            # K-means clustering
            from sklearn.cluster import KMeans

            kmeans = KMeans(
                n_clusters=self.pq_num_centroids,
                max_iter=num_kmeans_iters,
                n_init=1,
                random_state=42,
            )
            kmeans.fit(subvecs)

            centroids.append(kmeans.cluster_centers_.astype(np.float32))
            codes[:, m] = kmeans.labels_.astype(np.uint8)

        self._pq_centroids = np.stack(centroids)  # [M, K_pq, subvec_dim]
        self._pq_codes = codes  # [V, M]
        self._pq_built = True

    def select_topk_candidates(
        self, projected_context: tf.Tensor, top_k: int = 1024
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Select top-K candidates using PQ approximate search.

        Uses asymmetric distance computation (ADC) for fast approximate
        nearest neighbor search, then re-ranks with exact distances.

        Args:
            projected_context: Projected context [batch, seq_len, rank].
            top_k: Number of candidates to return per position.

        Returns:
            Tuple of:
            - candidates: Token indices [batch, seq_len, top_k]
            - scores: Similarity scores [batch, seq_len, top_k]
        """
        if not self._pq_built:
            raise RuntimeError("PQ index not built. Call build_pq_index() first.")

        batch_size = tf.shape(projected_context)[0]
        seq_len = tf.shape(projected_context)[1]

        # For now, implement in TensorFlow (C++ op would be faster)
        # This is a placeholder that can be replaced with native op

        # Flatten batch for processing
        flat_queries = tf.reshape(projected_context, [-1, self.rank])  # [B*L, r]
        num_queries = tf.shape(flat_queries)[0]

        # Convert tensors for numpy processing (TODO: move to C++ op)
        queries_np = flat_queries.numpy()

        # Compute distance tables and ADC
        top_k * self.pq_rerank_factor
        candidates_np = np.zeros((num_queries.numpy(), top_k), dtype=np.int32)
        scores_np = np.zeros((num_queries.numpy(), top_k), dtype=np.float32)

        subvec_dim = self._pq_centroids.shape[2]

        for q in range(num_queries.numpy()):
            query = queries_np[q]

            # Build distance table
            dist_table = np.zeros((self.pq_num_subvectors, self.pq_num_centroids), dtype=np.float32)
            for m in range(self.pq_num_subvectors):
                start = m * subvec_dim
                end = min(start + subvec_dim, self.rank)
                if start >= self.rank:
                    continue

                subvec = np.zeros(subvec_dim, dtype=np.float32)
                subvec[: end - start] = query[start:end]

                # Distance to each centroid
                diffs = subvec - self._pq_centroids[m]  # [K_pq, subvec_dim]
                dist_table[m] = np.sum(diffs**2, axis=1)

            # ADC distances
            adc_dists = np.zeros(self.vocab_size, dtype=np.float32)
            for v in range(self.vocab_size):
                for m in range(self.pq_num_subvectors):
                    adc_dists[v] += dist_table[m, self._pq_codes[v, m]]

            # Get top-K candidates (smallest distance = closest)
            top_indices = np.argpartition(adc_dists, top_k)[:top_k]
            top_indices = top_indices[np.argsort(adc_dists[top_indices])]

            candidates_np[q] = top_indices

            # Re-rank with exact U dot products
            U_np = self.U.numpy()
            for k, v in enumerate(top_indices):
                dot = np.dot(query, U_np[v])
                q_norm = np.linalg.norm(query)
                u_norm = np.linalg.norm(U_np[v])
                scores_np[q, k] = (dot / (q_norm * u_norm + 1e-8) + 1) * 0.5

        # Reshape back to [batch, seq_len, top_k]
        candidates = tf.reshape(
            tf.constant(candidates_np, dtype=tf.int32), [batch_size, seq_len, top_k]
        )
        scores = tf.reshape(tf.constant(scores_np, dtype=tf.float32), [batch_size, seq_len, top_k])

        return candidates, scores

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "pq_num_subvectors": self.pq_num_subvectors,
                "pq_num_centroids": self.pq_num_centroids,
                "pq_rerank_factor": self.pq_rerank_factor,
            }
        )
        return config
