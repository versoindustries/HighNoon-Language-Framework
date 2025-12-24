# highnoon/models/reasoning/self_consistency.py
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

"""Self-Consistency Verification Module.

This module implements self-consistency verification for reasoning outputs,
inspired by DeepSeek-R1's self-verification mechanism. The model generates
multiple reasoning paths and verifies their consistency.

Key Features:
    - Multiple path verification for reasoning confidence
    - Threshold-based consistency gating
    - O(d) verification per output position
    - CPU-friendly without quadratic complexity

CONSTRAINT: All computations use float32/64 precision only. No quantization.

Research Reference: DeepSeek-R1 Self-Verification
"""

from __future__ import annotations

import logging
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.config import CONSISTENCY_THRESHOLD, USE_SELF_CONSISTENCY

logger = logging.getLogger(__name__)


class SelfConsistencyVerifier(layers.Layer):
    """Verifies reasoning outputs against multiple paths for consistency.

    This layer compares multiple reasoning outputs to detect inconsistencies
    and produce confidence-weighted results.

    Attributes:
        embedding_dim: Dimension of embeddings.
        num_verification_heads: Number of parallel verification projections.
        threshold: Consistency threshold for gating.

    Complexity: O(d) per position, where d = embedding dimension.

    Example:
        >>> verifier = SelfConsistencyVerifier(embedding_dim=512)
        >>> reasoning_paths = [path1, path2, path3]  # Each [B, L, D]
        >>> verified, confidence = verifier(reasoning_paths)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_verification_heads: int = 4,
        threshold: float = CONSISTENCY_THRESHOLD,
        name: str = "self_consistency",
        **kwargs: Any,
    ) -> None:
        """Initialize SelfConsistencyVerifier.

        Args:
            embedding_dim: Embedding dimension.
            num_verification_heads: Number of parallel verification heads.
            threshold: Consistency threshold (0-1). Higher = stricter.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.

        Raises:
            ValueError: If embedding_dim is not positive.
            ValueError: If num_verification_heads is not positive.
            ValueError: If threshold is not in (0, 1].
        """
        super().__init__(name=name, **kwargs)

        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if num_verification_heads < 1:
            raise ValueError(
                f"num_verification_heads must be positive, got {num_verification_heads}"
            )
        if not (0 < threshold <= 1):
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")

        self.embedding_dim = embedding_dim
        self.num_verification_heads = num_verification_heads
        self.threshold = threshold

        # Initialize storage for sub-layers
        self.verification_projections = []
        self.aggregation = None
        self.confidence_projection = None
        self.layer_norm = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create layer weights."""
        if self.built:
            return

        # Verification head projections
        self.verification_projections = [
            layers.Dense(
                self.embedding_dim // self.num_verification_heads,
                name=f"{self.name}_proj_{i}",
            )
            for i in range(self.num_verification_heads)
        ]

        # Aggregation projection
        self.aggregation = layers.Dense(
            self.embedding_dim,
            name=f"{self.name}_aggregation",
        )

        # Confidence projection
        self.confidence_projection = layers.Dense(
            1,
            activation="sigmoid",
            name=f"{self.name}_confidence",
        )

        # Layer normalization
        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{self.name}_norm",
        )

        super().build(input_shape)

    def compute_pairwise_agreement(
        self,
        paths: list[tf.Tensor],
    ) -> tf.Tensor:
        """Compute agreement scores between all pairs of paths.

        Args:
            paths: List of reasoning paths, each [batch, seq_len, dim].

        Returns:
            Agreement matrix [batch, seq_len, num_paths, num_paths].
        """
        len(paths)
        tf.shape(paths[0])[0]
        tf.shape(paths[0])[1]

        # Stack paths: [batch, seq_len, num_paths, dim]
        stacked = tf.stack(paths, axis=2)

        # Normalize for cosine similarity
        normalized = tf.math.l2_normalize(stacked, axis=-1)

        # Compute pairwise cosine similarity
        # [B, L, P, D] @ [B, L, D, P] -> [B, L, P, P]
        agreement = tf.matmul(normalized, normalized, transpose_b=True)

        return agreement

    def compute_consistency_score(
        self,
        agreement: tf.Tensor,
    ) -> tf.Tensor:
        """Compute overall consistency score from agreement matrix.

        Args:
            agreement: Agreement matrix [batch, seq_len, num_paths, num_paths].

        Returns:
            Consistency scores [batch, seq_len].
        """
        # Average off-diagonal elements (exclude self-agreement)
        num_paths = tf.shape(agreement)[2]

        # Create mask for off-diagonal elements
        mask = 1.0 - tf.eye(num_paths, dtype=tf.float32)
        mask = mask[tf.newaxis, tf.newaxis, :, :]

        # Masked mean
        masked_agreement = agreement * mask
        sum_agreement = tf.reduce_sum(masked_agreement, axis=[-2, -1])
        count = tf.cast(num_paths * (num_paths - 1), tf.float32)

        consistency = sum_agreement / (count + 1e-8)

        return consistency

    def call(
        self,
        reasoning_paths: list[tf.Tensor] | tf.Tensor,
        return_confidence: bool = True,
        training: bool = False,
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
        """Verify consistency of multiple reasoning paths.

        Args:
            reasoning_paths: List of path tensors or stacked tensor.
                Each path: [batch, seq_len, embedding_dim].
                If stacked: [batch, seq_len, num_paths, embedding_dim].
            return_confidence: Whether to return confidence scores.
            training: Whether in training mode.

        Returns:
            If return_confidence is True:
                Tuple of (verified_output, confidence_scores)
            Else:
                verified_output only

            verified_output: [batch, seq_len, embedding_dim]
            confidence_scores: [batch, seq_len]
        """
        # Handle both list and stacked tensor input
        if isinstance(reasoning_paths, tf.Tensor):
            if len(reasoning_paths.shape) == 4:
                # Stacked tensor: [B, L, P, D] -> list of [B, L, D]
                num_paths = reasoning_paths.shape[2]
                if num_paths is None:
                    # Dynamic num_paths - use tf.unstack for graph mode
                    paths = tf.unstack(reasoning_paths, axis=2)
                else:
                    paths = [reasoning_paths[:, :, i, :] for i in range(num_paths)]
            elif len(reasoning_paths.shape) == 3:
                # Single path: [B, L, D] - wrap in list
                # Self-consistency with one path = trivially consistent
                paths = [reasoning_paths]
            else:
                raise ValueError(
                    f"reasoning_paths tensor must be 3D or 4D, got shape {reasoning_paths.shape}"
                )
        else:
            paths = reasoning_paths

        # Cast to float32 if needed - use tf.map_fn for graph mode compatibility
        if len(paths) > 0:
            paths = [tf.cast(p, tf.float32) for p in paths]

        # Compute pairwise agreement
        agreement = self.compute_pairwise_agreement(paths)

        # Compute consistency scores
        consistency = self.compute_consistency_score(agreement)

        # Apply verification heads to each path
        verified_paths = []
        for _i, path in enumerate(paths):
            head_outputs = []
            for proj in self.verification_projections:
                head_outputs.append(proj(path))
            # Concatenate heads
            combined = tf.concat(head_outputs, axis=-1)
            verified_paths.append(combined)

        # Stack verified paths: [B, L, P, D]
        verified_stack = tf.stack(verified_paths, axis=2)

        # Weight paths by consistency-informed attention
        # Higher consistency = more uniform weighting
        # Lower consistency = favor majority
        path_weights = tf.nn.softmax(
            consistency[:, :, tf.newaxis] * agreement[..., 0],
            axis=-1,
        )  # [B, L, P]

        # Weighted combination
        weighted = tf.reduce_sum(
            verified_stack * path_weights[:, :, :, tf.newaxis],
            axis=2,
        )  # [B, L, D]

        # Project back to full dimension
        output = self.aggregation(weighted)
        output = self.layer_norm(output)

        # Compute confidence from consistency and threshold
        confidence = consistency

        # Apply threshold gating: below threshold = reduced confidence
        gated_confidence = tf.where(
            confidence >= self.threshold,
            confidence,
            confidence * 0.5,  # Reduce confidence for inconsistent outputs
        )

        # Update output for low-confidence positions (optional smoothing)
        # Could blend with original input for safety

        if return_confidence:
            return output, gated_confidence
        return output

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_verification_heads": self.num_verification_heads,
                "threshold": self.threshold,
            }
        )
        return config


class ConsistencyGatedOutput(layers.Layer):
    """Applies consistency-based gating to final output.

    Uses consistency confidence to gate between reasoning output
    and a safer fallback (e.g., original hidden state).

    Example:
        >>> gater = ConsistencyGatedOutput(embedding_dim=512)
        >>> output = gater(reasoning_output, fallback, confidence)
    """

    def __init__(
        self,
        embedding_dim: int,
        threshold: float = CONSISTENCY_THRESHOLD,
        name: str = "consistency_gate",
        **kwargs: Any,
    ) -> None:
        """Initialize ConsistencyGatedOutput.

        Args:
            embedding_dim: Embedding dimension.
            threshold: Confidence threshold for full pass-through.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.
        """
        super().__init__(name=name, **kwargs)

        self.embedding_dim = embedding_dim
        self.threshold = threshold

        # Learned blending based on confidence
        self.blend_projection = layers.Dense(
            embedding_dim,
            name=f"{name}_blend",
        )

    def call(
        self,
        reasoning_output: tf.Tensor,
        fallback: tf.Tensor,
        confidence: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply consistency-gated blending.

        Args:
            reasoning_output: Output from reasoning [batch, seq_len, dim].
            fallback: Fallback output (e.g., original) [batch, seq_len, dim].
            confidence: Consistency confidence [batch, seq_len].
            training: Whether in training mode.

        Returns:
            Gated output [batch, seq_len, dim].
        """
        # Expand confidence for broadcasting
        confidence_expanded = confidence[:, :, tf.newaxis]

        # Smooth gating: high confidence = mostly reasoning, low = blend with fallback
        gate = tf.minimum(confidence_expanded / self.threshold, 1.0)

        # Blend outputs
        blended = gate * reasoning_output + (1 - gate) * fallback

        return self.blend_projection(blended)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "threshold": self.threshold,
            }
        )
        return config


def create_self_consistency_verifier(
    embedding_dim: int,
    **kwargs: Any,
) -> SelfConsistencyVerifier | layers.Layer:
    """Factory function for creating self-consistency verifier.

    Returns identity layer if USE_SELF_CONSISTENCY is False.

    Args:
        embedding_dim: Embedding dimension.
        **kwargs: Additional arguments for SelfConsistencyVerifier.

    Returns:
        SelfConsistencyVerifier if enabled, identity layer otherwise.
    """
    if not USE_SELF_CONSISTENCY:
        logger.debug("Self-consistency disabled (USE_SELF_CONSISTENCY=False)")
        # Return layer that just returns first path
        return layers.Lambda(
            lambda x: x[0] if isinstance(x, list) else x[:, :, 0, :],
            name="self_consistency_identity",
        )

    return SelfConsistencyVerifier(embedding_dim=embedding_dim, **kwargs)


__all__ = [
    "SelfConsistencyVerifier",
    "ConsistencyGatedOutput",
    "create_self_consistency_verifier",
]
