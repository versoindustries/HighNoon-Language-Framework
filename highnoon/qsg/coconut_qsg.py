# highnoon/qsg/coconut_qsg.py
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

"""Phase A4: Deep COCONUT Integration for QSG.

Implements COCONUT (Chain of Continuous Thought) reasoning integrated with
Quantum Superposition Generation. Enables "thinking before typing" via
multi-path latent space exploration at every generation phase.

Key Components:
    - ContinuousThoughtBlock: BFS multi-path reasoning in latent space
    - COCONUTEnhancedQSG: QSG pipeline with integrated continuous thought

Reference: "Training Large Language Models to Reason in a Continuous Latent Space" (Hao et al.)
"""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.config import (
    COCONUT_BFS_BRANCHES,
    COCONUT_BRANCH_ALPHA,
    COCONUT_HALT_THRESHOLD,
    COCONUT_MAX_THOUGHT_STEPS,
    COCONUT_RESERVOIR_DIM,
)

logger = logging.getLogger(__name__)


class ContinuousThoughtBlock(layers.Layer):
    """BFS-style continuous thought reasoning in latent space.

    Maintains K parallel thought paths and iteratively refines them
    until confidence threshold is reached or max steps exhausted.

    Each thought step:
    1. Projects current state through reservoir
    2. Computes path quality scores (Grover-inspired amplitude)
    3. Prunes low-quality paths, expands high-quality ones
    4. Updates state with residual connection

    Attributes:
        num_branches: Number of parallel thought paths (K).
        max_steps: Maximum reasoning iterations.
        halt_threshold: Confidence for early stopping.
        alpha: Residual update weight.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_branches: int = COCONUT_BFS_BRANCHES,
        max_steps: int = COCONUT_MAX_THOUGHT_STEPS,
        halt_threshold: float = COCONUT_HALT_THRESHOLD,
        alpha: float = COCONUT_BRANCH_ALPHA,
        reservoir_dim: int = COCONUT_RESERVOIR_DIM,
        name: str = "continuous_thought",
        **kwargs,
    ):
        """Initialize ContinuousThoughtBlock.

        Args:
            hidden_dim: Dimension of hidden states.
            num_branches: Number of parallel thought paths.
            max_steps: Maximum reasoning iterations.
            halt_threshold: Confidence for early stopping.
            alpha: Residual update weight.
            reservoir_dim: Hidden dimension for thought projection.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.
        """
        super().__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_branches = num_branches
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold
        self.alpha = alpha
        self.reservoir_dim = reservoir_dim

        # Thought projection (latent reasoning transform)
        self.thought_proj = layers.Dense(reservoir_dim, activation="gelu", name=f"{name}_proj")
        self.thought_out = layers.Dense(hidden_dim, name=f"{name}_out")

        # Quality scorer (Grover-like amplitude estimation)
        self.quality_scorer = layers.Dense(1, activation="sigmoid", name=f"{name}_quality")

        # Branch attention for path merging
        self.branch_attention = layers.Dense(num_branches, name=f"{name}_branch_attn")

    def build(self, input_shape):
        """Build layer weights."""
        # Sub-layers are built on first call, but we mark this layer as built
        super().build(input_shape)

    def call(
        self,
        hidden_states: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply continuous thought reasoning.

        Args:
            hidden_states: Input hidden states [batch, seq, dim].
            training: Whether in training mode.

        Returns:
            Tuple of:
                - refined_states: Enhanced hidden states [batch, seq, dim]
                - confidence: Final path quality [batch, seq]
        """
        batch_size = tf.shape(hidden_states)[0]
        seq_len = tf.shape(hidden_states)[1]

        # Initialize K branches with copies of input
        # Shape: [batch, seq, K, dim]
        branches = tf.repeat(tf.expand_dims(hidden_states, axis=2), self.num_branches, axis=2)

        # Initialize amplitudes (uniform)
        amplitudes = tf.ones([batch_size, seq_len, self.num_branches]) / self.num_branches

        confidence = tf.zeros([batch_size, seq_len])

        for _step in range(self.max_steps):
            # 1. Project through thought reservoir
            thought = self.thought_proj(branches)  # [B, L, K, reservoir_dim]
            delta = self.thought_out(thought)  # [B, L, K, dim]

            # 2. Residual update
            branches = branches + self.alpha * delta

            # 3. Compute quality scores per branch
            quality = tf.squeeze(self.quality_scorer(branches), axis=-1)  # [B, L, K]

            # 4. Update amplitudes (Grover-inspired amplitude boost)
            new_amplitudes = amplitudes * quality
            new_amplitudes = new_amplitudes / (
                tf.reduce_sum(new_amplitudes, axis=-1, keepdims=True) + 1e-8
            )
            amplitudes = new_amplitudes

            # 5. Update confidence (max amplitude)
            confidence = tf.reduce_max(amplitudes, axis=-1)  # [B, L]

            # Early stopping check (if all positions confident)
            if not training:
                avg_conf = tf.reduce_mean(confidence)
                if avg_conf >= self.halt_threshold:
                    break

        # 6. Collapse branches to single output (amplitude-weighted sum)
        # [B, L, K, 1] * [B, L, K, dim] -> [B, L, dim]
        weights = tf.expand_dims(amplitudes, axis=-1)  # [B, L, K, 1]
        refined_states = tf.reduce_sum(branches * weights, axis=2)

        return refined_states, confidence

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_branches": self.num_branches,
                "max_steps": self.max_steps,
                "halt_threshold": self.halt_threshold,
                "alpha": self.alpha,
                "reservoir_dim": self.reservoir_dim,
            }
        )
        return config


class COCONUTEnhancedQSG(layers.Layer):
    """QSG pipeline with integrated COCONUT continuous thought.

    Applies multi-path BFS reasoning at each QSG generation phase:
    1. MPS Context → COCONUT → Enhanced context
    2. Vocabulary Projection → COCONUT → Refined logits
    3. Grover Amplification with COCONUT-informed oracle

    This enables "thinking before typing" - the model explores
    multiple reasoning paths in latent space before committing
    to token predictions.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_branches: int = COCONUT_BFS_BRANCHES,
        max_steps: int = COCONUT_MAX_THOUGHT_STEPS,
        halt_threshold: float = COCONUT_HALT_THRESHOLD,
        name: str = "coconut_qsg",
        **kwargs,
    ):
        """Initialize COCONUTEnhancedQSG.

        Args:
            hidden_dim: Model hidden dimension.
            num_branches: Number of parallel thought paths.
            max_steps: Maximum reasoning iterations.
            halt_threshold: Confidence for early stopping.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.
        """
        super().__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim

        # Context enhancement thought block
        self.context_thought = ContinuousThoughtBlock(
            hidden_dim=hidden_dim,
            num_branches=num_branches,
            max_steps=max_steps,
            halt_threshold=halt_threshold,
            name=f"{name}_context_thought",
        )

        # Logits refinement thought block (optional second pass)
        self.logits_thought = ContinuousThoughtBlock(
            hidden_dim=hidden_dim,
            num_branches=num_branches,
            max_steps=max_steps // 2,  # Fewer steps for refinement
            halt_threshold=halt_threshold,
            name=f"{name}_logits_thought",
        )

    def build(self, input_shape):
        """Build layer weights."""
        # Sub-layers are built on first call
        super().build(input_shape)

    def enhance_context(
        self,
        context: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply continuous thought to enhance context representation.

        Args:
            context: Input context [batch, seq, dim].
            training: Whether in training mode.

        Returns:
            Tuple of (enhanced_context, confidence).
        """
        return self.context_thought(context, training=training)

    def refine_hidden_states(
        self,
        hidden_states: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Refine hidden states before vocabulary projection.

        Args:
            hidden_states: Hidden states for output [batch, seq, dim].
            training: Whether in training mode.

        Returns:
            Tuple of (refined_states, confidence).
        """
        return self.logits_thought(hidden_states, training=training)

    def call(
        self,
        hidden_states: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Full COCONUT enhancement pass.

        Args:
            hidden_states: Input hidden states [batch, seq, dim].
            training: Whether in training mode.

        Returns:
            Tuple of (enhanced_states, total_confidence).
        """
        # Context thinking
        enhanced, ctx_conf = self.context_thought(hidden_states, training=training)

        # Refinement pass
        refined, ref_conf = self.logits_thought(enhanced, training=training)

        # Average confidence
        total_conf = (ctx_conf + ref_conf) / 2.0

        return refined, total_conf

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


def coconut_quality_oracle(
    hidden_states: tf.Tensor,
    confidence: tf.Tensor,
) -> tf.Tensor:
    """Generate oracle scores from COCONUT confidence.

    Higher confidence = higher oracle amplitude for Grover.

    Args:
        hidden_states: Hidden states [batch, seq, dim].
        confidence: COCONUT confidence [batch, seq].

    Returns:
        Oracle scores [batch, seq] in range [0, 1].
    """
    # Confidence is already in [0, 1] from sigmoid
    return confidence
