# highnoon/inference/speculative.py
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

"""Phase 13.3: Speculative Decoding.

This module provides speculative decoding infrastructure for faster
autoregressive generation using a small draft model.

Speculative decoding works by:
1. Generate K tokens with a small, fast draft model
2. Verify all K tokens in parallel with the large target model
3. Accept verified tokens, reject and resample from target if needed

This can achieve 2-3x speedup when draft model alignment is good.

Example:
    >>> from highnoon.inference.speculative import SpeculativeGenerator
    >>>
    >>> generator = SpeculativeGenerator(
    ...     target_model=large_model,
    ...     draft_model=small_model,
    ...     num_speculative_tokens=4,
    ... )
    >>> output = generator.generate(input_ids, max_length=100)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import tensorflow as tf

from highnoon.config import SPECULATIVE_ENABLED, SPECULATIVE_TEMPERATURE, SPECULATIVE_TOKENS

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeDecodingConfig:
    """Configuration for speculative decoding.

    Attributes:
        num_speculative_tokens: Number of tokens to speculate (K).
        temperature: Sampling temperature for both models.
        top_k: Top-k filtering.
        top_p: Nucleus sampling threshold.
        use_sampling: Use sampling vs greedy decoding.
    """

    num_speculative_tokens: int = SPECULATIVE_TOKENS
    temperature: float = SPECULATIVE_TEMPERATURE
    top_k: int = 50
    top_p: float = 0.9
    use_sampling: bool = True


class SpeculativeGenerator:
    """Speculative decoding generator for faster autoregressive generation.

    Uses a smaller draft model to speculate multiple tokens ahead,
    then verifies with the target model in a single forward pass.

    The algorithm:
    1. Draft model generates K tokens autoregressively
    2. Target model processes all K tokens in parallel
    3. For each position, check if draft matches target distribution
    4. Accept matching tokens, reject and resample first mismatch
    5. Repeat from step 1 with accepted tokens

    This achieves speedup when K tokens can be verified faster than
    generating them one by one with the target model.
    """

    def __init__(
        self,
        target_model: tf.keras.Model,
        draft_model: tf.keras.Model,
        config: SpeculativeDecodingConfig | None = None,
    ):
        """Initialize speculative generator.

        Args:
            target_model: Large, accurate target model.
            draft_model: Small, fast draft model.
            config: Configuration for speculative decoding.
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = config or SpeculativeDecodingConfig()

        # Get vocab size from models
        self.vocab_size = getattr(target_model, "vocab_size", 32000)

        # Statistics tracking
        self.total_accepted = 0
        self.total_speculated = 0

    def generate(
        self,
        input_ids: tf.Tensor,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> tf.Tensor:
        """Generate tokens using speculative decoding.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Additional generation parameters.

        Returns:
            Generated token IDs [batch, seq_len + generated].
        """
        if not SPECULATIVE_ENABLED:
            # Fall back to standard generation
            return self._standard_generate(input_ids, max_new_tokens)

        temperature = kwargs.get("temperature", self.config.temperature)
        top_k = kwargs.get("top_k", self.config.top_k)

        generated = input_ids
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # Step 1: Draft speculation
            draft_tokens, draft_probs = self._draft_speculate(
                generated,
                num_tokens=self.config.num_speculative_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            # Step 2: Target verification
            num_accepted, next_token = self._verify_and_accept(
                generated,
                draft_tokens,
                draft_probs,
                temperature=temperature,
                top_k=top_k,
            )

            # Step 3: Update sequence with accepted tokens
            if num_accepted > 0:
                accepted = draft_tokens[:, :num_accepted]
                generated = tf.concat([generated, accepted], axis=1)
                tokens_generated += num_accepted

            # Add the resampled/bonus token
            if next_token is not None and tokens_generated < max_new_tokens:
                generated = tf.concat([generated, next_token], axis=1)
                tokens_generated += 1

            # Update statistics
            self.total_accepted += num_accepted
            self.total_speculated += self.config.num_speculative_tokens

            # Check for EOS
            eos_id = getattr(self.target_model, "eos_token_id", None)
            if eos_id is not None:
                if tf.reduce_any(generated[:, -1] == eos_id):
                    break

        return generated

    def _draft_speculate(
        self,
        input_ids: tf.Tensor,
        num_tokens: int,
        temperature: float,
        top_k: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Generate speculative tokens with draft model.

        Args:
            input_ids: Current sequence [batch, seq_len].
            num_tokens: Number of tokens to speculate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.

        Returns:
            Tuple of (draft_tokens, draft_probs).
        """
        current = input_ids
        draft_tokens = []
        draft_probs = []

        for _ in range(num_tokens):
            # Get draft model logits
            outputs = self.draft_model(current, training=False)
            if hasattr(outputs, "logits"):
                logits = outputs.logits[:, -1, :]
            else:
                logits = outputs[:, -1, :]

            # Sample next token
            next_token, probs = self._sample_with_probs(logits, temperature, top_k)

            draft_tokens.append(next_token)
            draft_probs.append(probs)

            # Extend sequence
            current = tf.concat([current, next_token], axis=1)

        # Stack drafts
        draft_tokens = tf.concat(draft_tokens, axis=1)  # [batch, num_tokens]
        draft_probs = tf.stack(draft_probs, axis=1)  # [batch, num_tokens, vocab]

        return draft_tokens, draft_probs

    def _verify_and_accept(
        self,
        prefix: tf.Tensor,
        draft_tokens: tf.Tensor,
        draft_probs: tf.Tensor,
        temperature: float,
        top_k: int,
    ) -> tuple[int, tf.Tensor | None]:
        """Verify draft tokens with target model.

        Args:
            prefix: Original prefix [batch, prefix_len].
            draft_tokens: Draft tokens [batch, num_speculative].
            draft_probs: Draft probabilities [batch, num_speculative, vocab].
            temperature: Sampling temperature.
            top_k: Top-k filtering.

        Returns:
            Tuple of (num_accepted, resampled_token).
        """
        num_speculative = draft_tokens.shape[1]

        # Concatenate prefix with all draft tokens for parallel verification
        full_sequence = tf.concat([prefix, draft_tokens], axis=1)

        # Get target model logits for all positions
        outputs = self.target_model(full_sequence, training=False)
        if hasattr(outputs, "logits"):
            all_logits = outputs.logits
        else:
            all_logits = outputs

        # Get logits at positions corresponding to draft tokens
        prefix_len = prefix.shape[1]
        target_logits = all_logits[:, prefix_len - 1 : -1, :]  # [batch, num_spec, vocab]

        # Apply temperature
        target_logits_scaled = target_logits / temperature
        target_probs = tf.nn.softmax(target_logits_scaled, axis=-1)

        # Verify each draft token using rejection sampling criterion
        num_accepted = 0

        for i in range(num_speculative):
            draft_token = draft_tokens[:, i : i + 1]  # [batch, 1]

            # Get probabilities for the drafted token
            batch_indices = tf.range(tf.shape(draft_token)[0])
            token_indices = tf.squeeze(draft_token, axis=1)

            p_target = tf.gather_nd(
                target_probs[:, i, :], tf.stack([batch_indices, token_indices], axis=1)
            )
            p_draft = tf.gather_nd(
                draft_probs[:, i, :], tf.stack([batch_indices, token_indices], axis=1)
            )

            # Acceptance criterion: accept if p_target >= p_draft
            # Or with probability p_target / p_draft
            accept_prob = tf.minimum(1.0, p_target / (p_draft + 1e-8))

            # Sample acceptance
            u = tf.random.uniform(tf.shape(accept_prob))
            accepted = tf.reduce_all(u < accept_prob)

            if accepted:
                num_accepted += 1
            else:
                # Reject this and all subsequent tokens
                break

        # Get bonus token from target distribution at rejection point
        bonus_idx = prefix_len + num_accepted - 1
        if bonus_idx < tf.shape(all_logits)[1]:
            bonus_logits = all_logits[:, bonus_idx, :]
            bonus_token, _ = self._sample_with_probs(bonus_logits, temperature, top_k)
        else:
            bonus_token = None

        return num_accepted, bonus_token

    def _sample_with_probs(
        self,
        logits: tf.Tensor,
        temperature: float,
        top_k: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Sample from logits and return token with full probability distribution.

        Args:
            logits: Logits [batch, vocab_size].
            temperature: Sampling temperature.
            top_k: Top-k filtering.

        Returns:
            Tuple of (sampled_token [batch, 1], probs [batch, vocab]).
        """
        # Apply temperature
        logits_scaled = logits / max(temperature, 1e-8)

        # Top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = tf.math.top_k(logits_scaled, k=top_k)
            # Create mask
            min_val = tf.reduce_min(top_k_logits, axis=-1, keepdims=True)
            logits_scaled = tf.where(
                logits_scaled < min_val, tf.ones_like(logits_scaled) * -1e9, logits_scaled
            )

        # Compute probabilities
        probs = tf.nn.softmax(logits_scaled, axis=-1)

        # Sample
        if self.config.use_sampling:
            sampled = tf.random.categorical(logits_scaled, num_samples=1)
        else:
            sampled = tf.argmax(logits_scaled, axis=-1, output_type=tf.int32)
            sampled = tf.expand_dims(sampled, axis=-1)

        return tf.cast(sampled, tf.int32), probs

    def _standard_generate(
        self,
        input_ids: tf.Tensor,
        max_new_tokens: int,
    ) -> tf.Tensor:
        """Standard autoregressive generation (fallback)."""
        return self.target_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
        )

    def get_acceptance_rate(self) -> float:
        """Get the acceptance rate of speculative tokens."""
        if self.total_speculated == 0:
            return 0.0
        return self.total_accepted / self.total_speculated

    def reset_statistics(self) -> None:
        """Reset acceptance statistics."""
        self.total_accepted = 0
        self.total_speculated = 0


class DraftModelFactory:
    """Factory for creating draft models from target models.

    Creates smaller draft models by:
    - Reducing number of layers
    - Reducing hidden dimensions
    - Sharing vocabulary embeddings
    """

    @staticmethod
    def create_draft_model(
        target_model: tf.keras.Model,
        size_ratio: float = 0.25,
    ) -> tf.keras.Model:
        """Create a draft model from a target model.

        Args:
            target_model: Target model to derive draft from.
            size_ratio: Size of draft relative to target (0.0-1.0).

        Returns:
            Smaller draft model.

        Note:
            This is a placeholder. Actual implementation depends on
            model architecture specifics.
        """
        # Get target model dimensions
        getattr(target_model, "vocab_size", 32000)
        embedding_dim = getattr(target_model, "embedding_dim", 512)
        num_blocks = getattr(target_model, "_num_reasoning_blocks", 6)

        # Scale down
        draft_dim = max(64, int(embedding_dim * size_ratio))
        draft_blocks = max(1, int(num_blocks * size_ratio))

        logger.info(
            f"Creating draft model: dim={draft_dim}, blocks={draft_blocks} "
            f"(target: dim={embedding_dim}, blocks={num_blocks})"
        )

        # Would create actual model here
        # For now, return None as placeholder
        return None


__all__ = [
    "SpeculativeDecodingConfig",
    "SpeculativeGenerator",
    "DraftModelFactory",
]
