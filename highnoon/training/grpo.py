# highnoon/training/grpo.py
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

"""Group Relative Policy Optimization (GRPO) Training Module.

This module implements GRPO for emergent reasoning capabilities, inspired by
DeepSeek-R1. GRPO enables reinforcement learning without explicit reward models
by using group-relative comparisons.

Key Features:
    - Group-relative reward computation (no separate reward model needed)
    - Multiple response sampling per prompt
    - Policy gradient with baseline subtraction
    - O(n) linear complexity for CPU-first training

CONSTRAINT: All computations use float32/64 precision only. No quantization.

Research Reference: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.config import GRPO_NUM_SAMPLES, USE_GRPO_TRAINING

logger = logging.getLogger(__name__)


class GRPOSampler(layers.Layer):
    """Generates multiple response samples per prompt for GRPO training.

    This layer samples K responses for each input prompt, enabling
    group-relative reward computation without an external reward model.

    Attributes:
        num_samples: Number of response samples per prompt.
        temperature: Sampling temperature for diversity.
        top_k: Optional top-k filtering for sampling.

    Example:
        >>> sampler = GRPOSampler(num_samples=4, temperature=0.8)
        >>> responses, log_probs = sampler(model, prompts)
        >>> # responses: [batch, num_samples, seq_len, vocab]
        >>> # log_probs: [batch, num_samples]
    """

    def __init__(
        self,
        num_samples: int = GRPO_NUM_SAMPLES,
        temperature: float = 1.0,
        top_k: int | None = None,
        name: str = "grpo_sampler",
        **kwargs: Any,
    ) -> None:
        """Initialize GRPOSampler.

        Args:
            num_samples: Number of response samples per prompt.
            temperature: Sampling temperature (higher = more diverse).
            top_k: Optional top-k filtering. None means no filtering.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.

        Raises:
            ValueError: If num_samples is not positive.
            ValueError: If temperature is not positive.
        """
        super().__init__(name=name, **kwargs)

        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be positive or None, got {top_k}")

        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k

    def sample_responses(
        self,
        model: tf.keras.Model,
        prompts: tf.Tensor,
        max_length: int = 128,
        training: bool = True,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Sample multiple responses for each prompt.

        Args:
            model: Language model to sample from.
            prompts: Input prompts [batch, prompt_len].
            max_length: Maximum response length.
            training: Whether in training mode.

        Returns:
            Tuple of:
                - responses: Generated token IDs [batch, num_samples, max_length]
                - log_probs: Log probabilities per sample [batch, num_samples]
        """
        tf.shape(prompts)[0]

        all_responses = []
        all_log_probs = []

        for _ in range(self.num_samples):
            # Get model logits
            logits = model(prompts, training=training)

            # Apply temperature
            scaled_logits = logits / self.temperature

            # Apply top-k filtering if specified
            if self.top_k is not None:
                top_k_logits, top_k_indices = tf.math.top_k(scaled_logits, k=self.top_k)
                # Mask out non-top-k values
                mask = tf.scatter_nd(
                    tf.expand_dims(top_k_indices, -1),
                    tf.ones_like(top_k_logits),
                    tf.shape(scaled_logits),
                )
                scaled_logits = tf.where(mask > 0, scaled_logits, tf.float32.min)

            # Sample from categorical distribution
            # Cast to float32 to ensure precision
            probs = tf.nn.softmax(tf.cast(scaled_logits, tf.float32), axis=-1)
            samples = tf.random.categorical(tf.math.log(probs + 1e-10), num_samples=1)
            samples = tf.squeeze(samples, axis=-1)

            # Compute log probabilities for sampled tokens
            sample_log_probs = tf.reduce_sum(
                tf.math.log(tf.gather(probs, samples, axis=-1, batch_dims=1) + 1e-10),
                axis=-1,
            )

            all_responses.append(samples)
            all_log_probs.append(sample_log_probs)

        # Stack samples: [batch, num_samples, seq_len]
        responses = tf.stack(all_responses, axis=1)
        # Stack log_probs: [batch, num_samples]
        log_probs = tf.stack(all_log_probs, axis=1)

        return responses, log_probs

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "num_samples": self.num_samples,
                "temperature": self.temperature,
                "top_k": self.top_k,
            }
        )
        return config


class GRPOLoss(layers.Layer):
    """Computes GRPO loss using group-relative rewards.

    GRPO compares responses within each group (same prompt) to compute
    relative advantages, eliminating the need for an external reward model.

    The loss is:
        L = -E[A(r) * log π(r|p)]
    where A(r) is the advantage computed relative to the group mean.

    Attributes:
        kl_coeff: KL penalty coefficient for policy regularization.
        entropy_coeff: Entropy bonus coefficient for exploration.

    Example:
        >>> loss_fn = GRPOLoss(kl_coeff=0.01)
        >>> loss = loss_fn(log_probs, rewards, ref_log_probs)
    """

    def __init__(
        self,
        kl_coeff: float = 0.01,
        entropy_coeff: float = 0.001,
        name: str = "grpo_loss",
        **kwargs: Any,
    ) -> None:
        """Initialize GRPOLoss.

        Args:
            kl_coeff: KL divergence penalty coefficient.
            entropy_coeff: Entropy bonus coefficient.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.

        Raises:
            ValueError: If kl_coeff is negative.
            ValueError: If entropy_coeff is negative.
        """
        super().__init__(name=name, **kwargs)

        if kl_coeff < 0:
            raise ValueError(f"kl_coeff must be non-negative, got {kl_coeff}")
        if entropy_coeff < 0:
            raise ValueError(f"entropy_coeff must be non-negative, got {entropy_coeff}")

        self.kl_coeff = kl_coeff
        self.entropy_coeff = entropy_coeff

    def compute_group_advantages(
        self,
        rewards: tf.Tensor,
    ) -> tf.Tensor:
        """Compute group-relative advantages.

        Subtracts group mean and normalizes by group std to compute
        relative advantages within each prompt group.

        Args:
            rewards: Reward scores [batch, num_samples].

        Returns:
            Group-relative advantages [batch, num_samples].
        """
        # Cast to float32 for numerical stability
        rewards = tf.cast(rewards, tf.float32)

        # Compute group statistics
        group_mean = tf.reduce_mean(rewards, axis=1, keepdims=True)
        group_std = tf.math.reduce_std(rewards, axis=1, keepdims=True)

        # Normalize: (reward - mean) / (std + eps)
        advantages = (rewards - group_mean) / (group_std + 1e-8)

        return advantages

    def call(
        self,
        log_probs: tf.Tensor,
        rewards: tf.Tensor,
        ref_log_probs: tf.Tensor | None = None,
        training: bool = True,
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """Compute GRPO loss.

        Args:
            log_probs: Current policy log probabilities [batch, num_samples].
            rewards: Reward scores for each sample [batch, num_samples].
            ref_log_probs: Reference policy log probs for KL penalty (optional).
            training: Whether in training mode.

        Returns:
            Tuple of:
                - total_loss: Scalar loss value
                - metrics: Dictionary of loss components
        """
        # Cast inputs to float32
        log_probs = tf.cast(log_probs, tf.float32)
        rewards = tf.cast(rewards, tf.float32)

        # Compute group-relative advantages
        advantages = self.compute_group_advantages(rewards)

        # Policy gradient loss: -E[A * log π]
        pg_loss = -tf.reduce_mean(advantages * log_probs)

        # KL penalty (if reference provided)
        kl_loss = tf.constant(0.0, dtype=tf.float32)
        if ref_log_probs is not None:
            ref_log_probs = tf.cast(ref_log_probs, tf.float32)
            # KL(π || π_ref) ≈ log π - log π_ref
            kl_div = tf.reduce_mean(log_probs - ref_log_probs)
            kl_loss = self.kl_coeff * kl_div

        # Entropy bonus (encourage exploration)
        # Approximate entropy from log probs
        entropy = -tf.reduce_mean(log_probs)
        entropy_bonus = -self.entropy_coeff * entropy

        # Total loss
        total_loss = pg_loss + kl_loss + entropy_bonus

        metrics = {
            "grpo/pg_loss": pg_loss,
            "grpo/kl_loss": kl_loss,
            "grpo/entropy": entropy,
            "grpo/advantage_mean": tf.reduce_mean(advantages),
            "grpo/advantage_std": tf.math.reduce_std(advantages),
            "grpo/total_loss": total_loss,
        }

        return total_loss, metrics

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "kl_coeff": self.kl_coeff,
                "entropy_coeff": self.entropy_coeff,
            }
        )
        return config


class GRPOTrainer:
    """Training wrapper for GRPO-based reasoning enhancement.

    Orchestrates the full GRPO training loop including sampling,
    reward computation, and policy optimization.

    Attributes:
        model: Language model to train.
        sampler: GRPO sampler for response generation.
        loss_fn: GRPO loss function.
        optimizer: Optimizer for policy updates.
        reward_fn: Callable that computes rewards for responses.

    Example:
        >>> trainer = GRPOTrainer(
        ...     model=language_model,
        ...     reward_fn=my_reward_function,
        ... )
        >>> metrics = trainer.train_step(prompts)
    """

    def __init__(
        self,
        model: tf.keras.Model,
        reward_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        optimizer: tf.keras.optimizers.Optimizer | None = None,
        num_samples: int = GRPO_NUM_SAMPLES,
        kl_coeff: float = 0.01,
        temperature: float = 1.0,
    ) -> None:
        """Initialize GRPOTrainer.

        Args:
            model: Language model to train.
            reward_fn: Function (prompts, responses) -> rewards [batch, num_samples].
            optimizer: Optimizer for updates. Defaults to Adam with lr=1e-5.
            num_samples: Number of samples per prompt.
            kl_coeff: KL penalty coefficient.
            temperature: Sampling temperature.
        """
        self.model = model
        self.reward_fn = reward_fn
        self.optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=1e-5)

        self.sampler = GRPOSampler(
            num_samples=num_samples,
            temperature=temperature,
        )
        self.loss_fn = GRPOLoss(kl_coeff=kl_coeff)

        # Reference model for KL penalty (frozen copy)
        self._ref_model: tf.keras.Model | None = None

    def set_reference_model(self, ref_model: tf.keras.Model) -> None:
        """Set reference model for KL penalty.

        Args:
            ref_model: Reference policy model (should be frozen).
        """
        self._ref_model = ref_model

    @tf.function
    def train_step(
        self,
        prompts: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        """Execute one GRPO training step.

        Args:
            prompts: Input prompts [batch, seq_len].

        Returns:
            Dictionary of training metrics.
        """
        with tf.GradientTape() as tape:
            # Sample responses from current policy
            responses, log_probs = self.sampler.sample_responses(self.model, prompts, training=True)

            # Compute rewards
            rewards = self.reward_fn(prompts, responses)

            # Get reference log probs if available
            ref_log_probs = None
            if self._ref_model is not None:
                _, ref_log_probs = self.sampler.sample_responses(
                    self._ref_model, prompts, training=False
                )

            # Compute loss
            loss, metrics = self.loss_fn(log_probs, rewards, ref_log_probs, training=True)

        # Compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Add reward statistics to metrics
        metrics["grpo/reward_mean"] = tf.reduce_mean(rewards)
        metrics["grpo/reward_std"] = tf.math.reduce_std(rewards)

        return metrics


def create_grpo_trainer(
    model: tf.keras.Model,
    reward_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    **kwargs: Any,
) -> GRPOTrainer | None:
    """Factory function for creating GRPO trainer.

    Returns None if USE_GRPO_TRAINING is False.

    Args:
        model: Language model to train.
        reward_fn: Reward function for responses.
        **kwargs: Additional arguments for GRPOTrainer.

    Returns:
        GRPOTrainer if enabled, None otherwise.
    """
    if not USE_GRPO_TRAINING:
        logger.info("GRPO training disabled (USE_GRPO_TRAINING=False)")
        return None

    return GRPOTrainer(model=model, reward_fn=reward_fn, **kwargs)


__all__ = [
    "GRPOSampler",
    "GRPOLoss",
    "GRPOTrainer",
    "create_grpo_trainer",
]
