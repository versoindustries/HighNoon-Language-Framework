# highnoon/quantum/coherence_bus.py
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

"""UQHA v3.1 Unified Coherence Bus.

Centralizes coherence metrics from all quantum-inspired components (QHD, PEPS, Hopfield)
to provide global routing and adaptive reasoning signals.
"""

from __future__ import annotations

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


class CoherenceBus:
    """Unified coherence tracking across all quantum components.

    Provides a shared signal that represents the 'order' or 'certainty' of the
    quantum state across multiple hierarchical or spatial blocks.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._coherence_scores: dict[str, tf.Tensor] = {}
        self._global_coherence: tf.Tensor = tf.constant(0.5, dtype=tf.float32)
        self._initialized = True
        logger.debug("CoherenceBus initialized")

    def reset(self) -> None:
        """Clear all registered coherence scores for a new sequence/batch."""
        self._coherence_scores.clear()
        self._global_coherence = tf.constant(0.5, dtype=tf.float32)

    def register(self, block_name: str, coherence: tf.Tensor) -> None:
        """Register coherence from a quantum block.

        Args:
            block_name: Unique identifier for the source block.
            coherence: Coherence metric [batch_size] or scalar.
        """
        # Ensure it's a tensor and float32
        coherence = tf.cast(coherence, tf.float32)
        self._coherence_scores[block_name] = coherence
        self._update_global()

    def _update_global(self) -> None:
        """Aggregate coherence via geometric mean."""
        if not self._coherence_scores:
            return

        values = list(self._coherence_scores.values())
        # geometric mean ensures that if any component is totally decoherent (0),
        # the global coherence is low.
        log_sum = 0.0
        for v in values:
            log_sum += tf.math.log(v + 1e-8)

        self._global_coherence = tf.exp(log_sum / len(values))

    def get_global_coherence(self) -> tf.Tensor:
        """Get aggregated coherence for routing/gating.

        Returns:
            Mean coherence across all registered components.
        """
        return self._global_coherence

    def get_coconut_threshold(self) -> float:
        """Get coherence-adjusted crystallization threshold.

        Higher global coherence allows for more selective (higher) thresholds.
        """
        # Base threshold 0.7 + up to 0.2 boost from coherence
        # We take the mean of the global coherence if it's a batch tensor
        avg_coherence = tf.reduce_mean(self._global_coherence)
        return 0.7 + 0.2 * float(avg_coherence)


# Global singleton instance
coherence_bus = CoherenceBus()
