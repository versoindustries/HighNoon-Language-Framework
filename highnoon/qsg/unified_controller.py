# highnoon/qsg/unified_controller.py
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

"""Phase A7: Unified QSG Generation Controller.

Provides a single API managing the 6-phase QSG pipeline:
1. MPS Context Entanglement (Phase A1)
2. Hopfield Vocabulary Projection (Phase A2)
3. VQC Adaptive Control (Phase A3)
4. COCONUT Continuous Thought (Phase A4)
5. HD Compression (Phase A5)
6. Speculative Draft Verification (Phase A6)

The UnifiedQSGController orchestrates all components and provides
a clean interface for generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.config import (
    COCONUT_BFS_BRANCHES,
    COCONUT_MAX_THOUGHT_STEPS,
    QSG_BOND_DIM,
    QSG_DEFAULT_TEMPERATURE,
    QSG_DEFAULT_TOP_K,
    QSG_DEFAULT_TOP_P,
    QSG_HOPFIELD_BETA,
    USE_SPECULATIVE,  # Unified speculative decoding flag
)
from highnoon.qsg.coconut_qsg import COCONUTEnhancedQSG
from highnoon.qsg.hd_compression import HDVocabularyEncoder
from highnoon.qsg.hopfield_vocab import HopfieldVocabularyProjector

# Import QSG components
from highnoon.qsg.mps_context import MPSContextEntangler, compute_position_importance
from highnoon.qsg.speculative_qsg import SpeculativeConfig, SpeculativeQSGPipeline
from highnoon.qsg.vqc_oracle_controller import VQCOracleController, compute_context_entropy

logger = logging.getLogger(__name__)


@dataclass
class QSGConfig:
    """Configuration for Unified QSG Controller."""

    # Model dimensions
    embedding_dim: int = 512
    vocab_size: int = 256000

    # Phase A1: MPS Context
    use_mps_context: bool = True
    mps_bond_dim: int = QSG_BOND_DIM

    # Phase A2: Hopfield Vocabulary
    use_hopfield_vocab: bool = True
    hopfield_beta: float = QSG_HOPFIELD_BETA
    learnable_beta: bool = True

    # Phase A3: VQC Adaptive Control
    use_vqc_control: bool = True

    # Phase A4: COCONUT
    use_coconut: bool = True
    coconut_branches: int = COCONUT_BFS_BRANCHES
    coconut_steps: int = COCONUT_MAX_THOUGHT_STEPS

    # Phase A5: HD Compression
    use_hd_compression: bool = True
    hd_active_vocab: int = 10000

    # Phase A6: Speculative - unified with global USE_SPECULATIVE config
    use_speculative: bool = USE_SPECULATIVE
    speculative_drafts: int = 4
    speculative_length: int = 8

    # Generation parameters
    temperature: float = QSG_DEFAULT_TEMPERATURE
    top_k: int = QSG_DEFAULT_TOP_K
    top_p: float = QSG_DEFAULT_TOP_P


class UnifiedQSGController(layers.Layer):
    """Unified controller for 6-phase QSG generation pipeline.

    Orchestrates MPS context, Hopfield vocab, VQC control, COCONUT thought,
    HD compression, and speculative decoding into a single generation API.
    """

    def __init__(self, config: QSGConfig | None = None, name: str = "unified_qsg", **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config or QSGConfig()

        # Initialize components based on config
        self._init_components()

    def _init_components(self):
        """Initialize QSG pipeline components."""
        cfg = self.config

        # Phase A1: MPS Context Entanglement
        self.mps_context = None
        if cfg.use_mps_context:
            self.mps_context = MPSContextEntangler(
                embedding_dim=cfg.embedding_dim, bond_dim=cfg.mps_bond_dim, name=f"{self.name}_mps"
            )

        # Phase A2: Hopfield Vocabulary
        self.hopfield_vocab = None
        if cfg.use_hopfield_vocab:
            self.hopfield_vocab = HopfieldVocabularyProjector(
                beta=cfg.hopfield_beta,
                learnable_beta=cfg.learnable_beta,
                name=f"{self.name}_hopfield",
            )

        # Phase A3: VQC Adaptive Control
        self.vqc_controller = None
        if cfg.use_vqc_control:
            self.vqc_controller = VQCOracleController(name=f"{self.name}_vqc_ctrl")

        # Phase A4: COCONUT Continuous Thought
        self.coconut = None
        if cfg.use_coconut:
            self.coconut = COCONUTEnhancedQSG(
                hidden_dim=cfg.embedding_dim,
                num_branches=cfg.coconut_branches,
                max_steps=cfg.coconut_steps,
                name=f"{self.name}_coconut",
            )

        # Phase A5: HD Compression (optional encoder)
        self.hd_encoder = None
        if cfg.use_hd_compression:
            self.hd_encoder = HDVocabularyEncoder(
                vocab_size=cfg.vocab_size,
                active_vocab_size=cfg.hd_active_vocab,
                embedding_dim=cfg.embedding_dim,
                name=f"{self.name}_hd",
            )

        # Phase A6: Speculative Pipeline
        self.speculative = None
        if cfg.use_speculative:
            spec_config = SpeculativeConfig(
                num_drafts=cfg.speculative_drafts,
                draft_length=cfg.speculative_length,
            )
            self.speculative = SpeculativeQSGPipeline(
                hidden_dim=cfg.embedding_dim, config=spec_config, name=f"{self.name}_spec"
            )

    def build(self, input_shape):
        """Build layer weights."""
        super().build(input_shape)

    def call(
        self,
        hidden_states: tf.Tensor,
        vocab_embeddings: tf.Tensor | None = None,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Run full QSG generation pipeline.

        Args:
            hidden_states: Input hidden states [batch, seq, dim].
            vocab_embeddings: Optional vocabulary embeddings for Hopfield [vocab, dim].
            training: Whether in training mode.

        Returns:
            Dict containing:
                - 'context': Enhanced context [batch, seq, dim]
                - 'entropy': MPS entropy [batch, seq-1] (if MPS enabled)
                - 'logits': Vocabulary logits [batch, seq, vocab] (if Hopfield enabled)
                - 'confidence': COCONUT confidence [batch, seq] (if COCONUT enabled)
                - 'params': Adaptive QSG parameters (if VQC enabled)
        """
        result = {}
        context = hidden_states

        # Phase A1: MPS Context Entanglement
        if self.mps_context is not None:
            try:
                context, entropy = self.mps_context(context)
                result["entropy"] = entropy
                result["importance"] = compute_position_importance(entropy)
            except Exception as e:
                logger.warning(f"MPS context failed, using fallback: {e}")

        # Phase A4: COCONUT Continuous Thought (applied to context)
        if self.coconut is not None:
            context, coconut_conf = self.coconut(context, training=training)
            result["confidence"] = coconut_conf

        result["context"] = context

        # Phase A3: VQC Adaptive Control
        if self.vqc_controller is not None:
            ctx_entropy = compute_context_entropy(context)
            params = self.vqc_controller(ctx_entropy)
            result["params"] = params

        # Phase A2: Hopfield Vocabulary Projection
        if self.hopfield_vocab is not None and vocab_embeddings is not None:
            logits, ood_mask = self.hopfield_vocab(context, vocab_embeddings)
            result["logits"] = logits
            result["ood_mask"] = ood_mask

        return result

    def generate_step(
        self,
        hidden_states: tf.Tensor,
        vocab_embeddings: tf.Tensor,
        logits: tf.Tensor,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Run one generation step with speculative decoding.

        Args:
            hidden_states: Context [batch, seq, dim].
            vocab_embeddings: Vocabulary [vocab, dim].
            logits: Current logits [batch, seq, vocab].
            training: Whether in training mode.

        Returns:
            Dict with speculative draft results.
        """
        if self.speculative is not None:
            return self.speculative(
                hidden_states, logits, temperature=self.config.temperature, training=training
            )

        # Fallback: simple argmax
        tokens = tf.argmax(logits[:, -1, :], axis=-1)
        return {
            "tokens": tokens,
            "draft_tokens": tf.expand_dims(tf.expand_dims(tokens, 1), 1),
        }

    def get_config(self) -> dict:
        config = super().get_config()
        # Note: Full config serialization would require QSGConfig serialization
        return config


def create_qsg_controller(
    embedding_dim: int = 512, vocab_size: int = 256000, enable_all: bool = True, **overrides
) -> UnifiedQSGController:
    """Factory function to create a QSG controller with default settings.

    Args:
        embedding_dim: Model hidden dimension.
        vocab_size: Vocabulary size.
        enable_all: If True, enable all QSG phases.
        **overrides: Override specific config options.

    Returns:
        Configured UnifiedQSGController instance.
    """
    config = QSGConfig(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        use_mps_context=enable_all,
        use_hopfield_vocab=enable_all,
        use_vqc_control=enable_all,
        use_coconut=enable_all,
        use_hd_compression=enable_all,
        use_speculative=enable_all,
    )

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return UnifiedQSGController(config=config)
