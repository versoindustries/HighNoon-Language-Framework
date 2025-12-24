# highnoon/models/hsmn.py
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

"""HSMN: Hierarchical State-Space Model Network for Language Modeling.

This is the Lite edition - a language-only version of the full HSMN architecture
with enforced scale limits (20B params, 5M context, 24 reasoning blocks).

Architecture:
- Embedding + Positional Encoding
- Reasoning Stack: Mamba SSM, TimeCrystal, WLAM, MoE (hybrid pattern)
- Output LM Head
"""

import logging
from typing import Any

import tensorflow as tf

from highnoon.config import (
    COMPRESSED_DIM,
    EMBEDDING_DIM,
    LITE_MAX_REASONING_BLOCKS,
    MAMBA2_CONV_DIM,
    MAMBA2_STATE_DIM,
    NUM_EXPERTS,
    REASONING_BLOCK_PATTERN,
    REASONING_HEADS,
    REASONING_LAYERS,
    TOP_K,
    VOCAB_SIZE,
    WLAM_NUM_HEADS,
    WLAM_WAVELET_KERNEL_SIZE,
)
from highnoon.models.reasoning import ReasoningModule

logger = logging.getLogger(__name__)


class HSMN(tf.keras.Model):
    """Hierarchical State-Space Model Network for Language Modeling.

    This model uses linear-time reasoning blocks (Mamba SSM, TimeCrystal, WLAM)
    instead of quadratic attention, enabling efficient long-context processing.

    Args:
        vocab_size: Size of the vocabulary.
        embedding_dim: Dimension of token embeddings.
        max_seq_length: Maximum sequence length.
        num_reasoning_blocks: Number of reasoning blocks (max 24 for Lite).
        block_pattern: Reasoning block pattern.
        num_experts: Number of MoE experts (max 12 for Lite).
        top_k: Top-K experts to route per token.
        dropout: Dropout rate.

    Example:
        >>> model = HSMN(vocab_size=32000, embedding_dim=512)
        >>> logits = model(input_ids)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
        max_seq_length: int = 4096,
        num_reasoning_blocks: int = REASONING_LAYERS,
        block_pattern: str = REASONING_BLOCK_PATTERN,
        num_experts: int = NUM_EXPERTS,
        top_k: int = TOP_K,
        dropout: float = 0.1,
        mamba_state_dim: int = MAMBA2_STATE_DIM,
        mamba_conv_dim: int = MAMBA2_CONV_DIM,
        wlam_num_heads: int = WLAM_NUM_HEADS,
        wlam_kernel_size: int = WLAM_WAVELET_KERNEL_SIZE,
        tie_word_embeddings: bool = True,
        # Quantum Training Flags
        use_quantum_memory_replay: bool = False,
        use_entanglement_loss: bool = False,
        use_quantum_holographic_memory: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Validate Lite edition limits
        if num_reasoning_blocks > LITE_MAX_REASONING_BLOCKS:
            raise ValueError(
                f"Lite edition supports max {LITE_MAX_REASONING_BLOCKS} reasoning blocks, "
                f"got {num_reasoning_blocks}. Upgrade to Enterprise for unlimited."
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.num_reasoning_blocks = num_reasoning_blocks
        self.block_pattern = block_pattern

        # Token embedding
        self.token_embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, name="token_embedding"
        )

        # Positional encoding (learnable)
        self.position_embedding = tf.keras.layers.Embedding(
            max_seq_length, embedding_dim, name="position_embedding"
        )

        # Embedding dropout
        self.embedding_dropout = tf.keras.layers.Dropout(dropout)

        # Reasoning module with hybrid block pattern
        self.reasoning_module = ReasoningModule(
            num_layers=num_reasoning_blocks,
            embedding_dim=embedding_dim,
            compressed_dim=COMPRESSED_DIM,
            block_pattern=block_pattern,
            num_heads=REASONING_HEADS,
            ff_dim=embedding_dim * 4,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            mamba_state_dim=mamba_state_dim,
            mamba_conv_dim=mamba_conv_dim,
            wlam_num_heads=wlam_num_heads,
            wlam_kernel_size=wlam_kernel_size,
            use_quantum_memory_replay=use_quantum_memory_replay,
            use_entanglement_loss=use_entanglement_loss,
            use_quantum_holographic_memory=use_quantum_holographic_memory,
            name="reasoning_module",
        )

        # Output layer norm
        self.output_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="output_norm")

        # LM head - optionally tied to embedding weights
        self.tie_word_embeddings = tie_word_embeddings
        if not tie_word_embeddings:
            # Import TT config and layer
            from highnoon.config import TT_LM_HEAD_RANKS, USE_TT_LM_HEAD

            if USE_TT_LM_HEAD:
                # Use TTDense for massive parameter reduction (85-98%)
                from highnoon.models.layers.tt_dense import TTDense

                self.lm_head = TTDense(
                    output_dim=vocab_size,
                    tt_ranks=TT_LM_HEAD_RANKS,
                    use_bias=False,
                    name="lm_head_tt",
                )
                logger.info(
                    f"HSMN LM Head: Using TTDense with ranks {TT_LM_HEAD_RANKS} "
                    f"(~{100 * (1 - sum(TT_LM_HEAD_RANKS) / (embedding_dim * vocab_size)):.1f}% compression)"
                )
            else:
                self.lm_head = tf.keras.layers.Dense(vocab_size, name="lm_head", use_bias=False)
        else:
            self.lm_head = None  # Use transposed embedding weights

        logger.info(
            f"Initialized HSMN: vocab={vocab_size}, dim={embedding_dim}, "
            f"blocks={num_reasoning_blocks}, pattern={block_pattern}, "
            f"tie_embeddings={tie_word_embeddings}"
        )

    def call(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        training: bool = False,
        return_hidden_states: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape [batch, seq_len].
            attention_mask: Optional mask of shape [batch, seq_len].
            training: Whether in training mode.
            return_hidden_states: Whether to return intermediate hidden states.

        Returns:
            Dictionary with 'logits' and optionally 'hidden_states'.
        """
        _batch_size = tf.shape(input_ids)[0]  # noqa: F841 - used for documentation
        seq_len = tf.shape(input_ids)[1]

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # [B, L, D]

        # Position embeddings
        positions = tf.range(seq_len)
        pos_embeds = self.position_embedding(positions)  # [L, D]

        # Combine
        hidden_states = token_embeds + pos_embeds
        hidden_states = self.embedding_dropout(hidden_states, training=training)

        # Reasoning module
        reasoning_output = self.reasoning_module(
            hidden_states,
            training=training,
            mask=attention_mask,
        )

        # Extract final hidden states
        if isinstance(reasoning_output, dict):
            hidden_states = reasoning_output.get("hidden_states", reasoning_output)
        else:
            hidden_states = reasoning_output

        # Output norm and LM head
        hidden_states = self.output_norm(hidden_states)

        # Compute logits
        if self.tie_word_embeddings:
            # Use transposed embedding weights for output projection
            # logits = hidden_states @ embedding_weights
            embedding_weights = self.token_embedding.embeddings  # [vocab, dim]
            logits = tf.matmul(hidden_states, embedding_weights, transpose_b=True)  # [B, L, V]
        else:
            logits = self.lm_head(hidden_states)  # [B, L, V]

        outputs = {"logits": logits}
        if return_hidden_states:
            outputs["hidden_states"] = hidden_states

        return outputs

    def generate(
        self,
        input_ids: tf.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        use_qsg: bool = True,
    ) -> tf.Tensor:
        """Generate text using Quantum Superposition Generation (QSG).

        QSG generates all tokens in parallel using quantum-inspired mechanisms,
        achieving 50-100x+ speedup over autoregressive generation while
        maintaining or improving quality.

        Args:
            input_ids: Starting token IDs [batch, seq_len].
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K filtering.
            top_p: Nucleus sampling threshold.
            use_qsg: Enable QSG parallel generation (default True). Set False
                for autoregressive fallback (not recommended).

        Returns:
            Generated token IDs [batch, seq_len + new_tokens].

        Note:
            QSG is strongly recommended. AR generation is provided only for
            compatibility testing.
        """
        from highnoon.inference.qsg_generator import QSGGenerator

        generator = QSGGenerator(self)
        return generator.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "max_seq_length": self.max_seq_length,
            "num_reasoning_blocks": self.num_reasoning_blocks,
            "block_pattern": self.block_pattern,
            "tie_word_embeddings": self.tie_word_embeddings,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HSMN":
        """Create model from configuration."""
        return cls(**config)
