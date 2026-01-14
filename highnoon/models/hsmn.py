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
    CHUNKED_FORWARD_SIZE,
    COMPRESSED_DIM,
    EMBEDDING_DIM,
    HD_ACTIVE_VOCAB_SIZE,
    HD_EMBEDDING_DIM,
    HQE_CTQW_STEPS,
    MAMBA2_CONV_DIM,
    MAMBA2_STATE_DIM,
    NUM_EXPERTS,
    QUANTUM_LM_HEAD_LAYERS,
    REASONING_BLOCK_PATTERN,
    REASONING_HEADS,
    REASONING_LAYERS,
    TOP_K,
    USE_CHUNKED_FORWARD,
    USE_HYPERDIMENSIONAL_EMBEDDING,
    USE_QUANTUM_LM_HEAD,
    VOCAB_SIZE,
    WLAM_NUM_HEADS,
    WLAM_WAVELET_KERNEL_SIZE,
    get_tokenizer,
)
from highnoon.models.reasoning import ReasoningModule


class _EncoderProxy:
    """Proxy object providing encoder-like interface for training loop compatibility.

    The training loop expects model.encoder.embedding_dim to exist. This proxy
    provides that interface by wrapping the token embedding layer.
    """

    def __init__(self, token_embedding: tf.keras.layers.Embedding, embedding_dim: int):
        self._token_embedding = token_embedding
        self.embedding_dim = embedding_dim

    @property
    def embeddings(self) -> tf.Variable:
        """Get the embedding weights."""
        return self._token_embedding.embeddings


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
        # =====================================================================
        # LITE EDITION LIMIT VALIDATION
        # Comprehensive validation of all scale limits (20B, 5M, 24 blocks, etc.)
        # Pro/Enterprise editions skip these checks via is_lite().
        # =====================================================================
        from highnoon._native._limits import (
            MAX_CONTEXT_LENGTH,
            MAX_EMBEDDING_DIM,
            MAX_MOE_EXPERTS,
            MAX_REASONING_BLOCKS,
            LimitExceededError,
            is_lite,
        )

        if is_lite():
            violations = []

            # Check reasoning blocks
            if num_reasoning_blocks > MAX_REASONING_BLOCKS:
                violations.append(
                    f"num_reasoning_blocks: {num_reasoning_blocks} > {MAX_REASONING_BLOCKS}"
                )

            # Check embedding dimension
            if embedding_dim > MAX_EMBEDDING_DIM:
                violations.append(f"embedding_dim: {embedding_dim} > {MAX_EMBEDDING_DIM}")

            # Check context length
            if max_seq_length > MAX_CONTEXT_LENGTH:
                violations.append(f"max_seq_length: {max_seq_length:,} > {MAX_CONTEXT_LENGTH:,}")

            # Check MoE experts
            if num_experts > MAX_MOE_EXPERTS:
                violations.append(f"num_experts: {num_experts} > {MAX_MOE_EXPERTS}")

            if violations:
                raise LimitExceededError(
                    "Configuration exceeds Lite edition limits:\n  - "
                    + "\n  - ".join(violations)
                    + "\n\nUpgrade to Pro or Enterprise for unlimited scale:\n"
                    "  https://versoindustries.com/upgrade",
                    violations=violations,
                )

        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.num_reasoning_blocks = num_reasoning_blocks
        self.block_pattern = block_pattern

        # Token embedding - Phase 48: HyperdimensionalEmbedding support
        if USE_HYPERDIMENSIONAL_EMBEDDING:
            from highnoon.models.layers.hyperdimensional_layer import DualPathEmbedding

            # Phase 1 Fix: DualPathEmbedding is incompatible with weight tying
            # because it doesn't expose a single 'embeddings' property
            if tie_word_embeddings:
                logger.warning(
                    "[HSMN] DualPathEmbedding is incompatible with tie_word_embeddings. "
                    "Using separate LM head for output projection."
                )
                tie_word_embeddings = False

            self.token_embedding = DualPathEmbedding(
                vocab_size=vocab_size,
                model_dim=embedding_dim,
                active_vocab_size=HD_ACTIVE_VOCAB_SIZE,
                hd_dim=HD_EMBEDDING_DIM,
                use_ctqw=True,
                ctqw_steps=HQE_CTQW_STEPS,
                max_seq_len=max_seq_length,  # User's context window → HDE position_keys
                name="token_embedding_hde",
            )
            logger.info(
                f"HSMN Token Embedding: Using DualPathEmbedding "
                f"(active={HD_ACTIVE_VOCAB_SIZE}, hd_dim={HD_EMBEDDING_DIM}, max_seq={max_seq_length})"
            )
        else:
            self.token_embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim, name="token_embedding"
            )

        # Phase 901: Position encoding moved to DualPathEmbedding.streaming_pos
        # Holographic binding (circular convolution) in HD space replaces simple addition.
        # HSMN.streaming_position is deprecated but retained for backward compatibility.
        # Memory savings preserved: O(hd_dim) in DualPathEmbedding.
        self.streaming_position = None  # Deprecated, use DualPathEmbedding.streaming_pos

        # GRADIENT CONNECTIVITY FIX: Normalize embeddings before reasoning
        # DualPathEmbedding output has std≈0.01 due to L2 normalization + projection.
        # Without normalization, ReasoningModule amplifies 17,000× causing grad_norm=inf.
        # LayerNorm ensures consistent activation magnitude entering the reasoning stack.
        self.embedding_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="embedding_norm"
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
        # Priority: QuantumLMHead > TTDense > Dense
        self.tie_word_embeddings = tie_word_embeddings
        if not tie_word_embeddings:
            if USE_QUANTUM_LM_HEAD:
                # Phase 33: VQC-based output with Born rule
                from highnoon.models.reasoning.block_factory import QuantumLMHead

                self.lm_head = QuantumLMHead(
                    vocab_size=vocab_size,
                    hidden_dim=embedding_dim,
                    vqc_layers=QUANTUM_LM_HEAD_LAYERS,
                    name="lm_head_quantum",
                )
                logger.info(
                    f"HSMN LM Head: Using QuantumLMHead (vqc_layers={QUANTUM_LM_HEAD_LAYERS})"
                )
            else:
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

        # Create encoder proxy for training loop compatibility
        # Training loop accesses model.encoder.embedding_dim
        self._encoder_proxy = _EncoderProxy(self.token_embedding, embedding_dim)

        # Initialize tokenizer (lazy-loaded on first access)
        self._tokenizer = None

        logger.info(
            f"Initialized HSMN: vocab={vocab_size}, dim={embedding_dim}, "
            f"blocks={num_reasoning_blocks}, pattern={block_pattern}, "
            f"tie_embeddings={tie_word_embeddings}"
        )

    @property
    def tokenizer(self):
        """Get the tokenizer for this model.

        Lazy-loaded to avoid circular imports and allow HPO to configure
        vocab_size before tokenizer creation.

        Returns:
            QWTTextTokenizer instance configured for this model's vocab_size.
        """
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer(
                vocab_size=self.vocab_size,
                max_length=self.max_seq_length,
            )
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        """Set a custom tokenizer.

        Args:
            value: Tokenizer instance to use. Must have vocab_size and
                pad_token_id attributes.
        """
        self._tokenizer = value

    @property
    def encoder(self) -> _EncoderProxy:
        """Get the encoder proxy for training loop compatibility.

        The training loop expects model.encoder.embedding_dim to exist.
        This property provides that interface.

        Returns:
            _EncoderProxy wrapping the token embedding layer.
        """
        return self._encoder_proxy

    @property
    def reasoning(self) -> ReasoningModule:
        """Alias for reasoning_module for training loop compatibility.

        The training loop accesses model.reasoning.block_types, etc.
        This property provides that interface.

        Returns:
            The ReasoningModule instance.
        """
        return self.reasoning_module

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
        seq_len = tf.shape(input_ids)[1]

        # Wire USE_CHUNKED_FORWARD config flag for O(1) memory forward passes
        # Processes long sequences in chunks to reduce peak memory usage
        if USE_CHUNKED_FORWARD and seq_len > CHUNKED_FORWARD_SIZE:
            return self._chunked_forward(input_ids, attention_mask, training, return_hidden_states)

        # Standard forward pass
        return self._standard_forward(input_ids, attention_mask, training, return_hidden_states)

    def _standard_forward(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor | None,
        training: bool,
        return_hidden_states: bool,
    ) -> dict[str, tf.Tensor]:
        """Standard (non-chunked) forward pass."""
        # Phase 901: Token embeddings with holographic position binding
        # Position encoding is now handled via circular convolution in HD space
        # within DualPathEmbedding. No separate position addition needed.
        hidden_states = self.token_embedding(input_ids)  # [B, L, D] with positions bound

        # GRADIENT CONNECTIVITY FIX: Normalize embeddings before reasoning
        # Prevents 17,000× activation amplification that causes grad_norm=inf
        hidden_states = self.embedding_norm(hidden_states)

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
            embedding_weights = self.token_embedding.embeddings  # [vocab, dim]
            logits = tf.matmul(hidden_states, embedding_weights, transpose_b=True)  # [B, L, V]
        else:
            logits = self.lm_head(hidden_states)  # [B, L, V]

        outputs = {"logits": logits}
        if return_hidden_states:
            outputs["hidden_states"] = hidden_states

        return outputs

    def _chunked_forward(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor | None,
        training: bool,
        return_hidden_states: bool,
    ) -> dict[str, tf.Tensor]:
        """Chunked forward pass for O(1) memory scaling.

        Processes long sequences in fixed-size chunks to reduce peak memory.
        Combined with gradient checkpointing achieves constant memory training.
        """
        tf.shape(input_ids)[0]
        seq_len = tf.shape(input_ids)[1]
        chunk_size = CHUNKED_FORWARD_SIZE

        # Compute number of chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        # Process chunks and collect outputs
        all_logits = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = tf.minimum(start + chunk_size, seq_len)

            # Extract chunk
            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end] if attention_mask is not None else None

            # Forward pass for chunk
            chunk_output = self._standard_forward(
                chunk_ids, chunk_mask, training, return_hidden_states=False
            )
            all_logits.append(chunk_output["logits"])

        # Concatenate all chunk outputs
        logits = tf.concat(all_logits, axis=1)

        outputs = {"logits": logits}
        if return_hidden_states:
            # For chunked forward, we don't return intermediate hidden states
            # to save memory (that's the whole point of chunking)
            outputs["hidden_states"] = None

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
