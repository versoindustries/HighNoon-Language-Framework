"""Streaming Inference Wrapper for HighNoon Models.

This module provides the StreamingHSMN wrapper which enables O(1) inference
step-by-step, managing the state cache for stateful blocks (TimeCrystal, QHDSpatial).

Key Features:
- O(1) per-token inference (no reprocessing of history)
- Explicit state management
- JIT-compiled step function
- Support for dual-path position-aware embeddings
"""

import logging
from typing import Any, List, Optional, Tuple

import tensorflow as tf

from highnoon import config
from highnoon.models.hsmn import HSMN

logger = logging.getLogger(__name__)


class StreamingHSMN:
    """Wrapper for performing streaming (O(1) per token) inference with HSMN."""

    def __init__(self, model: HSMN):
        self.model = model

    def initialize_cache(self, batch_size: int) -> tuple[list[Any], tf.Tensor]:
        """Initialize the inference state cache.

        Args:
            batch_size: Batch size.

        Returns:
            cache: (states_list, position_ids)
        """
        states_list = []

        # Iterate over reasoning blocks to create initial states
        # The reasoning module's blocks might be wrapped in QuantumEnhancedBlock_X
        # or HDCheckpointLayer_X (if enabled).

        for block in self.model.reasoning.reasoning_blocks:
            # Unwrap wrappers logic
            inner = block

            # Unwrap HD Checkpoint Layer
            if hasattr(inner, "sublayers") and len(inner.sublayers) > 0:
                inner = inner.sublayers[0]  # Often the wrapped block

            # Unwrap QuantumEnhancedBlock
            # It stores inner block in .inner_block
            if hasattr(inner, "inner_block"):
                inner = inner.inner_block

            block_type = inner.__class__.__name__

            # 1. QHDSpatialBlock State Initialization
            if block_type == "QHDSpatialBlock":
                # State shape: [Batch, NumPaths, D_inner, StateDim]
                # Attributes: num_paths, d_inner, state_dim

                # Verify attributes exist
                # QHDSpatial uses hd_dim as inner dimension
                if hasattr(inner, "hd_dim"):
                    d_inner = inner.hd_dim
                elif hasattr(inner, "d_inner"):
                    d_inner = inner.d_inner
                else:
                    d_inner = None

                if (
                    d_inner is not None
                    and hasattr(inner, "num_paths")
                    and hasattr(inner, "state_dim")
                ):
                    num_paths = inner.num_paths
                    state_dim = inner.state_dim

                    # Assume real-valued state for now (SSM typical)
                    initial_state = tf.zeros(
                        (batch_size, num_paths, d_inner, state_dim), dtype=tf.float32
                    )
                    states_list.append(initial_state)
                else:
                    logger.warning(
                        f"QHDSpatialBlock missing attributes for state init (hd_dim/d_inner={d_inner}). Using None."
                    )
                    states_list.append(None)

            # 2. TimeCrystalSequenceBlock State Initialization
            elif block_type == "TimeCrystalSequenceBlock":
                # Uses TimeCrystalBlock cell.
                # State is tuple: (h, aux)
                # h: [Batch, state_dim]
                # aux: [Batch, state_dim] (momentum p)

                if hasattr(inner, "state_dim"):
                    state_dim = inner.state_dim

                    h_init = tf.zeros((batch_size, state_dim), dtype=tf.float32)
                    aux_init = tf.zeros((batch_size, state_dim), dtype=tf.float32)

                    states_list.append((h_init, aux_init))
                else:
                    logger.warning("TimeCrystalSequenceBlock missing state_dim. Using None.")
                    states_list.append(None)

            # 3. ReasoningMamba2Block State Initialization
            elif block_type == "ReasoningMamba2Block":
                # Mamba2 state is usually complex or just [B, N, D]
                # Generally handled by mamba internal init if passed None?
                # But we need explicit zeros.
                # State shape: [B, state_dim, head_dim * num_heads]?
                # Depends greatly on implementation.
                # For now, pass None and see if step fails, or try zero init if dims known.
                # HighNoon wrapper usually Mamba2Block from specialized kernel.
                # Let's inspect known attributes or default to None if block handles it.
                # Warning: Passing None to step(x, None) might fail if step expects tensor.
                # We'll use None for now as Mamba2 support is secondary to TC/QHD.
                states_list.append(None)

            else:
                # Stateless or Unknown Block
                states_list.append(None)

        # Current Position (scalar sequence index per batch item for position embedding)
        # Initialize to 0.
        position_ids = tf.zeros((batch_size,), dtype=tf.int32)

        return (states_list, position_ids)

    @tf.function(jit_compile=True)
    def step(
        self, input_ids: tf.Tensor, cache: tuple[list[Any], tf.Tensor]
    ) -> tuple[tf.Tensor, tuple[list[Any], tf.Tensor]]:
        """Perform a single inference step.

        Args:
            input_ids: [Batch] or [Batch, 1] tensor of token IDs.
            cache: Tuple of (states_list, position_ids).

        Returns:
            logits: Output logits [Batch, vocab_size].
            new_cache: Updated (states_list, position_ids).
        """
        states_list, position_ids = cache

        # Expect input_ids to be [B] or [B, 1].
        # Embedding expects [B, L] usually?
        # DualPathEmbedding handles rank 2.
        # If input is [B], expand to [B, 1].
        if len(input_ids.shape) == 1:
            input_ids_expanded = tf.expand_dims(input_ids, axis=1)
        else:
            input_ids_expanded = input_ids

        # 1. Embeddings
        # Pass position_ids explicitly to handle streaming position encoding
        # Position ids should match input shape [B, 1]
        if len(position_ids.shape) == 1:
            pos_ids_expanded = tf.expand_dims(position_ids, axis=1)
        else:
            pos_ids_expanded = position_ids

        x = self.model.token_embedding(
            input_ids_expanded, training=False, position_ids=pos_ids_expanded
        )

        # Embeddings return [B, 1, D].
        # ReasoningModule.step expects [B, D].
        x_squeezed = tf.squeeze(x, axis=1)

        # 2. Reasoning
        # Performs O(1) step through all blocks
        x_out, new_states_list = self.model.reasoning.step(x_squeezed, states_list)

        # 3. Head (Logits)
        # QuantumLMHead expects [B, D] (if Dense) or [B, L, D] (if Factored?)
        # FactoredOutputHead (if used) usually handles [B, D] or [B, 1, D].
        # Let's assume standard [B, D] -> [B, V].
        logits = self.model.lm_head(x_out, training=False)

        # 4. Update Position
        new_position_ids = position_ids + 1

        return logits, (new_states_list, new_position_ids)


# Aliases for backward compatibility and standardized naming (Phase 13.8)
StreamingInferenceWrapper = StreamingHSMN
StreamingState = tuple  # (states_list, position_ids)
