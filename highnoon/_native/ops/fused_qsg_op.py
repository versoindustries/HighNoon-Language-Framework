# highnoon/_native/ops/fused_qsg_op.py
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

"""Python wrappers for Quantum Superposition Generation (QSG) ops.

This module provides Python interfaces to the C++ QSG operations.
Native C++ ops are REQUIRED - no Python fallbacks are provided.

Example:
    >>> from highnoon._native.ops.fused_qsg_op import entangled_coherence
    >>> output = entangled_coherence(position_states, coherence_range=64)
"""

from __future__ import annotations

import logging

import tensorflow as tf

from highnoon._native.ops.lib_loader import get_highnoon_core_path

logger = logging.getLogger(__name__)

# Lazy load native ops
_qsg_ops = None


def _get_native_ops():
    """Load QSG native ops library.

    Raises:
        RuntimeError: If native library cannot be loaded.
    """
    global _qsg_ops

    if _qsg_ops is not None:
        return _qsg_ops

    try:
        lib_path = get_highnoon_core_path()
        lib = tf.load_op_library(lib_path)
        _qsg_ops = lib
        logger.debug("QSG native ops loaded successfully from %s", lib_path)
        return _qsg_ops
    except Exception as e:
        raise RuntimeError(
            f"QSG native ops required but not available. "
            f"Please rebuild the native library with: ./build_secure.sh --lite --debug\n"
            f"Error: {e}"
        ) from e


# =============================================================================
# NATIVE OP WRAPPERS
# =============================================================================


def entangled_coherence(
    position_states: tf.Tensor,
    coherence_range: int = -1,
    temperature: float = 1.0,
) -> tf.Tensor:
    """Compute entangled bidirectional coherence between all positions.

    Unlike standard causal attention, this allows each position to see
    ALL other positions (including future), enabling superior quality
    for parallel generation.

    Args:
        position_states: Input states [batch, seq_len, dim] or [seq_len, dim].
        coherence_range: Maximum distance for coherence. -1 means all pairs.
        temperature: Softmax temperature for coherence weights.

    Returns:
        Updated states with same shape as input.

    Raises:
        RuntimeError: If C++ native ops are not available.

    Example:
        >>> states = tf.random.normal([8, 128, 512])
        >>> output = entangled_coherence(states, coherence_range=64)
    """
    ops = _get_native_ops()
    return ops.qsg_entangled_coherence(
        position_states,
        coherence_range=coherence_range,
        temperature=temperature,
    )


def grover_amplify(
    logits: tf.Tensor,
    oracle_scores: tf.Tensor,
    iterations: int = 3,
    amplification_strength: float = 1.5,
) -> tf.Tensor:
    """Grover-inspired amplitude amplification for token selection.

    Amplifies "good" tokens (high oracle score) and suppresses "bad" tokens
    through iterative reflection and oracle phase kicks.

    Args:
        logits: Token logits [batch, seq_len, vocab_size] or [seq_len, vocab_size].
        oracle_scores: Semantic consistency scores with same shape as logits.
            Values in [0, 1] where higher indicates better tokens.
        iterations: Number of Grover iterations (typically 2-4).
        amplification_strength: How strongly to amplify good tokens (1.0-2.0).

    Returns:
        Amplified logits with same shape as input.

    Raises:
        RuntimeError: If C++ native ops are not available.

    Example:
        >>> logits = model(input_ids)["logits"]
        >>> oracle = compute_oracle(context, vocab_embeddings)
        >>> amplified = grover_amplify(logits, oracle, iterations=3)
    """
    ops = _get_native_ops()
    return ops.qsg_grover_amplify(
        logits,
        oracle_scores,
        iterations=iterations,
        amplification_strength=amplification_strength,
    )


def semantic_oracle(
    vocab_embeddings: tf.Tensor,
    context_embedding: tf.Tensor,
) -> tf.Tensor:
    """Compute semantic consistency oracle scores.

    Evaluates cosine similarity between each vocabulary token and the
    context representation at each position.

    Args:
        vocab_embeddings: Vocabulary embedding matrix [vocab_size, dim].
        context_embedding: Context representation [seq_len, dim] or
            [batch, seq_len, dim].

    Returns:
        Oracle scores [seq_len, vocab_size] or [batch, seq_len, vocab_size]
        with values in [0, 1].

    Raises:
        RuntimeError: If C++ native ops are not available.

    Example:
        >>> vocab_emb = model.get_layer("token_embedding").embeddings
        >>> context = model.reasoning_module(input_embeddings)
        >>> oracle = semantic_oracle(vocab_emb, context)
    """
    ops = _get_native_ops()
    return ops.qsg_semantic_oracle(vocab_embeddings, context_embedding)


def jacobi_refine(
    token_logits: tf.Tensor,
    context_embedding: tf.Tensor,
    vocab_embeddings: tf.Tensor,
    iterations: int = 2,
    neighbor_window: int = 3,
) -> tf.Tensor:
    """Jacobi fixed-point iteration for local consistency refinement.

    After parallel generation, refines each position based on neighbor
    context to fix local inconsistencies.

    Args:
        token_logits: Current logits [seq_len, vocab_size].
        context_embedding: Context [seq_len, dim].
        vocab_embeddings: Vocabulary embeddings [vocab_size, dim].
        iterations: Number of refinement iterations.
        neighbor_window: Size of neighbor window for averaging.

    Returns:
        Refined logits with same shape as token_logits.

    Raises:
        RuntimeError: If C++ native ops are not available.

    Reference:
        Jacobi Forcing (ICML 2024)
    """
    ops = _get_native_ops()
    return ops.qsg_jacobi_refine(
        token_logits,
        context_embedding,
        vocab_embeddings,
        iterations=iterations,
        neighbor_window=neighbor_window,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "entangled_coherence",
    "grover_amplify",
    "semantic_oracle",
    "jacobi_refine",
]
