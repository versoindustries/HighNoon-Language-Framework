# highnoon/_native/ops/fused_coconut_ops.py
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

"""Phase 87: Python wrappers for CoCoNut multi-path BFS ops.

Provides Python interfaces to the fused C++ ops for multi-path thought
exploration, adaptive collapse, and crystallization.

Example:
    >>> from highnoon._native.ops.fused_coconut_ops import fused_coconut_bfs
    >>> output, amplitudes = fused_coconut_bfs(hidden_states, context, ...)
"""

from __future__ import annotations

import logging
from typing import Tuple

import tensorflow as tf

logger = logging.getLogger(__name__)

# Global state for op loading
_coconut_ops_lib = None
_coconut_ops_available = False


def _load_coconut_ops() -> bool:
    """Lazy load the native CoCoNut ops library.

    Uses the consolidated binary loading from `highnoon._native`.
    All CoCoNut ops are compiled into `_highnoon_core.so`.
    """
    global _coconut_ops_lib, _coconut_ops_available

    if _coconut_ops_lib is not None:
        return _coconut_ops_available

    try:
        from highnoon._native import _load_consolidated_binary

        _coconut_ops_lib = _load_consolidated_binary()
        _coconut_ops_available = _coconut_ops_lib is not None
        if _coconut_ops_available:
            logger.debug("CoCoNut native ops loaded successfully from consolidated binary")
        else:
            logger.warning("CoCoNut native ops: consolidated binary not available")
    except Exception as e:
        logger.warning(f"Failed to load CoCoNut native ops: {e}")
        _coconut_ops_lib = None
        _coconut_ops_available = False

    return _coconut_ops_available


def fused_coconut_bfs_available() -> bool:
    """Check if fused CoCoNut BFS op is available."""
    return _load_coconut_ops()


def fused_coconut_bfs(
    hidden_states: tf.Tensor,
    context: tf.Tensor,
    input_norm_gamma: tf.Tensor,
    input_norm_beta: tf.Tensor,
    aggregator_weight: tf.Tensor,
    aggregator_bias: tf.Tensor,
    projector_norm_gamma: tf.Tensor,
    projector_norm_beta: tf.Tensor,
    projector_dense1_weight: tf.Tensor,
    projector_dense1_bias: tf.Tensor,
    projector_dense2_weight: tf.Tensor,
    projector_dense2_bias: tf.Tensor,
    broadcast_weight: tf.Tensor,
    broadcast_bias: tf.Tensor,
    output_norm_gamma: tf.Tensor,
    output_norm_beta: tf.Tensor,
    num_paths: int = 2,
    num_thought_steps: int = 4,
    prune_threshold: float = 0.1,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Multi-path BFS thought exploration with Grover-inspired amplitude scoring.

    Expands hidden state to num_paths parallel thought paths, evolves them
    through num_thought_steps iterations, and aggregates using amplitude
    weighting.

    Args:
        hidden_states: Input hidden states [batch, seq_len, dim].
        context: Context for amplitude scoring [batch, dim].
        input_norm_gamma: Input LayerNorm gamma [dim].
        input_norm_beta: Input LayerNorm beta [dim].
        aggregator_weight: Aggregator dense weight [dim, dim].
        aggregator_bias: Aggregator dense bias [dim].
        projector_norm_gamma: Projector LayerNorm gamma [dim].
        projector_norm_beta: Projector LayerNorm beta [dim].
        projector_dense1_weight: First projector dense weight [dim, hidden_dim].
        projector_dense1_bias: First projector dense bias [hidden_dim].
        projector_dense2_weight: Second projector dense weight [hidden_dim, dim].
        projector_dense2_bias: Second projector dense bias [dim].
        broadcast_weight: Broadcast projection weight [dim, dim].
        broadcast_bias: Broadcast projection bias [dim].
        output_norm_gamma: Output LayerNorm gamma [dim].
        output_norm_beta: Output LayerNorm beta [dim].
        num_paths: Number of parallel thought paths (default 2, Lite max 8).
        num_thought_steps: Number of thought iterations per path.
        prune_threshold: Minimum amplitude to keep path (not used in current impl).

    Returns:
        Tuple of:
            - output: Enhanced hidden states [batch, seq_len, dim]
            - amplitudes: Final path amplitudes [batch, num_paths]

    Raises:
        RuntimeError: If native ops are not available.
    """
    if not _load_coconut_ops():
        raise RuntimeError(
            "FusedCoconutBFS C++ op not available. "
            "Build with: cd highnoon/_native && ./build_secure.sh --lite --debug"
        )

    return _coconut_ops_lib.fused_coconut_bfs(
        hidden_states,
        context,
        input_norm_gamma,
        input_norm_beta,
        aggregator_weight,
        aggregator_bias,
        projector_norm_gamma,
        projector_norm_beta,
        projector_dense1_weight,
        projector_dense1_bias,
        projector_dense2_weight,
        projector_dense2_bias,
        broadcast_weight,
        broadcast_bias,
        output_norm_gamma,
        output_norm_beta,
        num_paths=num_paths,
        num_thought_steps=num_thought_steps,
        prune_threshold=prune_threshold,
    )


def fused_coconut_dfs_collapse(
    path_states: tf.Tensor,
    path_amplitudes: tf.Tensor,
    collapse_threshold: float = 0.8,
    crystallize_threshold: float = 0.9,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Adaptive BFS→DFS collapse based on path confidence.

    When best path amplitude exceeds collapse_threshold, collapse to that
    single path. When it exceeds crystallize_threshold, flag for storage.

    Args:
        path_states: Current path states [batch, num_paths, dim].
        path_amplitudes: Path quality scores [batch, num_paths].
        collapse_threshold: Threshold to collapse to single path.
        crystallize_threshold: Threshold to flag for crystallization.

    Returns:
        Tuple of:
            - collapsed_state: Best/aggregated path [batch, dim]
            - should_crystallize: Bool flags [batch]
            - best_path_index: Index of best path [batch]
            - confidence: Collapse confidence [batch]
    """
    if not _load_coconut_ops():
        raise RuntimeError("FusedCoconutDFSCollapse C++ op not available.")

    return _coconut_ops_lib.fused_coconut_dfs_collapse(
        path_states,
        path_amplitudes,
        collapse_threshold=collapse_threshold,
        crystallize_threshold=crystallize_threshold,
    )


def fused_coconut_crystallize(
    thought_path: tf.Tensor,
    confidence: tf.Tensor,
    crystal_store: tf.Tensor,
    crystal_ages: tf.Tensor,
    crystallize_threshold: float = 0.9,
    max_crystals: int = 64,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Crystallize high-confidence thought paths for reuse.

    Stores thought paths exceeding threshold into a persistent store with
    LRU eviction when full.

    Args:
        thought_path: Thought path to potentially store [batch, dim].
        confidence: Confidence score [batch].
        crystal_store: Existing crystal store [max_crystals, dim].
        crystal_ages: Age counters for LRU [max_crystals].
        crystallize_threshold: Minimum confidence to store.
        max_crystals: Maximum crystals in store.

    Returns:
        Tuple of:
            - updated_store: Updated crystal store [max_crystals, dim]
            - updated_ages: Updated age counters [max_crystals]
            - crystal_indices: Where each was stored (-1 if not) [batch]
    """
    if not _load_coconut_ops():
        raise RuntimeError("FusedCoconutCrystallize C++ op not available.")

    return _coconut_ops_lib.fused_coconut_crystallize(
        thought_path,
        confidence,
        crystal_store,
        crystal_ages,
        crystallize_threshold=crystallize_threshold,
        max_crystals=max_crystals,
    )


def fused_coconut_retrieve(
    query: tf.Tensor,
    crystal_store: tf.Tensor,
    crystal_valid: tf.Tensor,
    top_k: int = 1,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Retrieve crystallized reasoning paths similar to query.

    Args:
        query: Query embedding [batch, dim].
        crystal_store: Crystal store [max_crystals, dim].
        crystal_valid: Which slots are valid [max_crystals].
        top_k: Number of crystals to retrieve.

    Returns:
        Tuple of:
            - retrieved: Best matching crystal [batch, dim]
            - similarity: Match similarity [batch]
    """
    if not _load_coconut_ops():
        raise RuntimeError("FusedCoconutRetrieve C++ op not available.")

    return _coconut_ops_lib.fused_coconut_retrieve(
        query,
        crystal_store,
        crystal_valid,
        top_k=top_k,
    )


# Custom gradient registration for training support
def fused_coconut_bfs_with_grad(
    hidden_states: tf.Tensor,
    context: tf.Tensor,
    input_norm_gamma: tf.Tensor,
    input_norm_beta: tf.Tensor,
    aggregator_weight: tf.Tensor,
    aggregator_bias: tf.Tensor,
    projector_norm_gamma: tf.Tensor,
    projector_norm_beta: tf.Tensor,
    projector_dense1_weight: tf.Tensor,
    projector_dense1_bias: tf.Tensor,
    projector_dense2_weight: tf.Tensor,
    projector_dense2_bias: tf.Tensor,
    broadcast_weight: tf.Tensor,
    broadcast_bias: tf.Tensor,
    output_norm_gamma: tf.Tensor,
    output_norm_beta: tf.Tensor,
    num_paths: int = 2,
    num_thought_steps: int = 4,
    prune_threshold: float = 0.1,
) -> tuple[tf.Tensor, tf.Tensor]:
    """CoCoNut BFS with custom gradient for training.

    This wrapper uses an inner function with @tf.custom_gradient to avoid
    issues with variable tracking when layer weights are passed as tensors.

    Args:
        hidden_states: Input hidden states [batch, seq_len, dim].
        context: Context for amplitude scoring [batch, dim].
        input_norm_gamma: Input LayerNorm gamma [dim].
        input_norm_beta: Input LayerNorm beta [dim].
        aggregator_weight: Aggregator dense weight [dim, dim].
        aggregator_bias: Aggregator dense bias [dim].
        projector_norm_gamma: Projector LayerNorm gamma [dim].
        projector_norm_beta: Projector LayerNorm beta [dim].
        projector_dense1_weight: First projector dense weight [dim, hidden_dim].
        projector_dense1_bias: First projector dense bias [hidden_dim].
        projector_dense2_weight: Second projector dense weight [hidden_dim, dim].
        projector_dense2_bias: Second projector dense bias [dim].
        broadcast_weight: Broadcast projection weight [dim, dim].
        broadcast_bias: Broadcast projection bias [dim].
        output_norm_gamma: Output LayerNorm gamma [dim].
        output_norm_beta: Output LayerNorm beta [dim].
        num_paths: Number of parallel thought paths.
        num_thought_steps: Number of thought iterations per path.
        prune_threshold: Minimum amplitude to keep path.

    Returns:
        Tuple of (output, amplitudes).
    """
    # Ensure float32
    hidden_states = tf.cast(hidden_states, tf.float32)
    context = tf.cast(context, tf.float32)
    input_norm_gamma = tf.cast(input_norm_gamma, tf.float32)
    input_norm_beta = tf.cast(input_norm_beta, tf.float32)
    aggregator_weight = tf.cast(aggregator_weight, tf.float32)
    aggregator_bias = tf.cast(aggregator_bias, tf.float32)
    projector_norm_gamma = tf.cast(projector_norm_gamma, tf.float32)
    projector_norm_beta = tf.cast(projector_norm_beta, tf.float32)
    projector_dense1_weight = tf.cast(projector_dense1_weight, tf.float32)
    projector_dense1_bias = tf.cast(projector_dense1_bias, tf.float32)
    projector_dense2_weight = tf.cast(projector_dense2_weight, tf.float32)
    projector_dense2_bias = tf.cast(projector_dense2_bias, tf.float32)
    broadcast_weight = tf.cast(broadcast_weight, tf.float32)
    broadcast_bias = tf.cast(broadcast_bias, tf.float32)
    output_norm_gamma = tf.cast(output_norm_gamma, tf.float32)
    output_norm_beta = tf.cast(output_norm_beta, tf.float32)

    @tf.custom_gradient
    def _fused_coconut_bfs_inner(
        hs,
        ctx,
        ing,
        inb,
        agg_w,
        agg_b,
        png,
        pnb,
        pd1_w,
        pd1_b,
        pd2_w,
        pd2_b,
        bc_w,
        bc_b,
        ong,
        onb,
    ):
        """Inner function with tensor-only signature for custom gradient."""
        output, amplitudes = fused_coconut_bfs(
            hs,
            ctx,
            ing,
            inb,
            agg_w,
            agg_b,
            png,
            pnb,
            pd1_w,
            pd1_b,
            pd2_w,
            pd2_b,
            bc_w,
            bc_b,
            ong,
            onb,
            num_paths=num_paths,
            num_thought_steps=num_thought_steps,
            prune_threshold=prune_threshold,
        )

        def grad(grad_output, grad_amplitudes):
            """Backward pass through CoCoNut BFS.

            Args:
                grad_output: Gradient w.r.t. output tensor.
                grad_amplitudes: Gradient w.r.t. amplitudes tensor.

            Returns:
                Tuple of gradients for all 16 tensor inputs.
            """
            # Return gradients for each of the 16 tensor inputs
            # hidden_states gets grad_output, all weight/bias tensors get None
            return (
                grad_output,  # grad for hs (hidden_states)
                None,  # grad for ctx (context)
                None,  # grad for ing (input_norm_gamma)
                None,  # grad for inb (input_norm_beta)
                None,  # grad for agg_w (aggregator_weight)
                None,  # grad for agg_b (aggregator_bias)
                None,  # grad for png (projector_norm_gamma)
                None,  # grad for pnb (projector_norm_beta)
                None,  # grad for pd1_w (projector_dense1_weight)
                None,  # grad for pd1_b (projector_dense1_bias)
                None,  # grad for pd2_w (projector_dense2_weight)
                None,  # grad for pd2_b (projector_dense2_bias)
                None,  # grad for bc_w (broadcast_weight)
                None,  # grad for bc_b (broadcast_bias)
                None,  # grad for ong (output_norm_gamma)
                None,  # grad for onb (output_norm_beta)
            )

        return (output, amplitudes), grad

    return _fused_coconut_bfs_inner(
        hidden_states,
        context,
        input_norm_gamma,
        input_norm_beta,
        aggregator_weight,
        aggregator_bias,
        projector_norm_gamma,
        projector_norm_beta,
        projector_dense1_weight,
        projector_dense1_bias,
        projector_dense2_weight,
        projector_dense2_bias,
        broadcast_weight,
        broadcast_bias,
        output_norm_gamma,
        output_norm_beta,
    )


__all__ = [
    "fused_coconut_bfs_available",
    "fused_coconut_bfs",
    "fused_coconut_bfs_with_grad",
    "fused_coconut_dfs_collapse",
    "fused_coconut_crystallize",
    "fused_coconut_retrieve",
]
