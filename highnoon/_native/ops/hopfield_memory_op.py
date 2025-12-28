# highnoon/_native/ops/hopfield_memory_op.py
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

"""Python wrapper for Modern Hopfield Network memory operations.

This module provides Python-accessible functions for Hopfield memory operations
implemented in C++ with SIMD optimization.

Ops:
    hopfield_memory_retrieve: Retrieve patterns via Hopfield update rule
    hopfield_energy: Compute energy landscape for OOD detection
    hopfield_moe_routing_bias: Compute MoE routing bias from Hopfield energy
    hopfield_kv_cache_compress: Compress KV cache using Hopfield patterns
    hopfield_kv_cache_retrieve: Retrieve from compressed KV cache

Reference: "Hopfield Networks is All You Need" (2020)

Example:
    >>> import tensorflow as tf
    >>> from highnoon._native.ops.hopfield_memory_op import hopfield_memory_retrieve
    >>> patterns = tf.random.normal([64, 256])  # 64 stored patterns
    >>> query = tf.random.normal([8, 256])      # 8 queries
    >>> retrieved = hopfield_memory_retrieve(query, patterns, beta=1.0)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tensorflow as tf

from highnoon._native.ops.lib_loader import get_highnoon_core_path

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Load the compiled ops library
_ops = None
_hopfield_ops_available = False

try:
    _lib_path = get_highnoon_core_path()
    _ops = tf.load_op_library(_lib_path)
    _hopfield_memory_retrieve = _ops.hopfield_memory_retrieve
    _hopfield_energy = _ops.hopfield_energy
    _hopfield_moe_routing_bias = _ops.hopfield_moe_routing_bias
    _hopfield_kv_cache_compress = _ops.hopfield_kv_cache_compress
    _hopfield_kv_cache_retrieve = _ops.hopfield_kv_cache_retrieve
    _hopfield_ops_available = True
    log.debug("Hopfield memory ops loaded successfully from %s", _lib_path)
except (RuntimeError, AttributeError) as e:
    log.warning(f"Hopfield memory ops not available: {e}")
    _hopfield_ops_available = False


def hopfield_ops_available() -> bool:
    """Check if Hopfield memory ops are available.

    Returns:
        True if C++ ops are loaded, False otherwise.
    """
    return _hopfield_ops_available


def hopfield_memory_retrieve(
    queries: tf.Tensor,
    patterns: tf.Tensor,
    beta: float = 1.0,
) -> tf.Tensor:
    """Retrieve patterns from Modern Hopfield Network memory.

    Performs single-step Hopfield retrieval:
        output = Σᵢ softmax(β * query^T * patterns[i]) * patterns[i]

    This is mathematically equivalent to softmax attention but with
    exponential storage capacity guarantee.

    Args:
        queries: Query vectors [batch_size, dim].
        patterns: Stored patterns [num_patterns, dim].
        beta: Inverse temperature (higher = sharper retrieval).

    Returns:
        Retrieved patterns [batch_size, dim].

    Raises:
        RuntimeError: If C++ ops are not available.
    """
    if not _hopfield_ops_available:
        raise RuntimeError(
            "Hopfield memory ops not available. " "Rebuild with build_secure.sh --lite --debug"
        )
    return _hopfield_memory_retrieve(queries, patterns, beta=beta)


def hopfield_energy(
    states: tf.Tensor,
    patterns: tf.Tensor,
    beta: float = 1.0,
) -> tf.Tensor:
    """Compute Modern Hopfield Network energy for states.

    Energy function:
        E(s) = -β⁻¹ log(Σᵢ exp(β s^T ξᵢ)) + ½||s||² + β⁻¹ log(M)

    Lower energy indicates better match to stored patterns.
    High energy indicates out-of-distribution states.

    Args:
        states: State vectors [batch_size, dim].
        patterns: Stored patterns [num_patterns, dim].
        beta: Inverse temperature.

    Returns:
        Energy values [batch_size].

    Raises:
        RuntimeError: If C++ ops are not available.
    """
    if not _hopfield_ops_available:
        raise RuntimeError("Hopfield memory ops not available.")
    return _hopfield_energy(states, patterns, beta=beta)


def hopfield_moe_routing_bias(
    token_states: tf.Tensor,
    expert_patterns: tf.Tensor,
    beta: float = 1.0,
    uncertainty_expert_idx: int = -1,
    energy_threshold: float = 0.0,
) -> tf.Tensor:
    """Compute MoE routing bias from Hopfield energy.

    High energy (out-of-distribution) tokens get boosted routing
    to the uncertainty expert. Low energy tokens use standard routing.

    This enables the MoE to detect anomalous tokens and route them
    to specialized experts for better handling.

    Args:
        token_states: Token embeddings [num_tokens, dim].
        expert_patterns: Expert prototype patterns [num_experts, dim].
        beta: Inverse temperature for energy calculation.
        uncertainty_expert_idx: Index of uncertainty expert (-1 = last expert).
        energy_threshold: Threshold above which tokens are considered OOD.

    Returns:
        Routing bias to add to logits [num_tokens, num_experts].

    Raises:
        RuntimeError: If C++ ops are not available.
    """
    if not _hopfield_ops_available:
        raise RuntimeError("Hopfield memory ops not available.")
    return _hopfield_moe_routing_bias(
        token_states,
        expert_patterns,
        beta=beta,
        uncertainty_expert_idx=uncertainty_expert_idx,
        energy_threshold=energy_threshold,
    )


def hopfield_kv_cache_compress(
    keys: tf.Tensor,
    values: tf.Tensor,
    num_patterns: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compress KV cache using Hopfield pattern storage.

    Reduces memory from O(seq_len * dim) to O(num_patterns * dim).
    For long contexts (100K+ tokens), this can provide significant
    memory savings while maintaining retrieval quality.

    Args:
        keys: Key vectors [seq_len, dim].
        values: Value vectors [seq_len, dim].
        num_patterns: Target number of patterns to store.

    Returns:
        Tuple of (pattern_keys, pattern_values), each [num_patterns, dim].

    Raises:
        RuntimeError: If C++ ops are not available.
    """
    if not _hopfield_ops_available:
        raise RuntimeError("Hopfield memory ops not available.")
    return _hopfield_kv_cache_compress(keys, values, num_patterns=num_patterns)


def hopfield_kv_cache_retrieve(
    queries: tf.Tensor,
    pattern_keys: tf.Tensor,
    pattern_values: tf.Tensor,
    beta: float = 1.0,
) -> tf.Tensor:
    """Retrieve values from compressed Hopfield KV cache.

    Uses Hopfield associative memory to reconstruct values from
    compressed pattern storage.

    Args:
        queries: Query vectors [batch_size, dim].
        pattern_keys: Stored pattern keys [num_patterns, dim].
        pattern_values: Stored pattern values [num_patterns, dim].
        beta: Inverse temperature.

    Returns:
        Retrieved values [batch_size, dim].

    Raises:
        RuntimeError: If C++ ops are not available.
    """
    if not _hopfield_ops_available:
        raise RuntimeError("Hopfield memory ops not available.")
    return _hopfield_kv_cache_retrieve(queries, pattern_keys, pattern_values, beta=beta)
