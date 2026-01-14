"""Unified Memory System Operations Python Wrapper.

Phase 4 of V2 Performance Optimization - Memory Op Unification

This module provides Python bindings for all 5 consolidated memory
operations via native C++ dispatch. NO PYTHON FALLBACKS.

Supported Memory Types:
- CONTENT_ADDRESSED: Standard attention-based memory
- PRODUCT_KEY: Sub-linear O(√M) lookup via product codebooks
- HOPFIELD: Energy-based associative memory (exponential capacity)
- ADAPTIVE: Learned gating with surprise-based writes
- HIERARCHICAL: Multi-level memory with CTQW traversal

Copyright 2025-2026 Verso Industries
"""

from dataclasses import dataclass
from enum import IntEnum

import tensorflow as tf

# Import native ops loader - REQUIRED, no fallback
from highnoon._native import load_highnoon_core

_highnoon_core = load_highnoon_core()


# =============================================================================
# MEMORY TYPE ENUM
# =============================================================================


class MemoryType(IntEnum):
    """Memory types matching C++ enum."""

    CONTENT_ADDRESSED = 0
    PRODUCT_KEY = 1
    HOPFIELD = 2
    ADAPTIVE = 3
    HIERARCHICAL = 4


# =============================================================================
# MEMORY CONFIGURATION
# =============================================================================


@dataclass
class MemoryConfig:
    """Configuration for memory operations."""

    # Core parameters
    mem_type: MemoryType = MemoryType.CONTENT_ADDRESSED
    batch_size: int = 1
    num_slots: int = 256
    slot_dim: int = 512
    query_dim: int = 512
    epsilon: float = 1e-6

    # Content-addressed parameters
    temperature: float = 1.0
    normalize_keys: bool = True

    # Product-key parameters
    codebook_size: int = 64
    subkey_dim: int = 256
    product_k: int = 8

    # Hopfield parameters
    beta: float = 1.0
    num_iterations: int = 1

    # Adaptive parameters
    surprise_threshold: float = 0.5
    decay_rate: float = 0.99
    write_strength: float = 0.1

    # Hierarchical parameters
    num_levels: int = 3
    ctqw_gamma: float = 0.1

    def validate(self) -> bool:
        """Validate configuration."""
        if self.mem_type not in MemoryType:
            return False
        if self.num_slots < 1 or self.slot_dim < 1:
            return False
        return True


# =============================================================================
# UNIFIED MEMORY DISPATCH (NATIVE ONLY)
# =============================================================================


def unified_memory_read(
    query: tf.Tensor, memory: tf.Tensor, config: MemoryConfig, aux_data: tf.Tensor | None = None
) -> tuple:
    """Unified memory read dispatcher.

    Dispatches to native C++ kernel. No Python fallback.

    Args:
        query: Query vector [query_dim] or [batch, query_dim]
        memory: Memory matrix [num_slots, slot_dim]
        config: Memory configuration
        aux_data: Auxiliary data (codebooks for product-key, etc.)

    Returns:
        Tuple of (output, attention_weights)

    Raises:
        RuntimeError: If native ops are not available
    """
    if aux_data is None:
        aux_data = tf.zeros([1], dtype=tf.float32)

    return _highnoon_core.unified_memory_system_op(
        query=query,
        memory=memory,
        aux_data=aux_data,
        mem_type=int(config.mem_type),
        batch_size=config.batch_size,
        num_slots=config.num_slots,
        slot_dim=config.slot_dim,
        query_dim=config.query_dim,
        codebook_size=config.codebook_size,
        subkey_dim=config.subkey_dim,
        product_k=config.product_k,
        temperature=config.temperature,
        beta=config.beta,
        num_iterations=config.num_iterations,
        epsilon=config.epsilon,
    )


# =============================================================================
# KERAS LAYERS
# =============================================================================


class UnifiedMemoryLayer(tf.keras.layers.Layer):
    """Keras layer wrapper for unified memory operations."""

    def __init__(self, config: MemoryConfig, name: str | None = None, **kwargs):
        super().__init__(name=name or f"memory_{config.mem_type.name.lower()}", **kwargs)
        self.config = config

    def build(self, input_shape):
        """Build layer - create memory bank."""
        self.memory = self.add_weight(
            name="memory",
            shape=(self.config.num_slots, self.config.slot_dim),
            initializer=tf.keras.initializers.Orthogonal(),
            trainable=True,
        )

        if self.config.mem_type == MemoryType.PRODUCT_KEY:
            # Create codebooks for product-key memory
            self.codebook_a = self.add_weight(
                name="codebook_a",
                shape=(self.config.codebook_size, self.config.subkey_dim),
                initializer="random_normal",
                trainable=True,
            )
            self.codebook_b = self.add_weight(
                name="codebook_b",
                shape=(self.config.codebook_size, self.config.subkey_dim),
                initializer="random_normal",
                trainable=True,
            )

        super().build(input_shape)

    def call(self, query, training=None):
        """Forward pass - memory read."""
        aux_data = None

        if self.config.mem_type == MemoryType.PRODUCT_KEY:
            aux_data = tf.concat(
                [tf.reshape(self.codebook_a, [-1]), tf.reshape(self.codebook_b, [-1])], axis=0
            )

        output, attention = unified_memory_read(query, self.memory, self.config, aux_data)
        return output

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "mem_type": int(self.config.mem_type),
                "num_slots": self.config.num_slots,
                "slot_dim": self.config.slot_dim,
                "query_dim": self.config.query_dim,
            }
        )
        return config


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def content_addressed_memory(
    query: tf.Tensor, memory: tf.Tensor, num_slots: int = 256, temperature: float = 1.0, **kwargs
) -> tuple:
    """Content-addressed memory read.

    Standard attention-based memory with cosine similarity.

    Args:
        query: Query vector [dim]
        memory: Memory matrix [num_slots, slot_dim]
        num_slots: Number of memory slots
        temperature: Softmax temperature

    Returns:
        Tuple of (output, attention_weights)
    """
    slot_dim = memory.shape[-1] or 512

    config = MemoryConfig(
        mem_type=MemoryType.CONTENT_ADDRESSED,
        num_slots=num_slots,
        slot_dim=int(slot_dim),
        query_dim=int(query.shape[-1] or slot_dim),
        temperature=temperature,
        **kwargs,
    )
    return unified_memory_read(query, memory, config)


def product_key_memory(
    query: tf.Tensor,
    memory: tf.Tensor,
    codebook_a: tf.Tensor,
    codebook_b: tf.Tensor,
    num_slots: int = 4096,
    product_k: int = 8,
    **kwargs,
) -> tuple:
    """Product-key memory read with O(√M) lookup.

    Args:
        query: Query vector [2 * subkey_dim]
        memory: Memory matrix [num_slots, slot_dim]
        codebook_a: First codebook [codebook_size, subkey_dim]
        codebook_b: Second codebook [codebook_size, subkey_dim]
        num_slots: Number of memory slots
        product_k: Top-k per sub-codebook

    Returns:
        Tuple of (output, attention_weights)
    """
    codebook_size = codebook_a.shape[0] or 64
    subkey_dim = codebook_a.shape[-1] or 256
    slot_dim = memory.shape[-1] or 512

    config = MemoryConfig(
        mem_type=MemoryType.PRODUCT_KEY,
        num_slots=num_slots,
        slot_dim=int(slot_dim),
        query_dim=2 * int(subkey_dim),
        codebook_size=int(codebook_size),
        subkey_dim=int(subkey_dim),
        product_k=product_k,
        **kwargs,
    )

    aux_data = tf.concat([tf.reshape(codebook_a, [-1]), tf.reshape(codebook_b, [-1])], axis=0)

    return unified_memory_read(query, memory, config, aux_data)


def hopfield_memory(
    query: tf.Tensor,
    patterns: tf.Tensor,
    num_patterns: int = 256,
    beta: float = 1.0,
    num_iterations: int = 1,
    **kwargs,
) -> tuple:
    """Hopfield memory retrieval.

    Energy-based associative memory with exponential capacity.

    Args:
        query: Query/state vector [dim]
        patterns: Stored patterns [num_patterns, dim]
        num_patterns: Number of stored patterns
        beta: Inverse temperature (higher = sharper retrieval)
        num_iterations: Number of Hopfield update iterations

    Returns:
        Tuple of (retrieved_pattern, attention_weights)
    """
    dim = patterns.shape[-1] or 512

    config = MemoryConfig(
        mem_type=MemoryType.HOPFIELD,
        num_slots=num_patterns,
        slot_dim=int(dim),
        query_dim=int(dim),
        beta=beta,
        num_iterations=num_iterations,
        **kwargs,
    )
    return unified_memory_read(query, patterns, config)


def adaptive_memory(
    query: tf.Tensor,
    memory: tf.Tensor,
    num_slots: int = 256,
    surprise_threshold: float = 0.5,
    **kwargs,
) -> tuple:
    """Adaptive memory with surprise-gated writes.

    Titans-inspired memory with novelty detection.

    Args:
        query: Query vector [dim]
        memory: Memory matrix [num_slots, slot_dim]
        num_slots: Number of memory slots
        surprise_threshold: Threshold for write gating

    Returns:
        Tuple of (output, attention_weights)
    """
    slot_dim = memory.shape[-1] or 512

    config = MemoryConfig(
        mem_type=MemoryType.ADAPTIVE,
        num_slots=num_slots,
        slot_dim=int(slot_dim),
        query_dim=int(query.shape[-1] or slot_dim),
        surprise_threshold=surprise_threshold,
        **kwargs,
    )
    return unified_memory_read(query, memory, config)


__all__ = [
    "MemoryType",
    "MemoryConfig",
    "unified_memory_read",
    "UnifiedMemoryLayer",
    "content_addressed_memory",
    "product_key_memory",
    "hopfield_memory",
    "adaptive_memory",
]
