# highnoon/_native/ops/fused_superposition_slots_op.py
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

"""Python wrapper for superposition slot C++ operations.

Enhancement 4: Quantum-Inspired Slot Superposition

Provides:
- superposition_collapse_read: Collapse superposition via attention
- superposition_write: Gated update to all superposition dims
"""

from __future__ import annotations

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

# Attempt to load the C++ operations
_collapse_read_op: tf.Operation | None = None
_collapse_read_grad_op: tf.Operation | None = None
_write_op: tf.Operation | None = None

try:
    from highnoon._native import _load_consolidated_binary

    _highnoon_core = _load_consolidated_binary()
    if _highnoon_core is not None:
        _collapse_read_op = _highnoon_core.superposition_collapse_read
        _collapse_read_grad_op = _highnoon_core.superposition_collapse_read_grad
        _write_op = _highnoon_core.superposition_write
        logger.debug("Superposition slots C++ ops loaded successfully.")
    else:
        logger.warning(
            "Superposition slots C++ ops not available: consolidated binary not found. "
            "Using Python fallback. Rebuild with ./build_secure.sh to enable."
        )
except (ImportError, AttributeError) as e:
    logger.warning(
        f"Superposition slots C++ ops not available: {e}. "
        "Using Python fallback. Rebuild with ./build_secure.sh to enable."
    )


def superposition_collapse_available() -> bool:
    """Check if C++ superposition collapse op is available."""
    return _collapse_read_op is not None


def superposition_write_available() -> bool:
    """Check if C++ superposition write op is available."""
    return _write_op is not None


def superposition_collapse_read(
    query: tf.Tensor,
    buffer: tf.Tensor,
    collapse_weight: tf.Tensor,
    collapse_bias: tf.Tensor,
    num_slots: int,
    superposition_dim: int,
    bus_dim: int,
    temperature: float = 1.0,
) -> tf.Tensor:
    """Collapse superposition buffer via query-conditioned attention.

    Uses C++ SIMD-optimized kernel when available, falls back to Python.

    Args:
        query: Query tensor [batch, query_dim].
        buffer: Superposition buffer [batch, num_slots, superposition_dim, bus_dim].
        collapse_weight: Projection weights [query_dim, bus_dim].
        collapse_bias: Projection bias [bus_dim].
        num_slots: Number of slots.
        superposition_dim: Number of superposition dimensions.
        bus_dim: Bus dimension.
        temperature: Softmax temperature for collapse attention.

    Returns:
        Collapsed slots [batch, num_slots, bus_dim].
    """
    if _collapse_read_op is not None:
        return _collapse_read_with_gradient(
            query,
            buffer,
            collapse_weight,
            collapse_bias,
            num_slots,
            superposition_dim,
            bus_dim,
            temperature,
        )
    else:
        return _collapse_read_fallback(query, buffer, collapse_weight, collapse_bias, temperature)


def _collapse_read_with_gradient(
    query: tf.Tensor,
    buffer: tf.Tensor,
    collapse_weight: tf.Tensor,
    collapse_bias: tf.Tensor,
    num_slots: int,
    superposition_dim: int,
    bus_dim: int,
    temperature: float,
) -> tf.Tensor:
    """C++ collapse with custom gradient."""

    @tf.custom_gradient
    def _inner(q, buf, w, b):
        result = _collapse_read_op(
            query=q,
            buffer=buf,
            collapse_weight=w,
            collapse_bias=b,
            num_slots=num_slots,
            superposition_dim=superposition_dim,
            bus_dim=bus_dim,
            temperature=temperature,
        )

        def grad(grad_output):
            if _collapse_read_grad_op is not None:
                grads = _collapse_read_grad_op(
                    grad_collapsed=grad_output,
                    query=q,
                    buffer=buf,
                    collapse_weight=w,
                    collapse_bias=b,
                    num_slots=num_slots,
                    superposition_dim=superposition_dim,
                    bus_dim=bus_dim,
                    temperature=temperature,
                )
                return grads
            # Fallback: zeros
            return (
                tf.zeros_like(q),
                tf.zeros_like(buf),
                tf.zeros_like(w),
                tf.zeros_like(b),
            )

        return result, grad

    return _inner(query, buffer, collapse_weight, collapse_bias)


def _collapse_read_fallback(
    query: tf.Tensor,
    buffer: tf.Tensor,
    collapse_weight: tf.Tensor,
    collapse_bias: tf.Tensor,
    temperature: float,
) -> tf.Tensor:
    """Python fallback for collapse read."""
    tf.shape(buffer)[0]
    tf.shape(buffer)[1]
    tf.shape(buffer)[2]
    bus_dim = tf.shape(buffer)[3]

    # Project query: [B, Q] @ [Q, D] + [D] -> [B, D]
    proj_query = tf.matmul(query, collapse_weight) + collapse_bias

    # Compute attention scores for each slot
    # buffer: [B, S, D, V], proj_query: [B, V]
    # scores: [B, S, D]
    scores = tf.einsum("bv,bsdv->bsd", proj_query, buffer)
    scores = scores / tf.sqrt(tf.cast(bus_dim, tf.float32))

    # Softmax with temperature
    weights = tf.nn.softmax(scores / temperature, axis=-1)

    # Weighted sum: [B, S, D] x [B, S, D, V] -> [B, S, V]
    collapsed = tf.einsum("bsd,bsdv->bsv", weights, buffer)

    return collapsed


def superposition_write(
    content: tf.Tensor,
    gate: tf.Tensor,
    buffer: tf.Tensor,
    num_slots: int,
    superposition_dim: int,
    bus_dim: int,
) -> tf.Tensor:
    """Gated write to all superposition dimensions.

    Uses C++ SIMD-optimized kernel when available, falls back to Python.

    Args:
        content: Content to write [batch, bus_dim].
        gate: Per-slot write gate [batch, num_slots].
        buffer: Input buffer [batch, num_slots, superposition_dim, bus_dim].
        num_slots: Number of slots.
        superposition_dim: Number of superposition dimensions.
        bus_dim: Bus dimension.

    Returns:
        Updated buffer [batch, num_slots, superposition_dim, bus_dim].
    """
    if _write_op is not None:
        return _write_op(
            content=content,
            gate=gate,
            buffer=buffer,
            num_slots=num_slots,
            superposition_dim=superposition_dim,
            bus_dim=bus_dim,
        )
    else:
        return _write_fallback(content, gate, buffer)


def _write_fallback(
    content: tf.Tensor,
    gate: tf.Tensor,
    buffer: tf.Tensor,
) -> tf.Tensor:
    """Python fallback for superposition write."""
    # content: [B, V] -> [B, 1, 1, V]
    content_expanded = content[:, None, None, :]

    # gate: [B, S] -> [B, S, 1, 1]
    gate_expanded = gate[:, :, None, None]

    # Blended update
    return buffer * (1.0 - gate_expanded) + content_expanded * gate_expanded
