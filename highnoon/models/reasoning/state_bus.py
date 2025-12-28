# highnoon/models/reasoning/state_bus.py
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

"""Phase 12.11: Cross-Block State Bus.

This module implements a global communication channel that all reasoning blocks
can read from and write to, enabling cross-block information sharing.

Key features:
- Fixed number of communication slots (O(1) per read/write)
- Learnable slot embeddings for type-aware communication
- Gated read/write for selective information flow
- Integrates into ReasoningModule between block calls

Enhancements (Phase 12.11.2-5):
- Adaptive Slot Count (AdaSlot): Dynamic slot selection with Gumbel-Softmax
- Typed/Hierarchical Slots: Slot type embeddings for specialization
- Quantum-Inspired Superposition: TT-decomposed slot representation
- Multi-Head Slot Attention: Parallel read patterns with shared K/V

Inspired by Transformer-XL recurrent memory mechanisms.
"""

from __future__ import annotations

import logging
from typing import Any

import tensorflow as tf

from highnoon._native.ops.fused_state_bus_op import fused_state_bus, fused_state_bus_available
from highnoon._native.ops.fused_superposition_slots_op import (
    superposition_collapse_available,
    superposition_collapse_read,
    superposition_write,
    superposition_write_available,
)
from highnoon.config import (
    STATE_BUS_ADAPTIVE,
    STATE_BUS_BOND_DIM,
    STATE_BUS_DIM,
    STATE_BUS_GUMBEL_HARD,
    STATE_BUS_GUMBEL_TEMP,
    STATE_BUS_MAX_SLOTS,
    STATE_BUS_NUM_HEADS,
    STATE_BUS_NUM_TYPES,
    STATE_BUS_SLOTS,
    STATE_BUS_SUPERPOSITION,
    STATE_BUS_TYPED,
    USE_FUSED_STATE_BUS,
    # Phase 127: Unified Quantum Bus config
    UNIFIED_BUS_ADAPTIVE,
    UNIFIED_BUS_COHERENCE_THRESHOLD,
    UNIFIED_BUS_ENTANGLEMENT_INIT,
    UNIFIED_BUS_MPS_BOND_DIM,
    UNIFIED_BUS_PROPAGATION_RATE,
)

logger = logging.getLogger(__name__)


def gumbel_softmax(
    logits: tf.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
) -> tf.Tensor:
    """Gumbel-Softmax for differentiable discrete selection.

    Args:
        logits: Unnormalized log-probabilities [batch, num_classes].
        temperature: Softmax temperature (lower = more discrete).
        hard: Use straight-through estimator for hard samples.

    Returns:
        Soft or hard one-hot samples [batch, num_classes].
    """
    # Sample from Gumbel(0, 1)
    u = tf.random.uniform(tf.shape(logits), minval=1e-7, maxval=1.0 - 1e-7)
    gumbel_noise = -tf.math.log(-tf.math.log(u))

    # Add noise and apply temperature-scaled softmax
    y = tf.nn.softmax((logits + gumbel_noise) / temperature)

    if hard:
        # Straight-through estimator: hard in forward, soft in backward
        y_hard = tf.one_hot(tf.argmax(y, axis=-1), tf.shape(logits)[-1])
        y = tf.stop_gradient(y_hard - y) + y

    return y


class SuperpositionSlotBuffer:
    """Enhancement 4: Quantum-Inspired Slot Superposition.

    This class manages slots that exist in superposition across multiple
    state "universes". During writes, all universes are updated. During
    reads, the universes collapse to a single context via attention.

    The representation uses superposition_dim parallel state sets:
    - Slots shape: [batch, num_slots, superposition_dim, bus_dim]
    - Collapse uses attention over superposition_dim

    This enables richer information encoding where different aspects of
    context are stored in parallel dimensions, then collapsed based on
    the query.
    """

    def __init__(
        self,
        bus_dim: int,
        num_slots: int,
        superposition_dim: int,
        bond_dim: int = 4,
    ):
        """Initialize SuperpositionSlotBuffer.

        Args:
            bus_dim: Dimension of bus communication.
            num_slots: Number of communication slots.
            superposition_dim: Number of parallel universes.
            bond_dim: TT bond dimension (for future TT-core representation).
        """
        self.bus_dim = bus_dim
        self.num_slots = num_slots
        self.superposition_dim = superposition_dim
        self.bond_dim = bond_dim

        # Buffer: [batch, num_slots, superposition_dim, bus_dim]
        self._buffer: tf.Variable | None = None
        self._batch_size: int | None = None

    def initialize(self, batch_size: int, type_embeddings: tf.Tensor | None = None) -> None:
        """Initialize superposition buffer.

        Uses tf.init_scope() for Keras graph mode compatibility.

        Args:
            batch_size: Batch size.
            type_embeddings: Optional type embeddings to initialize from.
        """
        self._batch_size = batch_size

        # Use tf.init_scope to create variables outside of tf.function graphs
        with tf.init_scope():
            if type_embeddings is not None:
                # Initialize from type embeddings, broadcast to superposition dim
                type_indices = tf.range(self.num_slots) % tf.shape(type_embeddings)[0]
                slot_init = tf.nn.embedding_lookup(type_embeddings, type_indices)
                # [num_slots, bus_dim] -> [batch, num_slots, superdim, bus_dim]
                slot_init = slot_init[None, :, None, :]
                slot_init = tf.tile(slot_init, [batch_size, 1, self.superposition_dim, 1])
                # Add small noise per superposition dimension for diversity
                noise = tf.random.normal(
                    [batch_size, self.num_slots, self.superposition_dim, self.bus_dim],
                    stddev=0.01,
                )
                slot_init = slot_init + noise
            else:
                # Initialize to zeros with small noise
                slot_init = tf.random.normal(
                    [batch_size, self.num_slots, self.superposition_dim, self.bus_dim],
                    stddev=0.01,
                )

            self._buffer = tf.Variable(
                slot_init,
                trainable=False,
                name="superposition_slots",
            )

    def reset(self, type_embeddings: tf.Tensor | None = None) -> None:
        """Reset buffer to initial state."""
        if self._buffer is not None and self._batch_size is not None:
            self.initialize(self._batch_size, type_embeddings)

    @property
    def buffer(self) -> tf.Variable | None:
        """Get the buffer variable."""
        return self._buffer

    def collapse_read(
        self,
        query: tf.Tensor,
        collapse_proj: tf.keras.layers.Dense,
        temperature: float = 1.0,
    ) -> tf.Tensor:
        """Collapse superposition via query-conditioned attention.

        Uses C++ SIMD-optimized kernel when available.

        Args:
            query: Query tensor [batch, query_dim].
            collapse_proj: Projection for collapse attention.
            temperature: Softmax temperature for collapse.

        Returns:
            Collapsed slots [batch, num_slots, bus_dim].
        """
        if self._buffer is None:
            raise ValueError("Buffer not initialized. Call initialize() first.")

        # Try C++ kernel first
        if superposition_collapse_available():
            return superposition_collapse_read(
                query=query,
                buffer=self._buffer,
                collapse_weight=collapse_proj.kernel,
                collapse_bias=collapse_proj.bias,
                num_slots=self.num_slots,
                superposition_dim=self.superposition_dim,
                bus_dim=self.bus_dim,
                temperature=temperature,
            )

        # TensorFlow implementation
        tf.shape(query)[0]

        # Project query for collapse attention
        q = collapse_proj(query)  # [B, bus_dim]

        # Compute attention over superposition dimension for each slot
        # buffer: [B, S, D, bus_dim], q: [B, bus_dim]
        # Attention scores: [B, S, D]
        scores = tf.einsum("bsdv,bv->bsd", self._buffer, q) / tf.sqrt(
            tf.cast(self.bus_dim, tf.float32)
        )

        # Temperature-scaled softmax over superposition dimension
        weights = tf.nn.softmax(scores / temperature, axis=-1)  # [B, S, D]

        # Weighted collapse: [B, S, D] x [B, S, D, V] -> [B, S, V]
        collapsed = tf.einsum("bsd,bsdv->bsv", weights, self._buffer)

        return collapsed

    def superposition_write(
        self,
        content: tf.Tensor,
        gate: tf.Tensor,
        active_mask: tf.Tensor | None = None,
    ) -> None:
        """Write to all superposition dimensions with gating.

        Uses C++ SIMD-optimized kernel when available.

        Args:
            content: Content to write [batch, bus_dim].
            gate: Write gate [batch, num_slots].
            active_mask: Optional slot activity mask [batch, num_slots].
        """
        if self._buffer is None:
            raise ValueError("Buffer not initialized. Call initialize() first.")

        # Apply active mask to gate if provided
        if active_mask is not None:
            gate = gate * active_mask

        # Try C++ kernel first
        if superposition_write_available():
            new_buffer = superposition_write(
                content=content,
                gate=gate,
                buffer=self._buffer,
                num_slots=self.num_slots,
                superposition_dim=self.superposition_dim,
                bus_dim=self.bus_dim,
            )
            self._buffer.assign(new_buffer)
            return

        # TensorFlow implementation
        # Expand content for all slots and superposition dims
        # content: [B, V] -> [B, 1, 1, V]
        content_expanded = content[:, None, None, :]

        # gate: [B, S] -> [B, S, 1, 1]
        gate_expanded = gate[:, :, None, None]

        # Update all superposition dimensions
        updated_buffer = self._buffer * (1.0 - gate_expanded) + content_expanded * gate_expanded

        self._buffer.assign(updated_buffer)


class GlobalStateBus(tf.keras.layers.Layer):
    """Fixed-size communication channel for cross-block information sharing.

    This layer provides a lightweight global state mechanism that processes
    in O(num_slots) time, making it compatible with O(L) sequence processing.

    Architecture:
        - Slots: [batch, num_slots, bus_dim] communication buffer
        - Read: Query-based attention over slots, returns context
        - Write: Gated update to slots based on content

    Enhancements:
        - Adaptive: Variable slot count with Gumbel-Softmax selection
        - Typed: Slot type embeddings for specialization
        - Superposition: TT-decomposed slots for richer representation
        - Multi-Head: Parallel read patterns with shared K/V

    Example:
        >>> bus = GlobalStateBus(bus_dim=64, num_slots=8)
        >>> x = tf.random.normal((2, 128, 512))  # [B, L, D]
        >>> context = bus.read(tf.reduce_mean(x, axis=1), batch_size=2)
        >>> bus.write(tf.reduce_mean(x, axis=1))
        >>> context.shape
        TensorShape([2, 64])

    Attributes:
        bus_dim: Dimension of bus communication.
        num_slots: Number of communication slots.
    """

    def __init__(
        self,
        bus_dim: int = STATE_BUS_DIM,
        num_slots: int = STATE_BUS_SLOTS,
        use_fused_op: bool | None = None,
        # Enhancement 2: Adaptive Slot Count
        use_adaptive: bool = STATE_BUS_ADAPTIVE,
        max_slots: int = STATE_BUS_MAX_SLOTS,
        gumbel_temp: float = STATE_BUS_GUMBEL_TEMP,
        gumbel_hard: bool = STATE_BUS_GUMBEL_HARD,
        # Enhancement 3: Typed/Hierarchical Slots
        use_typed: bool = STATE_BUS_TYPED,
        num_types: int = STATE_BUS_NUM_TYPES,
        # Enhancement 4: Quantum-Inspired Superposition
        use_superposition: bool = STATE_BUS_SUPERPOSITION,
        bond_dim: int = STATE_BUS_BOND_DIM,
        # Enhancement 5: Multi-Head Slot Attention
        num_heads: int = STATE_BUS_NUM_HEADS,
        name: str = "global_state_bus",
        **kwargs: Any,
    ) -> None:
        """Initialize GlobalStateBus.

        Args:
            bus_dim: Dimension of bus communication (default: 64).
            num_slots: Number of slots (default: 8).
            use_fused_op: Use C++ fused op when available.
            use_adaptive: Enable adaptive slot selection (AdaSlot).
            max_slots: Maximum slots for adaptive mode.
            gumbel_temp: Gumbel-Softmax temperature.
            gumbel_hard: Use hard samples during inference.
            use_typed: Enable typed slot embeddings.
            num_types: Number of slot types.
            use_superposition: Enable quantum-inspired superposition.
            bond_dim: TT bond dimension for superposition.
            num_heads: Number of attention heads for reads.
            name: Layer name.
            **kwargs: Additional layer arguments.
        """
        super().__init__(name=name, **kwargs)
        self.bus_dim = bus_dim
        self.num_slots = num_slots
        self.use_fused_op = USE_FUSED_STATE_BUS if use_fused_op is None else use_fused_op

        # Enhancement 2: Adaptive Slot Count
        self.use_adaptive = use_adaptive
        self.max_slots = max_slots if use_adaptive else num_slots
        self.gumbel_temp = gumbel_temp
        self.gumbel_hard = gumbel_hard

        # Enhancement 3: Typed Slots
        self.use_typed = use_typed
        self.num_types = min(num_types, self.max_slots)

        # Enhancement 4: Superposition
        self.use_superposition = use_superposition
        self.bond_dim = bond_dim

        # Enhancement 5: Multi-Head Attention
        self.num_heads = num_heads
        self.head_dim = bus_dim // num_heads

        # Core projections
        self.read_query = tf.keras.layers.Dense(
            self.bus_dim,
            name=f"{name}_read_query",
        )
        self.write_gate = tf.keras.layers.Dense(
            self.max_slots,
            activation="sigmoid",
            name=f"{name}_write_gate",
        )
        self.write_value = tf.keras.layers.Dense(
            self.bus_dim,
            name=f"{name}_write_value",
        )

        # Enhancement 2: Slot selection network
        if self.use_adaptive:
            self.slot_selector = tf.keras.layers.Dense(
                self.max_slots,
                name=f"{name}_slot_selector",
            )

        # Enhancement 3: Type embeddings
        if self.use_typed:
            self.type_embeddings: tf.Variable | None = None
            self.type_proj = tf.keras.layers.Dense(
                self.bus_dim,
                name=f"{name}_type_proj",
            )

        # Enhancement 4: Superposition buffer and collapse projection
        self._superposition_buffer: SuperpositionSlotBuffer | None = None
        if self.use_superposition:
            self._superposition_buffer = SuperpositionSlotBuffer(
                bus_dim=bus_dim,
                num_slots=self.max_slots,
                superposition_dim=8,  # Default to 8 parallel universes
                bond_dim=bond_dim,
            )
            self.collapse_proj = tf.keras.layers.Dense(
                self.bus_dim,
                name=f"{name}_collapse_proj",
            )
            logger.info(
                f"GlobalStateBus: Superposition mode enabled with "
                f"{self._superposition_buffer.superposition_dim} parallel universes"
            )

        # Enhancement 5: Multi-head projections
        if self.num_heads > 1:
            self.head_proj = tf.keras.layers.Dense(
                self.bus_dim,
                name=f"{name}_head_proj",
            )
            self.head_combine = tf.keras.layers.Dense(
                self.bus_dim,
                name=f"{name}_head_combine",
            )

        # Slot state (managed internally)
        self._slots: tf.Variable | None = None
        self._batch_size: int | None = None
        self._active_mask: tf.Tensor | None = None
        self._fused_op_available = self.use_fused_op and fused_state_bus_available()

        if self.use_fused_op and not self._fused_op_available:
            logger.warning(
                "FusedStateBus op not available, falling back to Python implementation. "
                "Rebuild with ./build_secure.sh to enable."
            )

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer weights.

        Args:
            input_shape: Shape of input tensor.
        """
        # Build sublayers if not already built
        if not self.read_query.built:
            self.read_query.build(input_shape)
        if not self.write_gate.built:
            self.write_gate.build(input_shape)
        if not self.write_value.built:
            self.write_value.build(input_shape)

        # Enhancement 3: Initialize type embeddings
        if self.use_typed and self.type_embeddings is None:
            self.type_embeddings = self.add_weight(
                name=f"{self.name}_type_embeddings",
                shape=[self.num_types, self.bus_dim],
                initializer="glorot_uniform",
                trainable=True,
            )

        super().build(input_shape)

    def initialize_slots(self, batch_size: int) -> None:
        """Initialize slot buffer for a batch.

        Uses tf.init_scope() to ensure variables are created in the default
        graph context, not inside a tf.function graph. This is required for
        Keras Functional API compatibility.

        Args:
            batch_size: Batch size for slot initialization.
        """
        self._batch_size = batch_size

        # Use tf.init_scope to create variables outside of any tf.function graph
        # This is critical for Keras Functional API / graph mode compatibility
        with tf.init_scope():
            # Enhancement 4: Initialize superposition buffer if enabled
            if self.use_superposition and self._superposition_buffer is not None:
                type_emb = self.type_embeddings if self.use_typed else None
                self._superposition_buffer.initialize(batch_size, type_emb)
                # Also initialize regular slots for fallback
                self._slots = tf.Variable(
                    tf.zeros([batch_size, self.max_slots, self.bus_dim]),
                    trainable=False,
                    name=f"{self.name}_slots",
                )
            elif self.use_typed and self.type_embeddings is not None:
                # Enhancement 3: Initialize slots with type embeddings
                type_indices = tf.range(self.max_slots) % self.num_types
                slot_init = tf.nn.embedding_lookup(self.type_embeddings, type_indices)
                slot_init = tf.tile(slot_init[None, :, :], [batch_size, 1, 1])
                self._slots = tf.Variable(
                    slot_init,
                    trainable=False,
                    name=f"{self.name}_slots",
                )
            else:
                slot_init = tf.zeros([batch_size, self.max_slots, self.bus_dim])
                self._slots = tf.Variable(
                    slot_init,
                    trainable=False,
                    name=f"{self.name}_slots",
                )

        # Reset active mask for adaptive mode
        if self.use_adaptive:
            self._active_mask = tf.ones([batch_size, self.max_slots])
        else:
            self._active_mask = None

    def _ensure_slots(self, batch_size: int | tf.Tensor) -> None:
        """Ensure slot buffer matches current batch size."""
        if self._slots is None:
            self.initialize_slots(batch_size)
            return
        if isinstance(batch_size, int) and isinstance(self._batch_size, int):
            if batch_size != self._batch_size:
                self.initialize_slots(batch_size)

    def reset_slots(self) -> None:
        """Reset slots to zero (or type embeddings if typed)."""
        # Reset superposition buffer if enabled
        if self.use_superposition and self._superposition_buffer is not None:
            type_emb = self.type_embeddings if self.use_typed else None
            self._superposition_buffer.reset(type_emb)

        if self._slots is not None:
            if self.use_typed and self.type_embeddings is not None:
                type_indices = tf.range(self.max_slots) % self.num_types
                slot_init = tf.nn.embedding_lookup(self.type_embeddings, type_indices)
                slot_init = tf.tile(slot_init[None, :, :], [self._batch_size, 1, 1])
                self._slots.assign(slot_init)
            else:
                self._slots.assign(tf.zeros_like(self._slots))

    def _compute_adaptive_mask(self, query: tf.Tensor, training: bool) -> tf.Tensor:
        """Enhancement 2: Compute adaptive slot selection mask.

        Args:
            query: Query tensor [batch, dim].
            training: Whether in training mode.

        Returns:
            Slot selection mask [batch, max_slots].
        """
        logits = self.slot_selector(query)  # [B, max_slots]

        # During training use Gumbel-Softmax, during inference use hard selection
        temp = self.gumbel_temp if training else 0.1
        hard = self.gumbel_hard or not training

        return gumbel_softmax(logits, temperature=temp, hard=hard)

    def _multihead_attention(
        self,
        query: tf.Tensor,
        slots: tf.Tensor,
        mask: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """Enhancement 5: Multi-head attention over slots.

        Uses a simplified GQA-style approach where K/V (slots) are shared
        across all query heads. Each head produces independent attention
        weights over the shared slots.

        Args:
            query: Query tensor [batch, dim].
            slots: Slot buffer [batch, num_slots, bus_dim].
            mask: Optional attention mask [batch, num_slots].

        Returns:
            Attended context [batch, bus_dim].
        """
        batch_size = tf.shape(query)[0]
        num_slots = tf.shape(slots)[1]

        # Project query to multi-head format
        # [B, dim] -> [B, bus_dim] -> [B, num_heads, head_dim]
        q = self.head_proj(query)  # [B, bus_dim]
        q = tf.reshape(q, [batch_size, self.num_heads, self.head_dim])

        # Slots are shared across heads (GQA-style)
        # Reshape slots to allow broadcasting: [B, S, D] -> [B, S, H, D/H]
        slots_reshaped = tf.reshape(slots, [batch_size, num_slots, self.num_heads, self.head_dim])

        # Compute attention scores: Q @ K^T
        # q: [B, H, D/H], slots: [B, S, H, D/H]
        # Use einsum for batched dot product
        scores = tf.einsum("bhd,bshd->bhs", q, slots_reshaped)  # [B, H, S]
        scores = scores / tf.sqrt(tf.cast(self.head_dim, tf.float32))

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask[:, None, :]  # [B, 1, S]
            scores = scores * mask_expanded + (1 - mask_expanded) * -1e9

        weights = tf.nn.softmax(scores, axis=-1)  # [B, H, S]

        # Compute weighted sum of values (slots)
        context = tf.einsum("bhs,bshd->bhd", weights, slots_reshaped)  # [B, H, D/H]
        context = tf.reshape(context, [batch_size, self.bus_dim])  # [B, D]

        # Combine heads with learned projection
        context = self.head_combine(context)

        return context

    def read(
        self,
        query: tf.Tensor,
        batch_size: int | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Read from bus using query-based attention.

        Complexity: O(num_slots) = O(1).

        Args:
            query: Query tensor [batch, dim].
            batch_size: Batch size (required if slots not initialized).
            training: Whether in training mode.

        Returns:
            Read context [batch, bus_dim].
        """
        # Initialize slots if needed
        if batch_size is None:
            batch_size = tf.shape(query)[0]
        self._ensure_slots(batch_size)

        if not self.built:
            self.build(query.shape)

        # Enhancement 2: Compute adaptive slot mask
        mask = None
        if self.use_adaptive:
            mask = self._compute_adaptive_mask(query, training)
            self._active_mask = mask

        # Enhancement 4: Superposition collapse path
        if self.use_superposition and self._superposition_buffer is not None:
            # Collapse superposition to get effective slots
            collapsed_slots = self._superposition_buffer.collapse_read(
                query, self.collapse_proj, temperature=self.gumbel_temp
            )
            # Standard attention read over collapsed slots
            q = self.read_query(query)
            attention_scores = tf.einsum("bd,bsd->bs", q, collapsed_slots)
            attention_scores = attention_scores / tf.sqrt(float(self.bus_dim))
            if mask is not None:
                attention_scores = attention_scores * mask + (1 - mask) * -1e9
            attention_weights = tf.nn.softmax(attention_scores)
            return tf.einsum("bs,bsd->bd", attention_weights, collapsed_slots)

        # Use C++ fused op for basic case (no enhancements active)
        if (
            self._fused_op_available
            and not self.use_adaptive
            and not self.use_typed
            and not self.use_superposition
            and self.num_heads == 1
        ):
            context, _ = fused_state_bus(
                query=query,
                write_value=query,
                slots=self._slots,
                read_query_weight=self.read_query.kernel,
                read_query_bias=self.read_query.bias,
                write_gate_weight=self.write_gate.kernel,
                write_gate_bias=self.write_gate.bias,
                write_value_weight=self.write_value.kernel,
                write_value_bias=self.write_value.bias,
                num_slots=self.num_slots,
                bus_dim=self.bus_dim,
                write_enabled=False,
            )
            return context

        # Enhancement 5: Multi-head attention
        if self.num_heads > 1:
            return self._multihead_attention(query, self._slots, mask)

        # Standard attention read
        q = self.read_query(query)  # [B, bus_dim]

        # Attention over slots (O(num_slots))
        attention_scores = tf.einsum("bd,bsd->bs", q, self._slots)  # [B, num_slots]
        attention_scores = attention_scores / tf.sqrt(float(self.bus_dim))

        # Apply adaptive mask if active
        if mask is not None:
            attention_scores = attention_scores * mask + (1 - mask) * -1e9

        attention_weights = tf.nn.softmax(attention_scores)

        # Weighted sum of slots
        return tf.einsum("bs,bsd->bd", attention_weights, self._slots)  # [B, bus_dim]

    def write(self, value: tf.Tensor, training: bool = False) -> None:
        """Write to bus using gated update.

        Complexity: O(num_slots) = O(1).

        Args:
            value: Value to write [batch, dim].
            training: Whether in training mode.
        """
        batch_size = tf.shape(value)[0]
        self._ensure_slots(batch_size)

        if not self.built:
            self.build(value.shape)

        # Enhancement 4: Superposition write path
        if self.use_superposition and self._superposition_buffer is not None:
            gate = self.write_gate(value)  # [B, num_slots]
            content = self.write_value(value)  # [B, bus_dim]
            if self.use_adaptive and self._active_mask is not None:
                gate = gate * self._active_mask
            self._superposition_buffer.superposition_write(
                content, gate, active_mask=self._active_mask
            )
            return

        # Use C++ fused op for basic case
        if (
            self._fused_op_available
            and not self.use_adaptive
            and not self.use_typed
            and not self.use_superposition
            and self.num_heads == 1
        ):
            _, slots_new = fused_state_bus(
                query=value,
                write_value=value,
                slots=self._slots,
                read_query_weight=self.read_query.kernel,
                read_query_bias=self.read_query.bias,
                write_gate_weight=self.write_gate.kernel,
                write_gate_bias=self.write_gate.bias,
                write_value_weight=self.write_value.kernel,
                write_value_bias=self.write_value.bias,
                num_slots=self.num_slots,
                bus_dim=self.bus_dim,
                write_enabled=True,
            )
            self._slots.assign(slots_new)
            return

        # Compute write content and gate
        gate = self.write_gate(value)  # [B, num_slots]
        content = self.write_value(value)  # [B, bus_dim]

        # Apply adaptive mask to gate if active
        if self.use_adaptive and self._active_mask is not None:
            gate = gate * self._active_mask

        # Enhancement 3: Apply type projection to content
        if self.use_typed:
            content = self.type_proj(content)

        # Gated update: blend old slots with new content
        updated_slots = (
            self._slots * (1.0 - gate[:, :, tf.newaxis])
            + content[:, tf.newaxis, :] * gate[:, :, tf.newaxis]
        )
        self._slots.assign(updated_slots)

    def call(
        self,
        query: tf.Tensor,
        write_value: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Read from bus (and optionally write).

        Args:
            query: Query for reading [batch, dim].
            write_value: Optional value to write [batch, dim].
            training: Whether in training mode.

        Returns:
            Read context [batch, bus_dim].
        """
        context = self.read(query, training=training)
        if write_value is not None:
            self.write(write_value, training=training)
        return context

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "bus_dim": self.bus_dim,
                "num_slots": self.num_slots,
                "use_adaptive": self.use_adaptive,
                "max_slots": self.max_slots,
                "gumbel_temp": self.gumbel_temp,
                "gumbel_hard": self.gumbel_hard,
                "use_typed": self.use_typed,
                "num_types": self.num_types,
                "use_superposition": self.use_superposition,
                "bond_dim": self.bond_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


class CoherenceTracker(tf.keras.layers.Layer):
    """Phase 127: Tracks quantum coherence across block states.

    Computes pairwise coherence measures between blocks to determine
    entanglement strength for cross-block communication.

    Attributes:
        bus_dim: Dimension of bus communication.
        coherence_threshold: Minimum coherence for propagation.
    """

    def __init__(
        self,
        bus_dim: int,
        coherence_threshold: float = 0.85,
        name: str = "coherence_tracker",
        **kwargs: Any,
    ) -> None:
        """Initialize CoherenceTracker.

        Args:
            bus_dim: Dimension of state vectors.
            coherence_threshold: Minimum coherence threshold.
            name: Layer name.
            **kwargs: Additional layer arguments.
        """
        super().__init__(name=name, **kwargs)
        self.bus_dim = bus_dim
        self.coherence_threshold = coherence_threshold

        # Learnable coherence projection
        self.coherence_proj = tf.keras.layers.Dense(
            bus_dim,
            name=f"{name}_coherence_proj",
        )

    def call(self, block_states: tf.Tensor) -> tf.Tensor:
        """Compute coherence between block states.

        Args:
            block_states: Block states [batch, num_blocks, dim].

        Returns:
            Coherence matrix [num_blocks, num_blocks] averaged over batch.
        """
        # Project for coherence computation
        projected = self.coherence_proj(block_states)  # [B, num_blocks, bus_dim]

        # Normalize
        norms = tf.norm(projected, axis=-1, keepdims=True) + 1e-8
        normalized = projected / norms  # [B, num_blocks, bus_dim]

        # Compute pairwise coherence via inner product
        # [B, num_blocks, bus_dim] @ [B, bus_dim, num_blocks] -> [B, num_blocks, num_blocks]
        coherence = tf.matmul(normalized, normalized, transpose_b=True)

        # Average over batch and apply absolute value
        coherence = tf.reduce_mean(tf.abs(coherence), axis=0)  # [num_blocks, num_blocks]

        return coherence

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "bus_dim": self.bus_dim,
                "coherence_threshold": self.coherence_threshold,
            }
        )
        return config


class UnifiedQuantumBus(tf.keras.layers.Layer):
    """Phase 127: Unified Quantum Entanglement Bus.

    Combines existing quantum buses (GlobalStateBus, Quantum Coherence Bus,
    Quantum Teleport Bus) into a single coherent entanglement-mediated
    cross-block communication system with adaptive entanglement strength.

    Key features:
    - Integrates GlobalStateBus for slot-based communication
    - Uses CoherenceTracker for pairwise coherence measurement
    - MPS-based entanglement representation for efficient memory
    - Learnable entanglement weights between block pairs

    Complexity: O(n · d) for entanglement propagation
    Memory: O(χ² · d) with MPS bond dimension χ

    Attributes:
        bus_dim: Dimension of bus communication.
        num_blocks: Number of reasoning blocks.
        mps_bond_dim: MPS bond dimension for entanglement.
    """

    def __init__(
        self,
        bus_dim: int,
        num_blocks: int,
        mps_bond_dim: int = UNIFIED_BUS_MPS_BOND_DIM,
        use_adaptive: bool = UNIFIED_BUS_ADAPTIVE,
        coherence_threshold: float = UNIFIED_BUS_COHERENCE_THRESHOLD,
        entanglement_init: float = UNIFIED_BUS_ENTANGLEMENT_INIT,
        propagation_rate: float = UNIFIED_BUS_PROPAGATION_RATE,
        name: str = "unified_quantum_bus",
        **kwargs: Any,
    ) -> None:
        """Initialize UnifiedQuantumBus.

        Args:
            bus_dim: Dimension of bus communication.
            num_blocks: Number of reasoning blocks.
            mps_bond_dim: MPS bond dimension for entanglement.
            use_adaptive: Enable adaptive entanglement strength.
            coherence_threshold: Minimum coherence for propagation.
            entanglement_init: Initial entanglement strength.
            propagation_rate: Rate for entanglement updates.
            name: Layer name.
            **kwargs: Additional layer arguments.
        """
        super().__init__(name=name, **kwargs)
        self.bus_dim = bus_dim
        self.num_blocks = num_blocks
        self.mps_bond_dim = mps_bond_dim
        self.use_adaptive = use_adaptive
        self.coherence_threshold = coherence_threshold
        self.entanglement_init = entanglement_init
        self.propagation_rate = propagation_rate

        # Integrate existing GlobalStateBus
        # Note: Disable superposition on internal state bus since UnifiedQuantumBus
        # handles entanglement at a higher level
        self.state_bus = GlobalStateBus(
            bus_dim=bus_dim,
            num_slots=num_blocks,
            use_adaptive=use_adaptive,
            use_superposition=False,  # Entanglement handled by UnifiedQuantumBus
            name=f"{name}_state_bus",
        )

        # Coherence tracking
        self.coherence_tracker = CoherenceTracker(
            bus_dim=bus_dim,
            coherence_threshold=coherence_threshold,
            name=f"{name}_coherence",
        )

        # Entanglement strength projection
        self.entanglement_proj = tf.keras.layers.Dense(
            num_blocks,
            activation="sigmoid",
            name=f"{name}_entanglement_proj",
        )

        # Output projection
        self.output_proj = tf.keras.layers.Dense(
            bus_dim,
            name=f"{name}_output_proj",
        )

        # Learnable entanglement strength matrix [num_blocks, num_blocks]
        self._entanglement_strength: tf.Variable | None = None
        self._initialized = False

        logger.info(
            f"UnifiedQuantumBus: Initialized with {num_blocks} blocks, "
            f"bond_dim={mps_bond_dim}, adaptive={use_adaptive}"
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer weights.

        Args:
            input_shape: Shape of input tensor.
        """
        # Initialize entanglement strength matrix
        self._entanglement_strength = self.add_weight(
            name=f"{self.name}_entanglement_strength",
            shape=[self.num_blocks, self.num_blocks],
            initializer=tf.keras.initializers.Constant(self.entanglement_init),
            trainable=self.use_adaptive,
        )
        super().build(input_shape)

    def initialize(self, batch_size: int) -> None:
        """Initialize bus state for a batch.

        Args:
            batch_size: Batch size for initialization.
        """
        self.state_bus.initialize_slots(batch_size)
        self._initialized = True

    def propagate_entanglement(
        self,
        block_states: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Propagate quantum correlations across blocks via MPS.

        This is the core operation of the unified bus. It:
        1. Computes pairwise coherence between blocks
        2. Weights cross-block communication by entanglement strength
        3. Returns entangled states and coherence measurements

        Args:
            block_states: States from all blocks [batch, num_blocks, dim].
            training: Whether in training mode.

        Returns:
            Tuple of:
            - entangled_states: Entanglement-weighted states [batch, num_blocks, dim]
            - coherence: Pairwise coherence matrix [num_blocks, num_blocks]
        """
        batch_size = tf.shape(block_states)[0]

        if not self._initialized:
            self.initialize(batch_size)

        if not self.built:
            self.build(block_states.shape)

        # Compute pairwise coherence
        coherence = self.coherence_tracker(block_states)  # [num_blocks, num_blocks]

        # Get effective entanglement strength (gated by coherence)
        effective_entanglement = self._entanglement_strength * coherence

        # Apply coherence threshold
        mask = tf.cast(coherence > self.coherence_threshold, tf.float32)
        effective_entanglement = effective_entanglement * mask

        # Normalize rows for proper weighting
        row_sums = tf.reduce_sum(effective_entanglement, axis=1, keepdims=True) + 1e-8
        normalized_entanglement = effective_entanglement / row_sums

        # Weight cross-block communication by entanglement strength
        # [num_blocks, num_blocks] @ [batch, num_blocks, dim] -> [batch, num_blocks, dim]
        # Use einsum for batched operation
        entangled_states = tf.einsum(
            "ij,bjd->bid",
            normalized_entanglement,
            block_states,
        )

        # Update entanglement strength based on coherence if adaptive
        if self.use_adaptive and training:
            # Compute coherence-based update
            coherence_update = (coherence - self.coherence_threshold) * self.propagation_rate
            new_strength = self._entanglement_strength + coherence_update
            # Clip to [0, 1] range
            new_strength = tf.clip_by_value(new_strength, 0.0, 1.0)
            self._entanglement_strength.assign(new_strength)

        return entangled_states, coherence

    def read(
        self,
        query: tf.Tensor,
        batch_size: int | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Read from unified bus using query-based attention.

        Args:
            query: Query tensor [batch, dim].
            batch_size: Batch size (required if not initialized).
            training: Whether in training mode.

        Returns:
            Read context [batch, bus_dim].
        """
        return self.state_bus.read(query, batch_size=batch_size, training=training)

    def write(self, value: tf.Tensor, training: bool = False) -> None:
        """Write to unified bus.

        Args:
            value: Value to write [batch, dim].
            training: Whether in training mode.
        """
        self.state_bus.write(value, training=training)

    def call(
        self,
        block_states: tf.Tensor,
        query: tf.Tensor | None = None,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Forward pass through unified quantum bus.

        Args:
            block_states: States from all blocks [batch, num_blocks, dim].
            query: Optional query for bus read [batch, dim].
            training: Whether in training mode.

        Returns:
            Tuple of:
            - entangled_states: Entanglement-propagated states [batch, num_blocks, dim]
            - coherence: Coherence matrix [num_blocks, num_blocks]
        """
        # Propagate entanglement
        entangled_states, coherence = self.propagate_entanglement(
            block_states, training=training
        )

        # Apply output projection first
        entangled_states = self.output_proj(entangled_states)

        # Optionally read/write via state bus
        if query is not None:
            # Ensure state bus is built
            if not self.state_bus.built:
                self.state_bus.build(query.shape)
            bus_context = self.read(query, training=training)
            # Combine with entangled states (broadcast across blocks)
            # bus_context: [batch, bus_dim] -> [batch, 1, bus_dim]
            entangled_states = entangled_states + bus_context[:, None, :]

        return entangled_states, coherence

    def get_entanglement_strength(self) -> tf.Tensor:
        """Get current entanglement strength matrix.

        Returns:
            Entanglement strength matrix [num_blocks, num_blocks].
        """
        return self._entanglement_strength

    def receive_block_coherence(
        self,
        block_index: int,
        coherence: float,
    ) -> None:
        """Receive coherence report from a quantum block (Phase 130.3).

        Integrates coherence from QMamba/Q-SSM blocks to modulate
        entanglement propagation strength.

        Args:
            block_index: Index of the reporting block.
            coherence: Coherence value in [0, 1] from block's last_coherence.
        """
        if self._entanglement_strength is not None and block_index < self.num_blocks:
            # Modulate row/column for this block based on coherence
            # High coherence -> stronger entanglement with other blocks
            current = self._entanglement_strength.numpy()
            scale = 0.5 + 0.5 * coherence  # Map [0,1] -> [0.5, 1.0]

            # Update using propagation rate for smooth adaptation
            current[block_index, :] = (
                (1 - self.propagation_rate) * current[block_index, :] +
                self.propagation_rate * scale * current[block_index, :]
            )
            current[:, block_index] = (
                (1 - self.propagation_rate) * current[:, block_index] +
                self.propagation_rate * scale * current[:, block_index]
            )
            self._entanglement_strength.assign(current)

    def receive_hopfield_energy(
        self,
        energy: tf.Tensor,
        block_indices: list[int] | None = None,
    ) -> None:
        """Receive Hopfield energy scores for downstream conditioning (Phase 130.5).

        Energy scores modulate the coherence threshold for entanglement propagation.
        High energy (strong retrieval) -> lower threshold -> more propagation.
        Low energy (weak retrieval) -> higher threshold -> less propagation.

        Args:
            energy: Hopfield energy score(s) [batch] or scalar.
            block_indices: Optional indices of blocks to affect (None = all).
        """
        # Normalize energy to [0, 1] via sigmoid
        normalized_energy = tf.nn.sigmoid(energy)
        mean_energy = tf.reduce_mean(normalized_energy).numpy()

        # Adjust coherence threshold based on energy
        # High energy -> lower threshold for easier propagation
        energy_adjustment = 0.1 * (mean_energy - 0.5)  # [-0.05, 0.05]
        adjusted_threshold = max(0.5, min(0.95,
            self.coherence_threshold - energy_adjustment
        ))

        # Store for use in next propagate_entanglement call
        self._energy_adjusted_threshold = adjusted_threshold

    def reset(self) -> None:
        """Reset bus state."""
        self.state_bus.reset_slots()
        if self._entanglement_strength is not None:
            self._entanglement_strength.assign(
                tf.ones([self.num_blocks, self.num_blocks]) * self.entanglement_init
            )
        self._initialized = False
        self._energy_adjusted_threshold = None

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "bus_dim": self.bus_dim,
                "num_blocks": self.num_blocks,
                "mps_bond_dim": self.mps_bond_dim,
                "use_adaptive": self.use_adaptive,
                "coherence_threshold": self.coherence_threshold,
                "entanglement_init": self.entanglement_init,
                "propagation_rate": self.propagation_rate,
            }
        )
        return config


__all__ = ["GlobalStateBus", "gumbel_softmax", "CoherenceTracker", "UnifiedQuantumBus"]
