# highnoon/models/reasoning/reasoning_module.py
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

"""ReasoningModule: Linear-time reasoning stack for LLMs.

This module provides the full HSMN reasoning architecture using linear-time
blocks (Mamba SSM, TimeCrystal, WLAM, MoE) via native C++ ops.

The reasoning module uses the 'mamba_timecrystal_wlam_moe_hybrid' block pattern:
- SpatialBlock (Mamba SSM): O(L) linear-time sequence modeling
- TimeCrystalSequenceBlock: O(L) energy-conserving Hamiltonian dynamics
- WLAMBlock: O(L log L) wavelet-based attention
- MoELayer: O(L·k) sparse expert routing

Requires compiled native ops from build_secure.sh.
"""

import logging
from typing import Any

import tensorflow as tf

# Phase 201.1: HD Activation Checkpointing
# Phase 48+: Memory optimization flags
from highnoon.config import (  # Phase 26-36 Quantum Architecture
    COMPRESSED_DIM,
    HD_ACTIVATION_CTQW,
    HD_ACTIVATION_DIM,
    MAMBA2_CONV_DIM,
    MAMBA2_EXPAND_FACTOR,
    MAMBA2_HEAD_DIM,
    MAMBA2_STATE_DIM,
    NUM_EXPERTS,
    REASONING_BLOCK_PATTERN,
    REASONING_FF_DIM,
    REASONING_HEADS,
    STATE_BUS_DIM,
    STATE_BUS_INJECT_CONTEXT,
    STATE_BUS_SLOTS,
    SUPERPOSITION_DIM,
    TOP_K,
    UNITARY_RESIDUAL_INIT_ANGLE,
    USE_GRADIENT_CHECKPOINTING,
    USE_HD_ACTIVATION_CHECKPOINT,
    USE_QUANTUM_NORM,
    USE_STATE_BUS,
    USE_UNITARY_RESIDUAL,
    WLAM_NUM_HEADS,
    WLAM_WAVELET_KERNEL_SIZE,
)
from highnoon.models.reasoning.block_factory import create_reasoning_stack
from highnoon.models.reasoning.state_bus import GlobalStateBus

# Lazy import for HD checkpoint layer
_hd_checkpoint_layer_cls = None
_hd_checkpoint_loaded = False


def _load_hd_checkpoint_lazy():
    """Lazy load HD checkpoint layer to avoid slowing down module import."""
    global _hd_checkpoint_layer_cls, _hd_checkpoint_loaded
    if _hd_checkpoint_loaded:
        return _hd_checkpoint_layer_cls is not None
    try:
        from highnoon.training.hd_activation_checkpoint import HDCheckpointLayer

        _hd_checkpoint_layer_cls = HDCheckpointLayer
        _hd_checkpoint_loaded = True
        return True
    except Exception as e:
        logger.warning(f"HD Activation Checkpoint not available: {e}")
        _hd_checkpoint_loaded = True
        return False


# Quantum ops (lazy import to avoid load at module level)
_quantum_ops_loaded = False
_unitary_residual_forward = None
_unitary_norm_forward = None


def _load_quantum_ops_lazy():
    """Lazy load quantum ops to avoid slowing down module import."""
    global _quantum_ops_loaded, _unitary_residual_forward, _unitary_norm_forward
    if _quantum_ops_loaded:
        return
    try:
        from highnoon._native.ops.quantum_ops import unitary_norm_forward, unitary_residual_forward

        _unitary_residual_forward = unitary_residual_forward
        _unitary_norm_forward = unitary_norm_forward
        _quantum_ops_loaded = True
        logger.info("Quantum ops loaded for reasoning module")
    except Exception as e:
        logger.warning(f"Quantum ops not available: {e}")
        _quantum_ops_loaded = False


logger = logging.getLogger(__name__)


class AdaptiveBlockRouter(tf.keras.layers.Layer):
    """Router for adaptive block selection in reasoning stack.

    Learns to skip blocks for simple inputs, saving compute while
    maintaining capacity for complex reasoning.

    Args:
        num_blocks: Number of blocks to route.
        embedding_dim: Input embedding dimension.
        skip_threshold: Probability threshold below which block is skipped.
    """

    def __init__(self, num_blocks: int, embedding_dim: int, skip_threshold: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.embedding_dim = embedding_dim
        self.skip_threshold = skip_threshold

        # Lightweight router: pool → dense → block scores
        self.router_proj = tf.keras.layers.Dense(
            num_blocks,
            activation="sigmoid",
            name="block_router",
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Compute execution probabilities for each block.

        Args:
            hidden_states: Input [batch, seq_len, dim].
            training: Whether in training mode.

        Returns:
            Block execution probabilities [batch, num_blocks].
        """
        # Global average pooling over sequence
        pooled = tf.reduce_mean(hidden_states, axis=1)  # [batch, dim]

        # Compute block execution scores (0-1)
        block_scores = self.router_proj(pooled)  # [batch, num_blocks]

        if training:
            # During training, use Gumbel-Softmax for differentiability
            # Add noise to prevent premature convergence
            noise = tf.random.uniform(tf.shape(block_scores), 0, 0.1)
            block_scores = block_scores + noise

        return block_scores

    def get_skip_mask(
        self,
        block_scores: tf.Tensor,
        training: bool = False,
    ) -> list[tf.Tensor]:
        """Convert scores to binary skip masks per block.

        Args:
            block_scores: [batch, num_blocks] execution probabilities.
            training: Whether in training mode.

        Returns:
            List of [batch, 1] masks, one per block.
        """
        masks = []
        for i in range(self.num_blocks):
            score = block_scores[:, i : i + 1]  # [batch, 1]
            if training:
                # Soft mask during training
                masks.append(score)
            else:
                # Hard mask during inference
                masks.append(tf.cast(score > self.skip_threshold, tf.float32))
        return masks

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_blocks": self.num_blocks,
                "embedding_dim": self.embedding_dim,
                "skip_threshold": self.skip_threshold,
            }
        )
        return config


class ReasoningModule(tf.keras.layers.Layer):
    """Linear-time reasoning module with HSMN blocks via native ops.

    Uses the 'mamba_timecrystal_wlam_moe_hybrid' block pattern:
    - SpatialBlock (Mamba SSM): O(L) linear-time sequence modeling
    - TimeCrystalSequenceBlock: O(L) energy-conserving Hamiltonian dynamics
    - WLAMBlock: O(L log L) wavelet-based attention
    - MoELayer: O(L·k) sparse expert routing

    Requires compiled native ops (build_secure.sh).

    Args:
        num_layers: Number of reasoning blocks.
        embedding_dim: Model embedding dimension.
        compressed_dim: Compressed representation dimension.
        block_pattern: Block pattern name.
        num_heads: Number of attention heads (for WLAM).
        ff_dim: Feedforward dimension (for MoE).
        num_experts: Number of MoE experts.
        top_k: Top-K experts per token.
        dropout: Dropout rate.
        mamba_state_dim: Mamba state dimension.
        mamba_conv_dim: Mamba convolution dimension.
        wlam_num_heads: WLAM attention heads.
        wlam_kernel_size: WLAM wavelet kernel size.

    Example:
        >>> rm = ReasoningModule(num_layers=4, embedding_dim=512)
        >>> output = rm(hidden_states, training=False)
    """

    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        compressed_dim: int = COMPRESSED_DIM,
        block_pattern: str = REASONING_BLOCK_PATTERN,
        num_heads: int = REASONING_HEADS,
        ff_dim: int = REASONING_FF_DIM,
        num_experts: int = NUM_EXPERTS,
        top_k: int = TOP_K,
        dropout: float = 0.1,
        mamba_state_dim: int = MAMBA2_STATE_DIM,
        mamba_conv_dim: int = MAMBA2_CONV_DIM,
        mamba_head_dim: int = MAMBA2_HEAD_DIM,
        mamba_expand_factor: int = MAMBA2_EXPAND_FACTOR,
        superposition_dim: int = SUPERPOSITION_DIM,
        wlam_num_heads: int = WLAM_NUM_HEADS,
        wlam_kernel_size: int = WLAM_WAVELET_KERNEL_SIZE,
        # Phase 5.1: Adaptive Block Selection
        use_adaptive_routing: bool = False,
        skip_threshold: float = 0.3,
        # Quantum Training Flags
        use_quantum_memory_replay: bool = False,
        use_entanglement_loss: bool = False,
        use_quantum_holographic_memory: bool = False,
        # Phase 201.1: HD Activation Checkpointing - override global HD_ACTIVATION_DIM
        hd_activation_dim: int | None = None,
        **kwargs,
    ):
        keras_kwargs = {k: v for k, v in kwargs.items() if k in ["name", "dtype", "trainable"]}
        super().__init__(**keras_kwargs)

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.compressed_dim = compressed_dim
        self.block_pattern = block_pattern
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout_rate = dropout

        # Build reasoning blocks using the factory (O(L) linear-time blocks)
        (
            self.reasoning_blocks,
            self._fused_block_weights,
            self._block_weight_counts,
            self._block_descriptors,
        ) = create_reasoning_stack(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            name="reasoning_stack",
            reasoning_heads=num_heads,
            reasoning_ff_dim=ff_dim,
            block_pattern=block_pattern,
            spatial_state_dim=mamba_state_dim,
            spatial_conv_dim=mamba_conv_dim,
            mamba2_head_dim=mamba_head_dim,
            mamba2_expand_factor=mamba_expand_factor,
            num_experts=num_experts,
            top_k=top_k,
            superposition_dim=superposition_dim,
            wlam_num_heads=wlam_num_heads,
            wlam_wavelet_kernel_size=wlam_kernel_size,
            # Quantum Flags
            use_quantum_memory_replay=use_quantum_memory_replay,
            use_entanglement_loss=use_entanglement_loss,
            use_quantum_holographic_memory=use_quantum_holographic_memory,
        )

        # Phase 30: Quantum Normalization flag (set early since it's used for layer_norm setup)
        self.use_quantum_norm = USE_QUANTUM_NORM

        # Layer normalization and dropout
        # When use_quantum_norm=True, we use QNorm instead of LayerNorm in forward pass
        # but still create LayerNorm as fallback (marked non-trainable to avoid gradient warnings)
        self.layer_norms = [
            tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"layer_norm_{i}")
            for i in range(len(self.reasoning_blocks))
        ]
        # Mark layer_norms as non-trainable when quantum norm is used to avoid gradient warnings
        if self.use_quantum_norm:
            for norm in self.layer_norms:
                norm.trainable = False
        self.dropout_layers = [
            tf.keras.layers.Dropout(dropout, name=f"dropout_{i}")
            for i in range(len(self.reasoning_blocks))
        ]

        # Phase 34: Unitary Residual Connections
        self.use_unitary_residual = USE_UNITARY_RESIDUAL
        if self.use_unitary_residual:
            _load_quantum_ops_lazy()
            if _quantum_ops_loaded:
                # Learnable blend angles per block
                self.blend_angles = [
                    self.add_weight(
                        name=f"blend_angle_{i}",
                        shape=(),
                        initializer=tf.keras.initializers.Constant(UNITARY_RESIDUAL_INIT_ANGLE),
                        trainable=True,
                    )
                    for i in range(len(self.reasoning_blocks))
                ]
                logger.info("  - Phase 34: Unitary Residual enabled")
            else:
                self.use_unitary_residual = False
                logger.warning("Phase 34: Unitary Residual disabled (ops not available)")

        # Phase 30: Quantum Normalization weights (replaces LayerNorm when enabled)
        if self.use_quantum_norm:
            _load_quantum_ops_lazy()
            if _quantum_ops_loaded:
                # Quantum norm uses scale+bias like LayerNorm
                self.qnorm_scales = [
                    self.add_weight(
                        name=f"qnorm_scale_{i}",
                        shape=(embedding_dim,),
                        initializer="ones",
                        trainable=True,
                    )
                    for i in range(len(self.reasoning_blocks))
                ]
                self.qnorm_biases = [
                    self.add_weight(
                        name=f"qnorm_bias_{i}",
                        shape=(embedding_dim,),
                        initializer="zeros",
                        trainable=True,
                    )
                    for i in range(len(self.reasoning_blocks))
                ]
                logger.info("  - Phase 30: Quantum Normalization enabled")
            else:
                self.use_quantum_norm = False
                logger.warning("Phase 30: Quantum Norm disabled (ops not available)")

        # Enhancement 6: GlobalStateBus for cross-block communication
        self.use_state_bus = USE_STATE_BUS
        self.inject_context = STATE_BUS_INJECT_CONTEXT
        if self.use_state_bus:
            self.state_bus = GlobalStateBus(
                bus_dim=STATE_BUS_DIM,
                num_slots=STATE_BUS_SLOTS,
                name="reasoning_state_bus",
            )
            # Context projection to match embedding dim for injection
            if self.inject_context:
                self.context_proj = tf.keras.layers.Dense(
                    embedding_dim,
                    name="bus_context_proj",
                )

        # Phase 5.1: Adaptive Block Selection Router
        self.use_adaptive_routing = use_adaptive_routing
        self.adaptive_router = None
        if use_adaptive_routing:
            self.adaptive_router = AdaptiveBlockRouter(
                num_blocks=len(self.reasoning_blocks),
                embedding_dim=embedding_dim,
                skip_threshold=skip_threshold,
                name="adaptive_block_router",
            )
            logger.info("  - Phase 5.1: Adaptive Block Routing enabled")

        # Phase 48+: Gradient Checkpointing for memory efficiency
        self.use_gradient_checkpointing = USE_GRADIENT_CHECKPOINTING

        # Phase 201.1: HD Activation Checkpointing (enhanced gradient checkpointing)
        # CRITICAL: Use hd_activation_dim parameter if provided, else fall back to global
        # This ensures trial-specific hd_dim is used, preventing memory bloat
        effective_hd_activation_dim = hd_activation_dim if hd_activation_dim else HD_ACTIVATION_DIM
        self.use_hd_activation_checkpoint = (
            USE_HD_ACTIVATION_CHECKPOINT and self.use_gradient_checkpointing
        )
        self._hd_checkpoint_wrappers = None
        if self.use_hd_activation_checkpoint:
            if _load_hd_checkpoint_lazy() and _hd_checkpoint_layer_cls is not None:
                # Create HD checkpoint wrappers for each reasoning block
                self._hd_checkpoint_wrappers = [
                    _hd_checkpoint_layer_cls(
                        sublayers=[block],
                        hd_dim=effective_hd_activation_dim,
                        use_ctqw=HD_ACTIVATION_CTQW,
                        name=f"hd_checkpoint_{i}",
                    )
                    for i, block in enumerate(self.reasoning_blocks)
                ]
                logger.info(
                    f"  - Phase 201.1: HD Activation Checkpointing enabled "
                    f"(hd_dim={effective_hd_activation_dim}, ctqw={HD_ACTIVATION_CTQW})"
                )
            else:
                self.use_hd_activation_checkpoint = False
                logger.warning("HD Activation Checkpoint disabled (layer not available)")

        if self.use_gradient_checkpointing and not self.use_hd_activation_checkpoint:
            logger.info(
                "  - Phase 48+: Standard Gradient Checkpointing enabled (reduces peak memory ~40-50%)"
            )

        logger.info(
            f"ReasoningModule: {len(self.reasoning_blocks)} blocks, dim={embedding_dim}, "
            f"pattern={block_pattern}, types={[type(b).__name__ for b in self.reasoning_blocks]}, "
            f"state_bus={'enabled' if self.use_state_bus else 'disabled'}"
        )

    def build(self, input_shape):
        """Build all child layers to ensure weights are created.

        This method is called automatically by Keras before the first call()
        when using graph mode or Functional API. We explicitly build all
        child layers to avoid warnings about unbuilt state.

        Args:
            input_shape: Input tensor shape [batch, seq_len, embedding_dim].
        """
        # Build reasoning blocks
        block_input_shape = input_shape
        for block in self.reasoning_blocks:
            if hasattr(block, "build") and not block.built:
                block.build(block_input_shape)

        # Build layer norms
        for norm in self.layer_norms:
            if not norm.built:
                norm.build(input_shape)

        # Build dropout layers (they don't need explicit building but for consistency)
        for dropout in self.dropout_layers:
            if hasattr(dropout, "build") and not dropout.built:
                dropout.build(input_shape)

        # Build adaptive router if enabled
        if self.adaptive_router is not None and not self.adaptive_router.built:
            self.adaptive_router.build(input_shape)

        # Build state bus components if enabled (only in eager mode during actual call)
        # State bus is initialized dynamically in call() when batch size is known

        # Build context projection if used
        if hasattr(self, "context_proj") and not self.context_proj.built:
            self.context_proj.build(
                (
                    None,
                    STATE_BUS_DIM if hasattr(self, "state_bus") else self.embedding_dim,
                )
            )

        super().build(input_shape)

    def call(
        self,
        hidden_states: tf.Tensor,
        training: bool = False,
        mask: tf.Tensor | None = None,
        reset_state_bus: bool = True,
    ) -> tf.Tensor:
        """Forward pass through linear-time reasoning blocks.

        Args:
            hidden_states: Input tensor [batch, seq_len, embedding_dim].
            training: Whether in training mode.
            mask: Optional attention mask.
            reset_state_bus: Whether to reset State Bus at sequence start.

        Returns:
            Output tensor [batch, seq_len, embedding_dim].
        """
        x = hidden_states

        # State Bus: Use graph-compatible stateless operations during training
        # The state is tracked as a tensor that flows through the blocks
        bus_slots = None
        if self.use_state_bus:
            batch_size = tf.shape(x)[0]
            # Initialize slots as zeros tensor (graph-compatible)
            # Use state_bus's actual max_slots and bus_dim to match write_gate output
            bus_slots = tf.zeros([batch_size, self.state_bus.max_slots, self.state_bus.bus_dim])

        # Phase 5.1: Compute block execution masks if adaptive routing enabled
        block_masks = None
        if self.use_adaptive_routing and self.adaptive_router is not None:
            block_scores = self.adaptive_router(x, training=training)
            block_masks = self.adaptive_router.get_skip_mask(block_scores, training)

        for i, (block, norm, dropout) in enumerate(
            zip(self.reasoning_blocks, self.layer_norms, self.dropout_layers)
        ):
            residual = x
            # Phase 30: Use quantum norm or standard LayerNorm
            if self.use_quantum_norm and _quantum_ops_loaded:
                x_norm, _ = _unitary_norm_forward(x, self.qnorm_scales[i], self.qnorm_biases[i])
                x = x_norm
            else:
                x = norm(x)

            # Phase 48+/201.1: Gradient Checkpointing - wrap block call to save memory
            # HD variant uses holographic encoding, standard uses tf.recompute_grad
            if self.use_hd_activation_checkpoint and training and self._hd_checkpoint_wrappers:
                # Use HD checkpoint wrapper for this block
                block_output = self._hd_checkpoint_wrappers[i](x, training=True)
            elif self.use_gradient_checkpointing and training:
                # Standard recompute_grad: discards activations and recomputes during backward
                @tf.recompute_grad
                def _checkpointed_block_call(block_input, blk=block):
                    return blk(block_input, training=True)

                block_output = _checkpointed_block_call(x)
            else:
                block_output = block(x, training=training)

            # Handle blocks that return tuples (stateful blocks)
            if isinstance(block_output, tuple):
                x = block_output[0]
            else:
                x = block_output

            x = dropout(x, training=training)

            # Phase 5.1: Apply block mask (soft during training, hard during inference)
            if block_masks is not None:
                mask_i = block_masks[i][:, :, None]  # [batch, 1, 1] for broadcasting
                # x = mask * block_output + (1 - mask) * 0 for residual
                # Final: residual + mask * block_contribution
                x = residual + mask_i * (x - residual)
            else:
                # Phase 34: Use unitary residual if enabled
                if self.use_unitary_residual and _quantum_ops_loaded:
                    angle = self.blend_angles[i]
                    x = _unitary_residual_forward(residual, x, angle)
                else:
                    x = residual + x

            # Enhancement 6: State Bus read/write after each block (graph-compatible)
            if self.use_state_bus and bus_slots is not None:
                # Pool sequence to get block summary
                x_pooled = tf.reduce_mean(x, axis=1)  # [batch, dim]

                # Graph-compatible read: query-based attention over slots
                q = self.state_bus.read_query(x_pooled)  # [B, bus_dim]
                attention_scores = tf.einsum("bd,bsd->bs", q, bus_slots)  # [B, num_slots]
                attention_scores = attention_scores / tf.sqrt(float(self.state_bus.bus_dim))
                attention_weights = tf.nn.softmax(attention_scores)
                bus_context = tf.einsum("bs,bsd->bd", attention_weights, bus_slots)  # [B, bus_dim]

                # Inject context back into hidden states if enabled
                if self.inject_context:
                    context_expanded = self.context_proj(bus_context)[:, None, :]
                    x = x + context_expanded * 0.1  # Scale factor for stability

                # Graph-compatible write: update slots tensor (not Variable.assign)
                gate = self.state_bus.write_gate(x_pooled)  # [B, num_slots]
                content = self.state_bus.write_value(x_pooled)  # [B, bus_dim]

                # Gated update: blend old slots with new content
                bus_slots = (
                    bus_slots * (1.0 - gate[:, :, tf.newaxis])
                    + content[:, tf.newaxis, :] * gate[:, :, tf.newaxis]
                )

        # Return tensor directly (not dict) for Keras Functional API compatibility
        return x

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape for Keras graph tracing.

        This is called during model building (graph mode) to determine
        output shapes without executing the layer.

        Args:
            input_shape: Input tensor shape.

        Returns:
            Output shape (same as input for this layer).
        """
        # ReasoningModule preserves input shape: [batch, seq, dim] -> [batch, seq, dim]
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        return {
            "num_layers": self.num_layers,
            "embedding_dim": self.embedding_dim,
            "compressed_dim": self.compressed_dim,
            "block_pattern": self.block_pattern,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "dropout": self.dropout_rate,
            "use_state_bus": self.use_state_bus,
            "inject_context": self.inject_context,
            # Phase 5.1: Adaptive Block Selection
            "use_adaptive_routing": self.use_adaptive_routing,
        }

    def reset_bus(self, batch_size: int) -> None:
        """Reset the State Bus for a new sequence.

        Args:
            batch_size: Batch size for slot initialization.
        """
        if self.use_state_bus:
            self.state_bus.initialize_slots(batch_size)

    @property
    def block_types(self) -> list[str]:
        """Get list of block type names."""
        return [type(b).__name__ for b in self.reasoning_blocks]

    @property
    def block_weight_counts(self) -> list[int]:
        """Get weight counts per block for fused ops."""
        return self._block_weight_counts

    @property
    def block_descriptors(self) -> list[str]:
        """Get JSON descriptors for blocks."""
        return self._block_descriptors
