# highnoon/models/reasoning/continuous_thought.py
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

"""COCONUT Continuous Thought Module.

This module implements Chain of Continuous Thought (COCONUT) for latent-space
reasoning without generating intermediate tokens. The model reasons in
continuous embedding space rather than discrete token space.

Key Features:
    - Latent-space reasoning (no token generation overhead)
    - O(k) thought steps where k is configurable
    - O(d²) per thought step (linear with embedding dim squared)
    - Integration with existing LatentReasoningBlock

Complexity: O(n + k·d) where n = sequence length, k = thought steps, d = dim

CONSTRAINT: All computations use float32/64 precision only. No quantization.

Research Reference: COCONUT: Chain of Continuous Thought
"""

from __future__ import annotations

import logging
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from highnoon._native.ops.fused_continuous_thought_op import (
    fused_continuous_thought,
    fused_continuous_thought_available,
)
from highnoon.config import (
    COCONUT_COLLAPSE_THRESHOLD,
    COCONUT_CRYSTALLIZE_THRESHOLD,
    COCONUT_NUM_PATHS,
    COCONUT_USE_VQC_AMPLITUDES,
    CONTINUOUS_THOUGHT_STEPS,
    USE_CONTINUOUS_THOUGHT,
)

logger = logging.getLogger(__name__)

# Lazy load multi-path ops
_multipath_ops_loaded = False
_fused_coconut_bfs = None
_fused_coconut_dfs = None
_fused_coconut_crystallize = None
_fused_coconut_retrieve = None


def _load_multipath_ops() -> bool:
    """Lazy load Phase 87 multi-path CoCoNut ops.

    Uses fused_coconut_bfs_with_grad which has @tf.custom_gradient
    for proper gradient flow during training.
    Also loads DFS/Crystallize/Retrieve for S19 persistent reasoning.
    """
    global _multipath_ops_loaded, _fused_coconut_bfs
    global _fused_coconut_dfs, _fused_coconut_crystallize, _fused_coconut_retrieve

    if _multipath_ops_loaded:
        return _fused_coconut_bfs is not None

    _multipath_ops_loaded = True
    try:
        from highnoon._native.ops.fused_coconut_ops import (
            fused_coconut_bfs_available,
            fused_coconut_bfs_with_grad,
            fused_coconut_crystallize,
            fused_coconut_dfs_collapse,
            fused_coconut_retrieve,
        )

        if fused_coconut_bfs_available():
            # Use gradient-aware version for training support
            _fused_coconut_bfs = fused_coconut_bfs_with_grad
            # S19: Load DFS/Crystallize/Retrieve for persistent reasoning memory
            _fused_coconut_dfs = fused_coconut_dfs_collapse
            _fused_coconut_crystallize = fused_coconut_crystallize
            _fused_coconut_retrieve = fused_coconut_retrieve
            logger.debug("Phase 87 CoCoNut multi-path ops loaded (with S19 crystallization)")
            return True
    except ImportError as e:
        logger.debug(f"Multi-path CoCoNut ops not available: {e}")
    return False


class ThoughtProjector(layers.Layer):
    """Projects thought state through learned transformation.

    Applies a non-linear transformation to evolve the thought state
    through reasoning iterations.

    Attributes:
        hidden_dim: Dimension of hidden transformation.
        output_dim: Dimension of output (same as input for iterative use).
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int | None = None,
        dropout_rate: float = 0.1,
        name: str = "thought_projector",
        **kwargs: Any,
    ) -> None:
        """Initialize ThoughtProjector.

        Args:
            embedding_dim: Input/output embedding dimension.
            hidden_dim: Hidden layer dimension. Defaults to 4x embedding_dim.
            dropout_rate: Dropout rate for regularization.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.
        """
        super().__init__(name=name, **kwargs)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim or (4 * embedding_dim)
        self.dropout_rate = dropout_rate

        # Two-layer MLP with GELU activation
        self.dense_1 = layers.Dense(
            self.hidden_dim,
            activation="gelu",
            name=f"{name}_dense_1",
        )
        self.dense_2 = layers.Dense(
            embedding_dim,
            name=f"{name}_dense_2",
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{name}_norm",
        )
        self.dropout = layers.Dropout(dropout_rate)

    def call(
        self,
        thought: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply thought projection.

        Args:
            thought: Current thought state [batch, dim].
            training: Whether in training mode.

        Returns:
            Evolved thought state [batch, dim].
        """
        # Residual connection with pre-norm
        residual = thought

        # Layer norm -> dense -> GELU -> dense -> dropout
        x = self.layer_norm(thought)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dropout(x, training=training)

        # Residual connection
        return residual + x

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer weights."""
        self.layer_norm.build(input_shape)
        self.dense_1.build(input_shape)
        self.dense_2.build(tf.TensorShape([None, self.hidden_dim]))
        super().build(input_shape)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class ContinuousThoughtBlock(layers.Layer):
    """COCONUT-style continuous thought block for latent-space reasoning.

    This block processes hidden states through iterative thought steps
    in continuous embedding space, enabling reasoning without generating
    intermediate tokens.

    Execution is provided by the fused C++ op (no Python fallback).

    The thought process extracts a summary from the last position and
    iteratively refines it, then broadcasts the result back to all positions.

    Attributes:
        embedding_dim: Dimension of embeddings.
        num_thought_steps: Number of thinking iterations.
        thought_projector: Learned thought transformation.

    Complexity:
        - Extraction: O(1) - take last position
        - Thought steps: O(k) iterations, O(d²) each
        - Broadcast: O(n) - add to all positions
        - Total: O(n + k·d²)

    Example:
        >>> block = ContinuousThoughtBlock(embedding_dim=512, num_thought_steps=4)
        >>> hidden = tf.random.normal([2, 128, 512])  # [B, L, D]
        >>> output = block(hidden)
        >>> # output: [2, 128, 512] - enhanced with continuous thought
    """

    def __init__(
        self,
        embedding_dim: int,
        num_thought_steps: int = CONTINUOUS_THOUGHT_STEPS,
        hidden_dim: int | None = None,
        dropout_rate: float = 0.0,
        use_gating: bool = True,
        num_paths: int = COCONUT_NUM_PATHS,
        collapse_threshold: float = COCONUT_COLLAPSE_THRESHOLD,
        crystallize_threshold: float = COCONUT_CRYSTALLIZE_THRESHOLD,
        name: str = "continuous_thought",
        **kwargs: Any,
    ) -> None:
        """Initialize ContinuousThoughtBlock.

        Args:
            embedding_dim: Input/output embedding dimension.
            num_thought_steps: Number of thought iterations.
            hidden_dim: Hidden dimension for thought projector.
            dropout_rate: Must be 0.0 (C++ op has no dropout support).
            use_gating: Whether to use gated residual connection.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.

        Raises:
            ValueError: If embedding_dim is not positive.
            ValueError: If num_thought_steps is not positive.
        """
        super().__init__(name=name, **kwargs)

        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if num_thought_steps < 1:
            raise ValueError(f"num_thought_steps must be positive, got {num_thought_steps}")
        if dropout_rate != 0.0:
            raise ValueError(
                "dropout_rate must be 0.0 for fused ContinuousThoughtBlock "
                "(C++ op does not implement dropout)."
            )

        self.embedding_dim = embedding_dim
        self.num_thought_steps = num_thought_steps
        self.dropout_rate = dropout_rate
        self.use_gating = use_gating

        # Phase 87: Multi-path BFS configuration
        # Default 2 paths; Lite edition limited to max 8 paths (enforced by C++ op)
        self.num_paths = num_paths
        self.collapse_threshold = collapse_threshold
        self.crystallize_threshold = crystallize_threshold

        # Track if multi-path mode is available and requested
        self._use_multipath = self.num_paths > 1 and _load_multipath_ops()

        # Phase 130.4: VQC amplitude provider callback
        self._vqc_amplitude_provider: callable | None = None
        self._use_vqc_amplitudes = COCONUT_USE_VQC_AMPLITUDES

        # Thought projector for iterative refinement
        self.thought_projector = ThoughtProjector(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name=f"{name}_projector",
        )

        # Thought aggregation: combine sequence summary into thought
        self.thought_aggregator = layers.Dense(
            embedding_dim,
            name=f"{name}_aggregator",
        )

        # Broadcast projection: project thought back to sequence
        self.broadcast_projection = layers.Dense(
            embedding_dim,
            name=f"{name}_broadcast",
        )

        # Gate weights are always created; use_gating controls execution.
        self.gate = layers.Dense(
            embedding_dim,
            activation="sigmoid",
            name=f"{name}_gate",
        )

        # Layer normalization
        self.input_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{name}_input_norm",
        )
        self.output_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{name}_output_norm",
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer weights."""
        self.input_norm.build(input_shape)
        projector_shape = tf.TensorShape([None, self.embedding_dim])
        self.thought_aggregator.build(projector_shape)
        self.thought_projector.build(projector_shape)
        self.broadcast_projection.build(projector_shape)
        self.gate.build(input_shape)
        self.output_norm.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass through continuous thought block.

        Args:
            hidden_states: Input hidden states [batch, seq_len, embedding_dim].
            training: Whether in training mode.

        Returns:
            Enhanced hidden states [batch, seq_len, embedding_dim].
        """
        if not fused_continuous_thought_available():
            raise RuntimeError(
                "FusedContinuousThought C++ op not available. "
                "Build with: cd highnoon/_native && ./build_secure.sh"
            )

        hidden_states = tf.cast(hidden_states, tf.float32)

        # Phase 87: Use multi-path BFS if num_paths > 1 and ops available
        if self._use_multipath and _fused_coconut_bfs is not None:
            # Extract context for amplitude scoring (mean pool last dim)
            context = tf.reduce_mean(hidden_states, axis=1)  # [batch, dim]

            output, path_amplitudes = _fused_coconut_bfs(
                hidden_states=hidden_states,
                context=context,
                input_norm_gamma=self.input_norm.gamma,
                input_norm_beta=self.input_norm.beta,
                aggregator_weight=self.thought_aggregator.kernel,
                aggregator_bias=self.thought_aggregator.bias,
                projector_norm_gamma=self.thought_projector.layer_norm.gamma,
                projector_norm_beta=self.thought_projector.layer_norm.beta,
                projector_dense1_weight=self.thought_projector.dense_1.kernel,
                projector_dense1_bias=self.thought_projector.dense_1.bias,
                projector_dense2_weight=self.thought_projector.dense_2.kernel,
                projector_dense2_bias=self.thought_projector.dense_2.bias,
                broadcast_weight=self.broadcast_projection.kernel,
                broadcast_bias=self.broadcast_projection.bias,
                output_norm_gamma=self.output_norm.gamma,
                output_norm_beta=self.output_norm.beta,
                num_paths=self.num_paths,
                num_thought_steps=self.num_thought_steps,
                prune_threshold=0.1,
            )

            # Phase 130.4: VQC amplitudes for analysis (do NOT replace forward amplitudes)
            # The path_amplitudes from C++ fused_coconut_bfs are connected to the gradient
            # graph. Replacing them breaks the gradient chain and causes segfault during
            # backward pass. Instead, we store VQC amplitudes separately for analysis.
            vqc_amps_for_analysis = None
            if self._use_vqc_amplitudes and self._vqc_amplitude_provider is not None:
                try:
                    vqc_amplitudes = self._vqc_amplitude_provider(self.num_paths)
                    if vqc_amplitudes is not None:
                        # Use stop_gradient to ensure VQC amplitudes don't interfere
                        # with C++ op's gradient computation
                        vqc_amps_for_analysis = tf.stop_gradient(vqc_amplitudes)
                        logger.debug("VQC amplitudes computed for analysis")
                except Exception as e:
                    logger.debug(f"VQC amplitude provider failed: {e}")

            # Store both: C++ op amplitudes for gradient, VQC for analysis
            self._last_amplitudes = path_amplitudes  # Original C++ amplitudes (gradient-connected)
            self._last_vqc_amplitudes = vqc_amps_for_analysis  # VQC amplitudes (for analysis)
            return output

        # Single-path mode: use original fused continuous thought

        output, _ = fused_continuous_thought(
            x=hidden_states,
            input_norm_gamma=self.input_norm.gamma,
            input_norm_beta=self.input_norm.beta,
            aggregator_weight=self.thought_aggregator.kernel,
            aggregator_bias=self.thought_aggregator.bias,
            projector_norm_gamma=self.thought_projector.layer_norm.gamma,
            projector_norm_beta=self.thought_projector.layer_norm.beta,
            projector_dense1_weight=self.thought_projector.dense_1.kernel,
            projector_dense1_bias=self.thought_projector.dense_1.bias,
            projector_dense2_weight=self.thought_projector.dense_2.kernel,
            projector_dense2_bias=self.thought_projector.dense_2.bias,
            broadcast_weight=self.broadcast_projection.kernel,
            broadcast_bias=self.broadcast_projection.bias,
            gate_weight=self.gate.kernel,
            gate_bias=self.gate.bias,
            output_norm_gamma=self.output_norm.gamma,
            output_norm_beta=self.output_norm.beta,
            num_thought_steps=self.num_thought_steps,
            use_gating=self.use_gating,
        )

        return output

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_thought_steps": self.num_thought_steps,
                "use_gating": self.use_gating,
                "dropout_rate": self.dropout_rate,
                # Phase 87: Multi-path config
                "num_paths": self.num_paths,
                "collapse_threshold": self.collapse_threshold,
                "crystallize_threshold": self.crystallize_threshold,
            }
        )
        return config

    def set_vqc_amplitude_provider(
        self,
        provider: callable,
    ) -> None:
        """Set VQC amplitude provider for quantum path selection (Phase 130.4).

        The provider should be a callable that takes num_paths and returns
        Born rule probabilities from a VQC measurement.

        Args:
            provider: Callable(num_paths) -> tf.Tensor [num_paths] probabilities.
        """
        self._vqc_amplitude_provider = provider
        logger.debug("VQC amplitude provider set for COCONUT path selection")

    def set_qmamba_amplitude_provider(
        self,
        qmamba_block: object,
    ) -> None:
        """Set QMamba block as amplitude provider (S2: QMamba → COCONUT).

        Feeds QMamba's learned complex amplitudes directly into COCONUT's
        BFS path selection for better long-range dependency capture.

        Args:
            qmamba_block: QMambaBlock or UnifiedQSSMBlock instance with
                amplitudes_real and amplitudes_imag weights.
        """
        from highnoon import config as cfg

        if not getattr(cfg, "COCONUT_USE_QMAMBA_AMPLITUDES", True):
            logger.debug("COCONUT_USE_QMAMBA_AMPLITUDES disabled, skipping")
            return

        def qmamba_amplitude_provider(num_paths: int) -> tf.Tensor:
            """Extract Born rule probabilities from QMamba amplitudes."""
            try:
                # Get complex amplitudes from QMamba block
                alpha_real = qmamba_block.amplitudes_real
                alpha_imag = qmamba_block.amplitudes_imag

                # Born rule: |α|² = α_real² + α_imag²
                probs = alpha_real**2 + alpha_imag**2

                # Normalize
                probs = probs / (tf.reduce_sum(probs) + 1e-10)

                # Resize to num_paths if needed
                if tf.shape(probs)[0] != num_paths:
                    # Interpolate or pad
                    if tf.shape(probs)[0] > num_paths:
                        probs = probs[:num_paths]
                    else:
                        padding = tf.zeros([num_paths - tf.shape(probs)[0]])
                        probs = tf.concat([probs, padding], axis=0)
                    probs = probs / (tf.reduce_sum(probs) + 1e-10)

                return probs
            except Exception as e:
                logger.debug(f"QMamba amplitude extraction failed: {e}")
                return None

        self._vqc_amplitude_provider = qmamba_amplitude_provider
        # Use object.__setattr__ to bypass Keras layer tracking
        object.__setattr__(self, "_qmamba_block", qmamba_block)
        logger.debug("QMamba amplitude provider set for COCONUT (S2)")

    def get_last_amplitudes(self) -> tf.Tensor | None:
        """Get the last path amplitudes (for debugging or analysis).

        Returns:
            Last path amplitude tensor or None if not available.
        """
        return getattr(self, "_last_amplitudes", None)

    def get_qmamba_block(self) -> object | None:
        """Get the connected QMamba block for S2 synergy.

        Returns:
            QMamba block reference or None if not set.
        """
        return getattr(self, "_qmamba_block", None)

    def set_hopfield_energy(self, energy: float) -> None:
        """S5: Receive Hopfield retrieval energy for dynamic crystallization threshold.

        Strong retrieval (low energy) indicates confident memory match,
        so we raise the crystallization threshold (be more selective).
        Weak retrieval (high energy) indicates uncertain match,
        so we lower the threshold (crystallize more readily for exploration).

        Args:
            energy: Hopfield energy value (lower = better match).
        """
        from highnoon import config as cfg

        if not getattr(cfg, "QHPM_USE_HOPFIELD_THRESHOLD", True):
            return

        # Store base threshold on first call
        if not hasattr(self, "_base_crystallize_threshold"):
            self._base_crystallize_threshold = self.crystallize_threshold

        # Normalize energy to [0, 1] range using sigmoid
        # energy < 0 (strong match) -> normalized < 0.5
        # energy > 0 (weak match) -> normalized > 0.5
        normalized_energy = 1.0 / (1.0 + tf.exp(-energy))
        if hasattr(normalized_energy, "numpy"):
            normalized_energy = float(normalized_energy.numpy())

        # Adjust threshold: strong match -> higher threshold, weak match -> lower
        # Range: [base - 0.15, base + 0.1]
        adjustment = 0.25 * (0.5 - normalized_energy)  # [-0.125, 0.125]
        new_threshold = self._base_crystallize_threshold + adjustment
        self.crystallize_threshold = max(0.5, min(0.99, new_threshold))

        logger.debug(
            "[COCONUT-S5] Hopfield energy=%.3f -> crystallize_threshold=%.3f",
            energy,
            self.crystallize_threshold,
        )

    def get_crystallize_threshold(self) -> float:
        """Get the current crystallization threshold.

        Returns:
            Current threshold (may be adapted from Hopfield energy if S5 enabled).
        """
        return self.crystallize_threshold


class ContinuousThoughtWrapper(layers.Layer):
    """Wrapper to integrate ContinuousThought with existing reasoning blocks.

    Provides a convenient interface for adding continuous thought capability
    to existing LatentReasoningBlock or other reasoning modules.

    Example:
        >>> wrapper = ContinuousThoughtWrapper(embedding_dim=512)
        >>> # Use with existing latent reasoning block
        >>> output = wrapper(hidden, latent_block_output)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_thought_steps: int = CONTINUOUS_THOUGHT_STEPS,
        integration_mode: str = "parallel",
        name: str = "continuous_thought_wrapper",
        **kwargs: Any,
    ) -> None:
        """Initialize ContinuousThoughtWrapper.

        Args:
            embedding_dim: Embedding dimension.
            num_thought_steps: Number of thought steps.
            integration_mode: "parallel" (add to existing) or "sequential" (after).
            name: Layer name.
            **kwargs: Additional Keras layer arguments.
        """
        super().__init__(name=name, **kwargs)

        self.embedding_dim = embedding_dim
        self.integration_mode = integration_mode

        self.thought_block = ContinuousThoughtBlock(
            embedding_dim=embedding_dim,
            num_thought_steps=num_thought_steps,
            name=f"{name}_block",
        )

        if integration_mode == "parallel":
            # Gate to blend continuous thought with other reasoning
            self.blend_gate = layers.Dense(
                embedding_dim,
                activation="sigmoid",
                name=f"{name}_blend_gate",
            )

    def call(
        self,
        hidden_states: tf.Tensor,
        reasoning_output: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply continuous thought with integration.

        Args:
            hidden_states: Original hidden states [batch, seq_len, dim].
            reasoning_output: Optional output from other reasoning (for parallel).
            training: Whether in training mode.

        Returns:
            Integrated output [batch, seq_len, dim].
        """
        # Apply continuous thought
        thought_output = self.thought_block(hidden_states, training=training)

        if self.integration_mode == "parallel" and reasoning_output is not None:
            # Blend continuous thought with other reasoning output
            blend = self.blend_gate(hidden_states)
            output = blend * thought_output + (1 - blend) * reasoning_output
        else:
            output = thought_output

        return output

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "integration_mode": self.integration_mode,
            }
        )
        return config


def create_continuous_thought_block(
    embedding_dim: int,
    **kwargs: Any,
) -> ContinuousThoughtBlock | layers.Layer:
    """Factory function for creating continuous thought block.

    Returns identity layer if USE_CONTINUOUS_THOUGHT is False.

    Args:
        embedding_dim: Embedding dimension.
        **kwargs: Additional arguments for ContinuousThoughtBlock.

    Returns:
        ContinuousThoughtBlock if enabled, identity layer otherwise.
    """
    if not USE_CONTINUOUS_THOUGHT:
        logger.debug("Continuous thought disabled (USE_CONTINUOUS_THOUGHT=False)")
        return layers.Lambda(lambda x: x, name="continuous_thought_identity")

    return ContinuousThoughtBlock(embedding_dim=embedding_dim, **kwargs)


__all__ = [
    "ThoughtProjector",
    "ContinuousThoughtBlock",
    "ContinuousThoughtWrapper",
    "create_continuous_thought_block",
]
