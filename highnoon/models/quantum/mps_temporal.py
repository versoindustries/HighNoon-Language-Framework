# highnoon/models/quantum/mps_temporal.py
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

"""MPS Temporal Integration Module.

This module provides a Keras-compatible adapter for using Matrix Product States
(MPS) for temporal sequence modeling in the language model context.

Key Features:
    - Tensor network-based temporal modeling
    - O(n·χ²) complexity where χ = bond dimension
    - CPU-native SIMD-friendly tensor contractions via C++ kernel
    - Hybrid mode: GRU for short sequences, true MPS for long sequences
    - Entanglement entropy output for analysis

CONSTRAINT: All computations use float32/64 precision only. No quantization.

Research References:
    - MPSTime Algorithm (2024) - Time-series ML with MPS
    - Uniform MPS for Probabilistic Modeling (2024)
"""

from __future__ import annotations

import logging
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.config import (
    MPS_BOND_DIM,
    MPS_COMPUTE_ENTROPY,
    MPS_HYBRID_THRESHOLD,
    MPS_TRUNCATION_THRESHOLD,
    MPS_UNIFORM_MODE,
    MPS_USE_TDVP_GRADIENTS,
    MPS_USE_TENSORIZED_ATTENTION,
    MPS_USE_TRUE_CONTRACTION,
    USE_MPS_TEMPORAL,
)

logger = logging.getLogger(__name__)

# Lazy imports for C++ MPS operations
_mps_contract_module = None
_mps_temporal_module = None


def _get_mps_contract():
    """Lazy load MPS contract module."""
    global _mps_contract_module
    if _mps_contract_module is None:
        try:
            from highnoon._native.ops.mps_contract import mps_contract

            _mps_contract_module = mps_contract
            logger.debug("Loaded C++ MPS contraction kernel")
        except ImportError as e:
            logger.warning(f"C++ MPS contraction kernel not available: {e}")
            _mps_contract_module = False
    return _mps_contract_module if _mps_contract_module else None


def _get_mps_temporal():
    """Lazy load MPS temporal module."""
    global _mps_temporal_module
    if _mps_temporal_module is None:
        try:
            # This would be the Python wrapper for MPSTemporalScan
            # For now, we assume it's in highnoon._native.ops.mps_temporal
            from highnoon._native.ops.mps_temporal import mps_temporal_scan

            _mps_temporal_module = mps_temporal_scan
            logger.debug("Loaded C++ MPS temporal scan kernel")
        except ImportError as e:
            logger.debug(f"C++ MPS temporal scan kernel not available: {e}")
            _mps_temporal_module = False
    return _mps_temporal_module if _mps_temporal_module else None


class MPSTemporalBlock(layers.Layer):
    """MPS-based temporal block for sequence modeling.

    Uses Matrix Product State tensor networks for capturing temporal
    dependencies with efficient O(n·χ²) complexity.

    Supports two modes:
        - **True MPS Mode** (default): Uses C++ SIMD-optimized kernel for
          actual tensor network contraction with SVD truncation.
        - **GRU Fallback Mode**: Uses GRU approximation for environments
          where C++ kernel is unavailable or for short sequences.

    The hybrid mode automatically selects GRU for short sequences (L < threshold)
    and true MPS for longer sequences where tensor network benefits apply.

    Attributes:
        embedding_dim: Input/output embedding dimension.
        bond_dim: MPS bond dimension (controls expressiveness vs cost).
        spatial_bond_dim: Internal spatial bond dimension.
        use_true_mps: Whether to use true MPS contraction (vs GRU).
        hybrid_threshold: Sequence length threshold for hybrid mode.

    Complexity: O(n·χ²) where n = sequence length, χ = bond_dim

    Example:
        >>> block = MPSTemporalBlock(embedding_dim=512, bond_dim=32)
        >>> x = tf.random.normal([2, 128, 512])  # [B, L, D]
        >>> output = block(x)
        >>> # output: [2, 128, 512]
    """

    def __init__(
        self,
        embedding_dim: int,
        bond_dim: int = MPS_BOND_DIM,
        spatial_bond_dim: int = 4,
        use_residual: bool = True,
        use_true_mps: bool = MPS_USE_TRUE_CONTRACTION,
        hybrid_threshold: int = MPS_HYBRID_THRESHOLD,
        compute_entropy: bool = MPS_COMPUTE_ENTROPY,
        truncation_threshold: float = MPS_TRUNCATION_THRESHOLD,
        uniform_mode: bool = MPS_UNIFORM_MODE,
        use_tdvp: bool = MPS_USE_TDVP_GRADIENTS,
        name: str = "mps_temporal",
        **kwargs: Any,
    ) -> None:
        """Initialize MPSTemporalBlock.

        Args:
            embedding_dim: Input/output embedding dimension.
            bond_dim: Temporal bond dimension for MPS.
            spatial_bond_dim: Spatial bond dimension.
            use_residual: Whether to use residual connection.
            use_true_mps: Use true MPS contraction via C++ kernel.
            hybrid_threshold: Sequence length threshold for hybrid mode.
                GRU is used for L < threshold, MPS for L >= threshold.
            compute_entropy: Compute entanglement entropy at each bond.
            truncation_threshold: SVD truncation threshold.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.

        Raises:
            ValueError: If embedding_dim is not positive.
            ValueError: If bond_dim is not positive.
        """
        super().__init__(name=name, **kwargs)

        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if bond_dim < 1:
            raise ValueError(f"bond_dim must be positive, got {bond_dim}")

        self.embedding_dim = embedding_dim
        self.bond_dim = bond_dim
        self.spatial_bond_dim = spatial_bond_dim
        self.use_residual = use_residual
        self.use_true_mps = use_true_mps
        self.hybrid_threshold = hybrid_threshold
        self.compute_entropy = compute_entropy
        self.truncation_threshold = truncation_threshold
        self.uniform_mode = uniform_mode
        self.use_tdvp = use_tdvp

        # Check if C++ kernels are available
        self._mps_contract_available = _get_mps_contract() is not None
        self._mps_temporal_available = _get_mps_temporal() is not None

        if self.use_true_mps and not self._mps_temporal_available:
            raise RuntimeError(
                "True MPS scan requested but C++ temporal kernel not available. "
                "Strict C++ compliance mode is enabled (no Python fallback allowed)."
            )

        # Shared input projection to MPS-friendly dimension
        self.input_projection = layers.Dense(
            bond_dim * spatial_bond_dim,
            name=f"{name}_input_proj",
        )

        # MPS core tensor weights (trainable for true MPS mode)
        # Physical dimension is spatial_bond_dim, bond dims are bond_dim
        # Shape: [bond_left, physical_dim, bond_right] for each site
        # For efficiency, we use a learned projection that creates site tensors
        self.site_tensor_proj = layers.Dense(
            bond_dim * spatial_bond_dim * bond_dim,
            name=f"{name}_site_tensor",
        )

        # GRU fallback layers removed for strict C++ compliance

        # Output bridge for true MPS mode (maps phys_dim to bond_dim)
        self.mps_bridge = layers.Dense(
            bond_dim,
            name=f"{name}_mps_bridge",
        )

        # Output projection back to embedding dim
        self.output_projection = layers.Dense(
            embedding_dim,
            name=f"{name}_output_proj",
        )

        # Layer normalization
        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{name}_norm",
        )

        # Gate for residual (if using)
        if use_residual:
            self.residual_gate = layers.Dense(
                embedding_dim,
                activation="sigmoid",
                name=f"{name}_gate",
            )

        # Storage for entanglement entropy (for analysis)
        self._last_entanglement_entropy = None

    def build_mps_cores(
        self,
        sequence: tf.Tensor,
    ) -> tuple[list[tf.Tensor], tf.Tensor, tf.Tensor]:
        """Build MPS core tensors from sequence embeddings.

        Constructs site tensors A[t] for each position in the sequence,
        suitable for contraction via the C++ MPS kernel.

        Args:
            sequence: Input sequence [batch, seq_len, embedding_dim].

        Returns:
            Tuple of:
                - mps_tensors: List of site tensors, each [bond_left, phys, bond_right]
                - physical_dims: Physical dimension at each site [seq_len]
                - bond_dims: Bond dimensions [seq_len + 1]
        """
        batch_size = tf.shape(sequence)[0]
        seq_len = tf.shape(sequence)[1]

        # Project to MPS dimension
        projected = self.input_projection(sequence)  # [B, L, bond*spatial]

        # Generate site tensors via learned projection
        site_tensors_flat = self.site_tensor_proj(projected)  # [B, L, bond*phys*bond]

        # Reshape to [B, L, bond_left, phys, bond_right]
        site_tensors = tf.reshape(
            site_tensors_flat,
            [batch_size, seq_len, self.bond_dim, self.spatial_bond_dim, self.bond_dim],
        )

        # Create list of tensors for each sequence position
        # For C++ kernel, we need boundary conditions: bond_dims[0] = bond_dims[-1] = 1
        # We'll process per-batch and contract within batch dimension
        mps_tensors = []

        # Left boundary tensor: [1, phys, bond_dim]
        left_tensor = tf.reshape(
            site_tensors[:, 0, 0:1, :, :],  # Take first bond slice
            [batch_size, 1, self.spatial_bond_dim, self.bond_dim],
        )
        mps_tensors.append(left_tensor)

        # Interior tensors: [bond_dim, phys, bond_dim]
        for t in range(1, seq_len - 1):
            mps_tensors.append(site_tensors[:, t, :, :, :])

        # Right boundary tensor: [bond_dim, phys, 1]
        if seq_len > 1:
            right_tensor = tf.reshape(
                site_tensors[:, -1, :, :, 0:1],  # Take first bond slice
                [batch_size, self.bond_dim, self.spatial_bond_dim, 1],
            )
            mps_tensors.append(right_tensor)

        # Physical dims: all sites have same physical dimension
        physical_dims = tf.fill([seq_len], self.spatial_bond_dim)

        # Bond dims: [1, bond_dim, ..., bond_dim, 1]
        bond_dims = tf.concat(
            [
                tf.constant([1], dtype=tf.int32),
                tf.fill([seq_len - 1], self.bond_dim),
                tf.constant([1], dtype=tf.int32),
            ],
            axis=0,
        )

        return mps_tensors, physical_dims, bond_dims

    def _call_true_mps(
        self,
        inputs: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass using optimized C++ MPSTemporalScan kernel."""
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Project input to MPS-friendly representation
        site_weights_flat = self.site_tensor_proj(inputs)  # [B, L, chi*phys*chi]
        site_weights = tf.reshape(
            site_weights_flat,
            [batch_size, seq_len, self.bond_dim, self.spatial_bond_dim, self.bond_dim],
        )

        initial_state = tf.zeros([batch_size, 1, self.bond_dim])

        # Call the C++ kernel with wired config flags
        mps_temporal_scan = _get_mps_temporal()
        outputs, log_probs, entropy = mps_temporal_scan(
            inputs,
            site_weights,
            initial_state,
            tf.constant(self.bond_dim, dtype=tf.int32),
            use_tdvp=self.use_tdvp,
            compute_entropy=self.compute_entropy,  # Wire MPS_COMPUTE_ENTROPY
            uniform_mode=self.uniform_mode,  # Wire MPS_UNIFORM_MODE
        )

        # Store log_probs for probability modeling if needed
        self._last_log_probs = log_probs

        # Store entanglement entropy if computed (wire MPS_COMPUTE_ENTROPY)
        if self.compute_entropy and entropy is not None:
            self._last_entanglement_entropy = entropy

        # Bridge physical output to bond dimension to match GRU path
        outputs = self.mps_bridge(outputs)

        return self.output_projection(outputs)

    def compute_log_prob(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compute sequence log-probability using MPS norm."""
        if not self.use_true_mps:
            return tf.zeros([tf.shape(inputs)[0], tf.shape(inputs)[1]])

        # Ensure forward pass has been run or run it now
        _ = self(inputs, training=False)
        return self._last_log_probs

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass through MPS temporal block.

        Uses hybrid mode: true MPS for long sequences, GRU for short ones.

        Args:
            inputs: Input tensor [batch, seq_len, embedding_dim].
            training: Whether in training mode.

        Returns:
            Output tensor [batch, seq_len, embedding_dim].
        """
        # Cast to float32 for precision
        inputs = tf.cast(inputs, tf.float32)

        tf.shape(inputs)[1]

        # Hybrid mode: choose based on sequence length
        # Hybrid mode: choose based on sequence length
        if self.use_true_mps:
            # Strictly use C++ MPS kernel
            x = self._call_true_mps(inputs, training)
        else:
            raise RuntimeError("MPSTemporalBlock requires use_true_mps=True in strict mode.")

        # Apply normalization
        x = self.layer_norm(x)

        # Residual connection if enabled
        if self.use_residual:
            gate = self.residual_gate(inputs)
            x = gate * x + (1 - gate) * inputs

        return x

    @property
    def last_entanglement_entropy(self) -> tf.Tensor | None:
        """Get entanglement entropy from last forward pass.

        Returns:
            Entanglement entropy tensor [num_bonds] or None if not computed.
        """
        return self._last_entanglement_entropy

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "bond_dim": self.bond_dim,
                "spatial_bond_dim": self.spatial_bond_dim,
                "use_residual": self.use_residual,
                "use_true_mps": self.use_true_mps,
                "hybrid_threshold": self.hybrid_threshold,
                "compute_entropy": self.compute_entropy,
                "truncation_threshold": self.truncation_threshold,
            }
        )
        return config


class MPSAttentionBlock(layers.Layer):
    """MPS-enhanced attention block with tensor network structure.

    Combines standard attention with MPS-based temporal modeling
    for improved long-range dependency capture.

    Example:
        >>> block = MPSAttentionBlock(embedding_dim=512)
        >>> output = block(hidden_states)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        bond_dim: int = MPS_BOND_DIM,
        dropout_rate: float = 0.1,
        use_true_mps: bool = MPS_USE_TRUE_CONTRACTION,
        use_tensorized_attention: bool = MPS_USE_TENSORIZED_ATTENTION,
        name: str = "mps_attention",
        **kwargs: Any,
    ) -> None:
        """Initialize MPSAttentionBlock.

        Args:
            embedding_dim: Embedding dimension.
            num_heads: Number of attention heads.
            bond_dim: MPS bond dimension.
            dropout_rate: Dropout rate.
            use_true_mps: Use true MPS contraction.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.
        """
        super().__init__(name=name, **kwargs)

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.bond_dim = bond_dim
        self.use_true_mps = use_true_mps
        self.use_tensorized_attention = use_tensorized_attention

        # MPS temporal layer
        self.mps_temporal = MPSTemporalBlock(
            embedding_dim=embedding_dim,
            bond_dim=bond_dim,
            use_residual=False,
            use_true_mps=use_true_mps,
            name=f"{name}_mps",
        )

        # Attention projection
        self.query = layers.Dense(embedding_dim, name=f"{name}_query")
        self.key = layers.Dense(embedding_dim, name=f"{name}_key")
        self.value = layers.Dense(embedding_dim, name=f"{name}_value")

        # Output projection
        self.output_proj = layers.Dense(embedding_dim, name=f"{name}_output")

        # Normalization and dropout
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")
        self.dropout = layers.Dropout(dropout_rate)

        # Blend gate between attention and MPS
        self.blend_gate = layers.Dense(
            embedding_dim,
            activation="sigmoid",
            name=f"{name}_blend",
        )

    def _tensorized_attention(self, q, k, v, training=False):
        """Implement O(n√n) attention by 2D sequence reshaping."""
        batch_size = tf.shape(q)[0]
        seq_len = tf.shape(q)[1]
        head_dim = self.embedding_dim // self.num_heads

        # Determine grid size for tensorization
        h = tf.cast(tf.math.sqrt(tf.cast(seq_len, tf.float32)), tf.int32)
        w = seq_len // h

        # Trim to exact grid size if necessary
        actual_len = h * w
        q = q[:, :actual_len, :]
        k = k[:, :actual_len, :]
        v = v[:, :actual_len, :]

        # Reshape to [B, h, w, D] and then to [B*h, w, D] for row attention
        q_row = tf.reshape(q, [batch_size * h, w, self.embedding_dim])
        k_row = tf.reshape(k, [batch_size * h, w, self.embedding_dim])
        v_row = tf.reshape(v, [batch_size * h, w, self.embedding_dim])

        # Row-wise attention: [B*h, w, w]
        row_attn = tf.nn.softmax(
            tf.matmul(q_row, k_row, transpose_b=True) / tf.math.sqrt(tf.cast(head_dim, tf.float32)),
            axis=-1,
        )
        row_out = tf.matmul(row_attn, v_row)

        # Reshape to [B, h, w, D] -> [B, w, h, D] -> [B*w, h, D] for column attention
        row_out_2d = tf.reshape(row_out, [batch_size, h, w, self.embedding_dim])
        q_col = tf.transpose(row_out_2d, [0, 2, 1, 3])
        q_col = tf.reshape(q_col, [batch_size * w, h, self.embedding_dim])

        # Use MPS keys/values for columns to inject global context
        mps_context = self.mps_temporal(q, training=training)
        mps_2d = tf.reshape(mps_context, [batch_size, h, w, self.embedding_dim])
        k_col = tf.transpose(mps_2d, [0, 2, 1, 3])
        k_col = tf.reshape(k_col, [batch_size * w, h, self.embedding_dim])

        col_attn = tf.nn.softmax(
            tf.matmul(q_col, k_col, transpose_b=True) / tf.math.sqrt(tf.cast(head_dim, tf.float32)),
            axis=-1,
        )
        col_out = tf.matmul(col_attn, k_col)

        # Reshape back to [B, L, D]
        col_out_2d = tf.reshape(col_out, [batch_size, w, h, self.embedding_dim])
        output = tf.transpose(col_out_2d, [0, 2, 1, 3])
        output = tf.reshape(output, [batch_size, actual_len, self.embedding_dim])

        # Pad back to original length if trimmed
        if actual_len < seq_len:
            padding = tf.zeros([batch_size, seq_len - actual_len, self.embedding_dim])
            output = tf.concat([output, padding], axis=1)

        return output

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass through MPS attention block."""
        # MPS temporal processing for enhanced state representation
        mps_out = self.mps_temporal(inputs, training=training)

        # Normalization and Projections
        norm_inputs = self.norm1(inputs)
        q = self.query(norm_inputs)
        k = self.key(mps_out)
        v = self.value(mps_out)

        if self.use_tensorized_attention:
            attn_out = self._tensorized_attention(q, k, v, training=training)
        else:
            # Import enforcement flag
            from highnoon.config import ENFORCE_LINEAR_ATTENTION

            if ENFORCE_LINEAR_ATTENTION:
                raise RuntimeError(
                    "Quadratic O(n²) attention is disabled when ENFORCE_LINEAR_ATTENTION=True. "
                    "Set MPS_USE_TENSORIZED_ATTENTION=True or disable enforcement."
                )
            # Legacy quadratic attention (only if enforcement disabled)
            head_dim = self.embedding_dim // self.num_heads
            scale = tf.math.sqrt(tf.cast(head_dim, tf.float32))
            attn_weights = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / scale, axis=-1)
            attn_out = tf.matmul(attn_weights, v)

        # Project and dropout
        attn_out = self.output_proj(attn_out)
        attn_out = self.dropout(attn_out, training=training)

        # Adaptive blend gate between attention and MPS path
        blend = self.blend_gate(inputs)
        combined = blend * attn_out + (1 - blend) * mps_out

        # Final normalization and residual
        output = inputs + self.norm2(combined)

        return output

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "bond_dim": self.bond_dim,
                "use_true_mps": self.use_true_mps,
            }
        )
        return config


def create_mps_temporal_block(
    embedding_dim: int,
    **kwargs: Any,
) -> MPSTemporalBlock | layers.Layer:
    """Factory function for creating MPS temporal block.

    Returns identity layer if USE_MPS_TEMPORAL is False.

    Args:
        embedding_dim: Embedding dimension.
        **kwargs: Additional arguments for MPSTemporalBlock.

    Returns:
        MPSTemporalBlock if enabled, identity layer otherwise.
    """
    if not USE_MPS_TEMPORAL:
        logger.debug("MPS temporal disabled (USE_MPS_TEMPORAL=False)")
        return layers.Lambda(lambda x: x, name="mps_temporal_identity")

    return MPSTemporalBlock(embedding_dim=embedding_dim, **kwargs)


__all__ = [
    "MPSTemporalBlock",
    "MPSAttentionBlock",
    "create_mps_temporal_block",
]
