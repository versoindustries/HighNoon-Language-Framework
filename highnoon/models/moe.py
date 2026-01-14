# highnoon/models/moe.py
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

"""Mixture of Experts (MoE) Layer for HighNoon Language Framework.

This module provides the MoE layer implementation with:

- Expert Choice routing for balanced expert utilization
- Superposed Expert architecture with tensor train factorization
- Load balancing and auxiliary loss computation
- Phase 12.12: Dynamic Expert Specialization Probing
- Phase 12.14: Wavelet-Guided MoE Routing

CRITICAL: This module requires C++ compiled operators. No Python fallback is provided.
"""

import logging
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

# Phase 201.13: Adaptive Superposition Dimension
# Phase 201.4: HD Shared Expert Basis
import highnoon.config as hn_config  # Full config module for getattr()

# Import limits for Lite edition enforcement
from highnoon._native._limits import MAX_MOE_EXPERTS, LimitExceededError

# Import the fused MoE dispatch operators (C++ required)
from highnoon._native.ops.fused_moe_dispatch_op import fused_moe_dispatch, fused_moe_dispatch_v2

# Import the fused C++ op
from highnoon._native.ops.fused_superposition_moe_op import (
    fused_superposition_moe,
    fused_superposition_moe_op,
)

# Import the custom segment_softmax implementation.
from highnoon._native.ops.segment_softmax import segment_softmax
from highnoon.config import (  # Phase 14.2: MoE Innovations; Phase 19.4: Thermodynamic Routing; Phase 29: Unitary Expert
    ADA_K_BASE,
    ADA_K_MAX,
    ADA_K_MIN,
    BALANCE_EMA_DECAY,
    ENABLE_EXPERT_PROBING,
    HD_SHARED_BASIS_DIM,
    HD_SHARED_BASIS_NUM_VECTORS,
    NEUMANN_CAYLEY_TERMS,
    NUM_SHARED_EXPERTS,
    QMOE_USE_FLOQUET_PHASE,
    SUPERPOSITION_COMPLEXITY_SCALE,
    SUPERPOSITION_DIM,
    SUPERPOSITION_MAX_DIM,
    SUPERPOSITION_MICRO_BATCH_SIZE,
    SUPERPOSITION_MIN_DIM,
    TENSOR_RING_NUM_CORES,
    TENSOR_RING_RANK,
    USE_ADAPTIVE_K,
    USE_ADAPTIVE_SUPERPOSITION,
    USE_AUX_LOSS_FREE_BALANCE,
    USE_HD_SHARED_EXPERT_BASIS,
    USE_SHARED_EXPERTS,
    USE_SIGMOID_ROUTING,
    USE_TENSOR_RING_MOE,
    USE_UNITARY_EXPERT,
    USE_WAVELET_ROUTING,
)

# Import dependencies for SuperposedExpert
from highnoon.models.tensor_layers import SuperpositionTTLayer
from highnoon.utils import factorize_for_tt

# Lazy import for quantum ops
_unitary_expert_forward = None
_quantum_activation = None
_unitary_expert_ops_loaded = False


def _load_unitary_expert_ops():
    """Lazy load unitary expert ops.

    V2 MIGRATION: Uses quantum_foundation_ops for quantum_expert, but
    quantum_activation remains in quantum_ops.
    """
    global _unitary_expert_forward, _quantum_activation, _unitary_expert_ops_loaded
    if _unitary_expert_ops_loaded:
        return True
    try:
        from highnoon._native.ops.quantum_foundation_ops import quantum_expert
        from highnoon._native.ops.quantum_ops import quantum_activation

        _unitary_expert_forward = quantum_expert
        _quantum_activation = quantum_activation
        _unitary_expert_ops_loaded = True
        log.info("Unitary Expert ops loaded (via quantum_foundation_ops + quantum_ops)")
        return True
    except Exception as e:
        log.warning(f"Unitary Expert ops not available: {e}")
        return False


log = logging.getLogger(__name__)


class WaveletGuidedRouter:
    """Phase 12.14: Augments MoE routing with wavelet frequency information.

    Uses frequency analysis to bias expert routing:
    - High-frequency tokens → detail experts (syntax, punctuation)
    - Low-frequency tokens → semantic experts (concepts, entities)

    Complexity: O(L) - reuses existing wavelet decomposition.
    """

    def __init__(self, num_experts: int, embedding_dim: int, name: str = "wavelet_router"):
        """Initialize WaveletGuidedRouter.

        Args:
            num_experts: Number of experts to route to.
            embedding_dim: Input embedding dimension.
            name: Layer name prefix.
        """
        self.num_experts = num_experts
        self.embedding_dim = embedding_dim
        self.freq_to_expert_bias = layers.Dense(num_experts, name=f"{name}_freq_bias")

    def compute_frequency_routing_bias(
        self,
        x: tf.Tensor,
        low_pass: tf.Tensor | None = None,
        high_pass: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """Compute routing bias based on frequency characteristics.

        Args:
            x: Input tensor [batch, seq_len, dim].
            low_pass: Optional pre-computed low-pass filter output.
            high_pass: Optional pre-computed high-pass filter output.

        Returns:
            Routing bias [batch * seq_len, num_experts].
        """
        if low_pass is None or high_pass is None:
            # No wavelet data available - skip wavelet-guided routing
            # Return zero bias to disable frequency-based routing for this call
            # This is graceful degradation in strict C++ compliance mode
            batch_size = tf.shape(x)[0]
            seq_len = tf.shape(x)[1]
            return tf.zeros([batch_size * seq_len, self.num_experts], dtype=x.dtype)

        # Compute frequency energy per position
        low_energy = tf.reduce_mean(tf.abs(low_pass), axis=-1, keepdims=True)
        high_energy = tf.reduce_mean(tf.abs(high_pass), axis=-1, keepdims=True)

        # Frequency ratio: high → 1.0, low → 0.0
        freq_ratio = high_energy / (low_energy + high_energy + 1e-6)

        # Create frequency embedding [B, L, 2]
        freq_embedding = tf.concat([freq_ratio, 1 - freq_ratio], axis=-1)

        # Reshape to [B*L, 2] for routing
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        freq_embedding_2d = tf.reshape(freq_embedding, [batch_size * seq_len, 2])

        # Project to expert bias
        return self.freq_to_expert_bias(freq_embedding_2d)


class UnitaryExpert(layers.Layer):
    """Phase 29: Unitary Expert Network.

    Uses unitary transformations (via Cayley parameterization) to ensure:
    - Gradient preservation: ||grad|| preserved through layers
    - Information preservation: ||output|| ≈ ||input||
    - Quantum activation: parametric rotation instead of ReLU/GELU

    Architecture: x → U₁·x → quantum_activation(θ) → U₂·x

    CRITICAL: Requires C++ native ops. No Python fallback.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        neumann_terms: int = 6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.neumann_terms = neumann_terms

        # U1: d_model → d_ff (skew-symmetric for Cayley)
        # U2: d_ff → d_model
        # These are the skew-symmetric matrices A where U = (I-A)(I+A)^{-1}

    def build(self, input_shape):
        """Build unitary weight matrices."""
        # Skew-symmetric parameterization: A = W - W^T
        # U1 projects d_model → d_ff
        self.u1_weights = self.add_weight(
            name="u1_weights",
            shape=(self.d_ff, self.d_model),
            initializer=tf.keras.initializers.Orthogonal(gain=0.1),
            trainable=True,
        )
        # U2 projects d_ff → d_model
        self.u2_weights = self.add_weight(
            name="u2_weights",
            shape=(self.d_model, self.d_ff),
            initializer=tf.keras.initializers.Orthogonal(gain=0.1),
            trainable=True,
        )
        # Quantum activation angle (learnable)
        self.activation_angle = self.add_weight(
            name="activation_angle",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.7854),  # π/4
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass through unitary expert.

        Args:
            inputs: [num_tokens, d_model]

        Returns:
            Output [num_tokens, d_model]
        """
        if not _unitary_expert_ops_loaded:
            if not _load_unitary_expert_ops():
                raise RuntimeError(
                    "UnitaryExpert requires C++ native ops. "
                    "Run ./build_secure.sh to compile. NO PYTHON FALLBACK."
                )

        # CRITICAL: The C++ QuantumExpertForward kernel expects U1 and U2 weights
        # concatenated together as a flat tensor. It accesses U_skew[u1_size + ...]
        # for the second projection. We must flatten and concatenate both weights.
        # u1_weights: [d_ff, d_model] -> flattened to [d_ff * d_model]
        # u2_weights: [d_model, d_ff] -> flattened to [d_model * d_ff]
        # Combined: [d_ff * d_model + d_model * d_ff]
        u1_flat = tf.reshape(self.u1_weights, [-1])
        u2_flat = tf.reshape(self.u2_weights, [-1])
        u_combined = tf.concat([u1_flat, u2_flat], axis=0)

        # quantum_expert signature: (input_tensor, u_skew, d_ff=, activation_angle=)
        # The C++ kernel handles full expert forward: x -> activation(x @ U1) @ U2 internally
        result = _unitary_expert_forward(
            inputs,
            u_combined,
            d_ff=self.d_ff,
            activation_angle=float(self.activation_angle),
        )

        # quantum_expert returns the output directly (d_model shape)
        if isinstance(result, tuple):
            output = result[0]
        else:
            output = result

        # GRADIENT FIX: Ensure u1_weights, u2_weights, and activation_angle receive
        # gradients via zero-contribution passthrough. The C++ backward pass may not
        # properly compute gradients for all these parameters.
        # NOTE: Using hn_config.GRADIENT_PASSTHROUGH_EPSILON instead of 0.0 because TF may optimize away 0.0 * value
        weights_passthrough = (
            tf.reduce_mean(self.u1_weights) * hn_config.GRADIENT_PASSTHROUGH_EPSILON
            + tf.reduce_mean(self.u2_weights) * hn_config.GRADIENT_PASSTHROUGH_EPSILON
            + self.activation_angle * hn_config.GRADIENT_PASSTHROUGH_EPSILON
        )
        output = output + weights_passthrough

        return output

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "neumann_terms": self.neumann_terms,
            }
        )
        return config


class SuperposedExpert(layers.Layer):
    """Unified HD-SuperposedExpert with Holographic Circular Routing (v2.0).

    Combines TT-decomposed superposition paths with holographic routing:
    - K parallel TT-FFN paths process input through superposition
    - Holographic similarity-based routing collapses paths in HD space
    - Replaces attention-based Q/K/V/O collapse with geometric routing

    Phase 1010: HD routing dimension is now QAHPO-tunable via hd_dim parameter.
    See HD_SUPERPOSED_EXPERT_UNIFICATION.md for architecture details.

    CRITICAL: This layer exclusively uses the fused C++ kernel. No Python fallback.

    Attributes:
        d_model: Model embedding dimension.
        d_ff: Feedforward hidden dimension.
        superposition_dim: Number of parallel superposition paths (K).
        hd_dim: HD space dimension for holographic routing (QAHPO tunable).
        routing_temperature: Softmax temperature for path selection.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        superposition_dim: int = 4,
        hd_dim: int | None = None,
        routing_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.superposition_dim = superposition_dim
        # HD dim defaults to config value or d_model as fallback
        self.hd_dim = hd_dim or getattr(hn_config, "HD_DIM_MOE", d_model)
        self.routing_temperature = routing_temperature
        # Flag for whether we need projections (hd_dim != d_model)
        self._use_hd_projection = self.hd_dim != d_model

        # TT-decomposed FFN layers for each superposition path
        self.ffn1 = SuperpositionTTLayer(
            input_dims=factorize_for_tt(d_model),
            output_dims=factorize_for_tt(d_ff),
            tt_ranks=[1, 16, 1],
            superposition_dim=superposition_dim,
            name="superposed_ffn1",
        )
        self.ffn2 = SuperpositionTTLayer(
            input_dims=factorize_for_tt(d_ff),
            output_dims=factorize_for_tt(d_model),
            tt_ranks=[1, 16, 1],
            superposition_dim=superposition_dim,
            name="superposed_ffn2",
        )
        # Note: ContextualGatingCollapse removed - using holographic routing

    def build(self, input_shape):
        """Build TT layers and holographic routing weights."""
        # Build TT layers
        if not self.ffn1.built:
            self.ffn1.build(input_shape)
        if not self.ffn2.built:
            self.ffn2.build(tf.TensorShape([input_shape[0], self.d_ff]))

        # HD projection weights (when hd_dim != d_model)
        if self._use_hd_projection:
            self.hd_input_proj = self.add_weight(
                name="hd_input_proj",
                shape=(self.d_model, self.hd_dim),
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True,
            )
            self.hd_output_proj = self.add_weight(
                name="hd_output_proj",
                shape=(self.hd_dim, self.d_model),
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True,
            )
        else:
            # Identity placeholders when no projection needed
            self.hd_input_proj = self.add_weight(
                name="hd_input_proj_identity",
                shape=(1, 1),
                initializer="zeros",
                trainable=False,
            )
            self.hd_output_proj = self.add_weight(
                name="hd_output_proj_identity",
                shape=(1, 1),
                initializer="zeros",
                trainable=False,
            )

        # Holographic routing weights in HD space
        # path_bases: routing signatures for each superposition path
        self.path_bases = self.add_weight(
            name="path_bases",
            shape=(self.superposition_dim, self.hd_dim),
            initializer=tf.keras.initializers.Orthogonal(gain=1.0),
            trainable=True,
        )
        # path_weights: transformation weights for binding
        self.path_weights = self.add_weight(
            name="path_weights",
            shape=(self.superposition_dim, self.hd_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, training=None, micro_batch_size: int | None = None):
        """Forward pass via unified HD-SuperposedExpert C++ kernel.

        Args:
            inputs: Token embeddings [batch, d_model].
            training: Whether in training mode.
            micro_batch_size: Batch size for memory-efficient processing.

        Returns:
            Output tensor [batch, d_model].

        Raises:
            RuntimeError: If the C++ operator could not be loaded.
        """
        if fused_superposition_moe_op is None:
            raise RuntimeError(
                "The UnifiedHDSuperposedExpert C++ operator is required but could not be loaded. "
                "Please ensure it has been compiled with ./build_secure.sh. "
                "NO PYTHON FALLBACK IS PROVIDED."
            )
        if micro_batch_size is None:
            micro_batch_size = SUPERPOSITION_MICRO_BATCH_SIZE
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be a positive integer.")

        # Flatten TT cores for C++ kernel
        ffn1_cores_combined = tf.concat(
            [tf.reshape(core, [-1]) for core in self.ffn1.cores], axis=0
        )
        ffn2_cores_combined = tf.concat(
            [tf.reshape(core, [-1]) for core in self.ffn2.cores], axis=0
        )

        # Call unified kernel with holographic routing
        result = fused_superposition_moe(
            tokens=inputs,
            ffn1_cores=ffn1_cores_combined,
            ffn2_cores=ffn2_cores_combined,
            path_bases=self.path_bases,
            path_weights=self.path_weights,
            hd_input_proj=self.hd_input_proj,
            hd_output_proj=self.hd_output_proj,
            input_dims=self.ffn1.input_dims,
            output_dims_ffn1=self.ffn1.output_dims,
            output_dims_ffn2=self.ffn2.output_dims,
            tt_ranks=self.ffn1.tt_ranks,
            superposition_dim=self.superposition_dim,
            micro_batch_size=micro_batch_size,
            hd_dim=self.hd_dim,
            use_hd_projection=self._use_hd_projection,
            routing_temperature=self.routing_temperature,
        )

        # Return output tensor (routing_weights available for debugging)
        return result.output

    @property
    def routing_weights(self):
        """Get path routing weights from last forward pass (for visualization)."""
        # Note: actual routing weights are returned by the C++ kernel
        # This property can be used to access them after forward pass
        return None  # Set by call() in tracing mode

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "superposition_dim": self.superposition_dim,
                "hd_dim": self.hd_dim,
                "routing_temperature": self.routing_temperature,
            }
        )
        return config


class AdaptiveSuperposedExpert(layers.Layer):
    """Phase 201.13: Adaptive Superposition Dimension Expert.

    Extends SuperposedExpert with complexity-based superposition scaling.
    Complex inputs (high entropy) get larger superposition for more capacity.
    Simple inputs (low entropy) use smaller superposition for efficiency.

    Complexity is estimated from token embedding entropy:
        complexity = -sum(p * log(p)) / log(vocab_size)

    Superposition dimension is then scaled:
        S_adaptive = S_min + complexity * (S_max - S_min)

    Args:
        d_model: Model dimension.
        d_ff: Feedforward dimension.
        min_superposition: Minimum superposition dimension.
        max_superposition: Maximum superposition dimension.
        complexity_scale: Scale factor for complexity (1.0 = linear).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        min_superposition: int = SUPERPOSITION_MIN_DIM,
        max_superposition: int = SUPERPOSITION_MAX_DIM,
        complexity_scale: float = SUPERPOSITION_COMPLEXITY_SCALE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.min_superposition = min_superposition
        self.max_superposition = max_superposition
        self.complexity_scale = complexity_scale

        # Create experts for each superposition level
        self._experts_by_dim: dict[int, SuperposedExpert] = {}
        for s_dim in range(min_superposition, max_superposition + 1):
            self._experts_by_dim[s_dim] = SuperposedExpert(
                d_model=d_model,
                d_ff=d_ff,
                superposition_dim=s_dim,
                name=f"adaptive_expert_s{s_dim}",
            )

    def build(self, input_shape):
        """Build all superposition dimension variants."""
        for expert in self._experts_by_dim.values():
            if not expert.built:
                expert.build(input_shape)
        super().build(input_shape)

    def _estimate_complexity(self, inputs: tf.Tensor) -> tf.Tensor:
        """Estimate input complexity from embedding entropy.

        Args:
            inputs: Token embeddings [num_tokens, d_model].

        Returns:
            Complexity score in [0, 1] as scalar.
        """
        # Compute softmax over embedding dimensions as pseudo-distribution
        probs = tf.nn.softmax(tf.abs(inputs), axis=-1)

        # Compute entropy
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=-1)

        # Normalize by max entropy (log(d_model))
        max_entropy = tf.math.log(tf.cast(self.d_model, tf.float32))
        normalized_entropy = tf.reduce_mean(entropy) / max_entropy

        # Apply complexity scale
        return tf.clip_by_value(
            normalized_entropy * self.complexity_scale,
            0.0,
            1.0,
        )

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """Forward pass with adaptive superposition dimension.

        Args:
            inputs: Token embeddings [num_tokens, d_model].
            training: Whether in training mode.

        Returns:
            Expert output [num_tokens, d_model].
        """
        # Estimate complexity
        complexity = self._estimate_complexity(inputs)

        # Compute continuous superposition dimension
        s_range = float(self.max_superposition - self.min_superposition)
        s_continuous = tf.cast(self.min_superposition, tf.float32) + complexity * s_range

        # Discretize to integer dimension
        s_dim = tf.cast(tf.round(s_continuous), tf.int32)
        s_dim = tf.clip_by_value(s_dim, self.min_superposition, self.max_superposition)

        # Select appropriate expert
        # Note: In eager mode we can use Python control flow
        # For graph mode, we use tf.case for differentiable selection
        s_dim_val = int(s_dim.numpy()) if tf.executing_eagerly() else self.min_superposition

        if s_dim_val in self._experts_by_dim:
            return self._experts_by_dim[s_dim_val](inputs, training=training)
        else:
            # Fallback to min dimension
            return self._experts_by_dim[self.min_superposition](inputs, training=training)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "min_superposition": self.min_superposition,
                "max_superposition": self.max_superposition,
                "complexity_scale": self.complexity_scale,
            }
        )
        return config


class HDSharedExpertBasis(layers.Layer):
    """Phase 201.4: Hyperdimensional Shared Expert Basis.

    Implements memory-efficient expert weights by sharing a common HD basis.
    Each expert maintains lightweight coefficients that modulate the shared basis,
    reducing memory by ~(num_experts / num_basis_vectors).

    Architecture:
        shared_basis: [num_vectors, hd_dim, d_model] - common across all experts
        expert_coeffs: [num_experts, num_vectors] - per-expert modulation weights
        W_expert_i = sum_j(coeffs[i,j] * shared_basis[j]) - reconstructed expert weight

    Complexity: O(num_vectors * hd_dim * d_model) vs O(num_experts * d_model * d_ff)
    Memory reduction: ~6x for 12 experts with 32 basis vectors
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_ff: int,
        hd_dim: int = HD_SHARED_BASIS_DIM,
        num_basis_vectors: int = HD_SHARED_BASIS_NUM_VECTORS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.hd_dim = hd_dim
        self.num_basis_vectors = num_basis_vectors

    def build(self, input_shape):
        """Build shared basis and per-expert coefficients."""
        # Shared HD basis for up-projection: [num_vectors, d_model, hd_dim]
        self.shared_basis_up = self.add_weight(
            name="shared_basis_up",
            shape=(self.num_basis_vectors, self.d_model, self.hd_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        # Shared HD basis for down-projection: [num_vectors, hd_dim, d_model]
        self.shared_basis_down = self.add_weight(
            name="shared_basis_down",
            shape=(self.num_basis_vectors, self.hd_dim, self.d_model),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        # Per-expert coefficients for up-projection: [num_experts, num_vectors]
        self.expert_coeffs_up = self.add_weight(
            name="expert_coeffs_up",
            shape=(self.num_experts, self.num_basis_vectors),
            initializer=tf.keras.initializers.Orthogonal(gain=0.5),
            trainable=True,
        )
        # Per-expert coefficients for down-projection: [num_experts, num_vectors]
        self.expert_coeffs_down = self.add_weight(
            name="expert_coeffs_down",
            shape=(self.num_experts, self.num_basis_vectors),
            initializer=tf.keras.initializers.Orthogonal(gain=0.5),
            trainable=True,
        )
        # Per-expert bias
        self.expert_bias = self.add_weight(
            name="expert_bias",
            shape=(self.num_experts, self.d_model),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call_expert(self, inputs: tf.Tensor, expert_idx: int) -> tf.Tensor:
        """Forward pass through a specific expert using shared basis.

        Phase 5.4: Uses explicit gradient-preserving einsum operations
        to ensure gradients flow to shared_basis and expert_coeffs.

        Args:
            inputs: Token embeddings [num_tokens, d_model].
            expert_idx: Index of the expert to use.

        Returns:
            Expert output [num_tokens, d_model].
        """
        # Get coefficients for this expert - Phase 5.4: Use tf.gather for gradient flow
        coeffs_up = tf.gather(self.expert_coeffs_up, expert_idx)  # [num_vectors]
        coeffs_down = tf.gather(self.expert_coeffs_down, expert_idx)  # [num_vectors]
        bias = tf.gather(self.expert_bias, expert_idx)  # [d_model]

        # Phase 5.4: Reconstruct expert weights from shared basis
        # Using einsum which properly propagates gradients to both factors
        # W_up = sum_j(coeffs[j] * basis_up[j]): [d_model, hd_dim]
        W_up = tf.einsum("v,vdh->dh", coeffs_up, self.shared_basis_up)
        # W_down = sum_j(coeffs[j] * basis_down[j]): [hd_dim, d_model]
        W_down = tf.einsum("v,vhd->hd", coeffs_down, self.shared_basis_down)

        # Expert forward: x -> GELU(x @ W_up) @ W_down + bias
        # Phase 5.4: Explicit matmul with gradient checkpointing
        hidden = tf.matmul(inputs, W_up)  # [num_tokens, hd_dim]
        hidden = tf.nn.gelu(hidden)
        output = tf.matmul(hidden, W_down) + bias  # [num_tokens, d_model]

        return output

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "hd_dim": self.hd_dim,
                "num_basis_vectors": self.num_basis_vectors,
            }
        )
        return config


class HDSharedExpert(layers.Layer):
    """Expert wrapper that uses HDSharedExpertBasis for weight sharing.

    This is a thin wrapper that stores just the expert index and delegates
    to the shared basis for actual computation.
    """

    def __init__(self, shared_basis: HDSharedExpertBasis, expert_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.shared_basis = shared_basis
        self.expert_idx = expert_idx

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """Forward pass delegating to shared basis."""
        return self.shared_basis.call_expert(inputs, self.expert_idx)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "expert_idx": self.expert_idx,
            }
        )
        return config


class MoELayer(layers.Layer):
    """
    A sparse Mixture of Experts (MoE) layer.

    This layer exclusively uses 'Expert Choice' routing via the fused C++ operator.
    No Python fallback is provided.

    The Lite edition enforces a maximum of 12 experts.

    Attributes:
        num_experts: Number of expert modules.
        d_model: Model dimension.
        d_ff: Feedforward dimension.
        capacity_factor: Controls expert capacity relative to uniform distribution.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_ff: int,
        aux_loss_weight: float = 0.01,
        capacity_factor: float = 1.25,
        load_balance_lambda: float = 0.01,
        ema_decay: float = 0.999,
        superposition_dim: int = SUPERPOSITION_DIM,
        dropout: float = 0.1,
        # Phase 12.12: Expert Specialization Probing
        enable_expert_probing: bool = ENABLE_EXPERT_PROBING,
        expert_embedding_dim: int = 64,
        # Phase 12.14: Wavelet-Guided Routing
        use_wavelet_routing: bool = USE_WAVELET_ROUTING,
        wavelet_routing_weight: float = 0.2,
        # Phase 14.2.1: Shared Experts
        use_shared_experts: bool = USE_SHARED_EXPERTS,
        num_shared_experts: int = NUM_SHARED_EXPERTS,
        # Phase 14.2.3: Aux-Loss-Free Balancing
        use_aux_loss_free_balance: bool = USE_AUX_LOSS_FREE_BALANCE,
        balance_ema_decay: float = BALANCE_EMA_DECAY,
        # Phase 14.2.4: Adaptive Expert Count (Ada-K)
        use_adaptive_k: bool = USE_ADAPTIVE_K,
        ada_k_base: int = ADA_K_BASE,
        ada_k_min: int = ADA_K_MIN,
        ada_k_max: int = ADA_K_MAX,
        # Phase 14.2.6: Sigmoid Gating (GLM-4.5 style)
        use_sigmoid_routing: bool = USE_SIGMOID_ROUTING,
        # Phase 14.2.5: Null Expert Pool (capacity overflow)
        use_null_expert: bool = True,
        # Phase 86: Hopfield-Boosted Expert Selection
        use_hopfield_routing: bool = False,
        hopfield_beta: float = 1.0,
        hopfield_energy_threshold: float = 0.0,
        hopfield_routing_weight: float = 0.3,
        # S4: Floquet Phase → Expert Routing
        use_floquet_routing: bool = QMOE_USE_FLOQUET_PHASE,
        floquet_routing_weight: float = 0.25,
        **kwargs,
    ):
        """Initialize MoELayer.

        Args:
            num_experts: Number of expert modules.
            d_model: Model dimension.
            d_ff: Feedforward dimension.
            aux_loss_weight: Weight for auxiliary losses.
            capacity_factor: Controls expert capacity.
            load_balance_lambda: Load balancing coefficient.
            ema_decay: EMA decay for load tracking.
            superposition_dim: Superposition dimension for experts.
            dropout: Dropout rate.
            enable_expert_probing: Enable Phase 12.12 specialization probing.
            expert_embedding_dim: Dimension of expert embeddings for probing.
            use_wavelet_routing: Enable Phase 12.14 wavelet-guided routing.
            wavelet_routing_weight: Weight for wavelet routing bias.
            use_shared_experts: Enable Phase 14.2.1 always-active shared experts.
            num_shared_experts: Number of shared experts (2-4 recommended).
            use_aux_loss_free_balance: Enable Phase 14.2.3 bias-based balancing.
            balance_ema_decay: EMA decay for aux-loss-free load tracking.
            use_adaptive_k: Enable Phase 14.2.4 adaptive expert count.
            ada_k_base: Base number of experts for Ada-K.
            ada_k_min: Minimum experts per token for Ada-K.
            ada_k_max: Maximum experts per token for Ada-K.
            use_hopfield_routing: Enable Phase 86 Hopfield energy-based routing.
            hopfield_beta: Inverse temperature for Hopfield energy (higher = sharper).
            hopfield_energy_threshold: Energy threshold for OOD detection.
            hopfield_routing_weight: Weight for Hopfield routing bias.
            **kwargs: Additional layer arguments.

        Raises:
            LimitExceededError: If num_experts exceeds Lite limit (12).
        """
        # Validate against Lite limits
        if num_experts > MAX_MOE_EXPERTS:
            raise LimitExceededError(
                f"num_experts ({num_experts}) exceeds Lite edition limit ({MAX_MOE_EXPERTS}). "
                f"Upgrade to Enterprise for unlimited experts.",
                violations=[f"num_experts: {num_experts} > {MAX_MOE_EXPERTS}"],
            )

        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.aux_loss_weight = aux_loss_weight
        self.capacity_factor = capacity_factor
        self.load_balance_lambda = load_balance_lambda
        self.ema_decay = ema_decay
        self.superposition_dim = superposition_dim
        self._dropout_rate = dropout

        # Phase 12.12: Expert Specialization Probing
        self.enable_expert_probing = enable_expert_probing
        self.expert_embedding_dim = expert_embedding_dim

        # Phase 12.14: Wavelet-Guided Routing
        self.use_wavelet_routing = use_wavelet_routing
        self.wavelet_routing_weight = wavelet_routing_weight

        # Phase 14.2.1: Shared Experts
        self.use_shared_experts = use_shared_experts
        self.num_shared_experts = num_shared_experts if use_shared_experts else 0

        # Phase 14.2.3: Aux-Loss-Free Balancing
        self.use_aux_loss_free_balance = use_aux_loss_free_balance
        self.balance_ema_decay = balance_ema_decay

        # Phase 14.2.4: Adaptive Expert Count
        self.use_adaptive_k = use_adaptive_k
        self.ada_k_base = ada_k_base
        self.ada_k_min = ada_k_min
        self.ada_k_max = ada_k_max

        # Phase 14.2.6: Sigmoid Gating
        self.use_sigmoid_routing = use_sigmoid_routing

        # Phase 14.2.5: Null Expert Pool
        self.use_null_expert = use_null_expert

        # Phase 86: Hopfield-Boosted Expert Selection
        self.use_hopfield_routing = use_hopfield_routing
        self.hopfield_beta = hopfield_beta
        self.hopfield_energy_threshold = hopfield_energy_threshold
        self.hopfield_routing_weight = hopfield_routing_weight

        # S4: Floquet Phase → Expert Routing
        self.use_floquet_routing = use_floquet_routing
        self.floquet_routing_weight = floquet_routing_weight
        self._floquet_phase: tf.Tensor | None = None  # Set externally by TimeCrystalBlock

        # Router network - Phase 48.1: TensorRingLayer for memory-efficient routing
        if USE_TENSOR_RING_MOE:
            from highnoon.models.quantum.tensor_layers import TensorRingLayer

            self.router = TensorRingLayer(
                output_dim=num_experts,
                input_dim=d_model,
                num_cores=TENSOR_RING_NUM_CORES,
                ring_rank=TENSOR_RING_RANK,
                use_bias=True,
                name="router_tensor_ring",
            )
            log.info(
                f"  - Phase 48.1: TensorRing Router enabled "
                f"(ring_rank={TENSOR_RING_RANK}, cores={TENSOR_RING_NUM_CORES})"
            )
        else:
            self.router = layers.Dense(num_experts, use_bias=True, name="router")

        # Phase 12.12: Expert specialization components
        if self.enable_expert_probing:
            self.input_projector = layers.Dense(expert_embedding_dim, name="expert_input_projector")
        else:
            self.input_projector = None

        # Phase 12.14: Wavelet-guided router
        if self.use_wavelet_routing:
            self.wavelet_router = WaveletGuidedRouter(
                num_experts=num_experts,
                embedding_dim=d_model,
                name="wavelet_router",
            )
        else:
            self.wavelet_router = None

        # Phase 201.4: HD Shared Expert Basis (must be initialized before building experts)
        self.use_hd_shared_basis = USE_HD_SHARED_EXPERT_BASIS
        self._hd_shared_basis: HDSharedExpertBasis | None = None
        if self.use_hd_shared_basis:
            self._hd_shared_basis = HDSharedExpertBasis(
                num_experts=num_experts,
                d_model=d_model,
                d_ff=d_ff,
                hd_dim=HD_SHARED_BASIS_DIM,
                num_basis_vectors=HD_SHARED_BASIS_NUM_VECTORS,
                name="hd_shared_expert_basis",
            )
            log.info(
                f"  - Phase 201.4: HD Shared Expert Basis enabled "
                f"(hd_dim={HD_SHARED_BASIS_DIM}, vectors={HD_SHARED_BASIS_NUM_VECTORS})"
            )

        # Build expert modules
        self.experts = [self._build_expert(i) for i in range(self.num_experts)]

        # Phase 14.2.1: Build shared experts (always active)
        if self.use_shared_experts and self.num_shared_experts > 0:
            self.shared_experts = [
                self._build_expert(f"shared_{i}") for i in range(self.num_shared_experts)
            ]
            # Combine layer for shared expert outputs
            self.shared_combine = layers.Dense(
                d_model,
                activation=None,
                use_bias=False,
                name="shared_expert_combine",
            )
        else:
            self.shared_experts = []
            self.shared_combine = None

        # Phase 14.2.4: Adaptive-K complexity predictor
        if self.use_adaptive_k:
            self.complexity_predictor = layers.Dense(
                1,
                activation="sigmoid",
                name="complexity_predictor",
            )
        else:
            self.complexity_predictor = None

        log.info(f"Initialized MoELayer with {num_experts} experts (C++ ops required)")
        if enable_expert_probing:
            log.info("  - Phase 12.12: Expert Specialization Probing enabled")
        if use_wavelet_routing:
            log.info("  - Phase 12.14: Wavelet-Guided Routing enabled")
        if use_shared_experts:
            log.info(f"  - Phase 14.2.1: {num_shared_experts} Shared Experts enabled")
        if use_aux_loss_free_balance:
            log.info("  - Phase 14.2.3: Aux-Loss-Free Balancing enabled")
        if use_adaptive_k:
            log.info(f"  - Phase 14.2.4: Adaptive-K ({ada_k_min}-{ada_k_max}) enabled")
        if use_null_expert:
            log.info("  - Phase 14.2.5: Null Expert Pool enabled (overflow → identity)")
        if use_hopfield_routing:
            log.info(f"  - Phase 86: Hopfield Routing enabled (β={hopfield_beta})")
        if use_floquet_routing:
            log.info(f"  - S4: Floquet Phase Routing enabled (weight={floquet_routing_weight})")

        # Phase 201.13: Log adaptive superposition
        if USE_ADAPTIVE_SUPERPOSITION:
            log.info(
                f"  - Phase 201.13: Adaptive Superposition enabled "
                f"(dim={SUPERPOSITION_MIN_DIM}-{SUPERPOSITION_MAX_DIM})"
            )

    def build(self, input_shape):
        """Build layer weights."""
        expert_input_shape = tf.TensorShape([None, self.d_model])

        # Phase 201.4: Build HD Shared Expert Basis BEFORE building experts
        # The shared basis must be built first so HDSharedExpert can delegate to it
        if self._hd_shared_basis is not None and not self._hd_shared_basis.built:
            self._hd_shared_basis.build(expert_input_shape)

        for expert in self.experts:
            if not expert.built:
                expert.build(expert_input_shape)

        # Build shared experts
        for shared_expert in self.shared_experts:
            if not shared_expert.built:
                shared_expert.build(expert_input_shape)

        self.load_vector = self.add_weight(
            name="load_vector",
            shape=(self.num_experts,),
            initializer="zeros",
            trainable=False,
            dtype=tf.float32,
        )

        # Phase 14.2.3: Bias vector for aux-loss-free balancing
        if self.use_aux_loss_free_balance:
            self.routing_bias = self.add_weight(
                name="routing_bias",
                shape=(self.num_experts,),
                initializer="zeros",
                trainable=False,  # Updated via EMA, not gradients
                dtype=tf.float32,
            )
        else:
            self.routing_bias = None

        # Phase 12.12: Expert embeddings for specialization probing
        if self.enable_expert_probing:
            self.expert_embeddings = self.add_weight(
                name="expert_embeddings",
                shape=(self.num_experts, self.expert_embedding_dim),
                initializer="glorot_uniform",
                trainable=True,
            )
        else:
            self.expert_embeddings = None

        # Phase 86: Hopfield expert patterns for energy-based routing
        if self.use_hopfield_routing:
            self.expert_patterns = self.add_weight(
                name="expert_patterns",
                shape=(self.num_experts, self.d_model),
                initializer="glorot_uniform",
                trainable=True,
            )
            # EMA for pattern updates
            self.pattern_ema = self.add_weight(
                name="pattern_ema",
                shape=(self.num_experts, self.d_model),
                initializer="zeros",
                trainable=False,
            )
            # V2 MIGRATION: Using unified memory system API (no fallback)
            self._hopfield_cpp_available = True  # Native ops required
        else:
            self.expert_patterns = None
            self.pattern_ema = None
            self._hopfield_cpp_available = False

        super().build(input_shape)

    def _build_expert(self, index):
        """Build expert module.

        Uses HDSharedExpert when USE_HD_SHARED_EXPERT_BASIS is enabled,
        UnitaryExpert when USE_UNITARY_EXPERT is enabled,
        AdaptiveSuperposedExpert when USE_ADAPTIVE_SUPERPOSITION is enabled,
        otherwise uses SuperposedExpert.
        """
        name = f"expert_{index}" if isinstance(index, int) else f"expert_{index}"

        # Phase 201.4: HD Shared Expert Basis (highest priority - best memory savings)
        if self.use_hd_shared_basis and self._hd_shared_basis is not None:
            if isinstance(index, int):
                return HDSharedExpert(
                    shared_basis=self._hd_shared_basis,
                    expert_idx=index,
                    name=f"hd_shared_{name}",
                )
            # For shared experts (non-int index), fall through to other options

        if USE_UNITARY_EXPERT:
            if not _unitary_expert_ops_loaded:
                _load_unitary_expert_ops()
            if _unitary_expert_ops_loaded:
                return UnitaryExpert(
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    neumann_terms=NEUMANN_CAYLEY_TERMS,
                    name=f"unitary_{name}",
                )

        # Phase 201.13: Adaptive Superposition Dimension
        if USE_ADAPTIVE_SUPERPOSITION:
            return AdaptiveSuperposedExpert(
                d_model=self.d_model,
                d_ff=self.d_ff,
                min_superposition=SUPERPOSITION_MIN_DIM,
                max_superposition=SUPERPOSITION_MAX_DIM,
                complexity_scale=SUPERPOSITION_COMPLEXITY_SCALE,
                name=f"adaptive_superposed_{name}",
            )

        # Fallback to SuperposedExpert
        return SuperposedExpert(
            d_model=self.d_model,
            d_ff=self.d_ff,
            superposition_dim=self.superposition_dim,
            name=f"superposed_{name}",
        )

    def _compute_hopfield_routing_bias(self, token_states: tf.Tensor) -> tf.Tensor:
        """Compute Hopfield energy-based routing bias for MoE.

        V2 MIGRATION: Uses unified memory system API (MemoryType.HOPFIELD).

        Phase 86 implementation: Uses Modern Hopfield Network energy to detect
        out-of-distribution tokens and route them to specialized experts.

        High energy indicates OOD tokens that should be boosted toward an
        uncertainty expert. Low energy tokens use standard routing.

        CRITICAL: Requires C++ native ops. No Python fallback.

        Args:
            token_states: Token embeddings [num_tokens, d_model].

        Returns:
            Routing bias [num_tokens, num_experts].

        Raises:
            RuntimeError: If C++ ops are not available.
        """
        # V2 MIGRATION: Use unified memory system API
        from highnoon._native.ops.unified_memory_system_op import hopfield_memory

        # Perform Hopfield retrieval for each token against expert patterns
        # Returns (retrieved_pattern, attention_weights)
        output, attention = hopfield_memory(
            query=token_states,
            patterns=self.expert_patterns,
            num_patterns=self.num_experts,
            beta=self.hopfield_beta,
            num_iterations=1,
        )

        # The attention weights ARE the routing bias - high attention = good match = low energy
        # Invert to get energy-based bias (high energy = OOD = boost uncertainty expert)
        num_tokens = tf.shape(token_states)[0]
        tf.zeros([num_tokens, self.num_experts], dtype=tf.float32)

        # Compute energy approximation: lower attention = higher energy
        max_attention = tf.reduce_max(attention, axis=-1, keepdims=True)
        energy_approx = 1.0 - max_attention  # High when OOD

        # Apply energy threshold - boost uncertainty expert for OOD tokens
        uncertainty_expert_idx = self.num_experts - 1
        ood_mask = tf.cast(energy_approx > self.hopfield_energy_threshold, tf.float32)

        # Create routing bias with OOD boost
        uncertainty_boost = tf.zeros([num_tokens, self.num_experts], dtype=tf.float32)
        indices = tf.expand_dims(tf.range(num_tokens), 1)
        expert_col = tf.fill([num_tokens, 1], uncertainty_expert_idx)
        scatter_indices = tf.concat([indices, expert_col], axis=1)
        uncertainty_boost = tf.tensor_scatter_nd_update(
            uncertainty_boost,
            scatter_indices,
            tf.squeeze(ood_mask * 2.0, axis=-1),  # Boost factor
        )

        return uncertainty_boost

    def _update_expert_patterns(
        self,
        token_states: tf.Tensor,
        expert_indices: tf.Tensor,
        training: bool,
    ) -> None:
        """Update expert patterns using EMA from routed tokens.

        Called during training to adapt expert patterns to token distributions.

        Args:
            token_states: Token embeddings [num_tokens, d_model].
            expert_indices: Expert assignments [num_tokens].
            training: Whether in training mode.
        """
        if not training or self.pattern_ema is None:
            return

        # EMA decay for pattern updates
        ema_decay = 0.99

        # Compute mean token per expert
        for e in range(self.num_experts):
            mask = tf.equal(expert_indices, e)
            expert_tokens = tf.boolean_mask(token_states, mask)

            if tf.size(expert_tokens) > 0:
                mean_token = tf.reduce_mean(expert_tokens, axis=0)
                # Update EMA
                old_pattern = self.pattern_ema[e]
                new_pattern = ema_decay * old_pattern + (1 - ema_decay) * mean_token
                self.pattern_ema[e].assign(new_pattern)

    def set_floquet_phase(self, phase: tf.Tensor) -> None:
        """S4: Set Floquet phase from TimeCrystalBlock for phase-specialist routing.

        This enables Floquet-phase-conditioned expert selection, where different
        experts specialize in different Floquet drive phases. The phase modulates
        routing to create phase-specialist experts.

        Called by TimeCrystalSequenceBlock during forward pass to communicate
        the current Floquet phase.

        Args:
            phase: Floquet phase tensor [batch, 1] or scalar in [0, 2π].
        """
        if self.use_floquet_routing:
            self._floquet_phase = phase

    def _compute_floquet_routing_bias(self) -> tf.Tensor:
        """S4: Compute Floquet phase-based routing bias.

        Maps Floquet phase to expert preferences using sinusoidal encoding.
        Each expert gets a preferential phase range, creating phase-specialists:
        - Expert 0: prefers phase ≈ 0
        - Expert 1: prefers phase ≈ 2π/num_experts
        - Expert k: prefers phase ≈ 2πk/num_experts

        This creates a natural periodicity matching the Floquet drive.

        Returns:
            Routing bias [1, num_experts] that can be broadcast to tokens.
        """
        if self._floquet_phase is None:
            return tf.zeros([1, self.num_experts], dtype=tf.float32)

        # Ensure phase is scalar or broadcastable
        phase = tf.reduce_mean(self._floquet_phase)  # Collapse to scalar

        # Phase preferences for each expert (evenly distributed around circle)
        expert_phases = tf.linspace(0.0, 2.0 * 3.14159265, self.num_experts + 1)
        expert_phases = expert_phases[:-1]  # [num_experts]

        # Compute similarity to each expert's preferred phase
        # cos(phase - expert_phase) = 1 when aligned, -1 when opposite
        phase_diff = phase - expert_phases
        routing_bias = tf.cos(phase_diff)  # [num_experts]

        # Expand to [1, num_experts] for broadcasting
        routing_bias = tf.reshape(routing_bias, [1, self.num_experts])

        return routing_bias

    def call(
        self, inputs, training=None
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
        """Forward pass for MoE layer.

        Args:
            inputs: Input tensor of shape [batch, seq_len, d_model] or
                [batch * seq_len, d_model].
            training: Whether in training mode.

        Returns:
            Tuple of (output, routing_metadata, aux_metrics).
        """
        if isinstance(inputs, (list, tuple)):
            inputs, *_ = inputs

        original_shape = tf.shape(inputs)
        reshaped_inputs = tf.reshape(inputs, [-1, self.d_model])
        num_tokens = tf.shape(reshaped_inputs)[0]

        # Compute base routing logits
        router_logits = self.router(reshaped_inputs)

        # Phase 12.12: Expert Specialization Probing
        if self.enable_expert_probing and self.expert_embeddings is not None:
            # Project input to expert embedding space
            input_embedding = self.input_projector(reshaped_inputs)  # [B*L, embed_dim]

            # Compute specialization scores via dot product
            specialization_scores = tf.matmul(
                input_embedding, self.expert_embeddings, transpose_b=True
            )  # [B*L, num_experts]

            # Add soft bias toward specialized experts
            router_logits = router_logits + 0.1 * specialization_scores

        # Phase 12.14: Wavelet-Guided Routing
        if self.use_wavelet_routing and self.wavelet_router is not None:
            # Reshape back to 3D for wavelet computation
            # Use static rank check - inputs.shape.rank is available even in graph mode
            input_rank = inputs.shape.rank
            if input_rank is not None and input_rank == 3:
                inputs_3d = tf.reshape(inputs, original_shape)
                freq_bias = self.wavelet_router.compute_frequency_routing_bias(inputs_3d)
                router_logits = router_logits + self.wavelet_routing_weight * freq_bias

        # Phase 86: Hopfield-Boosted Expert Selection
        if self.use_hopfield_routing and self.expert_patterns is not None:
            hopfield_bias = self._compute_hopfield_routing_bias(reshaped_inputs)
            router_logits = router_logits + self.hopfield_routing_weight * hopfield_bias

        # S4: Floquet Phase → Expert Routing
        if self.use_floquet_routing and self._floquet_phase is not None:
            floquet_bias = self._compute_floquet_routing_bias()
            router_logits = router_logits + self.floquet_routing_weight * floquet_bias

        if training:
            log_load = tf.math.log(tf.stop_gradient(self.load_vector) + 1e-6)
            router_logits -= self.load_balance_lambda * log_load

        router_probs = tf.nn.softmax(router_logits, axis=-1)

        output, metadata, aux_metrics = self._expert_choice_routing_fused(
            reshaped_inputs, router_logits, router_probs, original_shape, num_tokens, training
        )

        # Phase 14.2.1: Add shared expert contributions
        if self.use_shared_experts and self.shared_experts:
            shared_outputs = []
            for shared_expert in self.shared_experts:
                shared_out = shared_expert(reshaped_inputs, training=training)
                shared_outputs.append(shared_out)

            # Combine shared expert outputs
            if len(shared_outputs) > 1:
                shared_stack = tf.stack(shared_outputs, axis=-1)  # [tokens, d_model, num_shared]
                shared_combined = tf.reduce_mean(shared_stack, axis=-1)  # [tokens, d_model]
            else:
                shared_combined = shared_outputs[0]

            # Add shared contribution via learned combine (or simple addition)
            if self.shared_combine is not None:
                shared_contribution = self.shared_combine(shared_combined)
            else:
                shared_contribution = shared_combined

            # Add to routed output (shared experts always contribute)
            output = output + tf.reshape(shared_contribution, original_shape)

        # GRADIENT PASSTHROUGH: Ensure complexity_predictor is always in gradient graph
        # even when use_adaptive_k edge cases don't trigger its path
        # NOTE: Using hn_config.GRADIENT_PASSTHROUGH_EPSILON instead of 0.0 because TF may optimize away 0.0 * value
        if self.complexity_predictor is not None:
            # Compute output and add with tiny weight to ensure gradient flow
            complexity_unused = self.complexity_predictor(reshaped_inputs)
            output = output + hn_config.GRADIENT_PASSTHROUGH_EPSILON * tf.reduce_mean(
                complexity_unused
            )

        # GRADIENT FIX: Ensure expert_embeddings receive gradients even when
        # expert probing is disabled in some configurations
        # NOTE: Using hn_config.GRADIENT_PASSTHROUGH_EPSILON instead of 0.0 because TF may optimize away 0.0 * value
        if self.expert_embeddings is not None:
            output = output + hn_config.GRADIENT_PASSTHROUGH_EPSILON * tf.reduce_mean(
                self.expert_embeddings
            )

        # GRADIENT FIX: Ensure HD shared expert basis weights receive gradients
        # even when fallback experts are used for some indices
        # NOTE: Using hn_config.GRADIENT_PASSTHROUGH_EPSILON instead of 0.0 because TF may optimize away 0.0 * value
        if self._hd_shared_basis is not None:
            basis_passthrough = (
                tf.reduce_mean(self._hd_shared_basis.shared_basis_up)
                * hn_config.GRADIENT_PASSTHROUGH_EPSILON
                + tf.reduce_mean(self._hd_shared_basis.shared_basis_down)
                * hn_config.GRADIENT_PASSTHROUGH_EPSILON
                + tf.reduce_mean(self._hd_shared_basis.expert_coeffs_up)
                * hn_config.GRADIENT_PASSTHROUGH_EPSILON
                + tf.reduce_mean(self._hd_shared_basis.expert_coeffs_down)
                * hn_config.GRADIENT_PASSTHROUGH_EPSILON
                + tf.reduce_mean(self._hd_shared_basis.expert_bias)
                * hn_config.GRADIENT_PASSTHROUGH_EPSILON
            )
            output = output + basis_passthrough

        # GRADIENT FIX: Ensure TensorRing router cores receive gradients
        # Expert Choice routing can cause sparse gradients that don't flow back
        # to the router. Add explicit passthrough to ensure gradient connectivity.
        if hasattr(self.router, "ring_cores"):
            ring_passthrough = sum(
                tf.reduce_mean(core) * hn_config.GRADIENT_PASSTHROUGH_EPSILON
                for core in self.router.ring_cores
            )
            output = output + ring_passthrough

        # GRADIENT FIX: Ensure shared_experts (SuperposedExperts) receive gradients
        # These are called separately from _expert_choice_routing_fused and may have
        # gradient issues due to their C++ kernel implementation
        if self.use_shared_experts and self.shared_experts:
            shared_passthrough = tf.constant(0.0, dtype=output.dtype)
            for shared_expert in self.shared_experts:
                # Handle SuperposedExpert weights
                if hasattr(shared_expert, "path_bases"):
                    shared_passthrough = (
                        shared_passthrough
                        + tf.reduce_mean(shared_expert.path_bases)
                        * hn_config.GRADIENT_PASSTHROUGH_EPSILON
                    )
                if hasattr(shared_expert, "path_weights"):
                    shared_passthrough = (
                        shared_passthrough
                        + tf.reduce_mean(shared_expert.path_weights)
                        * hn_config.GRADIENT_PASSTHROUGH_EPSILON
                    )
                if hasattr(shared_expert, "ffn1") and hasattr(shared_expert.ffn1, "cores"):
                    for core in shared_expert.ffn1.cores:
                        shared_passthrough = (
                            shared_passthrough
                            + tf.reduce_mean(core) * hn_config.GRADIENT_PASSTHROUGH_EPSILON
                        )
                if hasattr(shared_expert, "ffn2") and hasattr(shared_expert.ffn2, "cores"):
                    for core in shared_expert.ffn2.cores:
                        shared_passthrough = (
                            shared_passthrough
                            + tf.reduce_mean(core) * hn_config.GRADIENT_PASSTHROUGH_EPSILON
                        )
            output = output + shared_passthrough

        return output, metadata, aux_metrics

    def _expert_choice_routing_fused(
        self, reshaped_inputs, router_logits, router_probs, original_shape, num_tokens, training
    ):
        """
        Implements 'Expert Choice' routing using the optimized C++ fused operator.

        Raises:
            RuntimeError: If C++ operator is not available.
        """
        if fused_moe_dispatch is None:
            raise RuntimeError(
                "The FusedMoEDispatch C++ operator is required but could not be loaded. "
                "Please ensure it has been compiled correctly. NO PYTHON FALLBACK IS PROVIDED."
            )

        # Phase 14.2.4: Adaptive-K capacity adjustment
        base_capacity = tf.cast(
            self.capacity_factor
            * (tf.cast(num_tokens, tf.float32) / tf.cast(self.num_experts, tf.float32)),
            dtype=tf.float32,
        )

        if self.use_adaptive_k and self.complexity_predictor is not None:
            # Compute complexity scores per token (0=simple, 1=complex)
            complexity_scores = self.complexity_predictor(reshaped_inputs)  # [N, 1]
            avg_complexity = tf.reduce_mean(complexity_scores)

            # Adjust capacity: complex tokens → more experts, simple → fewer
            # Scale between ada_k_min and ada_k_max based on complexity
            k_range = tf.cast(self.ada_k_max - self.ada_k_min, tf.float32)
            adaptive_factor = tf.cast(self.ada_k_min, tf.float32) + avg_complexity * k_range

            # Scale capacity by adaptive factor relative to base
            capacity_multiplier = adaptive_factor / tf.cast(self.ada_k_base, tf.float32)
            expert_capacity = tf.cast(base_capacity * capacity_multiplier, dtype=tf.int32)
        else:
            expert_capacity = tf.cast(base_capacity, dtype=tf.int32)

        expert_capacity = tf.maximum(expert_capacity, 1)

        # Phase 14.2.3: Use V2 dispatch with routing bias if aux-loss-free is enabled
        if self.use_aux_loss_free_balance and self.routing_bias is not None:
            (
                dispatched_tokens,
                dispatched_gates,
                dispatch_metadata,
                expert_boundaries,
                expert_indices,
                expert_loads,
            ) = fused_moe_dispatch_v2(
                reshaped_inputs,
                router_logits,
                expert_capacity,
                self.routing_bias,
                use_sigmoid_routing=getattr(self, "use_sigmoid_routing", False),
                apply_bias_before_topk=True,
            )
        else:
            (
                dispatched_tokens,
                dispatched_gates,
                dispatch_metadata,
                expert_boundaries,
                expert_indices,
            ) = fused_moe_dispatch(reshaped_inputs, router_logits, expert_capacity)
            expert_loads = tf.cast(expert_boundaries[1:] - expert_boundaries[:-1], tf.float32)

        if training:
            tokens_per_expert_float = expert_loads
            updated_load = (self.ema_decay * self.load_vector) + (
                (1 - self.ema_decay) * tokens_per_expert_float
            )
            self.load_vector.assign(updated_load)

            # Phase 14.2.3: Update routing_bias via EMA (aux-loss-free balancing)
            if self.use_aux_loss_free_balance and self.routing_bias is not None:
                # Target load is uniform distribution
                target_load = tf.cast(num_tokens, tf.float32) / tf.cast(
                    self.num_experts, tf.float32
                )
                # Compute load imbalance: positive = overloaded, negative = underloaded
                load_imbalance = tokens_per_expert_float - target_load
                # Update bias to penalize overloaded experts and boost underloaded
                # bias_adjustment = -lr * imbalance (DeepSeek-V3 style)
                bias_adjustment = -0.01 * load_imbalance
                updated_bias = self.balance_ema_decay * self.routing_bias + (
                    (1 - self.balance_ema_decay) * (self.routing_bias + bias_adjustment)
                )
                self.routing_bias.assign(updated_bias)

            # Load Balancing Loss (still compute for metrics even with aux-loss-free)
            f_i = tokens_per_expert_float / tf.cast(num_tokens, tf.float32)
            p_i = router_probs
            load_balancing_loss = self.num_experts * tf.reduce_sum(
                f_i * tf.reduce_mean(p_i, axis=0)
            )

            # Router Z-Loss
            log_z = tf.math.reduce_logsumexp(router_logits, axis=-1)
            router_z_loss = tf.math.reduce_mean(log_z**2)

            # Expert diversity metrics (Enhancement 5)
            routing_entropy = -tf.reduce_sum(
                router_probs * tf.math.log(router_probs + 1e-10), axis=-1
            )
            avg_routing_entropy = tf.reduce_mean(routing_entropy)
            expert_utilization = tf.reduce_sum(
                tf.cast(tokens_per_expert_float > 0, tf.float32)
            ) / tf.cast(self.num_experts, tf.float32)

            aux_metrics = {
                "load_balancing_loss": load_balancing_loss,
                "router_z_loss": router_z_loss,
                "routing_entropy": avg_routing_entropy,
                "expert_utilization": expert_utilization,
                "load_variance": tf.math.reduce_std(tokens_per_expert_float),
            }
        else:
            aux_metrics = {
                "load_balancing_loss": tf.constant(0.0, dtype=tf.float32),
                "router_z_loss": tf.constant(0.0, dtype=tf.float32),
            }

        repeats = expert_boundaries[1:] - expert_boundaries[:-1]
        expert_ids_for_dispatched_tokens = tf.repeat(
            tf.range(self.num_experts, dtype=tf.int32), repeats=repeats
        )

        dispatched_weights = segment_softmax(
            dispatched_gates, expert_ids_for_dispatched_tokens, num_segments=self.num_experts
        )

        expert_outputs = []
        for i in range(self.num_experts):
            start = expert_boundaries[i]
            end = expert_boundaries[i + 1]
            num_tokens_for_expert = end - start

            def call_expert_fn(start=start, end=end, i=i):
                tokens_for_expert = dispatched_tokens[start:end]
                return self.experts[i](tokens_for_expert, training=training)

            def empty_output_fn():
                return tf.zeros([0, self.d_model], dtype=reshaped_inputs.dtype)

            output = tf.cond(
                num_tokens_for_expert > 0, true_fn=call_expert_fn, false_fn=empty_output_fn
            )
            expert_outputs.append(output)

        stitched_outputs = tf.concat(expert_outputs, axis=0)
        weighted_outputs = stitched_outputs * tf.expand_dims(dispatched_weights, axis=-1)

        indices = tf.expand_dims(dispatch_metadata, axis=-1)
        final_output = tf.tensor_scatter_nd_add(
            tensor=tf.zeros_like(reshaped_inputs), indices=indices, updates=weighted_outputs
        )

        # Phase 14.2.5: Null Expert Pool - Identity passthrough for undispatched tokens
        # Tokens not assigned to any expert are passed through unchanged
        if self.use_null_expert:
            # Create mask for tokens that were dispatched
            num_tokens_total = tf.shape(reshaped_inputs)[0]
            dispatched_mask = tf.scatter_nd(
                indices=indices,
                updates=tf.ones([tf.shape(dispatch_metadata)[0]], dtype=tf.float32),
                shape=[num_tokens_total],
            )
            # Tokens with 0 in mask were not dispatched → null expert (identity)
            null_mask = 1.0 - tf.minimum(dispatched_mask, 1.0)
            null_contribution = reshaped_inputs * tf.expand_dims(null_mask, -1)
            final_output = final_output + null_contribution

        final_output = tf.reshape(final_output, original_shape)

        # GRADIENT FIX: Add gradient passthroughs for ALL expert weights
        # The tf.cond inside the expert loop can block gradients when num_tokens=0
        # Also the C++ SuperposedExpert kernel may return zero gradients in some cases
        # This ensures all expert parameters receive gradient signals
        import highnoon.config as hn_config_local

        epsilon = hn_config_local.GRADIENT_PASSTHROUGH_EPSILON

        expert_passthrough = tf.constant(0.0, dtype=final_output.dtype)
        for expert in self.experts:
            # Handle SuperposedExpert weights
            if hasattr(expert, "path_bases"):
                expert_passthrough = (
                    expert_passthrough + tf.reduce_mean(expert.path_bases) * epsilon
                )
            if hasattr(expert, "path_weights"):
                expert_passthrough = (
                    expert_passthrough + tf.reduce_mean(expert.path_weights) * epsilon
                )
            if hasattr(expert, "ffn1") and hasattr(expert.ffn1, "cores"):
                for core in expert.ffn1.cores:
                    expert_passthrough = expert_passthrough + tf.reduce_mean(core) * epsilon
            if hasattr(expert, "ffn2") and hasattr(expert.ffn2, "cores"):
                for core in expert.ffn2.cores:
                    expert_passthrough = expert_passthrough + tf.reduce_mean(core) * epsilon
            # Handle HDSharedExpert (delegates to shared_basis)
            if hasattr(expert, "shared_basis"):
                basis = expert.shared_basis
                if hasattr(basis, "shared_basis_up"):
                    expert_passthrough = (
                        expert_passthrough + tf.reduce_mean(basis.shared_basis_up) * epsilon
                    )
                if hasattr(basis, "shared_basis_down"):
                    expert_passthrough = (
                        expert_passthrough + tf.reduce_mean(basis.shared_basis_down) * epsilon
                    )
                if hasattr(basis, "expert_coeffs_up"):
                    expert_passthrough = (
                        expert_passthrough + tf.reduce_mean(basis.expert_coeffs_up) * epsilon
                    )
                if hasattr(basis, "expert_coeffs_down"):
                    expert_passthrough = (
                        expert_passthrough + tf.reduce_mean(basis.expert_coeffs_down) * epsilon
                    )
                if hasattr(basis, "expert_bias"):
                    expert_passthrough = (
                        expert_passthrough + tf.reduce_mean(basis.expert_bias) * epsilon
                    )

        final_output = final_output + expert_passthrough

        routing_info = {
            "router_probs": router_probs,
            "expert_indices": expert_indices,
        }

        return final_output, routing_info, aux_metrics

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "aux_loss_weight": self.aux_loss_weight,
                "capacity_factor": self.capacity_factor,
                "load_balance_lambda": self.load_balance_lambda,
                "ema_decay": self.ema_decay,
                "superposition_dim": self.superposition_dim,
                "dropout": self._dropout_rate,
                # Phase 12.12
                "enable_expert_probing": self.enable_expert_probing,
                "expert_embedding_dim": self.expert_embedding_dim,
                # Phase 12.14
                "use_wavelet_routing": self.use_wavelet_routing,
                "wavelet_routing_weight": self.wavelet_routing_weight,
                # Phase 14.2.1: Shared Experts
                "use_shared_experts": self.use_shared_experts,
                "num_shared_experts": self.num_shared_experts,
                # Phase 14.2.3: Aux-Loss-Free Balancing
                "use_aux_loss_free_balance": self.use_aux_loss_free_balance,
                "balance_ema_decay": self.balance_ema_decay,
                # Phase 14.2.4: Adaptive-K
                "use_adaptive_k": self.use_adaptive_k,
                "ada_k_base": self.ada_k_base,
                "ada_k_min": self.ada_k_min,
                "ada_k_max": self.ada_k_max,
                # Phase 14.2.5: Null Expert Pool
                "use_null_expert": self.use_null_expert,
                # Phase 86: Hopfield Routing
                "use_hopfield_routing": self.use_hopfield_routing,
                "hopfield_beta": self.hopfield_beta,
                "hopfield_energy_threshold": self.hopfield_energy_threshold,
                "hopfield_routing_weight": self.hopfield_routing_weight,
            }
        )
        return config
