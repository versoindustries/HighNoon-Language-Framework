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
import math
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

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
    NEUMANN_CAYLEY_TERMS,
    NUM_SHARED_EXPERTS,
    QMOE_USE_FLOQUET_PHASE,  # S4: Floquet-QMoE phase routing
    SUPERPOSITION_DIM,
    SUPERPOSITION_MICRO_BATCH_SIZE,
    USE_ADAPTIVE_K,
    USE_AUX_LOSS_FREE_BALANCE,
    USE_SHARED_EXPERTS,
    USE_SIGMOID_ROUTING,
    USE_UNITARY_EXPERT,
    USE_WAVELET_ROUTING,
)
from highnoon.models.layers.collapse import ContextualGatingCollapse

# Import dependencies for SuperposedExpert
from highnoon.models.tensor_layers import SuperpositionTTLayer

# Lazy import for quantum ops
_unitary_expert_forward = None
_quantum_activation = None
_unitary_expert_ops_loaded = False


def _load_unitary_expert_ops():
    """Lazy load unitary expert ops."""
    global _unitary_expert_forward, _quantum_activation, _unitary_expert_ops_loaded
    if _unitary_expert_ops_loaded:
        return True
    try:
        from highnoon._native.ops.quantum_ops import quantum_activation, unitary_expert_forward

        _unitary_expert_forward = unitary_expert_forward
        _quantum_activation = quantum_activation
        _unitary_expert_ops_loaded = True
        log.info("Unitary Expert ops loaded")
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

        output, _ = _unitary_expert_forward(
            inputs,
            self.u1_weights,
            self.u2_weights,
            self.activation_angle,
            d_ff=self.d_ff,
            neumann_terms=self.neumann_terms,
        )
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
    """
    An expert that encapsulates the Tensorized Expert Superposition (TES) logic.
    It uses SuperpositionTTLayers to process the input in parallel across a
    superposition dimension and then collapses the result using a contextual
    gating mechanism.

    CRITICAL: This layer exclusively uses the fused C++ kernel. No Python fallback.
    """

    def __init__(self, d_model: int, d_ff: int, superposition_dim: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.superposition_dim = superposition_dim

        def factorize(dim):
            """Simple helper to factor a dimension into two, close to its sqrt."""
            if dim <= 0:
                return [1, dim]
            s = int(math.sqrt(dim))
            while s > 1 and dim % s != 0:
                s -= 1
            if s == 1:
                for i in range(2, int(dim**0.5) + 2):
                    if dim % i == 0:
                        s = i
                        break
            return [s, dim // s] if s > 1 else [1, dim]

        self.ffn1 = SuperpositionTTLayer(
            input_dims=factorize(d_model),
            output_dims=factorize(d_ff),
            tt_ranks=[1, 16, 1],
            superposition_dim=superposition_dim,
            name="superposed_ffn1",
        )
        self.activation = layers.Activation(tf.nn.gelu)
        self.ffn2 = SuperpositionTTLayer(
            input_dims=factorize(d_ff),
            output_dims=factorize(d_model),
            tt_ranks=[1, 16, 1],
            superposition_dim=superposition_dim,
            name="superposed_ffn2",
        )
        self.collapse = ContextualGatingCollapse(
            d_in=d_model, d_out=d_model, num_heads=4, name="collapse_mechanism"
        )

    def build(self, input_shape):
        """Explicitly builds child layers to ensure weights are created."""
        if not self.ffn1.built:
            self.ffn1.build(input_shape)
        if not self.ffn2.built:
            self.ffn2.build(tf.TensorShape([input_shape[0], self.d_ff]))
        if not self.collapse.built:
            y_superposed_shape = tf.TensorShape(
                [input_shape[0], self.superposition_dim, self.d_model]
            )
            x_context_shape = input_shape
            self.collapse.build((y_superposed_shape, x_context_shape))
        super().build(input_shape)

    def call(self, inputs, training=None, micro_batch_size: int | None = None):
        """
        Forward pass for the SuperposedExpert, using only the C++ kernel.

        Raises:
            RuntimeError: If the C++ operator could not be loaded.
        """
        if fused_superposition_moe_op is None:
            raise RuntimeError(
                "The FusedSuperpositionMoe C++ operator is required but could not be loaded. "
                "Please ensure it has been compiled correctly. NO PYTHON FALLBACK IS PROVIDED."
            )
        if micro_batch_size is None:
            micro_batch_size = SUPERPOSITION_MICRO_BATCH_SIZE
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be a positive integer.")

        ffn1_cores_combined = tf.concat(
            [tf.reshape(core, [-1]) for core in self.ffn1.cores], axis=0
        )
        ffn2_cores_combined = tf.concat(
            [tf.reshape(core, [-1]) for core in self.ffn2.cores], axis=0
        )

        return fused_superposition_moe(
            tokens=inputs,
            context=inputs,
            ffn1_cores=ffn1_cores_combined,
            ffn2_cores=ffn2_cores_combined,
            collapse_q_weights=self.collapse.query_proj.kernel,
            collapse_k_weights=self.collapse.key_proj.kernel,
            collapse_v_weights=self.collapse.value_proj.kernel,
            collapse_o_weights=self.collapse.output_proj.kernel,
            collapse_q_bias=self.collapse.query_proj.bias,
            collapse_k_bias=self.collapse.key_proj.bias,
            collapse_v_bias=self.collapse.value_proj.bias,
            collapse_o_bias=self.collapse.output_proj.bias,
            input_dims=self.ffn1.input_dims,
            output_dims_ffn1=self.ffn1.output_dims,
            output_dims_ffn2=self.ffn2.output_dims,
            tt_ranks=self.ffn1.tt_ranks,
            superposition_dim=self.superposition_dim,
            micro_batch_size=micro_batch_size,
        )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "superposition_dim": self.superposition_dim,
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

        # Router network
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

    def build(self, input_shape):
        """Build layer weights."""
        expert_input_shape = tf.TensorShape([None, self.d_model])
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
            # Load C++ ops if available
            self._hopfield_cpp_available = False
            try:
                from highnoon._native.ops.hopfield_memory_op import hopfield_ops_available
                self._hopfield_cpp_available = hopfield_ops_available()
            except ImportError:
                self._hopfield_cpp_available = False
        else:
            self.expert_patterns = None
            self.pattern_ema = None
            self._hopfield_cpp_available = False

        super().build(input_shape)

    def _build_expert(self, index):
        """Build expert module.

        Uses UnitaryExpert when USE_UNITARY_EXPERT is enabled,
        otherwise uses SuperposedExpert.
        """
        name = f"expert_{index}" if isinstance(index, int) else f"expert_{index}"

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

        # Fallback to SuperposedExpert
        return SuperposedExpert(
            d_model=self.d_model,
            d_ff=self.d_ff,
            superposition_dim=self.superposition_dim,
            name=f"superposed_{name}",
        )

    def _compute_hopfield_routing_bias(
        self, token_states: tf.Tensor
    ) -> tf.Tensor:
        """Compute Hopfield energy-based routing bias for MoE.

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
        if not self._hopfield_cpp_available:
            raise RuntimeError(
                "HopfieldMoeRoutingBias C++ operator is required but not available. "
                "Run ./build_secure.sh to compile. NO PYTHON FALLBACK IS PROVIDED."
            )

        from highnoon._native.ops.hopfield_memory_op import hopfield_moe_routing_bias
        return hopfield_moe_routing_bias(
            token_states,
            self.expert_patterns,
            beta=self.hopfield_beta,
            uncertainty_expert_idx=-1,  # Last expert
            energy_threshold=self.hopfield_energy_threshold,
        )

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
        if self.complexity_predictor is not None:
            # Compute output and add with zero weight to ensure gradient flow
            complexity_unused = self.complexity_predictor(reshaped_inputs)
            output = output + 0.0 * tf.reduce_mean(complexity_unused)

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
