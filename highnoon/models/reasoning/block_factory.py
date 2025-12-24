# src/models/reasoning/block_factory.py
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

import json
from typing import Any

import tensorflow as tf

from highnoon import config

# MODIFIED: Delay imports to avoid circular dependency
# from ..spatial.mamba import ReasoningMamba2Block, SpatialBlock
# from ..spatial.kalman import KalmanBlock
# from ..hamiltonian import TimeCrystalBlock, TimeCrystalSequenceBlock
# from ..moe import MoELayer

# --- Logger Setup ---
logger = tf.get_logger()


# =============================================================================
# Phase 19-24: Unified Quantum Enhancements (Phases 19-24)
# =============================================================================


class QuantumEnhancedBlock(tf.keras.layers.Layer):
    """Wraps any reasoning block with unified quantum enhancements.

    Applies quantum-enhanced operations from Phases 19-24:
    - Port-Hamiltonian dynamics (energy-preserving with dissipation)
    - QSVT activations (Chebyshev polynomial approximations)
    - Orthogonalized keys (for attention blocks)

    All operations use float32 precision with NO quantization.
    """

    def __init__(
        self,
        inner_block: tf.keras.layers.Layer,
        embedding_dim: int,
        use_port_hamiltonian: bool = True,
        use_qsvt_activations: bool = True,
        use_entanglement_loss: bool = False,
        qsvt_degree: int = 8,
        **kwargs,
    ):
        """Initialize quantum-enhanced block wrapper.

        Args:
            inner_block: The underlying reasoning block to enhance.
            embedding_dim: Model embedding dimension.
            use_port_hamiltonian: Apply Port-Hamiltonian dynamics.
            use_qsvt_activations: Apply QSVT Chebyshev activations.
            use_entanglement_loss: Apply Entanglement Preservation Loss.
            qsvt_degree: Polynomial degree for QSVT.
        """
        name = kwargs.pop("name", f"quantum_{inner_block.name}")
        super().__init__(name=name, **kwargs)
        self.inner_block = inner_block
        self.embedding_dim = embedding_dim
        self.use_port_hamiltonian = use_port_hamiltonian and config.USE_PORT_HAMILTONIAN
        self.use_qsvt_activations = use_qsvt_activations and config.USE_QSVT_ACTIVATIONS
        self.use_entanglement_loss = use_entanglement_loss
        self.qsvt_degree = qsvt_degree

    def build(self, input_shape):
        """Build quantum enhancement layers."""
        if self.use_port_hamiltonian:
            # Skew-symmetric interconnection matrix
            self.j_upper = self.add_weight(
                name="j_upper",
                shape=(self.embedding_dim, self.embedding_dim),
                initializer="glorot_uniform",
                trainable=True,
            )
            # Dissipation (positive semi-definite diagonal)
            self.r_diag = self.add_weight(
                name="r_diag",
                shape=(self.embedding_dim,),
                initializer=tf.keras.initializers.Constant(0.01),
                trainable=True,
            )
            # Hamiltonian gradient projection
            self.h_proj = self.add_weight(
                name="h_proj",
                shape=(self.embedding_dim, self.embedding_dim),
                initializer="glorot_uniform",
                trainable=True,
            )

        if self.use_qsvt_activations:
            # Chebyshev coefficients approximating GELU
            self.qsvt_coefficients = self.add_weight(
                name="qsvt_coefficients",
                shape=(self.qsvt_degree + 1,),
                initializer=tf.keras.initializers.Constant(
                    [0.5, 0.5, 0.0, -0.044, 0.0, 0.003, 0.0, -0.0002, 0.0][: self.qsvt_degree + 1]
                ),
                trainable=True,
            )

        if not self.inner_block.built:
            self.inner_block.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False, **kwargs):
        """Forward pass with quantum enhancements.

        Args:
            inputs: Input tensor [B, L, D] or tuple (tensor, extra...) from previous block.
            training: Whether in training mode.

        Returns:
            Quantum-enhanced output [B, L, D] or tuple if inner block returns tuple.
        """
        # Handle tuple inputs from previous blocks in the reasoning stack
        # Some blocks return (output, aux_data) - we need to extract just the tensor
        if isinstance(inputs, tuple):
            actual_inputs = inputs[0]
            inputs[1:] if len(inputs) > 1 else None
        else:
            actual_inputs = inputs

        # Apply inner block first
        inner_result = self.inner_block(actual_inputs, training=training, **kwargs)

        # Handle tuple outputs (e.g., MoELayer returns (output, metadata, aux_metrics))
        extra_outputs = None
        if isinstance(inner_result, tuple):
            x = inner_result[0]
            extra_outputs = inner_result[1:]
        else:
            x = inner_result

        # Port-Hamiltonian dynamics (phase 19.2)
        if self.use_port_hamiltonian:
            x = self._apply_port_hamiltonian(x)

        # QSVT activation (phase 24.1)
        if self.use_qsvt_activations:
            x = self._apply_qsvt_activation(x)

        # Entanglement Preservation Loss (Phase 7)
        if self.use_entanglement_loss and training:
            self._apply_entanglement_loss(x)

        # Return tuple if inner block returned tuple
        if extra_outputs is not None:
            return (x,) + extra_outputs
        return x

    def _apply_port_hamiltonian(self, x: tf.Tensor) -> tf.Tensor:
        """Apply Port-Hamiltonian integration step."""
        # Make J skew-symmetric: J = J_upper - J_upper^T
        j_matrix = self.j_upper - tf.transpose(self.j_upper)
        # Make R positive semi-definite (diagonal with softplus)
        r_matrix = tf.linalg.diag(tf.nn.softplus(self.r_diag))

        # Compute gradient of Hamiltonian
        grad_h = tf.einsum("ij,blj->bli", self.h_proj, x)

        # Average over sequence for dynamics (reduce computation)
        x_mean = tf.reduce_mean(x, axis=1, keepdims=False)  # [B, D]
        grad_h_mean = tf.reduce_mean(grad_h, axis=1, keepdims=False)  # [B, D]

        # Port-Hamiltonian step: dx = [J - R] @ grad_H
        jmr = j_matrix - r_matrix  # [D, D]
        dx = tf.einsum("ij,bj->bi", jmr, grad_h_mean)  # [B, D]

        # Euler integration with small dt
        dt = 0.01
        x_evolved = x_mean + dt * dx  # [B, D]

        # Apply as residual to full sequence
        delta = x_evolved - x_mean  # [B, D]
        x = x + tf.expand_dims(delta, axis=1)  # Broadcast over L

        return x

    def _apply_qsvt_activation(self, x: tf.Tensor) -> tf.Tensor:
        """Apply QSVT Chebyshev polynomial activation."""
        # Normalize to [-1, 1] for Chebyshev stability
        x_norm = tf.nn.l2_normalize(x, axis=-1)

        # Chebyshev recurrence: T_0=1, T_1=x, T_{n+1}=2x*T_n - T_{n-1}
        t_nm1 = tf.ones_like(x_norm)
        t_n = x_norm
        result = self.qsvt_coefficients[0] * t_nm1

        if self.qsvt_degree >= 1:
            result = result + self.qsvt_coefficients[1] * t_n

        for n in range(2, self.qsvt_degree + 1):
            t_np1 = 2.0 * x_norm * t_n - t_nm1
            result = result + self.qsvt_coefficients[n] * t_np1
            t_nm1 = t_n
            t_n = t_np1

        # Small residual to not dominate the block's output
        return x + 0.1 * result

    def _apply_entanglement_loss(self, x: tf.Tensor) -> None:
        """Apply Phase 7 Entanglement Preservation Loss."""
        # Calculate bond entropy approximation (singular values of reshaped state)
        # Reshape [B, L, D] -> [B*L, D] for simplified bond entropy
        # In a real MPS, we'd reshape to [B*L, D_left, D_right]

        # SVD is expensive, so we use a proxy: variance of activations
        # or we use the custom EntanglementLoss op if inputs are tailored.
        # Here we assume x is the state.

        # Proxy: Compute variance across feature dimension (simple entropy proxy)
        # High variance suggests structure (low entropy), uniform suggests high entropy (tangled)
        # We want to MINIMIZE entanglement loss (keep entropy low? or high?
        # Area law says low entropy for ground states).

        # Load the custom op module
        from highnoon.models.layers.utils import load_custom_ops

        ops = load_custom_ops()
        if hasattr(ops, "EntanglementLoss"):
            # Compute a proxy for bond entropies
            # For simplicity, use the variance of the state vector as a proxy for singular values
            # S = svd(x). s^2 are eigenvalues of rho.

            # Use random projection to estimate singular value spread?
            # Or just use the custom op on the raw state if the op supports it.
            # The custom op takes "bond_entropies".

            # Let's compute a simple entropy per token
            # p = softmax(abs(x))
            # H = -sum(p log p)
            probs = tf.nn.softmax(tf.abs(x), axis=-1)
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=-1)  # [B, L]

            loss = ops.EntanglementLoss(
                bond_entropies=tf.reshape(entropy, [-1]), min_entropy=0.1, weight=0.01
            )
            self.add_loss(loss)

    def get_weights_for_fused_op(self):
        """Get weights for fused C++ op compatibility."""
        if hasattr(self.inner_block, "get_weights_for_fused_op"):
            return self.inner_block.get_weights_for_fused_op()
        return self.inner_block.weights

    def fused_metadata(self):
        """Get metadata for fused C++ op."""
        if hasattr(self.inner_block, "fused_metadata"):
            meta = self.inner_block.fused_metadata()
        else:
            meta = {"block_type": type(self.inner_block).__name__}
        meta["quantum_enhanced"] = True
        meta["use_port_hamiltonian"] = self.use_port_hamiltonian
        meta["use_qsvt_activations"] = self.use_qsvt_activations
        meta["use_entanglement_loss"] = self.use_entanglement_loss
        return meta

    @property
    def weights(self):
        """Return combined weights from inner block and quantum layers."""
        return self.inner_block.weights + super().weights


def _extract_tt_layer_info(
    block: tf.keras.layers.Layer, block_weights: list[tf.Tensor]
) -> list[dict[str, Any]]:
    """Extract TT layer metadata from a block for C++ kernel."""
    from ..tensor_layers import SuperpositionTTLayer, TTLayer

    tt_layers = []
    # Build a lookup so we can map core tensors back to their position in the
    # block's weight list. This ensures `core_indices` line up with the fused
    # kernel's flat weight ordering (which includes non-TT weights).
    weight_index_map = {id(w): idx for idx, w in enumerate(block_weights)}

    # Iterate using __dict__ to preserve attribute declaration order from __init__
    # (dir() is alphabetical and scrambled our TT core ordering).
    for attr_name, attr in block.__dict__.items():
        if isinstance(attr, (TTLayer, SuperpositionTTLayer)):
            core_indices: list[int] = []
            for core in getattr(attr, "cores", []):
                idx = weight_index_map.get(id(core))
                if idx is None:
                    logger.warning(
                        "[reasoning_stack] Skipping TT layer '%s' on block %s; core not found in weight list",
                        attr_name,
                        block.name,
                    )
                    core_indices = []
                    break
                core_indices.append(idx)

            if len(core_indices) != getattr(attr, "d", 0):
                logger.warning(
                    "[reasoning_stack] TT layer '%s' on block %s has %d cores but located %d in weights; "
                    "fused kernel will ignore this layer.",
                    attr_name,
                    block.name,
                    getattr(attr, "d", -1),
                    len(core_indices),
                )
                continue

            tt_layer_info = {
                "name": attr_name,
                "input_dims": list(attr.input_dims),
                "output_dims": list(attr.output_dims),
                "tt_ranks": list(attr.tt_ranks),
                "num_cores": int(attr.d),
                "core_indices": core_indices,
            }

            # Add superposition info if it's a SuperpositionTTLayer
            if isinstance(attr, SuperpositionTTLayer):
                tt_layer_info["superposition_dim"] = int(attr.superposition_dim)
                tt_layer_info["is_superposition"] = True
            else:
                tt_layer_info["is_superposition"] = False

            tt_layers.append(tt_layer_info)

    return tt_layers


def _build_block_descriptor(
    block: tf.keras.layers.Layer, block_weights: list[tf.Tensor], embedding_dim: int
) -> dict[str, Any]:
    """Constructs a serializable descriptor for a reasoning block."""
    weight_count = len(block_weights)
    descriptor: dict[str, Any]
    if hasattr(block, "get_fused_op_descriptor"):
        descriptor = dict(block.get_fused_op_descriptor())
    else:
        # Use class name comparison to avoid circular import issues
        stateful_types = {"ReasoningMamba2Block", "KalmanBlock", "TimeCrystalSequenceBlock"}
        descriptor = {
            "type": block.__class__.__name__,
            "stateful": block.__class__.__name__ in stateful_types,
            "metadata": {},
        }
    descriptor["weight_count"] = int(weight_count)
    metadata = dict(descriptor.get("metadata") or {})
    metadata["embedding_dim"] = int(embedding_dim)
    for attr in ("d_inner", "state_dim", "conv_dim", "num_heads", "num_experts"):
        if attr not in metadata and hasattr(block, attr):
            metadata[attr] = int(getattr(block, attr))

    # Add TT layer metadata if block contains TT layers
    tt_layers = _extract_tt_layer_info(block, block_weights)
    if tt_layers:
        metadata["tt_layers"] = tt_layers

    descriptor["metadata"] = metadata
    descriptor.setdefault("type", block.__class__.__name__)
    descriptor.setdefault("stateful", False)
    return descriptor


def create_reasoning_stack(
    num_layers: int,
    embedding_dim: int,
    name: str,  # ADDED: Accept a base name for the stack
    **kwargs,
) -> tuple[list[tf.keras.layers.Layer], list[tf.Tensor], list[int], list[str]]:
    """
    Creates the stack of reasoning blocks based on the global configuration pattern.

    MODIFIED: Now returns a tuple containing:
              1. The list of reasoning layer objects.
              2. A flat, ordered list of all trainable weight and constant
                 tensors required by all blocks in the sequence. This list is passed
                 to the single FusedReasoningStack C++ op (which now supports TT decomposition).
              3. A list of integers specifying the number of weights each block consumes.
              4. A list of JSON descriptor strings for blocks (including TT metadata).

    Args:
        num_layers (int): The total number of layers in the stack.
        embedding_dim (int): The embedding dimension for the models.
        name (str): The base name for the reasoning stack, used to scope its layers.
        **kwargs: Dictionary of hyperparameters from the tuning script.

    Returns:
        A tuple of (List[tf.keras.layers.Layer], List[tf.Tensor], List[int], List[str]).
    """
    # Delayed imports to avoid circular dependency
    from ..hamiltonian import TimeCrystalSequenceBlock
    from ..layers.wlam import WLAMBlock
    from ..moe import MoELayer
    from ..spatial.kalman import KalmanBlock
    from ..spatial.mamba import ReasoningMamba2Block, SpatialBlock
    from .latent_reasoning import LatentReasoningBlock

    reasoning_blocks = []
    blocks_tensors = []
    block_weight_counts = []
    block_descriptors: list[str] = []
    pattern = kwargs.get("block_pattern", config.REASONING_BLOCK_PATTERN)

    # Phase 10.4: Get latent reasoning config
    num_thought_steps = kwargs.get("num_thought_steps", config.NUM_THOUGHT_STEPS)

    # Handle 'mamba_only' pattern for HPO trials (uses only SpatialBlock)
    if pattern == "mamba_only":
        logger.info(
            "--- [create_reasoning_stack] Using 'mamba_only' pattern (SpatialBlocks only) ---"
        )
        for i in range(num_layers):
            mamba_state_dim = kwargs.get("spatial_state_dim", config.MAMBA2_STATE_DIM)
            mamba_conv_dim = kwargs.get("spatial_conv_dim", config.MAMBA2_CONV_DIM)
            mamba_expand_factor = kwargs.get("mamba2_expand_factor", config.MAMBA2_EXPAND_FACTOR)

            block = SpatialBlock(
                embedding_dim=embedding_dim,
                state_dim=mamba_state_dim,
                conv_dim=mamba_conv_dim,
                expand_factor=mamba_expand_factor,
                name=f"spatial_block_{i}",
            )
            block.build(tf.TensorShape([None, None, embedding_dim]))
            reasoning_blocks.append(block)

            weights = block.get_weights_for_fused_op()
            blocks_tensors.append(weights)
            block_weight_counts.append(len(weights))
            block_descriptors.append(json.dumps(block.fused_metadata()))

        logger.info(
            f"--- [create_reasoning_stack] Created {num_layers} SpatialBlocks (mamba_only) ---"
        )
        return reasoning_blocks, blocks_tensors, block_weight_counts, block_descriptors

    logger.info(
        "--- [create_reasoning_stack] Using focused block pattern: 'mamba_timecrystal_latent_wlam_moe_hybrid' ---"
    )
    for i in range(num_layers):
        mamba_state_dim = kwargs.get("spatial_state_dim", config.MAMBA2_STATE_DIM)
        kwargs.get("mamba2_head_dim", config.MAMBA2_HEAD_DIM)
        mamba_conv_dim = kwargs.get("spatial_conv_dim", config.MAMBA2_CONV_DIM)
        mamba_expand_factor = kwargs.get("mamba2_expand_factor", config.MAMBA2_EXPAND_FACTOR)
        d_inner = embedding_dim * mamba_expand_factor
        num_experts = kwargs.get("num_experts", config.NUM_EXPERTS)
        superposition_dim = kwargs.get("superposition_dim", config.SUPERPOSITION_DIM)
        wlam_num_heads = kwargs.get("wlam_num_heads", config.WLAM_NUM_HEADS)
        wlam_wavelet_kernel_size = kwargs.get(
            "wlam_wavelet_kernel_size", config.WLAM_WAVELET_KERNEL_SIZE
        )
        reasoning_ff_dim = kwargs.get("reasoning_ff_dim", config.REASONING_FF_DIM)

        block = None

        block_type = i % 6  # Phase 10.4: Pattern now has 6 unique blocks
        if block_type == 0:
            block = SpatialBlock(
                embedding_dim=embedding_dim,
                state_dim=mamba_state_dim,
                conv_dim=mamba_conv_dim,
                expand_factor=mamba_expand_factor,
                name=f"spatial_block_{i}",
            )
        elif block_type == 1:
            # Get Lorentzian config from kwargs or global config
            use_lorentzian = kwargs.get("use_lorentzian", config.USE_LORENTZIAN_TRANSFORM)
            lorentzian_dim = kwargs.get("lorentzian_dim", config.LORENTZIAN_HYPERBOLIC_DIM)

            block = TimeCrystalSequenceBlock(
                embedding_dim=embedding_dim,
                state_dim=embedding_dim // 2,
                hamiltonian_hidden_dim=embedding_dim,
                d_inner=d_inner,
                use_lorentzian=use_lorentzian,
                lorentzian_dim=lorentzian_dim,
                name=f"timecrystal_block_{i}",
            )

            # Phase 2: Add KalmanBlock after TimeCrystal for state estimation
            # Provides uncertainty tracking and corrects Hamiltonian integration drift
            use_kalman = kwargs.get("use_kalman", config.USE_KALMAN_FILTERING)
            if use_kalman:
                kalman_state_dim = kwargs.get("kalman_state_dim", config.KALMAN_STATE_DIM)
                kalman_block = KalmanBlock(
                    embedding_dim=embedding_dim,
                    state_dim=kalman_state_dim,
                    d_inner=32,
                    name=f"kalman_block_{i}",
                )
                # First add and wrap the TimeCrystal block
                if config.USE_PORT_HAMILTONIAN or config.USE_QSVT_ACTIVATIONS:
                    block = QuantumEnhancedBlock(
                        inner_block=block,
                        embedding_dim=embedding_dim,
                        use_port_hamiltonian=config.USE_PORT_HAMILTONIAN,
                        use_qsvt_activations=config.USE_QSVT_ACTIVATIONS,
                        qsvt_degree=config.QSVT_POLYNOMIAL_DEGREE,
                    )
                reasoning_blocks.append(block)

                # Then add the KalmanBlock (also quantum-enhanced)
                if config.USE_PORT_HAMILTONIAN or config.USE_QSVT_ACTIVATIONS:
                    kalman_block = QuantumEnhancedBlock(
                        inner_block=kalman_block,
                        embedding_dim=embedding_dim,
                        use_port_hamiltonian=config.USE_PORT_HAMILTONIAN,
                        use_qsvt_activations=config.USE_QSVT_ACTIVATIONS,
                        qsvt_degree=config.QSVT_POLYNOMIAL_DEGREE,
                    )
                reasoning_blocks.append(kalman_block)
                block = None  # Already added, skip the generic append
        elif block_type == 2:
            # Phase 10.4: LatentReasoningBlock for iterative thought refinement
            # Phase 14.3: Unified with ContinuousThought (COCONUT)
            use_continuous_thought = kwargs.get(
                "use_continuous_thought", config.USE_CONTINUOUS_THOUGHT
            )
            continuous_thought_steps = kwargs.get(
                "continuous_thought_steps", config.CONTINUOUS_THOUGHT_STEPS
            )
            # Phase 25: Quantum Holographic Memory (Persistent)
            use_holographic_memory = kwargs.get(
                "use_quantum_holographic_memory", config.USE_QUANTUM_HOLOGRAPHIC_MEMORY
            )

            block = LatentReasoningBlock(
                embedding_dim=embedding_dim,
                num_thought_steps=num_thought_steps,
                ff_expansion=4,
                adaptive_exit=True,
                use_continuous_thought=use_continuous_thought,
                continuous_thought_steps=continuous_thought_steps,
                use_holographic_memory=use_holographic_memory,
                name=f"latent_reasoning_{i}",
            )

            # Phase 5: Add SelfConsistencyVerifier after LatentReasoning for confidence scoring
            use_self_consistency = kwargs.get("use_self_consistency", config.USE_SELF_CONSISTENCY)
            if use_self_consistency:
                from .self_consistency import SelfConsistencyVerifier

                # First add and wrap the LatentReasoning block
                if (
                    config.USE_PORT_HAMILTONIAN
                    or config.USE_QSVT_ACTIVATIONS
                    or kwargs.get("use_entanglement_loss")
                ):
                    block = QuantumEnhancedBlock(
                        inner_block=block,
                        embedding_dim=embedding_dim,
                        use_port_hamiltonian=config.USE_PORT_HAMILTONIAN,
                        use_qsvt_activations=config.USE_QSVT_ACTIVATIONS,
                        use_entanglement_loss=kwargs.get("use_entanglement_loss", False),
                        qsvt_degree=config.QSVT_POLYNOMIAL_DEGREE,
                    )
                reasoning_blocks.append(block)

                # Then add the SelfConsistencyVerifier
                verifier = SelfConsistencyVerifier(
                    embedding_dim=embedding_dim,
                    num_verification_heads=4,
                    name=f"self_consistency_{i}",
                )
                # Wrap verifier if quantum enhancements enabled
                if config.USE_PORT_HAMILTONIAN or config.USE_QSVT_ACTIVATIONS:
                    verifier = QuantumEnhancedBlock(
                        inner_block=verifier,
                        embedding_dim=embedding_dim,
                        use_port_hamiltonian=config.USE_PORT_HAMILTONIAN,
                        use_qsvt_activations=config.USE_QSVT_ACTIVATIONS,
                        qsvt_degree=config.QSVT_POLYNOMIAL_DEGREE,
                    )
                reasoning_blocks.append(verifier)
                block = None  # Already added, skip the generic append
        elif block_type == 3:
            block = SpatialBlock(  # Second spatial block for more spatial reasoning
                embedding_dim=embedding_dim,
                state_dim=mamba_state_dim,
                conv_dim=mamba_conv_dim,
                expand_factor=mamba_expand_factor,
                name=f"spatial_block_{i}_alt",
            )
        elif block_type == 4:
            wlam_num_levels = kwargs.get("wlam_num_levels", config.WLAM_NUM_LEVELS)
            wlam_use_lifting = kwargs.get("wlam_use_lifting", config.WLAM_USE_LIFTING)
            wlam_scattering_layers = kwargs.get(
                "wlam_scattering_layers", config.WLAM_SCATTERING_LAYERS
            )
            wlam_scattering_pool = kwargs.get("wlam_scattering_pool", config.WLAM_SCATTERING_POOL)
            wlam_use_cross_attn = kwargs.get("wlam_use_cross_attn", config.WLAM_USE_CROSS_ATTN)
            block = WLAMBlock(
                embedding_dim=embedding_dim,
                num_heads=wlam_num_heads,
                wavelet_kernel_size=wlam_wavelet_kernel_size,
                num_levels=wlam_num_levels,
                use_lifting=wlam_use_lifting,
                scattering_layers=wlam_scattering_layers,
                scattering_pool=wlam_scattering_pool,
                use_cross_attn=wlam_use_cross_attn,
                name=f"wlam_block_{i}",
            )
        elif block_type == 5:
            block = MoELayer(
                num_experts=num_experts,
                d_model=embedding_dim,
                d_ff=reasoning_ff_dim,
                superposition_dim=superposition_dim,
                name=f"moe_layer_{i}",
            )

        if block:
            # Phase 19-24: Wrap with quantum enhancements if enabled
            # This adds Port-Hamiltonian dynamics and QSVT activations
            use_quantum = (
                config.USE_PORT_HAMILTONIAN
                or config.USE_QSVT_ACTIVATIONS
                or kwargs.get("use_entanglement_loss", False)
            )
            if use_quantum:
                block = QuantumEnhancedBlock(
                    inner_block=block,
                    embedding_dim=embedding_dim,
                    use_port_hamiltonian=config.USE_PORT_HAMILTONIAN,
                    use_qsvt_activations=config.USE_QSVT_ACTIVATIONS,
                    use_entanglement_loss=kwargs.get("use_entanglement_loss", False),
                    qsvt_degree=config.QSVT_POLYNOMIAL_DEGREE,
                )
            reasoning_blocks.append(block)
    for block in reasoning_blocks:
        # Build the layer to ensure weights are created
        if not block.built:
            # We provide a dummy shape that is sufficient for most layers to build their weights.
            # Stateful layers have more complex build requirements handled in the ReasoningModule.
            # Check inner block type for wrapped layers
            inner_block = getattr(block, "inner_block", block)
            if isinstance(inner_block, (ReasoningMamba2Block, KalmanBlock)):
                block.build((tf.TensorShape([None, None, embedding_dim]), (None, None)))
            else:
                block.build(tf.TensorShape([None, None, embedding_dim]))

        if hasattr(block, "get_weights_for_fused_op"):
            block_weights = list(block.get_weights_for_fused_op())
        elif isinstance(block, TimeCrystalSequenceBlock):
            block_weights = list(block.cell.weights)
        else:
            block_weights = list(block.weights)

        blocks_tensors.extend(block_weights)
        weight_delta = len(block_weights)
        block_weight_counts.append(weight_delta)
        descriptor = _build_block_descriptor(block, block_weights, embedding_dim)
        block_descriptors.append(json.dumps(descriptor, sort_keys=True))

    # --- START: Sanity Check Logging ---
    logger.info(
        f"--- [create_reasoning_stack] Created {len(reasoning_blocks)} reasoning blocks: "
        f"{', '.join(type(b).__name__ for b in reasoning_blocks)} ---"
    )
    logger.info(
        "--- [create_reasoning_stack] All blocks now support the C++ fused kernel (including TT layers)"
    )
    # --- END: Sanity Check Logging ---
    return reasoning_blocks, blocks_tensors, block_weight_counts, block_descriptors
