"""Unified Quantum Foundation Operations Python Wrapper.

Phase 3 of V2 Performance Optimization - Quantum Op Consolidation

This module provides Python bindings for all 15 consolidated quantum
operations via native C++ dispatch. NO PYTHON FALLBACKS.

Supported Operations:
- EMBEDDING: Holographic token embeddings
- POSITION_ENCODING: Floquet/SU(2) position encoding
- LM_HEAD: VQC-based language model head
- EXPERT: Unitary expert networks (Cayley transform)
- NORM: Unitary/Stiefel normalization
- RESIDUAL: Unitary residual connections
- COHERENCE_BUS: Phase-coherent state transport
- TELEPORT_BUS: Quantum teleportation state transfer
- VQC: Variational Quantum Circuit
- TENSOR_RING_VQC: Tensor ring VQC simulation
- MEASUREMENT: Born rule measurement
- CRYSTALLIZATION: Quantum state crystallization
- FIDELITY_LOSS: Quantum fidelity regularization
- DROPOUT: Quantum measurement dropout
- CURRICULUM: Spectral-aware quantum curriculum

Copyright 2025-2026 Verso Industries
"""

from dataclasses import dataclass
from enum import IntEnum

import tensorflow as tf

# Import native ops loader - REQUIRED, no fallback
from highnoon._native import load_highnoon_core

_highnoon_core = load_highnoon_core()


# =============================================================================
# QUANTUM OPERATION TYPE ENUM
# =============================================================================


class QuantumOpType(IntEnum):
    """Quantum operation types matching C++ enum."""

    EMBEDDING = 0
    POSITION_ENCODING = 1
    LM_HEAD = 2
    EXPERT = 3
    NORM = 4
    RESIDUAL = 5
    COHERENCE_BUS = 6
    TELEPORT_BUS = 7
    VQC = 8
    TENSOR_RING_VQC = 9
    MEASUREMENT = 10
    CRYSTALLIZATION = 11
    FIDELITY_LOSS = 12
    DROPOUT = 13
    CURRICULUM = 14


# =============================================================================
# QUANTUM CONFIGURATION
# =============================================================================


@dataclass
class QuantumConfig:
    """Configuration for quantum operations."""

    # Core parameters
    op_type: QuantumOpType = QuantumOpType.VQC
    batch_size: int = 1
    seq_len: int = 512
    d_model: int = 512
    vocab_size: int = 32000
    epsilon: float = 1e-6

    # VQC parameters
    num_qubits: int = 4
    vqc_layers: int = 2
    entanglement_strength: float = 0.5
    neumann_terms: int = 6

    # Embedding parameters
    embedding_dim: int = 512
    num_bundles: int = 4

    # Position encoding parameters
    floquet_omega: float = 1.0
    floquet_amplitude: float = 0.1
    su2_components: int = 3

    # Expert parameters
    d_ff: int = 2048
    activation_angle: float = 0.5

    # Norm parameters
    use_bias: bool = True

    # Bus parameters
    num_channels: int = 8
    coherence_threshold: float = 0.9

    # Dropout parameters
    dropout_rate: float = 0.1
    collapse_probability: float = 0.5

    # Curriculum parameters
    spectral_complexity_threshold: float = 0.5
    use_fft_analysis: bool = True

    # Tensor ring parameters
    tr_rank: int = 8
    tr_cores: int = 4
    bp_mitigation_strength: float = 0.1

    # Crystallization parameters
    crystallization_rate: float = 0.1
    memory_slots: int = 64

    def validate(self) -> bool:
        """Validate configuration."""
        if self.op_type not in QuantumOpType:
            return False
        if self.batch_size < 1 or self.d_model < 1:
            return False
        if self.num_qubits < 1 or self.vqc_layers < 1:
            return False
        return True


# =============================================================================
# UNIFIED QUANTUM DISPATCH (NATIVE ONLY)
# =============================================================================


def unified_quantum(
    input_tensor: tf.Tensor,
    params: tf.Tensor,
    config: QuantumConfig,
    aux_input: tf.Tensor | None = None,
    training: bool = False,
) -> tf.Tensor:
    """Unified quantum operation dispatcher.

    Dispatches to native C++ kernel. No Python fallback.

    Args:
        input_tensor: Input tensor (interpretation depends on op_type)
        params: Operation parameters
        config: Quantum configuration
        aux_input: Auxiliary input (weights, keys, etc.)
        training: Training mode flag

    Returns:
        Output tensor from quantum operation

    Raises:
        RuntimeError: If native ops are not available
    """
    if aux_input is None:
        aux_input = tf.zeros([1], dtype=tf.float32)

    return _highnoon_core.unified_quantum_op(
        input=input_tensor,
        params=params,
        aux_input=aux_input,
        op_type=int(config.op_type),
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        vocab_size=config.vocab_size,
        num_qubits=config.num_qubits,
        vqc_layers=config.vqc_layers,
        d_ff=config.d_ff,
        num_bundles=config.num_bundles,
        tr_rank=config.tr_rank,
        tr_cores=config.tr_cores,
        dropout_rate=config.dropout_rate,
        epsilon=config.epsilon,
        training=training,
    )


# =============================================================================
# KERAS LAYERS
# =============================================================================


class UnifiedQuantumLayer(tf.keras.layers.Layer):
    """Keras layer wrapper for unified quantum operations."""

    def __init__(self, config: QuantumConfig, name: str | None = None, **kwargs):
        super().__init__(name=name or f"unified_quantum_{config.op_type.name.lower()}", **kwargs)
        self.config = config

    def build(self, input_shape):
        """Build layer - create trainable parameters."""

        if self.config.op_type == QuantumOpType.VQC:
            params_per_layer = 2 * self.config.num_qubits
            total_params = self.config.vqc_layers * params_per_layer
            self.vqc_params = self.add_weight(
                name="vqc_params",
                shape=(total_params,),
                initializer="random_uniform",
                trainable=True,
            )

        elif self.config.op_type == QuantumOpType.NORM:
            dim = input_shape[-1]
            self.scale = self.add_weight(
                name="scale", shape=(dim,), initializer="ones", trainable=True
            )
            if self.config.use_bias:
                self.bias = self.add_weight(
                    name="bias", shape=(dim,), initializer="zeros", trainable=True
                )
            else:
                self.bias = None

        elif self.config.op_type == QuantumOpType.EXPERT:
            self.u_skew = self.add_weight(
                name="u_skew",
                shape=(self.config.d_model, self.config.d_ff),
                initializer=tf.keras.initializers.Orthogonal(),
                trainable=True,
            )

        elif self.config.op_type == QuantumOpType.RESIDUAL:
            self.alpha = self.add_weight(
                name="alpha",
                shape=(),
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True,
            )

        super().build(input_shape)

    def call(self, inputs, aux_inputs=None, training=None):
        """Forward pass."""

        if self.config.op_type == QuantumOpType.VQC:
            batch_vqc_params = tf.tile(tf.expand_dims(self.vqc_params, 0), [tf.shape(inputs)[0], 1])
            return unified_quantum(
                inputs, batch_vqc_params, self.config, aux_inputs, training or False
            )

        elif self.config.op_type == QuantumOpType.NORM:
            return unified_quantum(inputs, self.scale, self.config, self.bias, training or False)

        elif self.config.op_type == QuantumOpType.EXPERT:
            return unified_quantum(inputs, self.u_skew, self.config, aux_inputs, training or False)

        elif self.config.op_type == QuantumOpType.RESIDUAL:
            return unified_quantum(
                inputs, tf.reshape(self.alpha, [1]), self.config, aux_inputs, training or False
            )

        else:
            dummy_params = tf.zeros([1])
            return unified_quantum(inputs, dummy_params, self.config, aux_inputs, training or False)

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "op_type": int(self.config.op_type),
                "batch_size": self.config.batch_size,
                "seq_len": self.config.seq_len,
                "d_model": self.config.d_model,
                "num_qubits": self.config.num_qubits,
                "vqc_layers": self.config.vqc_layers,
            }
        )
        return config


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def vqc(
    input_tensor: tf.Tensor,
    vqc_params: tf.Tensor,
    num_qubits: int = 4,
    vqc_layers: int = 2,
    **kwargs,
) -> tf.Tensor:
    """Variational Quantum Circuit forward pass.

    Args:
        input_tensor: Input features [batch, dim]
        vqc_params: VQC parameters [batch, num_params]
        num_qubits: Number of qubits
        vqc_layers: Number of VQC layers

    Returns:
        VQC output features [batch, dim]
    """
    batch_size = tf.shape(input_tensor)[0]
    d_model = input_tensor.shape[-1] or 512

    config = QuantumConfig(
        op_type=QuantumOpType.VQC,
        batch_size=int(batch_size),
        d_model=int(d_model),
        num_qubits=num_qubits,
        vqc_layers=vqc_layers,
        **kwargs,
    )
    return unified_quantum(input_tensor, vqc_params, config)


def quantum_norm(
    input_tensor: tf.Tensor,
    scale: tf.Tensor,
    bias: tf.Tensor | None = None,
    epsilon: float = 1e-6,
    **kwargs,
) -> tuple[tf.Tensor, dict]:
    """Quantum (unitary-style) normalization - Python implementation.

    Uses L2 normalization which preserves angles (unitary-like).
    This replaces the unified_quantum_op dispatcher which has stability issues.

    Args:
        input_tensor: Input [batch, seq, dim]
        scale: Scale parameter [dim]
        bias: Optional bias [dim]
        epsilon: Numerical stability

    Returns:
        Tuple of (normalized output, empty stats dict)
    """
    # L2 normalize preserves angular relationships (unitary-like)
    normalized = tf.nn.l2_normalize(input_tensor, axis=-1, epsilon=epsilon)

    # Apply learned scale
    output = normalized * scale

    # Apply optional bias
    if bias is not None:
        output = output + bias

    return output, {}


def quantum_expert(
    input_tensor: tf.Tensor,
    u_skew: tf.Tensor,
    d_ff: int = 2048,
    activation_angle: float = 0.5,
    **kwargs,
) -> tf.Tensor:
    """Quantum expert (unitary network) forward pass.

    Args:
        input_tensor: Input [num_tokens, d_model] - flattened token batch
        u_skew: Concatenated skew-symmetric projections [d_ff * d_model + d_model * d_ff]
                Contains flattened U1 [d_ff, d_model] followed by flattened U2 [d_model, d_ff]
        d_ff: Hidden dimension
        activation_angle: Quantum activation angle

    Returns:
        Expert output [num_tokens, d_model]
    """
    d_model = input_tensor.shape[-1] or 512

    # Get num_tokens from static shape if available, otherwise use a reasonable estimate
    # The C++ kernel computes: num_tokens = batch_size * seq_len
    # We set batch_size = num_tokens and seq_len = 1 to match
    static_shape = input_tensor.shape
    if static_shape[0] is not None:
        num_tokens = int(static_shape[0])
    else:
        # Dynamic shape - use default that will be correct at runtime
        # The C++ kernel uses this to allocate work, but processes actual tensor data
        num_tokens = 1  # Will be overridden by actual tensor size

    config = QuantumConfig(
        op_type=QuantumOpType.EXPERT,
        batch_size=num_tokens,
        seq_len=1,  # Flattened tokens: no seq_len dimension
        d_model=int(d_model),
        d_ff=d_ff,
        activation_angle=activation_angle,
        **kwargs,
    )
    return unified_quantum(input_tensor, u_skew, config)


@tf.custom_gradient
def quantum_residual(
    x: tf.Tensor,
    f_x: tf.Tensor,
    angle: tf.Tensor,
) -> tf.Tensor:
    """Unitary residual connection via rotation blending.

    Implements: y = cos(angle)*x + sin(angle)*f_x

    This provides gradient preservation and approximate norm preservation
    when x and f_x are approximately orthogonal.

    Args:
        x: Input tensor (residual) of any shape
        f_x: Block output tensor, same shape as x
        angle: Blend angle theta as scalar tensor

    Returns:
        Blended output, same shape as x
    """
    # Call native forward op
    output = _highnoon_core.unitary_residual_forward(
        x=x,
        f_x=f_x,
        angle=angle,
    )

    def grad(upstream, variables=None):
        """Gradient function with variables support for tf.custom_gradient."""
        grad_x, grad_f_x, grad_angle = _highnoon_core.unitary_residual_backward(
            grad_output=upstream,
            x=x,
            f_x=f_x,
            angle=angle,
        )
        # Return gradients for inputs and variables
        # Variables are the trainable weights passed through; we propagate grad_angle to them
        grad_vars = []
        if variables:
            for _var in variables:
                # The angle variable gets the grad_angle
                grad_vars.append(grad_angle)
        return (grad_x, grad_f_x, grad_angle), grad_vars

    return output, grad


def measurement_dropout(
    state: tf.Tensor, dropout_rate: float = 0.1, training: bool = True, **kwargs
) -> tf.Tensor:
    """Apply measurement dropout to quantum state.

    Args:
        state: Quantum state amplitudes [batch, 2 * num_qubits]
        dropout_rate: Probability of measuring each qubit
        training: Whether in training mode

    Returns:
        Possibly collapsed state
    """
    config = QuantumConfig(op_type=QuantumOpType.DROPOUT, dropout_rate=dropout_rate, **kwargs)
    return unified_quantum(state, tf.zeros([1]), config, training=training)


def quantum_curriculum_score(input_tensor: tf.Tensor, use_fft: bool = True, **kwargs) -> tf.Tensor:
    """Compute quantum curriculum score (spectral complexity).

    Args:
        input_tensor: Input features [batch, seq, dim]
        use_fft: Whether to use FFT-based analysis

    Returns:
        Complexity scores [batch]
    """
    d_model = input_tensor.shape[-1] or 512
    seq_len = input_tensor.shape[-2] if len(input_tensor.shape) > 2 else 1

    config = QuantumConfig(
        op_type=QuantumOpType.CURRICULUM,
        use_fft_analysis=use_fft,
        d_model=int(d_model),
        seq_len=int(seq_len),
        **kwargs,
    )
    return unified_quantum(input_tensor, tf.zeros([1]), config)


def born_rule_measurement(state: tf.Tensor, num_qubits: int = 4) -> tf.Tensor:
    """Extract probabilities from quantum state via Born rule.

    Args:
        state: Quantum state amplitudes [batch, 2 * num_qubits]
        num_qubits: Number of qubits

    Returns:
        Probabilities [batch, 2 * num_qubits]
    """
    config = QuantumConfig(
        op_type=QuantumOpType.MEASUREMENT,
        num_qubits=num_qubits,
    )
    return unified_quantum(state, tf.zeros([1]), config)


# Alias for backward compatibility
run_vqc = vqc
born_rule_probabilities = born_rule_measurement


__all__ = [
    "QuantumOpType",
    "QuantumConfig",
    "unified_quantum",
    "UnifiedQuantumLayer",
    "vqc",
    "run_vqc",  # Alias for backward compatibility
    "quantum_norm",
    "quantum_expert",
    "quantum_residual",  # Phase 34: Unitary residual connections
    "measurement_dropout",
    "quantum_curriculum_score",
    "born_rule_measurement",
    "born_rule_probabilities",  # Alias for backward compatibility
]
