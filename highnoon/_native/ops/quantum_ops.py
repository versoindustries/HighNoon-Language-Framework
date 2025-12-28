# highnoon/_native/ops/quantum_ops.py
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

"""Python wrappers for quantum architecture ops (Phases 26-36).

NO PYTHON FALLBACKS: If native ops cannot be loaded, RuntimeError is raised.

Ops:
    - UnitaryResidualForward/Backward: Phase 34 rotation blend residuals
    - UnitaryNormForward/Backward: Phase 30 Stiefel normalization
    - UnitaryExpertForward/Backward: Phase 29 unitary experts
    - QuantumEmbeddingForward/Backward: Phase 26 holographic embeddings
    - FloquetPositionEncodingForward/Backward: Phase 27 time-crystal position
    - QuantumLMHeadForward/Backward: Phase 33 VQC output
    - GroverGuidedQSG: Phase 32 amplitude amplification for QSG
"""

import logging

import tensorflow as tf

from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

# Module-level op handle
_quantum_ops_module = None
_quantum_ops_available = False


def _load_quantum_ops():
    """Load quantum ops from consolidated binary."""
    global _quantum_ops_module, _quantum_ops_available

    if _quantum_ops_module is not None:
        return _quantum_ops_available

    try:
        lib_path = resolve_op_library(__file__, "_highnoon_core.so")
        if lib_path is None:
            raise RuntimeError("Could not find _highnoon_core.so")

        _quantum_ops_module = tf.load_op_library(lib_path)
        _quantum_ops_available = True
        logger.info(f"Quantum ops loaded from {lib_path}")

    except Exception as e:
        _quantum_ops_available = False
        logger.warning(f"Failed to load quantum ops: {e}")
        raise RuntimeError(
            "Quantum native ops not available. " "Run ./build_secure.sh to compile."
        ) from e

    return _quantum_ops_available


def quantum_ops_available() -> bool:
    """Check if quantum native ops are available."""
    try:
        _load_quantum_ops()
        return _quantum_ops_available
    except RuntimeError:
        return False


# =============================================================================
# Phase 34: Unitary Residual Connections
# =============================================================================


def unitary_residual_forward(
    x: tf.Tensor,
    f_x: tf.Tensor,
    angle: tf.Tensor,
) -> tf.Tensor:
    """Unitary residual: y = cos(angle)*x + sin(angle)*f_x.

    Args:
        x: Input tensor (identity path).
        f_x: Block output tensor (transform path).
        angle: Blend angle (scalar).

    Returns:
        Blended output tensor.
    """
    _load_quantum_ops()
    return _quantum_ops_module.unitary_residual_forward(x, f_x, angle)


def unitary_residual_backward(
    grad_output: tf.Tensor,
    x: tf.Tensor,
    f_x: tf.Tensor,
    angle: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Backward pass for unitary residual.

    Returns:
        (grad_x, grad_f_x, grad_angle)
    """
    _load_quantum_ops()
    return _quantum_ops_module.unitary_residual_backward(grad_output, x, f_x, angle)


# Register gradient for UnitaryResidualForward C++ op
@tf.RegisterGradient("UnitaryResidualForward")
def _unitary_residual_forward_grad(op, grad):
    """Gradient registration for UnitaryResidualForward.

    Connects the C++ forward op to its backward op for automatic differentiation.
    """
    x = op.inputs[0]
    f_x = op.inputs[1]
    angle = op.inputs[2]

    # Load ops and call backward
    _load_quantum_ops()
    grad_x, grad_f_x, grad_angle = _quantum_ops_module.unitary_residual_backward(
        grad, x, f_x, angle
    )

    return [grad_x, grad_f_x, grad_angle]


# =============================================================================
# Phase 30: Quantum Normalization
# =============================================================================


def unitary_norm_forward(
    input_tensor: tf.Tensor,
    scale: tf.Tensor,
    bias: tf.Tensor,
    eps: float = 1e-6,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Unitary norm: x_norm = (x / ||x||) * scale + bias.

    Args:
        input_tensor: Input [batch, seq_len, dim].
        scale: Scale parameter [dim].
        bias: Bias parameter [dim].
        eps: Numerical stability epsilon.

    Returns:
        (normalized_output, cached_norms)
    """
    _load_quantum_ops()
    return _quantum_ops_module.unitary_norm_forward(input_tensor, scale, bias, eps=eps)


def rms_norm_forward(
    input_tensor: tf.Tensor,
    scale: tf.Tensor,
    eps: float = 1e-6,
) -> tf.Tensor:
    """RMS normalization variant.

    Args:
        input_tensor: Input tensor.
        scale: Scale parameter.
        eps: Numerical stability epsilon.

    Returns:
        Normalized output.
    """
    _load_quantum_ops()
    return _quantum_ops_module.rms_norm_forward(input_tensor, scale, eps=eps)


def unitary_norm_backward(
    grad_output: tf.Tensor,
    input_tensor: tf.Tensor,
    scale: tf.Tensor,
    norms: tf.Tensor,
    eps: float = 1e-6,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Backward pass for unitary normalization.

    Args:
        grad_output: Gradient with respect to output.
        input_tensor: Original input tensor.
        scale: Scale parameter.
        norms: Cached L2 norms from forward pass.
        eps: Numerical stability epsilon.

    Returns:
        (grad_input, grad_scale, grad_bias)
    """
    _load_quantum_ops()
    return _quantum_ops_module.unitary_norm_backward(
        grad_output, input_tensor, scale, norms, eps=eps
    )


# Register gradient for UnitaryNormForward C++ op
@tf.RegisterGradient("UnitaryNormForward")
def _unitary_norm_forward_grad(op, grad_output, grad_norms):
    """Gradient registration for UnitaryNormForward.

    Connects the C++ forward op to its backward op for automatic differentiation.
    The forward op outputs (normalized_output, norms) so we receive gradients for both.
    We only need to propagate grad_output; grad_norms is typically None/zeros.
    """
    input_tensor = op.inputs[0]
    scale = op.inputs[1]
    norms = op.outputs[1]  # Cached norms from forward pass
    eps = op.get_attr("eps")

    # Load ops and call backward
    _load_quantum_ops()
    grad_input, grad_scale, grad_bias = _quantum_ops_module.unitary_norm_backward(
        grad_output, input_tensor, scale, norms, eps=eps
    )

    return [grad_input, grad_scale, grad_bias]


# =============================================================================
# Phase 29: Unitary Expert Networks
# =============================================================================


def unitary_expert_forward(
    input_tensor: tf.Tensor,
    u1_weights: tf.Tensor,
    u2_weights: tf.Tensor,
    activation_angle: tf.Tensor,
    d_ff: int,
    neumann_terms: int = 6,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Unitary expert FFN: x -> U1*x -> quantum_activation -> U2*x.

    Args:
        input_tensor: Input [num_tokens, d_model].
        u1_weights: First projection [d_ff, d_model].
        u2_weights: Second projection [d_model, d_ff].
        activation_angle: Quantum activation angle.
        d_ff: Feedforward dimension.
        neumann_terms: Cayley approximation terms.

    Returns:
        (output, hidden_cache)
    """
    _load_quantum_ops()
    return _quantum_ops_module.unitary_expert_forward(
        input_tensor,
        u1_weights,
        u2_weights,
        activation_angle,
        d_ff=d_ff,
        neumann_terms=neumann_terms,
    )


def quantum_activation(
    input_tensor: tf.Tensor,
    angle: tf.Tensor,
) -> tf.Tensor:
    """Quantum activation: parametric rotation in pairs.

    Args:
        input_tensor: Input with even last dimension.
        angle: Rotation angle.

    Returns:
        Rotated output.
    """
    _load_quantum_ops()
    return _quantum_ops_module.quantum_activation(input_tensor, angle)


def unitary_expert_backward(
    grad_output: tf.Tensor,
    input_tensor: tf.Tensor,
    u1_weights: tf.Tensor,
    u2_weights: tf.Tensor,
    activation_angle: tf.Tensor,
    hidden_cache: tf.Tensor,
    d_ff: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Backward pass for unitary expert.

    Args:
        grad_output: Gradient with respect to output.
        input_tensor: Original input tensor.
        u1_weights: First projection weights.
        u2_weights: Second projection weights.
        activation_angle: Quantum activation angle.
        hidden_cache: Cached hidden activations from forward.
        d_ff: Feedforward dimension.

    Returns:
        (grad_input, grad_u1, grad_u2, grad_angle)
    """
    _load_quantum_ops()
    return _quantum_ops_module.unitary_expert_backward(
        grad_output, input_tensor, u1_weights, u2_weights, activation_angle, hidden_cache, d_ff=d_ff
    )


# Register gradient for UnitaryExpertForward C++ op
@tf.RegisterGradient("UnitaryExpertForward")
def _unitary_expert_forward_grad(op, grad_output, grad_hidden_cache):
    """Gradient registration for UnitaryExpertForward.

    Forward op outputs (output, hidden_cache) so we receive gradients for both.
    """
    input_tensor = op.inputs[0]
    u1_weights = op.inputs[1]
    u2_weights = op.inputs[2]
    activation_angle = op.inputs[3]
    hidden_cache = op.outputs[1]  # Cached from forward pass
    d_ff = op.get_attr("d_ff")

    _load_quantum_ops()
    grad_input, grad_u1, grad_u2, grad_angle = _quantum_ops_module.unitary_expert_backward(
        grad_output, input_tensor, u1_weights, u2_weights, activation_angle, hidden_cache, d_ff=d_ff
    )

    return [grad_input, grad_u1, grad_u2, grad_angle]


# =============================================================================
# Phase 26: Quantum Embedding
# =============================================================================


def quantum_embedding_forward(
    token_ids: tf.Tensor,
    holographic_store: tf.Tensor,
    token_keys: tf.Tensor,
    vocab_size: int,
    dim: int,
    num_bundles: int = 4,
) -> tf.Tensor:
    """Holographic embedding lookup via FFT unbinding.

    Args:
        token_ids: Token IDs [batch, seq_len].
        holographic_store: Bundled representations [num_bundles, dim].
        token_keys: Token keys [vocab_size, dim].
        vocab_size: Vocabulary size.
        dim: Embedding dimension.
        num_bundles: Number of holographic bundles.

    Returns:
        Embeddings [batch, seq_len, dim].
    """
    _load_quantum_ops()
    return _quantum_ops_module.quantum_embedding_forward(
        token_ids,
        holographic_store,
        token_keys,
        vocab_size=vocab_size,
        dim=dim,
        num_bundles=num_bundles,
    )


def haar_random_key_init(
    shape_tensor: tf.Tensor,
    seed: int = 42,
) -> tf.Tensor:
    """Initialize Haar-random orthogonal token keys.

    Args:
        shape_tensor: Shape [vocab_size, dim].
        seed: Random seed.

    Returns:
        Keys [vocab_size, dim].
    """
    _load_quantum_ops()
    return _quantum_ops_module.haar_random_key_init(shape_tensor, seed=seed)


def quantum_embedding_backward(
    grad_output: tf.Tensor,
    token_ids: tf.Tensor,
    token_keys: tf.Tensor,
    vocab_size: int,
    dim: int,
    num_bundles: int = 4,
) -> tf.Tensor:
    """Backward pass for quantum embedding.

    Returns:
        grad_store: Gradient w.r.t. holographic store
    """
    _load_quantum_ops()
    return _quantum_ops_module.quantum_embedding_backward(
        grad_output, token_ids, token_keys, vocab_size=vocab_size, dim=dim, num_bundles=num_bundles
    )


# Register gradient for QuantumEmbeddingForward C++ op
@tf.RegisterGradient("QuantumEmbeddingForward")
def _quantum_embedding_forward_grad(op, grad_output):
    """Gradient registration for QuantumEmbeddingForward."""
    token_ids = op.inputs[0]
    token_keys = op.inputs[2]
    vocab_size = op.get_attr("vocab_size")
    dim = op.get_attr("dim")
    num_bundles = op.get_attr("num_bundles")

    _load_quantum_ops()
    grad_store = _quantum_ops_module.quantum_embedding_backward(
        grad_output, token_ids, token_keys, vocab_size=vocab_size, dim=dim, num_bundles=num_bundles
    )

    # No gradient for token_ids (discrete), return grad_store for holographic_store, None for token_keys
    return [None, grad_store, None]


# =============================================================================
# Phase 27: Floquet Position Encoding
# =============================================================================


def floquet_position_encoding_forward(
    base_embedding: tf.Tensor,
    floquet_angles: tf.Tensor,
    max_position: int = 100000,
) -> tf.Tensor:
    """Apply Floquet time-crystal position encoding.

    Args:
        base_embedding: Input [batch, seq_len, dim].
        floquet_angles: SU(2) angles [dim/2, 3].
        max_position: Maximum position scaling.

    Returns:
        Position-encoded embedding.
    """
    _load_quantum_ops()
    return _quantum_ops_module.floquet_position_encoding_forward(
        base_embedding, floquet_angles, max_position=max_position
    )


def init_floquet_angles(
    num_qubits: int,
    base_frequency: float = 10000.0,
) -> tf.Tensor:
    """Initialize Floquet angles with frequency scaling.

    Args:
        num_qubits: Number of qubit pairs (dim/2).
        base_frequency: Base frequency.

    Returns:
        Angles [num_qubits, 3].
    """
    _load_quantum_ops()
    return _quantum_ops_module.init_floquet_angles(
        tf.constant(num_qubits, dtype=tf.int32), base_frequency=base_frequency
    )


def floquet_position_encoding_backward(
    grad_output: tf.Tensor,
    base_embedding: tf.Tensor,
    floquet_angles: tf.Tensor,
) -> tf.Tensor:
    """Backward pass for Floquet position encoding.

    Returns:
        grad_angles: Gradient w.r.t. floquet angles
    """
    _load_quantum_ops()
    return _quantum_ops_module.floquet_position_encoding_backward(
        grad_output, base_embedding, floquet_angles
    )


# Register gradient for FloquetPositionEncodingForward C++ op
@tf.RegisterGradient("FloquetPositionEncodingForward")
def _floquet_position_encoding_forward_grad(op, grad_output):
    """Gradient registration for FloquetPositionEncodingForward."""
    base_embedding = op.inputs[0]
    floquet_angles = op.inputs[1]

    _load_quantum_ops()
    grad_angles = _quantum_ops_module.floquet_position_encoding_backward(
        grad_output, base_embedding, floquet_angles
    )

    # No gradient for base_embedding (inputs flow through), only for learnable angles
    return [None, grad_angles]


# =============================================================================
# Phase 33: Quantum LM Head
# =============================================================================


def quantum_lm_head_forward(
    hidden_states: tf.Tensor,
    rotation_params: tf.Tensor,
    token_weights: tf.Tensor,
    vocab_size: int,
    num_layers: int = 2,
) -> tf.Tensor:
    """VQC-based LM head with Born rule output.

    Args:
        hidden_states: Hidden states [batch, seq_len, d_model].
        rotation_params: VQC parameters [num_layers, d_model/2, 2].
        token_weights: Token-qubit weights [vocab_size, d_model/2].
        vocab_size: Vocabulary size.
        num_layers: VQC depth.

    Returns:
        Logits [batch, seq_len, vocab_size].
    """
    _load_quantum_ops()
    return _quantum_ops_module.quantum_lm_head_forward(
        hidden_states, rotation_params, token_weights, vocab_size=vocab_size, num_layers=num_layers
    )


def quantum_lm_head_backward(
    grad_logits: tf.Tensor,
    hidden_states: tf.Tensor,
    rotation_params: tf.Tensor,
    token_weights: tf.Tensor,
    vocab_size: int,
    num_layers: int = 2,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Backward pass for quantum LM head.

    Returns:
        (grad_rotation, grad_token_weights)
    """
    _load_quantum_ops()
    return _quantum_ops_module.quantum_lm_head_backward(
        grad_logits,
        hidden_states,
        rotation_params,
        token_weights,
        vocab_size=vocab_size,
        num_layers=num_layers,
    )


# Register gradient for QuantumLMHeadForward C++ op
@tf.RegisterGradient("QuantumLMHeadForward")
def _quantum_lm_head_forward_grad(op, grad_logits):
    """Gradient registration for QuantumLMHeadForward."""
    hidden_states = op.inputs[0]
    rotation_params = op.inputs[1]
    token_weights = op.inputs[2]
    vocab_size = op.get_attr("vocab_size")
    num_layers = op.get_attr("num_layers")

    _load_quantum_ops()
    grad_rotation, grad_token_weights = _quantum_ops_module.quantum_lm_head_backward(
        grad_logits,
        hidden_states,
        rotation_params,
        token_weights,
        vocab_size=vocab_size,
        num_layers=num_layers,
    )

    # No gradient for hidden_states (computed elsewhere), only for learnable params
    return [None, grad_rotation, grad_token_weights]


# =============================================================================
# Phase 32: Grover QSG Enhancement
# =============================================================================


def grover_guided_qsg(
    candidate_logits: tf.Tensor,
    quality_scores: tf.Tensor,
    quality_threshold: float = 0.7,
    grover_iterations: int = -1,
) -> tf.Tensor:
    """Grover-guided QSG for amplitude amplification.

    Args:
        candidate_logits: Candidates [batch, num_candidates, seq_len, vocab].
        quality_scores: Quality per candidate [batch, num_candidates].
        quality_threshold: Oracle threshold.
        grover_iterations: Iterations (-1 for auto).

    Returns:
        Selected logits [batch, seq_len, vocab].
    """
    _load_quantum_ops()
    return _quantum_ops_module.grover_guided_qsg(
        candidate_logits,
        quality_scores,
        quality_threshold=quality_threshold,
        grover_iterations=grover_iterations,
    )


def grover_single_iteration(
    amplitudes: tf.Tensor,
    quality_scores: tf.Tensor,
    quality_threshold: float = 0.7,
) -> tf.Tensor:
    """Single Grover iteration for fine-grained control.

    Args:
        amplitudes: State amplitudes [num_states].
        quality_scores: Quality scores [num_states].
        quality_threshold: Oracle threshold.

    Returns:
        Updated amplitudes.
    """
    _load_quantum_ops()
    return _quantum_ops_module.grover_single_iteration(
        amplitudes, quality_scores, quality_threshold=quality_threshold
    )


# =============================================================================
# Phase 51: Born Rule Loss
# =============================================================================


def born_rule_loss(
    logits: tf.Tensor,
    targets: tf.Tensor,
    temperature: float = 1.0,
    use_qfim: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute Born rule loss with QFIM gradients.

    Enforces |ψ|² = p normalization for VQC outputs. The loss measures
    how well the model's output amplitudes follow the Born rule when
    interpreted as quantum probability amplitudes.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size].
        targets: Target token IDs [batch, seq_len].
        temperature: Born rule temperature for sharpness control.
        use_qfim: If True, use Quantum Fisher Information Matrix for
            improved gradient flow.

    Returns:
        Tuple of (loss_per_sample [batch], grad_logits [batch, seq, vocab]).

    Raises:
        RuntimeError: If native op not available.

    Example:
        >>> logits = model(input_ids, training=True)
        >>> loss, grad = born_rule_loss(logits, targets)
        >>> total_loss = tf.reduce_mean(loss) * config.QULS_BORN_RULE_WEIGHT
    """
    _load_quantum_ops()
    return _quantum_ops_module.born_rule_loss(
        logits, targets, temperature=temperature, use_qfim=use_qfim
    )


def born_rule_loss_available() -> bool:
    """Check if Born rule loss op is available."""
    try:
        _load_quantum_ops()
        return hasattr(_quantum_ops_module, "born_rule_loss")
    except RuntimeError:
        return False


# =============================================================================
# Phase 52: Quantum Fidelity Loss
# =============================================================================


def quantum_fidelity_loss(
    pred_states: tf.Tensor,
    true_states: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute quantum fidelity loss between predicted and true states.

    Implements trace fidelity F(p,q) = (Σ√pᵢ√qᵢ)² for comparing
    quantum state distributions. Useful for training models to
    match target probability distributions in a quantum-aware manner.

    Args:
        pred_states: Predicted probability distributions [batch, dim].
        true_states: Target probability distributions [batch, dim].

    Returns:
        Tuple of (loss_per_sample [batch], grad_pred [batch, dim]).

    Raises:
        RuntimeError: If native op not available.

    Example:
        >>> pred = tf.nn.softmax(logits[:, -1, :])  # Last token probs
        >>> target = tf.nn.softmax(target_logits[:, -1, :])
        >>> loss, grad = quantum_fidelity_loss(pred, target)
        >>> total_loss = tf.reduce_mean(loss) * config.QULS_FIDELITY_WEIGHT
    """
    _load_quantum_ops()
    return _quantum_ops_module.quantum_fidelity_loss(pred_states, true_states)


def quantum_fidelity_loss_available() -> bool:
    """Check if quantum fidelity loss op is available."""
    try:
        _load_quantum_ops()
        return hasattr(_quantum_ops_module, "quantum_fidelity_loss")
    except RuntimeError:
        return False

