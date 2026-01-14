# highnoon/_native/ops/alphaqubit_ops.py
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

"""Python wrappers for AlphaQubit-2 Decoder C++ operations (Phase 61).

AlphaQubit-2 is a neural syndrome decoder for quantum error correction,
inspired by DeepMind's AlphaFold architecture.

NO PYTHON FALLBACKS: RuntimeError raised if native ops unavailable.

Ops:
    - alphaqubit_decode: Neural syndrome decoding for error classification
"""

import logging

import tensorflow as tf

from highnoon import config
from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

_module = None
_available = False


def _load_ops():
    """Load ops from consolidated binary."""
    global _module, _available
    if _module is not None:
        return _available

    try:
        lib_path = resolve_op_library(__file__, "_highnoon_core.so")
        if lib_path is None:
            raise RuntimeError("Could not find _highnoon_core.so")
        _module = tf.load_op_library(lib_path)
        _available = True
        logger.info(f"AlphaQubit ops loaded from {lib_path}")
    except Exception as e:
        _available = False
        logger.warning(f"Failed to load AlphaQubit ops: {e}")
        raise RuntimeError(
            "AlphaQubit native ops not available. Run ./build_secure.sh to compile."
        ) from e
    return _available


def ops_available() -> bool:
    """Check if native ops are available."""
    try:
        _load_ops()
        return _available
    except RuntimeError:
        return False


# =============================================================================
# Phase 61: AlphaQubit-2 Neural Syndrome Decoder
# =============================================================================


def alphaqubit_decode(
    syndrome: tf.Tensor,
    embed_weights: tf.Tensor,
    attention_weights: tf.Tensor,
    output_weights: tf.Tensor,
    syndrome_dim: int | None = None,
    hidden_dim: int = 128,
    num_layers: int | None = None,
) -> tf.Tensor:
    """AlphaQubit-2 neural syndrome decoder for error classification.

    Uses a transformer-like architecture to decode error syndromes
    from quantum circuits into error probability distributions.

    Args:
        syndrome: Error syndrome tensor [batch, syndrome_dim].
        embed_weights: Embedding weights [syndrome_dim, hidden_dim].
        attention_weights: Multi-head attention weights [num_layers, ...].
        output_weights: Output projection weights [hidden_dim, 4].
        syndrome_dim: Syndrome dimension (default 64).
        hidden_dim: Hidden layer dimension (default 128).
        num_layers: Number of attention layers (default from config).

    Returns:
        Error probability distribution [batch, 4] for:
        - Index 0: No error (I)
        - Index 1: X error
        - Index 2: Y error
        - Index 3: Z error

    Raises:
        RuntimeError: If native op not available.

    Example:
        >>> # During error correction
        >>> syndrome = measure_syndrome(quantum_state)
        >>> error_probs = alphaqubit_decode(syndrome, embed_w, attn_w, out_w)
        >>> predicted_error = tf.argmax(error_probs, axis=-1)
    """
    if not config.USE_ALPHAQUBIT_DECODER:
        # Return uniform distribution if disabled
        batch_size = tf.shape(syndrome)[0]
        return tf.ones([batch_size, 4], dtype=tf.float32) * 0.25

    _load_ops()
    syndrome_dim = syndrome_dim or 64
    num_layers = num_layers or config.ALPHAQUBIT_NUM_LAYERS

    return _module.alpha_qubit_decode(
        syndrome,
        embed_weights,
        attention_weights,
        output_weights,
        syndrome_dim=syndrome_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )


def create_alphaqubit_weights(
    syndrome_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int | None = None,
) -> tuple[tf.Variable, tf.Variable, tf.Variable]:
    """Create initialized AlphaQubit decoder weights.

    Args:
        syndrome_dim: Input syndrome dimension.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of attention layers (default from config).

    Returns:
        Tuple of (embed_weights, attention_weights, output_weights).
    """
    num_layers = num_layers or config.ALPHAQUBIT_NUM_LAYERS

    embed_w = tf.Variable(
        tf.random.normal([syndrome_dim, hidden_dim], stddev=0.02),
        trainable=True,
        name="alphaqubit_embed",
    )

    # Attention weights: [num_layers, 3, hidden_dim, hidden_dim] for Q, K, V
    attn_w = tf.Variable(
        tf.random.normal([num_layers, 3, hidden_dim, hidden_dim], stddev=0.02),
        trainable=True,
        name="alphaqubit_attention",
    )

    out_w = tf.Variable(
        tf.random.normal([hidden_dim, 4], stddev=0.02),
        trainable=True,
        name="alphaqubit_output",
    )

    return embed_w, attn_w, out_w


# =============================================================================
# S11: AlphaQubitCorrect - Unified Quantum Layer Correction
# =============================================================================


def alphaqubit_correct(
    quantum_output: tf.Tensor,
    qkv_weights: tf.Tensor,
    proj_weights: tf.Tensor,
    corr_w1: tf.Tensor,
    corr_w2: tf.Tensor,
    gate_w: tf.Tensor,
    gate_b: tf.Tensor,
    feature_dim: int | None = None,
    hidden_dim: int = 64,
    num_attn_layers: int | None = None,
    num_heads: int = 4,
) -> tf.Tensor:
    """S11: Apply AlphaQubit-style error correction to quantum layer outputs.

    Uses syndrome detection (self-attention) and learned correction with
    gated residual connection. This is the general-purpose correction op
    for any quantum layer output (VQC, QASA, Q-SSM, QMamba, etc.).

    Args:
        quantum_output: Output from quantum layer [batch, feature_dim].
        qkv_weights: Q/K/V projection weights [num_layers, 3, dim, hidden].
        proj_weights: Output projection weights [num_layers, hidden, dim].
        corr_w1: Correction dense layer 1 [dim, hidden].
        corr_w2: Correction dense layer 2 [hidden, dim].
        gate_w: Gate weights [dim, dim].
        gate_b: Gate bias [dim].
        feature_dim: Feature dimension (inferred from input if None).
        hidden_dim: Hidden layer dimension (default 64).
        num_attn_layers: Number of attention layers (default from config).
        num_heads: Number of attention heads (default 4).

    Returns:
        Error-corrected output [batch, feature_dim].

    Raises:
        RuntimeError: If native op not available.

    Example:
        >>> # Apply correction to VQC output
        >>> corrected = alphaqubit_correct(vqc_output, qkv_w, proj_w, ...)
    """
    if not config.USE_UNIFIED_ALPHAQUBIT:
        return quantum_output

    _load_ops()
    feature_dim = feature_dim or int(quantum_output.shape[-1])
    num_attn_layers = num_attn_layers or getattr(config, "ALPHAQUBIT_NUM_LAYERS", 2)

    return _module.alpha_qubit_correct(
        quantum_output,
        qkv_weights,
        proj_weights,
        corr_w1,
        corr_w2,
        gate_w,
        gate_b,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_attn_layers=num_attn_layers,
        num_heads=num_heads,
    )


def create_alphaqubit_correct_weights(
    feature_dim: int = 256,
    hidden_dim: int = 64,
    num_attn_layers: int | None = None,
) -> dict[str, tf.Variable]:
    """Create initialized weights for AlphaQubitCorrect op.

    Args:
        feature_dim: Input/output feature dimension.
        hidden_dim: Hidden layer dimension.
        num_attn_layers: Number of attention layers (default from config).

    Returns:
        Dictionary of weight tensors:
        - qkv_weights: [num_layers, 3, dim, hidden]
        - proj_weights: [num_layers, hidden, dim]
        - corr_w1: [dim, hidden]
        - corr_w2: [hidden, dim]
        - gate_w: [dim, dim]
        - gate_b: [dim]
    """
    num_attn_layers = num_attn_layers or getattr(config, "ALPHAQUBIT_NUM_LAYERS", 2)
    stddev = 0.02

    weights = {
        "qkv_weights": tf.Variable(
            tf.random.normal([num_attn_layers, 3, feature_dim, hidden_dim], stddev=stddev),
            trainable=True,
            name="alphaqubit_correct_qkv",
        ),
        "proj_weights": tf.Variable(
            tf.random.normal([num_attn_layers, hidden_dim, feature_dim], stddev=stddev),
            trainable=True,
            name="alphaqubit_correct_proj",
        ),
        "corr_w1": tf.Variable(
            tf.random.normal([feature_dim, hidden_dim], stddev=stddev),
            trainable=True,
            name="alphaqubit_correct_corr1",
        ),
        "corr_w2": tf.Variable(
            tf.random.normal([hidden_dim, feature_dim], stddev=stddev),
            trainable=True,
            name="alphaqubit_correct_corr2",
        ),
        "gate_w": tf.Variable(
            tf.random.normal([feature_dim, feature_dim], stddev=stddev),
            trainable=True,
            name="alphaqubit_correct_gate_w",
        ),
        "gate_b": tf.Variable(
            tf.zeros([feature_dim]),
            trainable=True,
            name="alphaqubit_correct_gate_b",
        ),
    }

    return weights


__all__ = [
    "alphaqubit_decode",
    "alphaqubit_correct",
    "create_alphaqubit_weights",
    "create_alphaqubit_correct_weights",
    "ops_available",
]
