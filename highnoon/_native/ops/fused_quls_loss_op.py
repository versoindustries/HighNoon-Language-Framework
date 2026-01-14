# highnoon/_native/ops/fused_quls_loss_op.py
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

"""Python wrapper for Fused QULS Loss native operations.

Provides optimized C++ implementations of the Quantum Unified Loss System
with automatic fallback to Python implementations when native ops are unavailable.

Example:
    >>> from highnoon._native.ops.fused_quls_loss_op import fused_quls_loss
    >>> total_loss, metrics = fused_quls_loss(logits, labels)
"""

from __future__ import annotations

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

# =============================================================================
# Load Native Ops
# =============================================================================

_fused_quls_loss_forward = None
_fused_quls_loss_backward = None
_NATIVE_OPS_AVAILABLE = False

try:
    from highnoon._native.ops.lib_loader import load_op_library

    _lib = load_op_library()
    if _lib is not None:
        _fused_quls_loss_forward = _lib.fused_quls_loss_forward
        _fused_quls_loss_backward = _lib.fused_quls_loss_backward
        _NATIVE_OPS_AVAILABLE = True
        logger.debug("[FusedQULSLoss] Native ops loaded successfully")
except Exception as e:
    logger.debug(f"[FusedQULSLoss] Native ops not available: {e}")


def is_native_available() -> bool:
    """Check if native QULS loss ops are available.

    Returns:
        True if C++ native ops are loaded and available.
    """
    return _NATIVE_OPS_AVAILABLE


# =============================================================================
# Python Fallback Implementation
# =============================================================================


def _python_quls_loss_forward(
    logits: tf.Tensor,
    labels: tf.Tensor,
    ce_weight: float = 1.0,
    fidelity_weight: float = 0.01,
    entropy_weight: float = 0.01,
    label_smoothing: float = 0.1,
    target_entropy: float = 0.5,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Python fallback for QULS forward pass.

    Args:
        logits: Input logits [batch, vocab_size].
        labels: Target label indices [batch].
        ce_weight: Cross-entropy weight.
        fidelity_weight: Fidelity loss weight.
        entropy_weight: Entropy regularization weight.
        label_smoothing: Label smoothing factor.
        target_entropy: Target entropy value.

    Returns:
        Tuple of (total_loss, ce_loss, fidelity_loss, entropy_loss).
    """
    epsilon = 1e-10
    vocab_size = tf.shape(logits)[-1]
    vocab_size_f = tf.cast(vocab_size, tf.float32)

    # Compute softmax probabilities
    probs = tf.nn.softmax(logits, axis=-1)
    probs_safe = tf.maximum(probs, epsilon)

    # Cross-entropy with label smoothing
    # Cast labels to int if needed
    labels_int = tf.cast(labels, tf.int64) if labels.dtype in (tf.float32, tf.float64) else labels
    one_hot = tf.one_hot(labels_int, depth=vocab_size, dtype=tf.float32)
    smoothed_labels = one_hot * (1.0 - label_smoothing) + label_smoothing / vocab_size_f

    log_probs = tf.math.log(probs_safe)
    ce_per_sample = -tf.reduce_sum(smoothed_labels * log_probs, axis=-1)
    ce_loss = tf.reduce_mean(ce_per_sample)

    # Fidelity loss: 1 - p_target (for one-hot target)
    target_probs = tf.reduce_sum(probs * one_hot, axis=-1)
    fidelity_loss = tf.reduce_mean(1.0 - target_probs)

    # Entropy: -sum(p * log(p))
    entropy_per_sample = -tf.reduce_sum(probs * log_probs, axis=-1)
    avg_entropy = tf.reduce_mean(entropy_per_sample)

    # Normalize entropy to [0, 1]
    max_entropy = tf.math.log(vocab_size_f)
    normalized_entropy = avg_entropy / (max_entropy + epsilon)

    # Entropy loss: deviation from target
    entropy_loss = tf.square(normalized_entropy - target_entropy)

    # Total weighted loss
    total_loss = (
        ce_weight * ce_loss + fidelity_weight * fidelity_loss + entropy_weight * entropy_loss
    )

    return total_loss, ce_loss, fidelity_loss, entropy_loss


def _python_quls_loss_backward(
    logits: tf.Tensor,
    labels: tf.Tensor,
    ce_weight: float = 1.0,
    fidelity_weight: float = 0.01,
    label_smoothing: float = 0.1,
) -> tf.Tensor:
    """Python fallback for QULS backward pass.

    Args:
        logits: Input logits [batch, vocab_size].
        labels: Target label indices [batch].
        ce_weight: Cross-entropy weight.
        fidelity_weight: Fidelity loss weight.
        label_smoothing: Label smoothing factor.

    Returns:
        Gradient w.r.t. logits [batch, vocab_size].
    """
    vocab_size = tf.shape(logits)[-1]
    vocab_size_f = tf.cast(vocab_size, tf.float32)
    batch_size_f = tf.cast(tf.shape(logits)[0], tf.float32)

    # Compute softmax probabilities
    probs = tf.nn.softmax(logits, axis=-1)

    # Cast labels to int if needed
    labels_int = tf.cast(labels, tf.int64) if labels.dtype in (tf.float32, tf.float64) else labels
    one_hot = tf.one_hot(labels_int, depth=vocab_size, dtype=tf.float32)

    # Smoothed target
    smoothed_labels = one_hot * (1.0 - label_smoothing) + label_smoothing / vocab_size_f

    # CE gradient: probs - smoothed_labels
    ce_grad = ce_weight * (probs - smoothed_labels) / batch_size_f

    # Fidelity gradient
    target_probs = tf.reduce_sum(probs * one_hot, axis=-1, keepdims=True)

    # d(1-p_target)/d_logit_i = -d(p_target)/d_logit_i
    # For softmax: d(p_j)/d(logit_i) = p_j * (delta_ij - p_i)
    # So for target j: d(p_j)/d(logit_i) = p_j * (delta_ij - p_i)
    # At i=j: p_j * (1 - p_j)
    # At i!=j: p_j * (-p_i) = -p_j * p_i

    # Simplified: grad = -p_target * (one_hot - probs)
    fid_grad = fidelity_weight * target_probs * (probs - one_hot) / batch_size_f

    return ce_grad + fid_grad


# =============================================================================
# Main API Functions
# =============================================================================


def fused_quls_loss(
    logits: tf.Tensor,
    labels: tf.Tensor,
    ce_weight: float = 1.0,
    fidelity_weight: float = 0.01,
    born_weight: float = 0.005,
    entropy_weight: float = 0.01,
    spectral_weight: float = 0.01,
    label_smoothing: float = 0.1,
    target_entropy: float = 0.5,
    use_native: bool = True,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Compute fused QULS loss with optional native C++ acceleration.

    Combines cross-entropy, fidelity, and entropy losses in a single pass.
    Uses SIMD-optimized C++ implementation when available.

    Args:
        logits: Input logits [batch, vocab_size] or [batch, seq, vocab_size].
        labels: Target label indices [batch] or [batch, seq].
        ce_weight: Weight for cross-entropy loss.
        fidelity_weight: Weight for quantum fidelity loss.
        born_weight: Weight for Born rule regularization (placeholder).
        entropy_weight: Weight for entropy regularization.
        spectral_weight: Weight for spectral flatness (placeholder).
        label_smoothing: Label smoothing factor epsilon.
        target_entropy: Target entropy for regularization.
        use_native: Whether to use native C++ ops if available.

    Returns:
        Tuple of (total_loss, metrics_dict) where metrics_dict contains:
            - ce_loss: Cross-entropy loss value
            - fidelity_loss: Fidelity loss value
            - entropy_loss: Entropy regularization loss value
            - total_loss: Weighted sum of losses

    Example:
        >>> logits = tf.random.normal([32, 50000])
        >>> labels = tf.random.uniform([32], 0, 50000, dtype=tf.int32)
        >>> total_loss, metrics = fused_quls_loss(logits, labels)
        >>> print(f"Total loss: {total_loss:.4f}")
    """
    # Handle 3D logits (batch, seq, vocab) by flattening
    original_shape = tf.shape(logits)
    if len(logits.shape) == 3:
        original_shape[0]
        original_shape[1]
        vocab_size = original_shape[2]
        logits = tf.reshape(logits, [-1, vocab_size])
        labels = tf.reshape(labels, [-1])

    # Try native implementation
    if use_native and _NATIVE_OPS_AVAILABLE and _fused_quls_loss_forward is not None:
        try:
            total_loss, ce_loss, fidelity_loss, entropy_loss = _fused_quls_loss_forward(
                logits=logits,
                labels=labels,
                ce_weight=ce_weight,
                fidelity_weight=fidelity_weight,
                born_weight=born_weight,
                entropy_weight=entropy_weight,
                spectral_weight=spectral_weight,
                label_smoothing=label_smoothing,
                target_entropy=target_entropy,
            )

            metrics = {
                "total_loss": total_loss,
                "ce_loss": ce_loss,
                "fidelity_loss": fidelity_loss,
                "entropy_loss": entropy_loss,
            }

            return total_loss, metrics

        except Exception as e:
            logger.warning(f"[FusedQULSLoss] Native op failed, using Python fallback: {e}")

    # Python fallback
    total_loss, ce_loss, fidelity_loss, entropy_loss = _python_quls_loss_forward(
        logits=logits,
        labels=labels,
        ce_weight=ce_weight,
        fidelity_weight=fidelity_weight,
        entropy_weight=entropy_weight,
        label_smoothing=label_smoothing,
        target_entropy=target_entropy,
    )

    metrics = {
        "total_loss": total_loss,
        "ce_loss": ce_loss,
        "fidelity_loss": fidelity_loss,
        "entropy_loss": entropy_loss,
    }

    return total_loss, metrics


def fused_quls_loss_with_gradient(
    logits: tf.Tensor,
    labels: tf.Tensor,
    ce_weight: float = 1.0,
    fidelity_weight: float = 0.01,
    label_smoothing: float = 0.1,
    use_native: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute QULS loss and gradients in a single pass.

    More efficient than computing loss and using tape.gradient() separately.

    Args:
        logits: Input logits [batch, vocab_size].
        labels: Target label indices [batch].
        ce_weight: Weight for cross-entropy loss.
        fidelity_weight: Weight for quantum fidelity loss.
        label_smoothing: Label smoothing factor epsilon.
        use_native: Whether to use native C++ ops if available.

    Returns:
        Tuple of (loss, grad_logits).
    """
    # Compute loss
    total_loss, _ = fused_quls_loss(
        logits=logits,
        labels=labels,
        ce_weight=ce_weight,
        fidelity_weight=fidelity_weight,
        label_smoothing=label_smoothing,
        use_native=use_native,
    )

    # Compute gradients
    if use_native and _NATIVE_OPS_AVAILABLE and _fused_quls_loss_backward is not None:
        try:
            grad_logits = _fused_quls_loss_backward(
                logits=logits,
                labels=labels,
                ce_weight=ce_weight,
                fidelity_weight=fidelity_weight,
                label_smoothing=label_smoothing,
            )
            return total_loss, grad_logits
        except Exception as e:
            logger.warning(f"[FusedQULSLoss] Native backward failed: {e}")

    # Python fallback
    grad_logits = _python_quls_loss_backward(
        logits=logits,
        labels=labels,
        ce_weight=ce_weight,
        fidelity_weight=fidelity_weight,
        label_smoothing=label_smoothing,
    )

    return total_loss, grad_logits


# =============================================================================
# TensorFlow Custom Gradient Registration
# =============================================================================


@tf.custom_gradient
def fused_quls_loss_differentiable(
    logits: tf.Tensor,
    labels: tf.Tensor,
    ce_weight: float = 1.0,
    fidelity_weight: float = 0.01,
    label_smoothing: float = 0.1,
):
    """QULS loss with custom gradient for training.

    Registers a custom gradient that uses the optimized backward pass.

    Args:
        logits: Input logits.
        labels: Target labels.
        ce_weight: CE weight.
        fidelity_weight: Fidelity weight.
        label_smoothing: Label smoothing.

    Returns:
        Total loss tensor.
    """
    total_loss, _ = fused_quls_loss(
        logits=logits,
        labels=labels,
        ce_weight=ce_weight,
        fidelity_weight=fidelity_weight,
        label_smoothing=label_smoothing,
    )

    def grad_fn(upstream):
        if _NATIVE_OPS_AVAILABLE and _fused_quls_loss_backward is not None:
            try:
                grad = _fused_quls_loss_backward(
                    logits=logits,
                    labels=labels,
                    ce_weight=ce_weight,
                    fidelity_weight=fidelity_weight,
                    label_smoothing=label_smoothing,
                )
                return upstream * grad, None  # No gradient for labels
            except Exception:
                pass

        # Fallback
        grad = _python_quls_loss_backward(
            logits=logits,
            labels=labels,
            ce_weight=ce_weight,
            fidelity_weight=fidelity_weight,
            label_smoothing=label_smoothing,
        )
        return upstream * grad, None

    return total_loss, grad_fn


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "fused_quls_loss",
    "fused_quls_loss_with_gradient",
    "fused_quls_loss_differentiable",
    "is_native_available",
]
