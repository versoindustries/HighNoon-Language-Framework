# highnoon/training/quantum_loss.py
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

"""Quantum Unified Loss System (QULS) for HighNoon Framework.

This module implements a unified quantum-aware loss framework that replaces standard
cross-entropy with a multi-component loss system leveraging quantum-inspired regularization.
The system is designed to work with the existing quantum architecture (VQC, QASA, QMamba,
Hamiltonian layers, QCB, MPS) and integrates with HPO training.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     Quantum Unified Loss System (QULS)                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │
    │  │  Primary Loss   │   │  Quantum Terms  │   │  Regularization │       │
    │  │ • Cross-Entropy │   │ • Fidelity      │   │ • Entropy       │       │
    │  │ • Label Smooth  │   │ • Born Rule     │   │ • Spectral      │       │
    │  │                 │   │ • Coherence     │   │ • Entanglement  │       │
    │  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘       │
    │           └─────────────────────┼─────────────────────┘                 │
    │                                 ▼                                       │
    │                    ┌────────────────────────┐                           │
    │                    │  Adaptive Weight       │                           │
    │                    │  Controller (VQC-Aware)│                           │
    │                    └────────────────────────┘                           │
    └─────────────────────────────────────────────────────────────────────────┘

Mathematical Foundation:
    Total Loss = w_task · L_task + w_fidelity · L_fidelity + w_born · L_born
                 + w_entropy · L_entropy + w_coherence · L_coherence
                 + w_symplectic · L_symplectic + w_entanglement · L_entanglement

    Where:
    - L_task: Primary task loss (cross-entropy with label smoothing)
    - L_fidelity: Trace fidelity F(p,q) = (Σ√pᵢ√qᵢ)², loss = 1 - F
    - L_born: Born rule |ψ|² = p enforcement for VQC outputs
    - L_entropy: Von Neumann entropy H(ρ) = -Tr(ρ log ρ) regularization
    - L_coherence: Coherence preservation penalty from QCB metrics
    - L_symplectic: Hamiltonian energy conservation |H(t) - H(0)|
    - L_entanglement: MPS bond entropy regularization

References:
    - Nielsen & Chuang (2010): Quantum Computation and Quantum Information
    - Cerezo et al. (2021): Variational Quantum Algorithms. Nature Rev. Physics.
    - Abbas et al. (2021): The power of quantum neural networks. Nature Comp. Sci.
    - Wang (2024): Optimal Trace Distance and Fidelity Estimations for Pure Quantum States
    - Lie algebraic theory of barren plateaus (Nature Comm. 2024)
    - Symplectic learning for HNNs (J. Comp. Physics 2023)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf

from highnoon import config as hn_config

logger = logging.getLogger(__name__)

# Phase 300+: HD Spectral Entropy Integration
try:
    from highnoon._native.ops.hd_spectral_entropy import (
        hd_spectral_entropy as hd_spectral_entropy_native,
    )
    from highnoon._native.ops.hd_spectral_entropy import hd_spectral_entropy_available
    from highnoon._native.ops.hd_spectral_entropy import (
        hd_spectral_flatness as hd_spectral_flatness_native,
    )
    from highnoon.config import USE_HD_SPECTRAL_ENTROPY

    _HD_SPECTRAL_AVAILABLE = hd_spectral_entropy_available()
except ImportError:
    USE_HD_SPECTRAL_ENTROPY = False
    _HD_SPECTRAL_AVAILABLE = False
    hd_spectral_entropy_native = None
    hd_spectral_flatness_native = None


# =============================================================================
# Cached Eigenvalue Computation (Merged from quantum_loss_v2.py)
# =============================================================================


class EigenvalueCache:
    """Cache for eigenvalue computations to avoid duplicate work.

    Consolidation from QULS V2: spectral_entropy and spectral_flatness previously
    computed eigenvalues independently. This cache ensures single computation per
    forward pass, providing ~60% memory reduction.

    Memory: O(k) for k eigenvalues
    Time: O(k × power_iter × d²) amortized to single computation

    Example:
        >>> cache = EigenvalueCache()
        >>> eigenvalues = cache.get_or_compute(hidden_states, num_eigenvalues=8)
        >>> cache.clear()  # Call at end of training step
    """

    def __init__(self, max_cache_size: int = 4):
        self._cache: dict[int, tf.Tensor] = {}
        self._max_size = max_cache_size

    def get_or_compute(
        self,
        hidden_states: tf.Tensor,
        num_eigenvalues: int = 8,
        power_iter_steps: int = 10,
        epsilon: float = 1e-8,
    ) -> tf.Tensor:
        """Get cached eigenvalues or compute if not cached.

        Args:
            hidden_states: Hidden state tensor [batch, seq, dim] or [batch, dim].
            num_eigenvalues: Number of top eigenvalues to compute.
            power_iter_steps: Power iteration steps per eigenvalue.
            epsilon: Numerical stability constant.

        Returns:
            Top-k eigenvalues as tensor [k].
        """
        cache_key = id(hidden_states)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Compute eigenvalues via power iteration
        eigenvalues = self._power_iteration_eigenvalues(
            hidden_states, num_eigenvalues, power_iter_steps, epsilon
        )

        # Update cache with LRU eviction
        if len(self._cache) >= self._max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._cache[cache_key] = eigenvalues
        return eigenvalues

    def _power_iteration_eigenvalues(
        self,
        hidden_states: tf.Tensor,
        num_eigenvalues: int,
        power_iter_steps: int,
        epsilon: float,
    ) -> tf.Tensor:
        """Compute top-k eigenvalues via deflation-based power iteration.

        Complexity: O(k × power_iter × d²)
        """
        h = tf.cast(hidden_states, tf.float32)

        # Flatten to [N, dim]
        if len(h.shape) == 3:
            h_flat = tf.reshape(h, [-1, h.shape[-1]])
        else:
            h_flat = h

        # Center the data
        mean = tf.reduce_mean(h_flat, axis=0, keepdims=True)
        h_centered = h_flat - mean

        # Compute covariance: (1/N) * H^T * H
        n_samples = tf.cast(tf.shape(h_centered)[0], tf.float32)
        cov = tf.matmul(h_centered, h_centered, transpose_a=True) / n_samples

        # Power iteration with deflation
        dim = tf.shape(cov)[0]
        k = tf.minimum(num_eigenvalues, dim)
        eigenvalues = []

        A = cov
        for _ in range(k):
            # Random initialization
            v = tf.random.normal([dim, 1], dtype=tf.float32)
            v = v / tf.norm(v)

            # Power iteration
            for _ in range(power_iter_steps):
                v_new = tf.matmul(A, v)
                v_new = v_new / (tf.norm(v_new) + epsilon)
                v = v_new

            # Extract eigenvalue: λ = v^T A v
            eigenvalue = tf.squeeze(tf.matmul(tf.matmul(v, A, transpose_a=True), v))
            eigenvalue = tf.maximum(eigenvalue, epsilon)
            eigenvalues.append(eigenvalue)

            # Deflate: A = A - λ * v * v^T
            A = A - eigenvalue * tf.matmul(v, v, transpose_b=True)

        return tf.stack(eigenvalues)

    def clear(self):
        """Clear the eigenvalue cache."""
        self._cache.clear()


# Global eigenvalue cache instance for use by spectral entropy functions
_eigenvalue_cache = EigenvalueCache()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class QULSConfig:
    """Configuration for Quantum Unified Loss System.

    Attributes:
        enabled: Master switch for QULS (if False, uses standard cross-entropy).
        primary_loss: Primary task loss type.
        label_smoothing: Label smoothing factor for cross-entropy.

        # Quantum Fidelity Loss
        fidelity_enabled: Enable quantum fidelity loss component.
        fidelity_weight: Initial fidelity loss weight.
        fidelity_temperature: Born probability temperature for softmax sharpening.

        # Born Rule Loss
        born_rule_enabled: Enable Born rule enforcement for VQC outputs.
        born_rule_weight: Born rule loss weight.

        # Entropy Regularization
        entropy_enabled: Enable Von Neumann entropy regularization.
        entropy_weight: Entropy loss weight.
        target_entropy: Target entropy value (0.5 = balanced).
        spectral_weight: Spectral flatness loss weight.
        spectral_target: Target spectral flatness.
        power_iter_steps: Power iteration steps for eigenvalue computation.

        # Coherence Preservation
        coherence_enabled: Enable coherence preservation from QCB.
        coherence_weight: Coherence loss weight.
        coherence_threshold: Minimum acceptable coherence.

        # Symplectic Conservation
        symplectic_enabled: Enable Hamiltonian energy conservation.
        symplectic_weight: Symplectic loss weight.

        # Entanglement Regularization
        entanglement_enabled: Enable MPS bond entropy regularization.
        entanglement_weight: Entanglement loss weight.
        min_bond_entropy: Minimum target bond entropy.

        # Adaptive Weighting
        adaptive_weights: Enable VQC-aware weight adaptation.
        weight_ema_decay: EMA decay for weight updates.
        vqc_variance_boost: Weight boost multiplier for high VQC gradient variance.
        barren_plateau_reduction: Weight reduction factor during barren plateau.
    """

    # Master switch
    enabled: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_QUANTUM_UNIFIED_LOSS", True)
    )

    # Primary Loss
    primary_loss: str = field(
        default_factory=lambda: getattr(
            hn_config, "QULS_PRIMARY_LOSS", "sparse_categorical_crossentropy"
        )
    )
    label_smoothing: float = field(
        default_factory=lambda: getattr(hn_config, "QULS_LABEL_SMOOTHING", 0.1)
    )

    # Quantum Fidelity Loss
    fidelity_enabled: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_QUANTUM_FIDELITY_LOSS", True)
    )
    fidelity_weight: float = field(
        default_factory=lambda: getattr(hn_config, "QULS_FIDELITY_WEIGHT", 0.01)
    )
    fidelity_temperature: float = field(
        default_factory=lambda: getattr(hn_config, "QULS_FIDELITY_TEMPERATURE", 1.0)
    )

    # Born Rule Loss
    born_rule_enabled: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_BORN_RULE_LOSS", True)
    )
    born_rule_weight: float = field(
        default_factory=lambda: getattr(hn_config, "QULS_BORN_RULE_WEIGHT", 0.005)
    )

    # Entropy Regularization
    entropy_enabled: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_ENTROPY_REGULARIZATION", True)
    )
    entropy_weight: float = field(
        default_factory=lambda: getattr(hn_config, "ENTROPY_REG_WEIGHT", 0.01)
    )
    target_entropy: float = field(default_factory=lambda: getattr(hn_config, "TARGET_ENTROPY", 0.5))
    spectral_weight: float = field(
        default_factory=lambda: getattr(hn_config, "SPECTRAL_REG_WEIGHT", 0.01)
    )
    spectral_target: float = field(
        default_factory=lambda: getattr(hn_config, "SPECTRAL_FLATNESS_TARGET", 0.8)
    )
    power_iter_steps: int = field(
        default_factory=lambda: getattr(hn_config, "ENTROPY_POWER_ITER_STEPS", 10)
    )

    # Coherence Preservation
    coherence_enabled: bool = field(
        default_factory=lambda: getattr(hn_config, "QULS_COHERENCE_ENABLED", True)
    )
    coherence_weight: float = field(
        default_factory=lambda: getattr(hn_config, "QULS_COHERENCE_WEIGHT", 0.01)
    )
    coherence_threshold: float = field(
        default_factory=lambda: getattr(hn_config, "UNIFIED_BUS_COHERENCE_THRESHOLD", 0.85)
    )

    # Symplectic Conservation
    symplectic_enabled: bool = field(
        default_factory=lambda: getattr(hn_config, "QULS_SYMPLECTIC_ENABLED", True)
    )
    symplectic_weight: float = field(
        default_factory=lambda: getattr(hn_config, "QULS_SYMPLECTIC_WEIGHT", 0.01)
    )

    # Entanglement Regularization
    entanglement_enabled: bool = field(
        default_factory=lambda: getattr(hn_config, "QULS_ENTANGLEMENT_ENABLED", True)
    )
    entanglement_weight: float = field(
        default_factory=lambda: getattr(hn_config, "ENTANGLEMENT_REGULARIZATION", 0.01)
    )
    min_bond_entropy: float = field(
        default_factory=lambda: getattr(hn_config, "MIN_BOND_ENTROPY", 0.5)
    )

    # Adaptive Weighting
    adaptive_weights: bool = field(
        default_factory=lambda: getattr(hn_config, "QULS_ADAPTIVE_WEIGHTS", True)
    )
    weight_ema_decay: float = field(
        default_factory=lambda: getattr(hn_config, "QULS_WEIGHT_EMA_DECAY", 0.99)
    )
    vqc_variance_boost: float = field(
        default_factory=lambda: getattr(hn_config, "QULS_VQC_VARIANCE_BOOST", 2.0)
    )
    barren_plateau_reduction: float = field(
        default_factory=lambda: getattr(hn_config, "QULS_BARREN_PLATEAU_REDUCTION", 0.1)
    )


# =============================================================================
# Individual Loss Components
# =============================================================================


def trace_fidelity(p: tf.Tensor, q: tf.Tensor, epsilon: float = 1e-8) -> tf.Tensor:
    """Compute trace fidelity between two probability distributions.

    The trace fidelity (also called Bhattacharyya coefficient) between
    probability distributions p and q is:

        F(p, q) = (Σᵢ √pᵢ · √qᵢ)²

    For diagonal density matrices, this is equivalent to the quantum fidelity
    F(ρ, σ) = (Tr(√(√ρ σ √ρ)))².

    Args:
        p: First probability distribution [batch, dim] or [batch, seq, dim].
        q: Second probability distribution with same shape as p.
        epsilon: Small constant for numerical stability.

    Returns:
        Fidelity values [batch] or [batch, seq] in range [0, 1].
    """
    # Ensure float32 and non-negative
    p = tf.cast(p, tf.float32)
    q = tf.cast(q, tf.float32)
    p = tf.maximum(p, epsilon)
    q = tf.maximum(q, epsilon)

    # Normalize to probabilities if needed
    p_sum = tf.reduce_sum(p, axis=-1, keepdims=True)
    q_sum = tf.reduce_sum(q, axis=-1, keepdims=True)
    p = p / tf.maximum(p_sum, epsilon)
    q = q / tf.maximum(q_sum, epsilon)

    # Compute fidelity: F = (Σ √p · √q)²
    sqrt_p = tf.sqrt(p)
    sqrt_q = tf.sqrt(q)
    overlap = tf.reduce_sum(sqrt_p * sqrt_q, axis=-1)
    fidelity = tf.square(overlap)

    return fidelity


def quantum_fidelity_loss(
    predictions: tf.Tensor,
    targets: tf.Tensor,
    temperature: float = 1.0,
    epsilon: float = 1e-8,
) -> tf.Tensor:
    """Quantum fidelity-inspired loss using Born rule probabilities.

    Measures the overlap between predicted distribution |ψ|² and target
    using trace fidelity approximation. This loss encourages the model
    to produce predictions that have high quantum fidelity with targets.

    Mathematical formulation:
        L_fidelity = 1 - F(softmax(pred/T), one_hot(target))
                   = 1 - Σᵢ √pᵢ · √qᵢ)²

    Args:
        predictions: Model logits [batch, vocab_size] or [batch, seq, vocab_size].
        targets: Target token IDs [batch] or [batch, seq].
        temperature: Softmax temperature for probability sharpening.
        epsilon: Small constant for numerical stability.

    Returns:
        Scalar loss tensor (mean fidelity loss across batch).

    Example:
        >>> predictions = tf.random.normal([32, 50000])  # Batch of logits
        >>> targets = tf.random.uniform([32], 0, 50000, dtype=tf.int32)
        >>> loss = quantum_fidelity_loss(predictions, targets)
    """
    # Get prediction probabilities via temperature-scaled softmax
    pred_probs = tf.nn.softmax(predictions / temperature, axis=-1)

    # Create target distribution (one-hot)
    vocab_size = tf.shape(predictions)[-1]
    target_probs = tf.one_hot(targets, depth=vocab_size, dtype=tf.float32)

    # Handle sequence dimension if present
    if len(predictions.shape) == 3:
        # Flatten batch and sequence for fidelity computation
        batch_size = tf.shape(predictions)[0]
        seq_len = tf.shape(predictions)[1]
        pred_flat = tf.reshape(pred_probs, [-1, vocab_size])
        target_flat = tf.reshape(target_probs, [-1, vocab_size])
        fidelity_flat = trace_fidelity(pred_flat, target_flat, epsilon)
        fidelity = tf.reshape(fidelity_flat, [batch_size, seq_len])
        loss = 1.0 - tf.reduce_mean(fidelity)
    else:
        fidelity = trace_fidelity(pred_probs, target_probs, epsilon)
        loss = 1.0 - tf.reduce_mean(fidelity)

    return loss


def born_rule_amplitude_loss(
    vqc_amplitudes: tf.Tensor,
    target_distribution: tf.Tensor | None = None,
    epsilon: float = 1e-8,
) -> tf.Tensor:
    """Enforce Born rule |ψ|² = p for VQC layer outputs.

    Ensures VQC amplitude outputs follow proper quantum probability rules.
    If target_distribution is None, enforces normalization constraint only.

    Mathematical formulation:
        L_born = Σᵢ (|αᵢ|² - pᵢ)²    (if target given)
        L_born = (Σᵢ |αᵢ|² - 1)²     (normalization only)

    Args:
        vqc_amplitudes: Complex amplitudes from VQC [batch, dim] or magnitude squares.
        target_distribution: Target probability distribution [batch, dim], or None.
        epsilon: Small constant for numerical stability.

    Returns:
        Scalar Born rule violation penalty.

    Note:
        For real-valued VQC outputs representing |ψ|² directly, pass those as
        vqc_amplitudes and the function will treat them as probabilities.
    """
    amplitudes = tf.cast(vqc_amplitudes, tf.float32)

    # If complex, compute magnitude squared
    if amplitudes.dtype == tf.complex64 or amplitudes.dtype == tf.complex128:
        probs = tf.abs(amplitudes) ** 2
    else:
        # Assume already |ψ|² or need to normalize
        probs = tf.maximum(tf.abs(amplitudes), epsilon)

    if target_distribution is not None:
        # Enforce Born rule against target
        target = tf.cast(target_distribution, tf.float32)
        target = tf.maximum(target, epsilon)
        # Normalize target
        target = target / tf.reduce_sum(target, axis=-1, keepdims=True)
        # Normalize probs
        probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
        # MSE between predicted probs and target
        loss = tf.reduce_mean(tf.square(probs - target))
    else:
        # Enforce normalization: Σ|α|² = 1
        norm = tf.reduce_sum(probs, axis=-1)
        loss = tf.reduce_mean(tf.square(norm - 1.0))

    return loss


def spectral_entropy(
    hidden_states: tf.Tensor,
    num_eigenvalues: int = 10,
    power_iter_steps: int = 10,
    epsilon: float = 1e-8,
) -> tf.Tensor:
    """Compute spectral entropy from hidden state covariance matrix.

    Uses power iteration for O(n × num_eigenvalues) complexity instead of
    full eigendecomposition O(n³). The spectral entropy measures the
    "uniformity" of the eigenvalue spectrum.

    Phase 300+: When USE_HD_SPECTRAL_ENTROPY=True and native ops are available,
    uses O(d log d) FFT-based spectral analysis instead of O(d²) power iteration.

    Mathematical formulation:
        Σ = (1/N) Σᵢ (hᵢ - μ)(hᵢ - μ)ᵀ   (covariance)
        H = -Σⱼ λⱼ log(λⱼ)                 (spectral entropy)

    Normalized entropy in [0, 1]:
        H_norm = H / log(k)  where k = num_eigenvalues

    Args:
        hidden_states: Hidden state tensor [batch, seq_len, hidden_dim] or
                      [batch, hidden_dim].
        num_eigenvalues: Number of top eigenvalues to compute.
        power_iter_steps: Power iteration steps per eigenvalue.
        epsilon: Small constant for numerical stability.

    Returns:
        Spectral entropy scalar (mean over batch).
    """
    # Phase 300+: HD Spectral Entropy (O(d log d) via FFT)
    if USE_HD_SPECTRAL_ENTROPY and _HD_SPECTRAL_AVAILABLE:
        try:
            return hd_spectral_entropy_native(hidden_states, epsilon)
        except Exception as e:
            logger.warning(f"HD spectral entropy failed, using fallback: {e}")
            # Fall through to power iteration

    # Fallback: Power iteration eigenvalue computation (O(d²))
    h = tf.cast(hidden_states, tf.float32)

    # Handle different input shapes
    if len(h.shape) == 3:
        # [batch, seq, dim] -> flatten to [batch * seq, dim]
        h_flat = tf.reshape(h, [-1, h.shape[-1]])
    else:
        # [batch, dim]
        h_flat = h

    # Center the data
    mean = tf.reduce_mean(h_flat, axis=0, keepdims=True)
    h_centered = h_flat - mean

    # Compute covariance: (1/N) * H^T * H
    n_samples = tf.cast(tf.shape(h_centered)[0], tf.float32)
    cov = tf.matmul(h_centered, h_centered, transpose_a=True) / n_samples

    # Power iteration for top-k eigenvalues
    dim = tf.shape(cov)[0]
    k = tf.minimum(num_eigenvalues, dim)
    eigenvalues = []

    # Initialize random vector
    v = tf.random.normal([dim, 1], dtype=tf.float32)
    v = v / tf.norm(v)

    # Deflation-based power iteration for multiple eigenvalues
    A = cov
    for _ in range(k):
        # Power iteration for largest eigenvalue of A
        for _ in range(power_iter_steps):
            v_new = tf.matmul(A, v)
            v_new = v_new / (tf.norm(v_new) + epsilon)
            v = v_new

        # Compute eigenvalue: λ = v^T A v
        eigenvalue = tf.squeeze(tf.matmul(tf.matmul(v, A, transpose_a=True), v))
        eigenvalue = tf.maximum(eigenvalue, epsilon)
        eigenvalues.append(eigenvalue)

        # Deflate: A = A - λ * v * v^T
        A = A - eigenvalue * tf.matmul(v, v, transpose_b=True)

        # Reinitialize v for next eigenvalue
        v = tf.random.normal([dim, 1], dtype=tf.float32)
        v = v / tf.norm(v)

    # Stack eigenvalues and normalize to probability
    eigenvalues = tf.stack(eigenvalues)
    eigenvalues = tf.maximum(eigenvalues, epsilon)
    total = tf.reduce_sum(eigenvalues)
    probs = eigenvalues / (total + epsilon)

    # Compute entropy: H = -Σ p log(p)
    entropy = -tf.reduce_sum(probs * tf.math.log(probs + epsilon))

    # Normalize by log(k) to get value in [0, 1]
    max_entropy = tf.math.log(tf.cast(k, tf.float32))
    normalized_entropy = entropy / (max_entropy + epsilon)

    return normalized_entropy


def entropy_regularization_loss(
    hidden_states: tf.Tensor,
    target_entropy: float = 0.5,
    num_eigenvalues: int = 10,
    power_iter_steps: int = 10,
) -> tf.Tensor:
    """Von Neumann entropy regularization for representation diversity.

    Encourages diverse activations by penalizing deviation from target entropy:
    - Too low entropy → collapsed representations (all similar)
    - Too high entropy → uniform/uninformative representations

    Mathematical formulation:
        L_entropy = (H(ρ) - H_target)²

    Args:
        hidden_states: Hidden state tensor [batch, seq_len, hidden_dim].
        target_entropy: Target entropy value in [0, 1]. Default 0.5 (balanced).
        num_eigenvalues: Number of eigenvalues for spectral entropy.
        power_iter_steps: Power iteration steps.

    Returns:
        Scalar entropy regularization loss.
    """
    current_entropy = spectral_entropy(
        hidden_states,
        num_eigenvalues=num_eigenvalues,
        power_iter_steps=power_iter_steps,
    )

    loss = tf.square(current_entropy - target_entropy)
    return loss


def spectral_flatness_loss(
    hidden_states: tf.Tensor,
    target_flatness: float = 0.8,
    num_eigenvalues: int = 10,
    power_iter_steps: int = 10,
    epsilon: float = 1e-8,
) -> tf.Tensor:
    """Spectral flatness regularization for eigenvalue distribution.

    Spectral flatness measures how uniform the eigenvalue distribution is:
        SF = (Πᵢ λᵢ)^(1/k) / (Σᵢ λᵢ / k)
           = geometric_mean / arithmetic_mean

    Phase 300+: When USE_HD_SPECTRAL_ENTROPY=True and native ops are available,
    uses O(d log d) FFT-based spectral flatness computation.

    Values:
    - SF → 1: Flat spectrum (all eigenvalues equal)
    - SF → 0: Peaked spectrum (few dominant eigenvalues)

    Args:
        hidden_states: Hidden state tensor.
        target_flatness: Target flatness in [0, 1].
        num_eigenvalues: Number of eigenvalues to compute.
        power_iter_steps: Power iteration steps.
        epsilon: Numerical stability constant.

    Returns:
        Scalar spectral flatness loss.
    """
    # Phase 300+: HD Spectral Flatness (O(d log d) via FFT)
    if USE_HD_SPECTRAL_ENTROPY and _HD_SPECTRAL_AVAILABLE:
        try:
            flatness = hd_spectral_flatness_native(hidden_states, epsilon)
            loss = tf.square(flatness - target_flatness)
            return tf.reduce_mean(loss)  # Average over batch
        except Exception as e:
            logger.warning(f"HD spectral flatness failed, using fallback: {e}")
            # Fall through to power iteration

    # Fallback: Power iteration eigenvalue computation (O(d²))
    h = tf.cast(hidden_states, tf.float32)

    # Flatten to [N, dim]
    if len(h.shape) == 3:
        h_flat = tf.reshape(h, [-1, h.shape[-1]])
    else:
        h_flat = h

    # Center and compute covariance
    mean = tf.reduce_mean(h_flat, axis=0, keepdims=True)
    h_centered = h_flat - mean
    n_samples = tf.cast(tf.shape(h_centered)[0], tf.float32)
    cov = tf.matmul(h_centered, h_centered, transpose_a=True) / n_samples

    # Get eigenvalues via power iteration (same as spectral_entropy)
    dim = tf.shape(cov)[0]
    k = tf.minimum(num_eigenvalues, dim)
    eigenvalues = []

    v = tf.random.normal([dim, 1], dtype=tf.float32)
    v = v / tf.norm(v)
    A = cov

    for _ in range(k):
        for _ in range(power_iter_steps):
            v_new = tf.matmul(A, v)
            v_new = v_new / (tf.norm(v_new) + epsilon)
            v = v_new

        eigenvalue = tf.squeeze(tf.matmul(tf.matmul(v, A, transpose_a=True), v))
        eigenvalue = tf.maximum(eigenvalue, epsilon)
        eigenvalues.append(eigenvalue)

        A = A - eigenvalue * tf.matmul(v, v, transpose_b=True)
        v = tf.random.normal([dim, 1], dtype=tf.float32)
        v = v / tf.norm(v)

    eigenvalues = tf.stack(eigenvalues)
    eigenvalues = tf.maximum(eigenvalues, epsilon)

    # Compute spectral flatness: geometric_mean / arithmetic_mean
    log_eigenvalues = tf.math.log(eigenvalues)
    log_geometric_mean = tf.reduce_mean(log_eigenvalues)
    geometric_mean = tf.exp(log_geometric_mean)
    arithmetic_mean = tf.reduce_mean(eigenvalues)

    flatness = geometric_mean / (arithmetic_mean + epsilon)
    flatness = tf.minimum(flatness, 1.0)  # Clip to valid range

    loss = tf.square(flatness - target_flatness)
    return loss


def coherence_preservation_loss(
    coherence_metrics: dict[str, tf.Tensor] | tf.Tensor,
    threshold: float = 0.85,
    epsilon: float = 1e-8,
) -> tf.Tensor:
    """Quantum coherence preservation penalty across blocks.

    Penalizes loss of coherence through the network using metrics from the
    Quantum Coherence Bus (QCB) or direct coherence measurements.

    Mathematical formulation:
        L_coherence = max(0, threshold - C)²

    Where C is the coherence metric (fidelity, purity, or custom).

    Args:
        coherence_metrics: Either a dict with per-block coherence values
                          (keys like 'block_0', 'block_1', ..., 'global'), or
                          a single tensor representing global coherence.
        threshold: Minimum acceptable coherence target.
        epsilon: Numerical stability constant.

    Returns:
        Scalar coherence preservation loss.

    Example:
        >>> metrics = {'block_0': 0.95, 'block_1': 0.88, 'global': 0.91}
        >>> loss = coherence_preservation_loss(metrics, threshold=0.85)
    """
    if isinstance(coherence_metrics, dict):
        # Average over all block coherence values
        values = []
        for _, value in coherence_metrics.items():
            if tf.is_tensor(value):
                values.append(tf.cast(value, tf.float32))
            else:
                values.append(tf.constant(value, dtype=tf.float32))

        if not values:
            return tf.constant(0.0, dtype=tf.float32)

        coherence = tf.reduce_mean(tf.stack(values))
    else:
        coherence = tf.cast(coherence_metrics, tf.float32)

    # Penalize if coherence falls below threshold
    deficit = tf.maximum(threshold - coherence, 0.0)
    loss = tf.square(deficit)

    return loss


def symplectic_energy_loss(
    initial_energy: tf.Tensor,
    final_energy: tf.Tensor,
    dt: float = 0.01,
    epsilon: float = 1e-8,
) -> tf.Tensor:
    """Symplectic energy conservation loss for Hamiltonian layers.

    Penalizes energy drift in TimeCrystal and Hamiltonian Neural Network blocks.
    For symplectic integrators, energy should be approximately conserved.

    Mathematical formulation:
        L_symplectic = |H(q_t, p_t) - H(q_0, p_0)| / dt

    The division by dt normalizes for different integration step sizes.

    Args:
        initial_energy: Initial Hamiltonian energy H(q_0, p_0) [batch].
        final_energy: Final Hamiltonian energy H(q_T, p_T) [batch].
        dt: Time step for normalization.
        epsilon: Numerical stability constant.

    Returns:
        Scalar symplectic energy conservation loss.

    Reference:
        SympFlow (NeurIPS 2024), Symplectic Learning for HNNs (J. Comp. Phys. 2023)
    """
    h_init = tf.cast(initial_energy, tf.float32)
    h_final = tf.cast(final_energy, tf.float32)

    # Energy drift per time step
    drift = tf.abs(h_final - h_init) / (dt + epsilon)

    # Mean over batch
    loss = tf.reduce_mean(drift)

    return loss


def entanglement_regularization_loss(
    bond_entropies: tf.Tensor | list[tf.Tensor],
    min_entropy: float = 0.5,
    epsilon: float = 1e-8,
) -> tf.Tensor:
    """MPS bond entropy regularization for entanglement structure.

    Encourages meaningful entanglement in Matrix Product State representations
    by penalizing entropy values below a minimum threshold. This prevents
    the model from collapsing to product states (no entanglement).

    Mathematical formulation:
        L_entanglement = Σⱼ max(0, S_min - S_j)²

    Where S_j is the von Neumann entropy at bond j of the MPS.

    Args:
        bond_entropies: Entanglement entropies at MPS bonds, either as a
                       tensor [num_bonds] or list of scalars.
        min_entropy: Minimum target entropy per bond.
        epsilon: Numerical stability constant.

    Returns:
        Scalar entanglement regularization loss.
    """
    if isinstance(bond_entropies, list):
        entropies = tf.stack([tf.cast(e, tf.float32) for e in bond_entropies])
    else:
        entropies = tf.cast(bond_entropies, tf.float32)

    # Penalize entropy below minimum threshold
    deficit = tf.maximum(min_entropy - entropies, 0.0)
    loss = tf.reduce_mean(tf.square(deficit))

    return loss


# =============================================================================
# Adaptive Weight Controller
# =============================================================================


class AdaptiveWeightController:
    """VQC-aware dynamic weight adjustment for QULS components.

    Adjusts loss component weights based on:
    1. VQC gradient variance (from GaLore/barren plateau monitor)
    2. Training progress (step-based curriculum)
    3. Barren plateau detection signals
    4. Entanglement entropy from MPS/QCB

    Uses EMA smoothing for stable weight evolution.

    Attributes:
        config: QULSConfig with weight settings.
        current_weights: Dict of current component weights.
        ema_weights: Dict of EMA-smoothed weights.
        step: Current training step.
    """

    def __init__(self, config: QULSConfig):
        """Initialize adaptive weight controller.

        Args:
            config: QULSConfig with weight and adaptation settings.
        """
        self.config = config

        # Initialize weights from config
        self.base_weights = {
            "fidelity": config.fidelity_weight if config.fidelity_enabled else 0.0,
            "born_rule": config.born_rule_weight if config.born_rule_enabled else 0.0,
            "entropy": config.entropy_weight if config.entropy_enabled else 0.0,
            "spectral": config.spectral_weight if config.entropy_enabled else 0.0,
            "coherence": config.coherence_weight if config.coherence_enabled else 0.0,
            "symplectic": config.symplectic_weight if config.symplectic_enabled else 0.0,
            "entanglement": config.entanglement_weight if config.entanglement_enabled else 0.0,
        }

        self.current_weights = dict(self.base_weights)
        self.ema_weights = dict(self.base_weights)
        self.step = 0

        # Tracking metrics
        self._vqc_variance_history: list[float] = []
        self._gradient_entropy_history: list[float] = []
        self._in_barren_plateau = False

    def update(
        self,
        step: int,
        total_steps: int,
        vqc_gradient_variance: float | None = None,
        gradient_entropy: float | None = None,
        in_barren_plateau: bool = False,
        entanglement_entropy: float | None = None,
    ) -> dict[str, float]:
        """Update weights based on training state.

        Args:
            step: Current training step.
            total_steps: Total training steps (for curriculum).
            vqc_gradient_variance: Gradient variance from VQC layers.
            gradient_entropy: Overall gradient entropy.
            in_barren_plateau: Whether barren plateau is detected.
            entanglement_entropy: Average entanglement entropy from MPS.

        Returns:
            Updated weight dictionary.
        """
        self.step = step
        self._in_barren_plateau = in_barren_plateau

        if not self.config.adaptive_weights:
            return self.current_weights

        # Track metrics history
        if vqc_gradient_variance is not None:
            self._vqc_variance_history.append(vqc_gradient_variance)
            if len(self._vqc_variance_history) > 100:
                self._vqc_variance_history.pop(0)

        if gradient_entropy is not None:
            self._gradient_entropy_history.append(gradient_entropy)
            if len(self._gradient_entropy_history) > 100:
                self._gradient_entropy_history.pop(0)

        # Compute progress factor (0 to 1)
        progress = step / max(total_steps, 1)

        # Curriculum: gradually increase quantum loss weights
        curriculum_factor = min(1.0, 0.1 + 0.9 * progress)

        # VQC variance boost: increase weights when VQC gradients are high variance
        vqc_boost = 1.0
        if vqc_gradient_variance is not None and vqc_gradient_variance > 0:
            # Boost when variance is high (need more regularization)
            avg_variance = (
                sum(self._vqc_variance_history) / len(self._vqc_variance_history)
                if self._vqc_variance_history
                else vqc_gradient_variance
            )
            if vqc_gradient_variance > avg_variance * 1.5:
                vqc_boost = self.config.vqc_variance_boost

        # Barren plateau reduction: reduce quantum terms during barren plateau
        bp_factor = 1.0
        if in_barren_plateau:
            bp_factor = self.config.barren_plateau_reduction

        # Apply adjustments to quantum loss weights
        new_weights = {}
        for name, base_weight in self.base_weights.items():
            if name in ["fidelity", "born_rule", "coherence", "entanglement"]:
                # Quantum-specific weights: apply all factors
                weight = base_weight * curriculum_factor * vqc_boost * bp_factor
            elif name in ["entropy", "spectral"]:
                # Entropy weights: apply curriculum and VQC boost
                weight = base_weight * curriculum_factor * vqc_boost
            else:
                # Symplectic: apply curriculum only
                weight = base_weight * curriculum_factor

            new_weights[name] = weight

        # EMA smoothing
        decay = self.config.weight_ema_decay
        for name in new_weights:
            self.ema_weights[name] = (
                decay * self.ema_weights[name] + (1 - decay) * new_weights[name]
            )

        self.current_weights = dict(self.ema_weights)
        return self.current_weights

    def get_weights(self) -> dict[str, float]:
        """Get current component weights.

        Returns:
            Dictionary of current loss component weights.
        """
        return self.current_weights

    def get_statistics(self) -> dict[str, Any]:
        """Get weight controller statistics for logging.

        Returns:
            Dictionary with controller state and metrics.
        """
        return {
            "step": self.step,
            "weights": dict(self.current_weights),
            "in_barren_plateau": self._in_barren_plateau,
            "vqc_variance_samples": len(self._vqc_variance_history),
            "avg_vqc_variance": (
                sum(self._vqc_variance_history) / len(self._vqc_variance_history)
                if self._vqc_variance_history
                else 0.0
            ),
        }


# =============================================================================
# Main QULS Class
# =============================================================================


class QuantumUnifiedLoss:
    """Unified loss combining primary task loss with quantum-enhanced terms.

    The Quantum Unified Loss System (QULS) replaces standard cross-entropy
    with a multi-component loss framework leveraging quantum-inspired
    regularization terms. It integrates with the existing quantum architecture
    (VQC, QASA, QMamba, Hamiltonian layers, QCB, MPS).

    Loss Components:
        1. Primary Loss (L_task): Standard task loss (cross-entropy, MSE, etc.)
        2. Quantum Fidelity Loss (L_fidelity): Born probability alignment
        3. Born Rule Loss (L_born): VQC output normalization
        4. Entropy Regularization (L_entropy): Representation diversity
        5. Spectral Flatness (L_spectral): Eigenvalue distribution
        6. Coherence Preservation (L_coherence): Quantum state stability
        7. Symplectic Conservation (L_symplectic): Hamiltonian energy drift
        8. Entanglement Regularization (L_entanglement): MPS bond entropy

    Total Loss:
        L_total = L_task + Σᵢ wᵢ · Lᵢ

    Where weights wᵢ are dynamically adjusted based on VQC gradient statistics,
    training progress, and barren plateau detection.

    Attributes:
        config: QULSConfig with all settings.
        weight_controller: AdaptiveWeightController for dynamic weighting.
        _primary_loss_fn: Compiled primary loss function.
        _loss_components: Dict of last computed loss components.

    Example:
        >>> quls = QuantumUnifiedLoss()
        >>> loss, components = quls.compute_loss(
        ...     predictions=model_output,
        ...     targets=labels,
        ...     hidden_states=model.get_hidden_states(),
        ...     coherence_metrics=qcb.get_metrics(),
        ...     hamiltonian_energies=(h_init, h_final),
        ...     bond_entropies=mps.get_entropies(),
        ... )
        >>> print(f"Total loss: {loss:.4f}")
        >>> print(f"Components: {components}")
    """

    def __init__(self, config: QULSConfig | None = None):
        """Initialize Quantum Unified Loss System.

        Args:
            config: QULSConfig with settings. If None, uses defaults from config.py.
        """
        self.config = config or QULSConfig()
        self.weight_controller = AdaptiveWeightController(self.config)
        self._loss_components: dict[str, float] = {}
        self._step = 0
        self._total_steps = 1

        # Build primary loss function
        self._primary_loss_fn = self._build_primary_loss_fn()

        logger.info(
            f"[QULS] Initialized with primary_loss={self.config.primary_loss}, "
            f"fidelity={self.config.fidelity_enabled}, "
            f"born_rule={self.config.born_rule_enabled}, "
            f"entropy={self.config.entropy_enabled}, "
            f"coherence={self.config.coherence_enabled}, "
            f"symplectic={self.config.symplectic_enabled}, "
            f"entanglement={self.config.entanglement_enabled}"
        )

    def _build_primary_loss_fn(self):
        """Build the primary task loss function.

        Returns:
            Callable loss function (labels, predictions) -> loss.
        """
        loss_type = self.config.primary_loss.lower()
        label_smoothing = self.config.label_smoothing

        if loss_type == "sparse_categorical_crossentropy":
            if label_smoothing > 0:
                # With label smoothing, need to convert to one-hot first
                def loss_fn(labels, predictions):
                    vocab_size = tf.shape(predictions)[-1]
                    one_hot = tf.one_hot(labels, depth=vocab_size)
                    # Apply label smoothing
                    smoothed = one_hot * (1 - label_smoothing) + label_smoothing / tf.cast(
                        vocab_size, tf.float32
                    )
                    loss = tf.keras.losses.categorical_crossentropy(
                        smoothed, predictions, from_logits=True
                    )
                    return tf.reduce_mean(loss)

                return loss_fn
            else:

                def loss_fn(labels, predictions):
                    loss = tf.keras.losses.sparse_categorical_crossentropy(
                        labels, predictions, from_logits=True
                    )
                    return tf.reduce_mean(loss)

                return loss_fn

        elif loss_type == "categorical_crossentropy":

            def loss_fn(labels, predictions):
                loss = tf.keras.losses.categorical_crossentropy(
                    labels, predictions, from_logits=True, label_smoothing=label_smoothing
                )
                return tf.reduce_mean(loss)

            return loss_fn

        elif loss_type == "mse":

            def loss_fn(labels, predictions):
                return tf.reduce_mean(tf.square(predictions - labels))

            return loss_fn

        elif loss_type == "mae":

            def loss_fn(labels, predictions):
                return tf.reduce_mean(tf.abs(predictions - labels))

            return loss_fn

        else:
            logger.warning(
                f"[QULS] Unknown loss type '{loss_type}', defaulting to sparse_categorical_crossentropy"
            )

            def loss_fn(labels, predictions):
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    labels, predictions, from_logits=True
                )
                return tf.reduce_mean(loss)

            return loss_fn

    def set_training_state(self, step: int, total_steps: int):
        """Set current training state for adaptive weighting.

        Args:
            step: Current training step.
            total_steps: Total training steps.
        """
        self._step = step
        self._total_steps = total_steps

    def compute_loss(
        self,
        predictions: tf.Tensor,
        targets: tf.Tensor,
        hidden_states: tf.Tensor | None = None,
        vqc_outputs: tf.Tensor | None = None,
        coherence_metrics: dict[str, tf.Tensor] | tf.Tensor | None = None,
        hamiltonian_energies: tuple[tf.Tensor, tf.Tensor] | None = None,
        bond_entropies: tf.Tensor | list[tf.Tensor] | None = None,
        vqc_gradient_variance: float | None = None,
        gradient_entropy: float | None = None,
        in_barren_plateau: bool = False,
        zne_statistics: dict | None = None,  # S23: Neural ZNE feedback
    ) -> tuple[tf.Tensor, dict[str, float]]:
        """Compute total QULS loss with all components.

        Args:
            predictions: Model output logits [batch, seq, vocab] or [batch, vocab].
            targets: Target token IDs [batch, seq] or [batch].
            hidden_states: Hidden states for entropy computation [batch, seq, dim].
            vqc_outputs: VQC layer outputs for Born rule [batch, dim].
            coherence_metrics: Coherence metrics from QCB or single tensor.
            hamiltonian_energies: Tuple of (initial_energy, final_energy) for symplectic.
            bond_entropies: MPS bond entropies for entanglement regularization.
            vqc_gradient_variance: VQC gradient variance for adaptive weights.
            gradient_entropy: Overall gradient entropy for adaptive weights.
            in_barren_plateau: Whether barren plateau is detected.

        Returns:
            Tuple of:
            - total_loss: Scalar total loss tensor.
            - components: Dict of individual loss component values.
        """
        if not self.config.enabled:
            # QULS disabled, return standard cross-entropy
            primary_loss = self._primary_loss_fn(targets, predictions)
            return primary_loss, {"primary": float(primary_loss.numpy())}

        # Update adaptive weights
        weights = self.weight_controller.update(
            step=self._step,
            total_steps=self._total_steps,
            vqc_gradient_variance=vqc_gradient_variance,
            gradient_entropy=gradient_entropy,
            in_barren_plateau=in_barren_plateau,
            entanglement_entropy=(
                float(tf.reduce_mean(bond_entropies).numpy())
                if bond_entropies is not None
                else None
            ),
        )

        # S23: Neural ZNE feedback - adjust weights based on ZNE effectiveness
        components: dict[str, float] = {}
        zne_scale = 1.0
        if zne_statistics is not None and zne_statistics.get("enabled", False):
            num_mitigators = zne_statistics.get("num_mitigators", 0)
            if num_mitigators > 0:
                # When ZNE is actively mitigating errors, boost quantum terms
                # More mitigators = more quantum layers = higher quantum contribution
                zne_scale = 1.0 + 0.1 * min(num_mitigators, 5)
                # Record in components
                components["zne_mitigators"] = float(num_mitigators)
                components["zne_scale"] = zne_scale
                # Apply scale to quantum terms
                for key in ["fidelity", "born_rule", "coherence", "entanglement"]:
                    if key in weights:
                        weights[key] *= zne_scale

        total_loss = tf.constant(0.0, dtype=tf.float32)

        # 1. Primary task loss (always computed)
        primary_loss = self._primary_loss_fn(targets, predictions)
        total_loss = total_loss + primary_loss
        components["primary"] = float(primary_loss.numpy())

        # 2. Quantum Fidelity Loss
        if self.config.fidelity_enabled and weights["fidelity"] > 0:
            fidelity_loss = quantum_fidelity_loss(
                predictions, targets, temperature=self.config.fidelity_temperature
            )
            total_loss = total_loss + weights["fidelity"] * fidelity_loss
            components["fidelity"] = float(fidelity_loss.numpy())
            components["fidelity_weight"] = weights["fidelity"]

        # 3. Born Rule Loss
        if self.config.born_rule_enabled and weights["born_rule"] > 0 and vqc_outputs is not None:
            born_loss = born_rule_amplitude_loss(vqc_outputs)
            total_loss = total_loss + weights["born_rule"] * born_loss
            components["born_rule"] = float(born_loss.numpy())
            components["born_rule_weight"] = weights["born_rule"]

        # 4. Entropy Regularization
        if self.config.entropy_enabled and weights["entropy"] > 0 and hidden_states is not None:
            entropy_loss = entropy_regularization_loss(
                hidden_states,
                target_entropy=self.config.target_entropy,
                num_eigenvalues=min(10, hidden_states.shape[-1] // 10 + 1),
                power_iter_steps=self.config.power_iter_steps,
            )
            total_loss = total_loss + weights["entropy"] * entropy_loss
            components["entropy"] = float(entropy_loss.numpy())
            components["entropy_weight"] = weights["entropy"]

        # 5. Spectral Flatness
        if self.config.entropy_enabled and weights["spectral"] > 0 and hidden_states is not None:
            spectral_loss = spectral_flatness_loss(
                hidden_states,
                target_flatness=self.config.spectral_target,
                num_eigenvalues=min(10, hidden_states.shape[-1] // 10 + 1),
                power_iter_steps=self.config.power_iter_steps,
            )
            total_loss = total_loss + weights["spectral"] * spectral_loss
            components["spectral"] = float(spectral_loss.numpy())
            components["spectral_weight"] = weights["spectral"]

        # 6. Coherence Preservation
        if (
            self.config.coherence_enabled
            and weights["coherence"] > 0
            and coherence_metrics is not None
        ):
            coherence_loss = coherence_preservation_loss(
                coherence_metrics, threshold=self.config.coherence_threshold
            )
            total_loss = total_loss + weights["coherence"] * coherence_loss
            components["coherence"] = float(coherence_loss.numpy())
            components["coherence_weight"] = weights["coherence"]

        # 7. Symplectic Conservation
        if (
            self.config.symplectic_enabled
            and weights["symplectic"] > 0
            and hamiltonian_energies is not None
        ):
            h_init, h_final = hamiltonian_energies
            symplectic_loss = symplectic_energy_loss(h_init, h_final)
            total_loss = total_loss + weights["symplectic"] * symplectic_loss
            components["symplectic"] = float(symplectic_loss.numpy())
            components["symplectic_weight"] = weights["symplectic"]

        # 8. Entanglement Regularization
        if (
            self.config.entanglement_enabled
            and weights["entanglement"] > 0
            and bond_entropies is not None
        ):
            entanglement_loss = entanglement_regularization_loss(
                bond_entropies, min_entropy=self.config.min_bond_entropy
            )
            total_loss = total_loss + weights["entanglement"] * entanglement_loss
            components["entanglement"] = float(entanglement_loss.numpy())
            components["entanglement_weight"] = weights["entanglement"]

        # Record total
        components["total"] = float(total_loss.numpy())
        self._loss_components = components

        return total_loss, components

    def get_last_components(self) -> dict[str, float]:
        """Get loss components from last compute_loss call.

        Returns:
            Dictionary of loss component values.
        """
        return dict(self._loss_components)

    def get_weight_statistics(self) -> dict[str, Any]:
        """Get weight controller statistics.

        Returns:
            Dictionary with controller state and metrics.
        """
        return self.weight_controller.get_statistics()

    def __call__(
        self,
        predictions: tf.Tensor,
        targets: tf.Tensor,
        **kwargs,
    ) -> tf.Tensor:
        """Callable interface for QULS.

        Args:
            predictions: Model output logits.
            targets: Target token IDs.
            **kwargs: Additional arguments passed to compute_loss.

        Returns:
            Total loss tensor.
        """
        loss, _ = self.compute_loss(predictions, targets, **kwargs)
        return loss


# =============================================================================
# Factory Functions
# =============================================================================


def create_quls_loss(
    config: QULSConfig | dict[str, Any] | None = None,
) -> QuantumUnifiedLoss:
    """Factory function to create QULS instance.

    Args:
        config: QULSConfig, dict of config values, or None for defaults.

    Returns:
        Configured QuantumUnifiedLoss instance.

    Example:
        >>> quls = create_quls_loss({"fidelity_weight": 0.02})
        >>> loss = quls(predictions, targets)
    """
    if config is None:
        quls_config = QULSConfig()
    elif isinstance(config, dict):
        # Create config from dict
        quls_config = QULSConfig()
        for key, value in config.items():
            if hasattr(quls_config, key):
                setattr(quls_config, key, value)
    else:
        quls_config = config

    return QuantumUnifiedLoss(quls_config)


def create_quls_from_hpo_config(hpo_config: dict[str, Any]) -> QuantumUnifiedLoss:
    """Create QULS from HPO sweep configuration.

    Maps HPO parameter names to QULS config fields.

    Args:
        hpo_config: Dictionary from HPO sweep.

    Returns:
        Configured QuantumUnifiedLoss instance.

    Example:
        >>> hpo_config = {
        ...     "quls_fidelity_weight": 0.02,
        ...     "quls_entropy_weight": 0.01,
        ...     "quls_target_entropy": 0.6,
        ... }
        >>> quls = create_quls_from_hpo_config(hpo_config)
    """
    # HPO parameter name to QULSConfig field mapping
    field_mappings = {
        "quls_enabled": "enabled",
        "quls_primary_loss": "primary_loss",
        "quls_label_smoothing": "label_smoothing",
        "quls_fidelity_enabled": "fidelity_enabled",
        "quls_fidelity_weight": "fidelity_weight",
        "quls_fidelity_temperature": "fidelity_temperature",
        "quls_born_rule_enabled": "born_rule_enabled",
        "quls_born_rule_weight": "born_rule_weight",
        "quls_entropy_enabled": "entropy_enabled",
        "quls_entropy_weight": "entropy_weight",
        "quls_target_entropy": "target_entropy",
        "quls_spectral_weight": "spectral_weight",
        "quls_spectral_target": "spectral_target",
        "quls_coherence_enabled": "coherence_enabled",
        "quls_coherence_weight": "coherence_weight",
        "quls_coherence_threshold": "coherence_threshold",
        "quls_symplectic_enabled": "symplectic_enabled",
        "quls_symplectic_weight": "symplectic_weight",
        "quls_entanglement_enabled": "entanglement_enabled",
        "quls_entanglement_weight": "entanglement_weight",
        "quls_min_bond_entropy": "min_bond_entropy",
        "quls_adaptive_weights": "adaptive_weights",
        "quls_weight_ema_decay": "weight_ema_decay",
        "quls_vqc_variance_boost": "vqc_variance_boost",
        "quls_barren_plateau_reduction": "barren_plateau_reduction",
    }

    config_dict = {}
    for hpo_key, config_key in field_mappings.items():
        if hpo_key in hpo_config:
            config_dict[config_key] = hpo_config[hpo_key]

    return create_quls_loss(config_dict)


# =============================================================================
# Backward-Compatibility Alias (HIGHNOON_UPGRADE_ROADMAP.md Section 1.1)
# =============================================================================

# Alias for compatibility with code importing UnifiedQuantumLoss from V2
UnifiedQuantumLoss = QuantumUnifiedLoss
