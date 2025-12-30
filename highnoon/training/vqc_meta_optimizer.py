# highnoon/training/vqc_meta_optimizer.py
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

"""VQC Meta-Optimizer for Quantum-Native Unified Tuning.

This module provides a Variational Quantum Circuit (VQC) based meta-optimizer
that learns optimal tuning policies from training dynamics. It replaces
heuristic decision-making with quantum-learned policies.

Key Features:
    - Amplitude encoding of training state (loss, grad_norm, Fisher info)
    - VQC-computed tuning decisions (LR, GaLore rank, clip norm, etc.)
    - Integration with native C++ VQC ops for high performance

Note:
    This module requires the native C++ VQC expectation ops to be compiled.
    See highnoon/_native/ops/vqc_expectation.py for the wrapper.

Example:
    >>> config = VQCMetaOptimizerConfig(num_qubits=8, num_layers=4)
    >>> vqc_meta = VQCMetaOptimizer(config)
    >>> decisions = vqc_meta.compute_tuning_decisions(
    ...     loss=2.5, gradient_norm=1.0,
    ...     layer_fisher_info={"layer1": 0.5},
    ...     barren_plateau_scores={"layer1": 0.1},
    ...     exploration_factor=0.8,
    ... )

Reference:
    Smart_Tuner_Upgrade.md - Section 6.2: VQC-Driven Meta-Optimizer
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class VQCMetaOptimizerConfig:
    """Configuration for VQC-based meta-optimization.

    Attributes:
        num_qubits: Number of qubits in the VQC. Determines state space size.
        num_layers: Number of VQC layers (depth). More layers = more expressivity.
        entangler: Entanglement topology ("linear", "circular", "all-to-all").
        use_amplitude_encoding: Use amplitude encoding vs angle encoding.
        normalize_inputs: Normalize inputs before encoding.
        num_output_observables: Number of output observables (tuning params).
        meta_lr: Learning rate for VQC parameter updates.
        update_frequency: Update VQC parameters every N training steps.
        use_rotosolve: Use gradient-free Rotosolve optimization for VQC.
        experience_buffer_size: Size of experience buffer for meta-learning.
        use_hd_fisher_compression: Use HD holographic bundling for Fisher info.
        hd_dim: HD space dimension for Fisher compression.
        hd_out_dim: Output dimension for compressed Fisher info.
    """

    # VQC architecture
    num_qubits: int = 8
    num_layers: int = 4
    entangler: str = "linear"  # "linear", "circular", "all-to-all"

    # Input encoding
    use_amplitude_encoding: bool = True
    normalize_inputs: bool = True

    # Output decoding
    num_output_observables: int = 6

    # Training
    meta_lr: float = 1e-3
    update_frequency: int = 50
    use_rotosolve: bool = True

    # Experience buffer
    experience_buffer_size: int = 1000

    # HD Fisher Compression (VQC-HD Enhancement #1)
    use_hd_fisher_compression: bool = False  # Opt-in via config.USE_HD_FISHER_COMPRESSION
    hd_dim: int = 4096
    hd_out_dim: int = 64


# =============================================================================
# META EXPERIENCE
# =============================================================================


@dataclass
class MetaExperience:
    """Record of a tuning decision and its outcome.

    Used for meta-learning VQC parameters based on training results.

    Attributes:
        step: Training step when decision was made.
        input_features: Encoded input features.
        decisions: Tuning decisions that were made.
        loss_before: Loss before applying decisions.
        loss_after: Loss after applying decisions.
        reward: Computed reward signal for meta-learning.
    """

    step: int
    input_features: np.ndarray
    decisions: dict[str, float]
    loss_before: float
    loss_after: float
    reward: float = 0.0

    def compute_reward(self) -> float:
        """Compute reward based on loss improvement.

        Returns:
            Reward signal (positive = loss decreased).
        """
        if self.loss_before > 1e-8:
            improvement = (self.loss_before - self.loss_after) / self.loss_before
            self.reward = np.tanh(improvement * 10.0)  # Scaled tanh
        else:
            self.reward = 0.0
        return self.reward


# =============================================================================
# VQC TUNING DECISIONS
# =============================================================================


@dataclass
class VQCTuningDecisions:
    """Container for VQC-computed tuning decisions.

    Attributes:
        learning_rate_multiplier: Multiplier for base learning rate.
        galore_rank_scale: Scale factor for GaLore rank.
        max_grad_norm_multiplier: Multiplier for gradient clipping norm.
        exploration_boost: Additional exploration factor.
        evolution_time_scale: Scale for meta-controller evolution time.
        barren_plateau_threshold_scale: Scale for BP threshold.
        vqc_computed: Whether these decisions came from VQC (vs fallback).
        raw_expectations: Raw VQC expectation values.
    """

    learning_rate_multiplier: float = 1.0
    galore_rank_scale: float = 1.0
    max_grad_norm_multiplier: float = 1.0
    exploration_boost: float = 0.0
    evolution_time_scale: float = 1.0
    barren_plateau_threshold_scale: float = 1.0
    vqc_computed: bool = False
    raw_expectations: np.ndarray = field(default_factory=lambda: np.zeros(6))


# =============================================================================
# VQC META-OPTIMIZER
# =============================================================================


class VQCMetaOptimizer:
    """VQC that outputs tuning decisions from training state.

    The VQC takes as input:
        - Encoded training metrics (loss, grad_norm, entropy, etc.)
        - Layer-wise Fisher Information estimates
        - Barren plateau indicators

    And outputs:
        - Global learning rate multiplier
        - Per-layer-group learning rate scales
        - GaLore rank allocation weights
        - Gradient clipping thresholds
        - Exploration/exploitation balance
        - Meta-controller evolution time

    This creates a closed-loop quantum control system where the VQC
    learns optimal tuning policies through training experience.

    Attributes:
        config: VQC meta-optimizer configuration.
        circuit_params: Trainable VQC rotation parameters.
        entangler_pairs: Fixed entanglement topology.

    Example:
        >>> optimizer = VQCMetaOptimizer()
        >>> decisions = optimizer.compute_tuning_decisions(
        ...     loss=2.5, gradient_norm=1.0,
        ...     layer_fisher_info={}, barren_plateau_scores={},
        ...     exploration_factor=0.8,
        ... )
        >>> print(f"LR multiplier: {decisions.learning_rate_multiplier:.2f}")
    """

    def __init__(
        self,
        config: VQCMetaOptimizerConfig | None = None,
    ):
        """Initialize VQC Meta-Optimizer.

        Args:
            config: Configuration for VQC architecture and training.
        """
        self.config = config or VQCMetaOptimizerConfig()

        # Initialize VQC parameters (rotation angles)
        self._init_circuit_params()

        # Experience buffer for meta-learning
        self._experience_buffer: deque[MetaExperience] = deque(
            maxlen=self.config.experience_buffer_size
        )

        # Step counter
        self._step = 0
        self._last_update_step = 0

        # Check for native VQC ops
        self._use_native_vqc = self._check_native_vqc_available()

        # Measurement observables (Pauli-Z on each output qubit)
        self._init_measurement_operators()

        logger.info(
            "[VQCMetaOptimizer] Initialized: %d qubits, %d layers, native=%s",
            self.config.num_qubits,
            self.config.num_layers,
            self._use_native_vqc,
        )

    def _check_native_vqc_available(self) -> bool:
        """Check if native C++ VQC ops are available.

        Raises:
            RuntimeError: If native VQC ops are not available.
        """
        try:
            from highnoon._native.ops.vqc_expectation import vqc_expectation_available

            available = vqc_expectation_available()
            if available:
                logger.info("[VQCMetaOptimizer] Using native C++ VQC ops")
                return True
            else:
                raise RuntimeError(
                    "VQC Meta-Optimizer requires native C++ VQC ops. "
                    "Please compile the vqc_expectation op."
                )
        except ImportError as e:
            raise RuntimeError(f"VQC Meta-Optimizer requires vqc_expectation module: {e}") from e

    def _init_circuit_params(self) -> None:
        """Initialize trainable VQC parameters.

        Uses Xavier/Glorot initialization to prevent barren plateaus.
        """
        n_params = self.config.num_qubits * self.config.num_layers * 3  # RY, RZ, RX

        # Xavier/Glorot initialization (prevents barren plateaus)
        scale = np.sqrt(2.0 / (self.config.num_qubits + self.config.num_layers))

        self.circuit_params = tf.Variable(
            tf.random.uniform([n_params], -scale * np.pi, scale * np.pi),
            trainable=True,
            name="vqc_meta_optimizer/circuit_params",
        )

        # Build entangler topology
        self.entangler_pairs = self._build_entangler_topology()

        logger.debug("[VQCMetaOptimizer] Initialized %d circuit parameters", n_params)

    def _build_entangler_topology(self) -> tf.Tensor:
        """Build entanglement pairs based on topology config.

        Returns:
            Tensor of shape [num_pairs, 2] with qubit index pairs.
        """
        n = self.config.num_qubits
        pairs = []

        if self.config.entangler == "linear":
            # Linear chain: (0,1), (1,2), ..., (n-2, n-1)
            pairs = [(i, i + 1) for i in range(n - 1)]

        elif self.config.entangler == "circular":
            # Circular: linear + (n-1, 0)
            pairs = [(i, (i + 1) % n) for i in range(n)]

        elif self.config.entangler == "all-to-all":
            # All-to-all connectivity
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        else:
            logger.warning(
                "[VQCMetaOptimizer] Unknown entangler '%s', using linear",
                self.config.entangler,
            )
            pairs = [(i, i + 1) for i in range(n - 1)]

        return tf.constant(pairs, dtype=tf.int32)

    def _init_measurement_operators(self) -> None:
        """Initialize Pauli measurement operators for output observables."""
        n_outputs = min(self.config.num_output_observables, self.config.num_qubits)

        # Measure Z on first n_outputs qubits
        # paulis: 0=I, 1=X, 2=Y, 3=Z
        self._measurement_paulis = tf.constant(
            [[3 if j == i else 0 for j in range(self.config.num_qubits)] for i in range(n_outputs)],
            dtype=tf.int32,
        )

        # Coefficients (all 1.0 for simple Z measurements)
        self._measurement_coeffs = tf.ones([n_outputs], dtype=tf.float32)

    def _encode_training_state(
        self,
        loss: float,
        gradient_norm: float,
        layer_fisher_info: dict[str, float],
        barren_plateau_scores: dict[str, float],
        exploration_factor: float,
    ) -> tf.Tensor:
        """Encode training state into VQC input angles.

        Uses amplitude encoding for efficiency:
            |ψ⟩ = Σ_i α_i|i⟩ where α_i encodes training metrics

        VQC-HD Enhancement #1: When use_hd_fisher_compression is enabled,
        layer Fisher info is compressed via holographic bundling for 10-50x
        memory reduction while preserving layer correlation structure.

        Args:
            loss: Current training loss.
            gradient_norm: Current gradient norm.
            layer_fisher_info: Dict of Fisher info per layer.
            barren_plateau_scores: Dict of BP scores per layer.
            exploration_factor: Current exploration factor (0-1).

        Returns:
            Tensor of encoded features for VQC input.
        """
        # Base features (always included)
        base_features = [
            np.tanh(loss / 100.0),  # Normalized loss
            np.tanh(np.log1p(gradient_norm) / 10.0),  # Log-normalized grad norm
            exploration_factor,
            np.mean(list(barren_plateau_scores.values())) if barren_plateau_scores else 0.0,
            np.max(list(barren_plateau_scores.values())) if barren_plateau_scores else 0.0,
            1.0 if gradient_norm > 1e3 else 0.0,  # Emergency indicator
        ]

        # HD Fisher compression path
        if self.config.use_hd_fisher_compression and layer_fisher_info:
            try:
                from highnoon._native.ops.hd_fisher_compression import (
                    hd_fisher_compress,
                    is_native_available,
                )

                # Convert Fisher dict to tensor
                fisher_values = tf.constant(list(layer_fisher_info.values()), dtype=tf.float32)

                # Initialize position keys if not already done
                if not hasattr(self, "_hd_fisher_keys"):
                    from highnoon._native.ops.hd_fisher_compression import create_position_keys

                    max_layers = max(100, len(layer_fisher_info))  # Support up to 100 layers
                    self._hd_fisher_keys = create_position_keys(
                        max_layers, self.config.hd_dim, seed=42
                    )
                    self._hd_proj_weights = tf.Variable(
                        tf.random.truncated_normal(
                            [self.config.hd_dim, self.config.hd_out_dim], stddev=0.01
                        ),
                        trainable=False,
                        name="vqc_meta_optimizer/hd_proj",
                    )

                # Compress Fisher info via holographic bundling
                num_layers = len(layer_fisher_info)
                compressed = hd_fisher_compress(
                    fisher_values,
                    self._hd_fisher_keys[:num_layers],
                    self._hd_proj_weights,
                    hd_dim=self.config.hd_dim,
                    out_dim=self.config.hd_out_dim,
                    normalize=True,
                    use_native=is_native_available(),
                )

                # Append compressed Fisher representation to features
                compressed_np = (
                    compressed.numpy() if hasattr(compressed, "numpy") else np.array(compressed)
                )
                base_features.extend(compressed_np.tolist())

                logger.debug(
                    "[VQCMetaOptimizer] HD Fisher compression: %d layers -> %d dims",
                    num_layers,
                    len(compressed_np),
                )

            except (ImportError, Exception) as e:
                # Fall back to simple mean/std encoding
                logger.debug("[VQCMetaOptimizer] HD compression unavailable, using fallback: %s", e)
                base_features.extend(
                    [
                        np.mean(list(layer_fisher_info.values())),
                        (
                            np.std(list(layer_fisher_info.values()))
                            if len(layer_fisher_info) > 1
                            else 0.0
                        ),
                    ]
                )
        else:
            # Original simple encoding
            base_features.extend(
                [
                    np.mean(list(layer_fisher_info.values())) if layer_fisher_info else 0.5,
                    np.std(list(layer_fisher_info.values())) if len(layer_fisher_info) > 1 else 0.0,
                ]
            )

        # Pad to power of 2 for amplitude encoding
        n_amplitudes = 2**self.config.num_qubits
        features = base_features + [0.0] * (n_amplitudes - len(base_features))

        # Normalize to unit norm (valid quantum state)
        features = np.array(features[:n_amplitudes], dtype=np.float32)
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features = features / norm
        else:
            # Default to uniform superposition
            features = np.ones(n_amplitudes, dtype=np.float32) / np.sqrt(n_amplitudes)

        return tf.constant(features, dtype=tf.float32)

    def compute_tuning_decisions(
        self,
        loss: float,
        gradient_norm: float,
        layer_fisher_info: dict[str, float],
        barren_plateau_scores: dict[str, float],
        exploration_factor: float,
    ) -> VQCTuningDecisions:
        """Run VQC to compute tuning decisions.

        Args:
            loss: Current training loss.
            gradient_norm: Current gradient norm.
            layer_fisher_info: Dict mapping layer names to Fisher info.
            barren_plateau_scores: Dict mapping layer names to BP scores.
            exploration_factor: Current exploration factor (0-1).

        Returns:
            VQCTuningDecisions with VQC-computed parameters.
        """
        # Encode training state
        data_angles = self._encode_training_state(
            loss,
            gradient_norm,
            layer_fisher_info,
            barren_plateau_scores,
            exploration_factor,
        )

        # Run native VQC (required)
        expectations = self._run_native_vqc(data_angles)

        # Decode VQC outputs to tuning decisions
        return self._decode_expectations(expectations)

    def _run_native_vqc(self, data_angles: tf.Tensor) -> tf.Tensor:
        """Run VQC using native C++ operator.

        Args:
            data_angles: Encoded training state angles.

        Returns:
            Expectation values tensor.
        """
        from highnoon._native.ops.vqc_expectation import run_vqc_expectation

        expectations = run_vqc_expectation(
            data_angles=data_angles,
            circuit_params=self.circuit_params,
            entangler_pairs=self.entangler_pairs,
            measurement_paulis=self._measurement_paulis,
            measurement_coeffs=self._measurement_coeffs,
        )
        return expectations

    def _decode_expectations(self, expectations: tf.Tensor) -> VQCTuningDecisions:
        """Decode VQC expectation values to tuning parameters.

        Maps expectation values (in [-1, 1]) to appropriate tuning ranges.

        Args:
            expectations: Tensor of expectation values.

        Returns:
            VQCTuningDecisions with decoded parameters.
        """
        exp_np = expectations.numpy() if hasattr(expectations, "numpy") else np.array(expectations)

        # Ensure we have enough expectation values
        while len(exp_np) < 6:
            exp_np = np.append(exp_np, 0.0)

        # Each expectation value maps to a tuning parameter
        # Output range is [-1, 1], we map to appropriate ranges

        # LR multiplier: exp[0] -> [0.1, 10.0] (log scale)
        lr_mult = float(10 ** (exp_np[0] * 1.0))  # [-1,1] -> [0.1, 10]
        lr_mult = np.clip(lr_mult, 0.1, 10.0)

        # GaLore rank scale: exp[1] -> [0.25, 4.0]
        galore_rank_scale = float(2 ** (exp_np[1] * 2.0))  # [-1,1] -> [0.25, 4]
        galore_rank_scale = np.clip(galore_rank_scale, 0.25, 4.0)

        # Grad clip relaxation: exp[2] -> [0.5, 2.0]
        grad_clip_mult = float(1.0 + exp_np[2] * 0.5)  # [-1,1] -> [0.5, 1.5]
        grad_clip_mult = np.clip(grad_clip_mult, 0.5, 2.0)

        # Exploration boost: exp[3] -> additional exploration
        exploration_boost = float(max(0, exp_np[3]))

        # Evolution time adjustment for meta-controller
        evolution_time_scale = float(1.0 + exp_np[4] * 0.3)  # [-1,1] -> [0.7, 1.3]
        evolution_time_scale = np.clip(evolution_time_scale, 0.7, 1.3)

        # Barren plateau aggressiveness: exp[5] -> mitigation threshold
        bp_threshold_scale = float(1.0 - exp_np[5] * 0.5)  # [-1,1] -> [0.5, 1.5]
        bp_threshold_scale = np.clip(bp_threshold_scale, 0.5, 1.5)

        return VQCTuningDecisions(
            learning_rate_multiplier=lr_mult,
            galore_rank_scale=galore_rank_scale,
            max_grad_norm_multiplier=grad_clip_mult,
            exploration_boost=exploration_boost,
            evolution_time_scale=evolution_time_scale,
            barren_plateau_threshold_scale=bp_threshold_scale,
            vqc_computed=True,
            raw_expectations=exp_np,
        )

    def record_experience(
        self,
        step: int,
        input_features: np.ndarray,
        decisions: dict[str, float],
        loss_before: float,
        loss_after: float,
    ) -> None:
        """Record an experience for meta-learning.

        Args:
            step: Training step.
            input_features: Encoded input features.
            decisions: Tuning decisions that were made.
            loss_before: Loss before applying decisions.
            loss_after: Loss after applying decisions.
        """
        experience = MetaExperience(
            step=step,
            input_features=input_features,
            decisions=decisions,
            loss_before=loss_before,
            loss_after=loss_after,
        )
        experience.compute_reward()
        self._experience_buffer.append(experience)

    def maybe_update_params(self, step: int) -> bool:
        """Potentially update VQC parameters from experience.

        Uses Rotosolve or gradient descent depending on config.

        Args:
            step: Current training step.

        Returns:
            True if parameters were updated.
        """
        if step - self._last_update_step < self.config.update_frequency:
            return False

        if len(self._experience_buffer) < 10:
            return False

        self._last_update_step = step

        if self.config.use_rotosolve:
            self._rotosolve_update()
        else:
            self._gradient_update()

        return True

    def _rotosolve_update(self) -> None:
        """Update VQC parameters using Rotosolve algorithm.

        Rotosolve is a gradient-free optimization for quantum circuits.
        """
        # Get recent experiences with positive reward
        recent = [e for e in list(self._experience_buffer)[-20:] if e.reward > 0]

        if not recent:
            return

        # For each parameter, find optimal rotation using reward signal
        params_np = self.circuit_params.numpy()
        n_params = len(params_np)

        # Sample a subset of parameters to update
        n_update = min(5, n_params)
        param_indices = np.random.choice(n_params, n_update, replace=False)

        for idx in param_indices:
            # Simplified Rotosolve: adjust based on reward-weighted gradient estimate
            total_shift = 0.0
            total_weight = 0.0

            for exp in recent:
                # Estimate gradient direction from reward
                shift = exp.reward * self.config.meta_lr
                total_shift += shift
                total_weight += abs(exp.reward)

            if total_weight > 1e-8:
                params_np[idx] += total_shift / total_weight

        self.circuit_params.assign(params_np)
        logger.debug("[VQCMetaOptimizer] Rotosolve updated %d parameters", n_update)

    def _gradient_update(self) -> None:
        """Update VQC parameters using gradient descent."""
        # Simplified gradient update based on experience buffer
        recent = list(self._experience_buffer)[-20:]

        if not recent:
            return

        # Compute average reward
        avg_reward = np.mean([e.reward for e in recent])

        # Simple parameter perturbation based on reward sign
        params_np = self.circuit_params.numpy()
        noise = np.random.randn(*params_np.shape) * self.config.meta_lr
        params_np += noise * np.sign(avg_reward)

        self.circuit_params.assign(params_np)
        logger.debug("[VQCMetaOptimizer] Gradient updated parameters")

    def get_statistics(self) -> dict[str, Any]:
        """Get VQC meta-optimizer statistics.

        Returns:
            Dictionary with current state and performance metrics.
        """
        stats = {
            "num_qubits": self.config.num_qubits,
            "num_layers": self.config.num_layers,
            "num_params": len(self.circuit_params.numpy()),
            "using_native_vqc": self._use_native_vqc,
            "experience_buffer_size": len(self._experience_buffer),
            "last_update_step": self._last_update_step,
        }

        if self._experience_buffer:
            rewards = [e.reward for e in self._experience_buffer]
            stats["mean_reward"] = float(np.mean(rewards))
            stats["std_reward"] = float(np.std(rewards))
            stats["positive_reward_ratio"] = float(sum(1 for r in rewards if r > 0) / len(rewards))

        return stats

    def reset(self) -> None:
        """Reset VQC meta-optimizer state."""
        self._experience_buffer.clear()
        self._step = 0
        self._last_update_step = 0
        self._init_circuit_params()
        logger.info("[VQCMetaOptimizer] State reset")


__all__ = [
    "VQCMetaOptimizer",
    "VQCMetaOptimizerConfig",
    "VQCTuningDecisions",
    "MetaExperience",
]
