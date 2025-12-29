# highnoon/training/barren_plateau_v2.py
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

"""Barren Plateau Prevention System - Phase 5 Implementation.

Integrates multi-pronged barren plateau mitigation based on 2024-2025 research:

1. Initialization Strategies
   - Xavier/He-style initialization for quantum circuits
   - Identity-biased initialization (small rotations)

2. Optimization Techniques
   - Layerwise training (progressive unfreezing)
   - Local cost functions

3. Architecture Design
   - Locality-preserving circuits
   - Depth limiting based on gradient health

4. Monitoring
   - Real-time gradient variance tracking
   - Early warning system with actionable recommendations

References:
    - "Investigating and Mitigating Barren Plateaus" (arXiv:2407.17706, 2024)
    - "Geometric Optimization on Lie Groups" (arXiv:2512.02078, 2025)
    - "Engineered Dissipation to Mitigate Barren Plateaus" (Nature, 2024)

Reference:
    QUANTUM_ROADMAP.md Phase 5: Barren Plateau Prevention
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


# =============================================================================
# Initialization Strategies
# =============================================================================


class InitializationStrategy(Enum):
    """VQC parameter initialization strategies."""

    RANDOM_UNIFORM = "random_uniform"  # Default (prone to BP)
    XAVIER_QUANTUM = "xavier_quantum"  # Xavier-scaled, depth-aware
    HE_QUANTUM = "he_quantum"  # He-scaled for ReLU-like activations
    IDENTITY_BIAS = "identity_bias"  # Near-identity (small rotations)
    ORTHOGONAL = "orthogonal"  # Orthogonal initialization


def initialize_vqc_params(
    num_qubits: int,
    num_layers: int,
    params_per_gate: int = 3,
    strategy: InitializationStrategy = InitializationStrategy.XAVIER_QUANTUM,
    dtype: tf.DType = tf.float32,
) -> tf.Tensor:
    """Initialize VQC parameters with barren plateau mitigation.

    Based on research showing that classical initialization strategies
    (Xavier, He) reduce gradient variance decay in quantum circuits.

    Args:
        num_qubits: Number of qubits in circuit.
        num_layers: Number of variational layers.
        params_per_gate: Parameters per gate (typically 3 for U3).
        strategy: Initialization strategy to use.
        dtype: Data type for parameters.

    Returns:
        Initialized parameters [num_layers, num_qubits, params_per_gate].
    """
    shape = (num_layers, num_qubits, params_per_gate)

    if strategy == InitializationStrategy.XAVIER_QUANTUM:
        # Scale by 1/sqrt(depth) to preserve variance through layers
        scale = 1.0 / np.sqrt(num_layers)
        params = tf.random.uniform(
            shape,
            minval=-np.pi * scale,
            maxval=np.pi * scale,
            dtype=dtype,
        )

    elif strategy == InitializationStrategy.HE_QUANTUM:
        # He initialization: sqrt(2/fan_in)
        fan_in = num_qubits * params_per_gate
        scale = np.sqrt(2.0 / fan_in)
        params = tf.random.normal(shape, mean=0.0, stddev=scale, dtype=dtype)

    elif strategy == InitializationStrategy.IDENTITY_BIAS:
        # Start near identity (small rotations)
        params = tf.random.normal(shape, mean=0.0, stddev=0.1, dtype=dtype)

    elif strategy == InitializationStrategy.ORTHOGONAL:
        # Orthogonal initialization per layer
        layers = []
        for _ in range(num_layers):
            layer_params = tf.Variable(
                tf.initializers.Orthogonal()(shape=(num_qubits, params_per_gate))
            )
            layers.append(layer_params)
        params = tf.stack(layers, axis=0)

    else:  # RANDOM_UNIFORM (baseline)
        params = tf.random.uniform(
            shape,
            minval=-np.pi,
            maxval=np.pi,
            dtype=dtype,
        )

    return params


# =============================================================================
# Gradient Health Monitoring
# =============================================================================


@dataclass
class GradientHealth:
    """Gradient health metrics."""

    variance: float
    max_abs: float
    mean_abs: float
    zero_ratio: float
    status: str  # "healthy", "warning", "barren_plateau"


class GradientHealthMonitor:
    """Real-time gradient health monitoring for VQC training.

    Tracks gradient statistics over time and provides actionable
    recommendations when barren plateaus are detected.

    Attributes:
        warning_threshold: Variance below this triggers warning.
        critical_threshold: Variance below this indicates BP.
        history_size: Number of steps to track.
    """

    def __init__(
        self,
        warning_threshold: float = 1e-5,
        critical_threshold: float = 1e-7,
        history_size: int = 100,
    ):
        """Initialize gradient monitor.

        Args:
            warning_threshold: Variance threshold for warnings.
            critical_threshold: Variance threshold for BP detection.
            history_size: Number of history entries to keep.
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.history_size = history_size

        self._history: list[GradientHealth] = []
        self._consecutive_bp_count = 0

    def check(self, gradients: tf.Tensor) -> GradientHealth:
        """Analyze gradient health.

        Args:
            gradients: Gradient tensor (any shape).

        Returns:
            GradientHealth metrics and status.
        """
        grads_flat = tf.reshape(gradients, [-1])

        variance = float(tf.math.reduce_variance(grads_flat).numpy())
        max_abs = float(tf.reduce_max(tf.abs(grads_flat)).numpy())
        mean_abs = float(tf.reduce_mean(tf.abs(grads_flat)).numpy())
        zero_ratio = float(tf.reduce_mean(tf.cast(tf.abs(grads_flat) < 1e-10, tf.float32)).numpy())

        # Determine status
        if variance < self.critical_threshold:
            status = "barren_plateau"
            self._consecutive_bp_count += 1
        elif variance < self.warning_threshold:
            status = "warning"
            self._consecutive_bp_count = 0
        else:
            status = "healthy"
            self._consecutive_bp_count = 0

        health = GradientHealth(
            variance=variance,
            max_abs=max_abs,
            mean_abs=mean_abs,
            zero_ratio=zero_ratio,
            status=status,
        )

        # Update history
        self._history.append(health)
        if len(self._history) > self.history_size:
            self._history.pop(0)

        return health

    def is_in_barren_plateau(self) -> bool:
        """Check if currently in barren plateau.

        Returns True if last 3+ checks indicated BP.
        """
        return self._consecutive_bp_count >= 3

    def recommend_action(self) -> str:
        """Recommend action based on gradient history.

        Returns:
            Action recommendation string.
        """
        if not self._history:
            return "continue"

        # Check recent history
        recent = self._history[-5:] if len(self._history) >= 5 else self._history
        bp_count = sum(1 for h in recent if h.status == "barren_plateau")
        warning_count = sum(1 for h in recent if h.status == "warning")

        if bp_count >= 3:
            return "reduce_circuit_depth"
        elif bp_count >= 1 or warning_count >= 3:
            return "switch_to_local_cost"
        elif warning_count >= 1:
            return "increase_learning_rate"
        else:
            return "continue"

    def get_statistics(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        if not self._history:
            return {"num_samples": 0}

        variances = [h.variance for h in self._history]
        bp_count = sum(1 for h in self._history if h.status == "barren_plateau")

        return {
            "num_samples": len(self._history),
            "mean_variance": np.mean(variances),
            "min_variance": np.min(variances),
            "max_variance": np.max(variances),
            "barren_plateau_ratio": bp_count / len(self._history),
            "consecutive_bp_count": self._consecutive_bp_count,
            "current_status": self._history[-1].status if self._history else "unknown",
            "recommendation": self.recommend_action(),
        }

    def reset(self):
        """Reset monitor state."""
        self._history.clear()
        self._consecutive_bp_count = 0


# =============================================================================
# Layerwise Training
# =============================================================================


class LayerwiseTrainingScheduler:
    """Progressive layer unfreezing to avoid barren plateaus.

    Based on research showing that training layers progressively
    (shallow first, then deeper) avoids the exponential gradient
    decay of deep circuits.

    Attributes:
        total_layers: Total number of layers.
        warmup_epochs: Epochs before starting progressive training.
        layers_per_epoch: Layers to unfreeze per epoch.
    """

    def __init__(
        self,
        total_layers: int,
        warmup_epochs: int = 1,
        layers_per_epoch: int = 1,
    ):
        """Initialize scheduler.

        Args:
            total_layers: Total layers in circuit.
            warmup_epochs: Warmup epochs (train first layer only).
            layers_per_epoch: Layers to add per epoch after warmup.
        """
        self.total_layers = total_layers
        self.warmup_epochs = warmup_epochs
        self.layers_per_epoch = max(1, layers_per_epoch)

        self._current_epoch = 0

    def get_active_layers(self) -> int:
        """Get number of active (trainable) layers for current epoch.

        Returns:
            Number of layers to train.
        """
        if self._current_epoch < self.warmup_epochs:
            return 1

        # After warmup, progressively add layers
        epochs_past_warmup = self._current_epoch - self.warmup_epochs
        # +1 so first epoch after warmup adds layers_per_epoch immediately
        active = 1 + (epochs_past_warmup + 1) * self.layers_per_epoch

        return min(active, self.total_layers)

    def should_train_layer(self, layer_idx: int) -> bool:
        """Check if layer should be trained.

        Args:
            layer_idx: Layer index (0-indexed).

        Returns:
            True if layer should be trained.
        """
        return layer_idx < self.get_active_layers()

    def step_epoch(self):
        """Advance to next epoch."""
        self._current_epoch += 1

    def get_layer_mask(self) -> tf.Tensor:
        """Get boolean mask for active layers.

        Returns:
            Boolean tensor [total_layers] where True = train.
        """
        active = self.get_active_layers()
        mask = tf.concat(
            [
                tf.ones([active], dtype=tf.bool),
                tf.zeros([self.total_layers - active], dtype=tf.bool),
            ],
            axis=0,
        )
        return mask

    def get_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "current_epoch": self._current_epoch,
            "total_layers": self.total_layers,
            "active_layers": self.get_active_layers(),
            "progress": self.get_active_layers() / self.total_layers,
        }


# =============================================================================
# Local Cost Function
# =============================================================================


def compute_local_cost(
    global_observable: tf.Tensor,
    qubit_index: int,
    locality_radius: int = 2,
) -> tf.Tensor:
    """Convert global observable to local cost function.

    Local cost functions have polynomial (not exponential) gradient decay,
    making them more trainable for deep circuits.

    Args:
        global_observable: Full observable tensor [num_qubits] or [batch, num_qubits].
        qubit_index: Central qubit for local region.
        locality_radius: Number of neighboring qubits to include.

    Returns:
        Local cost value (scalar or [batch]).
    """
    if len(global_observable.shape) == 1:
        num_qubits = global_observable.shape[0]
        start = max(0, qubit_index - locality_radius)
        end = min(num_qubits, qubit_index + locality_radius + 1)
        local_obs = global_observable[start:end]
        return tf.reduce_mean(local_obs)
    else:
        # Batched
        num_qubits = global_observable.shape[1]
        start = max(0, qubit_index - locality_radius)
        end = min(num_qubits, qubit_index + locality_radius + 1)
        local_obs = global_observable[:, start:end]
        return tf.reduce_mean(local_obs, axis=-1)


def global_to_local_cost(
    loss_fn,
    num_qubits: int,
    locality_radius: int = 2,
):
    """Wrapper to convert global loss to sum of local losses.

    Creates a new loss function that computes local costs and aggregates.

    Args:
        loss_fn: Original global loss function.
        num_qubits: Number of qubits.
        locality_radius: Locality radius for each qubit.

    Returns:
        Local loss function.
    """

    def local_loss_fn(*args, **kwargs):
        global_loss = loss_fn(*args, **kwargs)

        # If loss already per-qubit, use local aggregation
        if len(global_loss.shape) >= 1 and global_loss.shape[-1] == num_qubits:
            local_losses = []
            for q in range(num_qubits):
                local = compute_local_cost(global_loss, q, locality_radius)
                local_losses.append(local)
            return tf.reduce_mean(tf.stack(local_losses))

        return global_loss

    return local_loss_fn


# =============================================================================
# Integrated Barren Plateau Mitigator
# =============================================================================


@dataclass
class BarrenPlateauConfig:
    """Configuration for barren plateau mitigation."""

    initialization: InitializationStrategy = InitializationStrategy.XAVIER_QUANTUM
    use_layerwise_training: bool = True
    use_local_cost: bool = False
    locality_radius: int = 2
    warmup_epochs: int = 1
    layers_per_epoch: int = 1
    warning_threshold: float = 1e-5
    critical_threshold: float = 1e-7


class BarrenPlateauMitigator:
    """Integrated barren plateau prevention system.

    Combines:
    - Smart initialization
    - Gradient monitoring
    - Layerwise training
    - Local cost conversion (optional)

    Example:
        >>> mitigator = BarrenPlateauMitigator(total_layers=6)
        >>> params = mitigator.initialize_params(num_qubits=4)
        >>> # During training:
        >>> health = mitigator.check_gradients(gradients)
        >>> if health.status == "barren_plateau":
        ...     action = mitigator.recommend_action()
    """

    def __init__(
        self,
        total_layers: int,
        config: BarrenPlateauConfig | None = None,
    ):
        """Initialize mitigator.

        Args:
            total_layers: Total VQC layers.
            config: Configuration. If None, uses defaults.
        """
        self.config = config or BarrenPlateauConfig()
        self.total_layers = total_layers

        # Components
        self.monitor = GradientHealthMonitor(
            warning_threshold=self.config.warning_threshold,
            critical_threshold=self.config.critical_threshold,
        )

        self.scheduler = (
            LayerwiseTrainingScheduler(
                total_layers=total_layers,
                warmup_epochs=self.config.warmup_epochs,
                layers_per_epoch=self.config.layers_per_epoch,
            )
            if self.config.use_layerwise_training
            else None
        )

        logger.info(
            f"[BP] Mitigator initialized: init={self.config.initialization.value}, "
            f"layerwise={self.config.use_layerwise_training}, "
            f"local_cost={self.config.use_local_cost}"
        )

    def initialize_params(
        self,
        num_qubits: int,
        params_per_gate: int = 3,
    ) -> tf.Tensor:
        """Initialize VQC parameters with BP mitigation.

        Args:
            num_qubits: Number of qubits.
            params_per_gate: Parameters per gate.

        Returns:
            Initialized parameter tensor.
        """
        return initialize_vqc_params(
            num_qubits=num_qubits,
            num_layers=self.total_layers,
            params_per_gate=params_per_gate,
            strategy=self.config.initialization,
        )

    def check_gradients(self, gradients: tf.Tensor) -> GradientHealth:
        """Check gradient health.

        Args:
            gradients: Gradient tensor.

        Returns:
            Gradient health metrics.
        """
        return self.monitor.check(gradients)

    def should_train_layer(self, layer_idx: int) -> bool:
        """Check if layer should be trained.

        Args:
            layer_idx: Layer index.

        Returns:
            True if layer is active.
        """
        if self.scheduler is None:
            return True
        return self.scheduler.should_train_layer(layer_idx)

    def get_layer_mask(self) -> tf.Tensor:
        """Get layer training mask.

        Returns:
            Boolean mask [total_layers].
        """
        if self.scheduler is None:
            return tf.ones([self.total_layers], dtype=tf.bool)
        return self.scheduler.get_layer_mask()

    def step_epoch(self):
        """Advance scheduler epoch."""
        if self.scheduler is not None:
            self.scheduler.step_epoch()

    def is_in_barren_plateau(self) -> bool:
        """Check if in barren plateau."""
        return self.monitor.is_in_barren_plateau()

    def recommend_action(self) -> str:
        """Get action recommendation."""
        return self.monitor.recommend_action()

    def wrap_loss_function(self, loss_fn, num_qubits: int):
        """Wrap loss function with local cost if enabled.

        Args:
            loss_fn: Original loss function.
            num_qubits: Number of qubits.

        Returns:
            Wrapped loss function.
        """
        if not self.config.use_local_cost:
            return loss_fn

        return global_to_local_cost(loss_fn, num_qubits, self.config.locality_radius)

    def get_statistics(self) -> dict[str, Any]:
        """Get mitigator statistics."""
        stats = {
            "initialization": self.config.initialization.value,
            "use_layerwise_training": self.config.use_layerwise_training,
            "use_local_cost": self.config.use_local_cost,
            "gradient_health": self.monitor.get_statistics(),
        }

        if self.scheduler is not None:
            stats["scheduler"] = self.scheduler.get_statistics()

        return stats

    def reset(self):
        """Reset mitigator state."""
        self.monitor.reset()
        if self.scheduler is not None:
            self.scheduler = LayerwiseTrainingScheduler(
                total_layers=self.total_layers,
                warmup_epochs=self.config.warmup_epochs,
                layers_per_epoch=self.config.layers_per_epoch,
            )


# =============================================================================
# Factory Functions
# =============================================================================


def create_bp_mitigator(
    total_layers: int,
    initialization: str = "xavier_quantum",
    use_layerwise: bool = True,
    **kwargs,
) -> BarrenPlateauMitigator:
    """Factory for BarrenPlateauMitigator.

    Args:
        total_layers: Total VQC layers.
        initialization: Initialization strategy name.
        use_layerwise: Whether to use layerwise training.
        **kwargs: Additional config overrides.

    Returns:
        Configured mitigator.
    """
    strategy_map = {
        "random_uniform": InitializationStrategy.RANDOM_UNIFORM,
        "xavier_quantum": InitializationStrategy.XAVIER_QUANTUM,
        "he_quantum": InitializationStrategy.HE_QUANTUM,
        "identity_bias": InitializationStrategy.IDENTITY_BIAS,
        "orthogonal": InitializationStrategy.ORTHOGONAL,
    }

    config = BarrenPlateauConfig(
        initialization=strategy_map.get(initialization, InitializationStrategy.XAVIER_QUANTUM),
        use_layerwise_training=use_layerwise,
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return BarrenPlateauMitigator(total_layers=total_layers, config=config)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "InitializationStrategy",
    "initialize_vqc_params",
    "GradientHealth",
    "GradientHealthMonitor",
    "LayerwiseTrainingScheduler",
    "compute_local_cost",
    "global_to_local_cost",
    "BarrenPlateauConfig",
    "BarrenPlateauMitigator",
    "create_bp_mitigator",
]
