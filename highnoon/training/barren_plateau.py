# highnoon/training/barren_plateau.py
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

"""Barren Plateau Detection and Mitigation for Quantum-Enhanced Training.

This module provides automatic detection and mitigation of barren plateaus
(vanishing gradients) in quantum-enhanced neural network layers. When
gradient norms fall below a configurable threshold, the monitor applies
recovery strategies to maintain training stability.

Barren plateaus are a common challenge in variational quantum algorithms
and quantum neural networks, where gradients become exponentially small
with increasing circuit depth, making optimization intractable.

Recovery Strategies:
    1. Learning rate scaling: Temporarily increase LR for affected layers
    2. Noise injection: Add controlled noise to break symmetry
    3. Depth reduction: Reduce VQC depth temporarily (if applicable)
    4. Gradient clipping adjustment: Relax clipping for recovery

Example:
    >>> from highnoon.training.barren_plateau import BarrenPlateauMonitor
    >>> monitor = BarrenPlateauMonitor(threshold=1e-6)
    >>> mitigations = monitor.check_and_mitigate(gradient_norms)
    >>> for layer, strategy in mitigations.items():
    ...     apply_mitigation(layer, strategy)

References:
    - McClean et al., "Barren plateaus in quantum neural network training
      landscapes" (2018)
    - Cerezo et al., "Cost function dependent barren plateaus in shallow
      parametrized quantum circuits" (2021)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
import tensorflow as tf

from highnoon import config

logger = logging.getLogger(__name__)


class MitigationStrategy(Enum):
    """Available strategies for mitigating barren plateaus."""

    LR_SCALE_UP = auto()
    NOISE_INJECTION = auto()
    DEPTH_REDUCTION = auto()
    GRADIENT_CLIP_RELAX = auto()
    PARAMETER_REINITIALIZATION = auto()


@dataclass
class LayerMitigation:
    """Mitigation configuration for a specific layer.

    Attributes:
        layer_name: Name of the layer experiencing barren plateau.
        strategy: Primary mitigation strategy to apply.
        lr_scale_factor: Learning rate multiplier (default: 10.0).
        noise_scale: Standard deviation of injected noise (default: 1e-3).
        depth_reduction_factor: Factor to reduce VQC depth (default: 0.5).
        gradient_clip_multiplier: Gradient clipping relaxation (default: 2.0).
        consecutive_detections: Number of consecutive barren plateau detections.
        recovery_steps: Steps since mitigation was applied.
    """

    layer_name: str
    strategy: MitigationStrategy
    lr_scale_factor: float = 10.0
    noise_scale: float = 1e-3
    depth_reduction_factor: float = 0.5
    gradient_clip_multiplier: float = 2.0
    consecutive_detections: int = 1
    recovery_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "layer_name": self.layer_name,
            "strategy": self.strategy.name,
            "lr_scale_factor": self.lr_scale_factor,
            "noise_scale": self.noise_scale,
            "depth_reduction_factor": self.depth_reduction_factor,
            "gradient_clip_multiplier": self.gradient_clip_multiplier,
            "consecutive_detections": self.consecutive_detections,
            "recovery_steps": self.recovery_steps,
        }


@dataclass
class BarrenPlateauMonitor:
    """Monitors gradient norms and applies mitigation strategies.

    This monitor tracks gradient norms across training steps and detects
    when quantum-enhanced layers enter barren plateau regions. Upon
    detection, it returns mitigation configurations that the training
    loop should apply to recover training dynamics.

    Attributes:
        threshold: Gradient norm below which barren plateau is detected.
        hysteresis_factor: Factor to prevent oscillation (exit threshold =
            threshold * hysteresis_factor).
        recovery_window: Steps to maintain mitigation after detection.
        max_lr_scale: Maximum allowed learning rate scaling.
        quantum_layer_patterns: Layer name patterns to monitor.
        enabled: Whether monitoring is active.
    """

    threshold: float = field(default_factory=lambda: config.BARREN_PLATEAU_THRESHOLD)
    hysteresis_factor: float = 5.0
    recovery_window: int = 100
    max_lr_scale: float = 100.0
    quantum_layer_patterns: tuple[str, ...] = (
        "quantum",
        "vqc",
        "hamiltonian",
        "timecrystal",
        "port_hamiltonian",
        "mps",
        "evolution",
    )
    enabled: bool = field(default_factory=lambda: config.BARREN_PLATEAU_MONITOR)

    # Internal tracking state
    _gradient_history: dict[str, list[float]] = field(default_factory=dict)
    _active_mitigations: dict[str, LayerMitigation] = field(default_factory=dict)
    _detection_counts: dict[str, int] = field(default_factory=dict)
    _step_counter: int = 0

    def __post_init__(self) -> None:
        """Initialize internal state after dataclass construction."""
        if not hasattr(self, "_gradient_history") or self._gradient_history is None:
            self._gradient_history = {}
        if not hasattr(self, "_active_mitigations") or self._active_mitigations is None:
            self._active_mitigations = {}
        if not hasattr(self, "_detection_counts") or self._detection_counts is None:
            self._detection_counts = {}
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0

    def _is_quantum_layer(self, layer_name: str) -> bool:
        """Check if a layer name matches quantum layer patterns."""
        layer_lower = layer_name.lower()
        return any(pattern in layer_lower for pattern in self.quantum_layer_patterns)

    def _compute_ema(self, history: list[float], alpha: float = 0.1) -> float:
        """Compute exponential moving average of gradient norm history."""
        if not history:
            return float("inf")

        ema = history[0]
        for value in history[1:]:
            ema = alpha * value + (1 - alpha) * ema
        return ema

    def check_and_mitigate(
        self,
        gradient_norms: dict[str, float],
        variables: dict[str, tf.Variable] | None = None,
    ) -> dict[str, LayerMitigation]:
        """Check for barren plateaus and return mitigation strategies.

        This method should be called after each training step with the
        gradient norms for all layers. It returns a dictionary of
        mitigation configurations for layers that require intervention.

        Args:
            gradient_norms: Dictionary mapping layer names to gradient L2 norms.
            variables: Optional dictionary of layer variables for noise injection.

        Returns:
            Dictionary mapping layer names to `LayerMitigation` configurations.
            Only layers requiring mitigation are included.

        Example:
            >>> monitor = BarrenPlateauMonitor(threshold=1e-6)
            >>> norms = {"quantum_block_1": 1e-8, "dense_layer": 0.1}
            >>> mitigations = monitor.check_and_mitigate(norms)
            >>> # Returns {"quantum_block_1": LayerMitigation(...)}
        """
        if not self.enabled:
            return {}

        self._step_counter += 1
        mitigations: dict[str, LayerMitigation] = {}

        for layer_name, norm in gradient_norms.items():
            # Track gradient history
            if layer_name not in self._gradient_history:
                self._gradient_history[layer_name] = []
            self._gradient_history[layer_name].append(norm)

            # Keep history bounded
            if len(self._gradient_history[layer_name]) > 100:
                self._gradient_history[layer_name] = self._gradient_history[layer_name][-50:]

            # Only monitor quantum layers
            if not self._is_quantum_layer(layer_name):
                continue

            # Compute smoothed gradient norm
            ema_norm = self._compute_ema(self._gradient_history[layer_name])

            # Check for active mitigation in recovery phase
            if layer_name in self._active_mitigations:
                active = self._active_mitigations[layer_name]
                active.recovery_steps += 1

                # Check if recovery is complete
                exit_threshold = self.threshold * self.hysteresis_factor
                if ema_norm > exit_threshold:
                    logger.info(
                        "[BARREN PLATEAU] Layer '%s' recovered: norm=%.2e > %.2e",
                        layer_name,
                        ema_norm,
                        exit_threshold,
                    )
                    del self._active_mitigations[layer_name]
                    self._detection_counts[layer_name] = 0
                elif active.recovery_steps >= self.recovery_window:
                    # Escalate mitigation if not recovering
                    active.lr_scale_factor = min(
                        active.lr_scale_factor * 2.0,
                        self.max_lr_scale,
                    )
                    active.consecutive_detections += 1
                    active.recovery_steps = 0
                    logger.warning(
                        "[BARREN PLATEAU] Layer '%s' escalating mitigation: "
                        "lr_scale=%.1f, detections=%d",
                        layer_name,
                        active.lr_scale_factor,
                        active.consecutive_detections,
                    )
                    mitigations[layer_name] = active
                else:
                    # Continue current mitigation
                    mitigations[layer_name] = active
                continue

            # Check for new barren plateau detection
            if ema_norm < self.threshold:
                self._detection_counts[layer_name] = self._detection_counts.get(layer_name, 0) + 1

                # Require consecutive detections to reduce false positives
                if self._detection_counts[layer_name] >= 3:
                    mitigation = LayerMitigation(
                        layer_name=layer_name,
                        strategy=MitigationStrategy.LR_SCALE_UP,
                        lr_scale_factor=config.BARREN_PLATEAU_RECOVERY_LR_SCALE,
                        noise_scale=1e-3,
                        consecutive_detections=1,
                        recovery_steps=0,
                    )
                    self._active_mitigations[layer_name] = mitigation
                    mitigations[layer_name] = mitigation

                    logger.warning(
                        "[BARREN PLATEAU] Detected in layer '%s': "
                        "norm=%.2e < threshold=%.2e. Applying mitigation.",
                        layer_name,
                        ema_norm,
                        self.threshold,
                    )
            else:
                # Reset detection count if norm recovers
                self._detection_counts[layer_name] = 0

        return mitigations

    def apply_lr_scaling(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        mitigations: dict[str, LayerMitigation],
        layer_to_vars: dict[str, list[tf.Variable]],
    ) -> dict[str, float]:
        """Apply learning rate scaling to mitigate barren plateaus.

        This method adjusts learning rates for variables in layers
        experiencing barren plateaus. It returns the original learning
        rates for later restoration.

        Args:
            optimizer: The training optimizer.
            mitigations: Active mitigations from `check_and_mitigate`.
            layer_to_vars: Mapping of layer names to their variables.

        Returns:
            Dictionary mapping variable names to original learning rates.
        """
        original_lrs: dict[str, float] = {}

        for layer_name, mitigation in mitigations.items():
            if mitigation.strategy != MitigationStrategy.LR_SCALE_UP:
                continue

            if layer_name not in layer_to_vars:
                continue

            for var in layer_to_vars[layer_name]:
                # Store original learning rate
                original_lrs[var.name] = float(optimizer.learning_rate)

        return original_lrs

    def inject_noise(
        self,
        variables: dict[str, tf.Variable],
        mitigations: dict[str, LayerMitigation],
    ) -> None:
        """Inject controlled noise into layer variables.

        Noise injection can help break symmetry and escape barren plateau
        regions by perturbing the optimization trajectory.

        Args:
            variables: Dictionary of layer variables.
            mitigations: Active mitigations requiring noise injection.
        """
        for layer_name, mitigation in mitigations.items():
            if mitigation.strategy != MitigationStrategy.NOISE_INJECTION:
                continue

            if layer_name not in variables:
                continue

            var = variables[layer_name]
            noise = tf.random.normal(
                shape=var.shape,
                mean=0.0,
                stddev=mitigation.noise_scale,
                dtype=var.dtype,
            )
            var.assign_add(noise)

            logger.info(
                "[BARREN PLATEAU] Injected noise into '%s': scale=%.2e",
                layer_name,
                mitigation.noise_scale,
            )

    def get_statistics(self) -> dict[str, Any]:
        """Get current monitoring statistics for logging.

        Returns:
            Dictionary with monitoring statistics including active
            mitigations, detection counts, and gradient norm summaries.
        """
        stats = {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "step_counter": self._step_counter,
            "active_mitigations_count": len(self._active_mitigations),
            "active_mitigations": {
                name: mit.to_dict() for name, mit in self._active_mitigations.items()
            },
            "layers_monitored": len(
                [k for k in self._gradient_history.keys() if self._is_quantum_layer(k)]
            ),
        }

        # Add gradient norm summaries for monitored layers
        gradient_summaries = {}
        for layer_name, history in self._gradient_history.items():
            if self._is_quantum_layer(layer_name) and history:
                gradient_summaries[layer_name] = {
                    "current": history[-1],
                    "ema": self._compute_ema(history),
                    "min": min(history),
                    "max": max(history),
                }
        stats["gradient_summaries"] = gradient_summaries

        return stats

    def reset(self) -> None:
        """Reset all monitoring state.

        Call this when starting a new training run or after significant
        model changes.
        """
        self._gradient_history.clear()
        self._active_mitigations.clear()
        self._detection_counts.clear()
        self._step_counter = 0
        logger.info("[BARREN PLATEAU] Monitor state reset")


def compute_layer_gradient_norms(
    gradients: list[tf.Tensor],
    variables: list[tf.Variable],
) -> dict[str, float]:
    """Compute gradient L2 norms grouped by layer.

    This helper function takes raw gradients and variables from a
    training step and computes the L2 norm of gradients for each layer.

    Args:
        gradients: List of gradient tensors from `tape.gradient()`.
        variables: Corresponding list of trainable variables.

    Returns:
        Dictionary mapping layer names to gradient L2 norms.
    """
    layer_norms: dict[str, list[float]] = {}

    for grad, var in zip(gradients, variables):
        if grad is None:
            continue

        # Extract layer name from variable name
        # Format: "layer_name/kernel:0" or "model/layer_name/weight:0"
        name_parts = var.name.split("/")
        if len(name_parts) >= 2:
            layer_name = name_parts[-2]
        else:
            layer_name = name_parts[0].split(":")[0]

        # Compute L2 norm
        norm = float(tf.norm(grad, ord=2).numpy())

        if layer_name not in layer_norms:
            layer_norms[layer_name] = []
        layer_norms[layer_name].append(norm)

    # Aggregate norms per layer (RMS of individual norms)
    return {layer: np.sqrt(np.mean([n**2 for n in norms])) for layer, norms in layer_norms.items()}


__all__ = [
    "BarrenPlateauMonitor",
    "LayerMitigation",
    "MitigationStrategy",
    "compute_layer_gradient_norms",
]
