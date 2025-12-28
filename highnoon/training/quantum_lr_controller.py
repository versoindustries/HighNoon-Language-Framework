# highnoon/training/quantum_lr_controller.py
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

"""Quantum Adaptive Learning Rate Controller (QALRC).

This module provides a quantum-inspired self-tuning learning rate controller
that adapts the learning rate during training based on gradient landscape
analysis and quantum annealing principles.

Key Concepts:
- Gradient Entropy: Measures roughness of loss landscape from gradient distribution
- Adiabatic Schedule: Smooth transition from exploration (high LR) to exploitation (low LR)
- Quantum Tunneling: Occasional LR jumps to escape local minima

Mathematical Framework:
    Annealing schedule: s(t) = (t / total_steps)^power
    Temperature: T(t) = T_initial * (1 - s(t)) + T_final * s(t)
    Adaptive LR: lr(t) = base_lr * (1 + entropy_factor * entropy) * temperature_scale

Reference:
    - Quantum-Inspired Adaptive Learning Rate Optimization (QIALRO)
    - Quantum Annealing for Neural Network Optimization
    - Meta-Learning with Adaptive Learning rate (MALGO)

Example:
    >>> controller = QuantumAdaptiveLRController(initial_lr=3e-4)
    >>> lr = controller.get_learning_rate(step=100, total_steps=1000, gradients=grads)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf

from highnoon import config

logger = logging.getLogger(__name__)


@dataclass
class QALRCState:
    """Internal state for QALRC tracking.

    Attributes:
        step: Current optimization step.
        current_lr: Current learning rate.
        gradient_entropy: Smoothed gradient entropy estimate.
        temperature: Current annealing temperature.
        loss_history: Recent loss values for trend detection.
        tunneling_events: Count of tunneling events.
        lr_history: History of learning rate values.
    """

    step: int = 0
    current_lr: float = 3e-4
    gradient_entropy: float = 0.0
    temperature: float = 1.0
    loss_history: list[float] = field(default_factory=list)
    tunneling_events: int = 0
    lr_history: list[float] = field(default_factory=list)


@dataclass
class QuantumAdaptiveLRController:
    """Quantum-inspired self-tuning learning rate controller.

    Uses three quantum-inspired mechanisms to adaptively tune learning rate:
    1. Gradient Entropy Sensing - Measures loss landscape roughness
    2. Adiabatic Annealing - Smooth exploration→exploitation transition
    3. Quantum Tunneling - Probabilistic jumps to escape local minima

    Attributes:
        initial_lr: Starting learning rate.
        min_lr: Minimum allowed learning rate.
        max_lr: Maximum allowed learning rate.
        annealing_power: Exponent for annealing schedule (higher = faster focus).
        tunneling_probability: Base probability of quantum tunneling event.
        tunneling_magnitude: Magnitude of LR jump during tunneling.
        entropy_smoothing: EMA coefficient for gradient entropy.
        entropy_scale: How much entropy affects LR scaling.
        loss_window: Number of recent losses to track for trend detection.
        enabled: Whether the controller is active.
    """

    # Core LR bounds
    initial_lr: float = 3e-4
    min_lr: float = 1e-6
    max_lr: float = 1e-2

    # Quantum-inspired parameters
    annealing_power: float = field(
        default_factory=lambda: getattr(config, "QALRC_ANNEALING_POWER", 2.0)
    )
    tunneling_probability: float = field(
        default_factory=lambda: getattr(config, "QALRC_TUNNELING_PROBABILITY", 0.05)
    )
    tunneling_magnitude: float = 0.3
    entropy_smoothing: float = field(
        default_factory=lambda: getattr(config, "QALRC_ENTROPY_SMOOTHING", 0.9)
    )
    entropy_scale: float = 0.5
    loss_window: int = 20

    # Control flags
    enabled: bool = field(
        default_factory=lambda: getattr(config, "USE_QUANTUM_LR_CONTROLLER", True)
    )

    # Internal state
    _state: QALRCState = field(default_factory=QALRCState)

    def __post_init__(self) -> None:
        """Initialize controller state."""
        self._state = QALRCState(current_lr=self.initial_lr)
        logger.info(
            f"[QALRC] Initialized: initial_lr={self.initial_lr}, "
            f"annealing_power={self.annealing_power}, "
            f"tunneling_prob={self.tunneling_probability}"
        )

    def compute_gradient_entropy(
        self, gradients: list[tf.Tensor | None]
    ) -> float:
        """Compute entropy of gradient magnitude distribution.

        Higher entropy indicates a more "rough" or uncertain gradient landscape,
        suggesting exploration is beneficial. Lower entropy indicates consistent
        gradient direction, favoring exploitation.

        Args:
            gradients: List of gradient tensors (may contain None).

        Returns:
            Normalized entropy value in [0, 1].
        """
        valid_grads = [g for g in gradients if g is not None]
        if not valid_grads:
            return 0.0

        # Compute gradient magnitudes
        magnitudes = []
        for g in valid_grads:
            if isinstance(g, tf.IndexedSlices):
                mag = tf.reduce_mean(tf.abs(g.values))
            else:
                mag = tf.reduce_mean(tf.abs(g))
            magnitudes.append(float(mag.numpy()))

        if not magnitudes or max(magnitudes) == 0:
            return 0.0

        # Normalize to probability distribution
        magnitudes = np.array(magnitudes)
        magnitudes = magnitudes / (magnitudes.sum() + 1e-10)

        # Compute Shannon entropy
        entropy = -np.sum(magnitudes * np.log(magnitudes + 1e-10))

        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log(len(magnitudes))
        normalized_entropy = entropy / (max_entropy + 1e-10)

        return float(np.clip(normalized_entropy, 0.0, 1.0))

    def get_annealing_schedule(self, step: int, total_steps: int) -> float:
        """Compute adiabatic annealing schedule s(t) ∈ [0, 1].

        Schedule transitions from 0 (exploration) to 1 (exploitation)
        following a polynomial curve.

        Args:
            step: Current training step.
            total_steps: Total number of training steps.

        Returns:
            Schedule value s(t) in [0, 1].
        """
        if total_steps <= 0:
            return 1.0

        t = min(step / total_steps, 1.0)
        return math.pow(t, self.annealing_power)

    def get_temperature(self, schedule: float) -> float:
        """Compute current temperature from annealing schedule.

        Temperature controls exploration/exploitation trade-off:
        - High temperature (early training) → larger LR variance
        - Low temperature (late training) → stable, lower LR

        Args:
            schedule: Annealing schedule value s(t) ∈ [0, 1].

        Returns:
            Temperature value T(s) > 0.
        """
        # Temperature decays from 1.0 to 0.1 as training progresses
        initial_temp = 1.0
        final_temp = 0.1
        return initial_temp * (1.0 - schedule) + final_temp * schedule

    def should_tunnel(self, loss_trend: float) -> bool:
        """Determine if quantum tunneling should occur.

        Tunneling probability increases when:
        1. Loss is stagnating or increasing (positive trend)
        2. Random chance based on tunneling_probability

        Args:
            loss_trend: Slope of recent loss curve (positive = worsening).

        Returns:
            True if tunneling should occur.
        """
        # Base probability modified by loss trend
        adjusted_prob = self.tunneling_probability
        if loss_trend > 0:
            # Increase tunneling probability if loss is worsening
            adjusted_prob *= 1.0 + min(loss_trend * 10, 1.0)

        return np.random.random() < adjusted_prob

    def compute_loss_trend(self) -> float:
        """Compute slope of recent loss values.

        Returns:
            Positive value if loss is increasing, negative if decreasing.
        """
        if len(self._state.loss_history) < 3:
            return 0.0

        # Simple linear regression slope
        losses = np.array(self._state.loss_history[-self.loss_window :])
        x = np.arange(len(losses))

        # Compute slope via least squares
        n = len(losses)
        slope = (n * np.sum(x * losses) - np.sum(x) * np.sum(losses)) / (
            n * np.sum(x**2) - np.sum(x) ** 2 + 1e-10
        )

        return float(slope)

    def get_learning_rate(
        self,
        step: int,
        total_steps: int,
        gradients: list[tf.Tensor | None] | None = None,
        current_loss: float | None = None,
    ) -> float:
        """Compute adaptive learning rate for current step.

        This is the main entry point that combines all quantum-inspired
        mechanisms to produce an adaptive learning rate.

        Args:
            step: Current training step.
            total_steps: Total number of training steps.
            gradients: Optional list of gradient tensors for entropy computation.
            current_loss: Optional current loss value for trend detection.

        Returns:
            Adaptive learning rate value.
        """
        if not self.enabled:
            return self.initial_lr

        self._state.step = step

        # Update loss history
        if current_loss is not None and math.isfinite(current_loss):
            self._state.loss_history.append(current_loss)
            if len(self._state.loss_history) > self.loss_window:
                self._state.loss_history.pop(0)

        # 1. Compute annealing schedule
        schedule = self.get_annealing_schedule(step, total_steps)
        self._state.temperature = self.get_temperature(schedule)

        # 2. Compute gradient entropy (if gradients provided)
        if gradients is not None:
            raw_entropy = self.compute_gradient_entropy(gradients)
            # Exponential moving average for stability
            self._state.gradient_entropy = (
                self.entropy_smoothing * self._state.gradient_entropy
                + (1 - self.entropy_smoothing) * raw_entropy
            )

        # 3. Base learning rate with cosine decay
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * schedule))
        base_lr = self.initial_lr * (0.1 + 0.9 * cosine_factor)

        # 4. Entropy-based scaling (higher entropy → higher LR for exploration)
        entropy_factor = 1.0 + self.entropy_scale * self._state.gradient_entropy

        # 5. Temperature scaling
        temp_factor = 0.5 + 0.5 * self._state.temperature

        # Compute current LR
        current_lr = base_lr * entropy_factor * temp_factor

        # 6. Quantum tunneling check
        loss_trend = self.compute_loss_trend()
        if self.should_tunnel(loss_trend):
            # Random LR jump to escape local minima
            tunnel_direction = 1.0 if np.random.random() > 0.5 else -1.0
            tunnel_jump = 1.0 + tunnel_direction * self.tunneling_magnitude * np.random.random()
            current_lr *= tunnel_jump
            self._state.tunneling_events += 1
            logger.debug(
                f"[QALRC] Quantum tunneling at step {step}: "
                f"LR jump by {tunnel_jump:.3f}x"
            )

        # Clamp to bounds
        current_lr = max(self.min_lr, min(self.max_lr, current_lr))
        self._state.current_lr = current_lr

        # Track LR history
        self._state.lr_history.append(current_lr)
        if len(self._state.lr_history) > 100:
            self._state.lr_history.pop(0)

        return current_lr

    def get_statistics(self) -> dict[str, Any]:
        """Get controller statistics for logging/monitoring.

        Returns:
            Dictionary with current state and statistics.
        """
        return {
            "enabled": self.enabled,
            "step": self._state.step,
            "current_lr": self._state.current_lr,
            "gradient_entropy": self._state.gradient_entropy,
            "temperature": self._state.temperature,
            "tunneling_events": self._state.tunneling_events,
            "loss_trend": self.compute_loss_trend(),
            "lr_mean": np.mean(self._state.lr_history) if self._state.lr_history else 0,
            "lr_std": np.std(self._state.lr_history) if self._state.lr_history else 0,
        }

    def reset(self) -> None:
        """Reset controller state for a new trial."""
        self._state = QALRCState(current_lr=self.initial_lr)
        logger.info("[QALRC] Controller reset")


__all__ = [
    "QuantumAdaptiveLRController",
    "QALRCState",
]
