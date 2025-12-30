# highnoon/training/quls_feedback_callback.py
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

"""QULS Feedback Callback for QAHPO Integration.

This module provides the callback mechanism for QULS telemetry export to the
Quantum Adaptive HPO scheduler. It captures per-batch metrics and exposes them
for barren plateau detection and tunneling probability adjustment.

Reference: HIGHNOON_UPGRADE_ROADMAP.md Section 1.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class BaseCallback:
    """Base class for training callbacks."""

    def on_batch_end(self, step: int, result: dict[str, Any] | Any) -> bool:
        """Called at the end of each training batch."""
        return True


@dataclass
class QULSTelemetry:
    """Telemetry data from QULS for QAHPO feedback.

    Captured per-batch and exposed via QULSFeedbackCallback.get_telemetry().

    Attributes:
        barren_plateau_detected: Whether a barren plateau was detected this batch.
        vqc_gradient_variance: Variance of VQC layer gradients (low = barren).
        fidelity_loss: Quantum fidelity loss component value.
        entropy_loss: Entropy regularization loss component value.
        coherence_metric: Average coherence metric from QCB.
        gradient_entropy: Entropy of gradient distribution (low = uniform = barren).
        step: Training step number.
    """

    barren_plateau_detected: bool = False
    vqc_gradient_variance: float = 0.0
    fidelity_loss: float = 0.0
    entropy_loss: float = 0.0
    coherence_metric: float = 1.0
    gradient_entropy: float = 0.0
    step: int = 0


@dataclass
class BarrenPlateauStats:
    """Rolling statistics for barren plateau detection.

    Tracks plateau occurrences over a sliding window.
    """

    window_size: int = 10
    plateau_count: int = 0
    recent_detections: list[bool] = field(default_factory=list)

    def update(self, detected: bool) -> None:
        """Update with new detection result."""
        self.recent_detections.append(detected)
        if len(self.recent_detections) > self.window_size:
            removed = self.recent_detections.pop(0)
            if removed:
                self.plateau_count -= 1
        if detected:
            self.plateau_count += 1

    @property
    def count_in_window(self) -> int:
        """Get count of detections in current window."""
        return self.plateau_count


class QULSFeedbackCallback(BaseCallback):
    """Callback for QULS → QAHPO telemetry feedback.

    Captures QULS metrics each batch and exposes them for HPO scheduler
    barren plateau response. Implements escalation logic per roadmap:

    - 3+ barren plateaus in 10 batches → Signal for tunneling probability boost
    - 7+ barren plateaus in 10 batches → Signal for early-stop with perturbed config

    Example:
        >>> callback = QULSFeedbackCallback()
        >>> # During training, training_engine calls:
        >>> callback.on_batch_end(step=100, result=step_result)
        >>> # HPO scheduler queries:
        >>> telemetry = callback.get_telemetry()
        >>> if telemetry.barren_plateau_detected:
        ...     adjust_tunneling_probability()
    """

    def __init__(
        self,
        tunneling_threshold: int = 3,
        early_stop_threshold: int = 7,
        window_size: int = 10,
    ):
        """Initialize QULS feedback callback.

        Args:
            tunneling_threshold: Plateau count to trigger tunneling boost.
            early_stop_threshold: Plateau count to trigger early-stop signal.
            window_size: Sliding window size for plateau detection.
        """
        super().__init__()
        self.tunneling_threshold = tunneling_threshold
        self.early_stop_threshold = early_stop_threshold

        self._stats = BarrenPlateauStats(window_size=window_size)
        self._current_telemetry = QULSTelemetry()
        self._should_early_stop = False
        self._should_boost_tunneling = False

    def on_batch_end(self, step: int, result: dict[str, Any] | Any) -> bool:
        """Called at the end of each training batch.

        Captures QULS metrics from step result and updates telemetry.

        Args:
            step: Current training step.
            result: Training step result containing loss components.

        Returns:
            True to continue training, False to stop.
        """
        # Extract metrics from result
        if hasattr(result, "loss_components"):
            components = result.loss_components
        elif isinstance(result, dict):
            components = result.get("loss_components", result)
        else:
            components = {}

        # Detect barren plateau from VQC gradient variance
        vqc_variance = components.get("vqc_gradient_variance", 0.0)
        if isinstance(vqc_variance, (int, float)):
            barren_detected = vqc_variance < 1e-6
        else:
            barren_detected = False

        # Update rolling stats
        self._stats.update(barren_detected)

        # Build telemetry
        self._current_telemetry = QULSTelemetry(
            barren_plateau_detected=barren_detected,
            vqc_gradient_variance=float(vqc_variance) if vqc_variance else 0.0,
            fidelity_loss=float(components.get("fidelity", 0.0)),
            entropy_loss=float(components.get("entropy", 0.0)),
            coherence_metric=float(components.get("coherence", 1.0)),
            gradient_entropy=float(components.get("gradient_entropy", 0.0)),
            step=step,
        )

        # Update escalation signals
        count = self._stats.count_in_window
        self._should_boost_tunneling = count >= self.tunneling_threshold
        self._should_early_stop = count >= self.early_stop_threshold

        if self._should_early_stop:
            logger.warning(
                f"[QULS] Barren plateau escalation: {count}/{self.early_stop_threshold} "
                f"in window. Signaling early-stop."
            )
        elif self._should_boost_tunneling:
            logger.info(
                f"[QULS] Barren plateau detected: {count}/{self.tunneling_threshold} "
                f"in window. Boosting tunneling probability."
            )

        return True  # Continue training

    def get_telemetry(self) -> QULSTelemetry:
        """Get current QULS telemetry.

        Returns:
            QULSTelemetry dataclass with current metrics.
        """
        return self._current_telemetry

    def should_boost_tunneling(self) -> bool:
        """Check if tunneling probability should be boosted.

        Returns:
            True if barren plateaus detected >= tunneling_threshold.
        """
        return self._should_boost_tunneling

    def should_early_stop(self) -> bool:
        """Check if trial should early-stop.

        Returns:
            True if barren plateaus detected >= early_stop_threshold.
        """
        return self._should_early_stop

    def reset(self) -> None:
        """Reset callback state for new trial."""
        self._stats = BarrenPlateauStats(window_size=self._stats.window_size)
        self._current_telemetry = QULSTelemetry()
        self._should_early_stop = False
        self._should_boost_tunneling = False

    def get_summary(self) -> dict[str, Any]:
        """Get summary of callback state.

        Returns:
            Dict with plateau count, escalation status, and last telemetry.
        """
        return {
            "plateau_count_in_window": self._stats.count_in_window,
            "should_boost_tunneling": self._should_boost_tunneling,
            "should_early_stop": self._should_early_stop,
            "last_telemetry": {
                "barren_plateau_detected": self._current_telemetry.barren_plateau_detected,
                "vqc_gradient_variance": self._current_telemetry.vqc_gradient_variance,
                "step": self._current_telemetry.step,
            },
        }
