# highnoon/training/quantum_curriculum.py
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

"""Spectrally-Aware Quantum Curriculum (SAQC) - Phase 200+.

Synergizes QULS telemetry with curriculum scheduling for quantum-adaptive
data progression. Closes the loop between model quantum state and training
data complexity as proposed in quantum_enhancement_analysis.md.

Key Capabilities:
    - Entropy-Driven Retreat: Low spectral entropy → high-diversity batches
    - Barren Plateau Tunneling: Plateau detection → orthogonal dataset injection
    - Coherence-Gated Progression: High fidelity → rapid curriculum advancement

Integration:
    Works with QULSFeedbackCallback to receive quantum telemetry and
    adjusts curriculum based on spectral diagnostics (entropy, fidelity,
    coherence, barren plateau detection).

Reference:
    - HIGHNOON_UPGRADE_ROADMAP.md Phase 200+
    - quantum_enhancement_analysis.md (SAQC proposal)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from highnoon import config as hn_config
from highnoon.training.curriculum import AdaptiveCurriculumScheduler

if TYPE_CHECKING:
    from highnoon.training.quls_feedback_callback import QULSTelemetry

logger = logging.getLogger(__name__)


# =============================================================================
# Curriculum Mode Enumeration
# =============================================================================


class CurriculumMode(IntEnum):
    """SAQC operating modes driven by quantum telemetry.

    Matches C++ enum in quantum_curriculum_op.h for consistency.
    """

    NORMAL = 0  # Standard adaptive progression
    RETREAT = 1  # Low entropy → diversity restoration
    TUNNELING = 2  # Barren plateau → orthogonal injection
    ACCELERATE = 3  # High fidelity → rapid advancement


# =============================================================================
# SAQC Configuration
# =============================================================================


@dataclass
class SAQCConfig:
    """Configuration for Spectrally-Aware Quantum Curriculum.

    Attributes:
        enabled: Master switch for SAQC functionality.
        entropy_retreat_threshold: Spectral entropy below this triggers retreat mode.
        fidelity_advance_threshold: Fidelity above this enables acceleration mode.
        coherence_gate_threshold: Coherence required for progression gating.
        tunneling_dataset_ratio: Fraction of tunneling samples during plateau.
        fft_dim: FFT dimension for spectral analysis (C++ op).
        update_interval: Steps between curriculum state updates.
        min_stage_duration: Minimum steps before stage transition.
    """

    enabled: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_QUANTUM_CURRICULUM", True)
    )
    entropy_retreat_threshold: float = field(
        default_factory=lambda: getattr(hn_config, "SAQC_ENTROPY_RETREAT_THRESHOLD", 0.3)
    )
    fidelity_advance_threshold: float = field(
        default_factory=lambda: getattr(hn_config, "SAQC_FIDELITY_ADVANCE_THRESHOLD", 0.85)
    )
    coherence_gate_threshold: float = field(
        default_factory=lambda: getattr(hn_config, "SAQC_COHERENCE_GATE_THRESHOLD", 0.9)
    )
    tunneling_dataset_ratio: float = field(
        default_factory=lambda: getattr(hn_config, "SAQC_TUNNELING_DATASET_RATIO", 0.2)
    )
    fft_dim: int = field(default_factory=lambda: getattr(hn_config, "SAQC_FFT_DIM", 64))
    update_interval: int = field(
        default_factory=lambda: getattr(hn_config, "SAQC_UPDATE_INTERVAL", 10)
    )
    min_stage_duration: int = field(
        default_factory=lambda: getattr(hn_config, "SAQC_MIN_STAGE_DURATION", 100)
    )


# =============================================================================
# Native Op Loading
# =============================================================================


def _try_load_native_ops() -> bool:
    """Attempt to load native SAQC C++ ops.

    Returns:
        True if native ops are available, False otherwise.
    """
    try:
        from highnoon._native.ops import lib_loader

        lib = lib_loader.load_ops_library()
        return hasattr(lib, "QuantumCurriculumScoreForward")
    except (ImportError, OSError, AttributeError):
        return False


_NATIVE_OPS_AVAILABLE = _try_load_native_ops()


def saqc_available() -> bool:
    """Check if SAQC native ops are available.

    Returns:
        True if C++ SAQC ops are compiled and loadable.
    """
    return _NATIVE_OPS_AVAILABLE


# =============================================================================
# Main SAQC Class
# =============================================================================


class QuantumSynergyCurriculum(AdaptiveCurriculumScheduler):
    """SAQC: Telemetry-driven curriculum responding to QULS spectral state.

    Extends AdaptiveCurriculumScheduler with quantum-aware decisions based on
    QULS telemetry. Implements three key behaviors:

    1. **Entropy-Driven Retreat**: When spectral entropy collapses (representation
       rank drops), switch to high-diversity batches to restore representational
       capacity before tackling harder examples.

    2. **Barren Plateau Tunneling**: When QULS detects barren plateaus (low VQC
       gradient variance), inject orthogonal/edge-case datasets to break symmetry.

    3. **Coherence-Gated Progression**: When both fidelity and coherence are high,
       accelerate through curriculum stages.

    Example:
        >>> from highnoon.training.quantum_curriculum import QuantumSynergyCurriculum
        >>> from highnoon.training.quls_feedback_callback import QULSTelemetry
        >>>
        >>> curriculum = QuantumSynergyCurriculum()
        >>> telemetry = quls_callback.get_telemetry()
        >>> curriculum.update_from_quantum_state(telemetry)
    """

    def __init__(
        self,
        config: SAQCConfig | None = None,
        tunneling_indices: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """Initialize SAQC curriculum.

        Args:
            config: SAQC configuration. Uses defaults from config.py if None.
            tunneling_indices: Pre-computed indices for tunneling datasets.
                If None, will be computed from high-variance samples.
            **kwargs: Additional arguments for AdaptiveCurriculumScheduler.
        """
        super().__init__(**kwargs)

        self.saqc_config = config or SAQCConfig()
        self._mode = CurriculumMode.NORMAL
        self._previous_mode = CurriculumMode.NORMAL
        self._steps_in_mode = 0
        self._total_steps = 0
        self._tunneling_indices = tunneling_indices
        self._diversity_indices: np.ndarray | None = None

        # Track telemetry history for smoothing
        self._entropy_history: list[float] = []
        self._fidelity_history: list[float] = []
        self._history_window = 5

        # Validate native ops if required
        require_native = getattr(hn_config, "SAQC_REQUIRE_NATIVE_OP", True)
        if require_native and not saqc_available():
            raise RuntimeError(
                "SAQC native ops not available but SAQC_REQUIRE_NATIVE_OP=True. "
                "Build C++ ops with scripts/build.sh or set SAQC_REQUIRE_NATIVE_OP=False."
            )

        if self.saqc_config.enabled:
            logger.info(
                "[SAQC] Initialized QuantumSynergyCurriculum with config: "
                f"entropy_threshold={self.saqc_config.entropy_retreat_threshold}, "
                f"fidelity_threshold={self.saqc_config.fidelity_advance_threshold}"
            )

    @property
    def mode(self) -> CurriculumMode:
        """Current curriculum operating mode."""
        return self._mode

    @property
    def mode_name(self) -> str:
        """Human-readable mode name."""
        return self._mode.name

    def update_from_quantum_state(self, telemetry: QULSTelemetry) -> CurriculumMode:
        """Update curriculum based on QULS telemetry.

        Implements SAQC decision logic:
            - barren_plateau_detected → TUNNELING
            - low spectral entropy → RETREAT
            - high fidelity + high coherence → ACCELERATE
            - otherwise → NORMAL

        Args:
            telemetry: QULSTelemetry from QULSFeedbackCallback.

        Returns:
            New CurriculumMode after update.
        """
        if not self.saqc_config.enabled:
            return CurriculumMode.NORMAL

        self._total_steps += 1

        # Only update at configured interval
        if self._total_steps % self.saqc_config.update_interval != 0:
            return self._mode

        # Update history for smoothing
        self._entropy_history.append(telemetry.entropy_loss)
        self._fidelity_history.append(telemetry.fidelity_loss)
        if len(self._entropy_history) > self._history_window:
            self._entropy_history.pop(0)
            self._fidelity_history.pop(0)

        # Use smoothed values
        avg_entropy = np.mean(self._entropy_history) if self._entropy_history else 0.5
        avg_fidelity_loss = np.mean(self._fidelity_history) if self._fidelity_history else 0.5
        avg_fidelity = 1.0 - avg_fidelity_loss  # Convert loss to score

        # Determine new mode
        new_mode = self._determine_mode(
            spectral_entropy=avg_entropy,
            fidelity=avg_fidelity,
            coherence=telemetry.coherence_metric,
            barren_plateau=telemetry.barren_plateau_detected,
        )

        # Apply mode transition rules
        if new_mode != self._mode:
            if self._steps_in_mode >= self.saqc_config.min_stage_duration:
                self._previous_mode = self._mode
                self._mode = new_mode
                self._steps_in_mode = 0
                logger.info(
                    f"[SAQC] Mode transition: {self._previous_mode.name} → {self._mode.name} "
                    f"(entropy={avg_entropy:.3f}, fidelity={avg_fidelity:.3f}, "
                    f"coherence={telemetry.coherence_metric:.3f})"
                )
        else:
            self._steps_in_mode += self.saqc_config.update_interval

        return self._mode

    def _determine_mode(
        self,
        spectral_entropy: float,
        fidelity: float,
        coherence: float,
        barren_plateau: bool,
    ) -> CurriculumMode:
        """Determine curriculum mode from metrics.

        Priority order: TUNNELING > RETREAT > ACCELERATE > NORMAL
        """
        if barren_plateau:
            return CurriculumMode.TUNNELING

        if spectral_entropy < self.saqc_config.entropy_retreat_threshold:
            return CurriculumMode.RETREAT

        if (
            fidelity > self.saqc_config.fidelity_advance_threshold
            and coherence > self.saqc_config.coherence_gate_threshold
        ):
            return CurriculumMode.ACCELERATE

        return CurriculumMode.NORMAL

    def get_current_subset_indices(self, epoch: int) -> np.ndarray:
        """Get indices modified by current SAQC mode.

        Overrides base class to apply mode-specific index selection:
            - NORMAL: Standard curriculum progression
            - RETREAT: Prioritize high-diversity samples
            - TUNNELING: Mix in tunneling dataset samples
            - ACCELERATE: Advance to next stage faster

        Args:
            epoch: Current training epoch.

        Returns:
            Indices for current batch, modified by SAQC mode.
        """
        # Get base indices from parent
        base_indices = super().get_current_subset_indices(epoch)

        if not self.saqc_config.enabled:
            return base_indices

        if self._mode == CurriculumMode.RETREAT:
            return self._apply_retreat_selection(base_indices)
        elif self._mode == CurriculumMode.TUNNELING:
            return self._apply_tunneling_injection(base_indices)
        elif self._mode == CurriculumMode.ACCELERATE:
            return self._apply_acceleration(base_indices, epoch)
        else:
            return base_indices

    def _apply_retreat_selection(self, base_indices: np.ndarray) -> np.ndarray:
        """Apply retreat mode: prioritize high-diversity samples.

        When spectral entropy is low, we need to restore representation rank.
        Select samples with maximal diversity (spread across complexity range).
        """
        if self._diversity_indices is None or len(self._diversity_indices) == 0:
            # If no diversity indices precomputed, shuffle base indices
            # to increase diversity compared to sorted order
            shuffled = base_indices.copy()
            np.random.shuffle(shuffled)
            return shuffled

        # Preferentially use diversity indices, filling with base if needed
        num_needed = len(base_indices)
        available_diversity = len(self._diversity_indices)

        if available_diversity >= num_needed:
            return self._diversity_indices[:num_needed]
        else:
            # Mix diversity + some base indices
            combined = np.concatenate(
                [self._diversity_indices, base_indices[: num_needed - available_diversity]]
            )
            return combined

    def _apply_tunneling_injection(self, base_indices: np.ndarray) -> np.ndarray:
        """Apply tunneling mode: inject orthogonal/edge-case samples.

        During barren plateau, inject high-variance samples to break symmetry.
        """
        if self._tunneling_indices is None or len(self._tunneling_indices) == 0:
            return base_indices

        ratio = self.saqc_config.tunneling_dataset_ratio
        num_tunneling = int(len(base_indices) * ratio)
        num_base = len(base_indices) - num_tunneling

        # Randomly sample from tunneling indices
        tunneling_sample = np.random.choice(
            self._tunneling_indices,
            size=min(num_tunneling, len(self._tunneling_indices)),
            replace=False,
        )

        # Combine with base indices
        combined = np.concatenate(
            [
                base_indices[:num_base],
                tunneling_sample,
            ]
        )
        np.random.shuffle(combined)  # Shuffle to avoid ordering artifacts
        return combined

    def _apply_acceleration(self, base_indices: np.ndarray, epoch: int) -> np.ndarray:
        """Apply acceleration mode: include harder samples earlier.

        When fidelity and coherence are high, the model can handle harder data.
        Expand the curriculum fraction beyond what epoch alone would suggest.
        """
        if self.dataset_sorted_indices is None:
            return base_indices

        # Expand fraction by 20% (capped at full dataset)
        current_fraction = len(base_indices) / len(self.dataset_sorted_indices)
        accelerated_fraction = min(1.0, current_fraction * 1.2)

        num_to_include = int(len(self.dataset_sorted_indices) * accelerated_fraction)
        return self.dataset_sorted_indices[:num_to_include]

    def set_tunneling_indices(self, indices: np.ndarray) -> None:
        """Set indices for tunneling dataset samples.

        These should be high-variance, edge-case, or orthogonal samples
        that can help the model escape barren plateaus.

        Args:
            indices: Array of dataset indices for tunneling samples.
        """
        self._tunneling_indices = indices
        logger.info(f"[SAQC] Set {len(indices)} tunneling indices")

    def set_diversity_indices(self, indices: np.ndarray) -> None:
        """Set indices for high-diversity samples.

        These should be samples that maximize representation diversity,
        used during RETREAT mode to restore spectral rank.

        Args:
            indices: Array of dataset indices for diversity samples.
        """
        self._diversity_indices = indices
        logger.info(f"[SAQC] Set {len(indices)} diversity indices")

    def get_state(self) -> dict:
        """Get current SAQC state for logging/checkpointing.

        Returns:
            Dictionary with mode, step counts, and history.
        """
        return {
            "mode": self._mode.name,
            "previous_mode": self._previous_mode.name,
            "steps_in_mode": self._steps_in_mode,
            "total_steps": self._total_steps,
            "entropy_history": list(self._entropy_history),
            "fidelity_history": list(self._fidelity_history),
            "stage_info": self.get_stage_info(),
        }

    def reset(self) -> None:
        """Reset SAQC state for new training run."""
        self._mode = CurriculumMode.NORMAL
        self._previous_mode = CurriculumMode.NORMAL
        self._steps_in_mode = 0
        self._total_steps = 0
        self._entropy_history.clear()
        self._fidelity_history.clear()
        logger.info("[SAQC] Reset curriculum state")


# =============================================================================
# Factory Function
# =============================================================================


def create_quantum_curriculum(
    config: SAQCConfig | dict | None = None,
    **kwargs,
) -> QuantumSynergyCurriculum:
    """Create a QuantumSynergyCurriculum instance.

    Factory function for creating SAQC instances with flexible configuration.

    Args:
        config: SAQCConfig instance or dict of config values.
        **kwargs: Additional arguments for QuantumSynergyCurriculum.

    Returns:
        Configured QuantumSynergyCurriculum instance.

    Example:
        >>> curriculum = create_quantum_curriculum(
        ...     config={"entropy_retreat_threshold": 0.25},
        ...     num_stages=4,
        ... )
    """
    if isinstance(config, dict):
        saqc_config = SAQCConfig(**config)
    elif config is None:
        saqc_config = SAQCConfig()
    else:
        saqc_config = config

    return QuantumSynergyCurriculum(config=saqc_config, **kwargs)
