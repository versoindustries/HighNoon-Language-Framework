# highnoon/quantum_config.py
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

"""Unified Quantum Configuration System - Phase 6 Implementation.

Reduces 200+ quantum config flags to preset-based system:

Old approach (50+ individual flags):
    USE_PORT_HAMILTONIAN = True
    USE_QSVT_ACTIVATIONS = True
    USE_QUANTUM_STATE_BUS = True
    USE_QUANTUM_COHERENCE_BUS = True
    USE_QUANTUM_TELEPORT_BUS = True
    VQC_GRADIENT_MODE = "spsa"
    QULS_FIDELITY_WEIGHT = 0.1
    ... 190+ more

New approach (preset + overrides):
    config = QuantumConfig.from_preset("balanced")
    # or
    config = QuantumConfig(enhancement_level="full", mps_bond_dim=64)

Benefits:
    - 95% config complexity reduction
    - Sensible defaults for each use case
    - Easy A/B testing via preset switching
    - Backwards compatible via expand_flags()

Reference:
    QUANTUM_ROADMAP.md Phase 6: Configuration Simplification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Enhancement Levels
# =============================================================================


class QuantumEnhancementLevel(Enum):
    """Quantum enhancement levels for easy configuration.

    NONE: Pure classical mode (no quantum features)
    MINIMAL: MPS compression only (memory savings)
    BALANCED: MPS + simplified QULS (recommended)
    FULL: All quantum features enabled
    RESEARCH: Experimental features for development
    """

    NONE = "none"
    MINIMAL = "minimal"
    BALANCED = "balanced"
    FULL = "full"
    RESEARCH = "research"


# =============================================================================
# VQC Gradient Modes
# =============================================================================


class VQCGradientMode(Enum):
    """VQC gradient computation modes."""

    DISABLED = "disabled"
    FULL_PSR = "full_psr"  # Full parameter-shift rule
    GUIDED_SPSA = "guided_spsa"  # Hybrid PSR + SPSA
    PURE_SPSA = "pure_spsa"  # Pure SPSA (fastest)
    AUTO = "auto"  # Auto-select based on param count


# =============================================================================
# Main Configuration
# =============================================================================


@dataclass
class QuantumConfig:
    """Unified quantum configuration with preset support.

    Replaces 200+ individual flags with a single dataclass that:
    - Provides sensible presets for common use cases
    - Allows fine-grained overrides when needed
    - Expands to individual flags for backwards compatibility

    Attributes:
        enhancement_level: Overall quantum enhancement level.
        vqc_gradient_mode: How to compute VQC gradients.
        quls_alpha: Weight for unified quantum loss term.
        quls_beta: Weight for unified structural loss term.
        mps_bond_dim: MPS bond dimension (entanglement capacity).
        bus_num_sites: Number of sites in unified quantum bus.
        bp_initialization: Barren plateau initialization strategy.
        bp_use_layerwise: Whether to use layerwise training.
        zne_enabled: Whether neural ZNE is enabled.
        zne_hidden_dim: ZNE mitigator hidden dimension.
        custom_overrides: Optional dict of individual flag overrides.

    Example:
        >>> config = QuantumConfig.from_preset("balanced")
        >>> flags = config.expand_flags()
        >>> config.mps_bond_dim  # 32 for balanced preset
    """

    # Core settings
    enhancement_level: QuantumEnhancementLevel = QuantumEnhancementLevel.BALANCED

    # VQC gradient settings
    vqc_gradient_mode: VQCGradientMode = VQCGradientMode.AUTO
    vqc_spsa_ratio: float = 0.2  # For guided SPSA

    # QULS v2 settings
    quls_alpha: float = 0.1
    quls_beta: float = 0.01
    quls_target_entropy: float = 0.5
    quls_label_smoothing: float = 0.1

    # MPS/Bus settings
    mps_bond_dim: int = 32
    bus_num_sites: int = 8
    bus_physical_dim: int = 64

    # Barren plateau settings
    bp_initialization: str = "xavier_quantum"
    bp_use_layerwise: bool = True
    bp_warning_threshold: float = 1e-5

    # ZNE settings
    zne_enabled: bool = True
    zne_hidden_dim: int = 128
    zne_num_noise_levels: int = 5

    # Custom overrides (for backwards compat)
    custom_overrides: dict[str, Any] | None = field(default=None)

    @classmethod
    def from_preset(cls, preset: str) -> QuantumConfig:
        """Create config from preset name.

        Args:
            preset: Preset name ("none", "minimal", "balanced", "full", "research").

        Returns:
            Configured QuantumConfig instance.

        Raises:
            ValueError: If preset name is unknown.
        """
        presets = {
            "none": cls(
                enhancement_level=QuantumEnhancementLevel.NONE,
                vqc_gradient_mode=VQCGradientMode.DISABLED,
                quls_alpha=0.0,
                quls_beta=0.0,
                mps_bond_dim=0,
                zne_enabled=False,
            ),
            "minimal": cls(
                enhancement_level=QuantumEnhancementLevel.MINIMAL,
                vqc_gradient_mode=VQCGradientMode.DISABLED,
                quls_alpha=0.0,
                quls_beta=0.01,
                mps_bond_dim=16,
                zne_enabled=False,
            ),
            "balanced": cls(
                enhancement_level=QuantumEnhancementLevel.BALANCED,
                vqc_gradient_mode=VQCGradientMode.AUTO,
                quls_alpha=0.1,
                quls_beta=0.01,
                mps_bond_dim=32,
                zne_enabled=True,
            ),
            "full": cls(
                enhancement_level=QuantumEnhancementLevel.FULL,
                vqc_gradient_mode=VQCGradientMode.FULL_PSR,
                quls_alpha=0.15,
                quls_beta=0.02,
                mps_bond_dim=64,
                zne_enabled=True,
                zne_num_noise_levels=7,
            ),
            "research": cls(
                enhancement_level=QuantumEnhancementLevel.RESEARCH,
                vqc_gradient_mode=VQCGradientMode.GUIDED_SPSA,
                quls_alpha=0.2,
                quls_beta=0.03,
                mps_bond_dim=128,
                bp_use_layerwise=True,
                zne_enabled=True,
                zne_num_noise_levels=9,
            ),
        }

        if preset not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

        config = presets[preset]
        logger.info(f"[QuantumConfig] Created from preset '{preset}'")
        return config

    def expand_flags(self) -> dict[str, Any]:
        """Expand config to individual flags for backwards compatibility.

        Returns:
            Dictionary of individual config flags.
        """
        level = self.enhancement_level
        is_enabled = level != QuantumEnhancementLevel.NONE
        is_full = level in (QuantumEnhancementLevel.FULL, QuantumEnhancementLevel.RESEARCH)

        flags = {
            # Master switches
            "USE_QUANTUM_UNIFIED_LOSS": is_enabled,
            "USE_UNIFIED_QUANTUM_BUS": is_enabled and level != QuantumEnhancementLevel.MINIMAL,
            "USE_MPS_COMPRESSION": self.mps_bond_dim > 0,
            "USE_NEURAL_ZNE": self.zne_enabled,
            "USE_NEURAL_QEM": self.zne_enabled,
            "USE_UNIFIED_ALPHAQUBIT": is_full,
            # VQC settings
            "VQC_GRADIENT_MODE": self.vqc_gradient_mode.value,
            "VQC_SPSA_SAMPLE_RATIO": self.vqc_spsa_ratio,
            # QULS settings
            "QULS_ALPHA": self.quls_alpha,
            "QULS_BETA": self.quls_beta,
            "TARGET_ENTROPY": self.quls_target_entropy,
            "QULS_LABEL_SMOOTHING": self.quls_label_smoothing,
            # MPS/Bus settings
            "MPS_BOND_DIM": self.mps_bond_dim,
            "UNIFIED_BUS_NUM_SITES": self.bus_num_sites,
            "UNIFIED_BUS_PHYSICAL_DIM": self.bus_physical_dim,
            "UNIFIED_BUS_BOND_DIM": self.mps_bond_dim,
            # Barren plateau settings
            "BP_INITIALIZATION": self.bp_initialization,
            "BP_USE_LAYERWISE": self.bp_use_layerwise,
            "BP_WARNING_THRESHOLD": self.bp_warning_threshold,
            # ZNE settings
            "NEURAL_ZNE_HIDDEN_DIM": self.zne_hidden_dim,
            "NEURAL_QEM_HIDDEN_DIM": self.zne_hidden_dim,
            "NEURAL_QEM_NOISE_LEVELS": list(range(1, self.zne_num_noise_levels + 1)),
            # Legacy flags (for backwards compat)
            "USE_PORT_HAMILTONIAN": is_enabled,
            "USE_QSVT_ACTIVATIONS": is_full,
            "USE_QUANTUM_STATE_BUS": False,  # Replaced by unified bus
            "USE_QUANTUM_COHERENCE_BUS": False,  # Replaced by unified bus
            "USE_QUANTUM_TELEPORT_BUS": False,  # Replaced by unified bus
            # Fidelity/Born (now unified into alpha)
            "USE_QUANTUM_FIDELITY_LOSS": is_enabled,
            "QULS_FIDELITY_WEIGHT": self.quls_alpha * 0.6,
            "USE_BORN_RULE_LOSS": is_enabled,
            "QULS_BORN_RULE_WEIGHT": self.quls_alpha * 0.4,
            # Entropy/Spectral (now unified into beta)
            "USE_ENTROPY_REGULARIZATION": is_enabled,
            "ENTROPY_REG_WEIGHT": self.quls_beta * 0.5,
            "SPECTRAL_REG_WEIGHT": self.quls_beta * 0.5,
            # Entanglement
            "ENTANGLEMENT_REGULARIZATION": self.quls_beta,
        }

        # Apply custom overrides
        if self.custom_overrides:
            flags.update(self.custom_overrides)

        return flags

    def summary(self) -> str:
        """Get human-readable config summary.

        Returns:
            Summary string.
        """
        return (
            f"QuantumConfig(\n"
            f"  enhancement_level={self.enhancement_level.value},\n"
            f"  vqc_gradient_mode={self.vqc_gradient_mode.value},\n"
            f"  quls_alpha={self.quls_alpha}, quls_beta={self.quls_beta},\n"
            f"  mps_bond_dim={self.mps_bond_dim},\n"
            f"  zne_enabled={self.zne_enabled},\n"
            f")"
        )

    def __repr__(self) -> str:
        return self.summary()


# =============================================================================
# Global Configuration State
# =============================================================================


_global_config: QuantumConfig | None = None


def get_global_config() -> QuantumConfig:
    """Get global quantum configuration.

    Returns default 'balanced' config if not set.
    """
    global _global_config
    if _global_config is None:
        _global_config = QuantumConfig.from_preset("balanced")
    return _global_config


def set_global_config(config: QuantumConfig | str) -> QuantumConfig:
    """Set global quantum configuration.

    Args:
        config: QuantumConfig instance or preset name.

    Returns:
        The set configuration.
    """
    global _global_config

    if isinstance(config, str):
        config = QuantumConfig.from_preset(config)

    _global_config = config
    logger.info(f"[QuantumConfig] Global config set to {config.enhancement_level.value}")
    return config


# =============================================================================
# Convenience Functions
# =============================================================================


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "QuantumEnhancementLevel",
    "VQCGradientMode",
    "QuantumConfig",
    "get_global_config",
    "set_global_config",
    "set_global_config",
]
