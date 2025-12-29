# highnoon/training/unified_smart_tuner.py
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

"""Unified Smart Tuner - Enterprise Training Parameter Controller.

This module provides a single orchestrated controller that replaces independent
operation of QALRC, BarrenPlateauMonitor, GaLore, and Meta-Controller. It
provides cross-component awareness, global exploration/exploitation mode,
and optional cross-trial memory for learning from previous HPO sweeps.

Key Features:
    - Cross-component awareness: LR adjustments know barren plateau state
    - Global exploration/exploitation mode management
    - Cross-trial memory: Learns from previous HPO sweeps
    - Gradient-norm-aware emergency responses
    - Full observability via metrics and logging

Example:
    >>> config = UnifiedSmartTunerConfig(lr_initial=3e-4)
    >>> tuner = UnifiedSmartTuner(model, optimizer, config)
    >>> decisions = tuner.orchestrate(step, gradients, variables, loss, grad_norm)
    >>> # Apply decisions.learning_rate, decisions.max_grad_norm, etc.

Reference:
    Smart_Tuner_Upgrade.md - Enterprise Unified Smart Tuner specification
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from highnoon.training.vqc_meta_optimizer import VQCTuningDecisions

import numpy as np
import tensorflow as tf

from highnoon import config as hn_config
from highnoon.training.barren_plateau import BarrenPlateauMonitor, LayerMitigation
from highnoon.training.control_bridge import EvolutionTimeControlBridge
from highnoon.training.gradient_compression import TensorGaLoreCompressor
from highnoon.training.quantum_lr_controller import QuantumAdaptiveLRController

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class UnifiedSmartTunerConfig:
    """Configuration for the Unified Smart Tuner.

    Attributes:
        enabled: Whether the unified tuner is active.
        memory_enabled: Enable cross-trial learning.
        memory_path: Path for cross-trial memory persistence.
        coordination_mode: Control aggressiveness ("aggressive", "balanced", "conservative").
        exploration_decay: Exploration factor decay per epoch.
        lr_initial: Initial learning rate.
        lr_min: Minimum allowed learning rate.
        lr_max: Maximum allowed learning rate.
        galore_rank: Target GaLore compression rank.
        galore_adaptive_rank: Adjust rank based on gradient variance.
        barren_plateau_threshold: Gradient norm below which BP is detected.
        barren_plateau_aggressive: Apply multiple mitigations.
        meta_controller_frequency: Batches between meta-controller updates.
        evolution_time_enabled: Enable evolution time control.
        max_grad_norm: Default gradient clipping norm.
        warmup_steps: Number of warmup steps before full tuning.
        emergency_grad_threshold: Gradient norm triggering emergency mode.

    Quantum-Native Settings:
        use_vqc_meta_optimizer: Enable VQC meta-optimizer for tuning decisions.
        vqc_meta_qubits: Number of qubits in VQC.
        vqc_meta_layers: Number of VQC layers.
        vqc_meta_update_frequency: VQC parameter update frequency.
        use_fisher_grouping: Enable Fisher-based layer grouping.
        fisher_num_groups: Number of layer groups.
        fisher_regroup_frequency: Frequency of regrouping.
        fisher_use_qfim: Use QFIM for quantum layers.
        use_inline_qpbt: Enable inline quantum PBT.
        qpbt_population_size: Virtual population size.
        use_qng: Enable Quantum Natural Gradient.
        qng_damping: QNG damping factor.
    """

    # Global tuner settings
    enabled: bool = True
    memory_enabled: bool = True
    memory_path: Path = field(default_factory=lambda: Path("artifacts/tuner_memory"))

    # Coordination settings
    coordination_mode: str = "balanced"  # "aggressive", "balanced", "conservative"
    exploration_decay: float = 0.99

    # Learning rate settings
    lr_initial: float = 3e-4
    lr_min: float = 1e-7
    lr_max: float = 1e-2

    # GaLore settings
    galore_rank: int = field(default_factory=lambda: hn_config.GALORE_RANK)
    galore_adaptive_rank: bool = True
    galore_update_gap: int = field(default_factory=lambda: hn_config.GALORE_UPDATE_PROJ_GAP)
    galore_scale: float = field(default_factory=lambda: hn_config.GALORE_SCALE)

    # Barren plateau settings
    barren_plateau_threshold: float = field(
        default_factory=lambda: hn_config.BARREN_PLATEAU_THRESHOLD
    )
    barren_plateau_aggressive: bool = True

    # Meta-controller settings
    meta_controller_frequency: int = field(
        default_factory=lambda: hn_config.META_CONTROLLER_FREQUENCY
    )
    evolution_time_enabled: bool = True

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Phase control
    warmup_steps: int = 1000
    exploration_steps: int = 10000

    # Emergency thresholds
    emergency_grad_threshold: float = 1e6

    # =========================================================================
    # QUANTUM-NATIVE SETTINGS (Part 6 Enhancement)
    # =========================================================================

    # VQC Meta-Optimizer
    use_vqc_meta_optimizer: bool = True  # Enabled by default for testing
    vqc_meta_qubits: int = 8
    vqc_meta_layers: int = 4
    vqc_meta_update_frequency: int = 50
    vqc_meta_lr: float = 1e-3

    # Fisher Layer Grouping
    use_fisher_grouping: bool = True  # Enabled by default for testing
    fisher_num_groups: int = 4
    fisher_regroup_frequency: int = 100
    fisher_use_qfim: bool = True

    # Inline Quantum PBT
    use_inline_qpbt: bool = True  # Enabled by default for testing
    qpbt_population_size: int = 8
    qpbt_mutation_strength: float = 0.2
    qpbt_crossover_rate: float = 0.3
    qpbt_initial_temperature: float = 1.0
    qpbt_final_temperature: float = 0.01

    # Quantum Natural Gradient Integration
    use_qng: bool = True  # Enabled by default for testing
    qng_damping: float = 1e-4
    qng_apply_to_quantum_only: bool = True


# =============================================================================
# TUNING DECISIONS
# =============================================================================


@dataclass
class TuningDecisions:
    """Container for tuning decisions from the orchestrator.

    This dataclass holds all parameter adjustments decided by the
    UnifiedSmartTuner.orchestrate() method.

    Attributes:
        learning_rate: Computed learning rate to use.
        galore_rank: GaLore compression rank to use.
        max_grad_norm: Gradient clipping norm to use.
        barren_plateau_active: Whether barren plateau mitigation is active.
        bp_mitigations: Active barren plateau mitigations by layer.
        skip_meta_controller: Whether to skip meta-controller update.
        emergency_mode: Whether emergency mode is active.
        phase: Current training phase ("warmup", "exploration", "exploitation", "emergency").
        exploration_factor: Current exploration factor (0-1).
        lr_scale_factor: Applied LR scaling factor.
        additional_metrics: Additional metrics for logging.
    """

    learning_rate: float
    galore_rank: int = 32
    max_grad_norm: float = 1.0
    barren_plateau_active: bool = False
    bp_mitigations: dict[str, LayerMitigation] = field(default_factory=dict)
    skip_meta_controller: bool = False
    emergency_mode: bool = False
    phase: str = "warmup"
    exploration_factor: float = 1.0
    lr_scale_factor: float = 1.0
    additional_metrics: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# UNIFIED SMART TUNER
# =============================================================================


class UnifiedSmartTuner:
    """Enterprise-grade unified training parameter controller.

    This class orchestrates all live training adjustments through a single
    coordinated decision-making framework. It replaces the previous independent
    operation of QALRC, BarrenPlateauMonitor, GaLore, and Meta-Controller.

    Key Features:
        - Cross-component awareness: LR adjustments know barren plateau state
        - Global exploration/exploitation mode
        - Cross-trial memory: Learns from previous HPO sweeps
        - Gradient-norm-aware emergency responses
        - Full observability via metrics and logging

    Attributes:
        model: The Keras model being trained.
        optimizer: The training optimizer.
        config: Unified tuner configuration.

    Example:
        >>> tuner = UnifiedSmartTuner(model, optimizer)
        >>> for step, (inputs, targets) in enumerate(dataset):
        ...     with tf.GradientTape() as tape:
        ...         loss = compute_loss(model, inputs, targets)
        ...     gradients = tape.gradient(loss, model.trainable_variables)
        ...     grad_norm = tf.linalg.global_norm(gradients)
        ...     decisions = tuner.orchestrate(
        ...         step, gradients, model.trainable_variables, loss, grad_norm
        ...     )
        ...     optimizer.learning_rate.assign(decisions.learning_rate)
    """

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        config: UnifiedSmartTunerConfig | None = None,
        total_steps: int = 100000,
    ):
        """Initialize the Unified Smart Tuner.

        Args:
            model: The Keras model to tune.
            optimizer: The training optimizer.
            config: Tuner configuration. Defaults to UnifiedSmartTunerConfig().
            total_steps: Total expected training steps for scheduling.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config or UnifiedSmartTunerConfig()
        self._total_steps = total_steps

        # Global state
        self._global_step = 0
        self._exploration_factor = 1.0
        self._current_phase = "warmup"
        self._emergency_mode = False
        self._emergency_step_count = 0

        # Loss tracking for trend detection
        self._loss_history: list[float] = []
        self._grad_norm_history: list[float] = []

        # Initialize sub-components
        self._init_components()

        # Cross-trial memory (lazy init)
        self._memory = None
        if self.config.memory_enabled:
            self._init_memory()

        logger.info(
            "[SmartTuner] Initialized: mode=%s, lr_initial=%.2e, "
            "galore_rank=%d, bp_threshold=%.2e",
            self.config.coordination_mode,
            self.config.lr_initial,
            self.config.galore_rank,
            self.config.barren_plateau_threshold,
        )

    def _init_components(self) -> None:
        """Initialize wrapped sub-components."""
        # LR Controller
        self._lr_controller = QuantumAdaptiveLRController(
            initial_lr=self.config.lr_initial,
            min_lr=self.config.lr_min,
            max_lr=self.config.lr_max,
        )

        # Barren Plateau Monitor
        self._bp_monitor = BarrenPlateauMonitor(
            threshold=self.config.barren_plateau_threshold,
        )

        # GaLore Compressor
        self._galore = TensorGaLoreCompressor(
            rank=self.config.galore_rank,
            update_proj_gap=self.config.galore_update_gap,
            scale=self.config.galore_scale,
            enabled=self.config.galore_adaptive_rank,
        )

        # Evolution Time Control Bridge (lazy init after model built)
        self._evolution_bridge: EvolutionTimeControlBridge | None = None

        # =====================================================================
        # QUANTUM-NATIVE COMPONENTS (Part 6 Enhancement)
        # =====================================================================

        # VQC Meta-Optimizer
        self._vqc_meta = None
        if self.config.use_vqc_meta_optimizer:
            self._init_vqc_meta_optimizer()

        # Fisher Layer Grouper
        self._fisher_grouper = None
        if self.config.use_fisher_grouping:
            self._init_fisher_grouper()

        # Inline Quantum PBT
        self._inline_qpbt = None
        if self.config.use_inline_qpbt:
            self._init_inline_qpbt()

        # Quantum Natural Gradient
        self._qng = None
        if self.config.use_qng:
            self._init_qng()

    def _init_vqc_meta_optimizer(self) -> None:
        """Initialize VQC Meta-Optimizer component."""
        try:
            from highnoon.training.vqc_meta_optimizer import (
                VQCMetaOptimizer,
                VQCMetaOptimizerConfig,
            )

            vqc_config = VQCMetaOptimizerConfig(
                num_qubits=self.config.vqc_meta_qubits,
                num_layers=self.config.vqc_meta_layers,
                update_frequency=self.config.vqc_meta_update_frequency,
                meta_lr=self.config.vqc_meta_lr,
            )
            self._vqc_meta = VQCMetaOptimizer(vqc_config)
            logger.info("[SmartTuner] VQC Meta-Optimizer enabled")
        except Exception as e:
            logger.warning("[SmartTuner] VQC Meta-Optimizer not available: %s", e)
            self._vqc_meta = None

    def _init_fisher_grouper(self) -> None:
        """Initialize Fisher Layer Grouper component."""
        try:
            from highnoon.training.fisher_layer_grouper import (
                FisherLayerGrouper,
                FisherLayerGrouperConfig,
            )

            fisher_config = FisherLayerGrouperConfig(
                num_groups=self.config.fisher_num_groups,
                regroup_frequency=self.config.fisher_regroup_frequency,
                use_quantum_fisher=self.config.fisher_use_qfim,
            )
            self._fisher_grouper = FisherLayerGrouper(self.model, fisher_config)
            logger.info("[SmartTuner] Fisher Layer Grouper enabled")
        except Exception as e:
            logger.warning("[SmartTuner] Fisher Layer Grouper not available: %s", e)
            self._fisher_grouper = None

    def _init_inline_qpbt(self) -> None:
        """Initialize Inline Quantum PBT component."""
        try:
            from highnoon.training.inline_qpbt import InlineQPBTConfig, InlineQuantumPBT

            initial_config = {
                "lr": self.config.lr_initial,
                "galore_rank": self.config.galore_rank,
                "max_grad_norm": self.config.max_grad_norm,
            }
            qpbt_config = InlineQPBTConfig(
                population_size=self.config.qpbt_population_size,
                mutation_strength=self.config.qpbt_mutation_strength,
                crossover_rate=self.config.qpbt_crossover_rate,
                initial_temperature=self.config.qpbt_initial_temperature,
                final_temperature=self.config.qpbt_final_temperature,
            )
            self._inline_qpbt = InlineQuantumPBT(initial_config, qpbt_config)
            self._inline_qpbt.set_max_steps(self._total_steps)
            logger.info("[SmartTuner] Inline Quantum PBT enabled")
        except Exception as e:
            logger.warning("[SmartTuner] Inline Quantum PBT not available: %s", e)
            self._inline_qpbt = None

    def _init_qng(self) -> None:
        """Initialize Quantum Natural Gradient component."""
        try:
            from highnoon.training.quantum_gradient import QuantumNaturalGradient

            self._qng = QuantumNaturalGradient(
                enabled=True,
                damping=self.config.qng_damping,
                apply_to_quantum_only=self.config.qng_apply_to_quantum_only,
            )
            logger.info("[SmartTuner] Quantum Natural Gradient enabled")
        except Exception as e:
            logger.warning("[SmartTuner] Quantum Natural Gradient not available: %s", e)
            self._qng = None

    def _init_memory(self) -> None:
        """Initialize cross-trial memory."""
        try:
            from highnoon.training.tuner_memory import TunerMemory

            self._memory = TunerMemory(self.config.memory_path)
            logger.info(
                "[SmartTuner] Cross-trial memory enabled: %s",
                self.config.memory_path,
            )
        except ImportError:
            logger.warning(
                "[SmartTuner] Cross-trial memory not available (tuner_memory module missing)"
            )
            self._memory = None

    def ensure_evolution_bridge(self) -> None:
        """Initialize evolution time bridge after model is built."""
        if self._evolution_bridge is None and self.config.evolution_time_enabled:
            self._evolution_bridge = EvolutionTimeControlBridge(self.model)
            self._evolution_bridge.ensure_discovery()

    def orchestrate(
        self,
        step: int,
        gradients: list[tf.Tensor],
        variables: list[tf.Variable],
        loss: float,
        gradient_norm: float,
    ) -> TuningDecisions:
        """Single entry point for all live tuning decisions.

        This is the main method called by TrainingEngine.train_step().
        It coordinates all sub-components and returns unified decisions.

        Args:
            step: Current training step.
            gradients: Computed gradients (before compression/clipping).
            variables: Corresponding trainable variables.
            loss: Current loss value.
            gradient_norm: Pre-clipping gradient norm.

        Returns:
            TuningDecisions with all parameter adjustments.
        """
        if not self.config.enabled:
            return TuningDecisions(
                learning_rate=self.config.lr_initial,
                galore_rank=self.config.galore_rank,
                max_grad_norm=self.config.max_grad_norm,
            )

        self._global_step = step

        # Update history
        if math.isfinite(loss):
            self._loss_history.append(loss)
            if len(self._loss_history) > 100:
                self._loss_history = self._loss_history[-50:]

        if math.isfinite(gradient_norm):
            self._grad_norm_history.append(gradient_norm)
            if len(self._grad_norm_history) > 100:
                self._grad_norm_history = self._grad_norm_history[-50:]

        # 1. EMERGENCY CHECK: Gradient explosion or NaN
        if gradient_norm > self.config.emergency_grad_threshold or not math.isfinite(gradient_norm):
            return self._emergency_response(gradient_norm, loss)

        # Check if recovering from emergency
        if self._emergency_mode:
            self._emergency_step_count += 1
            if self._emergency_step_count > 50 and gradient_norm < 1e4:
                logger.info(
                    "[SmartTuner] Exiting emergency mode after %d recovery steps",
                    self._emergency_step_count,
                )
                self._emergency_mode = False
                self._emergency_step_count = 0

        # 2. PHASE DETERMINATION
        self._update_phase(step, loss, gradient_norm)

        # 3. GATHER SIGNALS from all sub-components
        bp_detected = self._bp_monitor.update(
            gradient_norm=gradient_norm,
            gradients=gradients,
            variables=variables,
        )
        entropy = self._lr_controller.compute_gradient_entropy(gradients)
        loss_trend = self._lr_controller.compute_loss_trend()

        # =====================================================================
        # QUANTUM-NATIVE INTEGRATION (Part 6 Enhancement)
        # =====================================================================

        # 3a. Update Fisher Layer Grouper with gradient info
        layer_fisher_info = {}
        if self._fisher_grouper is not None:
            self._fisher_grouper.update_fisher_estimates(gradients, variables)
            if step % self.config.fisher_regroup_frequency == 0:
                self._fisher_grouper.regroup_layers(step)
            layer_fisher_info = self._fisher_grouper.get_fisher_estimates()

        # 3b. Gather barren plateau scores per layer
        barren_plateau_scores = {}
        if hasattr(self._bp_monitor, "get_layer_scores"):
            barren_plateau_scores = self._bp_monitor.get_layer_scores()

        # 3c. VQC Meta-Optimizer: Quantum-learned tuning decisions
        vqc_decisions = None
        if self._vqc_meta is not None and not self._emergency_mode:
            try:

                vqc_decisions = self._vqc_meta.compute_tuning_decisions(
                    loss=loss,
                    gradient_norm=gradient_norm,
                    layer_fisher_info=layer_fisher_info,
                    barren_plateau_scores=barren_plateau_scores,
                    exploration_factor=self._exploration_factor,
                )
                # Update VQC params periodically
                self._vqc_meta.maybe_update_params(step)
            except Exception as e:
                logger.warning("[SmartTuner] VQC decision failed: %s", e)
                vqc_decisions = None

        # 3d. Inline QPBT: Record performance and evolve
        if self._inline_qpbt is not None:
            self._inline_qpbt.record_performance(loss, step)
            if self._inline_qpbt.maybe_evolve(step):
                qpbt_config = self._inline_qpbt.get_current_config()
                logger.info(
                    "[SmartTuner] QPBT evolved config: lr=%.2e, rank=%d",
                    qpbt_config.get("lr", self.config.lr_initial),
                    qpbt_config.get("galore_rank", self.config.galore_rank),
                )

        # 4. COORDINATED DECISION MAKING (with quantum enhancement)
        if bp_detected and self._current_phase != "emergency":
            return self._barren_plateau_response(entropy, gradient_norm, vqc_decisions)

        elif self._current_phase == "exploration":
            return self._exploration_response(entropy, loss_trend, gradient_norm, vqc_decisions)

        elif self._current_phase == "exploitation":
            return self._exploitation_response(step, gradient_norm, vqc_decisions)

        else:  # warmup
            return self._warmup_response(step, gradient_norm)

    def _update_phase(self, step: int, loss: float, gradient_norm: float) -> None:
        """Update the current training phase based on step and metrics."""
        if self._emergency_mode:
            self._current_phase = "emergency"
            return

        if step < self.config.warmup_steps:
            self._current_phase = "warmup"
        elif step < self.config.exploration_steps:
            self._current_phase = "exploration"
            # Decay exploration factor
            progress = (step - self.config.warmup_steps) / max(
                1, self.config.exploration_steps - self.config.warmup_steps
            )
            self._exploration_factor = 1.0 - 0.8 * progress
        else:
            self._current_phase = "exploitation"
            self._exploration_factor = 0.2

    def _emergency_response(self, gradient_norm: float, loss: float) -> TuningDecisions:
        """Emergency response for gradient explosion."""
        self._emergency_mode = True
        self._emergency_step_count = 0

        # Aggressive LR reduction
        emergency_lr = self.config.lr_min

        # Reduce GaLore rank for stability
        emergency_galore_rank = max(8, self.config.galore_rank // 4)

        logger.error(
            "[SmartTuner] EMERGENCY: grad_norm=%.2e, loss=%.2f. "
            "Reducing LR to %.2e, GaLore rank to %d",
            gradient_norm,
            loss,
            emergency_lr,
            emergency_galore_rank,
        )

        return TuningDecisions(
            learning_rate=emergency_lr,
            galore_rank=emergency_galore_rank,
            max_grad_norm=0.1,  # Aggressive clipping
            skip_meta_controller=True,
            emergency_mode=True,
            phase="emergency",
            exploration_factor=self._exploration_factor,
            lr_scale_factor=0.01,
            additional_metrics={
                "emergency_trigger_grad_norm": gradient_norm,
                "emergency_trigger_loss": loss,
            },
        )

    def _barren_plateau_response(
        self,
        entropy: float,
        gradient_norm: float,
        vqc_decisions: VQCTuningDecisions | None = None,
    ) -> TuningDecisions:
        """Coordinated response to barren plateau detection.

        Args:
            entropy: Gradient entropy measure.
            gradient_norm: Current gradient norm.
            vqc_decisions: Optional VQC-computed tuning decisions.

        Returns:
            TuningDecisions for barren plateau mitigation.
        """
        # Get LR scale from BP monitor
        lr_scale = self._bp_monitor.get_lr_scale_factor()

        # Apply VQC modulation if available
        if vqc_decisions is not None and vqc_decisions.vqc_computed:
            lr_scale *= vqc_decisions.learning_rate_multiplier
            logger.debug(
                "[SmartTuner] VQC modulating BP LR scale: %.2fx",
                vqc_decisions.learning_rate_multiplier,
            )

        # COORDINATION: If high entropy, scale LR more aggressively
        if entropy > 0.7:
            lr_scale *= 1.5
            logger.info(
                "[SmartTuner] High entropy during BP (%.2f), boosting LR scale to %.2fx",
                entropy,
                lr_scale,
            )

        # Compute coordinated LR
        base_lr = self._lr_controller.get_learning_rate(
            step=self._global_step,
            total_steps=self._total_steps,
            current_loss=None,  # Don't use loss trend during BP
        )
        effective_lr = base_lr * lr_scale
        effective_lr = max(self.config.lr_min, min(self.config.lr_max, effective_lr))

        # COORDINATION: Increase GaLore rank to preserve gradient information
        adjusted_galore_rank = min(64, self.config.galore_rank * 2)

        # Apply VQC GaLore rank scale if available
        if vqc_decisions is not None and vqc_decisions.vqc_computed:
            adjusted_galore_rank = int(adjusted_galore_rank * vqc_decisions.galore_rank_scale)
            adjusted_galore_rank = max(8, min(128, adjusted_galore_rank))

        # Relax gradient clipping during BP
        relaxed_clip = self.config.max_grad_norm * 2.0
        if vqc_decisions is not None and vqc_decisions.vqc_computed:
            relaxed_clip *= vqc_decisions.max_grad_norm_multiplier

        return TuningDecisions(
            learning_rate=effective_lr,
            galore_rank=adjusted_galore_rank,
            max_grad_norm=relaxed_clip,
            barren_plateau_active=True,
            bp_mitigations=self._bp_monitor.get_active_mitigations(),
            phase=self._current_phase,
            exploration_factor=self._exploration_factor,
            lr_scale_factor=lr_scale,
            additional_metrics={
                "bp_detected": True,
                "bp_lr_scale": lr_scale,
                "gradient_entropy": entropy,
                "vqc_computed": vqc_decisions.vqc_computed if vqc_decisions else False,
            },
        )

    def _exploration_response(
        self,
        entropy: float,
        loss_trend: float,
        gradient_norm: float,
        vqc_decisions: VQCTuningDecisions | None = None,
    ) -> TuningDecisions:
        """Response during exploration phase - higher variance allowed.

        Args:
            entropy: Gradient entropy measure.
            loss_trend: Recent loss trend.
            gradient_norm: Current gradient norm.
            vqc_decisions: Optional VQC-computed tuning decisions.

        Returns:
            TuningDecisions for exploration phase.
        """
        # Get base LR from controller
        base_lr = self._lr_controller.get_learning_rate(
            step=self._global_step,
            total_steps=self._total_steps,
            current_loss=self._loss_history[-1] if self._loss_history else None,
        )

        # Scale based on coordination mode
        if self.config.coordination_mode == "aggressive":
            lr_scale = 1.0 + 0.3 * self._exploration_factor
        elif self.config.coordination_mode == "conservative":
            lr_scale = 1.0 + 0.1 * self._exploration_factor
        else:  # balanced
            lr_scale = 1.0 + 0.2 * self._exploration_factor

        # Apply VQC modulation if available
        galore_rank = self.config.galore_rank
        max_grad_norm = self.config.max_grad_norm
        if vqc_decisions is not None and vqc_decisions.vqc_computed:
            lr_scale *= vqc_decisions.learning_rate_multiplier
            galore_rank = int(galore_rank * vqc_decisions.galore_rank_scale)
            galore_rank = max(8, min(128, galore_rank))
            max_grad_norm *= vqc_decisions.max_grad_norm_multiplier
            # Add VQC exploration boost
            lr_scale *= 1.0 + vqc_decisions.exploration_boost * 0.1

        effective_lr = base_lr * lr_scale
        effective_lr = max(self.config.lr_min, min(self.config.lr_max, effective_lr))

        return TuningDecisions(
            learning_rate=effective_lr,
            galore_rank=galore_rank,
            max_grad_norm=max_grad_norm,
            phase="exploration",
            exploration_factor=self._exploration_factor,
            lr_scale_factor=lr_scale,
            additional_metrics={
                "gradient_entropy": entropy,
                "loss_trend": loss_trend,
                "vqc_computed": vqc_decisions.vqc_computed if vqc_decisions else False,
            },
        )

    def _exploitation_response(
        self,
        step: int,
        gradient_norm: float,
        vqc_decisions: VQCTuningDecisions | None = None,
    ) -> TuningDecisions:
        """Response during exploitation phase - stable, lower LR.

        Args:
            step: Current training step.
            gradient_norm: Current gradient norm.
            vqc_decisions: Optional VQC-computed tuning decisions.

        Returns:
            TuningDecisions for exploitation phase.
        """
        # Get base LR with full annealing
        base_lr = self._lr_controller.get_learning_rate(
            step=step,
            total_steps=self._total_steps,
            current_loss=self._loss_history[-1] if self._loss_history else None,
        )

        # Conservative scaling in exploitation
        if self.config.coordination_mode == "aggressive":
            lr_scale = 0.9
        elif self.config.coordination_mode == "conservative":
            lr_scale = 0.7
        else:  # balanced
            lr_scale = 0.8

        # Apply VQC modulation if available
        galore_rank = self.config.galore_rank
        max_grad_norm = self.config.max_grad_norm
        if vqc_decisions is not None and vqc_decisions.vqc_computed:
            lr_scale *= vqc_decisions.learning_rate_multiplier
            galore_rank = int(galore_rank * vqc_decisions.galore_rank_scale)
            galore_rank = max(8, min(128, galore_rank))
            max_grad_norm *= vqc_decisions.max_grad_norm_multiplier

        effective_lr = base_lr * lr_scale
        effective_lr = max(self.config.lr_min, min(self.config.lr_max, effective_lr))

        return TuningDecisions(
            learning_rate=effective_lr,
            galore_rank=galore_rank,
            max_grad_norm=max_grad_norm,
            phase="exploitation",
            exploration_factor=self._exploration_factor,
            lr_scale_factor=lr_scale,
            additional_metrics={
                "vqc_computed": vqc_decisions.vqc_computed if vqc_decisions else False,
            },
        )

    def _warmup_response(
        self,
        step: int,
        gradient_norm: float,
    ) -> TuningDecisions:
        """Response during warmup phase - linear LR ramp."""
        # Linear warmup
        warmup_progress = step / max(1, self.config.warmup_steps)
        warmup_lr = self.config.lr_initial * warmup_progress
        warmup_lr = max(self.config.lr_min, warmup_lr)

        # Conservative gradient clipping during warmup
        warmup_clip = self.config.max_grad_norm * 0.5

        return TuningDecisions(
            learning_rate=warmup_lr,
            galore_rank=self.config.galore_rank,
            max_grad_norm=warmup_clip,
            phase="warmup",
            exploration_factor=1.0,
            lr_scale_factor=warmup_progress,
            additional_metrics={
                "warmup_progress": warmup_progress,
            },
        )

    def compress_gradients(
        self,
        gradients: list[tf.Tensor],
        variables: list[tf.Variable],
        decisions: TuningDecisions,
    ) -> list[tuple[tf.Tensor, str]]:
        """Compress gradients using GaLore with dynamic rank from decisions.

        Args:
            gradients: List of gradient tensors.
            variables: Corresponding trainable variables.
            decisions: Tuning decisions with galore_rank.

        Returns:
            List of (compressed_gradient, variable_id) tuples.
        """
        if not self.config.galore_adaptive_rank:
            return [(g, str(id(v))) for g, v in zip(gradients, variables)]

        # Update GaLore rank based on decisions
        compressed = []
        for grad, var in zip(gradients, variables):
            if grad is None:
                compressed.append((None, str(id(var))))
            else:
                comp_grad, var_id = self._galore.compress(grad, var)
                compressed.append((comp_grad, var_id))

        self._galore.step()
        return compressed

    def decompress_gradients(
        self,
        compressed_grads: list[tuple[tf.Tensor, str]],
    ) -> list[tf.Tensor]:
        """Decompress gradients after optimizer update.

        Args:
            compressed_grads: List of (compressed_gradient, variable_id) tuples.

        Returns:
            List of decompressed gradient tensors.
        """
        return [
            self._galore.decompress(cg, vid) if cg is not None else None
            for cg, vid in compressed_grads
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get unified tuner statistics for logging/monitoring.

        Returns:
            Dictionary with current state and sub-component statistics.
        """
        stats = {
            "enabled": self.config.enabled,
            "global_step": self._global_step,
            "current_phase": self._current_phase,
            "exploration_factor": self._exploration_factor,
            "emergency_mode": self._emergency_mode,
            "coordination_mode": self.config.coordination_mode,
        }

        # Add sub-component stats
        stats["lr_controller"] = self._lr_controller.get_statistics()
        stats["bp_monitor"] = self._bp_monitor.get_statistics()
        stats["galore"] = self._galore.get_statistics()

        # Add loss/grad history summaries
        if self._loss_history:
            stats["loss_mean"] = np.mean(self._loss_history)
            stats["loss_std"] = np.std(self._loss_history)
        if self._grad_norm_history:
            stats["grad_norm_mean"] = np.mean(self._grad_norm_history)
            stats["grad_norm_std"] = np.std(self._grad_norm_history)

        return stats

    def record_trial(
        self,
        trial_id: str,
        architecture_config: dict,
        hyperparameters: dict,
        final_loss: float,
        best_epoch: int,
    ) -> None:
        """Record trial results for cross-trial learning.

        Args:
            trial_id: Unique identifier for the trial.
            architecture_config: Model architecture configuration.
            hyperparameters: Training hyperparameters.
            final_loss: Final validation loss.
            best_epoch: Epoch with best validation loss.
        """
        if self._memory is None:
            return

        # Build trajectory from history
        trajectory = []
        for i, (loss, grad_norm) in enumerate(zip(self._loss_history, self._grad_norm_history)):
            trajectory.append(
                {
                    "step": i,
                    "learning_rate": (
                        self._lr_controller._state.lr_history[i]
                        if i < len(self._lr_controller._state.lr_history)
                        else self.config.lr_initial
                    ),
                    "galore_rank": self.config.galore_rank,
                    "loss": loss,
                    "grad_norm": grad_norm,
                }
            )

        self._memory.record_trial(
            trial_id=trial_id,
            architecture_config=architecture_config,
            hyperparameters=hyperparameters,
            final_loss=final_loss,
            best_epoch=best_epoch,
            tuner_trajectory=trajectory,
        )

    def suggest_initial_config(self, architecture_config: dict) -> dict[str, Any]:
        """Get suggested initial configuration from cross-trial memory.

        Args:
            architecture_config: Current model architecture configuration.

        Returns:
            Dictionary with suggested initial parameters.
        """
        if self._memory is None:
            return {}

        return self._memory.suggest_initial_config(architecture_config)

    def reset(self) -> None:
        """Reset all tuner state for a new trial."""
        self._global_step = 0
        self._exploration_factor = 1.0
        self._current_phase = "warmup"
        self._emergency_mode = False
        self._emergency_step_count = 0
        self._loss_history.clear()
        self._grad_norm_history.clear()

        # Reset sub-components
        self._lr_controller.reset()
        self._bp_monitor.reset()
        self._galore.reset()

        logger.info("[SmartTuner] State reset for new trial")


__all__ = [
    "UnifiedSmartTuner",
    "UnifiedSmartTunerConfig",
    "TuningDecisions",
]
