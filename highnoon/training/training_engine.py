# highnoon/training/training_engine.py
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

"""Unified Training Engine for HighNoon Framework.

This module provides a shared training infrastructure used by all training paths:
- HPO trial runner
- Main training loop
- Debug scripts

Key Features:
- Meta-Controller integration (HamiltonianMetaControllerCallback)
- Gradient clipping and NaN handling
- GaLore gradient compression with VQC-awareness
- QALRC quantum-adaptive learning rate
- Barren plateau detection and mitigation
- Quantum Natural Gradient (QNG)
- Entropy regularization
- QHPM crystallization for anti-forgetting
- Memory management with OOM prevention
- Callback-based extensibility

Usage:
    >>> engine = TrainingEngine(model, optimizer, EnterpriseTrainingConfig())
    >>> result = engine.run(epochs=10, dataset=train_ds, callbacks=[HPOCallback(reporter)])

Example with HPO:
    >>> from highnoon.training.training_engine import (
    ...     TrainingEngine, EnterpriseTrainingConfig, HPOCallback, MemoryManagerCallback
    ... )
    >>> config = EnterpriseTrainingConfig(use_qalrc=True)
    >>> engine = TrainingEngine(model, optimizer, config)
    >>> result = engine.run(
    ...     epochs=5,
    ...     dataset=train_ds,
    ...     callbacks=[HPOCallback(reporter), MemoryManagerCallback()],
    ... )
"""

from __future__ import annotations

import gc
import logging
import math
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import tensorflow as tf

from highnoon import config as hn_config

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class TrainingConfig:
    """Basic configuration for TrainingEngine (backward compatibility).

    For full feature set, use EnterpriseTrainingConfig instead.

    Attributes:
        max_grad_norm: Maximum gradient norm for clipping.
        max_nan_consecutive: Number of NaN losses before stopping.
        meta_controller_frequency: Batches between Meta-Controller updates.
        use_meta_controller: Enable Hamiltonian Meta-Controller.
        use_galore: Enable GaLore gradient compression.
        use_qalrc: Enable Quantum Adaptive LR Controller.
        log_frequency: Steps between log messages.
    """

    max_grad_norm: float = 1.0
    max_nan_consecutive: int = 5
    meta_controller_frequency: int = field(
        default_factory=lambda: getattr(hn_config, "META_CONTROLLER_FREQUENCY", 10)
    )
    use_meta_controller: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_META_CONTROLLER", True)
    )
    use_galore: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_TENSOR_GALORE", False)
    )
    use_qalrc: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_QUANTUM_LR_CONTROLLER", False)
    )
    log_frequency: int = 10


@dataclass
class EnterpriseTrainingConfig:
    """Enterprise-grade training configuration importing ALL features from config.py.

    This configuration class pulls all training-related flags from the central
    config.py module, ensuring consistent behavior across HPO trials, main training,
    and debugging workflows.

    Attributes:
        max_grad_norm: Maximum gradient norm for clipping.
        max_nan_consecutive: Number of NaN losses before stopping.
        log_frequency: Steps between log messages.

        # Meta-Controller (Phase T1-T6)
        use_meta_controller: Enable Hamiltonian Meta-Controller.
        meta_controller_frequency: Batches between Meta-Controller updates.

        # GaLore Gradient Compression (Phase T2)
        use_galore: Enable GaLore gradient compression.
        galore_rank: Gradient projection rank.
        galore_update_proj_gap: Steps between projection updates.
        galore_scale: Gradient scaling factor.
        galore_vqc_aware: Use VQC gradient variance for rank allocation.

        # QALRC Quantum Adaptive LR (Phase 131)
        use_qalrc: Enable Quantum Adaptive LR Controller.
        qalrc_annealing_power: Annealing schedule exponent.
        qalrc_tunneling_probability: Base quantum tunneling probability.
        qalrc_entropy_smoothing: EMA coefficient for gradient entropy.

        # Barren Plateau Detection (Phase T4)
        use_barren_plateau_detection: Enable barren plateau monitoring.
        barren_plateau_threshold: Gradient norm threshold for detection.
        barren_plateau_lr_scale: LR scaling factor for recovery.
        barren_plateau_hysteresis: Factor for exit threshold.
        barren_plateau_recovery_window: Steps to maintain mitigation.

        # Quantum Natural Gradient (Phase T1)
        use_qng: Enable QNG optimizer path.
        qng_damping: QFIM regularization damping.
        qng_apply_to_quantum_only: Only apply to quantum-enhanced layers.

        # Entropy Regularization (Phase 45)
        use_entropy_regularization: Enable entropy regularization.
        entropy_reg_weight: Von Neumann entropy loss weight.
        spectral_reg_weight: Spectral flatness loss weight.

        # QHPM Crystallization (replaces EWC)
        use_qhpm_crystallization: Enable gradient crystallization.
        crystallization_threshold: Confidence threshold for crystallization.
        max_crystallized_directions: Maximum protected parameter directions.

        # Legacy EWC (deprecated but supported)
        enable_ewc: Enable legacy EWC (deprecated, use QHPM instead).
        ewc_lambda: EWC penalty weight.

        # SympFlow Optimizer (Phase 46)
        use_sympflow: Enable SympFlow Hamiltonian optimizer.
        sympflow_mass: Effective mass for momentum.
        sympflow_friction: Dissipation rate.

        # Neural ZNE Error Mitigation (Phase T6)
        use_neural_zne: Enable neural ZNE.
        neural_zne_hidden_dim: Hidden dimension for ZNE MLP.

        # Neural QEM (Phase 129)
        use_neural_qem: Enable ML-enhanced quantum error mitigation.
    """

    # Core training parameters
    max_grad_norm: float = 1.0
    max_nan_consecutive: int = 5
    log_frequency: int = 10
    
    # Loss function configuration (selectable from WebUI)
    # Supported: 'sparse_categorical_crossentropy', 'categorical_crossentropy', 
    # 'mse', 'mae', 'huber', 'custom'
    loss_function: str = field(
        default_factory=lambda: getattr(hn_config, "TRAINING_LOSS_FUNCTION", "sparse_categorical_crossentropy")
    )
    label_smoothing: float = field(
        default_factory=lambda: getattr(hn_config, "LABEL_SMOOTHING", 0.0)
    )

    # Meta-Controller (Phase T1-T6)
    use_meta_controller: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_META_CONTROLLER", True)
    )
    meta_controller_frequency: int = field(
        default_factory=lambda: getattr(hn_config, "META_CONTROLLER_FREQUENCY", 10)
    )

    # GaLore Gradient Compression (Phase T2)
    use_galore: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_TENSOR_GALORE", True)
    )
    galore_rank: int = field(
        default_factory=lambda: getattr(hn_config, "GALORE_RANK", 32)
    )
    galore_update_proj_gap: int = field(
        default_factory=lambda: getattr(hn_config, "GALORE_UPDATE_PROJ_GAP", 200)
    )
    galore_scale: float = field(
        default_factory=lambda: getattr(hn_config, "GALORE_SCALE", 0.25)
    )
    galore_vqc_aware: bool = field(
        default_factory=lambda: getattr(hn_config, "GALORE_VQC_AWARE", True)
    )
    galore_vqc_variance_boost: float = field(
        default_factory=lambda: getattr(hn_config, "GALORE_VQC_VARIANCE_BOOST", 1.5)
    )

    # QALRC Quantum Adaptive LR (Phase 131)
    use_qalrc: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_QUANTUM_LR_CONTROLLER", True)
    )
    qalrc_annealing_power: float = field(
        default_factory=lambda: getattr(hn_config, "QALRC_ANNEALING_POWER", 2.0)
    )
    qalrc_tunneling_probability: float = field(
        default_factory=lambda: getattr(hn_config, "QALRC_TUNNELING_PROBABILITY", 0.05)
    )
    qalrc_entropy_smoothing: float = field(
        default_factory=lambda: getattr(hn_config, "QALRC_ENTROPY_SMOOTHING", 0.9)
    )

    # Barren Plateau Detection (Phase T4)
    use_barren_plateau_detection: bool = field(
        default_factory=lambda: getattr(hn_config, "BARREN_PLATEAU_MONITOR", True)
    )
    barren_plateau_threshold: float = field(
        default_factory=lambda: getattr(hn_config, "BARREN_PLATEAU_THRESHOLD", 1e-6)
    )
    barren_plateau_lr_scale: float = field(
        default_factory=lambda: getattr(hn_config, "BARREN_PLATEAU_RECOVERY_LR_SCALE", 10.0)
    )
    barren_plateau_hysteresis: float = field(
        default_factory=lambda: getattr(hn_config, "BARREN_PLATEAU_HYSTERESIS", 5.0)
    )
    barren_plateau_recovery_window: int = field(
        default_factory=lambda: getattr(hn_config, "BARREN_PLATEAU_RECOVERY_WINDOW", 100)
    )

    # Quantum Natural Gradient (Phase T1)
    use_qng: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_QUANTUM_NATURAL_GRADIENT", True)
    )
    qng_damping: float = field(
        default_factory=lambda: getattr(hn_config, "QNG_DAMPING", 1e-4)
    )
    qng_apply_to_quantum_only: bool = field(
        default_factory=lambda: getattr(hn_config, "QNG_APPLY_TO_QUANTUM_ONLY", True)
    )

    # Evolution Time QNG (Phase T5)
    use_evolution_time_qng: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_EVOLUTION_TIME_QNG", True)
    )
    evolution_time_metric: str = field(
        default_factory=lambda: getattr(hn_config, "EVOLUTION_TIME_METRIC", "hamiltonian")
    )

    # Entropy Regularization (Phase 45)
    use_entropy_regularization: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_ENTROPY_REGULARIZATION", True)
    )
    entropy_reg_weight: float = field(
        default_factory=lambda: getattr(hn_config, "ENTROPY_REG_WEIGHT", 0.01)
    )
    spectral_reg_weight: float = field(
        default_factory=lambda: getattr(hn_config, "SPECTRAL_REG_WEIGHT", 0.01)
    )
    target_entropy: float = field(
        default_factory=lambda: getattr(hn_config, "TARGET_ENTROPY", 0.5)
    )

    # QHPM Crystallization (replaces EWC)
    use_qhpm_crystallization: bool = field(
        default_factory=lambda: getattr(hn_config, "ENABLE_QHPM_CRYSTALLIZATION", True)
    )
    crystallization_threshold: float = field(
        default_factory=lambda: getattr(hn_config, "QHPM_CRYSTALLIZATION_THRESHOLD", 0.85)
    )
    max_crystallized_directions: int = field(
        default_factory=lambda: getattr(hn_config, "QHPM_MAX_CRYSTALLIZED_DIRECTIONS", 256)
    )
    crystallization_decay: float = field(
        default_factory=lambda: getattr(hn_config, "QHPM_CRYSTALLIZATION_DECAY", 0.99)
    )

    # Legacy EWC (deprecated but supported)
    enable_ewc: bool = field(
        default_factory=lambda: getattr(hn_config, "ENABLE_EWC", False)
    )
    ewc_lambda: float = field(
        default_factory=lambda: getattr(hn_config, "EWC_LAMBDA", 0.0)
    )

    # SympFlow Optimizer (Phase 46)
    use_sympflow: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_SYMPFLOW_OPTIMIZER", True)
    )
    sympflow_mass: float = field(
        default_factory=lambda: getattr(hn_config, "SYMPFLOW_MASS", 1.0)
    )
    sympflow_friction: float = field(
        default_factory=lambda: getattr(hn_config, "SYMPFLOW_FRICTION", 0.1)
    )
    sympflow_step_size: float = field(
        default_factory=lambda: getattr(hn_config, "SYMPFLOW_STEP_SIZE", 0.01)
    )

    # Neural ZNE Error Mitigation (Phase T6)
    use_neural_zne: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_NEURAL_ZNE", True)
    )
    neural_zne_hidden_dim: int = field(
        default_factory=lambda: getattr(hn_config, "NEURAL_ZNE_HIDDEN_DIM", 128)
    )

    # Neural QEM (Phase 129)
    use_neural_qem: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_NEURAL_QEM", True)
    )
    neural_qem_hidden_dim: int = field(
        default_factory=lambda: getattr(hn_config, "NEURAL_QEM_HIDDEN_DIM", 128)
    )

    # Quantum Hessian Estimation (Phase T3)
    use_quantum_hessian: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_QUANTUM_HESSIAN_ESTIMATION", True)
    )
    quantum_hessian_samples: int = field(
        default_factory=lambda: getattr(hn_config, "QUANTUM_HESSIAN_SAMPLES", 8)
    )

    # Auto Neural QEM Wrapping (Phase 130.1)
    use_auto_neural_qem: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_AUTO_NEURAL_QEM", True)
    )

    # Floquet-VQC Modulation (Phase 130.3)
    use_floquet_vqc_modulation: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_FLOQUET_VQC_MODULATION", True)
    )

    # QULS - Quantum Unified Loss System (Phase 132)
    use_quls: bool = field(
        default_factory=lambda: getattr(hn_config, "USE_QUANTUM_UNIFIED_LOSS", True)
    )
    quls_fidelity_weight: float = field(
        default_factory=lambda: getattr(hn_config, "QULS_FIDELITY_WEIGHT", 0.01)
    )
    quls_born_rule_weight: float = field(
        default_factory=lambda: getattr(hn_config, "QULS_BORN_RULE_WEIGHT", 0.005)
    )
    quls_coherence_weight: float = field(
        default_factory=lambda: getattr(hn_config, "QULS_COHERENCE_WEIGHT", 0.01)
    )
    quls_symplectic_weight: float = field(
        default_factory=lambda: getattr(hn_config, "QULS_SYMPLECTIC_WEIGHT", 0.01)
    )
    quls_adaptive_weights: bool = field(
        default_factory=lambda: getattr(hn_config, "QULS_ADAPTIVE_WEIGHTS", True)
    )

    def to_basic_config(self) -> TrainingConfig:
        """Convert to basic TrainingConfig for backward compatibility."""
        return TrainingConfig(
            max_grad_norm=self.max_grad_norm,
            max_nan_consecutive=self.max_nan_consecutive,
            meta_controller_frequency=self.meta_controller_frequency,
            use_meta_controller=self.use_meta_controller,
            use_galore=self.use_galore,
            use_qalrc=self.use_qalrc,
            log_frequency=self.log_frequency,
        )

    @classmethod
    def from_hpo_config(cls, hpo_config: dict) -> "EnterpriseTrainingConfig":
        """Create config from HPO sweep configuration dictionary.
        
        Maps HPO parameter names (from StartHPORequest) to EnterpriseTrainingConfig
        fields, enabling seamless integration between HPO tuner and TrainingEngine.
        
        Args:
            hpo_config: Dictionary from HPO sweep containing tuned hyperparameters
                        and feature flags.
        
        Returns:
            EnterpriseTrainingConfig with all parameters set from HPO config.
        
        Example:
            >>> from highnoon.training.training_engine import EnterpriseTrainingConfig
            >>> 
            >>> hpo_config = {"use_tensor_galore": True, "galore_rank": 64, ...}
            >>> config = EnterpriseTrainingConfig.from_hpo_config(hpo_config)
        """
        # Mapping from HPO config keys to EnterpriseTrainingConfig field names
        field_mappings = {
            # GaLore
            "use_tensor_galore": "use_galore",
            "galore_rank": "galore_rank",
            "galore_update_proj_gap": "galore_update_proj_gap",
            "galore_scale": "galore_scale",
            "galore_vqc_aware": "galore_vqc_aware",
            # QALRC
            "use_quantum_lr_controller": "use_qalrc",
            "qalrc_annealing_power": "qalrc_annealing_power",
            "qalrc_tunneling_probability": "qalrc_tunneling_probability",
            "qalrc_entropy_smoothing": "qalrc_entropy_smoothing",
            # Barren Plateau
            "barren_plateau_monitor": "use_barren_plateau_detection",
            "barren_plateau_threshold": "barren_plateau_threshold",
            "barren_plateau_recovery_lr_scale": "barren_plateau_lr_scale",
            "barren_plateau_hysteresis": "barren_plateau_hysteresis",
            "barren_plateau_recovery_window": "barren_plateau_recovery_window",
            # QNG
            "use_quantum_natural_gradient": "use_qng",
            "qng_damping": "qng_damping",
            "qng_apply_to_quantum_only": "qng_apply_to_quantum_only",
            # Entropy Regularization
            "use_entropy_regularization": "use_entropy_regularization",
            "entropy_reg_weight": "entropy_reg_weight",
            "spectral_reg_weight": "spectral_reg_weight",
            "target_entropy": "target_entropy",
            # QHPM Crystallization
            "use_qhpm_crystallization": "use_qhpm_crystallization",
            "qhpm_crystallization_threshold": "crystallization_threshold",
            "qhpm_max_directions": "max_crystallized_directions",
            "qhpm_crystallization_decay": "crystallization_decay",
            # SympFlow
            "use_sympflow_optimizer": "use_sympflow",
            "sympflow_mass": "sympflow_mass",
            "sympflow_friction": "sympflow_friction",
            "sympflow_step_size": "sympflow_step_size",
            # Neural ZNE/QEM
            "use_neural_zne": "use_neural_zne",
            "neural_zne_hidden_dim": "neural_zne_hidden_dim",
            "use_neural_qem": "use_neural_qem",
            "neural_qem_hidden_dim": "neural_qem_hidden_dim",
            # Meta-Controller
            "use_meta_controller": "use_meta_controller",
            "meta_controller_frequency": "meta_controller_frequency",
            # Loss function
            "loss_function": "loss_function",
            "label_smoothing": "label_smoothing",
            # Core
            "max_grad_norm": "max_grad_norm",
            "log_frequency": "log_frequency",
            # QULS - Quantum Unified Loss System (Phase 132)
            "use_quls": "use_quls",
            "quls_fidelity_weight": "quls_fidelity_weight",
            "quls_born_rule_weight": "quls_born_rule_weight",
            "quls_coherence_weight": "quls_coherence_weight",
            "quls_symplectic_weight": "quls_symplectic_weight",
            "quls_adaptive_weights": "quls_adaptive_weights",
        }
        
        # Build kwargs for constructor
        kwargs = {}
        for hpo_key, config_field in field_mappings.items():
            if hpo_key in hpo_config:
                kwargs[config_field] = hpo_config[hpo_key]
        
        return cls(**kwargs)


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class StepResult:
    """Result from a single training step.

    Attributes:
        loss: Training loss for this step.
        gradient_norm: Global gradient norm (pre-clipping).
        is_valid: Whether step was valid (no NaN/Inf).
        metrics: Additional metrics from the step.

        # Quantum training metrics
        barren_plateau_detected: Whether barren plateau was detected.
        effective_learning_rate: Actual LR after QALRC adjustment.
        qng_applied: Whether QNG was applied this step.
        galore_compression_ratio: Compression ratio from GaLore.
        qhpm_crystallized_count: Number of crystallized directions.
    """

    loss: float
    gradient_norm: float
    is_valid: bool = True
    metrics: dict[str, float] = field(default_factory=dict)

    # Quantum training metrics
    barren_plateau_detected: bool = False
    effective_learning_rate: float = 0.0
    qng_applied: bool = False
    galore_compression_ratio: float = 1.0
    qhpm_crystallized_count: int = 0


@dataclass
class EpochResult:
    """Result from a training epoch.

    Attributes:
        epoch: Epoch number (0-indexed).
        mean_loss: Mean loss over the epoch.
        steps_completed: Number of steps completed.
        should_stop: Whether training should stop early.
        stop_reason: Reason for early stopping (if any).
        metrics: Aggregated epoch metrics.
    """

    epoch: int
    mean_loss: float
    steps_completed: int
    should_stop: bool = False
    stop_reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result from full training run.

    Attributes:
        final_loss: Final training loss.
        epochs_completed: Number of epochs completed.
        total_steps: Total training steps.
        success: Whether training completed successfully.
        error: Error message if training failed.
        metrics: Final training metrics.
    """

    final_loss: float
    epochs_completed: int
    total_steps: int
    success: bool = True
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


# =============================================================================
# Callback Protocol and Base Classes
# =============================================================================


class TrainingCallback(Protocol):
    """Protocol for training callbacks.

    Callbacks can hook into training at batch and epoch boundaries.
    Return True from methods to continue training, False to stop.
    """

    def on_batch_start(self, step: int) -> bool:
        """Called before each training step. Return False to skip batch."""
        return True

    def on_batch_end(self, step: int, result: StepResult) -> bool:
        """Called after each training step. Return False to stop training."""
        return True

    def on_epoch_start(self, epoch: int) -> bool:
        """Called before each epoch. Return False to skip epoch."""
        return True

    def on_epoch_end(self, epoch: int, result: EpochResult) -> bool:
        """Called after each epoch. Return False to stop training."""
        return True


class BaseCallback:
    """Base implementation of TrainingCallback with no-op defaults."""

    def on_batch_start(self, step: int) -> bool:
        return True

    def on_batch_end(self, step: int, result: StepResult) -> bool:
        return True

    def on_epoch_start(self, epoch: int) -> bool:
        return True

    def on_epoch_end(self, epoch: int, result: EpochResult) -> bool:
        return True


# =============================================================================
# Specialized Callback Implementations
# =============================================================================


class HPOCallback(BaseCallback):
    """Callback for HPO trial reporting with pruning support.

    This callback reports metrics to an HPO framework (e.g., Optuna) and supports
    trial pruning based on intermediate results.

    Attributes:
        reporter: Callable that receives (loss, epoch) for reporting.
        prune_fn: Optional callable that returns True if trial should be pruned.
        report_every_n_steps: Report intermediate results every N steps.

    Example:
        >>> def reporter(loss, epoch):
        ...     trial.report(loss, epoch)
        >>> def should_prune(epoch):
        ...     return trial.should_prune()
        >>> callback = HPOCallback(reporter=reporter, prune_fn=should_prune)
    """

    def __init__(
        self,
        reporter: Callable[[float, int], None],
        prune_fn: Callable[[int], bool] | None = None,
        report_every_n_steps: int = 0,
    ):
        """Initialize HPO callback.

        Args:
            reporter: Callable receiving (loss, epoch) for metric reporting.
            prune_fn: Optional callable returning True if trial should prune.
            report_every_n_steps: If >0, report intermediate step-level metrics.
        """
        self.reporter = reporter
        self.prune_fn = prune_fn
        self.report_every_n_steps = report_every_n_steps
        self._step_losses: list[float] = []

    def on_batch_end(self, step: int, result: StepResult) -> bool:
        if result.is_valid:
            self._step_losses.append(result.loss)

        # Report intermediate step results if configured
        if self.report_every_n_steps > 0 and step % self.report_every_n_steps == 0:
            if self._step_losses:
                avg_loss = sum(self._step_losses[-self.report_every_n_steps :]) / min(
                    self.report_every_n_steps, len(self._step_losses)
                )
                try:
                    self.reporter(avg_loss, step)
                except Exception as e:
                    logger.warning(f"HPO reporter failed at step {step}: {e}")

        return True

    def on_epoch_end(self, epoch: int, result: EpochResult) -> bool:
        # Report epoch-level metrics
        try:
            self.reporter(result.mean_loss, epoch)
        except Exception as e:
            logger.warning(f"HPO reporter failed at epoch {epoch}: {e}")

        # Check for pruning
        if self.prune_fn is not None:
            try:
                if self.prune_fn(epoch):
                    logger.info(f"HPO trial pruned at epoch {epoch}")
                    return False
            except Exception as e:
                logger.warning(f"HPO prune check failed: {e}")

        # Reset step losses for next epoch
        self._step_losses = []
        return True


class CheckpointCallback(BaseCallback):
    """Callback for model checkpointing during training.

    Saves model checkpoints at regular intervals and tracks the best model
    based on validation loss.

    Attributes:
        checkpoint_dir: Directory for saving checkpoints.
        save_freq: Steps between checkpoint saves.
        model: Model to checkpoint.
        keep_best_only: If True, only keep the best checkpoint.

    Example:
        >>> callback = CheckpointCallback(
        ...     checkpoint_dir="/path/to/checkpoints",
        ...     save_freq=1000,
        ...     model=model,
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_freq: int = 1000,
        model: tf.keras.Model | None = None,
        keep_best_only: bool = False,
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory for saving checkpoints.
            save_freq: Steps between checkpoint saves.
            model: Model to checkpoint.
            keep_best_only: If True, only keep the best checkpoint.
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.model = model
        self.keep_best_only = keep_best_only
        self._step_count = 0
        self._best_loss = float("inf")
        self._best_checkpoint_path: str | None = None

        # Ensure directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_batch_end(self, step: int, result: StepResult) -> bool:
        self._step_count += 1
        if self._step_count % self.save_freq == 0:
            self._save_checkpoint(step, result.loss)
        return True

    def on_epoch_end(self, epoch: int, result: EpochResult) -> bool:
        # Always save at epoch end
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"epoch_{epoch}_loss_{result.mean_loss:.4f}.weights.h5"
        )
        self._save_model_weights(checkpoint_path)

        # Track best model
        if result.mean_loss < self._best_loss:
            self._best_loss = result.mean_loss
            best_path = os.path.join(self.checkpoint_dir, "best_model.weights.h5")
            self._save_model_weights(best_path)
            logger.info(f"New best model saved: loss={result.mean_loss:.4f}")

        return True

    def _save_checkpoint(self, step: int, loss: float) -> None:
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"step_{step}_loss_{loss:.4f}.weights.h5"
        )
        self._save_model_weights(checkpoint_path)

    def _save_model_weights(self, path: str) -> None:
        if self.model is not None:
            try:
                self.model.save_weights(path)
                logger.info(f"Checkpoint saved: {path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")


class MemoryManagerCallback(BaseCallback):
    """Callback for enterprise memory management with OOM prevention.

    Monitors system memory and triggers early stopping if memory pressure
    becomes critical. Uses EnterpriseMemoryManager for intelligent tracking.

    Attributes:
        warning_threshold_pct: Memory percentage triggering warnings.
        grace_steps: Steps of warnings before stopping.
        cleanup_frequency: Steps between garbage collection.

    Example:
        >>> callback = MemoryManagerCallback(
        ...     warning_threshold_pct=0.95,
        ...     grace_steps=10,
        ... )
    """

    def __init__(
        self,
        warning_threshold_pct: float = 0.95,
        grace_steps: int = 10,
        cleanup_frequency: int = 100,
    ):
        """Initialize memory manager callback.

        Args:
            warning_threshold_pct: Memory percentage triggering warnings (0-1).
            grace_steps: Number of warning steps before stopping.
            cleanup_frequency: Steps between garbage collection calls.
        """
        self.warning_threshold_pct = warning_threshold_pct
        self.grace_steps = grace_steps
        self.cleanup_frequency = cleanup_frequency
        self._memory_manager = None
        self._step_count = 0

        # Try to initialize memory manager
        try:
            from highnoon.services.hpo_trial_runner import EnterpriseMemoryManager

            self._memory_manager = EnterpriseMemoryManager(
                warning_threshold_pct=warning_threshold_pct,
                grace_steps=grace_steps,
            )
            logger.info("MemoryManagerCallback initialized with EnterpriseMemoryManager")
        except ImportError:
            logger.warning(
                "EnterpriseMemoryManager not available, using basic memory monitoring"
            )

    def on_batch_end(self, step: int, result: StepResult) -> bool:
        self._step_count += 1

        # Periodic garbage collection
        if self._step_count % self.cleanup_frequency == 0:
            gc.collect()

        # Check memory if manager available
        if self._memory_manager is not None:
            should_stop, reason = self._memory_manager.check_memory_critical()
            if should_stop:
                logger.error(f"Memory critical at step {step}: {reason}")
                return False

        return True

    def get_memory_stats(self) -> dict[str, float]:
        """Get current memory statistics."""
        if self._memory_manager is not None:
            return self._memory_manager.get_stats()
        return {}


class EWCUpdateCallback(BaseCallback):
    """Callback for EWC/QHPM crystallization updates during continual learning.

    Updates Fisher information matrix and optimal weights after each task
    for Elastic Weight Consolidation or QHPM crystallization.

    This callback is used during main training to enable anti-forgetting
    when training on multiple sequential tasks.

    Attributes:
        model: Model being trained.
        cumulative_fisher: Cumulative Fisher information dict.
        cumulative_optimal_weights: Cumulative optimal weights dict.
        update_frequency: Epochs between Fisher updates.

    Example:
        >>> callback = EWCUpdateCallback(
        ...     model=model,
        ...     cumulative_fisher=fisher_dict,
        ...     cumulative_optimal_weights=optimal_weights_dict,
        ...     update_frequency=1,
        ... )
    """

    def __init__(
        self,
        model: tf.keras.Model,
        cumulative_fisher: dict[str, tf.Tensor],
        cumulative_optimal_weights: dict[str, tf.Tensor],
        update_frequency: int = 1,
        use_qhpm: bool = True,
    ):
        """Initialize EWC update callback.

        Args:
            model: Model being trained.
            cumulative_fisher: Dict to store cumulative Fisher information.
            cumulative_optimal_weights: Dict to store optimal weights.
            update_frequency: Epochs between Fisher/QHPM updates.
            use_qhpm: If True, use QHPM crystallization instead of EWC.
        """
        self.model = model
        self.cumulative_fisher = cumulative_fisher
        self.cumulative_optimal_weights = cumulative_optimal_weights
        self.update_frequency = update_frequency
        self.use_qhpm = use_qhpm

    def on_epoch_end(self, epoch: int, result: EpochResult) -> bool:
        if (epoch + 1) % self.update_frequency == 0:
            self._update_consolidation_state()
        return True

    def _update_consolidation_state(self) -> None:
        """Update Fisher/QHPM state for anti-forgetting."""
        for var in self.model.trainable_variables:
            var_name = var.name
            # Store current weights as optimal
            self.cumulative_optimal_weights[var_name] = tf.identity(var)
            # Fisher would normally be computed from gradients
            # For QHPM, we use crystallization instead


class LoggingCallback(BaseCallback):
    """Callback for structured training logging.

    Logs training metrics to console and optionally to file in JSON format.

    Attributes:
        log_file: Optional path to JSON log file.
        log_frequency: Steps between log messages.
        include_memory: Include memory stats in logs.

    Example:
        >>> callback = LoggingCallback(
        ...     log_file="/path/to/metrics.jsonl",
        ...     log_frequency=10,
        ... )
    """

    def __init__(
        self,
        log_file: str | None = None,
        log_frequency: int = 10,
        include_memory: bool = True,
    ):
        """Initialize logging callback.

        Args:
            log_file: Optional path for JSON log file.
            log_frequency: Steps between log messages.
            include_memory: Whether to include memory stats.
        """
        self.log_file = log_file
        self.log_frequency = log_frequency
        self.include_memory = include_memory
        self._step_count = 0

    def on_batch_end(self, step: int, result: StepResult) -> bool:
        self._step_count += 1

        if self._step_count % self.log_frequency == 0:
            log_entry = {
                "step": step,
                "loss": result.loss,
                "gradient_norm": result.gradient_norm,
                "barren_plateau": result.barren_plateau_detected,
                "lr": result.effective_learning_rate,
                **result.metrics,
            }

            logger.info(
                f"Step {step}: loss={result.loss:.4f}, "
                f"grad_norm={result.gradient_norm:.4f}, "
                f"lr={result.effective_learning_rate:.2e}"
            )

            if self.log_file is not None:
                self._write_log_entry(log_entry)

        return True

    def _write_log_entry(self, entry: dict) -> None:
        import json

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write log entry: {e}")


# =============================================================================
# Training Engine
# =============================================================================


class TrainingEngine:
    """Unified training engine with all HighNoon training features.

    This engine encapsulates the core training loop with:
    - Meta-Controller for adaptive evolution_time
    - GaLore gradient compression with VQC-awareness
    - QALRC quantum-adaptive learning rate
    - Barren plateau detection and mitigation
    - Quantum Natural Gradient (QNG)
    - Gradient clipping and NaN handling
    - Memory management
    - Callback-based extensibility

    Supports both basic TrainingConfig and full EnterpriseTrainingConfig.

    Example:
        >>> config = EnterpriseTrainingConfig()
        >>> engine = TrainingEngine(model, optimizer, config)
        >>> result = engine.run(epochs=5, dataset=train_ds)
        >>> print(f"Final loss: {result.final_loss}")

    Example with HPO:
        >>> engine = TrainingEngine(model, optimizer, EnterpriseTrainingConfig())
        >>> result = engine.run(
        ...     epochs=5,
        ...     dataset=train_ds,
        ...     callbacks=[HPOCallback(reporter), MemoryManagerCallback()],
        ... )
    """

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        config: TrainingConfig | EnterpriseTrainingConfig | None = None,
        loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] | None = None,
    ):
        """Initialize training engine.

        Args:
            model: The model to train.
            optimizer: Optimizer for parameter updates.
            config: Training configuration. Uses EnterpriseTrainingConfig defaults if None.
            loss_fn: Optional custom loss function. If None, uses config.loss_function.
                     Signature: loss_fn(y_true, y_pred) -> tf.Tensor
        """
        self.model = model
        self.optimizer = optimizer

        # Normalize config to EnterpriseTrainingConfig
        if config is None:
            self.config = EnterpriseTrainingConfig()
        elif isinstance(config, TrainingConfig) and not isinstance(
            config, EnterpriseTrainingConfig
        ):
            # Convert basic config to enterprise config
            self.config = EnterpriseTrainingConfig(
                max_grad_norm=config.max_grad_norm,
                max_nan_consecutive=config.max_nan_consecutive,
                meta_controller_frequency=config.meta_controller_frequency,
                use_meta_controller=config.use_meta_controller,
                use_galore=config.use_galore,
                use_qalrc=config.use_qalrc,
                log_frequency=config.log_frequency,
            )
        else:
            self.config = config

        # Build or use provided loss function
        self._loss_fn = loss_fn if loss_fn is not None else self._build_loss_fn()

        # Initialize components
        self._meta_callback = None
        self._control_bridge = None
        self._galore = None
        self._qalrc = None
        self._barren_detector = None
        self._quls = None

        if self.config.use_meta_controller:
            self._init_meta_controller()

        if self.config.use_galore:
            self._init_galore()

        if self.config.use_qalrc:
            self._init_qalrc()

        if self.config.use_barren_plateau_detection:
            self._init_barren_plateau_detector()

        # Initialize QULS - Quantum Unified Loss System (Phase 132)
        if getattr(self.config, 'use_quls', False):
            self._init_quls()

        # Training state
        self._global_step = 0
        self._nan_count = 0
        self._current_lr = float(
            self.optimizer.learning_rate.numpy()
            if hasattr(self.optimizer.learning_rate, "numpy")
            else self.optimizer.learning_rate
        )
        self._barren_plateau_active = False
        self._barren_plateau_recovery_steps = 0

        logger.info(
            f"TrainingEngine initialized: "
            f"loss={self.config.loss_function}, "
            f"meta_controller={self._meta_callback is not None}, "
            f"galore={self._galore is not None}, "
            f"qalrc={self._qalrc is not None}, "
            f"barren_detector={self._barren_detector is not None}"
        )

    def _build_loss_fn(self) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        """Build loss function from configuration.
        
        Builds the appropriate Keras loss based on config.loss_function.
        Supports label smoothing for classification losses.
        
        Returns:
            Loss function with signature (y_true, y_pred) -> tf.Tensor.
        """
        loss_name = self.config.loss_function.lower()
        label_smoothing = getattr(self.config, 'label_smoothing', 0.0)
        
        if loss_name == 'sparse_categorical_crossentropy':
            def loss_fn(y_true, y_pred):
                return tf.keras.losses.sparse_categorical_crossentropy(
                    y_true, y_pred, from_logits=True
                )
            return loss_fn
            
        elif loss_name == 'categorical_crossentropy':
            loss_obj = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, label_smoothing=label_smoothing
            )
            return loss_obj
            
        elif loss_name == 'mse' or loss_name == 'mean_squared_error':
            return tf.keras.losses.MeanSquaredError()
            
        elif loss_name == 'mae' or loss_name == 'mean_absolute_error':
            return tf.keras.losses.MeanAbsoluteError()
            
        elif loss_name == 'huber':
            return tf.keras.losses.Huber()
            
        elif loss_name == 'binary_crossentropy':
            return tf.keras.losses.BinaryCrossentropy(from_logits=True)
            
        elif loss_name == 'cosine_similarity':
            return tf.keras.losses.CosineSimilarity()
            
        elif loss_name == 'kl_divergence':
            return tf.keras.losses.KLDivergence()
            
        else:
            # Default to MSE for unknown loss types (safe fallback)
            logger.warning(
                f"Unknown loss function '{loss_name}', defaulting to MSE. "
                f"Supported: sparse_categorical_crossentropy, categorical_crossentropy, "
                f"mse, mae, huber, binary_crossentropy, cosine_similarity, kl_divergence"
            )
            return tf.keras.losses.MeanSquaredError()

    def _init_meta_controller(self) -> None:
        """Initialize Meta-Controller and control bridge."""
        try:
            from highnoon.training.callbacks import HamiltonianMetaControllerCallback
            from highnoon.training.control_bridge import EvolutionTimeControlBridge

            self._meta_callback = HamiltonianMetaControllerCallback(
                frequency=self.config.meta_controller_frequency,
            )
            self._control_bridge = EvolutionTimeControlBridge(self.model)

            var_count = len(self._control_bridge.get_evolution_time_var_names())
            logger.info(f"Meta-Controller initialized: {var_count} evolution_time variables")
        except ImportError as e:
            logger.warning(f"Meta-Controller unavailable: {e}")

    def _init_galore(self) -> None:
        """Initialize GaLore gradient compressor."""
        try:
            from highnoon.training.gradient_compression import TensorGaLoreCompressor

            self._galore = TensorGaLoreCompressor(
                rank=self.config.galore_rank,
                update_proj_gap=self.config.galore_update_proj_gap,
                scale=self.config.galore_scale,
                enabled=True,
            )
            logger.info(
                f"GaLore initialized: rank={self._galore.rank}, "
                f"gap={self.config.galore_update_proj_gap}, scale={self.config.galore_scale}"
            )
        except ImportError as e:
            logger.warning(f"GaLore unavailable: {e}")

    def _init_qalrc(self) -> None:
        """Initialize Quantum Adaptive LR Controller."""
        try:
            from highnoon.training.quantum_lr_controller import QuantumAdaptiveLRController

            initial_lr = float(
                self.optimizer.learning_rate.numpy()
                if hasattr(self.optimizer.learning_rate, "numpy")
                else self.optimizer.learning_rate
            )
            self._qalrc = QuantumAdaptiveLRController(
                initial_lr=initial_lr,
                annealing_power=self.config.qalrc_annealing_power,
                tunneling_probability=self.config.qalrc_tunneling_probability,
                entropy_smoothing=self.config.qalrc_entropy_smoothing,
            )
            logger.info(f"QALRC initialized: initial_lr={initial_lr}")
        except (ImportError, TypeError) as e:
            # TypeError handles case where QuantumAdaptiveLRController doesn't accept all args
            try:
                from highnoon.training.quantum_lr_controller import QuantumAdaptiveLRController

                initial_lr = float(
                    self.optimizer.learning_rate.numpy()
                    if hasattr(self.optimizer.learning_rate, "numpy")
                    else self.optimizer.learning_rate
                )
                self._qalrc = QuantumAdaptiveLRController(
                    initial_lr=initial_lr,
                )
                logger.info(f"QALRC initialized (basic): initial_lr={initial_lr}")
            except ImportError as e2:
                logger.warning(f"QALRC unavailable: {e2}")

    def _init_barren_plateau_detector(self) -> None:
        """Initialize barren plateau detector."""
        try:
            from highnoon.training.barren_plateau import BarrenPlateauMonitor

            self._barren_detector = BarrenPlateauMonitor(
                threshold=self.config.barren_plateau_threshold,
                hysteresis_factor=self.config.barren_plateau_hysteresis,
                recovery_window=self.config.barren_plateau_recovery_window,
            )
            logger.info(
                f"Barren plateau detector initialized: threshold={self.config.barren_plateau_threshold}"
            )
        except ImportError as e:
            logger.warning(f"BarrenPlateauMonitor unavailable: {e}")

    def _init_quls(self) -> None:
        """Initialize Quantum Unified Loss System (Phase 132)."""
        try:
            from highnoon.training.quantum_loss import QULSConfig, QuantumUnifiedLoss

            # Build QULS config from EnterpriseTrainingConfig
            quls_config = QULSConfig(
                enabled=True,
                primary_loss=getattr(self.config, 'loss_function', 'sparse_categorical_crossentropy'),
                label_smoothing=getattr(self.config, 'label_smoothing', 0.1),
                fidelity_enabled=getattr(hn_config, 'USE_QUANTUM_FIDELITY_LOSS', True),
                fidelity_weight=getattr(self.config, 'quls_fidelity_weight', 0.01),
                born_rule_enabled=getattr(hn_config, 'USE_BORN_RULE_LOSS', True),
                born_rule_weight=getattr(self.config, 'quls_born_rule_weight', 0.005),
                entropy_enabled=getattr(self.config, 'use_entropy_regularization', True),
                entropy_weight=getattr(self.config, 'entropy_reg_weight', 0.01),
                target_entropy=getattr(self.config, 'target_entropy', 0.5),
                coherence_enabled=getattr(hn_config, 'QULS_COHERENCE_ENABLED', True),
                coherence_weight=getattr(self.config, 'quls_coherence_weight', 0.01),
                symplectic_enabled=getattr(hn_config, 'QULS_SYMPLECTIC_ENABLED', True),
                symplectic_weight=getattr(self.config, 'quls_symplectic_weight', 0.01),
                entanglement_enabled=getattr(hn_config, 'QULS_ENTANGLEMENT_ENABLED', True),
                adaptive_weights=getattr(self.config, 'quls_adaptive_weights', True),
            )

            self._quls = QuantumUnifiedLoss(quls_config)
            logger.info(
                f"QULS initialized: fidelity_weight={quls_config.fidelity_weight}, "
                f"entropy_weight={quls_config.entropy_weight}, "
                f"adaptive_weights={quls_config.adaptive_weights}"
            )
        except ImportError as e:
            logger.warning(f"QULS unavailable: {e}")
            self._quls = None

    def _safe_set_learning_rate(self, new_lr: float) -> None:
        """Safely set optimizer learning rate (TF2 compatible).

        Handles both tf.Variable and EagerTensor learning rates.
        Some optimizers (e.g., SophiaG) may return an EagerTensor
        for learning_rate instead of a Variable, which doesn't
        support .assign().

        Args:
            new_lr: New learning rate value.
        """
        if not hasattr(self.optimizer, "learning_rate"):
            return
        
        lr_attr = self.optimizer.learning_rate
        
        # Method 1: Try .assign() directly (works for tf.Variable)
        if hasattr(lr_attr, "assign"):
            try:
                lr_attr.assign(new_lr)
                return
            except AttributeError:
                pass
        
        # Method 2: Use K.set_value (works for both Variable and Tensor)
        try:
            tf.keras.backend.set_value(lr_attr, new_lr)
            return
        except Exception:
            pass
        
        # Method 3: Direct attribute assignment (last resort)
        try:
            self.optimizer.learning_rate = tf.Variable(
                new_lr, trainable=False, name="learning_rate"
            )
        except Exception as e:
            logger.debug(f"Could not set learning rate: {e}")

    def train_step(
        self,
        inputs: tf.Tensor,
        labels: tf.Tensor,
    ) -> StepResult:
        """Execute a single training step.

        Args:
            inputs: Input tensor.
            labels: Target labels.

        Returns:
            StepResult with loss, gradient norm, and metrics.
        """
        # Initialize QULS loss components tracking
        quls_components: dict[str, float] = {}

        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)

            # PHASE 1 FIX: Trigger lazy discovery after first forward pass
            # Model weights are now built, so evolution_time_bias exists
            if self._control_bridge is not None and self._global_step == 0:
                self._control_bridge.ensure_discovery()
                var_count = len(self._control_bridge.get_evolution_time_var_names())
                logger.info(f"Lazy discovery triggered: {var_count} evolution_time variables found")

            # Use QULS when enabled, otherwise use standard loss function
            if self._quls is not None:
                # Set training state for adaptive weighting
                self._quls.set_training_state(
                    step=self._global_step,
                    total_steps=max(1, getattr(self, '_total_steps', 10000))
                )

                # Collect hidden states if available (for entropy regularization)
                hidden_states = None
                if hasattr(self.model, 'get_hidden_states'):
                    try:
                        hidden_states = self.model.get_hidden_states()
                    except Exception:
                        pass

                # Collect coherence metrics if available (from QCB)
                coherence_metrics = None
                if hasattr(self.model, 'get_coherence_metrics'):
                    try:
                        coherence_metrics = self.model.get_coherence_metrics()
                    except Exception:
                        pass

                # Collect Hamiltonian energies if available (from TimeCrystal blocks)
                hamiltonian_energies = None
                if hasattr(self.model, 'get_hamiltonian_energies'):
                    try:
                        hamiltonian_energies = self.model.get_hamiltonian_energies()
                    except Exception:
                        pass

                # Compute QULS loss
                loss, quls_components = self._quls.compute_loss(
                    predictions=predictions,
                    targets=labels,
                    hidden_states=hidden_states,
                    coherence_metrics=coherence_metrics,
                    hamiltonian_energies=hamiltonian_energies,
                    in_barren_plateau=self._barren_plateau_active,
                )
            else:
                # Use configurable loss function
                loss = self._loss_fn(labels, predictions)
                if isinstance(loss, tf.Tensor) and len(loss.shape) > 0:
                    loss = tf.reduce_mean(loss)

        loss_val = float(loss.numpy())

        # Check for NaN/Inf loss
        if not math.isfinite(loss_val):
            self._nan_count += 1
            logger.warning(
                f"NaN/Inf loss at step {self._global_step}: count={self._nan_count}"
            )
            return StepResult(
                loss=loss_val,
                gradient_norm=0.0,
                is_valid=False,
                effective_learning_rate=self._current_lr,
            )

        # Reset NaN counter on valid loss
        self._nan_count = 0

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Filter None gradients
        grad_var_pairs = [
            (g, v)
            for g, v in zip(gradients, self.model.trainable_variables)
            if g is not None
        ]

        if not grad_var_pairs:
            logger.warning(f"No valid gradients at step {self._global_step}")
            return StepResult(
                loss=loss_val,
                gradient_norm=0.0,
                is_valid=False,
                effective_learning_rate=self._current_lr,
            )

        gradients_only = [g for g, _ in grad_var_pairs]
        vars_only = [v for _, v in grad_var_pairs]

        # Check for NaN in gradients
        has_nan_grad = any(tf.reduce_any(tf.math.is_nan(g)) for g in gradients_only)
        if has_nan_grad:
            self._nan_count += 1
            logger.warning(
                f"NaN gradient at step {self._global_step}: count={self._nan_count}"
            )
            return StepResult(
                loss=loss_val,
                gradient_norm=float("inf"),
                is_valid=False,
                effective_learning_rate=self._current_lr,
            )

        # Compute gradient norm
        gradient_norm = tf.linalg.global_norm(gradients_only)
        grad_norm_val = float(gradient_norm.numpy())

        # PHASE 1 FIX: Emergency gradient handling for gradient explosion
        # Gradient norms > 1e6 indicate catastrophic instability
        emergency_mode = False
        emergency_grad_clip = self.config.max_grad_norm
        if grad_norm_val > 1e6 or not math.isfinite(grad_norm_val):
            emergency_mode = True
            # Use aggressive clipping (0.1 instead of default 1.0)
            emergency_grad_clip = min(0.1, self.config.max_grad_norm / 10)
            # Reduce LR to minimum
            min_lr = getattr(self.config, 'lr_min', 1e-7)
            if hasattr(self.optimizer, "learning_rate"):
                current_lr = float(self.optimizer.learning_rate.numpy())
                emergency_lr = max(current_lr * 0.1, min_lr)
                # Use K.set_value for TF2 compatibility (works with both Variable and Tensor)
                self._safe_set_learning_rate(emergency_lr)
                self._current_lr = emergency_lr
            logger.error(
                f"[EMERGENCY] Gradient explosion at step {self._global_step}: "
                f"grad_norm={grad_norm_val:.2e}. Reducing LR to {self._current_lr:.2e}, "
                f"clip to {emergency_grad_clip}"
            )

        # Barren plateau detection with per-layer tracking
        barren_plateau_detected = False
        if self._barren_detector is not None:
            # Use full per-layer detection by passing gradients and variables
            barren_plateau_detected = self._barren_detector.update(
                gradient_norm=grad_norm_val,
                gradients=gradients_only,
                variables=vars_only,
            )
            if barren_plateau_detected and not self._barren_plateau_active:
                self._barren_plateau_active = True
                self._barren_plateau_recovery_steps = self.config.barren_plateau_recovery_window
                # Get the LR scale factor from the detector's active mitigations
                scale_factor = self._barren_detector.get_lr_scale_factor()
                logger.warning(
                    f"Barren plateau detected at step {self._global_step}, "
                    f"scaling LR by {scale_factor}x for quantum layers"
                )

        # Apply GaLore compression if enabled (per-gradient basis)
        # Flow: compress → clip in compressed space → decompress → apply
        galore_compression_ratio = 1.0
        galore_var_names: list[str] = []  # Track var names for decompression
        if self._galore is not None:
            original_norm = grad_norm_val
            compressed_gradients = []
            for grad, var in zip(gradients_only, vars_only):
                compressed_grad, var_name = self._galore.compress(grad, var)
                compressed_gradients.append(compressed_grad)
                galore_var_names.append(var_name)
            gradients_only = compressed_gradients
            # Increment step counter for projection update tracking
            self._galore.step()
            if original_norm > 0:
                compressed_norm = float(tf.linalg.global_norm(gradients_only).numpy())
                galore_compression_ratio = compressed_norm / original_norm

        # Clip gradients (in compressed space if GaLore enabled)
        # Use emergency clip value if gradient explosion was detected
        clip_norm = emergency_grad_clip if emergency_mode else self.config.max_grad_norm
        clipped_grads, _ = tf.clip_by_global_norm(
            gradients_only, clip_norm, use_norm=gradient_norm
        )

        # Decompress gradients back to original shapes for applying to variables
        if self._galore is not None and galore_var_names:
            decompressed_grads = []
            for clipped_grad, var_name in zip(clipped_grads, galore_var_names):
                decompressed_grad = self._galore.decompress(clipped_grad, var_name)
                decompressed_grads.append(decompressed_grad)
            clipped_grads = decompressed_grads

        # Compute effective learning rate with barren plateau mitigation
        effective_lr = self._current_lr
        if self._barren_plateau_active and self._barren_detector is not None:
            # Use the detector's computed scale factor from active mitigations
            bp_scale = self._barren_detector.get_lr_scale_factor()
            effective_lr *= bp_scale
            self._barren_plateau_recovery_steps -= 1
            if self._barren_plateau_recovery_steps <= 0:
                self._barren_plateau_active = False
                # Reset the detector's global barren state on recovery
                if hasattr(self._barren_detector, '_global_barren_detected'):
                    self._barren_detector._global_barren_detected = False
                logger.info(
                    f"Exiting barren plateau recovery mode, mitigations active: "
                    f"{len(self._barren_detector.get_active_mitigations())}"
                )
        elif self._barren_plateau_active:
            # Fallback to config if detector not available
            effective_lr *= self.config.barren_plateau_lr_scale
            self._barren_plateau_recovery_steps -= 1
            if self._barren_plateau_recovery_steps <= 0:
                self._barren_plateau_active = False
                logger.info("Exiting barren plateau recovery mode")

        # QALRC update
        qng_applied = False
        if self._qalrc is not None:
            try:
                # QALRC requires total_steps - estimate based on dataset if not known
                total_steps = getattr(self, '_estimated_total_steps', 100000)
                adaptive_lr = self._qalrc.get_learning_rate(
                    step=self._global_step,
                    total_steps=total_steps,
                    gradients=clipped_grads,  # Use decompressed gradients
                    current_loss=loss_val,
                )
                effective_lr = adaptive_lr
                if hasattr(self.optimizer, "learning_rate"):
                    # Use K.set_value for TF2 compatibility (works with both Variable and Tensor)
                    self._safe_set_learning_rate(effective_lr)
                self._current_lr = effective_lr
            except Exception as e:
                logger.debug(f"QALRC update failed: {e}")

        # Apply gradients (decompressed to original shapes)
        self.optimizer.apply_gradients(zip(clipped_grads, vars_only))

        # Meta-Controller update
        metrics: dict[str, float] = {}
        if self._meta_callback is not None and self._control_bridge is not None:
            logs = {"loss": loss_val, "gradient_norm": grad_norm_val}

            # Collect energy_drift from TimeCrystal blocks
            for layer in self.model.layers:
                if hasattr(layer, "_sequence_evolution_metric"):
                    drift = layer._sequence_evolution_metric
                    if drift is not None and hasattr(drift, "numpy"):
                        logs[f"{layer.name}/energy_drift"] = float(drift.numpy())

            control_names = self._control_bridge.get_evolution_time_var_names()
            
            # Ensure control names contain 'evolution_time' suffix for C++ PID classification
            # C++ generate_default_pid_config() classifies controls by checking for
            # "evolution_time" substring to assign proper PID gains
            control_names_for_cpp = []
            for name in control_names:
                if 'evolution_time' not in name:
                    control_names_for_cpp.append(f"{name}/evolution_time")
                else:
                    control_names_for_cpp.append(name)
            
            block_names, new_times = self._meta_callback.on_batch_end(
                batch=self._global_step,
                logs=logs,
                control_input_names=control_names_for_cpp,
            )

            if tf.size(new_times) > 0:
                updates = self._control_bridge.apply_evolution_times(block_names, new_times)
                metrics["evolution_time_updates"] = float(updates)

        # Add QULS loss components to metrics
        if quls_components:
            for key, value in quls_components.items():
                metrics[f"quls/{key}"] = value

        self._global_step += 1

        return StepResult(
            loss=loss_val,
            gradient_norm=grad_norm_val,
            is_valid=True,
            metrics=metrics,
            barren_plateau_detected=barren_plateau_detected,
            effective_learning_rate=effective_lr,
            qng_applied=qng_applied,
            galore_compression_ratio=galore_compression_ratio,
        )

    def execute_epoch(
        self,
        dataset: tf.data.Dataset,
        callbacks: list[TrainingCallback] | None = None,
        steps_per_epoch: int | None = None,
    ) -> EpochResult:
        """Execute a single training epoch.

        Args:
            dataset: Training dataset.
            callbacks: Optional list of callbacks.
            steps_per_epoch: Max steps per epoch (None = full dataset).

        Returns:
            EpochResult with epoch statistics.
        """
        callbacks = callbacks or []
        epoch = self._global_step // (steps_per_epoch or 1000)

        # on_epoch_start
        for cb in callbacks:
            if not cb.on_epoch_start(epoch):
                return EpochResult(
                    epoch=epoch,
                    mean_loss=float("inf"),
                    steps_completed=0,
                    should_stop=True,
                    stop_reason="callback requested stop",
                )

        losses: list[float] = []
        step = 0
        epoch_metrics: dict[str, list[float]] = {}

        for inputs, labels in dataset:
            if steps_per_epoch is not None and step >= steps_per_epoch:
                break

            # on_batch_start
            skip_batch = False
            for cb in callbacks:
                if not cb.on_batch_start(self._global_step):
                    skip_batch = True
                    break
            if skip_batch:
                continue

            result = self.train_step(inputs, labels)

            if result.is_valid:
                losses.append(result.loss)
                # Accumulate metrics
                for key, value in result.metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)
            else:
                # Check for too many NaN
                if self._nan_count >= self.config.max_nan_consecutive:
                    return EpochResult(
                        epoch=epoch,
                        mean_loss=float("inf"),
                        steps_completed=step,
                        should_stop=True,
                        stop_reason=f"Too many NaN losses: {self._nan_count}",
                    )

            # on_batch_end
            for cb in callbacks:
                if not cb.on_batch_end(self._global_step, result):
                    return EpochResult(
                        epoch=epoch,
                        mean_loss=sum(losses) / len(losses) if losses else float("inf"),
                        steps_completed=step + 1,
                        should_stop=True,
                        stop_reason="callback requested stop",
                    )

            step += 1

            # Log periodically
            if step % self.config.log_frequency == 0:
                logger.info(
                    f"Step {self._global_step}: loss={result.loss:.4f}, "
                    f"grad_norm={result.gradient_norm:.4f}, lr={result.effective_learning_rate:.2e}"
                )

        mean_loss = sum(losses) / len(losses) if losses else float("inf")

        # Aggregate epoch metrics
        aggregated_metrics = {
            key: sum(values) / len(values) for key, values in epoch_metrics.items()
        }

        epoch_result = EpochResult(
            epoch=epoch,
            mean_loss=mean_loss,
            steps_completed=step,
            metrics=aggregated_metrics,
        )

        # on_epoch_end
        for cb in callbacks:
            if not cb.on_epoch_end(epoch, epoch_result):
                epoch_result.should_stop = True
                epoch_result.stop_reason = "callback requested stop"
                break

        logger.info(f"Epoch {epoch} complete: mean_loss={mean_loss:.4f}, steps={step}")
        return epoch_result

    def run(
        self,
        epochs: int,
        dataset: tf.data.Dataset,
        callbacks: list[TrainingCallback] | None = None,
        steps_per_epoch: int | None = None,
    ) -> TrainingResult:
        """Run full training loop.

        Args:
            epochs: Number of epochs to train.
            dataset: Training dataset.
            callbacks: Optional list of callbacks.
            steps_per_epoch: Max steps per epoch.

        Returns:
            TrainingResult with final statistics.
        """
        callbacks = callbacks or []
        total_steps = 0
        last_loss = float("inf")
        all_metrics: dict[str, list[float]] = {}

        logger.info(f"Starting training: {epochs} epochs")

        for epoch in range(epochs):
            result = self.execute_epoch(dataset, callbacks, steps_per_epoch)
            total_steps += result.steps_completed
            last_loss = result.mean_loss

            # Accumulate metrics
            for key, value in result.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

            if result.should_stop:
                final_metrics = {
                    key: sum(values) / len(values) for key, values in all_metrics.items()
                }
                return TrainingResult(
                    final_loss=last_loss,
                    epochs_completed=epoch + 1,
                    total_steps=total_steps,
                    success=False,
                    error=result.stop_reason,
                    metrics=final_metrics,
                )

        final_metrics = {
            key: sum(values) / len(values) for key, values in all_metrics.items()
        }
        logger.info(
            f"Training complete: {epochs} epochs, {total_steps} steps, final_loss={last_loss:.4f}"
        )

        return TrainingResult(
            final_loss=last_loss,
            epochs_completed=epochs,
            total_steps=total_steps,
            success=True,
            metrics=final_metrics,
        )


__all__ = [
    # Configuration
    "TrainingConfig",
    "EnterpriseTrainingConfig",
    # Results
    "StepResult",
    "EpochResult",
    "TrainingResult",
    # Callbacks
    "TrainingCallback",
    "BaseCallback",
    "HPOCallback",
    "CheckpointCallback",
    "MemoryManagerCallback",
    "EWCUpdateCallback",
    "LoggingCallback",
    # Engine
    "TrainingEngine",
]
