# highnoon/services/hpo_training_bridge.py
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

"""HPO-TrainingEngine Integration Bridge.

This module provides a unified bridge between HPO sweep configurations and
the TrainingEngine, ensuring all feature flags and hyperparameters are
properly transferred between the HPO tuner and training workflows.

Usage:
    # After HPO sweep completes
    from highnoon.services.hpo_training_bridge import HPOTrainingConfig

    config = HPOTrainingConfig.from_hpo_sweep(sweep_result)
    config.save("sweep_config.json")

    # In training page/workflow
    config = HPOTrainingConfig.load("sweep_config.json")
    engine_config = config.to_enterprise_training_config()
    engine = TrainingEngine(model, optimizer, engine_config)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from highnoon.training.training_engine import EnterpriseTrainingConfig


@dataclass
class HPOTrainingConfig:
    """Complete configuration from HPO sweep for training.

    This dataclass bridges the gap between the WebUI HPO Smart Tuner
    (StartHPORequest) and the TrainingEngine (EnterpriseTrainingConfig).
    It stores all tuned hyperparameters and feature flags in a serializable
    format that can be saved after HPO completion and loaded for training.

    Attributes:
        sweep_id: Unique identifier of the HPO sweep.
        vocab_size: Tokenizer vocabulary size.
        context_window: Maximum context length.
        param_budget: Parameter count budget.
        embedding_dim: Model embedding dimension.
        num_reasoning_blocks: Number of reasoning blocks.
        num_moe_experts: Number of MoE experts.
        learning_rate: Tuned learning rate.
        batch_size: Tuned batch size.
        optimizer: Optimizer name (sophiag, adamw, lion, etc.).
        weight_decay: Weight decay for regularization.
        loss_function: Loss function name.
        feature_flags: All quantum/training feature flags.

    Example:
        >>> config = HPOTrainingConfig.from_hpo_sweep(sweep_result)
        >>> config.save("config.json")
        >>>
        >>> # Later, for training
        >>> config = HPOTrainingConfig.load("config.json")
        >>> engine_config = config.to_enterprise_training_config()
    """

    # Sweep metadata
    sweep_id: str = ""

    # Model architecture (from HPO model_config)
    vocab_size: int = 32000
    context_window: int = 4096
    param_budget: int = 1_000_000_000
    embedding_dim: int = 512
    num_reasoning_blocks: int = 8
    num_moe_experts: int = 8
    position_embedding: str = "rope"

    # Mamba2 SSM parameters
    mamba_state_dim: int = 64
    mamba_conv_dim: int = 4
    mamba_expand: int = 2

    # WLAM parameters
    wlam_num_heads: int = 8
    wlam_kernel_size: int = 3
    wlam_num_landmarks: int = 32

    # MoE parameters
    moe_top_k: int = 2
    moe_capacity_factor: float = 1.25

    # FFN and TT
    ff_expansion: int = 4
    tt_rank_middle: int = 16

    # Quantum/superposition
    superposition_dim: int = 2
    hamiltonian_hidden_dim: int = 256

    # Regularization
    dropout_rate: float = 0.1
    weight_decay: float = 0.01

    # Training hyperparameters (tuned by HPO)
    learning_rate: float = 1e-4
    batch_size: int = 32
    optimizer: str = "sophiag"
    epochs: int = 10

    # Loss function configuration
    loss_function: str = "sparse_categorical_crossentropy"
    label_smoothing: float = 0.0

    # All feature flags (quantum phases, training loop optimizations)
    feature_flags: dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: str = ""
    best_loss: float | None = None
    best_composite_score: float | None = None

    def to_enterprise_training_config(self) -> EnterpriseTrainingConfig:
        """Convert to TrainingEngine's EnterpriseTrainingConfig.

        Maps all HPO-tuned parameters to the corresponding fields in
        EnterpriseTrainingConfig for use with TrainingEngine.

        Returns:
            EnterpriseTrainingConfig instance with all tuned parameters.
        """
        from highnoon.training.training_engine import EnterpriseTrainingConfig

        # Start with feature flags
        flags = self.feature_flags.copy()

        # Build kwargs for EnterpriseTrainingConfig
        kwargs = {
            # Core training
            "max_grad_norm": flags.get("max_grad_norm", 1.0),
            "max_nan_consecutive": flags.get("max_nan_consecutive", 5),
            "log_frequency": flags.get("log_frequency", 10),
            "loss_function": self.loss_function,
            "label_smoothing": self.label_smoothing,
            # Meta-Controller
            "use_meta_controller": flags.get("use_meta_controller", True),
            "meta_controller_frequency": flags.get("meta_controller_frequency", 10),
            # GaLore
            "use_galore": flags.get("use_tensor_galore", True),
            "galore_rank": flags.get("galore_rank", 32),
            "galore_update_proj_gap": flags.get("galore_update_proj_gap", 200),
            "galore_scale": flags.get("galore_scale", 0.25),
            "galore_vqc_aware": flags.get("galore_vqc_aware", True),
            # QALRC
            "use_qalrc": flags.get("use_quantum_lr_controller", True),
            "qalrc_annealing_power": flags.get("qalrc_annealing_power", 2.0),
            "qalrc_tunneling_probability": flags.get("qalrc_tunneling_probability", 0.05),
            "qalrc_entropy_smoothing": flags.get("qalrc_entropy_smoothing", 0.9),
            # Barren Plateau
            "use_barren_plateau_detection": flags.get("barren_plateau_monitor", True),
            "barren_plateau_threshold": flags.get("barren_plateau_threshold", 1e-6),
            "barren_plateau_lr_scale": flags.get("barren_plateau_recovery_lr_scale", 10.0),
            # QNG
            "use_qng": flags.get("use_quantum_natural_gradient", True),
            "qng_damping": flags.get("qng_damping", 1e-4),
            "qng_apply_to_quantum_only": flags.get("qng_apply_to_quantum_only", True),
            # Entropy Regularization
            "use_entropy_regularization": flags.get("use_entropy_regularization", True),
            "entropy_reg_weight": flags.get("entropy_reg_weight", 0.01),
            # QHPM Crystallization
            "use_qhpm_crystallization": flags.get("use_qhpm_crystallization", True),
            "crystallization_threshold": flags.get("qhpm_crystallization_threshold", 0.85),
            "max_crystallized_directions": flags.get("qhpm_max_directions", 256),
            # SympFlow
            "use_sympflow": flags.get("use_sympflow_optimizer", True),
            "sympflow_mass": flags.get("sympflow_mass", 1.0),
            "sympflow_friction": flags.get("sympflow_friction", 0.1),
            # Neural ZNE/QEM
            "use_neural_zne": flags.get("use_neural_zne", True),
            "use_neural_qem": flags.get("use_neural_qem", True),
        }

        return EnterpriseTrainingConfig(**kwargs)

    def to_model_build_config(self) -> dict[str, Any]:
        """Get configuration dictionary for model building.

        Returns config format expected by hpo_trial_runner.build_hsmn_model().

        Returns:
            Dictionary with all model architecture parameters.
        """
        config = {
            # Core architecture
            "vocab_size": self.vocab_size,
            "hidden_dim": self.embedding_dim,
            "num_reasoning_blocks": self.num_reasoning_blocks,
            "num_moe_experts": self.num_moe_experts,
            "position_embedding": self.position_embedding,
            # Mamba2
            "mamba_state_dim": self.mamba_state_dim,
            "mamba_conv_dim": self.mamba_conv_dim,
            "mamba_expand": self.mamba_expand,
            # WLAM
            "wlam_num_heads": self.wlam_num_heads,
            "wlam_kernel_size": self.wlam_kernel_size,
            "wlam_num_landmarks": self.wlam_num_landmarks,
            # MoE
            "moe_top_k": self.moe_top_k,
            "moe_capacity_factor": self.moe_capacity_factor,
            # FFN and TT
            "ff_expansion": self.ff_expansion,
            "tt_rank_middle": self.tt_rank_middle,
            # Quantum
            "superposition_dim": self.superposition_dim,
            "hamiltonian_hidden_dim": self.hamiltonian_hidden_dim,
            # Regularization
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            # Training
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
        }

        # Add all feature flags
        config.update(self.feature_flags)

        return config

    def to_optimizer_config(self) -> dict[str, Any]:
        """Get configuration for optimizer creation.

        Returns config format expected by hpo_trial_runner.create_optimizer().

        Returns:
            Dictionary with optimizer configuration.
        """
        return {
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
        }

    @classmethod
    def from_hpo_sweep(
        cls,
        sweep_result: dict[str, Any],
        model_config: dict[str, Any] | None = None,
    ) -> HPOTrainingConfig:
        """Create config from completed HPO sweep results.

        Args:
            sweep_result: Sweep result dictionary containing best_hyperparams.
            model_config: Optional model configuration override.

        Returns:
            HPOTrainingConfig with all tuned parameters.
        """
        best_hp = sweep_result.get("best_hyperparams", {})
        mc = model_config or sweep_result.get("model_config", {})
        config = sweep_result.get("config", {})

        # Extract feature flags from best_hyperparams and config
        feature_flags = {}

        # Training loop flags
        for key in [
            "use_tensor_galore",
            "galore_rank",
            "galore_scale",
            "use_quantum_natural_gradient",
            "qng_damping",
            "barren_plateau_monitor",
            "barren_plateau_threshold",
            "use_meta_controller",
            "meta_controller_frequency",
            "use_quantum_lr_controller",
            "use_entropy_regularization",
            "entropy_reg_weight",
            "use_qhpm_crystallization",
            "qhpm_crystallization_threshold",
            "use_sympflow_optimizer",
            "use_neural_zne",
            "use_neural_qem",
        ]:
            if key in best_hp:
                feature_flags[key] = best_hp[key]
            elif key in config:
                feature_flags[key] = config[key]

        # Quantum phase flags (from StartHPORequest)
        quantum_flags = [
            "use_quantum_embedding",
            "use_floquet_position",
            "use_quantum_feature_maps",
            "use_unitary_expert",
            "use_quantum_norm",
            "use_superposition_bpe",
            "use_grover_qsg",
            "use_quantum_lm_head",
            "use_unitary_residual",
            "use_quantum_state_bus",
            "use_quantum_measurement_dropout",
            "use_q_ssm_gating",
            "use_qasa_attention",
            "use_mpqr_reasoning",
            "use_td_moe",
            "use_vqem",
            "use_qcot_reasoning",
        ]
        for key in quantum_flags:
            if key in best_hp:
                feature_flags[key] = best_hp[key]
            elif key in config:
                feature_flags[key] = config[key]

        return cls(
            sweep_id=sweep_result.get("sweep_id", ""),
            vocab_size=mc.get("vocab_size", 32000),
            context_window=mc.get("context_window", 4096),
            param_budget=mc.get("param_budget", 1_000_000_000),
            embedding_dim=mc.get("embedding_dim", mc.get("hidden_dim", 512)),
            num_reasoning_blocks=mc.get("num_reasoning_blocks", 8),
            num_moe_experts=mc.get("num_moe_experts", 8),
            position_embedding=mc.get("position_embedding", "rope"),
            mamba_state_dim=mc.get("mamba_state_dim", 64),
            mamba_conv_dim=mc.get("mamba_conv_dim", 4),
            mamba_expand=mc.get("mamba_expand", 2),
            wlam_num_heads=mc.get("wlam_num_heads", 8),
            wlam_kernel_size=mc.get("wlam_kernel_size", 3),
            wlam_num_landmarks=mc.get("wlam_num_landmarks", 32),
            moe_top_k=mc.get("moe_top_k", 2),
            moe_capacity_factor=mc.get("moe_capacity_factor", 1.25),
            ff_expansion=mc.get("ff_expansion", 4),
            tt_rank_middle=mc.get("tt_rank_middle", 16),
            superposition_dim=mc.get("superposition_dim", 2),
            hamiltonian_hidden_dim=mc.get("hamiltonian_hidden_dim", 256),
            dropout_rate=best_hp.get("dropout_rate", mc.get("dropout_rate", 0.1)),
            weight_decay=best_hp.get("weight_decay", mc.get("weight_decay", 0.01)),
            learning_rate=best_hp.get("learning_rate", 1e-4),
            batch_size=best_hp.get("batch_size", 32),
            optimizer=best_hp.get("optimizer", config.get("optimizers", ["sophiag"])[0]),
            epochs=config.get("epochs", 10),
            loss_function=best_hp.get("loss_function", "sparse_categorical_crossentropy"),
            feature_flags=feature_flags,
            best_loss=sweep_result.get("best_loss"),
            best_composite_score=sweep_result.get("best_composite_score"),
        )

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save configuration.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"[HPO Bridge] Saved training config to {path}")

    @classmethod
    def load(cls, path: str | Path) -> HPOTrainingConfig:
        """Load configuration from JSON file.

        Args:
            path: Path to configuration file.

        Returns:
            HPOTrainingConfig instance.
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        logger.info(f"[HPO Bridge] Loaded training config from {path}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of config.
        """
        return asdict(self)


__all__ = ["HPOTrainingConfig"]
