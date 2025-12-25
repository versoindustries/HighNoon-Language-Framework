# highnoon/webui/app.py
# Copyright 2025 Verso Industries
#
# FastAPI application for the HighNoon Language Framework WebUI.
# Provides a web interface for curriculum building, training management,
# and model configuration.

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models for API
# ============================================================================


class AddHuggingFaceDataset(BaseModel):
    """Request body for adding a HuggingFace dataset."""

    dataset_id: str
    config_name: str | None = None
    splits: list[str] = ["train"]


class UseDatasetRequest(BaseModel):
    """Request body for using a dataset in curriculum."""

    dataset_id: str
    stage_name: str
    weight: float = 1.0


class CreateStageRequest(BaseModel):
    """Request body for creating a new curriculum stage."""

    name: str
    display_name: str
    module: str = "language_modeling"
    epochs: int = 1
    learning_rate: str = "1e-4"
    batch_size: int = 8


class UpdateStageRequest(BaseModel):
    """Request body for updating a curriculum stage."""

    display_name: str | None = None
    module: str | None = None
    epochs: int | None = None
    learning_rate: str | None = None
    batch_size: int | None = None


class AddDatasetToStageRequest(BaseModel):
    """Request body for adding a dataset to a stage."""

    dataset_id: str
    weight: float = 1.0


class CurriculumDatasetEntry(BaseModel):
    """A dataset entry within a curriculum stage."""

    dataset_id: str
    weight: float = 1.0


class SaveCurriculumStage(BaseModel):
    """A stage within a saved curriculum."""

    name: str
    display_name: str
    module: str = "language_modeling"
    datasets: list[CurriculumDatasetEntry] = []
    epochs: int = 1
    learning_rate: str = "1e-4"
    batch_size: int = 8
    weight: float = 1.0


class SaveCurriculumRequest(BaseModel):
    """Request body for saving a complete curriculum."""

    id: str
    name: str
    stages: list[SaveCurriculumStage]
    vocab_size: int = 32000
    context_window: int = 4096
    created_at: str | None = None
    updated_at: str | None = None


class CurriculumStage(BaseModel):
    """Configuration for a single curriculum stage."""

    name: str
    datasets: list[str]
    epochs: int = 1
    learning_rate: float = 1e-4
    batch_size: int = 8


class TrainingConfig(BaseModel):
    """Training configuration submitted from the UI."""

    model_name: str = "highnoon-base"
    stages: list[CurriculumStage] = []
    output_dir: str = "./outputs"
    resume_from_checkpoint: bool = True


class TrainingStatus(BaseModel):
    """Status of a training run."""

    running: bool = False
    current_stage: int = 0
    current_epoch: int = 0
    global_step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0


class DatasetInfo(BaseModel):
    """Information about an available dataset."""

    name: str
    source: str
    num_examples: int
    description: str = ""


# ============================================================================
# Attribution Configuration Models (Pro/Enterprise Feature)
# ============================================================================


class AttributionConfig(BaseModel):
    """Custom attribution configuration for Pro/Enterprise editions.

    Pro and Enterprise users can customize the attribution that appears
    in trained models. Lite edition keeps default Verso Industries attribution.
    """

    framework_name: str = Field(
        default="HighNoon Language Framework",
        min_length=1,
        max_length=100,
        description="The framework name to display in attribution",
    )
    author: str = Field(default="", max_length=100, description="The author or company name")
    copyright_notice: str = Field(
        default="", max_length=200, description="The copyright notice string"
    )
    version: str = Field(default="1.0.0", max_length=20, description="The version string")
    support_url: str = Field(default="", max_length=200, description="The support URL")


class AttributionResponse(BaseModel):
    """Response model for attribution API with edition info."""

    attribution: AttributionConfig
    edition: str  # "Lite", "Pro", "Enterprise"
    edition_code: int  # 0, 1, 2
    is_customizable: bool  # True for Pro/Enterprise
    is_custom: bool  # True if currently using custom attribution


# ============================================================================
# Training State Machine
# ============================================================================


class TrainingMode(str, Enum):
    """Training mode selection."""

    QUICK_TRAIN = "quick_train"  # Uses pre-validated hyperparameters
    AUTO_TUNE = "auto_tune"  # Runs Stage 1 (COARSE) HPO sweep first
    FULL_SWEEP = "full_sweep"  # Runs all 3 HPO stages


class TrainingState(str, Enum):
    """Training job state machine states."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HPOStage(str, Enum):
    """HPO sweep stages for hierarchical optimization."""

    COARSE = "coarse"  # Architecture discovery
    REFINE = "refine"  # Training hyperparameters
    FINE = "fine"  # Fine-tuning parameters


class StartTrainingRequest(BaseModel):
    """Request body for starting a training job."""

    mode: TrainingMode = TrainingMode.QUICK_TRAIN
    curriculum_id: str | None = None
    sweep_id: str | None = None  # HPO sweep ID to load best hyperparameters from
    output_dir: str = "./outputs"
    learning_rate: str | None = None
    batch_size: int | None = None
    epochs: int = 10


class StartHPORequest(BaseModel):
    """Request body for starting an HPO sweep.

    All model configuration values are validated against Lite edition limits:
    - vocab_size: max 256000 (256K)
    - context_window: max 5000000 (5M)
    - embedding_dim: max 4096
    - num_reasoning_blocks: max 24
    - num_moe_experts: max 12

    Training uses epoch-based budgets with automatic early stopping:
    - HPO trials: 5 epochs per trial by default
    - Full training: 200 epochs budget with convergence-based early stop
    """

    # HPO sweep settings
    stage: HPOStage = HPOStage.COARSE
    max_trials: int = Field(default=10, ge=1, le=999)  # Allow up to 999 for Convergence mode
    epochs_per_trial: int = Field(default=5, ge=1, le=20)  # Replaces steps_per_trial
    search_strategy: str = "bayesian"  # random, bayesian, hyperband, successive_halving, pbt
    curriculum_id: str | None = None

    # Advanced HPO options
    use_optuna: bool = False  # Use Optuna sampler for advanced optimization
    hyperband_eta: int = Field(default=3, ge=2, le=4)  # Hyperband reduction factor
    pbt_population_size: int = Field(default=8, ge=4, le=32)  # PBT population

    # Learning rate configuration (HPO will optimize within this range)
    lr_min: str = "1e-5"
    lr_max: str = "1e-3"
    batch_sizes: list[int] = Field(default=[16, 32, 64])
    optimizers: list[str] = Field(default=["sophiag"])

    # Model configuration (expanded vocab support up to 256K)
    vocab_size: int = Field(default=32000, ge=1000, le=256000)
    context_window: int = Field(default=4096, ge=128, le=5000000)
    embedding_dim: int | None = Field(default=512)  # Make optional with default
    num_reasoning_blocks: int | None = Field(default=8)  # Make optional with default
    num_moe_experts: int | None = Field(default=8)  # Make optional with default
    position_embedding: str = Field(default="rope")

    # Parameter budget constraint - HPO will skip architectures exceeding this limit
    # Range: 100M (small efficient models) to 20B (Lite edition max)
    param_budget: int = Field(default=1_000_000_000, ge=100_000_000, le=20_000_000_000)

    # =========================================================================
    # HSMN-Specific Trainable Parameters
    # =========================================================================

    # Mamba2 SSM parameters
    mamba_state_dim: int = Field(default=64, ge=16, le=128, description="Mamba state dimension")
    mamba_conv_dim: int = Field(default=4, ge=2, le=8, description="Mamba convolution dimension")
    mamba_expand: int = Field(default=2, ge=1, le=4, description="Mamba expansion factor")

    # WLAM (Wavelet Linear Attention Module) parameters
    wlam_num_heads: int = Field(default=8, ge=1, le=16, description="WLAM attention heads")
    wlam_kernel_size: int = Field(default=3, ge=1, le=7, description="WLAM wavelet kernel size")
    wlam_num_landmarks: int = Field(default=32, ge=16, le=128, description="WLAM Nyström landmarks")

    # MoE (Mixture of Experts) parameters
    moe_top_k: int = Field(default=2, ge=1, le=4, description="Number of experts to route to")
    moe_capacity_factor: float = Field(
        default=1.25, ge=1.0, le=2.0, description="Expert capacity factor"
    )

    # Feed-forward network
    ff_expansion: int = Field(default=4, ge=2, le=8, description="FFN expansion factor")

    # TensorTrain decomposition
    tt_rank_middle: int = Field(default=16, ge=4, le=64, description="TT-decomposition middle rank")

    # Quantum/Superposition parameters
    superposition_dim: int = Field(
        default=2, ge=1, le=2, description="Superposition dimension (max 2 Lite)"
    )

    # TimeCrystal/Hamiltonian parameters
    hamiltonian_hidden_dim: int = Field(
        default=256, ge=64, le=1024, description="HNN hidden dimension"
    )

    # Regularization
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout rate")
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.1, description="Weight decay")

    # =========================================================================
    # Quantum Enhancement Parameters (Phases 26-36)
    # =========================================================================

    # Phase 26: Quantum Embedding
    use_quantum_embedding: bool = Field(
        default=True, description="Enable quantum Hadamard embedding layer"
    )
    quantum_embedding_num_qubits: int = Field(
        default=8, ge=4, le=16, description="Virtual qubits for embedding"
    )

    # Phase 27: Floquet Position Encoding
    use_floquet_position: bool = Field(
        default=True, description="Enable time-crystal SU(2) position encoding"
    )
    floquet_num_layers: int = Field(
        default=2, ge=1, le=4, description="SU(2) rotation layers per qubit"
    )

    # Phase 28: Quantum Feature Maps (already in WLAM via flash attention)
    use_quantum_feature_maps: bool = Field(
        default=True, description="Enable quantum feature maps in attention"
    )
    quantum_feature_rotation_depth: int = Field(
        default=2, ge=1, le=4, description="VQC rotation depth"
    )

    # Phase 29: Unitary Expert Networks
    use_unitary_expert: bool = Field(
        default=True, description="Enable Cayley-parameterized unitary experts"
    )
    neumann_cayley_terms: int = Field(
        default=6, ge=2, le=12, description="Neumann series truncation terms"
    )

    # Phase 30: Quantum Normalization
    use_quantum_norm: bool = Field(
        default=True, description="Enable Stiefel manifold normalization"
    )

    # Phase 31: Superposition BPE
    use_superposition_bpe: bool = Field(
        default=True, description="Enable multi-segmentation tokenization"
    )
    sbpe_max_superposition: int = Field(
        default=4, ge=2, le=8, description="Max parallel segmentations"
    )

    # Phase 32: Grover QSG
    use_grover_qsg: bool = Field(default=True, description="Enable Grover-guided generation")
    qsg_quality_threshold: float = Field(
        default=0.7, ge=0.5, le=0.95, description="Oracle quality threshold"
    )

    # Phase 33: Quantum LM Head
    use_quantum_lm_head: bool = Field(
        default=True, description="Enable Born-rule output distribution"
    )
    quantum_lm_head_layers: int = Field(default=2, ge=1, le=4, description="VQC circuit depth")

    # Phase 34: Unitary Residual Connections
    use_unitary_residual: bool = Field(default=True, description="Enable rotation-blend residuals")
    unitary_residual_init_angle: float = Field(
        default=0.7854, ge=0.0, le=1.571, description="Initial blend angle (π/4 default)"
    )

    # Phase 35: Quantum State Bus
    use_quantum_state_bus: bool = Field(
        default=True, description="Enable entanglement-mediated cross-block comm"
    )

    # =========================================================================
    # Quantum Training Loop Optimization (Phases T1-T6)
    # =========================================================================

    # Phase T1: Quantum Natural Gradient
    use_quantum_natural_gradient: bool = Field(
        default=True, description="Enable QFIM-preconditioned gradient"
    )
    qng_damping: float = Field(
        default=1e-4, ge=1e-6, le=1e-2, description="QFIM regularization damping"
    )

    # Phase T2: Tensor-GaLore Gradient Compression
    use_tensor_galore: bool = Field(
        default=True, description="Enable gradient low-rank projection (50-75% memory savings)"
    )
    galore_rank: int = Field(default=32, ge=8, le=128, description="Gradient projection rank")

    # Phase T4: Barren Plateau Mitigation
    barren_plateau_monitor: bool = Field(
        default=True, description="Enable barren plateau detection and recovery"
    )
    barren_plateau_threshold: float = Field(
        default=1e-6, ge=1e-8, le=1e-4, description="Gradient norm threshold for detection"
    )

    # =========================================================================
    # Quantum Constant-Memory Training (Phases 1-7)
    # =========================================================================

    # Phase 1: Neumann-Cayley Unitary Mamba Gates
    use_unitary_mamba_gates: bool = Field(
        default=True, description="Enable Neumann-Cayley for Mamba B/C projections"
    )
    neumann_series_terms: int = Field(
        default=4, ge=2, le=8, description="Neumann series truncation terms"
    )

    # Phase 2: SPRK TimeCrystal Integrator
    use_sprk_timecrystal: bool = Field(
        default=True, description="Use 6th-order Yoshida symplectic integrator"
    )
    sprk_order: int = Field(default=6, ge=4, le=6, description="Integrator order (4 or 6)")

    # Phase 5: QNG Geodesic MoE Routing
    use_qng_geodesic: bool = Field(default=True, description="Enable geodesic QNG for MoE router")
    qng_geodesic_order: int = Field(default=2, ge=1, le=2, description="Geodesic correction order")

    # Phase 7: Entanglement Preservation
    entanglement_regularization: float = Field(
        default=0.01, ge=0.0, le=0.1, description="Weight for entropy loss"
    )

    # =========================================================================
    # Phase 17 Hamiltonian Enhancements
    # =========================================================================

    # Phase 17.1: Hamiltonian Superposition
    hamiltonian_basis_size: int = Field(
        default=4, ge=2, le=8, description="Number of basis Hamiltonians"
    )
    hamiltonian_enable_superposition: bool = Field(
        default=True, description="Enable superposition mode"
    )

    # Phase 17.3: Magnus Integrator
    hamiltonian_integrator: str = Field(
        default="yoshida", description="Integrator: yoshida, magnus, euler"
    )

    # Phase 17.4: Neural Quantum State
    hamiltonian_enable_nqs: bool = Field(default=True, description="Enable NQS wavefunction mode")
    hamiltonian_nqs_hidden_dim: int = Field(
        default=32, ge=16, le=128, description="RBM hidden dimension"
    )

    # =========================================================================
    # Quantum Superposition Generation (QSG) Parameters
    # =========================================================================
    qsg_bond_dim: int = Field(
        default=32, ge=16, le=128, description="MPS bond dimension for context"
    )
    qsg_grover_iterations: int = Field(
        default=3, ge=1, le=10, description="Grover amplitude amplification iterations"
    )
    qsg_jacobi_iterations: int = Field(
        default=2, ge=1, le=5, description="Jacobi consistency refinement iterations"
    )
    qsg_hopfield_beta: float = Field(
        default=1.0, ge=0.1, le=5.0, description="Modern Hopfield inverse temperature"
    )

    # =========================================================================
    # Quantum Control Parameters (Phase 2.1)
    # =========================================================================
    use_rls_sysid: bool = Field(default=True, description="Enable fast RLS system identification")
    rls_forgetting_factor: float = Field(
        default=0.998, ge=0.95, le=0.999, description="RLS forgetting factor"
    )
    use_hybrid_pid: bool = Field(default=True, description="Enable Relay + Adam hybrid PID tuning")
    pid_learning_rate: float = Field(
        default=0.001, ge=1e-4, le=0.01, description="Adam learning rate for PID tuning"
    )
    use_tensor_network_kalman: bool = Field(
        default=False, description="Enable TT-compressed Kalman filter"
    )
    tnkf_max_rank: int = Field(default=8, ge=4, le=32, description="Maximum TT rank for TNKF")

    # =========================================================================
    # Anti-Forgetting Configuration (QHPM Crystallization)
    # =========================================================================
    # QHPM Crystallization replaces legacy EWC for preventing catastrophic forgetting.
    # Uses orthogonal gradient projection for mathematically superior protection.

    use_qhpm_crystallization: bool = Field(
        default=True, description="Enable QHPM gradient crystallization for anti-forgetting"
    )
    qhpm_crystallization_threshold: float = Field(
        default=0.85, ge=0.5, le=0.99, description="Confidence threshold for memory crystallization"
    )
    qhpm_max_directions: int = Field(
        default=256, ge=32, le=1024, description="Maximum crystallized parameter directions"
    )
    qhpm_crystallization_decay: float = Field(
        default=0.99, ge=0.9, le=1.0, description="Decay rate for crystallization strength"
    )

    # =========================================================================
    # Quantum Enhancement Parameters v5.0 (Phases 47-84)
    # =========================================================================

    # Pillar 1: Foundation (Critical)
    # Phase 47: Quantum Measurement Dropout
    use_quantum_measurement_dropout: bool = Field(
        default=True, description="Enable QMD for ensemble circuits"
    )
    qmd_drop_rate: float = Field(
        default=0.1, ge=0.0, le=0.3, description="QMD measurement probability"
    )
    qmd_softening_temp: float = Field(
        default=1.0, ge=0.1, le=5.0, description="Soft collapse temperature"
    )

    # Phase 69: Q-SSM Quantum State Space Gating
    use_q_ssm_gating: bool = Field(default=True, description="Enable VQC gating for Mamba")
    q_ssm_vqc_layers: int = Field(default=2, ge=1, le=4, description="VQC circuit layers")
    q_ssm_num_qubits: int = Field(default=4, ge=2, le=8, description="Virtual qubits for Q-SSM")

    # Phase 71: Intrinsic Plasticity Preservation
    use_intrinsic_plasticity: bool = Field(
        default=True, description="Enable Stiefel manifold plasticity"
    )
    plasticity_learning_rate: float = Field(
        default=0.01, ge=0.001, le=0.1, description="Plasticity update rate"
    )

    # Phase 76: Unified Quantum Coherence Bus
    use_quantum_coherence_bus: bool = Field(
        default=True, description="Enable QCB for block coordination"
    )
    qcb_num_nodes: int = Field(default=8, ge=4, le=24, description="Coherence mesh nodes")
    qcb_fidelity_threshold: float = Field(
        default=0.9, ge=0.8, le=0.99, description="Minimum entanglement fidelity"
    )

    # Pillar 2: Input/Output Enhancement
    # Phase 48: Hyperdimensional Quantum Embeddings
    use_hyperdimensional_embedding: bool = Field(
        default=True, description="Enable HQE holographic bundling"
    )
    hqe_ctqw_steps: int = Field(default=3, ge=1, le=10, description="CTQW spreading walk steps")

    # Phase 49: Holographic Hypertokens
    use_hypertokens: bool = Field(default=True, description="Enable HDRAM spread-spectrum encoding")
    hypertoken_spread_factor: int = Field(
        default=4, ge=2, le=8, description="Spectrum spreading factor"
    )

    # Phase 50: Majorana Position Encoding
    use_majorana_position: bool = Field(
        default=True, description="Enable topological position encoding"
    )
    majorana_floquet_period: int = Field(
        default=16, ge=8, le=64, description="Floquet drive period"
    )

    # Phase 51: Born Rule Loss
    use_born_rule_loss: bool = Field(default=True, description="Enable QBRL amplitude loss")
    born_rule_temperature: float = Field(
        default=1.0, ge=0.1, le=5.0, description="Born rule temperature"
    )

    # Phase 52: Quantum Fidelity Regularization
    use_quantum_fidelity_loss: bool = Field(default=True, description="Enable fidelity loss term")
    fidelity_loss_weight: float = Field(
        default=0.01, ge=0.0, le=0.1, description="Fidelity regularization weight"
    )

    # Pillar 3: Topological Reasoning
    # Phase 53: QASA Attention
    use_qasa_attention: bool = Field(default=True, description="Enable VQC attention scoring")
    qasa_vqc_layers: int = Field(default=2, ge=1, le=4, description="QASA VQC depth")
    qasa_entanglement_strength: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Entanglement contribution"
    )

    # Phase 55: MPQR Multi-Path Reasoning
    use_mpqr_reasoning: bool = Field(default=True, description="Enable Grover path amplification")
    mpqr_num_paths: int = Field(default=8, ge=4, le=16, description="Reasoning paths")
    mpqr_grover_iterations: int = Field(default=3, ge=1, le=5, description="Grover iterations")

    # Phase 56: Topological Wavelet Attention
    use_topological_wavelet: bool = Field(default=True, description="Enable TWA Betti number bias")
    twa_num_scales: int = Field(default=4, ge=2, le=8, description="Wavelet scales")

    # Phase 57: TD-MoE Tucker Decomposition
    use_td_moe: bool = Field(default=True, description="Enable Tucker-decomposed experts")
    td_moe_tucker_rank: int = Field(
        default=16, ge=8, le=64, description="Tucker decomposition rank"
    )

    # Phase 58: Symplectic GNN Kalman
    use_symplectic_gnn_kalman: bool = Field(
        default=True, description="Enable Hamiltonian GNN dynamics"
    )
    sgkf_dt: float = Field(default=0.01, ge=0.001, le=0.1, description="Symplectic timestep")

    # Pillar 4: Training & Optimization
    # Phase 59: Quantum Adiabatic Optimizer
    use_adiabatic_optimizer: bool = Field(
        default=True, description="Enable QAO for global optimization"
    )
    qao_initial_temp: float = Field(
        default=10.0, ge=1.0, le=100.0, description="QAO initial temperature"
    )
    qao_final_temp: float = Field(
        default=0.01, ge=0.001, le=1.0, description="QAO final temperature"
    )

    # Phase 60: Geodesic Optimizer
    use_geodesic_optimizer: bool = Field(
        default=True, description="Enable manifold-aware optimization"
    )
    geodesic_momentum: float = Field(default=0.9, ge=0.0, le=0.99, description="Geodesic momentum")

    # Phase 61: AlphaQubit-2 Decoder
    use_alphaqubit_decoder: bool = Field(default=True, description="Enable neural syndrome decoder")
    alphaqubit_num_layers: int = Field(
        default=2, ge=1, le=4, description="Decoder attention layers"
    )

    # Phase 62: VQEM Error Mitigation
    use_vqem: bool = Field(default=True, description="Enable variational error mitigation")
    vqem_num_params: int = Field(default=32, ge=16, le=128, description="VQEM circuit parameters")

    # Phase 64: Gradient Teleportation
    use_gradient_teleportation: bool = Field(
        default=True, description="Enable distributed gradient teleport"
    )

    # Pillar 5: Memory & Continual Learning
    # Phase 65/83: Quantum Crystallization
    use_quantum_crystallization: bool = Field(
        default=True, description="Enable knowledge crystallization"
    )
    crystallization_threshold: float = Field(
        default=0.5, ge=0.1, le=0.9, description="Importance threshold"
    )

    # Phase 68: Quantum Neuromorphic Memory
    use_neuromorphic_memory: bool = Field(
        default=True, description="Enable spiking quantum neurons"
    )
    neuromorphic_tau: float = Field(
        default=10.0, ge=1.0, le=100.0, description="Membrane time constant"
    )

    # Pillar 6: Coherence Mesh
    # Phase 70: Multi-Stage Hamiltonian
    use_multi_stage_hamiltonian: bool = Field(
        default=True, description="Enable staged Hamiltonian learning"
    )
    hamiltonian_num_stages: int = Field(default=4, ge=2, le=8, description="Hamiltonian stages")

    # Phase 72: Random Natural Gradient
    use_random_natural_gradient: bool = Field(
        default=True, description="Enable Monte Carlo QNG approximation"
    )
    rng_num_samples: int = Field(default=10, ge=5, le=50, description="RNG Monte Carlo samples")

    # Pillar 7: Advanced Quantum Intelligence
    # Phase 78: SPINI Integrator
    use_spini_integrator: bool = Field(
        default=True, description="Enable symplectic neural integrator"
    )
    spini_friction: float = Field(
        default=0.1, ge=0.0, le=0.5, description="SPINI friction coefficient"
    )

    # Phase 79: QCOT Reasoning
    use_qcot_reasoning: bool = Field(default=True, description="Enable quantum chain-of-thought")
    qcot_reasoning_steps: int = Field(default=3, ge=1, le=8, description="QCOT reasoning depth")

    # Phase 80: Waveform Attention
    use_waveform_attention: bool = Field(
        default=True, description="Enable phase-coherent attention"
    )

    # =========================================================================
    # Architecture tuning flags (enabled by default - HPO will optimize these)
    # =========================================================================
    tune_embedding_dim: bool = True
    tune_reasoning_blocks: bool = True
    tune_moe_experts: bool = True
    tune_mamba_params: bool = True  # Tune Mamba state/conv dims
    tune_wlam_params: bool = True  # Tune WLAM heads/landmarks
    tune_ff_expansion: bool = True  # Tune FFN expansion
    tune_tt_rank: bool = True  # Tune TT decomposition rank
    tune_regularization: bool = True  # Tune dropout/weight decay
    tune_quantum_params: bool = True  # Tune quantum enhancement parameters (Phases 26-36)
    tune_training_loop: bool = True  # Tune training loop optimization (QNG, GaLore, etc.)
    tune_hamiltonian: bool = True  # Tune Hamiltonian/TimeCrystal parameters
    tune_qsg: bool = True  # Tune QSG generation parameters
    tune_quantum_control: bool = True  # Tune RLS/PID/TNKF control parameters
    tune_qhpm_crystallization: bool = True  # Tune QHPM crystallization parameters
    tune_quantum_phases_v5: bool = True  # Tune Phases 47-84 parameters


class TrainingJobInfo(BaseModel):
    """Information about a training job."""

    job_id: str
    state: TrainingState
    mode: TrainingMode
    model_size: str
    current_stage: int = 0
    current_epoch: int = 0
    global_step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    throughput: float = 0.0
    progress_percent: float = 0.0
    started_at: str | None = None
    updated_at: str | None = None
    hpo_trial_current: int = 0
    hpo_trial_total: int = 0
    best_hyperparams: dict[str, Any] | None = None


class HPOTrialInfo(BaseModel):
    """Information about an HPO trial."""

    trial_id: int
    status: str  # running, completed, failed
    learning_rate: float
    batch_size: int
    optimizer: str
    loss: float | None = None
    duration_seconds: float | None = None
    memory_mb: float | None = None  # Current RSS memory in MB
    peak_memory_mb: float | None = None  # Peak RSS memory in MB
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    # Multi-objective quality metrics
    perplexity: float | None = None
    mean_confidence: float | None = None
    expected_calibration_error: float | None = None
    composite_score: float | None = None


class HPOSweepInfo(BaseModel):
    """Information about an HPO sweep."""

    sweep_id: str
    stage: HPOStage
    state: str  # pending, running, completed, cancelled
    max_trials: int
    completed_trials: int = 0
    best_trial_id: str | int | None = None
    best_loss: float | None = None
    best_composite_score: float | None = None  # Multi-objective best score
    best_perplexity: float | None = None  # Best trial's perplexity
    best_hyperparams: dict[str, Any] | None = None
    started_at: str | None = None
    trials: list[HPOTrialInfo] = Field(default_factory=list)


# ============================================================================
# Distributed Training Models
# ============================================================================


class DistributedConfigRequest(BaseModel):
    """Request body for configuring distributed training."""

    role: str = Field(default="standalone", pattern="^(host|worker|standalone)$")
    host_address: str | None = None
    port: int = Field(default=12345, ge=1024, le=65535)
    cluster_secret: str | None = None
    shared_checkpoint_dir: str = "/shared/checkpoints"
    communication_protocol: str = Field(default="ring", pattern="^(ring|auto)$")


class StartHostRequest(BaseModel):
    """Request body for starting as a Host node."""

    port: int = Field(default=12345, ge=1024, le=65535)
    cluster_secret: str | None = None
    shared_checkpoint_dir: str = "/shared/checkpoints"
    communication_protocol: str = Field(default="ring", pattern="^(ring|auto)$")


class JoinClusterRequest(BaseModel):
    """Request body for joining a cluster as a Worker."""

    host_address: str
    cluster_secret: str


class WorkerInfoResponse(BaseModel):
    """Information about a connected worker node."""

    worker_id: str
    hostname: str
    address: str
    status: str  # connected, training, disconnected
    cpu_count: int = 0
    memory_gb: float = 0.0
    connected_at: str = ""
    last_heartbeat: str = ""
    task_index: int = -1


class ClusterStatusResponse(BaseModel):
    """Current status of the distributed training cluster."""

    role: str  # host, worker, standalone
    cluster_secret: str | None = None
    workers: list[WorkerInfoResponse] = Field(default_factory=list)
    is_ready: bool = False
    tf_config: dict[str, Any] | None = None
    host_address: str | None = None
    is_training: bool = False
    error: str | None = None


class StartDistributedTrainingRequest(BaseModel):
    """Request body for starting distributed training."""

    training_config: dict[str, Any] = Field(default_factory=dict)
    curriculum_id: str | None = None


# ============================================================================
# Training Presets (Pre-validated HPO Configurations)
# ============================================================================

TRAINING_PRESETS = {
    "1b": {
        "name": "HighNoon 1B",
        "params": "1B",
        "ram_estimate_gb": 4,
        "vram_estimate_gb": 8,
        "optimal_config": {
            "learning_rate": 2e-4,
            "batch_size": 32,
            "optimizer": "sophiag",
            "num_reasoning_blocks": 8,
            "block_pattern": "mamba_timecrystal",
            "warmup_steps": 500,
        },
    },
    "3b": {
        "name": "HighNoon 3B",
        "params": "3B",
        "ram_estimate_gb": 12,
        "vram_estimate_gb": 16,
        "optimal_config": {
            "learning_rate": 1e-4,
            "batch_size": 16,
            "optimizer": "sophiag",
            "num_reasoning_blocks": 12,
            "block_pattern": "mamba_timecrystal",
            "warmup_steps": 1000,
        },
    },
    "7b": {
        "name": "HighNoon 7B",
        "params": "7B",
        "ram_estimate_gb": 28,
        "vram_estimate_gb": 24,
        "optimal_config": {
            "learning_rate": 5e-5,
            "batch_size": 8,
            "optimizer": "sophiag",
            "num_reasoning_blocks": 16,
            "block_pattern": "hybrid",
            "warmup_steps": 2000,
        },
    },
    "12b": {
        "name": "HighNoon 12B",
        "params": "12B",
        "ram_estimate_gb": 48,
        "vram_estimate_gb": 40,
        "optimal_config": {
            "learning_rate": 3e-5,
            "batch_size": 4,
            "optimizer": "sophiag",
            "num_reasoning_blocks": 20,
            "block_pattern": "hybrid",
            "warmup_steps": 3000,
        },
    },
    "20b": {
        "name": "HighNoon 20B (Max)",
        "params": "20B",
        "ram_estimate_gb": 64,
        "vram_estimate_gb": 80,
        "optimal_config": {
            "learning_rate": 2e-5,
            "batch_size": 2,
            "optimizer": "sophiag",
            "num_reasoning_blocks": 24,
            "block_pattern": "hybrid",
            "warmup_steps": 5000,
        },
        "lite_limit": True,
    },
}


# ============================================================================
# Application Factory
# ============================================================================


def create_app(debug: bool = False) -> FastAPI:
    """Create and configure the FastAPI application.

    This is an API-only backend. The React frontend handles all UI rendering.

    Args:
        debug: Enable debug mode.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="HighNoon Language Framework API",
        description="API backend for curriculum building and training management",
        version="1.0.0",
        debug=debug,
    )

    # CORS middleware for React frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # ========================================================================
    # Health Check
    # ========================================================================

    @app.get("/")
    async def health_check():
        """API health check endpoint."""
        return {"status": "ok", "service": "highnoon-api", "version": "1.0.0"}

    # ========================================================================
    # Local Dataset Catalog (in-memory for now)
    # ========================================================================

    local_datasets: list[dict[str, Any]] = [
        {
            "dataset_id": "openwebtext",
            "provider": "huggingface",
            "total_size_bytes": 8_000_000_000,
            "media_types": ["text"],
            "description": "Web content extracted from Reddit URLs",
            "downloads": 500000,
            "objects": [{"key": "train", "media_type": "parquet"}],
        },
        {
            "dataset_id": "bigcode/commitpackft",
            "provider": "huggingface",
            "total_size_bytes": 2_500_000_000,
            "media_types": ["text"],
            "description": "Code commits for instruction fine-tuning",
            "downloads": 150000,
            "objects": [{"key": "train", "media_type": "parquet"}],
        },
        {
            "dataset_id": "cognitivecomputations/wizard_vicuna_70k_unfiltered",
            "provider": "huggingface",
            "total_size_bytes": 500_000_000,
            "media_types": ["text"],
            "description": "Vicuna-style conversation dataset",
            "downloads": 75000,
            "objects": [{"key": "train", "media_type": "jsonl"}],
        },
    ]

    # ========================================================================
    # Curriculum Storage (Persistent to disk)
    # ========================================================================

    CURRICULA_FILE = PROJECT_ROOT / "artifacts" / "curricula.json"

    def load_curricula() -> list[dict[str, Any]]:
        """Load curricula from persistent storage."""
        if CURRICULA_FILE.exists():
            try:
                with open(CURRICULA_FILE, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Curriculum] Failed to load curricula: {e}")
        return []

    def save_curricula(data: list[dict[str, Any]]) -> None:
        """Save curricula to persistent storage."""
        CURRICULA_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(CURRICULA_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Curriculum] Failed to save curricula: {e}")

    def resolve_curriculum_to_datasets(
        curriculum_id: str | None,
        curricula_list: list[dict[str, Any]],
    ) -> tuple[str | None, list[str]]:
        """Resolve curriculum_id to HuggingFace dataset names.

        Checks user-created curricula first, then falls back to CURRICULUM_PRESETS.

        Args:
            curriculum_id: The curriculum ID to resolve
            curricula_list: List of user-created curricula

        Returns:
            Tuple of (primary_dataset_name, all_dataset_names)
        """
        if not curriculum_id:
            return None, []

        curriculum_datasets: list[str] = []

        # First, check user-created curricula
        for c in curricula_list:
            if c.get("id") == curriculum_id:
                for stage in c.get("stages", []):
                    for ds_entry in stage.get("datasets", []):
                        ds_id = (
                            ds_entry.get("dataset_id") if isinstance(ds_entry, dict) else ds_entry
                        )
                        if ds_id and ds_id not in curriculum_datasets:
                            curriculum_datasets.append(ds_id)
                break

        # Fallback: Check predefined CURRICULUM_PRESETS
        if not curriculum_datasets and curriculum_id in CURRICULUM_PRESETS:
            preset = CURRICULUM_PRESETS[curriculum_id]
            curriculum_datasets = preset.get("hf_datasets", [])
            print(f"[Curriculum] Using preset '{preset['name']}': {curriculum_datasets}")

        primary_dataset = curriculum_datasets[0] if curriculum_datasets else None
        if primary_dataset:
            print(f"[Curriculum] Resolved {curriculum_id} -> {primary_dataset}")

        return primary_dataset, curriculum_datasets

    # Load curricula from disk on startup
    curricula: list[dict[str, Any]] = load_curricula()
    print(f"[Curriculum] Loaded {len(curricula)} curricula from disk")

    # Individual stages (legacy, kept for backwards compatibility)
    curriculum_stages: list[dict[str, Any]] = []

    # ========================================================================
    # Predefined Curriculum Presets (with HuggingFace dataset mappings)
    # ========================================================================
    # These are presets that map curriculum IDs to HuggingFace datasets.
    # When users select a preset curriculum in the UI, this resolves
    # the curriculum_id to an actual dataset for training.

    CURRICULUM_PRESETS: dict[str, dict[str, Any]] = {
        # Chat / Conversational
        "chat-conversational": {
            "name": "Chat / Conversational",
            "hf_datasets": [
                "databricks/databricks-dolly-15k",  # Primary conversational dataset
                "OpenAssistant/oasst1",
                "stingning/ultrachat",
            ],
            "description": "Multi-turn dialogue and conversational AI training",
        },
        # Code / Programming
        "code-programming": {
            "name": "Code / Programming",
            "hf_datasets": [
                "bigcode/the-stack-dedup",
                "bigcode/starcoderdata",
                "codeparrot/github-code",
            ],
            "description": "Code generation and programming assistance",
        },
        # Reasoning / Math
        "reasoning-math": {
            "name": "Reasoning / Math",
            "hf_datasets": [
                "lighteval/MATH",
                "gsm8k",
                "meta-math/MetaMathQA",
            ],
            "description": "Mathematical reasoning and problem solving",
        },
        # General Language / Web
        "general-language": {
            "name": "General Language",
            "hf_datasets": [
                "openwebtext",
                "wikipedia",
                "c4",
            ],
            "description": "General language understanding from web text",
        },
        # Instruction Following
        "instruction-following": {
            "name": "Instruction Following",
            "hf_datasets": [
                "HuggingFaceH4/ultrafeedback_binarized",
                "Anthropic/hh-rlhf",
                "teknium/OpenHermes-2.5",
            ],
            "description": "Instruction-following and alignment tuning",
        },
        # Verso Baseline (comprehensive)
        "verso-baseline": {
            "name": "Verso Baseline",
            "hf_datasets": [
                "databricks/databricks-dolly-15k",
                "openwebtext",
                "bigcode/starcoderdata",
            ],
            "description": "Comprehensive baseline curriculum for HighNoon training",
        },
    }

    # ========================================================================
    # API Routes - HuggingFace Hub Integration
    # ========================================================================

    @app.get("/api/huggingface/search")
    async def search_huggingface_datasets(
        query: str = "",
        limit: int = 20,
        offset: int = 0,
    ):
        """Search HuggingFace Hub for datasets."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                params = {
                    "search": query,
                    "limit": min(limit, 50),
                    "offset": offset,
                    "sort": "downloads",
                    "direction": "-1",
                }
                response = await client.get(
                    "https://huggingface.co/api/datasets",
                    params=params,
                )
                response.raise_for_status()
                datasets = response.json()

                # Transform to our format
                results = []
                for ds in datasets:
                    results.append(
                        {
                            "dataset_id": ds.get("id", ""),
                            "provider": "huggingface",
                            "description": (
                                ds.get("description", "")[:200] if ds.get("description") else ""
                            ),
                            "downloads": ds.get("downloads", 0),
                            "likes": ds.get("likes", 0),
                            "tags": ds.get("tags", [])[:5],
                            "last_modified": ds.get("lastModified", ""),
                        }
                    )
                return {"datasets": results, "total": len(results)}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail="HuggingFace API error")
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503, detail=f"Failed to connect to HuggingFace: {str(e)}"
            )

    @app.get("/api/huggingface/dataset/{dataset_id:path}")
    async def get_huggingface_dataset_info(dataset_id: str):
        """Get detailed info about a HuggingFace dataset."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"https://huggingface.co/api/datasets/{dataset_id}",
                )
                response.raise_for_status()
                data = response.json()

                # Also try to get dataset info
                info_response = await client.get(
                    f"https://datasets-server.huggingface.co/info?dataset={dataset_id}",
                )
                dataset_info = {}
                if info_response.status_code == 200:
                    dataset_info = info_response.json()

                return {
                    "dataset_id": data.get("id", dataset_id),
                    "provider": "huggingface",
                    "description": data.get("description", ""),
                    "downloads": data.get("downloads", 0),
                    "likes": data.get("likes", 0),
                    "tags": data.get("tags", []),
                    "card_data": data.get("cardData", {}),
                    "configs": (
                        list(dataset_info.get("dataset_info", {}).keys()) if dataset_info else []
                    ),
                    "dataset_info": dataset_info.get("dataset_info", {}),
                }
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail="Dataset not found")
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503, detail=f"Failed to connect to HuggingFace: {str(e)}"
            )

    @app.get("/api/huggingface/dataset/{dataset_id:path}/preview")
    async def preview_huggingface_dataset(
        dataset_id: str,
        config: str = "default",
        split: str = "train",
        rows: int = 10,
    ):
        """Get sample rows from a HuggingFace dataset."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                params = {
                    "dataset": dataset_id,
                    "config": config,
                    "split": split,
                    "offset": 0,
                    "length": min(rows, 100),
                }
                response = await client.get(
                    "https://datasets-server.huggingface.co/rows",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                # Extract features (column info)
                features = data.get("features", [])
                rows_data = data.get("rows", [])

                # Simplify rows to just the row content
                simplified_rows = [r.get("row", r) for r in rows_data]

                return {
                    "dataset_id": dataset_id,
                    "config": config,
                    "split": split,
                    "features": features,
                    "rows": simplified_rows[:rows],
                    "num_rows_total": data.get("num_rows_total", len(rows_data)),
                }
        except httpx.HTTPStatusError as e:
            # Try without config
            if config != "default":
                return await preview_huggingface_dataset(dataset_id, "default", split, rows)
            raise HTTPException(status_code=e.response.status_code, detail="Preview not available")
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503, detail=f"Failed to connect to HuggingFace: {str(e)}"
            )

    # ========================================================================
    # API Routes - Local Datasets
    # ========================================================================

    @app.get("/api/datasets")
    async def list_datasets():
        """List available datasets in local catalog."""
        return {"datasets": local_datasets}

    @app.post("/api/datasets/add-huggingface")
    async def add_huggingface_dataset(payload: AddHuggingFaceDataset):
        """Add a HuggingFace dataset to the local catalog."""
        # Check if already exists
        for ds in local_datasets:
            if ds["dataset_id"] == payload.dataset_id:
                return {"status": "exists", "dataset": ds}

        # Fetch info from HuggingFace
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"https://huggingface.co/api/datasets/{payload.dataset_id}",
                )
                response.raise_for_status()
                data = response.json()

                # Fetch actual row count from datasets-server API
                num_examples = 0
                try:
                    info_response = await client.get(
                        f"https://datasets-server.huggingface.co/info?dataset={payload.dataset_id}",
                        timeout=10.0,
                    )
                    if info_response.status_code == 200:
                        info_data = info_response.json()
                        dataset_info = info_data.get("dataset_info", {})
                        # Sum up rows from all configs and splits
                        for _config_name, config_data in dataset_info.items():
                            splits_info = config_data.get("splits", {})
                            for _split_name, split_data in splits_info.items():
                                num_examples += split_data.get("num_examples", 0)
                except Exception:
                    # If we can't get row count, fall back to 0
                    pass

                new_dataset = {
                    "dataset_id": payload.dataset_id,
                    "provider": "huggingface",
                    "description": (data.get("description", "") or "")[:200],
                    "downloads": data.get("downloads", 0),
                    "num_examples": num_examples,
                    "total_size_bytes": num_examples * 1000,  # Approximate for backwards compat
                    "media_types": ["text"],
                    "objects": [
                        {"key": split, "media_type": "parquet"} for split in payload.splits
                    ],
                    "config_name": payload.config_name,
                }
                local_datasets.append(new_dataset)
                return {"status": "added", "dataset": new_dataset}
        except httpx.HTTPStatusError:
            raise HTTPException(status_code=404, detail="Dataset not found on HuggingFace")
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503, detail=f"Failed to connect to HuggingFace: {str(e)}"
            )

    @app.delete("/api/datasets/{dataset_id:path}")
    async def remove_dataset(dataset_id: str):
        """Remove a dataset from the local catalog."""
        for i, ds in enumerate(local_datasets):
            if ds["dataset_id"] == dataset_id:
                removed = local_datasets.pop(i)
                return {"status": "removed", "dataset": removed}
        raise HTTPException(status_code=404, detail="Dataset not found in catalog")

    @app.post("/api/datasets/use-in-curriculum")
    async def use_dataset_in_curriculum(payload: UseDatasetRequest):
        """Add a dataset to a curriculum stage."""
        # Find the stage
        target_stage = None
        for stage in curriculum_stages:
            if stage["name"] == payload.stage_name:
                target_stage = stage
                break

        if not target_stage:
            raise HTTPException(status_code=404, detail=f"Stage '{payload.stage_name}' not found")

        # Check if dataset already in stage
        for ds in target_stage["datasets"]:
            if ds["dataset_id"] == payload.dataset_id:
                return {
                    "status": "exists",
                    "message": f"Dataset '{payload.dataset_id}' already in stage '{payload.stage_name}'",
                }

        # Add dataset to stage
        target_stage["datasets"].append(
            {
                "dataset_id": payload.dataset_id,
                "weight": payload.weight,
            }
        )

        return {
            "status": "added_to_curriculum",
            "dataset_id": payload.dataset_id,
            "stage": payload.stage_name,
            "message": f"Dataset '{payload.dataset_id}' added to stage '{payload.stage_name}'",
        }

    @app.get("/api/datasets/{dataset_id:path}")
    async def get_local_dataset_info(dataset_id: str):
        """Get detailed information about a specific local dataset."""
        for ds in local_datasets:
            if ds["dataset_id"] == dataset_id:
                return ds
        raise HTTPException(status_code=404, detail="Dataset not found")

    @app.get("/datasets/{dataset_id}")
    async def get_dataset_info(dataset_id: str):
        """Get detailed information about a specific dataset."""
        return {
            "dataset_id": dataset_id,
            "provider": "huggingface",
            "total_size_bytes": 1_000_000_000,
            "media_types": ["text"],
            "objects": [{"key": "train", "media_type": "parquet"}],
        }

    @app.get("/datasets/{dataset_id}/inspection")
    async def inspect_dataset(dataset_id: str, object_index: int = 0):
        """Inspect a dataset."""
        return {
            "dataset_id": dataset_id,
            "object_index": object_index,
            "columns": ["text", "label"],
            "row_count": 1000000,
            "sample": {"text": "Example text...", "label": 1},
        }

    @app.get("/datasets/{dataset_id}/spec")
    async def suggest_spec(dataset_id: str, object_index: int = 0, path_key: str = "data_path"):
        """Suggest a preprocessor spec for a dataset."""
        return {
            "dataset_id": dataset_id,
            "spec": {
                "input_columns": ["text"],
                "output_format": "tfrecord",
                "tokenizer": "qwt",
            },
        }

    @app.post("/datasets/{dataset_id}/download")
    async def download_object(dataset_id: str, object_index: int = 0, overwrite: bool = False):
        """Download a dataset object."""
        return {"status": "downloaded", "dataset_id": dataset_id, "object_index": object_index}

    @app.post("/datasets/{dataset_id}/materialize")
    async def materialize_dataset(dataset_id: str, object_index: int = 0):
        """Materialize a dataset spec."""
        return {"status": "materialized", "dataset_id": dataset_id}

    @app.post("/datasets/{dataset_id}/validate")
    async def validate_dataset(dataset_id: str, object_index: int = 0):
        """Validate a dataset preprocessor."""
        return {"status": "validated", "dataset_id": dataset_id}

    @app.get("/jobs/{job_id}")
    async def job_status(job_id: str):
        """Get job status."""
        return {"job_id": job_id, "status": "completed"}

    # ========================================================================
    # API Routes - Curriculum
    # ========================================================================

    @app.get("/api/curriculum/presets")
    async def get_curriculum_presets():
        """Get predefined curriculum presets with HuggingFace dataset mappings.

        These presets can be selected in the HPO page for quick curriculum selection
        without needing to manually configure stages.
        """
        presets = []
        for preset_id, preset_data in CURRICULUM_PRESETS.items():
            presets.append(
                {
                    "id": preset_id,
                    "name": preset_data["name"],
                    "description": preset_data.get("description", ""),
                    "hf_datasets": preset_data.get("hf_datasets", []),
                    "stages": [
                        {
                            "name": "default",
                            "display_name": preset_data["name"],
                            "datasets": [
                                {"dataset_id": ds, "weight": 1.0}
                                for ds in preset_data.get("hf_datasets", [])
                            ],
                        }
                    ],
                }
            )
        return {"presets": presets}

    @app.get("/api/curriculum")
    async def list_curricula():
        """List all saved curricula.

        Returns curricula in the format expected by the HPO page:
        [{ id, name, stages, vocab_size, context_window, created_at, updated_at }]
        """
        return curricula

    @app.post("/api/curriculum")
    async def save_curriculum(payload: SaveCurriculumRequest):
        """Save or update a complete curriculum.

        If a curriculum with the same ID exists, it will be updated.
        Otherwise, a new curriculum will be created.
        """
        now = datetime.utcnow().isoformat()

        # Convert Pydantic model to dict for storage
        curriculum_data = {
            "id": payload.id,
            "name": payload.name,
            "stages": [stage.model_dump() for stage in payload.stages],
            "vocab_size": payload.vocab_size,
            "context_window": payload.context_window,
            "created_at": payload.created_at or now,
            "updated_at": now,
        }

        # Check if curriculum already exists (update)
        for i, c in enumerate(curricula):
            if c["id"] == payload.id:
                curricula[i] = curriculum_data
                save_curricula(curricula)  # Persist to disk
                return {"status": "updated", "curriculum": curriculum_data}

        # Create new curriculum
        curricula.append(curriculum_data)
        save_curricula(curricula)  # Persist to disk
        return {"status": "created", "curriculum": curriculum_data}

    @app.get("/api/curriculum/{curriculum_id}")
    async def get_curriculum(curriculum_id: str):
        """Get a specific curriculum by ID."""
        for c in curricula:
            if c["id"] == curriculum_id:
                return c
        raise HTTPException(status_code=404, detail="Curriculum not found")

    @app.delete("/api/curriculum/{curriculum_id}")
    async def delete_curriculum(curriculum_id: str):
        """Delete a curriculum by ID."""
        for i, c in enumerate(curricula):
            if c["id"] == curriculum_id:
                removed = curricula.pop(i)
                save_curricula(curricula)  # Persist to disk
                return {"status": "deleted", "curriculum": removed}
        raise HTTPException(status_code=404, detail="Curriculum not found")

    # ========================================================================
    # API Routes - Curriculum Stages (Legacy)
    # ========================================================================

    @app.get("/api/curriculum/stages")
    async def get_curriculum_stages():
        """Get all curriculum stages."""
        return {"stages": curriculum_stages}

    @app.post("/api/curriculum/stages")
    async def create_curriculum_stage(payload: CreateStageRequest):
        """Create a new curriculum stage."""
        # Check if stage already exists
        for stage in curriculum_stages:
            if stage["name"] == payload.name:
                raise HTTPException(status_code=400, detail="Stage already exists")

        new_stage = {
            "name": payload.name,
            "display_name": payload.display_name,
            "module": payload.module,
            "datasets": [],
            "epochs": payload.epochs,
            "learning_rate": payload.learning_rate,
            "batch_size": payload.batch_size,
        }
        curriculum_stages.append(new_stage)
        return {"status": "created", "stage": new_stage}

    @app.get("/api/curriculum/stages/{stage_name}")
    async def get_curriculum_stage(stage_name: str):
        """Get a specific curriculum stage."""
        for stage in curriculum_stages:
            if stage["name"] == stage_name:
                return stage
        raise HTTPException(status_code=404, detail="Stage not found")

    @app.put("/api/curriculum/stages/{stage_name}")
    async def update_curriculum_stage(stage_name: str, payload: UpdateStageRequest):
        """Update a curriculum stage."""
        for stage in curriculum_stages:
            if stage["name"] == stage_name:
                if payload.display_name is not None:
                    stage["display_name"] = payload.display_name
                if payload.module is not None:
                    stage["module"] = payload.module
                if payload.epochs is not None:
                    stage["epochs"] = payload.epochs
                if payload.learning_rate is not None:
                    stage["learning_rate"] = payload.learning_rate
                if payload.batch_size is not None:
                    stage["batch_size"] = payload.batch_size
                return {"status": "updated", "stage": stage}
        raise HTTPException(status_code=404, detail="Stage not found")

    @app.delete("/api/curriculum/stages/{stage_name}")
    async def delete_curriculum_stage(stage_name: str):
        """Delete a curriculum stage."""
        for i, stage in enumerate(curriculum_stages):
            if stage["name"] == stage_name:
                removed = curriculum_stages.pop(i)
                return {"status": "deleted", "stage": removed}
        raise HTTPException(status_code=404, detail="Stage not found")

    @app.post("/api/curriculum/stages/{stage_name}/datasets")
    async def add_dataset_to_stage(stage_name: str, payload: AddDatasetToStageRequest):
        """Add a dataset to a curriculum stage."""
        for stage in curriculum_stages:
            if stage["name"] == stage_name:
                # Check if dataset already in stage
                for ds in stage["datasets"]:
                    if ds["dataset_id"] == payload.dataset_id:
                        return {"status": "exists", "message": "Dataset already in stage"}

                stage["datasets"].append(
                    {
                        "dataset_id": payload.dataset_id,
                        "weight": payload.weight,
                    }
                )
                return {
                    "status": "added",
                    "stage": stage_name,
                    "dataset_id": payload.dataset_id,
                }
        raise HTTPException(status_code=404, detail="Stage not found")

    @app.delete("/api/curriculum/stages/{stage_name}/datasets/{dataset_id:path}")
    async def remove_dataset_from_stage(stage_name: str, dataset_id: str):
        """Remove a dataset from a curriculum stage."""
        for stage in curriculum_stages:
            if stage["name"] == stage_name:
                for i, ds in enumerate(stage["datasets"]):
                    if ds["dataset_id"] == dataset_id:
                        stage["datasets"].pop(i)
                        return {
                            "status": "removed",
                            "stage": stage_name,
                            "dataset_id": dataset_id,
                        }
                raise HTTPException(status_code=404, detail="Dataset not found in stage")
        raise HTTPException(status_code=404, detail="Stage not found")

    # ========================================================================
    # API Routes - Bulk Operations
    # ========================================================================

    @app.post("/datasets/materialize_all")
    async def materialize_all():
        """Materialize all datasets."""
        return {"status": "pending", "job_id": "bulk_materialize_001"}

    @app.post("/datasets/validate_all")
    async def validate_all():
        """Validate all datasets."""
        return {"status": "pending", "job_id": "bulk_validate_001"}

    @app.get("/outputs/spec_summary")
    async def spec_summary():
        """Get spec summary."""
        return []

    @app.post("/datasets/register_remote")
    async def register_remote(payload: dict[str, Any]):
        """Register a remote dataset."""
        return {"status": "registered", "dataset_id": payload.get("dataset_id")}

    # ========================================================================
    # API Routes - Operations Console
    # ========================================================================

    @app.post("/ops/{name}/start")
    async def start_process(name: str):
        """Start a process."""
        return {"name": name, "running": True, "pid": 12345}

    @app.post("/ops/{name}/stop")
    async def stop_process(name: str):
        """Stop a process."""
        return {"name": name, "running": False}

    @app.get("/ops/{name}/status")
    async def process_status(name: str):
        """Get process status."""
        return {"name": name, "running": False, "pid": None}

    @app.get("/ops/{name}/logs")
    async def process_logs(name: str, tail: int = 200):
        """Get process logs."""
        return {"name": name, "tail": tail, "lines": []}

    # ========================================================================
    # API Routes - Training
    # ========================================================================

    # In-memory training job storage (for demo purposes)
    training_jobs: dict[str, dict[str, Any]] = {}
    hpo_sweeps: dict[str, dict[str, Any]] = {}
    active_websockets: dict[str, list[WebSocket]] = {}

    # Store active SweepExecutor instances for multi-trial orchestration
    from highnoon.services.scheduler_factory import create_scheduler
    from highnoon.services.sweep_executor import SweepConfig, SweepExecutor

    active_sweeps: dict[str, SweepExecutor] = {}

    @app.get("/api/training/presets")
    async def get_training_presets():
        """Get pre-validated training presets with RAM/VRAM estimates."""
        presets = []
        for size_key, preset in TRAINING_PRESETS.items():
            presets.append(
                {
                    "size": size_key,
                    "name": preset["name"],
                    "params": preset["params"],
                    "ram_estimate_gb": preset["ram_estimate_gb"],
                    "vram_estimate_gb": preset["vram_estimate_gb"],
                    "lite_limit": preset.get("lite_limit", False),
                }
            )
        return {"presets": presets}

    @app.get("/api/training/estimate-resources")
    async def estimate_resources(model_size: str = "3b", batch_size: int = 8):
        """Estimate RAM and VRAM requirements for a model configuration."""
        preset = TRAINING_PRESETS.get(model_size)
        if not preset:
            raise HTTPException(status_code=404, detail=f"Unknown model size: {model_size}")

        # Adjust estimates based on batch size
        base_ram = preset["ram_estimate_gb"]
        base_vram = preset["vram_estimate_gb"]
        batch_multiplier = batch_size / preset["optimal_config"]["batch_size"]

        return {
            "model_size": model_size,
            "batch_size": batch_size,
            "ram_estimate_gb": round(base_ram * (1 + 0.1 * (batch_multiplier - 1)), 1),
            "vram_estimate_gb": round(base_vram * (1 + 0.2 * (batch_multiplier - 1)), 1),
            "optimal_batch_size": preset["optimal_config"]["batch_size"],
        }

    @app.post("/api/training/start")
    async def start_training(payload: StartTrainingRequest):
        """Start a new training job.

        Supports three modes:
        - quick_train: Uses baseline presets from TRAINING_PRESETS based on model_size
        - auto_tune/full_sweep: If sweep_id provided, uses best_hyperparams from that HPO sweep
        """
        job_id = str(uuid.uuid4())[:8]

        # Get model size from payload or default to 3b
        model_size = getattr(payload, "model_size", None) or "3b"
        preset = TRAINING_PRESETS.get(model_size, TRAINING_PRESETS["3b"])
        optimal_config = preset.get("optimal_config", {})

        # Load best hyperparameters from HPO sweep if provided
        best_hyperparams = None
        if payload.sweep_id:
            sweep = hpo_sweeps.get(payload.sweep_id)
            if sweep and sweep.get("state") == "completed":
                best_hyperparams = sweep.get("best_hyperparams", {})
                print(f"[Training] Loading best hyperparams from HPO sweep {payload.sweep_id}")
                print(f"[Training] Best hyperparams: {best_hyperparams}")
            else:
                print(
                    f"[Training] Warning: HPO sweep {payload.sweep_id} not found or not completed"
                )

        # Determine config source priority:
        # 1. HPO best_hyperparams (if sweep_id provided and sweep completed)
        # 2. Payload values (if explicitly provided)
        # 3. Preset defaults (for QUICK_TRAIN mode)
        if best_hyperparams:
            # Use HPO-optimized values
            learning_rate = best_hyperparams.get("learning_rate", 1e-4)
            batch_size = best_hyperparams.get("batch_size", 8)
            num_reasoning_blocks = best_hyperparams.get("num_reasoning_blocks", 8)
            optimizer = best_hyperparams.get("optimizer", "sophiag")
            embedding_dim = best_hyperparams.get("embedding_dim", 512)
            num_moe_experts = best_hyperparams.get("num_moe_experts", 8)
            mamba_state_dim = best_hyperparams.get("mamba_state_dim", 64)
        elif payload.mode == TrainingMode.QUICK_TRAIN:
            learning_rate = optimal_config.get("learning_rate", 1e-4)
            batch_size = optimal_config.get("batch_size", 8)
            num_reasoning_blocks = optimal_config.get("num_reasoning_blocks", 8)
            optimizer = optimal_config.get("optimizer", "sophiag")
            embedding_dim = 512
            num_moe_experts = 8
            mamba_state_dim = 64
        else:
            learning_rate = float(payload.learning_rate) if payload.learning_rate else 1e-4
            batch_size = payload.batch_size if payload.batch_size else 8
            num_reasoning_blocks = 8
            optimizer = "sophiag"
            embedding_dim = 512
            num_moe_experts = 8
            mamba_state_dim = 64

        # Determine HPO trials based on mode (0 if using completed sweep)
        hpo_trials = 0
        if not best_hyperparams:
            if payload.mode == TrainingMode.AUTO_TUNE:
                hpo_trials = 15  # Stage 1 only
            elif payload.mode == TrainingMode.FULL_SWEEP:
                hpo_trials = 50  # All 3 stages

        # Resolve curriculum to HuggingFace datasets
        hf_dataset_name, curriculum_datasets = resolve_curriculum_to_datasets(
            payload.curriculum_id, curricula
        )
        if hf_dataset_name:
            print(f"[Training] Using dataset from curriculum: {hf_dataset_name}")

        job = {
            "job_id": job_id,
            "state": TrainingState.RUNNING.value,
            "mode": payload.mode.value,
            "model_size": model_size,
            "current_stage": 0,
            "current_epoch": 0,
            "total_epochs": payload.epochs,
            "global_step": 0,
            "loss": 0.0,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "throughput": 0.0,
            "progress_percent": 0.0,
            "started_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "hpo_trial_current": 0,
            "hpo_trial_total": hpo_trials,
            "best_hyperparams": best_hyperparams,
            "sweep_id": payload.sweep_id,
            "config": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "optimizer": optimizer,
                "num_reasoning_blocks": num_reasoning_blocks,
                "embedding_dim": embedding_dim,
                "num_moe_experts": num_moe_experts,
                "mamba_state_dim": mamba_state_dim,
                "curriculum_id": payload.curriculum_id,
                "hf_dataset_name": hf_dataset_name,
                "curriculum_datasets": curriculum_datasets,
            },
        }
        training_jobs[job_id] = job

        if best_hyperparams:
            print(f"[Training] Started job {job_id} using HPO sweep {payload.sweep_id}")
        else:
            print(f"[Training] Started job {job_id} in {payload.mode.value} mode")
        print(f"[Training] Model size: {preset['name']} ({preset['params']} params)")
        print(
            f"[Training] Config: lr={learning_rate}, bs={batch_size}, blocks={num_reasoning_blocks}"
        )

        return {
            "status": "started",
            "job_id": job_id,
            "mode": payload.mode.value,
            "model_size": model_size,
            "hpo_trials": hpo_trials,
            "using_hpo_sweep": payload.sweep_id is not None,
        }

    @app.get("/api/training/{job_id}/status")
    async def get_training_status(job_id: str):
        """Get the status of a training job."""
        job = training_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        return TrainingJobInfo(**job)

    @app.get("/api/training/{job_id}/metrics")
    async def get_training_metrics(job_id: str):
        """Get current training metrics."""
        job = training_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        return {
            "job_id": job_id,
            "loss": job["loss"],
            "learning_rate": job["learning_rate"],
            "throughput": job["throughput"],
            "global_step": job["global_step"],
            "current_epoch": job["current_epoch"],
            "progress_percent": job["progress_percent"],
        }

    @app.post("/api/training/{job_id}/pause")
    async def pause_training(job_id: str):
        """Pause a running training job."""
        job = training_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        if job["state"] != TrainingState.RUNNING.value:
            raise HTTPException(status_code=400, detail="Job is not running")

        job["state"] = TrainingState.PAUSED.value
        job["updated_at"] = datetime.utcnow().isoformat()
        return {"status": "paused", "job_id": job_id}

    @app.post("/api/training/{job_id}/resume")
    async def resume_training(job_id: str):
        """Resume a paused training job."""
        job = training_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        if job["state"] != TrainingState.PAUSED.value:
            raise HTTPException(status_code=400, detail="Job is not paused")

        job["state"] = TrainingState.RUNNING.value
        job["updated_at"] = datetime.utcnow().isoformat()
        return {"status": "resumed", "job_id": job_id}

    @app.post("/api/training/{job_id}/cancel")
    async def cancel_training(job_id: str):
        """Cancel a training job."""
        job = training_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        if job["state"] in [TrainingState.COMPLETED.value, TrainingState.CANCELLED.value]:
            raise HTTPException(status_code=400, detail="Job already finished")

        job["state"] = TrainingState.CANCELLED.value
        job["updated_at"] = datetime.utcnow().isoformat()
        return {"status": "cancelled", "job_id": job_id}

    @app.get("/api/training/{job_id}/checkpoints")
    async def list_checkpoints(job_id: str):
        """List available checkpoints for a training job."""
        job = training_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        # Mock checkpoint list
        checkpoints = []
        for step in range(1000, job.get("global_step", 0) + 1, 1000):
            checkpoints.append(
                {
                    "step": step,
                    "epoch": step // 100,
                    "loss": 4.0 - (step * 0.001),
                    "path": f"./outputs/{job_id}/checkpoint-{step}",
                }
            )
        return {"checkpoints": checkpoints}

    @app.get("/api/training/jobs")
    async def list_training_jobs():
        """List all training jobs."""
        return {"jobs": list(training_jobs.values())}

    # ========================================================================
    # API Routes - HPO
    # ========================================================================

    @app.get("/api/hpo/configs")
    async def list_hpo_configs():
        """List available HPO configurations."""
        return {
            "configs": [
                {
                    "name": "codex_cli_base",
                    "description": "Tool-calling optimized for 100M-500M models",
                    "stages": ["COARSE", "REFINE", "FINE"],
                    "total_trials": 50,
                },
                {
                    "name": "reasoning_7b",
                    "description": "Long-context reasoning for 7B models",
                    "stages": ["COARSE", "REFINE"],
                    "total_trials": 30,
                },
                {
                    "name": "code_13b",
                    "description": "Code generation focus for 13B models",
                    "stages": ["COARSE"],
                    "total_trials": 15,
                },
            ],
        }

    @app.get("/api/hpo/schedulers")
    async def list_hpo_schedulers():
        """List available HPO schedulers and their capabilities.

        Returns scheduler metadata including:
        - Name and description
        - Whether it supports adaptive resource allocation
        - Supported parameters
        """
        return {
            "schedulers": [
                {
                    "name": "random",
                    "display_name": "Random Search",
                    "description": "Random sampling without early stopping",
                    "adaptive": False,
                    "supports_pruning": False,
                },
                {
                    "name": "bayesian",
                    "display_name": "Bayesian Optimization",
                    "description": "TPE-based sampling with learned surrogate model",
                    "adaptive": True,
                    "supports_pruning": False,
                },
                {
                    "name": "hyperband",
                    "display_name": "Hyperband",
                    "description": "Adaptive resource allocation with successive halving across brackets",
                    "adaptive": True,
                    "supports_pruning": True,
                    "parameters": {
                        "eta": {"type": "int", "default": 3, "min": 2, "max": 4},
                    },
                },
                {
                    "name": "successive_halving",
                    "display_name": "Successive Halving",
                    "description": "Single-bracket early stopping for known budgets",
                    "adaptive": True,
                    "supports_pruning": True,
                },
                {
                    "name": "pbt",
                    "display_name": "Population-Based Training",
                    "description": "Evolutionary optimization with exploit/explore",
                    "adaptive": True,
                    "supports_pruning": False,
                    "parameters": {
                        "population_size": {"type": "int", "default": 8, "min": 4, "max": 32},
                    },
                },
            ],
            "optuna_available": True,  # Report whether Optuna integration is ready
            "lite_edition": True,
            "max_params": 20_000_000_000,  # 20B Lite limit
        }

    @app.get("/api/hpo/configs/{config_name}")
    async def get_hpo_config(config_name: str):
        """Get details of an HPO configuration."""
        configs = {
            "codex_cli_base": {
                "name": "codex_cli_base",
                "description": "Tool-calling optimized for 100M-500M models",
                "parameter_groups": {
                    "architecture": [
                        {"name": "num_reasoning_blocks", "type": "int", "min": 4, "max": 16},
                        {
                            "name": "block_pattern",
                            "type": "choice",
                            "options": ["mamba_timecrystal", "hybrid"],
                        },
                    ],
                    "training": [
                        {"name": "learning_rate", "type": "log_uniform", "min": 1e-5, "max": 1e-3},
                        {
                            "name": "optimizer",
                            "type": "choice",
                            "options": [
                                "sophiag",
                                "qiao",
                                "grover",
                                "sympflow",
                                "adamw",
                                "adam",
                                "lion",
                            ],
                        },
                    ],
                },
            },
        }
        config = configs.get(config_name)
        if not config:
            raise HTTPException(status_code=404, detail="HPO config not found")
        return config

    @app.post("/api/hpo/sweep/start")
    async def start_hpo_sweep(payload: StartHPORequest):
        """Start a new HPO sweep with full multi-trial orchestration.

        This is the FIXED implementation that runs ALL requested trials,
        not just a single trial. Uses SweepExecutor for enterprise-grade
        orchestration with retry logic, checkpointing, and scheduler integration.
        """
        sweep_id = str(uuid.uuid4())[:8]

        sweep = {
            "sweep_id": sweep_id,
            "stage": payload.stage.value,
            "state": "running",
            "max_trials": payload.max_trials,
            "completed_trials": 0,
            "best_trial_id": None,
            "best_loss": None,
            "best_hyperparams": None,
            "started_at": datetime.utcnow().isoformat(),
            "trials": [],
            "config": {
                "epochs_per_trial": payload.epochs_per_trial,
                "search_strategy": payload.search_strategy,
                "lr_min": payload.lr_min,
                "lr_max": payload.lr_max,
                "batch_sizes": payload.batch_sizes,
                "optimizers": payload.optimizers,
                "curriculum_id": payload.curriculum_id,
            },
            # Model configuration for C++ orchestrator
            "model_config": {
                "vocab_size": payload.vocab_size,
                "context_window": payload.context_window,
                "embedding_dim": payload.embedding_dim,
                "num_reasoning_blocks": payload.num_reasoning_blocks,
                "num_moe_experts": payload.num_moe_experts,
                "position_embedding": payload.position_embedding,
                "param_budget": payload.param_budget,
            },
            # Architecture tuning flags
            "architecture_tuning": {
                "tune_embedding_dim": payload.tune_embedding_dim,
                "tune_reasoning_blocks": payload.tune_reasoning_blocks,
                "tune_moe_experts": payload.tune_moe_experts,
            },
            # Scheduler configuration (new)
            "scheduler_config": {
                "search_strategy": payload.search_strategy,
                "use_optuna": payload.use_optuna,
                "hyperband_eta": payload.hyperband_eta,
                "pbt_population_size": payload.pbt_population_size,
            },
        }
        hpo_sweeps[sweep_id] = sweep

        # Write config to disk for trial runner
        config_dir = PROJECT_ROOT / "artifacts" / "hpo_trials"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / f"sweep_{sweep_id}_config.json"

        # Resolve curriculum datasets using shared helper function
        hf_dataset_name, curriculum_datasets = resolve_curriculum_to_datasets(
            payload.curriculum_id, curricula
        )
        if hf_dataset_name:
            print(f"[HPO] Using curriculum dataset: {hf_dataset_name}")

        # Parse learning rate
        try:
            lr_value = float(payload.lr_max) if payload.lr_max else 1e-4
        except (ValueError, TypeError):
            lr_value = 1e-4

        # Build model config for executor
        model_config = {
            "sweep_id": sweep_id,
            # Core architecture
            "vocab_size": payload.vocab_size,
            "hidden_dim": payload.embedding_dim or 512,
            "num_reasoning_blocks": payload.num_reasoning_blocks or 8,
            "num_moe_experts": payload.num_moe_experts or 8,
            "sequence_length": min(payload.context_window or 512, 512),
            "batch_size": payload.batch_sizes[0] if payload.batch_sizes else 8,
            "learning_rate": lr_value,
            "optimizer": payload.optimizers[0] if payload.optimizers else "sophiag",
            "param_budget": payload.param_budget,
            # Mamba2 SSM parameters
            "mamba_state_dim": payload.mamba_state_dim,
            "mamba_conv_dim": payload.mamba_conv_dim,
            "mamba_expand": payload.mamba_expand,
            # WLAM parameters
            "wlam_num_heads": payload.wlam_num_heads,
            "wlam_kernel_size": payload.wlam_kernel_size,
            "wlam_num_landmarks": payload.wlam_num_landmarks,
            # MoE parameters
            "moe_top_k": payload.moe_top_k,
            "moe_capacity_factor": payload.moe_capacity_factor,
            # FFN and TT
            "ff_expansion": payload.ff_expansion,
            "tt_rank_middle": payload.tt_rank_middle,
            # Quantum/superposition
            "superposition_dim": payload.superposition_dim,
            "hamiltonian_hidden_dim": payload.hamiltonian_hidden_dim,
            # Regularization
            "dropout_rate": payload.dropout_rate,
            "weight_decay": payload.weight_decay,
            # Dataset configuration
            "hf_dataset_name": hf_dataset_name,
            "curriculum_id": payload.curriculum_id,
            "curriculum_datasets": curriculum_datasets,
            # Quantum Enhancement Parameters
            "use_quantum_embedding": payload.use_quantum_embedding,
            "use_floquet_position": payload.use_floquet_position,
            "use_quantum_feature_maps": payload.use_quantum_feature_maps,
            "use_unitary_expert": payload.use_unitary_expert,
            "neumann_cayley_terms": payload.neumann_cayley_terms,
            "use_quantum_norm": payload.use_quantum_norm,
            "use_superposition_bpe": payload.use_superposition_bpe,
            "use_grover_qsg": payload.use_grover_qsg,
            "qsg_quality_threshold": payload.qsg_quality_threshold,
            "use_quantum_lm_head": payload.use_quantum_lm_head,
            "use_unitary_residual": payload.use_unitary_residual,
            "unitary_residual_init_angle": payload.unitary_residual_init_angle,
            "use_quantum_state_bus": payload.use_quantum_state_bus,
        }

        # Save config for reference
        with open(config_file, "w") as f:
            json.dump(model_config, f, indent=2)

        # Add initial log entry
        hpo_logs[sweep_id] = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": f"Starting HPO sweep {sweep_id} with {payload.max_trials} trials",
                "step": None,
                "loss": None,
            }
        ]

        # Create SweepConfig for executor
        sweep_config = SweepConfig(
            max_trials=payload.max_trials,
            max_parallel=1,  # Sequential execution for now
            epochs_per_trial=payload.epochs_per_trial or 3,
            steps_per_epoch=50,
            search_strategy=payload.search_strategy or "random",
            use_optuna=payload.use_optuna,
            hyperband_eta=payload.hyperband_eta or 3,
            param_budget=payload.param_budget or 1_000_000_000,
            sweep_id=sweep_id,
            model_config=model_config,
        )

        # Create scheduler based on strategy
        scheduler = create_scheduler(
            strategy=payload.search_strategy or "random",
            config=sweep_config,
        )

        # Create log callback to update hpo_logs
        def log_callback(entry: dict[str, Any]) -> None:
            if sweep_id in hpo_logs:
                hpo_logs[sweep_id].append(entry)
                # Keep log buffer bounded
                if len(hpo_logs[sweep_id]) > 1000:
                    hpo_logs[sweep_id] = hpo_logs[sweep_id][-1000:]

        # Create SweepExecutor (THE FIX - this runs ALL trials!)
        executor = SweepExecutor(
            sweep_id=sweep_id,
            config=sweep_config,
            scheduler=scheduler,
            max_parallel=1,
            storage_path=str(config_dir / sweep_id),
            log_callback=log_callback,
        )

        # Store executor for status queries
        active_sweeps[sweep_id] = executor

        # Background task to run the full sweep
        async def run_sweep_background():
            """Run complete HPO sweep with all trials."""
            try:
                result = await executor.run()

                # Update sweep state
                sweep["state"] = result.state
                sweep["completed_trials"] = len(result.completed_trials)
                sweep["best_trial_id"] = result.best_trial_id
                sweep["best_loss"] = result.best_loss
                sweep["best_hyperparams"] = result.best_hyperparams
                sweep["trials"] = [t.to_dict() for t in result.completed_trials]
                # Multi-objective best scores
                if result.best_trial:
                    sweep["best_composite_score"] = result.best_trial.composite_score
                    sweep["best_perplexity"] = result.best_trial.perplexity

                hpo_logs[sweep_id].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "level": "INFO",
                        "message": f"HPO sweep completed: {len(result.completed_trials)} trials, "
                        f"best loss: {result.best_loss:.6f if result.best_loss else 'N/A'}",
                    }
                )

            except Exception as e:
                sweep["state"] = "failed"
                hpo_logs[sweep_id].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "level": "ERROR",
                        "message": f"Sweep failed: {str(e)}",
                    }
                )

            finally:
                # Cleanup executor
                active_sweeps.pop(sweep_id, None)

        # Start the sweep in background (don't await)
        asyncio.create_task(run_sweep_background())

        return {
            "status": "started",
            "sweep_id": sweep_id,
            "stage": payload.stage.value,
            "max_trials": payload.max_trials,
            "config_file": str(config_file),
        }

    @app.get("/api/hpo/sweep/{sweep_id}/status")
    async def get_hpo_sweep_status(sweep_id: str):
        """Get HPO sweep status with real-time progress from executor."""
        sweep = hpo_sweeps.get(sweep_id)
        if not sweep:
            raise HTTPException(status_code=404, detail="HPO sweep not found")

        # Get live progress from active executor if running
        executor = active_sweeps.get(sweep_id)
        if executor:
            completed = len(executor.completed_trials)
            best_loss = executor.best_trial.loss if executor.best_trial else None
            best_trial_id = executor.best_trial.trial_id if executor.best_trial else None
            best_composite_score = (
                executor.best_trial.composite_score if executor.best_trial else None
            )
            best_perplexity = executor.best_trial.perplexity if executor.best_trial else None
        else:
            completed = sweep["completed_trials"]
            best_loss = sweep.get("best_loss")
            best_trial_id = sweep.get("best_trial_id")
            best_composite_score = sweep.get("best_composite_score")
            best_perplexity = sweep.get("best_perplexity")

        return HPOSweepInfo(
            sweep_id=sweep["sweep_id"],
            stage=HPOStage(sweep["stage"]),
            state=sweep["state"],
            max_trials=sweep["max_trials"],
            completed_trials=completed,
            best_trial_id=best_trial_id,
            best_loss=best_loss,
            best_composite_score=best_composite_score,
            best_perplexity=best_perplexity,
            best_hyperparams=sweep.get("best_hyperparams"),
            started_at=sweep.get("started_at"),
            trials=[],  # Exclude trials for status endpoint
        )

    @app.get("/api/hpo/sweep/{sweep_id}/trials")
    async def get_hpo_trials(sweep_id: str):
        """Get all trials for an HPO sweep."""
        sweep = hpo_sweeps.get(sweep_id)
        if not sweep:
            raise HTTPException(status_code=404, detail="HPO sweep not found")

        # Get live trials from executor if running
        executor = active_sweeps.get(sweep_id)
        if executor:
            trials = [t.to_dict() for t in executor.completed_trials]
        else:
            trials = sweep.get("trials", [])

        return {"trials": trials}

    @app.get("/api/hpo/sweep/{sweep_id}/skipped")
    async def get_hpo_skipped_trials(sweep_id: str):
        """Get trials that were skipped due to parameter budget constraints.

        Returns list of skipped trial records with:
        - trial_id: ID that would have been assigned
        - reason: Why it was skipped (e.g., 'exceeded_budget_after_50_attempts')
        - estimated_params: Estimated parameter count
        - param_budget: The configured budget
        - architecture_signature: Architecture pattern for clustering
        """
        sweep = hpo_sweeps.get(sweep_id)
        if not sweep:
            raise HTTPException(status_code=404, detail="HPO sweep not found")

        # Get from executor if running
        executor = active_sweeps.get(sweep_id)
        if executor and hasattr(executor, "smart_sampler") and executor.smart_sampler:
            skipped = executor.smart_sampler.get_skipped_for_webui()
            stats = executor.smart_sampler.get_statistics()
        else:
            # Check for persisted skipped records
            from highnoon.services.hpo_metrics import OversizedConfigTracker

            tracker = OversizedConfigTracker(
                param_budget=sweep.get("model_config", {}).get("param_budget", 1_000_000_000),
                hpo_root=Path("artifacts/hpo_trials"),
            )
            skipped = [r.to_dict() for r in tracker.get_skipped_records()]
            stats = tracker.get_statistics()

        return {
            "skipped_trials": skipped,
            "statistics": stats,
        }

    @app.get("/api/hpo/sweep/{sweep_id}/budget")
    async def get_hpo_budget_stats(sweep_id: str):
        """Get parameter budget enforcement statistics.

        Returns:
        - param_budget: The configured budget
        - total_skipped: Number of trials skipped
        - skip_rate: Percentage of trials skipped
        - safe_bounds: Recommended safe architecture bounds
        - top_failing_architectures: Most common failing patterns
        """
        sweep = hpo_sweeps.get(sweep_id)
        if not sweep:
            raise HTTPException(status_code=404, detail="HPO sweep not found")

        # Get budget from sweep config
        param_budget = sweep.get("model_config", {}).get("param_budget", 1_000_000_000)

        # Get from executor if running
        executor = active_sweeps.get(sweep_id)
        if executor and hasattr(executor, "smart_sampler") and executor.smart_sampler:
            stats = executor.smart_sampler.get_statistics()
        else:
            from highnoon.services.hpo_metrics import OversizedConfigTracker

            tracker = OversizedConfigTracker(
                param_budget=param_budget,
                hpo_root=Path("artifacts/hpo_trials"),
            )
            stats = tracker.get_statistics()
            stats["enabled"] = True
            stats["param_budget"] = param_budget

        return stats

    @app.get("/api/hpo/sweep/{sweep_id}/best")
    async def get_hpo_best(sweep_id: str):
        """Get best hyperparameters from an HPO sweep."""
        sweep = hpo_sweeps.get(sweep_id)
        if not sweep:
            raise HTTPException(status_code=404, detail="HPO sweep not found")

        # Check executor for live best
        executor = active_sweeps.get(sweep_id)
        if executor and executor.best_trial:
            return {
                "best_trial_id": executor.best_trial.trial_id,
                "best_loss": executor.best_trial.loss,
                "best_hyperparams": executor.best_trial.hyperparams,
            }

        if not sweep.get("best_hyperparams"):
            return {"status": "no_results", "message": "No completed trials yet"}
        return {
            "best_trial_id": sweep["best_trial_id"],
            "best_loss": sweep["best_loss"],
            "best_hyperparams": sweep["best_hyperparams"],
        }

    @app.post("/api/hpo/sweep/{sweep_id}/cancel")
    async def cancel_hpo_sweep(sweep_id: str):
        """Cancel an HPO sweep and stop the executor."""
        sweep = hpo_sweeps.get(sweep_id)
        if not sweep:
            raise HTTPException(status_code=404, detail="HPO sweep not found")
        if sweep["state"] in ["completed", "cancelled"]:
            raise HTTPException(status_code=400, detail="Sweep already finished")

        # Actually cancel the executor if running
        executor = active_sweeps.get(sweep_id)
        if executor:
            await executor.cancel()
            active_sweeps.pop(sweep_id, None)

        sweep["state"] = "cancelled"
        return {"status": "cancelled", "sweep_id": sweep_id}

    @app.get("/api/hpo/sweep/{sweep_id}/pareto")
    async def get_hpo_pareto_frontier(sweep_id: str):
        """Get Pareto frontier trials from an HPO sweep."""
        sweep = hpo_sweeps.get(sweep_id)
        if not sweep:
            raise HTTPException(status_code=404, detail="HPO sweep not found")

        # Get from executor if running
        executor = active_sweeps.get(sweep_id)
        if executor:
            pareto = [t.to_dict() for t in executor.scorer.pareto_frontier]
        else:
            # Fall back to trials marked as on frontier
            pareto = [t for t in sweep.get("trials", []) if t.get("is_on_pareto_frontier")]

        return {"pareto_frontier": pareto}

    @app.get("/api/hpo/sweep/{sweep_id}/metrics")
    async def stream_hpo_metrics(sweep_id: str, request: Request):
        """Stream HPO metrics in real-time via Server-Sent Events.

        This endpoint provides live updates on sweep progress including:
        - Trial completions
        - Current best trial
        - Running trial count
        - Completed trial count
        """
        from starlette.responses import StreamingResponse

        sweep = hpo_sweeps.get(sweep_id)
        if not sweep:
            raise HTTPException(status_code=404, detail="HPO sweep not found")

        async def event_generator():
            """Generate SSE events for sweep progress."""
            last_completed = 0
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                executor = active_sweeps.get(sweep_id)
                if executor:
                    completed = len(executor.completed_trials)
                    running = len(executor.running_trials)
                    best_loss = executor.best_trial.loss if executor.best_trial else None
                    best_id = executor.best_trial.trial_id if executor.best_trial else None
                    state = "running"
                else:
                    completed = sweep.get("completed_trials", 0)
                    running = 0
                    best_loss = sweep.get("best_loss")
                    best_id = sweep.get("best_trial_id")
                    state = sweep.get("state", "unknown")

                # Send update if there's new data or periodically
                if completed != last_completed or state != "running":
                    data = {
                        "sweep_id": sweep_id,
                        "state": state,
                        "completed_trials": completed,
                        "running_trials": running,
                        "max_trials": sweep.get("max_trials", 0),
                        "best_trial_id": best_id,
                        "best_loss": best_loss,
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    last_completed = completed

                    # Stop streaming if sweep is complete
                    if state in ["completed", "cancelled", "failed"]:
                        break

                await asyncio.sleep(1)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/hpo/sweep/{sweep_id}/resume")
    async def resume_hpo_sweep(sweep_id: str, background_tasks: BackgroundTasks):
        """Resume a cancelled or crashed HPO sweep from checkpoint.

        This endpoint restores sweep state from the last checkpoint
        and continues running remaining trials.
        """
        sweep = hpo_sweeps.get(sweep_id)
        if not sweep:
            raise HTTPException(status_code=404, detail="HPO sweep not found")

        if sweep_id in active_sweeps:
            raise HTTPException(status_code=400, detail="Sweep is already running")

        if sweep.get("state") == "completed":
            raise HTTPException(status_code=400, detail="Sweep already completed")

        # Try to restore from checkpoint
        storage_path = Path(f"artifacts/hpo_trials/{sweep_id}")
        checkpoint_file = storage_path / f"{sweep_id}_checkpoint.json"

        if not checkpoint_file.exists():
            raise HTTPException(status_code=400, detail="No checkpoint found for this sweep")

        # Update sweep state
        sweep["state"] = "resuming"

        async def resume_sweep_background():
            """Resume sweep in background."""
            try:
                # Create new executor with checkpoint restore
                sweep_config = SweepConfig(
                    max_trials=sweep.get("max_trials", 50),
                    max_parallel=1,
                    sweep_id=sweep_id,
                )
                scheduler = create_scheduler(
                    sweep.get("search_strategy", "random"),
                    sweep_config,
                )
                executor = SweepExecutor(
                    sweep_id=sweep_id,
                    config=sweep_config,
                    scheduler=scheduler,
                    storage_path=str(storage_path),
                )
                active_sweeps[sweep_id] = executor

                sweep["state"] = "running"
                result = await executor.run()

                # Update sweep state on completion
                sweep["state"] = result.state
                sweep["completed_trials"] = len(result.completed_trials)
                if result.best_trial:
                    sweep["best_trial_id"] = result.best_trial.trial_id
                    sweep["best_loss"] = result.best_trial.loss
                    sweep["best_hyperparams"] = result.best_trial.hyperparams
                sweep["trials"] = [t.to_dict() for t in result.completed_trials]

                active_sweeps.pop(sweep_id, None)
                logger.info(f"[HPO] Resumed sweep {sweep_id} completed")

            except Exception as e:
                sweep["state"] = "failed"
                sweep["error"] = str(e)
                active_sweeps.pop(sweep_id, None)
                logger.error(f"[HPO] Resume sweep {sweep_id} failed: {e}")

        background_tasks.add_task(resume_sweep_background)

        return {
            "status": "resuming",
            "sweep_id": sweep_id,
            "message": "Sweep is being resumed from checkpoint",
        }

    @app.get("/api/hpo/sweeps")
    async def list_hpo_sweeps():
        """List all HPO sweeps."""
        return {"sweeps": list(hpo_sweeps.values())}

    # ========================================================================
    # HPO Log Streaming - Real-time training logs
    # ========================================================================

    # In-memory log storage per sweep (limited to last 1000 entries)
    hpo_logs: dict[str, list[dict[str, Any]]] = {}
    MAX_LOG_ENTRIES = 1000

    @app.post("/api/hpo/sweep/{sweep_id}/log")
    async def add_hpo_log(sweep_id: str, log_entry: dict[str, Any]):
        """Add a log entry for an HPO sweep (called by trial runner)."""
        if sweep_id not in hpo_logs:
            hpo_logs[sweep_id] = []

        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": log_entry.get("level", "INFO"),
            "message": log_entry.get("message", ""),
            "step": log_entry.get("step"),
            "loss": log_entry.get("loss"),
            "gradient_norm": log_entry.get("gradient_norm"),
            "learning_rate": log_entry.get("learning_rate"),
            "epoch": log_entry.get("epoch"),
            "trial_id": log_entry.get("trial_id"),
            # Memory metrics
            "memory_mb": log_entry.get("memory_mb"),
            "peak_memory_mb": log_entry.get("peak_memory_mb"),
            # Quality metrics (sent on trial completion)
            "perplexity": log_entry.get("perplexity"),
            "mean_confidence": log_entry.get("mean_confidence"),
            "composite_score": log_entry.get("composite_score"),
            "efficiency_score": log_entry.get("efficiency_score"),
            "param_count": log_entry.get("param_count"),
        }
        hpo_logs[sweep_id].append(entry)

        # Keep only last MAX_LOG_ENTRIES
        if len(hpo_logs[sweep_id]) > MAX_LOG_ENTRIES:
            hpo_logs[sweep_id] = hpo_logs[sweep_id][-MAX_LOG_ENTRIES:]

        return {"status": "ok"}

    @app.get("/api/hpo/sweep/{sweep_id}/logs")
    async def get_hpo_logs(sweep_id: str, since_index: int = 0, limit: int = 100):
        """Get logs for an HPO sweep starting from a specific index."""
        logs = hpo_logs.get(sweep_id, [])

        # Return logs starting from since_index
        result_logs = logs[since_index : since_index + limit]

        # Sanitize float values to ensure JSON compliance (NaN/Inf -> None)
        def sanitize_value(val: Any) -> Any:
            """Convert NaN/Inf floats to None for JSON serialization."""
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
            elif isinstance(val, dict):
                return {k: sanitize_value(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [sanitize_value(item) for item in val]
            return val

        sanitized_logs = [sanitize_value(log) for log in result_logs]

        return {
            "logs": sanitized_logs,
            "total": len(logs),
            "next_index": min(since_index + limit, len(logs)),
        }

    @app.delete("/api/hpo/sweep/{sweep_id}/logs")
    async def clear_hpo_logs(sweep_id: str):
        """Clear logs for an HPO sweep."""
        if sweep_id in hpo_logs:
            hpo_logs[sweep_id] = []
        return {"status": "cleared"}

    # ========================================================================
    # API Routes - Distributed Training
    # ========================================================================

    from .distributed_manager import ClusterRole, get_distributed_manager

    # Get the global distributed manager instance
    dist_manager = get_distributed_manager()

    @app.get("/api/distributed/status")
    async def get_distributed_status() -> ClusterStatusResponse:
        """Get current distributed cluster status.

        Returns the current role, connected workers, and cluster configuration.
        """
        status = dist_manager.get_cluster_status()
        return ClusterStatusResponse(
            role=status.role.value if hasattr(status.role, "value") else status.role,
            cluster_secret=status.cluster_secret,
            workers=[
                WorkerInfoResponse(
                    worker_id=w.worker_id,
                    hostname=w.hostname,
                    address=w.address,
                    status=w.status.value if hasattr(w.status, "value") else w.status,
                    cpu_count=w.cpu_count,
                    memory_gb=w.memory_gb,
                    connected_at=w.connected_at,
                    last_heartbeat=w.last_heartbeat,
                    task_index=w.task_index,
                )
                for w in status.workers
            ],
            is_ready=status.is_ready,
            tf_config=status.tf_config,
            host_address=status.host_address,
            is_training=status.is_training,
            error=status.error,
        )

    @app.post("/api/distributed/start-host")
    async def start_distributed_host(payload: StartHostRequest):
        """Start this node as a distributed training Host.

        The Host coordinates the cluster, accepts worker connections,
        and generates TF_CONFIG for distributed TensorFlow training.

        Returns the cluster secret that workers need to join.
        """
        try:
            cluster_secret = await dist_manager.start_host(
                port=payload.port,
                cluster_secret=payload.cluster_secret,
                shared_checkpoint_dir=payload.shared_checkpoint_dir,
                communication_protocol=payload.communication_protocol,
            )
            return {
                "status": "hosting",
                "cluster_secret": cluster_secret,
                "port": payload.port,
                "message": f"Host started on port {payload.port}. Workers can join with the cluster secret.",
            }
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start host: {e}")

    @app.post("/api/distributed/join-cluster")
    async def join_distributed_cluster(payload: JoinClusterRequest):
        """Join an existing distributed training cluster as a Worker.

        Connects to the Host and receives TF_CONFIG for training.
        """
        try:
            await dist_manager.join_cluster(
                host_address=payload.host_address,
                cluster_secret=payload.cluster_secret,
            )
            status = dist_manager.get_cluster_status()
            return {
                "status": "joined",
                "host_address": payload.host_address,
                "tf_config": status.tf_config,
                "message": "Successfully joined the cluster.",
            }
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))
        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to join cluster: {e}")

    @app.post("/api/distributed/disconnect")
    async def disconnect_from_cluster():
        """Disconnect from the cluster (Worker) or stop hosting (Host).

        Returns this node to standalone mode.
        """
        try:
            await dist_manager.disconnect()
            return {
                "status": "disconnected",
                "message": "Returned to standalone mode.",
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to disconnect: {e}")

    @app.get("/api/distributed/workers")
    async def list_distributed_workers():
        """List all connected workers (Host only).

        Returns worker details including hostname, resources, and status.
        """
        if dist_manager.role != ClusterRole.HOST:
            raise HTTPException(
                status_code=400, detail="Not in host mode. Only the host can list workers."
            )

        status = dist_manager.get_cluster_status()
        return {
            "workers": [w.to_dict() for w in status.workers],
            "total": len(status.workers),
            "connected": sum(
                1
                for w in status.workers
                if (w.status.value if hasattr(w.status, "value") else w.status) == "connected"
            ),
        }

    @app.delete("/api/distributed/workers/{worker_id}")
    async def remove_distributed_worker(worker_id: str):
        """Remove a worker from the cluster (Host only)."""
        if dist_manager.role != ClusterRole.HOST:
            raise HTTPException(
                status_code=400, detail="Not in host mode. Only the host can remove workers."
            )

        success = await dist_manager.remove_worker(worker_id)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Worker '{worker_id}' not found or cannot be removed."
            )

        return {"status": "removed", "worker_id": worker_id}

    @app.post("/api/distributed/start-training")
    async def start_distributed_training(payload: StartDistributedTrainingRequest):
        """Start distributed training across all connected workers (Host only).

        Broadcasts the training configuration to all workers and coordinates
        the training start.
        """
        if dist_manager.role != ClusterRole.HOST:
            raise HTTPException(
                status_code=400,
                detail="Not in host mode. Only the host can start distributed training.",
            )

        if not dist_manager._is_cluster_ready():
            raise HTTPException(
                status_code=400,
                detail="Cluster is not ready. Ensure at least one worker is connected.",
            )

        success = await dist_manager.start_distributed_training(
            training_config=payload.training_config,
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to start distributed training.")

        return {
            "status": "started",
            "workers": len(dist_manager._workers),
            "tf_config": dist_manager._tf_config,
            "message": f"Distributed training started across {len(dist_manager._workers)} workers.",
        }

    @app.post("/api/distributed/stop-training")
    async def stop_distributed_training():
        """Stop distributed training on all workers (Host only)."""
        if dist_manager.role != ClusterRole.HOST:
            raise HTTPException(
                status_code=400,
                detail="Not in host mode. Only the host can stop distributed training.",
            )

        await dist_manager.stop_distributed_training()
        return {"status": "stopped", "message": "Distributed training stopped."}

    @app.get("/api/distributed/system-info")
    async def get_system_info():
        """Get system information for this node.

        Returns CPU count, memory, hostname, and local IP.
        """
        from .distributed_manager import _get_system_info

        info = _get_system_info()
        return {
            "hostname": info["hostname"],
            "cpu_count": info["cpu_count"],
            "memory_gb": info["memory_gb"],
            "local_ip": info["local_ip"],
        }

    @app.websocket("/ws/distributed")
    async def distributed_websocket(websocket: WebSocket):
        """WebSocket endpoint for real-time cluster updates.

        Streams cluster status changes, worker joins/leaves, and training metrics.
        """
        await websocket.accept()

        try:
            last_status = None
            while True:
                status = dist_manager.get_cluster_status()
                status_dict = status.to_dict()

                # Only send if status changed
                if status_dict != last_status:
                    await websocket.send_json(
                        {
                            "type": "cluster_status",
                            "data": status_dict,
                        }
                    )
                    last_status = status_dict

                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    # ========================================================================
    # Dev Mode Detection
    # ========================================================================

    def check_debug_build() -> dict[str, Any]:
        """Check if running with debug build (no symbol stripping)."""
        import os
        import subprocess

        dev_mode = False
        debug_build = False
        native_lib_info = {}

        # Check environment variable
        if os.environ.get("HIGHNOON_DEV_MODE", "").lower() in ("1", "true", "yes"):
            dev_mode = True

        # Check if Flask/FastAPI debug mode
        if debug:  # From create_app parameter
            dev_mode = True

        # Check native library for debug symbols
        native_lib = (
            Path(__file__).parent.parent / "_native" / "bin" / "x86_64" / "_highnoon_core.so"
        )
        if native_lib.exists():
            try:
                # Check for debug sections
                result = subprocess.run(
                    ["readelf", "-S", str(native_lib)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if ".debug" in result.stdout:
                    debug_build = True
                    dev_mode = True

                # Check for symbols
                result = subprocess.run(
                    ["nm", str(native_lib)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                symbol_count = (
                    len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
                )
                native_lib_info = {
                    "path": str(native_lib),
                    "has_debug_sections": debug_build,
                    "symbol_count": symbol_count,
                }
            except (subprocess.TimeoutExpired, FileNotFoundError):
                native_lib_info = {
                    "path": str(native_lib),
                    "check_error": "readelf/nm not available",
                }
        else:
            native_lib_info = {"path": str(native_lib), "exists": False}

        return {
            "dev_mode": dev_mode,
            "debug_build": debug_build,
            "native_lib": native_lib_info,
            "environment": {
                "HIGHNOON_DEV_MODE": os.environ.get("HIGHNOON_DEV_MODE", ""),
                "flask_debug": debug,
            },
        }

    @app.get("/api/dev/status")
    async def get_dev_status():
        """Check if running in dev mode (debug build with verbose logging)."""
        return check_debug_build()

    @app.get("/api/dev/verbose-logs")
    async def get_verbose_logs_enabled():
        """Check if verbose logging is enabled."""
        status = check_debug_build()
        return {
            "verbose_logs_enabled": status["dev_mode"] or status["debug_build"],
            "reason": (
                "Debug build detected"
                if status["debug_build"]
                else ("HIGHNOON_DEV_MODE=1" if status["dev_mode"] else "Production mode")
            ),
        }

    # ========================================================================
    # WebSocket - Real-time Training Updates
    # ========================================================================

    @app.websocket("/ws/training/{job_id}")
    async def training_websocket(websocket: WebSocket, job_id: str):
        """WebSocket endpoint for real-time training metrics."""
        await websocket.accept()

        if job_id not in active_websockets:
            active_websockets[job_id] = []
        active_websockets[job_id].append(websocket)

        try:
            while True:
                job = training_jobs.get(job_id)
                if not job:
                    await websocket.send_json({"error": "Job not found"})
                    break

                # Send current metrics
                await websocket.send_json(
                    {
                        "type": "metrics",
                        "data": {
                            "state": job["state"],
                            "global_step": job["global_step"],
                            "current_epoch": job["current_epoch"],
                            "loss": job["loss"],
                            "learning_rate": job["learning_rate"],
                            "throughput": job["throughput"],
                            "progress_percent": job["progress_percent"],
                            "hpo_trial_current": job.get("hpo_trial_current", 0),
                            "hpo_trial_total": job.get("hpo_trial_total", 0),
                        },
                    }
                )

                # Check if job finished
                if job["state"] in [
                    TrainingState.COMPLETED.value,
                    TrainingState.CANCELLED.value,
                    TrainingState.FAILED.value,
                ]:
                    await websocket.send_json({"type": "finished", "state": job["state"]})
                    break

                await asyncio.sleep(1)  # Update every second
        except WebSocketDisconnect:
            pass
        finally:
            if job_id in active_websockets:
                active_websockets[job_id].remove(websocket)

    @app.websocket("/ws/hpo/{sweep_id}")
    async def hpo_websocket(websocket: WebSocket, sweep_id: str):
        """WebSocket endpoint for real-time HPO trial updates."""
        await websocket.accept()

        try:
            while True:
                sweep = hpo_sweeps.get(sweep_id)
                if not sweep:
                    await websocket.send_json({"error": "Sweep not found"})
                    break

                await websocket.send_json(
                    {
                        "type": "sweep_update",
                        "data": {
                            "state": sweep["state"],
                            "completed_trials": sweep["completed_trials"],
                            "max_trials": sweep["max_trials"],
                            "best_loss": sweep.get("best_loss"),
                            "best_trial_id": sweep.get("best_trial_id"),
                        },
                    }
                )

                if sweep["state"] in ["completed", "cancelled"]:
                    await websocket.send_json({"type": "finished", "state": sweep["state"]})
                    break

                await asyncio.sleep(2)
        except WebSocketDisconnect:
            pass

    # ========================================================================
    # Attribution Configuration (Pro/Enterprise Feature)
    # ========================================================================

    @app.get("/api/attribution", response_model=AttributionResponse)
    async def get_attribution():
        """Get current attribution configuration and edition info.

        Returns the effective attribution values. For Pro/Enterprise editions
        with custom attribution set, returns the custom values. Otherwise,
        returns the default Verso Industries attribution.

        Returns:
            AttributionResponse with current attribution and edition info.
        """
        try:
            from highnoon import attribution as attr

            current = attr.get_current_attribution()
            return AttributionResponse(
                attribution=AttributionConfig(
                    framework_name=current.framework_name,
                    author=current.author,
                    copyright_notice=current.copyright_notice,
                    version=current.version,
                    support_url=current.support_url,
                ),
                edition=current.edition,
                edition_code=current.edition_code,
                is_customizable=current.is_customizable,
                is_custom=current.is_custom,
            )
        except ImportError:
            # Fallback if attribution module not available
            return AttributionResponse(
                attribution=AttributionConfig(),
                edition="Lite",
                edition_code=0,
                is_customizable=False,
                is_custom=False,
            )

    @app.put("/api/attribution", response_model=AttributionResponse)
    async def update_attribution(config: AttributionConfig):
        """Update custom attribution configuration.

        Only functional in Pro and Enterprise editions. Lite edition will
        return a 403 Forbidden error.

        Args:
            config: The new attribution configuration.

        Returns:
            Updated AttributionResponse.

        Raises:
            HTTPException 403: If custom attribution is not allowed (Lite edition).
            HTTPException 400: If validation fails (empty framework name).
        """
        try:
            from highnoon import attribution as attr

            # Check if customization is allowed
            if not attr.is_custom_attribution_allowed():
                raise HTTPException(
                    status_code=403,
                    detail="Custom attribution requires Pro or Enterprise edition. "
                    "See https://versoindustries.com/upgrade",
                )

            # Set custom attribution
            success = attr.set_custom_attribution(
                framework_name=config.framework_name,
                author=config.author,
                copyright_notice=config.copyright_notice,
                version=config.version,
                support_url=config.support_url,
            )

            if not success:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to set custom attribution. Framework name is required.",
                )

            # Return updated attribution
            current = attr.get_current_attribution()
            return AttributionResponse(
                attribution=AttributionConfig(
                    framework_name=current.framework_name,
                    author=current.author,
                    copyright_notice=current.copyright_notice,
                    version=current.version,
                    support_url=current.support_url,
                ),
                edition=current.edition,
                edition_code=current.edition_code,
                is_customizable=current.is_customizable,
                is_custom=current.is_custom,
            )
        except ImportError:
            raise HTTPException(status_code=500, detail="Attribution module not available")

    @app.post("/api/attribution/reset", response_model=AttributionResponse)
    async def reset_attribution():
        """Reset attribution to default Verso Industries values.

        Clears any custom attribution and reverts to the default attribution.
        For Lite edition, this is a no-op since custom attribution is never set.

        Returns:
            AttributionResponse with default attribution values.
        """
        try:
            from highnoon import attribution as attr

            attr.clear_custom_attribution()

            current = attr.get_current_attribution()
            return AttributionResponse(
                attribution=AttributionConfig(
                    framework_name=current.framework_name,
                    author=current.author,
                    copyright_notice=current.copyright_notice,
                    version=current.version,
                    support_url=current.support_url,
                ),
                edition=current.edition,
                edition_code=current.edition_code,
                is_customizable=current.is_customizable,
                is_custom=current.is_custom,
            )
        except ImportError:
            return AttributionResponse(
                attribution=AttributionConfig(),
                edition="Lite",
                edition_code=0,
                is_customizable=False,
                is_custom=False,
            )

    # ========================================================================
    # Health Check
    # ========================================================================

    @app.get("/health")
    async def health_status():
        """Health check endpoint."""
        return {"status": "healthy", "version": "1.0.0"}

    return app


# ============================================================================
# Development Server
# ============================================================================


def run_dev_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the development server.

    Args:
        host: Host address to bind.
        port: Port number.
    """
    import uvicorn

    app = create_app(debug=True)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dev_server()
