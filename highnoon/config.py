# highnoon/config.py
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

"""Configuration system for HighNoon Language Framework.

This module defines the configuration dataclasses used throughout the framework.
The Lite edition enforces hard limits on model scale, which are validated both
here in Python and in the compiled C++ binaries.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# =============================================================================
# LITE EDITION LIMITS
# =============================================================================
# These limits are enforced at the Python level for early validation with
# helpful error messages. They are also hard-coded in the compiled C++ binaries
# and cannot be bypassed.

LITE_MAX_PARAMS = 20_000_000_000  # 20B parameters
LITE_MAX_REASONING_BLOCKS = 24  # Maximum reasoning depth
LITE_MAX_MOE_EXPERTS = 12  # Maximum MoE experts
LITE_MAX_CONTEXT_LENGTH = 5_000_000  # 5M tokens
LITE_MAX_EMBEDDING_DIM = 4096  # Maximum embedding dimension

# =============================================================================
# SUPERPOSITION PARAMETERS (Used by C++ ops)
# =============================================================================
SUPERPOSITION_DIM = 4  # Default superposition dimension
LITE_MAX_SUPERPOSITION_DIM = 2  # Maximum superposition dimension for Lite edition
SUPERPOSITION_MICRO_BATCH_SIZE = 32  # Micro-batch size for superposition ops

# =============================================================================
# MAMBA SSM PARAMETERS (Spatial Blocks)
# =============================================================================
MAMBA2_STATE_DIM = 16  # State dimension for Mamba SSM
MAMBA2_CONV_DIM = 4  # Conv dimension for Mamba
MAMBA2_HEAD_DIM = 64  # Head dimension for Mamba-2
MAMBA2_EXPAND_FACTOR = 2  # Expansion factor for inner dimension

# =============================================================================
# WLAM (Wavelet Attention) PARAMETERS
# =============================================================================
WLAM_NUM_HEADS = 8  # Number of heads for wavelet attention
WLAM_WAVELET_KERNEL_SIZE = 5  # Kernel size for wavelet transform
WLAM_USE_FLASH_LINEAR: bool = True  # Use FlashLinearAttention for low-freq (5-8x speedup)

# =============================================================================
# REASONING MODULE PARAMETERS
# =============================================================================
REASONING_BLOCK_PATTERN = "mamba_timecrystal_wlam_moe_hybrid"  # Block pattern
REASONING_HEADS = 8  # Number of reasoning attention heads
REASONING_FF_DIM = 2048  # Feedforward dimension
NUM_EXPERTS = 8  # Number of MoE experts
TOP_K = 2  # Top-K experts to route
ADAPTER_DIM = 64  # Adapter layer dimension
MAX_HIERARCHY_LEVELS = 8  # Maximum memory hierarchy levels

# =============================================================================
# TENSOR TRAIN PARAMETERS
# =============================================================================
TT_LAYER_CONFIGS = {
    "default": {"tt_ranks": [1, 16, 1]},
    "small": {"tt_ranks": [1, 8, 1]},
    "large": {"tt_ranks": [1, 32, 1]},
}

# TT Integration Flags - Safe Layers Only (High Confidence, Low Quality Risk)
# These layers have been validated to tolerate TT decomposition well.
# All maintain O(d) complexity with 85-98% parameter reduction.

# LM Head / Output Projection (Very Low Risk)
# Compresses vocab_size × embedding_dim projection - largest parameter sink
USE_TT_LM_HEAD: bool = True  # Enable TT for LM head
TT_LM_HEAD_RANKS: list[int] = [1, 8, 8, 1]  # Ranks for 32K+ vocab

# Token Embeddings (Very Low Risk) - Only when tie_embeddings=True
USE_TT_EMBEDDINGS: bool = True  # Enable TT for token embeddings
TT_EMBEDDING_RANKS: list[int] = [1, 8, 8, 1]  # Ranks for 32K+ vocab

# FlashLinearAttention Q/K/V/O Projections (Low Risk)
# Linear attention already uses feature maps that smooth gradients
USE_TT_ATTENTION_PROJECTIONS: bool = True  # Enable TT for Q/K/V/O
TT_ATTENTION_RANKS: list[int] = [1, 4, 4, 1]  # Conservative ranks

# LatentReasoningBlock FFN Projections (Low Risk)
# Standard FFN layers widely shown to tolerate compression
USE_TT_FFN_PROJECTIONS: bool = True  # Enable TT for thought projectors
TT_FFN_RANKS: list[int] = [1, 8, 8, 1]  # Ranks for d→4d→d
TT_FFN_MIN_DIM: int = 512  # Only apply TT for dims >= this (Phase 201.2)

# KalmanBlock Projections (Low Risk)
# Input/output/state projections - not the Kalman filter parameters
USE_TT_KALMAN_PROJECTIONS: bool = True  # Enable TT for Kalman projections
TT_KALMAN_RANKS: list[int] = [1, 4, 1]  # Simple projections need fewer ranks

# =============================================================================
# DEBUG MODE
# =============================================================================
DEBUG_MODE = False  # Enable verbose logging and fallback ops

# =============================================================================
# META CONTROLLER PARAMETERS
# =============================================================================
USE_META_CONTROLLER: bool = True  # Master switch for quantum-enhanced meta-controller
META_CONTROLLER_FREQUENCY = 10  # Batches between meta-controller updates
META_CONTROLLER_SYSID_INTERVAL = 850  # Legacy: Full N4SID interval (kept for compatibility)

# =============================================================================
# UNIFIED SMART TUNER (Enterprise Training Parameter Controller)
# =============================================================================
# The Unified Smart Tuner replaces independent operation of QALRC,
# BarrenPlateauMonitor, GaLore, and Meta-Controller with a single orchestrated
# system. It provides cross-component awareness, global exploration/exploitation
# mode, and optional cross-trial memory for learning from previous HPO sweeps.
# See Smart_Tuner_Upgrade.md for full specification.

USE_UNIFIED_SMART_TUNER: bool = True  # Master switch for unified smart tuner
SMART_TUNER_MODE: str = "balanced"  # Coordination mode: "aggressive", "balanced", "conservative"
SMART_TUNER_WARMUP_STEPS: int = 1000  # Steps in warmup phase
SMART_TUNER_EXPLORATION_STEPS: int = 10000  # Steps in exploration phase
SMART_TUNER_MEMORY_ENABLED: bool = True  # Enable cross-trial memory learning
SMART_TUNER_EMERGENCY_GRAD_THRESHOLD: float = 1e6  # Gradient norm triggering emergency mode

# =============================================================================
# QUANTUM-ENHANCED CONTROL SYSTEM (Phase 2.1)
# =============================================================================
# These parameters control the quantum-enhanced control components that replace
# the legacy N4SID/Relay-based control system. All components are C++ implemented
# for performance and security. See QUANTUM_CONTROL_ENHANCEMENT_ROADMAP.md.

# RLS System Identifier (replaces 850-batch N4SID with O(n²) per-batch updates)
USE_RLS_SYSID: bool = True  # Enable fast RLS system identification
RLS_SYSID_INTERVAL: int = 5  # RLS update every N batches (vs 850 for N4SID)
RLS_FULL_SYSID_INTERVAL: int = 500  # Full N4SID reset every N batches (hybrid approach)
RLS_FORGETTING_FACTOR: float = 0.998  # RLS forgetting factor (0.95-0.999)
RLS_STATE_ORDER: int = 4  # ARX model order for RLS

# Hybrid PID Tuner (Relay/Ziegler-Nichols + Adam gradient descent)
USE_HYBRID_PID: bool = True  # Enable hybrid PID tuning
PID_LEARNING_RATE: float = 0.001  # Adam optimizer learning rate
PID_MAX_GAIN_CHANGE_RATE: float = 0.1  # Max gain change per step (rate limiting)
PID_KP_BOUNDS: tuple[float, float] = (0.001, 10.0)  # Kp safe range
PID_KI_BOUNDS: tuple[float, float] = (0.0, 2.0)  # Ki safe range
PID_KD_BOUNDS: tuple[float, float] = (0.0, 1.0)  # Kd safe range
PID_DIVERGENCE_THRESHOLD: float = 10.0  # Loss ratio triggering fallback to relay gains
PID_DIVERGENCE_WINDOW: int = 50  # Samples for divergence detection
PID_ZN_TYPE: str = (
    "some_overshoot"  # Ziegler-Nichols type: "classic", "some_overshoot", "no_overshoot", "pessen"
)

# Extended Kalman Filter (for nonlinear dynamics - 35-45% RMSE improvement)
USE_EXTENDED_KALMAN: bool = True  # Enable EKF for nonlinear state estimation
EKF_ADAPTIVE_NOISE: bool = True  # Enable adaptive Q/R estimation
EKF_ADAPTATION_RATE: float = 0.01  # Rate for adaptive noise learning

# Tensor Network Kalman Filter (O(n×r²) memory via TT decomposition)
USE_TENSOR_NETWORK_KALMAN: bool = True  # Enable TNKF for high-dimensional states
TNKF_MAX_RANK: int = 8  # Maximum TT rank for compression
TNKF_MIN_DIM_FOR_TT: int = 16  # Minimum state dim to use TT (else dense)
TNKF_ROUNDING_TOLERANCE: float = 1e-6  # TT rounding tolerance

# =============================================================================
# QUANTUM SUPERPOSITION GENERATION (QSG)
# =============================================================================
# QSG replaces autoregressive generation with parallel token generation using
# quantum-inspired mechanisms. Achieves 50-100x speedup while maintaining quality.
# See HOLOGRAPHIC_GENERATION_RESEARCH.md for algorithm details.

USE_QSG_GENERATION: bool = True  # Enable QSG for parallel generation
QSG_BOND_DIM: int = 32  # MPS bond dimension for context entanglement
QSG_COHERENCE_RANGE: int = 64  # Maximum position coherence distance (-1 = all)
QSG_GROVER_ITERATIONS: int = 3  # Grover amplitude amplification iterations
QSG_JACOBI_ITERATIONS: int = 2  # Jacobi consistency refinement iterations
QSG_HOPFIELD_BETA: float = 1.0  # Modern Hopfield inverse temperature
QSG_DEFAULT_TEMPERATURE: float = 1.0  # Default sampling temperature
QSG_DEFAULT_TOP_K: int = 50  # Default top-k filtering
QSG_DEFAULT_TOP_P: float = 0.9  # Default nucleus sampling threshold

# =============================================================================
# LORENTZIAN/HYPERBOLIC GEOMETRY PARAMETERS
# =============================================================================
# Enables hyperbolic feature transforms on TimeCrystal blocks for hierarchical
# language structure representation. Uses Lie algebra matrix exponential.
USE_LORENTZIAN_TRANSFORM = True  # Enable hyperbolic feature transforms
LORENTZIAN_BOOST_DIM = 4  # Spatial dimension for boost vectors (D_spatial)
LORENTZIAN_HYPERBOLIC_DIM = 5  # Full hyperbolic dimension (D_spatial + 1)

# =============================================================================
# KALMAN FILTERING PARAMETERS
# =============================================================================
# Enables Kalman state estimation block after TimeCrystal for uncertainty tracking
USE_KALMAN_FILTERING: bool = True  # Add KalmanBlock after TimeCrystal blocks
KALMAN_STATE_DIM: int = 16  # Kalman filter state dimension

# =============================================================================
# SPARSE ATTENTION PARAMETERS
# =============================================================================
# Band-diagonal sparse matrix multiplication for efficient local attention
USE_SPARSE_ATTENTION = True  # Enable band-diagonal sparse attention
SPARSE_ATTENTION_LOWER_BANDS = 64  # Lower bandwidth (tokens to look back)
SPARSE_ATTENTION_UPPER_BANDS = 64  # Upper bandwidth (tokens to look forward)


# =============================================================================
# MODEL ARCHITECTURE PARAMETERS
# =============================================================================
VOCAB_SIZE = 256000  # Vocabulary size (256k for frontier models)
CHUNK_SIZE = 128  # Chunk size for processing
CHUNK_STRIDE = 64  # Stride between chunks
EMBEDDING_DIM = 512  # Embedding dimension
COMPRESSED_DIM = 128  # Compressed representation dim
AGGREGATOR_RANK = 64  # Aggregator rank for compression
REASONING_LAYERS = 6  # Number of reasoning layers
MAX_CONTEXT_LEN = 256000  # 1M tokens (Lite limit: 5M)

# =============================================================================
# QUANTUM INTEGRATION PARAMETERS
# =============================================================================

QUANTUM_BACKEND = os.getenv("HSMN_QUANTUM_BACKEND", "default.qubit")
QUANTUM_NOISE_MODEL = os.getenv("HSMN_QUANTUM_NOISE_MODEL", "depolarizing")
QUANTUM_NOISE_STRENGTH = float(os.getenv("HSMN_QUANTUM_NOISE_STRENGTH", "0.02"))
QUANTUM_VQC_QUBITS = 2  # VQC qubits
QUANTUM_VQC_LAYERS = 2  # VQC layers
QUANTUM_VQC_SHOTS = None  # Shots (None = analytic)
QUANTUM_ENTANGLEMENT = "linear"  # Entanglement type
QUANTUM_KERNEL_QUBITS = 2
QUANTUM_KERNEL_LAYERS = 2
QUANTUM_ENERGY_QUBITS = 2
QUANTUM_ENABLE_SAMPLING = False
QUANTUM_ZNE_SCALES = (1.0, 2.0, 3.0)  # Zero-noise extrapolation scales
QGAN_SAMPLES_PER_BATCH = 1  # Quantum GAN samples per batch

# =============================================================================
# UNIFIED QUANTUM ARCHITECTURE (Phases 19-24)
# =============================================================================
# These enhancements are applied automatically to the reasoning stack.
# All operations use float32/float64 precision with NO QUANTIZATION.
# Disable only if memory-constrained or for ablation studies.

# Phase 19: Core Physics Enhancements (enabled by default)
USE_HOLOGRAPHIC_MEMORY: bool = True  # 19.1 - O(n log n) FFT binding
USE_PORT_HAMILTONIAN: bool = True  # 19.2 - Dissipative dynamics (extends SPHNN)
USE_TT_SSM: bool = True  # 19.3 - Tensor-train SSM
USE_THERMODYNAMIC_ROUTING: bool = True  # 19.4 - Annealed MoE routing
USE_QUANTUM_FEATURE_MAPS: bool = True  # 19.5 - RKHS feature maps

# Phase 20-24: Advanced Quantum (enabled by default)
USE_QUANTUM_WALK_EMBEDDINGS: bool = True  # 20.2 - CTQW attention
USE_ORTHOGONAL_KEYS: bool = True  # 22.2 - Anti-collapse keys
USE_HYPERBOLIC_EVOLUTION: bool = True  # 22.2 - Poincaré ball dynamics
USE_QSVT_ACTIVATIONS: bool = True  # 24.1 - Chebyshev activations
USE_MIXED_STATE_ATTENTION: bool = True  # 24.2 - Density matrix attention
USE_QUANTUM_RESERVOIR: bool = True  # 24.3 - Quantum reservoir (enabled)

# Tunable parameters (float32 precision throughout)
HOLOGRAPHIC_DIM: int = 512  # Must be power of 2 for FFT
THERMODYNAMIC_TEMP_INITIAL: float = 1.0  # Starting temperature
THERMODYNAMIC_TEMP_FINAL: float = 0.1  # Target annealed temperature
THERMODYNAMIC_ANNEAL_STEPS: int = 10000  # Steps for temperature annealing
TT_SSM_RANKS: list[int] = [4, 8, 4]  # Tensor-train ranks
HYPERBOLIC_CURVATURE: float = -1.0  # Negative for Poincaré ball
QSVT_POLYNOMIAL_DEGREE: int = 8  # Chebyshev approximation degree
QUANTUM_RESERVOIR_DIM: int = 64  # Reservoir hidden dimension
ORTHOGONAL_PENALTY_WEIGHT: float = 0.01  # Weight for orthogonality loss

# =============================================================================
# UNIFIED QUANTUM ARCHITECTURE PHASES 26-36 (mamba_timecrystal_wlam_moe_hybrid)
# =============================================================================
# These enhancements unify ALL layers under quantum-coherent architecture.
# All operations are C++ native with NO PYTHON FALLBACKS.
# See final_enhancements.md for full implementation details.

# Phase 26: Quantum Embedding Layer (QEL)
USE_QUANTUM_EMBEDDING: bool = True  # Holographic hyperdimensional embeddings
QUANTUM_EMBEDDING_BUNDLES: int = 4  # Number of holographic bundles
QUANTUM_EMBEDDING_SEED: int = 42  # Haar-random key initialization seed

# Phase 27: Floquet Position Encoding
USE_FLOQUET_POSITION: bool = True  # Time-crystal SU(2) position encoding
FLOQUET_NUM_LAYERS: int = 2  # SU(2) rotation layers per qubit
FLOQUET_BASE_FREQUENCY: float = 10000.0  # Base frequency for angle initialization

# Phase 28: Quantum Feature Maps for FlashLinearAttention
# (Extends existing USE_QUANTUM_FEATURE_MAPS flag)
QUANTUM_FEATURE_ROTATION_DEPTH: int = 2  # VQC rotation depth for feature maps

# Phase 29: Unitary Expert Networks
USE_UNITARY_EXPERT: bool = True  # Cayley-parameterized expert FFN
NEUMANN_CAYLEY_TERMS: int = 6  # Neumann series truncation terms

# Phase 30: Quantum Normalization (QNorm)
USE_QUANTUM_NORM: bool = True  # Stiefel manifold normalization
QUANTUM_NORM_EPS: float = 1e-6  # Epsilon for numerical stability

# Phase 31: Superposition BPE
USE_SUPERPOSITION_BPE: bool = True  # Multi-segmentation tokenization
SBPE_MAX_SUPERPOSITION: int = 4  # Maximum parallel segmentations
SBPE_AMPLITUDE_THRESHOLD: float = 0.1  # Minimum amplitude to keep segmentation

# Phase 32: Enhanced Grover QSG
QSG_QUALITY_THRESHOLD: float = 0.7  # Oracle threshold for "good" sequences
QSG_AUTO_ITERATIONS: bool = True  # Use optimal √N iteration count

# Phase 33: Quantum LM Head
USE_QUANTUM_LM_HEAD: bool = True  # VQC-based output distribution (Born rule)
QUANTUM_LM_HEAD_LAYERS: int = 2  # VQC circuit depth

# Phase 34: Unitary Residual Connections
USE_UNITARY_RESIDUAL: bool = True  # Rotation-blend residuals (cos/sin)
UNITARY_RESIDUAL_INIT_ANGLE: float = 0.7854  # π/4 for balanced initialization

# Phase 35: Quantum State Bus
USE_QUANTUM_STATE_BUS: bool = True  # Entanglement-mediated cross-block comm
QUANTUM_STATE_BUS_TELEPORT: bool = True  # Enable teleportation-style state transfer

# =============================================================================
# QUANTUM TRAINING LOOP ENHANCEMENTS (Phases T1-T4)
# =============================================================================
# These parameters control quantum-enhanced training loop optimizations.
# All maintain O(n) complexity and focus on performance/memory improvements.

# Phase T4: Barren Plateau Detection and Mitigation (HIGH)
# Monitors gradient norms and applies recovery strategies for quantum layers
BARREN_PLATEAU_MONITOR: bool = True  # Enable barren plateau monitoring
BARREN_PLATEAU_THRESHOLD: float = 1e-6  # Gradient norm threshold for detection
BARREN_PLATEAU_RECOVERY_LR_SCALE: float = 10.0  # LR scaling factor for recovery
BARREN_PLATEAU_HYSTERESIS: float = 5.0  # Factor for exit threshold (exit = threshold * hysteresis)
BARREN_PLATEAU_RECOVERY_WINDOW: int = 100  # Steps to maintain mitigation after detection

# Phase 2-4: Gradient-Norm-Aware Dampening for Barren Plateau Recovery
# Prevents LR explosion when BP scaling is applied with high gradients.
# See PHASE2_4_SMART_TUNER_FIX.md for full design rationale.
GRADIENT_DAMPENING_THRESHOLD: float = 100.0  # Grad norm above which dampening activates
MIN_GRADIENT_DAMPENING: float = 0.001  # Floor for dampening factor (prevent stalls)

# OPTIMIZER LR CAPS (Fallback when optimizer max_safe_lr trait unavailable)
# These caps prevent catastrophic LR during barren plateau recovery.
OPTIMIZER_LR_CAPS: dict[str, float] = {
    "sympflowqngoptimizer": 1e-3,
    "sophiag": 5e-4,
    "lion": 3e-4,
    "qiao": 5e-4,
}

# Phase T2: Tensor-GaLore Gradient Compression (HIGH)
# Reduces optimizer memory by 50-75% via Tucker decomposition
USE_TENSOR_GALORE: bool = True  # Enable gradient low-rank projection
GALORE_RANK: int = 32  # Gradient projection rank
GALORE_UPDATE_PROJ_GAP: int = 200  # Steps between projection updates
GALORE_SCALE: float = 0.25  # Gradient scaling factor

# Phase T1: Quantum Natural Gradient (HIGH)
# Replaces standard gradient with QFIM-preconditioned gradient
USE_QUANTUM_NATURAL_GRADIENT: bool = True  # Enable QNG optimizer path
QNG_DAMPING: float = 1e-4  # QFIM regularization damping
QNG_APPLY_TO_QUANTUM_ONLY: bool = True  # Only apply to quantum-enhanced layers

# Phase T3: Quantum-Inspired Hessian Estimation (MEDIUM)
# Better Hessian approximation via quantum random projections
USE_QUANTUM_HESSIAN_ESTIMATION: bool = True  # Enable quantum hessian estimation
QUANTUM_HESSIAN_SAMPLES: int = 8  # Samples for hessian estimation

# Phase T5: Evolution Time QNG (MEDIUM)
# Riemannian optimization for Hamiltonian evolution time
USE_EVOLUTION_TIME_QNG: bool = True  # Enable evolution time QNG
EVOLUTION_TIME_METRIC: str = "hamiltonian"  # Metric type: "hamiltonian", "euclidean"

# Phase T6: Neural Zero-Noise Extrapolation (LOW)
# Error mitigation for quantum layer outputs
USE_NEURAL_ZNE: bool = True  # Enable neural ZNE
NEURAL_ZNE_HIDDEN_DIM: int = 128  # Hidden dimension for ZNE MLP
NEURAL_ZNE_TRAIN_SAMPLES: int = 1000  # Training samples for ZNE model
NEURAL_ZNE_FEEDBACK: bool = True  # S23: Feed ZNE statistics into QULS weight controller

# Phase 129: ML-Enhanced Quantum Error Mitigation (Neural QEM)
# Extends Neural ZNE with learned error models for quantum circuit outputs
USE_NEURAL_QEM: bool = True  # Enable ML-enhanced quantum error mitigation
NEURAL_QEM_NOISE_LEVELS: tuple[float, ...] = (1.0, 1.5, 2.0)  # Noise levels for extrapolation
NEURAL_QEM_HIDDEN_DIM: int = 128  # Error model hidden dimension
NEURAL_QEM_LEARNING_RATE: float = 1e-4  # Error model learning rate

# =============================================================================
# QUANTUM ENHANCEMENT INTEGRATIONS (Phase 130+)
# =============================================================================
# These flags enable synergistic connections between quantum components.

# Phase 130.1: Auto Neural QEM Wrapping
# Automatically apply Neural QEM to all quantum layer outputs
USE_AUTO_NEURAL_QEM: bool = True  # Auto-apply Neural QEM to VQC, QASA, Q-SSM outputs

# Phase 130.2: GaLore-VQC Gradient Awareness
# Use VQC gradient statistics to inform GaLore rank allocation
GALORE_VQC_AWARE: bool = True  # Use VQC gradient variance for rank allocation
GALORE_VQC_VARIANCE_BOOST: float = 1.5  # Rank multiplier for high-variance VQC layers

# Phase 130.3: Floquet-VQC Modulation
# Floquet phase modulates VQC entanglement strengths
USE_FLOQUET_VQC_MODULATION: bool = True  # Floquet phase modulates VQC circuits

# Phase 130.4: COCONUT-VQC Amplitude Selection
# Use VQC-derived amplitudes for COCONUT path weighting
COCONUT_USE_VQC_AMPLITUDES: bool = True  # Use VQC for path weighting instead of softmax

# Phase 130.5: Hopfield-QuantumBus Integration
# Publish Hopfield energy scores to quantum bus for downstream blocks
HOPFIELD_PUBLISH_TO_BUS: bool = True  # Publish energy scores to quantum bus

# Phase 130.6: QASA-MPS Entanglement Gating
# Gate QASA quantum contribution by MPS entanglement entropy
QASA_MPS_GATING: bool = True  # Gate QASA by MPS entanglement entropy
QASA_MPS_ENTROPY_THRESHOLD: float = 0.5  # Min entropy for full quantum contribution

# =============================================================================
# QUANTUM SYNERGY INTEGRATIONS (Phase 131+)
# =============================================================================
# These flags enable additional synergistic connections between quantum components.

# S1: QMamba ↔ Q-SSM Gating Unification
# Share VQC gates between QMamba superposition and Q-SSM gating decisions
USE_UNIFIED_QSSM_GATING: bool = True  # Enable unified Q-SSM/QMamba gating
UNIFIED_QSSM_SHARE_VQC: bool = True  # Share VQC parameters between components

# S2: QMamba Amplitudes → COCONUT Path Weighting
# Feed QMamba learned amplitudes into COCONUT BFS path selection
COCONUT_USE_QMAMBA_AMPLITUDES: bool = True  # Use QMamba amplitudes as path prior

# S3: TimeCrystal Evolution Time → VQC Circuit Depth
# Dynamically adjust VQC circuit depth based on predicted evolution time
USE_ADAPTIVE_VQC_DEPTH: bool = True  # Enable evolution-time-dependent VQC depth
VQC_MIN_LAYERS: int = 1  # Minimum VQC layers
VQC_MAX_LAYERS: int = 4  # Maximum VQC layers

# S18: CayleyDense → VQC Data Encoder (Orthogonal Weight Stability)
# Use Cayley-parameterized orthogonal weights for VQC data encoder layers
# Improves gradient stability in deep VQC circuits via unitary constraint
USE_CAYLEY_VQC: bool = True  # Use CayleyDense for VQC data encoding


# S4: Floquet Phase → QMoE Expert Selection
# Condition QMoE expert selection on Floquet phase for phase-specialist experts
QMOE_USE_FLOQUET_PHASE: bool = True  # Floquet phase modulates expert routing

# S5: Hopfield Energy → QHPM Crystallization Threshold
# Use Hopfield retrieval energy to dynamically adjust crystallization threshold
QHPM_USE_HOPFIELD_THRESHOLD: bool = True  # Dynamic crystallization threshold

# S6: MPS Bond Entropy → Hopfield β (Inverse Temperature)
# Dynamically adjust Hopfield β based on MPS entanglement entropy
HOPFIELD_ADAPTIVE_BETA: bool = True  # Context-aware memory retrieval sharpness
HOPFIELD_BASE_BETA: float = 1.0  # Base inverse temperature

# S7: QASA Attention Weights → QMamba Selective Gating
# Feed QASA attention patterns as prior for QMamba selectivity
QMAMBA_USE_QASA_PRIOR: bool = True  # Attention-informed recurrence

# S8: Q-SSM Gate Statistics → Neural ZNE Calibration
# Use Q-SSM gate variance/entropy as features for Neural ZNE error prediction
NEURAL_ZNE_USE_QSSM_STATS: bool = True  # Q-SSM statistics for error mitigation

# S9: Quantum Coherence Bus → All VQC Layers
# Broadcast QCB coherence to VQC layers to modulate entanglement strength
VQC_USE_COHERENCE_BUS: bool = True  # Global coherence modulates VQC circuits

# S10: Unified Bus Entanglement → TimeCrystal Floquet Period
# Modulate Floquet drive period based on bus entanglement strength
DTC_ADAPTIVE_PERIOD: bool = True  # Entanglement-stabilized time crystal

# S11: AlphaQubit Decoder → All Quantum Layer Outputs
# Unified post-processing with AlphaQubit-style error decoding
USE_UNIFIED_ALPHAQUBIT: bool = True  # Systematic error correction
ALPHAQUBIT_ENABLED_LAYERS: tuple[str, ...] = (
    "VQCLayer",
    "QMambaBlock",
    "QASAAttention",
    "QMoERouter",
)

# S12: SympFlow Optimizer → QNG Geodesic Integration
# Combine symplectic momentum with quantum-natural geodesic corrections
SYMPFLOW_USE_QNG_GEODESIC: bool = True  # Superior optimization landscape navigation
SYMPFLOW_GEODESIC_WEIGHT: float = 0.1  # Weight for geodesic corrections


# =============================================================================
# QUANTUM CONSTANT-MEMORY TRAINING (Phases 1-7 + 25)
# =============================================================================
# Implementation of O(1) or O(depth) memory training from QUANTUM_TRAINING_RESEARCH.md.
# All enhancements are CPU-only, float32 precision, and maintain linear complexity.

# Phase 1: Neumann-Cayley Unitary Mamba Gates (CRITICAL)
# Constrains B/C projections to be unitary via Cayley transform
USE_UNITARY_MAMBA_GATES: bool = True  # Enable Neumann-Cayley for Mamba
NEUMANN_SERIES_TERMS: int = 4  # Neumann series truncation (4-8 typical)

# Phase 2: SPRK-Enhanced TimeCrystal (HIGH)
# Higher-order symplectic integrator for Hamiltonian evolution
USE_SPRK_TIMECRYSTAL: bool = True  # Use 6th-order Yoshida (vs 4th)
SPRK_ORDER: int = 6  # Integrator order (4 or 6)

# Phase 3: CA-TDVP Isometric MPS (HIGH)
# Clifford-augmented TDVP with isometry constraints
USE_CATDVP_MPS: bool = True  # Enable CA-TDVP for MPS gradients
USE_ISOMETRIC_MPS_CORES: bool = True  # Enforce left-canonical isometry

# Phase 4: Lifting Scheme Reversible WLAM (MEDIUM)
# Ensures perfect invertibility for gradient reconstruction
WLAM_VALIDATE_REVERSIBILITY: bool = True  # Validate lifting invertibility

# Phase 5: QNG Geodesic MoE Routing (HIGH)
# Quantum Natural Gradient with geodesic corrections
USE_QNG_GEODESIC: bool = True  # Enable geodesic QNG for MoE router
QNG_GEODESIC_ORDER: int = 2  # Geodesic correction order (1 or 2)

# Phase 6: Quantum Memory Replay (CRITICAL)
# Custom autograd with logarithmic checkpointing for O(log n) memory
ENABLE_QUANTUM_MEMORY_REPLAY: bool = True  # Enable QMR autograd wrapper
QMR_CHECKPOINT_STRATEGY: str = "logarithmic"  # "logarithmic" or "fixed"
QMR_LOG_CHECKPOINT_FACTOR: int = 2  # Checkpoint density factor

# Phase 7: Entanglement Preservation Loss (MEDIUM)
# Regularizes training to maintain context-carrying capacity
ENTANGLEMENT_REGULARIZATION: float = 0.01  # Weight for entropy loss (0 = disabled)
MIN_BOND_ENTROPY: float = 0.5  # Minimum MPS bond entropy target
USE_TRUNCATION_NOISE: bool = True  # Inject gradient noise at truncation

# Phase 25: Quantum-Holographic Persistent Memory [EXPERIMENTAL]
# Silica-inspired persistent memory with holographic binding
USE_QUANTUM_HOLOGRAPHIC_MEMORY: bool = True  # Enable QHPM layer
QHPM_HOLOGRAPHIC_DIM: int = 512  # Holographic vector dimension
QHPM_MPS_BOND_DIM: int = 32  # MPS bond dimension for storage
QHPM_HOPFIELD_BETA: float = 1.0  # Modern Hopfield inverse temperature
QHPM_CRYSTALLIZATION_RATE: float = 0.1  # Orthogonal gradient projection rate

# =============================================================================
# QUANTUM ENHANCEMENT PHASES 37-46 (mamba_timecrystal_wlam_moe_hybrid)
# =============================================================================
# These enhancements extend the unified quantum architecture with advanced
# quantum-inspired operations. All are C++ native with NO PYTHON FALLBACKS.
# See docs/final_enhancements.md for full implementation details.

# Phase 37: QMamba - Quantum-Enhanced State Space Model
# Extends Mamba SSM with quantum superposition and entanglement
USE_QMAMBA: bool = True  # Enable quantum-enhanced Mamba
QMAMBA_NUM_SUPERPOSITION_STATES: int = 4  # K parallel quantum state paths
QMAMBA_ENTANGLEMENT_DEPTH: int = 2  # VQC entanglement layers per step
QMAMBA_ENTANGLEMENT_STRENGTH: float = 0.3  # α ∈ [0,1] for quantum mixing
QMAMBA_USE_BORN_RULE: bool = True  # Born rule (vs Gumbel-Softmax) collapse
QMAMBA_GUMBEL_TEMPERATURE: float = 1.0  # Temperature for Gumbel-Softmax collapse

# Phase 38: Discrete Time Crystal State Protection
# Stabilizes TimeCrystal hidden states using DTC Floquet dynamics
USE_DTC_PROTECTION: bool = True  # Enable DTC state protection
DTC_FLOQUET_PERIOD: int = 4  # Floquet driving period (steps)
DTC_COUPLING_J: float = 1.0  # Heisenberg coupling strength
DTC_DISORDER_W: float = 0.5  # MBL disorder strength
DTC_PI_PULSE_ERROR: float = 0.01  # π-pulse imperfection ε
DTC_USE_PRETHERMAL: bool = True  # Enable prethermal DTC regime
DTC_NUM_CYCLES: int = 1  # Floquet cycles per evolution step

# Phase 128: Floquet Time Crystal Evolution Optimization
# Enhances TimeCrystalBlock with Floquet-engineered periodic driving for stability
USE_FLOQUET_EVOLUTION: bool = True  # Enable Floquet-engineered Hamiltonian evolution
FLOQUET_DRIVE_AMPLITUDE: float = 0.1  # Periodic drive amplitude for H(t) = H_0 + V(t)
FLOQUET_USE_MBL: bool = True  # Enable Many-Body Localization for error mitigation


# Phase 39: Coconut Continuous Latent Reasoning
# BFS-style thought exploration with quantum reservoir
USE_COCONUT_REASONING: bool = True  # Enable Coconut thought refinement
COCONUT_MAX_THOUGHT_STEPS: int = 8  # Maximum reasoning iterations
COCONUT_BFS_BRANCHES: int = 4  # Parallel thought branches (K)
COCONUT_HALT_THRESHOLD: float = 0.9  # Confidence for early stopping
COCONUT_BRANCH_ALPHA: float = 0.1  # Residual update weight
COCONUT_RESERVOIR_DIM: int = 64  # Quantum reservoir hidden dimension
COCONUT_DISSIPATION_RATE: float = 0.3  # γ ∈ [0,1] for tunable info loss
COCONUT_USE_ECHO_STATE: bool = True  # Enforce echo state property

# Phase 40: Holographic State Binding (extends existing QHPM)
# Enhanced holographic binding for cross-block state persistence
USE_HOLOGRAPHIC_STATE_BINDING: bool = True  # Enable enhanced holographic binding
HOLOGRAPHIC_BINDING_DIMENSION: int = 256  # Binding vector dimension
HOLOGRAPHIC_BINDING_STRENGTH: float = 0.5  # Blend weight with existing states

# Phase 41: Learnable Multi-Scale Wavelet Transformer (LMWT)
# Replaces WLAM fixed wavelets with learnable parameters
USE_LMWT_ATTENTION: bool = True  # Enable learnable wavelet attention
LMWT_NUM_SCALES: int = 4  # Wavelet decomposition levels
LMWT_LEARN_FILTERS: bool = True  # Enable gradient updates to α, β
LMWT_ALPHA_INIT: float = 0.7071067811865476  # Low-pass filter init (1/√2)
LMWT_BETA_INIT: float = 0.7071067811865476  # High-pass filter init (1/√2)
LMWT_NUM_HEADS: int = 8  # Cross-scale attention heads

# Phase 42: Quantum Mixture of Experts (QMoE) Routing
# VQC-based routing with Born rule selection
USE_QMOE_ROUTING: bool = True  # Enable quantum MoE routing
QMOE_NUM_QUBITS: int = 4  # Effective qubits (log2 num_experts)
QMOE_VQC_LAYERS: int = 2  # VQC rotation layers
QMOE_MEASUREMENT_TEMP: float = 1.0  # Born rule temperature
QMOE_USE_ENTANGLEMENT: bool = True  # Cross-token routing correlations

# Phase 43: Neural Kalman with Learned Covariance
# GRU-based learned Kalman gain
USE_NEURAL_KALMAN: bool = True  # Enable neural Kalman filter
NEURAL_KALMAN_HIDDEN_DIM: int = 128  # GRU hidden dimension
NEURAL_KALMAN_PROPAGATE_COV: bool = False  # Track full covariance (expensive)
NEURAL_KALMAN_INITIAL_P: float = 1.0  # Initial covariance diagonal
NEURAL_KALMAN_PROCESS_NOISE: float = 0.01  # Initial Q estimate
NEURAL_KALMAN_MEASUREMENT_NOISE: float = 0.1  # Initial R estimate

# Phase 44: Quantum-Teleported State Bus Communication
# Entanglement-mediated cross-block state transfer
USE_QUANTUM_TELEPORT_BUS: bool = True  # Enable teleport-style state bus
TELEPORT_ENTANGLEMENT_DIM: int = 64  # Entangled state dimension
TELEPORT_FIDELITY_THRESHOLD: float = 0.9  # Minimum fidelity for success
TELEPORT_USE_CORRECTION: bool = True  # Apply Pauli corrections

# Phase 45: Von Neumann Entropy Regularization
# Entropy-based regularization for representation diversity
USE_ENTROPY_REGULARIZATION: bool = True  # Enable entropy regularization
ENTROPY_REG_WEIGHT: float = 0.01  # Von Neumann entropy loss weight
SPECTRAL_REG_WEIGHT: float = 0.01  # Spectral flatness loss weight
TARGET_ENTROPY: float = 0.5  # Target entropy (higher = diverse)
SPECTRAL_FLATNESS_TARGET: float = 0.8  # Target flatness ∈ [0,1]
ENTROPY_POWER_ITER_STEPS: int = 10  # Power iteration steps for eigenvalues

# Phase 46: SympFlow Hamiltonian Optimizer
# Symplectic integrator-based optimizer
USE_SYMPFLOW_OPTIMIZER: bool = True  # Enable SympFlow (can be used alongside SophiaG)
SYMPFLOW_MASS: float = 1.0  # Effective mass for momentum
SYMPFLOW_FRICTION: float = 0.1  # Dissipation rate γ
SYMPFLOW_STEP_SIZE: float = 0.01  # Leapfrog step size (learning rate)
SYMPFLOW_NUM_LEAPFROG_STEPS: int = 1  # Leapfrog sub-steps per update


# =============================================================================
# ANTI-FORGETTING CONFIGURATION (EWC Deprecated, QHPM Preferred)
# =============================================================================
# EWC (Elastic Weight Consolidation) has been superseded by QHPM crystallization.
# QHPM provides mathematically superior anti-forgetting via hard orthogonal
# gradient projection, whereas EWC only applies a soft penalty.

# Legacy EWC Configuration [DEPRECATED]
ENABLE_EWC: bool = False  # [DEPRECATED] Set True to enable legacy EWC
EWC_LAMBDA: float = 0.0  # [DEPRECATED] EWC penalty weight (0 = disabled)

# QHPM Crystallization Configuration [PREFERRED] - Replaces EWC
# Uses orthogonal gradient projection to prevent overwrites of crystallized memories.
ENABLE_QHPM_CRYSTALLIZATION: bool = True  # Enable gradient crystallization
QHPM_CRYSTALLIZATION_THRESHOLD: float = 0.85  # Confidence threshold for crystallization
QHPM_MAX_CRYSTALLIZED_DIRECTIONS: int = 256  # Maximum protected parameter directions
QHPM_CRYSTALLIZATION_DECAY: float = 0.99  # Gradual decay of crystallization strength

# =============================================================================
# QUANTUM ENHANCEMENT PHASES 47-84 v5.0 (Unified Architecture)
# =============================================================================
# These enhancements complete the unified quantum architecture with advanced
# quantum-inspired operations across all pillars. All are C++ native.
# See docs/final_enhancements.md for full implementation details.

# Pillar 1: Foundation (Critical)
# Phase 47: Quantum Measurement Dropout
USE_QUANTUM_MEASUREMENT_DROPOUT: bool = True  # Enable QMD for ensemble circuits
QMD_DROP_RATE: float = 0.1  # QMD measurement probability
QMD_SOFTENING_TEMP: float = 1.0  # Soft collapse temperature

# Phase 69: Q-SSM Quantum State Space Gating
USE_Q_SSM_GATING: bool = True  # Enable VQC gating for Mamba
Q_SSM_VQC_LAYERS: int = 2  # VQC circuit layers
Q_SSM_NUM_QUBITS: int = 4  # Virtual qubits for Q-SSM

# Phase 71: Intrinsic Plasticity Preservation
USE_INTRINSIC_PLASTICITY: bool = True  # Enable Stiefel manifold plasticity
PLASTICITY_LEARNING_RATE: float = 0.01  # Plasticity update rate

# Phase 76: Unified Quantum Coherence Bus
USE_QUANTUM_COHERENCE_BUS: bool = True  # Enable QCB for block coordination
QCB_NUM_NODES: int = 8  # Coherence mesh nodes
QCB_FIDELITY_THRESHOLD: float = 0.9  # Minimum entanglement fidelity

# =============================================================================
# PHASE 127: UNIFIED QUANTUM ENTANGLEMENT BUS
# =============================================================================
# Unifies existing quantum buses (State Bus, Coherence Bus, Teleport Bus) into
# a single coherent entanglement-mediated cross-block communication system.
# Complexity: O(n · d) for entanglement propagation, O(χ² · d) memory with MPS.

USE_UNIFIED_QUANTUM_BUS: bool = True  # Enable unified quantum bus
UNIFIED_BUS_ADAPTIVE: bool = True  # Enable adaptive entanglement strength learning
UNIFIED_BUS_MPS_BOND_DIM: int = 32  # MPS bond dimension for entanglement representation
UNIFIED_BUS_COHERENCE_THRESHOLD: float = 0.85  # Minimum coherence for propagation
UNIFIED_BUS_ENTANGLEMENT_INIT: float = 0.5  # Initial entanglement strength
UNIFIED_BUS_PROPAGATION_RATE: float = 0.1  # Entanglement update rate during forward pass

# Pillar 2: Input/Output Enhancement
# Phase 48: Hyperdimensional Quantum Embeddings
USE_HYPERDIMENSIONAL_EMBEDDING: bool = True  # Enable HQE holographic bundling
HQE_CTQW_STEPS: int = 3  # CTQW spreading walk steps
HD_EMBEDDING_DIM: int = 4096  # Hyperdimensional embedding dimension
HD_ACTIVE_VOCAB_SIZE: int = 10000  # Active vocabulary for DualPathEmbedding

# =============================================================================
# PHASE 200+: BLOCK-INTEGRATED HD STREAMING (HIGHNOON_UPGRADE_ROADMAP.md)
# =============================================================================
# When enabled, reasoning blocks operate natively on HD bundles instead of
# converting back to model_dim at block boundaries. Requires native C++ ops.

USE_HD_SPATIAL_BLOCK: bool = True  # Replace SpatialBlock with HDSpatialBlock
USE_HD_TIMECRYSTAL_BLOCK: bool = True  # Replace TimeCrystalBlock with HDTimeCrystal
USE_HD_MOE_BLOCK: bool = True  # Replace FusedMoEDispatch with HDMoEDispatch
USE_HOLOGRAPHIC_LOSS: bool = True  # Use HD-space cross-entropy (no logits tensor)
HD_PROJECTION_FREEZE_EPOCHS: int = 2  # Freeze HD projections for N epochs
HD_REQUIRE_NATIVE_OPS: bool = True  # Raise error if C++ ops unavailable

# =============================================================================
# PHASE 300+: HD UPGRADE INTEGRATION (hd_upgrade.md)
# =============================================================================
# These flags enable the Phase 300+ HD upgrade integration, including optimizer
# state compression, FFT-based spectral entropy for QULS, HD gradient projection,
# holographic attention similarity, and HD KV cache for inference.

# Phase 1: HD Optimizer State Compression
# Compresses optimizer states (momentum, Hessian, QFIM) via HD projection
USE_HD_OPTIMIZER_STATES: bool = True  # Compress optimizer states (50-60% memory reduction)
HD_OPTIMIZER_COMPRESSION_RATIO: int = 8  # Compression ratio for optimizer states
HD_OPTIMIZER_USE_SPARSE: bool = True  # Use sparse random projection (faster)

# Phase 2: HD QULS + Gradient Compression
# FFT-based spectral entropy (O(d log d)) replaces power iteration (O(d²))
USE_HD_SPECTRAL_ENTROPY: bool = True  # FFT-based spectral entropy for QULS
# HD gradient projection replaces Tucker decomposition (no periodic SVD)
USE_HD_GRADIENT_PROJECTION: bool = True  # Phase 300+: HD projection for GaLore
HD_GRADIENT_RANK: int = 128  # Target rank for HD gradient projection (legacy)
# Phase 300+ HD-native: Frequency bandwidth for gradient compression
HD_GRADIENT_BANDWIDTH: int = 256  # Top-K frequencies to retain in FFT compression

# Phase 3: HD Attention Layers
# Holographic Q·K similarity via FFT correlation (O(d log d) per pair)
USE_HD_HOLOGRAPHIC_ATTENTION: bool = True  # FFT-based attention similarity
# HD KV cache compression for inference (8-16x memory reduction)
USE_HD_KV_CACHE: bool = True  # HD compressed KV cache for inference
HD_KV_COMPRESSION_RATIO: int = 8  # Tokens per HD bundle for KV cache

# Phase 4: HD Reasoning Enhancements
# HD thought trace storage for COCONUT continuous reasoning
USE_HD_THOUGHT_TRACE: bool = True  # HD thought trace for COCONUT

# =============================================================================
# PHASE 400: HD-ENHANCED META CONTROLLER INTEGRATION (implementation_plan.md)
# =============================================================================
# Deep integration of HD embeddings with Hamiltonian Meta Controller and Kalman
# filters for improved control, stagnation detection, and memory efficiency.

# HD Control Fingerprinting: Detect training stagnation via HD innovation similarity
USE_HD_CONTROL_FINGERPRINTING: bool = True  # HD innovation fingerprinting in EKF/TNKF
HD_CONTROL_FINGERPRINT_DIM: int = 256  # HD fingerprint dimension
HD_STAGNATION_THRESHOLD: float = 0.95  # Similarity threshold for stagnation detection

# HD Kalman State: HD-compressed state representation in Python KalmanBlock
USE_HD_KALMAN_STATE: bool = True  # Enable HD state bundling in Python Kalman
HD_KALMAN_STATE_COMPRESSION: int = 8  # Compression ratio for Kalman state

# HD Online Tuning: HD-based evolution_time trend analysis and control smoothing
USE_HD_EVOLUTION_TREND: bool = True  # Track evolution_time trends via HD unbinding
HD_CONTROL_SMOOTHING: bool = True  # Apply HD low-pass filter to control actions


# =============================================================================
# QULS → QAHPO FEEDBACK (HIGHNOON_UPGRADE_ROADMAP.md Phase 1.2)
# =============================================================================
# Enables QULS telemetry export to QuantumAdaptiveHPO for barren plateau
# detection and tunneling probability adjustment.

USE_QULS_HPO_FEEDBACK: bool = True  # Enable QULS → QAHPO feedback loop
BARREN_PLATEAU_TUNNELING_THRESHOLD: int = 3  # Plateau count to boost tunneling
BARREN_PLATEAU_EARLY_STOP_THRESHOLD: int = 7  # Plateau count for early-stop
QULS_FEEDBACK_WINDOW_SIZE: int = 10  # Sliding window for plateau detection
VQC_GRADIENT_VARIANCE_MIN: float = 1e-6  # Threshold for barren detection

# =============================================================================
# PHASE 200+: SPECTRALLY-AWARE QUANTUM CURRICULUM (SAQC)
# =============================================================================
# Synergizes QULS telemetry with curriculum scheduling for quantum-adaptive
# data progression. Complements QULS→QAHPO feedback by closing the loop between
# model quantum state and training data complexity. See quantum_enhancement_analysis.md.
#
# SAQC Modes:
#   - NORMAL: Standard adaptive curriculum progression
#   - RETREAT: Low spectral entropy → high-diversity batches to restore rank
#   - TUNNELING: Barren plateau → inject orthogonal/edge-case datasets
#   - ACCELERATE: High fidelity → rapid curriculum advancement

USE_QUANTUM_CURRICULUM: bool = True  # Master switch for SAQC
SAQC_ENTROPY_RETREAT_THRESHOLD: float = 0.3  # Spectral entropy below this triggers retreat
SAQC_FIDELITY_ADVANCE_THRESHOLD: float = 0.85  # Fidelity above this enables acceleration
SAQC_TUNNELING_DATASET_RATIO: float = 0.2  # Fraction of tunneling samples during plateau
SAQC_COHERENCE_GATE_THRESHOLD: float = 0.9  # Coherence required for progression
SAQC_REQUIRE_NATIVE_OP: bool = True  # Fail if C++ op unavailable
SAQC_FFT_DIM: int = 64  # FFT dimension for spectral analysis
SAQC_UPDATE_INTERVAL: int = 10  # Steps between curriculum state updates
SAQC_MIN_STAGE_DURATION: int = 100  # Minimum steps before stage transition

# =============================================================================
# PHASE 500+: VQC-HD INTEGRATION ENHANCEMENTS
# =============================================================================
# Three-phase integration to tighten VQC-HD connections for improved accuracy,
# quality, expressiveness, and memory efficiency.

# Phase 1: HD Fisher Compression in VQC Meta-Optimizer
# Compresses layer-wise Fisher info via holographic bundling (10-50x memory reduction)
USE_HD_FISHER_COMPRESSION: bool = True  # HD compression for VQC meta-opt
HD_FISHER_COMPRESSION_DIM: int = 4096  # HD space dimension
HD_FISHER_OUTPUT_DIM: int = 64  # Compressed output dimension

# Phase 2: QuantumLMHead VQC-HD Feedback Loop
# Injects HD spectral entropy as VQC rotation modulation (+2-5% rare token accuracy)
QUANTUM_LM_HEAD_USE_HD_ENTROPY: bool = True  # HD entropy feedback
QUANTUM_LM_HEAD_ENTROPY_SCALE: float = 0.1  # Scaling factor for entropy injection

# Phase 3: QWT → HD Continuous Connection
# QWT outputs continuous VQC amplitudes that modulate HD base vectors (gradient through tokenization)
QWT_CONTINUOUS_HD_OUTPUT: bool = True  # Continuous QWT→HD path
QWT_CONTINUOUS_VQC_DIM: int = 256  # VQC amplitude dimension for continuous output
USE_TENSOR_RING_MOE: bool = True  # Enable Tensor-Ring decomposition for MoE gates
TENSOR_RING_RANK: int = 8  # Ring rank for TR decomposition
TENSOR_RING_NUM_CORES: int = 4  # Number of TR cores

# Phase 49: Holographic Hypertokens
USE_HYPERTOKENS: bool = True  # Enable HDRAM spread-spectrum encoding
HYPERTOKEN_SPREAD_FACTOR: int = 4  # Spectrum spreading factor

# Phase 50: Majorana Position Encoding
USE_MAJORANA_POSITION: bool = True  # Enable topological position encoding
MAJORANA_FLOQUET_PERIOD: int = 16  # Floquet drive period

# Phase 51: Born Rule Loss
USE_BORN_RULE_LOSS: bool = True  # Enable QBRL amplitude loss
BORN_RULE_TEMPERATURE: float = 1.0  # Born rule temperature

# Phase 52: Quantum Fidelity Regularization
USE_QUANTUM_FIDELITY_LOSS: bool = True  # Enable fidelity loss term
FIDELITY_LOSS_WEIGHT: float = 0.01  # Fidelity regularization weight

# =============================================================================
# PHASE 132: QUANTUM UNIFIED LOSS SYSTEM (QULS)
# =============================================================================
# Unified loss framework combining primary task loss with quantum-enhanced
# regularization terms. Replaces standard cross-entropy with multi-component
# loss leveraging VQC, QCB, MPS, and Hamiltonian layer metrics.
# Maintains O(n) complexity with VQC-aware adaptive weighting.
# References:
#   - Cerezo et al. (2021): Variational Quantum Algorithms
#   - Lie algebraic theory of barren plateaus (Nature Comm. 2024)
#   - Symplectic learning for HNNs (J. Comp. Physics 2023)

# Master switch for QULS (if False, uses standard cross-entropy)
USE_QUANTUM_UNIFIED_LOSS: bool = True

# Primary Loss Configuration
QULS_PRIMARY_LOSS: str = "sparse_categorical_crossentropy"  # Primary task loss
QULS_LABEL_SMOOTHING: float = 0.1  # Label smoothing factor (0.0-0.2 typical)

# Quantum Fidelity Loss: Trace fidelity F(p,q) = (Σ√pᵢ√qᵢ)², loss = 1 - F
QULS_FIDELITY_WEIGHT: float = 0.01  # Initial fidelity weight
QULS_FIDELITY_TEMPERATURE: float = 1.0  # Born probability temperature

# Born Rule Loss: |ψ|² = p enforcement for VQC outputs
QULS_BORN_RULE_WEIGHT: float = 0.005  # Born rule penalty weight

# Coherence Preservation: Penalty for coherence drop across blocks
QULS_COHERENCE_ENABLED: bool = True  # Enable coherence preservation
QULS_COHERENCE_WEIGHT: float = 0.01  # Coherence loss weight

# Symplectic Conservation: Hamiltonian energy drift penalty
QULS_SYMPLECTIC_ENABLED: bool = True  # Enable energy conservation
QULS_SYMPLECTIC_WEIGHT: float = 0.01  # Symplectic loss weight

# Entanglement Regularization: MPS bond entropy regularization
QULS_ENTANGLEMENT_ENABLED: bool = True  # Enable entanglement regularization
# Uses existing ENTANGLEMENT_REGULARIZATION and MIN_BOND_ENTROPY

# Adaptive Weighting: VQC-aware dynamic weight adjustment
QULS_ADAPTIVE_WEIGHTS: bool = True  # Enable adaptive weights
QULS_WEIGHT_EMA_DECAY: float = 0.99  # EMA decay for weight updates
QULS_VQC_VARIANCE_BOOST: float = 2.0  # Weight boost for high VQC variance
QULS_BARREN_PLATEAU_REDUCTION: float = 0.1  # Weight reduction during barren plateau

# Phase 53: QASA Attention
USE_QASA_ATTENTION: bool = True  # Enable VQC attention scoring
QASA_VQC_LAYERS: int = 2  # QASA VQC depth
QASA_ENTANGLEMENT_STRENGTH: float = 0.5  # Entanglement contribution

# Phase 54: Quantum Grouped Query Attention
# Uses VQC-enhanced Q/K/V projections with TensorPilot optimization
USE_QUANTUM_GQA: bool = True  # Enable QuantumGQA in block factory
QUANTUM_GQA_NUM_HEADS: int = 8  # Number of attention heads
QUANTUM_GQA_KV_HEADS: int = 2  # Key/Value heads (grouped)

# Phase 55: MPQR Multi-Path Reasoning
USE_MPQR_REASONING: bool = True  # Enable Grover path amplification
MPQR_NUM_PATHS: int = 8  # Reasoning paths
MPQR_GROVER_ITERATIONS: int = 3  # Grover iterations

# Phase 56: Topological Wavelet Attention
USE_TOPOLOGICAL_WAVELET: bool = True  # Enable TWA Betti number bias
TWA_NUM_SCALES: int = 4  # Wavelet scales

# Phase 57: TD-MoE Tucker Decomposition
USE_TD_MOE: bool = True  # Enable Tucker-decomposed experts
TD_MOE_TUCKER_RANK: int = 16  # Tucker decomposition rank

# Phase 58: Symplectic GNN Kalman
USE_SYMPLECTIC_GNN_KALMAN: bool = True  # Enable Hamiltonian GNN dynamics
SGKF_DT: float = 0.01  # Symplectic timestep

# Pillar 4: Training & Optimization
# Phase 59: Quantum Adiabatic Optimizer
USE_ADIABATIC_OPTIMIZER: bool = True  # Enable QAO for global optimization
QAO_INITIAL_TEMP: float = 10.0  # QAO initial temperature
QAO_FINAL_TEMP: float = 0.01  # QAO final temperature

# Phase 60: Geodesic Optimizer
USE_GEODESIC_OPTIMIZER: bool = True  # Enable manifold-aware optimization
GEODESIC_MOMENTUM: float = 0.9  # Geodesic momentum

# Phase 61: AlphaQubit-2 Decoder
USE_ALPHAQUBIT_DECODER: bool = True  # Enable neural syndrome decoder
ALPHAQUBIT_NUM_LAYERS: int = 2  # Decoder attention layers

# Phase 62: VQEM Error Mitigation
USE_VQEM: bool = True  # Enable variational error mitigation
VQEM_NUM_PARAMS: int = 32  # VQEM circuit parameters

# Phase 64: Gradient Teleportation
USE_GRADIENT_TELEPORTATION: bool = True  # Enable distributed gradient teleport

# Pillar 5: Memory & Continual Learning
# Phase 65/83: Quantum Crystallization
USE_QUANTUM_CRYSTALLIZATION: bool = True  # Enable knowledge crystallization
CRYSTALLIZATION_THRESHOLD: float = 0.5  # Importance threshold

# Phase 68: Quantum Neuromorphic Memory
USE_NEUROMORPHIC_MEMORY: bool = True  # Enable spiking quantum neurons
NEUROMORPHIC_TAU: float = 10.0  # Membrane time constant

# Pillar 6: Coherence Mesh
# Phase 70: Multi-Stage Hamiltonian
USE_MULTI_STAGE_HAMILTONIAN: bool = True  # Enable staged Hamiltonian learning
HAMILTONIAN_NUM_STAGES: int = 4  # Hamiltonian stages

# Phase 72: Random Natural Gradient
USE_RANDOM_NATURAL_GRADIENT: bool = True  # Enable Monte Carlo QNG approximation
RNG_NUM_SAMPLES: int = 10  # RNG Monte Carlo samples

# Pillar 7: Advanced Quantum Intelligence
# Phase 78: SPINI Integrator
USE_SPINI_INTEGRATOR: bool = True  # Enable symplectic neural integrator
SPINI_FRICTION: float = 0.1  # SPINI friction coefficient

# Phase 79: QCOT Reasoning
USE_QCOT_REASONING: bool = True  # Enable quantum chain-of-thought
QCOT_REASONING_STEPS: int = 3  # QCOT reasoning depth

# Phase 80: Waveform Attention
USE_WAVEFORM_ATTENTION: bool = True  # Enable phase-coherent attention

# =============================================================================
# PHASE 131: QUANTUM ADAPTIVE LEARNING RATE CONTROLLER (QALRC)
# =============================================================================
# Self-tuning learning rate controller using quantum-inspired mechanisms.
# Replaces static LR schedules with adaptive, entropy-driven LR evolution.

USE_QUANTUM_LR_CONTROLLER: bool = True  # Enable QALRC for HPO trials
QALRC_ANNEALING_POWER: float = 2.0  # Annealing schedule exponent (higher = faster focus)
QALRC_TUNNELING_PROBABILITY: float = 0.05  # Base quantum tunneling probability
QALRC_ENTROPY_SMOOTHING: float = 0.9  # EMA coefficient for gradient entropy


# QWT Tokenizer Parameters
QWT_EMBEDDING_DIM = 512  # QWT embedding dimension
QWT_DWT_FILTER_SIZE = 2  # DWT filter size
QWT_VQC_QUBITS = 2  # QWT VQC qubits
QWT_VQC_LAYERS = 2  # QWT VQC layers
QWT_CTQW_STEPS = 0  # CTQW steps (0 = disabled)

# =============================================================================
# PHASE 17 HAMILTONIAN LAYER ENHANCEMENTS
# =============================================================================
# These parameters control the 6 major Hamiltonian/TimeCrystal enhancements.
# All maintain O(n) linear complexity, CPU-only, float32/cfloat32 precision.

# Phase 17.1: Quantum Superposition of Hamiltonians (CRITICAL)
# Enables learning a superposition of basis Hamiltonians with complex amplitudes
HAMILTONIAN_BASIS_SIZE: int = 4  # Number of basis Hamiltonians
HAMILTONIAN_ENABLE_SUPERPOSITION: bool = True  # Enable superposition mode
HAMILTONIAN_SUPERPOSITION_NORMALIZE: bool = True  # Normalize amplitudes (Σ|α|²=1)

# Phase 17.2: Lie-Poisson Neural Network Structure (CRITICAL)
# Generalizes dynamics from canonical T*Q to Lie algebra g* duals
HAMILTONIAN_MANIFOLD: str = "canonical"  # "canonical", "lie_poisson", "port_hamiltonian"
HAMILTONIAN_LIE_GROUP: str = "su_n"  # "so_3", "se_3", "su_n", "gl_n"
HAMILTONIAN_TRACK_CASIMIR: bool = True  # Track Casimir invariant conservation

# Phase 17.3: Magnus Expansion Geometric Integrator (HIGH)
# Unitary-preserving integration for time-dependent Hamiltonians
HAMILTONIAN_INTEGRATOR: str = "yoshida"  # "yoshida", "magnus", "euler"
HAMILTONIAN_MAGNUS_LEVEL: int = 2  # Magnus series truncation (2 or 4)
HAMILTONIAN_PADE_SCALING: int = 4  # Padé matrix exp scaling iterations

# Phase 17.4: Neural Quantum State (NQS) Ansatz (HIGH)
# Complex-valued wavefunction representation via RBM
HAMILTONIAN_ENABLE_NQS: bool = True  # Enable NQS wavefunction mode
HAMILTONIAN_NQS_HIDDEN_DIM: int = 32  # RBM hidden dimension

# Phase 17.5: Time Crystal-Enhanced VQC (MEDIUM)
# Many-body localization for enhanced evolution time prediction
HAMILTONIAN_MBL_ENABLED: bool = True  # Enable MBL-enhanced VQC

# Phase 17.6: sPHNN Structure (Port-Hamiltonian) (MEDIUM)
# Port-Hamiltonian framework with Lyapunov stability guarantees
HAMILTONIAN_ENABLE_SPHNN: bool = True  # Enable sPHNN mode
HAMILTONIAN_SPHNN_DISSIPATION: float = 0.01  # Dissipation strength
HAMILTONIAN_SPHNN_MIN_DISSIPATION: float = 1e-6  # Minimum dissipation

# Soft-Potential Regularization (Phase 3.2 - existing)
HAMILTONIAN_REG_ALPHA: float = 0.0  # Regularization strength (0=disabled)
HAMILTONIAN_REG_SCALE: float = 10.0  # Characteristic scale

# =============================================================================
# PHASE 17 QWT TOKENIZER ENHANCEMENTS
# =============================================================================
# These parameters control the 5 major QWT enhancements for improved performance
# and accuracy. All maintain O(n) linear complexity and float32 precision.

# Phase 17.1: Lifting Scheme DWT (CRITICAL)
# Replaces FIR convolution DWT with lifting scheme for ~50% fewer FLOPs.
# The lifting scheme performs predict/update steps in-place.
QWT_USE_LIFTING_SCHEME: bool = True  # Enable lifting scheme (default: True)

# Phase 17.2: Padé[m/m] Matrix Exponential (HIGH)
# Replaces 2nd-order Cayley approximation with higher-order Padé.
# Order 4 provides 8th-order accuracy, enabling larger step sizes.
QWT_PADE_ORDER: int = 4  # Padé order (1=Cayley, 2-4 supported)

# Phase 17.3: Jacobi Preconditioner (HIGH)
# Adds diagonal preconditioning to BiCGSTAB solver for 2-4x fewer iterations.
# Negligible O(n) overhead with significant convergence improvement.
QWT_USE_JACOBI_PRECONDITIONER: bool = True  # Enable Jacobi preconditioning

# Phase 17.4: Skip-Connection Hamiltonian (MEDIUM)
# Extends the Hamiltonian beyond adjacent-node coupling with learnable
# skip connections for improved long-range dependency capture.
QWT_SKIP_STRIDE: int = 0  # Skip connection stride (0 = disabled)
QWT_MAX_SKIPS_PER_NODE: int = 2  # Maximum skip connections per node

# Phase 17.5: Parallel DWT Levels (MEDIUM)
# Restructures multi-scale DWT to compute levels in parallel where possible.
# Uses parallel::ForShard across (batch, level) pairs for better utilization.
QWT_PARALLEL_DWT_LEVELS: bool = True  # Enable parallel cascade processing

# =============================================================================
# PHASE 10 ARCHITECTURE ENHANCEMENT PARAMETERS
# =============================================================================
# These parameters control the "thought simulation" capabilities added in Phase 10.

# Thinking Token Configuration
THINKING_TOKENS_ENABLED: bool = True  # Enable thinking token injection

# Multi-Scale Wavelet Configuration
NUM_WAVELET_LEVELS: int = 3  # Number of cascaded DWT levels (1-5)

# Latent Reasoning Configuration
NUM_THOUGHT_STEPS: int = 4  # Iterative refinement steps in latent space

# Tokenizer Mode Configuration
TOKENIZER_MODE: str = "semantic"  # "semantic" | "byte_stream"

# Superword Merger Configuration
ENABLE_SUPERWORDS: bool = True  # Enable semantic n-gram grouping
SUPERWORD_MIN_FREQUENCY: int = 100  # Minimum n-gram frequency to learn
SUPERWORD_MAX_VOCAB_SIZE: int = 100000  # Maximum superwords to learn (100k capacity)
SUPERWORD_MIN_NGRAM: int = 2  # Minimum n-gram size
SUPERWORD_MAX_NGRAM: int = 5  # Maximum n-gram size

# Byte-Stream Frontend Configuration
BYTE_STREAM_STRIDE_FACTOR: int = 4  # Reduces byte-level seq length by 4x

# =============================================================================
# PHASE 48+: INTELLIGENT VOCAB CONTROLLER (Quantum Tokenization Pipeline)
# =============================================================================
# Controls the IntelligentVocabController which automatically determines
# effective vocabulary size from the tokenizer's learned codebook.
# Replaces manual vocab_size slider with automatic determination.

USE_INTELLIGENT_VOCAB_CONTROLLER: bool = True  # Enable automatic vocab sizing
VOCAB_CONTROLLER_AUTO_TRAIN: bool = True  # Auto-train codebook from curriculum data
VOCAB_CONTROLLER_SAMPLE_SIZE: int = (
    2000  # Corpus sample size for training (reduced from 10K for memory)
)
VOCAB_CONTROLLER_MIN_NGRAM_FREQ: int = 10  # Minimum n-gram frequency

# =============================================================================
# PHASE 48+: MEMORY OPTIMIZATION FLAGS
# =============================================================================
# Controls memory-efficient training features.

USE_DUAL_PATH_EMBEDDING: bool = True  # Use DualPathEmbedding instead of HyperdimensionalEmbedding
# DualPathEmbedding uses standard embedding for active vocab + HDE for rare tokens
# Reduces embedding memory by ~90% (from 2GB to ~200MB for 120K vocab)
DUAL_PATH_ACTIVE_VOCAB_SIZE: int = 10000  # Tokens to use standard embedding for

USE_GRADIENT_CHECKPOINTING: bool = True  # Enable gradient checkpointing for reasoning blocks
# Trades compute for memory by recomputing activations during backward pass
# Reduces peak memory by ~40-50% at ~20% compute cost

# Phase 201.1: HD Activation Checkpointing
# Uses holographic encoding for activation storage instead of full tensors
# Further reduces memory by 2-4x on top of standard gradient checkpointing
USE_HD_ACTIVATION_CHECKPOINT: bool = True  # Enable holographic activation encoding
HD_ACTIVATION_DIM: int = 512  # Holographic encoding dimension
HD_ACTIVATION_CTQW: bool = True  # Enable CTQW spreading for noise robustness

# Phase 201.4: HD Shared Expert Basis
# Enables hyperdimensional shared basis for MoE experts, reducing memory
# by having experts share a common HD-encoded basis with per-expert coefficients
USE_HD_SHARED_EXPERT_BASIS: bool = True  # Enable HD shared basis for MoE
HD_SHARED_BASIS_DIM: int = 256  # HD dimension for shared basis
HD_SHARED_BASIS_NUM_VECTORS: int = 32  # Number of HD basis vectors

# Phase 201.13: Adaptive Superposition Dimension
# Dynamically scales superposition dimension based on input complexity
# Complex inputs (high entropy) get larger superposition for more capacity
# Simple inputs (low entropy) use smaller superposition for efficiency
USE_ADAPTIVE_SUPERPOSITION: bool = True  # Enable adaptive superposition dimension
SUPERPOSITION_MIN_DIM: int = 2  # Minimum superposition dimension
SUPERPOSITION_MAX_DIM: int = 8  # Maximum superposition dimension
SUPERPOSITION_COMPLEXITY_SCALE: float = 1.0  # Scale factor for complexity

# Phase 201.5: Native Sparse Attention for Long Sequences
# Routes to DeepSeek NSA for sequences >= min length, O(n log n) vs O(n²)
USE_NATIVE_SPARSE_ATTENTION: bool = True  # Enable NSA routing for long sequences
NATIVE_SPARSE_BLOCK_SIZE: int = 64  # Block compression size
NATIVE_SPARSE_SELECTED_BLOCKS: int = 8  # Blocks to attend after compression
NATIVE_SPARSE_SELECTED_TOKENS: int = 32  # Tokens per block for fine-grained
NATIVE_SPARSE_SLIDING_WINDOW: int = 128  # Local attention window
NATIVE_SPARSE_MIN_SEQ_LEN: int = 4096  # Only use NSA for sequences >= this

# Phase 201.6: Sweep Executor Memory Compression
# Reduces memory for large HPO sweeps via LRU eviction and checkpoint compression
SWEEP_MAX_IN_MEMORY_TRIALS: int = 50  # Max completed trials to keep in RAM
SWEEP_COMPRESS_CHECKPOINTS: bool = True  # Use gzip for checkpoint files
SWEEP_USE_HD_TRIAL_ENCODING: bool = True  # HD-encode trial configs for similarity

# Phase 201.7: HD Superposition Embedding
# Unifies tokenizer superposition with HD embedding for quality enhancement
USE_HD_SUPERPOSITION_EMBEDDING: bool = True  # Enable superposition embedding path
HD_SUPERPOSITION_BRANCHES: int = 4  # Max superposition tokenizations to process

# Phase 201.8: State Bus HD Checkpoint Integration
# Store HD-encoded activations in State Bus slots for unified memory path
HD_CHECKPOINT_USE_STATE_BUS: bool = True  # Store HD bundles in State Bus
HD_CHECKPOINT_BUS_SLOTS: int = 8  # Number of State Bus slots for checkpoints

USE_CHUNKED_FORWARD: bool = False  # Enable chunked forward passes for O(1) memory
# Processes long sequences in chunks with gradient accumulation
# Combined with gradient checkpointing achieves constant memory training
CHUNKED_FORWARD_SIZE: int = 512  # Chunk size for chunked forward passes

GALORE_ADAPTIVE_RANK: bool = True  # Dynamically adjust GaLore rank based on model size
# Larger models get higher rank (up to GALORE_MAX_RANK), smaller models use less
GALORE_MAX_RANK: int = 128  # Maximum rank for adaptive GaLore

# Phase 201.9: HD + QMR Combined Checkpointing
# Combines holographic encoding with logarithmic checkpointing for multiplicative savings
# Memory reduction: O(log n / compression_ratio) vs O(n) standard checkpointing
USE_HD_QMR_COMBINED: bool = True  # Enable combined HD + logarithmic checkpointing
HD_QMR_LOG_FACTOR: int = 2  # Log base for checkpoint spacing (2 = log2(n) checkpoints)
HD_QMR_MIN_LAYERS: int = 4  # Minimum layers to activate combined strategy
HD_QMR_QUALITY_THRESHOLD: float = 0.95  # Minimum reconstruction quality before fallback

# Phase 201.10: Unified Tensor Budget Controller
# Orchestrates rank allocation across all tensor-decomposed layers (TT, Tucker, TensorRing)
# Based on Fisher Information importance and global memory budget
USE_TENSOR_BUDGET_CONTROLLER: bool = True  # Enable unified tensor budget allocation
TENSOR_BUDGET_MB: float = 1000.0  # Memory budget for all TN layers (MB)
TENSOR_BUDGET_REALLOC_INTERVAL: int = 1000  # Steps between rank reallocation
TENSOR_BUDGET_MIN_RANK: int = 4  # Minimum rank for any layer
TENSOR_BUDGET_MAX_RANK: int = 64  # Maximum rank for any layer
TENSOR_BUDGET_LM_HEAD_PRIORITY: float = 1.5  # Priority multiplier for LM head
TENSOR_BUDGET_EMBEDDING_PRIORITY: float = 1.3  # Priority multiplier for embeddings
TENSOR_BUDGET_MOE_PRIORITY: float = 1.0  # Priority multiplier for MoE layers

# Phase 201.11: Fisher-based TN Rank Allocation
# Extends FisherLayerGrouper to allocate ranks for TT/Tucker/TensorRing layers
# Layers with higher Fisher importance get larger ranks for capacity preservation
USE_FISHER_TN_RANKS: bool = True  # Enable Fisher-based TN rank allocation
FISHER_TN_RANK_MIN: int = 4  # Minimum TN rank regardless of Fisher score
FISHER_TN_RANK_MAX: int = 32  # Maximum TN rank for high-importance layers
FISHER_TN_RANK_SCALE: float = 1.0  # Scale factor for rank computation

# Phase 201.12: GaLore TT-Manifold Projection
# Projects gradients to TT tangent space for better convergence on TT layers
# Ensures gradient updates stay compatible with TT structure
GALORE_TT_MANIFOLD_PROJECTION: bool = True  # Enable TT tangent space projection
GALORE_TT_PROJECTION_EPS: float = 1e-6  # Numerical stability for projection


# =============================================================================
# PHASE 12 REASONING ENHANCEMENT PARAMETERS
# =============================================================================
# These parameters control the advanced reasoning capabilities added in Phase 12.
# All enhancements maintain O(L) linear complexity for CPU inference.

# Phase 12.1: Adaptive Exit (ACT-Lite)
PONDER_COST: float = 0.01  # Regularization weight for ponder cost

# Phase 12.2: Thought Memory Attention
THOUGHT_MEMORY_SIZE: int = 8  # Number of compressed thought slots

# Phase 12.3: Hierarchical WLAM (Enhanced in WLAM C++ Integration)
WLAM_NUM_LEVELS: int = 3  # Number of wavelet decomposition levels (1-5)
WLAM_USE_LIFTING: bool = True  # Lifting scheme: learnable predict/update wavelets (C++ op)
WLAM_SCATTERING_LAYERS: int = 0  # Scattering transform layers (0=disabled)
WLAM_SCATTERING_POOL: int = 4  # Scattering average pooling size
WLAM_USE_CROSS_ATTN: bool = True  # Cross-frequency attention

# Phase 12.4: Gated External Memory (GEM)
USE_EXTERNAL_MEMORY: bool = True  # Enable external memory bank
MEMORY_SLOTS: int = 64  # Number of memory slots (fixed, O(1))
MEMORY_SLOT_DIM: int = 256  # Dimension of each memory slot

# Phase 12.4 Enhancement 1: Surprise-Based Write Gating (Titans-Inspired) [CRITICAL]
# Only write to memory when input significantly differs from memory prediction
MEMORY_USE_SURPRISE_GATING: bool = True  # Enable surprise-based write gating
MEMORY_SURPRISE_THRESHOLD: float = 0.1  # Threshold for surprise signal
MEMORY_LEARNABLE_THRESHOLD: bool = True  # Make threshold learnable

# Phase 12.4 Enhancement 2: Product-Key Memory (Sub-Linear Lookup) [CRITICAL]
# Two-level key decomposition for O(√M) instead of O(M) lookup
MEMORY_USE_PRODUCT_KEYS: bool = True  # Enable product-key decomposition (O(√M) lookup)
MEMORY_PRODUCT_K: int = 16  # Top-k per sub-codebook
MEMORY_SUBKEY_DIM: int | None = None  # Sub-key dimension (None = slot_dim // 2)

# Phase 12.4 Enhancement 3: Multi-Head External Memory [HIGH]
# Multiple parallel read/write heads for increased capacity utilization
MEMORY_NUM_HEADS: int = 4  # Number of memory heads (1 = single-head)
MEMORY_HEAD_DIM: int | None = None  # Per-head dim (None = slot_dim // num_heads)

# Phase 12.4 Enhancement 4: TT-Decomposed Projections [MEDIUM]
# Use Tensor-Train factorization for projection matrices
USE_TT_MEMORY_PROJECTIONS: bool = True  # Enable TT-factorized projections
MEMORY_TT_RANK: int = 16  # TT decomposition rank

# Phase 12.4 Enhancement 5: Quantum-Inspired Associative Memory (QIAM) [RESEARCH]
# MPS-based quantum-inspired memory with exponential capacity potential
USE_QUANTUM_MEMORY: bool = True  # Enable quantum-inspired memory
MEMORY_MPS_BOND_DIM: int = 32  # MPS bond dimension for quantum memory

# Phase 12.4 Enhancement 6: Sparse Memory Finetuning [HIGH]
# Update only top-k most activated slots during training
MEMORY_USE_SPARSE_TRAINING: bool = True  # Enable sparse training updates
MEMORY_SPARSE_TOPK: int = 8  # Number of slots to update during training
MEMORY_EXPLORATION_PROB: float = 0.1  # Probability of random slot exploration

# Phase 12.5: Differential State Sharpening
SHARPEN_INTERVAL: int = 256  # Steps between state sharpening

# Phase 12.6: Entropy-Guided Thought Allocation
UNCERTAINTY_THRESHOLD: float = 0.5  # Threshold for position-level uncertainty

# Phase 12.7: Rotary Position Embeddings (RoPE)
USE_ROPE: bool = True  # Enable RoPE for SSM blocks

# Phase 12.8: Configuration-Level Optimizations (Guidance)
# -----------------------------------------------------------------------------
# These existing parameters can be tuned for enhanced reasoning:
#
# NUM_THOUGHT_STEPS: int = 4
#   → Recommended: 6-8 for complex reasoning tasks
#   → Effect: More internal hops per token in LatentReasoningBlock
#
# NUM_WAVELET_LEVELS: int = 3
#   → Recommended: 4-5 for document-level tasks
#   → Effect: Captures longer-range dependencies in tokenizer
#
# QWT_CTQW_STEPS: int = 0 (disabled)
#   → Recommended: 4-8 to enable quantum dynamics
#   → Effect: Semantic energy spreading on token graph
#   → Caution: Tune evolution_time carefully via HPO
#
# TOP_K: int = 2
#   → Recommended: 3-4 for knowledge-diverse tasks
#   → Effect: Activates more experts per token
#   → Trade-off: Compute scales linearly with TOP_K
#
# MAMBA2_STATE_DIM: int = 16
#   → Recommended: 32-64 for very long context (1M+)
#   → Effect: Higher fidelity long-term memory
#   → Trade-off: Memory usage scales with state dim
# -----------------------------------------------------------------------------

# Phase 12.9: Training Configuration Optimizations
CURRICULUM_PHASES: list = [
    {"context_len": 1024, "learning_rate": 1e-4, "steps": 10000},
    {"context_len": 16384, "learning_rate": 5e-5, "steps": 20000},
    {"context_len": 1000000, "learning_rate": 1e-5, "steps": 50000},
]
# Recommended gradient_accumulation_steps: 4-8 for long-context training
# Recommended warmup_steps: 2000-5000 for larger models

# Phase 12.10: Sparse State Propagation
USE_SPARSE_STATE: bool = True  # Enable importance-gated state updates

# Phase 12.11: Cross-Block State Bus
USE_STATE_BUS: bool = True  # Enable global communication bus
STATE_BUS_DIM: int = 64  # Dimension of bus communication
STATE_BUS_SLOTS: int = 8  # Number of bus slots (O(1))
USE_FUSED_STATE_BUS: bool = True  # Prefer C++ fused State Bus op when available

# Phase 12.11 Enhancement 2: Adaptive Slot Count (AdaSlot)
# Allows dynamic slot count per-input using differentiable slot selection
STATE_BUS_MAX_SLOTS: int = 16  # Maximum slot count for adaptive selection
STATE_BUS_ADAPTIVE: bool = True  # Enable adaptive slot selection
STATE_BUS_GUMBEL_TEMP: float = 1.0  # Gumbel-Softmax temperature (anneals during training)
STATE_BUS_GUMBEL_HARD: bool = False  # Use hard samples during inference

# Phase 12.11 Enhancement 3: Typed/Hierarchical Slots
# Specializes slots by type with learnable type embeddings
STATE_BUS_NUM_TYPES: int = 4  # Number of slot types
STATE_BUS_TYPED: bool = True  # Enable typed slots

# Phase 12.11 Enhancement 4: Quantum-Inspired Slot Superposition
# Represents slots in superposition using TT decomposition
STATE_BUS_SUPERPOSITION: bool = True  # Enable superposition slots
STATE_BUS_BOND_DIM: int = 4  # TT bond dimension for superposition

# Phase 12.11 Enhancement 5: Multi-Head Slot Attention
# Multiple attention heads over slots for parallel read patterns
STATE_BUS_NUM_HEADS: int = 2  # Attention heads for reads

# Phase 12.11 Enhancement 6: ReasoningModule Integration
# Wire GlobalStateBus into ReasoningModule for cross-block communication
STATE_BUS_INJECT_CONTEXT: bool = True  # Inject read context into blocks

# Phase 12.12: Dynamic Expert Specialization Probing
ENABLE_EXPERT_PROBING: bool = True  # Enable specialization-aware routing

# Phase 12.13: Hierarchical Thought Steps
HIERARCHICAL_THOUGHT_LEVELS: list = [2, 4, 8]  # Coarse → Fine thought pyramid

# Phase 12.14: Wavelet-Guided MoE Routing
USE_WAVELET_ROUTING: bool = True  # Enable frequency-aware expert routing

# Phase 12.15: Temporal Aliasing Prevention
STATE_CRYSTALLIZE_INTERVAL: int = 100000  # Steps between state crystallization

# Phase 12.16: Thought Token Semantic Coupling
THINKING_TOKEN_MULTIPLIER: float = 2.0  # Thought budget multiplier for <think> tokens


# =============================================================================
# TOOL INTEGRATION PARAMETERS
# =============================================================================
SIMULATION_TOOL_NAME = "run_simulation"  # Simulation tool name
MAX_TOOL_PAYLOAD_CHARS = 24576  # Max tool payload size

# =============================================================================
# SELECTIVE SCAN CACHE
# =============================================================================
SELECTIVE_SCAN_CACHE_MAX_SEQ_LEN = 4096  # Max cached sequence length


# =============================================================================
# PHASE 13 LINEAR EFFICIENCY PARAMETERS
# =============================================================================
# These parameters control inference optimizations and efficiency improvements
# added in Phase 13. All maintain O(L) or O(L log L) complexity for CPU viability.

# Phase 13.1: Mamba-2 SSD (State Space Duality)
USE_MAMBA2_SSD: bool = True  # Use SSD formulation for faster training
MAMBA2_SSD_HEADDIM: int = 64  # Head dimension for SSD matmul pattern
USE_SSD_FORMULATION: bool = True  # Alias for USE_MAMBA2_SSD
SSD_CHUNK_SIZE: int = 64  # Chunk size for SSD parallel computation

# Phase 13.3: Speculative Decoding
SPECULATIVE_ENABLED: bool = True  # Enable speculative decoding
SPECULATIVE_TOKENS: int = 4  # Number of tokens to speculate ahead
SPECULATIVE_TEMPERATURE: float = 1.0  # Temperature for sampling

# Phase 13.7: Flash Linear Attention
FLASH_LINEAR_CHUNK_SIZE: int = 64  # Chunk size for memory-efficient backward

# Phase 13.2: State Caching
USE_STATE_CACHE: bool = True  # Enable SSM state caching for O(1) generation
STATE_CACHE_MAX_LEN: int = 1000000  # Maximum cached sequence length

# Phase 13.3: Speculative Decoding
USE_SPECULATIVE: bool = False  # Enable speculative decoding (requires draft model)
SPECULATIVE_LOOKAHEAD: int = 5  # Number of tokens to speculate ahead
DRAFT_MODEL_PATH: str = ""  # Path to draft model checkpoint

# =============================================================================
# QUANTUM SUPERPOSITION GENERATION (QSG) PARAMETERS
# =============================================================================
# QSG COMPLETELY REPLACES autoregressive generation with quantum-inspired
# parallel generation, achieving 3000x+ speedup while maintaining quality.
# All operations are O(n) or O(n²) with sparse optimizations. Float32 only.
#
# NOTE: Autoregressive generation has been REMOVED from the codebase.
# These parameters tune QSG behavior - there is no fallback.

QSG_BOND_DIM: int = 32  # MPS bond dimension for entanglement
QSG_COHERENCE_RANGE: int = 64  # Max distance for position coherence (-1 = all)
QSG_GROVER_ITERATIONS: int = 3  # Grover amplitude amplification iterations
QSG_JACOBI_ITERATIONS: int = 2  # Jacobi consistency refinement iterations
QSG_HOPFIELD_BETA: float = 1.0  # Modern Hopfield inverse temperature
QSG_AMPLIFICATION_STRENGTH: float = 1.5  # Grover amplification factor (1.0-2.0)

# Phase 13.4: Quantization-Aware Training
QUANTIZATION_BITS: int = 16  # Quantization bits (4, 8, or 16)
USE_QAT: bool = False  # Enable quantization-aware training

# Phase 13.5: xLSTM (Extended LSTM)
USE_XLSTM: bool = True  # Use xLSTM instead of Mamba in blocks
XLSTM_VARIANT: str = "mlstm"  # "slstm" (scalar) or "mlstm" (matrix memory)

# xLSTM Enhancement Flags (Phase 13.5+)
XLSTM_USE_TFLA_KERNEL: bool = True  # Enhancement 1: TFLA-style C++ kernel for mLSTM
XLSTM_QUANTUM_GATES: bool = True  # Enhancement 2: Quantum-enhanced exponential gating
XLSTM_MPS_MEMORY: bool = True  # Enhancement 3: MPS-compressed matrix memory
XLSTM_MPS_BOND_DIM: int = 32  # Enhancement 3: MPS bond dimension (χ)
XLSTM_SPARSE_UPDATES: bool = True  # Enhancement 4: Dynamic sparse neuron updates
XLSTM_SPARSE_TOPK_RATIO: float = 0.5  # Enhancement 4: Ratio of neurons to update (0.5 = 50%)
XLSTM_LOG_SPACE_PARALLEL: bool = True  # Enhancement 5: Log-space parallel sequence evaluation

# Phase 13.6: Local Attention Hybrid
USE_LOCAL_ATTENTION: bool = True  # Enable Griffin-style local attention
LOCAL_ATTENTION_WINDOW: int = 256  # Window size for local attention

# Phase 13.6 Enhancement 1: Multi-Scale Window Attention (MSWA)
LOCAL_ATTENTION_MULTI_SCALE: bool = True  # Enable per-head variable window sizes
LOCAL_ATTENTION_WINDOW_MIN: int = 64  # Smallest window (head 0)
LOCAL_ATTENTION_WINDOW_MAX: int = 512  # Largest window (head N-1)

# Phase 13.6 Enhancement 2: Sigmoid Attention (SWAT-inspired)
LOCAL_ATTENTION_USE_SIGMOID: bool = True  # Replace softmax with sigmoid
LOCAL_ATTENTION_SIGMOID_TEMP: float = 1.0  # Temperature scaling for sigmoid

# Phase 13.6 Enhancement 3: ALiBi Position Bias
LOCAL_ATTENTION_USE_ALIBI: bool = True  # Enable ALiBi position encoding
LOCAL_ATTENTION_ALIBI_SLOPE_BASE: float = 8.0  # Base for slope computation

# Phase 13.6 Enhancement 4: Block-Sparse Optimization
LOCAL_ATTENTION_BLOCK_SPARSE: bool = True  # Enable block-sparse pruning
LOCAL_ATTENTION_BLOCK_SIZE: int = 32  # Block size for sparse attention
LOCAL_ATTENTION_SPARSITY_RATIO: float = 0.5  # Ratio of blocks to keep (0.5 = 50%)

# Phase 13.6 Enhancement 5: Quantum Kernel Approximation (Research)
LOCAL_ATTENTION_QUANTUM_KERNEL: bool = True  # Enable quantum-inspired attention
LOCAL_ATTENTION_QUANTUM_RANK: int = 16  # Low-rank approximation for O(n)

# Phase 13.6 Enhancement 6: Mask Caching
LOCAL_ATTENTION_CACHE_MASK: bool = True  # Cache masks for recurring sequences

# Phase 13.7: Flash Linear Attention
USE_FLASH_LINEAR_ATTENTION: bool = True  # Memory-efficient attention backward

# Phase 13.8: KV-Free State Streaming
USE_STATE_STREAMING: bool = True  # Enable infinite context via state compression
STATE_COMPRESSION_RATIO: float = 0.1  # Target compression ratio for state

# Phase 13.9: Grouped-Query Attention (GQA)
USE_GQA: bool = True  # Enable grouped-query attention
GQA_NUM_KV_HEADS: int = 4  # Number of key-value heads (shared across query heads)

# Phase 15: Enhanced GQA (O(n) Linear Complexity)
# These parameters control the GQA upgrade from O(n²) to O(n) complexity.

# Phase 15.1: Linear GQA via Kernel Approximation (CRITICAL)
GQA_USE_LINEAR: bool = True  # Enable O(n) linear attention (default: enabled)
GQA_FEATURE_MAP: str = "favor"  # Feature map: "elu", "exp", "favor"
GQA_RANDOM_FEATURES: int = 256  # Random features for FAVOR variants

# Phase 15.2: Tensor Product Attention (TPA) Integration (CRITICAL)
GQA_USE_TPA: bool = True  # Enable tensor product attention (10x+ KV cache reduction)
GQA_TPA_RANK: int = 4  # Tensor decomposition rank
GQA_TPA_CONTEXT_DIM: int = 64  # Context embedding dimension

# Phase 15.3: Sliding Window Hybrid GQA (HIGH)
GQA_USE_SLIDING_WINDOW: bool = True  # Enable sliding window mode
GQA_WINDOW_SIZE: int = 512  # Local attention window size
GQA_GLOBAL_TOKENS: int = 32  # Sparse global attention positions

# Phase 15.4: Quantum-Enhanced Attention (RESEARCH)
GQA_EXPERIMENTAL_QUANTUM: bool = True  # Enable quantum-enhanced attention (bugs fixed)
GQA_QUANTUM_QUBITS: int = 8  # Number of qubits for quantum attention
GQA_QUANTUM_LAYERS: int = 2  # VQC circuit depth

# Phase 15.5: Asymmetric & Dynamic GQA (MEDIUM)
GQA_ASYMMETRIC: bool = True  # Enable asymmetric KV head assignment
GQA_LEARNED_GROUPING: bool = True  # Enable learned grouping weights
GQA_PER_LAYER_KV_HEADS: list | None = None  # Per-layer KV head counts

# Phase 13.10: Tool Use / Function Calling
ENABLE_TOOL_USE: bool = True  # Enable function calling capabilities
MAX_TOOL_CALLS: int = 10  # Maximum tool calls per generation
TOOL_SPECIAL_TOKENS: list = ["<tool>", "<call>", "<result>", "</tool>"]

# =============================================================================
# PHASE 16 CONTEXTUAL GATING COLLAPSE PARAMETERS
# =============================================================================
# These parameters control the collapse mechanism for superposition states.
# Used by ContextualGatingCollapse layer and fused MoE superposition ops.
# All operations maintain O(n) linear complexity and float32 precision.

# Phase 16.1: Gumbel-Softmax Unified Collapse Path (CRITICAL)
COLLAPSE_USE_GUMBEL_SOFTMAX: bool = True  # Unified differentiable train/infer path
COLLAPSE_GUMBEL_INITIAL_TEMP: float = 1.0  # Initial Gumbel temperature
COLLAPSE_GUMBEL_FINAL_TEMP: float = 0.1  # Final annealed temperature
COLLAPSE_GUMBEL_ANNEAL_STEPS: int = 10000  # Steps to anneal temperature
COLLAPSE_HARD_SAMPLES: bool = False  # Use straight-through hard samples

# Phase 16.2: Fused C++ Collapse Kernel (CRITICAL)
USE_FUSED_COLLAPSE_OP: bool = True  # Enable C++ SIMD kernel
FUSED_COLLAPSE_SIMD_THRESHOLD: int = 8  # Min superposition size for SIMD

# Phase 16.3: Kernel Attention Linearization (HIGH)
COLLAPSE_USE_KERNEL_ATTENTION: bool = True  # Use kernel feature maps
COLLAPSE_KERNEL_FEATURE_MAP: str = "elu_plus_one"  # "elu_plus_one", "relu_squared"
COLLAPSE_KERNEL_THRESHOLD: int = 16  # Use kernel when S > threshold

# Phase 16.4: TT Q/K/V Compression (MEDIUM)
COLLAPSE_USE_TT_PROJECTIONS: bool = True  # TT-factorized projections
COLLAPSE_TT_RANK: int = 8  # TT decomposition rank

# Phase 16.5: Adaptive Collapse Strategy (LOW - Experimental)
COLLAPSE_ADAPTIVE_STRATEGY: bool = True  # Entropy-based soft/hard blend
COLLAPSE_ENTROPY_THRESHOLD: float = 1.0  # Blend threshold


# Phase 13.11: RAG (Retrieval-Augmented Generation)
USE_RAG: bool = False  # Enable retrieval-augmented generation
RAG_TOP_K: int = 5  # Number of documents to retrieve
RAG_INDEX_PATH: str = ""  # Path to FAISS/Annoy index

# Phase 13.12: Model Distillation
DISTILLATION_TEMPERATURE: float = 3.0  # Temperature for soft targets
DISTILLATION_ALPHA: float = 0.5  # Weight for distillation loss vs task loss


# =============================================================================
# PHASE 14 ADVANCED ARCHITECTURE PARAMETERS
# =============================================================================
# These parameters control advanced architecture enhancements added in Phase 14.
# All enhancements maintain O(n) linear complexity and use float32/64 precision.
# CONSTRAINT: No quantization - all operations use full float32/float64 precision.

# Phase 14.0: CPU Optimization Foundation
USE_NOMAD_ATTENTION: bool = False  # NoMAD-style attention with float32 lookup tables
NOMAD_TABLE_SIZE: int = 256  # Lookup table entries (256 for float32)
CACHE_TILE_SIZE: int = 64  # L2 cache-friendly tile size for tiling
PREFETCH_DISTANCE: int = 8  # Software prefetch lookahead distance

# Phase 14.1: Linear SSM Enhancements
USE_DATA_DEPENDENT_SHIFT: bool = True  # RWKV-6 data-dependent token shift
TOKEN_SHIFT_DECAY: float = 0.5  # Shift decay factor
USE_RG_LRU: bool = True  # Griffin RG-LRU (alternative to Mamba)
RG_LRU_STATE_DIM: int = 256  # RG-LRU hidden state dimension

# RG-LRU Enhancements (Phase 14.1.2)
# Enhancement 1: Parallel Prefix Scan (CRITICAL) - 2-4x training speedup
RG_LRU_USE_PARALLEL_SCAN: bool = True  # Enable parallel prefix scan for training
RG_LRU_SCAN_CHUNK_SIZE: int = 64  # Chunk size for work-efficient Blelloch scan

# Enhancement 2: Matrix-Valued State Extension (HIGH) - Improved in-context learning
RG_LRU_MATRIX_STATE: bool = True  # Enable low-rank matrix state H = UV^T
RG_LRU_STATE_RANK: int = 16  # Rank of factored state matrix (r << d)

# Enhancement 3: Selective Content-Aware Gating (HIGH) - Mamba-style selective forgetting
RG_LRU_SELECTIVE_GATING: bool = True  # Gate depends on input AND hidden state
RG_LRU_GATING_MODE: str = "concat"  # "concat", "delta", or "bilinear"

# Enhancement 4: Quantum-Inspired Rotation Gating (MEDIUM) - Gradient preservation
RG_LRU_QUANTUM_GATING: bool = True  # Use Givens rotations instead of sigmoid
RG_LRU_ROTATION_GROUPS: int = 2  # 2 for pairwise, 4+ for block rotations

# Enhancement 5: Tensor Train Weight Compression (MEDIUM) - 10-100x parameter reduction
RG_LRU_TT_COMPRESSION: bool = True  # Enable TT-factorized weight matrices
RG_LRU_TT_RANKS: list = [4, 4]  # TT-ranks for weight factorization

# Phase 14.2: Token Shift Enhancements (All five roadmap improvements)
TOKEN_SHIFT_MODE: str = (
    "standard"  # "standard", "simplified", "fourier", "delta", "hierarchical", "multi_position"
)
USE_SIMPLIFIED_TOKEN_SHIFT: bool = False  # RWKV-7 style 3x faster (no input-dependent gate)
USE_FOURIER_TOKEN_SHIFT: bool = True  # Global frequency patterns via FFT
FOURIER_SHIFT_MIN_SEQ_LEN: int = 256  # Only use FFT for sequences >= this length
USE_HIERARCHICAL_DECAY: bool = True  # Layer-position aware decay rates
HIERARCHICAL_DECAY_FACTOR: float = 2.0  # Scaling factor for hierarchy (decay^(1/(layer+factor)))
USE_DELTA_TOKEN_SHIFT: bool = False  # Gated delta rule (erase/write gates) for precise memory
TOKEN_SHIFT_DISTANCES: list = [1]  # Multi-position: [1,2,4] for multi-scale look-ahead

# MinGRU Configuration (Enhanced in Phase 17)
USE_MIN_GRU_BLOCKS: bool = True  # minGRU blocks (production-ready)
MIN_GRU_PARALLEL_CHUNK_SIZE: int = 256  # Parallel scan chunk size
MIN_GRU_USE_PARALLEL_SCAN: bool = True  # Use parallel scan for training (10-175x speedup)
MIN_GRU_DYNAMIC_RATIO: float = 1.0  # Fraction of dimensions to update (1.0=all, 0.5=half)
MIN_GRU_QUANTUM_INSPIRED: bool = True  # Enable gradient norm preservation
MIN_GRU_ORTHOG_INTERVAL: int = 64  # Steps between gradient normalization (0=disabled)
MIN_GRU_IMPORTANCE_THRESHOLD: float = 0.5  # Threshold for dynamic gating importance


# Phase 14.2: CPU-Optimized MoE Innovations
USE_SHARED_EXPERTS: bool = True  # Always-active shared experts (DeepSeek-style)
NUM_SHARED_EXPERTS: int = 2  # Number of shared experts (2-4 recommended)
EXPERT_SEGMENT_FACTOR: int = 4  # Fine-grained expert segmentation factor
SEGMENT_TOP_K: int = 8  # Top-K segments to select
USE_AUX_LOSS_FREE_BALANCE: bool = True  # Bias-based load balancing (no aux loss)
BALANCE_EMA_DECAY: float = 0.99  # EMA decay for load tracking
USE_ADAPTIVE_K: bool = True  # Dynamic expert count (Ada-K)
ADA_K_BASE: int = 2  # Base expert count
ADA_K_MIN: int = 1  # Minimum experts per token
ADA_K_MAX: int = 6  # Maximum experts per token
USE_SIGMOID_ROUTING: bool = True  # GLM-4.5 style sigmoid gating instead of softmax

# Phase 14.3: Advanced Reasoning Architecture
USE_GRPO_TRAINING: bool = True  # Group Relative Policy Optimization (training-only)
GRPO_NUM_SAMPLES: int = 4  # Samples per GRPO optimization step
USE_CONTINUOUS_THOUGHT: bool = True  # COCONUT continuous thought (latent reasoning)
CONTINUOUS_THOUGHT_STEPS: int = 4  # Number of thought iteration steps

# Phase 87: Enhanced CoCoNut Multi-Path Exploration
# Multi-path BFS exploration with Grover-inspired amplitude scoring
# Edition limits enforced at runtime: Lite max 8 paths, Pro/Enterprise unlimited
COCONUT_NUM_PATHS: int = 2  # Number of parallel thought paths (default 2)
COCONUT_PRUNE_THRESHOLD: float = 0.1  # Amplitude threshold for path pruning
COCONUT_COLLAPSE_THRESHOLD: float = 0.8  # Confidence for BFS→DFS collapse
COCONUT_CRYSTALLIZE_THRESHOLD: float = 0.9  # Threshold to freeze reasoning path
COCONUT_MAX_CRYSTALS: int = 64  # Max crystallized reasoning primitives (LRU eviction)
# HPO-tunable ranges:
#   - COCONUT_NUM_PATHS: Lite [1, 8], Pro [1, 32]
#   - COCONUT_COLLAPSE_THRESHOLD: [0.5, 0.95]
#   - COCONUT_CRYSTALLIZE_THRESHOLD: [0.7, 0.99]

USE_SELF_CONSISTENCY: bool = True  # Self-consistency verification (confidence scoring)
CONSISTENCY_THRESHOLD: float = 0.8  # Verification agreement threshold

# Phase 14.4: Quantum-Inspired Enhancements (Already CPU-Native)
VQC_ENCODING: str = "hybrid"  # VQC encoding: "amplitude", "angle", or "hybrid"
VQC_REUPLOADING_DEPTH: int = 2  # Data re-uploading depth for VQC
USE_MPS_TEMPORAL: bool = True  # MPS for temporal sequence modeling
MPS_BOND_DIM: int = 32  # MPS bond dimension (χ)
USE_HYBRID_TN: bool = True  # Hybrid Tensor Network with neural pre/post processing

# Phase 14.4.1: True MPS Contraction Configuration (Enhancement 1)
# These flags control the integration of C++ SIMD-optimized MPS kernel
MPS_USE_TRUE_CONTRACTION: bool = True  # Use C++ MPS kernel instead of GRU approximation
MPS_TRUNCATION_THRESHOLD: float = 1e-10  # SVD truncation threshold for bond truncation
MPS_COMPUTE_ENTROPY: bool = False  # Compute entanglement entropy at each bond
MPS_HYBRID_THRESHOLD: int = 128  # Seq length threshold: GRU for L<threshold, MPS for L>=threshold
MPS_USE_TDVP_GRADIENTS: bool = True  # Project gradients onto MPS tangent space (TDVP)
MPS_USE_TENSORIZED_ATTENTION: bool = (
    True  # Enable sub-quadratic O(n√n) attention (REQUIRED for linear scaling)
)
MPS_UNIFORM_MODE: bool = False  # Enable uniform MPS (translation invariant)

# Strict Linear Complexity Enforcement
# When True, prevents any O(n²) quadratic attention fallbacks
ENFORCE_LINEAR_ATTENTION: bool = True  # Enforce linear complexity in all attention ops

# Phase 14.5: Long-Context Innovations
USE_COMPRESSIVE_MEMORY: bool = True  # Compressive memory update for GEM
MEMORY_COMPRESSION_RATIO: float = 0.25  # Target compression ratio

# Phase 14.6: Neurosymbolic Reasoning (Optional)
USE_KNOWLEDGE_GRAPH: bool = True  # Knowledge graph integration
KG_ENTITY_DIM: int = 256  # Entity embedding dimension
KG_MAX_HOPS: int = 2  # Maximum graph traversal hops (keep low for CPU)


# =============================================================================
# PHASE 16 FLASH LINEAR ATTENTION ENHANCEMENTS
# =============================================================================
# These parameters control the enhanced Flash Linear Attention with all
# roadmap enhancements. All maintain O(n) complexity and float32/64 precision.

# Phase 16.1: Gated Linear Attention (GLA) - CRITICAL
# Uses 2D forget gates for selective memory: St = Gt ⊙ St-1 + kt ⊗ vt
FLASH_LINEAR_USE_GATING: bool = True  # Enable gated linear attention
FLASH_LINEAR_GATE_INIT_BIAS: float = -2.0  # exp(-exp(-2)) ≈ 0.87 initial retention

# Phase 16.2: Learnable Feature Maps (Random Maclaurin) - HIGH
# Replaces fixed ELU+1 with learnable spectral features
FLASH_LINEAR_FEATURE_MAP: str = "elu"  # "elu", "random_maclaurin", "learnable", "quantum"
FLASH_LINEAR_FEATURE_DIM_MULT: float = 2.0  # Feature dim = head_dim * mult

# Phase 16.3: Hybrid Sliding Window + Linear Attention - HIGH
# Combines local precision with global linear for best of both
FLASH_LINEAR_HYBRID_WINDOW: int = 0  # 0 disables hybrid mode
FLASH_LINEAR_HYBRID_ALPHA: float = 0.5  # Mix: α*local + (1-α)*linear

# Phase 16.4: Chunkwise Parallel Training - MEDIUM
# Parallel chunk processing for training speedup (2-3x on multi-core)
FLASH_LINEAR_TRAIN_CHUNK_SIZE: int = 256  # Chunk size for chunkwise parallel
FLASH_LINEAR_USE_CHUNKWISE: bool = True  # Enable chunkwise mode

# Phase 16.5: Quantum-Inspired Feature Maps - R&D
# Parameterized rotation matrices for non-classical correlations
FLASH_LINEAR_QUANTUM_INSPIRED: bool = True  # Enable quantum-inspired features
FLASH_LINEAR_QUANTUM_DEPTH: int = 4  # Number of rotation layers

# Phase 16.6: RALA (Rank-Augmented Linear Attention) - HIGH
# Low-rank augmentation for improved expressivity with minimal overhead
FLASH_LINEAR_AUGMENT_RANK: int = 4  # 0 disables RALA
FLASH_LINEAR_USE_RALA: bool = True  # Enable rank augmentation

# Phase 16.7: Forget Gate (Selective Forgetting) - HIGH
# Learnable forget gate that down-weights attention to distant tokens
FLASH_LINEAR_USE_FORGET_GATE: bool = True  # Enable forget gate mechanism
FLASH_LINEAR_FORGET_INIT: float = -2.0  # Initial bias (controls retention rate)

# =============================================================================
# PHASE 18 FRONTIER MEMORY INNOVATIONS
# =============================================================================
# These parameters control advanced memory and attention innovations for
# enhanced long-context processing. All maintain O(n) complexity.

# Phase 18.1: Latent KV Compression - CRITICAL
# Compresses K/V to low-dimensional latent space for 10-28x memory reduction
USE_LATENT_KV_ATTENTION: bool = True  # Enable latent compression
LATENT_KV_DIM: int = 64  # Latent dimension (smaller = more compression)
LATENT_KV_USE_QUANTUM: bool = True  # Use float64 quantum-enhanced compression
LATENT_KV_FEATURE_MAP: str = "elu"  # Feature map for linear attention: elu, relu, softmax

# Phase 18.2: Tensor Factorized Attention - HIGH
# Rank-factorized Q/K/V projections for memory-efficient attention
USE_TENSOR_FACTORIZED_ATTENTION: bool = True  # Enable tensor factorization
TENSOR_FACTORIZED_RANK: int = 4  # Factorization rank (lower = more efficient)
TENSOR_FACTORIZED_CONTEXT_DIM: int = 128  # Context projection dimension

# Phase 18.3: Adaptive Memory (Test-Time Learning) - EXPERIMENTAL
# Memory that updates its weights during inference based on surprise signals
USE_ADAPTIVE_MEMORY: bool = True  # Enable test-time weight updates
ADAPTIVE_MEMORY_LEARNING_RATE: float = 0.01  # Test-time learning rate
ADAPTIVE_MEMORY_SURPRISE_THRESHOLD: float = 0.5  # Surprise threshold for updates
ADAPTIVE_MEMORY_MLP_DIM: int = 256  # Hidden dimension for memory MLP


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for HighNoon language models.

    Attributes:
        vocab_size: Size of the vocabulary.
        embedding_dim: Dimension of token embeddings.
        num_reasoning_blocks: Number of reasoning blocks in the model.
            Lite edition capped at 24.
        max_seq_length: Maximum sequence length the model can process.
            Lite edition capped at 256k.
        num_moe_experts: Number of experts in MoE layers.
            Lite edition capped at 12.
        use_fused_ops: Whether to use fused C++ operations for performance.
        dtype: Data type for model parameters ('float32' or 'float16').
        block_pattern: Pattern for reasoning blocks ('transformer', 'mamba', 'hybrid').
        compressed_dim: Dimension after tensor compression.
        chunk_size: Size of chunks for sequence processing.
        chunk_stride: Stride between chunks.
        dropout: Dropout rate for regularization.
        attention_dropout: Dropout rate for attention layers.
        use_residual_connections: Whether to use residual connections.
    """

    vocab_size: int = 32000
    embedding_dim: int = 768
    num_reasoning_blocks: int = 4
    max_seq_length: int = 4096
    num_moe_experts: int = 8
    use_fused_ops: bool = True
    dtype: str = "float32"
    block_pattern: str = "transformer"
    compressed_dim: int = 256
    chunk_size: int = 512
    chunk_stride: int = 256
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_residual_connections: bool = True

    def __post_init__(self):
        """Validate configuration against Lite edition limits."""
        from highnoon._native._limits import validate_model_config

        validate_model_config(self)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_reasoning_blocks": self.num_reasoning_blocks,
            "max_seq_length": self.max_seq_length,
            "num_moe_experts": self.num_moe_experts,
            "use_fused_ops": self.use_fused_ops,
            "dtype": self.dtype,
            "block_pattern": self.block_pattern,
            "compressed_dim": self.compressed_dim,
            "chunk_size": self.chunk_size,
            "chunk_stride": self.chunk_stride,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "use_residual_connections": self.use_residual_connections,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for training HighNoon models.

    Attributes:
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        warmup_steps: Number of warmup steps for learning rate schedule.
        max_steps: Maximum number of training steps.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        max_grad_norm: Maximum gradient norm for clipping.
        weight_decay: Weight decay coefficient.
        curriculum_strategy: Strategy for curriculum learning.
        allowed_stages: List of allowed curriculum stages.
        checkpoint_interval: Steps between checkpoints.
        eval_interval: Steps between evaluations.
        log_interval: Steps between logging.
    """

    batch_size: int = 8
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    curriculum_strategy: str = "adaptive"
    allowed_stages: list[str] = field(
        default_factory=lambda: [
            "code_foundation",
            "code_instruction",
            "cli_specialist",
            "tool_use",
            "chat_integration",
            "reasoning_enhanced",
        ]
    )
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 10

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "curriculum_strategy": self.curriculum_strategy,
            "allowed_stages": self.allowed_stages,
            "checkpoint_interval": self.checkpoint_interval,
            "eval_interval": self.eval_interval,
            "log_interval": self.log_interval,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# =============================================================================
# UNIFIED CONFIGURATION
# =============================================================================


@dataclass
class Config:
    """Unified configuration for HighNoon Language Framework.

    This class combines model and training configurations into a single
    object for convenient management.

    Attributes:
        model: Model configuration.
        training: Training configuration.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to nested dictionary."""
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create configuration from nested dictionary."""
        model_config = ModelConfig.from_dict(config_dict.get("model", {}))
        training_config = TrainingConfig.from_dict(config_dict.get("training", {}))
        return cls(model=model_config, training=training_config)

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        import json

        with open(path) as f:
            config_dict = json.load(f)
        log.info(f"Configuration loaded from {path}")
        return cls.from_dict(config_dict)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================


def get_preset_config(name: str) -> Config:
    """Get a preset configuration by name.

    Available presets:
        - 'highnoon-small': Small model for development (125M params)
        - 'highnoon-base': Base model (350M params)
        - 'highnoon-large': Large model (1.3B params)
        - 'highnoon-3b': 3B parameter model
        - 'highnoon-7b': 7B parameter model (approaching Lite limit)

    Args:
        name: Name of the preset configuration.

    Returns:
        Config object with preset values.

    Raises:
        ValueError: If preset name is not recognized.
    """
    presets = {
        "highnoon-small": Config(
            model=ModelConfig(
                vocab_size=32000,
                embedding_dim=768,
                num_reasoning_blocks=4,
                max_seq_length=2048,
                num_moe_experts=4,
            ),
        ),
        "highnoon-base": Config(
            model=ModelConfig(
                vocab_size=32000,
                embedding_dim=1024,
                num_reasoning_blocks=8,
                max_seq_length=4096,
                num_moe_experts=8,
            ),
        ),
        "highnoon-large": Config(
            model=ModelConfig(
                vocab_size=32000,
                embedding_dim=2048,
                num_reasoning_blocks=12,
                max_seq_length=8192,
                num_moe_experts=8,
            ),
        ),
        "highnoon-3b": Config(
            model=ModelConfig(
                vocab_size=32000,
                embedding_dim=2560,
                num_reasoning_blocks=16,
                max_seq_length=16384,
                num_moe_experts=8,
            ),
        ),
        "highnoon-7b": Config(
            model=ModelConfig(
                vocab_size=32000,
                embedding_dim=4096,
                num_reasoning_blocks=24,
                max_seq_length=32768,
                num_moe_experts=12,
            ),
        ),
    }

    if name not in presets:
        raise ValueError(f"Unknown preset '{name}'. Available presets: {list(presets.keys())}")

    return presets[name]


# =============================================================================
# TOKENIZER FACTORY
# =============================================================================

# Cached tokenizer instance (singleton pattern)
_TOKENIZER_INSTANCE = None


def get_tokenizer(
    vocab_size: int | None = None,
    max_length: int | None = None,
    force_new: bool = False,
):
    """Get or create the global tokenizer instance.

    Factory function that returns a configured QWTTextTokenizer. Uses a
    singleton pattern by default to ensure consistent tokenization across
    the framework. The tokenizer can be customized via HPO or manual config.

    Args:
        vocab_size: Vocabulary size. Defaults to VOCAB_SIZE from config.
        max_length: Maximum sequence length. Defaults to MAX_CONTEXT_LEN.
        force_new: If True, create a new instance instead of using cached.

    Returns:
        Configured QWTTextTokenizer instance.

    Example:
        >>> tokenizer = get_tokenizer()
        >>> tokens = tokenizer("Hello, world!", return_tensors="tf")
        >>> tokenizer.vocab_size
        60000
    """
    global _TOKENIZER_INSTANCE

    # Use defaults from config if not specified
    vocab_size = vocab_size or VOCAB_SIZE
    max_length = max_length or MAX_CONTEXT_LEN

    # Return cached instance if available and not forcing new
    if _TOKENIZER_INSTANCE is not None and not force_new:
        # Check if cached tokenizer matches requested config
        if (
            _TOKENIZER_INSTANCE.vocab_size == vocab_size
            and _TOKENIZER_INSTANCE.model_max_length == max_length
        ):
            return _TOKENIZER_INSTANCE

    # Import here to avoid circular dependency
    from highnoon.tokenization import QWTTextTokenizer

    _TOKENIZER_INSTANCE = QWTTextTokenizer(
        vocab_size=vocab_size,
        model_max_length=max_length,
        enable_thinking_tokens=True,
    )

    log.info(f"Created tokenizer: vocab_size={vocab_size}, max_length={max_length}")

    return _TOKENIZER_INSTANCE


def reset_tokenizer():
    """Reset the cached tokenizer instance.

    Use this when changing tokenizer configuration (e.g., during HPO trials
    with different vocab sizes).
    """
    global _TOKENIZER_INSTANCE
    _TOKENIZER_INSTANCE = None
