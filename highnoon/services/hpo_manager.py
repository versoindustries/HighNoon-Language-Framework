"""HPO Trial Manager for orchestrating hyperparameter optimization.

This module manages the lifecycle of HPO trials, including spawning,
monitoring, and coordinating with the C++ HPO orchestrator.

Phase 5 Update: Integrated grouped parameter search for multi-stage HPO.
Phase 6 Update: Automated learning rate selection with optimizer-aware defaults.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from highnoon import config as hn_config
from highnoon.services.hpo_metrics import HPOMetricsCollector, OversizedConfigTracker, TrialStatus
from highnoon.services.hpo_utils import convert_numpy_types

# Phase 5: Import grouped parameter sampler (TPE-style forest sampling)
try:
    from highnoon.services.hpo_grouped_sampling import ParameterGroupSampler

    GROUPED_SAMPLING_AVAILABLE = True
except ImportError as e:
    GROUPED_SAMPLING_AVAILABLE = False
    # Only log at debug level - not critical for operation
    logging.getLogger(__name__).debug(f"[HPO Manager] Grouped sampling not available: {e}")

logger = logging.getLogger(__name__)

# =============================================================================
# OPTIMIZER-AWARE LEARNING RATE DEFAULTS
# =============================================================================
# Each optimizer has an optimal learning rate range based on its characteristics.
# Second-order optimizers (SophiaG) need lower LRs, while Lion is sensitive to high LRs.
# Log-uniform sampling ensures small LRs are as likely as large ones.

OPTIMIZER_LR_RANGES: dict[str, tuple[float, float]] = {
    "adam": (1e-5, 1e-3),
    "adamw": (1e-5, 3e-4),
    "sophiag": (1e-5, 1e-4),  # Conservative for second-order optimizer
    "qiao": (1e-5, 5e-4),  # Quantum-inspired alternating optimizer (mixer steps add stability)
    "grover": (1e-5, 5e-4),  # Grover-enhanced optimizer
    "sympflow": (1e-5, 5e-4),  # Symplectic Hamiltonian flow (symplectic integration is stable)
    "sympflowqng": (
        1e-5,
        2e-4,
    ),  # S12: SympFlow + QNG geodesic (conservative for geodesic corrections)
    "lion": (1e-5, 1e-4),  # Lion needs lower LR than Adam-family
}


# Default LR range when optimizer is not specified or unknown
DEFAULT_LR_RANGE: tuple[float, float] = (1e-5, 3e-4)


def load_hpo_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load HPO configuration from JSON file.

    Args:
        config_path: Path to config file. Defaults to artifacts/hpo_trials/hpo_config.json

    Returns:
        Dictionary with configuration values
    """
    if config_path is None:
        config_path = Path("artifacts/hpo_trials/hpo_config.json")

    if not config_path.exists():
        logger.warning(f"[HPO Manager] Config file not found: {config_path}, using defaults")
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        logger.info(f"[HPO Manager] Loaded config from: {config_path}")
        return config
    except Exception as exc:
        logger.error(f"[HPO Manager] Failed to load config: {exc}")
        return {}


def compute_max_vocab_for_budget(
    param_budget: int,
    embedding_dim: int = 256,
    use_hqe: bool = True,
    hd_dim: int | None = None,
    model_overhead_fraction: float = 0.5,
) -> int:
    """Compute maximum sustainable vocab_size for a given parameter budget.

    When HQE is enabled, embedding parameters dominate the model size.
    This function calculates the maximum vocab_size that leaves room for
    the rest of the model (blocks, experts, output layers).

    Args:
        param_budget: Target parameter budget (e.g., 100_000_000)
        embedding_dim: Model embedding dimension
        use_hqe: Whether Hyperdimensional Embedding is enabled
        hd_dim: Explicit HD dimension (defaults to embedding_dim * 8 if HQE)
        model_overhead_fraction: Fraction of budget reserved for non-embedding params

    Returns:
        Maximum vocab_size that fits within the budget.
    """
    # Reserve fraction for model layers (blocks, MoE, output)
    embedding_budget = int(param_budget * (1 - model_overhead_fraction))

    if use_hqe:
        # HQE: embed_params = vocab_size * hd_dim + hd_dim * embedding_dim
        effective_hd_dim = hd_dim or (embedding_dim * 8)
        projection_params = effective_hd_dim * embedding_dim
        vocab_budget = embedding_budget - projection_params
        max_vocab = vocab_budget // effective_hd_dim
    else:
        # Standard: embed_params = vocab_size * embedding_dim
        max_vocab = embedding_budget // embedding_dim

    # Ensure minimum viable vocab (at least 8K for basic tokenization)
    # Phase 200+: Increased max to 300K for large vocabulary models
    return max(8000, min(max_vocab, 300000))


def estimate_model_params(config: dict[str, Any]) -> int:
    """Estimate total model parameters from architecture configuration.

    Phase 11 Update: Support HQE (Hyperdimensional Embedding), QuantumLMHead,
    and MoE Tucker decomposition aware estimation.

    Phase 201.7 Update: Budget-aware vocab_size default when param_budget is set.

    Args:
        config: Architecture configuration containing:
            - vocab_size: Vocabulary size (default 32000)
            - embedding_dim: Embedding dimension (default 512)
            - num_reasoning_blocks: Number of reasoning blocks (default 8)
            - num_moe_experts: Number of MoE experts (default 8)
            - mamba_state_dim: Mamba state dimension (default 64)
            - use_hyperdimensional_embedding: Whether HQE is used
            - hd_dim: Explicit HQE dimension (defaults to embedding_dim * 8)
            - use_quantum_lm_head: Whether VQC-based head is used
            - use_td_moe: Whether Tucker-decomposed MoE is used
            - td_moe_tucker_rank: Rank for MoE decomposition
            - param_budget: Optional parameter budget for smart defaults

    Returns:
        Estimated total parameter count.
    """
    embedding_dim = config.get("embedding_dim") or config.get("hidden_dim") or 512
    use_hqe = config.get(
        "use_hyperdimensional_embedding", getattr(hn_config, "USE_HYPERDIMENSIONAL_EMBEDDING", True)
    )
    param_budget = config.get("param_budget")
    config_hd_dim = config.get("hd_dim")

    # Budget-aware vocab_size default: when param_budget is set and vocab_size is null,
    # compute max sustainable vocab instead of defaulting to 128K
    vocab_size = config.get("vocab_size")
    if vocab_size is None:
        if param_budget is not None:
            vocab_size = compute_max_vocab_for_budget(
                param_budget=param_budget,
                embedding_dim=embedding_dim,
                use_hqe=use_hqe,
                hd_dim=config_hd_dim,
                model_overhead_fraction=0.5,
            )
            logger.debug(
                f"[HPO] Budget-aware vocab default: {vocab_size} "
                f"(budget={param_budget/1e6:.0f}M, hqe={use_hqe})"
            )
        else:
            # No budget specified - use conservative default
            vocab_size = 32000
    num_blocks = config.get("num_reasoning_blocks") or 8
    num_experts = config.get("num_moe_experts") or 8
    mamba_state = config.get("mamba_state_dim") or 64

    # Phase 201.2: Account for HQE impact on embedding parameters
    # Note: use_hqe is already set above for budget-aware vocab calculation
    if use_hqe:
        # HQE uses hd_dim (default embedding_dim * 8) for base vectors
        # hd_dim = config.get("hd_dim") or (embedding_dim * 8)
        # Note: In hpo_trial_runner.py, hd_dim is calculated as hidden_dim * 8
        hd_dim = config.get("hd_dim") or (embedding_dim * 8)
        # Base vectors: [vocab_size, hd_dim]
        # Position keys: [max_seq, hd_dim] (small compared to base)
        # Projection: [hd_dim, embedding_dim]
        embed_params = vocab_size * hd_dim + hd_dim * embedding_dim
    else:
        # Embedding: vocab × embedding_dim
        embed_params = vocab_size * embedding_dim

    # Per-block parameter estimates based on HSMN architecture
    ff_dim = embedding_dim * 4  # ff_expansion=4
    d_inner = embedding_dim * 2  # expand_factor=2

    # SpatialBlock (Mamba-2): approx 4 * embedding_dim * d_inner + state params
    spatial_params = 4 * embedding_dim * d_inner + mamba_state * embedding_dim * 2

    # TimeCrystalSequenceBlock: Hamiltonian weights + output proj
    timecrystal_params = 8 * embedding_dim * embedding_dim

    # LatentReasoningBlock: FFN layers
    latent_params = 3 * embedding_dim * ff_dim

    # WLAMBlock: wavelet transforms + attention
    wlam_params = 4 * embedding_dim * embedding_dim

    # MoELayer estimation
    use_td_moe = config.get("use_td_moe", getattr(hn_config, "USE_TD_MOE", False))
    if use_td_moe:
        # Tucker decomposition reduces parameters: O(rank * (dim + ff_dim))
        tucker_rank = config.get("td_moe_tucker_rank", 16)
        # Approx: num_experts * tucker_rank * (embedding_dim + ff_dim)
        moe_params = num_experts * (tucker_rank * (embedding_dim + ff_dim))
    else:
        # Standard MoE: num_experts * (2 * embedding_dim * ff_dim)
        moe_params = num_experts * (2 * embedding_dim * ff_dim) + embedding_dim * num_experts

    # Blocks per 6-block pattern: [Spatial, TimeCrystal, Latent, Spatial, WLAM, MoE]
    num_full_patterns = num_blocks // 6
    remaining_blocks = num_blocks % 6

    params_per_pattern = (
        spatial_params * 2 + timecrystal_params + latent_params + wlam_params + moe_params
    )
    total_reasoning_params = num_full_patterns * params_per_pattern

    # Add parameters for remaining blocks in order
    rem_block_params = [
        spatial_params,
        timecrystal_params,
        latent_params,
        spatial_params,
        wlam_params,
        moe_params,
    ]
    for i in range(remaining_blocks):
        total_reasoning_params += rem_block_params[i]

    # Output layer (LM head)
    # Phase 11: QuantumLMHead estimation
    use_qlm = config.get("use_quantum_lm_head", getattr(hn_config, "USE_QUANTUM_LM_HEAD", True))
    if use_qlm:
        # VQC head uses VQC layers + projection
        # [embedding_dim, vqc_input_dim] + VQC weights
        output_params = embedding_dim * 128 + 1024  # Simplified VQC estimate
    else:
        # standard LM head: embedding_dim × vocab
        output_params = embedding_dim * vocab_size

    total = embed_params + total_reasoning_params + output_params
    return int(total)


@dataclass
class HyperparameterConfig:
    """Configuration for a single hyperparameter."""

    name: str
    value: Any
    type: str  # "float", "int", "categorical"
    min_value: Any | None = None
    max_value: Any | None = None
    choices: list[Any] | None = None


@dataclass
class HPOSearchSpace:
    """Defines the search space for hyperparameter optimization.

    Lite Edition Limits (enforced by C++ binaries):
    - vocab_size: max 65536
    - context_window: max 5000000
    - embedding_dim: max 4096
    - num_reasoning_blocks: max 24
    - num_moe_experts: max 12

    Quantum Parameter Limits (Lite vs Enterprise):
    - VQC layers: max 8 (Lite) vs 16+ (Enterprise)
    - MPQR paths: max 8 (Lite) vs 32+ (Enterprise)
    - Grover iterations: max 5 (Lite) vs 12+ (Enterprise)
    - QCOT steps: max 5 (Lite) vs 12+ (Enterprise)

    Memory-Aware Shrinking:
    When memory_failure_count > 0, the sampler automatically reduces
    architecture sizes to prevent repeated OOM/swap failures.
    """

    # =========================================================================
    # CORE TRAINING HYPERPARAMETERS
    # =========================================================================
    # Note: LR range is now auto-derived from optimizer selection.
    # This default is a safe fallback; actual range comes from OPTIMIZER_LR_RANGES.
    learning_rate: tuple[float, float] = (1e-5, 3e-4)
    batch_size: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    optimizer: list[str] = field(default_factory=lambda: ["sophiag", "adam"])
    warmup_steps: tuple[int, int] = (0, 1000)
    weight_decay: tuple[float, float] = (0.0, 0.1)

    # =========================================================================
    # MODEL ARCHITECTURE (Lite edition limits)
    # Phase 201.7: Expanded ranges for better exploration across budget sizes
    # - Minimum values for tiny models (sub-100M): 1 block, 64 dim, 2 experts
    # - Maximum values at Lite limits: 24 blocks, 12 experts, 2048 dim
    # - Dense intermediate values for smooth fANOVA-guided search
    # =========================================================================
    num_reasoning_blocks: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    )
    hidden_dim: list[int] = field(
        default_factory=lambda: [
            64,
            96,
            128,
            160,
            192,
            224,
            256,
            320,
            384,
            448,
            512,
            576,
            640,
            704,
            768,
            896,
            1024,
            1280,
            1536,
            1792,
            2048,
        ]
    )
    embedding_dim: list[int] = field(
        default_factory=lambda: [
            64,
            96,
            128,
            192,
            256,
            384,
            512,
            640,
            768,
            1024,
            1280,
            1536,
            2048,
            2560,
            3072,
            3584,
            4096,
        ]
    )
    num_moe_experts: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # =========================================================================
    # FEATURE FLAGS DESIGN PRINCIPLE
    # =========================================================================
    # Feature flags (use_hyperdimensional_embedding, use_quantum_norm, etc.) are
    # NOT tuned by HPO. They are set globally in highnoon/config.py and pulled
    # at runtime by model/training components. This ensures consistent behavior.
    #
    # hd_dim is defined in PHASE 200+ section as list[int] for HPO sampling
    # For budget estimation, use hd_dim from config or calculate from embedding_dim * 8
    # Phase 200+: vocab_size is now TUNABLE with curriculum-aware range
    # Range from 16k (small models) to 300k (large tokenizers)
    vocab_size: list[int] = field(
        default_factory=lambda: [
            16000,
            24000,
            32000,
            48000,
            64000,
            96000,
            128000,
            160000,
            200000,
            256000,
            300000,
        ]
    )
    context_window: int | None = None
    param_budget: int | None = None

    # HSMN-specific architecture params
    # Phase 201.7: Fully expanded for comprehensive architecture search
    mamba_state_dim: list[int] = field(
        default_factory=lambda: [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    )
    moe_top_k: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8])
    superposition_dim: list[int] = field(default_factory=lambda: [1, 2])  # Max 2 for Lite
    tt_rank_middle: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    hamiltonian_hidden_dim: list[int] = field(
        default_factory=lambda: [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    )

    # =========================================================================
    # PHASE 44: QUANTUM TELEPORT BUS
    # Phase 201.7: Fully expanded entanglement dimensions
    # =========================================================================
    teleport_entanglement_dim: list[int] = field(
        default_factory=lambda: [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    )
    teleport_fidelity_threshold: tuple[float, float] = (0.7, 0.995)

    # =========================================================================
    # PHASE 45: ENTROPY REGULARIZATION
    # =========================================================================
    entropy_reg_weight: tuple[float, float] = (0.001, 0.05)
    spectral_reg_weight: tuple[float, float] = (0.001, 0.05)
    target_entropy: tuple[float, float] = (0.3, 0.8)

    # =========================================================================
    # PHASE 46: SYMPFLOW OPTIMIZER
    # =========================================================================
    sympflow_mass: tuple[float, float] = (0.5, 2.0)
    sympflow_friction: tuple[float, float] = (0.05, 0.3)
    sympflow_step_size: tuple[float, float] = (0.005, 0.05)

    # =========================================================================
    # PHASE 47: QUANTUM MEASUREMENT DROPOUT
    # =========================================================================
    qmd_drop_rate: tuple[float, float] = (0.05, 0.25)
    qmd_softening_temp: tuple[float, float] = (0.5, 2.0)

    # =========================================================================
    # PHASE 50: MAJORANA POSITION ENCODING
    # Phase 201.7: Expanded Floquet period options
    # =========================================================================
    majorana_floquet_period: list[int] = field(
        default_factory=lambda: [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    )

    # =========================================================================
    # PHASE 51-52: QUANTUM LOSS FUNCTIONS
    # =========================================================================
    born_rule_temperature: tuple[float, float] = (0.5, 2.0)
    fidelity_loss_weight: tuple[float, float] = (0.005, 0.05)

    # =========================================================================
    # PHASE 55: MPQR MULTI-PATH REASONING (Lite: max 8 paths, max 5 Grover)
    # Phase 201.7: Expanded path and iteration counts
    # =========================================================================
    mpqr_num_paths: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    mpqr_grover_iterations: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    # =========================================================================
    # PHASE 56: TOPOLOGICAL WAVELET ATTENTION
    # Phase 201.7: Expanded scale counts
    # =========================================================================
    twa_num_scales: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8])

    # =========================================================================
    # PHASE 57: TD-MOE TUCKER DECOMPOSITION
    # Phase 201.7: Expanded Tucker ranks
    # =========================================================================
    td_moe_tucker_rank: list[int] = field(default_factory=lambda: [4, 6, 8, 12, 16, 24, 32, 48, 64])

    # =========================================================================
    # PHASE 58: SYMPLECTIC GNN KALMAN
    # =========================================================================
    sgkf_dt: tuple[float, float] = (0.005, 0.05)

    # =========================================================================
    # PHASE 59: ADIABATIC OPTIMIZER
    # =========================================================================
    qao_initial_temp: tuple[float, float] = (5.0, 20.0)
    qao_final_temp: tuple[float, float] = (0.005, 0.05)

    # =========================================================================
    # PHASE 60: GEODESIC OPTIMIZER
    # =========================================================================
    geodesic_momentum: tuple[float, float] = (0.8, 0.99)

    # =========================================================================
    # PHASE 61: ALPHAQUBIT DECODER (Lite: max 4 layers)
    # =========================================================================
    alphaqubit_num_layers: list[int] = field(default_factory=lambda: [1, 2, 3, 4])

    # =========================================================================
    # PHASE 62: VQEM ERROR MITIGATION
    # Phase 201.7: Fully expanded parameter counts
    # =========================================================================
    vqem_num_params: list[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    )

    # =========================================================================
    # PHASE 65: QUANTUM CRYSTALLIZATION
    # =========================================================================
    crystallization_threshold: tuple[float, float] = (0.4, 0.8)
    qhpm_crystallization_rate: tuple[float, float] = (0.05, 0.2)

    # =========================================================================
    # PHASE 68: NEUROMORPHIC MEMORY
    # =========================================================================
    neuromorphic_tau: tuple[float, float] = (5.0, 20.0)

    # =========================================================================
    # PHASE 70: MULTI-STAGE HAMILTONIAN (Lite: max 6 stages)
    # Phase 201.7: Expanded stage counts
    # =========================================================================
    hamiltonian_num_stages: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])

    # =========================================================================
    # PHASE 71: INTRINSIC PLASTICITY
    # =========================================================================
    plasticity_learning_rate: tuple[float, float] = (0.005, 0.05)

    # =========================================================================
    # PHASE 72: RANDOM NATURAL GRADIENT
    # Phase 201.7: Expanded sample counts
    # =========================================================================
    rng_num_samples: list[int] = field(default_factory=lambda: [3, 5, 8, 10, 15, 20, 30, 50])

    # =========================================================================
    # PHASE 76: QUANTUM COHERENCE BUS
    # Phase 201.7: Fully expanded node counts
    # =========================================================================
    qcb_num_nodes: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 8, 10, 12, 16])
    qcb_fidelity_threshold: tuple[float, float] = (0.7, 0.995)

    # =========================================================================
    # PHASE 78: SPINI INTEGRATOR
    # =========================================================================
    spini_friction: tuple[float, float] = (0.05, 0.3)

    # =========================================================================
    # PHASE 79: QCOT REASONING (Lite: max 5 steps)
    # Phase 201.7: Expanded step counts
    # =========================================================================
    qcot_reasoning_steps: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    # =========================================================================
    # PHASE 80: WAVEFORM ATTENTION
    # =========================================================================
    # (no tunable params beyond enable/disable)

    # =========================================================================
    # PHASE 87: COCONUT MULTI-PATH EXPLORATION (Lite: max 8 paths)
    # Phase 201.7: Expanded path counts and thresholds
    # =========================================================================
    coconut_num_paths: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    coconut_collapse_threshold: tuple[float, float] = (0.3, 0.98)
    coconut_crystallize_threshold: tuple[float, float] = (0.5, 0.995)

    # =========================================================================
    # PHASE 127: UNIFIED QUANTUM BUS
    # Phase 201.7: Fully expanded bond dimensions
    # =========================================================================
    unified_bus_mps_bond_dim: list[int] = field(
        default_factory=lambda: [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    )
    unified_bus_coherence_threshold: tuple[float, float] = (0.6, 0.98)

    # =========================================================================
    # PHASE 130: QUANTUM SYNERGY PARAMETERS
    # Phase 201.7: Expanded VQC and synergy parameters
    # =========================================================================
    # S3: VQC Adaptive Depth (Lite: max 8 layers)
    vqc_min_layers: list[int] = field(default_factory=lambda: [1, 2, 3])
    vqc_max_layers: list[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8])
    # S6: Hopfield base beta (MPS Entropy → Hopfield β adaptation)
    hopfield_base_beta: tuple[float, float] = (0.1, 3.0)
    hopfield_beta_min: tuple[float, float] = (0.1, 1.0)
    hopfield_beta_max: tuple[float, float] = (1.5, 10.0)
    # S10: DTC Floquet Period (Bus Entanglement → Floquet Period)
    dtc_floquet_period: list[int] = field(default_factory=lambda: [2, 3, 4, 6, 8, 12, 16])
    # S12: SympFlow + QNG Geodesic Weight
    sympflow_geodesic_weight: tuple[float, float] = (0.01, 0.5)

    # =========================================================================
    # DEPRECATED: QMAMBA / Q-SSM PARAMETERS (Removed per HIGHNOON_UPGRADE_ROADMAP.md)
    # QMamba blocks replaced by HDSpatialBlock when HD streaming is enabled.
    # These parameters are no longer tuned by QAHPO.
    # =========================================================================
    # NOTE: Q-SSM VQC layers moved to HD block configuration
    # q_ssm_vqc_layers retained for backward compatibility with Q-SSM gated blocks
    q_ssm_vqc_layers: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])

    # =========================================================================
    # QASA ATTENTION PARAMETERS
    # Phase 201.7: Expanded VQC layers and entanglement
    # =========================================================================
    qasa_vqc_layers: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    qasa_entanglement_strength: tuple[float, float] = (0.1, 0.95)

    # =========================================================================
    # QMOE ROUTING PARAMETERS
    # Phase 201.7: Expanded routing parameters
    # =========================================================================
    qmoe_vqc_layers: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    qmoe_measurement_temp: tuple[float, float] = (0.1, 3.0)

    # =========================================================================
    # PHASE 200+: HD STREAMING CORPUS PARAMETERS
    # =========================================================================
    # Hyperdimensional corpus compression for memory-efficient training
    # Note: use_hd_streaming flag is pulled from global config.py at runtime
    # Phase 201.7: Fully expanded ranges for comprehensive HD architecture search
    hd_reservoir_size: list[int] = field(
        default_factory=lambda: [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 8000, 10000]
    )
    hd_dim: list[int] = field(
        default_factory=lambda: [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    )
    # Vocab sample size for tokenizer learning (lower = less memory)
    vocab_sample_size: list[int] = field(
        default_factory=lambda: [500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000]
    )

    # =========================================================================
    # PHASE 201: QUANTUM MEMORY OPTIMIZATION PARAMETERS
    # =========================================================================
    # Phase 201.2: TTDense compression for FFN layers
    # Note: use_tt_ffn flag is pulled from global config.py at runtime
    tt_ffn_ranks: list[list[int]] = field(
        default_factory=lambda: [
            [1, 4, 4, 1],  # Conservative
            [1, 8, 8, 1],  # Balanced (default)
            [1, 16, 16, 1],  # Higher expressiveness
        ]
    )
    # Phase 201.3: LatentKVAttention for KV cache compression
    # Phase 201.7: Fully expanded KV compression dimensions
    # Note: use_latent_kv flag is pulled from global config.py at runtime
    latent_kv_dim: list[int] = field(
        default_factory=lambda: [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    )

    # =========================================================================
    # PHASE 132: QULS (QUANTUM UNIFIED LOSS SYSTEM) PARAMETERS
    # =========================================================================
    # Label smoothing for softmax targets (reduces overconfidence)
    quls_label_smoothing: tuple[float, float] = (0.0, 0.2)
    # Quantum fidelity loss weight (trace fidelity with target distribution)
    quls_fidelity_weight: tuple[float, float] = (0.005, 0.05)
    # Born probability temperature for fidelity computation
    quls_fidelity_temperature: tuple[float, float] = (0.5, 2.0)
    # Born rule |ψ|² = p enforcement penalty weight
    quls_born_rule_weight: tuple[float, float] = (0.001, 0.02)
    # Coherence preservation loss weight (penalizes coherence drop)
    quls_coherence_weight: tuple[float, float] = (0.005, 0.05)
    # Symplectic energy conservation loss weight
    quls_symplectic_weight: tuple[float, float] = (0.005, 0.05)
    # VQC variance boost factor for adaptive weighting
    quls_vqc_variance_boost: tuple[float, float] = (1.0, 4.0)
    # Weight reduction during barren plateau detection
    quls_barren_plateau_reduction: tuple[float, float] = (0.05, 0.3)
    # Weight EMA decay for adaptive weight updates
    quls_weight_ema_decay: tuple[float, float] = (0.95, 0.999)

    # =========================================================================
    # PHASE 131: QALRC (QUANTUM ADAPTIVE LR CONTROLLER) PARAMETERS
    # =========================================================================
    # Annealing schedule exponent (higher = faster convergence focus)
    qalrc_annealing_power: tuple[float, float] = (1.0, 4.0)
    # Base quantum tunneling probability for escaping local minima
    qalrc_tunneling_probability: tuple[float, float] = (0.01, 0.15)
    # EMA coefficient for gradient entropy smoothing
    qalrc_entropy_smoothing: tuple[float, float] = (0.8, 0.99)

    # =========================================================================
    # PHASE 200+: SAQC (SPECTRALLY-AWARE QUANTUM CURRICULUM) PARAMETERS
    # =========================================================================
    # Fraction of tunneling/orthogonal samples during plateau
    saqc_tunneling_dataset_ratio: tuple[float, float] = (0.1, 0.4)
    # Minimum steps before curriculum stage transition
    saqc_min_stage_duration: list[int] = field(
        default_factory=lambda: [50, 100, 150, 200, 300, 500]
    )
    # Steps between curriculum state updates
    saqc_update_interval: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 30, 50])
    # Spectral entropy threshold triggering retreat mode
    saqc_entropy_retreat_threshold: tuple[float, float] = (0.2, 0.5)
    # Fidelity threshold enabling acceleration mode
    saqc_fidelity_advance_threshold: tuple[float, float] = (0.75, 0.95)
    # Coherence gate threshold for curriculum progression
    saqc_coherence_gate_threshold: tuple[float, float] = (0.8, 0.98)

    # =========================================================================
    # PHASE 17: HAMILTONIAN ENHANCEMENT PARAMETERS
    # =========================================================================
    # Number of basis Hamiltonians for superposition mode
    hamiltonian_basis_size: list[int] = field(default_factory=lambda: [2, 4, 6, 8])
    # RBM hidden dimension for Neural Quantum State ansatz
    hamiltonian_nqs_hidden_dim: list[int] = field(
        default_factory=lambda: [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    )
    # Magnus expansion truncation level (2 or 4 terms)
    hamiltonian_magnus_level: list[int] = field(default_factory=lambda: [2, 4])
    # Padé matrix exponential scaling iterations
    hamiltonian_pade_scaling: list[int] = field(default_factory=lambda: [2, 4, 6, 8])

    # =========================================================================
    # PHASE T1-T6: QUANTUM TRAINING LOOP PARAMETERS
    # =========================================================================
    # Barren plateau gradient norm threshold for detection
    barren_plateau_threshold: tuple[float, float] = (1e-7, 1e-5)
    # LR scaling factor during barren plateau recovery
    barren_plateau_recovery_lr_scale: tuple[float, float] = (5.0, 20.0)
    # QNG damping for QFIM regularization
    qng_damping: tuple[float, float] = (1e-5, 1e-3)
    # GaLore gradient projection rank
    galore_rank: list[int] = field(default_factory=lambda: [8, 16, 24, 32, 48, 64, 96, 128])
    # Steps between GaLore projection updates
    galore_update_proj_gap: list[int] = field(default_factory=lambda: [100, 150, 200, 300, 500])
    # GaLore gradient scaling factor
    galore_scale: tuple[float, float] = (0.1, 0.5)

    # =========================================================================
    # USER-SET FIXED PARAMETERS (not tuned)
    # =========================================================================
    # Note: vocab_size is now tunable (see FEATURE FLAGS section)
    context_window: int | None = None
    position_embedding: str | None = None

    # Parameter budget constraint (user-set) - configs exceeding this are rejected
    param_budget: int | None = None
    lambda_da: tuple[float, float] | None = None  # Domain adaptation weight
    unfreeze_schedule: list[str] | None = None  # Unfreezing strategy
    unfreeze_interval: tuple[int, int] | None = None  # Steps between unfreezing

    # Memory-aware shrinking: count of consecutive memory failures
    # When > 0, sample() reduces architecture sizes to prevent OOM
    memory_failure_count: int = 0

    # Smart tuner: track skipped trials for WebUI and adaptive sampling

    _budget_tracker: OversizedConfigTracker | None = None

    def record_memory_failure(self) -> None:
        """Record a memory-related trial failure (OOM, swap spike, etc.).

        Increments memory_failure_count which causes subsequent samples to use
        smaller architectures to prevent repeated failures.
        """
        self.memory_failure_count += 1
        logger.warning(
            f"[HPO Manager] Memory failure recorded. "
            f"Shrink level now: {self.memory_failure_count}"
        )

    def record_trial_success(self) -> None:
        """Record a successful trial completion.

        Resets memory_failure_count since the current config fits in memory.
        """
        if self.memory_failure_count > 0:
            logger.info(
                f"[HPO Manager] Trial succeeded. Resetting memory shrink level "
                f"from {self.memory_failure_count} to 0"
            )
            self.memory_failure_count = 0

    def _compute_progression_factor(self, trial_id: int) -> float:
        """Compute budget progression factor based on trial index.

        Progressive sizing ensures early trials start with small architectures
        that fit well within the parameter budget, then gradually explore
        larger configs as trials progress.

        Args:
            trial_id: Trial index (0-based)

        Returns:
            Progression factor (0.1 to 0.9) to multiply param_budget by.
        """
        if trial_id < 10:
            # Trials 0-9: Start at 10-30% of budget (minimum viable configs)
            return 0.1 + (0.2 * trial_id / 10)
        elif trial_id < 50:
            # Trials 10-49: Scale from 30% to 70% of budget
            return 0.3 + (0.4 * (trial_id - 10) / 40)
        else:
            # Trials 50+: Allow up to 90% of budget
            return min(0.9, 0.7 + (0.2 * (trial_id - 50) / 100))

    def _estimate_max_sizes_for_budget(
        self,
        target_budget: int,
    ) -> tuple[int, int, int]:
        """Estimate max architecture sizes that fit within target param budget.

        Uses inverse of estimate_model_params() to find compatible architectures.
        Iterates through dimension/block/expert combinations to find the largest
        config that fits.

        Args:
            target_budget: Target parameter count limit

        Returns:
            Tuple of (max_blocks, max_experts, max_dim) that fit within budget.
        """
        # CRITICAL: self.vocab_size is a list[int] in HPOSearchSpace
        # Use conservative (min) value for budget estimation
        vocab_size = (
            min(self.vocab_size)
            if isinstance(self.vocab_size, list) and self.vocab_size
            else (self.vocab_size if isinstance(self.vocab_size, int) else 32000)
        )

        # Start with minimum viable config
        best_blocks = 2
        best_experts = 2
        best_dim = 64
        best_params = 0  # Track the largest fitting config's param count

        # Try dimensions from smallest to largest
        sorted_dims = sorted(self.embedding_dim)
        sorted_blocks = sorted(self.num_reasoning_blocks)
        sorted_experts = sorted(self.num_moe_experts)

        for dim in sorted_dims:
            for blocks in sorted_blocks:
                for experts in sorted_experts:
                    # Use representative hd_dim value (min of list if available)
                    # This ensures estimate_model_params gets a scalar, not list
                    representative_hd_dim = (
                        min(self.hd_dim)
                        if self.hd_dim and isinstance(self.hd_dim, list)
                        else (self.hd_dim if isinstance(self.hd_dim, int) else dim * 8)
                    )
                    test_config = {
                        "vocab_size": vocab_size,
                        "embedding_dim": dim,
                        "hidden_dim": dim,
                        "num_reasoning_blocks": blocks,
                        "num_moe_experts": experts,
                        "hd_dim": representative_hd_dim,
                    }
                    estimated = estimate_model_params(test_config)
                    # Only update if this config fits AND is larger than current best
                    if estimated <= target_budget and estimated > best_params:
                        best_dim = dim
                        best_blocks = blocks
                        best_experts = experts
                        best_params = estimated

        return (best_blocks, best_experts, best_dim)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> HPOSearchSpace:
        """Create HPOSearchSpace from config dictionary.

        Phase 5 Update: Stores original config for grouped format detection.

        Args:
            config: Configuration dictionary with 'search_space' section

        Returns:
            HPOSearchSpace instance
        """
        search_space_config = config.get("search_space", {})
        model_config = config.get("model_config", {})

        # Create instance with default values
        instance = cls()

        # Update from search_space config if provided
        for key, value in search_space_config.items():
            if hasattr(instance, key):
                # Handle nested dicts for legacy flat format
                if isinstance(value, dict) and "choices" in value:
                    setattr(instance, key, value["choices"])
                elif isinstance(value, dict) and "min" in value and "max" in value:
                    setattr(instance, key, (value["min"], value["max"]))
                else:
                    setattr(instance, key, value)

        # Sync model-specific limits/flags for parameter estimation
        # If vocab_size is not in search_space, use default from model_config
        if not hasattr(instance, "vocab_size") or instance.vocab_size is None:
            instance.vocab_size = model_config.get("vocab_size")

        instance.context_window = model_config.get("sequence_length") or model_config.get(
            "context_window"
        )
        instance.param_budget = config.get("param_budget") or model_config.get("param_budget")

        # Capture hd_dim if provided (needed for accurate estimation)
        instance.hd_dim = model_config.get("hd_dim")

        # Store full config for grouped detection
        instance._grouped_config = config
        instance._is_grouped = "parameters" in search_space_config

        return instance

    def sample(self, trial_id: int, stage: int = 1) -> dict[str, Any]:
        """Sample a configuration from the search space.

        Phase 5 Update: Uses grouped parameter search when available,
        supporting multi-stage sampling (COARSE → REFINE → FINE).

        Args:
            trial_id: Integer trial ID for deterministic sampling
            stage: Sampling stage (1=COARSE, 2=REFINE, 3=FINE)

        Returns:
            Dictionary of hyperparameter values
        """
        # Phase 5: Use grouped sampling if available and config is grouped
        if GROUPED_SAMPLING_AVAILABLE and getattr(self, "_is_grouped", False):
            sampler = ParameterGroupSampler(seed=trial_id)
            config = sampler.sample_grouped_configuration(self._grouped_config, trial_id, stage)

            # Apply any fixed parameters from the config
            if "fixed_parameters" in self._grouped_config:
                for key, value in self._grouped_config["fixed_parameters"].items():
                    if key != "comment":
                        config[key] = value

            logger.info(
                f"[HPO Manager] Sampled Stage {stage} config for trial {trial_id}: "
                f"keys={sorted(config.keys())}"
            )
            return config

        # Legacy flat sampling for backward compatibility
        import random

        random.seed(trial_id)  # Deterministic for reproducibility

        # Memory-aware shrinking: reduce architecture when previous trials failed due to OOM
        # Each memory failure decreases max allowed sizes to avoid repeated OOM
        memory_shrink_level = getattr(self, "memory_failure_count", 0)

        if memory_shrink_level >= 3:
            # Severe memory pressure: use minimum viable config
            max_blocks = 4
            max_experts = 4
            max_dim = 256
            max_batch = 8
            logger.warning(
                f"[HPO Manager] Memory shrink level {memory_shrink_level}: "
                "Using minimum viable config (4 blocks, 4 experts, 256 dim)"
            )
        elif memory_shrink_level >= 2:
            # High memory pressure: significantly reduced
            max_blocks = 6
            max_experts = 6
            max_dim = 512
            max_batch = 16
            logger.warning(
                f"[HPO Manager] Memory shrink level {memory_shrink_level}: "
                "Using reduced config (6 blocks, 6 experts, 512 dim)"
            )
        elif memory_shrink_level >= 1:
            # Moderate memory pressure: slightly reduced
            max_blocks = 8
            max_experts = 8
            max_dim = 768
            max_batch = 32
            logger.info(
                f"[HPO Manager] Memory shrink level {memory_shrink_level}: "
                "Using conservative config (8 blocks, 8 experts, 768 dim)"
            )
        else:
            # No memory pressure: use full search space
            max_blocks = 24
            max_experts = 12
            max_dim = 4096
            max_batch = 128

        # =====================================================================
        # PROGRESSIVE BUDGET-AWARE SIZING
        # When param_budget is set, progressively scale architecture limits:
        # - Early trials (0-9): 10-30% of budget → explore small configs first
        # - Middle trials (10-49): 30-70% of budget → expand search
        # - Late trials (50+): up to 90% of budget → find optimal near limit
        # =====================================================================
        if self.param_budget is not None:
            prog_factor = self._compute_progression_factor(trial_id)
            target_budget = int(self.param_budget * prog_factor)
            prog_max_blocks, prog_max_experts, prog_max_dim = self._estimate_max_sizes_for_budget(
                target_budget
            )

            # Use stricter of progressive and memory limits
            max_blocks = min(max_blocks, prog_max_blocks)
            max_experts = min(max_experts, prog_max_experts)
            max_dim = min(max_dim, prog_max_dim)

            logger.debug(
                f"[SmartTuner] Trial {trial_id}: progression={prog_factor:.2f}, "
                f"target_budget={target_budget / 1e6:.0f}M, max_dim={max_dim}, "
                f"max_blocks={max_blocks}, max_experts={max_experts}"
            )

        # Filter search space by memory and budget constraints
        filtered_blocks = [b for b in self.num_reasoning_blocks if b <= max_blocks] or [4]
        filtered_experts = [e for e in self.num_moe_experts if e <= max_experts] or [4]
        filtered_dims = [d for d in self.embedding_dim if d <= max_dim] or [256]
        filtered_batch = [b for b in self.batch_size if b <= max_batch] or [8]

        # Select optimizer first so we can derive LR range
        selected_optimizer = random.choice(self.optimizer)

        # Get optimizer-specific LR range, falling back to default if unknown
        lr_range = OPTIMIZER_LR_RANGES.get(selected_optimizer.lower(), DEFAULT_LR_RANGE)

        # Use log-uniform sampling for learning rate
        # This ensures small LRs (1e-5) are as likely as large LRs (1e-4)
        # which is critical for stable hyperparameter search
        lr_log_min = math.log10(lr_range[0])
        lr_log_max = math.log10(lr_range[1])
        sampled_lr = 10 ** random.uniform(lr_log_min, lr_log_max)

        # Sample vocab_size if it's a list or tuple
        sampled_vocab_size = self.vocab_size
        if isinstance(self.vocab_size, (list, tuple)):
            if len(self.vocab_size) == 2 and all(isinstance(x, int) for x in self.vocab_size):
                sampled_vocab_size = random.randint(*self.vocab_size)
            else:
                sampled_vocab_size = random.choice(self.vocab_size)

        config = {
            # Metadata for internal tracking
            "_trial_id": trial_id,
            "vocab_size": sampled_vocab_size,
            "param_budget": self.param_budget,
            # Training hyperparameters (LR auto-derived from optimizer)
            "learning_rate": sampled_lr,
            "batch_size": random.choice(filtered_batch),
            "optimizer": selected_optimizer,
            "warmup_steps": random.randint(*self.warmup_steps),
            "weight_decay": random.uniform(*self.weight_decay),
            # Model architecture params (memory-aware)
            "num_reasoning_blocks": random.choice(filtered_blocks),
            "hidden_dim": random.choice(filtered_dims),
            "embedding_dim": random.choice(filtered_dims),
            "num_moe_experts": random.choice(filtered_experts),
            # HSMN-specific params
            "mamba_state_dim": random.choice(self.mamba_state_dim),
            "moe_top_k": random.choice(self.moe_top_k),
            "superposition_dim": random.choice(self.superposition_dim),
            "tt_rank_middle": random.choice(self.tt_rank_middle),
            "hamiltonian_hidden_dim": random.choice(self.hamiltonian_hidden_dim),
            # =================================================================
            # PHASE 44: Quantum Teleport Bus
            # =================================================================
            "teleport_entanglement_dim": random.choice(self.teleport_entanglement_dim),
            "teleport_fidelity_threshold": random.uniform(*self.teleport_fidelity_threshold),
            # =================================================================
            # PHASE 45: Entropy Regularization
            # =================================================================
            "entropy_reg_weight": random.uniform(*self.entropy_reg_weight),
            "spectral_reg_weight": random.uniform(*self.spectral_reg_weight),
            "target_entropy": random.uniform(*self.target_entropy),
            # =================================================================
            # PHASE 46: SympFlow Optimizer
            # =================================================================
            "sympflow_mass": random.uniform(*self.sympflow_mass),
            "sympflow_friction": random.uniform(*self.sympflow_friction),
            "sympflow_step_size": random.uniform(*self.sympflow_step_size),
            # =================================================================
            # PHASE 47: Quantum Measurement Dropout
            # =================================================================
            "qmd_drop_rate": random.uniform(*self.qmd_drop_rate),
            "qmd_softening_temp": random.uniform(*self.qmd_softening_temp),
            # =================================================================
            # PHASE 50: Majorana Position Encoding
            # =================================================================
            "majorana_floquet_period": random.choice(self.majorana_floquet_period),
            # =================================================================
            # PHASE 51-52: Quantum Loss Functions
            # =================================================================
            "born_rule_temperature": random.uniform(*self.born_rule_temperature),
            "fidelity_loss_weight": random.uniform(*self.fidelity_loss_weight),
            # =================================================================
            # PHASE 55: MPQR Multi-Path Reasoning
            # =================================================================
            "mpqr_num_paths": random.choice(self.mpqr_num_paths),
            "mpqr_grover_iterations": random.choice(self.mpqr_grover_iterations),
            # =================================================================
            # PHASE 56: Topological Wavelet Attention
            # =================================================================
            "twa_num_scales": random.choice(self.twa_num_scales),
            # =================================================================
            # PHASE 57: TD-MoE Tucker Decomposition
            # =================================================================
            "td_moe_tucker_rank": random.choice(self.td_moe_tucker_rank),
            # =================================================================
            # PHASE 58: Symplectic GNN Kalman
            # =================================================================
            "sgkf_dt": random.uniform(*self.sgkf_dt),
            # =================================================================
            # PHASE 59: Adiabatic Optimizer
            # =================================================================
            "qao_initial_temp": random.uniform(*self.qao_initial_temp),
            "qao_final_temp": random.uniform(*self.qao_final_temp),
            # =================================================================
            # PHASE 60: Geodesic Optimizer
            # =================================================================
            "geodesic_momentum": random.uniform(*self.geodesic_momentum),
            # =================================================================
            # PHASE 61: AlphaQubit Decoder
            # =================================================================
            "alphaqubit_num_layers": random.choice(self.alphaqubit_num_layers),
            # =================================================================
            # PHASE 62: VQEM Error Mitigation
            # =================================================================
            "vqem_num_params": random.choice(self.vqem_num_params),
            # =================================================================
            # PHASE 65: Quantum Crystallization
            # =================================================================
            "crystallization_threshold": random.uniform(*self.crystallization_threshold),
            "qhpm_crystallization_rate": random.uniform(*self.qhpm_crystallization_rate),
            # =================================================================
            # PHASE 68: Neuromorphic Memory
            # =================================================================
            "neuromorphic_tau": random.uniform(*self.neuromorphic_tau),
            # =================================================================
            # PHASE 70: Multi-Stage Hamiltonian
            # =================================================================
            "hamiltonian_num_stages": random.choice(self.hamiltonian_num_stages),
            # =================================================================
            # PHASE 71: Intrinsic Plasticity
            # =================================================================
            "plasticity_learning_rate": random.uniform(*self.plasticity_learning_rate),
            # =================================================================
            # PHASE 72: Random Natural Gradient
            # =================================================================
            "rng_num_samples": random.choice(self.rng_num_samples),
            # =================================================================
            # PHASE 76: Quantum Coherence Bus
            # =================================================================
            "qcb_num_nodes": random.choice(self.qcb_num_nodes),
            "qcb_fidelity_threshold": random.uniform(*self.qcb_fidelity_threshold),
            # =================================================================
            # PHASE 78: SPINI Integrator
            # =================================================================
            "spini_friction": random.uniform(*self.spini_friction),
            # =================================================================
            # PHASE 79: QCOT Reasoning
            # =================================================================
            "qcot_reasoning_steps": random.choice(self.qcot_reasoning_steps),
            # =================================================================
            # PHASE 87: CoCoNut Multi-Path Exploration
            # =================================================================
            "coconut_num_paths": random.choice(self.coconut_num_paths),
            "coconut_collapse_threshold": random.uniform(*self.coconut_collapse_threshold),
            "coconut_crystallize_threshold": random.uniform(*self.coconut_crystallize_threshold),
            # =================================================================
            # PHASE 127: Unified Quantum Bus
            # =================================================================
            "unified_bus_mps_bond_dim": random.choice(self.unified_bus_mps_bond_dim),
            "unified_bus_coherence_threshold": random.uniform(
                *self.unified_bus_coherence_threshold
            ),
            # =================================================================
            # PHASE 130: Quantum Synergy Parameters
            # =================================================================
            # S3: VQC Adaptive Depth
            "vqc_min_layers": random.choice(self.vqc_min_layers),
            "vqc_max_layers": random.choice(self.vqc_max_layers),
            # S6: Hopfield adaptive beta
            "hopfield_base_beta": random.uniform(*self.hopfield_base_beta),
            "hopfield_beta_min": random.uniform(*self.hopfield_beta_min),
            "hopfield_beta_max": random.uniform(*self.hopfield_beta_max),
            # S10: DTC Floquet Period
            "dtc_floquet_period": random.choice(self.dtc_floquet_period),
            # S12: SympFlow Geodesic Weight
            "sympflow_geodesic_weight": random.uniform(*self.sympflow_geodesic_weight),
            # =================================================================
            # Q-SSM Parameters (QMamba deprecated per HIGHNOON_UPGRADE_ROADMAP.md)
            # HD blocks now replace QMamba when USE_HD_STREAMING is enabled
            # =================================================================
            "q_ssm_vqc_layers": random.choice(self.q_ssm_vqc_layers),
            # =================================================================
            # QASA Attention Parameters
            # =================================================================
            "qasa_vqc_layers": random.choice(self.qasa_vqc_layers),
            "qasa_entanglement_strength": random.uniform(*self.qasa_entanglement_strength),
            # =================================================================
            # QMoE Routing Parameters
            # =================================================================
            "qmoe_vqc_layers": random.choice(self.qmoe_vqc_layers),
            "qmoe_measurement_temp": random.uniform(*self.qmoe_measurement_temp),
            # =================================================================
            # PHASE 200+: HD Streaming Corpus Parameters
            # Note: use_hd_streaming pulled from global config.py at runtime
            # =================================================================
            "hd_reservoir_size": random.choice(self.hd_reservoir_size),
            # Defensive null handling: use default [512, 1024] if hd_dim is None
            "hd_dim": random.choice(self.hd_dim if self.hd_dim else [512, 1024]),
            "vocab_sample_size": random.choice(self.vocab_sample_size),
            # =================================================================
            # PHASE 201: Quantum Memory Optimization Parameters
            # Note: use_tt_ffn, use_latent_kv pulled from global config.py
            # =================================================================
            "tt_ffn_ranks": random.choice(self.tt_ffn_ranks),
            "latent_kv_dim": random.choice(self.latent_kv_dim),
            # =================================================================
            # PHASE 132: QULS (Quantum Unified Loss System) Parameters
            # =================================================================
            "quls_label_smoothing": random.uniform(*self.quls_label_smoothing),
            "quls_fidelity_weight": random.uniform(*self.quls_fidelity_weight),
            "quls_fidelity_temperature": random.uniform(*self.quls_fidelity_temperature),
            "quls_born_rule_weight": random.uniform(*self.quls_born_rule_weight),
            "quls_coherence_weight": random.uniform(*self.quls_coherence_weight),
            "quls_symplectic_weight": random.uniform(*self.quls_symplectic_weight),
            "quls_vqc_variance_boost": random.uniform(*self.quls_vqc_variance_boost),
            "quls_barren_plateau_reduction": random.uniform(*self.quls_barren_plateau_reduction),
            "quls_weight_ema_decay": random.uniform(*self.quls_weight_ema_decay),
            # =================================================================
            # PHASE 131: QALRC (Quantum Adaptive LR Controller) Parameters
            # =================================================================
            "qalrc_annealing_power": random.uniform(*self.qalrc_annealing_power),
            "qalrc_tunneling_probability": random.uniform(*self.qalrc_tunneling_probability),
            "qalrc_entropy_smoothing": random.uniform(*self.qalrc_entropy_smoothing),
            # =================================================================
            # PHASE 200+: SAQC (Spectrally-Aware Quantum Curriculum) Parameters
            # =================================================================
            "saqc_tunneling_dataset_ratio": random.uniform(*self.saqc_tunneling_dataset_ratio),
            "saqc_min_stage_duration": random.choice(self.saqc_min_stage_duration),
            "saqc_update_interval": random.choice(self.saqc_update_interval),
            "saqc_entropy_retreat_threshold": random.uniform(*self.saqc_entropy_retreat_threshold),
            "saqc_fidelity_advance_threshold": random.uniform(
                *self.saqc_fidelity_advance_threshold
            ),
            "saqc_coherence_gate_threshold": random.uniform(*self.saqc_coherence_gate_threshold),
            # =================================================================
            # PHASE 17: Hamiltonian Enhancement Parameters
            # =================================================================
            "hamiltonian_basis_size": random.choice(self.hamiltonian_basis_size),
            "hamiltonian_nqs_hidden_dim": random.choice(self.hamiltonian_nqs_hidden_dim),
            "hamiltonian_magnus_level": random.choice(self.hamiltonian_magnus_level),
            "hamiltonian_pade_scaling": random.choice(self.hamiltonian_pade_scaling),
            # =================================================================
            # PHASE T1-T6: Quantum Training Loop Parameters
            # =================================================================
            "barren_plateau_threshold": random.uniform(*self.barren_plateau_threshold),
            "barren_plateau_recovery_lr_scale": random.uniform(
                *self.barren_plateau_recovery_lr_scale
            ),
            "qng_damping": random.uniform(*self.qng_damping),
            "galore_rank": random.choice(self.galore_rank),
            "galore_update_proj_gap": random.choice(self.galore_update_proj_gap),
            "galore_scale": random.uniform(*self.galore_scale),
        }

        # Add user-set tokenizer/context config if provided
        # Note: vocab_size is now TUNABLE (sampled at line 880), no longer fixed
        if self.context_window is not None:
            config["context_window"] = self.context_window
        if self.position_embedding is not None:
            config["position_embedding"] = self.position_embedding

        # Phase 3: Add transfer learning params if configured
        if self.lambda_da is not None:
            config["lambda_da"] = random.uniform(*self.lambda_da)
        if self.unfreeze_schedule is not None:
            config["unfreeze_schedule"] = random.choice(self.unfreeze_schedule)
        if self.unfreeze_interval is not None:
            config["unfreeze_interval"] = random.randint(*self.unfreeze_interval)

        # =================================================================
        # PHASE 200+: Enforce hd_dim divisibility constraint
        # hd_dim must be divisible by hidden_dim for HDStreamingAdapter projection
        # =================================================================
        hidden_dim = config.get("hidden_dim", 256)
        hd_dim_val = config.get("hd_dim", 1024)
        if hd_dim_val % hidden_dim != 0:
            # Find nearest valid hd_dim that is divisible by hidden_dim
            # Round up to next multiple
            valid_hd_dim = ((hd_dim_val // hidden_dim) + 1) * hidden_dim
            # But don't exceed max from search space
            max_hd_dim = max(self.hd_dim) if self.hd_dim else 4096
            if valid_hd_dim > max_hd_dim:
                # Round down instead
                valid_hd_dim = (hd_dim_val // hidden_dim) * hidden_dim
            if valid_hd_dim > 0:
                config["hd_dim"] = valid_hd_dim
                logger.debug(
                    f"[HPO Manager] Adjusted hd_dim {hd_dim_val} -> {valid_hd_dim} "
                    f"(divisible by hidden_dim={hidden_dim})"
                )

        # Parameter budget constraint check with smart tuner integration
        if self.param_budget is not None:
            # Re-fetch target budget for stricter enforcement
            prog_factor = self._compute_progression_factor(trial_id)
            trial_target_budget = int(self.param_budget * prog_factor)

            # Initialize budget tracker if not already done
            if not hasattr(self, "_budget_tracker") or self._budget_tracker is None:
                from highnoon.services.hpo_metrics import OversizedConfigTracker

                self._budget_tracker = OversizedConfigTracker(
                    param_budget=self.param_budget,
                )

            # Check initial estimate
            # CRITICAL: Pass trial_target_budget (not full param_budget) so that
            # compute_max_vocab_for_budget uses the progressive target, preventing
            # initial configs from being 2-3x larger than allowed.
            config_for_estimate = config.copy()
            config_for_estimate["param_budget"] = trial_target_budget
            initial_estimated = estimate_model_params(config_for_estimate)
            max_attempts = 100  # Stricter enforcement requires more attempts
            attempt = 0

            # Log if initial config exceeds the progressive target
            if initial_estimated > trial_target_budget:
                logger.debug(
                    f"[SmartTuner] Trial {trial_id}: initial config exceeds progressive budget "
                    f"({initial_estimated / 1e6:.1f}M > {trial_target_budget / 1e6:.1f}M), resampling..."
                )

            estimated_params = initial_estimated

            while estimated_params > trial_target_budget and attempt < max_attempts:
                # Re-sample with progressively smaller architecture
                attempt += 1
                random.seed(trial_id + attempt * 1000)

                # Progressively tighten constraints based on attempt number
                if attempt < max_attempts // 2:
                    current_dim = config["embedding_dim"]
                    smaller_dims = [d for d in filtered_dims if d < current_dim]
                    if smaller_dims:
                        new_dim = random.choice(smaller_dims)
                        config["embedding_dim"] = new_dim
                        config["hidden_dim"] = new_dim
                    else:
                        # If dim is already min, try reducing blocks
                        current_blocks = config["num_reasoning_blocks"]
                        smaller_blocks = [b for b in filtered_blocks if b < current_blocks]
                        if smaller_blocks:
                            config["num_reasoning_blocks"] = random.choice(smaller_blocks)
                        else:
                            # If also at min blocks, try reducing experts
                            current_experts = config["num_moe_experts"]
                            smaller_experts = [e for e in filtered_experts if e < current_experts]
                            if smaller_experts:
                                config["num_moe_experts"] = random.choice(smaller_experts)
                else:
                    # Radical reduction phase: slam to minimums
                    config["num_reasoning_blocks"] = min(filtered_blocks)
                    config["num_moe_experts"] = min(filtered_experts)
                    config["embedding_dim"] = min(filtered_dims)
                    config["hidden_dim"] = config["embedding_dim"]

                # Use progressive target budget for estimation during resampling
                config_for_estimate = config.copy()
                config_for_estimate["param_budget"] = trial_target_budget
                estimated_params = estimate_model_params(config_for_estimate)

                if attempt % 20 == 0:
                    logger.debug(
                        f"[SmartTuner] Trial {trial_id} attempt {attempt}: current {estimated_params/1e6:.1f}M"
                    )

            if attempt > 0:
                logger.info(
                    f"[SmartTuner] Trial {trial_id}: config adjusted to fit progressive target "
                    f"({estimated_params / 1e6:.1f}M <= {trial_target_budget / 1e6:.1f}M) "
                    f"after {attempt} attempts"
                )

            # Record if even after adjustment we are over the ABSOLUTE budget
            if estimated_params > self.param_budget:
                self._budget_tracker.record_oversized(
                    trial_id=f"trial_{trial_id}",
                    config=config.copy(),
                    estimated_params=estimated_params,
                    reason="even_minimum_config_exceeds_budget_after_max_attempts",
                )
                logger.warning(
                    f"[SmartTuner] Trial {trial_id}: ABSOLUTE budget exceeded "
                    f"({estimated_params / 1e6:.1f}M > {self.param_budget / 1e6:.1f}M). "
                    f"Consider increasing budget or reducing vocab_size/HQE base."
                )

        return config

    def get_skipped_trials(self) -> list[dict[str, Any]]:
        """Get list of trials that were skipped due to budget constraints.

        Returns:
            List of skipped trial records formatted for WebUI
        """
        if self._budget_tracker is None:
            return []
        return [r.to_dict() for r in self._budget_tracker.get_skipped_records()]

    def get_budget_statistics(self) -> dict[str, Any]:
        """Get statistics about parameter budget enforcement.

        Returns:
            Dictionary with budget enforcement statistics
        """
        if self._budget_tracker is None:
            return {"enabled": False}

        stats = self._budget_tracker.get_statistics()
        stats["enabled"] = True
        stats["param_budget"] = self.param_budget
        return stats


class HPOTrialManager:
    """Manages the lifecycle of HPO trials."""

    def __init__(
        self,
        search_space: HPOSearchSpace | None = None,
        metrics_collector: HPOMetricsCollector | None = None,
        max_trials: int = 10,
        hpo_binary: Path | None = None,
        config_path: Path | None = None,
    ):
        """Initialize the HPO trial manager.

        Args:
            search_space: Search space configuration
            metrics_collector: Metrics collector instance
            max_trials: Maximum number of trials to run
            hpo_binary: Path to the compiled HPO binary (hpo_main)
            config_path: Path to HPO config JSON file
        """
        # Load config from file if not provided
        self.config = load_hpo_config(config_path)

        # Use search space from config if not explicitly provided
        if search_space is None and self.config:
            self.search_space = HPOSearchSpace.from_config(self.config)
        else:
            self.search_space = search_space or HPOSearchSpace()

        self.metrics_collector = metrics_collector or HPOMetricsCollector()

        # Get max_trials from config if available
        hpo_runner = self.config.get("hpo_runner", {})
        trial_settings = hpo_runner.get("trial_settings", {})
        config_max_trials = trial_settings.get("max_trials", max_trials)
        self.max_trials = max_trials if max_trials != 10 else config_max_trials

        # Find the HPO binary
        if hpo_binary is None:
            # Try standard locations (in order of preference)
            import platform

            arch = platform.machine()
            if arch in ("x86_64", "AMD64"):
                arch = "x86_64"
            elif arch in ("aarch64", "arm64"):
                arch = "arm64"

            candidates = [
                # HighNoon native binary locations
                Path(__file__).parent.parent / "_native" / "bin" / arch / "_hpo_main.so",
                Path(__file__).parent.parent / "_native" / "bin" / arch / "hpo_main",
                Path(__file__).parent.parent / "_native" / "ops" / "_hpo_main.so",
                # Legacy HSMN locations
                Path("src/ops/_hpo_main.so"),
                Path("src/ops/_hpo_main.x86_64.so"),
                Path("_hpo_main.so"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    self.hpo_binary = candidate.resolve()
                    break
            else:
                # Don't fail - HPO can work without the binary in many cases
                self.hpo_binary = None
                logger.warning(
                    "[HPO Manager] HPO binary not found. Some features may be unavailable. "
                    "Build with: cd highnoon/_native && mkdir -p build && cd build && "
                    "cmake .. -DCMAKE_BUILD_TYPE=Release && make hpo_main"
                )
        else:
            self.hpo_binary = Path(hpo_binary).resolve()

        if self.hpo_binary:
            logger.info(f"[HPO Manager] Using binary: {self.hpo_binary}")
        logger.info(f"[HPO Manager] Max trials: {max_trials}")

    def write_trial_config(self, trial_id: str, config: dict[str, Any]) -> Path:
        """Write trial configuration to disk for the C++ orchestrator.

        Phase 5 Update: Converts numpy types to Python native types for JSON serialization.

        Args:
            trial_id: Trial identifier
            config: Hyperparameter configuration

        Returns:
            Path to the written config file
        """
        trial_dir = self.metrics_collector.get_trial_dir(trial_id)
        config_file = trial_dir / "config.json"

        # Convert numpy types to Python native types
        config_clean = convert_numpy_types(config)

        with config_file.open("w", encoding="utf-8") as f:
            json.dump(config_clean, f, indent=2)
            f.write("\n")

        logger.info(f"[HPO Manager] Wrote config for trial {trial_id}: {config_file}")
        return config_file

    def initialize_trial(self, trial_id: str, config: dict[str, Any]) -> None:
        """Initialize a trial with pending status.

        Args:
            trial_id: Trial identifier
            config: Hyperparameter configuration
        """
        # Compute estimated parameter count for efficiency scoring
        param_count = estimate_model_params(config)

        status = TrialStatus(
            trial_id=trial_id,
            status="pending",
            hyperparameters=config,
            param_count=param_count,
        )
        self.metrics_collector.update_trial_status(status)
        logger.info(
            f"[HPO Manager] Initialized trial {trial_id} (~{param_count / 1e6:.1f}M params)"
        )

    def start_trial(self, trial_id: str) -> None:
        """Mark a trial as running.

        Args:
            trial_id: Trial identifier
        """
        status = self.metrics_collector.load_trial_status(trial_id)
        if status:
            status.status = "running"
            status.start_time = time.time()
            self.metrics_collector.update_trial_status(status)
        logger.info(f"[HPO Manager] Started trial {trial_id}")

    def complete_trial(
        self, trial_id: str, success: bool = True, error_message: str | None = None
    ) -> None:
        """Mark a trial as completed or failed.

        Args:
            trial_id: Trial identifier
            success: Whether the trial completed successfully
            error_message: Error message if failed
        """
        status = self.metrics_collector.load_trial_status(trial_id)
        if status:
            status.status = "completed" if success else "failed"
            status.end_time = time.time()
            if error_message:
                status.error_message = error_message
            self.metrics_collector.update_trial_status(status)

        # Memory-aware shrinking: track memory failures to shrink future configs
        if not success and error_message:
            # Detect memory-related failures (swap spike, OOM, memory critical)
            memory_keywords = ["memory", "swap", "oom", "resourceexhausted"]
            is_memory_failure = any(kw in error_message.lower() for kw in memory_keywords)
            if is_memory_failure:
                self.search_space.record_memory_failure()
                logger.info(
                    f"[HPO Manager] Memory failure detected in trial {trial_id}, "
                    f"next trials will use smaller configs"
                )
        elif success:
            # Reset memory failure count on successful trial
            self.search_space.record_trial_success()

        logger.info(
            f"[HPO Manager] Trial {trial_id} {'completed' if success else 'failed'}"
            + (f": {error_message}" if error_message else "")
        )

    def generate_trials(self) -> list[tuple[str, dict[str, Any], int]]:
        """Generate trial configurations for the HPO sweep.

        Phase 5 Update: Returns (trial_id, config, stage) tuples for multi-stage HPO.

        Returns:
            List of (trial_id, config, stage) tuples
        """
        trials = []

        # Phase 5: Check if using grouped search with multi-stage settings
        is_grouped = getattr(self.search_space, "_is_grouped", False)
        if is_grouped and GROUPED_SAMPLING_AVAILABLE:
            # Multi-stage trial generation
            hpo_runner = self.config.get("hpo_runner", {})
            trial_settings = hpo_runner.get("trial_settings", {})
            max_stage1 = trial_settings.get("max_trials_stage1", 15)
            max_stage2 = trial_settings.get("max_trials_stage2", 25)
            max_stage3 = trial_settings.get("max_trials_stage3", 30)

            trial_idx = 0
            # Stage 1: COARSE
            for _ in range(max_stage1):
                trial_id = f"trial_{trial_idx:04d}"
                config = self.search_space.sample(trial_idx, stage=1)
                trials.append((trial_id, config, 1))
                trial_idx += 1

            # Stage 2: REFINE
            for _ in range(max_stage2):
                trial_id = f"trial_{trial_idx:04d}"
                config = self.search_space.sample(trial_idx, stage=2)
                trials.append((trial_id, config, 2))
                trial_idx += 1

            # Stage 3: FINE
            for _ in range(max_stage3):
                trial_id = f"trial_{trial_idx:04d}"
                config = self.search_space.sample(trial_idx, stage=3)
                trials.append((trial_id, config, 3))
                trial_idx += 1

            logger.info(
                f"[HPO Manager] Generated {len(trials)} multi-stage trial configs: "
                f"Stage1={max_stage1}, Stage2={max_stage2}, Stage3={max_stage3}"
            )
        else:
            # Legacy flat search: all trials are stage 1
            for i in range(self.max_trials):
                trial_id = f"trial_{i:04d}"
                config = self.search_space.sample(i, stage=1)
                trials.append((trial_id, config, 1))

            logger.info(f"[HPO Manager] Generated {len(trials)} trial configurations")

        return trials

    def prepare_sweep(self) -> Path:
        """Prepare a full HPO sweep by writing all trial configs.

        Returns:
            Path to the sweep configuration file
        """
        trials = self.generate_trials()

        # Write individual trial configs
        for trial_entry in trials:
            if len(trial_entry) == 3:
                trial_id, config, stage = trial_entry
            else:
                # Backward compatibility
                trial_id, config = trial_entry

            self.write_trial_config(trial_id, config)
            self.initialize_trial(trial_id, config)

        # Write master sweep config
        is_grouped = getattr(self.search_space, "_is_grouped", False)

        if is_grouped and GROUPED_SAMPLING_AVAILABLE:
            # Phase 5: Multi-stage sweep config with stage metadata
            sweep_config = {
                "max_trials": len(trials),
                "multi_stage": True,
                "search_space_type": "grouped",
                "grouped_config": self.search_space._grouped_config,
                "trials": [
                    {
                        "trial_id": tid,
                        "config": cfg,
                        "stage": stage,
                        "stage_name": ["COARSE", "REFINE", "FINE"][stage - 1],
                    }
                    for tid, cfg, stage in trials
                ],
                "created_at": time.time(),
            }
        else:
            # Legacy flat sweep config
            sweep_config = {
                "max_trials": self.max_trials,
                "multi_stage": False,
                "search_space_type": "flat",
                "search_space": {
                    "learning_rate": self.search_space.learning_rate,
                    "batch_size": self.search_space.batch_size,
                    "optimizer": self.search_space.optimizer,
                    "warmup_steps": self.search_space.warmup_steps,
                    "weight_decay": self.search_space.weight_decay,
                    "num_reasoning_blocks": self.search_space.num_reasoning_blocks,
                    "hidden_dim": self.search_space.hidden_dim,
                },
                "trials": [
                    {"trial_id": tid, "config": cfg, "stage": stage} for tid, cfg, stage in trials
                ],
                "created_at": time.time(),
            }

        sweep_file = self.metrics_collector.hpo_root / "sweep_config.json"
        with sweep_file.open("w", encoding="utf-8") as f:
            # Convert numpy types to Python native types for JSON serialization
            sweep_config_clean = convert_numpy_types(sweep_config)
            json.dump(sweep_config_clean, f, indent=2)
            f.write("\n")

        logger.info(f"[HPO Manager] Prepared sweep config: {sweep_file}")
        return sweep_file

    def get_trial_env(self, trial_id: str) -> dict[str, str]:
        """Get environment variables for a trial.

        Args:
            trial_id: Trial identifier

        Returns:
            Dictionary of environment variables
        """
        env = os.environ.copy()
        env["HPO_TRIAL_ID"] = trial_id
        env["HPO_TRIAL_DIR"] = str(self.metrics_collector.get_trial_dir(trial_id))
        return env

    def monitor_trials(self) -> dict[str, Any]:
        """Monitor all active trials and return their status.

        Returns:
            Dictionary with trial statuses and progress
        """
        trials = self.metrics_collector.list_trials()

        running_count = 0
        completed_count = 0
        failed_count = 0

        trial_statuses = []
        for trial_id in trials:
            status = self.metrics_collector.load_trial_status(trial_id)
            if status:
                trial_statuses.append(status.to_dict())

                if status.status == "running":
                    running_count += 1
                elif status.status == "completed":
                    completed_count += 1
                elif status.status == "failed":
                    failed_count += 1

        best = self.metrics_collector.get_best_trial()

        return {
            "total_trials": len(trials),
            "running": running_count,
            "completed": completed_count,
            "failed": failed_count,
            "trials": trial_statuses,
            "best_trial": (
                {
                    "trial_id": best[0],
                    "best_loss": best[1],
                }
                if best
                else None
            ),
        }
