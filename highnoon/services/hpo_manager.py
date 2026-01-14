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
from functools import lru_cache
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


# =============================================================================
# CACHED PARAMETER ESTIMATION
# =============================================================================
# Phase 950: LRU cache for estimate_model_params to avoid redundant computation
# in _estimate_max_sizes_for_budget triple loop (2000-3500+ iterations).


@lru_cache(maxsize=4096)
def _estimate_model_params_cached(
    vocab_size: int,
    embedding_dim: int,
    hd_dim: int,
    max_seq_len: int,
    num_blocks: int,
    num_experts: int,
    state_dim: int,
    floquet_modes: int,
    wavelet_kernel: int,
    qsvt_degree: int,
    vqc_layers: int,
    vqc_qubits: int,
    ff_expansion: int,
    use_qhd_spatial: bool,
    skip_connection_type: str,
    num_paths: int,
    entanglement_depth: int,
    entanglement_topology: str,
    superposition_dim: int,
    hd_dim_moe: int,
    active_vocab_size: int,
    # TT decomposition flags for accurate parameter estimation
    use_tt_embeddings: bool,
    use_tt_attention: bool,
    use_tt_ffn: bool,
    use_tt_kalman: bool,
    use_tt_lm_head: bool,
    tt_embedding_ranks: tuple[int, ...],
    tt_attention_ranks: tuple[int, ...],
    tt_ffn_ranks: tuple[int, ...],
    tt_kalman_ranks: tuple[int, ...],
    tt_lm_head_ranks: tuple[int, ...],
) -> int:
    """Cached inner implementation of parameter estimation.

    All parameters are passed as primitives to enable LRU caching.

    TT Decomposition Accuracy (Phase update):
    When USE_TT_* flags are enabled, this function uses TT-compressed
    parameter counts instead of dense counts, providing accurate estimates
    that reflect 85-98% parameter reduction from TT decomposition.
    """
    # FFN dimensions
    d_inner = embedding_dim * ff_expansion

    # ==========================================================================
    # 1. EMBEDDING LAYERS
    # ==========================================================================
    max_char_per_token = 64

    # TT-aware embedding parameter calculation
    if use_tt_embeddings and len(tt_embedding_ranks) >= 2:
        # TT decomposition: vocab_size × embedding_dim → Σ(r_i × d_i × r_{i+1})
        # Approximate as: vocab_size × r + r × embedding_dim (simplified 2-core)
        tt_r = tt_embedding_ranks[1] if len(tt_embedding_ranks) > 1 else 8
        standard_embed_params = active_vocab_size * tt_r + tt_r * embedding_dim
    else:
        standard_embed_params = active_vocab_size * embedding_dim

    hde_params = (
        256 * hd_dim  # hd_char_basis
        + max_char_per_token * hd_dim  # hd_position_basis
        + hd_dim * embedding_dim  # hd_projection
    )
    position_params = max_seq_len * embedding_dim
    total_embed_params = standard_embed_params + hde_params + position_params

    # ==========================================================================
    # 2. HD SPATIAL BLOCK
    # ==========================================================================
    needs_projection = embedding_dim != hd_dim
    hd_projection_params = (2 * embedding_dim * hd_dim) if needs_projection else 0

    hd_spatial_params = (
        state_dim  # a_log
        + hd_dim * state_dim  # b_proj
        + hd_dim * state_dim  # c_proj
        + hd_dim  # dt_proj
        + (
            hd_dim * hd_dim
            if skip_connection_type == "dense"
            else (
                hd_dim
                if skip_connection_type == "diagonal"
                else (1 if skip_connection_type == "learned_scalar" else 0)
            )
        )
        + hd_projection_params
    )

    # ==========================================================================
    # 2b. QHD SPATIAL BLOCK
    # ==========================================================================
    qhd_superposition_params = (
        num_paths  # amplitudes_real
        + num_paths  # amplitudes_imag
        + entanglement_depth * num_paths  # rotation_angles
    )
    qhd_walk_hamiltonian_params = (num_paths * num_paths) if entanglement_topology == "walk" else 0
    qhd_spatial_params = hd_spatial_params + qhd_superposition_params + qhd_walk_hamiltonian_params

    # ==========================================================================
    # 3. HD TIME CRYSTAL BLOCK
    # ==========================================================================
    hd_timecrystal_params = (
        floquet_modes * hd_dim
        + floquet_modes
        + floquet_modes * floquet_modes
        + hd_projection_params
    )

    # ==========================================================================
    # 4. LATENT REASONING BLOCK
    # ==========================================================================
    # TT-aware FFN parameter calculation
    if use_tt_ffn and len(tt_ffn_ranks) >= 2:
        # TT decomposition for FFN projections
        tt_r = tt_ffn_ranks[1] if len(tt_ffn_ranks) > 1 else 8
        ffn1_params = embedding_dim * tt_r + tt_r * d_inner + d_inner
        ffn2_params = d_inner * tt_r + tt_r * embedding_dim + embedding_dim
        latent_params = ffn1_params + ffn2_params + 4 * embedding_dim
    else:
        latent_params = (
            embedding_dim * d_inner
            + d_inner
            + d_inner * embedding_dim
            + embedding_dim
            + 4 * embedding_dim
        )

    # ==========================================================================
    # 4b. UNIFIED ATTENTION (Phase 2 V2 Update - O(n) Linear)
    # ==========================================================================
    # The unified_attention_op provides O(n) linear attention with holographic
    # FFT features, replacing O(n²) quadratic attention. Key differences:
    # - Old: Q×K^T attention matrix = O(n²) params for positional biases
    # - New: KV state = O(d²) per head, independent of sequence length
    #
    # Parameters include:
    # - Q/K/V projection weights: 3 × embedding_dim × embedding_dim
    # - Output projection: embedding_dim × embedding_dim
    # - Holographic FFT has no learnable params (just FFT/iFFT operations)
    # - Linear attention KV state is an activation, not a parameter
    num_heads_attn = 8  # Default, matches config
    embedding_dim // num_heads_attn if num_heads_attn > 0 else 64

    # TT-aware attention projection calculation
    if use_tt_attention and len(tt_attention_ranks) >= 2:
        # TT decomposition for Q/K/V/O projections
        tt_r = tt_attention_ranks[1] if len(tt_attention_ranks) > 1 else 4
        # Each projection: embedding_dim × tt_r + tt_r × embedding_dim
        proj_params = embedding_dim * tt_r + tt_r * embedding_dim
        unified_attention_params = (
            4 * proj_params + embedding_dim  # Q, K, V, O projections (TT-compressed)  # Layer norm
        )
    else:
        unified_attention_params = (
            3 * embedding_dim * embedding_dim  # Q, K, V projections
            + embedding_dim * embedding_dim  # Output projection
            + embedding_dim  # Layer norm
        )

    # ==========================================================================
    # 4c. KALMAN BLOCK (after TimeCrystal when USE_KALMAN_FILTERING=True)
    # ==========================================================================
    # KalmanBlock provides state estimation with uncertainty tracking.
    # Added after TimeCrystalBlock in block_factory.py block_type==1.
    # Parameters: state projections, Kalman gain, measurement matrix
    kalman_state_dim = 32  # Default from config.KALMAN_STATE_DIM

    # TT-aware Kalman projection calculation
    if use_tt_kalman and len(tt_kalman_ranks) >= 2:
        # TT decomposition for Kalman projections
        tt_r = tt_kalman_ranks[1] if len(tt_kalman_ranks) > 1 else 4
        state_proj_params = embedding_dim * tt_r + tt_r * kalman_state_dim
        measurement_proj_params = kalman_state_dim * tt_r + tt_r * embedding_dim
        kalman_params = (
            state_proj_params  # state_proj (TT)
            + measurement_proj_params  # measurement_proj (TT)
            + kalman_state_dim * kalman_state_dim  # kalman_gain
            + kalman_state_dim  # innovation_bias
            + embedding_dim  # output_bias
        )
    else:
        kalman_params = (
            embedding_dim * kalman_state_dim  # state_proj
            + kalman_state_dim * embedding_dim  # measurement_proj
            + kalman_state_dim * kalman_state_dim  # kalman_gain
            + kalman_state_dim  # innovation_bias
            + embedding_dim  # output_bias
        )
    # Kalman is added only on TimeCrystal blocks (1 out of 6 in pattern)
    # So we count 1 Kalman per 6 blocks
    num_blocks // 6

    # ==========================================================================
    # 5. WLAM BLOCK (block_type==4 in block_factory.py)
    # ==========================================================================
    # WLAMBlock: Wavelet-based Linear Attention Module
    # Uses multi-scale wavelet decomposition with linear attention
    num_heads = 8  # Default from config.WLAM_NUM_HEADS
    wlam_params = (
        4 * wavelet_kernel * embedding_dim * embedding_dim  # wavelet filters
        + 2 * embedding_dim  # layer norms
        + num_heads * (embedding_dim // num_heads) * 3  # QKV per head
    )

    # ==========================================================================
    # 6. MOE BLOCK
    # ==========================================================================
    tt_rank = 16
    d_ff = embedding_dim * ff_expansion
    ffn1_tt_params = (embedding_dim * tt_rank + tt_rank * d_ff) * superposition_dim
    ffn2_tt_params = (d_ff * tt_rank + tt_rank * embedding_dim) * superposition_dim
    moe_uses_hd_projection = hd_dim_moe != embedding_dim
    moe_hd_projection_params = (2 * embedding_dim * hd_dim_moe) if moe_uses_hd_projection else 0
    holographic_routing_params = 2 * superposition_dim * hd_dim_moe
    superposed_expert_params = (
        ffn1_tt_params + ffn2_tt_params + moe_hd_projection_params + holographic_routing_params
    )
    router_params = embedding_dim * num_experts + num_experts
    moe_params = superposed_expert_params + router_params

    # ==========================================================================
    # 7. QUANTUM ENHANCED BLOCK WRAPPER
    # ==========================================================================
    quantum_enhanced_overhead = (
        embedding_dim * embedding_dim
        + embedding_dim
        + embedding_dim * embedding_dim
        + (qsvt_degree + 1)
    )

    # ==========================================================================
    # BLOCK PATTERN ASSEMBLY (matches block_factory.py create_reasoning_stack)
    # ==========================================================================
    # Phase 2 V2 Update: Unified attention (O(n) linear) replaces O(n²) attention
    # Blocks are now more parameter-efficient due to linear attention's O(d²) KV state
    #
    # Pattern: 6 unique blocks repeating, matching block_factory.py block_type % 6:
    #   0: QHDSpatialBlock (or HDSpatialBlock)
    #   1: HDTimeCrystalBlock + KalmanBlock
    #   2: LatentReasoningBlock + SelfConsistencyVerifier
    #   3: LatentKVAttention / QuantumGQA / LocalAttentionBlock
    #   4: WLAMBlock
    #   5: MoELayer (SuperposedExpert)
    spatial_block_params = qhd_spatial_params if use_qhd_spatial else hd_spatial_params
    blocks_in_pattern = [
        # block_type 0: QHDSpatialBlock
        spatial_block_params + quantum_enhanced_overhead,
        # block_type 1: HDTimeCrystalBlock + KalmanBlock (when USE_KALMAN_FILTERING)
        hd_timecrystal_params + kalman_params + quantum_enhanced_overhead,
        # block_type 2: LatentReasoningBlock (uses unified attention internally)
        latent_params + unified_attention_params + quantum_enhanced_overhead,
        # block_type 3: Attention block (LatentKV/QuantumGQA/Local/Spatial)
        unified_attention_params + quantum_enhanced_overhead,
        # block_type 4: WLAMBlock (uses wavelet + unified attention)
        wlam_params + unified_attention_params + quantum_enhanced_overhead,
        # block_type 5: MoELayer (SuperposedExpert with holographic routing)
        moe_params + quantum_enhanced_overhead,
    ]
    num_full_patterns = num_blocks // 6
    remaining_blocks = num_blocks % 6
    params_per_pattern = sum(blocks_in_pattern)
    total_reasoning_params = num_full_patterns * params_per_pattern
    for i in range(remaining_blocks):
        total_reasoning_params += blocks_in_pattern[i]

    # ==========================================================================
    # 8. QUANTUM LM HEAD
    # ==========================================================================
    vqc_dim = 2**vqc_qubits

    # TT-aware LM head parameter calculation
    if use_tt_lm_head and len(tt_lm_head_ranks) >= 2:
        # TT decomposition for output projection: embedding_dim × vocab_size
        # → embedding_dim × r + r × vocab_size
        tt_r = tt_lm_head_ranks[1] if len(tt_lm_head_ranks) > 1 else 8
        output_proj_params = embedding_dim * tt_r + tt_r * vocab_size + vocab_size
        qlm_params = (
            embedding_dim * vqc_dim
            + vqc_layers * vqc_qubits * 3
            + vqc_layers * (vqc_qubits - 1)
            + vqc_dim * tt_r
            + tt_r * vocab_size
            + vocab_size  # VQC output (TT)
            + output_proj_params  # Final projection (TT)
            + 1
        )
    else:
        qlm_params = (
            embedding_dim * vqc_dim
            + vqc_layers * vqc_qubits * 3
            + vqc_layers * (vqc_qubits - 1)
            + vqc_dim * vocab_size
            + vocab_size
            + embedding_dim * vocab_size
            + vocab_size
            + 1
        )

    return int(total_embed_params + total_reasoning_params + qlm_params)


def estimate_model_params(config: dict[str, Any]) -> int:
    """Estimate total model parameters from architecture configuration.

    Phase 300 Update: Accurate HD-only architecture parameter estimation.
    Phase 950 Update: LRU-cached implementation for 10-50x speedup in HPO loops.
    Phase 2 V2 Update: Unified attention (O(n) linear) replaces O(n²) quadratic,
        reducing attention parameter overhead for long sequences.

    Args:
        config: Architecture configuration containing:
            - vocab_size: Vocabulary size (default 32000)
            - embedding_dim: Embedding dimension (default 512)
            - hd_dim: HD space dimension (default embedding_dim * 8)
            - max_seq_len: Maximum sequence length (default 4096)
            - num_reasoning_blocks: Number of reasoning blocks (default 8)
            - num_moe_experts: Number of MoE experts (default 8)
            - mamba_state_dim: Mamba state dimension for HDSpatialBlock (default 16)
            - floquet_modes: Floquet modes for HDTimeCrystalBlock (default 16)
            - wavelet_kernel_size: Wavelet kernel for WLAMBlock (default 4)
            - vqc_layers: VQC layers for QuantumLMHead (default 2)
            - vqc_qubits: VQC qubits for QuantumLMHead (default 8)
            - qsvt_degree: QSVT degree for QuantumEnhancedBlock (default 8)
            - param_budget: Optional parameter budget for smart defaults

    Returns:
        Estimated total parameter count.
    """
    # ==========================================================================
    # EXTRACT ALL CONFIG VALUES (Phase 950: for cache key hashing)
    # ==========================================================================
    embedding_dim = (
        config.get("embedding_dim")
        or config.get("hidden_dim")
        or getattr(hn_config, "EMBEDDING_DIM", 512)
    )
    hd_dim = config.get("hd_dim") or getattr(hn_config, "HD_EMBEDDING_DIM", embedding_dim * 8)
    max_seq_len = config.get("max_seq_len") or config.get("context_window") or 4096
    param_budget = config.get("param_budget")

    # Budget-aware vocab_size default
    vocab_size = config.get("vocab_size") or config.get("target_vocab_size")
    if vocab_size is None:
        if param_budget is not None:
            vocab_size = compute_max_vocab_for_budget(
                param_budget=param_budget,
                embedding_dim=embedding_dim,
                use_hqe=True,
                hd_dim=hd_dim,
                model_overhead_fraction=0.5,
            )
            logger.debug(
                f"[HPO] Budget-aware vocab default: {vocab_size} (budget={param_budget / 1e6:.0f}M)"
            )
        else:
            vocab_size = 32000

    num_blocks = config.get("num_reasoning_blocks") or 8
    num_experts = config.get("num_moe_experts") or 8
    state_dim = config.get("mamba_state_dim") or 16
    floquet_modes = config.get("floquet_modes") or config.get("majorana_floquet_period") or 16
    wavelet_kernel = config.get("wavelet_kernel_size") or 4
    qsvt_degree = config.get("qsvt_degree") or 8
    vqc_layers = config.get("vqc_layers") or config.get("vqc_max_layers") or 2
    vqc_qubits = config.get("vqc_qubits") or 8
    ff_expansion = config.get("ff_expansion") or 4
    use_qhd_spatial = (
        config.get("use_qhd_spatial_block")
        if config.get("use_qhd_spatial_block") is not None
        else True
    )
    skip_connection_type = config.get("skip_connection_type", "diagonal")
    num_paths = config.get("qhd_num_paths") or getattr(hn_config, "QHD_NUM_PATHS", 2)
    entanglement_depth = config.get("qhd_entanglement_depth") or getattr(
        hn_config, "QHD_ENTANGLEMENT_DEPTH", 2
    )
    entanglement_topology = config.get("qhd_entanglement_topology") or getattr(
        hn_config, "QHD_ENTANGLEMENT_TOPOLOGY", "walk"
    )
    superposition_dim = config.get("superposition_dim") or getattr(
        hn_config, "SUPERPOSITION_DIM", 4
    )
    hd_dim_moe = config.get("hd_dim_moe") or getattr(hn_config, "HD_DIM_MOE", embedding_dim)
    active_vocab_size = config.get("hd_active_vocab_size") or getattr(
        hn_config, "HD_ACTIVE_VOCAB_SIZE", 10000
    )

    # ==========================================================================
    # TT DECOMPOSITION FLAGS (for accurate parameter estimation)
    # ==========================================================================
    # Extract TT flags from config, falling back to global config defaults
    use_tt_embeddings = (
        config.get("use_tt_embeddings")
        if config.get("use_tt_embeddings") is not None
        else getattr(hn_config, "USE_TT_EMBEDDINGS", True)
    )
    use_tt_attention = (
        config.get("use_tt_attention_projections")
        if config.get("use_tt_attention_projections") is not None
        else getattr(hn_config, "USE_TT_ATTENTION_PROJECTIONS", True)
    )
    use_tt_ffn = (
        config.get("use_tt_ffn_projections")
        if config.get("use_tt_ffn_projections") is not None
        else getattr(hn_config, "USE_TT_FFN_PROJECTIONS", True)
    )
    use_tt_kalman = (
        config.get("use_tt_kalman_projections")
        if config.get("use_tt_kalman_projections") is not None
        else getattr(hn_config, "USE_TT_KALMAN_PROJECTIONS", True)
    )
    use_tt_lm_head = (
        config.get("use_tt_lm_head")
        if config.get("use_tt_lm_head") is not None
        else getattr(hn_config, "USE_TT_LM_HEAD", True)
    )

    # TT rank configurations (convert lists to tuples for hashability in LRU cache)
    tt_embedding_ranks = tuple(
        config.get("tt_embedding_ranks") or getattr(hn_config, "TT_EMBEDDING_RANKS", [1, 8, 8, 1])
    )
    tt_attention_ranks = tuple(
        config.get("tt_attention_ranks") or getattr(hn_config, "TT_ATTENTION_RANKS", [1, 4, 4, 1])
    )
    tt_ffn_ranks = tuple(
        config.get("tt_ffn_ranks") or getattr(hn_config, "TT_FFN_RANKS", [1, 8, 8, 1])
    )
    tt_kalman_ranks = tuple(
        config.get("tt_kalman_ranks") or getattr(hn_config, "TT_KALMAN_RANKS", [1, 4, 1])
    )
    tt_lm_head_ranks = tuple(
        config.get("tt_lm_head_ranks") or getattr(hn_config, "TT_LM_HEAD_RANKS", [1, 8, 8, 1])
    )

    # ==========================================================================
    # DELEGATE TO CACHED IMPLEMENTATION (Phase 950: 10-50x speedup)
    # ==========================================================================
    return _estimate_model_params_cached(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hd_dim=hd_dim,
        max_seq_len=max_seq_len,
        num_blocks=num_blocks,
        num_experts=num_experts,
        state_dim=state_dim,
        floquet_modes=floquet_modes,
        wavelet_kernel=wavelet_kernel,
        qsvt_degree=qsvt_degree,
        vqc_layers=vqc_layers,
        vqc_qubits=vqc_qubits,
        ff_expansion=ff_expansion,
        use_qhd_spatial=use_qhd_spatial,
        skip_connection_type=skip_connection_type,
        num_paths=num_paths,
        entanglement_depth=entanglement_depth,
        entanglement_topology=entanglement_topology,
        superposition_dim=superposition_dim,
        hd_dim_moe=hd_dim_moe,
        active_vocab_size=active_vocab_size,
        # TT decomposition flags
        use_tt_embeddings=use_tt_embeddings,
        use_tt_attention=use_tt_attention,
        use_tt_ffn=use_tt_ffn,
        use_tt_kalman=use_tt_kalman,
        use_tt_lm_head=use_tt_lm_head,
        tt_embedding_ranks=tt_embedding_ranks,
        tt_attention_ranks=tt_attention_ranks,
        tt_ffn_ranks=tt_ffn_ranks,
        tt_kalman_ranks=tt_kalman_ranks,
        tt_lm_head_ranks=tt_lm_head_ranks,
    )


def estimate_peak_memory(config: dict[str, Any]) -> int:
    """Phase 900.2: Estimate peak memory in bytes for a model configuration.

    Unlike estimate_model_params which counts trainable parameters, this function
    estimates total memory consumption including:
    - Model parameters (weights)
    - Optimizer state (momentum, QFIM diagonal for SympFlowQNG)
    - Activation memory during forward pass
    - HD intermediate tensors (tokens_hd, bound_hd)
    - FFT buffers for circular convolution (in-place: 4× reduction)
    - dt tensor (1D since Phase 900.2 C++ broadcast)

    Phase 1 Memory Roadmap Update:
    - Streaming buffers now use STREAMING_CHUNK_SIZE instead of context_window
    - Memory is O(batch × chunk_size × max(hd_dim, embedding_dim)) for streaming ops
    - This makes memory independent of sequence length for long sequences

    This helps prevent OOM errors during HPO by rejecting configs that would
    exceed available memory before model construction.

    Args:
        config: Architecture configuration (same as estimate_model_params)

    Returns:
        Estimated peak memory in bytes.

    Example:
        >>> memory = estimate_peak_memory({"context_window": 4096, "embedding_dim": 512})
        >>> print(f"Estimated memory: {memory / 1e9:.2f} GB")
    """
    # Core dimensions
    embedding_dim = config.get("embedding_dim") or config.get("hidden_dim") or 512
    hd_dim = config.get("hd_dim") or embedding_dim * 8
    context_window = config.get("context_window") or config.get("max_seq_len") or 4096
    batch_size = config.get("batch_size") or 1
    num_blocks = config.get("num_reasoning_blocks") or 8

    # Phase 1 Memory Roadmap: Get streaming chunk size from config
    # Streaming-enabled ops use fixed chunk sizes independent of seq_len
    # Note: Use explicit None checks to respect False values
    streaming_enabled = config.get("streaming_enabled")
    if streaming_enabled is None:
        streaming_enabled = getattr(hn_config, "STREAMING_ENABLED", True)
    streaming_chunk_size = config.get("streaming_chunk_size") or getattr(
        hn_config, "STREAMING_CHUNK_SIZE", 128
    )

    # HD bundling configuration from Phase 2
    use_hd_bundling = config.get("use_hd_token_bundling")
    if use_hd_bundling is None:
        use_hd_bundling = getattr(hn_config, "USE_HD_TOKEN_BUNDLING", True)
    hd_bundle_size = config.get("hd_bundle_size") or getattr(hn_config, "HD_BUNDLE_SIZE", 128)

    # Clamp hd_dim to max (Phase 900.2 memory guard)
    hd_dim_max = getattr(hn_config, "HD_EMBEDDING_DIM_MAX", 8192)
    if hd_dim > hd_dim_max:
        logger.warning(f"[Memory] hd_dim {hd_dim} exceeds max {hd_dim_max}, clamping")
        hd_dim = hd_dim_max

    # 1. Model parameters (from estimate_model_params)
    param_count = estimate_model_params(config)
    param_memory = param_count * 4  # float32

    # 2. Optimizer state: SympFlowQNG uses 3x params (weights + momentum + qfim_diag)
    # AdamW uses 2x (momentum + variance)
    optimizer = config.get("optimizer", "sympflowqng")
    if optimizer in ("sympflowqng", "qiao", "sophia"):
        optimizer_multiplier = 3
    else:
        optimizer_multiplier = 2
    optimizer_memory = param_count * 4 * optimizer_multiplier

    # 3. Position encoding memory
    # Phase 900.1: StreamingHDPosition uses O(D) instead of O(L×D)
    position_memory = embedding_dim * 4  # O(D) with streaming positions

    # 4. Activation memory during forward pass
    # Input/output: [batch, seq, dim] × 2
    # Note: This is still O(seq) because TensorFlow needs full input/output
    activation_memory = batch_size * context_window * embedding_dim * 4 * 2

    # 5. HD intermediate tensors (Phase 1 Memory Roadmap: streaming-aware)
    # With streaming enabled, HD ops use fixed chunk size instead of full context
    # With HD bundling (Phase 2), we compress tokens by bundle_size factor
    if streaming_enabled:
        # Streaming mode: memory proportional to chunk_size, not context_window
        effective_seq_for_hd = streaming_chunk_size
    elif use_hd_bundling:
        # HD Bundling: effective sequence is reduced by bundle factor
        num_bundles = max(1, context_window // hd_bundle_size)
        effective_seq_for_hd = num_bundles
    else:
        # Legacy: full context window (memory explosion risk)
        effective_seq_for_hd = context_window

    hd_activation_memory = batch_size * effective_seq_for_hd * hd_dim * 4 * 2

    # 6. FFT buffers for circular convolution
    # Phase 1 Memory Roadmap: FFT buffers use streaming chunk size
    # Phase 900.2: In-place FFT uses 2 buffers of [batch_size, chunk, hd_dim * 2] (complex)
    fft_buffer_memory = batch_size * effective_seq_for_hd * hd_dim * 8 * 2  # 2 complex buffers

    # 7. Unified Attention Memory (Phase 2 V2 update)
    # The unified_attention_op uses O(n) linear attention with holographic FFT features.
    # Memory includes: Q, K, V tensors + attention output + FFT feature maps
    num_heads = config.get("num_heads", 8) or config.get("wlam_num_heads", 8)
    head_dim = embedding_dim // num_heads if num_heads > 0 else 64

    # Q, K, V tensors: [batch, heads, seq, head_dim] × 3
    qkv_memory = batch_size * num_heads * context_window * head_dim * 4 * 3

    # Attention output: [batch, seq, embedding_dim]
    attn_output_memory = batch_size * context_window * embedding_dim * 4

    # Holographic FFT feature maps when use_holographic_features=True (default for HD models)
    # Features are [batch, heads, seq, head_dim] real + imaginary parts
    use_holographic = config.get("use_holographic_features", True)
    if use_holographic:
        holographic_feature_memory = (
            batch_size * num_heads * context_window * head_dim * 4 * 4
        )  # Q/K real+imag
    else:
        holographic_feature_memory = 0

    # Linear attention KV state: [batch, heads, head_dim, head_dim] - O(d²) not O(n²)
    linear_kv_state_memory = batch_size * num_heads * head_dim * head_dim * 4

    unified_attention_memory = (
        qkv_memory + attn_output_memory + holographic_feature_memory + linear_kv_state_memory
    )

    # 8. QHD block intermediate memory
    # Phase 1 Memory Roadmap: dt and state use streaming chunk size
    # Phase 900.2: dt is now 1D [hd_dim] per block (C++ broadcasts internally)
    dt_memory_per_block = hd_dim * 4  # 1D dt (broadcasted in C++)
    block_state_memory = batch_size * hd_dim * 8  # SSM state (complex)
    block_memory = (dt_memory_per_block + block_state_memory) * num_blocks

    # 9. TensorFlow graph overhead (~20% of computed tensors)
    computed_memory = (
        activation_memory
        + hd_activation_memory
        + fft_buffer_memory
        + unified_attention_memory
        + block_memory
    )
    tf_overhead = int(computed_memory * 0.2)

    # Total
    total = (
        param_memory
        + optimizer_memory
        + position_memory
        + activation_memory
        + hd_activation_memory
        + fft_buffer_memory
        + unified_attention_memory
        + block_memory
        + tf_overhead
    )

    logger.debug(
        f"[Memory Estimate] params={param_memory / 1e6:.1f}MB, "
        f"optimizer={optimizer_memory / 1e6:.1f}MB, "
        f"positions={position_memory / 1e3:.1f}KB, "
        f"activations={activation_memory / 1e6:.1f}MB, "
        f"hd_activations={hd_activation_memory / 1e6:.1f}MB (streaming={streaming_enabled}), "
        f"fft_buffers={fft_buffer_memory / 1e6:.1f}MB, "
        f"attention={unified_attention_memory / 1e6:.1f}MB, "
        f"blocks={block_memory / 1e6:.1f}MB, "
        f"overhead={tf_overhead / 1e6:.1f}MB, "
        f"total={total / 1e9:.2f}GB"
    )

    return total


def check_memory_budget(
    config: dict[str, Any],
    available_memory_gb: float = 32.0,
    safety_margin: float = 0.8,
) -> tuple[bool, str]:
    """Phase 900.1: Check if a config fits within available memory.

    Phase 950 Update: Pre-computed context window sizes for O(1) suggestion lookup
    instead of iterative halving.

    Args:
        config: Model configuration.
        available_memory_gb: Available system memory in GB.
        safety_margin: Fraction of available memory to use (default 0.8 = 80%).

    Returns:
        Tuple of (fits, message) where fits is True if config fits in memory.
    """
    estimated = estimate_peak_memory(config)
    budget = int(available_memory_gb * 1e9 * safety_margin)

    if estimated <= budget:
        return True, f"Estimated {estimated / 1e9:.2f}GB fits within {budget / 1e9:.2f}GB budget"

    # Phase 950: Pre-computed common context window sizes (power-of-2 sequence)
    # Binary search to find largest fitting context_window
    context_window = config.get("context_window") or 4096
    common_contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    # Filter to sizes smaller than current and use binary search via list comprehension
    candidates = [c for c in common_contexts if c < context_window]

    suggested_context = 512  # Fallback minimum
    for ctx in reversed(candidates):  # Check from largest to smallest
        test_config = {**config, "context_window": ctx}
        if estimate_peak_memory(test_config) <= budget:
            suggested_context = ctx
            break

    return False, (
        f"Estimated {estimated / 1e9:.2f}GB exceeds {budget / 1e9:.2f}GB budget. "
        f"Consider reducing context_window from {context_window} to {suggested_context}."
    )


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
    learning_rate: tuple[float, float] = (1e-6, 5e-4)  # Widened min for frontier
    batch_size: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16]
    )  # Capped at 16 for 64GB local testing
    optimizer: list[str] = field(default_factory=lambda: ["sophiag", "adam", "adamw", "lion"])
    warmup_steps: tuple[int, int] = (0, 2500)  # Widened for frontier models
    weight_decay: tuple[float, float] = (0.0, 0.3)  # Widened for stronger regularization

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
    #
    # Phase 1 Tokenizer Fix: target_vocab_size replaces vocab_size
    # This is the TARGET size for the tokenizer to learn. The actual model vocab_size
    # is derived from tokenizer.vocab_size after learning, ensuring zero dead embeddings.
    target_vocab_size: list[int] = field(
        default_factory=lambda: [
            1000,
            2000,
            4000,
            8000,
            12000,
            16000,
            24000,
            32000,
            48000,
            64000,
            96000,
            128000,
            192000,
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
    superposition_dim: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6]
    )  # MoE superposition
    tt_rank_middle: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    hamiltonian_hidden_dim: list[int] = field(
        default_factory=lambda: [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    )

    # =========================================================================
    # UQHA PHASE 1: DIAGONAL SKIP CONNECTION (P0)
    # =========================================================================
    # "dense" is excluded from default search due to extreme memory cost (O(D²)).
    # "diagonal" matches performance of dense but with O(D) params.
    skip_connection_type: list[str] = field(
        default_factory=lambda: ["diagonal", "learned_scalar", "identity"]
    )
    skip_diagonal_init: tuple[float, float] = (0.5, 1.5)

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
    entropy_reg_weight: tuple[float, float] = (0.0005, 0.15)  # Widened
    spectral_reg_weight: tuple[float, float] = (0.0005, 0.1)  # Widened
    target_entropy: tuple[float, float] = (0.2, 0.9)  # Widened

    # =========================================================================
    # PHASE 46: SYMPFLOW OPTIMIZER
    # =========================================================================
    sympflow_mass: tuple[float, float] = (0.2, 4.0)  # Widened
    sympflow_friction: tuple[float, float] = (0.02, 0.5)  # Widened
    sympflow_step_size: tuple[float, float] = (0.002, 0.1)  # Widened

    # =========================================================================
    # PHASE 47: QUANTUM MEASUREMENT DROPOUT
    # =========================================================================
    qmd_drop_rate: tuple[float, float] = (0.02, 0.5)  # Widened
    qmd_softening_temp: tuple[float, float] = (0.3, 3.0)  # Widened

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
    # PHASE 43: NEURAL KALMAN PARAMETERS
    # GRU-based learned Kalman gain with tunable uncertainty calibration.
    # See config.py L629-632 for QAHPO tunable range documentation.
    # =========================================================================
    neural_kalman_hidden_dim: list[int] = field(
        default_factory=lambda: [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
    )
    neural_kalman_process_noise: tuple[float, float] = (1e-5, 0.5)
    neural_kalman_measurement_noise: tuple[float, float] = (1e-4, 2.0)

    # =========================================================================
    # PHASE 58: SYMPLECTIC GNN KALMAN
    # =========================================================================
    sgkf_dt: tuple[float, float] = (0.005, 0.05)

    # =========================================================================
    # PHASE 59: ADIABATIC OPTIMIZER
    # =========================================================================
    qao_initial_temp: tuple[float, float] = (3.0, 30.0)  # Widened
    qao_final_temp: tuple[float, float] = (0.001, 0.1)  # Widened

    # =========================================================================
    # PHASE 60: GEODESIC OPTIMIZER
    # =========================================================================
    geodesic_momentum: tuple[float, float] = (0.7, 0.995)  # Widened

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
    crystallization_threshold: tuple[float, float] = (0.2, 0.9)  # Widened
    qhpm_crystallization_rate: tuple[float, float] = (0.02, 0.4)  # Widened

    # =========================================================================
    # PHASE 68: NEUROMORPHIC MEMORY
    # =========================================================================
    neuromorphic_tau: tuple[float, float] = (2.0, 50.0)  # Widened

    # =========================================================================
    # PHASE 70: MULTI-STAGE HAMILTONIAN (Lite: max 6 stages)
    # Phase 201.7: Expanded stage counts
    # =========================================================================
    hamiltonian_num_stages: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])

    # =========================================================================
    # PHASE 71: INTRINSIC PLASTICITY
    # =========================================================================
    plasticity_learning_rate: tuple[float, float] = (0.001, 0.1)  # Widened

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
    coconut_num_paths: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4]
    )  # BFS branches for thought exploration
    coconut_max_thought_steps: list[int] = field(
        default_factory=lambda: [2, 4, 6, 8, 10, 12, 14, 16]
    )  # Max reasoning iterations
    coconut_collapse_threshold: tuple[float, float] = (0.3, 0.98)
    coconut_crystallize_threshold: tuple[float, float] = (0.5, 0.995)

    # =========================================================================
    # PHASE 127: UNIFIED QUANTUM BUS
    # Phase 201.7: Fully expanded bond dimensions
    # =========================================================================
    unified_bus_mps_bond_dim: list[int] = field(
        default_factory=lambda: [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    )
    unified_bus_coherence_threshold: tuple[float, float] = (0.5, 0.99)  # Wide range
    unified_bus_entanglement_init: tuple[float, float] = (0.1, 0.9)  # Wide range
    unified_bus_propagation_rate: tuple[float, float] = (0.01, 0.5)  # Wide range

    # =========================================================================
    # ATTENTION CONFIGURATION PARAMETERS - WIDE RANGES
    # =========================================================================
    local_attention_window: list[int] = field(
        default_factory=lambda: [32, 64, 128, 192, 256, 384, 512, 768, 1024, 2048]  # Wide
    )
    local_attention_window_min: list[int] = field(
        default_factory=lambda: [16, 32, 48, 64, 96, 128]  # Multi-scale min
    )
    local_attention_window_max: list[int] = field(
        default_factory=lambda: [256, 384, 512, 768, 1024, 2048]  # Multi-scale max
    )
    local_attention_sigmoid_temp: tuple[float, float] = (0.1, 4.0)  # Wide range
    local_attention_sparsity_ratio: tuple[float, float] = (0.1, 0.95)  # Wide range

    # =========================================================================
    # FLASH LINEAR ATTENTION PARAMETERS - WIDE RANGES
    # =========================================================================
    flash_linear_chunk_size: list[int] = field(
        default_factory=lambda: [16, 32, 48, 64, 96, 128, 192, 256]  # Wide range
    )
    flash_linear_train_chunk_size: list[int] = field(
        default_factory=lambda: [64, 128, 192, 256, 384, 512, 768, 1024]  # Wide range
    )
    flash_linear_hybrid_window: list[int] = field(
        default_factory=lambda: [0, 32, 64, 128, 256, 512, 1024]  # 0 = disable
    )
    flash_linear_hybrid_alpha: tuple[float, float] = (0.0, 1.0)  # Full range
    flash_linear_augment_rank: list[int] = field(
        default_factory=lambda: [0, 2, 4, 8, 12, 16, 24, 32]  # RALA rank
    )
    flash_linear_gate_init_bias: tuple[float, float] = (-6.0, 0.0)  # Wide range

    # =========================================================================
    # STATE BUS PARAMETERS - WIDE RANGES
    # =========================================================================
    state_bus_dim: list[int] = field(
        default_factory=lambda: [16, 32, 48, 64, 96, 128, 192, 256]  # Wide range
    )
    state_bus_slots: list[int] = field(
        default_factory=lambda: [2, 4, 6, 8, 12, 16, 24, 32]  # Wide range
    )
    state_bus_max_slots: list[int] = field(
        default_factory=lambda: [8, 12, 16, 24, 32, 48, 64]  # Wide range
    )
    state_bus_num_types: list[int] = field(
        default_factory=lambda: [2, 3, 4, 6, 8, 10, 12]  # Wide range
    )
    state_bus_bond_dim: list[int] = field(
        default_factory=lambda: [2, 4, 6, 8, 12, 16]  # TT superposition bond
    )

    # =========================================================================
    # EXTERNAL MEMORY (GEM) PARAMETERS - WIDE RANGES
    # =========================================================================
    memory_slots: list[int] = field(
        default_factory=lambda: [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]  # Wide
    )
    memory_slot_dim: list[int] = field(
        default_factory=lambda: [64, 128, 192, 256, 384, 512, 768, 1024]  # Wide range
    )
    memory_surprise_threshold: tuple[float, float] = (0.001, 0.5)  # Wide range
    memory_product_k: list[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 24, 32, 48, 64]  # Wide range
    )
    memory_num_heads: list[int] = field(
        default_factory=lambda: [1, 2, 4, 6, 8, 12, 16]  # Wide range
    )
    memory_mps_bond_dim: list[int] = field(
        default_factory=lambda: [4, 8, 16, 24, 32, 48, 64]  # Wide range
    )

    # =========================================================================
    # CORE CHUNKING PARAMETERS - WIDE RANGES
    # =========================================================================
    chunk_size: list[int] = field(
        default_factory=lambda: [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]  # Wide
    )
    chunk_stride: list[int] = field(
        default_factory=lambda: [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]  # Wide
    )

    # =========================================================================
    # SSM MAMBA2 PARAMETERS - WIDE RANGES
    # =========================================================================
    mamba2_conv_dim: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 8])  # Wide range
    mamba2_head_dim: list[int] = field(
        default_factory=lambda: [16, 32, 48, 64, 96, 128, 192, 256]  # Wide range
    )
    mamba2_expand_factor: list[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]  # Wide range
    )

    # =========================================================================
    # DEPRECATED: RG-LRU and MinGRU blocks removed from architecture in Phase 200+
    # These parameters are no longer tuned (HD blocks replace legacy recurrent units)
    # =========================================================================

    # =========================================================================
    # CPU-SPECIFIC OPTIMIZATION PARAMETERS - WIDE RANGES
    # =========================================================================
    cache_tile_size: list[int] = field(
        default_factory=lambda: [16, 32, 48, 64, 96, 128, 192, 256]  # L2 cache
    )
    prefetch_distance: list[int] = field(
        default_factory=lambda: [2, 4, 6, 8, 12, 16, 24, 32]  # Wide range
    )
    ssd_chunk_size: list[int] = field(
        default_factory=lambda: [16, 32, 48, 64, 96, 128, 192, 256]  # Wide range
    )
    superposition_micro_batch_size: list[int] = field(
        default_factory=lambda: [8, 16, 24, 32, 48, 64, 96, 128]  # Wide range
    )

    # =========================================================================
    # PHASE 130: QUANTUM SYNERGY PARAMETERS
    # Phase 201.7: Expanded VQC and synergy parameters with WIDE RANGES
    # =========================================================================
    # S3: VQC Adaptive Depth (expanded for frontier exploration)
    vqc_min_layers: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    vqc_max_layers: list[int] = field(default_factory=lambda: [4, 5, 6, 7, 8, 10, 12])  # Wide range
    vqc_qubits: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 8])  # Circuit width
    vqc_feature_rotation_depth: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6]  # Feature encoding depth
    )
    neumann_cayley_terms: list[int] = field(
        default_factory=lambda: [2, 3, 4, 6, 8, 10, 12]  # Unitary approximation order
    )
    neumann_series_terms: list[int] = field(
        default_factory=lambda: [2, 3, 4, 5, 6, 8, 10]  # General Neumann series
    )
    # S6: Hopfield base beta (MPS Entropy → Hopfield β adaptation)
    hopfield_base_beta: tuple[float, float] = (0.05, 5.0)  # Wide range
    hopfield_beta_min: tuple[float, float] = (0.05, 1.5)
    hopfield_beta_max: tuple[float, float] = (1.0, 15.0)  # Wide range
    # S10: DTC Floquet Period (Bus Entanglement → Floquet Period)
    dtc_floquet_period: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 8, 10, 12, 16])
    dtc_coupling_j: tuple[float, float] = (0.05, 3.0)  # Wide Heisenberg coupling
    dtc_disorder_w: tuple[float, float] = (0.05, 2.0)  # Wide MBL disorder
    dtc_num_cycles: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])  # Floquet cycles
    # S12: SympFlow + QNG Geodesic Weight
    sympflow_geodesic_weight: tuple[float, float] = (0.005, 0.8)  # Wide range

    # =========================================================================
    # PHASE V2.0-P1: PATH SYSTEM INDEPENDENCE
    # All three path systems remain INDEPENDENT per Section 11.6:
    # - qhd_num_paths: QHD superposition paths in FFT-domain SSM
    # - superposition_dim: TT-FFN parallel paths for MoE expert diversity
    # - coconut_num_paths: BFS thought exploration (gradient-connected)
    #
    # v2.0 adds HPO correlation hints via PATH_CORRELATION_GROUPS in config.py
    # to help faNOVA sample related params together without merging them.
    # See HIGHNOON_V2_PERFORMANCE_ANALYSIS.md Section 11.5-11.6.
    # =========================================================================

    # =========================================================================
    # PHASE 600+: QHD SPATIAL BLOCK (Replaces QMAMBA/HDSpatialBlock)
    # QHDSpatialBlock combines FFT-domain SSM with quantum superposition.
    # K superposition paths provide quantum expressiveness at K× memory cost.
    # =========================================================================
    qhd_num_paths: list[int] = field(
        default_factory=lambda: [1, 2]  # QAHPO tunable: 1-2 for Lite (K× memory)
    )
    qhd_entanglement_depth: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4]  # VQC entanglement layers
    )
    qhd_entanglement_strength: tuple[float, float] = (0.1, 0.6)  # CNOT mixing strength
    qhd_gumbel_temperature: tuple[float, float] = (0.5, 2.0)  # Born rule collapse temperature

    # =========================================================================
    # PHASE 800: QHD HIERARCHICAL BLOCK (Multi-Scale Reasoning)
    # Extends QHDSpatialBlock with inline hierarchical memory using:
    # - CTQW quantum walk aggregation
    # - Adaptive semantic chunking
    # - Cross-level attention injection
    # Complexity: O(K × L × D log D) - same as QHDSpatialBlock
    # =========================================================================
    hd_hierarchical_levels: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4]  # Hierarchy levels (2-3 recommended)
    )
    hd_hierarchical_pooling_ratio: list[int] = field(
        default_factory=lambda: [2, 3, 4, 6, 8]  # Compression ratio per level
    )
    hd_hierarchical_use_ctqw: list[bool] = field(
        default_factory=lambda: [True, False]  # CTQW quantum walk aggregation
    )
    hd_hierarchical_use_cross_attention: list[bool] = field(
        default_factory=lambda: [True, False]  # Cross-level attention injection
    )
    hd_hierarchical_ctqw_time: tuple[float, float] = (0.5, 2.0)  # Quantum walk time
    hd_hierarchical_ema_base_rate: tuple[float, float] = (0.05, 0.2)  # Base EMA rate for level 0
    # Phase 900.2: Quantum Feature Map Cross-Level Attention (replaces EMA blending)
    hd_cross_attn_qfm_depth: list[int] = field(
        default_factory=lambda: [2, 3, 4, 5, 6, 8]  # VQC rotation layers for QFM attention
    )
    hd_use_quantum_cross_attention: list[bool] = field(
        default_factory=lambda: [True, False]  # Enable quantum feature map attention
    )

    # =========================================================================
    # PHASE 850-880: UQHA (Unified Quantum-Hierarchical Architecture) v3.0
    # Replaces cross-level attention with O(K²) quantum walk entanglement.
    # Memory savings: ~576 MB → ~128 KB per hierarchical block.
    # =========================================================================
    # Phase 850: Frequency Stratification (implicit hierarchical encoding)
    freq_stratification_overlap: tuple[float, float] = (0.1, 0.5)  # Band overlap
    freq_stratification_mode: list[str] = field(
        default_factory=lambda: ["exponential", "linear", "learned"]
    )
    # Phase 860: Quantum Walk Entanglement (O(K²) cross-scale mixing)
    qhd_entanglement_topology: list[str] = field(
        default_factory=lambda: ["walk", "hierarchical", "adjacent"]
    )
    qhd_walk_evolution_time: tuple[float, float] = (0.5, 2.0)  # Unitary evolution time
    # Phase 880: Quantum Bus Hierarchical Injection
    bus_injection_strength: tuple[float, float] = (0.05, 0.3)  # Injection strength
    bus_injection_decay: tuple[float, float] = (0.3, 0.7)  # Hierarchical decay

    # =========================================================================
    # PHASE 700: FLOQUET POSITION ENCODING FOR CONTEXT EXTRAPOLATION
    # Enables infinite context extrapolation: train on 32K -> infer on 5M+.
    # The base frequency controls the geometric frequency progression.
    # =========================================================================
    hd_floquet_base_freq: tuple[float, float] = (1000.0, 100000.0)  # Log-uniform sampling

    # =========================================================================
    # PHASE 900: STREAMING HIERARCHICAL MEMORY (O(1) Sequence-Length Memory)
    # From MEMORY_ARCHITECTURE.md - enables O(1) memory via fixed-slot banks
    # =========================================================================
    # Fixed-slot memory bank configuration
    hierarchical_memory_slots: list[int] = field(
        default_factory=lambda: [32, 48, 64, 96, 128, 192, 256]  # M slots per level
    )
    # Uncertainty-gated pooling configuration
    uncertainty_pool_threshold: tuple[float, float] = (0.2, 0.9)  # Kalman trace threshold
    streaming_max_chunk_size: list[int] = field(
        default_factory=lambda: [16, 24, 32, 48, 64, 96, 128]  # Max tokens per chunk
    )
    # Online CTQW spectral approximation (O(k²) instead of O(M³))
    online_ctqw_rank: list[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 24, 32]  # Top-k eigenvectors
    )
    online_ctqw_ema_decay: tuple[float, float] = (0.9, 0.999)  # Spectral EMA decay
    # CTQW RFF approximation dimension (higher = more accurate, more memory)
    ctqw_rff_dim: list[int] = field(
        default_factory=lambda: [32, 48, 64, 80, 96, 128]  # QAHPO tunable: 32-128
    )
    # HD position encoding (Kanerva permutation)
    hd_position_decay_rate: tuple[float, float] = (0.999, 0.99999)  # Long-range decay
    # Phase 900.3: Batch cross-attention memory guard (RAM-dependent)
    # Higher values allow more batch processing, lower values force streaming earlier
    auto_streaming_batch_threshold: list[int] = field(
        default_factory=lambda: [2048, 4096, 8192, 12288, 16384, 24576, 32768]
    )

    # =========================================================================
    # QASA ATTENTION PARAMETERS
    # Phase 201.7: Expanded VQC layers and entanglement - WIDE RANGES
    # =========================================================================
    qasa_vqc_layers: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10])  # Wide
    qasa_entanglement_strength: tuple[float, float] = (0.05, 0.99)  # Wide range
    qasa_feature_rotation_depth: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6]  # Feature map rotation layers
    )

    # =========================================================================
    # QMOE ROUTING PARAMETERS
    # Phase 201.7: Expanded routing parameters - WIDE RANGES
    # =========================================================================
    qmoe_vqc_layers: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8])  # Wide
    qmoe_measurement_temp: tuple[float, float] = (0.05, 5.0)  # Wide range

    # =========================================================================
    # PHASE 69: Q-SSM QUANTUM STATE SPACE GATING
    # VQC gating for Mamba SSM for quantum-enhanced state updates
    # =========================================================================
    q_ssm_vqc_layers: list[int] = field(default_factory=lambda: [1, 2, 3, 4])  # VQC circuit layers
    q_ssm_num_qubits: list[int] = field(
        default_factory=lambda: [2, 3, 4, 5, 6, 8]
    )  # Virtual qubits

    # =========================================================================
    # QSG (QUANTUM SUPERPOSITION GENERATION) PARAMETERS
    # WIDE RANGES for frontier exploration
    # =========================================================================
    qsg_bond_dim: list[int] = field(
        default_factory=lambda: [4, 8, 16, 24, 32, 48, 64, 96, 128]  # MPS entanglement
    )
    qsg_coherence_range: list[int] = field(
        default_factory=lambda: [8, 16, 32, 64, 128, 256, 512, -1]  # -1 = infinite
    )
    qsg_grover_iterations: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 8]  # Amplitude amplification
    )
    qsg_jacobi_iterations: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6]  # Consistency refinement
    )
    qsg_hopfield_beta: tuple[float, float] = (0.1, 5.0)  # Wide energy landscape
    qsg_amplification_strength: tuple[float, float] = (0.5, 4.0)  # Wide Grover amp

    # =========================================================================
    # QSG ENTERPRISE OPTIMIZATION (Phases 1-3)
    # See QSG_ENTERPRISE_OPTIMIZATION_ROADMAP.md for full specification.
    # These affect training via factored embeddings and ensemble refinement.
    # =========================================================================
    # Phase 1.1: Top-K Candidate Pruning (PQ-based approximate search)
    qsg_candidate_k: list[int] = field(
        default_factory=lambda: [512, 768, 1024, 1536, 2048]  # Top-K candidates per position
    )
    qsg_pq_num_subvectors: list[int] = field(
        default_factory=lambda: [4, 6, 8, 12, 16]  # PQ subvector count M (speed vs accuracy)
    )
    # Phase 1.2: Vocabulary Factorization (E ≈ U × V^T)
    qsg_factorization_rank: list[int] = field(
        default_factory=lambda: [
            16,
            24,
            32,
            48,
            64,
        ]  # Factorization rank r (higher = more capacity)
    )
    # Phase 3.1: Ensemble Parallel Refinement
    qsg_ensemble_num_paths: list[int] = field(
        default_factory=lambda: [2, 3, 4, 6, 8]  # Parallel ensemble paths P (quality vs compute)
    )
    qsg_ensemble_coherence_weight: tuple[float, float] = (
        0.5,
        2.0,
    )  # Cross-path consistency weighting
    qsg_ensemble_temp_range_min: tuple[float, float] = (0.5, 0.9)  # Path temperature min
    qsg_ensemble_temp_range_max: tuple[float, float] = (1.1, 1.5)  # Path temperature max
    # Phase 3.2: Self-Consistency Filtering
    qsg_consistency_window: list[int] = field(
        default_factory=lambda: [2, 3, 4, 5]  # Cross-position consistency window
    )
    qsg_consistency_softening: tuple[float, float] = (0.25, 1.0)  # Softening for low-consistency
    # Phase 3.2: Grover Quality Boost (amplitude amplification on ensemble output)
    qsg_grover_boost_iterations: list[int] = field(
        default_factory=lambda: [1, 2, 3]  # Grover diffusion iterations on ensemble
    )

    # =========================================================================
    # QHPM (QUANTUM HOLOGRAPHIC PERSISTENT MEMORY) PARAMETERS
    # WIDE RANGES for frontier exploration
    # =========================================================================
    qhpm_holographic_dim: list[int] = field(
        default_factory=lambda: [128, 256, 384, 512, 768, 1024, 1536, 2048]  # Memory dimension
    )
    qhpm_mps_bond_dim: list[int] = field(
        default_factory=lambda: [4, 8, 16, 24, 32, 48, 64, 96]  # Quantum memory bond
    )
    qhpm_hopfield_beta: tuple[float, float] = (0.1, 5.0)  # Retrieval sharpness
    qhpm_max_crystallized_directions: list[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 24, 32, 48, 64]  # Max crystallized
    )

    # =========================================================================
    # PHASE 200+: HD STREAMING CORPUS PARAMETERS (DEPRECATED)
    # =========================================================================
    # DEPRECATED (Phase 500+): HD streaming is disabled in favor of DualPathEmbedding.
    # These parameters are kept for backward compatibility but no longer affect training.
    # HD enhancements are now integrated directly into the embedding layer via CTQW spreading.
    hd_reservoir_size: list[int] = field(
        default_factory=lambda: [2000]  # Single value - parameter is deprecated
    )
    hd_dim: list[int] = field(
        # Phase 1010: Deprecated - use per-layer dims (hd_dim_embedding, etc.) instead
        # Kept for backward compatibility. Max capped at 4096 to prevent CPU saturation
        # (12288 causes ~5.5B FFT ops at seq_len=32K). Use per-layer dims for larger values.
        default_factory=lambda: [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
    )
    # =========================================================================
    # PHASE 1010: PER-LAYER HD DIMENSION SEARCH SPACES
    # =========================================================================
    # Per-layer HD dimensions for independent QAHPO optimization.
    # Enable layer-type-specific memory/quality trade-offs.
    # Minimum is 64 (HD_DIM_MIN in config.py), maximum is 12288.
    # Lower values are listed first to encourage initial exploration
    # at smaller dimensions where HD scaling differs from transformers.
    hd_dim_embedding: list[int] = field(
        default_factory=lambda: [
            64,
            128,
            256,
            512,
            768,
            1024,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
            10240,
            12288,
        ]
    )
    hd_dim_spatial: list[int] = field(
        default_factory=lambda: [
            64,
            128,
            256,
            512,
            768,
            1024,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
            10240,
            12288,
        ]
    )
    hd_dim_timecrystal: list[int] = field(
        default_factory=lambda: [
            64,
            128,
            256,
            512,
            768,
            1024,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
            10240,
            12288,
        ]
    )
    hd_dim_moe: list[int] = field(
        default_factory=lambda: [
            64,
            128,
            256,
            512,
            768,
            1024,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
            10240,
            12288,
        ]
    )
    # Vocab sample size for tokenizer learning (lower = less memory)
    vocab_sample_size: list[int] = field(
        default_factory=lambda: [500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000]
    )
    # HD sample length: max tokens per individual HD sample
    # DEPRECATED (Phase 500+): HD streaming is disabled in favor of DualPathEmbedding.
    # This parameter is kept for backward compatibility but no longer affects training.
    # HD enhancements are now integrated directly into the embedding layer via CTQW spreading.
    hd_sample_length: list[int] = field(
        default_factory=lambda: [512]  # Single value - parameter is deprecated
    )

    # =========================================================================
    # PHASE 300+: HD UPGRADE PARAMETERS
    # =========================================================================
    # Phase 1: Optimizer Compression
    hd_optimizer_compression_ratio: list[int] = field(
        default_factory=lambda: [2, 4, 6, 8, 12, 16, 24, 32]  # Wide range
    )

    # Phase 2: Gradient Projection (replacing GaLore when HD enabled)
    hd_gradient_rank: list[int] = field(
        default_factory=lambda: [16, 32, 64, 96, 128, 192, 256, 384, 512]  # Wide range
    )
    hd_gradient_bandwidth: list[int] = field(
        default_factory=lambda: [32, 64, 128, 192, 256, 384, 512, 768, 1024]  # FFT bandwidth
    )

    # Phase 3: KV Cache Compression
    hd_kv_compression_ratio: list[int] = field(
        default_factory=lambda: [2, 4, 6, 8, 12, 16, 24, 32]  # Wide range
    )

    # Phase 4: HD Control & Stagnation Detection
    hd_control_fingerprint_dim: list[int] = field(
        default_factory=lambda: [64, 128, 192, 256, 384, 512]  # Fingerprint dimension
    )
    hd_stagnation_threshold: tuple[float, float] = (0.80, 0.99)  # Wide range for tuning
    hd_kalman_state_compression: list[int] = field(
        default_factory=lambda: [2, 4, 6, 8, 12, 16]  # Kalman state compression
    )

    # Phase 5: HD Projection Training
    hd_projection_freeze_epochs: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 8]  # Warmup epochs
    )
    hqe_ctqw_steps: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10]  # CTQW walk steps
    )
    hd_active_vocab_size: list[int] = field(
        default_factory=lambda: [
            1000,
            2000,
            5000,
            10000,
            20000,
            50000,
            100000,
            150000,
            200000,  # Frontier models may use large active vocabs
        ]
    )

    # =========================================================================
    # QWT (QUANTUM WAVELET TOKENIZER) PARAMETERS
    # See config.py QWT_* and PHASE 17 QWT TOKENIZER ENHANCEMENTS
    # =========================================================================
    # Core QWT parameters
    qwt_embedding_dim: list[int] = field(
        default_factory=lambda: [128, 256, 384, 512, 768, 1024]  # QWT embedding dimension
    )
    qwt_dwt_filter_size: list[int] = field(
        default_factory=lambda: [2, 3, 4, 5, 6]  # DWT filter size
    )
    qwt_vqc_qubits: list[int] = field(
        default_factory=lambda: [2, 3, 4, 5, 6, 8]  # VQC qubits for tokenizer
    )
    qwt_vqc_layers: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6]  # VQC layers for tokenizer
    )
    qwt_ctqw_steps: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5]  # CTQW steps (0 = disabled)
    )
    # Phase 17 QWT Enhancements
    qwt_pade_order: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4]  # Padé order (1=Cayley, higher = more accurate)
    )
    qwt_skip_stride: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4]  # Skip connection stride (0 = disabled)
    )
    qwt_max_skips_per_node: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4]  # Maximum skip connections per node
    )
    # QWT Text Tokenizer Parameters (QWTTextTokenizer)
    qwt_byte_offset: list[int] = field(
        default_factory=lambda: [16, 24, 32, 48, 64]  # Byte offset in vocabulary
    )
    qwt_max_superposition: list[int] = field(
        default_factory=lambda: [
            1,
            2,
        ]  # Max alt tokenizations - low since MoE provides path diversity
    )
    qwt_superposition_amplitude_threshold: tuple[float, float] = (
        0.05,
        0.3,
    )  # Min amplitude threshold
    # Thinking token complexity (affects injection rate)
    qwt_thinking_token_complexity_scale: tuple[float, float] = (0.5, 2.0)  # Complexity scaling
    # Multi-Scale DWT
    num_wavelet_levels: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5]  # Cascaded DWT levels
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
    # PHASE T1-T6: QUANTUM TRAINING LOOP PARAMETERS - WIDE RANGES
    # =========================================================================
    # Barren plateau gradient norm threshold for detection
    barren_plateau_threshold: tuple[float, float] = (1e-10, 1e-3)  # Ultra-wide for tuning
    # LR scaling factor during barren plateau recovery
    barren_plateau_recovery_lr_scale: tuple[float, float] = (1.5, 20.0)  # Wide range
    # Barren plateau recovery window (steps)
    barren_plateau_recovery_window: list[int] = field(
        default_factory=lambda: [25, 50, 75, 100, 150, 200, 300, 500]  # Wide range
    )
    # Barren plateau hysteresis (exit threshold factor)
    barren_plateau_hysteresis: tuple[float, float] = (1.5, 20.0)  # Wide range
    # Gradient dampening threshold
    gradient_dampening_threshold: tuple[float, float] = (10.0, 1000.0)  # Wide range
    # Minimum gradient dampening floor
    min_gradient_dampening: tuple[float, float] = (1e-5, 0.1)  # Wide range
    # QNG damping for QFIM regularization
    qng_damping: tuple[float, float] = (1e-6, 1e-2)  # Wide range
    # GaLore gradient projection rank
    galore_rank: list[int] = field(
        default_factory=lambda: [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]  # Wide
    )
    # Steps between GaLore projection updates
    galore_update_proj_gap: list[int] = field(
        default_factory=lambda: [50, 100, 150, 200, 300, 500, 750, 1000]  # Wide
    )
    # GaLore gradient scaling factor
    galore_scale: tuple[float, float] = (0.05, 0.8)  # Wide range

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

    # =========================================================================
    # PHASE 600: ENHANCED BUDGET PROGRESSION
    # =========================================================================
    # Minimum budget fraction for trial 0 (prevents undersized configs)
    min_budget_fraction: float = 0.25  # Was effectively 0.10, now 25% minimum
    # Maximum budget fraction cap
    max_budget_fraction: float = 0.95
    # Progression curve type: "linear", "sqrt", "log", "cosine"
    # sqrt: Faster initial ramp, good for exploration
    # log: Even faster initial ramp
    # cosine: Smooth S-curve
    budget_progression_curve: str = "sqrt"
    # Population initialization budget spread (min_frac, max_frac)
    # Ensures initial population has diverse config sizes
    initial_population_budget_spread: tuple[float, float] = (0.25, 0.70)

    # Memory-aware shrinking: count of consecutive memory failures
    # When > 0, sample() reduces architecture sizes to prevent OOM
    memory_failure_count: int = 0

    # Smart tuner: track skipped trials for WebUI and adaptive sampling

    _budget_tracker: OversizedConfigTracker | None = None

    # Phase 500: faNOVA importance scores for adaptive progression
    # Updated by QuantumAdaptiveHPOScheduler during search
    _importance_scores: dict[str, float] = field(default_factory=dict)

    # Architecture params that benefit from budget acceleration
    # Phase 600+: Updated for QHDSpatial/QHDHierarchical scaling
    # Ordered by compute impact (most impactful first)
    _architecture_params: list[str] = field(
        default_factory=lambda: [
            # === Primary Compute Drivers (QHDSpatial) ===
            "qhd_num_paths",  # K× linear multiplier (most critical!)
            "num_reasoning_blocks",  # Depth multiplier
            # === Phase 1010: Per-Layer HD Dimensions ===
            "hd_dim_embedding",  # Vocabulary encoding capacity
            "hd_dim_spatial",  # FFT SSM dimension (D log D)
            "hd_dim_timecrystal",  # Floquet modes × D
            "hd_dim_moe",  # Routing similarity (smallest impact)
            "hd_dim",  # DEPRECATED: Global fallback
            # === Secondary (linear scaling) ===
            "hd_hierarchical_levels",  # Hierarchical memory depth
            "hidden_dim",  # Linear projections
            "embedding_dim",  # Embedding table
            "mamba_state_dim",  # SSM state dimension
            # === Tertiary (sparse/shared activation) ===
            "num_moe_experts",  # Sparse MoE (top-k active)
            "superposition_dim",  # MoE superposition (sparse)
            "latent_kv_dim",  # Compressed KV cache
            # === QHD-Specific (moderate impact) ===
            "qhd_entanglement_depth",  # VQC layers
        ]
    )

    def update_importance_scores(self, scores: dict[str, float]) -> None:
        """Update faNOVA importance scores for adaptive progression.

        Called by QuantumAdaptiveHPOScheduler when faNOVA refits.

        Args:
            scores: Dict mapping param names to importance (0-1).
        """
        self._importance_scores = scores.copy()
        arch_importance = self._compute_architecture_importance()
        logger.info(
            f"[HPO Manager] Updated importance scores. "
            f"Architecture importance: {arch_importance:.2f}"
        )

    def _compute_architecture_importance(self) -> float:
        """Compute total importance of architecture parameters.

        Returns:
            Sum of importance for params that affect model architecture.
        """
        if not self._importance_scores:
            return 0.0
        return sum(self._importance_scores.get(p, 0.0) for p in self._architecture_params)

    def record_memory_failure(self) -> None:
        """Record a memory-related trial failure (OOM, swap spike, etc.).

        Increments memory_failure_count which causes subsequent samples to use
        smaller architectures to prevent repeated failures.
        """
        self.memory_failure_count += 1
        logger.warning(
            f"[HPO Manager] Memory failure recorded. Shrink level now: {self.memory_failure_count}"
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
        """Compute budget progression factor based on trial index and faNOVA importance.

        Phase 600 Enhancement: Configurable progression curves for better exploration.
        - sqrt: Faster initial ramp (recommended for exploration)
        - log: Even faster initial ramp for aggressive exploration
        - cosine: Smooth S-curve for gradual transition
        - linear: Original behavior, slowest ramp

        When faNOVA indicates architecture params are important (>30% combined
        importance), accelerates budget progression to explore larger architectures.

        Args:
            trial_id: Trial index (0-based)

        Returns:
            Progression factor (min_budget_fraction to max_budget_fraction).
        """
        import math

        min_frac = self.min_budget_fraction
        max_frac = self.max_budget_fraction
        curve = self.budget_progression_curve

        # Estimate trials for schedule (adaptive based on observed progress)
        total_trials_estimate = max(50, trial_id + 20)
        progress = min(1.0, trial_id / total_trials_estimate)

        # Apply progression curve
        if curve == "sqrt":
            # Faster initial ramp, slower later (good for exploration)
            raw_factor = math.sqrt(progress)
        elif curve == "log":
            # Even faster initial ramp
            raw_factor = math.log1p(progress * (math.e - 1)) / math.log(math.e)
        elif curve == "cosine":
            # Smooth S-curve (slow start, fast middle, slow end)
            raw_factor = 0.5 * (1 - math.cos(math.pi * progress))
        else:  # linear
            raw_factor = progress

        # Scale to [min_frac, max_frac]
        base_factor = min_frac + (max_frac - min_frac) * raw_factor

        # Phase 600+: faNOVA-adaptive acceleration (tuned for QHDSpatial)
        # QHD has fewer high-importance params, so lower trigger threshold
        arch_importance = self._compute_architecture_importance()
        if arch_importance > 0.2:  # Was 0.3 (lower for QHD's concentrated importance)
            # Architecture params are important - accelerate progression!
            # Boost factor by up to 60% when arch importance is high
            acceleration = 1.0 + (0.6 * min(1.0, arch_importance / 0.4))
            accelerated_factor = base_factor * acceleration

            logger.debug(
                f"[HPO Manager] faNOVA-adaptive boost: base={base_factor:.2f}, "
                f"arch_importance={arch_importance:.2f}, "
                f"accelerated={accelerated_factor:.2f}"
            )

            # Cap at max_frac
            return min(max_frac, accelerated_factor)

        return base_factor

    def _sample_progressive_dim(
        self, trial_id: int, dim_list: list[int], default: int = 256
    ) -> int:
        """Sample dimension with progressive filtering for early trials.

        Phase 1020: Applies to ALL dimension parameters (embedding_dim, hidden_dim,
        hd_dim_*, etc.) to ensure early trials explore smaller dimensions first.
        HD and standard dimensions scale differently from transformers - smaller
        dimensions are often more efficient and should be explored first.

        Progressive schedule:
        - Trial 0-9: Sample from lower 50% of range (establishes baseline)
        - Trial 10-29: Sample from lower 75% of range (expands exploration)
        - Trial 30+: Sample from full range (final optimization)

        Args:
            trial_id: Trial index (0-based)
            dim_list: Full list of dimension options (e.g., embedding_dim, hd_dim)
            default: Default value if dim_list is empty

        Returns:
            Sampled dimension from the appropriate progressive range.
        """
        import random

        if not dim_list:
            return default

        sorted_dims = sorted(dim_list)  # Ensure sorted (low to high)

        # Progressive filtering based on trial index
        if trial_id < 10:
            # Early trials: explore lower 50% of range
            cutoff_idx = max(1, len(sorted_dims) // 2)
            filtered = sorted_dims[:cutoff_idx]
        elif trial_id < 30:
            # Middle trials: expand to lower 75% of range
            cutoff_idx = max(1, (len(sorted_dims) * 3) // 4)
            filtered = sorted_dims[:cutoff_idx]
        else:
            # Later trials: full range
            filtered = sorted_dims

        # Also respect memory_failure_count if set
        memory_shrink = getattr(self, "memory_failure_count", 0)
        if memory_shrink >= 2:
            # Severe memory pressure: use lower 25% only
            cutoff_idx = max(1, len(filtered) // 4)
            filtered = filtered[:cutoff_idx]
        elif memory_shrink >= 1:
            # Moderate memory pressure: use lower 50%
            cutoff_idx = max(1, len(filtered) // 2)
            filtered = filtered[:cutoff_idx]

        return random.choice(filtered)

    def _estimate_max_sizes_for_budget(
        self,
        target_budget: int,
    ) -> tuple[int, int, int]:
        """Estimate max architecture sizes that fit within target param budget.

        Phase 950 Update: Optimized with early exit for O(D×B×E) -> O(D×B) typical case.
        Uses inverse of estimate_model_params() to find compatible architectures.
        Iterates through dimension/block/expert combinations to find the largest
        config that fits.

        Args:
            target_budget: Target parameter count limit

        Returns:
            Tuple of (max_blocks, max_experts, max_dim) that fit within budget.
        """
        # CRITICAL: self.target_vocab_size is a list[int] in HPOSearchSpace
        # Use conservative (min) value for budget estimation
        vocab_size = (
            min(self.target_vocab_size)
            if isinstance(self.target_vocab_size, list) and self.target_vocab_size
            else (self.target_vocab_size if isinstance(self.target_vocab_size, int) else 32000)
        )

        # Start with minimum viable config
        best_blocks = 2
        best_experts = 2
        best_dim = 64
        best_params = 0  # Track the largest fitting config's param count

        # Use representative hd_dim value (computed once, not per-iteration)
        representative_hd_dim = (
            min(self.hd_dim)
            if self.hd_dim and isinstance(self.hd_dim, list)
            else (self.hd_dim if isinstance(self.hd_dim, int) else 64 * 8)
        )

        # Try dimensions from LARGEST to SMALLEST (Phase 950: enable early exit)
        # Since we want the LARGEST fitting config, start big and exit early
        sorted_dims = sorted(self.embedding_dim, reverse=True)
        sorted_blocks = sorted(self.num_reasoning_blocks, reverse=True)
        sorted_experts = sorted(self.num_moe_experts, reverse=True)

        for dim in sorted_dims:
            # Quick check: if smallest config at this dim exceeds budget, skip dim entirely
            min_check_config = {
                "vocab_size": vocab_size,
                "embedding_dim": dim,
                "hidden_dim": dim,
                "num_reasoning_blocks": min(self.num_reasoning_blocks),
                "num_moe_experts": min(self.num_moe_experts),
                "hd_dim": dim * 8,  # Scale hd_dim with dim for quick check
            }
            if estimate_model_params(min_check_config) > target_budget:
                # Even smallest config at this dim is too big; smaller dims may fit
                continue

            for blocks in sorted_blocks:
                # Phase 950: Track if ANY expert count worked at this (dim, blocks) combo
                found_fit_at_this_level = False

                for experts in sorted_experts:
                    test_config = {
                        "vocab_size": vocab_size,
                        "embedding_dim": dim,
                        "hidden_dim": dim,
                        "num_reasoning_blocks": blocks,
                        "num_moe_experts": experts,
                        "hd_dim": representative_hd_dim,
                    }
                    estimated = estimate_model_params(test_config)

                    if estimated <= target_budget:
                        # This config fits! Check if it's larger than current best
                        if estimated > best_params:
                            best_dim = dim
                            best_blocks = blocks
                            best_experts = experts
                            best_params = estimated
                        found_fit_at_this_level = True
                        # Since experts are sorted descending, first fit is largest
                        break

                # Phase 950: Early exit - if we found a fit at max dim/blocks, we're done
                # (since sorting is descending, first overall fit is optimal)
                if found_fit_at_this_level and best_dim == dim:
                    # We found the largest fitting config at this dimension
                    # But continue checking other block counts at this dim
                    pass

            # If we found any fit at the largest dimension, no need to check smaller dims
            if best_dim == dim and best_params > 0:
                break

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
        # If target_vocab_size is not in search_space, use default from model_config
        if not hasattr(instance, "target_vocab_size") or instance.target_vocab_size is None:
            instance.target_vocab_size = model_config.get("target_vocab_size", [32000])

        # Phase 200: Unified to context_window (minimum 8K)
        # Phase 900.1: Default reduced to 4K for HPO exploration - 128K was causing 25GB+ memory
        # Full context training should explicitly set context_window in sweep config
        instance.context_window = model_config.get("context_window", 4096)
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

        # =====================================================================
        # PHASE 500+: BUDGET-AWARE HD_DIM AND VOCAB FILTERING
        # With HQE: embed_params = vocab_size × hd_dim + hd_dim × embedding_dim
        # These dominate parameter count and MUST be budget-constrained.
        # =====================================================================
        if self.param_budget is not None and self.hd_dim:
            # Compute max affordable hd_dim for this budget
            # Reserve 50% for model layers, 50% for embeddings (conservative)
            embedding_budget = int(target_budget * 0.5)
            selected_dim = max(filtered_dims)  # Use max filtered dim for estimation

            # Filter hd_dim: embed_params = vocab × hd + hd × dim
            # For budget V, max hd ≈ V / (vocab + dim) with V = embedding_budget
            # Use min vocab for max hd_dim estimation
            min_vocab = min(self.target_vocab_size) if self.target_vocab_size else 8000
            max_hd_for_budget = (
                embedding_budget // (min_vocab + selected_dim)
                if (min_vocab + selected_dim) > 0
                else 1024
            )

            filtered_hd_dim = [h for h in self.hd_dim if h <= max_hd_for_budget] or [
                min(self.hd_dim)
            ]

            # Now filter vocab based on selected max hd_dim
            max_filtered_hd = max(filtered_hd_dim)
            # max_vocab ≈ (embedding_budget - hd × dim) / hd
            max_vocab_for_budget = (
                (embedding_budget - max_filtered_hd * selected_dim) // max_filtered_hd
                if max_filtered_hd > 0
                else 32000
            )
            max_vocab_for_budget = max(
                8000, min(max_vocab_for_budget, 300000)
            )  # Clamp to reasonable range

            filtered_vocab = [v for v in self.target_vocab_size if v <= max_vocab_for_budget] or [
                min(self.target_vocab_size)
            ]

            logger.debug(
                f"[SmartTuner] HD budget filter: target_budget={target_budget / 1e6:.0f}M, "
                f"max_hd={max_hd_for_budget}, max_vocab={max_vocab_for_budget}, "
                f"filtered_hd_dim={filtered_hd_dim[:3]}..., filtered_vocab={filtered_vocab[:3]}..."
            )
        else:
            filtered_hd_dim = self.hd_dim if self.hd_dim else [512, 1024]
            filtered_vocab = self.target_vocab_size if self.target_vocab_size else [32000]

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

        # Phase 1 Tokenizer Fix: Sample target_vocab_size from FILTERED list
        # Model vocab_size is derived from tokenizer.vocab_size in trial runner
        sampled_target_vocab = random.choice(filtered_vocab)

        # Task 1.2.2: Explicit vocab size validation against parameter budget
        # This prevents vocab sizes that would blow the embedding parameter budget
        if self.param_budget is not None:
            # Get sampled embedding_dim for budget calculation
            sampled_embedding_dim = self._sample_progressive_dim(
                trial_id, filtered_dims, default=256
            )
            sampled_hd_dim = (
                random.choice(filtered_hd_dim) if filtered_hd_dim else (sampled_embedding_dim * 8)
            )

            max_vocab = compute_max_vocab_for_budget(
                param_budget=self.param_budget,
                embedding_dim=sampled_embedding_dim,
                use_hqe=True,  # Conservative: assume HQE for max params
                hd_dim=sampled_hd_dim,
            )

            if sampled_target_vocab > max_vocab:
                logger.info(
                    f"[QAHPO] Clamping target_vocab_size: {sampled_target_vocab} → {max_vocab} "
                    f"(budget={self.param_budget / 1e6:.0f}M params, embed_dim={sampled_embedding_dim})"
                )
                sampled_target_vocab = max_vocab
        else:
            # No budget constraint - use progressive sampling for dimensions
            sampled_embedding_dim = self._sample_progressive_dim(
                trial_id, filtered_dims, default=256
            )
            sampled_hd_dim = (
                random.choice(filtered_hd_dim) if filtered_hd_dim else (sampled_embedding_dim * 8)
            )

        config = {
            # Metadata for internal tracking
            "_trial_id": trial_id,
            "target_vocab_size": sampled_target_vocab,
            "param_budget": self.param_budget,
            # Training hyperparameters (LR auto-derived from optimizer)
            "learning_rate": sampled_lr,
            "batch_size": random.choice(filtered_batch),
            "optimizer": selected_optimizer,
            "warmup_steps": random.randint(*self.warmup_steps),
            "weight_decay": random.uniform(*self.weight_decay),
            # Model architecture params (memory-aware)
            "num_reasoning_blocks": random.choice(filtered_blocks),
            # Phase 1020: Progressive sampling for architecture dimensions
            # Early trials explore smaller dimensions first (trials 0-9: lower 50%)
            "hidden_dim": self._sample_progressive_dim(trial_id, filtered_dims, default=256),
            "embedding_dim": self._sample_progressive_dim(trial_id, filtered_dims, default=256),
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
            # PHASE 43: Neural Kalman Parameters
            # =================================================================
            "neural_kalman_hidden_dim": random.choice(self.neural_kalman_hidden_dim),
            "neural_kalman_process_noise": random.uniform(*self.neural_kalman_process_noise),
            "neural_kalman_measurement_noise": random.uniform(
                *self.neural_kalman_measurement_noise
            ),
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
            "q_ssm_num_qubits": random.choice(self.q_ssm_num_qubits),
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
            # Note: Global hd_dim removed from sampling (Phase 1010).
            # Per-layer dims (hd_dim_embedding, etc.) are now the primary mechanism.
            # The filtered_hd_dim budget logic above is retained for constraint validation.
            # =================================================================
            "hd_reservoir_size": random.choice(self.hd_reservoir_size),
            "vocab_sample_size": random.choice(self.vocab_sample_size),
            # =================================================================
            # PHASE 1010: Per-Layer HD Dimensions (independent tuning)
            # Early trials use lower HD dims to match HD scaling (differs from transformers)
            # Progressive filtering: trial 0-9 → lower half, trial 10-29 → lower 3/4, trial 30+ → full range
            # =================================================================
            "hd_dim_embedding": self._sample_progressive_dim(
                trial_id, self.hd_dim_embedding, default=512
            ),
            "hd_dim_spatial": self._sample_progressive_dim(
                trial_id, self.hd_dim_spatial, default=512
            ),
            "hd_dim_timecrystal": self._sample_progressive_dim(
                trial_id, self.hd_dim_timecrystal, default=64
            ),
            "hd_dim_moe": self._sample_progressive_dim(trial_id, self.hd_dim_moe, default=512),
            # HD sample length: controls max tokens per individual HD sample
            # This is separate from context_window and directly affects position keys memory
            "hd_sample_length": random.choice(self.hd_sample_length),
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
            "barren_plateau_recovery_window": random.choice(self.barren_plateau_recovery_window),
            "barren_plateau_hysteresis": random.uniform(*self.barren_plateau_hysteresis),
            "gradient_dampening_threshold": random.uniform(*self.gradient_dampening_threshold),
            "min_gradient_dampening": random.uniform(*self.min_gradient_dampening),
            "qng_damping": random.uniform(*self.qng_damping),
            "galore_rank": random.choice(self.galore_rank),
            "galore_update_proj_gap": random.choice(self.galore_update_proj_gap),
            "galore_scale": random.uniform(*self.galore_scale),
            # =================================================================
            # PHASE 2.0: NEW PARAMETER EXPANSION - Chunking Parameters
            # =================================================================
            "chunk_size": random.choice(self.chunk_size),
            "chunk_stride": random.choice(self.chunk_stride),
            # =================================================================
            # Attention Configuration Parameters
            # =================================================================
            "local_attention_window": random.choice(self.local_attention_window),
            "local_attention_window_min": random.choice(self.local_attention_window_min),
            "local_attention_window_max": random.choice(self.local_attention_window_max),
            "local_attention_sigmoid_temp": random.uniform(*self.local_attention_sigmoid_temp),
            "local_attention_sparsity_ratio": random.uniform(*self.local_attention_sparsity_ratio),
            # =================================================================
            # Flash Linear Attention Parameters
            # =================================================================
            "flash_linear_chunk_size": random.choice(self.flash_linear_chunk_size),
            "flash_linear_train_chunk_size": random.choice(self.flash_linear_train_chunk_size),
            "flash_linear_hybrid_window": random.choice(self.flash_linear_hybrid_window),
            "flash_linear_hybrid_alpha": random.uniform(*self.flash_linear_hybrid_alpha),
            "flash_linear_augment_rank": random.choice(self.flash_linear_augment_rank),
            "flash_linear_gate_init_bias": random.uniform(*self.flash_linear_gate_init_bias),
            # =================================================================
            # State Bus Parameters
            # =================================================================
            "state_bus_dim": random.choice(self.state_bus_dim),
            "state_bus_slots": random.choice(self.state_bus_slots),
            "state_bus_max_slots": random.choice(self.state_bus_max_slots),
            "state_bus_num_types": random.choice(self.state_bus_num_types),
            "state_bus_bond_dim": random.choice(self.state_bus_bond_dim),
            # =================================================================
            # External Memory (GEM) Parameters
            # =================================================================
            "memory_slots": random.choice(self.memory_slots),
            "memory_slot_dim": random.choice(self.memory_slot_dim),
            "memory_surprise_threshold": random.uniform(*self.memory_surprise_threshold),
            "memory_product_k": random.choice(self.memory_product_k),
            "memory_num_heads": random.choice(self.memory_num_heads),
            "memory_mps_bond_dim": random.choice(self.memory_mps_bond_dim),
            # =================================================================
            # CPU-Specific Optimization Parameters
            # =================================================================
            "cache_tile_size": random.choice(self.cache_tile_size),
            "prefetch_distance": random.choice(self.prefetch_distance),
            "ssd_chunk_size": random.choice(self.ssd_chunk_size),
            "superposition_micro_batch_size": random.choice(self.superposition_micro_batch_size),
            # =================================================================
            # VQC Extended Parameters
            # =================================================================
            "vqc_qubits": random.choice(self.vqc_qubits),
            "vqc_feature_rotation_depth": random.choice(self.vqc_feature_rotation_depth),
            "neumann_cayley_terms": random.choice(self.neumann_cayley_terms),
            "neumann_series_terms": random.choice(self.neumann_series_terms),
            # =================================================================
            # DTC Extended Parameters
            # =================================================================
            "dtc_coupling_j": random.uniform(*self.dtc_coupling_j),
            "dtc_disorder_w": random.uniform(*self.dtc_disorder_w),
            "dtc_num_cycles": random.choice(self.dtc_num_cycles),
            # =================================================================
            # QSG Parameters
            # =================================================================
            "qsg_bond_dim": random.choice(self.qsg_bond_dim),
            "qsg_coherence_range": random.choice(self.qsg_coherence_range),
            "qsg_grover_iterations": random.choice(self.qsg_grover_iterations),
            "qsg_jacobi_iterations": random.choice(self.qsg_jacobi_iterations),
            "qsg_hopfield_beta": random.uniform(*self.qsg_hopfield_beta),
            "qsg_amplification_strength": random.uniform(*self.qsg_amplification_strength),
            # =================================================================
            # QHPM Parameters
            # =================================================================
            "qhpm_holographic_dim": random.choice(self.qhpm_holographic_dim),
            "qhpm_mps_bond_dim": random.choice(self.qhpm_mps_bond_dim),
            "qhpm_hopfield_beta": random.uniform(*self.qhpm_hopfield_beta),
            "qhpm_max_crystallized_directions": random.choice(
                self.qhpm_max_crystallized_directions
            ),
            # =================================================================
            # HD Extended Parameters
            # =================================================================
            "hd_optimizer_compression_ratio": random.choice(self.hd_optimizer_compression_ratio),
            "hd_gradient_rank": random.choice(self.hd_gradient_rank),
            "hd_gradient_bandwidth": random.choice(self.hd_gradient_bandwidth),
            "hd_kv_compression_ratio": random.choice(self.hd_kv_compression_ratio),
            "hd_control_fingerprint_dim": random.choice(self.hd_control_fingerprint_dim),
            "hd_stagnation_threshold": random.uniform(*self.hd_stagnation_threshold),
            "hd_kalman_state_compression": random.choice(self.hd_kalman_state_compression),
            "hd_projection_freeze_epochs": random.choice(self.hd_projection_freeze_epochs),
            "hqe_ctqw_steps": random.choice(self.hqe_ctqw_steps),
            # Phase 500.1: Filter hd_active_vocab_size <= target_vocab_size at sample time
            # This ensures active vocab (dense embeddings) never exceeds total vocab
            "hd_active_vocab_size": random.choice(
                [v for v in self.hd_active_vocab_size if v <= sampled_target_vocab]
                or [min(10000, sampled_target_vocab)]  # Fallback for very small vocabs
            ),
            # =================================================================
            # Unified Bus Extended Parameters
            # =================================================================
            "unified_bus_entanglement_init": random.uniform(*self.unified_bus_entanglement_init),
            "unified_bus_propagation_rate": random.uniform(*self.unified_bus_propagation_rate),
            # =================================================================
            # QASA Extended Parameters
            # =================================================================
            "qasa_feature_rotation_depth": random.choice(self.qasa_feature_rotation_depth),
            # =================================================================
            # PHASE 600+: QHD SPATIAL BLOCK Parameters
            # =================================================================
            "qhd_num_paths": random.choice(self.qhd_num_paths),
            "qhd_entanglement_depth": random.choice(self.qhd_entanglement_depth),
            "qhd_entanglement_strength": random.uniform(*self.qhd_entanglement_strength),
            "qhd_gumbel_temperature": random.uniform(*self.qhd_gumbel_temperature),
            # =================================================================
            # PHASE 700: Floquet Position Encoding
            # =================================================================
            "hd_floquet_base_freq": 10
            ** random.uniform(
                math.log10(self.hd_floquet_base_freq[0]), math.log10(self.hd_floquet_base_freq[1])
            ),  # Log-uniform sampling
            # =================================================================
            # PHASE 800: QHD HIERARCHICAL BLOCK Parameters
            # =================================================================
            "hd_hierarchical_levels": random.choice(self.hd_hierarchical_levels),
            "hd_hierarchical_pooling_ratio": random.choice(self.hd_hierarchical_pooling_ratio),
            "hd_hierarchical_use_ctqw": random.choice(self.hd_hierarchical_use_ctqw),
            "hd_hierarchical_use_cross_attention": random.choice(
                self.hd_hierarchical_use_cross_attention
            ),
            "hd_hierarchical_ctqw_time": random.uniform(*self.hd_hierarchical_ctqw_time),
            "hd_hierarchical_ema_base_rate": random.uniform(*self.hd_hierarchical_ema_base_rate),
            # Phase 900.2: Quantum Feature Map Cross-Level Attention (replaces EMA)
            "hd_cross_attn_qfm_depth": random.choice(self.hd_cross_attn_qfm_depth),
            "hd_use_quantum_cross_attention": random.choice(self.hd_use_quantum_cross_attention),
            # =================================================================
            # PHASE 900: STREAMING HIERARCHICAL MEMORY Parameters
            # =================================================================
            "hierarchical_memory_slots": random.choice(self.hierarchical_memory_slots),
            "uncertainty_pool_threshold": random.uniform(*self.uncertainty_pool_threshold),
            "streaming_max_chunk_size": random.choice(self.streaming_max_chunk_size),
            "online_ctqw_rank": random.choice(self.online_ctqw_rank),
            "online_ctqw_ema_decay": random.uniform(*self.online_ctqw_ema_decay),
            "ctqw_rff_dim": random.choice(self.ctqw_rff_dim),
            "hd_position_decay_rate": random.uniform(*self.hd_position_decay_rate),
        }

        # Add user-set tokenizer/context config if provided
        # Note: target_vocab_size is sampled above; model vocab derived from tokenizer
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
                        # Phase 1.1 Fix: Force hd_dim recalculation when shrinking embedding
                        if "hd_dim" in config:
                            del config["hd_dim"]
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
                    if "hd_dim" in config:
                        del config["hd_dim"]

                # Use progressive target budget for estimation during resampling
                config_for_estimate = config.copy()
                config_for_estimate["param_budget"] = trial_target_budget
                estimated_params = estimate_model_params(config_for_estimate)

                if attempt % 20 == 0:
                    logger.debug(
                        f"[SmartTuner] Trial {trial_id} attempt {attempt}: current {estimated_params / 1e6:.1f}M"
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

        # =================================================================
        # PHASE 2.1: STRUCTURAL CONSTRAINT VALIDATION
        # Enforce constraints between interdependent parameters
        # =================================================================
        config = self._validate_and_fix_constraints(config)

        return config

    def _validate_and_fix_constraints(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate and fix structural constraints between sampled parameters.

        This method enforces:
        - Ordering constraints (stride ≤ size, min < max)
        - Divisibility constraints (dim % heads == 0)
        - Proportionality constraints (rank ≤ dim / N)
        - Memory budget constraints (position keys ≤ 50MB)

        Args:
            config: Independently sampled configuration dict.

        Returns:
            Config dict with constraints validated and fixed.
        """
        import random

        # ===================================================================
        # 1. ORDERING CONSTRAINTS
        # ===================================================================
        # chunk_stride ≤ chunk_size
        if "chunk_stride" in config and "chunk_size" in config:
            if config["chunk_stride"] > config["chunk_size"]:
                valid = [s for s in self.chunk_stride if s <= config["chunk_size"]]
                original = config["chunk_stride"]
                config["chunk_stride"] = random.choice(valid) if valid else config["chunk_size"]
                logger.info(
                    f"[Constraint] Fixed chunk_stride: {original} -> {config['chunk_stride']} "
                    f"(≤ {config['chunk_size']})"
                )

        # local_attention_window_min < local_attention_window_max
        if "local_attention_window_min" in config and "local_attention_window_max" in config:
            if config["local_attention_window_min"] >= config["local_attention_window_max"]:
                # Swap and ensure gap
                min_val = min(
                    config["local_attention_window_min"], config["local_attention_window_max"]
                )
                max_val = max(
                    config["local_attention_window_min"], config["local_attention_window_max"]
                )
                config["local_attention_window_min"] = min_val
                config["local_attention_window_max"] = (
                    max_val + 64 if min_val == max_val else max_val
                )
                logger.info(
                    f"[Constraint] Fixed window ordering: min={config['local_attention_window_min']}, "
                    f"max={config['local_attention_window_max']}"
                )

        # vqc_min_layers < vqc_max_layers
        if "vqc_min_layers" in config and "vqc_max_layers" in config:
            if config["vqc_min_layers"] >= config["vqc_max_layers"]:
                config["vqc_max_layers"] = config["vqc_min_layers"] + 1
                logger.info(
                    f"[Constraint] Fixed VQC layers: min={config['vqc_min_layers']}, "
                    f"max={config['vqc_max_layers']}"
                )

        # hopfield_beta_min < hopfield_beta_max
        if "hopfield_beta_min" in config and "hopfield_beta_max" in config:
            if config["hopfield_beta_min"] >= config["hopfield_beta_max"]:
                min_val = min(config["hopfield_beta_min"], config["hopfield_beta_max"])
                max_val = max(config["hopfield_beta_min"], config["hopfield_beta_max"])
                config["hopfield_beta_min"] = min_val * 0.5
                config["hopfield_beta_max"] = max_val * 1.5 if min_val == max_val else max_val
                logger.info(
                    f"[Constraint] Fixed hopfield_beta: min={config['hopfield_beta_min']:.2f}, "
                    f"max={config['hopfield_beta_max']:.2f}"
                )

        # qao_initial_temp > qao_final_temp (annealing goes from high to low)
        if "qao_initial_temp" in config and "qao_final_temp" in config:
            if config["qao_initial_temp"] <= config["qao_final_temp"]:
                # Swap
                config["qao_initial_temp"], config["qao_final_temp"] = (
                    config["qao_final_temp"] * 2.0,
                    config["qao_initial_temp"] * 0.1,
                )
                logger.info(
                    f"[Constraint] Fixed QAOA temps: initial={config['qao_initial_temp']:.2f}, "
                    f"final={config['qao_final_temp']:.2f}"
                )

        # state_bus_slots ≤ state_bus_max_slots
        if "state_bus_slots" in config and "state_bus_max_slots" in config:
            if config["state_bus_slots"] > config["state_bus_max_slots"]:
                config["state_bus_max_slots"] = config["state_bus_slots"]
                logger.info(
                    f"[Constraint] Fixed state_bus_max_slots to {config['state_bus_max_slots']}"
                )

        # ===================================================================
        # 2. DIVISIBILITY CONSTRAINTS
        # ===================================================================

        # memory_slot_dim % memory_num_heads == 0
        if "memory_slot_dim" in config and "memory_num_heads" in config:
            slot_dim = config["memory_slot_dim"]
            heads = config["memory_num_heads"]
            if slot_dim % heads != 0:
                valid = [h for h in self.memory_num_heads if slot_dim % h == 0]
                if valid:
                    config["memory_num_heads"] = random.choice(valid)
                else:
                    config["memory_num_heads"] = 4  # Safe fallback
                logger.info(
                    f"[Constraint] memory_num_heads adjusted to {config['memory_num_heads']}"
                )

        # ===================================================================
        # 3. PROPORTIONALITY CONSTRAINTS
        # ===================================================================

        # hd_gradient_bandwidth ≤ hd_dim / 2
        if "hd_gradient_bandwidth" in config and "hd_dim" in config:
            max_bw = config["hd_dim"] // 2
            if config["hd_gradient_bandwidth"] > max_bw:
                valid = [b for b in self.hd_gradient_bandwidth if b <= max_bw]
                config["hd_gradient_bandwidth"] = random.choice(valid) if valid else max_bw
                logger.info(
                    f"[Constraint] hd_gradient_bandwidth capped to "
                    f"{config['hd_gradient_bandwidth']} (max={max_bw})"
                )

        # hd_gradient_rank ≤ hd_dim / 4
        if "hd_gradient_rank" in config and "hd_dim" in config:
            max_rank = config["hd_dim"] // 4
            if config["hd_gradient_rank"] > max_rank:
                valid = [r for r in self.hd_gradient_rank if r <= max_rank]
                config["hd_gradient_rank"] = random.choice(valid) if valid else max_rank
                logger.info(f"[Constraint] hd_gradient_rank capped to {config['hd_gradient_rank']}")

        # flash_linear_augment_rank ≤ hidden_dim / 8
        if "flash_linear_augment_rank" in config and "hidden_dim" in config:
            max_rank = config["hidden_dim"] // 8
            if config["flash_linear_augment_rank"] > max_rank:
                valid = [r for r in self.flash_linear_augment_rank if r <= max_rank]
                config["flash_linear_augment_rank"] = random.choice(valid) if valid else max_rank
                logger.info(
                    f"[Constraint] flash_linear_augment_rank capped to "
                    f"{config['flash_linear_augment_rank']}"
                )

        # latent_kv_dim ≤ hidden_dim
        if "latent_kv_dim" in config and "hidden_dim" in config:
            if config["latent_kv_dim"] > config["hidden_dim"]:
                valid = [d for d in self.latent_kv_dim if d <= config["hidden_dim"]]
                config["latent_kv_dim"] = random.choice(valid) if valid else config["hidden_dim"]
                logger.info(f"[Constraint] latent_kv_dim capped to {config['latent_kv_dim']}")

        # ===================================================================
        # 4. MEMORY BUDGET CONSTRAINTS
        # ===================================================================

        # Position keys: hd_sample_length × hd_dim × 4 bytes ≤ 50MB
        if "hd_sample_length" in config and "hd_dim" in config:
            position_mb = (config["hd_sample_length"] * config["hd_dim"] * 4) / (1024 * 1024)
            MAX_POSITION_MB = 50
            if position_mb > MAX_POSITION_MB:
                max_length = int(MAX_POSITION_MB * 1024 * 1024 / (config["hd_dim"] * 4))
                valid = [l for l in self.hd_sample_length if l <= max_length]
                original = config["hd_sample_length"]
                config["hd_sample_length"] = random.choice(valid) if valid else max_length
                logger.info(
                    f"[Constraint] hd_sample_length {original} -> {config['hd_sample_length']} "
                    f"(memory: {position_mb:.1f}MB -> ~{MAX_POSITION_MB}MB)"
                )

        # GEM memory: memory_slots × memory_slot_dim × 4 ≤ 100MB
        if "memory_slots" in config and "memory_slot_dim" in config:
            gem_mb = (config["memory_slots"] * config["memory_slot_dim"] * 4) / (1024 * 1024)
            MAX_GEM_MB = 100
            if gem_mb > MAX_GEM_MB:
                max_slots = int(MAX_GEM_MB * 1024 * 1024 / (config["memory_slot_dim"] * 4))
                valid = [s for s in self.memory_slots if s <= max_slots]
                original = config["memory_slots"]
                config["memory_slots"] = random.choice(valid) if valid else max_slots
                logger.info(
                    f"[Constraint] memory_slots {original} -> {config['memory_slots']} "
                    f"(GEM memory: {gem_mb:.1f}MB -> ~{MAX_GEM_MB}MB)"
                )

        # ===================================================================
        # 5. VOCAB SIZE CONSTRAINTS (Phase 500.1)
        # ===================================================================

        # hd_active_vocab_size ≤ target_vocab_size
        # Active vocab is the set of tokens using dense embeddings; remaining tokens
        # use character-level HD encoding. Cannot exceed total vocabulary.
        if "hd_active_vocab_size" in config and "target_vocab_size" in config:
            if config["hd_active_vocab_size"] > config["target_vocab_size"]:
                max_active = config["target_vocab_size"]
                valid = [v for v in self.hd_active_vocab_size if v <= max_active]
                original = config["hd_active_vocab_size"]
                # Use largest valid size, or 10% of target if none available
                config["hd_active_vocab_size"] = (
                    max(valid) if valid else min(10000, int(max_active * 0.1))
                )
                logger.info(
                    f"[Constraint] hd_active_vocab_size {original} -> "
                    f"{config['hd_active_vocab_size']} (≤ target_vocab_size={max_active})"
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
