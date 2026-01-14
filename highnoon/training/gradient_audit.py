"""Gradient audit utilities for TrainingEngine diagnostics.

Provides a lightweight analyzer for gradient coverage across model variables,
with optional grouping and expected no-grad patterns for conditional layers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf


@dataclass
class GradientAuditConfig:
    """Configuration for gradient audit analysis.

    Phase 2026.1: Comprehensive expected no-grad patterns for enterprise deployment.

    Categories of intentionally non-trainable variables:
    1. Meta-Controller controlled variables (updated at step 5-10+)
    2. Kalman GRU weights (auxiliary, not gradient-trained)
    3. HD identity projections (when use_hd_projection=False)
    4. EMA trackers and running statistics
    5. Floquet/evolution time variables
    6. Special placeholder weights

    Note: Some variables have DELAYED gradient activation:
    - Kalman identification: gradients activate at step 850
    - Meta-Controller: activates at step 5-10
    - These are correctly shown as zero initially.
    """

    expected_no_grad_patterns: tuple[str, ...] = (
        # =====================================================================
        # CONTINUOUS THOUGHT / COCONUT GATES (Phase 14.3)
        # These use stop_gradient for straight-through estimators
        # =====================================================================
        "continuous_thought_gate",
        "blend_gate",
        "thought_exit_gate",
        # =====================================================================
        # HD EMBEDDING BASIS VECTORS (Phase 201)
        # Fixed random basis vectors, not learned
        # =====================================================================
        "hd_char_basis",
        "hd_position_basis",
        "hd_document_boundary",
        # =====================================================================
        # KALMAN FILTER AUXILIARY GRU WEIGHTS (Phase 43.2)
        # GRU is only for adaptive noise estimation, not gradient-trained
        # Main Kalman params (A, H, Q, R) ARE trainable
        # =====================================================================
        "gru_w_z",
        "gru_w_r",
        "gru_w_h",
        "gru_b_z",
        "gru_b_r",
        "gru_b_h",
        "gru_w_out",
        # =====================================================================
        # EVOLUTION TIME VARIABLES (Meta-Controller managed, Phase 2.1)
        # Updated by Meta-Controller at step 5-10, not by SGD
        # =====================================================================
        "evolution_time",
        "evolution_time_gain",
        "evolution_time_shift",
        "evolution_time_cap",
        "last_stable_evolution_time",
        # =====================================================================
        # MOE ROUTING STATISTICS (EMA-updated, not gradient-trained)
        # =====================================================================
        "load_vector",
        "routing_bias",
        "pattern_ema",
        "expert_load_ema",
        "routing_entropy_ema",
        # =====================================================================
        # HD IDENTITY PROJECTIONS (when use_hd_projection=False)
        # Placeholder identity matrices, not learned
        # =====================================================================
        "hd_input_proj_identity",
        "hd_output_proj_identity",
        # =====================================================================
        # FLOQUET / TIME CRYSTAL PHASE VARIABLES
        # Controlled by external dynamics, not SGD
        # =====================================================================
        "floquet_phase",
        "time_crystal_phase",
        # =====================================================================
        # REASONING STATE BUS VARIABLES
        # Internal coordination state, not gradient-trained
        # =====================================================================
        "bus_state",
        "coherence_accumulator",
        # =====================================================================
        # VQC ENTROPY/MEASUREMENT OBSERVABLES (non-trainable statistics)
        # =====================================================================
        "vqc_entropy_ema",
        "measurement_observable",
    )
    group_patterns: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            # =====================================================================
            # SPATIAL BLOCKS
            # =====================================================================
            "QHDSpatialBlock": (
                "qhd_spatial_block",
                "qhd_spatial",
                "quantum_walk",
                "hd_proj",
            ),
            "SpatialBlock": (
                "spatial_block",
                "mamba",
                "selective_scan",
            ),
            # =====================================================================
            # TIME CRYSTAL BLOCKS
            # =====================================================================
            "HDTimeCrystalBlock": (
                "hd_timecrystal_block",
                "timecrystal_block",
                "timecrystal",
                "hamiltonian",
                "hnn_w",
                "hnn_b",
            ),
            # =====================================================================
            # REASONING BLOCKS
            # =====================================================================
            "ContinuousThoughtBlock": (
                "continuous_thought",
                "coconut",
            ),
            "LatentReasoningBlock": (
                "latent_reasoning",
                "thought_proj",
                "confidence_proj",
            ),
            "SelfConsistencyBlock": (
                "self_consistency",
                "verification",
            ),
            # =====================================================================
            # ATTENTION BLOCKS
            # =====================================================================
            "WLAMBlock": (
                "wlam_block",
                "wlam",
                "wavelet",
                "scattering",
            ),
            "QuantumGQA": (
                "quantum_gqa",
                "gqa",
            ),
            "LatentKVAttention": (
                "latent_kv_attention",
                "latent_kv",
            ),
            # =====================================================================
            # MOE BLOCKS
            # =====================================================================
            "MoELayer": (
                "moe_layer",
                "moe",
                "router",
                "expert_embeddings",
            ),
            "HDSharedExpertBasis": (
                "shared_basis_up",
                "shared_basis_down",
                "expert_coeffs_up",
                "expert_coeffs_down",
                "expert_bias",
                "hd_shared",
            ),
            "SuperposedExpert": (
                "path_bases",
                "path_weights",
                "superposition_tt_core",
                "ffn1",
                "ffn2",
                "hd_input_proj",
                "hd_output_proj",
            ),
            # =====================================================================
            # QUANTUM ENHANCEMENTS
            # =====================================================================
            "PortHamiltonian": (
                "port_hamiltonian",
                "j_upper",
                "r_diag",
                "h_proj",
            ),
            "QSVT": (
                "qsvt",
                "qsvt_coefficients",
            ),
            "QuantumLMHead": (
                "quantum_lm_head",
                "lm_head_quantum",
                "vqc_params",
                "entangle_params",
                "fallback_weight",
            ),
            "TensorRingLayer": (
                "tensor_ring",
                "ring_core",
            ),
            # =====================================================================
            # KALMAN / STATE ESTIMATION
            # =====================================================================
            "KalmanBlock": (
                "kalman_block",
                "kalman",
                "neural_kalman",
                "state_transition",
                "observation_matrix",
                "process_noise",
                "observation_noise",
            ),
            # =====================================================================
            # EMBEDDINGS / OUTPUT
            # =====================================================================
            "DualPathEmbedding": (
                "dual_path_embedding",
                "token_embedding",
                "position_encoding",
            ),
            "TTLayer": (
                "tt_core",
                "tt_layer",
            ),
        }
    )
    zero_tol: float = 0.0
    sample_limit: int = 25
    include_per_variable_stats: bool = False
    per_variable_limit: int = 0


@dataclass
class GradientAuditReport:
    """Result summary for gradient audit analysis."""

    total_vars: int
    nonzero_vars: int
    none_vars: int
    zero_vars: int
    nonfinite_vars: int
    expected_missing_or_zero: int
    unexpected_missing_or_zero: int
    coverage_ratio: float
    group_stats: dict[str, dict[str, int | float]]
    expected_samples: list[str]
    unexpected_samples: list[str]
    variable_stats: list[dict[str, Any]]

    def summary_metrics(self, prefix: str = "grad_audit") -> dict[str, float]:
        """Return flattened summary metrics for logging."""
        return {
            f"{prefix}/total_vars": float(self.total_vars),
            f"{prefix}/nonzero_vars": float(self.nonzero_vars),
            f"{prefix}/none_vars": float(self.none_vars),
            f"{prefix}/zero_vars": float(self.zero_vars),
            f"{prefix}/nonfinite_vars": float(self.nonfinite_vars),
            f"{prefix}/expected_missing_or_zero": float(self.expected_missing_or_zero),
            f"{prefix}/unexpected_missing_or_zero": float(self.unexpected_missing_or_zero),
            f"{prefix}/coverage_ratio": float(self.coverage_ratio),
        }


class GradientAudit:
    """Analyze gradient coverage and collect per-group summaries."""

    def __init__(self, config: GradientAuditConfig | None = None) -> None:
        self.config = config or GradientAuditConfig()

    def analyze(
        self,
        variables: Iterable[tf.Variable],
        gradients: Iterable[tf.Tensor | tf.IndexedSlices | None],
    ) -> GradientAuditReport:
        total = 0
        nonzero = 0
        none_count = 0
        zero_count = 0
        nonfinite_count = 0
        expected_count = 0
        unexpected_count = 0
        expected_samples: list[str] = []
        unexpected_samples: list[str] = []
        variable_stats: list[dict[str, Any]] = []
        include_var_stats = self.config.include_per_variable_stats
        var_stat_limit = self.config.per_variable_limit
        unlimited_var_stats = var_stat_limit <= 0

        group_stats: dict[str, dict[str, int | float]] = {}
        for group in self.config.group_patterns:
            group_stats[group] = {
                "total": 0,
                "nonzero": 0,
                "none": 0,
                "zero": 0,
                "nonfinite": 0,
                "coverage_ratio": 0.0,
            }

        for grad, var in zip(gradients, variables):
            total += 1
            name = var.name
            expected_no_grad = any(pat in name for pat in self.config.expected_no_grad_patterns)

            group_matches = [
                group
                for group, patterns in self.config.group_patterns.items()
                if any(pat in name for pat in patterns)
            ]
            for group in group_matches:
                group_stats[group]["total"] = int(group_stats[group]["total"]) + 1

            if grad is None:
                none_count += 1
                if expected_no_grad:
                    expected_count += 1
                    if len(expected_samples) < self.config.sample_limit:
                        expected_samples.append(name)
                else:
                    unexpected_count += 1
                    if len(unexpected_samples) < self.config.sample_limit:
                        unexpected_samples.append(name)
                for group in group_matches:
                    group_stats[group]["none"] = int(group_stats[group]["none"]) + 1
                if include_var_stats and (
                    unlimited_var_stats or len(variable_stats) < var_stat_limit
                ):
                    variable_stats.append(
                        {
                            "name": name,
                            "shape": var.shape.as_list(),
                            "dtype": (
                                var.dtype.name if hasattr(var.dtype, "name") else str(var.dtype)
                            ),
                            "grad_kind": "none",
                            "expected_no_grad": expected_no_grad,
                            "status": "none",
                            "max_abs": None,
                            "mean_abs": None,
                            "norm": None,
                            "has_nan": False,
                            "has_inf": False,
                            "is_zero": None,
                            "is_nonfinite": False,
                        }
                    )
                continue

            values = grad.values if isinstance(grad, tf.IndexedSlices) else grad
            if values is None:
                none_count += 1
                if expected_no_grad:
                    expected_count += 1
                    if len(expected_samples) < self.config.sample_limit:
                        expected_samples.append(name)
                else:
                    unexpected_count += 1
                    if len(unexpected_samples) < self.config.sample_limit:
                        unexpected_samples.append(name)
                for group in group_matches:
                    group_stats[group]["none"] = int(group_stats[group]["none"]) + 1
                if include_var_stats and (
                    unlimited_var_stats or len(variable_stats) < var_stat_limit
                ):
                    variable_stats.append(
                        {
                            "name": name,
                            "shape": var.shape.as_list(),
                            "dtype": (
                                var.dtype.name if hasattr(var.dtype, "name") else str(var.dtype)
                            ),
                            "grad_kind": "none",
                            "expected_no_grad": expected_no_grad,
                            "status": "none",
                            "max_abs": None,
                            "mean_abs": None,
                            "norm": None,
                            "has_nan": False,
                            "has_inf": False,
                            "is_zero": None,
                            "is_nonfinite": False,
                        }
                    )
                continue

            abs_values = tf.abs(values) if values.dtype.is_complex else tf.abs(values)
            max_abs = float(tf.reduce_max(abs_values).numpy())
            mean_abs = float(tf.reduce_mean(abs_values).numpy())
            is_zero = max_abs <= self.config.zero_tol

            has_nan = bool(tf.reduce_any(tf.math.is_nan(values)).numpy())
            has_inf = bool(tf.reduce_any(tf.math.is_inf(values)).numpy())
            is_nonfinite = has_nan or has_inf
            norm_val = float(tf.linalg.global_norm([values]).numpy())

            if is_nonfinite:
                nonfinite_count += 1
                for group in group_matches:
                    group_stats[group]["nonfinite"] = int(group_stats[group]["nonfinite"]) + 1

            if is_zero:
                zero_count += 1
                if expected_no_grad:
                    expected_count += 1
                    if len(expected_samples) < self.config.sample_limit:
                        expected_samples.append(name)
                else:
                    unexpected_count += 1
                    if len(unexpected_samples) < self.config.sample_limit:
                        unexpected_samples.append(name)
                for group in group_matches:
                    group_stats[group]["zero"] = int(group_stats[group]["zero"]) + 1
            else:
                nonzero += 1
                for group in group_matches:
                    group_stats[group]["nonzero"] = int(group_stats[group]["nonzero"]) + 1

            if include_var_stats and (unlimited_var_stats or len(variable_stats) < var_stat_limit):
                status = "nonzero"
                if is_nonfinite:
                    status = "nonfinite"
                elif is_zero:
                    status = "zero"
                variable_stats.append(
                    {
                        "name": name,
                        "shape": var.shape.as_list(),
                        "dtype": var.dtype.name if hasattr(var.dtype, "name") else str(var.dtype),
                        "grad_kind": type(grad).__name__,
                        "expected_no_grad": expected_no_grad,
                        "status": status,
                        "max_abs": max_abs,
                        "mean_abs": mean_abs,
                        "norm": norm_val,
                        "has_nan": has_nan,
                        "has_inf": has_inf,
                        "is_zero": is_zero,
                        "is_nonfinite": is_nonfinite,
                    }
                )

        for group, stats in group_stats.items():
            total_group = stats["total"]
            if total_group:
                stats["coverage_ratio"] = float(stats["nonzero"]) / float(total_group)

        coverage_ratio = float(nonzero) / float(total) if total else 0.0

        return GradientAuditReport(
            total_vars=total,
            nonzero_vars=nonzero,
            none_vars=none_count,
            zero_vars=zero_count,
            nonfinite_vars=nonfinite_count,
            expected_missing_or_zero=expected_count,
            unexpected_missing_or_zero=unexpected_count,
            coverage_ratio=coverage_ratio,
            group_stats=group_stats,
            expected_samples=expected_samples,
            unexpected_samples=unexpected_samples,
            variable_stats=variable_stats,
        )


__all__ = ["GradientAuditConfig", "GradientAuditReport", "GradientAudit"]
