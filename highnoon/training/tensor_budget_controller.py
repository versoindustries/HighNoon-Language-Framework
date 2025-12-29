# highnoon/training/tensor_budget_controller.py
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

"""Phase 201.10: Unified Tensor Budget Controller.

This module provides cross-layer orchestration of tensor decomposition ranks
across all TN layer types (TT, Tucker, TensorRing) based on Fisher Information
importance and a global memory budget.

Key Features:
- Memory-aware rank allocation that respects a global MB budget
- Fisher-based importance weighting from FisherLayerGrouper
- Priority tiers for different layer types (LM Head > Embeddings > MoE > FFN)
- Dynamic reallocation during training based on loss sensitivity

References:
    - Tensor Network Compression for Neural Networks (NeurIPS 2020)
    - Fisher Information in Deep Learning (ICML 2019)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from highnoon.config import (
    TENSOR_BUDGET_EMBEDDING_PRIORITY,
    TENSOR_BUDGET_LM_HEAD_PRIORITY,
    TENSOR_BUDGET_MAX_RANK,
    TENSOR_BUDGET_MB,
    TENSOR_BUDGET_MIN_RANK,
    TENSOR_BUDGET_MOE_PRIORITY,
    TENSOR_BUDGET_REALLOC_INTERVAL,
    USE_TENSOR_BUDGET_CONTROLLER,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LAYER TYPE CLASSIFICATION
# =============================================================================


@dataclass
class LayerInfo:
    """Information about a tensor-decomposed layer.

    Attributes:
        name: Layer name/path.
        layer_type: Classification (embedding, lm_head, moe, attention, ffn).
        original_params: Original parameter count without compression.
        tn_type: Tensor network type (tt, tucker, tensor_ring).
        current_rank: Current rank/ranks for the TN decomposition.
        fisher_importance: Fisher Information score (if available).
        priority: Priority multiplier for rank allocation.
    """

    name: str
    layer_type: str
    original_params: int
    tn_type: str
    current_rank: int | list[int]
    fisher_importance: float = 1.0
    priority: float = 1.0


# =============================================================================
# TENSOR BUDGET CONTROLLER
# =============================================================================


class TensorBudgetController:
    """Orchestrates rank allocation across all tensor-decomposed layers.

    Manages a global memory budget for TN layers by:
    1. Classifying layers by type and importance
    2. Allocating ranks proportionally to Fisher importance with priority tiers
    3. Respecting min/max rank constraints
    4. Periodically reallocating based on training dynamics

    Example:
        >>> controller = TensorBudgetController(memory_budget_mb=1000)
        >>> controller.register_layer("encoder/tt_ffn", "tt", [512, 2048], "ffn")
        >>> controller.register_layer("decoder/lm_head", "tt", [2048, 50000], "lm_head")
        >>> # After Fisher updates
        >>> controller.update_fisher_scores(grouper.get_fisher_estimates())
        >>> # Get optimized ranks
        >>> ranks = controller.allocate_ranks()
        >>> print(ranks["encoder/tt_ffn"])  # {"tt_rank": 16}

    Attributes:
        memory_budget_mb: Total memory budget for TN layers in MB.
        min_rank: Minimum rank for any layer.
        max_rank: Maximum rank for any layer.
        realloc_interval: Training steps between reallocations.
    """

    def __init__(
        self,
        memory_budget_mb: float = TENSOR_BUDGET_MB,
        min_rank: int = TENSOR_BUDGET_MIN_RANK,
        max_rank: int = TENSOR_BUDGET_MAX_RANK,
        realloc_interval: int = TENSOR_BUDGET_REALLOC_INTERVAL,
    ):
        """Initialize tensor budget controller.

        Args:
            memory_budget_mb: Total memory budget for TN layers (MB).
            min_rank: Minimum rank for any layer.
            max_rank: Maximum rank for any layer.
            realloc_interval: Steps between rank reallocation.
        """
        self.memory_budget_mb = memory_budget_mb
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.realloc_interval = realloc_interval

        # Layer registry
        self._layers: dict[str, LayerInfo] = {}

        # Current rank allocations
        self._allocations: dict[str, dict[str, int | list[int]]] = {}

        # Training state
        self._step_counter: int = 0
        self._last_realloc_step: int = 0

        # Priority multipliers by layer type
        self._priority_map = {
            "lm_head": TENSOR_BUDGET_LM_HEAD_PRIORITY,
            "embedding": TENSOR_BUDGET_EMBEDDING_PRIORITY,
            "moe": TENSOR_BUDGET_MOE_PRIORITY,
            "attention": 1.0,
            "ffn": 0.8,
            "other": 0.5,
        }

        logger.info(
            "[TensorBudget] Initialized: budget=%.1f MB, ranks=[%d, %d]",
            memory_budget_mb,
            min_rank,
            max_rank,
        )

    def register_layer(
        self,
        name: str,
        tn_type: str,
        shape: tuple[int, ...] | list[int],
        layer_type: str = "ffn",
        current_rank: int | list[int] = 8,
    ) -> None:
        """Register a tensor-decomposed layer for budget management.

        Args:
            name: Unique layer name/path.
            tn_type: Tensor network type ("tt", "tucker", "tensor_ring").
            shape: Original weight shape (before decomposition).
            layer_type: Layer classification for priority.
            current_rank: Current rank or ranks.
        """
        original_params = int(np.prod(shape))
        priority = self._priority_map.get(layer_type, 1.0)

        self._layers[name] = LayerInfo(
            name=name,
            layer_type=layer_type,
            original_params=original_params,
            tn_type=tn_type,
            current_rank=current_rank,
            priority=priority,
        )

        logger.debug(
            "[TensorBudget] Registered %s: type=%s, tn=%s, params=%d, priority=%.2f",
            name,
            layer_type,
            tn_type,
            original_params,
            priority,
        )

    def classify_layer(self, layer_name: str) -> str:
        """Classify a layer by name pattern.

        Args:
            layer_name: Layer name/path.

        Returns:
            Layer type classification.
        """
        name_lower = layer_name.lower()

        if "lm_head" in name_lower or "output_proj" in name_lower:
            return "lm_head"
        elif "embed" in name_lower or "token" in name_lower:
            return "embedding"
        elif "moe" in name_lower or "expert" in name_lower:
            return "moe"
        elif "attn" in name_lower or "attention" in name_lower:
            return "attention"
        elif "ffn" in name_lower or "dense" in name_lower or "mlp" in name_lower:
            return "ffn"
        else:
            return "other"

    def update_fisher_scores(
        self,
        fisher_estimates: dict[str, float],
    ) -> None:
        """Update Fisher importance scores for registered layers.

        Args:
            fisher_estimates: Dictionary mapping variable names to Fisher values.
        """
        for layer_name, layer_info in self._layers.items():
            # Find matching Fisher estimate (may have partial name match)
            matching_fisher = []
            for var_name, fisher_val in fisher_estimates.items():
                if layer_name in var_name or var_name in layer_name:
                    matching_fisher.append(fisher_val)

            if matching_fisher:
                layer_info.fisher_importance = float(np.mean(matching_fisher))
            else:
                # Default to neutral if no match
                layer_info.fisher_importance = 1.0

    def _estimate_memory_mb(self, layer: LayerInfo, rank: int) -> float:
        """Estimate memory usage for a layer with given rank.

        Args:
            layer: Layer information.
            rank: Proposed rank.

        Returns:
            Estimated memory in MB.
        """
        # Approximate memory based on TN type
        # TT: O(d * r^2) per core, typically 3-4 cores
        # Tucker: O(r^d) core + sum(d_i * r) factors
        # TensorRing: O(d * r^2) similar to TT

        if layer.tn_type == "tt":
            # Assume 3-4 cores with given rank
            num_cores = 4
            params_per_core = rank * rank
            total_params = num_cores * params_per_core * 32  # float32
            return total_params * 4 / (1024 * 1024)  # bytes to MB

        elif layer.tn_type == "tucker":
            # Core + factors
            core_params = rank**2  # Simplified for 2D
            factor_params = 2 * rank * 512  # Assume avg dim 512
            total_params = (core_params + factor_params) * 4
            return total_params / (1024 * 1024)

        elif layer.tn_type == "tensor_ring":
            # Similar to TT
            num_cores = 4
            params_per_core = rank * rank
            total_params = num_cores * params_per_core * 4
            return total_params / (1024 * 1024)

        else:
            # Fallback: linear estimate
            return rank * 0.01  # ~10KB per rank unit

    def allocate_ranks(self, force: bool = False) -> dict[str, dict[str, Any]]:
        """Allocate ranks to all layers based on importance and budget.

        Args:
            force: Force reallocation regardless of interval.

        Returns:
            Dictionary mapping layer names to rank allocations.
        """
        if not USE_TENSOR_BUDGET_CONTROLLER:
            # Return current ranks unchanged
            return {
                name: {"rank": layer.current_rank, "tn_type": layer.tn_type}
                for name, layer in self._layers.items()
            }

        # Check if reallocation is due
        if not force and self._step_counter - self._last_realloc_step < self.realloc_interval:
            return self._allocations

        self._last_realloc_step = self._step_counter

        if not self._layers:
            return {}

        # Compute weighted importance scores
        total_weighted = 0.0
        weighted_scores: dict[str, float] = {}

        for name, layer in self._layers.items():
            weighted = layer.fisher_importance * layer.priority
            weighted_scores[name] = weighted
            total_weighted += weighted

        if total_weighted < 1e-10:
            total_weighted = len(self._layers)
            weighted_scores = dict.fromkeys(self._layers, 1.0)

        # Allocate ranks proportionally
        allocations: dict[str, dict[str, Any]] = {}
        total_memory = 0.0

        for name, layer in self._layers.items():
            proportion = weighted_scores[name] / total_weighted

            # Scale rank by proportion
            rank_range = self.max_rank - self.min_rank
            allocated_rank = self.min_rank + int(proportion * rank_range * len(self._layers))

            # Clamp to bounds
            allocated_rank = max(self.min_rank, min(self.max_rank, allocated_rank))

            # Check memory budget
            layer_memory = self._estimate_memory_mb(layer, allocated_rank)
            if total_memory + layer_memory > self.memory_budget_mb:
                # Reduce rank to fit budget
                while (
                    allocated_rank > self.min_rank
                    and total_memory + self._estimate_memory_mb(layer, allocated_rank)
                    > self.memory_budget_mb
                ):
                    allocated_rank -= 1

            layer_memory = self._estimate_memory_mb(layer, allocated_rank)
            total_memory += layer_memory

            # Store allocation based on TN type
            if layer.tn_type == "tt":
                allocations[name] = {"tt_rank": allocated_rank, "tn_type": "tt"}
            elif layer.tn_type == "tucker":
                allocations[name] = {
                    "tucker_ranks": [allocated_rank, allocated_rank],
                    "tn_type": "tucker",
                }
            elif layer.tn_type == "tensor_ring":
                allocations[name] = {
                    "tensor_ring_rank": allocated_rank,
                    "tn_type": "tensor_ring",
                }
            else:
                allocations[name] = {"rank": allocated_rank, "tn_type": layer.tn_type}

            # Update layer's current rank
            layer.current_rank = allocated_rank

        self._allocations = allocations

        logger.info(
            "[TensorBudget] Allocated ranks for %d layers, total memory=%.1f MB",
            len(allocations),
            total_memory,
        )

        return allocations

    def step(self) -> None:
        """Increment step counter. Call once per training step."""
        self._step_counter += 1

    def get_layer_rank(self, layer_name: str) -> dict[str, Any] | None:
        """Get current rank allocation for a layer.

        Args:
            layer_name: Layer name/path.

        Returns:
            Rank allocation dict or None if not registered.
        """
        return self._allocations.get(layer_name)

    def get_statistics(self) -> dict[str, Any]:
        """Get controller statistics.

        Returns:
            Dictionary with budget usage and allocation details.
        """
        total_memory = sum(
            self._estimate_memory_mb(layer, layer.current_rank) for layer in self._layers.values()
        )

        return {
            "enabled": USE_TENSOR_BUDGET_CONTROLLER,
            "memory_budget_mb": self.memory_budget_mb,
            "estimated_usage_mb": total_memory,
            "budget_utilization": total_memory / max(self.memory_budget_mb, 1),
            "num_layers": len(self._layers),
            "step_counter": self._step_counter,
            "last_realloc_step": self._last_realloc_step,
            "layer_types": {
                lt: sum(1 for layer in self._layers.values() if layer.layer_type == lt)
                for lt in ["lm_head", "embedding", "moe", "attention", "ffn", "other"]
            },
            "rank_distribution": {
                name: layer.current_rank
                for name, layer in list(self._layers.items())[:10]  # First 10
            },
        }

    def reset(self) -> None:
        """Reset controller state."""
        self._layers.clear()
        self._allocations.clear()
        self._step_counter = 0
        self._last_realloc_step = 0
        logger.info("[TensorBudget] Controller state reset")


__all__ = [
    "LayerInfo",
    "TensorBudgetController",
]
