# highnoon/models/spatial/__init__.py
# Copyright 2025 Verso Industries
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

"""Spatial (Mamba SSM) blocks for HSMN architecture.

This module provides linear-time sequence modeling blocks based on
state-space models (Mamba) for efficient O(L) processing.

Phase 12.4 Enhancements:
- GatedExternalMemory: Fixed-size external memory with multi-head gated access
- CompressiveGEM: Infini-attention style compressive memory
- ProductKeyMemory: Sub-linear O(√M) lookup via product-key decomposition
- QuantumInspiredMemory: MPS-based quantum-inspired associative memory
- TTProjectionMemory: TT-decomposed projections for parameter efficiency

Note: RG-LRU and SSD blocks removed in favor of Mamba as primary SSM.
"""

# Kalman Filter for state estimation (integrated into reasoning stack)
from highnoon.models.spatial.kalman import KalmanBlock
from highnoon.models.spatial.mamba import (
    QMambaBlock,
    QSSMGatedBlock,
    ReasoningMamba2Block,
    SpatialBlock,
    create_band_diagonal_mask,
)

# Phase 12.4: Gated External Memory (GEM) with Enhancements
from highnoon.models.spatial.memory_bank import (
    AdaptiveMemory,
    CompressiveGEM,
    GatedExternalMemory,
    ProductKeyMemory,
    QuantumInspiredMemory,
    TTProjectionMemory,
    create_external_memory,
)

__all__ = [
    "SpatialBlock",
    "ReasoningMamba2Block",
    "QMambaBlock",
    "QSSMGatedBlock",
    "create_band_diagonal_mask",
    # Kalman Filter
    "KalmanBlock",
    # Phase 12.4: Gated External Memory
    "GatedExternalMemory",
    "CompressiveGEM",
    "ProductKeyMemory",
    "QuantumInspiredMemory",
    "TTProjectionMemory",
    "create_external_memory",
    # Phase 18: Adaptive Memory
    "AdaptiveMemory",
]
