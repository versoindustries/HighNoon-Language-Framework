# highnoon/models/reasoning/__init__.py
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

"""HighNoon Reasoning Module.

This module provides multi-layer reasoning capabilities with:

- ReasoningModule: Main reasoning architecture with configurable blocks
- create_reasoning_stack: Factory function for creating reasoning block stacks
- MemoryBuilder: Hierarchical memory construction
- FusedReasoningBlockMixin: Base mixin for fused reasoning blocks

The reasoning module uses the 'mamba_timecrystal_wlam_moe_hybrid' block pattern:
- SpatialBlock (Mamba SSM): O(L) linear-time sequence modeling
- TimeCrystalSequenceBlock: Energy-conserving Hamiltonian dynamics
- WLAMBlock: O(L log L) wavelet-based attention
- MoELayer: Sparse expert routing
"""

from highnoon.models.reasoning.block_factory import create_reasoning_stack
from highnoon.models.reasoning.fused_contract import FusedReasoningBlockMixin, TTBasedReasoningMixin
from highnoon.models.reasoning.latent_reasoning import LatentReasoningBlock
from highnoon.models.reasoning.reasoning_module import ReasoningModule

__all__ = [
    "FusedReasoningBlockMixin",
    "TTBasedReasoningMixin",
    "create_reasoning_stack",
    "ReasoningModule",
    "LatentReasoningBlock",
]
