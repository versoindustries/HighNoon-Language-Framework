# highnoon/models/controllers/__init__.py
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

"""Hamiltonian Controllers for HighNoon Language Framework.

This module provides Hamiltonian Neural Network (HNN) based controllers
for energy-conserving dynamics in language modeling.

Components:
- HamiltonianNN: Lightweight HNN for physics-informed training
- TimeCrystalBlock: Single-step HNN recurrent cell
- HamiltonianMetaController: Full meta-learning controller
"""

from highnoon.models.hamiltonian import HamiltonianNN, TimeCrystalBlock, TimeCrystalSequenceBlock

__all__ = [
    "HamiltonianNN",
    "TimeCrystalBlock",
    "TimeCrystalSequenceBlock",
]
