# highnoon/services/__init__.py
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

"""HighNoon Services Module.

Provides service components for the HighNoon Language Framework:

- ACI: Agent Communication Interface for structured message parsing
- Tooling: Tool dispatch and management services
- HPO: Hyperparameter Optimization services
"""

from highnoon.services.aci import AciMessageParser, AciSchemaValidator
from highnoon.services.hpo_manager import HPOSearchSpace, HPOTrialManager
from highnoon.services.hpo_metrics import HPOMetricsCollector, TrialStatus
from highnoon.services.tooling import SimulationDispatcher, ToolCallParser

__all__ = [
    "AciMessageParser",
    "AciSchemaValidator",
    "SimulationDispatcher",
    "ToolCallParser",
    "HPOTrialManager",
    "HPOSearchSpace",
    "HPOMetricsCollector",
    "TrialStatus",
]
