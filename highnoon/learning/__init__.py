# highnoon/learning/__init__.py
# Copyright 2025-2026 Verso Industries (Author: Michael B. Zimmerman)
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

"""Model Inference Update (MIU) Learning Module.

This module provides user-activated continual learning capabilities through
hyperdimensional memory systems with QHPM anti-forgetting protection.

Key Components:
    - MIUController: Main orchestrator for learning sessions
    - ContentIndexer: Content processing and HD encoding
    - HDLearner: Hyperdimensional gradient computation
    - CrystallizedOptimizer: QHPM-protected weight updates

Example:
    >>> from highnoon.learning import MIUController, ContentSource
    >>> controller = MIUController(model)
    >>> session = controller.begin_session([
    ...     ContentSource(source_type="codebase", path="./my_project"),
    ...     ContentSource(source_type="document", path="./docs/*.md"),
    ... ])
    >>> result = controller.finalize_session(session)
    >>> print(f"Learned from {result.num_updates} updates")
"""

from highnoon.learning.content_indexer import ContentIndexer, ContentSource
from highnoon.learning.crystallized_optimizer import CrystallizedOptimizer
from highnoon.learning.hd_learner import HDLearner
from highnoon.learning.miu_controller import MIUController, MIUResult, MIUSession

__all__ = [
    "MIUController",
    "MIUSession",
    "MIUResult",
    "ContentIndexer",
    "ContentSource",
    "HDLearner",
    "CrystallizedOptimizer",
]
