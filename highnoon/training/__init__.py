# highnoon/training/__init__.py
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

"""HighNoon Training Infrastructure.

This module provides training utilities for HighNoon language models, including:

- Trainer: High-level training orchestrator with curriculum learning
- CurriculumScheduler: Manages progressive training stages
- Distributed: Multi-node CPU training utilities
- Callbacks: Training event hooks

Example:
    >>> import highnoon as hn
    >>> model = hn.create_model("highnoon-3b")
    >>> trainer = hn.Trainer(model)
    >>> trainer.add_curriculum_stage("code_foundation", datasets=["the_stack_v2"])
    >>> trainer.train(epochs_per_stage=5, checkpoint_dir="./checkpoints")

Distributed Training Example:
    >>> from highnoon.training.distributed import create_cpu_strategy
    >>> strategy = create_cpu_strategy()
    >>> with strategy.scope():
    ...     model = hn.create_model("7b")
    ...     trainer = hn.Trainer(model)
    ...     trainer.train(...)
"""

from highnoon.training.curriculum import AdaptiveCurriculumScheduler, CurriculumScheduler
from highnoon.training.distributed import (
    ClusterConfig,
    create_cpu_strategy,
    get_distributed_dataset,
    get_worker_info,
    setup_cpu_threading,
    validate_tf_config,
)
from highnoon.training.trainer import Trainer

__all__ = [
    "AdaptiveCurriculumScheduler",
    "ClusterConfig",
    "CurriculumScheduler",
    "Trainer",
    "create_cpu_strategy",
    "get_distributed_dataset",
    "get_worker_info",
    "setup_cpu_threading",
    "validate_tf_config",
]
