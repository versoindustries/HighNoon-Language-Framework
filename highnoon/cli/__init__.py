# highnoon/cli/__init__.py
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

"""HighNoon CLI - Agentic Code Intelligence Interface.

This module provides the Codex CLI integration for HighNoon, enabling
agentic code understanding and generation capabilities.

Components:
- CodexRunner: Main runner for executing agentic tasks
- ToolManifest: Registry for available tools
"""

from highnoon.cli.manifest import ToolManifest
from highnoon.cli.runner import CodexRunner

__all__ = [
    "CodexRunner",
    "ToolManifest",
]
