# highnoon/tools/__init__.py
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

"""HighNoon Tools Module - Function Calling and Tool Use.

This module provides infrastructure for model-tool interaction:
- FunctionRegistry: Register and manage callable functions
- ToolParser: Parse tool calls from model output
- ToolExecutor: Execute tools and format results
"""

from highnoon.tools.function_registry import (
    FunctionRegistry,
    ToolCall,
    ToolDefinition,
    ToolResult,
    format_tool_result,
    parse_tool_calls,
)

__all__ = [
    "FunctionRegistry",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "parse_tool_calls",
    "format_tool_result",
]
