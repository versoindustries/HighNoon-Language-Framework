# highnoon/tools/function_registry.py
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

"""Phase 13.10: Tool Use / Function Calling Infrastructure.

This module provides the infrastructure for native tool calling with
MCP (Model Context Protocol) support, enabling agents, code execution,
and external API integration.

Special tokens used:
- <tool>: Start of tool definition/call block
- <call>: Invoke a tool with arguments
- <result>: Tool execution result
- </tool>: End of tool block

Example:
    >>> registry = FunctionRegistry()
    >>> @registry.register("get_weather")
    ... def get_weather(city: str) -> str:
    ...     return f"Weather in {city}: Sunny, 72°F"
    >>>
    >>> # Parse tool call from model output
    >>> text = "<tool><call>get_weather</call>{\"city\": \"NYC\"}</tool>"
    >>> calls = parse_tool_calls(text)
    >>> result = registry.execute(calls[0])
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from highnoon.config import MAX_TOOL_CALLS, TOOL_SPECIAL_TOKENS

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a callable tool.

    Attributes:
        name: Unique tool name.
        description: Human-readable description of what the tool does.
        parameters: JSON schema for tool parameters.
        function: The callable function to execute.
        returns: Description of return value.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]
    returns: str = "string"

    def to_schema(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible function schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolCall:
    """Represents a parsed tool call from model output.

    Attributes:
        name: Name of the tool to call.
        arguments: Dictionary of arguments to pass.
        call_id: Optional unique identifier for this call.
    """

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass
class ToolResult:
    """Result of executing a tool.

    Attributes:
        call_id: ID of the original call (if provided).
        name: Tool name that was called.
        result: The return value from the tool.
        error: Error message if execution failed.
        success: Whether execution succeeded.
    """

    call_id: str | None
    name: str
    result: Any = None
    error: str | None = None
    success: bool = True


class FunctionRegistry:
    """Registry for callable tools/functions.

    This class manages tool registration, schema generation, and execution.
    It supports both decorator-based and explicit registration.

    Example:
        >>> registry = FunctionRegistry()
        >>>
        >>> @registry.register("calculator")
        ... def calculator(expression: str) -> float:
        ...     '''Evaluate a mathematical expression.'''
        ...     return eval(expression)
        >>>
        >>> # Or explicit registration
        >>> def search(query: str) -> list[str]:
        ...     return ["result1", "result2"]
        >>> registry.add_tool("search", search, description="Search the web")
    """

    def __init__(self, max_calls: int = MAX_TOOL_CALLS):
        """Initialize the function registry.

        Args:
            max_calls: Maximum tool calls allowed per generation.
        """
        self._tools: dict[str, ToolDefinition] = {}
        self.max_calls = max_calls

    def register(
        self,
        name: str,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Callable[[Callable], Callable]:
        """Decorator to register a function as a tool.

        Args:
            name: Unique tool name.
            description: Tool description (defaults to docstring).
            parameters: JSON schema for parameters (auto-generated if not provided).

        Returns:
            Decorator function.

        Example:
            >>> @registry.register("greet")
            ... def greet(name: str) -> str:
            ...     '''Greet a person by name.'''
            ...     return f"Hello, {name}!"
        """

        def decorator(func: Callable) -> Callable:
            self.add_tool(name, func, description, parameters)
            return func

        return decorator

    def add_tool(
        self,
        name: str,
        function: Callable,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Add a tool to the registry.

        Args:
            name: Unique tool name.
            function: Callable function to execute.
            description: Tool description (defaults to docstring).
            parameters: JSON schema for parameters (auto-generated if not provided).

        Raises:
            ValueError: If tool with same name already exists.
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")

        # Use docstring as description if not provided
        if description is None:
            description = inspect.getdoc(function) or f"Execute {name}"

        # Auto-generate parameter schema from function signature
        if parameters is None:
            parameters = self._generate_schema(function)

        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
        )
        logger.info(f"Registered tool: {name}")

    def _generate_schema(self, function: Callable) -> dict[str, Any]:
        """Generate JSON schema from function signature.

        Args:
            function: Function to analyze.

        Returns:
            JSON schema dict for parameters.
        """
        sig = inspect.signature(function)
        hints = getattr(function, "__annotations__", {})

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Get type hint
            param_type = hints.get(param_name, Any)
            json_type = self._python_type_to_json(param_type)

            properties[param_name] = {"type": json_type}

            # Check if required (no default value)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _python_type_to_json(self, python_type: type) -> str:
        """Convert Python type to JSON schema type."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        # Handle Optional, List, etc.
        origin = getattr(python_type, "__origin__", None)
        if origin is not None:
            if origin is list:
                return "array"
            if origin is dict:
                return "object"

        return type_map.get(python_type, "string")

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible schemas for all tools."""
        return [tool.to_schema() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call.

        Args:
            call: The tool call to execute.

        Returns:
            ToolResult with execution result or error.
        """
        tool = self._tools.get(call.name)

        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                error=f"Unknown tool: {call.name}",
                success=False,
            )

        try:
            result = tool.function(**call.arguments)
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                result=result,
                success=True,
            )
        except Exception as e:
            logger.error(f"Tool execution failed: {call.name} - {e}")
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                error=str(e),
                success=False,
            )

    def execute_all(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls.

        Args:
            calls: List of tool calls to execute.

        Returns:
            List of ToolResults.

        Raises:
            ValueError: If number of calls exceeds max_calls.
        """
        if len(calls) > self.max_calls:
            raise ValueError(f"Too many tool calls: {len(calls)} > {self.max_calls}")

        return [self.execute(call) for call in calls]


# Special token patterns
TOOL_START = TOOL_SPECIAL_TOKENS[0]  # <tool>
CALL_START = TOOL_SPECIAL_TOKENS[1]  # <call>
RESULT_START = TOOL_SPECIAL_TOKENS[2]  # <result>
TOOL_END = TOOL_SPECIAL_TOKENS[3]  # </tool>

# Regex pattern for tool calls
TOOL_CALL_PATTERN = re.compile(
    rf"{re.escape(TOOL_START)}\s*"
    rf"{re.escape(CALL_START)}(\w+){re.escape(CALL_START.replace('<', '</'))}?"
    rf"\s*(.*?)\s*{re.escape(TOOL_END)}",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Parse tool calls from model output text.

    Supports formats:
    - <tool><call>name</call>{"arg": "value"}</tool>
    - <tool><call>name{"arg": "value"}</tool>

    Args:
        text: Model output text potentially containing tool calls.

    Returns:
        List of parsed ToolCall objects.

    Example:
        >>> text = '<tool><call>search</call>{"query": "weather"}</tool>'
        >>> calls = parse_tool_calls(text)
        >>> print(calls[0].name)  # "search"
        >>> print(calls[0].arguments)  # {"query": "weather"}
    """
    calls = []

    # More flexible pattern for tool calls
    pattern = re.compile(
        rf"{re.escape(TOOL_START)}\s*"
        rf"{re.escape(CALL_START)}(\w+)"
        rf"(?:</call>)?\s*(.*?)\s*{re.escape(TOOL_END)}",
        re.DOTALL,
    )

    for match in pattern.finditer(text):
        name = match.group(1)
        args_str = match.group(2).strip()

        # Parse arguments as JSON
        try:
            if args_str:
                arguments = json.loads(args_str)
            else:
                arguments = {}
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool arguments: {e}")
            arguments = {}

        calls.append(ToolCall(name=name, arguments=arguments))

    return calls


def format_tool_result(result: ToolResult) -> str:
    """Format a tool result for injection into model context.

    Args:
        result: The tool result to format.

    Returns:
        Formatted string with result or error.

    Example:
        >>> result = ToolResult(call_id=None, name="search", result=["item1"])
        >>> print(format_tool_result(result))
        # <tool><result>search: ["item1"]</result></tool>
    """
    if result.success:
        content = json.dumps(result.result) if not isinstance(result.result, str) else result.result
        return f"{TOOL_START}{RESULT_START}{result.name}: {content}</result>{TOOL_END}"
    else:
        return f"{TOOL_START}{RESULT_START}{result.name} ERROR: {result.error}</result>{TOOL_END}"


def inject_tool_descriptions(
    system_prompt: str,
    registry: FunctionRegistry,
) -> str:
    """Inject tool descriptions into a system prompt.

    Args:
        system_prompt: Original system prompt.
        registry: FunctionRegistry with available tools.

    Returns:
        System prompt with tool descriptions appended.
    """
    if not registry.list_tools():
        return system_prompt

    tool_section = "\n\n## Available Tools\n\n"
    for tool_name in registry.list_tools():
        tool = registry.get_tool(tool_name)
        if tool:
            tool_section += f"### {tool.name}\n"
            tool_section += f"{tool.description}\n"
            tool_section += f"Parameters: {json.dumps(tool.parameters, indent=2)}\n\n"

    tool_section += "\nTo use a tool, format your response as:\n"
    tool_section += f'{TOOL_START}{CALL_START}tool_name</call>{{"param": "value"}}{TOOL_END}\n'

    return system_prompt + tool_section
