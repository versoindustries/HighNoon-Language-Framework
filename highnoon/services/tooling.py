# highnoon/services/tooling.py
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

"""Tooling Services for HighNoon Language Framework.

Provides tool call parsing and simulation dispatch services for
agent-based workflows.
"""

import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Parsed tool call from model output.

    Attributes:
        name: Tool name.
        arguments: Tool arguments dictionary.
        metadata: Additional metadata.
        invocation_id: Unique invocation identifier.
        raw_block: Original raw text block.
    """

    name: str
    arguments: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    invocation_id: str | None = None
    raw_block: str = ""


@dataclass
class DispatchResult:
    """Result from tool dispatch.

    Attributes:
        status: Status string ('ok', 'error', 'ignored').
        message: Human-readable message.
        data: Result data payload.
    """

    status: str
    message: str
    data: Any | None = None


class ToolCallParser:
    """Parser for extracting tool calls from model output.

    Extracts structured tool calls from various formats:
    - XML-style: <tool:name>{...}</tool>
    - Tag-style: <tool_call name="name">{...}</tool_call>
    """

    # Pattern for <tool:name>{json}</tool> format
    TOOL_TAG_PATTERN = re.compile(r"<tool:(\w+)>(.*?)</tool>", re.DOTALL | re.IGNORECASE)

    # Pattern for <tool_call name="name">{json}</tool_call> format
    TOOL_CALL_PATTERN = re.compile(
        r'<tool_call\s+name=["\']([^"\']+)["\']\s*>(.*?)</tool_call>', re.DOTALL | re.IGNORECASE
    )

    def extract_first(self, text: str) -> ToolCall | None:
        """Extract the first tool call from text.

        Args:
            text: Model output text.

        Returns:
            ToolCall if found, None otherwise.
        """
        # Try <tool:name> format first
        match = self.TOOL_TAG_PATTERN.search(text)
        if match:
            return self._parse_match(match, text)

        # Try <tool_call name="name"> format
        match = self.TOOL_CALL_PATTERN.search(text)
        if match:
            return self._parse_match(match, text)

        return None

    def extract_all(self, text: str) -> list[ToolCall]:
        """Extract all tool calls from text.

        Args:
            text: Model output text.

        Returns:
            List of ToolCall objects.
        """
        tool_calls = []

        # Find all <tool:name> matches
        for match in self.TOOL_TAG_PATTERN.finditer(text):
            tc = self._parse_match(match, text)
            if tc:
                tool_calls.append(tc)

        # Find all <tool_call name="name"> matches
        for match in self.TOOL_CALL_PATTERN.finditer(text):
            tc = self._parse_match(match, text)
            if tc:
                tool_calls.append(tc)

        return tool_calls

    def _parse_match(self, match: re.Match, full_text: str) -> ToolCall | None:
        """Parse a regex match into a ToolCall.

        Args:
            match: Regex match object.
            full_text: Original full text.

        Returns:
            ToolCall if successfully parsed.
        """
        try:
            tool_name = match.group(1)
            json_content = match.group(2).strip()
            raw_block = match.group(0)

            # Parse JSON content
            try:
                data = json.loads(json_content)
            except json.JSONDecodeError:
                # Try extracting just arguments if there's extra text
                data = {"raw": json_content}

            # Extract arguments
            if isinstance(data, dict):
                arguments = data.get("arguments", data)
                metadata = data.get("metadata", {})
                invocation_id = data.get("invocation_id", str(uuid.uuid4()))
            else:
                arguments = {"value": data}
                metadata = {}
                invocation_id = str(uuid.uuid4())

            return ToolCall(
                name=tool_name,
                arguments=arguments,
                metadata=metadata,
                invocation_id=invocation_id,
                raw_block=raw_block,
            )
        except Exception as e:
            log.warning(f"Failed to parse tool call: {e}")
            return None


class SimulationDispatcher:
    """Dispatcher for simulation tool calls.

    Routes simulation requests to appropriate handlers and returns
    results in a standardized format.

    Attributes:
        handlers: Mapping of simulation types to handler functions.
    """

    def __init__(self):
        """Initialize the dispatcher."""
        self._handlers: dict[str, Callable] = {}
        self._default_simulation_tool = "run_simulation"

    def register_handler(
        self,
        simulation_type: str,
        handler: Callable[[dict[str, Any]], DispatchResult],
    ) -> None:
        """Register a simulation handler.

        Args:
            simulation_type: Type of simulation to handle.
            handler: Handler function.
        """
        self._handlers[simulation_type] = handler
        log.info(f"Registered simulation handler: {simulation_type}")

    def dispatch(self, tool_call: ToolCall) -> DispatchResult:
        """Dispatch a tool call to the appropriate handler.

        Args:
            tool_call: ToolCall to dispatch.

        Returns:
            DispatchResult from handler.
        """
        if tool_call.name != self._default_simulation_tool:
            return DispatchResult(
                status="ignored",
                message=f"Tool '{tool_call.name}' is not a simulation tool.",
            )

        # Determine simulation type from arguments
        sim_type = tool_call.arguments.get("type", "default")

        handler = self._handlers.get(sim_type)
        if handler is None:
            return DispatchResult(
                status="error",
                message=f"No handler registered for simulation type: {sim_type}",
            )

        try:
            log.debug(f"Dispatching simulation: {sim_type}")
            result = handler(tool_call.arguments)
            return result
        except Exception as e:
            log.error(f"Simulation dispatch failed: {e}")
            return DispatchResult(
                status="error",
                message=str(e),
            )

    def list_handlers(self) -> list[str]:
        """List registered simulation types.

        Returns:
            List of registered simulation type names.
        """
        return list(self._handlers.keys())
