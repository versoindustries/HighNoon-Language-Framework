# highnoon/services/aci.py
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

"""Agent Communication Interface (ACI) for HighNoon.

Provides structured message parsing and validation for agent-model
communication, enabling reliable tool call extraction and response
formatting.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ParsedMessage:
    """Parsed message from agent output.

    Attributes:
        thought: Model's reasoning/thought content.
        content: Main textual content.
        tool_call: Extracted tool call if present.
    """

    thought: str | None = None
    content: str | None = None
    tool_call: dict[str, Any] | None = None


class AciMessageParser:
    """Parser for ACI-formatted agent messages.

    Extracts structured components from model output including:
    - Thought blocks (<thought>...</thought>)
    - Tool calls (<tool_call>...</tool_call>)
    - Tool responses (<tool_response>...</tool_response>)
    """

    # Regex patterns for message components
    THOUGHT_PATTERN = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)
    TOOL_CALL_PATTERN = re.compile(
        r'<tool_call\s*(?:name=["\']([^"\']+)["\'])?\s*>(.*?)</tool_call>',
        re.DOTALL | re.IGNORECASE,
    )
    TOOL_RESPONSE_PATTERN = re.compile(
        r"<tool_response>(.*?)</tool_response>", re.DOTALL | re.IGNORECASE
    )

    def parse(self, text: str) -> ParsedMessage:
        """Parse an agent message.

        Args:
            text: Raw message text.

        Returns:
            ParsedMessage with extracted components.
        """
        result = ParsedMessage()

        # Extract thought
        thought_match = self.THOUGHT_PATTERN.search(text)
        if thought_match:
            result.thought = thought_match.group(1).strip()

        # Extract tool call
        tool_match = self.TOOL_CALL_PATTERN.search(text)
        if tool_match:
            tool_name = tool_match.group(1)
            tool_content = tool_match.group(2).strip()

            try:
                tool_data = json.loads(tool_content)
                result.tool_call = {
                    "name": tool_name or tool_data.get("name"),
                    "arguments": tool_data.get("arguments", tool_data),
                    "invocation_id": tool_data.get("invocation_id"),
                }
            except json.JSONDecodeError:
                log.warning(f"Failed to parse tool call JSON: {tool_content[:100]}")

        # Extract remaining content
        clean_text = text
        for pattern in [self.THOUGHT_PATTERN, self.TOOL_CALL_PATTERN, self.TOOL_RESPONSE_PATTERN]:
            clean_text = pattern.sub("", clean_text)
        clean_text = clean_text.strip()
        if clean_text:
            result.content = clean_text

        return result

    def extract_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract all tool calls from text.

        Args:
            text: Raw message text.

        Returns:
            List of tool call dictionaries.
        """
        tool_calls = []
        for match in self.TOOL_CALL_PATTERN.finditer(text):
            tool_name = match.group(1)
            tool_content = match.group(2).strip()

            try:
                tool_data = json.loads(tool_content)
                tool_calls.append(
                    {
                        "name": tool_name or tool_data.get("name"),
                        "arguments": tool_data.get("arguments", tool_data),
                        "invocation_id": tool_data.get("invocation_id"),
                    }
                )
            except json.JSONDecodeError:
                log.warning(f"Failed to parse tool call: {tool_content[:100]}")

        return tool_calls


class AciSchemaValidator:
    """Validator for ACI message schemas.

    Validates that agent output conforms to expected schemas and
    that tool calls reference allowlisted tools.

    Attributes:
        allowlisted_tools: Set of allowed tool names.
    """

    def __init__(self, allowlisted_tools: set[str] | None = None):
        """Initialize validator.

        Args:
            allowlisted_tools: Set of allowed tool names.
        """
        self.allowlisted_tools = allowlisted_tools or set()
        self._parser = AciMessageParser()

    def validate_text(self, text: str) -> bool:
        """Validate agent output text.

        Args:
            text: Agent output text.

        Returns:
            True if valid.

        Raises:
            ValueError: If validation fails.
        """
        parsed = self._parser.parse(text)

        if parsed.tool_call:
            tool_name = parsed.tool_call.get("name")
            if tool_name and self.allowlisted_tools:
                if tool_name not in self.allowlisted_tools:
                    raise ValueError(
                        f"Tool '{tool_name}' is not in allowlist. "
                        f"Allowed: {sorted(self.allowlisted_tools)}"
                    )

        return True

    def validate_tool_call(self, tool_call: dict[str, Any]) -> bool:
        """Validate a tool call structure.

        Args:
            tool_call: Tool call dictionary.

        Returns:
            True if valid.

        Raises:
            ValueError: If validation fails.
        """
        if not isinstance(tool_call, dict):
            raise ValueError("Tool call must be a dictionary")

        if "name" not in tool_call:
            raise ValueError("Tool call must have 'name' field")

        tool_name = tool_call["name"]
        if self.allowlisted_tools and tool_name not in self.allowlisted_tools:
            raise ValueError(f"Tool '{tool_name}' is not allowlisted")

        return True
