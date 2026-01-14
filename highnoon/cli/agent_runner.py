# highnoon/cli/agent_runner.py
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

"""Phase B1: Unified AgentRunner Architecture.

Provides an abstract base class for agent runners and concrete implementations
for different model backends (CodexRunner, ClaudeRunner, etc.).

Key Components:
    - AgentRunner: Abstract base class defining the agent interface
    - ClaudeRunner: Runner for Claude API integration
    - CodexRunner (in runner.py): Runner for local HighNoon models
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from highnoon.cli.manifest import ToolManifest

log = logging.getLogger(__name__)

DEFAULT_MAX_CHAIN = 8


@dataclass
class Message:
    """A message in the conversation history."""

    role: str  # "user", "assistant", "tool"
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentRunner(ABC):
    """Abstract base class for agent runners.

    Defines the common interface for all agent runners regardless of
    the underlying model backend (local model, Claude API, etc.).

    Subclasses must implement:
        - _generate(): Generate a response from the model
        - _parse_tool_call(): Parse tool calls from model response
    """

    def __init__(
        self,
        manifest: ToolManifest | None = None,
        *,
        max_chain: int = DEFAULT_MAX_CHAIN,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the AgentRunner.

        Args:
            manifest: Tool manifest for available tools.
            max_chain: Maximum agent iterations.
            system_prompt: Optional system prompt for the model.
        """
        self.manifest = manifest or ToolManifest()
        self.max_chain = max_chain
        self.system_prompt = system_prompt
        self.history: list[Message] = []

        log.info(
            f"Initialized {self.__class__.__name__}: "
            f"max_chain={max_chain}, tools={len(self.manifest.names)}"
        )

    def run(self, user_prompt: str) -> str:
        """Execute the agent loop for a user prompt.

        Args:
            user_prompt: The user's input prompt.

        Returns:
            The final response from the agent.

        Raises:
            RuntimeError: If max_chain is exceeded without a final answer.
        """
        self.history.append(Message(role="user", content=user_prompt))

        for iteration in range(self.max_chain):
            log.debug(f"Agent iteration {iteration + 1}/{self.max_chain}")

            # Generate response
            response = self._generate()
            self.history.append(Message(role="assistant", content=response.get("content", "")))

            # Check for tool calls
            tool_call = self._parse_tool_call(response)
            if tool_call:
                invocation_id = tool_call.get("invocation_id") or f"agent-{uuid.uuid4()}"

                try:
                    result = self.manifest.dispatch(
                        tool_call["name"],
                        tool_call.get("arguments", {}),
                        invocation_id=invocation_id,
                    )
                except Exception as exc:
                    result = {
                        "status": "error",
                        "message": str(exc),
                        "invocation_id": invocation_id,
                    }

                self._inject_tool_response(tool_call["name"], result, invocation_id)
                continue

            # No tool call - return final answer
            final = response.get("content", "")
            log.info(f"Agent completed in {iteration + 1} iterations")
            return final

        raise RuntimeError("Agent exceeded max_chain without producing a final answer.")

    @abstractmethod
    def _generate(self) -> dict[str, Any]:
        """Generate a response from the model.

        Returns:
            Dict containing at minimum {"content": str}.
            May also contain tool_call information.
        """
        pass

    @abstractmethod
    def _parse_tool_call(self, response: dict[str, Any]) -> dict[str, Any] | None:
        """Parse tool call from model response.

        Args:
            response: The model response dict.

        Returns:
            Tool call dict with "name" and "arguments", or None.
        """
        pass

    def _inject_tool_response(
        self,
        name: str,
        payload: dict[str, Any],
        tool_call_id: str,
    ) -> None:
        """Inject a tool response into the conversation history."""
        self.history.append(
            Message(
                role="tool",
                content=json.dumps(payload),
                name=name,
                tool_call_id=tool_call_id,
            )
        )

    def reset(self) -> None:
        """Reset the conversation history."""
        self.history.clear()

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get tool schemas for API calls."""
        return self.manifest.to_openai_tools()


class ClaudeRunner(AgentRunner):
    """Agent runner for Claude API integration.

    Uses the Anthropic Claude API for model generation.
    Supports Claude's native tool calling format.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        manifest: ToolManifest | None = None,
        *,
        max_chain: int = DEFAULT_MAX_CHAIN,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize ClaudeRunner.

        Args:
            api_key: Anthropic API key (uses env var if None).
            model: Claude model to use.
            manifest: Tool manifest.
            max_chain: Maximum agent iterations.
            system_prompt: System prompt.
            max_tokens: Maximum tokens per response.
        """
        super().__init__(manifest=manifest, max_chain=max_chain, system_prompt=system_prompt)

        self.model = model
        self.max_tokens = max_tokens
        self._client = None
        self._api_key = api_key

    def _get_client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError("anthropic package required for ClaudeRunner")
        return self._client

    def _generate(self) -> dict[str, Any]:
        """Generate response using Claude API."""
        client = self._get_client()

        # Convert history to Claude format
        messages = []
        for msg in self.history:
            if msg.role == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == "tool":
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )

        # Build request
        request_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        if self.system_prompt:
            request_kwargs["system"] = self.system_prompt

        # Add tools if available
        tools = self.get_tools_schema()
        if tools:
            request_kwargs["tools"] = self._convert_to_claude_tools(tools)

        # Call API
        response = client.messages.create(**request_kwargs)

        # Parse response
        result = {"content": ""}
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_call"] = {
                    "name": block.name,
                    "arguments": block.input,
                    "invocation_id": block.id,
                }

        return result

    def _parse_tool_call(self, response: dict[str, Any]) -> dict[str, Any] | None:
        """Parse tool call from Claude response."""
        return response.get("tool_call")

    def _convert_to_claude_tools(self, openai_tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool format to Claude format."""
        claude_tools = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool["function"]
                claude_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object"}),
                    }
                )
        return claude_tools
