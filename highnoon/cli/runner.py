# highnoon/cli/runner.py
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

"""Codex CLI Runner for HighNoon Language Framework.

Provides the main runner for executing agentic tasks using HighNoon
language models. The runner manages conversation history, tool dispatch,
and the agent loop for multi-step reasoning.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from highnoon.cli.manifest import ToolManifest

log = logging.getLogger(__name__)

DEFAULT_MAX_CHAIN = 8


@dataclass
class ConversationTurn:
    """A single turn in the conversation history."""

    role: str
    content: str
    name: str | None = None


@dataclass
class ConversationHistory:
    """Manages the conversation history for the agent loop."""

    turns: list[ConversationTurn] = field(default_factory=list)

    def add(self, role: str, content: str, name: str | None = None) -> None:
        """Add a turn to the history."""
        self.turns.append(ConversationTurn(role=role, content=content, name=name))

    def render(self) -> str:
        """Render the history as a string for model context."""
        rendered: list[str] = []
        for turn in self.turns:
            prefix = turn.role
            if turn.name:
                prefix += f"({turn.name})"
            rendered.append(f"{prefix}: {turn.content}".strip())
        return "\n".join(rendered)

    def clear(self) -> None:
        """Clear the conversation history."""
        self.turns.clear()


class CodexRunner:
    """Runner for executing agentic tasks with HighNoon models.

    The CodexRunner implements an agent loop that:
    1. Generates responses using the language model
    2. Parses tool calls from the response
    3. Dispatches tool calls to the manifest
    4. Injects tool results back into the conversation
    5. Continues until a final answer is produced

    Attributes:
        model: HighNoon language model instance.
        manifest: Tool manifest for available tools.
        max_chain: Maximum number of agent iterations.
        history: Conversation history.
    """

    def __init__(
        self,
        model: Any | None = None,
        manifest: ToolManifest | None = None,
        *,
        max_chain: int = DEFAULT_MAX_CHAIN,
        dry_run: bool = False,
    ) -> None:
        """Initialize the CodexRunner.

        Args:
            model: HighNoon language model (or None for dry-run).
            manifest: Tool manifest (created if not provided).
            max_chain: Maximum agent iterations.
            dry_run: If True, bypass model inference.
        """
        self.model = model
        self.manifest = manifest or ToolManifest()
        self.max_chain = max_chain
        self.dry_run = dry_run
        self.history = ConversationHistory()
        self._dry_run_phase = 0

        log.info(
            f"Initialized CodexRunner: max_chain={max_chain}, "
            f"dry_run={dry_run}, tools={len(self.manifest.names)}"
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
        self.history.add("user", user_prompt)

        for iteration in range(self.max_chain):
            log.debug(f"Agent iteration {iteration + 1}/{self.max_chain}")

            # Generate response
            assistant_text, payload = self._generate()
            self.history.add("assistant", assistant_text)

            # Check for tool calls
            tool_call = payload.get("tool_call")
            if tool_call:
                invocation_id = tool_call.get("invocation_id") or f"cli-{uuid.uuid4()}"

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

                self._inject_tool_response(tool_call["name"], result)
                continue

            # No tool call - return final answer
            final = payload.get("content") or assistant_text
            log.info(f"Agent completed in {iteration + 1} iterations")
            return final

        raise RuntimeError("Model exceeded max_chain without producing a final answer.")

    def _generate(self) -> tuple[str, dict[str, Any]]:
        """Generate a response from the model.

        Returns:
            Tuple of (raw_text, parsed_payload).
        """
        context_text = self.history.render()

        if self.dry_run:
            return self._dry_run_generate()

        if self.model is None:
            raise RuntimeError("Model is not loaded; run with dry_run=True or provide a model.")

        # Generate with the model
        if hasattr(self.model, "generate_with_tools"):
            response = self.model.generate_with_tools(context=context_text, max_length=512)
            raw = response.get("raw_response") or response.get("content") or ""
            return raw, response
        else:
            # Fallback to basic generation
            raw = self.model.generate(context_text, max_length=512)
            return raw, {"type": "text", "content": raw}

    def _dry_run_generate(self) -> tuple[str, dict[str, Any]]:
        """Generate a dry-run response for testing."""
        if self._dry_run_phase == 0:
            self._dry_run_phase = 1
            fake_tool = {
                "name": "run_unit_tests",
                "arguments": {"targets": ["tests/"]},
                "invocation_id": f"dry-{uuid.uuid4()}",
            }
            raw = (
                "<thought>Executing dry-run test.</thought>"
                f'<tool_call name="{fake_tool["name"]}">{json.dumps(fake_tool)}</tool_call>'
            )
            return raw, {"type": "tool_call", "tool_call": fake_tool, "content": raw}

        final = "<thought>Dry-run complete. Tests passed.</thought>"
        return final, {"type": "text", "content": final}

    def _inject_tool_response(self, name: str, payload: dict[str, Any]) -> None:
        """Inject a tool response into the conversation history."""
        block = f"<tool_response>{json.dumps(payload)}</tool_response>"
        self.history.add("tool", block, name=name)

    def reset(self) -> None:
        """Reset the conversation history."""
        self.history.clear()
        self._dry_run_phase = 0
