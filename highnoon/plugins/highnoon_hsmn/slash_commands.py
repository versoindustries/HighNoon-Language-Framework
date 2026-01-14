# highnoon/plugins/highnoon-hsmn/slash_commands.py
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

"""Phase B3: Claude Code Slash Command Registration.

Implements the /hsmn slash command for Claude Code integration.

Usage:
    /hsmn generate <prompt>    - Generate code using HSMN
    /hsmn analyze <code>       - Analyze code structure
    /hsmn train                - Check training status
    /hsmn status               - Get model status
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class SlashCommand:
    """A registered slash command."""

    name: str
    description: str
    handler: callable
    schema: dict[str, Any] | None = None


class SlashCommandRegistry:
    """Registry for slash commands.

    Manages registration and dispatch of slash commands
    for Claude Code integration.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._commands: dict[str, SlashCommand] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default slash commands."""
        self.register(
            SlashCommand(
                name="hsmn",
                description="HighNoon HSMN model commands",
                handler=self._handle_hsmn,
                schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["generate", "analyze", "train", "status"],
                        },
                        "prompt": {"type": "string"},
                    },
                    "required": ["action"],
                },
            )
        )

    def register(self, command: SlashCommand) -> None:
        """Register a slash command."""
        self._commands[command.name] = command
        log.info(f"Registered slash command: /{command.name}")

    def get(self, name: str) -> SlashCommand | None:
        """Get a command by name."""
        return self._commands.get(name)

    def list_commands(self) -> list[dict[str, str]]:
        """List all registered commands."""
        return [
            {"name": f"/{cmd.name}", "description": cmd.description}
            for cmd in self._commands.values()
        ]

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a slash command.

        Args:
            name: Command name (without leading /).
            args: Command arguments.

        Returns:
            Command result.

        Raises:
            ValueError: If command not found.
        """
        command = self._commands.get(name)
        if command is None:
            raise ValueError(f"Unknown command: /{name}")

        return command.handler(args)

    def _handle_hsmn(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle /hsmn command."""
        action = args.get("action", "status")
        prompt = args.get("prompt", "")

        if action == "generate":
            return self._hsmn_generate(prompt)
        elif action == "analyze":
            return self._hsmn_analyze(prompt)
        elif action == "train":
            return self._hsmn_train_status()
        elif action == "status":
            return self._hsmn_status()
        else:
            return {"error": f"Unknown action: {action}"}

    def _hsmn_generate(self, prompt: str) -> dict[str, Any]:
        """Generate code using HSMN."""
        # Placeholder - would invoke actual model
        return {
            "action": "generate",
            "prompt": prompt,
            "status": "success",
            "message": "HSMN generation requires model to be loaded. Use 'highnoon train' first.",
        }

    def _hsmn_analyze(self, code: str) -> dict[str, Any]:
        """Analyze code structure."""
        return {
            "action": "analyze",
            "input_length": len(code),
            "status": "success",
            "message": "Code analysis requires model. Use 'highnoon train' first.",
        }

    def _hsmn_train_status(self) -> dict[str, Any]:
        """Get training status."""
        return {
            "action": "train",
            "status": "idle",
            "message": "No training run in progress.",
        }

    def _hsmn_status(self) -> dict[str, Any]:
        """Get model status."""
        return {
            "action": "status",
            "framework": "HighNoon",
            "version": "1.0.0",
            "qsg_enabled": True,
            "model_loaded": False,
            "message": "HSMN ready. Load model to enable generation.",
        }


# Global registry instance
_registry: SlashCommandRegistry | None = None


def get_registry() -> SlashCommandRegistry:
    """Get the global slash command registry."""
    global _registry
    if _registry is None:
        _registry = SlashCommandRegistry()
    return _registry


def dispatch_command(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a slash command using the global registry."""
    return get_registry().dispatch(name, args)
