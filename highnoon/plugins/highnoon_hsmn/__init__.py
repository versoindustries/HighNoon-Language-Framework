# highnoon/plugins/highnoon_hsmn/__init__.py
# Claude Code Plugin for HighNoon HSMN
"""HighNoon HSMN Plugin for Claude Code integration."""

from highnoon.plugins.highnoon_hsmn.slash_commands import (
    SlashCommand,
    SlashCommandRegistry,
    dispatch_command,
    get_registry,
)

__all__ = [
    "SlashCommand",
    "SlashCommandRegistry",
    "get_registry",
    "dispatch_command",
]
