# highnoon/cli/tools/__init__.py
# Copyright 2025 Verso Industries
#
# HighNoon Language Framework - CLI Tools Registration
#
# This module provides the tool registration system for the Codex CLI
# agentic capabilities. Tools are registered with safety classes, rate
# limiting, and audit logging.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from highnoon.cli.manifest import ToolManifest

from highnoon.cli.tools.base import read_file, run_command, run_unit_tests, search_files, write_file


def register_core_tools(manifest: ToolManifest) -> None:
    """Register all built-in tools with the manifest.

    These are the core tools available in the HighNoon Language Framework
    for agentic code generation and manipulation tasks.

    Args:
        manifest: Tool manifest to register tools with.
    """
    manifest.register_simple(
        name="run_unit_tests",
        handler=run_unit_tests,
        description="Execute pytest against specified test targets.",
        safety_class="trusted",
    )

    manifest.register_simple(
        name="read_file",
        handler=read_file,
        description="Read the contents of a file from disk.",
        safety_class="trusted",
    )

    manifest.register_simple(
        name="write_file",
        handler=write_file,
        description="Write content to a file on disk.",
        safety_class="sensitive",
    )

    manifest.register_simple(
        name="search_files",
        handler=search_files,
        description="Search for files matching a pattern in a directory.",
        safety_class="trusted",
    )

    manifest.register_simple(
        name="run_command",
        handler=run_command,
        description="Execute a shell command with safety checks.",
        safety_class="critical",
    )


__all__ = ["register_core_tools"]
