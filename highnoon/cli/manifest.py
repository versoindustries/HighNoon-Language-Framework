# highnoon/cli/manifest.py
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

"""Tool Manifest for HighNoon CLI.

Provides the tool registry and dispatch system for the Codex CLI,
including rate limiting and safety classification.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class ToolExecutionError(RuntimeError):
    """Raised when a tool invocation fails validation or runtime execution."""

    pass


@dataclass
class RateLimiter:
    """Rate limiter for tool invocations.

    Attributes:
        max_calls: Maximum number of calls allowed in the interval.
        interval_sec: Time window in seconds.
    """

    max_calls: int
    interval_sec: float
    _recent: list[float] = field(default_factory=list)

    def allow(self) -> bool:
        """Check if a call is allowed under the rate limit.

        Returns:
            True if the call is allowed, False otherwise.
        """
        now = time.monotonic()
        window_start = now - self.interval_sec
        self._recent = [ts for ts in self._recent if ts >= window_start]
        if len(self._recent) >= self.max_calls:
            return False
        self._recent.append(now)
        return True

    def reset(self) -> None:
        """Reset the rate limiter."""
        self._recent.clear()


@dataclass
class ToolRegistration:
    """Registration information for a tool.

    Attributes:
        name: Unique tool name.
        handler: Callable that implements the tool.
        description: Human-readable description.
        safety_class: Safety classification ('trusted', 'sandboxed', etc.).
        rate_limiter: Optional rate limiter.
        audit_tags: Metadata tags for auditing.
    """

    name: str
    handler: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    description: str
    safety_class: str = "trusted"
    rate_limiter: RateLimiter | None = None
    audit_tags: Mapping[str, str] = field(default_factory=dict)


class ToolManifest:
    """Allow-list backed tool registry for Codex CLI.

    The manifest manages tool registrations and dispatches tool calls
    from the agent loop. It supports:

    - Tool registration with metadata
    - Rate limiting per tool
    - Safety classification
    - Audit logging

    Attributes:
        base_dir: Base directory for file operations.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize the ToolManifest.

        Args:
            base_dir: Base directory for file operations (default: cwd).
        """
        self._base_dir = base_dir or Path.cwd()
        self._tools: dict[str, ToolRegistration] = {}
        log.debug(f"Initialized ToolManifest with base_dir={self._base_dir}")

    def register(
        self,
        name_or_registration: str | ToolRegistration,
    ) -> Callable | None:
        """Register a tool.

        Can be used as a decorator or called directly with a ToolRegistration.

        Args:
            name_or_registration: Tool name (for decorator) or ToolRegistration.

        Returns:
            Decorator function if name is provided, None otherwise.

        Examples:
            # As decorator
            @manifest.register("my_tool")
            def my_tool(args):
                return {"result": "done"}

            # Direct registration
            manifest.register(ToolRegistration(name="my_tool", ...))
        """
        if isinstance(name_or_registration, ToolRegistration):
            self._tools[name_or_registration.name] = name_or_registration
            log.info(f"Registered tool: {name_or_registration.name}")
            return None

        # Decorator mode
        def decorator(func: Callable) -> Callable:
            registration = ToolRegistration(
                name=name_or_registration,
                handler=func,
                description=func.__doc__ or "",
            )
            self._tools[name_or_registration] = registration
            log.info(f"Registered tool: {name_or_registration}")
            return func

        return decorator

    @property
    def names(self) -> Iterable[str]:
        """Get all registered tool names."""
        return tuple(self._tools.keys())

    def get(self, name: str) -> ToolRegistration | None:
        """Get a tool registration by name."""
        return self._tools.get(name)

    def dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        invocation_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Dispatch a tool call.

        Args:
            name: Tool name to invoke.
            arguments: Arguments to pass to the tool.
            invocation_id: Optional ID for tracking.

        Returns:
            Tool result with metadata.

        Raises:
            ToolExecutionError: If tool is not registered, rate limited, or fails.
        """
        registration = self._tools.get(name)
        if registration is None:
            raise ToolExecutionError(f"Tool '{name}' is not registered.")

        # Check rate limit
        limiter = registration.rate_limiter
        if limiter is not None and not limiter.allow():
            raise ToolExecutionError(
                f"Tool '{name}' exceeded rate limit {limiter.max_calls}/{limiter.interval_sec}s."
            )

        # Execute the tool
        try:
            log.debug(f"Dispatching tool: {name} with args: {arguments}")
            result = registration.handler(arguments)
        except Exception as exc:
            log.error(f"Tool '{name}' failed: {exc}")
            raise ToolExecutionError(f"Tool '{name}' failed: {exc}") from exc

        # Build response payload
        payload = dict(result) if isinstance(result, Mapping) else {"result": result}
        payload.setdefault("status", "ok")
        payload.setdefault("invocation_id", invocation_id)
        payload.setdefault("tool", name)
        payload.setdefault("audit", dict(registration.audit_tags))
        payload["safety_class"] = registration.safety_class

        log.debug(f"Tool '{name}' completed with status: {payload.get('status')}")
        return payload

    def list_tools(self) -> list[dict[str, str]]:
        """List all registered tools with descriptions.

        Returns:
            List of tool info dictionaries.
        """
        return [
            {
                "name": reg.name,
                "description": reg.description,
                "safety_class": reg.safety_class,
            }
            for reg in self._tools.values()
        ]

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert tool registrations to OpenAI function calling format.

        Returns:
            List of tool definitions in OpenAI format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": reg.name,
                    "description": reg.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
            for reg in self._tools.values()
        ]


# Built-in tools
def _create_default_tools() -> list[ToolRegistration]:
    """Create default tool registrations."""

    def run_command(args: Mapping[str, Any]) -> dict[str, Any]:
        """Execute a shell command."""
        import subprocess

        cmd = args.get("command", "")
        cwd = args.get("cwd")
        timeout = args.get("timeout", 30)

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out", "returncode": -1}

    def read_file(args: Mapping[str, Any]) -> dict[str, Any]:
        """Read contents of a file."""
        path = Path(args.get("path", ""))
        if not path.exists():
            return {"error": f"File not found: {path}"}
        try:
            content = path.read_text()
            return {"content": content, "path": str(path)}
        except Exception as e:
            return {"error": str(e)}

    def write_file(args: Mapping[str, Any]) -> dict[str, Any]:
        """Write contents to a file."""
        path = Path(args.get("path", ""))
        content = args.get("content", "")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return {"path": str(path), "bytes_written": len(content)}
        except Exception as e:
            return {"error": str(e)}

    def search_files(args: Mapping[str, Any]) -> dict[str, Any]:
        """Search for files matching a pattern."""
        import glob

        pattern = args.get("pattern", "*")
        directory = args.get("directory", ".")
        matches = glob.glob(f"{directory}/**/{pattern}", recursive=True)
        return {"matches": matches[:100], "count": len(matches)}

    def run_unit_tests(args: Mapping[str, Any]) -> dict[str, Any]:
        """Run pytest unit tests."""
        import subprocess

        targets = args.get("targets", ["tests/"])
        if isinstance(targets, str):
            targets = [targets]

        cmd = ["python", "-m", "pytest", "-v"] + targets
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return {
                "passed": result.returncode == 0,
                "stdout": result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout,
                "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Tests timed out"}

    return [
        ToolRegistration(
            name="run_command",
            handler=run_command,
            description="Execute a shell command",
            safety_class="sandboxed",
            rate_limiter=RateLimiter(max_calls=10, interval_sec=60),
        ),
        ToolRegistration(
            name="read_file",
            handler=read_file,
            description="Read contents of a file",
            safety_class="trusted",
        ),
        ToolRegistration(
            name="write_file",
            handler=write_file,
            description="Write contents to a file",
            safety_class="sandboxed",
            rate_limiter=RateLimiter(max_calls=20, interval_sec=60),
        ),
        ToolRegistration(
            name="search_files",
            handler=search_files,
            description="Search for files matching a pattern",
            safety_class="trusted",
        ),
        ToolRegistration(
            name="run_unit_tests",
            handler=run_unit_tests,
            description="Run pytest unit tests",
            safety_class="sandboxed",
            rate_limiter=RateLimiter(max_calls=5, interval_sec=300),
        ),
    ]


def create_default_manifest(base_dir: Path | None = None) -> ToolManifest:
    """Create a manifest with default tools registered.

    Args:
        base_dir: Base directory for file operations.

    Returns:
        ToolManifest with default tools.
    """
    manifest = ToolManifest(base_dir)
    for tool in _create_default_tools():
        manifest.register(tool)
    return manifest
