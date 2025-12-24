# highnoon/cli/tools/base.py
# Copyright 2025 Verso Industries
#
# Core tool implementations for HighNoon CLI agentic capabilities.
# These tools provide basic file system operations, testing, and
# command execution with appropriate safety checks.

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Safety: Allowlisted test directories
ALLOWED_TEST_PATHS = [
    "tests/",
    "src/tests/",
    "highnoon/tests/",
]

# Safety: Blocked command patterns
BLOCKED_COMMANDS = [
    "rm -rf",
    "sudo",
    "chmod 777",
    "> /dev",
    "mkfs",
    "dd if=",
]


def run_unit_tests(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Execute pytest against specified test targets.

    Args:
        arguments: Dict with:
            - target: Test file or directory (must be in allowed paths)
            - markers: Optional pytest markers to filter tests
            - verbose: Whether to use verbose output

    Returns:
        Dict with status, stdout, stderr, and return code.
    """
    target = str(arguments.get("target", "tests/"))
    markers = arguments.get("markers", "")
    verbose = arguments.get("verbose", True)

    # Security: Validate target is in allowed paths
    target_path = Path(target)
    is_allowed = any(str(target_path).startswith(allowed) for allowed in ALLOWED_TEST_PATHS)

    if not is_allowed:
        return {
            "status": "error",
            "message": f"Target '{target}' not in allowed test paths: {ALLOWED_TEST_PATHS}",
        }

    # Build pytest command
    cmd = ["python", "-m", "pytest", target]
    if verbose:
        cmd.append("-v")
    if markers:
        cmd.extend(["-m", markers])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        return {
            "status": "ok",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Test execution timed out (5 min limit)"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def read_file(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Read file contents from disk.

    Args:
        arguments: Dict with:
            - path: File path to read
            - encoding: Text encoding (default: utf-8)

    Returns:
        Dict with status and file contents.
    """
    path = arguments.get("path")
    encoding = arguments.get("encoding", "utf-8")

    if not path:
        return {"status": "error", "message": "Missing 'path' argument"}

    file_path = Path(path)

    if not file_path.exists():
        return {"status": "error", "message": f"File not found: {path}"}

    if not file_path.is_file():
        return {"status": "error", "message": f"Not a file: {path}"}

    try:
        content = file_path.read_text(encoding=encoding)
        return {
            "status": "ok",
            "content": content,
            "size_bytes": len(content.encode(encoding)),
            "path": str(file_path.absolute()),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def write_file(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Write content to a file on disk.

    Args:
        arguments: Dict with:
            - path: File path to write
            - content: Content to write
            - encoding: Text encoding (default: utf-8)
            - create_dirs: Create parent directories if needed (default: True)

    Returns:
        Dict with status and written file info.
    """
    path = arguments.get("path")
    content = arguments.get("content")
    encoding = arguments.get("encoding", "utf-8")
    create_dirs = arguments.get("create_dirs", True)

    if not path:
        return {"status": "error", "message": "Missing 'path' argument"}

    if content is None:
        return {"status": "error", "message": "Missing 'content' argument"}

    file_path = Path(path)

    try:
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text(content, encoding=encoding)

        return {
            "status": "ok",
            "path": str(file_path.absolute()),
            "size_bytes": len(content.encode(encoding)),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def search_files(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Search for files matching a pattern.

    Args:
        arguments: Dict with:
            - directory: Directory to search in
            - pattern: Glob pattern to match (e.g., "*.py")
            - recursive: Whether to search recursively (default: True)
            - max_results: Maximum number of results (default: 100)

    Returns:
        Dict with status and list of matching files.
    """
    directory = arguments.get("directory", ".")
    pattern = arguments.get("pattern", "*")
    recursive = arguments.get("recursive", True)
    max_results = arguments.get("max_results", 100)

    dir_path = Path(directory)

    if not dir_path.exists():
        return {"status": "error", "message": f"Directory not found: {directory}"}

    if not dir_path.is_dir():
        return {"status": "error", "message": f"Not a directory: {directory}"}

    try:
        if recursive:
            matches = list(dir_path.rglob(pattern))
        else:
            matches = list(dir_path.glob(pattern))

        # Limit results
        matches = matches[:max_results]

        files = [
            {
                "path": str(m.absolute()),
                "is_dir": m.is_dir(),
                "size": m.stat().st_size if m.is_file() else None,
            }
            for m in matches
        ]

        return {
            "status": "ok",
            "matches": files,
            "count": len(files),
            "truncated": len(matches) >= max_results,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def run_command(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Execute a shell command with safety checks.

    This is a critical safety tool - commands are validated against
    a blocklist before execution. Use with caution.

    Args:
        arguments: Dict with:
            - command: Command string to execute
            - cwd: Working directory (optional)
            - timeout: Timeout in seconds (default: 60)

    Returns:
        Dict with status, stdout, stderr, and return code.
    """
    command = arguments.get("command")
    cwd = arguments.get("cwd", None)
    timeout = arguments.get("timeout", 60)

    if not command:
        return {"status": "error", "message": "Missing 'command' argument"}

    # Security: Check against blocked patterns
    for blocked in BLOCKED_COMMANDS:
        if blocked in command:
            return {
                "status": "blocked",
                "message": f"Command contains blocked pattern: '{blocked}'",
            }

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )

        return {
            "status": "ok",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": f"Command timed out ({timeout}s limit)"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
