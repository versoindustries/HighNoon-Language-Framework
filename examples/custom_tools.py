#!/usr/bin/env python3
# examples/custom_tools.py
# Copyright 2025 Verso Industries
#
# Custom tool registration example for HighNoon Language Framework.

"""Custom Tools Example.

Demonstrates how to register custom tools with the ToolManifest.

Usage:
    python examples/custom_tools.py

Requirements:
    - HighNoon Language Framework installed
"""

from highnoon.cli import CodexRunner, ToolManifest


def main():
    """Run custom tools example."""
    print("HighNoon Language Framework - Custom Tools Example")
    print("-" * 50)

    # Create a new tool manifest
    print("\n[1] Creating ToolManifest...")
    manifest = ToolManifest()

    # Register a custom tool using the decorator pattern
    print("\n[2] Registering custom tools...")

    @manifest.register("analyze_code")
    def analyze_code(args: dict) -> dict:
        """Analyze a Python file for complexity metrics.

        Args:
            args: Dictionary containing:
                - filepath: Path to the Python file

        Returns:
            Analysis results including complexity score.
        """
        filepath = args.get("filepath", "")

        # Simple demo analysis
        try:
            with open(filepath) as f:
                content = f.read()

            lines = content.split("\n")
            return {
                "filepath": filepath,
                "lines": len(lines),
                "functions": content.count("def "),
                "classes": content.count("class "),
                "complexity_estimate": "medium",
            }
        except FileNotFoundError:
            return {"error": f"File not found: {filepath}"}

    @manifest.register("format_code")
    def format_code(args: dict) -> dict:
        """Format Python code using standard formatting.

        Args:
            args: Dictionary containing:
                - code: Python code string to format

        Returns:
            Formatted code string.
        """
        code = args.get("code", "")

        # Simple demo: just add consistent spacing
        formatted = code.strip()
        return {
            "formatted_code": formatted,
            "changes_made": True,
        }

    # List registered tools
    print("\n[3] Registered tools:")
    for tool_info in manifest.list_tools():
        print(f"    - {tool_info['name']}: {tool_info['description'][:50]}...")

    # Dispatch a tool call
    print("\n[4] Dispatching 'analyze_code' tool...")
    result = manifest.dispatch(
        "analyze_code",
        {"filepath": __file__},  # Analyze this file
        invocation_id="demo-001",
    )
    print(f"    Result: {result}")

    # Create runner with custom manifest
    print("\n[5] Creating CodexRunner with custom manifest...")
    runner = CodexRunner(
        model=None,
        manifest=manifest,
        dry_run=True,
    )
    print(f"    Runner initialized with {len(runner.manifest.names)} custom tools")

    print("\n" + "-" * 50)
    print("Custom tools example complete!")


if __name__ == "__main__":
    main()
