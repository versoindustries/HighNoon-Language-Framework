#!/usr/bin/env python3
# examples/codex_cli.py
# Copyright 2025 Verso Industries
#
# Codex CLI agent example for HighNoon Language Framework.

"""Codex CLI Agent Example.

Demonstrates the CodexRunner for agentic task execution.

Usage:
    python examples/codex_cli.py

Requirements:
    - HighNoon Language Framework installed
    - TensorFlow 2.x
"""

import highnoon as hn


def main():
    """Run Codex CLI agent example."""
    print("HighNoon Language Framework - Codex CLI Example")
    print(f"Version: {hn.__version__}")
    print(f"Edition: {hn.__edition__}")
    print("-" * 50)

    # Create a model (using small for demo)
    print("\n[1] Creating model...")
    model = hn.create_model("highnoon-small")

    # Create the CodexRunner
    print("\n[2] Initializing CodexRunner...")
    runner = hn.CodexRunner(
        model,
        max_chain=5,  # Maximum agent iterations
    )
    print(f"    Initialized with {len(runner.manifest.names)} tools")
    print(f"    Available tools: {list(runner.manifest.names)}")

    # For demonstration, run in dry-run mode which simulates responses
    print("\n[3] Running in dry-run mode...")
    runner_dry = hn.CodexRunner(model=None, dry_run=True)

    # Execute an agentic task (dry run)
    result = runner_dry.run("List all Python files in the current directory")
    print(f"    Result: {result[:200]}...")

    # Show conversation history
    print("\n[4] Conversation history:")
    for turn in runner_dry.history.turns:
        role = turn.role.upper()
        content_preview = turn.content[:80].replace("\n", " ")
        print(f"    [{role}] {content_preview}...")

    print("\n" + "-" * 50)
    print("Codex CLI example complete!")
    print("\nNote: For full functionality, provide a trained model and run")
    print("      with actual tool execution enabled.")


if __name__ == "__main__":
    main()
