#!/usr/bin/env python3
"""
Codebase Analysis Script
------------------------
Analyzes the codebase for:
1. Dead code (unused functions, classes, variables) using Vulture.
2. Duplicate code (functions/classes with identical logic) using AST hashing.

Generates a markdown report: CODE_ANALYSIS_REPORT.md
"""

import ast
import hashlib
import logging
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
REPORT_FILE = PROJECT_ROOT / "CODE_ANALYSIS_REPORT.md"
MIN_DUPLICATE_LINES = 4  # Minimum lines of code to consider for duplication
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "migrations",
    "artifacts",
    ".gemini",
    "tests",  # Often tests have repetitive setup code
}
EXCLUDE_FILES = {"analyze_codebase.py", "setup.py", "conftest.py"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DeadCodeItem:
    filename: str
    line: int
    name: str
    confidence: int
    message: str


@dataclass
class DuplicateItem:
    name: str
    type: str  # 'function' or 'class'
    locations: list[tuple[str, int]]  # (filename, line_number)
    line_count: int


class ASTCleaner(ast.NodeTransformer):
    """Cleans AST for fingerprinting by removing docstrings and type hints."""

    def visit_FunctionDef(self, node):
        self._clean_node(node)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._clean_node(node)
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        self._clean_node(node)
        return self.generic_visit(node)

    def _clean_node(self, node):
        # Remove docstrings
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)

        # Remove type annotations (simplistic approach)
        # We generally want to treat typed/untyped same logic as duplicates?
        # For strictness, maybe keep them. Let's remove them for structural equality.
        if hasattr(node, "result"):
            node.result = None

        # Remove annotations from args if present
        if hasattr(node, "args"):
            for arg in node.args.args:
                arg.annotation = None
            for arg in node.args.kwonlyargs:
                arg.annotation = None
            if node.args.vararg:
                node.args.vararg.annotation = None
            if node.args.kwarg:
                node.args.kwarg.annotation = None

        return node


def get_ast_hash(node: ast.AST) -> str:
    """Computes a hash of the AST node."""
    # Clean the node first
    cleaner = ASTCleaner()
    cleaner.visit(node)

    # Dump string representation, ignoring attributes like lineno/col_offset
    dump = ast.dump(node, include_attributes=False)
    return hashlib.md5(dump.encode("utf-8")).hexdigest()


class DuplicateDetector:
    def __init__(self):
        self.hashes: dict[str, list[ast.AST]] = defaultdict(list)
        self.files_scanned = 0

    def scan_file(self, file_path: Path):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content, filename=str(file_path))
            self.files_scanned += 1

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Filter out small functions/classes
                    if node.end_lineno - node.lineno < MIN_DUPLICATE_LINES:
                        continue

                    # Store original location info before cleaning
                    node._file_path = str(file_path.relative_to(PROJECT_ROOT))

                    node_hash = get_ast_hash(node)
                    self.hashes[node_hash].append(node)

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

    def find_duplicates(self) -> list[DuplicateItem]:
        duplicates = []
        for _, nodes in self.hashes.items():
            if len(nodes) > 1:
                # Pick the first one for metadata
                first = nodes[0]
                locations = [(n._file_path, n.lineno) for n in nodes]
                # Sort locations by filename
                locations.sort()

                duplicates.append(
                    DuplicateItem(
                        name=first.name,
                        type=type(first).__name__.replace("Def", ""),
                        locations=locations,
                        line_count=first.end_lineno - first.lineno,
                    )
                )

        # Sort by line count (impact) descending
        duplicates.sort(key=lambda x: x.line_count, reverse=True)
        return duplicates


class DeadCodeAnalyzer:
    def __init__(self):
        self.items: list[DeadCodeItem] = []

    def run(self) -> bool:
        """Runs vulture and parses output. Returns True if successful."""
        logger.info("Running Vulture for dead code analysis...")
        try:
            # We use subprocess to run vulture as a CLI tool
            # Assuming vulture is installed
            cmd = [
                sys.executable,
                "-m",
                "vulture",
                str(PROJECT_ROOT),
                "--exclude",
                ",".join(EXCLUDE_DIRS),
                "--min-confidence",
                "60",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            # Vulture returns exit code 1 if dead code is found, which is expected.
            # We only error on real failures (like command not found)
            if result.returncode not in (0, 1):
                logger.error(f"Vulture failed: {result.stderr}")
                return False

            self._parse_output(result.stdout)
            return True

        except FileNotFoundError:
            logger.error("Vulture not found. Please install it with 'pip install vulture'.")
            return False

    def _parse_output(self, output: str):
        # Example line: "highnoon/utils.py:15: unused function 'helper' (60% confidence)"
        for line in output.splitlines():
            if not line.strip():
                continue

            try:
                parts = line.split(":")
                if len(parts) >= 3:
                    rel_path = parts[0].strip()
                    lineno = int(parts[1])
                    message = ":".join(parts[2:]).strip()

                    # Extract name and confidence from message if possible
                    # message like: "unused function 'helper' (60% confidence)"
                    name = "unknown"
                    if "'" in message:
                        name = message.split("'")[1]

                    confidence = 0
                    if "(" in message and "% confidence)" in message:
                        conf_str = message.split("(")[-1].replace("% confidence)", "")
                        if conf_str.isdigit():
                            confidence = int(conf_str)

                    # Filter out excluded files if vulture missed them
                    if any(ex in rel_path for ex in EXCLUDE_FILES):
                        continue

                    self.items.append(
                        DeadCodeItem(
                            filename=rel_path,
                            line=lineno,
                            name=name,
                            confidence=confidence,
                            message=message,
                        )
                    )
            except ValueError:
                continue

        # Sort by confidence
        self.items.sort(key=lambda x: x.confidence, reverse=True)


def generate_report(dead_code: list[DeadCodeItem], duplicates: list[DuplicateItem]):
    logger.info(f"Generating report at {REPORT_FILE}...")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# Code Analysis Report\n\n")
        f.write("> **Auto-generated by `scripts/analyze_codebase.py`**\n")
        f.write(f"> **Date**: {subprocess.getoutput('date')}\n\n")

        # Summary
        f.write("## 📊 Summary\n\n")
        f.write(f"- **Dead Code Candidates**: {len(dead_code)}\n")
        f.write(f"- **Duplicate Blocks**: {len(duplicates)}\n\n")

        # Duplicates
        f.write("## 👯 Duplicate Code\n\n")
        if not duplicates:
            f.write("✅ No significant duplicates found.\n\n")
        else:
            f.write("| Type | Name | Lines | Locations |\n")
            f.write("|------|------|-------|-----------|\n")
            for dup in duplicates:
                locs = "<br>".join([f"`{path}:{line}`" for path, line in dup.locations])
                f.write(f"| {dup.type} | `{dup.name}` | {dup.line_count} | {locs} |\n")
            f.write("\n")

        # Dead Code
        f.write("## 💀 Dead Code Candidates\n\n")
        if not dead_code:
            f.write("✅ No dead code found (high confidence).\n\n")
        else:
            f.write("> Note: confidence > 60%. Always verify before deleting.\n\n")
            f.write("| Confidence | File | Line | Symbol | Message |\n")
            f.write("|------------|------|------|--------|---------|\n")
            for item in dead_code:
                f.write(
                    f"| {item.confidence}% | `{item.filename}` | {item.line} | `{item.name}` | {item.message} |\n"
                )
            f.write("\n")


def main():
    logger.info("Starting codebase analysis...")

    # 1. Duplicate Detection
    logger.info("Phase 1: Detecting Duplicates...")
    detector = DuplicateDetector()

    python_files = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip excluded dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in files:
            if file.endswith(".py") and file not in EXCLUDE_FILES:
                python_files.append(Path(root) / file)

    for p in python_files:
        detector.scan_file(p)

    duplicates = detector.find_duplicates()
    logger.info(f"Found {len(duplicates)} duplicate items.")

    # 2. Dead Code Detection
    logger.info("Phase 2: Detecting Dead Code...")
    dead_analyzer = DeadCodeAnalyzer()
    has_vulture = dead_analyzer.run()

    if not has_vulture:
        logger.warning("Vulture analysis skipped (not installed).")

    # 3. Report
    generate_report(dead_analyzer.items, duplicates)
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
