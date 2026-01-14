# Saguaro Usage Guide

This guide covers the command-line interface (CLI) for Saguaro.

## 1. Lifecycle Commands

### `init`
Initialize a new Saguaro project. Creates `.saguaro/` directory and config.
```bash
saguaro init
```

### `index`
Build or update the holographic knowledge graph.
```bash
saguaro index
# Options:
#   --verbose   Show detailed logs
#   --clean     Force full re-indexing
```

### `watch`
Run in daemon mode, watching for file changes and re-indexing incrementally.
```bash
saguaro watch
```

## 2. Intelligence Commands

### `query`
Semantic search.
```bash
saguaro query "How is user auth handled?"
# Options:
#   --json      Output JSON
#   --k 10      Number of results
```

### `agent` (SSAI)
Tools for AI perception.
*   `skeleton <file>`: View structural outline.
*   `slice <symbol>`: View code implementation with dependencies.
*   `impact <sandbox_id>`: Predict risk of proposed changes.

### `simulate`
Predictive modeling throughout the codebase.
*   `volatility`: Heatmap of unstable code regions.
*   `regression <file>`: Predict regression probability.

## 3. Governance Commands

### `verify` (The Sentinel)
Check compliance against rules.
```bash
saguaro verify .
# Options:
#   --fix       Auto-fix violations
#   --engines   Select engines (native, ruff, semantic)
```

### `chronicle`
Manage semantic history.
*   `snapshot`: Save current memory state.
*   `diff`: Compare current state to last snapshot.

### `health`
System diagnostics.
```bash
saguaro health
```

### `coverage`
AST parsing coverage report.
```bash
saguaro coverage
```
