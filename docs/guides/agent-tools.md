# Agent Tools Guide

> HighNoon Language Framework - Codex CLI and Custom Tools

This guide covers building agentic applications with the Codex CLI runner and custom tools.

---

## Overview

The Codex CLI enables HighNoon models to:

- **Execute tools** - Run commands, read/write files, call APIs
- **Multi-step reasoning** - Chain multiple tool calls
- **Iterative refinement** - Use tool outputs to inform next steps

---

## Quick Start

```python
import highnoon as hn

# Create model and runner
model = hn.create_model("highnoon-3b")
runner = hn.CodexRunner(model)

# Execute an agentic task
result = runner.run("List all Python files and count their lines")
print(result)
```

---

## CodexRunner

### Basic Usage

```python
from highnoon import CodexRunner

runner = CodexRunner(
    model=model,          # HighNoon model
    max_chain=8,          # Max iterations
    dry_run=False,        # Real execution
)

result = runner.run("Summarize the README.md file")
```

### Agent Loop

The runner executes an agent loop:

```
┌─────────────────────────────────────────┐
│  User: "List Python files"              │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│  Model generates response               │
│  → Includes tool call: search_files     │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│  Tool executed                          │
│  → Results: ["main.py", "utils.py"]     │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│  Model generates final response         │
│  → "Found 2 Python files: main.py..."   │
└─────────────────────────────────────────┘
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chain` | `8` | Maximum tool-call iterations |
| `dry_run` | `False` | Skip actual tool execution |

---

## Built-in Tools

### Available Tools

| Tool | Description |
|------|-------------|
| `run_command` | Execute shell commands |
| `read_file` | Read file contents |
| `write_file` | Write to files |
| `search_files` | Find files by pattern |
| `run_unit_tests` | Run pytest |

### run_command

Execute shell commands:

```
Tool: run_command
Arguments: {"command": "ls -la", "timeout": 30}
```

### read_file

Read file contents:

```
Tool: read_file
Arguments: {"path": "src/main.py"}
```

### write_file

Write content to file:

```
Tool: write_file
Arguments: {"path": "output.txt", "content": "Hello, world!"}
```

### search_files

Search for files:

```
Tool: search_files
Arguments: {"pattern": "*.py", "directory": "src/"}
```

### run_unit_tests

Run tests:

```
Tool: run_unit_tests
Arguments: {"targets": ["tests/"]}
```

---

## Creating Custom Tools

### Decorator Syntax

```python
from highnoon.cli import ToolManifest

manifest = ToolManifest()

@manifest.register("get_weather")
def get_weather(args):
    """Get weather for a location."""
    location = args.get("location", "New York")

    # Your implementation
    return {
        "location": location,
        "temperature": 72,
        "conditions": "Sunny"
    }
```

### Registration Syntax

```python
from highnoon.cli import ToolManifest, ToolRegistration

manifest = ToolManifest()

def my_handler(args):
    return {"result": args["input"].upper()}

manifest.register(ToolRegistration(
    name="uppercase",
    handler=my_handler,
    description="Convert text to uppercase",
))
```

### Using Custom Tools

```python
from highnoon import CodexRunner

runner = CodexRunner(model, manifest=manifest)
result = runner.run("Get the weather in London")
```

---

## Tool Patterns

### API Integration

```python
import requests

@manifest.register("fetch_api")
def fetch_api(args):
    """Fetch data from an API endpoint."""
    url = args["url"]
    method = args.get("method", "GET")

    response = requests.request(method, url, timeout=30)

    return {
        "status": response.status_code,
        "data": response.json() if response.ok else None,
        "error": None if response.ok else response.text
    }
```

### Database Query

```python
import sqlite3

@manifest.register("query_db")
def query_db(args):
    """Execute a read-only SQL query."""
    query = args["query"]

    # Validate read-only
    if not query.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries allowed")

    conn = sqlite3.connect("database.db")
    cursor = conn.execute(query)
    results = cursor.fetchall()
    conn.close()

    return {"rows": results, "count": len(results)}
```

### File Processing

```python
import os

@manifest.register("analyze_directory")
def analyze_directory(args):
    """Analyze files in a directory."""
    path = args["path"]

    files = []
    for root, _, filenames in os.walk(path):
        for name in filenames:
            filepath = os.path.join(root, name)
            files.append({
                "path": filepath,
                "size": os.path.getsize(filepath),
                "ext": os.path.splitext(name)[1]
            })

    return {
        "total_files": len(files),
        "total_size": sum(f["size"] for f in files),
        "by_extension": {ext: len([f for f in files if f["ext"] == ext])
                        for ext in set(f["ext"] for f in files)}
    }
```

---

## Rate Limiting

Protect tools from abuse:

```python
from highnoon.cli import ToolManifest, ToolRegistration, RateLimiter

# Max 5 calls per minute
limiter = RateLimiter(max_calls=5, interval_sec=60)

manifest.register(ToolRegistration(
    name="expensive_api",
    handler=call_expensive_api,
    description="Call external API",
    rate_limiter=limiter,
))
```

---

## Safety Classes

Categorize tools by risk:

```python
manifest.register(ToolRegistration(
    name="read_only_tool",
    handler=read_handler,
    description="Read-only operation",
    safety_class="trusted",  # Safe
))

manifest.register(ToolRegistration(
    name="write_tool",
    handler=write_handler,
    description="Modifies files",
    safety_class="sandboxed",  # Restricted
))

manifest.register(ToolRegistration(
    name="external_api",
    handler=api_handler,
    description="External network call",
    safety_class="external",  # Network
))
```

---

## Conversation History

### Viewing History

```python
runner.run("Do something")

for turn in runner.history.turns:
    print(f"[{turn.role}] {turn.content[:100]}...")
```

### Clearing History

```python
runner.reset()  # Clear history for new conversation
```

### Continuing Conversations

```python
# First query
result1 = runner.run("List Python files")

# Follow-up uses context
result2 = runner.run("Now count lines in each one")
```

---

## Error Handling

### Tool Errors

Errors are captured and returned to the model:

```python
@manifest.register("might_fail")
def might_fail(args):
    if args.get("fail"):
        raise ValueError("Something went wrong")
    return {"success": True}
```

The model sees:
```json
{"status": "error", "message": "Something went wrong"}
```

### Validation

Validate inputs early:

```python
@manifest.register("validated_tool")
def validated_tool(args):
    # Validate required fields
    if "input" not in args:
        raise ValueError("'input' is required")

    # Validate types
    if not isinstance(args["input"], str):
        raise TypeError("'input' must be a string")

    # Validate values
    if len(args["input"]) > 1000:
        raise ValueError("'input' too long (max 1000 chars)")

    return {"result": process(args["input"])}
```

---

## Complete Example

```python
import highnoon as hn
from highnoon.cli import ToolManifest, ToolRegistration, RateLimiter

# Create custom manifest
manifest = ToolManifest()

# Add custom tools
@manifest.register("count_words")
def count_words(args):
    """Count words in text."""
    text = args["text"]
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "unique_words": len(set(words))
    }

@manifest.register("summarize_file")
def summarize_file(args):
    """Summarize a file's contents."""
    with open(args["path"]) as f:
        content = f.read()

    lines = content.split("\n")
    return {
        "path": args["path"],
        "lines": len(lines),
        "words": len(content.split()),
        "preview": content[:200]
    }

# Rate-limited API tool
api_limiter = RateLimiter(max_calls=10, interval_sec=60)

manifest.register(ToolRegistration(
    name="fetch_url",
    handler=lambda args: {"content": "..."},
    description="Fetch URL content",
    rate_limiter=api_limiter,
    safety_class="external",
))

# Create runner
model = hn.create_model("highnoon-3b")
runner = hn.CodexRunner(model, manifest=manifest, max_chain=10)

# Execute task
result = runner.run("""
Analyze the README.md file:
1. Count the words
2. Summarize its contents
3. Report your findings
""")

print(result)

# View what happened
print("\n--- Conversation History ---")
for turn in runner.history.turns:
    role = turn.role.upper()
    preview = turn.content[:80].replace('\n', ' ')
    print(f"[{role}] {preview}...")
```

---

## Best Practices

### 1. Clear Tool Descriptions

```python
# Good - clear description
@manifest.register("calculate_hash")
def calculate_hash(args):
    """Calculate SHA-256 hash of a file.

    Args:
        path: Path to file to hash

    Returns:
        hash: Hex-encoded SHA-256 hash
    """
```

### 2. Validate All Inputs

```python
def safe_tool(args):
    # Check required
    assert "input" in args, "Missing 'input'"

    # Check types
    assert isinstance(args["input"], str)

    # Check bounds
    assert len(args["input"]) <= 10000
```

### 3. Return Structured Data

```python
# Good - structured response
return {
    "success": True,
    "data": result,
    "metadata": {"elapsed_ms": 42}
}

# Bad - unstructured string
return f"Result: {result}"
```

### 4. Handle Errors Gracefully

```python
try:
    result = do_operation()
    return {"success": True, "data": result}
except SpecificError as e:
    return {"success": False, "error": str(e)}
```

---

[← Curriculum Learning](curriculum-learning.md) | [HPO Guide →](hpo.md)
