# CLI API Reference

> HighNoon Language Framework - Codex CLI Agent

This document covers the `CodexRunner` agent, `ToolManifest`, and tool system.

---

## CodexRunner

Runner for executing agentic tasks with HighNoon models. Implements an agent loop that generates responses, parses tool calls, dispatches them, and continues until a final answer is produced.

```python
import highnoon as hn

model = hn.create_model("highnoon-3b")
runner = hn.CodexRunner(model, max_chain=5)

result = runner.run("List all Python files in the current directory")
print(result)
```

### Constructor

```python
from highnoon import CodexRunner

runner = CodexRunner(
    model,
    manifest=None,
    max_chain=8,
    dry_run=False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `HSMN` | *required* | HighNoon language model (or `None` for dry-run) |
| `manifest` | `ToolManifest` | `None` | Tool manifest (auto-created if not provided) |
| `max_chain` | `int` | `8` | Maximum agent iterations |
| `dry_run` | `bool` | `False` | Bypass model inference for testing |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `history` | `ConversationHistory` | Conversation history |
| `manifest` | `ToolManifest` | Tool manifest |
| `model` | `HSMN` | Language model |

### Methods

#### run()

Execute the agent loop for a user prompt.

```python
result = runner.run("Analyze the code in src/models/")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_prompt` | `str` | The user's input prompt |

Returns: `str` - Final response from the agent

Raises: `RuntimeError` if max_chain exceeded without final answer

#### reset()

Clear the conversation history.

```python
runner.reset()
```

---

## ToolManifest

Allow-list registry for tools available to the Codex CLI agent.

```python
from highnoon.cli import ToolManifest

manifest = ToolManifest(base_dir=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_dir` | `Path` | `None` | Base directory for file operations (default: cwd) |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `names` | `list[str]` | List of registered tool names |

### Methods

#### register()

Register a tool. Can be used as a decorator or called directly.

```python
# As decorator
@manifest.register("my_tool")
def my_tool(args):
    return {"result": args["input"].upper()}

# Direct registration
from highnoon.cli import ToolRegistration

manifest.register(ToolRegistration(
    name="another_tool",
    handler=my_handler,
    description="Does something useful",
))
```

#### get()

Get a tool registration by name.

```python
tool = manifest.get("run_command")
```

#### dispatch()

Dispatch a tool call.

```python
result = manifest.dispatch(
    name="run_command",
    arguments={"command": "ls -la"},
    invocation_id="call-123",
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Tool name |
| `arguments` | `dict` | Tool arguments |
| `invocation_id` | `str` | Optional tracking ID |

Returns: `dict` - Tool result with metadata

Raises: `ToolExecutionError` if tool not registered, rate limited, or fails

#### list_tools()

List all registered tools.

```python
tools = manifest.list_tools()
# [{"name": "run_command", "description": "Execute shell command", ...}, ...]
```

---

## Built-in Tools

The default manifest includes these tools:

| Tool | Description | Arguments |
|------|-------------|-----------|
| `run_command` | Execute a shell command | `command: str`, `timeout: int` |
| `read_file` | Read file contents | `path: str` |
| `write_file` | Write content to file | `path: str`, `content: str` |
| `search_files` | Search for files by pattern | `pattern: str`, `directory: str` |
| `run_unit_tests` | Run pytest tests | `targets: list[str]` |

### Tool Arguments

#### run_command

```python
{
    "command": "ls -la",
    "timeout": 30  # optional, seconds
}
```

#### read_file

```python
{
    "path": "src/main.py"
}
```

#### write_file

```python
{
    "path": "output.txt",
    "content": "Hello, world!"
}
```

#### search_files

```python
{
    "pattern": "*.py",
    "directory": "src/"  # optional
}
```

#### run_unit_tests

```python
{
    "targets": ["tests/unit/", "tests/integration/"]
}
```

---

## Creating Custom Tools

### Basic Tool

```python
from highnoon.cli import ToolManifest, ToolRegistration

manifest = ToolManifest()

@manifest.register("fetch_weather")
def fetch_weather(args):
    """Fetch weather for a location."""
    location = args.get("location", "New York")
    # Your implementation here
    return {
        "location": location,
        "temperature": 72,
        "conditions": "Sunny"
    }
```

### Tool with Rate Limiting

```python
from highnoon.cli import ToolManifest, ToolRegistration, RateLimiter

manifest = ToolManifest()

# Allow max 5 calls per 60 seconds
rate_limiter = RateLimiter(max_calls=5, interval_sec=60)

manifest.register(ToolRegistration(
    name="api_call",
    handler=my_api_handler,
    description="Call external API",
    safety_class="trusted",
    rate_limiter=rate_limiter,
))
```

### Tool with Validation

```python
@manifest.register("calculate")
def calculate(args):
    """Perform calculation."""
    if "expression" not in args:
        raise ValueError("expression is required")

    expr = args["expression"]
    # Validate expression is safe
    if any(c in expr for c in "import exec eval"):
        raise ValueError("Invalid expression")

    result = eval(expr)  # Simplified - use safe eval in production
    return {"result": result}
```

---

## ToolRegistration

Dataclass for tool registration metadata.

```python
from highnoon.cli import ToolRegistration, RateLimiter

registration = ToolRegistration(
    name="my_tool",
    handler=my_handler_function,
    description="What this tool does",
    safety_class="trusted",  # or "sandboxed", "external"
    rate_limiter=RateLimiter(max_calls=10, interval_sec=60),
    audit_tags={"category": "file_operations"},
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *required* | Unique tool identifier |
| `handler` | `Callable` | *required* | Function implementing the tool |
| `description` | `str` | *required* | Human-readable description |
| `safety_class` | `str` | `"trusted"` | Safety classification |
| `rate_limiter` | `RateLimiter` | `None` | Optional rate limiting |
| `audit_tags` | `dict` | `{}` | Metadata for auditing |

---

## RateLimiter

Rate limiter for tool invocations.

```python
from highnoon.cli import RateLimiter

limiter = RateLimiter(max_calls=10, interval_sec=60)

if limiter.allow():
    # Execute tool
    pass
else:
    # Rate limited
    pass

limiter.reset()  # Reset rate limit
```

---

## ConversationHistory

Manages conversation history for the agent loop.

```python
runner.history.add("user", "What is 2+2?")
runner.history.add("assistant", "2+2 equals 4.")

# Render as string
context = runner.history.render()
print(context)

# Clear history
runner.history.clear()
```

---

## Complete Example

```python
import highnoon as hn
from highnoon.cli import ToolManifest, ToolRegistration

# Create custom manifest with additional tools
manifest = ToolManifest()

@manifest.register("summarize_file")
def summarize_file(args):
    """Read and summarize a file."""
    path = args["path"]
    with open(path) as f:
        content = f.read()
    # In production, use model for summarization
    lines = len(content.split("\n"))
    words = len(content.split())
    return {
        "path": path,
        "lines": lines,
        "words": words,
        "preview": content[:200]
    }

# Create runner with custom manifest
model = hn.create_model("highnoon-3b")
runner = hn.CodexRunner(model, manifest=manifest, max_chain=10)

# Run agent
result = runner.run(
    "Summarize the README.md file and then run the unit tests"
)
print(result)

# Check history
for turn in runner.history.turns:
    print(f"[{turn.role}] {turn.content[:100]}...")
```

---

[← Inference API](inference.md) | [Config API →](config.md)
