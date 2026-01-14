# Model Inference Update (MIU)

Teach your model new knowledge from chats, codebases, documents, and more—without forgetting what it already knows.

## Overview

Model Inference Update (MIU) enables **user-activated continual learning**. Unlike automatic fine-tuning, MIU gives you complete control over what content the model learns and when.

**Key Features:**
- 🎯 **User-Controlled**: You decide what to teach—never automatic
- 🧠 **Anti-Forgetting**: QHPM crystallization protects existing knowledge
- ⚡ **HD-Accelerated**: Hyperdimensional embeddings for efficient encoding
- 🔄 **Safe Rollback**: Automatic checkpoints before any learning

## Quick Start

```python
from highnoon.api import ModelInferenceUpdate, ContentSource

# Initialize with your model
miu = ModelInferenceUpdate(model)

# Teach from a codebase
result = miu.learn_from_content([
    ContentSource(source_type="codebase", path="./my_project"),
])

if result.success:
    print(f"✓ Learned from {result.chunks_learned} chunks!")
else:
    print(f"✗ Failed: {result.errors}")
    miu.rollback(result.checkpoint_path)  # Undo if needed
```

## Content Sources

MIU can learn from multiple content types:

### Codebases
```python
ContentSource(
    source_type="codebase",
    path="./my_project",
    filters={
        "extensions": [".py", ".js", ".md"],
        "exclude": ["**/node_modules/**", "**/tests/**"]
    }
)
```

### Documents
```python
ContentSource(
    source_type="document",
    path="./docs/**/*.md"  # Supports glob patterns
)
```

### Chat Exports
```python
# OpenAI-format JSON: {"messages": [{"role": "user", "content": "..."}]}
ContentSource(
    source_type="chat",
    path="./conversation.json"
)
```

### Raw Text
```python
ContentSource(
    source_type="text",
    content="Important information the model should remember..."
)
```

## Configuration

Customize learning behavior with `MIUConfig`:

```python
from highnoon.api.miu import MIUConfig

config = MIUConfig(
    learning_rate=1e-6,      # Extra conservative (default: 1e-5)
    max_steps=50,            # Fewer updates (default: 100)
    crystallize_after=True,  # Protect new knowledge (default: True)
    checkpoint_name="v1.0",  # Custom checkpoint name
)

result = miu.learn_from_content(sources, config=config)
```

## Progress Tracking

Monitor learning progress with a callback:

```python
def on_progress(chunks_done, total):
    print(f"Processing: {chunks_done} chunks...")

result = miu.learn_from_content(
    sources,
    on_progress=on_progress
)
```

## Checkpoints & Rollback

MIU automatically creates a checkpoint before learning, enabling safe rollback:

```python
# List available checkpoints
for cp in miu.list_checkpoints():
    print(f"{cp.name}: {cp.created_at}")

# Rollback to a specific checkpoint
miu.rollback("/path/to/checkpoint")

# Or rollback to most recent
miu.rollback()
```

## Best Practices

1. **Start Small**: Test with a small content set first
2. **Review Results**: Check model behavior after learning
3. **Use Checkpoints**: Keep rollback points for safety
4. **Crystallize Important Knowledge**: Enable `crystallize_after=True` for content that must not be forgotten

## Configuration Reference

Global settings in `highnoon/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_MIU` | `True` | Master enable switch |
| `MIU_HD_DIM` | `4096` | HD embedding dimension |
| `MIU_LEARNING_RATE` | `1e-5` | Default learning rate |
| `MIU_MAX_STEPS` | `100` | Max gradient updates per session |
| `MIU_USE_CRYSTALLIZATION` | `True` | Enable anti-forgetting |
| `MIU_CHECKPOINT_ENABLED` | `True` | Enable automatic checkpoints |
| `MIU_MAX_CHECKPOINTS` | `10` | Max retained checkpoints |

## API Reference

### ModelInferenceUpdate

```python
class ModelInferenceUpdate:
    def __init__(self, model, checkpoint_dir=None)
    def learn_from_content(self, sources, config=None, on_progress=None) -> MIUResult
    def rollback(self, checkpoint=None) -> bool
    def list_checkpoints(self) -> list[CheckpointInfo]
    @property
    def is_enabled(self) -> bool
```

### ContentSource

```python
@dataclass
class ContentSource:
    source_type: Literal["text", "codebase", "document", "chat", "web"]
    path: Path | str | None = None
    content: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
```

### MIUResult

```python
@dataclass
class MIUResult:
    success: bool
    chunks_learned: int
    num_updates: int
    checkpoint_path: Path | None
    rollback_available: bool
    message: str
    metrics: dict[str, float]
    errors: list[str]
```
