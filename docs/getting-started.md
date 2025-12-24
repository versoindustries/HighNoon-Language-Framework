# Getting Started

> HighNoon Language Framework - Installation & Quick Start Guide

This guide covers installing HighNoon and creating your first language model.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| TensorFlow | 2.15+ |
| NumPy | 1.24+ |
| **OS (Native Ops)** | **Linux x86_64** (Ubuntu 22.04+) |

> [!IMPORTANT]
> **Platform Support:** Native C++ operations are compiled for **Linux x86_64 only**. Python fallback implementations are available for development on macOS and Windows, but production performance requires Linux.

**Memory**: Minimum 16GB RAM (32GB+ recommended for larger models)

---

## Installation

### From PyPI (Recommended)

```bash
pip install highnoon
```

### From Source

```bash
git clone https://github.com/versoindustries/HighNoon-Language-Framework.git
cd HighNoon-Language-Framework
pip install -e .
```

### Optional Dependencies

```bash
# Development tools (testing, linting)
pip install highnoon[dev]

# Training extras (tensorboard, datasets)
pip install highnoon[training]

# Everything
pip install highnoon[full]
```

### Verify Installation

```python
import highnoon as hn

print(f"Version: {hn.__version__}")
print(f"Edition: {hn.__edition__}")
```

Expected output:
```
Version: 1.0.0
Edition: lite
```

---

## Quick Start

### Create a Model

```python
import highnoon as hn

# Create from preset
model = hn.create_model("highnoon-small")
print(f"Model: vocab_size={model.vocab_size}, embedding_dim={model.embedding_dim}")

# Or with custom configuration
from highnoon.config import Config, ModelConfig

config = Config(
    model=ModelConfig(
        vocab_size=32000,
        embedding_dim=768,
        num_reasoning_blocks=8,
        num_moe_experts=8,
    )
)
model = hn.create_model(config)
```

### Generate Text

```python
import highnoon as hn

model = hn.create_model("highnoon-3b")

# Basic generation
response = model.generate(
    "Explain how neural networks learn",
    max_length=256,
    temperature=0.7,
)
print(response)
```

### Train a Model

```python
import highnoon as hn

model = hn.create_model("highnoon-small")

# Create trainer
trainer = hn.Trainer(
    model,
    learning_rate=1e-4,
    batch_size=8,
)

# Add curriculum stages
trainer.add_curriculum_stage("foundation", datasets=["your_dataset"])
trainer.add_curriculum_stage("instruction", datasets=["instruction_dataset"])

# Train
trainer.train(epochs_per_stage=5)
```

### Use the Codex CLI Agent

```python
import highnoon as hn

model = hn.create_model("highnoon-3b")

# Create agent runner
runner = hn.CodexRunner(model, max_chain=5)

# Execute an agentic task
result = runner.run("List all Python files in the current directory")
print(result)
```

---

## Your First Model

Let's walk through creating and using your first HighNoon model step by step.

### Step 1: Import and Check

```python
import highnoon as hn

# Verify the framework is loaded
print(f"HighNoon {hn.__version__} - {hn.__edition__} edition")
```

### Step 2: Choose a Preset

Available presets:

| Preset | Parameters | Embedding | Blocks | Use Case |
|--------|------------|-----------|--------|----------|
| `highnoon-small` | ~125M | 512 | 4 | Development, testing |
| `highnoon-base` | ~350M | 768 | 6 | Small-scale training |
| `highnoon-3b` | ~3B | 2048 | 12 | Production inference |
| `highnoon-7b` | ~7B | 3072 | 16 | High-quality generation |
| `highnoon-13b` | ~13B | 4096 | 20 | Maximum Lite capability |

```python
# For learning and development
model = hn.create_model("highnoon-small")

# For production use
model = hn.create_model("highnoon-3b")
```

### Step 3: Run Inference

```python
import tensorflow as tf

# Create input tokens (in practice, use a tokenizer)
dummy_input = tf.random.uniform(
    (1, 32), minval=0, maxval=model.vocab_size, dtype=tf.int32
)

# Forward pass
output = model(dummy_input, training=False)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
```

### Step 4: Configure for Your Use Case

```python
from highnoon.config import Config, ModelConfig, TrainingConfig

config = Config(
    model=ModelConfig(
        vocab_size=50000,          # Custom vocabulary
        embedding_dim=1024,         # Larger embeddings
        num_reasoning_blocks=12,    # More reasoning depth
        max_seq_length=8192,        # Longer context
    ),
    training=TrainingConfig(
        learning_rate=1e-4,
        batch_size=16,
        warmup_steps=1000,
    )
)

model = hn.create_model(config)
```

---

## Next Steps

Now that you have HighNoon installed and running:

1. **[API Reference](api/models.md)** - Learn about all available classes and methods
2. **[Training Guide](guides/training.md)** - Train your own models
3. **[Curriculum Learning](guides/curriculum-learning.md)** - Advanced training strategies
4. **[Agent Tools](guides/agent-tools.md)** - Build tool-using agents

---

## Troubleshooting

### ImportError: No module named 'highnoon'

```bash
# Ensure you're in the correct environment
which python
pip list | grep highnoon

# Reinstall if needed
pip install --force-reinstall highnoon
```

### TensorFlow GPU not detected

HighNoon is CPU-first (32-bit, no GPU). This is by design for the Lite edition.

### Out of Memory

Reduce model size or batch size:

```python
model = hn.create_model("highnoon-small")  # Smaller model
trainer = hn.Trainer(model, batch_size=4)  # Smaller batch
```

### Native operations not available

If you see warnings about native ops, ensure the compiled binaries are present:

```bash
ls highnoon/_native/linux_x86_64/
```

The framework will fall back to pure Python implementations if native ops are unavailable.

---

## Getting Help

- **Community Forum**: [versoindustries.com/messages](https://versoindustries.com/messages)
- **GitHub Issues**: Report bugs and feature requests
- **Enterprise Support**: [sales@versoindustries.com](mailto:sales@versoindustries.com)

---

[← Back to Index](index.md) | [Models API →](api/models.md)
