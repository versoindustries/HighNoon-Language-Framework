# Configuration API Reference

> HighNoon Language Framework - Configuration System

This document covers the hierarchical configuration system including `Config`, `ModelConfig`, and `TrainingConfig`.

---

## Overview

HighNoon uses dataclass-based configuration with sensible defaults. Configuration can be customized via:

1. **Presets**: Named configurations (e.g., `"highnoon-3b"`)
2. **Config objects**: Programmatic configuration
3. **Overrides**: Keyword arguments to factory functions

```python
import highnoon as hn
from highnoon.config import Config, ModelConfig, TrainingConfig

# From preset
model = hn.create_model("highnoon-3b")

# From config object
config = Config(
    model=ModelConfig(embedding_dim=1024),
    training=TrainingConfig(learning_rate=1e-4),
)
model = hn.create_model(config)

# With overrides
model = hn.create_model("highnoon-base", num_reasoning_blocks=12)
```

---

## Config

Top-level configuration container.

```python
from highnoon.config import Config

config = Config(
    model=ModelConfig(...),
    training=TrainingConfig(...),
)
```

| Field | Type | Description |
|-------|------|-------------|
| `model` | `ModelConfig` | Model architecture configuration |
| `training` | `TrainingConfig` | Training hyperparameters |

---

## ModelConfig

Model architecture configuration.

```python
from highnoon.config import ModelConfig

model_config = ModelConfig(
    vocab_size=32000,
    embedding_dim=768,
    num_reasoning_blocks=8,
    max_seq_length=4096,
    num_moe_experts=8,
    block_pattern="mamba_timecrystal_wlam_moe_hybrid",
)
```

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | `60000` | Vocabulary size |
| `embedding_dim` | `int` | `512` | Hidden dimension (max 4096 in Lite) |
| `num_reasoning_blocks` | `int` | `6` | Reasoning depth (max 24 in Lite) |
| `max_seq_length` | `int` | `4096` | Maximum sequence length |
| `num_moe_experts` | `int` | `8` | MoE experts (max 12 in Lite) |
| `block_pattern` | `str` | `"mamba_timecrystal_wlam_moe_hybrid"` | Block architecture |

### Block Patterns

| Pattern | Description |
|---------|-------------|
| `mamba_timecrystal_wlam_moe_hybrid` | Full HSMN architecture (default) |
| `mamba_only` | Mamba-2 blocks only |
| `transformer` | Standard transformer |
| `moe_only` | MoE blocks only |

### Methods

#### to_dict()

Convert configuration to dictionary.

```python
config_dict = model_config.to_dict()
```

---

## TrainingConfig

Training hyperparameters.

```python
from highnoon.config import TrainingConfig

training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=8,
    epochs=100,
    warmup_steps=1000,
    gradient_clip=1.0,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | `float` | `1e-4` | Initial learning rate |
| `batch_size` | `int` | `8` | Training batch size |
| `epochs` | `int` | `100` | Total training epochs |
| `warmup_steps` | `int` | `1000` | LR warmup steps |
| `gradient_clip` | `float` | `1.0` | Gradient clipping norm |
| `weight_decay` | `float` | `0.01` | Weight decay coefficient |
| `adam_beta1` | `float` | `0.9` | Adam β₁ |
| `adam_beta2` | `float` | `0.999` | Adam β₂ |
| `adam_epsilon` | `float` | `1e-8` | Adam ε |

---

## Preset Configurations

Get a preset configuration programmatically:

```python
from highnoon.config import get_preset_config

config = get_preset_config("highnoon-3b")
print(config.model.embedding_dim)  # 2048
```

### Available Presets

| Preset | embedding_dim | num_blocks | num_experts | vocab_size |
|--------|---------------|------------|-------------|------------|
| `highnoon-small` | 512 | 4 | 4 | 32000 |
| `highnoon-base` | 768 | 6 | 8 | 32000 |
| `highnoon-3b` | 2048 | 12 | 8 | 50000 |
| `highnoon-7b` | 3072 | 16 | 12 | 50000 |
| `highnoon-13b` | 4096 | 20 | 12 | 60000 |
| `highnoon-20b` | 4096 | 24 | 12 | 60000 |

---

## Advanced Configuration

HighNoon exposes many advanced configuration options via module-level constants. These are defined in `highnoon/config.py`.

### Limits (Lite Edition)

```python
from highnoon import config

print(config.LITE_MAX_PARAMS)           # 20,000,000,000
print(config.LITE_MAX_REASONING_BLOCKS) # 24
print(config.LITE_MAX_MOE_EXPERTS)      # 12
print(config.LITE_MAX_CONTEXT_LENGTH)   # 5,000,000
print(config.LITE_MAX_EMBEDDING_DIM)    # 4096
```

### MoE Configuration

```python
# Number of experts activated per token
config.TOP_K_EXPERTS = 2

# Expert capacity factor
config.EXPERT_CAPACITY_FACTOR = 1.25

# Auxiliary loss weight
config.AUX_LOSS_WEIGHT = 0.01
```

### WLAM (Wavelet Attention)

```python
config.WLAM_NUM_HEADS = 8
config.WLAM_WAVELET_KERNEL_SIZE = 5
config.WLAM_USE_FLASH_LINEAR = True  # 5-8x speedup
```

### Tensor-Train Compression

```python
config.USE_TT_LM_HEAD = True
config.TT_LM_HEAD_RANKS = [1, 8, 8, 1]
config.USE_TT_EMBEDDINGS = True
```

### Meta Controller

```python
config.USE_META_CONTROLLER = True
config.META_CONTROLLER_FREQUENCY = 10  # batches
config.USE_HYBRID_PID = True
```

### Quantum Superposition Generation

```python
config.USE_QSG = True
config.QSG_NUM_SUPERPOSITION_STATES = 4
config.QSG_COLLAPSE_STRATEGY = "greedy"
```

### Speculative Decoding

```python
config.USE_SPECULATIVE_DECODING = True
config.SPECULATIVE_LOOKAHEAD = 4
```

---

## Environment Variables

Some configuration can be set via environment variables:

| Variable | Description |
|----------|-------------|
| `HIGHNOON_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING) |
| `HIGHNOON_NATIVE_OPS` | Enable/disable native ops (`1`/`0`) |
| `TF_NUM_INTEROP_THREADS` | TensorFlow inter-op parallelism |
| `TF_NUM_INTRAOP_THREADS` | TensorFlow intra-op parallelism |
| `OMP_NUM_THREADS` | OpenMP thread count |

```bash
export HIGHNOON_LOG_LEVEL=DEBUG
export TF_NUM_INTRAOP_THREADS=16
python train.py
```

---

## Configuration Validation

Configuration is validated at model creation time:

```python
import highnoon as hn
from highnoon.config import Config, ModelConfig

# This will raise LimitExceededError
config = Config(model=ModelConfig(num_reasoning_blocks=30))

try:
    model = hn.create_model(config)
except Exception as e:
    print(f"Configuration error: {e}")
    # Configuration error: num_reasoning_blocks (30) exceeds Lite limit (24)
```

---

## Complete Example

```python
import highnoon as hn
from highnoon.config import Config, ModelConfig, TrainingConfig

# Create custom configuration
config = Config(
    model=ModelConfig(
        vocab_size=50000,
        embedding_dim=1024,
        num_reasoning_blocks=12,
        max_seq_length=8192,
        num_moe_experts=8,
        block_pattern="mamba_timecrystal_wlam_moe_hybrid",
    ),
    training=TrainingConfig(
        learning_rate=5e-5,
        batch_size=16,
        warmup_steps=2000,
        gradient_clip=0.5,
        weight_decay=0.1,
    ),
)

# Create model
model = hn.create_model(config)

# Create trainer using training config
trainer = hn.Trainer(
    model,
    learning_rate=config.training.learning_rate,
    batch_size=config.training.batch_size,
)
```

---

[← CLI API](cli.md) | [Tokenization API →](tokenization.md)
