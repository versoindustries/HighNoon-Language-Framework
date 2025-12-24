# Models API Reference

> HighNoon Language Framework - Model Classes and Factory Functions

This document covers the core model classes and the `create_model()` factory function.

---

## create_model()

Factory function for creating HighNoon language models.

```python
from highnoon import create_model

model = create_model(name_or_config, **kwargs)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_or_config` | `str` or `Config` | *required* | Preset name or configuration object |
| `**kwargs` | `dict` | `{}` | Overrides for model configuration |

### Returns

`HSMN` - Instantiated HighNoon language model (TensorFlow Keras model)

### Examples

```python
import highnoon as hn

# From preset name
model = hn.create_model("highnoon-3b")

# From configuration
from highnoon.config import Config, ModelConfig
config = Config(model=ModelConfig(embedding_dim=1024))
model = hn.create_model(config)

# With overrides
model = hn.create_model("highnoon-base", num_reasoning_blocks=8)
```

### Available Presets

| Preset | Parameters | Embedding Dim | Reasoning Blocks | MoE Experts |
|--------|------------|---------------|------------------|-------------|
| `highnoon-small` | ~125M | 512 | 4 | 4 |
| `highnoon-base` | ~350M | 768 | 6 | 8 |
| `highnoon-3b` | ~3B | 2048 | 12 | 8 |
| `highnoon-7b` | ~7B | 3072 | 16 | 12 |
| `highnoon-13b` | ~13B | 4096 | 20 | 12 |
| `highnoon-20b` | ~20B | 4096 | 24 | 12 |

> **Note**: The Lite edition is capped at 20B parameters, 24 reasoning blocks, and 12 MoE experts.

---

## HSMN

The main HighNoon language model class. This is a TensorFlow Keras model implementing the Hierarchical State-Space Model Network architecture.

```python
from highnoon.models import HSMN

model = HSMN(
    vocab_size=32000,
    embedding_dim=768,
    max_seq_length=4096,
    num_reasoning_blocks=8,
    block_pattern="mamba_timecrystal_wlam_moe_hybrid",
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | `32000` | Vocabulary size |
| `embedding_dim` | `int` | `768` | Hidden dimension size |
| `max_seq_length` | `int` | `4096` | Maximum sequence length |
| `num_reasoning_blocks` | `int` | `4` | Number of reasoning blocks |
| `block_pattern` | `str` | `"mamba_timecrystal_wlam_moe_hybrid"` | Block architecture pattern |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `vocab_size` | `int` | Vocabulary size |
| `embedding_dim` | `int` | Hidden dimension |
| `num_blocks` | `int` | Number of reasoning blocks |
| `max_seq_length` | `int` | Maximum sequence length |

### Methods

#### call()

Forward pass through the model.

```python
output = model(input_ids, training=False)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_ids` | `tf.Tensor` | Token IDs, shape `(batch, seq_len)` |
| `training` | `bool` | Whether in training mode |

Returns: `tf.Tensor` - Logits, shape `(batch, seq_len, vocab_size)`

#### generate()

Generate text continuation.

```python
output = model.generate(
    prompt,
    max_length=256,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` or `tf.Tensor` | *required* | Input prompt or token IDs |
| `max_length` | `int` | `256` | Maximum generation length |
| `temperature` | `float` | `1.0` | Sampling temperature |
| `top_k` | `int` | `50` | Top-k sampling |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold |

Returns: `str` or `tf.Tensor` - Generated text or tokens

#### get_config()

Get model configuration dictionary.

```python
config = model.get_config()
```

Returns: `dict` - Configuration dictionary

### Block Patterns

The `block_pattern` parameter controls the architecture of each reasoning block:

| Pattern | Description |
|---------|-------------|
| `mamba_timecrystal_wlam_moe_hybrid` | Full hybrid block (default) |
| `mamba_only` | Mamba-2 blocks only |
| `transformer` | Standard transformer blocks |
| `moe_only` | MoE blocks only |

---

## MoELayer

Mixture of Experts layer with Expert Choice routing.

```python
from highnoon.models import MoELayer

moe = MoELayer(
    num_experts=8,
    expert_dim=1024,
    num_experts_per_tok=2,
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_experts` | `int` | `8` | Number of experts (max 12 in Lite) |
| `expert_dim` | `int` | `1024` | Expert hidden dimension |
| `num_experts_per_tok` | `int` | `2` | Experts activated per token |
| `router_type` | `str` | `"expert_choice"` | Routing algorithm |

### Methods

#### call()

```python
output, aux_loss = moe(x, training=False)
```

Returns tuple of:
- `output`: Transformed tensor, same shape as input
- `aux_loss`: Auxiliary load balancing loss (for training)

---

## Additional Model Classes

These classes are available via `highnoon.models`:

| Class | Description |
|-------|-------------|
| `HamiltonianNN` | Hamiltonian neural network component |
| `TimeCrystalBlock` | Time-crystal attention block |
| `TimeCrystalSequenceBlock` | Sequence of time-crystal blocks |
| `SpatialBlock` | Spatial attention block |
| `ReasoningMamba2Block` | Mamba-2 reasoning block |
| `SuperposedExpert` | Single expert with superposition |

---

## Lite Edition Limits

The Lite edition enforces the following limits via compiled binaries:

| Limit | Value |
|-------|-------|
| Max Parameters | 20B |
| Max Reasoning Blocks | 24 |
| Max MoE Experts | 12 |
| Max Context Length | 5M tokens |
| Max Embedding Dim | 4096 |

Exceeding these limits raises `LimitExceededError`.

```python
try:
    model = hn.create_model(config)
except hn.LimitExceededError as e:
    print(f"Limit exceeded: {e}")
```

---

[← Back to Index](../index.md) | [Training API →](training.md)
