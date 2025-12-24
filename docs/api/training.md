# Training API Reference

> HighNoon Language Framework - Trainer and Curriculum Classes

This document covers the training infrastructure including `Trainer`, `CurriculumScheduler`, and related utilities.

---

## Trainer

High-level training orchestrator for HighNoon language models.

```python
from highnoon import Trainer

trainer = Trainer(
    model,
    learning_rate=1e-4,
    batch_size=8,
    max_seq_length=2048,
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `tf.keras.Model` | *required* | HighNoon language model |
| `learning_rate` | `float` | `1e-4` | Initial learning rate |
| `batch_size` | `int` | `8` | Training batch size |
| `max_seq_length` | `int` | `2048` | Maximum sequence length |
| `gradient_accumulation_steps` | `int` | `1` | Gradient accumulation steps |

### Methods

#### add_curriculum_stage()

Add a curriculum stage to the training pipeline.

```python
trainer.add_curriculum_stage(
    name,
    datasets,
    epochs=None,
    weight=1.0,
    **config_overrides
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Human-readable stage name |
| `datasets` | `str` or `list[str]` | *required* | Dataset name(s) or paths |
| `epochs` | `int` | `None` | Epochs for this stage (overrides default) |
| `weight` | `float` | `1.0` | Stage importance weight |
| `**config_overrides` | `dict` | `{}` | Per-stage configuration |

Returns: `self` for method chaining

```python
trainer.add_curriculum_stage(
    "code_foundation",
    datasets=["the_stack_v2"],
    epochs=5,
)

trainer.add_curriculum_stage(
    "tool_use",
    datasets=["toolbench"],
    epochs=3,
    weight=1.5,  # Higher priority
)
```

#### add_callback()

Add a training callback function.

```python
def my_callback(event: dict):
    if event["event"] == "epoch_end":
        print(f"Epoch {event['epoch']}: loss={event['loss']:.4f}")

trainer.add_callback(my_callback)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `callback` | `Callable` | Function called with event dictionary |

**Callback Events:**

| Event | Fields | Description |
|-------|--------|-------------|
| `epoch_end` | `stage`, `epoch`, `loss` | End of epoch |
| `stage_complete` | `stage`, `metrics` | Stage finished |
| `checkpoint_saved` | `path`, `step` | Checkpoint saved |

#### train()

Train the model through all curriculum stages.

```python
summary = trainer.train(
    epochs_per_stage=5,
    save_checkpoints=True,
    checkpoint_dir="./checkpoints",
    checkpoint_interval=1000,
    log_interval=100,
    eval_interval=500,
    resume_from=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs_per_stage` | `int` | `10` | Default epochs per stage |
| `save_checkpoints` | `bool` | `True` | Whether to save checkpoints |
| `checkpoint_dir` | `str` | `None` | Checkpoint directory |
| `checkpoint_interval` | `int` | `1000` | Steps between checkpoints |
| `log_interval` | `int` | `100` | Steps between log messages |
| `eval_interval` | `int` | `500` | Steps between evaluations |
| `resume_from` | `str` | `None` | Path to resume from |

Returns: `dict` - Training summary

```python
{
    "total_steps": 15000,
    "stages_completed": [
        {"name": "code_foundation", "epochs": 5, "final_loss": 2.34},
        {"name": "tool_use", "epochs": 3, "final_loss": 1.89},
    ]
}
```

#### save()

Save trainer state and model weights.

```python
trainer.save("./my_trainer")
```

#### load() (classmethod)

Load trainer state from a directory.

```python
trainer = Trainer.load("./my_trainer", model)
```

---

## CurriculumScheduler

Curriculum learning scheduler for progressive training.

```python
from highnoon import CurriculumScheduler

scheduler = CurriculumScheduler(
    strategy="size_based",
    num_stages=5,
    warmup_epochs=10,
    stage_increment_epochs=20,
    initial_fraction=0.2,
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | `str` | `"size_based"` | Curriculum strategy |
| `num_stages` | `int` | `5` | Number of curriculum stages |
| `warmup_epochs` | `int` | `10` | Epochs before increasing |
| `stage_increment_epochs` | `int` | `20` | Epochs per stage increase |
| `initial_fraction` | `float` | `0.2` | Initial data fraction |
| `verbose` | `bool` | `True` | Print progress info |

### Strategies

| Strategy | Description |
|----------|-------------|
| `size_based` | Order by data size/length |
| `difficulty_based` | Order by computed difficulty |
| `mixed` | Combination of strategies |

### Methods

#### compute_complexity_scores()

Compute complexity scores for data samples.

```python
scores = scheduler.compute_complexity_scores(samples, strategy=None)
```

Returns: `np.ndarray` - Complexity scores (higher = more complex)

#### prepare_curriculum()

Prepare curriculum by sorting samples.

```python
sorted_indices = scheduler.prepare_curriculum(samples)
```

Returns: `np.ndarray` - Indices sorted by complexity (easy to hard)

#### get_current_subset_indices()

Get indices for current training epoch.

```python
indices = scheduler.get_current_subset_indices(epoch=5)
```

Returns: `np.ndarray` - Indices of samples to include

#### get_stage_info()

Get current stage information.

```python
info = scheduler.get_stage_info()
# {"current_stage": 2, "fraction": 0.6, "num_samples": 6000}
```

---

## AdaptiveCurriculumScheduler

Adaptive scheduler that adjusts pacing based on model performance.

```python
from highnoon.training import AdaptiveCurriculumScheduler

scheduler = AdaptiveCurriculumScheduler(
    loss_threshold_for_progression=0.1,
    min_epochs_per_stage=10,
    max_epochs_per_stage=50,
)
```

### Methods

#### update_with_loss()

Update curriculum based on training loss.

```python
progressed = scheduler.update_with_loss(epoch=15, current_loss=2.1)
if progressed:
    print("Advanced to next curriculum stage!")
```

---

## Training Utilities

### CurriculumStage

Dataclass representing a training stage.

```python
from highnoon.training.trainer import CurriculumStage

stage = CurriculumStage(
    name="foundation",
    datasets=["dataset1", "dataset2"],
    epochs=10,
    weight=1.0,
    config_overrides={"learning_rate": 5e-5},
)
```

### create_curriculum_batches()

Create batches from curriculum-selected data.

```python
from highnoon.training import create_curriculum_batches

batches = create_curriculum_batches(
    molecules=data,
    labels=labels,
    curriculum_indices=indices,
    batch_size=32,
    shuffle=True,
)
```

---

## Complete Training Example

```python
import highnoon as hn

# Create model
model = hn.create_model("highnoon-3b")

# Create trainer with custom settings
trainer = hn.Trainer(
    model,
    learning_rate=1e-4,
    batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch = 32
)

# Define curriculum
trainer.add_curriculum_stage(
    "foundation",
    datasets=["pile_subset"],
    epochs=5,
)

trainer.add_curriculum_stage(
    "code",
    datasets=["the_stack_v2"],
    epochs=3,
)

trainer.add_curriculum_stage(
    "instruction",
    datasets=["open_assistant"],
    epochs=2,
    weight=2.0,  # Higher priority
)

# Add monitoring callback
def monitor(event):
    if event["event"] == "epoch_end":
        print(f"[{event['stage']}] Epoch {event['epoch']}: loss={event['loss']:.4f}")

trainer.add_callback(monitor)

# Train
summary = trainer.train(
    checkpoint_dir="./checkpoints",
    checkpoint_interval=500,
)

print(f"Training complete! Total steps: {summary['total_steps']}")
```

---

[← Models API](models.md) | [Inference API →](inference.md)
