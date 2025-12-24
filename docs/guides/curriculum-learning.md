# Curriculum Learning Guide

> HighNoon Language Framework - Advanced Training Strategies

This guide covers curriculum learning concepts and strategies for training HighNoon models.

---

## What is Curriculum Learning?

Curriculum learning trains models on progressively harder examples, mimicking how humans learn from simple to complex concepts. Benefits include:

- **Faster convergence** - Easier examples provide clear gradients
- **Better generalization** - Progressive complexity prevents overfitting
- **Improved stability** - Avoids chaotic early training

---

## Curriculum Strategies

### Stage-Based Learning

Define discrete training stages:

```python
import highnoon as hn

model = hn.create_model("highnoon-3b")
trainer = hn.Trainer(model, learning_rate=1e-4)

# Stage 1: Simple, general text
trainer.add_curriculum_stage(
    "foundation",
    datasets=["simple_wiki", "books_subset"],
    epochs=3,
)

# Stage 2: Domain-specific
trainer.add_curriculum_stage(
    "domain",
    datasets=["arxiv_papers", "documentation"],
    epochs=3,
)

# Stage 3: Task-specific, complex
trainer.add_curriculum_stage(
    "task",
    datasets=["instructions", "conversations"],
    epochs=2,
    weight=2.0,
)
```

### Built-in Strategies

The `CurriculumScheduler` provides automatic ordering:

```python
from highnoon.training import CurriculumScheduler

scheduler = CurriculumScheduler(
    strategy="size_based",
    num_stages=5,
    initial_fraction=0.2,
)
```

| Strategy | Ordering Criterion |
|----------|-------------------|
| `size_based` | Sequence length (short → long) |
| `difficulty_based` | Computed complexity score |
| `mixed` | Combination of factors |

---

## Defining Curriculum Stages

### Stage Parameters

```python
trainer.add_curriculum_stage(
    name="code_instruction",      # Stage identifier
    datasets=["commitpackft"],    # Dataset(s) to use
    epochs=5,                     # Epochs for this stage
    weight=1.5,                   # Importance weight
    learning_rate=5e-5,           # Override LR for stage
    batch_size=4,                 # Override batch size
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Unique stage name |
| `datasets` | `list[str]` | *required* | Datasets to train on |
| `epochs` | `int` | `None` | Override default epochs |
| `weight` | `float` | `1.0` | Importance multiplier |
| `**config_overrides` | `dict` | `{}` | Per-stage config overrides |

### Stage Weighting

Higher weights emphasize certain stages:

```python
# High weight = more emphasis on this stage
trainer.add_curriculum_stage(
    "tool_use",
    datasets=["toolbench"],
    weight=2.0,  # 2x importance
)
```

The weight affects:
- Loss contribution during multi-task training
- Sampling probability when mixing datasets

---

## Multi-Dataset Training

### Sequential Datasets

Train on datasets in order:

```python
# Each stage uses its own dataset
trainer.add_curriculum_stage("stage1", datasets=["dataset_a"])
trainer.add_curriculum_stage("stage2", datasets=["dataset_b"])
trainer.add_curriculum_stage("stage3", datasets=["dataset_c"])
```

### Mixed Datasets

Train on multiple datasets within a stage:

```python
# Mix datasets within a stage
trainer.add_curriculum_stage(
    "mixed_training",
    datasets=["wiki", "books", "news"],
    epochs=5,
)
```

### Weighted Mixing

Control dataset proportions:

```python
trainer.add_curriculum_stage(
    "balanced",
    datasets=[
        ("dataset_a", 0.5),  # 50% of samples
        ("dataset_b", 0.3),  # 30%
        ("dataset_c", 0.2),  # 20%
    ],
)
```

---

## Adaptive Curriculum

Adjust pacing based on model performance:

```python
from highnoon.training import AdaptiveCurriculumScheduler

scheduler = AdaptiveCurriculumScheduler(
    loss_threshold_for_progression=0.1,
    min_epochs_per_stage=10,
    max_epochs_per_stage=50,
)

# During training, update with loss
for epoch in range(100):
    loss = train_epoch()

    progressed = scheduler.update_with_loss(epoch, loss)
    if progressed:
        print(f"Advanced to stage {scheduler.get_stage_info()['current_stage']}")
```

### Automatic Progression

The model advances when:
- Loss improvement < `loss_threshold_for_progression`
- Minimum epochs have passed
- Convergence is detected

### Adaptive Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss_threshold_for_progression` | `0.1` | Min improvement to stay |
| `min_epochs_per_stage` | `10` | Earliest advancement |
| `max_epochs_per_stage` | `50` | Force advancement |

---

## Complexity Scoring

Automatically order data by complexity:

```python
from highnoon.training import CurriculumScheduler

scheduler = CurriculumScheduler(strategy="difficulty_based")

# Compute complexity scores
scores = scheduler.compute_complexity_scores(samples)

# Prepare sorted curriculum
sorted_indices = scheduler.prepare_curriculum(samples)

# Get subset for current epoch
indices = scheduler.get_current_subset_indices(epoch=5)
```

### Complexity Factors

The `difficulty_based` strategy considers:

| Factor | Weight | Description |
|--------|--------|-------------|
| Length | 0.3 | Longer = harder |
| Vocabulary | 0.2 | Rare words = harder |
| Syntax | 0.2 | Complex structure = harder |
| Domain | 0.3 | Specialized = harder |

---

## Recommended Curriculum Patterns

### Code Model Curriculum

```python
# 1. General language understanding
trainer.add_curriculum_stage("foundation", datasets=["pile"], epochs=5)

# 2. Code syntax and patterns
trainer.add_curriculum_stage("code_base", datasets=["the_stack"], epochs=5)

# 3. Code with comments/docstrings
trainer.add_curriculum_stage("code_docs", datasets=["code_search"], epochs=3)

# 4. Instruction following for code
trainer.add_curriculum_stage("code_instruct", datasets=["commitpackft"], epochs=3)

# 5. Tool use and agentic behavior
trainer.add_curriculum_stage("tools", datasets=["toolbench"], epochs=2, weight=2.0)
```

### Chat Model Curriculum

```python
# 1. General knowledge
trainer.add_curriculum_stage("knowledge", datasets=["wiki", "books"], epochs=5)

# 2. Conversational patterns
trainer.add_curriculum_stage("dialogue", datasets=["dailydialog"], epochs=3)

# 3. Instruction following
trainer.add_curriculum_stage("instruction", datasets=["dolly"], epochs=3)

# 4. Multi-turn conversations
trainer.add_curriculum_stage("multi_turn", datasets=["oasst"], epochs=3)

# 5. Preference alignment (RLHF/DPO data)
trainer.add_curriculum_stage("alignment", datasets=["hh_rlhf"], epochs=2, weight=2.0)
```

---

## Stage Transitions

### Callbacks for Stage Changes

```python
def on_stage_change(event):
    if event["event"] == "stage_complete":
        stage = event["stage"]
        metrics = event["metrics"]

        print(f"\n{'='*50}")
        print(f"Stage '{stage}' Complete")
        print(f"  Final loss: {metrics['final_loss']:.4f}")
        print(f"  Total steps: {metrics['steps']}")
        print(f"{'='*50}\n")

        # Optional: Save stage checkpoint
        trainer.save(f"./checkpoints/stage_{stage}")

trainer.add_callback(on_stage_change)
```

### Per-Stage Configuration

Override configuration for specific stages:

```python
# Lower LR for fine-tuning stage
trainer.add_curriculum_stage(
    "fine_tune",
    datasets=["task_specific"],
    learning_rate=1e-5,  # Override LR
    batch_size=4,        # Override batch size
)
```

---

## Best Practices

### 1. Start Simple

Begin with easy examples:
- Short sequences
- Common vocabulary
- Simple structure

### 2. Gradual Progression

Increase complexity gradually:
- Avoid large jumps between stages
- 3-5 stages typically sufficient

### 3. Monitor Convergence

Watch for signs of difficulty:
- Loss spikes at stage transitions
- Slow convergence on later stages
- Degradation on earlier capabilities

### 4. Use Validation

Validate on all capability levels:
```python
# Validate at each stage
for stage in ["simple", "medium", "hard"]:
    val_loss = evaluate(validation_sets[stage])
    print(f"{stage}: {val_loss:.4f}")
```

### 5. Save Checkpoints

Save at stage boundaries:
```python
trainer.train(
    checkpoint_dir="./checkpoints",
    checkpoint_interval=1000,
    save_checkpoints=True,
)
```

---

## Complete Example

```python
import highnoon as hn
from highnoon.training import AdaptiveCurriculumScheduler

# Create model
model = hn.create_model("highnoon-3b")

# Create trainer with adaptive scheduler
trainer = hn.Trainer(model, learning_rate=1e-4, batch_size=8)

# Define multi-stage curriculum
stages = [
    ("foundation", ["wiki_simple"], 5, 1.0),
    ("general", ["pile_subset"], 5, 1.0),
    ("domain", ["code_subset"], 4, 1.2),
    ("instruction", ["alpaca"], 3, 1.5),
    ("alignment", ["preference"], 2, 2.0),
]

for name, datasets, epochs, weight in stages:
    trainer.add_curriculum_stage(
        name,
        datasets=datasets,
        epochs=epochs,
        weight=weight,
    )

# Monitor progress
def monitor(event):
    if event["event"] == "stage_complete":
        print(f"✓ {event['stage']}: loss={event['metrics']['final_loss']:.4f}")

trainer.add_callback(monitor)

# Train with automatic curriculum progression
summary = trainer.train(
    checkpoint_dir="./checkpoints",
    epochs_per_stage=None,  # Use stage-defined epochs
)

print(f"Training complete: {len(summary['stages_completed'])} stages")
```

---

[← Training Guide](training.md) | [Agent Tools Guide →](agent-tools.md)
