# Training Workflow Guide

> HighNoon Language Framework - Complete Training Guide

This guide walks through the complete process of training a HighNoon language model from scratch.

---

## Overview

Training a HighNoon model involves:

1. **Data Preparation** - Prepare and format your training data
2. **Model Configuration** - Configure model architecture
3. **Training Setup** - Set up the trainer with curriculum stages
4. **Training Execution** - Run training with monitoring
5. **Evaluation** - Evaluate model quality

---

## Step 1: Data Preparation

### Dataset Format

Training data should be in one of these formats:

**Plain Text Files**
```
One text sample per file or line.
Larger context is better for language modeling.
```

**JSONL Format**
```json
{"text": "First training sample..."}
{"text": "Second training sample..."}
{"text": "Third training sample..."}
```

**HuggingFace Datasets**
```python
from datasets import load_dataset

dataset = load_dataset("your_dataset_name")
```

### Data Quality Considerations

| Aspect | Recommendation |
|--------|----------------|
| **Size** | Minimum 1GB text for small models, 10GB+ for larger |
| **Diversity** | Mix of domains improves generalization |
| **Quality** | Filter low-quality samples (spam, duplicates) |
| **Length** | Include variety of lengths; match target use case |

### Creating a Dataset

```python
from highnoon.data import TextDataset

# From files
dataset = TextDataset.from_files(
    paths=["data/*.txt"],
    tokenizer=tokenizer,
    max_length=2048,
)

# From HuggingFace
dataset = TextDataset.from_hf(
    name="pile",
    split="train",
    tokenizer=tokenizer,
    max_length=2048,
)
```

---

## Step 2: Model Configuration

### Choosing Model Size

| Size | Use Case | Training Requirements |
|------|----------|----------------------|
| `highnoon-small` | Experimentation, debugging | 16GB RAM, hours |
| `highnoon-base` | Small-scale deployments | 32GB RAM, days |
| `highnoon-3b` | Production quality | 64GB RAM, weeks |
| `highnoon-7b+` | High-quality generation | 128GB+ RAM, weeks-months |

### Custom Configuration

```python
import highnoon as hn
from highnoon.config import Config, ModelConfig, TrainingConfig

config = Config(
    model=ModelConfig(
        vocab_size=50000,
        embedding_dim=1024,
        num_reasoning_blocks=12,
        max_seq_length=4096,
        num_moe_experts=8,
    ),
    training=TrainingConfig(
        learning_rate=1e-4,
        batch_size=8,
        warmup_steps=1000,
        gradient_clip=1.0,
    ),
)

model = hn.create_model(config)
```

---

## Step 3: Training Setup

### Basic Trainer Setup

```python
import highnoon as hn

model = hn.create_model("highnoon-3b")

trainer = hn.Trainer(
    model,
    learning_rate=1e-4,
    batch_size=8,
    max_seq_length=2048,
    gradient_accumulation_steps=4,  # Effective batch = 32
)
```

### Adding Curriculum Stages

Curriculum learning improves training by progressively introducing harder examples:

```python
# Stage 1: General text
trainer.add_curriculum_stage(
    "foundation",
    datasets=["pile_subset"],
    epochs=5,
)

# Stage 2: Domain-specific
trainer.add_curriculum_stage(
    "code",
    datasets=["the_stack_v2"],
    epochs=3,
)

# Stage 3: Task-specific
trainer.add_curriculum_stage(
    "instruction",
    datasets=["open_assistant"],
    epochs=2,
    weight=2.0,  # Higher importance
)
```

### Adding Callbacks

Monitor training progress with callbacks:

```python
def training_monitor(event):
    event_type = event.get("event")

    if event_type == "epoch_end":
        print(f"[{event['stage']}] Epoch {event['epoch']}: "
              f"loss={event['loss']:.4f}")

    elif event_type == "stage_complete":
        print(f"Stage '{event['stage']}' complete!")
        print(f"  Final loss: {event['metrics']['final_loss']:.4f}")

    elif event_type == "checkpoint_saved":
        print(f"Checkpoint saved: {event['path']}")

trainer.add_callback(training_monitor)
```

---

## Step 4: Training Execution

### Running Training

```python
summary = trainer.train(
    epochs_per_stage=5,           # Override: epochs per stage
    save_checkpoints=True,         # Enable checkpointing
    checkpoint_dir="./checkpoints",
    checkpoint_interval=1000,      # Steps between checkpoints
    log_interval=100,              # Steps between logs
    eval_interval=500,             # Steps between evaluation
)
```

### Resuming Training

If training is interrupted:

```python
summary = trainer.train(
    resume_from="./checkpoints/latest",
    checkpoint_dir="./checkpoints",
)
```

### Training Output

```python
print(f"Total steps: {summary['total_steps']}")
print(f"Stages completed: {len(summary['stages_completed'])}")

for stage in summary['stages_completed']:
    print(f"  {stage['name']}: {stage['epochs']} epochs, "
          f"final_loss={stage['final_loss']:.4f}")
```

---

## Step 5: Monitoring Training

### TensorBoard Integration

```python
import tensorflow as tf

# Enable TensorBoard logging
log_dir = "./logs/training"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# View with: tensorboard --logdir ./logs/training
```

### Key Metrics to Watch

| Metric | Healthy Range | Action if Outside |
|--------|---------------|-------------------|
| **Loss** | Decreasing | Check LR, data quality |
| **Gradient Norm** | < 10 | Increase clip, check data |
| **Learning Rate** | Scheduled | Check warmup config |
| **GPU/CPU Util** | > 80% | Check data pipeline |

### Early Stopping

The trainer includes automatic early stopping:

- Monitors validation loss
- Stops after `EARLY_STOPPING_PATIENCE` epochs without improvement
- Minimum improvement threshold: `EARLY_STOPPING_MIN_DELTA`

---

## Step 6: Saving and Loading

### Save Final Model

```python
# Save trainer state (includes curriculum progress)
trainer.save("./final_model")

# Save just the model weights
model.save_weights("./final_model/weights")
```

### Load for Inference

```python
import highnoon as hn

# Load from trainer save
trainer = hn.Trainer.load("./final_model", model)

# Or just load weights
model = hn.create_model("highnoon-3b")
model.load_weights("./final_model/weights")
```

---

## Complete Training Script

```python
#!/usr/bin/env python3
"""Complete training script for HighNoon."""

import highnoon as hn
from highnoon.config import Config, ModelConfig, TrainingConfig

def main():
    # Configuration
    config = Config(
        model=ModelConfig(
            vocab_size=50000,
            embedding_dim=1024,
            num_reasoning_blocks=8,
        ),
        training=TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            warmup_steps=1000,
        ),
    )

    # Create model
    print("Creating model...")
    model = hn.create_model(config)
    print(f"Parameters: {model.count_params():,}")

    # Setup trainer
    trainer = hn.Trainer(
        model,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        gradient_accumulation_steps=4,
    )

    # Define curriculum
    trainer.add_curriculum_stage("foundation", datasets=["pile"], epochs=5)
    trainer.add_curriculum_stage("code", datasets=["the_stack"], epochs=3)
    trainer.add_curriculum_stage("instruction", datasets=["oasst"], epochs=2)

    # Monitoring callback
    def monitor(event):
        if event["event"] == "epoch_end":
            print(f"[{event['stage']}] Epoch {event['epoch']}: "
                  f"loss={event['loss']:.4f}")

    trainer.add_callback(monitor)

    # Train
    print("Starting training...")
    summary = trainer.train(
        checkpoint_dir="./checkpoints",
        checkpoint_interval=500,
    )

    # Save
    print("Saving model...")
    trainer.save("./trained_model")

    print(f"Training complete! Total steps: {summary['total_steps']}")

if __name__ == "__main__":
    main()
```

---

## Next Steps

- [Curriculum Learning Guide](curriculum-learning.md) - Advanced curriculum strategies
- [Distributed Training](../distributed_training.md) - Multi-node training
- [HPO Guide](hpo.md) - Hyperparameter optimization

---

[← Tokenization API](../api/tokenization.md) | [Curriculum Learning →](curriculum-learning.md)
