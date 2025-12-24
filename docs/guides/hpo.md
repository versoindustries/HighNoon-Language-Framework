# Hyperparameter Optimization Guide

> HighNoon Language Framework - HPO with Optuna Integration

This guide covers hyperparameter optimization for training HighNoon models.

---

## Overview

HighNoon provides integrated HPO capabilities:

- **HPOTrialManager** - Manages optimization trials
- **Optuna integration** - Advanced sampling strategies
- **WebUI dashboard** - Visual trial management
- **Distributed HPO** - Scale across nodes

---

## Quick Start

```python
import highnoon as hn

# Create HPO manager
hpo = hn.HPOTrialManager(
    study_name="my_hpo_study",
    n_trials=50,
    direction="minimize",  # Minimize loss
)

# Define search space
search_space = {
    "learning_rate": ("log_uniform", 1e-5, 1e-3),
    "batch_size": ("categorical", [4, 8, 16, 32]),
    "embedding_dim": ("int", 256, 1024, 128),  # step=128
    "num_reasoning_blocks": ("int", 4, 12),
}

# Run HPO
best_params = hpo.optimize(
    objective=train_and_evaluate,
    search_space=search_space,
)

print(f"Best parameters: {best_params}")
```

---

## HPOTrialManager

### Initialization

```python
from highnoon import HPOTrialManager

hpo = HPOTrialManager(
    study_name="experiment_001",
    n_trials=100,
    direction="minimize",
    storage="sqlite:///hpo.db",  # Persistent storage
    pruner="median",
    sampler="tpe",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `study_name` | `str` | *required* | Unique study identifier |
| `n_trials` | `int` | `50` | Number of trials to run |
| `direction` | `str` | `"minimize"` | `"minimize"` or `"maximize"` |
| `storage` | `str` | `None` | Database URL for persistence |
| `pruner` | `str` | `"median"` | Early stopping strategy |
| `sampler` | `str` | `"tpe"` | Sampling algorithm |

### Samplers

| Sampler | Description | Best For |
|---------|-------------|----------|
| `tpe` | Tree-structured Parzen Estimator | Default, most cases |
| `cmaes` | Covariance Matrix Adaptation | Continuous params |
| `random` | Random sampling | Baseline comparison |
| `grid` | Grid search | Small spaces |

### Pruners

| Pruner | Description |
|--------|-------------|
| `median` | Prune if below median |
| `hyperband` | Successive halving |
| `percentile` | Prune bottom percentile |
| `none` | No early stopping |

---

## Defining Search Space

### Parameter Types

```python
search_space = {
    # Continuous (log scale)
    "learning_rate": ("log_uniform", 1e-5, 1e-3),

    # Continuous (linear scale)
    "dropout": ("uniform", 0.0, 0.5),

    # Integer
    "num_layers": ("int", 2, 12),

    # Integer with step
    "hidden_dim": ("int", 128, 1024, 64),  # step=64

    # Categorical
    "optimizer": ("categorical", ["adam", "sgd", "adamw"]),

    # Boolean (as categorical)
    "use_dropout": ("categorical", [True, False]),
}
```

### HSMN-Specific Parameters

```python
hsmn_search_space = {
    # Model architecture
    "embedding_dim": ("categorical", [512, 768, 1024, 2048]),
    "num_reasoning_blocks": ("int", 4, 16),
    "num_moe_experts": ("categorical", [4, 8, 12]),

    # Training
    "learning_rate": ("log_uniform", 1e-5, 1e-3),
    "batch_size": ("categorical", [4, 8, 16]),
    "warmup_steps": ("int", 500, 5000, 500),
    "gradient_clip": ("uniform", 0.5, 2.0),
    "weight_decay": ("log_uniform", 0.001, 0.1),

    # Mamba parameters
    "mamba_d_state": ("categorical", [16, 32, 64]),
    "mamba_d_conv": ("categorical", [3, 4, 5]),
    "mamba_expand": ("int", 2, 4),

    # MoE parameters
    "top_k_experts": ("categorical", [1, 2, 4]),
    "aux_loss_weight": ("log_uniform", 0.001, 0.1),

    # Regularization
    "dropout_rate": ("uniform", 0.0, 0.3),
}
```

---

## Objective Function

### Basic Structure

```python
def objective(trial_config):
    """Train and evaluate with given config.

    Args:
        trial_config: Dictionary of hyperparameters

    Returns:
        float: Metric to optimize (e.g., validation loss)
    """
    # Create model with trial config
    model = hn.create_model(config_from_trial(trial_config))

    # Train
    trainer = hn.Trainer(
        model,
        learning_rate=trial_config["learning_rate"],
        batch_size=trial_config["batch_size"],
    )

    trainer.add_curriculum_stage("train", datasets=["train_data"])
    summary = trainer.train(epochs_per_stage=5)

    # Evaluate
    val_loss = evaluate(model, val_dataset)

    return val_loss
```

### With Early Stopping

```python
def objective(trial_config):
    model = hn.create_model(config_from_trial(trial_config))

    best_val_loss = float("inf")
    patience = 3
    no_improve = 0

    for epoch in range(20):
        train_loss = train_epoch(model)
        val_loss = evaluate(model)

        # Report for pruning
        hpo.report(val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        # Check if pruned
        if hpo.should_prune():
            raise hpo.TrialPruned()

    return best_val_loss
```

---

## Running HPO

### Basic Run

```python
best_params = hpo.optimize(
    objective=objective,
    search_space=search_space,
)
```

### With Callbacks

```python
def on_trial_complete(study, trial):
    print(f"Trial {trial.number}: {trial.value:.4f}")
    print(f"  Params: {trial.params}")

best_params = hpo.optimize(
    objective=objective,
    search_space=search_space,
    callbacks=[on_trial_complete],
)
```

### Resume Interrupted Study

```python
hpo = HPOTrialManager(
    study_name="experiment_001",
    storage="sqlite:///hpo.db",
    load_if_exists=True,  # Resume existing study
)

best_params = hpo.optimize(
    objective=objective,
    search_space=search_space,
    n_trials=50,  # Additional trials
)
```

---

## Results Analysis

### View Best Parameters

```python
print(f"Best value: {hpo.best_value}")
print(f"Best params: {hpo.best_params}")
```

### Trial History

```python
for trial in hpo.trials:
    print(f"Trial {trial.number}:")
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    print(f"  State: {trial.state}")
```

### Parameter Importance

```python
importance = hpo.get_param_importance()
for param, score in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"  {param}: {score:.3f}")
```

---

## WebUI Dashboard

### Starting the Dashboard

```python
from highnoon.webui import app

# Start WebUI server
app.run(host="0.0.0.0", port=8000)
```

Access at `http://localhost:8000/hpo`

### Dashboard Features

| Feature | Description |
|---------|-------------|
| **Study Browser** | View all HPO studies |
| **Trial List** | Browse individual trials |
| **Parameter Plots** | Visualize parameter effects |
| **Optimization History** | Track convergence |
| **Best Config Export** | Download best parameters |

---

## Distributed HPO

### Multi-Worker Setup

```python
# Worker 1
hpo = HPOTrialManager(
    study_name="distributed_study",
    storage="postgresql://user:pass@db:5432/optuna",
)
hpo.optimize(objective, search_space, n_trials=25)

# Worker 2 (same study)
hpo = HPOTrialManager(
    study_name="distributed_study",
    storage="postgresql://user:pass@db:5432/optuna",
)
hpo.optimize(objective, search_space, n_trials=25)
```

### Database Requirements

| Database | Storage URL |
|----------|-------------|
| SQLite | `sqlite:///hpo.db` |
| PostgreSQL | `postgresql://user:pass@host/db` |
| MySQL | `mysql://user:pass@host/db` |

---

## Complete Example

```python
import highnoon as hn
from highnoon.config import Config, ModelConfig

def create_config_from_trial(params):
    """Create model config from trial parameters."""
    return Config(
        model=ModelConfig(
            vocab_size=50000,
            embedding_dim=params["embedding_dim"],
            num_reasoning_blocks=params["num_blocks"],
            num_moe_experts=params["num_experts"],
        )
    )

def objective(params):
    """Train and evaluate with given hyperparameters."""
    # Create model
    config = create_config_from_trial(params)
    model = hn.create_model(config)

    # Setup training
    trainer = hn.Trainer(
        model,
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
    )

    trainer.add_curriculum_stage(
        "foundation",
        datasets=["train_subset"],
        epochs=3,
    )

    # Train
    summary = trainer.train(
        save_checkpoints=False,
        log_interval=500,
    )

    # Get final validation loss
    val_loss = summary["stages_completed"][-1]["final_loss"]

    return val_loss

# Define search space
search_space = {
    "learning_rate": ("log_uniform", 1e-5, 1e-3),
    "batch_size": ("categorical", [4, 8, 16]),
    "embedding_dim": ("categorical", [512, 768, 1024]),
    "num_blocks": ("int", 4, 12),
    "num_experts": ("categorical", [4, 8]),
}

# Run HPO
hpo = hn.HPOTrialManager(
    study_name="highnoon_hpo",
    n_trials=30,
    direction="minimize",
    storage="sqlite:///hpo_study.db",
    sampler="tpe",
    pruner="median",
)

best_params = hpo.optimize(
    objective=objective,
    search_space=search_space,
)

# Results
print("\n" + "="*50)
print("HPO Complete!")
print(f"Best validation loss: {hpo.best_value:.4f}")
print(f"Best parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# Train final model with best params
print("\nTraining final model with best parameters...")
final_model = hn.create_model(create_config_from_trial(best_params))
final_trainer = hn.Trainer(final_model, **best_params)
final_trainer.train(checkpoint_dir="./best_model")
```

---

## Best Practices

### 1. Start with Large Ranges

```python
# Good - wide initial search
"learning_rate": ("log_uniform", 1e-6, 1e-2)

# Then narrow down
"learning_rate": ("log_uniform", 1e-5, 1e-4)
```

### 2. Use Appropriate Scales

```python
# Log scale for learning rates
"learning_rate": ("log_uniform", ...)

# Linear for bounded params
"dropout": ("uniform", 0.0, 0.5)
```

### 3. Enable Pruning

```python
hpo = HPOTrialManager(
    pruner="median",  # Stop unpromising trials early
)
```

### 4. Persist Results

```python
hpo = HPOTrialManager(
    storage="sqlite:///hpo.db",  # Save results
)
```

### 5. Use Sufficient Trials

| Space Size | Recommended Trials |
|------------|-------------------|
| Small (<10 params) | 50-100 |
| Medium (10-20 params) | 100-200 |
| Large (>20 params) | 200+ |

---

[← Agent Tools](agent-tools.md) | [WebUI Guide →](webui.md)
