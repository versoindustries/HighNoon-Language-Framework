# WebUI Guide

> HighNoon Language Framework - Curriculum Builder and Training Console

This guide covers the HighNoon WebUI for managing training, curriculum building, and HPO.

---

## Overview

The HighNoon WebUI provides:

- **Curriculum Builder** - Visual curriculum stage management
- **Training Console** - Real-time training monitoring
- **HPO Dashboard** - Hyperparameter optimization management
- **Model Browser** - View and manage trained models
- **Attribution Settings** - Customize attribution (Pro/Enterprise)

---

## Starting the WebUI

### Basic Start

```python
from highnoon.webui import app

# Start on default port
app.run(host="0.0.0.0", port=8000)
```

### Command Line

```bash
python -m highnoon.webui --host 0.0.0.0 --port 8000
```

### With Configuration

```python
from highnoon.webui import app

app.run(
    host="0.0.0.0",
    port=8000,
    debug=False,
    workers=4,
)
```

Access the WebUI at `http://localhost:8000`

---

## Curriculum Builder

The curriculum builder provides a visual interface for designing training curricula.

### Creating a Curriculum

1. Navigate to **Curriculum Builder** in the sidebar
2. Click **New Curriculum**
3. Add stages using the **+ Add Stage** button

### Stage Configuration

For each stage, configure:

| Field | Description |
|-------|-------------|
| **Name** | Stage identifier (e.g., "foundation") |
| **Datasets** | Select or add datasets |
| **Epochs** | Number of training epochs |
| **Weight** | Importance multiplier (1.0 = normal) |
| **Learning Rate** | Optional per-stage override |

### Drag-and-Drop Ordering

Reorder stages by dragging them in the stage list. Training progresses through stages in order.

### Saving Curricula

Click **Save Curriculum** to save for later use. Curricula are stored in the database and can be:

- Exported as JSON
- Imported from JSON
- Cloned and modified

### Example Curriculum

```
┌─────────────────────────────────────┐
│ Stage 1: Foundation                 │
│ Datasets: wiki_simple, books_easy   │
│ Epochs: 5                           │
│ Weight: 1.0                         │
├─────────────────────────────────────┤
│ Stage 2: Domain Text                │
│ Datasets: arxiv_papers              │
│ Epochs: 4                           │
│ Weight: 1.2                         │
├─────────────────────────────────────┤
│ Stage 3: Instructions               │
│ Datasets: alpaca, dolly             │
│ Epochs: 3                           │
│ Weight: 1.5                         │
├─────────────────────────────────────┤
│ Stage 4: Alignment                  │
│ Datasets: preference_data           │
│ Epochs: 2                           │
│ Weight: 2.0                         │
└─────────────────────────────────────┘
```

---

## Training Console

Monitor and control training runs in real-time.

### Starting Training

1. Select a curriculum from the dropdown
2. Choose a model preset or configuration
3. Set training parameters:
   - Checkpoint directory
   - Checkpoint interval
   - Logging interval
4. Click **Start Training**

### Real-Time Metrics

The console displays:

| Metric | Description |
|--------|-------------|
| **Current Stage** | Active curriculum stage |
| **Epoch Progress** | Current/total epochs |
| **Step Progress** | Current training step |
| **Loss** | Training loss (chart) |
| **Learning Rate** | Current LR (chart) |
| **ETA** | Estimated time remaining |

### Training Controls

| Control | Action |
|---------|--------|
| **Pause** | Pause training (saves state) |
| **Resume** | Resume paused training |
| **Stop** | Stop and save checkpoint |
| **Cancel** | Stop without saving |

### Log Viewer

View real-time training logs with filtering:

- **Info** - Normal progress messages
- **Warning** - Non-critical issues
- **Error** - Errors and exceptions

---

## HPO Dashboard

Manage hyperparameter optimization studies.

### Creating a Study

1. Navigate to **HPO Dashboard**
2. Click **New Study**
3. Configure:
   - Study name
   - Objective (minimize/maximize)
   - Number of trials
   - Sampler type
   - Pruner type

### Defining Search Space

Use the visual search space builder:

```
┌─────────────────────────────────────┐
│ Parameter: learning_rate            │
│ Type: Log Uniform                   │
│ Min: 1e-5       Max: 1e-3           │
├─────────────────────────────────────┤
│ Parameter: batch_size               │
│ Type: Categorical                   │
│ Values: [4, 8, 16, 32]              │
├─────────────────────────────────────┤
│ Parameter: num_blocks               │
│ Type: Integer                       │
│ Min: 4          Max: 12             │
└─────────────────────────────────────┘
```

### Running Trials

- **Start Study** - Begin optimization
- **Pause Study** - Pause after current trial
- **Resume Study** - Continue paused study

### Viewing Results

| View | Description |
|------|-------------|
| **Trial List** | All trials with metrics |
| **Best Parameters** | Optimal configuration |
| **Optimization History** | Convergence plot |
| **Parameter Importance** | Feature importance |
| **Parallel Coordinates** | Multi-param visualization |

### Exporting Results

- **Export Best Config** - JSON configuration
- **Export All Trials** - CSV of all trials
- **Export Charts** - PNG/SVG visualizations

---

## Model Browser

View and manage trained models.

### Model List

Browse all saved models with:

- Model name and path
- Training date
- Configuration summary
- Checkpoint status

### Model Details

View detailed information:

- Full configuration
- Training history
- Curriculum stages completed
- Validation metrics

### Model Actions

| Action | Description |
|--------|-------------|
| **Load** | Load model for inference |
| **Clone** | Create copy with new name |
| **Export** | Download model files |
| **Delete** | Remove model |

---

## Attribution Settings

> **Note**: Attribution customization requires Pro or Enterprise edition.

### Viewing Attribution

All Lite models display "Powered by HSMN" attribution.

### Customizing Attribution (Pro/Enterprise)

1. Navigate to **Settings > Attribution**
2. Configure:
   - Display text
   - Logo (if allowed by license)
   - Visibility settings

```python
# API equivalent
from highnoon import attribution

attribution.set_text("Powered by MyCompany")
attribution.set_visible(True)  # or False for Enterprise
```

---

## API Access

The WebUI exposes a REST API for programmatic access.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/curricula` | GET | List curricula |
| `/api/curricula` | POST | Create curriculum |
| `/api/training/start` | POST | Start training |
| `/api/training/status` | GET | Training status |
| `/api/hpo/studies` | GET | List studies |
| `/api/hpo/studies/{id}` | GET | Study details |

### Example: Start Training via API

```python
import requests

response = requests.post(
    "http://localhost:8000/api/training/start",
    json={
        "curriculum_id": "my_curriculum",
        "model_preset": "highnoon-3b",
        "checkpoint_dir": "./checkpoints",
    }
)

job_id = response.json()["job_id"]
```

### Example: Check Training Status

```python
response = requests.get(
    f"http://localhost:8000/api/training/status/{job_id}"
)

status = response.json()
print(f"Stage: {status['current_stage']}")
print(f"Progress: {status['progress']}%")
print(f"Loss: {status['current_loss']}")
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HIGHNOON_WEBUI_HOST` | `0.0.0.0` | Server host |
| `HIGHNOON_WEBUI_PORT` | `8000` | Server port |
| `HIGHNOON_WEBUI_DEBUG` | `false` | Debug mode |
| `HIGHNOON_WEBUI_WORKERS` | `4` | Worker processes |
| `HIGHNOON_DB_URL` | `sqlite:///highnoon.db` | Database URL |

### Configuration File

Create `webui_config.yaml`:

```yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  debug: false

database:
  url: sqlite:///highnoon.db

auth:
  enabled: false
  # secret_key: your-secret-key

storage:
  checkpoints: ./checkpoints
  curricula: ./curricula
  hpo_studies: ./hpo
```

Load with:

```python
from highnoon.webui import app

app.run(config="webui_config.yaml")
```

---

## Complete Example

### Launch WebUI and Train

```python
#!/usr/bin/env python3
"""Launch WebUI and run training example."""

import threading
import time
import requests
from highnoon.webui import app

def start_webui():
    app.run(host="0.0.0.0", port=8000)

# Start WebUI in background
webui_thread = threading.Thread(target=start_webui, daemon=True)
webui_thread.start()
time.sleep(2)  # Wait for startup

# Create curriculum via API
curriculum = {
    "name": "api_curriculum",
    "stages": [
        {"name": "foundation", "datasets": ["wiki"], "epochs": 3},
        {"name": "instruction", "datasets": ["alpaca"], "epochs": 2},
    ]
}

response = requests.post(
    "http://localhost:8000/api/curricula",
    json=curriculum
)
curriculum_id = response.json()["id"]

# Start training
response = requests.post(
    "http://localhost:8000/api/training/start",
    json={
        "curriculum_id": curriculum_id,
        "model_preset": "highnoon-small",
        "checkpoint_dir": "./checkpoints",
    }
)
job_id = response.json()["job_id"]

# Monitor progress
while True:
    status = requests.get(
        f"http://localhost:8000/api/training/status/{job_id}"
    ).json()

    print(f"Stage: {status['current_stage']}, "
          f"Progress: {status['progress']:.1f}%, "
          f"Loss: {status.get('current_loss', 'N/A')}")

    if status["state"] in ("completed", "failed"):
        break

    time.sleep(10)

print(f"Training {status['state']}!")
```

---

[← HPO Guide](hpo.md) | [Back to Index →](../index.md)
