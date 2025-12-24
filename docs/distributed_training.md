# Distributed Training Guide

> **CPU-Only Multi-Node Training** | HighNoon Language Framework

This guide covers setting up distributed training across multiple CPU nodes for HighNoon models.

---

## Overview

HighNoon is a **CPU-first architecture** (32-bit, no GPU/quantization). For large-scale training, the framework supports:

| Strategy | Best For | Scaling |
|----------|----------|---------|
| `MultiWorkerMirroredStrategy` | Synchronous data parallelism | 2-16 workers |
| `ParameterServerStrategy` | Large clusters, async training | 16+ workers |
| `CentralStorageStrategy` | Single-node multi-CPU | 1 node |

---

## Quick Start

### 1. Install HighNoon on All Nodes

```bash
# On each node
pip install highnoon
# Or from source
git clone https://github.com/versoindustries/highnoon.git
pip install -e highnoon/
```

### 2. Configure TF_CONFIG

Each worker needs a `TF_CONFIG` environment variable:

```bash
# Worker 0 (chief)
export TF_CONFIG='{
  "cluster": {
    "worker": ["node1:12345", "node2:12345", "node3:12345"]
  },
  "task": {"type": "worker", "index": 0}
}'

# Worker 1
export TF_CONFIG='{
  "cluster": {
    "worker": ["node1:12345", "node2:12345", "node3:12345"]
  },
  "task": {"type": "worker", "index": 1}
}'

# Worker 2
export TF_CONFIG='{
  "cluster": {
    "worker": ["node1:12345", "node2:12345", "node3:12345"]
  },
  "task": {"type": "worker", "index": 2}
}'
```

### 3. Training Script

```python
import highnoon as hn
from highnoon.training.distributed import create_cpu_strategy

# Auto-detects TF_CONFIG and creates appropriate strategy
strategy = create_cpu_strategy()

print(f"Number of replicas: {strategy.num_replicas_in_sync}")

with strategy.scope():
    # Create model inside strategy scope
    model = hn.create_model("7b")

    # Dataset is automatically sharded across workers
    trainer = hn.Trainer(model, learning_rate=1e-4)
    trainer.add_curriculum_stage("foundation", datasets=["your_dataset"])
    trainer.train(epochs_per_stage=10, checkpoint_dir="/shared/checkpoints")
```

### 4. Launch on All Nodes

```bash
# Run on each node simultaneously
python train_distributed.py
```

Or use the launcher script:

```bash
./scripts/launch_distributed.sh --hosts "node1,node2,node3" --script train.py
```

---

## Strategy Selection

### MultiWorkerMirroredStrategy (Recommended)

Best for most use cases. Synchronous training with all-reduce gradients.

```python
from highnoon.training.distributed import create_cpu_strategy

strategy = create_cpu_strategy(
    strategy_type="multi_worker",
    communication="ring"  # Options: ring, nccl, auto
)
```

**Pros:**
- Simple setup
- Consistent training dynamics
- Linear scaling efficiency up to ~16 workers

**Cons:**
- Slowest worker bottlenecks training
- Synchronization overhead on high-latency networks

---

### ParameterServerStrategy

For large-scale async training with dedicated parameter servers.

```python
strategy = create_cpu_strategy(
    strategy_type="parameter_server",
    num_ps=2,  # Number of parameter servers
    variable_partitioner="min_max"  # Partition large variables
)
```

**TF_CONFIG for PS Strategy:**

```bash
export TF_CONFIG='{
  "cluster": {
    "worker": ["worker0:12345", "worker1:12345"],
    "ps": ["ps0:12346", "ps1:12346"],
    "chief": ["chief0:12347"]
  },
  "task": {"type": "worker", "index": 0}
}'
```

---

## CPU Performance Tuning

### Thread Configuration

```python
import os

# Set before importing TensorFlow
os.environ["OMP_NUM_THREADS"] = "16"  # Match physical cores
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "16"

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(16)
```

### Memory Optimization

```python
# Enable memory growth to avoid pre-allocation
os.environ["TF_FORCE_CPU_ALLOW_GROWTH"] = "true"

# Use XLA for CPU
tf.config.optimizer.set_jit(True)
```

### Batch Size Scaling

Scale batch size linearly with worker count:

```python
base_batch_size = 8
global_batch_size = base_batch_size * strategy.num_replicas_in_sync

# Learning rate scaling (linear or sqrt)
base_lr = 1e-4
scaled_lr = base_lr * strategy.num_replicas_in_sync  # Linear
# OR
scaled_lr = base_lr * math.sqrt(strategy.num_replicas_in_sync)  # Conservative
```

---

## Checkpointing

### Shared Filesystem (Required)

All nodes must access the same checkpoint directory via NFS, Lustre, or similar:

```python
checkpoint_dir = "/shared/nfs/checkpoints"  # All nodes can read/write

trainer.train(
    checkpoint_dir=checkpoint_dir,
    checkpoint_interval=1000,
    save_checkpoints=True
)
```

### Fault Tolerance

HighNoon automatically handles:
- Worker failure detection
- Checkpoint restoration on recovery
- Step synchronization after restart

```python
# Resume from latest checkpoint
trainer.train(
    resume_from="/shared/checkpoints/latest",
    checkpoint_dir="/shared/checkpoints"
)
```

---

## Network Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Bandwidth | 1 Gbps | 10+ Gbps |
| Latency | < 10ms | < 1ms |
| Ports | 1 per worker | 1 per worker |
| Protocol | TCP | TCP or RDMA |

### Firewall Configuration

Open the training port on all nodes:

```bash
# On each node
sudo ufw allow 12345/tcp
# Or for a range
sudo ufw allow 12345:12400/tcp
```

---

## Troubleshooting

### "Connection refused" errors

```bash
# Check if port is open
nc -zv node1 12345

# Check firewall
sudo iptables -L -n | grep 12345
```

### Workers not synchronizing

```bash
# Verify TF_CONFIG is correct on all nodes
echo $TF_CONFIG | python -m json.tool

# Check all workers are reachable
for host in node1 node2 node3; do
  ssh $host "hostname && echo TF_CONFIG set: \$TF_CONFIG"
done
```

### Out of memory

```bash
# Reduce batch size
trainer = hn.Trainer(model, batch_size=4)  # Smaller per-replica batch

# Monitor memory
watch -n 1 'free -h'
```

### Slow training

```python
# Profile to find bottlenecks
import tensorflow as tf

tf.profiler.experimental.start('/tmp/logdir')
# ... training steps ...
tf.profiler.experimental.stop()

# View with: tensorboard --logdir=/tmp/logdir
```

---

## Example: 4-Node Training

Complete example for a 4-node CPU cluster:

```bash
#!/bin/bash
# launch_4node.sh

HOSTS=("node1" "node2" "node3" "node4")
PORT=12345
WORKERS=$(IFS=,; echo "${HOSTS[*]/%/:$PORT}" | tr ',' '","')

for i in "${!HOSTS[@]}"; do
  ssh ${HOSTS[$i]} "
    export TF_CONFIG='{\"cluster\":{\"worker\":[\"$WORKERS\"]},\"task\":{\"type\":\"worker\",\"index\":$i}}'
    cd /path/to/project
    python train.py
  " &
done

wait
echo "All workers finished"
```

---

## See Also

- [Cluster Setup Guide](cluster_setup.md) - Infrastructure configuration
- [Training Guide](guides/training.md) - Training workflow
- [HPO Guide](guides/hpo.md) - Hyperparameter optimization
- [Enterprise Upgrade](enterprise-upgrade.md) - Unlimited scale options

---

*For production deployments requiring dedicated support, see [Enterprise Edition](enterprise-upgrade.md)*
