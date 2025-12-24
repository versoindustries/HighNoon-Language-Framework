# Cluster Setup Guide

> **Infrastructure Configuration for Multi-Node Training** | HighNoon Language Framework

This guide covers setting up CPU cluster infrastructure for distributed HighNoon training.

---

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| **OS** | Ubuntu 22.04+ / RHEL 8+ / Debian 11+ |
| **Python** | 3.10+ |
| **Memory** | 32GB+ RAM per node (64GB recommended) |
| **Network** | 1 Gbps minimum (10 Gbps recommended) |
| **Storage** | Shared filesystem accessible from all nodes |

---

## Network Configuration

### Port Requirements

```
Training Port: 12345 (configurable)
SSH: 22
NFS: 2049, 111
```

### Open Firewall on All Nodes

```bash
# Ubuntu/Debian
sudo ufw allow 12345/tcp
sudo ufw allow from 10.0.0.0/8 to any port 12345

# RHEL/CentOS
sudo firewall-cmd --permanent --add-port=12345/tcp
sudo firewall-cmd --reload
```

### Verify Connectivity

```bash
# From any node, test all others
for host in node1 node2 node3 node4; do
  echo -n "$host: "
  nc -zv $host 12345 2>&1 | grep -o "succeeded\|failed"
done
```

---

## Shared Filesystem

### Option 1: NFS (Simplest)

**On NFS Server:**

```bash
sudo apt install nfs-kernel-server
sudo mkdir -p /exports/highnoon

# Add to /etc/exports
echo "/exports/highnoon *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports

sudo exportfs -ra
sudo systemctl enable --now nfs-server
```

**On All Worker Nodes:**

```bash
sudo apt install nfs-common
sudo mkdir -p /shared/highnoon
echo "nfs-server:/exports/highnoon /shared/highnoon nfs defaults,_netdev 0 0" | sudo tee -a /etc/fstab
sudo mount -a
```

### Option 2: Lustre (High Performance)

For HPC clusters, use existing Lustre mounts:

```bash
# Typically mounted by cluster admin
mount | grep lustre
# /mnt/lustre type lustre ...

# Use as checkpoint directory
export CHECKPOINT_DIR=/mnt/lustre/users/$USER/highnoon/checkpoints
```

---

## SSH Configuration

### Passwordless SSH Between Nodes

```bash
# On primary node, generate key if needed
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# Copy to all nodes
for host in node1 node2 node3 node4; do
  ssh-copy-id -i ~/.ssh/id_ed25519.pub $host
done

# Verify
for host in node1 node2 node3 node4; do
  ssh $host hostname
done
```

---

## Environment Setup

### Install HighNoon on All Nodes

```bash
# Create identical environment on all nodes
for host in node1 node2 node3 node4; do
  ssh $host "
    cd /shared/highnoon
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -e .
  "
done
```

### CPU Optimization

Add to `~/.bashrc` on all nodes:

```bash
# CPU threading for TensorFlow
export OMP_NUM_THREADS=16          # Physical cores per node
export TF_NUM_INTEROP_THREADS=2
export TF_NUM_INTRAOP_THREADS=16
export TF_ENABLE_ONEDNN_OPTS=1     # Intel optimizations

# Memory
export TF_FORCE_CPU_ALLOW_GROWTH=true
```

---

## SLURM Integration

### Job Script

```bash
#!/bin/bash
#SBATCH --job-name=highnoon-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.log

# Get node list
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
PORT=12345

# Build worker list
WORKERS=""
for node in "${NODES[@]}"; do
  WORKERS="${WORKERS}\"${node}:${PORT}\","
done
WORKERS="[${WORKERS%,}]"

# Launch workers
for i in "${!NODES[@]}"; do
  srun --nodes=1 --ntasks=1 -w ${NODES[$i]} bash -c "
    export TF_CONFIG='{\"cluster\":{\"worker\":${WORKERS}},\"task\":{\"type\":\"worker\",\"index\":$i}}'
    source /shared/highnoon/venv/bin/activate
    python /shared/highnoon/train.py
  " &
done

wait
```

### Submit Job

```bash
sbatch train_job.slurm
squeue -u $USER  # Monitor
```

---

## Kubernetes Deployment

### ConfigMap for TF_CONFIG

```yaml
# tf-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: highnoon-cluster
data:
  cluster.json: |
    {
      "cluster": {
        "worker": [
          "highnoon-worker-0.highnoon:12345",
          "highnoon-worker-1.highnoon:12345",
          "highnoon-worker-2.highnoon:12345",
          "highnoon-worker-3.highnoon:12345"
        ]
      }
    }
```

### StatefulSet

```yaml
# workers.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: highnoon-worker
spec:
  serviceName: highnoon
  replicas: 4
  selector:
    matchLabels:
      app: highnoon
  template:
    metadata:
      labels:
        app: highnoon
    spec:
      containers:
      - name: worker
        image: versoindustries/highnoon:latest
        resources:
          requests:
            cpu: "16"
            memory: "64Gi"
          limits:
            cpu: "32"
            memory: "128Gi"
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: TF_CONFIG
          value: |
            {
              "cluster": {"worker": ["highnoon-worker-0.highnoon:12345", "highnoon-worker-1.highnoon:12345", "highnoon-worker-2.highnoon:12345", "highnoon-worker-3.highnoon:12345"]},
              "task": {"type": "worker", "index": $(echo $POD_NAME | grep -o '[0-9]*$')}
            }
        volumeMounts:
        - name: checkpoints
          mountPath: /shared/checkpoints
  volumeClaimTemplates:
  - metadata:
      name: checkpoints
    spec:
      accessModes: ["ReadWriteMany"]
      resources:
        requests:
          storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: highnoon
spec:
  clusterIP: None
  selector:
    app: highnoon
  ports:
  - port: 12345
```

---

## Validation Script

Run this to verify cluster is ready:

```bash
#!/bin/bash
# validate_cluster.sh

NODES=("node1" "node2" "node3" "node4")
PORT=12345
ERRORS=0

echo "=== HighNoon Cluster Validation ==="

# Check SSH
echo -e "\n[1/5] SSH Connectivity"
for node in "${NODES[@]}"; do
  if ssh -o ConnectTimeout=5 $node "exit" 2>/dev/null; then
    echo "  ✓ $node"
  else
    echo "  ✗ $node - SSH failed"
    ((ERRORS++))
  fi
done

# Check port
echo -e "\n[2/5] Port $PORT"
for node in "${NODES[@]}"; do
  if nc -z -w5 $node $PORT 2>/dev/null; then
    echo "  ✓ $node:$PORT open"
  else
    echo "  ⚠ $node:$PORT closed (will open during training)"
  fi
done

# Check shared filesystem
echo -e "\n[3/5] Shared Filesystem"
TESTFILE="/shared/highnoon/.cluster_test_$$"
echo "test" > $TESTFILE 2>/dev/null
for node in "${NODES[@]}"; do
  if ssh $node "cat $TESTFILE" 2>/dev/null | grep -q "test"; then
    echo "  ✓ $node can access shared storage"
  else
    echo "  ✗ $node cannot access shared storage"
    ((ERRORS++))
  fi
done
rm -f $TESTFILE

# Check Python/HighNoon
echo -e "\n[4/5] HighNoon Installation"
for node in "${NODES[@]}"; do
  VERSION=$(ssh $node "source /shared/highnoon/venv/bin/activate && python -c 'import highnoon; print(highnoon.__version__)'" 2>/dev/null)
  if [ -n "$VERSION" ]; then
    echo "  ✓ $node - highnoon $VERSION"
  else
    echo "  ✗ $node - highnoon not found"
    ((ERRORS++))
  fi
done

# Check resources
echo -e "\n[5/5] Node Resources"
for node in "${NODES[@]}"; do
  MEM=$(ssh $node "free -g | awk '/Mem:/ {print \$2}'" 2>/dev/null)
  CPUS=$(ssh $node "nproc" 2>/dev/null)
  echo "  $node: ${CPUS} CPUs, ${MEM}GB RAM"
done

echo -e "\n=== Validation Complete ==="
if [ $ERRORS -eq 0 ]; then
  echo "✓ Cluster is ready for distributed training"
  exit 0
else
  echo "✗ $ERRORS error(s) found - fix before training"
  exit 1
fi
```

---

## See Also

- [Distributed Training Guide](distributed_training.md) - Training configuration
- [Enterprise Upgrade](enterprise-upgrade.md) - Dedicated support options
