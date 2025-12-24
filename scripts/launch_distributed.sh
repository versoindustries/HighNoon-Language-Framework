#!/bin/bash
# scripts/launch_distributed.sh
# Copyright 2025 Verso Industries
#
# Launch distributed training across multiple nodes.
#
# Usage:
#   ./scripts/launch_distributed.sh --hosts "node1,node2,node3" --script train.py
#   ./scripts/launch_distributed.sh -h node1,node2 -s examples/distributed_training.py -p 12345
#
# Prerequisites:
#   - Passwordless SSH to all nodes
#   - Shared filesystem for code and checkpoints
#   - Python environment available on all nodes

set -euo pipefail

# Default configuration
PORT=12345
SCRIPT=""
HOSTS=""
PYTHON_ENV="/shared/highnoon/venv/bin/activate"
WORKDIR=""
EXTRA_ARGS=""
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Launch distributed HighNoon training across multiple CPU nodes.

Required:
  -h, --hosts HOSTS     Comma-separated list of hostnames (e.g., "node1,node2,node3")
  -s, --script SCRIPT   Path to training script to run

Optional:
  -p, --port PORT       Communication port (default: 12345)
  -e, --env PATH        Path to Python virtualenv activate script
                        (default: /shared/highnoon/venv/bin/activate)
  -w, --workdir DIR     Working directory on remote nodes (default: script directory)
  -a, --args ARGS       Additional arguments to pass to the training script
  -n, --dry-run         Print commands without executing
  --help                Show this help message

Examples:
  # Basic usage
  $(basename "$0") -h node1,node2,node3 -s train.py

  # With custom port and extra args
  $(basename "$0") -h node1,node2 -s train.py -p 12346 -a "--epochs 10"

  # Dry run to see what would be executed
  $(basename "$0") -h node1,node2 -s train.py -n

EOF
    exit 1
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--hosts)
            HOSTS="$2"
            shift 2
            ;;
        -s|--script)
            SCRIPT="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -e|--env)
            PYTHON_ENV="$2"
            shift 2
            ;;
        -w|--workdir)
            WORKDIR="$2"
            shift 2
            ;;
        -a|--args)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$HOSTS" ]]; then
    log_error "Missing required argument: --hosts"
    usage
fi

if [[ -z "$SCRIPT" ]]; then
    log_error "Missing required argument: --script"
    usage
fi

# Parse hosts into array
IFS=',' read -ra HOST_ARRAY <<< "$HOSTS"
NUM_WORKERS=${#HOST_ARRAY[@]}

if [[ $NUM_WORKERS -lt 2 ]]; then
    log_warn "Only 1 host specified. For single-node training, run the script directly."
    log_warn "Proceeding with single-node distributed training..."
fi

# Default workdir to script directory
if [[ -z "$WORKDIR" ]]; then
    WORKDIR=$(dirname "$(realpath "$SCRIPT")")
fi

# Build worker list for TF_CONFIG
WORKER_LIST=""
for host in "${HOST_ARRAY[@]}"; do
    WORKER_LIST="${WORKER_LIST}\"${host}:${PORT}\","
done
WORKER_LIST="[${WORKER_LIST%,}]"  # Remove trailing comma

log_info "=========================================="
log_info "HighNoon Distributed Training Launcher"
log_info "=========================================="
log_info "Nodes:    ${HOST_ARRAY[*]}"
log_info "Workers:  $NUM_WORKERS"
log_info "Port:     $PORT"
log_info "Script:   $SCRIPT"
log_info "Workdir:  $WORKDIR"
log_info "Python:   $PYTHON_ENV"
if [[ -n "$EXTRA_ARGS" ]]; then
    log_info "Args:     $EXTRA_ARGS"
fi
log_info "=========================================="

# Verify SSH connectivity
log_info "Checking SSH connectivity..."
for host in "${HOST_ARRAY[@]}"; do
    if $DRY_RUN; then
        echo "[DRY RUN] Would check: ssh -o ConnectTimeout=5 $host hostname"
    else
        if ssh -o ConnectTimeout=5 "$host" "hostname" > /dev/null 2>&1; then
            log_info "  ✓ $host"
        else
            log_error "  ✗ $host - SSH connection failed"
            exit 1
        fi
    fi
done

# Launch workers
log_info "Launching workers..."
PIDS=()

for i in "${!HOST_ARRAY[@]}"; do
    host="${HOST_ARRAY[$i]}"

    # Build TF_CONFIG for this worker
    TF_CONFIG="{\"cluster\":{\"worker\":${WORKER_LIST}},\"task\":{\"type\":\"worker\",\"index\":$i}}"

    # Build command
    CMD="export TF_CONFIG='${TF_CONFIG}' && \
         source '${PYTHON_ENV}' && \
         cd '${WORKDIR}' && \
         python '${SCRIPT}' ${EXTRA_ARGS}"

    if $DRY_RUN; then
        log_info "Worker $i ($host):"
        echo "  TF_CONFIG='$TF_CONFIG'"
        echo "  Command: python ${SCRIPT} ${EXTRA_ARGS}"
        echo ""
    else
        log_info "Starting worker $i on $host..."

        # Launch in background, capturing output
        ssh "$host" "$CMD" 2>&1 | sed "s/^/[$host] /" &
        PIDS+=($!)
    fi
done

if $DRY_RUN; then
    log_info "Dry run complete. No workers were launched."
    exit 0
fi

log_info "All workers launched (PIDs: ${PIDS[*]})"
log_info "Waiting for training to complete..."
log_info "(Press Ctrl+C to stop all workers)"

# Trap Ctrl+C to kill all workers
cleanup() {
    log_warn "Interrupted! Stopping all workers..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done

    # Also kill remote processes
    for host in "${HOST_ARRAY[@]}"; do
        ssh "$host" "pkill -f '$SCRIPT'" 2>/dev/null || true
    done

    log_info "Cleanup complete"
    exit 1
}
trap cleanup SIGINT SIGTERM

# Wait for all workers
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    host="${HOST_ARRAY[$i]}"

    if wait "$pid"; then
        log_info "Worker $i ($host) completed successfully"
    else
        log_error "Worker $i ($host) failed with exit code $?"
        FAILED=$((FAILED + 1))
    fi
done

log_info "=========================================="
if [[ $FAILED -eq 0 ]]; then
    log_info "All workers completed successfully!"
    exit 0
else
    log_error "$FAILED worker(s) failed"
    exit 1
fi
