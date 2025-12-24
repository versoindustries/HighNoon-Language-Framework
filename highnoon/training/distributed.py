# highnoon/training/distributed.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distributed Training Utilities for CPU Multi-Node Training.

This module provides helper functions for setting up distributed training
across multiple CPU nodes. HighNoon is a CPU-first architecture and does
not use GPU acceleration.

Supported Strategies:
    - MultiWorkerMirroredStrategy: Synchronous data parallelism (recommended)
    - ParameterServerStrategy: Async training for large clusters
    - CentralStorageStrategy: Single-node multi-CPU fallback

Example:
    >>> from highnoon.training.distributed import create_cpu_strategy
    >>> strategy = create_cpu_strategy()
    >>> with strategy.scope():
    ...     model = hn.create_model("7b")
    ...     trainer = hn.Trainer(model)
    ...     trainer.train(...)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

import tensorflow as tf

log = logging.getLogger(__name__)

# Communication protocols for CPU clusters
CommunicationProtocol = Literal["ring", "auto"]

# Strategy types
StrategyType = Literal["multi_worker", "parameter_server", "central_storage", "auto"]


@dataclass
class ClusterConfig:
    """Configuration for a distributed training cluster.

    Attributes:
        workers: List of worker addresses in "host:port" format.
        parameter_servers: Optional list of PS addresses for PS strategy.
        chief: Optional chief worker address.
        task_type: This node's role ("worker", "ps", or "chief").
        task_index: This node's index within its role.
    """

    workers: list[str]
    parameter_servers: list[str] | None = None
    chief: str | None = None
    task_type: str = "worker"
    task_index: int = 0

    @classmethod
    def from_tf_config(cls) -> ClusterConfig | None:
        """Parse cluster configuration from TF_CONFIG environment variable.

        Returns:
            ClusterConfig if TF_CONFIG is set and valid, None otherwise.

        Raises:
            ValueError: If TF_CONFIG is malformed.
        """
        tf_config_str = os.environ.get("TF_CONFIG")
        if not tf_config_str:
            return None

        try:
            tf_config = json.loads(tf_config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"TF_CONFIG is not valid JSON: {e}") from e

        cluster = tf_config.get("cluster", {})
        task = tf_config.get("task", {})

        workers = cluster.get("worker", [])
        if not workers:
            raise ValueError("TF_CONFIG must contain 'cluster.worker' list")

        return cls(
            workers=workers,
            parameter_servers=cluster.get("ps"),
            chief=cluster.get("chief", [None])[0] if cluster.get("chief") else None,
            task_type=task.get("type", "worker"),
            task_index=task.get("index", 0),
        )

    def to_tf_config(self) -> dict[str, Any]:
        """Convert to TF_CONFIG dictionary format."""
        cluster: dict[str, list[str]] = {"worker": self.workers}
        if self.parameter_servers:
            cluster["ps"] = self.parameter_servers
        if self.chief:
            cluster["chief"] = [self.chief]

        return {
            "cluster": cluster,
            "task": {"type": self.task_type, "index": self.task_index},
        }


def validate_tf_config() -> tuple[bool, str]:
    """Validate the TF_CONFIG environment variable.

    Returns:
        Tuple of (is_valid, message).

    Example:
        >>> valid, msg = validate_tf_config()
        >>> if not valid:
        ...     print(f"Error: {msg}")
    """
    tf_config_str = os.environ.get("TF_CONFIG")

    if not tf_config_str:
        return True, "TF_CONFIG not set - will use single-node strategy"

    try:
        config = ClusterConfig.from_tf_config()
        if config is None:
            return True, "TF_CONFIG not set"

        # Validate worker list
        if len(config.workers) < 2:
            return False, "TF_CONFIG has fewer than 2 workers - use single-node training instead"

        # Validate task index
        max_index = len(config.workers) - 1
        if config.task_type == "worker" and config.task_index > max_index:
            return (
                False,
                f"task.index {config.task_index} exceeds worker count {len(config.workers)}",
            )

        # Validate addresses
        for i, addr in enumerate(config.workers):
            if ":" not in addr:
                return False, f"Worker {i} address '{addr}' missing port (expected host:port)"

        return (
            True,
            f"Valid config: {len(config.workers)} workers, task {config.task_type}:{config.task_index}",
        )

    except ValueError as e:
        return False, str(e)


def create_cpu_strategy(
    strategy_type: StrategyType = "auto",
    communication: CommunicationProtocol = "auto",
    cluster_config: ClusterConfig | None = None,
) -> tf.distribute.Strategy:
    """Create a TensorFlow distribute strategy for CPU training.

    This function automatically detects the cluster configuration from
    TF_CONFIG and creates the appropriate strategy.

    Args:
        strategy_type: Type of strategy to create:
            - "auto": Detect from TF_CONFIG (default)
            - "multi_worker": MultiWorkerMirroredStrategy
            - "parameter_server": ParameterServerStrategy
            - "central_storage": CentralStorageStrategy (single-node)
        communication: Communication protocol for gradient reduction:
            - "auto": Let TensorFlow choose (default)
            - "ring": Ring-based all-reduce (CPU optimized)
        cluster_config: Optional explicit cluster configuration.
            If not provided, parsed from TF_CONFIG.

    Returns:
        A TensorFlow distribute strategy configured for CPU training.

    Example:
        >>> strategy = create_cpu_strategy()
        >>> print(f"Replicas: {strategy.num_replicas_in_sync}")
        >>> with strategy.scope():
        ...     model = create_model()
    """
    # Parse cluster config
    if cluster_config is None:
        cluster_config = ClusterConfig.from_tf_config()

    # Validate
    valid, msg = validate_tf_config()
    if not valid:
        raise ValueError(f"Invalid cluster configuration: {msg}")
    log.info(f"[DISTRIBUTED] {msg}")

    # Auto-detect strategy type
    if strategy_type == "auto":
        if cluster_config is None:
            strategy_type = "central_storage"
        elif cluster_config.parameter_servers:
            strategy_type = "parameter_server"
        else:
            strategy_type = "multi_worker"

    log.info(f"[DISTRIBUTED] Creating {strategy_type} strategy")

    # Configure communication
    if communication == "ring":
        communication_options = tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CommunicationImplementation.RING
        )
    else:
        communication_options = tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CommunicationImplementation.AUTO
        )

    # Create strategy
    if strategy_type == "multi_worker":
        strategy = tf.distribute.MultiWorkerMirroredStrategy(
            communication_options=communication_options
        )
        log.info(
            f"[DISTRIBUTED] MultiWorkerMirroredStrategy with {strategy.num_replicas_in_sync} replicas"
        )

    elif strategy_type == "parameter_server":
        if cluster_config is None or not cluster_config.parameter_servers:
            raise ValueError("ParameterServerStrategy requires 'ps' entries in TF_CONFIG")

        cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        variable_partitioner = tf.distribute.experimental.partitioners.MinSizePartitioner(
            min_shard_bytes=256 << 10,  # 256KB
            max_shards=len(cluster_config.parameter_servers),
        )
        strategy = tf.distribute.ParameterServerStrategy(
            cluster_resolver=cluster_resolver,
            variable_partitioner=variable_partitioner,
        )
        log.info(
            f"[DISTRIBUTED] ParameterServerStrategy with {len(cluster_config.workers)} workers, "
            f"{len(cluster_config.parameter_servers)} parameter servers"
        )

    elif strategy_type == "central_storage":
        # Single-node fallback - uses all CPU cores
        strategy = tf.distribute.experimental.CentralStorageStrategy()
        log.info("[DISTRIBUTED] CentralStorageStrategy (single-node)")

    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return strategy


def get_distributed_dataset(
    dataset: tf.data.Dataset,
    strategy: tf.distribute.Strategy,
    batch_size: int,
    shuffle: bool = True,
    shuffle_buffer: int = 10000,
) -> tf.distribute.DistributedDataset:
    """Prepare a dataset for distributed training.

    Handles batching, shuffling, and sharding automatically based on
    the strategy configuration.

    Args:
        dataset: Base TensorFlow dataset (unbatched).
        strategy: Distribute strategy to use.
        batch_size: Per-replica batch size.
        shuffle: Whether to shuffle the dataset.
        shuffle_buffer: Buffer size for shuffling.

    Returns:
        A DistributedDataset ready for training.

    Example:
        >>> strategy = create_cpu_strategy()
        >>> dist_dataset = get_distributed_dataset(
        ...     dataset, strategy, batch_size=8
        ... )
        >>> for batch in dist_dataset:
        ...     # batch is distributed across replicas
    """
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.batch(global_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Distribute with automatic sharding
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    return strategy.experimental_distribute_dataset(dataset)


def setup_cpu_threading(
    intra_op_threads: int | None = None,
    inter_op_threads: int | None = None,
) -> None:
    """Configure TensorFlow threading for optimal CPU utilization.

    Should be called before any TensorFlow operations.

    Args:
        intra_op_threads: Threads for parallelizing individual ops.
            Defaults to number of physical cores.
        inter_op_threads: Threads for parallelizing independent ops.
            Defaults to 2.

    Example:
        >>> setup_cpu_threading(intra_op_threads=16, inter_op_threads=2)
    """
    import multiprocessing

    if intra_op_threads is None:
        # Use physical cores (not hyperthreads)
        try:
            intra_op_threads = len(os.sched_getaffinity(0))
        except AttributeError:
            intra_op_threads = multiprocessing.cpu_count()

    if inter_op_threads is None:
        inter_op_threads = 2

    tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)

    log.info(
        f"[CPU] Threading configured: {intra_op_threads} intra-op, {inter_op_threads} inter-op"
    )


def get_worker_info() -> dict[str, Any]:
    """Get information about the current worker in a distributed setup.

    Returns:
        Dictionary with worker information including:
            - is_distributed: Whether running in distributed mode
            - is_chief: Whether this is the chief worker
            - worker_index: Index of this worker
            - num_workers: Total number of workers
            - cluster_spec: Full cluster specification

    Example:
        >>> info = get_worker_info()
        >>> if info["is_chief"]:
        ...     print("This is the chief worker")
    """
    config = ClusterConfig.from_tf_config()

    if config is None:
        return {
            "is_distributed": False,
            "is_chief": True,
            "worker_index": 0,
            "num_workers": 1,
            "cluster_spec": None,
        }

    is_chief = config.task_type == "worker" and config.task_index == 0

    return {
        "is_distributed": True,
        "is_chief": is_chief,
        "worker_index": config.task_index if config.task_type == "worker" else -1,
        "num_workers": len(config.workers),
        "cluster_spec": config.to_tf_config(),
    }


__all__ = [
    "ClusterConfig",
    "create_cpu_strategy",
    "get_distributed_dataset",
    "get_worker_info",
    "setup_cpu_threading",
    "validate_tf_config",
]
