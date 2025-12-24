# highnoon/webui/distributed_manager.py
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

"""Distributed Training Manager for WebUI-driven Cluster Coordination.

This module provides a WebSocket-based distributed training coordinator that
enables users to configure Host/Worker roles through the WebUI and manage
multi-node CPU training clusters without manual TF_CONFIG setup.

Architecture:
    - Host Mode: Runs a WebSocket server, accepts worker connections, generates
      and broadcasts TF_CONFIG, coordinates training start/stop
    - Worker Mode: Connects to a Host via WebSocket, receives TF_CONFIG,
      executes training under Host coordination
    - Standalone Mode: Default single-node training (no cluster)

Example:
    >>> from highnoon.webui.distributed_manager import DistributedManager
    >>> manager = DistributedManager()
    >>> await manager.start_host(port=12345, cluster_secret="my-secret")
    >>> # Workers connect via WebSocket
    >>> status = manager.get_cluster_status()
    >>> print(f"Connected workers: {len(status.workers)}")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class ClusterRole(str, Enum):
    """Role of this node in the distributed training cluster."""

    HOST = "host"
    """Chief coordinator that manages the cluster."""

    WORKER = "worker"
    """Worker node that joins an existing cluster."""

    STANDALONE = "standalone"
    """Single-node mode (default, no cluster)."""


class WorkerStatus(str, Enum):
    """Status of a connected worker."""

    CONNECTED = "connected"
    """Worker is connected and ready."""

    TRAINING = "training"
    """Worker is actively training."""

    DISCONNECTED = "disconnected"
    """Worker has disconnected."""

    ERROR = "error"
    """Worker encountered an error."""


class MessageType(str, Enum):
    """WebSocket message types for cluster coordination."""

    # Connection
    AUTH = "auth"
    AUTH_OK = "auth_ok"
    AUTH_FAILED = "auth_failed"

    # Registration
    REGISTER = "register"
    REGISTER_OK = "register_ok"

    # Health
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"

    # Cluster management
    WORKER_LIST = "worker_list"
    WORKER_JOINED = "worker_joined"
    WORKER_LEFT = "worker_left"

    # Training coordination
    TF_CONFIG = "tf_config"
    START_TRAINING = "start_training"
    STOP_TRAINING = "stop_training"
    TRAINING_STATUS = "training_status"
    TRAINING_METRICS = "training_metrics"

    # Errors
    ERROR = "error"


@dataclass
class WorkerInfo:
    """Information about a connected worker node.

    Attributes:
        worker_id: Unique identifier for this worker.
        hostname: Hostname of the worker machine.
        address: IP address and port for TensorFlow RPC.
        status: Current status of the worker.
        cpu_count: Number of CPU cores available.
        memory_gb: Total RAM in gigabytes.
        connected_at: ISO timestamp when worker connected.
        last_heartbeat: ISO timestamp of last heartbeat.
        task_index: TensorFlow task index assigned to this worker.
    """

    worker_id: str
    hostname: str
    address: str
    status: WorkerStatus = WorkerStatus.CONNECTED
    cpu_count: int = 0
    memory_gb: float = 0.0
    connected_at: str = ""
    last_heartbeat: str = ""
    task_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "address": self.address,
            "status": self.status.value if isinstance(self.status, WorkerStatus) else self.status,
            "cpu_count": self.cpu_count,
            "memory_gb": self.memory_gb,
            "connected_at": self.connected_at,
            "last_heartbeat": self.last_heartbeat,
            "task_index": self.task_index,
        }


@dataclass
class ClusterStatus:
    """Current status of the distributed training cluster.

    Attributes:
        role: Current role of this node.
        cluster_secret: The cluster secret (Host only, for display/copy).
        workers: List of connected workers.
        is_ready: Whether the cluster is ready for training.
        tf_config: Generated TF_CONFIG dictionary.
        host_address: Address of the Host (Worker only).
        is_training: Whether distributed training is in progress.
        error: Error message if any.
    """

    role: ClusterRole = ClusterRole.STANDALONE
    cluster_secret: str | None = None
    workers: list[WorkerInfo] = field(default_factory=list)
    is_ready: bool = False
    tf_config: dict[str, Any] | None = None
    host_address: str | None = None
    is_training: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role.value if isinstance(self.role, ClusterRole) else self.role,
            "cluster_secret": self.cluster_secret,
            "workers": [w.to_dict() for w in self.workers],
            "is_ready": self.is_ready,
            "tf_config": self.tf_config,
            "host_address": self.host_address,
            "is_training": self.is_training,
            "error": self.error,
        }


def _hash_secret(secret: str) -> str:
    """Hash a cluster secret for secure comparison."""
    return hashlib.sha256(secret.encode()).hexdigest()


def _generate_cluster_secret() -> str:
    """Generate a secure, human-readable cluster secret."""
    # Format: XXXX-XXXX-XXXX (12 chars, easy to type)
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # Exclude confusing chars
    parts = []
    for _ in range(3):
        part = "".join(secrets.choice(chars) for _ in range(4))
        parts.append(part)
    return "-".join(parts)


def _get_local_ip() -> str:
    """Get the local IP address for external connections."""
    try:
        # Create a socket to determine the outbound IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _get_system_info() -> dict[str, Any]:
    """Get system information for worker registration."""
    import multiprocessing

    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = multiprocessing.cpu_count()

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_kb = int(line.split()[1])
                    memory_gb = mem_kb / (1024 * 1024)
                    break
            else:
                memory_gb = 0.0
    except Exception:
        memory_gb = 0.0

    return {
        "hostname": socket.gethostname(),
        "cpu_count": cpu_count,
        "memory_gb": round(memory_gb, 2),
        "local_ip": _get_local_ip(),
    }


class DistributedManager:
    """Manages distributed training cluster coordination via WebSocket.

    This class handles both Host and Worker modes:
    - Host: Runs WebSocket server, tracks workers, generates TF_CONFIG
    - Worker: Connects to Host, receives configuration, reports status

    The manager is designed to be used as a singleton within the WebUI backend.

    Attributes:
        role: Current cluster role.
        port: Port for WebSocket connections.
        cluster_secret_hash: Hashed cluster secret for authentication.
        workers: Dictionary of connected workers (Host mode).
        host_address: Address of the Host (Worker mode).

    Example:
        >>> manager = DistributedManager()
        >>> # As Host
        >>> secret = await manager.start_host(port=12345)
        >>> print(f"Workers can join with secret: {secret}")
        >>> # As Worker (on another machine)
        >>> await manager.join_cluster("192.168.1.100:12345", "XXXX-XXXX-XXXX")
    """

    def __init__(self) -> None:
        """Initialize the distributed manager in standalone mode."""
        self.role: ClusterRole = ClusterRole.STANDALONE
        self.port: int = 12345
        self.cluster_secret: str | None = None
        self.cluster_secret_hash: str | None = None
        self.shared_checkpoint_dir: str = "/shared/checkpoints"
        self.communication_protocol: str = "ring"

        # Host mode state
        self._workers: dict[str, WorkerInfo] = {}
        self._worker_websockets: dict[str, Any] = {}  # worker_id -> WebSocket
        self._server: asyncio.Server | None = None
        self._next_task_index: int = 1  # 0 is reserved for Host/chief

        # Worker mode state
        self._host_address: str | None = None
        self._host_ws: Any = None
        self._worker_id: str | None = None

        # Shared state
        self._is_training: bool = False
        self._tf_config: dict[str, Any] | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._status_callbacks: list[Callable[[ClusterStatus], None]] = []

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        log.info("[DISTRIBUTED] Manager initialized in standalone mode")

    def get_cluster_status(self) -> ClusterStatus:
        """Get the current cluster status.

        Returns:
            ClusterStatus with current role, workers, and configuration.
        """
        return ClusterStatus(
            role=self.role,
            cluster_secret=self.cluster_secret if self.role == ClusterRole.HOST else None,
            workers=list(self._workers.values()),
            is_ready=self._is_cluster_ready(),
            tf_config=self._tf_config,
            host_address=self._host_address,
            is_training=self._is_training,
        )

    def _is_cluster_ready(self) -> bool:
        """Check if the cluster is ready for training."""
        if self.role == ClusterRole.STANDALONE:
            return True
        if self.role == ClusterRole.HOST:
            # Need at least one worker besides ourselves
            connected = sum(1 for w in self._workers.values() if w.status == WorkerStatus.CONNECTED)
            return connected >= 1
        if self.role == ClusterRole.WORKER:
            return self._tf_config is not None
        return False

    async def start_host(
        self,
        port: int = 12345,
        cluster_secret: str | None = None,
        shared_checkpoint_dir: str = "/shared/checkpoints",
        communication_protocol: str = "ring",
    ) -> str:
        """Start as a Host node, accepting worker connections.

        Args:
            port: Port for the WebSocket server.
            cluster_secret: Optional pre-defined secret. If None, one is generated.
            shared_checkpoint_dir: Path to shared checkpoint directory.
            communication_protocol: TensorFlow communication protocol (ring, auto).

        Returns:
            The cluster secret that workers need to join.

        Raises:
            RuntimeError: If already in Host or Worker mode.
        """
        if self.role != ClusterRole.STANDALONE:
            raise RuntimeError(f"Cannot start host: already in {self.role.value} mode")

        self.port = port
        self.shared_checkpoint_dir = shared_checkpoint_dir
        self.communication_protocol = communication_protocol

        # Generate or use provided secret
        if cluster_secret is None:
            self.cluster_secret = _generate_cluster_secret()
        else:
            self.cluster_secret = cluster_secret
        self.cluster_secret_hash = _hash_secret(self.cluster_secret)

        # Register ourselves as worker 0 (chief)
        sys_info = _get_system_info()
        local_ip = sys_info["local_ip"]
        chief_address = f"{local_ip}:{port + 1}"  # TF uses port+1 for gRPC

        self._workers["chief"] = WorkerInfo(
            worker_id="chief",
            hostname=sys_info["hostname"],
            address=chief_address,
            status=WorkerStatus.CONNECTED,
            cpu_count=sys_info["cpu_count"],
            memory_gb=sys_info["memory_gb"],
            connected_at=datetime.now(timezone.utc).isoformat(),
            last_heartbeat=datetime.now(timezone.utc).isoformat(),
            task_index=0,
        )

        # Start WebSocket server
        try:
            self._server = await asyncio.start_server(
                self._handle_worker_connection,
                "0.0.0.0",
                port,
            )
            self.role = ClusterRole.HOST
            self._update_tf_config()

            log.info(
                f"[DISTRIBUTED] Started Host on port {port}, "
                f"secret: {self.cluster_secret}, address: {chief_address}"
            )

            # Start heartbeat monitoring
            self._heartbeat_task = asyncio.create_task(self._monitor_heartbeats())

            return self.cluster_secret

        except Exception as e:
            log.error(f"[DISTRIBUTED] Failed to start host: {e}")
            self._workers.clear()
            raise RuntimeError(f"Failed to start host: {e}") from e

    async def stop_host(self) -> None:
        """Stop hosting and return to standalone mode."""
        if self.role != ClusterRole.HOST:
            return

        log.info("[DISTRIBUTED] Stopping host...")

        # Cancel heartbeat monitoring
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Close all worker connections
        for ws in list(self._worker_websockets.values()):
            try:
                ws.close()
            except Exception:
                pass

        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Reset state
        self._workers.clear()
        self._worker_websockets.clear()
        self._next_task_index = 1
        self.cluster_secret = None
        self.cluster_secret_hash = None
        self._tf_config = None
        self._is_training = False
        self.role = ClusterRole.STANDALONE

        log.info("[DISTRIBUTED] Host stopped, returned to standalone mode")

    async def join_cluster(
        self,
        host_address: str,
        cluster_secret: str,
    ) -> None:
        """Join an existing cluster as a Worker.

        Args:
            host_address: Host address in "ip:port" format.
            cluster_secret: The cluster secret provided by the Host.

        Raises:
            RuntimeError: If already in Host or Worker mode.
            ConnectionError: If unable to connect to Host.
            ValueError: If authentication fails.
        """
        if self.role != ClusterRole.STANDALONE:
            raise RuntimeError(f"Cannot join cluster: already in {self.role.value} mode")

        # Parse host address
        if ":" not in host_address:
            host_address = f"{host_address}:12345"

        host, port_str = host_address.rsplit(":", 1)
        port = int(port_str)

        log.info(f"[DISTRIBUTED] Joining cluster at {host}:{port}")

        try:
            # Connect to host
            reader, writer = await asyncio.open_connection(host, port)
            self._host_ws = (reader, writer)
            self._host_address = host_address

            # Send authentication
            auth_msg = {
                "type": MessageType.AUTH.value,
                "secret_hash": _hash_secret(cluster_secret),
            }
            writer.write((json.dumps(auth_msg) + "\n").encode())
            await writer.drain()

            # Wait for auth response
            response_line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            response = json.loads(response_line.decode())

            if response.get("type") == MessageType.AUTH_FAILED.value:
                writer.close()
                await writer.wait_closed()
                raise ValueError("Authentication failed: invalid cluster secret")

            if response.get("type") != MessageType.AUTH_OK.value:
                writer.close()
                await writer.wait_closed()
                raise ValueError(f"Unexpected response: {response}")

            # Send registration info
            sys_info = _get_system_info()
            self._worker_id = f"worker-{secrets.token_hex(4)}"

            reg_msg = {
                "type": MessageType.REGISTER.value,
                "worker_id": self._worker_id,
                "hostname": sys_info["hostname"],
                "cpu_count": sys_info["cpu_count"],
                "memory_gb": sys_info["memory_gb"],
                "local_ip": sys_info["local_ip"],
            }
            writer.write((json.dumps(reg_msg) + "\n").encode())
            await writer.drain()

            # Wait for registration response with TF_CONFIG
            reg_response_line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            reg_response = json.loads(reg_response_line.decode())

            if reg_response.get("type") != MessageType.REGISTER_OK.value:
                writer.close()
                await writer.wait_closed()
                raise ValueError(f"Registration failed: {reg_response}")

            self._tf_config = reg_response.get("tf_config")
            self.role = ClusterRole.WORKER

            log.info(
                f"[DISTRIBUTED] Joined cluster as {self._worker_id}, "
                f"task_index: {reg_response.get('task_index')}"
            )

            # Start message handler
            asyncio.create_task(self._handle_host_messages(reader))

            # Start heartbeat sender
            self._heartbeat_task = asyncio.create_task(self._send_heartbeats(writer))

        except asyncio.TimeoutError:
            raise ConnectionError("Connection to host timed out") from None
        except ConnectionRefusedError:
            raise ConnectionError(f"Unable to connect to host at {host_address}") from None
        except Exception as e:
            log.error(f"[DISTRIBUTED] Failed to join cluster: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from cluster (Worker) or stop hosting (Host)."""
        if self.role == ClusterRole.HOST:
            await self.stop_host()
        elif self.role == ClusterRole.WORKER:
            await self._disconnect_from_host()

    async def _disconnect_from_host(self) -> None:
        """Disconnect from the Host cluster."""
        if self.role != ClusterRole.WORKER:
            return

        log.info("[DISTRIBUTED] Disconnecting from host...")

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Close connection
        if self._host_ws:
            _, writer = self._host_ws
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            self._host_ws = None

        # Reset state
        self._host_address = None
        self._worker_id = None
        self._tf_config = None
        self._is_training = False
        self.role = ClusterRole.STANDALONE

        log.info("[DISTRIBUTED] Disconnected from host, returned to standalone mode")

    async def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker from the cluster (Host only).

        Args:
            worker_id: ID of the worker to remove.

        Returns:
            True if worker was removed, False if not found.
        """
        if self.role != ClusterRole.HOST:
            return False

        async with self._lock:
            if worker_id not in self._workers or worker_id == "chief":
                return False

            # Close WebSocket if connected
            if worker_id in self._worker_websockets:
                try:
                    _, writer = self._worker_websockets[worker_id]
                    writer.close()
                except Exception:
                    pass
                del self._worker_websockets[worker_id]

            del self._workers[worker_id]
            self._update_tf_config()

            log.info(f"[DISTRIBUTED] Removed worker: {worker_id}")
            await self._broadcast_worker_list()

        return True

    async def start_distributed_training(
        self,
        training_config: dict[str, Any],
    ) -> bool:
        """Start distributed training across all workers (Host only).

        Args:
            training_config: Training configuration to broadcast.

        Returns:
            True if training started successfully.
        """
        if self.role != ClusterRole.HOST:
            return False

        if not self._is_cluster_ready():
            return False

        self._is_training = True

        # Update all worker statuses
        for worker in self._workers.values():
            if worker.status == WorkerStatus.CONNECTED:
                worker.status = WorkerStatus.TRAINING

        # Broadcast start command
        msg = {
            "type": MessageType.START_TRAINING.value,
            "tf_config": self._tf_config,
            "training_config": training_config,
        }
        await self._broadcast_message(msg)

        log.info("[DISTRIBUTED] Started distributed training")
        return True

    async def stop_distributed_training(self) -> None:
        """Stop distributed training on all workers (Host only)."""
        if self.role != ClusterRole.HOST:
            return

        self._is_training = False

        # Update all worker statuses
        for worker in self._workers.values():
            if worker.status == WorkerStatus.TRAINING:
                worker.status = WorkerStatus.CONNECTED

        # Broadcast stop command
        msg = {"type": MessageType.STOP_TRAINING.value}
        await self._broadcast_message(msg)

        log.info("[DISTRIBUTED] Stopped distributed training")

    def _update_tf_config(self) -> None:
        """Update the TF_CONFIG based on current workers."""
        if self.role != ClusterRole.HOST:
            return

        # Build worker list sorted by task index
        workers_sorted = sorted(
            self._workers.values(),
            key=lambda w: w.task_index,
        )

        worker_addresses = [w.address for w in workers_sorted]

        self._tf_config = {
            "cluster": {"worker": worker_addresses},
            "task": {"type": "worker", "index": 0},  # Chief's view
        }

    async def _handle_worker_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a new worker connection (Host mode)."""
        peer = writer.get_extra_info("peername")
        log.info(f"[DISTRIBUTED] New connection from {peer}")

        worker_id = None

        try:
            # Read auth message
            auth_line = await asyncio.wait_for(reader.readline(), timeout=30.0)
            auth_msg = json.loads(auth_line.decode())

            if auth_msg.get("type") != MessageType.AUTH.value:
                await self._send_error(writer, "Expected AUTH message")
                return

            # Verify secret
            if auth_msg.get("secret_hash") != self.cluster_secret_hash:
                fail_msg = {"type": MessageType.AUTH_FAILED.value}
                writer.write((json.dumps(fail_msg) + "\n").encode())
                await writer.drain()
                log.warning(f"[DISTRIBUTED] Auth failed for {peer}")
                return

            # Send auth OK
            ok_msg = {"type": MessageType.AUTH_OK.value}
            writer.write((json.dumps(ok_msg) + "\n").encode())
            await writer.drain()

            # Read registration
            reg_line = await asyncio.wait_for(reader.readline(), timeout=30.0)
            reg_msg = json.loads(reg_line.decode())

            if reg_msg.get("type") != MessageType.REGISTER.value:
                await self._send_error(writer, "Expected REGISTER message")
                return

            # Register worker
            worker_id = reg_msg["worker_id"]
            worker_ip = reg_msg.get("local_ip", peer[0])
            worker_port = self.port + 1 + self._next_task_index

            async with self._lock:
                task_index = self._next_task_index
                self._next_task_index += 1

                worker_info = WorkerInfo(
                    worker_id=worker_id,
                    hostname=reg_msg.get("hostname", "unknown"),
                    address=f"{worker_ip}:{worker_port}",
                    status=WorkerStatus.CONNECTED,
                    cpu_count=reg_msg.get("cpu_count", 0),
                    memory_gb=reg_msg.get("memory_gb", 0.0),
                    connected_at=datetime.now(timezone.utc).isoformat(),
                    last_heartbeat=datetime.now(timezone.utc).isoformat(),
                    task_index=task_index,
                )

                self._workers[worker_id] = worker_info
                self._worker_websockets[worker_id] = (reader, writer)
                self._update_tf_config()

            # Build worker-specific TF_CONFIG
            worker_tf_config = {
                "cluster": self._tf_config["cluster"],
                "task": {"type": "worker", "index": task_index},
            }

            # Send registration OK with TF_CONFIG
            reg_ok_msg = {
                "type": MessageType.REGISTER_OK.value,
                "task_index": task_index,
                "tf_config": worker_tf_config,
            }
            writer.write((json.dumps(reg_ok_msg) + "\n").encode())
            await writer.drain()

            log.info(f"[DISTRIBUTED] Worker {worker_id} registered with task_index {task_index}")

            # Broadcast updated worker list to all workers
            await self._broadcast_worker_list()

            # Handle messages from this worker
            while True:
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=60.0)
                    if not line:
                        break

                    msg = json.loads(line.decode())
                    await self._handle_worker_message(worker_id, msg)

                except asyncio.TimeoutError:
                    # Check if worker is still alive
                    if worker_id in self._workers:
                        worker = self._workers[worker_id]
                        last_hb = datetime.fromisoformat(worker.last_heartbeat)
                        if (datetime.now(timezone.utc) - last_hb).total_seconds() > 120:
                            log.warning(f"[DISTRIBUTED] Worker {worker_id} timed out")
                            break
                    continue

        except Exception as e:
            log.error(f"[DISTRIBUTED] Error handling connection from {peer}: {e}")

        finally:
            # Clean up worker
            if worker_id:
                async with self._lock:
                    if worker_id in self._workers:
                        self._workers[worker_id].status = WorkerStatus.DISCONNECTED
                        del self._workers[worker_id]
                    if worker_id in self._worker_websockets:
                        del self._worker_websockets[worker_id]
                    self._update_tf_config()

                await self._broadcast_worker_list()
                log.info(f"[DISTRIBUTED] Worker {worker_id} disconnected")

            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_worker_message(self, worker_id: str, msg: dict[str, Any]) -> None:
        """Handle a message from a worker."""
        msg_type = msg.get("type")

        if msg_type == MessageType.HEARTBEAT.value:
            async with self._lock:
                if worker_id in self._workers:
                    self._workers[worker_id].last_heartbeat = datetime.now(timezone.utc).isoformat()

            # Send heartbeat ack
            if worker_id in self._worker_websockets:
                _, writer = self._worker_websockets[worker_id]
                ack = {"type": MessageType.HEARTBEAT_ACK.value}
                writer.write((json.dumps(ack) + "\n").encode())
                await writer.drain()

        elif msg_type == MessageType.TRAINING_METRICS.value:
            # Forward metrics to status callbacks
            metrics = msg.get("metrics", {})
            metrics["worker_id"] = worker_id
            for callback in self._status_callbacks:
                try:
                    callback(self.get_cluster_status())
                except Exception:
                    pass

    async def _handle_host_messages(self, reader: asyncio.StreamReader) -> None:
        """Handle messages from Host (Worker mode)."""
        try:
            while self.role == ClusterRole.WORKER:
                line = await reader.readline()
                if not line:
                    break

                msg = json.loads(line.decode())
                msg_type = msg.get("type")

                if msg_type == MessageType.TF_CONFIG.value:
                    self._tf_config = msg.get("tf_config")

                elif msg_type == MessageType.START_TRAINING.value:
                    self._is_training = True
                    self._tf_config = msg.get("tf_config")
                    # Training config is passed to actual training system
                    log.info("[DISTRIBUTED] Received START_TRAINING command")

                elif msg_type == MessageType.STOP_TRAINING.value:
                    self._is_training = False
                    log.info("[DISTRIBUTED] Received STOP_TRAINING command")

                elif msg_type == MessageType.WORKER_LIST.value:
                    # Update local worker list for status display
                    pass

                elif msg_type == MessageType.HEARTBEAT_ACK.value:
                    pass

        except Exception as e:
            log.error(f"[DISTRIBUTED] Error in host message handler: {e}")
            await self._disconnect_from_host()

    async def _send_heartbeats(self, writer: asyncio.StreamWriter) -> None:
        """Send periodic heartbeats to Host (Worker mode)."""
        try:
            while self.role == ClusterRole.WORKER:
                msg = {"type": MessageType.HEARTBEAT.value}
                writer.write((json.dumps(msg) + "\n").encode())
                await writer.drain()
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"[DISTRIBUTED] Heartbeat error: {e}")

    async def _monitor_heartbeats(self) -> None:
        """Monitor worker heartbeats and remove stale workers (Host mode)."""
        try:
            while self.role == ClusterRole.HOST:
                await asyncio.sleep(60)

                now = datetime.now(timezone.utc)
                stale_workers = []

                async with self._lock:
                    for worker_id, worker in self._workers.items():
                        if worker_id == "chief":
                            continue
                        last_hb = datetime.fromisoformat(worker.last_heartbeat)
                        if (now - last_hb).total_seconds() > 120:
                            stale_workers.append(worker_id)

                for worker_id in stale_workers:
                    log.warning(f"[DISTRIBUTED] Worker {worker_id} stale, removing")
                    await self.remove_worker(worker_id)

        except asyncio.CancelledError:
            pass

    async def _broadcast_message(self, msg: dict[str, Any]) -> None:
        """Broadcast a message to all connected workers."""
        data = (json.dumps(msg) + "\n").encode()
        for worker_id, (_, writer) in list(self._worker_websockets.items()):
            try:
                writer.write(data)
                await writer.drain()
            except Exception as e:
                log.warning(f"[DISTRIBUTED] Failed to send to {worker_id}: {e}")

    async def _broadcast_worker_list(self) -> None:
        """Broadcast the current worker list to all workers."""
        msg = {
            "type": MessageType.WORKER_LIST.value,
            "workers": [w.to_dict() for w in self._workers.values()],
        }
        await self._broadcast_message(msg)

    async def _send_error(self, writer: asyncio.StreamWriter, error: str) -> None:
        """Send an error message and close connection."""
        msg = {"type": MessageType.ERROR.value, "error": error}
        try:
            writer.write((json.dumps(msg) + "\n").encode())
            await writer.drain()
        except Exception:
            pass


# Global singleton instance
_manager_instance: DistributedManager | None = None


def get_distributed_manager() -> DistributedManager:
    """Get the global DistributedManager instance.

    Returns:
        The singleton DistributedManager.
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = DistributedManager()
    return _manager_instance


__all__ = [
    "ClusterRole",
    "ClusterStatus",
    "DistributedManager",
    "MessageType",
    "WorkerInfo",
    "WorkerStatus",
    "get_distributed_manager",
]
