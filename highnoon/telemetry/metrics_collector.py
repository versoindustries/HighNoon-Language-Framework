"""
HSMN-specific metrics collection for ops, controllers, and training dynamics.

This module provides observability hooks for:
- HNN operations (energy drift, symplectic integration errors)
- Controller telemetry (Kalman filter states, MPC solver convergence)
- MoE expert utilization (Shannon entropy, capacity usage)
- Hardware health (CPU temperature, memory bandwidth, thermal throttling)

Phase 9.2 Deliverable: Performance metrics per technical_roadmap.md section 9.2.
"""

from __future__ import annotations

import platform
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from highnoon.telemetry.prometheus_exporter import get_metrics_registry


@dataclass
class HNNEnergyMetrics:
    """Metrics for Hamiltonian energy conservation tracking."""

    initial_energy: float
    current_energy: float
    drift: float
    max_drift: float
    steps: int


@dataclass
class MoEExpertMetrics:
    """Metrics for Mixture-of-Experts utilization."""

    num_experts: int
    expert_counts: np.ndarray
    shannon_entropy: float
    gini_coefficient: float
    max_expert_ratio: float


class HSMNMetricsCollector:
    """
    Central collector for HSMN architecture telemetry.

    Provides thread-safe metric recording with minimal performance overhead.
    Integrates with PrometheusExporter for standardized observability.
    """

    def __init__(self) -> None:
        self._registry = get_metrics_registry()
        self._lock = threading.RLock()

        # Op latency histograms
        self._op_latency_hist = {}

        # Energy drift tracking
        self._energy_states: dict[str, HNNEnergyMetrics] = {}

        # MoE expert utilization
        self._moe_states: dict[str, MoEExpertMetrics] = {}

        # Hardware telemetry
        self._cpu_temp_gauge = self._registry.gauge(
            "hsmn_cpu_temperature_celsius",
            "CPU temperature in degrees Celsius",
        )
        self._memory_usage_gauge = self._registry.gauge(
            "hsmn_memory_usage_bytes",
            "Process memory usage in bytes",
        )
        self._thermal_throttle_counter = self._registry.counter(
            "hsmn_thermal_throttle_events_total",
            "Total thermal throttling events detected",
        )

        # Training dynamics
        self._training_loss_gauge = self._registry.gauge(
            "hsmn_training_loss",
            "Current training loss",
        )
        self._training_step_counter = self._registry.counter(
            "hsmn_training_steps_total",
            "Total training steps executed",
        )

    def record_op_latency(self, op_name: str, latency_sec: float, device: str = "cpu") -> None:
        """
        Record latency for a custom TensorFlow op.

        Args:
            op_name: Name of the op (e.g., "fused_hnn_step")
            latency_sec: Execution time in seconds
            device: Device where op executed (cpu/gpu)
        """
        key = f"{op_name}_{device}"
        if key not in self._op_latency_hist:
            self._op_latency_hist[key] = self._registry.histogram(
                "hsmn_op_latency_seconds",
                "Latency distribution for TensorFlow custom ops",
                labels={"op": op_name, "device": device},
            )
        self._op_latency_hist[key].observe(latency_sec)

    def record_energy_drift(
        self,
        layer_name: str,
        initial_energy: float,
        current_energy: float,
        steps: int,
    ) -> None:
        """
        Record Hamiltonian energy drift for symplectic integrator.

        Args:
            layer_name: Name of HNN layer
            initial_energy: Initial Hamiltonian value
            current_energy: Current Hamiltonian value
            steps: Number of integration steps
        """
        drift = abs(current_energy - initial_energy)

        with self._lock:
            if layer_name not in self._energy_states:
                self._energy_states[layer_name] = HNNEnergyMetrics(
                    initial_energy=initial_energy,
                    current_energy=current_energy,
                    drift=drift,
                    max_drift=drift,
                    steps=steps,
                )
            else:
                state = self._energy_states[layer_name]
                state.current_energy = current_energy
                state.drift = drift
                state.max_drift = max(state.max_drift, drift)
                state.steps = steps

        # Update Prometheus metrics
        drift_gauge = self._registry.gauge(
            "hsmn_energy_drift",
            "Current Hamiltonian energy drift",
            labels={"layer": layer_name},
        )
        drift_gauge.set(drift)

        max_drift_gauge = self._registry.gauge(
            "hsmn_energy_drift_max",
            "Maximum observed Hamiltonian energy drift",
            labels={"layer": layer_name},
        )
        max_drift_gauge.set(self._energy_states[layer_name].max_drift)

        # Alert if drift exceeds threshold
        threshold = 1e-5
        if drift > threshold:
            alert_counter = self._registry.counter(
                "hsmn_energy_drift_violations_total",
                "Energy drift threshold violations",
                labels={"layer": layer_name},
            )
            alert_counter.inc()

    def record_moe_expert_utilization(
        self,
        layer_name: str,
        expert_indices: np.ndarray,
        num_experts: int,
    ) -> None:
        """
        Record MoE expert utilization and compute Shannon entropy.

        Args:
            layer_name: Name of MoE layer
            expert_indices: Array of selected expert indices [batch, top_k]
            num_experts: Total number of experts
        """
        # Count expert usage
        expert_counts = np.bincount(expert_indices.flatten(), minlength=num_experts)
        total_selections = expert_counts.sum()

        # Compute Shannon entropy
        probs = expert_counts / max(total_selections, 1)
        probs = probs[probs > 0]  # Remove zeros
        shannon_entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(num_experts)
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0

        # Compute Gini coefficient (expert imbalance)
        sorted_counts = np.sort(expert_counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_counts) - (n + 1) * cumsum[-1]) / (
            n * cumsum[-1] + 1e-10
        )

        # Max expert ratio (detect collapse)
        max_expert_ratio = expert_counts.max() / max(total_selections, 1)

        with self._lock:
            self._moe_states[layer_name] = MoEExpertMetrics(
                num_experts=num_experts,
                expert_counts=expert_counts,
                shannon_entropy=normalized_entropy,
                gini_coefficient=gini,
                max_expert_ratio=max_expert_ratio,
            )

        # Update Prometheus metrics
        entropy_gauge = self._registry.gauge(
            "hsmn_moe_shannon_entropy",
            "Normalized Shannon entropy of expert selection",
            labels={"layer": layer_name},
        )
        entropy_gauge.set(normalized_entropy)

        gini_gauge = self._registry.gauge(
            "hsmn_moe_gini_coefficient",
            "Gini coefficient of expert utilization (0=uniform, 1=collapsed)",
            labels={"layer": layer_name},
        )
        gini_gauge.set(gini)

        max_ratio_gauge = self._registry.gauge(
            "hsmn_moe_max_expert_ratio",
            "Fraction of selections going to most-used expert",
            labels={"layer": layer_name},
        )
        max_ratio_gauge.set(max_expert_ratio)

        # Alert on expert collapse (entropy < 0.5 or Gini > 0.8)
        if normalized_entropy < 0.5:
            collapse_counter = self._registry.counter(
                "hsmn_moe_collapse_events_total",
                "Expert collapse events (Shannon entropy < 0.5)",
                labels={"layer": layer_name},
            )
            collapse_counter.inc()

    def record_controller_state(
        self,
        controller_name: str,
        state_vector: np.ndarray,
        covariance_trace: float | None = None,
        solver_iterations: int | None = None,
    ) -> None:
        """
        Record controller state telemetry (Kalman/MPC).

        Args:
            controller_name: Name of controller
            state_vector: State estimate [state_dim]
            covariance_trace: Trace of covariance matrix (uncertainty)
            solver_iterations: Iterations for MPC solver convergence
        """
        # State magnitude
        state_norm = np.linalg.norm(state_vector)
        state_gauge = self._registry.gauge(
            "hsmn_controller_state_norm",
            "L2 norm of controller state vector",
            labels={"controller": controller_name},
        )
        state_gauge.set(float(state_norm))

        # Covariance (uncertainty)
        if covariance_trace is not None:
            cov_gauge = self._registry.gauge(
                "hsmn_controller_covariance_trace",
                "Trace of covariance matrix (total uncertainty)",
                labels={"controller": controller_name},
            )
            cov_gauge.set(covariance_trace)

        # MPC solver convergence
        if solver_iterations is not None:
            iter_hist = self._registry.histogram(
                "hsmn_mpc_solver_iterations",
                "Distribution of MPC solver iterations to convergence",
                labels={"controller": controller_name},
                buckets=(1, 5, 10, 20, 50, 100, 200, 500),
            )
            iter_hist.observe(solver_iterations)

    def record_hardware_telemetry(self) -> None:
        """
        Record system hardware telemetry (CPU temp, memory).

        Reads from /sys/class/thermal and /proc/self/status on Linux.
        """
        # CPU temperature
        if platform.system() == "Linux":
            try:
                thermal_zones = Path("/sys/class/thermal").glob("thermal_zone*")
                temps = []
                for zone in thermal_zones:
                    temp_file = zone / "temp"
                    if temp_file.exists():
                        temp_millidegrees = int(temp_file.read_text().strip())
                        temps.append(temp_millidegrees / 1000.0)
                if temps:
                    max_temp = max(temps)
                    self._cpu_temp_gauge.set(max_temp)

                    # Thermal throttle detection (> 80C)
                    if max_temp > 80.0:
                        self._thermal_throttle_counter.inc()
            except Exception:
                pass  # Ignore hardware read failures

        # Memory usage
        try:
            import psutil

            process = psutil.Process()
            memory_bytes = process.memory_info().rss
            self._memory_usage_gauge.set(memory_bytes)
        except ImportError:
            pass  # psutil not available

    def record_training_step(self, loss: float, step: int) -> None:
        """
        Record training dynamics.

        Args:
            loss: Current loss value
            step: Global training step
        """
        self._training_loss_gauge.set(loss)
        self._training_step_counter.inc()

    def get_energy_metrics(self, layer_name: str) -> HNNEnergyMetrics | None:
        """Get current energy metrics for a layer."""
        with self._lock:
            return self._energy_states.get(layer_name)

    def get_moe_metrics(self, layer_name: str) -> MoEExpertMetrics | None:
        """Get current MoE metrics for a layer."""
        with self._lock:
            return self._moe_states.get(layer_name)

    def export_metrics(self) -> None:
        """Force immediate export of all metrics to disk."""
        self._registry.export_to_file()


# Global singleton
_GLOBAL_COLLECTOR: HSMNMetricsCollector | None = None
_COLLECTOR_LOCK = threading.Lock()


def get_metrics_collector() -> HSMNMetricsCollector:
    """Get or create global metrics collector singleton."""
    global _GLOBAL_COLLECTOR
    if _GLOBAL_COLLECTOR is None:
        with _COLLECTOR_LOCK:
            if _GLOBAL_COLLECTOR is None:
                _GLOBAL_COLLECTOR = HSMNMetricsCollector()
    return _GLOBAL_COLLECTOR


__all__ = [
    "HSMNMetricsCollector",
    "HNNEnergyMetrics",
    "MoEExpertMetrics",
    "get_metrics_collector",
]
