# src/telemetry/energy_monitor.py
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

"""
Energy Conservation Monitor for TimeCrystal Blocks

This module provides monitoring and logging functionality for Hamiltonian energy
conservation in TimeCrystal blocks during molecular property prediction.

The monitor tracks:
- Energy drift per evolution step: |H_final - H_initial|
- Conservation rate: fraction of steps with drift < threshold
- State convergence: variance of state norms over time
- Evolution time statistics: mean, variance, min, max

Usage:
    monitor = EnergyConservationMonitor(name="timecrystal_layer_1")

    # During training/inference
    monitor.record_step(h_initial, h_final, evolution_time, state_norm)

    # At epoch end
    stats = monitor.get_statistics()
    print(f"Energy conservation rate: {stats['conservation_rate']*100:.1f}%")

    # Reset for next epoch
    monitor.reset()
"""

import logging
from collections import deque

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class EnergyConservationMonitor:
    """
    Monitors Hamiltonian energy conservation in TimeCrystal blocks.

    Tracks energy drift, conservation rate, and state convergence metrics
    for each evolution step in the reasoning stack.
    """

    def __init__(
        self,
        name: str = "energy_monitor",
        drift_threshold: float = 0.1,
        history_size: int = 1000,
        enable_logging: bool = True,
    ):
        """
        Initialize energy conservation monitor.

        Args:
            name: Monitor name for logging
            drift_threshold: Maximum acceptable energy drift (default: 0.1)
            history_size: Maximum number of steps to keep in history
            enable_logging: Whether to log warnings for high drift
        """
        self.name = name
        self.drift_threshold = drift_threshold
        self.history_size = history_size
        self.enable_logging = enable_logging

        # Energy drift tracking
        self.energy_drifts: list[float] = []
        self.initial_energies: list[float] = []
        self.final_energies: list[float] = []

        # Evolution time tracking
        self.evolution_times: list[float] = []

        # State norm tracking (for convergence analysis)
        self.state_norms: list[float] = []

        # Step counter
        self.total_steps = 0
        self.conservation_violations = 0

        # Moving window for recent statistics
        self.recent_drifts = deque(maxlen=100)

        logger.info(f"Initialized {self.name} with drift_threshold={drift_threshold}")

    def record_step(
        self,
        h_initial: tf.Tensor,
        h_final: tf.Tensor,
        evolution_time: tf.Tensor | None = None,
        state_norm: tf.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Record energy drift for a single evolution step.

        Args:
            h_initial: Initial Hamiltonian energy [batch] or scalar
            h_final: Final Hamiltonian energy [batch] or scalar
            evolution_time: Evolution time step (dt) [batch] or scalar (optional)
            state_norm: Norm of state vector (||q|| + ||p||) [batch] or scalar (optional)

        Returns:
            Dictionary with step metrics:
                - drift: Energy drift |H_final - H_initial|
                - relative_drift: Normalized drift / |H_initial|
                - conserved: Whether drift < threshold
        """
        # Convert to numpy for storage
        h_init = tf.reduce_mean(h_initial).numpy() if tf.is_tensor(h_initial) else float(h_initial)
        h_fin = tf.reduce_mean(h_final).numpy() if tf.is_tensor(h_final) else float(h_final)

        # Compute drift
        drift = abs(h_fin - h_init)
        relative_drift = drift / (abs(h_init) + 1e-8)

        # Check conservation
        conserved = drift < self.drift_threshold

        # Update counters
        self.total_steps += 1
        if not conserved:
            self.conservation_violations += 1
            if self.enable_logging and self.conservation_violations % 10 == 0:
                logger.warning(
                    f"{self.name}: Conservation violation #{self.conservation_violations} "
                    f"(drift={drift:.4f}, threshold={self.drift_threshold})"
                )

        # Store metrics (with history limit)
        if len(self.energy_drifts) >= self.history_size:
            self.energy_drifts.pop(0)
            self.initial_energies.pop(0)
            self.final_energies.pop(0)

        self.energy_drifts.append(float(drift))
        self.initial_energies.append(float(h_init))
        self.final_energies.append(float(h_fin))
        self.recent_drifts.append(float(drift))

        # Store evolution time if provided
        if evolution_time is not None:
            dt = (
                tf.reduce_mean(evolution_time).numpy()
                if tf.is_tensor(evolution_time)
                else float(evolution_time)
            )
            if len(self.evolution_times) >= self.history_size:
                self.evolution_times.pop(0)
            self.evolution_times.append(float(dt))

        # Store state norm if provided
        if state_norm is not None:
            norm = (
                tf.reduce_mean(state_norm).numpy()
                if tf.is_tensor(state_norm)
                else float(state_norm)
            )
            if len(self.state_norms) >= self.history_size:
                self.state_norms.pop(0)
            self.state_norms.append(float(norm))

        # Return step metrics
        return {
            "drift": float(drift),
            "relative_drift": float(relative_drift),
            "conserved": bool(conserved),
        }

    def get_statistics(self) -> dict[str, float]:
        """
        Compute energy conservation statistics over recorded history.

        Returns:
            Dictionary with statistics:
                - total_steps: Total number of recorded steps
                - mean_drift: Mean energy drift
                - median_drift: Median energy drift
                - max_drift: Maximum energy drift
                - std_drift: Standard deviation of drift
                - conservation_rate: Fraction of steps with drift < threshold
                - conservation_violations: Number of steps with drift >= threshold
                - mean_evolution_time: Mean evolution time (if recorded)
                - std_evolution_time: Std of evolution time (if recorded)
                - mean_state_norm: Mean state norm (if recorded)
                - state_norm_variance: Variance of state norms (for convergence check)
        """
        if not self.energy_drifts:
            return {
                "total_steps": 0,
                "mean_drift": 0.0,
                "median_drift": 0.0,
                "max_drift": 0.0,
                "std_drift": 0.0,
                "conservation_rate": 1.0,
                "conservation_violations": 0,
                "mean_evolution_time": 0.0,
                "std_evolution_time": 0.0,
                "mean_state_norm": 0.0,
                "state_norm_variance": 0.0,
            }

        drifts = np.array(self.energy_drifts)

        # Energy drift statistics
        mean_drift = float(np.mean(drifts))
        median_drift = float(np.median(drifts))
        max_drift = float(np.max(drifts))
        std_drift = float(np.std(drifts))

        # Conservation rate (fraction within threshold)
        conservation_rate = float(np.mean(drifts < self.drift_threshold))

        stats = {
            "total_steps": self.total_steps,
            "mean_drift": mean_drift,
            "median_drift": median_drift,
            "max_drift": max_drift,
            "std_drift": std_drift,
            "conservation_rate": conservation_rate,
            "conservation_violations": self.conservation_violations,
        }

        # Evolution time statistics
        if self.evolution_times:
            times = np.array(self.evolution_times)
            stats["mean_evolution_time"] = float(np.mean(times))
            stats["std_evolution_time"] = float(np.std(times))
            stats["min_evolution_time"] = float(np.min(times))
            stats["max_evolution_time"] = float(np.max(times))
        else:
            stats["mean_evolution_time"] = 0.0
            stats["std_evolution_time"] = 0.0

        # State norm statistics (for convergence analysis)
        if self.state_norms:
            norms = np.array(self.state_norms)
            stats["mean_state_norm"] = float(np.mean(norms))
            stats["std_state_norm"] = float(np.std(norms))

            # Compute variance of recent norms (low variance = converged)
            if len(norms) >= 5:
                recent_norms = norms[-5:]
                stats["state_norm_variance"] = float(np.var(recent_norms))
                stats["converged"] = bool(stats["state_norm_variance"] < 1.0)
            else:
                stats["state_norm_variance"] = float(np.var(norms))
                stats["converged"] = False
        else:
            stats["mean_state_norm"] = 0.0
            stats["state_norm_variance"] = 0.0
            stats["converged"] = False

        return stats

    def check_health(self, verbose: bool = True) -> tuple[bool, str]:
        """
        Check health of energy conservation.

        Args:
            verbose: Whether to log detailed health status

        Returns:
            Tuple of (is_healthy, message):
                - is_healthy: True if conservation rate > 90%
                - message: Human-readable health status
        """
        stats = self.get_statistics()

        if stats["total_steps"] == 0:
            return True, "No steps recorded yet"

        conservation_rate = stats["conservation_rate"]
        mean_drift = stats["mean_drift"]
        max_drift = stats["max_drift"]

        # Health criteria
        is_healthy = conservation_rate >= 0.90

        # Build message
        if is_healthy:
            status = "✓ HEALTHY"
            level = logging.INFO
        elif conservation_rate >= 0.80:
            status = "⚠ WARNING"
            level = logging.WARNING
        else:
            status = "✗ CRITICAL"
            level = logging.ERROR

        message = (
            f"{self.name} Energy Conservation {status}:\n"
            f"  Conservation rate: {conservation_rate*100:.1f}% "
            f"({stats['total_steps']} steps)\n"
            f"  Mean drift: {mean_drift:.4f} (threshold: {self.drift_threshold})\n"
            f"  Max drift: {max_drift:.4f}\n"
            f"  Violations: {stats['conservation_violations']}"
        )

        if verbose:
            logger.log(level, message)

        return is_healthy, message

    def reset(self) -> None:
        """Reset monitoring statistics for next epoch."""
        self.energy_drifts.clear()
        self.initial_energies.clear()
        self.final_energies.clear()
        self.evolution_times.clear()
        self.state_norms.clear()
        self.total_steps = 0
        self.conservation_violations = 0
        self.recent_drifts.clear()

        logger.debug(f"{self.name}: Reset monitoring statistics")

    def get_recent_statistics(self, window_size: int = 100) -> dict[str, float]:
        """
        Get statistics over recent window (for real-time monitoring).

        Args:
            window_size: Size of recent window (default: 100)

        Returns:
            Dictionary with recent statistics (same format as get_statistics)
        """
        if not self.recent_drifts:
            return self.get_statistics()

        recent = np.array(list(self.recent_drifts))

        return {
            "total_steps": len(recent),
            "mean_drift": float(np.mean(recent)),
            "median_drift": float(np.median(recent)),
            "max_drift": float(np.max(recent)),
            "std_drift": float(np.std(recent)),
            "conservation_rate": float(np.mean(recent < self.drift_threshold)),
            "conservation_violations": int(np.sum(recent >= self.drift_threshold)),
        }

    def plot_history(self, save_path: str | None = None) -> None:
        """
        Plot energy conservation history (requires matplotlib).

        Args:
            save_path: Path to save plot (if None, display only)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, cannot plot history")
            return

        if not self.energy_drifts:
            logger.warning("No data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{self.name} Energy Conservation History")

        # Plot 1: Energy drift over time
        ax = axes[0, 0]
        ax.plot(self.energy_drifts, label="Energy drift", alpha=0.7)
        ax.axhline(
            y=self.drift_threshold,
            color="r",
            linestyle="--",
            label=f"Threshold ({self.drift_threshold})",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("|H_final - H_initial|")
        ax.set_title("Energy Drift")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Initial vs. Final energy
        ax = axes[0, 1]
        ax.scatter(self.initial_energies, self.final_energies, alpha=0.5, s=10)
        min_e = min(min(self.initial_energies), min(self.final_energies))
        max_e = max(max(self.initial_energies), max(self.final_energies))
        ax.plot([min_e, max_e], [min_e, max_e], "r--", label="Perfect conservation")
        ax.set_xlabel("H_initial")
        ax.set_ylabel("H_final")
        ax.set_title("Energy Conservation")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Evolution time (if available)
        ax = axes[1, 0]
        if self.evolution_times:
            ax.plot(self.evolution_times, label="Evolution time (dt)", alpha=0.7)
            ax.set_xlabel("Step")
            ax.set_ylabel("dt")
            ax.set_title("Evolution Time")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No evolution time data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Evolution Time (N/A)")

        # Plot 4: State norm (if available)
        ax = axes[1, 1]
        if self.state_norms:
            ax.plot(self.state_norms, label="State norm", alpha=0.7)
            ax.set_xlabel("Step")
            ax.set_ylabel("||q|| + ||p||")
            ax.set_title("State Norm (Convergence)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No state norm data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("State Norm (N/A)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved energy conservation plot to {save_path}")
        else:
            plt.show()

        plt.close()


# Convenience function for creating monitors
def create_monitor(
    name: str, drift_threshold: float = 0.1, history_size: int = 1000
) -> EnergyConservationMonitor:
    """
    Factory function to create an energy conservation monitor.

    Args:
        name: Monitor name
        drift_threshold: Maximum acceptable energy drift
        history_size: Maximum number of steps to keep in history

    Returns:
        EnergyConservationMonitor instance
    """
    return EnergyConservationMonitor(
        name=name, drift_threshold=drift_threshold, history_size=history_size
    )
