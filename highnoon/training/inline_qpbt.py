# highnoon/training/inline_qpbt.py
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

"""Inline Quantum Population-Based Training.

This module provides inline Population-Based Training (PBT) with quantum-inspired
evolution mechanisms. Instead of running separate trials, it maintains a "virtual
population" of hyperparameter configurations and evolves them during training.

Key Innovations:
    1. Amplitude-based selection: Configs have quantum amplitudes based on
       recent performance, used for probabilistic selection (Born rule).
    2. Quantum tunneling: Escape local minima by large perturbations when
       loss stagnates, inspired by quantum annealing.
    3. Phase interference: Crossover between configs uses phase to
       determine which parent contributes each parameter.
    4. Thermal annealing: Temperature decreases over training, shifting
       from exploration to exploitation.

Example:
    >>> config = {"lr": 3e-4, "galore_rank": 32}
    >>> qpbt = InlineQuantumPBT(config, population_size=8)
    >>> for step in training_loop:
    ...     current_config = qpbt.get_current_config()
    ...     # Apply config and train...
    ...     qpbt.record_performance(loss, step)
    ...     if qpbt.maybe_evolve(step):
    ...         print("Switched to new config")

Reference:
    Smart_Tuner_Upgrade.md - Section 6.4: Quantum-Enhanced Population-Based Training
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class InlineQPBTConfig:
    """Configuration for Inline Quantum PBT.

    Attributes:
        population_size: Number of virtual population members.
        mutation_strength: Base strength of mutations (log scale).
        crossover_rate: Probability of crossover during evolution.
        initial_temperature: Starting temperature for annealing.
        final_temperature: Ending temperature for annealing.
        evolution_interval_min: Minimum steps between evolutions.
        evolution_interval_max: Maximum steps between evolutions.
        tunneling_probability: Base probability of quantum tunneling.
        stagnation_threshold: Std threshold for detecting stagnation.
        performance_history_size: Size of performance history per config.
    """

    population_size: int = 8
    mutation_strength: float = 0.2
    crossover_rate: float = 0.3
    initial_temperature: float = 1.0
    final_temperature: float = 0.01
    evolution_interval_min: int = 50
    evolution_interval_max: int = 500
    tunneling_probability: float = 0.15
    stagnation_threshold: float = 1e-6
    performance_history_size: int = 20


# =============================================================================
# POPULATION MEMBER
# =============================================================================


@dataclass
class PopulationMember:
    """A single member of the virtual population.

    Attributes:
        config: Hyperparameter configuration dictionary.
        amplitude: Quantum amplitude (probability proportional to |amplitude|²).
        phase: Quantum phase for interference during crossover.
        performance_history: Recent loss values.
        total_steps: Total steps this config has been active.
        best_loss: Best loss achieved with this config.
    """

    config: dict[str, Any]
    amplitude: float = 1.0
    phase: float = 0.0
    performance_history: deque = field(default_factory=lambda: deque(maxlen=20))
    total_steps: int = 0
    best_loss: float = float("inf")

    def get_mean_loss(self) -> float:
        """Get mean loss from performance history."""
        if not self.performance_history:
            return float("inf")
        return float(np.mean(list(self.performance_history)))

    def get_loss_std(self) -> float:
        """Get loss standard deviation from performance history."""
        if len(self.performance_history) < 2:
            return 0.0
        return float(np.std(list(self.performance_history)))


# =============================================================================
# INLINE QUANTUM PBT
# =============================================================================


class InlineQuantumPBT:
    """Inline Population-Based Training with quantum-inspired evolution.

    Instead of running separate trials, maintains a "virtual population"
    of hyperparameter configurations and evolves them during training.

    The quantum-inspired mechanisms include:
    - Amplitude-based selection following the Born rule
    - Quantum tunneling for escaping local minima
    - Phase interference for crossover
    - Thermal annealing for exploration/exploitation balance

    Attributes:
        config: InlineQPBTConfig settings.
        population: List of PopulationMember objects.

    Example:
        >>> qpbt = InlineQuantumPBT({"lr": 3e-4, "galore_rank": 32})
        >>> config = qpbt.get_current_config()
        >>> qpbt.record_performance(loss=2.5, step=100)
        >>> if qpbt.maybe_evolve(step=100):
        ...     new_config = qpbt.get_current_config()
    """

    def __init__(
        self,
        initial_config: dict[str, Any],
        config: InlineQPBTConfig | None = None,
    ):
        """Initialize Inline Quantum PBT.

        Args:
            initial_config: Initial hyperparameter configuration.
            config: QPBT configuration settings.
        """
        self.config = config or InlineQPBTConfig()

        # Initialize population from initial config with perturbations
        self._population = self._init_population(initial_config)

        # Current active config index
        self._active_idx = 0

        # Step counter for annealing
        self._step = 0
        self._max_steps = 100000  # Will be updated

        # Last evolution step
        self._last_evolution_step = 0

        # Parameter bounds for clamping
        self._param_bounds = self._infer_bounds(initial_config)

        logger.info(
            "[InlineQPBT] Initialized: pop_size=%d, mutation=%.2f, crossover=%.2f",
            self.config.population_size,
            self.config.mutation_strength,
            self.config.crossover_rate,
        )

    def _init_population(self, initial_config: dict[str, Any]) -> list[PopulationMember]:
        """Initialize population from initial config with perturbations.

        Args:
            initial_config: Base configuration to perturb.

        Returns:
            List of PopulationMember objects.
        """
        population = []

        # First member is the original config
        population.append(
            PopulationMember(
                config=initial_config.copy(),
                amplitude=1.0 / np.sqrt(self.config.population_size),
                phase=0.0,
            )
        )

        # Generate perturbed variants
        for _ in range(1, self.config.population_size):
            perturbed = {}
            for key, value in initial_config.items():
                if isinstance(value, float):
                    # Log-scale perturbation
                    log_val = np.log(value + 1e-10)
                    perturbation = np.random.randn() * self.config.mutation_strength
                    perturbed[key] = np.exp(log_val + perturbation)
                elif isinstance(value, int):
                    # Multiplicative perturbation
                    factor = 0.5 + np.random.random()
                    perturbed[key] = max(1, int(value * factor))
                elif isinstance(value, bool):
                    # Random flip with low probability
                    perturbed[key] = value if np.random.random() > 0.1 else not value
                else:
                    perturbed[key] = value

            population.append(
                PopulationMember(
                    config=perturbed,
                    amplitude=1.0 / np.sqrt(self.config.population_size),
                    phase=np.random.uniform(0, 2 * np.pi),
                )
            )

        return population

    def _infer_bounds(self, config: dict[str, Any]) -> dict[str, tuple[float, float]]:
        """Infer reasonable bounds for each parameter.

        Args:
            config: Configuration to analyze.

        Returns:
            Dict mapping param names to (min, max) bounds.
        """
        bounds = {}
        for key, value in config.items():
            if isinstance(value, float):
                # Assume 2 orders of magnitude range
                bounds[key] = (value * 0.01, value * 100.0)
            elif isinstance(value, int):
                bounds[key] = (max(1, value // 4), value * 4)
        return bounds

    def set_max_steps(self, max_steps: int) -> None:
        """Set maximum training steps for annealing schedule.

        Args:
            max_steps: Total training steps.
        """
        self._max_steps = max_steps

    def get_current_config(self) -> dict[str, Any]:
        """Get the currently active hyperparameter config.

        Returns:
            Current configuration dictionary.
        """
        return self._population[self._active_idx].config.copy()

    def record_performance(self, loss: float, step: int) -> None:
        """Record performance of current config.

        Updates the amplitude based on loss (lower loss = higher amplitude).

        Args:
            loss: Current loss value.
            step: Current training step.
        """
        member = self._population[self._active_idx]
        member.performance_history.append(loss)
        member.total_steps += 1
        self._step = step

        if loss < member.best_loss:
            member.best_loss = loss

        # Update amplitude based on performance
        self._update_amplitude(self._active_idx, loss)

    def _update_amplitude(self, idx: int, loss: float) -> None:
        """Update quantum amplitude based on loss.

        Lower loss -> higher amplitude (Born rule probability).

        Args:
            idx: Population member index.
            loss: Current loss value.
        """
        # Convert loss to "fitness" (higher is better)
        fitness = 1.0 / (1.0 + loss)

        # EMA update of amplitude
        member = self._population[idx]
        member.amplitude = 0.9 * member.amplitude + 0.1 * np.sqrt(fitness)

        # Renormalize to valid quantum state
        self._normalize_amplitudes()

    def _normalize_amplitudes(self) -> None:
        """Normalize amplitudes to form valid quantum state."""
        total = sum(m.amplitude**2 for m in self._population)
        if total > 1e-10:
            norm = np.sqrt(total)
            for member in self._population:
                member.amplitude /= norm

    def _get_temperature(self) -> float:
        """Get current temperature for annealing.

        Returns:
            Current temperature (1.0 = hot/exploration, 0.0 = cold/exploitation).
        """
        if self._max_steps <= 0:
            return self.config.initial_temperature

        progress = min(1.0, self._step / self._max_steps)
        temp_range = self.config.initial_temperature - self.config.final_temperature
        return self.config.initial_temperature - temp_range * progress

    def maybe_evolve(self, step: int, force: bool = False) -> bool:
        """Potentially evolve population and switch config.

        Evolution includes:
        1. Quantum-inspired selection (Born rule)
        2. Quantum tunneling if stagnating
        3. Phase-based crossover with best config
        4. Light mutation

        Args:
            step: Current training step.
            force: Force evolution regardless of interval.

        Returns:
            True if config was switched.
        """
        # Evolution interval based on annealing schedule
        temperature = self._get_temperature()
        interval_range = self.config.evolution_interval_max - self.config.evolution_interval_min
        evolution_interval = int(
            self.config.evolution_interval_min + interval_range * (1 - temperature)
        )

        if not force and step - self._last_evolution_step < evolution_interval:
            return False

        self._last_evolution_step = step

        # Quantum-inspired selection: measure amplitude to select next config
        probabilities = np.array([m.amplitude**2 for m in self._population])
        probabilities = probabilities / (probabilities.sum() + 1e-10)

        # Add temperature-based randomness
        if temperature > 0.1:
            uniform = np.ones_like(probabilities) / len(probabilities)
            probabilities = (1 - temperature) * probabilities + temperature * uniform

        new_idx = np.random.choice(len(self._population), p=probabilities)

        # If selected config is underperforming, apply quantum tunneling
        if self._should_tunnel(new_idx):
            self._apply_tunneling(new_idx)

        # Crossover with exploitation probability
        if np.random.random() < self.config.crossover_rate * (1 - temperature):
            self._crossover(new_idx)

        # Light mutation
        self._mutate(new_idx, temperature)

        old_idx = self._active_idx
        self._active_idx = new_idx

        if old_idx != new_idx:
            logger.debug(
                "[InlineQPBT] Switched config %d -> %d (temp=%.2f)",
                old_idx,
                new_idx,
                temperature,
            )
            return True

        return False

    def _should_tunnel(self, idx: int) -> bool:
        """Determine if quantum tunneling should occur.

        Tunneling happens when a config shows stagnation (flat loss curve).

        Args:
            idx: Population member index.

        Returns:
            True if tunneling should be applied.
        """
        member = self._population[idx]
        history = list(member.performance_history)

        if len(history) < 5:
            return False

        # Check for stagnation
        recent = history[-5:]
        std = np.std(recent)
        return (
            std < self.config.stagnation_threshold
            and np.random.random() < self.config.tunneling_probability
        )

    def _apply_tunneling(self, idx: int) -> None:
        """Apply quantum tunneling: large random perturbation.

        This allows escaping local minima by making a large jump in
        hyperparameter space.

        Args:
            idx: Population member index.
        """
        member = self._population[idx]

        for key, value in member.config.items():
            if isinstance(value, float):
                # Large perturbation in log space
                log_val = np.log(value + 1e-10)
                log_val += np.random.randn() * 1.0  # Large perturbation
                new_val = np.exp(log_val)

                # Clamp to bounds
                if key in self._param_bounds:
                    low, high = self._param_bounds[key]
                    new_val = np.clip(new_val, low, high)

                member.config[key] = float(new_val)

            elif isinstance(value, int):
                factor = 0.5 + np.random.random()
                new_val = int(value * factor)

                # Clamp to bounds
                if key in self._param_bounds:
                    low, high = self._param_bounds[key]
                    new_val = int(np.clip(new_val, low, high))

                member.config[key] = max(1, new_val)

        logger.info("[InlineQPBT] Applied quantum tunneling to config %d", idx)

    def _crossover(self, idx: int) -> None:
        """Phase-based crossover with best performing config.

        Uses quantum phase interference to determine which parent
        contributes each parameter.

        Args:
            idx: Population member index to update.
        """
        # Find best config by amplitude
        best_idx = max(range(len(self._population)), key=lambda i: self._population[i].amplitude)

        if best_idx == idx:
            return

        best_member = self._population[best_idx]
        current_member = self._population[idx]

        # Phase interference determines inheritance
        phase_diff = current_member.phase - best_member.phase
        inheritance_prob = (1 + np.cos(phase_diff)) / 2  # [0, 1]

        for key in current_member.config:
            if key in best_member.config:
                if np.random.random() < inheritance_prob:
                    current_member.config[key] = best_member.config[key]

        # Evolve phase
        current_member.phase += np.random.randn() * 0.1

    def _mutate(self, idx: int, temperature: float) -> None:
        """Apply light mutation to a config.

        Mutation strength scales with temperature.

        Args:
            idx: Population member index.
            temperature: Current annealing temperature.
        """
        member = self._population[idx]

        # Mutation probability scales with temperature
        mutation_prob = 0.1 + 0.3 * temperature

        for key, value in member.config.items():
            if np.random.random() > mutation_prob:
                continue

            if isinstance(value, float):
                # Small log-scale perturbation
                log_val = np.log(value + 1e-10)
                perturbation = np.random.randn() * self.config.mutation_strength * temperature
                new_val = np.exp(log_val + perturbation)

                # Clamp to bounds
                if key in self._param_bounds:
                    low, high = self._param_bounds[key]
                    new_val = np.clip(new_val, low, high)

                member.config[key] = float(new_val)

            elif isinstance(value, int):
                if np.random.random() < 0.5:
                    delta = np.random.choice([-1, 1])
                    new_val = value + delta

                    if key in self._param_bounds:
                        low, high = self._param_bounds[key]
                        new_val = int(np.clip(new_val, low, high))

                    member.config[key] = max(1, new_val)

    def get_best_config(self) -> dict[str, Any]:
        """Get the best performing config from the population.

        Returns:
            Configuration with highest amplitude.
        """
        best_idx = max(
            range(len(self._population)),
            key=lambda i: self._population[i].amplitude,
        )
        return self._population[best_idx].config.copy()

    def get_population_summary(self) -> list[dict[str, Any]]:
        """Get summary of all population members.

        Returns:
            List of dicts with member statistics.
        """
        summaries = []
        for i, member in enumerate(self._population):
            summaries.append(
                {
                    "index": i,
                    "amplitude": member.amplitude,
                    "phase": member.phase,
                    "mean_loss": member.get_mean_loss(),
                    "best_loss": member.best_loss,
                    "total_steps": member.total_steps,
                    "is_active": i == self._active_idx,
                }
            )
        return summaries

    def get_statistics(self) -> dict[str, Any]:
        """Get QPBT statistics for logging/monitoring.

        Returns:
            Dictionary with current state and population statistics.
        """
        amplitudes = [m.amplitude for m in self._population]
        losses = [m.get_mean_loss() for m in self._population if m.performance_history]

        stats = {
            "population_size": len(self._population),
            "active_idx": self._active_idx,
            "current_step": self._step,
            "temperature": self._get_temperature(),
            "last_evolution_step": self._last_evolution_step,
            "amplitude_max": max(amplitudes) if amplitudes else 0.0,
            "amplitude_mean": np.mean(amplitudes) if amplitudes else 0.0,
            "loss_min": min(losses) if losses else float("inf"),
            "loss_mean": np.mean(losses) if losses else float("inf"),
        }

        return stats

    def reset(self, initial_config: dict[str, Any] | None = None) -> None:
        """Reset QPBT state.

        Args:
            initial_config: Optional new initial config.
        """
        if initial_config:
            self._population = self._init_population(initial_config)
            self._param_bounds = self._infer_bounds(initial_config)
        else:
            # Reset existing population
            for member in self._population:
                member.amplitude = 1.0 / np.sqrt(len(self._population))
                member.phase = np.random.uniform(0, 2 * np.pi)
                member.performance_history.clear()
                member.total_steps = 0
                member.best_loss = float("inf")

        self._active_idx = 0
        self._step = 0
        self._last_evolution_step = 0
        logger.info("[InlineQPBT] State reset")


__all__ = [
    "InlineQuantumPBT",
    "InlineQPBTConfig",
    "PopulationMember",
]
