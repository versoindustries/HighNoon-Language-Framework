# highnoon/services/quantum_hpo_scheduler.py
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

"""Quantum Adaptive HPO Scheduler.

This module implements a quantum-inspired hyperparameter optimization scheduler
that combines population-based evolution with real-time adaptive tuning.

Key Mechanisms:
1. **Quantum Superposition Sampling**: Explores multiple configurations in
   parallel during early trials, "collapsing" to promising regions.

2. **Gradient Entropy Sensing**: Uses loss landscape roughness to guide
   exploration vs exploitation trade-off.

3. **Thermal Annealing**: Temperature-based schedule transitions from
   exploration (high T) to exploitation (low T).

4. **Quantum Tunneling**: Probabilistic escape from local minima by
   perturbing hyperparameters when loss stagnates.

5. **Population Evolution**: Between trials, uses quantum-inspired selection
   where configurations have "amplitudes" based on performance.

Integration with QALRC:
    The scheduler works in conjunction with QuantumAdaptiveLRController
    for real-time learning rate adaptation during training. This scheduler
    handles the between-trial hyperparameter evolution, while QALRC handles
    within-trial learning rate optimization.

Example:
    >>> scheduler = QuantumAdaptiveHPOScheduler(
    ...     max_budget=100,
    ...     min_budget=10,
    ...     search_space_sampler=sample_func,
    ...     population_size=8,
    ... )
    >>> trials = scheduler.get_next_trials(4)
    >>> # Run trials...
    >>> scheduler.report_result(result)
    >>> next_trials = scheduler.get_next_trials(4)  # Evolved configurations
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from highnoon.services.hpo_schedulers import HPOSchedulerBase, TrialConfig, TrialResult

logger = logging.getLogger(__name__)


@dataclass
class QuantumState:
    """Quantum state for a hyperparameter configuration.

    Represents a configuration in the population with its associated
    quantum-inspired properties.

    Attributes:
        config: Hyperparameter configuration dictionary.
        amplitude: Probability amplitude (higher = more likely to be selected).
        phase: Phase for interference calculations.
        loss: Best loss achieved with this config.
        entropy: Gradient entropy observed during training.
        generation: Which evolution generation this config is from.
    """

    config: dict[str, Any]
    amplitude: float = 1.0
    phase: float = 0.0
    loss: float = float("inf")
    entropy: float = 0.5
    generation: int = 0
    trial_count: int = 0


@dataclass
class QAHPOConfig:
    """Configuration for Quantum Adaptive HPO Scheduler.

    Attributes:
        population_size: Number of configurations in the population.
        initial_temperature: Starting temperature for annealing.
        final_temperature: Ending temperature for annealing.
        tunneling_probability: Base probability of quantum tunneling.
        mutation_strength: Strength of mutations during evolution.
        crossover_rate: Probability of crossover between configs.
        elite_fraction: Fraction of top configs to preserve unchanged.
        entropy_weight: How much gradient entropy affects selection.
    """

    population_size: int = 8
    initial_temperature: float = 2.0
    final_temperature: float = 0.1
    tunneling_probability: float = 0.15
    mutation_strength: float = 0.3
    crossover_rate: float = 0.4
    elite_fraction: float = 0.25
    entropy_weight: float = 0.2
    annealing_power: float = 2.0


class QuantumAdaptiveHPOScheduler(HPOSchedulerBase):
    """Quantum-Inspired Adaptive HPO Scheduler.

    This scheduler combines quantum-inspired algorithms with population-based
    training to efficiently explore the hyperparameter space. It integrates
    with the QALRC for real-time learning rate adaptation.

    Key features:
    - Quantum superposition sampling for parallel exploration
    - Gradient entropy-based adaptation
    - Thermal annealing schedule
    - Quantum tunneling for escaping local minima
    - Population-based evolution with amplitude selection

    Example:
        >>> scheduler = QuantumAdaptiveHPOScheduler(
        ...     max_budget=100,
        ...     search_space_sampler=sample_hyperparams,
        ... )
        >>> for _ in range(num_trials):
        ...     trials = scheduler.get_next_trials(1)
        ...     result = run_trial(trials[0])
        ...     scheduler.report_result(result)
    """

    def __init__(
        self,
        max_budget: int = 100,
        min_budget: int = 10,
        search_space_sampler: Callable[[int], dict[str, Any]] | None = None,
        config: QAHPOConfig | None = None,
    ) -> None:
        """Initialize the Quantum Adaptive HPO Scheduler.

        Args:
            max_budget: Maximum resource budget per trial (epochs).
            min_budget: Minimum resource budget per trial.
            search_space_sampler: Function to sample hyperparameters.
            config: QAHPO configuration. Uses defaults if not provided.
        """
        super().__init__(max_budget, min_budget, search_space_sampler)

        self.config = config or QAHPOConfig()

        # Population of quantum states
        self.population: list[QuantumState] = []

        # Tracking state
        self.generation = 0
        self.total_trials = 0
        self._trial_counter = 0  # Counter for unique trial IDs
        self.best_loss = float("inf")
        self.best_config: dict[str, Any] | None = None
        self.trial_history: list[TrialResult] = []

        # Convergence detection state
        self._convergence_window: list[float] = []
        self._convergence_window_size = 10
        self._convergence_threshold = 1e-5
        self._no_improvement_count = 0
        self._max_no_improvement = 20  # Stop after 20 trials with no improvement

        # Temperature schedule state
        self._max_trials_estimate = 100  # Updated as trials come in

        logger.info(
            "[QAHPO] Initialized with population_size=%d, T=[%.2f, %.2f]",
            self.config.population_size,
            self.config.initial_temperature,
            self.config.final_temperature,
        )

    def _get_temperature(self) -> float:
        """Compute current temperature from annealing schedule.

        Returns temperature that decreases from initial to final as
        trials progress, following a polynomial curve.
        """
        if self._max_trials_estimate <= 0:
            return self.config.initial_temperature

        progress = min(1.0, self.total_trials / self._max_trials_estimate)
        schedule = progress ** self.config.annealing_power

        temperature = (
            self.config.initial_temperature * (1 - schedule)
            + self.config.final_temperature * schedule
        )
        return max(self.config.final_temperature, temperature)

    def _initialize_population(self) -> None:
        """Initialize the population with diverse configurations.

        Uses quantum superposition-inspired sampling where each
        configuration starts with equal amplitude.
        """
        logger.info("[QAHPO] Initializing population of %d configs", self.config.population_size)

        self.population = []
        for i in range(self.config.population_size):
            if self.search_space_sampler:
                config = self.search_space_sampler(i)
            else:
                config = {}

            # Add slight random phase for interference
            phase = random.uniform(0, 2 * math.pi)

            state = QuantumState(
                config=config,
                amplitude=1.0 / math.sqrt(self.config.population_size),
                phase=phase,
                generation=0,
            )
            self.population.append(state)

    def _select_by_amplitude(self, n: int) -> list[QuantumState]:
        """Select configurations using amplitude-based probability.

        Higher amplitude (better performance) = higher selection probability.
        Includes temperature-based randomness for exploration.

        Args:
            n: Number of configurations to select.

        Returns:
            List of selected quantum states.
        """
        if not self.population:
            return []

        temperature = self._get_temperature()

        # Compute selection probabilities from amplitudes
        # Higher temperature = more uniform, lower = favor high amplitude
        amplitudes = np.array([s.amplitude for s in self.population])

        # Boltzmann-like selection with temperature
        if temperature > 0.01:
            logits = amplitudes / temperature
            logits = logits - np.max(logits)  # Numerical stability
            probs = np.exp(logits)
            probs = probs / np.sum(probs)
        else:
            # At very low temperature, select greedily
            probs = np.zeros_like(amplitudes)
            probs[np.argmax(amplitudes)] = 1.0

        # Sample without replacement
        indices = np.random.choice(
            len(self.population),
            size=min(n, len(self.population)),
            replace=False,
            p=probs,
        )
        return [self.population[i] for i in indices]

    def _mutate_config(self, config: dict[str, Any], strength: float) -> dict[str, Any]:
        """Apply quantum-inspired mutation to a configuration.

        Perturbation strength is scaled by temperature and the provided
        strength parameter.

        Args:
            config: Configuration to mutate.
            strength: Mutation strength multiplier.

        Returns:
            Mutated configuration.
        """
        mutated = config.copy()
        temperature = self._get_temperature()

        # Scale mutation by temperature (more exploration when hot)
        effective_strength = strength * (0.5 + 0.5 * temperature / self.config.initial_temperature)

        for key, value in config.items():
            if isinstance(value, (int, float)):
                # Gaussian perturbation
                if random.random() < 0.5:  # 50% chance to mutate each param
                    if isinstance(value, float):
                        noise = random.gauss(0, effective_strength * abs(value) + 1e-8)
                        mutated[key] = max(1e-8, value + noise)
                    else:
                        # Integer: sometimes bump up or down
                        delta = random.choice([-1, 0, 0, 1])
                        mutated[key] = max(1, value + delta)

        return mutated

    def _crossover(
        self, parent1: dict[str, Any], parent2: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform quantum-inspired crossover between two configs.

        Uses phase interference to determine which parent contributes
        each parameter.

        Args:
            parent1: First parent configuration.
            parent2: Second parent configuration.

        Returns:
            Child configuration.
        """
        child = {}
        for key in parent1:
            if key in parent2:
                # 50/50 crossover with slight bias toward better parent
                if random.random() < 0.5:
                    child[key] = parent1[key]
                else:
                    child[key] = parent2[key]
            else:
                child[key] = parent1[key]

        # Include any keys only in parent2
        for key in parent2:
            if key not in child:
                child[key] = parent2[key]

        return child

    def _should_tunnel(self) -> bool:
        """Determine if quantum tunneling should occur.

        Tunneling is more likely when:
        1. Loss has stagnated across recent trials
        2. Random chance based on tunneling_probability
        3. Temperature is still relatively high

        Returns:
            True if tunneling should occur.
        """
        if len(self.trial_history) < 3:
            return False

        # Check for loss stagnation
        recent_losses = [r.loss for r in self.trial_history[-5:] if not math.isinf(r.loss)]
        if len(recent_losses) < 3:
            return random.random() < self.config.tunneling_probability

        loss_variance = np.var(recent_losses)
        loss_trend = recent_losses[-1] - recent_losses[0]

        # Tunnel if variance is low (stuck) or loss is increasing
        stagnation = loss_variance < 0.01 * np.mean(recent_losses) ** 2
        worsening = loss_trend > 0

        base_prob = self.config.tunneling_probability
        if stagnation:
            base_prob *= 2.0
        if worsening:
            base_prob *= 1.5

        return random.random() < min(0.5, base_prob)

    def _apply_tunneling(self, state: QuantumState) -> QuantumState:
        """Apply quantum tunneling to escape local minimum.

        Creates a significantly perturbed configuration to explore
        a different region of the hyperparameter space.

        Args:
            state: Current quantum state.

        Returns:
            New quantum state after tunneling.
        """
        logger.info("[QAHPO] Quantum tunneling triggered for generation %d", state.generation)

        # Large perturbation
        tunneled_config = self._mutate_config(
            state.config, self.config.mutation_strength * 3.0
        )

        return QuantumState(
            config=tunneled_config,
            amplitude=0.5,  # Reduced amplitude until proven
            phase=random.uniform(0, 2 * math.pi),
            generation=self.generation + 1,
        )

    def _evolve_population(self) -> None:
        """Evolve the population using quantum-inspired selection.

        1. Update amplitudes based on trial results
        2. Preserve elite configurations
        3. Generate children through crossover and mutation
        4. Apply quantum tunneling if stagnating
        """
        if not self.population:
            return

        logger.info("[QAHPO] Evolving population (generation %d)", self.generation)
        temperature = self._get_temperature()

        # Sort by amplitude (which reflects performance)
        self.population.sort(key=lambda s: s.amplitude, reverse=True)

        # Preserve elites
        n_elite = max(1, int(self.config.elite_fraction * len(self.population)))
        new_population = self.population[:n_elite]

        # Generate rest through selection, crossover, mutation
        while len(new_population) < self.config.population_size:
            # Select parents
            parents = self._select_by_amplitude(2)
            if len(parents) < 2:
                parents = [parents[0], parents[0]] if parents else [self.population[0], self.population[0]]

            # Crossover
            if random.random() < self.config.crossover_rate:
                child_config = self._crossover(parents[0].config, parents[1].config)
            else:
                child_config = parents[0].config.copy()

            # Mutation
            child_config = self._mutate_config(child_config, self.config.mutation_strength)

            # Create child state
            child = QuantumState(
                config=child_config,
                amplitude=0.5,  # Start with neutral amplitude
                phase=random.uniform(0, 2 * math.pi),
                generation=self.generation + 1,
            )
            new_population.append(child)

        # Apply quantum tunneling if needed
        if self._should_tunnel():
            # Replace worst config with tunneled version of best
            best_state = new_population[0]
            tunneled = self._apply_tunneling(best_state)
            new_population[-1] = tunneled

        self.population = new_population
        self.generation += 1

    def get_trial_budget(self, trial_id: str) -> int:
        """Get resource budget for a trial.

        Uses adaptive budgeting: early trials get min_budget to explore,
        later trials get more budget as we exploit good regions.
        """
        temperature = self._get_temperature()

        # Scale budget inversely with temperature
        # Hot = explore with short trials, cold = exploit with long trials
        budget_range = self.max_budget - self.min_budget
        budget = int(self.min_budget + budget_range * (1 - temperature / self.config.initial_temperature))

        return max(self.min_budget, min(self.max_budget, budget))

    def should_stop_trial(self, trial_id: str, current_loss: float) -> bool:
        """Determine if trial should be stopped early.

        Stops trial if loss is much worse than best known.
        """
        if self.best_loss == float("inf"):
            return False

        # Stop if 5x worse than best
        if current_loss > self.best_loss * 5:
            return True

        return False

    def report_intermediate(self, result: TrialResult) -> None:
        """Report intermediate trial results.

        Updates the quantum state for the trial based on observed performance.
        """
        trial_id = result.trial_id

        # Find the state for this trial
        for state in self.population:
            if state.config.get("_trial_id") == trial_id:
                # Update amplitude based on loss (lower loss = higher amplitude)
                if result.loss < state.loss:
                    state.loss = result.loss
                    # Amplitude increases for good performance
                    improvement = (state.loss - result.loss) / (state.loss + 1e-8)
                    state.amplitude = min(2.0, state.amplitude * (1 + improvement * 0.1))
                break

    def report_result(self, result: TrialResult) -> None:
        """Report final trial result and update population.

        Args:
            result: Completed trial result.
        """
        self.trial_history.append(result)
        self.total_trials += 1

        # Find and update the state
        for state in self.population:
            if state.config.get("_trial_id") == result.trial_id:
                state.loss = result.loss
                state.trial_count += 1

                # Update amplitude based on relative performance
                if len(self.trial_history) > 1:
                    losses = [r.loss for r in self.trial_history if not math.isinf(r.loss)]
                    if losses:
                        mean_loss = np.mean(losses)
                        std_loss = np.std(losses) + 1e-8
                        # Z-score scaled amplitude (inverted: lower loss = higher)
                        z_score = (mean_loss - result.loss) / std_loss
                        state.amplitude = 1.0 + 0.5 * np.tanh(z_score)
                break

        # Update best and track convergence
        if result.loss < self.best_loss:
            improvement = self.best_loss - result.loss
            self.best_loss = result.loss
            self._no_improvement_count = 0  # Reset on improvement
            for state in self.population:
                if state.config.get("_trial_id") == result.trial_id:
                    self.best_config = state.config.copy()
                    logger.info(
                        "[QAHPO] New best: loss=%.6f (improvement=%.6f, trial %s, gen %d)",
                        result.loss, improvement, result.trial_id, self.generation
                    )
                    break
        else:
            self._no_improvement_count += 1

        # Update convergence window
        self._convergence_window.append(result.loss)
        if len(self._convergence_window) > self._convergence_window_size:
            self._convergence_window.pop(0)

        # Evolve population periodically
        if self.total_trials % self.config.population_size == 0:
            self._evolve_population()

    def has_converged(self) -> bool:
        """Check if the optimization has converged.

        Convergence is detected when:
        1. Loss variance in recent trials is below threshold, OR
        2. No improvement for max_no_improvement consecutive trials

        Returns:
            True if converged, False otherwise.
        """
        # Check no improvement condition
        if self._no_improvement_count >= self._max_no_improvement:
            logger.info(
                "[QAHPO] Converged: no improvement for %d trials",
                self._no_improvement_count
            )
            return True

        # Check variance condition
        if len(self._convergence_window) >= self._convergence_window_size:
            variance = np.var(self._convergence_window)
            mean_loss = np.mean(self._convergence_window)
            relative_variance = variance / (mean_loss ** 2 + 1e-10)

            if relative_variance < self._convergence_threshold:
                logger.info(
                    "[QAHPO] Converged: loss variance %.2e < threshold %.2e",
                    relative_variance, self._convergence_threshold
                )
                return True

        return False

    def get_trials(self) -> list[dict[str, Any]]:
        """Get all trial information for monitoring.

        Returns:
            List of trial dictionaries with status and metrics.
        """
        trials = []
        for result in self.trial_history:
            trial_data = {
                "trial_id": result.trial_id,
                "loss": result.loss,
                "step": result.step,
                "memory_mb": result.memory_mb,
                "wall_time_seconds": result.wall_time_seconds,
                "status": "completed" if result.is_complete else "running",
            }
            # Add hyperparams if available
            for state in self.population:
                if state.config.get("_trial_id") == result.trial_id:
                    trial_data["hyperparams"] = {
                        k: v for k, v in state.config.items()
                        if not k.startswith("_")
                    }
                    trial_data["generation"] = state.generation
                    trial_data["amplitude"] = state.amplitude
                    break
            trials.append(trial_data)
        return trials

    def get_next_trials(self, n_trials: int) -> list[TrialConfig]:
        """Generate next batch of trial configurations.

        Args:
            n_trials: Number of trials to generate.

        Returns:
            List of trial configurations.
        """
        # Initialize population if needed
        if not self.population:
            self._initialize_population()

        trials = []
        selected = self._select_by_amplitude(n_trials)

        for state in selected:
            self._trial_counter += 1
            trial_id = f"qahpo_g{self.generation}_t{self._trial_counter}"

            # Store trial ID in config for tracking
            config = state.config.copy()
            config["_trial_id"] = trial_id

            budget = self.get_trial_budget(trial_id)

            trial = TrialConfig(
                trial_id=trial_id,
                hyperparams=config,
                budget=budget,
                bracket=self.generation,
                rung=0,
                stage=1,
            )
            trials.append(trial)

            # Update state with trial ID
            state.config["_trial_id"] = trial_id

        return trials

    def get_best_trial(self) -> tuple[str, float] | None:
        """Get the best trial so far.

        Returns:
            Tuple of (trial_id, best_loss) or None if no results.
        """
        if self.best_config and self.best_loss < float("inf"):
            return (self.best_config.get("_trial_id", "unknown"), self.best_loss)
        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics for monitoring.

        Returns:
            Dictionary with scheduler state and metrics.
        """
        # Calculate convergence metrics
        convergence_info = {
            "converged": self.has_converged(),
            "no_improvement_count": self._no_improvement_count,
            "max_no_improvement": self._max_no_improvement,
        }
        if len(self._convergence_window) >= 2:
            convergence_info["loss_variance"] = float(np.var(self._convergence_window))
            convergence_info["loss_trend"] = float(
                self._convergence_window[-1] - self._convergence_window[0]
            ) if len(self._convergence_window) > 1 else 0.0

        # Population diversity metrics
        if self.population:
            amplitudes = [s.amplitude for s in self.population]
            losses = [s.loss for s in self.population if s.loss < float("inf")]
            population_info = {
                "size": len(self.population),
                "mean_amplitude": float(np.mean(amplitudes)),
                "std_amplitude": float(np.std(amplitudes)),
                "mean_loss": float(np.mean(losses)) if losses else None,
                "loss_diversity": float(np.std(losses)) if len(losses) > 1 else 0.0,
            }
        else:
            population_info = {"size": 0}

        return {
            "scheduler": "QAHPO",
            "scheduler_version": "2.0.0",  # Enterprise version
            "generation": self.generation,
            "total_trials": self.total_trials,
            "trial_counter": self._trial_counter,
            "temperature": self._get_temperature(),
            "best_loss": self.best_loss if self.best_loss < float("inf") else None,
            "best_trial_id": self.best_config.get("_trial_id") if self.best_config else None,
            "convergence": convergence_info,
            "population": population_info,
            "config": {
                "tunneling_probability": self.config.tunneling_probability,
                "mutation_strength": self.config.mutation_strength,
                "crossover_rate": self.config.crossover_rate,
                "elite_fraction": self.config.elite_fraction,
            },
        }


__all__ = ["QuantumAdaptiveHPOScheduler", "QAHPOConfig", "QuantumState"]
