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
from dataclasses import dataclass
from typing import Any

import numpy as np

from highnoon.services.hpo_manager import (
    DEFAULT_LR_RANGE,
    OPTIMIZER_LR_RANGES,
    estimate_model_params,
)
from highnoon.services.hpo_schedulers import HPOSchedulerBase, TrialConfig, TrialResult
from highnoon.services.hpo_utils import next_power_of_2, snap_to_multiple

# Import new enhancement modules
try:
    from highnoon.services.hpo_surrogate import ConstrainedAcquisition, TPESurrogate

    TPE_AVAILABLE = True
except ImportError:
    TPE_AVAILABLE = False

try:
    from highnoon.services.hpo_multi_fidelity import HyperbandSchedule

    HYPERBAND_AVAILABLE = True
except ImportError:
    HYPERBAND_AVAILABLE = False

try:
    from highnoon.services.hpo_meta_cache import MetaLearningCache

    META_CACHE_AVAILABLE = True
except ImportError:
    META_CACHE_AVAILABLE = False

try:
    from highnoon.services.hpo_importance import HyperparameterImportanceAnalyzer

    FANOVA_AVAILABLE = True
except ImportError:
    FANOVA_AVAILABLE = False

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

    Enhanced Attributes (BOHB/DEHB/c-TPE/PBT/Meta-Learning):
        mutation_strategy: DE mutation strategy ("gaussian", "de_rand_1", "de_best_1")
        de_F: Differential Evolution scaling factor
        de_CR: Differential Evolution crossover rate
        enable_multi_fidelity: Enable Hyperband-style successive halving
        fidelity_eta: Hyperband reduction factor (keep top 1/eta)
        param_budget: Maximum parameter count constraint (c-TPE)
        budget_penalty_lambda: Penalty strength for budget violations
        enable_tpe_surrogate: Use TPE surrogate model for informed sampling
        tpe_gamma: Good/bad partition ratio for TPE
        enable_pbt: Enable Population-Based Training weight transfer
        pbt_exploit_threshold: Loss ratio to trigger exploit (0.8 = 20% worse)
        enable_meta_learning: Use cross-sweep knowledge transfer
        meta_warm_start_count: Number of prior configs for warm-start
    """

    # Core quantum-inspired parameters
    population_size: int = 8
    initial_temperature: float = 2.0
    final_temperature: float = 0.1
    tunneling_probability: float = 0.15
    mutation_strength: float = 0.3
    crossover_rate: float = 0.4
    elite_fraction: float = 0.25
    entropy_weight: float = 0.2
    annealing_power: float = 2.0

    # DEHB: Differential Evolution mutation
    mutation_strategy: str = "de_rand_1"  # "gaussian", "de_rand_1", "de_best_1"
    de_F: float = 0.8  # DE scaling factor
    de_CR: float = 0.9  # DE crossover rate

    # BOHB: Multi-fidelity scheduling
    enable_multi_fidelity: bool = True
    fidelity_eta: int = 3  # Halving rate

    # c-TPE: Constrained acquisition
    param_budget: int | None = None  # Parameter count limit
    budget_penalty_lambda: float = 1.0

    # TPE Surrogate
    enable_tpe_surrogate: bool = True
    tpe_gamma: float = 0.25  # Top 25% are "good"

    # PBT: Population-Based Training
    enable_pbt: bool = False  # Disabled by default (requires checkpoint hooks)
    pbt_exploit_threshold: float = 0.8  # Exploit if 20% worse than best

    # Meta-Learning
    enable_meta_learning: bool = True
    meta_warm_start_count: int = 4

    # fANOVA: Importance-guided mutation
    # Phase 500: Aggressive settings for faster layer importance learning
    enable_fanova_guidance: bool = True
    fanova_refit_interval: int = 5  # Refit importance every N trials (was 10)
    fanova_min_trials: int = 5  # Min trials before using importance (was 10)
    fanova_importance_boost: float = 3.0  # Multiply mutation for important params (was 2.0)
    fanova_importance_dampen: float = 0.3  # Multiply mutation for unimportant params
    fanova_importance_threshold: float = 0.1  # Below this = unimportant


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
        task_hash: str | None = None,
        hpo_search_space: Any | None = None,  # Phase 500: HPOSearchSpace for adaptive progression
    ) -> None:
        """Initialize the Quantum Adaptive HPO Scheduler.

        Args:
            max_budget: Maximum resource budget per trial (epochs).
            min_budget: Minimum resource budget per trial.
            search_space_sampler: Function to sample hyperparameters.
            config: QAHPO configuration. Uses defaults if not provided.
            task_hash: Task hash for meta-learning (optional).
            hpo_search_space: HPOSearchSpace instance for faNOVA-adaptive progression.
        """
        super().__init__(max_budget, min_budget, search_space_sampler)

        self.config = config or QAHPOConfig()
        self._task_hash = task_hash
        self._hpo_search_space = hpo_search_space  # For importance score sharing

        # Population of quantum states
        self.population: list[QuantumState] = []

        # Tracking state
        self.generation = 0
        self.total_trials = 0
        self._trial_counter = 0  # Counter for unique trial IDs
        self.best_loss = float("inf")
        self.best_config: dict[str, Any] | None = None
        self.trial_history: list[TrialResult] = []
        self._config_history: list[dict[str, Any]] = []  # For surrogate training
        self._used_fingerprints: set[str] = set()  # Global tracking for duplicate prevention

        # Convergence detection state
        self._convergence_window: list[float] = []
        self._convergence_window_size = 10
        self._convergence_threshold = 1e-5
        self._no_improvement_count = 0
        self._max_no_improvement = 20  # Stop after 20 trials with no improvement

        # Temperature schedule state
        self._max_trials_estimate = 100  # Updated as trials come in

        # ====================================================================
        # ENHANCEMENT 1: TPE Surrogate + c-TPE Constrained Acquisition
        # ====================================================================
        self._surrogate: TPESurrogate | None = None
        self._constrained_acq: ConstrainedAcquisition | None = None

        if self.config.enable_tpe_surrogate and TPE_AVAILABLE:
            self._surrogate = TPESurrogate(
                gamma=self.config.tpe_gamma,
                min_observations=8,
            )
            self._constrained_acq = ConstrainedAcquisition(
                tpe_surrogate=self._surrogate,
                param_budget=self.config.param_budget,
                budget_penalty_lambda=self.config.budget_penalty_lambda,
            )
            logger.info("[QAHPO] TPE surrogate + c-TPE enabled")

        # ====================================================================
        # ENHANCEMENT 2: BOHB Multi-Fidelity (Hyperband)
        # ====================================================================
        self._hyperband: HyperbandSchedule | None = None

        if self.config.enable_multi_fidelity and HYPERBAND_AVAILABLE:
            self._hyperband = HyperbandSchedule(
                max_epochs=max_budget,
                min_epochs=min_budget,
                eta=self.config.fidelity_eta,
            )
            logger.info("[QAHPO] BOHB multi-fidelity enabled (η=%d)", self.config.fidelity_eta)

        # ====================================================================
        # ENHANCEMENT 3: Meta-Learning Cache
        # ====================================================================
        self._meta_cache: MetaLearningCache | None = None

        if self.config.enable_meta_learning and META_CACHE_AVAILABLE:
            self._meta_cache = MetaLearningCache()
            logger.info("[QAHPO] Meta-learning cache enabled")

        # ====================================================================
        # ENHANCEMENT 4: DEHB State (DE mutation tracking)
        # ====================================================================
        self._de_best_state: QuantumState | None = None  # Best individual for de_best_1

        # ====================================================================
        # ENHANCEMENT 5: fANOVA Importance-Guided Mutation
        # ====================================================================
        self._fanova: HyperparameterImportanceAnalyzer | None = None
        self._importance_scores: dict[str, float] = {}  # param_name -> importance
        self._last_fanova_fit: int = 0  # Trial count at last refit

        if self.config.enable_fanova_guidance and FANOVA_AVAILABLE:
            self._fanova = HyperparameterImportanceAnalyzer(
                n_trees=50,  # Lighter than visualization (faster)
                min_trials=self.config.fanova_min_trials,
            )
            logger.info("[QAHPO] fANOVA importance-guided mutation enabled")

        logger.info(
            "[QAHPO] Initialized with population_size=%d, T=[%.2f, %.2f], "
            "mutation=%s, multi_fidelity=%s, tpe=%s, meta=%s, fanova=%s",
            self.config.population_size,
            self.config.initial_temperature,
            self.config.final_temperature,
            self.config.mutation_strategy,
            self.config.enable_multi_fidelity,
            self.config.enable_tpe_surrogate,
            self.config.enable_meta_learning,
            self.config.enable_fanova_guidance,
        )

    def _get_temperature(self) -> float:
        """Compute current temperature from annealing schedule.

        Returns temperature that decreases from initial to final as
        trials progress, following a polynomial curve.
        """
        if self._max_trials_estimate <= 0:
            return self.config.initial_temperature

        progress = min(1.0, self.total_trials / self._max_trials_estimate)
        schedule = progress**self.config.annealing_power

        temperature = (
            self.config.initial_temperature * (1 - schedule)
            + self.config.final_temperature * schedule
        )
        return max(self.config.final_temperature, temperature)

    def _initialize_population(self) -> None:
        """Initialize the population with diverse configurations.

        Uses quantum superposition-inspired sampling where each
        configuration starts with equal amplitude. Learning rate is
        clamped to optimizer-specific bounds after sampling.

        Budget-aware: Uses spread trial IDs (0, 5, 10, 15...) to ensure
        HPOSearchSpace.sample() produces a range of config sizes from
        the progressive budget logic (10-70% of param_budget).
        """
        logger.info("[QAHPO] Initializing population of %d configs", self.config.population_size)

        self.population = []
        for i in range(self.config.population_size):
            if self.search_space_sampler:
                # Use spread trial IDs to get budget-aware configs
                # Early IDs (0-7 at step 5 = 0,5,10,15,20,25,30,35) → diverse budget targets
                # This ensures population has a mix of small and medium configs
                trial_id = i * 5  # Spread across progressive budget tiers
                config = self.search_space_sampler(trial_id)
                logger.debug(f"[QAHPO] Population[{i}] sampled with trial_id={trial_id}")
            else:
                config = {}

            # Clamp LR to optimizer-specific bounds after sampling
            config = self._clamp_learning_rate(config)

            # Enforce architecture constraints (multiples of 8, powers of 2)
            config = self._enforce_architecture_constraints(config)

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

        c-TPE Enhancement: Penalizes oversized configs using constraint probability.

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

        # === c-TPE: Apply budget constraint penalty ===
        if self._constrained_acq is not None and self.config.param_budget is not None:
            constraint_probs = []
            for state in self.population:
                # Estimate params for this config
                try:
                    estimated = estimate_model_params(state.config)
                    prob = self._constrained_acq.compute_constraint_probability(estimated)

                    # Record for learning
                    violated = estimated > self.config.param_budget
                    self._constrained_acq.record_constraint_violation(
                        state.config, estimated, violated
                    )
                except Exception:
                    prob = 1.0  # Assume feasible if estimation fails

                constraint_probs.append(prob)

            constraint_probs = np.array(constraint_probs)
            amplitudes = amplitudes * constraint_probs  # Penalize oversized

        # === TPE Surrogate: Boost configs in good region ===
        if self._surrogate is not None and self._surrogate._is_fitted:
            acq_scores = []
            for state in self.population:
                try:
                    acq = self._surrogate.acquisition(state.config)
                    acq_scores.append(acq)
                except Exception:
                    acq_scores.append(1.0)

            acq_scores = np.array(acq_scores)
            # Normalize and blend with amplitudes
            if acq_scores.max() > acq_scores.min():
                acq_norm = (acq_scores - acq_scores.min()) / (
                    acq_scores.max() - acq_scores.min() + 1e-8
                )
                amplitudes = amplitudes * (0.5 + 0.5 * acq_norm)

        # Boltzmann-like selection with temperature
        if temperature > 0.01:
            logits = amplitudes / temperature
            logits = logits - np.max(logits)  # Numerical stability
            probs = np.exp(logits)
            probs = probs / (np.sum(probs) + 1e-10)
        else:
            # At very low temperature, select greedily
            probs = np.zeros_like(amplitudes)
            probs[np.argmax(amplitudes)] = 1.0

        # Ensure valid probability distribution
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / np.sum(probs)

        # Sample without replacement
        indices = np.random.choice(
            len(self.population),
            size=min(n, len(self.population)),
            replace=False,
            p=probs,
        )
        return [self.population[i] for i in indices]

    def _clamp_learning_rate(self, config: dict[str, Any]) -> dict[str, Any]:
        """Clamp learning_rate to optimizer-specific safe bounds.

        This prevents QAHPO mutations from pushing LR outside the safe range
        for sensitive optimizers like SympFlowQNG which are prone to gradient
        explosions at high learning rates.

        Args:
            config: Configuration dictionary (may contain optimizer and learning_rate).

        Returns:
            Configuration with learning_rate clamped to optimizer-appropriate range.
        """
        if "learning_rate" not in config:
            return config

        optimizer = config.get("optimizer", "").lower()
        lr_min, lr_max = OPTIMIZER_LR_RANGES.get(optimizer, DEFAULT_LR_RANGE)
        original_lr = config["learning_rate"]
        clamped_lr = max(lr_min, min(lr_max, original_lr))

        if clamped_lr != original_lr:
            logger.debug(
                "[QAHPO] Clamped LR for %s: %.2e -> %.2e (bounds: %.2e - %.2e)",
                optimizer or "unknown",
                original_lr,
                clamped_lr,
                lr_min,
                lr_max,
            )

        config["learning_rate"] = clamped_lr
        return config

    def _enforce_architecture_constraints(self, config: dict[str, Any]) -> dict[str, Any]:
        """Enforce divisibility and power-of-2 constraints on architecture.

        This ensures that embedding_dim % num_heads == 0 and other
        constraints are met, preventing ValueError in model layers.

        Phase 2.1 Update: Delegates to HPOSearchSpace._validate_and_fix_constraints
        for comprehensive checking (ordering, divisibility, memory budget).

        Args:
            config: Configuration dictionary to validate/adjust.

        Returns:
            Configuration with snapped architecture parameters.
        """
        # Phase 2.1: Use comprehensive validator from HPOSearchSpace if available
        if self._hpo_search_space is not None and hasattr(
            self._hpo_search_space, "_validate_and_fix_constraints"
        ):
            # This handles ordering, divisibility, proportionality, and memory budget
            config = self._hpo_search_space._validate_and_fix_constraints(config)

        # 1. Snap embedding and hidden dimensions to multiples of 8
        for key in ["embedding_dim", "hidden_dim", "compressed_dim", "latent_kv_dim"]:
            if key in config:
                config[key] = snap_to_multiple(config[key], 8, min_val=32)

        # 2. Ensure num_heads is a power of 2 (if it's being tuned)
        for key in ["num_heads", "quantum_gqa_num_heads", "wlam_num_heads"]:
            if key in config:
                config[key] = next_power_of_2(config[key])

        # 3. Special case: QuantumGQA requires num_heads % num_kv_heads == 0
        if "quantum_gqa_num_heads" in config and "num_kv_heads" in config:
            kv_heads = config["num_kv_heads"]
            if config["quantum_gqa_num_heads"] % kv_heads != 0:
                # Snap to next multiple of kv_heads
                config["quantum_gqa_num_heads"] = snap_to_multiple(
                    config["quantum_gqa_num_heads"], kv_heads, min_val=kv_heads
                )

        # 4. HD Superposition/Embedding dimensions usually expect power of 2
        if "hde_hd_dim" in config:
            config["hde_hd_dim"] = next_power_of_2(config["hde_hd_dim"])

        return config

    def _mutate_config(self, config: dict[str, Any], strength: float) -> dict[str, Any]:
        """Apply mutation to a configuration based on strategy.

        Supports multiple mutation strategies:
        - "gaussian": Original quantum-inspired Gaussian perturbation
        - "de_rand_1": DE/rand/1/bin differential evolution
        - "de_best_1": DE/best/1/bin differential evolution

        Args:
            config: Configuration to mutate.
            strength: Mutation strength multiplier.

        Returns:
            Mutated configuration with LR clamped to safe bounds.
        """
        strategy = self.config.mutation_strategy

        if strategy == "de_rand_1" and len(self.population) >= 3:
            mutated = self._de_rand_1_mutation(config)
        elif strategy == "de_best_1" and len(self.population) >= 2 and self._de_best_state:
            mutated = self._de_best_1_mutation(config)
        else:
            # Fallback to Gaussian mutation
            mutated = self._gaussian_mutation(config, strength)

        # Clamp learning_rate to optimizer-specific safe bounds
        mutated = self._clamp_learning_rate(mutated)

        # Enforce architecture constraints
        mutated = self._enforce_architecture_constraints(mutated)

        return mutated

    def _gaussian_mutation(self, config: dict[str, Any], strength: float) -> dict[str, Any]:
        """Apply Gaussian perturbation mutation with importance weighting.

        When fANOVA importance is available, mutation strength is scaled:
        - Important params (>threshold): boost mutation for more exploration
        - Unimportant params (<threshold): dampen mutation, less exploration

        Args:
            config: Configuration to mutate.
            strength: Mutation strength multiplier.

        Returns:
            Mutated configuration.
        """
        mutated = config.copy()
        temperature = self._get_temperature()

        # Scale mutation by temperature (more exploration when hot)
        base_strength = strength * (0.5 + 0.5 * temperature / self.config.initial_temperature)

        for key, value in config.items():
            if key.startswith("_"):
                continue  # Skip internal keys
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # 50% chance to mutate each param
                if random.random() < 0.5:
                    # === fANOVA Importance-Guided Mutation ===
                    param_strength = base_strength
                    if self._importance_scores:
                        importance = self._importance_scores.get(key, 0.0)
                        if importance > self.config.fanova_importance_threshold:
                            # Important param: explore more aggressively
                            param_strength *= self.config.fanova_importance_boost
                        else:
                            # Unimportant param: explore less
                            param_strength *= self.config.fanova_importance_dampen

                    if isinstance(value, float):
                        noise = random.gauss(0, param_strength * abs(value) + 1e-8)
                        mutated[key] = max(1e-8, value + noise)
                    else:
                        # Integer: sometimes bump up or down
                        delta = random.choice([-1, 0, 0, 1])
                        mutated[key] = max(1, value + delta)

        return mutated

    def get_reduction_guidance(self) -> dict[str, float]:
        """Get guidance on which parameters to reduce based on fANOVA importance.

        Returns:
            Dictionary mapping parameter names to importance scores.
        """
        if self._fanova and self._importance_scores:
            return self._importance_scores
        return {}

    def _de_rand_1_mutation(self, config: dict[str, Any]) -> dict[str, Any]:
        """DE/rand/1/bin mutation operator.

        Creates mutant: x_mutant = x_a + F * (x_b - x_c)
        where a, b, c are random distinct individuals from population.

        Args:
            config: Base configuration for crossover.

        Returns:
            Mutated configuration.
        """
        if len(self.population) < 3:
            return self._gaussian_mutation(config, self.config.mutation_strength)

        # Select 3 random distinct individuals
        selected = random.sample(self.population, 3)
        a, b, c = selected[0].config, selected[1].config, selected[2].config

        F = self.config.de_F
        CR = self.config.de_CR

        mutated = config.copy()

        for key in config:
            if key.startswith("_"):
                continue

            # Crossover decision
            if random.random() < CR:
                val_a = a.get(key)
                val_b = b.get(key)
                val_c = c.get(key)

                if all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in [val_a, val_b, val_c]
                    if v is not None
                ):
                    # DE mutation: x_a + F * (x_b - x_c)
                    diff = (val_b or 0) - (val_c or 0)
                    new_val = (val_a or config[key]) + F * diff

                    if isinstance(config[key], int):
                        mutated[key] = max(1, int(round(new_val)))
                    else:
                        mutated[key] = max(1e-8, new_val)
                elif all(v is not None for v in [val_a, val_b, val_c]):
                    # Categorical: majority vote
                    from collections import Counter

                    votes = Counter([val_a, val_b, val_c])
                    mutated[key] = votes.most_common(1)[0][0]

        return mutated

    def _de_best_1_mutation(self, config: dict[str, Any]) -> dict[str, Any]:
        """DE/best/1/bin mutation operator.

        Creates mutant: x_mutant = x_best + F * (x_a - x_b)
        where x_best is the best individual found so far.

        Args:
            config: Base configuration for crossover.

        Returns:
            Mutated configuration.
        """
        if len(self.population) < 2 or self._de_best_state is None:
            return self._gaussian_mutation(config, self.config.mutation_strength)

        # Select 2 random distinct individuals
        selected = random.sample(self.population, 2)
        a, b = selected[0].config, selected[1].config
        best = self._de_best_state.config

        F = self.config.de_F
        CR = self.config.de_CR

        mutated = config.copy()

        for key in config:
            if key.startswith("_"):
                continue

            # Crossover decision
            if random.random() < CR:
                val_best = best.get(key)
                val_a = a.get(key)
                val_b = b.get(key)

                if all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in [val_best, val_a, val_b]
                    if v is not None
                ):
                    # DE mutation: x_best + F * (x_a - x_b)
                    diff = (val_a or 0) - (val_b or 0)
                    new_val = (val_best or config[key]) + F * diff

                    if isinstance(config[key], int):
                        mutated[key] = max(1, int(round(new_val)))
                    else:
                        mutated[key] = max(1e-8, new_val)

        return mutated

    def _crossover(self, parent1: dict[str, Any], parent2: dict[str, Any]) -> dict[str, Any]:
        """Perform quantum-inspired crossover between two configs.

        Uses phase interference to determine which parent contributes
        each parameter. Learning rate is clamped to optimizer-specific
        bounds after crossover.

        Args:
            parent1: First parent configuration.
            parent2: Second parent configuration.

        Returns:
            Child configuration with LR clamped to safe bounds.
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

        # Clamp LR to optimizer-specific bounds after crossover
        child = self._clamp_learning_rate(child)

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
        tunneled_config = self._mutate_config(state.config, self.config.mutation_strength * 3.0)

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
                parents = (
                    [parents[0], parents[0]]
                    if parents
                    else [self.population[0], self.population[0]]
                )

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

        With multi-fidelity (BOHB): Returns budget from Hyperband schedule.
        Without: Uses adaptive budgeting based on temperature.

        Args:
            trial_id: Trial identifier.

        Returns:
            Number of epochs to train.
        """
        # === BOHB Multi-Fidelity: Use Hyperband schedule ===
        if self._hyperband is not None:
            bracket = self._hyperband.get_active_bracket()
            if bracket is not None:
                budget = bracket.current_epochs
                logger.debug(
                    "[QAHPO-BOHB] Trial %s: fidelity level %d, budget=%d epochs",
                    trial_id,
                    bracket.current_level,
                    budget,
                )
                return budget

        # Fallback: Adaptive budgeting based on temperature
        temperature = self._get_temperature()

        # Scale budget inversely with temperature
        # Hot = explore with short trials, cold = exploit with long trials
        budget_range = self.max_budget - self.min_budget
        budget = int(
            self.min_budget + budget_range * (1 - temperature / self.config.initial_temperature)
        )

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

        Also updates:
        - TPE surrogate model with new observation
        - DE best state for de_best_1 mutation
        - Multi-fidelity bracket promotions

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

                # Store config for surrogate/history
                self._config_history.append(state.config.copy())
                break

        # Update best and track convergence
        if result.loss < self.best_loss:
            improvement = self.best_loss - result.loss
            self.best_loss = result.loss
            self._no_improvement_count = 0  # Reset on improvement
            for state in self.population:
                if state.config.get("_trial_id") == result.trial_id:
                    self.best_config = state.config.copy()

                    # === DEHB: Update best state for de_best_1 ===
                    self._de_best_state = state

                    logger.info(
                        "[QAHPO] New best: loss=%.6f (improvement=%.6f, trial %s, gen %d)",
                        result.loss,
                        improvement,
                        result.trial_id,
                        self.generation,
                    )
                    break
        else:
            self._no_improvement_count += 1

        # Update convergence window
        self._convergence_window.append(result.loss)
        if len(self._convergence_window) > self._convergence_window_size:
            self._convergence_window.pop(0)

        # === TPE SURROGATE: Add observation and refit ===
        if self._surrogate is not None and self.best_config is not None:
            # Find config for this trial
            trial_config = None
            for state in self.population:
                if state.config.get("_trial_id") == result.trial_id:
                    trial_config = state.config
                    break

            if trial_config:
                self._surrogate.add_observation(trial_config, result.loss)

                # Refit periodically
                if self.total_trials % 5 == 0:
                    self._surrogate.fit()

        # === MULTI-FIDELITY: Report to Hyperband ===
        if self._hyperband is not None:
            bracket_id = None
            for state in self.population:
                if state.config.get("_trial_id") == result.trial_id:
                    bracket_id = state.config.get("_bracket_id")
                    trial_config = state.config
                    break

            promoted = self._hyperband.report_result(
                trial_config if trial_config else {},
                result.loss,
                bracket_id=bracket_id,
            )

            if promoted:
                logger.info(
                    "[QAHPO-BOHB] Hyperband promoted %d configs to next fidelity",
                    len(promoted),
                )

        # === fANOVA: Refit importance scores periodically ===
        if self._fanova is not None and self.total_trials >= self.config.fanova_min_trials:
            # Check if we should refit
            if self.total_trials - self._last_fanova_fit >= self.config.fanova_refit_interval:
                self._refit_fanova()

        # Evolve population periodically
        if self.total_trials % self.config.population_size == 0:
            self._evolve_population()

    def report_budget_violation(
        self,
        config: dict[str, Any],
        estimated_params: int,
    ) -> None:
        """Report that a config exceeded param_budget for c-TPE learning.

        This enables the constrained acquisition function to learn from
        real budget violations, not just estimates at selection time.
        Also penalizes matching configs in the population by reducing
        their amplitude.

        Args:
            config: The hyperparameter configuration that violated budget.
            estimated_params: Actual parameter count that exceeded budget.
        """
        if self._constrained_acq is not None and self.config.param_budget is not None:
            violated = estimated_params > self.config.param_budget
            self._constrained_acq.record_constraint_violation(
                config,
                estimated_params,
                violated=violated,
            )
            logger.info(
                f"[QAHPO] Recorded budget violation: {estimated_params / 1e6:.1f}M > "
                f"{self.config.param_budget / 1e6:.1f}M"
            )
        elif self.config.param_budget is not None:
            logger.warning(
                f"[QAHPO] Budget violation not recorded (c-TPE not enabled): "
                f"{estimated_params / 1e6:.1f}M > {self.config.param_budget / 1e6:.1f}M"
            )

        # Reduce amplitude for oversized configs in population to avoid re-selection
        for state in self.population:
            if self._configs_similar(state.config, config):
                old_amplitude = state.amplitude
                state.amplitude *= 0.3  # Heavy penalty for budget violation
                logger.debug(
                    f"[QAHPO] Reduced amplitude for oversized config: "
                    f"{old_amplitude:.3f} → {state.amplitude:.3f}"
                )
                break

    def _configs_similar(self, config1: dict[str, Any], config2: dict[str, Any]) -> bool:
        """Check if two configs are similar in architecture parameters.

        Compares key architecture parameters that affect model size.

        Args:
            config1: First configuration.
            config2: Second configuration.

        Returns:
            True if architectures are similar.
        """
        arch_keys = [
            "num_reasoning_blocks",
            "num_moe_experts",
            "hidden_dim",
            "embedding_dim",
        ]
        for key in arch_keys:
            v1 = config1.get(key)
            v2 = config2.get(key)
            if v1 is not None and v2 is not None and v1 != v2:
                return False
        return True

    def _refit_fanova(self) -> None:
        """Refit fANOVA importance analyzer on completed trials.

        Updates self._importance_scores which is used by _gaussian_mutation
        to weight mutation strength per parameter.

        Called periodically (every fanova_refit_interval trials).
        """
        if self._fanova is None:
            return

        # Collect completed trial configs and losses
        configs = []
        losses = []

        for result in self.trial_history:
            if result.loss is not None and not math.isinf(result.loss):
                # Find matching config
                for hist_config in self._config_history:
                    if hist_config.get("_trial_id") == result.trial_id:
                        # Strip internal keys
                        clean_config = {
                            k: v for k, v in hist_config.items() if not k.startswith("_")
                        }
                        configs.append(clean_config)
                        losses.append(result.loss)
                        break

        if len(configs) < self.config.fanova_min_trials:
            return

        # Fit fANOVA
        success = self._fanova.fit(configs, losses)

        if success:
            self._last_fanova_fit = self.total_trials
            result = self._fanova.get_importance()

            if result is not None:
                # Update importance scores dict
                self._importance_scores = {p.name: p.importance for p in result.individual}

                # Phase 500: Share importance with HPOSearchSpace for adaptive progression
                if self._hpo_search_space is not None:
                    if hasattr(self._hpo_search_space, "update_importance_scores"):
                        self._hpo_search_space.update_importance_scores(self._importance_scores)

                # Log top importance
                top_params = sorted(result.individual, key=lambda x: x.importance, reverse=True)[:3]
                logger.info(
                    "[QAHPO-fANOVA] Updated importance (R²=%.2f): %s",
                    result.explained_variance,
                    ", ".join(f"{p.name}={p.importance*100:.1f}%" for p in top_params),
                )

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
                "[QAHPO] Converged: no improvement for %d trials", self._no_improvement_count
            )
            return True

        # Check variance condition
        if len(self._convergence_window) >= self._convergence_window_size:
            variance = np.var(self._convergence_window)
            mean_loss = np.mean(self._convergence_window)
            relative_variance = variance / (mean_loss**2 + 1e-10)

            if relative_variance < self._convergence_threshold:
                logger.info(
                    "[QAHPO] Converged: loss variance %.2e < threshold %.2e",
                    relative_variance,
                    self._convergence_threshold,
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
                        k: v for k, v in state.config.items() if not k.startswith("_")
                    }
                    trial_data["generation"] = state.generation
                    trial_data["amplitude"] = state.amplitude
                    break
            trials.append(trial_data)
        return trials

    def _config_fingerprint(self, config: dict[str, Any]) -> str:
        """Create fingerprint for architecture-defining parameters.

        Used to detect duplicate configurations within a batch or across trials.
        The fingerprint captures the key parameters that define model architecture
        and training behavior.

        Args:
            config: Configuration dictionary to fingerprint.

        Returns:
            String fingerprint suitable for set membership testing.
        """
        arch_keys = [
            "num_reasoning_blocks",
            "num_moe_experts",
            "embedding_dim",
            "hidden_dim",
            "hd_dim",
            "target_vocab_size",
            "batch_size",
            "learning_rate",
        ]
        return "|".join(f"{k}={config.get(k)}" for k in sorted(arch_keys) if k in config)

    def get_next_trials(self, n_trials: int) -> list[TrialConfig]:
        """Generate next batch of trial configurations.

        Ensures configuration diversity by:
        1. Applying mutation to configs that have already been run in prior trials
        2. Detecting and mutating duplicate configs within the same batch
        3. Tracking which quantum states have been used via trial_count

        Args:
            n_trials: Number of trials to generate.

        Returns:
            List of trial configurations with guaranteed diversity.
        """
        # Initialize population if needed
        if not self.population:
            self._initialize_population()

        trials = []
        selected = self._select_by_amplitude(n_trials)
        # Use global fingerprint history for cross-sweep duplicate prevention
        # (self._used_fingerprints persists across get_next_trials() calls)

        for state in selected:
            self._trial_counter += 1
            trial_id = f"qahpo_g{self.generation}_t{self._trial_counter}"

            # Start with a copy of the state's config
            config = state.config.copy()

            # === FIX: Apply mutation if this config was already used in a prior trial ===
            if state.trial_count > 0:
                logger.info(
                    f"[QAHPO] Config reused {state.trial_count}x, applying mutation for {trial_id}"
                )
                # Use progressively stronger mutation for heavily reused configs
                mutation_multiplier = min(1.0 + 0.25 * state.trial_count, 2.0)
                config = self._mutate_config(
                    config, self.config.mutation_strength * mutation_multiplier
                )

            # === FIX: Prevent duplicates globally across all trials ===
            fingerprint = self._config_fingerprint(config)
            retry_count = 0
            max_retries = 5
            while fingerprint in self._used_fingerprints and retry_count < max_retries:
                logger.info(
                    f"[QAHPO] Duplicate detected (attempt {retry_count + 1}), mutating for {trial_id}"
                )
                # Use progressively stronger mutation on each retry
                config = self._mutate_config(
                    config, self.config.mutation_strength * (1.5 + 0.5 * retry_count)
                )
                fingerprint = self._config_fingerprint(config)
                retry_count += 1

            self._used_fingerprints.add(fingerprint)
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

            # Update state with trial ID and mark as used
            state.config["_trial_id"] = trial_id
            state.trial_count += 1  # Mark this state as having been used

        return trials

    def get_best_trial(self) -> tuple[str, float] | None:
        """Get the best trial so far.
        Returns:
            Tuple of (trial_id, best_loss) or None if no results.
        """
        if self.best_config and self.best_loss < float("inf"):
            return (self.best_config.get("_trial_id", "unknown"), self.best_loss)
        return None

    def save_to_meta_cache(self, sweep_id: str) -> None:
        """Save sweep results to meta-learning cache.

        Called at the end of a sweep to enable knowledge transfer
        to future sweeps with similar tasks.

        Args:
            sweep_id: Unique identifier for this sweep.
        """
        if self._meta_cache is None or self.best_config is None:
            logger.debug("[QAHPO] Meta-cache not available or no best config")
            return

        # Compute task hash if not provided
        task_hash = self._task_hash
        if task_hash is None:
            task_hash = MetaLearningCache.compute_task_hash(
                param_budget=self.config.param_budget,
                optimizer_choices=["sophiag", "adam", "grover"],  # Default options
            )

        # Collect all configs and losses
        all_configs = []
        all_losses = []
        for result in self.trial_history:
            if not math.isinf(result.loss):
                # Find matching config
                for state in self.population:
                    if state.config.get("_trial_id") == result.trial_id:
                        clean_config = {
                            k: v for k, v in state.config.items() if not k.startswith("_")
                        }
                        all_configs.append(clean_config)
                        all_losses.append(result.loss)
                        break

        # Clean best config
        clean_best = {k: v for k, v in self.best_config.items() if not k.startswith("_")}

        # Save to cache
        self._meta_cache.save_sweep_results(
            sweep_id=sweep_id,
            task_hash=task_hash,
            best_config=clean_best,
            best_loss=self.best_loss,
            all_configs=all_configs,
            all_losses=all_losses,
            metadata={
                "generation": self.generation,
                "total_trials": self.total_trials,
                "config": {
                    "mutation_strategy": self.config.mutation_strategy,
                    "enable_multi_fidelity": self.config.enable_multi_fidelity,
                },
            },
        )

        logger.info(
            "[QAHPO] Saved sweep %s to meta-cache (task_hash=%s, %d trials)",
            sweep_id,
            task_hash,
            len(all_configs),
        )

    def warm_start_from_meta_cache(self) -> int:
        """Warm-start population from meta-learning cache.

        Returns:
            Number of configs transferred from prior sweeps.
        """
        if self._meta_cache is None:
            return 0

        task_hash = self._task_hash
        if task_hash is None:
            task_hash = MetaLearningCache.compute_task_hash(
                param_budget=self.config.param_budget,
            )

        # Get warm-start configs
        warm_configs = self._meta_cache.get_warm_start_configs(
            task_hash,
            n_configs=self.config.meta_warm_start_count,
        )

        if not warm_configs:
            return 0

        # Initialize population if needed
        if not self.population:
            self._initialize_population()

        # Inject warm-start configs with high initial amplitude
        for i, config in enumerate(warm_configs):
            if i < len(self.population):
                self.population[i].config.update(config)
                self.population[i].amplitude = 1.2  # Boost warm-start configs

        # Also warm-start TPE surrogate
        if self._surrogate is not None:
            transferred = self._meta_cache.warm_start_surrogate(
                self._surrogate,
                task_hash,
                n_prior_observations=20,
            )
            if transferred > 0:
                self._surrogate.fit()

        logger.info(
            "[QAHPO] Warm-started with %d configs from meta-cache (task_hash=%s)",
            len(warm_configs),
            task_hash,
        )

        return len(warm_configs)

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
            convergence_info["loss_trend"] = (
                float(self._convergence_window[-1] - self._convergence_window[0])
                if len(self._convergence_window) > 1
                else 0.0
            )

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

        # Enhancement status
        enhancement_info = {
            "tpe_surrogate": self._surrogate is not None,
            "tpe_fitted": self._surrogate._is_fitted if self._surrogate else False,
            "tpe_observations": len(self._surrogate.observations) if self._surrogate else 0,
            "multi_fidelity": self._hyperband is not None,
            "meta_cache": self._meta_cache is not None,
            "mutation_strategy": self.config.mutation_strategy,
            "param_budget": self.config.param_budget,
        }

        # c-TPE constraint stats
        if self._constrained_acq is not None:
            enhancement_info["constraint_stats"] = (
                self._constrained_acq.get_feasibility_statistics()
            )

        # Hyperband stats
        if self._hyperband is not None:
            enhancement_info["hyperband"] = self._hyperband.get_statistics()

        # fANOVA importance-guided mutation stats
        enhancement_info["fanova_enabled"] = self._fanova is not None
        enhancement_info["fanova_last_fit"] = self._last_fanova_fit
        if self._importance_scores:
            enhancement_info["fanova_importance"] = self._importance_scores
            # Sort by importance for display
            sorted_importance = sorted(
                self._importance_scores.items(), key=lambda x: x[1], reverse=True
            )
            enhancement_info["fanova_top_params"] = [p[0] for p in sorted_importance[:3]]

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
            "enhancements": enhancement_info,
            "config": {
                "tunneling_probability": self.config.tunneling_probability,
                "mutation_strength": self.config.mutation_strength,
                "crossover_rate": self.config.crossover_rate,
                "elite_fraction": self.config.elite_fraction,
            },
        }


__all__ = ["QuantumAdaptiveHPOScheduler", "QAHPOConfig", "QuantumState"]
