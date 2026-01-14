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

# =============================================================================
# ENTERPRISE HPO ENHANCEMENTS (Phase 1 & 4)
# =============================================================================
try:
    from highnoon.services.hpo_surrogate_gp import (
        GPYTORCH_AVAILABLE,
        AcquisitionPortfolio,
        DeepKernelGPConfig,
        DeepKernelGPSurrogate,
    )

    DEEP_KERNEL_GP_AVAILABLE = True
except ImportError:
    DEEP_KERNEL_GP_AVAILABLE = False
    GPYTORCH_AVAILABLE = False

try:
    from highnoon.services.hpo_shapley import SHAP_AVAILABLE, ShapleyImportanceAnalyzer

    SHAPLEY_AVAILABLE = True
except ImportError:
    SHAPLEY_AVAILABLE = False
    SHAP_AVAILABLE = False

# =============================================================================
# ENTERPRISE HPO ENHANCEMENTS (Phases 2, 3, 5, 6, 7)
# =============================================================================
try:
    from highnoon.services.hpo_nas import DARTSSearcher, EarlyStoppingPredictor, ZeroCostProxy

    NAS_AVAILABLE = True
except ImportError:
    NAS_AVAILABLE = False

try:
    from highnoon.services.hpo_asha import ASHAScheduler, FreezeThawBO, MOBSTERScheduler

    ASHA_AVAILABLE = True
except ImportError:
    ASHA_AVAILABLE = False

try:
    from highnoon.services.hpo_pbt import MultiObjectivePBT, PBTBacktrackScheduler

    PBT_ADVANCED_AVAILABLE = True
except ImportError:
    PBT_ADVANCED_AVAILABLE = False

try:
    from highnoon.services.hpo_fanova_advanced import (
        CausalImportanceAnalyzer,
        GraphfANOVA,
        StreamingfANOVA,
    )

    FANOVA_ADVANCED_AVAILABLE = True
except ImportError:
    FANOVA_ADVANCED_AVAILABLE = False

try:
    from highnoon.services.hpo_theoretical import (
        HPOExperimentFramework,
        InformationGainAcquisition,
        RegretTracker,
    )

    THEORETICAL_HPO_AVAILABLE = True
except ImportError:
    THEORETICAL_HPO_AVAILABLE = False

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
    # Phase 600+: Tuned for QHDSpatial/QHDHierarchical scaling
    # QHD has fewer high-importance params (qhd_num_paths dominant)
    enable_fanova_guidance: bool = True
    fanova_refit_interval: int = 5  # Refit importance every N trials
    fanova_min_trials: int = 5  # Min trials before using importance
    fanova_importance_boost: float = 2.5  # Mutation boost for important params (was 3.0)
    fanova_importance_dampen: float = 0.4  # Mutation dampen for unimportant (was 0.3)
    fanova_importance_threshold: float = 0.05  # Lower threshold for QHD (was 0.1)

    # =========================================================================
    # Phase 600: Enhanced fANOVA Integration
    # =========================================================================
    # Boost mutation for parameters with significant interactions
    enable_interaction_boost: bool = True
    interaction_boost_factor: float = 1.5  # Additional boost for interacting params
    # Use marginal curves to guide sampling toward optimal regions
    enable_marginal_guidance: bool = True
    marginal_guidance_weight: float = 0.4  # Blend weight (0=ignore, 1=full bias)
    # Bootstrap importance estimation for early trials (before fanova_min_trials)
    enable_bootstrap_importance: bool = True
    bootstrap_min_trials: int = 3  # Min trials for bootstrap estimation

    # =========================================================================
    # Phase 600: DEHB-Inspired Subpopulation Tracking
    # =========================================================================
    # Track separate subpopulations per fidelity level
    enable_fidelity_subpopulations: bool = True
    # Fidelity levels as fractions of max budget
    fidelity_levels: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    # Rate at which mutations incorporate winners from lower fidelity
    cross_fidelity_transfer_rate: float = 0.3
    # Max configs to track per fidelity level
    max_configs_per_fidelity: int = 16

    # Enhancement: Earlier population evolution (Phase 600)
    # Evolve population more frequently for faster adaptation
    evolution_interval: int = 4  # Evolve every N trials (was population_size)

    # Enhancement: Aggressive meta-learning warm-start
    meta_warm_start_ratio: float = 0.5  # Fill 50% of population with warm-start configs

    # =========================================================================
    # Phase 700: Adaptive Meta-Tuning
    # These parameters control how QAHPO adapts its own behavior based on trial
    # signals (convergence rate, stagnation, etc.)
    # =========================================================================
    enable_adaptive_meta_tuning: bool = True  # Enable self-adaptation

    # Adaptation bounds (min, max) for each meta-parameter
    # Widened for full exploration across small to frontier models
    adaptive_mutation_range: tuple[float, float] = (0.1, 0.8)  # Was 0.6
    adaptive_tunneling_range: tuple[float, float] = (0.05, 0.5)  # Was 0.4
    adaptive_crossover_range: tuple[float, float] = (0.2, 0.8)  # Was 0.6
    adaptive_evolution_interval_range: tuple[int, int] = (2, 15)  # Was 10
    adaptive_population_size_range: tuple[int, int] = (6, 32)  # Was 26

    # Adaptation triggers
    stagnation_trials_threshold: int = 5  # Trials without improvement to trigger boost
    fast_convergence_threshold: float = 0.1  # Loss improvement ratio for speedup

    # =========================================================================
    # ENTERPRISE PHASE 1: Deep Kernel GP Surrogate
    # =========================================================================
    # Enable GP-based surrogate with neural network feature extraction
    enable_deep_kernel_gp: bool = True
    # Use Deep Kernel GP instead of TPE when available
    prefer_gp_over_tpe: bool = False  # Default: use TPE, GP is optional
    # GP configuration
    gp_hidden_dims: tuple[int, ...] = (64, 32)  # Feature extractor architecture
    gp_n_epochs: int = 50  # Training epochs for GP fitting
    gp_min_observations: int = 15  # Min observations before fitting GP

    # =========================================================================
    # ENTERPRISE PHASE 1: Acquisition Portfolio (EXP3)
    # =========================================================================
    # Enable ensemble acquisition function selection
    enable_acquisition_portfolio: bool = True
    # Acquisition functions in portfolio
    portfolio_acquisitions: tuple[str, ...] = ("ei", "ucb", "pi")
    # EXP3 learning rate for portfolio updates
    portfolio_eta: float = 0.1
    # Exploration probability
    portfolio_gamma: float = 0.1

    # =========================================================================
    # ENTERPRISE PHASE 4: Shapley Importance Analysis
    # =========================================================================
    # Enable Shapley-value importance (alternative to fANOVA)
    enable_shapley_importance: bool = True
    # Number of Monte Carlo samples for Shapley estimation
    shapley_n_samples: int = 500
    # Blend weight: 0 = pure fANOVA, 1 = pure Shapley, 0.5 = average
    shapley_blend_weight: float = 0.3  # Use fANOVA primarily, Shapley as supplement

    # =========================================================================
    # ENTERPRISE PHASE 2: Neural Architecture Search (NAS)
    # =========================================================================
    enable_nas_features: bool = True
    # Enable DARTS for architecture search
    enable_darts: bool = False  # Disabled by default (heavyweight)
    # Enable zero-cost proxy pre-filtering
    enable_zero_cost_proxy: bool = True
    # Proxies to use: naswot, gradnorm, synflow
    zero_cost_proxies: tuple[str, ...] = ("naswot", "synflow")
    # Enable early stopping predictor
    enable_early_stopping_predictor: bool = True
    # Minimum curve points before prediction
    early_stopping_min_points: int = 5

    # =========================================================================
    # ENTERPRISE PHASE 3: Advanced Multi-Fidelity Scheduling
    # =========================================================================
    # Scheduler type: hyperband, asha, mobster, freeze_thaw
    multi_fidelity_scheduler: str = "hyperband"  # Default existing
    # Enable ASHA async scheduling (requires Ray or async execution)
    enable_asha: bool = False  # Disabled until Ray available
    # Enable MOBSTER (Model-Based ASHA)
    enable_mobster: bool = False
    # ASHA grace period (min epochs before stopping)
    asha_grace_period: int = 5
    # ASHA reduction factor
    asha_reduction_factor: int = 3

    # =========================================================================
    # ENTERPRISE PHASE 4: Advanced fANOVA
    # =========================================================================
    # fANOVA analyzer type: standard, graph, causal, streaming
    fanova_analyzer_type: str = "standard"  # Default to existing
    # Enable streaming fANOVA for O(1) updates
    enable_streaming_fanova: bool = True
    # Streaming window size (0 = unbounded)
    streaming_fanova_window: int = 0

    # =========================================================================
    # ENTERPRISE PHASE 5: Advanced PBT
    # =========================================================================
    # Enable PBT with backtracking
    enable_pbt_backtrack: bool = True
    # Maximum checkpoint tree depth
    pbt_max_tree_depth: int = 50
    # Enable multi-objective PBT (Pareto optimization)
    enable_multi_objective_pbt: bool = False
    # Multi-objective names (if enabled)
    pbt_objectives: tuple[str, ...] = ("loss",)

    # =========================================================================
    # ENTERPRISE PHASE 7: Theoretical Foundations
    # =========================================================================
    # Enable regret tracking
    enable_regret_tracking: bool = True
    # Enable information-gain acquisition (MES, GIBBON)
    enable_information_gain_acquisition: bool = False  # Heavyweight
    # Enable A/B testing framework for strategy comparison
    enable_hpo_experiment_framework: bool = False


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

        # Fingerprint collision metrics (Enhancement: observability)
        self._fingerprint_collisions: int = 0
        self._fingerprint_retries: int = 0
        self._total_fingerprint_checks: int = 0

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

        # Phase 600+: QHD-Specific Prior Importance for Bootstrap
        # Used before faNOVA has enough data (< fanova_min_trials)
        # Reflects QHDSpatial/QHDHierarchical compute scaling
        self._qhd_prior_importance: dict[str, float] = {
            # === Primary Compute Drivers (linear multipliers) ===
            "qhd_num_paths": 0.20,  # K× most critical
            "num_reasoning_blocks": 0.15,  # Depth multiplier
            # === Phase 1010: Per-Layer HD Dimensions ===
            "hd_dim_embedding": 0.10,  # Vocabulary encoding capacity
            "hd_dim_spatial": 0.08,  # FFT SSM dimension
            "hd_dim_timecrystal": 0.05,  # Floquet modes × D
            "hd_dim_moe": 0.03,  # Routing similarity (smallest)
            "hd_dim": 0.02,  # DEPRECATED: Global fallback
            # === Secondary (linear but cheaper) ===
            "hd_hierarchical_levels": 0.08,
            "hidden_dim": 0.06,
            "embedding_dim": 0.05,
            "mamba_state_dim": 0.03,
            # === Sparse/Shared Activation ===
            "num_moe_experts": 0.04,
            "superposition_dim": 0.03,  # MoE sparse
            # === Regularization/Training ===
            "learning_rate": 0.03,
            "weight_decay": 0.02,
            "qhd_entanglement_depth": 0.02,  # VQC layers
        }

        # Initialize importance with priors for bootstrap (before faNOVA fits)
        if self.config.enable_bootstrap_importance:
            self._importance_scores = self._qhd_prior_importance.copy()
            logger.info("[QAHPO] Initialized with QHD prior importance (bootstrap)")

        if self.config.enable_fanova_guidance and FANOVA_AVAILABLE:
            self._fanova = HyperparameterImportanceAnalyzer(
                n_trees=50,  # Lighter than visualization (faster)
                min_trials=self.config.fanova_min_trials,
            )
            logger.info("[QAHPO] fANOVA importance-guided mutation enabled")

        # ====================================================================
        # ENHANCEMENT 6: Phase 600 - DEHB Fidelity Subpopulations
        # ====================================================================
        self._fidelity_populations: dict[float, list[QuantumState]] = {}
        self._interaction_importance: dict[tuple[str, str], float] = {}  # (p1, p2) -> importance

        if self.config.enable_fidelity_subpopulations:
            for fidelity in self.config.fidelity_levels:
                self._fidelity_populations[fidelity] = []
            logger.info(
                "[QAHPO] DEHB fidelity subpopulations enabled: %s",
                self.config.fidelity_levels,
            )

        logger.info(
            "[QAHPO] Initialized with population_size=%d, T=[%.2f, %.2f], "
            "mutation=%s, multi_fidelity=%s, tpe=%s, meta=%s, fanova=%s, dehb_subpop=%s",
            self.config.population_size,
            self.config.initial_temperature,
            self.config.final_temperature,
            self.config.mutation_strategy,
            self.config.enable_multi_fidelity,
            self.config.enable_tpe_surrogate,
            self.config.enable_meta_learning,
            self.config.enable_fanova_guidance,
            self.config.enable_fidelity_subpopulations,
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

        Enhancement (Phase 600): Aggressive meta-learning warm-start.
        Uses prior sweep results to seed a portion of the population,
        giving proven configs a head start.

        Budget-aware: Uses spread trial IDs (0, 5, 10, 15...) to ensure
        HPOSearchSpace.sample() produces a range of config sizes from
        the progressive budget logic (10-70% of param_budget).
        """
        logger.info("[QAHPO] Initializing population of %d configs", self.config.population_size)

        self.population = []
        warm_start_configs: list[dict[str, Any]] = []

        # === Enhancement: Meta-learning warm-start ===
        if self._meta_cache is not None and self._task_hash:
            warm_start_count = int(self.config.population_size * self.config.meta_warm_start_ratio)
            warm_start_count = min(warm_start_count, self.config.meta_warm_start_count)

            if warm_start_count > 0:
                warm_start_configs = self._meta_cache.get_warm_start_configs(
                    self._task_hash, n_configs=warm_start_count
                )
                logger.info(
                    f"[QAHPO] Meta-learning: loaded {len(warm_start_configs)} warm-start configs "
                    f"(requested {warm_start_count})"
                )

        # Fill population with warm-start configs first (with higher amplitude)
        for i, warm_config in enumerate(warm_start_configs):
            config = warm_config.copy()

            # Clamp LR to optimizer-specific bounds
            config = self._clamp_learning_rate(config)
            config = self._enforce_architecture_constraints(config)

            # Warm-start configs get higher amplitude (proven performers)
            warm_amplitude = 1.5 / math.sqrt(self.config.population_size)
            phase = random.uniform(0, 2 * math.pi)

            state = QuantumState(
                config=config,
                amplitude=warm_amplitude,
                phase=phase,
                generation=0,
            )
            self.population.append(state)
            logger.debug(
                f"[QAHPO] Population[{i}] from warm-start (amplitude={warm_amplitude:.3f})"
            )

        # Fill remaining slots with fresh samples
        remaining = self.config.population_size - len(self.population)
        for i in range(remaining):
            if self.search_space_sampler:
                # Use spread trial IDs to get budget-aware configs
                # Early IDs (0-7 at step 5 = 0,5,10,15,20,25,30,35) → diverse budget targets
                # This ensures population has a mix of small and medium configs
                trial_id = i * 5  # Spread across progressive budget tiers
                config = self.search_space_sampler(trial_id)
                logger.debug(
                    f"[QAHPO] Population[{len(self.population)}] sampled with trial_id={trial_id}"
                )
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

        Phase 3.2.1 Enhancement: Added hard cap for SympFlowQNG and barren
        plateau scale awareness to prevent LR explosion during recovery.

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

        # Phase 3.2.1: SympFlowQNG-specific hard cap (symplectic integrator sensitivity)
        # SympFlowQNGOptimizer.max_safe_lr defines stability threshold
        SYMPFLOWQNG_HARD_CAP = 1e-3
        if optimizer == "sympflowqng":
            lr_max = min(lr_max, SYMPFLOWQNG_HARD_CAP)

            # Also account for barren plateau LR scale if present
            # The UnifiedSmartTuner may boost LR during plateau recovery
            barren_scale = config.get("barren_plateau_lr_scale", 1.0)
            if barren_scale > 1.0:
                # Reduce max_lr by the scale factor to keep effective LR bounded
                # effective_lr = lr * barren_scale, so lr_max = cap / barren_scale
                effective_max = SYMPFLOWQNG_HARD_CAP / barren_scale
                lr_max = min(lr_max, effective_max)
                logger.debug(
                    "[QAHPO] SympFlowQNG with barren_scale=%.1f: lr_max reduced to %.2e",
                    barren_scale,
                    lr_max,
                )

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
        """Apply Gaussian perturbation mutation with importance and interaction weighting.

        Phase 600 Enhancement: Includes interaction-aware mutation boost.
        When fANOVA importance is available, mutation strength is scaled:
        - Important params (>threshold): boost mutation for more exploration
        - Unimportant params (<threshold): dampen mutation, less exploration
        - Parameters with significant interactions: additional boost

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

        # Build set of params with significant interactions for fast lookup
        interacting_params: set[str] = set()
        if self.config.enable_interaction_boost and self._interaction_importance:
            for (p1, p2), imp in self._interaction_importance.items():
                if imp > 0.01:  # Significant interaction threshold
                    interacting_params.add(p1)
                    interacting_params.add(p2)

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

                    # === Phase 600: Interaction Boost ===
                    if self.config.enable_interaction_boost and key in interacting_params:
                        param_strength *= self.config.interaction_boost_factor

                    if isinstance(value, float):
                        noise = random.gauss(0, param_strength * abs(value) + 1e-8)
                        mutated[key] = max(1e-8, value + noise)
                    else:
                        # Integer: scale mutation by param_strength
                        if param_strength > 0.5:
                            # Stronger mutation for important/interacting int params
                            delta = random.choice([-2, -1, 0, 1, 2])
                        else:
                            delta = random.choice([-1, 0, 0, 1])
                        mutated[key] = max(1, value + delta)

        # === Phase 600: Cross-Fidelity Transfer ===
        if self.config.enable_fidelity_subpopulations:
            mutated = self._apply_cross_fidelity_transfer(mutated)

        return mutated

    def _apply_cross_fidelity_transfer(self, config: dict[str, Any]) -> dict[str, Any]:
        """Blend configuration with winning configs from lower fidelity levels.

        DEHB-inspired enhancement: Uses information from low-fidelity trials
        to guide high-fidelity exploration.

        Args:
            config: Base configuration to potentially modify.

        Returns:
            Configuration with cross-fidelity influence applied.
        """
        if not self._fidelity_populations:
            return config

        if random.random() > self.config.cross_fidelity_transfer_rate:
            return config  # Skip transfer this time

        # Find populated lower fidelity levels with results
        populated_levels = [
            level
            for level, pop in self._fidelity_populations.items()
            if pop and level < 1.0  # Only consider lower than full fidelity
        ]

        if not populated_levels:
            return config

        # Select random lower fidelity level
        source_level = random.choice(populated_levels)
        source_pop = self._fidelity_populations[source_level]

        # Find best config from that level
        best_state = min(source_pop, key=lambda s: s.loss)

        # Blend: bias current config toward best from lower fidelity
        blended = config.copy()
        blend_weight = 0.3  # Light influence from lower fidelity

        for key, val in best_state.config.items():
            if key.startswith("_"):
                continue
            if key in blended and isinstance(val, (int, float)) and not isinstance(val, bool):
                current_val = blended[key]
                if isinstance(val, float):
                    blended[key] = (1 - blend_weight) * current_val + blend_weight * val
                elif isinstance(val, int):
                    blended[key] = int(round((1 - blend_weight) * current_val + blend_weight * val))

        logger.debug(
            "[QAHPO] Cross-fidelity transfer from level=%.2f (loss=%.4f)",
            source_level,
            best_state.loss,
        )

        return blended

    def _update_fidelity_population(self, result: TrialResult) -> None:
        """Update fidelity subpopulations with completed trial result.

        DEHB-inspired: Tracks best configs at each fidelity level to enable
        cross-fidelity information transfer during mutation.

        Args:
            result: Completed trial result.
        """
        if not self._fidelity_populations:
            return

        # Find the config for this trial
        trial_config = None
        for state in self.population:
            if state.config.get("_trial_id") == result.trial_id:
                trial_config = state.config
                break

        if trial_config is None:
            return

        # Determine fidelity level from budget fraction
        # Use param_budget or architecture size as proxy for fidelity
        if self.config.param_budget:
            from highnoon.services.hpo_manager import estimate_model_params

            try:
                estimated = estimate_model_params(trial_config)
                fidelity = estimated / self.config.param_budget
            except Exception:
                fidelity = 0.5  # Default to mid-fidelity
        else:
            # Without param_budget, estimate fidelity from embedding_dim relative to max
            dim = trial_config.get("embedding_dim", 256)
            fidelity = min(1.0, dim / 2048)

        # Snap to nearest defined fidelity level
        fidelity_levels = list(self.config.fidelity_levels)
        closest_level = min(fidelity_levels, key=lambda f: abs(f - fidelity))

        # Create state for this result
        new_state = QuantumState(
            config=trial_config.copy(),
            loss=result.loss,
            generation=self.generation,
        )

        # Add to appropriate fidelity population
        pop = self._fidelity_populations[closest_level]
        pop.append(new_state)

        # Limit population size per fidelity
        max_per_fidelity = self.config.max_configs_per_fidelity
        if len(pop) > max_per_fidelity:
            # Keep best configs (lowest loss)
            pop.sort(key=lambda s: s.loss)
            self._fidelity_populations[closest_level] = pop[:max_per_fidelity]

        logger.debug(
            "[QAHPO] Added to fidelity=%.2f population (size=%d, loss=%.4f)",
            closest_level,
            len(self._fidelity_populations[closest_level]),
            result.loss,
        )

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
        # Guard against None loss values from failed/crashed trials
        if current_loss is None:
            return False  # Don't stop trial if loss is unknown

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

        # === Phase 600: DEHB Fidelity Subpopulation Tracking ===
        if self.config.enable_fidelity_subpopulations and self._fidelity_populations:
            self._update_fidelity_population(result)

        # === fANOVA: Refit importance scores periodically ===
        if self._fanova is not None and self.total_trials >= self.config.fanova_min_trials:
            # Check if we should refit
            if self.total_trials - self._last_fanova_fit >= self.config.fanova_refit_interval:
                self._refit_fanova()

        # === Phase 700: Adaptive Meta-Tuning ===
        # Adapt QAHPO parameters based on trial signals
        if self.config.enable_adaptive_meta_tuning:
            self._adapt_meta_parameters()

        # Evolve population more frequently for faster adaptation (Phase 600 enhancement)
        evolution_interval = self.config.evolution_interval or self.config.population_size
        if self.total_trials > 0 and self.total_trials % evolution_interval == 0:
            logger.info(
                f"[QAHPO] Evolution triggered at trial {self.total_trials} (interval={evolution_interval})"
            )
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
                    ", ".join(f"{p.name}={p.importance * 100:.1f}%" for p in top_params),
                )

                # === Phase 600: Cache interaction importance ===
                if self.config.enable_interaction_boost and result.interactions:
                    self._interaction_importance.clear()
                    for interaction in result.interactions:
                        key = (interaction.param1, interaction.param2)
                        self._interaction_importance[key] = interaction.importance
                    if result.interactions:
                        top_int = result.interactions[0]
                        logger.debug(
                            "[QAHPO-fANOVA] Top interaction: %s×%s=%.1f%%",
                            top_int.param1,
                            top_int.param2,
                            top_int.importance * 100,
                        )

    def _adapt_meta_parameters(self) -> None:
        """Adapt QAHPO meta-parameters based on trial signals.

        Phase 700: Self-adaptive meta-tuning. Adjusts mutation_strength,
        tunneling_probability, and evolution_interval based on:
        - Stagnation: Boost exploration when no improvement for N trials
        - Fast convergence: Speed up evolution when making good progress
        - Temperature: Already annealed via _get_temperature()

        This is called after each trial result is reported.
        """
        if self.total_trials < 3:
            return  # Need minimum history

        # Get adaptation bounds from config
        min_mut, max_mut = self.config.adaptive_mutation_range
        min_tun, max_tun = self.config.adaptive_tunneling_range
        min_evo, max_evo = self.config.adaptive_evolution_interval_range

        # === Signal 1: Stagnation Detection ===
        # Boost mutation and tunneling when stuck
        if self._no_improvement_count >= self.config.stagnation_trials_threshold:
            # Increase mutation strength
            old_mutation = self.config.mutation_strength
            self.config.mutation_strength = min(max_mut, old_mutation * 1.2)

            # Increase tunneling probability
            old_tunneling = self.config.tunneling_probability
            self.config.tunneling_probability = min(max_tun, old_tunneling * 1.3)

            if old_mutation != self.config.mutation_strength:
                logger.info(
                    "[QAHPO-Adapt] Stagnation detected (%d trials), boosting: "
                    "mutation=%.2f→%.2f, tunneling=%.2f→%.2f",
                    self._no_improvement_count,
                    old_mutation,
                    self.config.mutation_strength,
                    old_tunneling,
                    self.config.tunneling_probability,
                )

            # Also grow population for more diversity during stagnation
            min_pop, max_pop = self.config.adaptive_population_size_range
            if self.config.population_size < max_pop:
                old_pop = self.config.population_size
                self.config.population_size = min(max_pop, old_pop + 2)
                if old_pop != self.config.population_size:
                    logger.info(
                        "[QAHPO-Adapt] Growing population for diversity: %d→%d",
                        old_pop,
                        self.config.population_size,
                    )
        else:
            # Decay back toward defaults when making progress
            target_mutation = 0.3  # Default
            target_tunneling = 0.15  # Default
            target_crossover = 0.4  # Default

            self.config.mutation_strength = max(
                min_mut, self.config.mutation_strength * 0.95 + target_mutation * 0.05
            )
            self.config.tunneling_probability = max(
                min_tun, self.config.tunneling_probability * 0.95 + target_tunneling * 0.05
            )
            # Crossover also decays toward default
            min_cross, max_cross = self.config.adaptive_crossover_range
            self.config.crossover_rate = max(
                min_cross,
                min(max_cross, self.config.crossover_rate * 0.95 + target_crossover * 0.05),
            )

        # === Signal 2: Convergence Rate ===
        # Speed up evolution when making good progress
        if len(self._convergence_window) >= 3:
            recent_losses = self._convergence_window[-3:]
            if len(recent_losses) >= 3:
                # Check improvement rate
                first_loss = recent_losses[0]
                last_loss = recent_losses[-1]
                improvement = (first_loss - last_loss) / (abs(first_loss) + 1e-8)

                if improvement > self.config.fast_convergence_threshold:
                    # Good progress - speed up evolution
                    old_interval = self.config.evolution_interval
                    self.config.evolution_interval = max(min_evo, old_interval - 1)

                    # Also increase crossover for faster mixing
                    min_cross, max_cross = self.config.adaptive_crossover_range
                    self.config.crossover_rate = min(max_cross, self.config.crossover_rate * 1.1)

                    if old_interval != self.config.evolution_interval:
                        logger.info(
                            "[QAHPO-Adapt] Fast convergence (%.1f%% improvement), "
                            "evolution_interval=%d→%d, crossover=%.2f",
                            improvement * 100,
                            old_interval,
                            self.config.evolution_interval,
                            self.config.crossover_rate,
                        )
                elif improvement < 0:
                    # Getting worse - slow down evolution to stabilize
                    old_interval = self.config.evolution_interval
                    self.config.evolution_interval = min(max_evo, old_interval + 1)

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
            "active_vocab_size",
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
            self._total_fingerprint_checks += 1
            retry_count = 0
            max_retries = 5
            initial_collision = fingerprint in self._used_fingerprints
            if initial_collision:
                self._fingerprint_collisions += 1

            while fingerprint in self._used_fingerprints and retry_count < max_retries:
                logger.info(
                    f"[QAHPO] Duplicate detected (attempt {retry_count + 1}), mutating for {trial_id}"
                )
                self._fingerprint_retries += 1
                # Use progressively stronger mutation on each retry
                config = self._mutate_config(
                    config, self.config.mutation_strength * (1.5 + 0.5 * retry_count)
                )
                fingerprint = self._config_fingerprint(config)
                retry_count += 1

            if retry_count > 0:
                logger.info(
                    f"[QAHPO] Fingerprint resolved after {retry_count} retries "
                    f"(total collisions: {self._fingerprint_collisions}, "
                    f"total retries: {self._fingerprint_retries})"
                )

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

            # Log trial config with explicit dimension info
            logger.info(
                f"[QAHPO] Trial {trial_id} config: "
                f"hidden_dim={config.get('hidden_dim')}, "
                f"embedding_dim={config.get('embedding_dim')}, "
                f"hd_dim={config.get('hd_dim')}, "
                f"blocks={config.get('num_reasoning_blocks')}, "
                f"experts={config.get('num_moe_experts')}"
            )

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
            "scheduler_version": "2.1.0",  # Enhanced version with diversity fix
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
                "evolution_interval": self.config.evolution_interval,
            },
            # Fingerprint collision metrics (Phase 600 observability)
            "fingerprint_metrics": {
                "total_checks": self._total_fingerprint_checks,
                "collisions": self._fingerprint_collisions,
                "retries": self._fingerprint_retries,
                "collision_rate": (
                    self._fingerprint_collisions / max(1, self._total_fingerprint_checks)
                ),
                "unique_fingerprints": len(self._used_fingerprints),
            },
        }


__all__ = ["QuantumAdaptiveHPOScheduler", "QAHPOConfig", "QuantumState"]
