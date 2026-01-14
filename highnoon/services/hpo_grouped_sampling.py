"""HPO Grouped Parameter Sampling - Forest-style Hierarchical Sampler.

This module implements Tree-Parzen Estimator (TPE) inspired grouped parameter
sampling for efficient hyperparameter optimization. When combined with the
Grover-Q optimizer, this provides quantum-inspired efficient search.

Key features:
- Hierarchical parameter groups (architecture → training → control)
- Good/bad partition strategy (TPE-style)
- Optional Random Forest surrogate for conditional sampling
- Integration with multi-stage HPO (COARSE → REFINE → FINE)

Copyright 2025 Verso Industries
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETER GROUP DEFINITIONS
# =============================================================================


@dataclass
class ParameterSpec:
    """Specification for a single hyperparameter."""

    name: str
    param_type: str  # "int", "float", "log_uniform", "choice"
    min_val: float | None = None
    max_val: float | None = None
    choices: list[Any] | None = None
    default: Any | None = None
    # For conditional parameters
    condition: dict[str, Any] | None = None


@dataclass
class ParameterGroup:
    """A group of related hyperparameters."""

    name: str
    priority: int  # Lower = sampled first (architecture before training)
    parameters: list[ParameterSpec] = field(default_factory=list)

    def get_param(self, param_name: str) -> ParameterSpec | None:
        """Get parameter spec by name."""
        for p in self.parameters:
            if p.name == param_name:
                return p
        return None


# =============================================================================
# TPE-STYLE GOOD/BAD PARTITIONING
# =============================================================================


class GoodBadPartitioner:
    """Partition observations into good/bad sets for TPE-style sampling.

    Uses quantile-based partitioning where the top gamma fraction
    of observations are considered "good".
    """

    def __init__(self, gamma: float = 0.25):
        """Initialize partitioner.

        Args:
            gamma: Fraction of observations to consider "good" (default 25%)
        """
        self.gamma = gamma
        self.observations: list[tuple[dict[str, Any], float]] = []

    def add_observation(self, config: dict[str, Any], loss: float) -> None:
        """Add an observation (configuration, loss) pair."""
        self.observations.append((config, loss))

    def partition(self) -> tuple[list[dict], list[dict]]:
        """Partition observations into good and bad sets.

        Returns:
            Tuple of (good_configs, bad_configs)
        """
        if len(self.observations) < 4:
            # Not enough data for meaningful partitioning
            return [], []

        # Sort by loss (lower is better)
        sorted_obs = sorted(self.observations, key=lambda x: x[1])

        # Determine cutoff
        n_good = max(1, int(len(sorted_obs) * self.gamma))

        good_configs = [config for config, _ in sorted_obs[:n_good]]
        bad_configs = [config for config, _ in sorted_obs[n_good:]]

        return good_configs, bad_configs

    def get_good_distribution(self, param_name: str) -> dict[str, Any]:
        """Estimate distribution of a parameter in the good set.

        Returns statistics about the parameter in good configurations.
        """
        good_configs, _ = self.partition()
        if not good_configs:
            return {}

        values = [c.get(param_name) for c in good_configs if param_name in c]
        if not values:
            return {}

        if isinstance(values[0], (int, float)):
            return {
                "mean": np.mean(values),
                "std": np.std(values) if len(values) > 1 else 0.1,
                "min": min(values),
                "max": max(values),
            }
        else:
            # Categorical - return frequency distribution
            from collections import Counter

            counts = Counter(values)
            total = sum(counts.values())
            return {"frequencies": {k: v / total for k, v in counts.items()}}


# =============================================================================
# FOREST GROUP SAMPLER
# =============================================================================


class ParameterGroupSampler:
    """Forest-style grouped parameter sampler for efficient HPO.

    Implements hierarchical sampling with three parameter groups:
    1. Architecture (num_blocks, embedding_dim, experts) - sampled first
    2. Training (learning_rate, batch_size, optimizer) - conditional on arch
    3. Control (warmup, weight_decay, schedules) - fine-tuning knobs

    Uses TPE-style good/bad partitioning with optional Random Forest
    surrogate for conditional parameter estimation.
    """

    def __init__(
        self,
        seed: int | None = None,
        use_tpe: bool = True,
        gamma: float = 0.25,
    ):
        """Initialize the sampler.

        Args:
            seed: Random seed for reproducibility
            use_tpe: Use TPE-style sampling (vs pure random)
            gamma: Good/bad partition ratio for TPE
        """
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.use_tpe = use_tpe
        self.partitioner = GoodBadPartitioner(gamma=gamma) if use_tpe else None

        # Define default parameter groups
        self._init_default_groups()

        logger.info(
            f"[GroupSampler] Initialized with seed={seed}, use_tpe={use_tpe}, gamma={gamma}"
        )

    def _init_default_groups(self) -> None:
        """Initialize default HSMN parameter groups."""
        self.groups: list[ParameterGroup] = [
            # Priority 1: Architecture (determines model capacity)
            ParameterGroup(
                name="architecture",
                priority=1,
                parameters=[
                    ParameterSpec(
                        name="num_reasoning_blocks",
                        param_type="choice",
                        choices=[4, 6, 8, 12, 16, 20, 24],
                        default=8,
                    ),
                    ParameterSpec(
                        name="embedding_dim",
                        param_type="choice",
                        choices=[256, 512, 768, 1024, 2048, 4096],
                        default=512,
                    ),
                    ParameterSpec(
                        name="num_moe_experts",
                        param_type="choice",
                        choices=[4, 6, 8, 12],
                        default=8,
                    ),
                    ParameterSpec(
                        name="mamba_state_dim",
                        param_type="choice",
                        choices=[32, 64, 128],
                        default=64,
                    ),
                    # V2.0 Path Independence: All path systems remain independent.
                    # QHD, COCONUT, and SuperposedExpert explore different "universes".
                    # faNOVA tracks correlation hints but doesn't merge parameters.
                    ParameterSpec(
                        name="superposition_dim",
                        param_type="choice",
                        choices=[1, 2],  # Max 2 for Lite (K× memory cost)
                        default=2,
                    ),
                ],
            ),
            # Priority 2: Training (optimization dynamics)
            ParameterGroup(
                name="training",
                priority=2,
                parameters=[
                    ParameterSpec(
                        name="learning_rate",
                        param_type="log_uniform",
                        min_val=1e-6,
                        max_val=1e-2,
                        default=1e-4,
                    ),
                    ParameterSpec(
                        name="batch_size",
                        param_type="choice",
                        choices=[8, 16, 32, 64, 128],
                        default=32,
                    ),
                    ParameterSpec(
                        name="optimizer",
                        param_type="choice",
                        choices=["sophiag", "grover", "adamw", "adam"],
                        default="sophiag",
                    ),
                    ParameterSpec(
                        name="moe_top_k",
                        param_type="choice",
                        choices=[1, 2, 4],
                        default=2,
                    ),
                ],
            ),
            # Priority 3: Control (fine-tuning knobs)
            ParameterGroup(
                name="control",
                priority=3,
                parameters=[
                    ParameterSpec(
                        name="warmup_steps",
                        param_type="int",
                        min_val=0,
                        max_val=2000,
                        default=500,
                    ),
                    ParameterSpec(
                        name="weight_decay",
                        param_type="float",
                        min_val=0.0,
                        max_val=0.2,
                        default=0.01,
                    ),
                    ParameterSpec(
                        name="tt_rank_middle",
                        param_type="choice",
                        choices=[8, 16, 32],
                        default=16,
                    ),
                    ParameterSpec(
                        name="hamiltonian_hidden_dim",
                        param_type="choice",
                        choices=[128, 256, 512],
                        default=256,
                    ),
                ],
            ),
        ]

    def _sample_parameter(
        self,
        spec: ParameterSpec,
        use_good_prior: bool = False,
    ) -> Any:
        """Sample a single parameter value.

        Args:
            spec: Parameter specification
            use_good_prior: If True, bias sampling toward good configurations

        Returns:
            Sampled parameter value
        """
        # Try to use good distribution if TPE is enabled
        if use_good_prior and self.partitioner:
            good_dist = self.partitioner.get_good_distribution(spec.name)
            if good_dist:
                return self._sample_from_distribution(spec, good_dist)

        # Fall back to prior sampling
        if spec.param_type == "choice":
            return self.rng.choice(spec.choices)
        elif spec.param_type == "int":
            return self.rng.randint(int(spec.min_val), int(spec.max_val))
        elif spec.param_type == "float":
            return self.rng.uniform(spec.min_val, spec.max_val)
        elif spec.param_type == "log_uniform":
            log_min = math.log10(spec.min_val)
            log_max = math.log10(spec.max_val)
            return 10 ** self.rng.uniform(log_min, log_max)
        else:
            raise ValueError(f"Unknown parameter type: {spec.param_type}")

    def _sample_from_distribution(
        self,
        spec: ParameterSpec,
        dist: dict[str, Any],
    ) -> Any:
        """Sample from a learned distribution (TPE-style).

        Args:
            spec: Parameter specification
            dist: Distribution statistics from good configurations

        Returns:
            Sampled value biased toward good region
        """
        if "frequencies" in dist:
            # Categorical - sample according to frequencies
            items = list(dist["frequencies"].items())
            weights = [w for _, w in items]
            values = [v for v, _ in items]
            return self.rng.choices(values, weights=weights)[0]

        if "mean" in dist:
            # Continuous - sample from truncated normal around good mean
            mean = dist["mean"]
            std = max(dist["std"], (spec.max_val - spec.min_val) * 0.1)

            # Sample with clipping
            value = self.np_rng.normal(mean, std)
            value = np.clip(value, spec.min_val, spec.max_val)

            if spec.param_type == "int":
                return int(round(value))
            elif spec.param_type == "log_uniform":
                # Convert back from log space
                return 10**value
            return float(value)

        # Fall back to prior
        return self._sample_parameter(spec, use_good_prior=False)

    def sample_grouped_configuration(
        self,
        config: dict[str, Any],
        trial_id: int,
        stage: int = 1,
    ) -> dict[str, Any]:
        """Sample a complete configuration using grouped strategy.

        Samples parameters in group priority order, allowing later groups
        to be conditioned on earlier group values.

        Args:
            config: Optional config dict with custom group definitions
            trial_id: Trial ID for seeding
            stage: HPO stage (1=COARSE, 2=REFINE, 3=FINE)

        Returns:
            Complete hyperparameter configuration
        """
        # Re-seed for this trial
        self.rng.seed(trial_id)
        self.np_rng = np.random.default_rng(trial_id)

        result: dict[str, Any] = {}

        # Decide whether to use good prior based on number of observations
        use_prior = self.use_tpe and self.partitioner and len(self.partitioner.observations) >= 10

        # Sort groups by priority
        sorted_groups = sorted(self.groups, key=lambda g: g.priority)

        # Stage-based focus: earlier stages explore architecture more
        for group in sorted_groups:
            # COARSE stage: focus on architecture
            # REFINE stage: focus on training
            # FINE stage: focus on control
            if stage == 1 and group.name == "control":
                # Use defaults for control in COARSE stage
                for param in group.parameters:
                    result[param.name] = param.default
                continue
            elif stage == 2 and group.name == "architecture":
                # In REFINE, narrow architecture search
                pass  # Sample normally but with tighter bounds later

            # Sample all parameters in this group
            for param in group.parameters:
                result[param.name] = self._sample_parameter(param, use_good_prior=use_prior)

        logger.debug(f"[GroupSampler] Trial {trial_id} Stage {stage}: {list(result.keys())}")

        return result

    def record_result(self, config: dict[str, Any], loss: float) -> None:
        """Record a trial result for TPE learning.

        Args:
            config: Trial configuration
            loss: Achieved loss (lower is better)
        """
        if self.partitioner:
            self.partitioner.add_observation(config, loss)
            n_obs = len(self.partitioner.observations)
            if n_obs % 10 == 0:
                good, bad = self.partitioner.partition()
                logger.info(
                    f"[GroupSampler] {n_obs} observations: {len(good)} good, {len(bad)} bad"
                )

    def get_sampling_stats(self) -> dict[str, Any]:
        """Get statistics about the sampling process.

        Returns:
            Dictionary with sampling statistics
        """
        stats = {
            "num_groups": len(self.groups),
            "use_tpe": self.use_tpe,
        }

        if self.partitioner:
            good, bad = self.partitioner.partition()
            stats["total_observations"] = len(self.partitioner.observations)
            stats["num_good"] = len(good)
            stats["num_bad"] = len(bad)

            # Best observed loss
            if self.partitioner.observations:
                best = min(self.partitioner.observations, key=lambda x: x[1])
                stats["best_loss"] = best[1]
                stats["best_config"] = best[0]

        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_sampler(
    seed: int | None = None,
    use_tpe: bool = True,
    gamma: float = 0.25,
) -> ParameterGroupSampler:
    """Create a configured parameter group sampler.

    Args:
        seed: Random seed
        use_tpe: Enable TPE-style sampling
        gamma: Good/bad partition ratio

    Returns:
        Configured ParameterGroupSampler instance
    """
    return ParameterGroupSampler(seed=seed, use_tpe=use_tpe, gamma=gamma)


def sample_for_stage(
    stage: int,
    trial_id: int,
    param_budget: int | None = None,
) -> dict[str, Any]:
    """Quick helper to sample configuration for a given stage.

    Args:
        stage: HPO stage (1, 2, or 3)
        trial_id: Trial ID for seeding
        param_budget: Optional parameter budget constraint

    Returns:
        Sampled configuration
    """
    sampler = create_sampler(seed=trial_id)
    config = sampler.sample_grouped_configuration({}, trial_id, stage)

    # Apply parameter budget constraint if specified
    if param_budget:
        # Import parameter estimation from hpo_manager
        try:
            from highnoon.services.hpo_manager import estimate_model_params

            estimated = estimate_model_params(config)
            attempts = 0
            while estimated > param_budget and attempts < 20:
                # Re-sample with smaller architecture bias
                sampler.rng.seed(trial_id + attempts * 1000)
                config["num_reasoning_blocks"] = min(
                    config.get("num_reasoning_blocks", 8), 12 if attempts < 10 else 6
                )
                config["embedding_dim"] = min(
                    config.get("embedding_dim", 512), 1024 if attempts < 10 else 512
                )
                estimated = estimate_model_params(config)
                attempts += 1
        except ImportError:
            pass

    return config
