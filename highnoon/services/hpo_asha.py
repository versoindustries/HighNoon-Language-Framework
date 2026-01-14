# highnoon/services/hpo_asha.py
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

"""Advanced Multi-Fidelity Scheduling for QAHPO.

Enterprise Enhancement Phase 3: Advanced scheduling including:
    - ASHA: Asynchronous Successive Halving Algorithm
    - MOBSTER: Model-Based ASHA with GP prioritization
    - Freeze-Thaw BO: Dynamic pause/resume based on curve predictions

References:
    - Li et al., "A System for Massively Parallel Hyperparameter Tuning" (MLSys 2020)
    - Klein et al., "MOBSTER: Model-Based Asynchronous Successive Halving" (2020)
    - Swersky et al., "Freeze-Thaw Bayesian Optimization" (ICML 2015)

Example:
    >>> asha = ASHAScheduler(max_t=100, grace_period=5, reduction_factor=3)
    >>> rung = asha.on_trial_start(trial_id, config)
    >>> asha.on_trial_result(trial_id, epoch=10, loss=0.5)
    >>> action = asha.decide(trial_id)  # "continue" or "stop"
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TrialAction(str, Enum):
    """Action for a trial."""

    CONTINUE = "continue"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    PROMOTE = "promote"


@dataclass
class ASHATrial:
    """State of a trial in ASHA scheduler.

    Attributes:
        trial_id: Unique trial identifier.
        config: Hyperparameter configuration.
        current_rung: Current rung level (0 = lowest fidelity).
        resource: Current resource usage (epochs).
        loss: Best loss observed.
        status: Current status.
        history: Loss history at each resource level.
    """

    trial_id: str
    config: dict[str, Any]
    current_rung: int = 0
    resource: int = 0
    loss: float = float("inf")
    status: str = "running"
    history: list[tuple[int, float]] = field(default_factory=list)


class ASHAScheduler:
    """Asynchronous Successive Halving Algorithm (ASHA).

    Unlike synchronous Hyperband, ASHA makes promotion decisions
    asynchronously as soon as trials complete their current rung.

    Key Features:
        - No bracket synchronization → better GPU utilization
        - Immediate promotion decisions
        - Early stopping for unpromising trials

    Attributes:
        max_t: Maximum resource budget (e.g., epochs).
        grace_period: Minimum resource before early stopping.
        reduction_factor: Halving factor (keep top 1/eta).
        rungs: Resource levels for evaluation.
    """

    def __init__(
        self,
        max_t: int = 100,
        grace_period: int = 5,
        reduction_factor: int = 3,
        mode: str = "min",
    ) -> None:
        """Initialize ASHA scheduler.

        Args:
            max_t: Maximum resource budget.
            grace_period: Minimum resources before stopping.
            reduction_factor: Factor for successive halving.
            mode: Optimization mode ("min" or "max").
        """
        self.max_t = max_t
        self.grace_period = grace_period
        self.reduction_factor = reduction_factor
        self.mode = mode

        # Compute rungs: geometric sequence from grace_period to max_t
        self.rungs = []
        rung = grace_period
        while rung <= max_t:
            self.rungs.append(rung)
            rung = int(rung * reduction_factor)
        if self.rungs[-1] != max_t:
            self.rungs.append(max_t)

        # Trial tracking
        self._trials: dict[str, ASHATrial] = {}
        self._rung_results: dict[int, list[tuple[str, float]]] = {r: [] for r in self.rungs}

        logger.info(f"[ASHA] Initialized with rungs: {self.rungs}")

    def on_trial_start(self, trial_id: str, config: dict[str, Any]) -> int:
        """Register a new trial.

        Args:
            trial_id: Unique trial identifier.
            config: Trial configuration.

        Returns:
            First rung level (resource target).
        """
        self._trials[trial_id] = ASHATrial(
            trial_id=trial_id,
            config=config.copy(),
            current_rung=0,
        )
        return self.rungs[0]

    def on_trial_result(self, trial_id: str, resource: int, loss: float) -> TrialAction:
        """Report trial result and get decision.

        Args:
            trial_id: Trial identifier.
            resource: Current resource (epoch).
            loss: Current loss.

        Returns:
            Action for the trial.
        """
        if trial_id not in self._trials:
            return TrialAction.STOP

        trial = self._trials[trial_id]
        trial.resource = resource
        trial.history.append((resource, loss))

        # Update best loss
        if self.mode == "min":
            trial.loss = min(trial.loss, loss)
        else:
            trial.loss = max(trial.loss, loss)

        # Check if at a rung boundary
        for i, rung in enumerate(self.rungs):
            if resource == rung:
                trial.current_rung = i
                self._rung_results[rung].append((trial_id, trial.loss))
                return self._decide_promotion(trial_id, i)

        return TrialAction.CONTINUE

    def _decide_promotion(self, trial_id: str, rung_idx: int) -> TrialAction:
        """Decide if trial should be promoted to next rung.

        Args:
            trial_id: Trial identifier.
            rung_idx: Current rung index.

        Returns:
            Action for the trial.
        """
        if rung_idx >= len(self.rungs) - 1:
            # At maximum rung, trial is complete
            self._trials[trial_id].status = "completed"
            return TrialAction.STOP

        rung = self.rungs[rung_idx]
        results = self._rung_results[rung]

        if len(results) < 2:
            # Not enough trials to compare, continue
            return TrialAction.CONTINUE

        # Sort by loss
        if self.mode == "min":
            sorted_results = sorted(results, key=lambda x: x[1])
        else:
            sorted_results = sorted(results, key=lambda x: -x[1])

        # Check if in top 1/eta
        num_promote = max(1, len(sorted_results) // self.reduction_factor)
        top_trials = [tid for tid, _ in sorted_results[:num_promote]]

        if trial_id in top_trials:
            # Promote to next rung
            return TrialAction.PROMOTE
        else:
            # Stop this trial
            self._trials[trial_id].status = "stopped"
            return TrialAction.STOP

    def get_next_rung(self, trial_id: str) -> int | None:
        """Get target resource for next rung.

        Args:
            trial_id: Trial identifier.

        Returns:
            Target resource or None if complete.
        """
        if trial_id not in self._trials:
            return None

        trial = self._trials[trial_id]
        if trial.current_rung >= len(self.rungs) - 1:
            return None

        return self.rungs[trial.current_rung + 1]

    def get_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "type": "ASHA",
            "max_t": self.max_t,
            "grace_period": self.grace_period,
            "reduction_factor": self.reduction_factor,
            "rungs": self.rungs,
            "num_trials": len(self._trials),
            "completed": sum(1 for t in self._trials.values() if t.status == "completed"),
            "stopped": sum(1 for t in self._trials.values() if t.status == "stopped"),
        }


class MOBSTERScheduler(ASHAScheduler):
    """Model-Based Asynchronous Successive Halving (MOBSTER).

    Extends ASHA with GP-based prioritization for promotion decisions.
    Uses surrogate predictions instead of just observed losses.

    Attributes:
        gp_surrogate: Gaussian Process surrogate for predictions.
        use_predictions: Whether to use GP predictions for promotion.
    """

    def __init__(
        self,
        max_t: int = 100,
        grace_period: int = 5,
        reduction_factor: int = 3,
        mode: str = "min",
        use_predictions: bool = True,
    ) -> None:
        """Initialize MOBSTER scheduler.

        Args:
            max_t: Maximum resource budget.
            grace_period: Minimum resources before stopping.
            reduction_factor: Successive halving factor.
            mode: Optimization mode.
            use_predictions: Use GP predictions for promotion.
        """
        super().__init__(max_t, grace_period, reduction_factor, mode)
        self.use_predictions = use_predictions

        # Simple GP approximation (in practice would use real GP)
        self._config_losses: list[tuple[dict[str, Any], float]] = []
        self._mean_loss: float = 0.0
        self._std_loss: float = 1.0

    def _update_surrogate(self, config: dict[str, Any], loss: float) -> None:
        """Update the surrogate model.

        Args:
            config: Configuration.
            loss: Observed loss.
        """
        self._config_losses.append((config, loss))

        # Update statistics
        losses = [l for _, l in self._config_losses]
        self._mean_loss = np.mean(losses)
        self._std_loss = np.std(losses) + 1e-8

    def _predict_final_loss(self, trial: ASHATrial) -> tuple[float, float]:
        """Predict final loss for a trial.

        Args:
            trial: Trial state.

        Returns:
            Tuple of (mean_prediction, uncertainty).
        """
        if len(trial.history) < 2:
            return trial.loss, 1.0

        # Simple linear extrapolation
        resources = [r for r, _ in trial.history]
        losses = [l for _, l in trial.history]

        if len(resources) >= 3:
            # Use last 3 points for trend
            slope = (losses[-1] - losses[-3]) / (resources[-1] - resources[-3] + 1e-10)
            remaining = self.max_t - trial.resource
            predicted = losses[-1] + slope * remaining
        else:
            # Fall back to current loss
            predicted = trial.loss

        # Uncertainty decreases with more observations
        uncertainty = self._std_loss / math.sqrt(len(trial.history))

        return predicted, uncertainty

    def _decide_promotion(self, trial_id: str, rung_idx: int) -> TrialAction:
        """Model-based promotion decision using GP predictions.

        Args:
            trial_id: Trial identifier.
            rung_idx: Current rung index.

        Returns:
            Action for the trial.
        """
        if not self.use_predictions:
            return super()._decide_promotion(trial_id, rung_idx)

        if rung_idx >= len(self.rungs) - 1:
            self._trials[trial_id].status = "completed"
            return TrialAction.STOP

        rung = self.rungs[rung_idx]
        trial = self._trials[trial_id]

        # Update surrogate
        self._update_surrogate(trial.config, trial.loss)

        # Get predictions for all trials at this rung
        predictions = []
        for results_trial_id, _ in self._rung_results[rung]:
            if results_trial_id in self._trials:
                t = self._trials[results_trial_id]
                pred, unc = self._predict_final_loss(t)
                predictions.append((results_trial_id, pred, unc))

        if len(predictions) < 2:
            return TrialAction.CONTINUE

        # Sort by predicted loss (with UCB for exploration)
        if self.mode == "min":
            # LCB for minimization
            sorted_preds = sorted(predictions, key=lambda x: x[1] - 0.5 * x[2])
        else:
            sorted_preds = sorted(predictions, key=lambda x: -(x[1] + 0.5 * x[2]))

        # Promote top 1/eta based on predictions
        num_promote = max(1, len(sorted_preds) // self.reduction_factor)
        top_trials = [tid for tid, _, _ in sorted_preds[:num_promote]]

        if trial_id in top_trials:
            return TrialAction.PROMOTE
        else:
            self._trials[trial_id].status = "stopped"
            return TrialAction.STOP

    def get_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        stats = super().get_statistics()
        stats["type"] = "MOBSTER"
        stats["use_predictions"] = self.use_predictions
        stats["surrogate_samples"] = len(self._config_losses)
        return stats


@dataclass
class FreezeThawTrial:
    """Trial state for Freeze-Thaw BO.

    Attributes:
        trial_id: Trial identifier.
        config: Configuration.
        resource: Current resource.
        loss: Current loss.
        status: running, paused, completed.
        curve: Full learning curve.
        predicted_final: Predicted final loss.
        priority: Priority for resumption.
    """

    trial_id: str
    config: dict[str, Any]
    resource: int = 0
    loss: float = float("inf")
    status: str = "running"
    curve: list[float] = field(default_factory=list)
    predicted_final: float = float("inf")
    priority: float = 0.0


class FreezeThawBO:
    """Freeze-Thaw Bayesian Optimization.

    Dynamically pauses and resumes trials based on learning curve
    predictions, allocating resources to most promising trials.

    Key Features:
        - Predicts full learning curves
        - Pauses unpromising trials
        - Resumes when uncertainty increases

    Attributes:
        max_t: Maximum resources.
        max_parallel: Maximum parallel trials.
        pause_threshold: Predicted loss threshold for pausing.
    """

    def __init__(
        self,
        max_t: int = 100,
        max_parallel: int = 4,
        pause_threshold_quantile: float = 0.5,
        min_observations: int = 5,
    ) -> None:
        """Initialize Freeze-Thaw BO.

        Args:
            max_t: Maximum resource budget.
            max_parallel: Max parallel running trials.
            pause_threshold_quantile: Quantile for pause decision.
            min_observations: Min observations before pausing.
        """
        self.max_t = max_t
        self.max_parallel = max_parallel
        self.pause_threshold_quantile = pause_threshold_quantile
        self.min_observations = min_observations

        self._trials: dict[str, FreezeThawTrial] = {}
        self._paused_queue: list[tuple[float, str]] = []  # Min-heap by priority
        self._best_final: float = float("inf")

    def add_trial(self, trial_id: str, config: dict[str, Any]) -> None:
        """Add a new trial.

        Args:
            trial_id: Trial identifier.
            config: Configuration.
        """
        self._trials[trial_id] = FreezeThawTrial(
            trial_id=trial_id,
            config=config.copy(),
        )

    def update(self, trial_id: str, loss: float) -> TrialAction:
        """Update trial with new observation.

        Args:
            trial_id: Trial identifier.
            loss: New loss observation.

        Returns:
            Action for the trial.
        """
        if trial_id not in self._trials:
            return TrialAction.STOP

        trial = self._trials[trial_id]
        trial.curve.append(loss)
        trial.resource = len(trial.curve)
        trial.loss = min(trial.loss, loss)

        if trial.resource >= self.max_t:
            trial.status = "completed"
            self._best_final = min(self._best_final, trial.loss)
            return TrialAction.STOP

        # Predict final loss
        predicted, uncertainty = self._predict_final(trial)
        trial.predicted_final = predicted
        trial.priority = -uncertainty  # Higher uncertainty = higher priority

        # Decide whether to pause
        if trial.resource >= self.min_observations:
            threshold = self._get_pause_threshold()
            if predicted > threshold * 1.2:  # 20% worse than threshold
                trial.status = "paused"
                heapq.heappush(self._paused_queue, (trial.priority, trial_id))
                return TrialAction.PAUSE

        return TrialAction.CONTINUE

    def _predict_final(self, trial: FreezeThawTrial) -> tuple[float, float]:
        """Predict final loss from partial curve.

        Args:
            trial: Trial state.

        Returns:
            Tuple of (predicted_loss, uncertainty).
        """
        if len(trial.curve) < 3:
            return trial.loss, 1.0

        # Exponential decay model: loss(t) = a * exp(-b*t) + c
        # Simplified: use log-linear extrapolation
        x = np.arange(1, len(trial.curve) + 1)
        y = np.array(trial.curve)

        try:
            # Fit log-linear model
            log_y = np.log(y + 1e-10)
            coeffs = np.polyfit(x, log_y, 1)
            slope, intercept = coeffs

            # Extrapolate
            predicted_log = intercept + slope * self.max_t
            predicted = np.exp(predicted_log)

            # Uncertainty from residuals
            residuals = log_y - (intercept + slope * x)
            uncertainty = np.std(residuals) * (self.max_t - len(trial.curve))

            return float(predicted), float(uncertainty)
        except Exception:
            return trial.loss, 1.0

    def _get_pause_threshold(self) -> float:
        """Get threshold for pausing decisions."""
        if not self._trials:
            return float("inf")

        predicted_losses = [t.predicted_final for t in self._trials.values()]
        return float(np.quantile(predicted_losses, self.pause_threshold_quantile))

    def get_trial_to_resume(self) -> str | None:
        """Get next trial to resume from paused queue.

        Returns:
            Trial ID to resume or None.
        """
        running = sum(1 for t in self._trials.values() if t.status == "running")

        if running >= self.max_parallel:
            return None

        while self._paused_queue:
            _, trial_id = heapq.heappop(self._paused_queue)
            if trial_id in self._trials and self._trials[trial_id].status == "paused":
                self._trials[trial_id].status = "running"
                return trial_id

        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "type": "FreezeThawBO",
            "max_t": self.max_t,
            "max_parallel": self.max_parallel,
            "num_trials": len(self._trials),
            "running": sum(1 for t in self._trials.values() if t.status == "running"),
            "paused": sum(1 for t in self._trials.values() if t.status == "paused"),
            "completed": sum(1 for t in self._trials.values() if t.status == "completed"),
            "best_final": self._best_final if self._best_final < float("inf") else None,
        }


__all__ = [
    "ASHAScheduler",
    "MOBSTERScheduler",
    "FreezeThawBO",
    "TrialAction",
    "ASHATrial",
    "FreezeThawTrial",
]
