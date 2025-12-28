# highnoon/training/pbt_callback.py
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

"""Population-Based Training (PBT) Callback for HPO.

This module implements Population-Based Training weight transfer, enabling
trials to inherit weights from better-performing trials during training.

Key Features:
1. Periodic checkpoint saving with trial metadata
2. Exploit: Copy weights from better performing trials
3. Explore: Perturb hyperparameters after exploitation
4. Cross-trial communication via shared checkpoint directory

References:
- Jaderberg et al., "Population Based Training of Neural Networks" (2017)
- FIRE-PBT: Prioritizing long-term improvement rate
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class PBTConfig:
    """Configuration for Population-Based Training.

    Attributes:
        checkpoint_dir: Shared directory for PBT checkpoints
        checkpoint_freq_steps: Steps between checkpoint saves
        exploit_check_freq_steps: Steps between exploit checks
        exploit_threshold: Loss ratio to trigger exploit (1.2 = 20% worse)
        explore_perturbation: Perturbation factor for explore phase
        min_steps_before_exploit: Minimum steps before first exploit
        exploit_probability: Probability of exploiting when conditions met
        perturb_params: List of hyperparameters to perturb
    """

    checkpoint_dir: str = "pbt_checkpoints"
    checkpoint_freq_steps: int = 100
    exploit_check_freq_steps: int = 200
    exploit_threshold: float = 1.2  # Exploit if 20% worse than best
    explore_perturbation: float = 0.2  # ±20% perturbation
    min_steps_before_exploit: int = 500
    exploit_probability: float = 0.5
    perturb_params: list[str] = field(default_factory=lambda: ["learning_rate", "batch_size"])


@dataclass
class TrialCheckpoint:
    """Metadata for a PBT checkpoint.

    Attributes:
        trial_id: Trial identifier
        step: Training step when checkpoint was saved
        loss: Loss at checkpoint time
        hyperparams: Hyperparameters at checkpoint time
        checkpoint_path: Path to weights file
        timestamp: Unix timestamp when saved
    """

    trial_id: str
    step: int
    loss: float
    hyperparams: dict[str, Any]
    checkpoint_path: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trial_id": self.trial_id,
            "step": self.step,
            "loss": self.loss,
            "hyperparams": self.hyperparams,
            "checkpoint_path": self.checkpoint_path,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrialCheckpoint:
        """Create from dictionary."""
        return cls(
            trial_id=data["trial_id"],
            step=data["step"],
            loss=data["loss"],
            hyperparams=data["hyperparams"],
            checkpoint_path=data["checkpoint_path"],
            timestamp=data["timestamp"],
        )


class PBTCheckpointManager:
    """Manager for PBT checkpoint storage and retrieval.

    Handles saving and loading checkpoints across trials in a sweep.
    Uses a shared directory with JSON metadata files.

    Attributes:
        checkpoint_dir: Root directory for checkpoints
        sweep_id: Sweep identifier for namespacing
    """

    def __init__(self, checkpoint_dir: str | Path, sweep_id: str = "default"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Root directory for PBT checkpoints
            sweep_id: Sweep identifier for namespacing
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.sweep_id = sweep_id
        self._sweep_dir = self.checkpoint_dir / sweep_id

        # Ensure directories exist
        self._sweep_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[PBT] Checkpoint manager initialized at {self._sweep_dir}")

    def save_checkpoint(
        self,
        trial_id: str,
        step: int,
        loss: float,
        hyperparams: dict[str, Any],
        model: tf.keras.Model,
    ) -> TrialCheckpoint:
        """Save a checkpoint for a trial.

        Args:
            trial_id: Trial identifier
            step: Current training step
            loss: Current loss value
            hyperparams: Current hyperparameters
            model: Model to checkpoint

        Returns:
            TrialCheckpoint metadata
        """
        # Create trial directory
        trial_dir = self._sweep_dir / trial_id
        trial_dir.mkdir(exist_ok=True)

        # Save weights
        weights_path = trial_dir / f"step_{step}.weights.h5"
        try:
            model.save_weights(str(weights_path))
        except Exception as e:
            logger.warning(f"[PBT] Failed to save weights: {e}")
            return None

        # Create checkpoint metadata
        checkpoint = TrialCheckpoint(
            trial_id=trial_id,
            step=step,
            loss=loss,
            hyperparams={k: v for k, v in hyperparams.items() if not k.startswith("_")},
            checkpoint_path=str(weights_path),
            timestamp=time.time(),
        )

        # Save metadata
        meta_path = trial_dir / f"step_{step}.json"
        with open(meta_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        # Also update "latest" pointer
        latest_path = trial_dir / "latest.json"
        with open(latest_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        logger.debug(f"[PBT] Saved checkpoint: {trial_id} step={step} loss={loss:.4f}")
        return checkpoint

    def get_all_latest_checkpoints(self) -> list[TrialCheckpoint]:
        """Get latest checkpoints from all trials in the sweep.

        Returns:
            List of TrialCheckpoint for all active trials
        """
        checkpoints = []

        if not self._sweep_dir.exists():
            return checkpoints

        for trial_dir in self._sweep_dir.iterdir():
            if not trial_dir.is_dir():
                continue

            latest_path = trial_dir / "latest.json"
            if latest_path.exists():
                try:
                    with open(latest_path) as f:
                        data = json.load(f)
                    checkpoint = TrialCheckpoint.from_dict(data)
                    checkpoints.append(checkpoint)
                except Exception as e:
                    logger.warning(f"[PBT] Failed to load checkpoint {latest_path}: {e}")

        return checkpoints

    def get_best_checkpoint(self, exclude_trial_id: str | None = None) -> TrialCheckpoint | None:
        """Get checkpoint from the best performing trial.

        Args:
            exclude_trial_id: Trial ID to exclude (current trial)

        Returns:
            Best checkpoint or None if no checkpoints available
        """
        checkpoints = self.get_all_latest_checkpoints()

        # Filter out excluded trial
        if exclude_trial_id:
            checkpoints = [c for c in checkpoints if c.trial_id != exclude_trial_id]

        if not checkpoints:
            return None

        # Sort by loss (lower is better)
        checkpoints.sort(key=lambda c: c.loss)
        return checkpoints[0]

    def load_checkpoint(
        self,
        checkpoint: TrialCheckpoint,
        model: tf.keras.Model,
    ) -> bool:
        """Load weights from a checkpoint into a model.

        Args:
            checkpoint: Checkpoint to load
            model: Model to load weights into

        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(checkpoint.checkpoint_path):
                model.load_weights(checkpoint.checkpoint_path)
                logger.info(
                    f"[PBT] Loaded weights from {checkpoint.trial_id} "
                    f"(step={checkpoint.step}, loss={checkpoint.loss:.4f})"
                )
                return True
            else:
                logger.warning(f"[PBT] Checkpoint file not found: {checkpoint.checkpoint_path}")
                return False
        except Exception as e:
            logger.error(f"[PBT] Failed to load checkpoint: {e}")
            return False


class PBTCallback:
    """Population-Based Training callback for TrainingEngine.

    Implements exploit (copy weights from better trial) and explore
    (perturb hyperparameters) during training.

    Example:
        >>> callback = PBTCallback(
        ...     trial_id="trial_001",
        ...     hyperparams={"learning_rate": 0.001},
        ...     model=model,
        ...     optimizer=optimizer,
        ...     sweep_id="sweep_001",
        ... )
        >>> engine.run(epochs=10, callbacks=[callback])
    """

    def __init__(
        self,
        trial_id: str,
        hyperparams: dict[str, Any],
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        sweep_id: str = "default",
        config: PBTConfig | None = None,
        on_hyperparams_updated: Callable[[dict[str, Any]], None] | None = None,
    ):
        """Initialize PBT callback.

        Args:
            trial_id: Unique trial identifier
            hyperparams: Current hyperparameters
            model: Model being trained
            optimizer: Optimizer being used
            sweep_id: Sweep identifier
            config: PBT configuration
            on_hyperparams_updated: Callback when hyperparams are perturbed
        """
        self.trial_id = trial_id
        self.hyperparams = hyperparams.copy()
        self.model = model
        self.optimizer = optimizer
        self.config = config or PBTConfig()
        self.on_hyperparams_updated = on_hyperparams_updated

        # Initialize checkpoint manager
        self.checkpoint_manager = PBTCheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            sweep_id=sweep_id,
        )

        # State tracking
        self._step_count = 0
        self._current_loss = float("inf")
        self._exploit_count = 0
        self._explore_count = 0

    def on_batch_start(self, step: int) -> bool:
        """Called before each training step."""
        return True

    def on_batch_end(self, step: int, result: Any) -> bool:
        """Called after each training step.

        Handles checkpoint saving and exploit/explore decisions.

        Args:
            step: Current step number
            result: StepResult from training

        Returns:
            True to continue training
        """
        self._step_count += 1
        self._current_loss = result.loss

        # Save checkpoint periodically
        if self._step_count % self.config.checkpoint_freq_steps == 0:
            self.checkpoint_manager.save_checkpoint(
                trial_id=self.trial_id,
                step=self._step_count,
                loss=result.loss,
                hyperparams=self.hyperparams,
                model=self.model,
            )

        # Check for exploit opportunity
        if (
            self._step_count >= self.config.min_steps_before_exploit
            and self._step_count % self.config.exploit_check_freq_steps == 0
        ):
            self._maybe_exploit()

        return True

    def on_epoch_start(self, epoch: int) -> bool:
        """Called before each epoch."""
        return True

    def on_epoch_end(self, epoch: int, result: Any) -> bool:
        """Called after each epoch.

        Always saves checkpoint at epoch end.

        Args:
            epoch: Current epoch number
            result: EpochResult from training

        Returns:
            True to continue training
        """
        self.checkpoint_manager.save_checkpoint(
            trial_id=self.trial_id,
            step=self._step_count,
            loss=result.mean_loss,
            hyperparams=self.hyperparams,
            model=self.model,
        )
        return True

    def _maybe_exploit(self) -> None:
        """Check if we should exploit (copy from better trial)."""
        # Get best checkpoint from other trials
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint(
            exclude_trial_id=self.trial_id
        )

        if best_checkpoint is None:
            logger.debug("[PBT] No other trials available for exploit")
            return

        # Check if we're significantly worse than best
        if self._current_loss > best_checkpoint.loss * self.config.exploit_threshold:
            # Probabilistic exploit
            if random.random() < self.config.exploit_probability:
                self._exploit(best_checkpoint)

    def _exploit(self, checkpoint: TrialCheckpoint) -> None:
        """Exploit: Copy weights from better trial.

        Args:
            checkpoint: Checkpoint to exploit from
        """
        logger.info(
            f"[PBT] Trial {self.trial_id} EXPLOITING from {checkpoint.trial_id} "
            f"(our loss={self._current_loss:.4f}, their loss={checkpoint.loss:.4f})"
        )

        # Load weights from better trial
        success = self.checkpoint_manager.load_checkpoint(checkpoint, self.model)

        if success:
            self._exploit_count += 1

            # Explore: Perturb hyperparameters
            self._explore(checkpoint.hyperparams)

    def _explore(self, base_hyperparams: dict[str, Any]) -> None:
        """Explore: Perturb hyperparameters after exploitation.

        Args:
            base_hyperparams: Hyperparameters from exploited checkpoint
        """
        new_hyperparams = base_hyperparams.copy()
        perturbation = self.config.explore_perturbation

        for param_name in self.config.perturb_params:
            if param_name not in new_hyperparams:
                continue

            value = new_hyperparams[param_name]

            if isinstance(value, float):
                # Random multiplicative perturbation
                factor = random.uniform(1 - perturbation, 1 + perturbation)
                new_value = value * factor
                new_hyperparams[param_name] = new_value
                logger.debug(f"[PBT] Perturbed {param_name}: {value:.4e} -> {new_value:.4e}")

            elif isinstance(value, int):
                # Random additive perturbation
                delta = max(1, int(value * perturbation))
                new_value = value + random.randint(-delta, delta)
                new_hyperparams[param_name] = max(1, new_value)
                logger.debug(
                    f"[PBT] Perturbed {param_name}: {value} -> {new_hyperparams[param_name]}"
                )

        # Apply learning rate if perturbed
        if "learning_rate" in new_hyperparams:
            new_lr = new_hyperparams["learning_rate"]
            try:
                if hasattr(self.optimizer, "learning_rate"):
                    lr_var = self.optimizer.learning_rate
                    if hasattr(lr_var, "assign"):
                        lr_var.assign(new_lr)
                        logger.info(f"[PBT] Updated optimizer LR to {new_lr:.4e}")
            except Exception as e:
                logger.warning(f"[PBT] Failed to update learning rate: {e}")

        # Update local hyperparams
        self.hyperparams.update(new_hyperparams)
        self._explore_count += 1

        # Notify callback if provided
        if self.on_hyperparams_updated:
            self.on_hyperparams_updated(self.hyperparams)

        logger.info(
            f"[PBT] Trial {self.trial_id} EXPLORED with {len(self.config.perturb_params)} perturbations"
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get PBT statistics for monitoring.

        Returns:
            Dictionary with PBT state and metrics
        """
        return {
            "trial_id": self.trial_id,
            "step_count": self._step_count,
            "current_loss": self._current_loss,
            "exploit_count": self._exploit_count,
            "explore_count": self._explore_count,
            "hyperparams": self.hyperparams,
            "config": {
                "exploit_threshold": self.config.exploit_threshold,
                "explore_perturbation": self.config.explore_perturbation,
            },
        }


__all__ = [
    "PBTConfig",
    "TrialCheckpoint",
    "PBTCheckpointManager",
    "PBTCallback",
]
