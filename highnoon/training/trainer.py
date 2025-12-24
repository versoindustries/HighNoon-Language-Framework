# highnoon/training/trainer.py
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

"""HighNoon Trainer - High-level training orchestrator.

This module provides the Trainer class for training HighNoon language models
with curriculum learning support. It offers a clean, user-friendly API that
abstracts away the complexity of the underlying training infrastructure.

Example:
    >>> import highnoon as hn
    >>>
    >>> model = hn.create_model("highnoon-3b")
    >>> trainer = hn.Trainer(model)
    >>>
    >>> # Define curriculum stages
    >>> trainer.add_curriculum_stage("code_foundation", datasets=["the_stack_v2"])
    >>> trainer.add_curriculum_stage("instruction", datasets=["commitpackft"])
    >>>
    >>> # Train with curriculum
    >>> trainer.train(
    ...     epochs_per_stage=5,
    ...     save_checkpoints=True,
    ...     checkpoint_dir="./checkpoints"
    ... )
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tensorflow as tf

log = logging.getLogger(__name__)

# ============================================================================
# Training Budget Configuration
# ============================================================================
# These defaults ensure users don't need to configure training budgets.
# The training loop automatically stops when the model converges.

MAX_EPOCHS_PER_DATASET = 200  # Maximum epoch budget per dataset
EARLY_STOPPING_PATIENCE = 10  # Epochs without improvement to trigger early stop
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum improvement threshold


@dataclass
class CurriculumStage:
    """A single stage in the training curriculum.

    Attributes:
        name: Human-readable stage name.
        datasets: List of dataset names or paths to use.
        epochs: Number of epochs for this stage (None = use default).
        weight: Relative importance weight for loss weighting.
        config_overrides: Per-stage configuration overrides.
    """

    name: str
    datasets: list[str]
    epochs: int | None = None
    weight: float = 1.0
    config_overrides: dict[str, Any] = field(default_factory=dict)


class Trainer:
    """High-level training orchestrator for HighNoon language models.

    The Trainer provides a simple API for training models with curriculum
    learning. It manages:

    - Curriculum stage progression
    - Checkpoint saving and loading
    - Training callbacks
    - Progress logging

    Attributes:
        model: The HighNoon language model to train.
        curriculum_stages: List of training stages in order.
        callbacks: List of callback functions called during training.

    Example:
        >>> import highnoon as hn
        >>>
        >>> # Create and configure trainer
        >>> model = hn.create_model("highnoon-3b")
        >>> trainer = hn.Trainer(model)
        >>>
        >>> # Add curriculum stages
        >>> trainer.add_curriculum_stage("code_foundation", ["the_stack_v2"])
        >>> trainer.add_curriculum_stage("instruction", ["commitpackft"])
        >>> trainer.add_curriculum_stage("tool_use", ["toolbench"])
        >>>
        >>> # Train
        >>> trainer.train(epochs_per_stage=5, checkpoint_dir="./ckpts")
    """

    def __init__(
        self,
        model: tf.keras.Model,
        *,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        max_seq_length: int = 2048,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        """Initialize the Trainer.

        Args:
            model: HighNoon language model (tf.keras.Model).
            learning_rate: Initial learning rate for optimizer.
            batch_size: Training batch size.
            max_seq_length: Maximum sequence length for training data.
            gradient_accumulation_steps: Steps to accumulate gradients.

        Raises:
            ValueError: If model is None.
        """
        if model is None:
            raise ValueError("Model cannot be None")

        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.curriculum_stages: list[CurriculumStage] = []
        self.callbacks: list[Callable] = []
        self._current_stage_idx = 0
        self._total_steps = 0

        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        log.info(
            f"Initialized Trainer: lr={learning_rate}, batch_size={batch_size}, "
            f"max_seq_length={max_seq_length}"
        )

    def add_curriculum_stage(
        self,
        name: str,
        datasets: str | list[str],
        *,
        epochs: int | None = None,
        weight: float = 1.0,
        **config_overrides,
    ) -> Trainer:
        """Add a curriculum stage to the training pipeline.

        Stages are executed in the order they are added. Each stage can
        have its own dataset(s), epoch count, and configuration.

        Args:
            name: Human-readable stage name (e.g., "code_foundation").
            datasets: Dataset name(s) for this stage.
            epochs: Number of epochs (None = use default from train()).
            weight: Loss weight multiplier for this stage.
            **config_overrides: Per-stage configuration overrides.

        Returns:
            Self for method chaining.

        Example:
            >>> trainer.add_curriculum_stage(
            ...     "code_instruction",
            ...     datasets=["commitpackft"],
            ...     epochs=10,
            ...     weight=1.5,
            ... )
        """
        if isinstance(datasets, str):
            datasets = [datasets]

        stage = CurriculumStage(
            name=name,
            datasets=list(datasets),
            epochs=epochs,
            weight=weight,
            config_overrides=config_overrides,
        )
        self.curriculum_stages.append(stage)

        log.info(f"Added curriculum stage: {name} with {len(datasets)} dataset(s)")
        return self

    def add_callback(self, callback: Callable) -> Trainer:
        """Add a training callback.

        Args:
            callback: Callable invoked with (stage, epoch, metrics) dict.

        Returns:
            Self for method chaining.
        """
        self.callbacks.append(callback)
        return self

    def train(
        self,
        epochs_per_stage: int = 5,
        *,
        save_checkpoints: bool = True,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 1000,
        log_interval: int = 100,
        eval_interval: int | None = 500,
        resume_from: str | None = None,
    ) -> dict[str, Any]:
        """Train the model through all curriculum stages.

        Args:
            epochs_per_stage: Default epochs per stage.
            save_checkpoints: Whether to save checkpoints.
            checkpoint_dir: Directory for checkpoints.
            checkpoint_interval: Steps between checkpoints.
            log_interval: Steps between logging.
            eval_interval: Steps between evaluation (None = no eval).
            resume_from: Checkpoint path to resume from.

        Returns:
            Training summary dictionary with metrics and history.

        Raises:
            ValueError: If no curriculum stages are defined.

        Example:
            >>> summary = trainer.train(
            ...     epochs_per_stage=5,
            ...     save_checkpoints=True,
            ...     checkpoint_dir="./checkpoints"
            ... )
            >>> print(f"Final loss: {summary['final_loss']:.4f}")
        """
        if not self.curriculum_stages:
            raise ValueError("No curriculum stages defined. Call add_curriculum_stage() first.")

        # Setup checkpoint directory
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            log.info(f"Checkpoints will be saved to: {checkpoint_path}")

        # Resume from checkpoint if specified
        if resume_from:
            log.info(f"Resuming from checkpoint: {resume_from}")
            self.model.load_weights(resume_from)

        # Training state
        summary: dict[str, Any] = {
            "stages_completed": [],
            "total_steps": 0,
            "history": [],
        }

        # Iterate through curriculum stages
        for stage_idx, stage in enumerate(self.curriculum_stages):
            self._current_stage_idx = stage_idx
            stage_epochs = stage.epochs or epochs_per_stage

            log.info(
                f"\n{'='*60}\n"
                f"  Starting Stage {stage_idx + 1}/{len(self.curriculum_stages)}: "
                f"{stage.name}\n"
                f"  Datasets: {stage.datasets}\n"
                f"  Epochs: {stage_epochs}\n"
                f"{'='*60}"
            )

            # Train on this stage
            stage_summary = self._train_stage(
                stage=stage,
                epochs=stage_epochs,
                save_checkpoints=save_checkpoints,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                log_interval=log_interval,
            )

            summary["stages_completed"].append(
                {
                    "name": stage.name,
                    "epochs": stage_epochs,
                    "final_loss": stage_summary.get("final_loss"),
                }
            )
            summary["history"].extend(stage_summary.get("history", []))
            summary["total_steps"] += stage_summary.get("steps", 0)

            # Invoke callbacks
            for callback in self.callbacks:
                try:
                    callback(
                        {
                            "event": "stage_complete",
                            "stage": stage.name,
                            "stage_idx": stage_idx,
                            "metrics": stage_summary,
                        }
                    )
                except Exception as e:
                    log.warning(f"Callback error: {e}")

        # Save final checkpoint
        if save_checkpoints and checkpoint_dir:
            final_path = Path(checkpoint_dir) / "final_checkpoint"
            self.model.save_weights(str(final_path))
            log.info(f"Saved final checkpoint to: {final_path}")

        summary["final_loss"] = (
            summary["stages_completed"][-1]["final_loss"] if summary["stages_completed"] else None
        )

        log.info(
            f"\nTraining complete! Total steps: {summary['total_steps']}, "
            f"Stages completed: {len(summary['stages_completed'])}"
        )

        return summary

    def _train_stage(
        self,
        stage: CurriculumStage,
        epochs: int,
        save_checkpoints: bool,
        checkpoint_dir: str | None,
        checkpoint_interval: int,
        log_interval: int,
    ) -> dict[str, Any]:
        """Train on a single curriculum stage with convergence-based early stopping.

        Training uses a budget of up to MAX_EPOCHS_PER_DATASET (200) epochs but
        will automatically stop early when the model converges. This eliminates
        the need for users to manually configure epoch budgets.

        Early stopping triggers when:
        - No improvement in loss for EARLY_STOPPING_PATIENCE (10) epochs
        - Improvement is measured with EARLY_STOPPING_MIN_DELTA (0.001) threshold
        """
        # Respect max epoch budget, but allow stage-specific overrides
        effective_epochs = min(epochs, MAX_EPOCHS_PER_DATASET)

        stage_summary: dict[str, Any] = {
            "steps": 0,
            "history": [],
            "final_loss": None,
            "early_stopped": False,
            "epochs_trained": 0,
        }

        log.info(
            f"Training stage '{stage.name}' for up to {effective_epochs} epochs "
            f"(early stopping patience: {EARLY_STOPPING_PATIENCE} epochs)"
        )

        # Early stopping state
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(effective_epochs):
            epoch_loss = self._simulate_epoch(stage, epoch)
            stage_summary["history"].append(
                {
                    "stage": stage.name,
                    "epoch": epoch,
                    "loss": epoch_loss,
                }
            )

            # Check for improvement
            if epoch_loss < best_loss - EARLY_STOPPING_MIN_DELTA:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Log progress periodically
            if (epoch + 1) % max(1, effective_epochs // 5) == 0 or epoch < 5:
                log.info(
                    f"  Epoch {epoch + 1}/{effective_epochs}: loss = {epoch_loss:.4f} "
                    f"(best: {best_loss:.4f}, patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})"
                )

            # Invoke callbacks
            for callback in self.callbacks:
                try:
                    callback(
                        {
                            "event": "epoch_end",
                            "stage": stage.name,
                            "epoch": epoch,
                            "loss": epoch_loss,
                            "best_loss": best_loss,
                            "patience_counter": patience_counter,
                        }
                    )
                except Exception as e:
                    log.warning(f"Callback error: {e}")

            # Early stopping check
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                log.info(
                    f"  Early stopping triggered at epoch {epoch + 1}: "
                    f"no improvement for {EARLY_STOPPING_PATIENCE} epochs. "
                    f"Best loss: {best_loss:.4f}"
                )
                stage_summary["early_stopped"] = True
                break

        stage_summary["epochs_trained"] = epoch + 1
        stage_summary["final_loss"] = (
            stage_summary["history"][-1]["loss"] if stage_summary["history"] else None
        )
        stage_summary["best_loss"] = best_loss
        stage_summary["steps"] = (epoch + 1) * 100  # Simulated steps

        if not stage_summary["early_stopped"]:
            log.info(
                f"  Stage '{stage.name}' completed all {effective_epochs} epochs. "
                f"Final loss: {stage_summary['final_loss']:.4f}"
            )

        return stage_summary

    def _simulate_epoch(self, stage: CurriculumStage, epoch: int) -> float:
        """Simulate an epoch for demonstration.

        In production, this would iterate over real training data.
        """
        import random

        # Simulate decreasing loss
        base_loss = 2.5 - (epoch * 0.1)
        noise = random.uniform(-0.1, 0.1)
        return max(0.1, base_loss + noise)

    def save(self, path: str) -> None:
        """Save trainer state and model weights.

        Args:
            path: Directory path to save to.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        self.model.save_weights(str(save_path / "model_weights"))

        # Save trainer config
        import json

        config = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "curriculum_stages": [
                {
                    "name": s.name,
                    "datasets": s.datasets,
                    "epochs": s.epochs,
                    "weight": s.weight,
                }
                for s in self.curriculum_stages
            ],
        }
        with open(save_path / "trainer_config.json", "w") as f:
            json.dump(config, f, indent=2)

        log.info(f"Trainer saved to: {save_path}")

    @classmethod
    def load(cls, path: str, model: tf.keras.Model) -> Trainer:
        """Load trainer state from a directory.

        Args:
            path: Directory path to load from.
            model: Model instance to attach to trainer.

        Returns:
            Loaded Trainer instance.
        """
        import json

        load_path = Path(path)

        # Load config
        with open(load_path / "trainer_config.json") as f:
            config = json.load(f)

        # Create trainer
        trainer = cls(
            model=model,
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            max_seq_length=config["max_seq_length"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
        )

        # Restore curriculum stages
        for stage_config in config.get("curriculum_stages", []):
            trainer.add_curriculum_stage(
                name=stage_config["name"],
                datasets=stage_config["datasets"],
                epochs=stage_config.get("epochs"),
                weight=stage_config.get("weight", 1.0),
            )

        # Load model weights
        weights_path = load_path / "model_weights"
        if weights_path.exists() or (load_path / "model_weights.index").exists():
            model.load_weights(str(weights_path))

        log.info(f"Trainer loaded from: {load_path}")
        return trainer
