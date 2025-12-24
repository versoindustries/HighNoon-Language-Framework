"""
Intelligent checkpoint manager with Pareto-optimal model saving.

Implements a 5-checkpoint strategy:
- Checkpoint 1: Best accuracy (no parameter consideration)
- Checkpoints 2-5: Pareto frontier of accuracy vs parameter count

This ensures we save:
1. The absolute best performing model
2-5. Models that offer good accuracy with fewer parameters (efficiency-accuracy trade-off)
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetrics:
    """Metrics for a saved checkpoint."""

    checkpoint_id: int
    accuracy: float
    loss: float
    param_count: int
    step: int
    epoch: int
    model_path: str
    timestamp: str
    is_best_accuracy: bool = False
    is_pareto_optimal: bool = False
    pareto_rank: int | None = None


class ParetoCheckpointManager:
    """
    Manages 5 checkpoints using Pareto optimization.

    The manager maintains:
    - 1 checkpoint for best accuracy
    - 4 checkpoints on the Pareto frontier (accuracy vs parameter count)
    """

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Base directory for saving checkpoints
            max_checkpoints: Total number of checkpoints to maintain (default: 5)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.checkpoints: list[CheckpointMetrics] = []
        self.best_accuracy = -float("inf")
        self.checkpoint_counter = 0

        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        """Load existing checkpoint metadata if available."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    data = json.load(f)
                    self.checkpoints = [
                        CheckpointMetrics(**cp) for cp in data.get("checkpoints", [])
                    ]
                    self.best_accuracy = data.get("best_accuracy", -float("inf"))
                    self.checkpoint_counter = data.get("checkpoint_counter", 0)
                    logger.info(f"[CKPT] Loaded {len(self.checkpoints)} existing checkpoints")
            except Exception as e:
                logger.warning(f"[CKPT] Failed to load checkpoint metadata: {e}")
                self.checkpoints = []

    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        try:
            data = {
                "checkpoints": [asdict(cp) for cp in self.checkpoints],
                "best_accuracy": self.best_accuracy,
                "checkpoint_counter": self.checkpoint_counter,
            }
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[CKPT] Failed to save checkpoint metadata: {e}")

    def count_parameters(self, model: tf.keras.Model) -> int:
        """
        Count total trainable parameters in the model.

        Args:
            model: TensorFlow model

        Returns:
            Total number of trainable parameters
        """
        return sum(tf.size(w).numpy() for w in model.trainable_weights)

    def _is_pareto_dominated(self, accuracy: float, param_count: int) -> bool:
        """
        Check if a point is dominated by any existing checkpoint.

        A point (a1, p1) is dominated by (a2, p2) if:
        - a2 >= a1 AND p2 <= p1 (better or equal accuracy with fewer or equal params)
        - AND at least one inequality is strict

        Args:
            accuracy: Model accuracy
            param_count: Model parameter count

        Returns:
            True if dominated, False otherwise
        """
        for cp in self.checkpoints:
            if cp.accuracy >= accuracy and cp.param_count <= param_count:
                # Check if at least one inequality is strict
                if cp.accuracy > accuracy or cp.param_count < param_count:
                    return True
        return False

    def _find_dominated_checkpoints(self, accuracy: float, param_count: int) -> list[int]:
        """
        Find checkpoints that would be dominated by the new point.

        Args:
            accuracy: New model accuracy
            param_count: New model parameter count

        Returns:
            List of indices of dominated checkpoints
        """
        dominated = []
        for i, cp in enumerate(self.checkpoints):
            # Skip the best accuracy checkpoint (index 0)
            if i == 0 and cp.is_best_accuracy:
                continue

            # New point dominates if it has better or equal accuracy with fewer params
            if accuracy >= cp.accuracy and param_count <= cp.param_count:
                if accuracy > cp.accuracy or param_count < cp.param_count:
                    dominated.append(i)

        return dominated

    def should_save_checkpoint(self, accuracy: float, model: tf.keras.Model) -> tuple[bool, str]:
        """
        Determine if a checkpoint should be saved.

        Args:
            accuracy: Current model accuracy
            model: TensorFlow model

        Returns:
            (should_save, reason) tuple
        """
        param_count = self.count_parameters(model)

        # Always save if we have fewer than max checkpoints
        if len(self.checkpoints) < self.max_checkpoints:
            if accuracy > self.best_accuracy:
                return True, "best_accuracy"
            elif not self._is_pareto_dominated(accuracy, param_count):
                return True, "pareto_optimal"
            else:
                return False, "dominated"

        # Check if this is the best accuracy
        if accuracy > self.best_accuracy:
            return True, "best_accuracy"

        # Check if this point is on the Pareto frontier
        if not self._is_pareto_dominated(accuracy, param_count):
            # Check if it dominates any existing checkpoints
            dominated = self._find_dominated_checkpoints(accuracy, param_count)
            if dominated:
                return True, "pareto_dominant"
            else:
                return True, "pareto_optimal"

        return False, "dominated"

    def save_checkpoint(
        self,
        model: tf.keras.Model,
        accuracy: float,
        loss: float,
        step: int,
        epoch: int,
    ) -> str | None:
        """
        Save a checkpoint if it meets the criteria.

        Args:
            model: TensorFlow model to save
            accuracy: Model accuracy
            loss: Model loss
            step: Training step
            epoch: Training epoch

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        should_save, reason = self.should_save_checkpoint(accuracy, model)

        if not should_save:
            logger.debug(f"[CKPT] Skipping checkpoint (accuracy={accuracy:.4f}, reason={reason})")
            return None

        param_count = self.count_parameters(model)
        self.checkpoint_counter += 1

        # Create checkpoint directory
        timestamp = tf.timestamp().numpy()
        checkpoint_id = self.checkpoint_counter
        checkpoint_name = f"ckpt_{checkpoint_id:04d}_acc{accuracy:.4f}_params{param_count}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = str(checkpoint_path / "model")
        try:
            model.save(model_path, save_format="tf")
            logger.info(f"[CKPT] Saved model to {model_path}")
        except Exception as e:
            logger.error(f"[CKPT] Failed to save model: {e}")
            return None

        # Create checkpoint metrics
        is_best_accuracy = accuracy > self.best_accuracy
        if is_best_accuracy:
            self.best_accuracy = accuracy

        checkpoint = CheckpointMetrics(
            checkpoint_id=checkpoint_id,
            accuracy=accuracy,
            loss=loss,
            param_count=param_count,
            step=step,
            epoch=epoch,
            model_path=str(checkpoint_path),
            timestamp=str(timestamp),
            is_best_accuracy=is_best_accuracy,
            is_pareto_optimal=(reason in ["pareto_optimal", "pareto_dominant"]),
        )

        # Handle checkpoint replacement logic
        if reason == "best_accuracy":
            # Remove old best accuracy checkpoint if it exists
            old_best = [i for i, cp in enumerate(self.checkpoints) if cp.is_best_accuracy]
            if old_best:
                old_cp = self.checkpoints.pop(old_best[0])
                self._remove_checkpoint(old_cp)
                logger.info("[CKPT] Replaced old best accuracy checkpoint")

            # Insert at front
            self.checkpoints.insert(0, checkpoint)
            logger.info(f"[CKPT] Saved NEW BEST ACCURACY checkpoint: {accuracy:.4f}")

        elif reason in ["pareto_optimal", "pareto_dominant"]:
            if reason == "pareto_dominant":
                # Remove dominated checkpoints
                dominated = self._find_dominated_checkpoints(accuracy, param_count)
                for idx in sorted(dominated, reverse=True):
                    old_cp = self.checkpoints.pop(idx)
                    self._remove_checkpoint(old_cp)
                    logger.info(
                        f"[CKPT] Removed dominated checkpoint (acc={old_cp.accuracy:.4f}, params={old_cp.param_count})"
                    )

            # Add new checkpoint
            self.checkpoints.append(checkpoint)
            logger.info(
                f"[CKPT] Saved Pareto-optimal checkpoint: acc={accuracy:.4f}, params={param_count}"
            )

        # Ensure we don't exceed max checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            # Remove the worst Pareto point (lowest accuracy, excluding best)
            pareto_only = [cp for cp in self.checkpoints if not cp.is_best_accuracy]
            if pareto_only:
                worst = min(pareto_only, key=lambda cp: cp.accuracy)
                idx = self.checkpoints.index(worst)
                old_cp = self.checkpoints.pop(idx)
                self._remove_checkpoint(old_cp)
                logger.info("[CKPT] Removed excess checkpoint to maintain limit")

        # Update Pareto ranks
        self._update_pareto_ranks()

        # Save metadata
        self._save_metadata()

        return str(checkpoint_path)

    def _update_pareto_ranks(self):
        """Update Pareto rank for all checkpoints."""
        # Sort by accuracy descending, then by param_count ascending
        pareto_checkpoints = sorted(
            [cp for cp in self.checkpoints if not cp.is_best_accuracy],
            key=lambda cp: (-cp.accuracy, cp.param_count),
        )

        for rank, cp in enumerate(pareto_checkpoints, start=1):
            cp.pareto_rank = rank

    def _remove_checkpoint(self, checkpoint: CheckpointMetrics):
        """Remove a checkpoint from disk."""
        try:
            checkpoint_path = Path(checkpoint.model_path)
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                logger.debug(f"[CKPT] Removed checkpoint directory: {checkpoint_path}")
        except Exception as e:
            logger.error(f"[CKPT] Failed to remove checkpoint: {e}")

    def get_best_checkpoint(self) -> CheckpointMetrics | None:
        """Get the checkpoint with best accuracy."""
        best = [cp for cp in self.checkpoints if cp.is_best_accuracy]
        return best[0] if best else None

    def get_pareto_checkpoints(self) -> list[CheckpointMetrics]:
        """Get checkpoints on the Pareto frontier."""
        return [cp for cp in self.checkpoints if cp.is_pareto_optimal]

    def print_summary(self):
        """Print a summary of all checkpoints."""
        if not self.checkpoints:
            logger.info("[CKPT] No checkpoints saved yet")
            return

        logger.info("=" * 80)
        logger.info(" " * 25 + "CHECKPOINT SUMMARY" + " " * 27)
        logger.info("=" * 80)

        for i, cp in enumerate(self.checkpoints, start=1):
            status = []
            if cp.is_best_accuracy:
                status.append("BEST ACCURACY")
            if cp.is_pareto_optimal and cp.pareto_rank:
                status.append(f"Pareto Rank {cp.pareto_rank}")

            status_str = f" [{', '.join(status)}]" if status else ""

            logger.info(f"Checkpoint {i}{status_str}:")
            logger.info(f"  Accuracy: {cp.accuracy:.4f}")
            logger.info(f"  Loss: {cp.loss:.4f}")
            logger.info(f"  Parameters: {cp.param_count:,}")
            logger.info(f"  Step: {cp.step}, Epoch: {cp.epoch}")
            logger.info(f"  Path: {cp.model_path}")
            logger.info("-" * 80)

        logger.info(f"Total checkpoints: {len(self.checkpoints)}/{self.max_checkpoints}")
        logger.info("=" * 80)
