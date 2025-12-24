"""HPO Bridge for integrating HPO metrics reporting into the training loop.

This module provides utilities for detecting HPO mode, loading trial configs,
and reporting metrics back to the HPO orchestrator during training.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import psutil

from highnoon.services.hpo_metrics import HPOMetricsCollector, TrialMetrics, TrialStatus

logger = logging.getLogger(__name__)


def is_hpo_mode() -> bool:
    """Check if the current process is running as an HPO trial.

    Returns:
        True if HPO_TRIAL_ID environment variable is set
    """
    return "HPO_TRIAL_ID" in os.environ


def get_trial_id() -> str | None:
    """Get the current trial ID if running in HPO mode.

    Returns:
        Trial ID or None if not in HPO mode
    """
    return os.environ.get("HPO_TRIAL_ID")


def get_trial_dir() -> Path | None:
    """Get the trial directory if running in HPO mode.

    Returns:
        Path to trial directory or None if not in HPO mode
    """
    trial_dir = os.environ.get("HPO_TRIAL_DIR")
    if trial_dir:
        return Path(trial_dir)
    return None


def load_trial_config() -> dict[str, Any] | None:
    """Load the hyperparameter configuration for the current trial.

    Returns:
        Dictionary of hyperparameters or None if not in HPO mode
    """
    if not is_hpo_mode():
        return None

    trial_dir = get_trial_dir()
    if not trial_dir:
        logger.warning("[HPO Bridge] HPO mode enabled but trial directory not set")
        return None

    config_file = trial_dir / "config.json"
    if not config_file.exists():
        logger.warning(f"[HPO Bridge] Trial config not found: {config_file}")
        return None

    try:
        with config_file.open("r", encoding="utf-8") as f:
            config = json.load(f)
        logger.info(f"[HPO Bridge] Loaded trial config: {config}")
        return config
    except Exception as exc:
        logger.error(f"[HPO Bridge] Failed to load trial config: {exc}")
        return None


class HPOReporter:
    """Reports metrics from the training loop to the HPO orchestrator."""

    def __init__(self):
        """Initialize the HPO reporter."""
        self.trial_id = get_trial_id()
        self.enabled = is_hpo_mode()

        if not self.enabled:
            logger.info("[HPO Reporter] HPO mode not enabled, metrics will not be reported")
            return

        self.collector = HPOMetricsCollector()
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.process = psutil.Process()

        # Load trial config to get hyperparameters
        self.config = load_trial_config()

        # Initialize trial status
        self._init_trial_status()

        logger.info(f"[HPO Reporter] Initialized for trial {self.trial_id}")

    def _init_trial_status(self) -> None:
        """Initialize the trial status file."""
        if not self.enabled or not self.trial_id:
            return

        status = TrialStatus(
            trial_id=self.trial_id,
            status="running",
            start_time=self.start_time,
            hyperparameters=self.config or {},
        )
        self.collector.update_trial_status(status)

    def report(
        self,
        step: int,
        loss: float | None = None,
        gradient_norm: float | None = None,
        learning_rate: float | None = None,
        **extras: Any,
    ) -> None:
        """Report metrics for the current training step.

        Args:
            step: Current training step
            loss: Training loss
            gradient_norm: Gradient norm
            learning_rate: Current learning rate
            **extras: Additional metrics to report
        """
        if not self.enabled or not self.trial_id:
            return

        current_time = time.time()
        wall_time = current_time - self.start_time

        # Get memory usage
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
        except Exception:
            memory_mb = None

        # Create metrics object
        metrics = TrialMetrics(
            trial_id=self.trial_id,
            step=step,
            timestamp=current_time,
            loss=loss,
            gradient_norm=gradient_norm,
            learning_rate=learning_rate,
            memory_mb=memory_mb,
            wall_time_seconds=wall_time,
            extras=extras,
        )

        # Write to disk
        self.collector.report_metrics(metrics)

        # Update trial status with best loss
        status = self.collector.load_trial_status(self.trial_id)
        if status:
            if loss is not None:
                if status.best_loss is None or loss < status.best_loss:
                    status.best_loss = loss
                    status.best_step = step
            status.total_steps = step
            self.collector.update_trial_status(status)

        self.last_report_time = current_time

    def report_completion(self, success: bool = True, error_message: str | None = None) -> None:
        """Report that the trial has completed.

        Args:
            success: Whether the trial completed successfully
            error_message: Error message if failed
        """
        if not self.enabled or not self.trial_id:
            return

        status = self.collector.load_trial_status(self.trial_id)
        if status:
            status.status = "completed" if success else "failed"
            status.end_time = time.time()
            if error_message:
                status.error_message = error_message
            self.collector.update_trial_status(status)

        logger.info(
            f"[HPO Reporter] Trial {self.trial_id} {'completed' if success else 'failed'}"
            + (f": {error_message}" if error_message else "")
        )

    def should_report(self, report_frequency: int = 10) -> bool:
        """Check if metrics should be reported at this step.

        Args:
            report_frequency: Report every N steps

        Returns:
            True if metrics should be reported
        """
        # In HPO mode, report more frequently for real-time monitoring
        return self.enabled


def apply_trial_config(config: dict[str, Any] | None, model_config: dict[str, Any]) -> None:
    """Apply trial hyperparameters to the model configuration.

    Args:
        config: Trial hyperparameter configuration
        model_config: Model configuration dictionary to update
    """
    if not config:
        return

    # Map HPO config to model config
    if "learning_rate" in config:
        model_config["learning_rate"] = config["learning_rate"]

    if "batch_size" in config:
        model_config["batch_size"] = config["batch_size"]

    if "optimizer" in config:
        model_config["optimizer_name"] = config["optimizer"]

    if "warmup_steps" in config:
        model_config["warmup_steps"] = config["warmup_steps"]

    if "weight_decay" in config:
        model_config["weight_decay"] = config["weight_decay"]

    if "num_reasoning_blocks" in config:
        model_config["num_reasoning_blocks"] = config["num_reasoning_blocks"]

    if "hidden_dim" in config:
        model_config["hidden_dim"] = config["hidden_dim"]

    logger.info(f"[HPO Bridge] Applied trial config to model: {config}")
