"""
HPO Bridge - Communication bridge for HPO trial reporting.

This module provides the HPOReporter class that enables trials to report
their progress and results back to the HPO orchestrator and WebUI.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class HPOReporter:
    """Reports HPO trial metrics to the orchestrator and WebUI.

    The HPOReporter writes metrics to JSON files that the HPO orchestrator
    monitors, and also streams logs to the WebUI backend API for real-time
    display in the Training Console.

    Args:
        sweep_id: HPO sweep identifier for log streaming to WebUI.
        trial_dir: Directory for trial artifacts. If not provided,
                   uses HPO_TRIAL_DIR or HPO_TRIAL_ID environment variables.
        api_host: Backend API host for log streaming (default: 127.0.0.1:8000).
    """

    def __init__(
        self,
        sweep_id: str | None = None,
        trial_dir: Path | str | None = None,
        api_host: str | None = None,
    ):
        """Initialize the HPO reporter.

        Args:
            sweep_id: HPO sweep identifier for WebUI log streaming.
            trial_dir: Directory for trial artifacts.
            api_host: Backend API host (default: from env or 127.0.0.1:8000).
        """
        self._enabled = False
        self._trial_dir: Path | None = None
        self._trial_id: str | None = None
        self._metrics_file: Path | None = None
        self._sweep_id = sweep_id or os.getenv("HPO_SWEEP_ID")
        self._api_host = api_host or os.getenv("HPO_API_HOST", "127.0.0.1:8000")
        self._log_queue: list[dict[str, Any]] = []
        self._log_lock = threading.Lock()

        # Determine trial directory from params or environment
        if trial_dir:
            self._trial_dir = Path(trial_dir)
        elif os.getenv("HPO_TRIAL_DIR"):
            self._trial_dir = Path(os.getenv("HPO_TRIAL_DIR"))  # type: ignore
        elif os.getenv("HPO_TRIAL_ID"):
            self._trial_id = os.getenv("HPO_TRIAL_ID")
            hpo_root = Path(os.getenv("HPO_ROOT", "artifacts/hpo_trials"))
            self._trial_dir = hpo_root / self._trial_id

        if self._trial_dir:
            self._trial_dir.mkdir(parents=True, exist_ok=True)
            self._metrics_file = self._trial_dir / "metrics.jsonl"
            self._enabled = True
            logger.info(f"[HPO Reporter] Enabled for trial: {self._trial_dir}")
        else:
            # Enable anyway if sweep_id is provided for API-only logging
            if self._sweep_id:
                self._enabled = True
                logger.info(f"[HPO Reporter] Enabled for sweep: {self._sweep_id}")
            else:
                logger.info("[HPO Reporter] Disabled (no trial directory or sweep_id)")

    @property
    def enabled(self) -> bool:
        """Whether the reporter is enabled."""
        return self._enabled

    def _send_to_api(self, log_entry: dict[str, Any]) -> None:
        """Send a log entry to the WebUI backend API.

        This runs in a background thread to avoid blocking training.

        Args:
            log_entry: Log data to send.
        """
        if not self._sweep_id:
            return

        def _do_send():
            try:
                url = f"http://{self._api_host}/api/hpo/sweep/{self._sweep_id}/log"
                data = json.dumps(log_entry).encode("utf-8")
                req = Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req, timeout=2) as response:
                    _ = response.read()
            except URLError as e:
                # Log silently - don't spam if API is unavailable
                logger.debug(f"[HPO Reporter] API log failed: {e}")
            except Exception as e:
                logger.debug(f"[HPO Reporter] API log error: {e}")

        # Run in background thread
        thread = threading.Thread(target=_do_send, daemon=True)
        thread.start()

    def report(
        self,
        step: int,
        loss: float,
        gradient_norm: float | None = None,
        learning_rate: float | None = None,
        epoch: int | None = None,
        memory_mb: float | None = None,
        peak_memory_mb: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Report a metric update.

        Args:
            step: Current training step
            loss: Current loss value
            gradient_norm: Optional gradient norm
            learning_rate: Optional learning rate
            epoch: Optional current epoch
            memory_mb: Current RSS memory usage in MB
            peak_memory_mb: Peak RSS memory during trial in MB
            **kwargs: Additional metrics to report
        """
        if not self._enabled:
            return

        metrics = {
            "step": step,
            "loss": loss,
        }

        if gradient_norm is not None:
            metrics["gradient_norm"] = gradient_norm
        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate
        if epoch is not None:
            metrics["epoch"] = epoch
        if memory_mb is not None:
            metrics["memory_mb"] = memory_mb
        if peak_memory_mb is not None:
            metrics["peak_memory_mb"] = peak_memory_mb

        metrics.update(kwargs)

        # Append to JSONL file
        if self._metrics_file:
            with self._metrics_file.open("a") as f:
                f.write(json.dumps(metrics) + "\n")

        # Send to WebUI API for real-time display
        log_entry = {
            "level": "INFO",
            "message": f"Step {step}: loss={loss:.6f}",
            "step": step,
            "loss": loss,
            "gradient_norm": gradient_norm,
            "learning_rate": learning_rate,
            "epoch": epoch,
            "memory_mb": memory_mb,
            "trial_id": self._trial_id,
        }
        self._send_to_api(log_entry)

        # Log progress locally
        if step % 50 == 0:
            mem_str = f", mem={memory_mb:.0f}MB" if memory_mb else ""
            logger.info(f"[HPO Reporter] Step {step}: loss={loss:.6f}{mem_str}")

    def log(
        self,
        message: str,
        level: str = "INFO",
        **kwargs: Any,
    ) -> None:
        """Log a message to the WebUI console.

        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            **kwargs: Additional data to include
        """
        if not self._enabled:
            return

        log_entry = {
            "level": level,
            "message": message,
            "trial_id": self._trial_id,
            **kwargs,
        }
        self._send_to_api(log_entry)

        # Also log locally
        if level == "ERROR":
            logger.error(f"[HPO] {message}")
        elif level == "WARNING":
            logger.warning(f"[HPO] {message}")
        else:
            logger.info(f"[HPO] {message}")

    def complete(
        self,
        success: bool,
        final_loss: float | None = None,
        error: str | None = None,
        param_count: int | None = None,
        efficiency_score: float | None = None,
    ) -> None:
        """Report trial completion.

        Args:
            success: Whether the trial completed successfully
            final_loss: Final loss achieved
            error: Error message if failed
            param_count: Estimated parameter count of the model
            efficiency_score: Efficiency-penalized score (loss × (1 + λ × norm_params))
        """
        if not self._enabled:
            return

        # Write status file if we have a trial directory
        if self._trial_dir:
            status = {
                "status": "completed" if success else "failed",
                "best_loss": final_loss,
                "error": error,
                "param_count": param_count,
                "efficiency_score": efficiency_score,
            }

            status_file = self._trial_dir / "status.json"
            with status_file.open("w") as f:
                json.dump(status, f, indent=2)

        # Send completion message to WebUI
        if success:
            eff_str = f", efficiency={efficiency_score:.6f}" if efficiency_score else ""
            params_str = f" ({param_count / 1e6:.1f}M params)" if param_count else ""
            self.log(
                (
                    f"Trial completed with loss={final_loss:.6f}{eff_str}{params_str}"
                    if final_loss
                    else "Trial completed"
                ),
                level="INFO",
                loss=final_loss,
                efficiency_score=efficiency_score,
                param_count=param_count,
            )
            logger.info(
                f"[HPO Reporter] Trial completed with loss={final_loss}, efficiency={efficiency_score}"
            )
        else:
            self.log(f"Trial failed: {error}", level="ERROR")
            logger.error(f"[HPO Reporter] Trial failed: {error}")
