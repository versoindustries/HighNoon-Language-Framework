"""
HPO Bridge - Communication bridge for HPO trial reporting.

This module provides the HPOReporter class that enables trials to report
their progress and results back to the HPO orchestrator and WebUI.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# API communication settings (more robust than silent 2-second timeout)
API_TIMEOUT_SECONDS = 5  # Increased from 2 for reliability
API_MAX_RETRIES = 3  # Retry failed API calls
API_RETRY_DELAY_SECONDS = 0.5  # Delay between retries
API_FAILURE_WARNING_THRESHOLD = 5  # Warn after this many consecutive failures


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

        # API failure tracking for diagnostics
        self._api_failure_count = 0
        self._api_success_count = 0
        self._last_api_warning_time = 0.0

        # Determine trial directory from params or environment
        if trial_dir:
            self._trial_dir = Path(trial_dir)
        elif os.getenv("HPO_TRIAL_DIR"):
            self._trial_dir = Path(os.getenv("HPO_TRIAL_DIR"))  # type: ignore
        elif os.getenv("HPO_TRIAL_ID"):
            self._trial_id = os.getenv("HPO_TRIAL_ID")
            hpo_root = Path(os.getenv("HPO_ROOT", "artifacts/hpo_trials"))
            self._trial_dir = hpo_root / self._trial_id

        # Step tracking for should_report (instance-level)
        self._report_step_counter = 0
        self._report_interval = 10  # Report every N steps

        if self._trial_dir:
            self._trial_dir.mkdir(parents=True, exist_ok=True)
            self._metrics_file = self._trial_dir / "metrics.jsonl"
            self._enabled = True
            # Extract trial_id from directory name if not explicitly set
            # This ensures trial logs show correct IDs like "trial_0" instead of "None"
            if self._trial_id is None:
                self._trial_id = self._trial_dir.name
            logger.info(
                f"[HPO Reporter] Enabled for trial: {self._trial_dir} (id={self._trial_id})"
            )
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

    def should_report(self) -> bool:
        """Check if metrics should be reported this step.

        Returns True every `_report_interval` steps to avoid flooding
        the HPO orchestrator with too frequent updates.

        Returns:
            True if metrics should be reported this step.
        """
        self._report_step_counter += 1
        return self._report_step_counter % self._report_interval == 0

    def report_completion(
        self,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """Report trial completion (wrapper for complete() method).

        This method provides backward compatibility with training loops
        that use the report_completion() signature.

        Args:
            success: Whether the trial completed successfully.
            error_message: Error message if failed.
        """
        self.complete(success=success, error=error_message)

    def _sanitize_float(self, value: float | None, name: str = "metric") -> float | None:
        """Sanitize float values, converting NaN/Inf to None and logging warnings.

        NaN or Inf values in training metrics indicate numerical issues that
        should be investigated (e.g., gradient explosion, division by zero).
        This method logs a warning and returns None to prevent JSON serialization
        errors while flagging the underlying training problem.

        Args:
            value: Float value to sanitize
            name: Name of the metric for logging

        Returns:
            The original value if valid, or None if NaN/Inf
        """
        if value is None:
            return None
        if not isinstance(value, (int, float)):
            return value
        if math.isnan(value) or math.isinf(value):
            logger.warning(
                f"[HPO Reporter] {name}={value} is not finite - "
                "possible gradient explosion or numerical instability"
            )
            return None
        return float(value)

    def _send_to_api(self, log_entry: dict[str, Any]) -> None:
        """Send a log entry to the WebUI backend API with retry logic.

        This runs in a background thread to avoid blocking training.
        Uses exponential backoff retry for reliability.

        Args:
            log_entry: Log data to send.
        """
        if not self._sweep_id:
            return

        def _do_send():
            url = f"http://{self._api_host}/api/hpo/sweep/{self._sweep_id}/log"
            data = json.dumps(log_entry).encode("utf-8")
            last_error = None

            for attempt in range(API_MAX_RETRIES):
                try:
                    req = Request(
                        url,
                        data=data,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(req, timeout=API_TIMEOUT_SECONDS) as response:
                        _ = response.read()

                    # Success - reset failure count and track
                    self._api_success_count += 1
                    if self._api_failure_count > 0:
                        logger.info(
                            f"[HPO Reporter] API connection restored after "
                            f"{self._api_failure_count} failures"
                        )
                    self._api_failure_count = 0
                    return

                except URLError as e:
                    last_error = e
                    if attempt < API_MAX_RETRIES - 1:
                        time.sleep(API_RETRY_DELAY_SECONDS * (2**attempt))
                except Exception as e:
                    last_error = e
                    if attempt < API_MAX_RETRIES - 1:
                        time.sleep(API_RETRY_DELAY_SECONDS * (2**attempt))

            # All retries failed
            self._api_failure_count += 1
            current_time = time.time()

            # Warn user if API has been failing (but don't spam)
            if (
                self._api_failure_count >= API_FAILURE_WARNING_THRESHOLD
                and current_time - self._last_api_warning_time > 60  # Max 1 warning per minute
            ):
                self._last_api_warning_time = current_time
                logger.warning(
                    f"[HPO Reporter] WebUI API unreachable ({self._api_failure_count} consecutive failures). "
                    f"Metrics ARE being saved to disk at {self._trial_dir or 'N/A'}. "
                    f"Last error: {last_error}"
                )

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
            **kwargs: Additional metrics to report (batch_size, tokens_per_sec, etc.)
        """
        if not self._enabled:
            return

        # Sanitize all float values to prevent NaN/Inf from causing JSON errors
        # This also logs warnings to flag numerical instability in training
        loss = self._sanitize_float(loss, "loss")
        gradient_norm = self._sanitize_float(gradient_norm, "gradient_norm")
        learning_rate = self._sanitize_float(learning_rate, "learning_rate")
        memory_mb = self._sanitize_float(memory_mb, "memory_mb")
        peak_memory_mb = self._sanitize_float(peak_memory_mb, "peak_memory_mb")

        # Diagnostic: warn if loss is suspiciously large (suggests aggregation issue)
        # For cross-entropy with typical vocab sizes, mean loss should be < 20
        if loss is not None and loss > 100:
            logger.warning(
                f"[HPO Reporter] Suspiciously large loss={loss:.2f} at step {step}. "
                "Expected cross-entropy should be < 20 for most vocabularies. "
                "This may indicate loss is being summed instead of averaged."
            )

        # Compute real-time perplexity estimate from loss
        # PPL = exp(loss) for cross-entropy loss
        perplexity = None
        if loss is not None and loss < 100:  # Only compute for reasonable loss values
            perplexity = math.exp(min(loss, 20))  # Cap to avoid overflow

        # Sanitize any additional kwargs that might contain floats
        sanitized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, float):
                sanitized_kwargs[key] = self._sanitize_float(value, key)
            else:
                sanitized_kwargs[key] = value

        metrics: dict[str, Any] = {
            "step": step,
        }

        # Only include loss if it's valid (not NaN/Inf)
        if loss is not None:
            metrics["loss"] = loss
        if perplexity is not None:
            metrics["perplexity"] = perplexity
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

        metrics.update(sanitized_kwargs)

        # Append to JSONL file
        if self._metrics_file:
            with self._metrics_file.open("a") as f:
                f.write(json.dumps(metrics) + "\n")

        # Build comprehensive log message with all available metrics
        parts = []
        if loss is not None:
            parts.append(f"loss={loss:.6f}")
        if perplexity is not None:
            parts.append(f"ppl={perplexity:.2f}")
        if gradient_norm is not None:
            parts.append(f"grad={gradient_norm:.4f}")
        if learning_rate is not None:
            parts.append(f"lr={learning_rate:.2e}")
        if memory_mb is not None:
            parts.append(f"mem={memory_mb:.0f}MB")
        message = " | ".join(parts) if parts else f"step={step}"

        # Send to WebUI API for real-time display
        log_entry = {
            "level": "WARNING" if loss is None or (loss is not None and loss > 100) else "INFO",
            "message": message,
            "step": step,
            "loss": loss,
            "perplexity": perplexity,
            "gradient_norm": gradient_norm,
            "learning_rate": learning_rate,
            "epoch": epoch,
            "memory_mb": memory_mb,
            "peak_memory_mb": peak_memory_mb,
            "trial_id": self._trial_id,
            # Pass through additional metrics for WebUI display
            **{
                k: v
                for k, v in sanitized_kwargs.items()
                if k
                in [
                    "batch_size",
                    "tokens_per_sec",
                    "optimizer",
                    "phase",
                    "barren_plateau_detected",
                    "lr_scaled",
                    "crystallization_active",
                ]
            },
        }
        self._send_to_api(log_entry)

        # Log progress locally with rich formatting
        if step % 50 == 0:
            epoch_str = f" [Epoch {epoch}]" if epoch is not None else ""
            logger.info(f"[HPO Reporter]{epoch_str} Step {step}: {message}")

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

    def log_trial_start(
        self,
        config: dict[str, Any],
        param_count: int | None = None,
    ) -> None:
        """Log trial configuration at the start for WebUI visibility.

        This sends a comprehensive summary of the trial configuration
        to the WebUI training console, making it easy to see what
        architecture and hyperparameters are being tested.

        Args:
            config: Trial configuration dictionary
            param_count: Estimated parameter count
        """
        if not self._enabled:
            return

        # Extract key configuration values
        hidden_dim = config.get("hidden_dim") or config.get("embedding_dim", "N/A")
        num_blocks = config.get("num_reasoning_blocks", "N/A")
        num_experts = config.get("num_moe_experts", "N/A")
        batch_size = config.get("batch_size", "N/A")
        learning_rate = config.get("learning_rate", "N/A")
        optimizer = config.get("optimizer", "N/A")
        vocab_size = config.get("vocab_size", "N/A")
        sequence_length = config.get("sequence_length", "N/A")

        # Format learning rate
        lr_str = f"{learning_rate:.2e}" if isinstance(learning_rate, float) else str(learning_rate)

        # Format parameter count
        param_str = f"{param_count / 1e6:.1f}M" if param_count else "N/A"

        # Build configuration summary message
        arch_msg = f"Architecture: dim={hidden_dim}, blocks={num_blocks}, experts={num_experts}"
        train_msg = f"Training: lr={lr_str}, batch={batch_size}, optimizer={optimizer}"
        data_msg = f"Data: vocab={vocab_size}, seq_len={sequence_length}"
        param_msg = f"Parameters: ~{param_str}"

        # Log each section
        self.log(f"=== Trial {self._trial_id} Started ===", level="INFO")
        self.log(
            arch_msg,
            level="INFO",
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_experts=num_experts,
        )
        self.log(
            train_msg,
            level="INFO",
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer=optimizer,
        )
        self.log(data_msg, level="INFO", vocab_size=vocab_size, sequence_length=sequence_length)
        self.log(param_msg, level="INFO", param_count=param_count)

        # Also log to local logger
        logger.info(f"[HPO Trial] {arch_msg}")
        logger.info(f"[HPO Trial] {train_msg}")
        logger.info(f"[HPO Trial] {data_msg}")
        logger.info(f"[HPO Trial] {param_msg}")

    def complete(
        self,
        success: bool,
        final_loss: float | None = None,
        error: str | None = None,
        param_count: int | None = None,
        efficiency_score: float | None = None,
        perplexity: float | None = None,
        mean_confidence: float | None = None,
        expected_calibration_error: float | None = None,
        composite_score: float | None = None,
        memory_peak_mb: float | None = None,
        epochs_completed: int | None = None,
        throughput_tokens_per_sec: float | None = None,
    ) -> None:
        """Report trial completion.

        Args:
            success: Whether the trial completed successfully
            final_loss: Final loss achieved
            error: Error message if failed
            param_count: Estimated parameter count of the model
            efficiency_score: Efficiency-penalized score (loss × (1 + λ × norm_params))
            perplexity: Validation perplexity (model quality metric)
            mean_confidence: Mean prediction confidence (entropy-based)
            expected_calibration_error: ECE (calibration quality, lower is better)
            composite_score: Multi-objective composite score for ranking
            memory_peak_mb: Peak memory usage during training (for multi-objective optimization)
            epochs_completed: Number of epochs completed before stopping
            throughput_tokens_per_sec: Generation throughput in tokens/second
        """
        if not self._enabled:
            return

        # Sanitize all float metrics to prevent NaN/Inf from causing JSON errors
        final_loss = self._sanitize_float(final_loss, "final_loss")
        efficiency_score = self._sanitize_float(efficiency_score, "efficiency_score")
        perplexity = self._sanitize_float(perplexity, "perplexity")
        mean_confidence = self._sanitize_float(mean_confidence, "mean_confidence")
        expected_calibration_error = self._sanitize_float(
            expected_calibration_error, "expected_calibration_error"
        )
        composite_score = self._sanitize_float(composite_score, "composite_score")
        memory_peak_mb = self._sanitize_float(memory_peak_mb, "memory_peak_mb")
        throughput_tokens_per_sec = self._sanitize_float(
            throughput_tokens_per_sec, "throughput_tokens_per_sec"
        )

        # Write status file if we have a trial directory
        if self._trial_dir:
            status = {
                "status": "completed" if success else "failed",
                "best_loss": final_loss,
                "error": error,
                "param_count": param_count,
                "efficiency_score": efficiency_score,
                # Multi-objective quality metrics
                "perplexity": perplexity,
                "mean_confidence": mean_confidence,
                "expected_calibration_error": expected_calibration_error,
                "composite_score": composite_score,
                # Memory and progress tracking
                "memory_peak_mb": memory_peak_mb,
                "epochs_completed": epochs_completed,
                "throughput_tokens_per_sec": throughput_tokens_per_sec,
            }

            status_file = self._trial_dir / "status.json"
            with status_file.open("w") as f:
                json.dump(status, f, indent=2)

        # Send completion message to WebUI
        if success:
            eff_str = f", efficiency={efficiency_score:.6f}" if efficiency_score else ""
            comp_str = f", composite={composite_score:.6f}" if composite_score else ""
            ppl_str = f", ppl={perplexity:.2f}" if perplexity else ""
            params_str = f" ({param_count / 1e6:.1f}M params)" if param_count else ""
            loss_str = f"loss={final_loss:.6f}" if final_loss else "loss=NaN"
            self.log(
                (
                    f"Trial completed with {loss_str}{eff_str}{comp_str}{ppl_str}{params_str}"
                    if final_loss
                    else "Trial completed"
                ),
                level="INFO",
                loss=final_loss,
                efficiency_score=efficiency_score,
                composite_score=composite_score,
                perplexity=perplexity,
                param_count=param_count,
            )
            logger.info(
                f"[HPO Reporter] Trial completed: loss={final_loss}, "
                f"composite={composite_score}, ppl={perplexity}"
            )
        else:
            self.log(f"Trial failed: {error}", level="ERROR")
            logger.error(f"[HPO Reporter] Trial failed: {error}")
