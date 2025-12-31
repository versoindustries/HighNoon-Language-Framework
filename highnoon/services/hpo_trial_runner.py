"""
HPO Trial Runner - Trains real HSMN models for hyperparameter optimization.

This module serves as the bridge between the C++ HPO orchestrator and Python model training.
It builds and trains full HSMN models with specified hyperparameters.

Includes RSS memory tracking via psutil for trial resource monitoring.
"""

import argparse
import faulthandler
import gc
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

# =============================================================================
# CRITICAL: Enable faulthandler BEFORE importing TensorFlow
# =============================================================================
# This enables Python to dump a stack trace on SIGSEGV, SIGFPE, SIGABRT, etc.
# Without this, native code crashes (C++ ops, CUDA, XLA) produce no Python trace.
faulthandler.enable(file=sys.stderr, all_threads=True)
print("[HPO] Faulthandler enabled: SIGSEGV will dump Python stack trace", file=sys.stderr)

import tensorflow as tf

# =============================================================================
# FORCE CPU-ONLY MODE FOR HPO TRIALS
# =============================================================================
# HighNoon C++ ops are CPU-optimized with AVX2/AVX512 SIMD. GPU acceleration
# provides no benefit for our custom ops and causes SIGSEGV crashes when:
# 1. CUDA FFT tries to allocate 6GB+ for HD bundles (hd_dim=1536)
# 2. XLA auto-dispatches to GPU for FFT ops even when ops are CPU-native
# 3. Memory pressure causes CUDA driver to crash
#
# CRITICAL: This MUST run BEFORE any TensorFlow operations to ensure no
# GPU kernels are ever loaded or initialized.

GPU_AVAILABLE = False  # Always False for HPO trials - CPU-only design
print("[HPO] CPU-ONLY MODE: HighNoon C++ ops are CPU-optimized (AVX2/AVX512)", file=sys.stderr)
print("[HPO] Disabling GPU to prevent CUDA FFT SIGSEGV crashes", file=sys.stderr)

try:
    # Hide ALL GPUs from TensorFlow BEFORE any other TF operations
    tf.config.set_visible_devices([], "GPU")

    # Verify GPUs are hidden
    _visible_gpus = tf.config.get_visible_devices("GPU")
    if _visible_gpus:
        print(f"[HPO] WARNING: Failed to hide GPUs: {_visible_gpus}", file=sys.stderr)
    else:
        print("[HPO] GPU devices hidden successfully", file=sys.stderr)
except Exception as e:
    print(f"[HPO] GPU hiding failed (non-fatal): {e}", file=sys.stderr)

# Force XLA to use CPU-only compiler to prevent CUDA kernel generation
os.environ["TF_XLA_FLAGS"] = os.environ.get("TF_XLA_FLAGS", "") + " --tf_xla_cpu_global_jit"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Belt-and-suspenders: hide at driver level

from highnoon.data.loaders import load_training_dataset
from highnoon.models.reasoning.reasoning_module import ReasoningModule
from highnoon.services.hpo_manager import estimate_model_params
from highnoon.services.hpo_metrics import (
    DEFAULT_ALPHA_LOSS,
    DEFAULT_BETA_PERPLEXITY,
    DEFAULT_GAMMA_CALIBRATION,
    DEFAULT_LAMBDA_EFFICIENCY,
    compute_composite_score,
    compute_efficiency_score,
)
from highnoon.training.hpo_bridge import HPOReporter

# Quantum control callback (optional - may not be available if native ops missing)
META_CONTROLLER_AVAILABLE = False
HamiltonianMetaControllerCallback = None
EvolutionTimeControlBridge = None
try:
    from highnoon.training.callbacks import HamiltonianMetaControllerCallback
    from highnoon.training.control_bridge import (  # noqa: F401 - availability check
        EvolutionTimeControlBridge,
    )

    META_CONTROLLER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass

# Import config module for global feature flags
# Note: Using `hn_config` to avoid conflict with local `config` dict parameter
import highnoon.config as hn_config  # noqa: E402

# Quantum control and QSG imports
from highnoon.config import (  # noqa: E402  # Quantum training features; Model architecture limits; Quantum tokenization pipeline (Phase 48+)
    META_CONTROLLER_FREQUENCY,
    USE_HYBRID_PID,
    USE_INTELLIGENT_VOCAB_CONTROLLER,
    USE_QSG_GENERATION,
    USE_RLS_SYSID,
    VOCAB_CONTROLLER_AUTO_TRAIN,
)

# QULS - Quantum Unified Loss System (Phase 132)
QULS_AVAILABLE = False
QuantumUnifiedLoss = None
QULSConfig = None
create_quls_from_hpo_config = None
try:
    from highnoon.training.quantum_loss import (  # noqa: F401 - availability check
        QuantumUnifiedLoss,
        QULSConfig,
        create_quls_from_hpo_config,
    )

    QULS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass

# TrainingEngine - Enterprise training with GaLore, QNG, Barren Plateau Monitor
TRAINING_ENGINE_AVAILABLE = False
TrainingEngine = None
EnterpriseTrainingConfig = None
TrainingResult = None
try:
    from highnoon.training.training_engine import (
        BaseCallback,
        EnterpriseTrainingConfig,
        StepResult,
        TrainingEngine,
        TrainingResult,
    )

    TRAINING_ENGINE_AVAILABLE = True
    print("[HPO] TrainingEngine loaded successfully", file=sys.stderr)
except (ImportError, ModuleNotFoundError) as e:
    # Log the error so we can diagnose import failures
    import sys

    print(f"[HPO] WARNING: TrainingEngine import failed: {e}", file=sys.stderr)
    TRAINING_ENGINE_AVAILABLE = False


# QSG evaluation (lazy import to avoid circular dependencies)
def get_qsg_generator():
    """Lazy import QSG generator."""
    try:
        from highnoon.inference.qsg_generator import QSGGenerator

        return QSGGenerator
    except ImportError:
        return None


# Memory tracking (optional but recommended)
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EnterpriseMemoryManager:
    """Enterprise-grade memory management for HPO trials.

    Features:
    - System memory detection (total, available, cached)
    - Swap usage monitoring
    - Memory trend analysis (rising/falling)
    - Intelligent early stopping with grace period
    - Linux buffer/cache accounting

    Memory thresholds are dynamically calculated based on system resources:
    - WARNING_THRESHOLD: 95% of total memory consumed
    - CRITICAL_THRESHOLD: Swap usage detected or memory sustained above warning

    Args:
        warning_threshold_pct: Memory usage percentage to trigger warning (default 95%)
        grace_steps: Number of steps to allow high memory before early stop (default 10)
        trend_window: Number of samples for trend analysis (default 5)
    """

    def __init__(
        self,
        warning_threshold_pct: float = 0.90,
        grace_steps: int = 10,
        trend_window: int = 5,
    ):
        """Initialize enterprise memory manager."""
        self.warning_threshold_pct = warning_threshold_pct
        self.grace_steps = grace_steps
        self.trend_window = trend_window

        # Memory tracking state
        self.peak_memory_mb: float = 0.0
        self._process = None
        self._memory_history: list[float] = []
        self._warning_count: int = 0
        self._last_swap_mb: float = 0.0

        # System memory info (detected once at init)
        self.total_memory_mb: float = 0.0
        self.warning_threshold_mb: float = 60_000.0  # Default fallback

        if PSUTIL_AVAILABLE:
            self._process = psutil.Process()
            self._detect_system_memory()
            logger.info(
                f"[Memory] EnterpriseMemoryManager: total={self.total_memory_mb:.0f}MB, "
                f"warning_threshold={self.warning_threshold_mb:.0f}MB ({warning_threshold_pct*100:.0f}%)"
            )
        else:
            logger.warning("[Memory] psutil not available, using fallback thresholds")

    def _detect_system_memory(self) -> None:
        """Detect system memory and set dynamic thresholds."""
        try:
            mem = psutil.virtual_memory()
            self.total_memory_mb = mem.total / (1024 * 1024)

            # Calculate warning threshold as percentage of total
            # This accounts for the 95% warning point
            self.warning_threshold_mb = self.total_memory_mb * self.warning_threshold_pct

            # Log system memory details
            available_mb = mem.available / (1024 * 1024)
            cached_mb = getattr(mem, "cached", 0) / (1024 * 1024)
            buffers_mb = getattr(mem, "buffers", 0) / (1024 * 1024)

            logger.debug(
                f"[Memory] System: total={self.total_memory_mb:.0f}MB, "
                f"available={available_mb:.0f}MB, cached={cached_mb:.0f}MB, "
                f"buffers={buffers_mb:.0f}MB"
            )
        except Exception as e:
            logger.warning(f"[Memory] Failed to detect system memory: {e}")

    def get_current_memory_mb(self) -> float:
        """Get current RSS memory usage of the process in MB.

        Returns:
            Current memory usage in megabytes
        """
        if self._process:
            try:
                mem_info = self._process.memory_info()
                return mem_info.rss / (1024 * 1024)
            except Exception:
                pass

        return 0.0

    def get_system_memory_pct(self) -> float:
        """Get system-wide memory usage percentage (accounting for Linux cache).

        On Linux, 'available' memory includes reclaimable cache/buffers,
        which is a better indicator of actual pressure than 'used'.

        Returns:
            Memory usage as percentage (0.0 to 1.0)
        """
        if not PSUTIL_AVAILABLE:
            return 0.0

        try:
            mem = psutil.virtual_memory()
            # Use percent from psutil which accounts for available (cache-aware)
            return mem.percent / 100.0
        except Exception:
            return 0.0

    def get_swap_usage_mb(self) -> float:
        """Get current swap usage in MB.

        Returns:
            Swap usage in megabytes
        """
        if not PSUTIL_AVAILABLE:
            return 0.0

        try:
            swap = psutil.swap_memory()
            return swap.used / (1024 * 1024)
        except Exception:
            return 0.0

    def update_peak(self) -> float:
        """Update and return peak memory usage.

        Returns:
            Peak memory usage in MB
        """
        current = self.get_current_memory_mb()
        if current > self.peak_memory_mb:
            self.peak_memory_mb = current
        return self.peak_memory_mb

    def _update_history(self, current_mb: float) -> None:
        """Update memory history for trend analysis."""
        self._memory_history.append(current_mb)
        if len(self._memory_history) > self.trend_window:
            self._memory_history.pop(0)

    def get_memory_trend(self) -> str:
        """Analyze memory trend from recent history.

        Returns:
            "rising", "falling", or "stable"
        """
        if len(self._memory_history) < 2:
            return "stable"

        # Linear regression on recent history
        n = len(self._memory_history)
        x_mean = (n - 1) / 2
        y_mean = sum(self._memory_history) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(self._memory_history))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Threshold for significant trend (MB per step)
        if slope > 50:  # Rising more than 50MB per step
            return "rising"
        elif slope < -50:  # Falling more than 50MB per step
            return "falling"
        return "stable"

    def check_memory_critical(self) -> tuple[bool, str]:
        """Check if memory situation is critical and trial should stop.

        This implements intelligent early stopping:
        1. Triggers warning at 95% memory usage
        2. Tracks warning count (grace period)
        3. Detects swap usage spikes
        4. Considers memory trend (don't stop if falling)

        Returns:
            Tuple of (should_stop, reason). should_stop is True if trial
            should be terminated immediately.
        """
        if not PSUTIL_AVAILABLE:
            return False, ""

        current_mb = self.get_current_memory_mb()
        self._update_history(current_mb)
        self.update_peak()

        system_pct = self.get_system_memory_pct()
        swap_mb = self.get_swap_usage_mb()
        swap_delta = swap_mb - self._last_swap_mb
        self._last_swap_mb = swap_mb

        trend = self.get_memory_trend()

        # Check swap spike (significant new swap usage indicates OOM pressure)
        if swap_delta > 500:  # More than 500MB new swap
            logger.error(
                f"[Memory] CRITICAL: Swap spike detected! "
                f"Delta={swap_delta:.0f}MB, Total Swap={swap_mb:.0f}MB"
            )
            return True, f"swap_spike:{swap_delta:.0f}MB"

        # Check system memory percentage
        if system_pct >= self.warning_threshold_pct:
            self._warning_count += 1
            logger.warning(
                f"[Memory] WARNING ({self._warning_count}/{self.grace_steps}): "
                f"System memory at {system_pct*100:.1f}% (process={current_mb:.0f}MB), "
                f"trend={trend}"
            )

            # Early stop if sustained high usage AND memory is rising or stable
            if self._warning_count >= self.grace_steps:
                if trend == "rising":
                    return True, f"sustained_high_memory_rising:{system_pct*100:.1f}%"
                elif trend == "stable":
                    return True, f"sustained_high_memory_stable:{system_pct*100:.1f}%"
                else:
                    # Memory is falling, give it more grace
                    logger.info("[Memory] High memory but trend falling, extending grace period")
                    self._warning_count = self.grace_steps // 2
        else:
            # Reset warning count when memory drops
            if self._warning_count > 0:
                logger.info(
                    f"[Memory] Memory normalized: {system_pct*100:.1f}%, "
                    f"resetting warning count"
                )
            self._warning_count = 0

        return False, ""

    def check_oom_risk(self, threshold_mb: float | None = None) -> bool:
        """Check if memory usage exceeds threshold (backward-compatible API).

        Args:
            threshold_mb: Legacy threshold parameter (ignored, uses dynamic threshold)

        Returns:
            True if memory exceeds warning threshold
        """
        system_pct = self.get_system_memory_pct()
        return system_pct >= self.warning_threshold_pct

    def get_stats(self) -> dict[str, float]:
        """Get comprehensive memory statistics.

        Returns:
            Dictionary with memory stats including system-wide info
        """
        current = self.get_current_memory_mb()
        self.update_peak()

        stats = {
            "current_mb": current,
            "peak_mb": self.peak_memory_mb,
            "warning_count": float(self._warning_count),
        }

        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                swap = psutil.swap_memory()
                stats.update(
                    {
                        "system_total_mb": mem.total / (1024 * 1024),
                        "system_available_mb": mem.available / (1024 * 1024),
                        "system_percent": mem.percent,
                        "swap_used_mb": swap.used / (1024 * 1024),
                        "swap_percent": swap.percent,
                    }
                )
            except Exception:
                pass

        return stats

    def cleanup(self) -> None:
        """Perform memory cleanup (gc.collect).

        Call this after each epoch or on trial failure to release Python objects.
        """
        import gc

        collected = gc.collect()
        if collected > 0:
            logger.debug(f"[Memory] gc.collect() freed {collected} objects")


# Backward compatibility alias
MemoryTracker = EnterpriseMemoryManager


class HPOReporterCallback(BaseCallback):
    """Callback that reports per-batch metrics to WebUI via HPOReporter.

    This bridges TrainingEngine's step results to the WebUI Training Console
    by calling HPOReporter.report() for each valid batch, enabling real-time
    loss visibility in the HPO dashboard.
    """

    def __init__(
        self,
        reporter: HPOReporter,
        memory_manager: "EnterpriseMemoryManager",
        epoch_tracker: dict | None = None,
    ):
        """Initialize HPO reporter callback.

        Args:
            reporter: HPOReporter instance for WebUI logging.
            memory_manager: EnterpriseMemoryManager for memory stats.
            epoch_tracker: Optional dict to track current epoch (mutated externally).
        """
        self.reporter = reporter
        self.memory_manager = memory_manager
        self.epoch_tracker = epoch_tracker or {"epoch": 0}
        self._step_count = 0

    def on_batch_end(self, step: int, result: "StepResult") -> bool:
        """Report batch metrics to WebUI if valid.

        Args:
            step: Current training step.
            result: StepResult from TrainingEngine.train_step().

        Returns:
            True to continue training.
        """
        self._step_count += 1
        if result.is_valid and self.reporter.should_report():
            mem_stats = self.memory_manager.get_stats()

            # Compute real-time perplexity estimate from loss
            # PPL = exp(cross_entropy_loss) for next-token prediction
            perplexity = None
            if result.loss is not None and result.loss < 100:
                import math

                perplexity = math.exp(min(result.loss, 20))  # Cap to avoid overflow

            self.reporter.report(
                step=step,
                loss=result.loss,
                gradient_norm=result.gradient_norm,
                learning_rate=result.effective_learning_rate,
                epoch=self.epoch_tracker.get("epoch", 0),
                memory_mb=mem_stats.get("current_mb"),
                peak_memory_mb=mem_stats.get("peak_mb"),
                perplexity=perplexity,  # Real-time PPL for WebUI display
            )
        return True

    def on_epoch_start(self, epoch: int) -> bool:
        """Track current epoch for logging."""
        self.epoch_tracker["epoch"] = epoch
        return True


def evaluate_qsg_generation(
    model: tf.keras.Model,
    tokenizer,
    sample_prompts: list[str] | None = None,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    """Evaluate generation quality using Quantum Superposition Generation (QSG).

    QSG generates tokens in parallel using quantum-inspired mechanisms,
    achieving 50-100x speedup over autoregressive generation.

    Args:
        model: HSMN model with generate() method
        tokenizer: Tokenizer for encoding prompts
        sample_prompts: List of prompts to evaluate (default: built-in samples)
        max_new_tokens: Maximum tokens to generate per sample

    Returns:
        Dictionary with QSG evaluation metrics:
        - qsg_samples: Number of samples evaluated
        - avg_generation_time_ms: Average generation time per sample
        - tokens_per_second: Generation throughput
    """
    if not USE_QSG_GENERATION:
        logger.debug("[HPO] QSG evaluation disabled")
        return {}

    QSGGenerator = get_qsg_generator()
    if QSGGenerator is None:
        logger.warning("[HPO] QSG generator not available")
        return {}

    # Default sample prompts for quality evaluation
    if sample_prompts is None:
        sample_prompts = [
            "The future of artificial intelligence is",
            "In the quantum realm, particles can",
            "Machine learning has transformed",
            "The key to understanding language is",
            "Neural networks learn by",
        ]

    import time

    try:
        generator = QSGGenerator(model)

        total_time = 0.0
        total_tokens = 0

        for prompt in sample_prompts[:5]:  # Limit to 5 for efficiency
            # Encode prompt
            if hasattr(tokenizer, "encode"):
                input_ids = tokenizer.encode(prompt)
                if isinstance(input_ids, list):
                    input_ids = tf.constant([input_ids], dtype=tf.int32)
            else:
                # Fallback: create simple token IDs
                input_ids = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)

            # Time generation
            start = time.perf_counter()
            generator.generate(input_ids, max_new_tokens=max_new_tokens)
            elapsed = time.perf_counter() - start

            total_time += elapsed
            total_tokens += max_new_tokens

        # Compute metrics
        num_samples = min(len(sample_prompts), 5)
        avg_time_ms = (total_time / num_samples) * 1000
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

        metrics = {
            "qsg_samples": num_samples,
            "avg_generation_time_ms": avg_time_ms,
            "tokens_per_second": tokens_per_sec,
        }

        logger.info(
            f"[HPO] QSG Evaluation: {num_samples} samples, "
            f"{tokens_per_sec:.1f} tok/s, {avg_time_ms:.1f}ms/sample"
        )

        return metrics

    except Exception as e:
        logger.warning(f"[HPO] QSG evaluation failed: {e}")
        return {"qsg_error": str(e)}


def _evaluate_quality_metrics_inline(
    model: tf.keras.Model,
    vocab_size: int,
    seq_length: int = 256,
    num_samples: int = 50,
) -> dict[str, float]:
    """Evaluate quality metrics inline without external benchmark dependencies.

    This is a fallback implementation that computes perplexity and confidence
    directly using the trained model without requiring benchmark module imports.

    Args:
        model: Trained model to evaluate.
        vocab_size: Model vocabulary size for confidence normalization.
        seq_length: Sequence length for evaluation.
        num_samples: Number of samples to evaluate.

    Returns:
        Dictionary with perplexity, mean_confidence, expected_calibration_error.
    """
    import math

    import numpy as np

    logger.info(f"[HPO] Running inline quality evaluation on {num_samples} samples...")

    # Generate synthetic evaluation data with Zipf distribution
    np.random.seed(42)
    batch_size = 8

    all_losses = []
    all_confidences = []
    all_accuracies = []

    for _ in range(max(1, num_samples // batch_size)):
        # Generate tokens with Zipf distribution (realistic token distribution)
        tokens = np.random.zipf(1.5, size=(batch_size, seq_length)).astype(np.int32)
        tokens = np.clip(tokens, 1, vocab_size - 1)
        input_ids = tf.constant(tokens)
        target_ids = tf.roll(input_ids, -1, axis=1)

        try:
            # Forward pass
            outputs = model(input_ids, training=False)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs

            # Ensure logits have correct shape
            if len(logits.shape) != 3:
                logger.warning(f"[HPO] Unexpected logits shape: {logits.shape}")
                continue

            # Compute cross-entropy loss per token
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                target_ids, logits, from_logits=True
            )
            batch_mean_loss = tf.reduce_mean(loss).numpy()
            if math.isfinite(batch_mean_loss):
                all_losses.append(batch_mean_loss)

            # Compute entropy-based confidence
            # Entropy = -sum(p * log(p))
            probs = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.math.log(probs + 1e-10)
            entropy = -tf.reduce_sum(probs * log_probs, axis=-1)  # [batch, seq]

            # Normalize entropy to [0, 1] and convert to confidence
            max_entropy = math.log(vocab_size)
            normalized_entropy = entropy / max_entropy
            confidence = 1.0 - tf.clip_by_value(normalized_entropy, 0.0, 1.0)

            all_confidences.extend(tf.reduce_mean(confidence, axis=1).numpy().tolist())

            # Compute accuracy for calibration error estimation
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            correct = tf.cast(predictions == target_ids, tf.float32)
            all_accuracies.extend(tf.reduce_mean(correct, axis=1).numpy().tolist())

        except Exception as e:
            logger.debug(f"[HPO] Inline eval batch error: {e}")
            continue

    if not all_losses:
        logger.warning("[HPO] Inline evaluation produced no valid samples")
        return {}

    # Aggregate metrics
    mean_loss = float(np.mean(all_losses))
    overall_perplexity = math.exp(min(mean_loss, 20))  # Cap to avoid overflow
    mean_confidence = float(np.mean(all_confidences)) if all_confidences else 0.5

    # Compute simple ECE (binned calibration error)
    ece = 0.0
    if all_confidences and all_accuracies:
        confidences = np.array(all_confidences)
        accuracies = np.array(all_accuracies)

        # 10-bin ECE
        num_bins = 10
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        for i in range(num_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                avg_conf = confidences[in_bin].mean()
                avg_acc = accuracies[in_bin].mean()
                ece += abs(avg_acc - avg_conf) * prop_in_bin

    logger.info(
        f"[HPO] Inline quality: PPL={overall_perplexity:.2f}, "
        f"Conf={mean_confidence:.3f}, ECE={ece:.4f}"
    )

    return {
        "perplexity": overall_perplexity,
        "mean_confidence": mean_confidence,
        "expected_calibration_error": ece,
    }


def evaluate_quality_metrics(
    model: tf.keras.Model,
    vocab_size: int,
    seq_length: int = 256,
    num_samples: int = 50,
    dataset_name: str = "synthetic",
) -> dict[str, float]:
    """Evaluate model quality metrics (perplexity, confidence, calibration).

    This runs a lightweight evaluation on a validation set to compute:
    - Perplexity: Cross-entropy based language modeling quality
    - Mean Confidence: Average prediction confidence (entropy-based)
    - ECE: Expected Calibration Error (how well confidence matches accuracy)

    Args:
        model: Trained model to evaluate
        vocab_size: Model vocabulary size
        seq_length: Sequence length for evaluation
        num_samples: Number of samples to evaluate
        dataset_name: Dataset for perplexity ("synthetic" or HuggingFace name)

    Returns:
        Dictionary with perplexity, mean_confidence, expected_calibration_error
    """
    import math

    import numpy as np

    # Lazy imports to avoid circular dependencies
    use_inline_fallback = False
    try:
        from benchmarks.bench_confidence import (
            compute_confidence_from_entropy,
            compute_expected_calibration_error,
            compute_token_entropy,
        )
        from benchmarks.bench_perplexity import load_dataset_as_batches
    except ImportError as e:
        logger.warning(f"[HPO] Benchmark modules not available: {e}, using inline evaluation")
        use_inline_fallback = True

    # If benchmark imports failed, use inline evaluation
    if use_inline_fallback:
        return _evaluate_quality_metrics_inline(model, vocab_size, seq_length, num_samples)

    logger.info(f"[HPO] Evaluating quality metrics on {num_samples} samples...")

    perplexities = []
    all_confidences = []
    all_accuracies = []

    batch_size = 8
    try:
        for input_ids, target_ids in load_dataset_as_batches(
            dataset_name, vocab_size, seq_length, batch_size, num_samples
        ):
            # Forward pass
            outputs = model(input_ids, training=False)
            logits = outputs if isinstance(outputs, tf.Tensor) else outputs

            # Perplexity from cross-entropy
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                target_ids, logits, from_logits=True
            )
            mean_loss = float(tf.reduce_mean(loss).numpy())
            perplexities.append(math.exp(min(mean_loss, 100)))

            # Confidence from entropy
            entropy = compute_token_entropy(logits)
            confidence = compute_confidence_from_entropy(entropy, vocab_size)

            # Accuracy for calibration
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            correct = tf.cast(predictions == target_ids, tf.float32)

            all_confidences.extend(confidence.numpy().flatten().tolist())
            all_accuracies.extend(correct.numpy().flatten().tolist())

    except Exception as e:
        logger.warning(f"[HPO] Quality evaluation error during batch processing: {e}")
        return {}

    if not perplexities:
        logger.warning("[HPO] No perplexity samples collected")
        return {}

    # Aggregate metrics
    overall_perplexity = float(np.mean(perplexities))
    mean_confidence = float(np.mean(all_confidences)) if all_confidences else 0.0

    ece = 0.0
    if all_confidences and all_accuracies:
        ece, _ = compute_expected_calibration_error(
            np.array(all_confidences),
            np.array(all_accuracies),
            num_bins=10,
        )

    logger.info(
        f"[HPO] Quality: PPL={overall_perplexity:.2f}, "
        f"Conf={mean_confidence:.3f}, ECE={ece:.4f}"
    )

    return {
        "perplexity": overall_perplexity,
        "mean_confidence": mean_confidence,
        "expected_calibration_error": ece,
    }


def build_hsmn_model(
    config: dict[str, Any],
    vocab_size: int,
    hidden_dim_override: int | None = None,
    is_hd_mode: bool = False,
    hd_dim: int = 1024,
) -> tf.keras.Model:
    """
    Build HSMN model from hyperparameter configuration.

    Args:
        config: Hyperparameter configuration dictionary
        vocab_size: Vocabulary size for embedding and output layers
        hidden_dim_override: Optional override for hidden dimension
        is_hd_mode: If True, build model for HD streaming (float32 bundle inputs)
        hd_dim: HD bundle dimension (only used when is_hd_mode=True)

    Returns:
        Compiled tf.keras.Model ready for language modeling
    """
    # Extract hyperparameters with defaults
    hidden_dim = config.get("hidden_dim", 512)
    num_reasoning_blocks = config.get("num_reasoning_blocks", 8)
    num_heads = config.get("num_heads", 8)
    config.get("dropout_rate", 0.1)

    # =========================================================================
    # LITE EDITION LIMIT VALIDATION
    # Validates HPO trial config against scale limits BEFORE model construction.
    # This enables fail-fast behavior to save compute on overlimit trials.
    # Pro/Enterprise editions skip this via is_lite() check.
    # =========================================================================
    from types import SimpleNamespace

    from highnoon._native._limits import (
        LimitExceededError,
        is_lite,
        validate_model_config,
        validate_param_count,
    )

    # MoE parameters (extracted early for validation)
    moe_num_experts = config.get("num_moe_experts", 8)

    if is_lite():
        config_ns = SimpleNamespace(
            num_reasoning_blocks=num_reasoning_blocks,
            num_moe_experts=moe_num_experts,
            embedding_dim=hidden_dim,
            max_seq_length=config.get("context_window", 4096),
            vocab_size=vocab_size,
        )

        # Validate model configuration (raises LimitExceededError if exceeded)
        validate_model_config(config_ns)
        validate_param_count(config_ns)

        logger.info(
            f"[HPO] Lite edition validation passed: blocks={num_reasoning_blocks}, "
            f"experts={moe_num_experts}, dim={hidden_dim}, vocab={vocab_size}"
        )

    # FFN dimension computed from hidden_dim and expansion factor
    ff_expansion = config.get("ff_expansion", 4)
    ff_dim = config.get("ff_dim", hidden_dim * ff_expansion)

    # Mamba2 parameters
    mamba_state_dim = config.get("mamba_state_dim", 64)
    config.get("mamba_conv_dim", 4)
    config.get("mamba_expand", 2)

    # MoE parameters
    moe_num_experts = config.get("num_moe_experts", 8)
    moe_top_k = config.get("moe_top_k", 2)
    config.get("moe_capacity_factor", 1.25)

    # WLAM parameters
    wlam_num_heads = config.get("wlam_num_heads", 8)
    wlam_kernel_size = config.get("wlam_kernel_size", 3)
    config.get("wlam_num_landmarks", 32)

    # TensorTrain decomposition
    tt_rank_middle = config.get("tt_rank_middle", 16)

    # Quantum/superposition parameters
    superposition_dim = config.get("superposition_dim", 2)

    # Hamiltonian parameters
    config.get("hamiltonian_hidden_dim", 256)

    # Legacy parameters (kept for backward compatibility)
    config.get("spatial_hidden_dim", 256)
    config.get("spatial_num_layers", 2)
    config.get("graph_num_layers", 3)
    config.get("graph_hidden_dim", 128)
    config.get("graph_k_neighbors", 5)
    config.get("adapter_rank", 8)
    config.get("adapter_scale", 0.1)
    config.get("lsh_num_hashes", 4)
    config.get("lsh_bucket_size", 32)

    logger.info("[HPO] Building HSMN model with config:")
    logger.info(f"  - hidden_dim: {hidden_dim}")
    logger.info(f"  - num_reasoning_blocks: {num_reasoning_blocks}")
    logger.info(f"  - num_heads: {num_heads}")
    logger.info(f"  - ff_dim: {ff_dim}")
    logger.info(f"  - moe_num_experts: {moe_num_experts}, top_k: {moe_top_k}")
    logger.info(f"  - mamba_state_dim: {mamba_state_dim}")
    logger.info(f"  - superposition_dim: {superposition_dim}")
    logger.info(f"  - tt_rank_middle: {tt_rank_middle}")

    # Extract quantum enhancement parameters (Phases 26-36)
    use_unitary_residual = config.get("use_unitary_residual", True)
    use_quantum_norm = config.get("use_quantum_norm", True)
    use_unitary_expert = config.get("use_unitary_expert", True)
    config.get("unitary_residual_init_angle", 0.7854)
    config.get("neumann_cayley_terms", 6)

    logger.info(
        f"  - Quantum Ops: unitary_residual={use_unitary_residual}, qnorm={use_quantum_norm}, unitary_expert={use_unitary_expert}"
    )

    # Extract QHPM crystallization parameters (replaces EWC)
    use_qhpm_crystallization = config.get("use_qhpm_crystallization", True)
    qhpm_crystallization_threshold = config.get("qhpm_crystallization_threshold", 0.85)
    qhpm_max_directions = config.get("qhpm_max_directions", 256)

    logger.info(
        f"  - QHPM Crystallization: enabled={use_qhpm_crystallization}, threshold={qhpm_crystallization_threshold}, max_dirs={qhpm_max_directions}"
    )

    # Extract Quantum Enhancement Parameters v5.0 (Phases 47-84)
    # Pillar 1: Foundation
    use_qmd = config.get("use_quantum_measurement_dropout", True)
    config.get("qmd_drop_rate", 0.1)
    use_q_ssm = config.get("use_q_ssm_gating", True)
    use_qcb = config.get("use_quantum_coherence_bus", True)

    # Pillar 2: I/O Enhancement
    use_hqe = config.get("use_hyperdimensional_embedding", True)
    config.get("use_hypertokens", True)
    config.get("use_majorana_position", True)
    config.get("use_born_rule_loss", True)

    # Pillar 3: Topological Reasoning
    use_qasa = config.get("use_qasa_attention", True)
    use_mpqr = config.get("use_mpqr_reasoning", True)
    mpqr_num_paths = config.get("mpqr_num_paths", 8)
    use_td_moe = config.get("use_td_moe", True)

    # Pillar 4: Training
    use_vqem = config.get("use_vqem", True)
    use_rng = config.get("use_random_natural_gradient", True)

    # Pillar 5: Memory
    config.get("use_quantum_crystallization", True)

    # Pillar 6: Coherence
    config.get("use_multi_stage_hamiltonian", True)

    # Pillar 7: Advanced QI
    use_spini = config.get("use_spini_integrator", True)
    use_qcot = config.get("use_qcot_reasoning", True)

    logger.info(f"  - Quantum v5.0: QMD={use_qmd}, Q-SSM={use_q_ssm}, QCB={use_qcb}, HQE={use_hqe}")
    logger.info(
        f"  - Topological: QASA={use_qasa}, MPQR={use_mpqr}(paths={mpqr_num_paths}), TD-MoE={use_td_moe}"
    )
    logger.info(f"  - Training: VQEM={use_vqem}, RNG={use_rng}, SPINI={use_spini}, QCOT={use_qcot}")

    # Override hidden_dim if specified
    if hidden_dim_override is not None:
        hidden_dim = hidden_dim_override

    # Build model using Keras Functional API with full ReasoningModule
    if is_hd_mode:
        # HD Streaming Mode: float32 bundle inputs from HolographicCorpus
        # Use C++ HDStreamingAdapter op for projection
        logger.info(
            f"[HPO] Building HD-mode HSMN model: hd_dim={hd_dim} -> hidden_dim={hidden_dim}"
        )
        input_layer = tf.keras.layers.Input(shape=(hd_dim,), dtype=tf.float32, name="hd_bundle")

        # Project HD bundle to hidden dim using C++ op
        from highnoon._native.ops.hd_streaming_adapter import HDStreamingAdapter

        x = HDStreamingAdapter(
            hidden_dim=hidden_dim,
            hd_dim=hd_dim,
            use_bias=True,
            name="hd_streaming_adapter",
        )(input_layer)
        # x is now (batch, 1, hidden_dim) - ready for ReasoningModule
    else:
        # Standard mode: int32 token IDs
        input_layer = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_ids")

    # Use HyperdimensionalEmbedding for quantum-enhanced embedding (matching debug script)
    # Skip embedding in HD mode - HDStreamingAdapter already provides (batch, 1, hidden_dim)
    if is_hd_mode:
        # x is already set from HDStreamingAdapter above
        logger.info("[HPO] Skipping embedding layer in HD mode (HD bundles are pre-embedded)")
    elif use_hqe:
        try:
            # Check if we should use memory-efficient DualPathEmbedding
            use_dual_path = getattr(hn_config, "USE_DUAL_PATH_EMBEDDING", True)
            active_vocab_size = getattr(hn_config, "DUAL_PATH_ACTIVE_VOCAB_SIZE", 10000)

            if use_dual_path:
                from highnoon.models.layers.hyperdimensional_layer import DualPathEmbedding

                # PHASE 500+: Use the passed hd_dim from trial config (budget-filtered)
                # Previously used hidden_dim * 8 which bypassed budget constraints
                # Ensure divisibility: round up to nearest multiple of hidden_dim
                embed_hd_dim = (
                    hd_dim
                    if hd_dim % hidden_dim == 0
                    else ((hd_dim // hidden_dim) + 1) * hidden_dim
                )
                x = DualPathEmbedding(
                    vocab_size=vocab_size,
                    model_dim=hidden_dim,
                    active_vocab_size=min(active_vocab_size, vocab_size),  # Don't exceed vocab
                    hd_dim=embed_hd_dim,
                    use_ctqw=True,
                    ctqw_steps=3,
                    name="dual_path_embedding",
                )(input_layer)
                logger.info(
                    f"[HPO] Using DualPathEmbedding: vocab={vocab_size}, active={min(active_vocab_size, vocab_size)}, hd_dim={embed_hd_dim}"
                )
            else:
                from highnoon.models.layers.hyperdimensional_layer import HyperdimensionalEmbedding

                # PHASE 500+: Use the passed hd_dim from trial config (budget-filtered)
                # Previously used hidden_dim * 8 which bypassed budget constraints
                embed_hd_dim = (
                    hd_dim
                    if hd_dim % hidden_dim == 0
                    else ((hd_dim // hidden_dim) + 1) * hidden_dim
                )
                x = HyperdimensionalEmbedding(
                    vocab_size=vocab_size,
                    model_dim=hidden_dim,
                    hd_dim=embed_hd_dim,
                    use_ctqw=True,
                    ctqw_steps=3,
                    name="hde_embedding",
                )(input_layer)
                logger.info(
                    f"[HPO] Using HyperdimensionalEmbedding: vocab={vocab_size}, hd_dim={embed_hd_dim}"
                )
        except (ImportError, Exception) as e:
            logger.warning(
                f"[HPO] HyperdimensionalEmbedding failed ({e}), falling back to standard"
            )
            x = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=hidden_dim,
                name="token_embedding",
            )(input_layer)
    else:
        # Fallback: standard embedding layer
        x = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_dim,
            name="token_embedding",
        )(input_layer)

    # Phase 2.1: Extract expanded architecture parameters for ReasoningModule
    expanded_params = {
        # Chunking
        "chunk_size": config.get("chunk_size", 128),
        "chunk_stride": config.get("chunk_stride", 64),
        # Attention
        "local_attention_window": config.get("local_attention_window", 256),
        "local_attention_window_min": config.get("local_attention_window_min", 64),
        "local_attention_window_max": config.get("local_attention_window_max", 512),
        "local_attention_sigmoid_temp": config.get("local_attention_sigmoid_temp", 0.1),
        "local_attention_sparsity_ratio": config.get("local_attention_sparsity_ratio", 0.5),
        # Flash Linear
        "flash_linear_chunk_size": config.get("flash_linear_chunk_size", 64),
        "flash_linear_train_chunk_size": config.get("flash_linear_train_chunk_size", 64),
        "flash_linear_hybrid_window": config.get("flash_linear_hybrid_window", 128),
        "flash_linear_hybrid_alpha": config.get("flash_linear_hybrid_alpha", 0.5),
        "flash_linear_augment_rank": config.get("flash_linear_augment_rank", 32),
        "flash_linear_gate_init_bias": config.get("flash_linear_gate_init_bias", -3.0),
        # State Bus
        "state_bus_dim": config.get("state_bus_dim", 64),
        "state_bus_slots": config.get("state_bus_slots", 16),
        "state_bus_max_slots": config.get("state_bus_max_slots", 64),
        "state_bus_num_types": config.get("state_bus_num_types", 4),
        "state_bus_bond_dim": config.get("state_bus_bond_dim", 8),
        # Memory
        "memory_slots": config.get("memory_slots", 128),
        "memory_slot_dim": config.get("memory_slot_dim", 64),
        "memory_surprise_threshold": config.get("memory_surprise_threshold", 0.8),
        "memory_product_k": config.get("memory_product_k", 4),
        "memory_num_heads": config.get("memory_num_heads", 4),
        "memory_mps_bond_dim": config.get("memory_mps_bond_dim", 8),
        # CPU/HD Params
        "cache_tile_size": config.get("cache_tile_size", 64),
        "prefetch_distance": config.get("prefetch_distance", 2),
        "ssd_chunk_size": config.get("ssd_chunk_size", 128),
        "superposition_micro_batch_size": config.get("superposition_micro_batch_size", 4),
        # VQC Extended
        "vqc_qubits": config.get("vqc_qubits", 4),
        "vqc_feature_rotation_depth": config.get("vqc_feature_rotation_depth", 2),
        "neumann_cayley_terms": config.get("neumann_cayley_terms", 6),
        "neumann_series_terms": config.get("neumann_series_terms", 6),
        # DTC Extended
        "dtc_coupling_j": config.get("dtc_coupling_j", 1.0),
        "dtc_disorder_w": config.get("dtc_disorder_w", 0.5),
        "dtc_num_cycles": config.get("dtc_num_cycles", 10),
        # QSG
        "qsg_bond_dim": config.get("qsg_bond_dim", 8),
        "qsg_coherence_range": config.get("qsg_coherence_range", 0.5),
        "qsg_grover_iterations": config.get("qsg_grover_iterations", 1),
        "qsg_jacobi_iterations": config.get("qsg_jacobi_iterations", 5),
        "qsg_hopfield_beta": config.get("qsg_hopfield_beta", 10.0),
        "qsg_amplification_strength": config.get("qsg_amplification_strength", 1.5),
        # QHPM
        "qhpm_holographic_dim": config.get("qhpm_holographic_dim", 1024),
        "qhpm_mps_bond_dim": config.get("qhpm_mps_bond_dim", 16),
        "qhpm_hopfield_beta": config.get("qhpm_hopfield_beta", 10.0),
        "qhpm_max_crystallized_directions": config.get("qhpm_max_crystallized_directions", 128),
        # HD Extended
        "hd_optimizer_compression_ratio": config.get("hd_optimizer_compression_ratio", 0.5),
        "hd_gradient_rank": config.get("hd_gradient_rank", 16),
        "hd_gradient_bandwidth": config.get("hd_gradient_bandwidth", 128),
        "hd_kv_compression_ratio": config.get("hd_kv_compression_ratio", 2.0),
        "hd_control_fingerprint_dim": config.get("hd_control_fingerprint_dim", 64),
        "hd_stagnation_threshold": config.get("hd_stagnation_threshold", 0.1),
        "hd_kalman_state_compression": config.get("hd_kalman_state_compression", 4),
        "hd_projection_freeze_epochs": config.get("hd_projection_freeze_epochs", 2),
        "hqe_ctqw_steps": config.get("hqe_ctqw_steps", 3),
        "hd_active_vocab_size": config.get("hd_active_vocab_size", 5000),
        # Unified Bus
        "unified_bus_entanglement_init": config.get("unified_bus_entanglement_init", 0.5),
        "unified_bus_propagation_rate": config.get("unified_bus_propagation_rate", 0.1),
        # QASA
        "qasa_feature_rotation_depth": config.get("qasa_feature_rotation_depth", 2),
    }

    # Build the full HSMN reasoning module with hybrid block pattern
    # Note: Quantum parameters are read from global config by ReasoningModule
    # CRITICAL: Pass hd_dim so HD Activation Checkpointing uses trial-specific value
    reasoning_module = ReasoningModule(
        num_layers=num_reasoning_blocks,
        embedding_dim=hidden_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_experts=moe_num_experts,
        wlam_num_heads=wlam_num_heads,
        wlam_kernel_size=wlam_kernel_size,
        hd_activation_dim=hd_dim if is_hd_mode else None,  # Pass trial's hd_dim
        **expanded_params,
    )

    x = reasoning_module(x)

    # Output projection to vocabulary logits
    # Use QuantumLMHead for VQC-based output if available (matching debug script)
    use_quantum_lm_head = config.get("use_quantum_lm_head", True)
    if use_quantum_lm_head:
        try:
            from highnoon.models.reasoning.block_factory import QuantumLMHead

            output_layer = QuantumLMHead(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                vqc_layers=2,
                vqc_qubits=8,
                name="quantum_lm_head",
            )(x)
            logger.info(f"[HPO] Using QuantumLMHead: vocab={vocab_size}, vqc_qubits=8")
        except (ImportError, Exception) as e:
            logger.warning(f"[HPO] QuantumLMHead failed ({e}), falling back to Dense")
            output_layer = tf.keras.layers.Dense(vocab_size, name="lm_head")(x)
    else:
        output_layer = tf.keras.layers.Dense(vocab_size, name="lm_head")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="HSMN_LM")

    logger.info(f"[HPO] Built HSMN_LM model: vocab_size={vocab_size}, hidden_dim={hidden_dim}")
    return model


def create_optimizer(
    config: dict[str, Any],
    model: tf.keras.Model | None = None,
) -> tf.keras.optimizers.Optimizer:
    """
    Create optimizer from hyperparameter configuration.

    Args:
        config: Hyperparameter configuration dictionary
        model: The model to optimize (required for SophiaG optimizer)

    Returns:
        tf.keras.optimizers.Optimizer
    """
    learning_rate = config.get("learning_rate", 1e-4)
    optimizer_name = config.get("optimizer", "adam").lower()
    weight_decay = config.get("weight_decay", 0.01)
    total_steps = config.get("total_training_steps", 10000)

    # Ensure decay_steps is at least 1
    decay_steps = max(1, total_steps)

    # Simple cosine decay schedule (no warmup for HPO trials)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        alpha=0.1,  # Final LR = 10% of initial
    )

    # Phase 300+: HD Upgrade Parameters
    hd_compression_ratio = config.get("hd_optimizer_compression_ratio")

    # Create optimizer
    if optimizer_name == "sophiag":
        # Import SophiaG if available
        try:
            from highnoon.training.optimizers import SophiaG

            if model is None:
                logger.warning("[HPO] SophiaG requires model, falling back to AdamW")
                optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=lr_schedule,
                    weight_decay=weight_decay,
                )
            else:
                optimizer = SophiaG(
                    model=model,
                    learning_rate=lr_schedule,
                    weight_decay=weight_decay,
                    hd_compression_ratio=hd_compression_ratio,
                )
        except ImportError:
            logger.warning("[HPO] SophiaG not available, falling back to AdamW")
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
            )
    elif optimizer_name == "grover":
        # Phase 20.1: Grover-Q (Quantum-Enhanced) Optimizer
        # Uses amplitude-inspired learning rate scheduling for faster convergence
        # Classical simulation - ready for quantum backend via Qiskit
        grover_lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=learning_rate * 1.5,  # Quadratic-like amplification
            first_decay_steps=max(1, decay_steps // 4),  # Faster initial decay
            t_mul=2.0,  # Double restart period each time
            m_mul=0.9,  # Slightly reduce peak each restart
            alpha=0.05,  # Lower minimum for sharper focus
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=grover_lr,
            weight_decay=weight_decay * 0.8,  # Slightly less regularization
            beta_1=0.9,
            beta_2=0.98,  # Higher for quantum-inspired stability
        )
        logger.info("[HPO] Using Grover-Q (quantum-enhanced) optimizer with restarts")
    elif optimizer_name == "adamw":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "lion":
        # Lion optimizer - sign-based momentum, memory efficient
        try:
            from highnoon.training.optimizers import Lion

            optimizer = Lion(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
            )
            logger.info("[HPO] Using Lion (EvoLved Sign Momentum) optimizer")
        except ImportError:
            logger.warning("[HPO] Lion not available, falling back to AdamW")
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
            )
    elif optimizer_name == "qiao":
        # QIAO: Quantum-Inspired Alternating Optimizer
        # Alternates between SophiaG-style cost updates and exploration steps
        try:
            from highnoon.training.optimizers import QIAO

            if model is None:
                logger.warning("[HPO] QIAO requires model, falling back to AdamW")
                optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=lr_schedule,
                    weight_decay=weight_decay,
                )
            else:
                optimizer = QIAO(
                    model=model,
                    learning_rate=lr_schedule,
                    weight_decay=weight_decay,
                    mixer_frequency=config.get("qiao_mixer_frequency", 10),
                    mixer_strength=config.get("qiao_mixer_strength", 0.1),
                )
                logger.info("[HPO] Using QIAO (Quantum-Inspired Alternating) optimizer")
        except ImportError:
            logger.warning("[HPO] QIAO not available, falling back to AdamW")
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
            )
    elif optimizer_name == "sympflow":
        # SympFlow: Symplectic Hamiltonian Optimizer
        # Uses Hamiltonian dynamics with symplectic integration for optimization
        try:
            from highnoon.config import (  # noqa: F401 - imported for future use
                SYMPFLOW_FRICTION,
                SYMPFLOW_MASS,
                SYMPFLOW_NUM_LEAPFROG_STEPS,
                SYMPFLOW_STEP_SIZE,
            )

            # SympFlow wraps a base optimizer with Hamiltonian dynamics
            # For now, use a similar approach to Grover-Q with specialized LR scheduling
            sympflow_lr = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=learning_rate
                * config.get("sympflow_step_size", SYMPFLOW_STEP_SIZE)
                * 100,
                decay_steps=decay_steps,
                alpha=0.01,  # Low minimum for stable Hamiltonian evolution
            )
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=sympflow_lr,
                weight_decay=weight_decay * config.get("sympflow_friction", SYMPFLOW_FRICTION),
                beta_1=0.95,  # Higher momentum for symplectic evolution
                beta_2=0.999,
            )
            logger.info(
                f"[HPO] Using SympFlow (Symplectic Hamiltonian) optimizer with "
                f"mass={config.get('sympflow_mass', SYMPFLOW_MASS)}, "
                f"friction={config.get('sympflow_friction', SYMPFLOW_FRICTION)}"
            )
        except ImportError:
            logger.warning("[HPO] SympFlow config not available, using default AdamW")
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
            )
    elif optimizer_name == "sympflowqng":
        # S12: SympFlow + QNG Geodesic optimizer (Phase 131 Synergy)
        # Combines symplectic momentum with quantum natural geodesic corrections
        try:
            from highnoon.training.optimizers import SympFlowQNGOptimizer

            optimizer = SympFlowQNGOptimizer(
                learning_rate=learning_rate,
                mass=config.get("sympflowqng_mass", 1.0),
                friction=config.get("sympflowqng_friction", 0.01),
                geodesic_weight=config.get("sympflowqng_geodesic_weight", 0.1),
            )
            logger.info(
                "[HPO] Using SympFlowQNG (Synergy S12: Symplectic + QNG Geodesic) optimizer"
            )
        except ImportError:
            logger.warning("[HPO] SympFlowQNGOptimizer not available, falling back to AdamW")
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
            )
    else:  # adam
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
        )

    logger.info(
        f"[HPO] Created {optimizer_name} optimizer with lr={learning_rate}, wd={weight_decay}"
    )

    return optimizer


def create_engine_config(config: dict[str, Any]) -> "EnterpriseTrainingConfig":
    """Create EnterpriseTrainingConfig from trial config dictionary.

    This enables using TrainingEngine with all enterprise features (GaLore,
    QNG, Barren Plateau Monitor, Meta-Controller) in HPO trials, matching
    the debug script's training approach.

    Args:
        config: Trial configuration dictionary with hyperparameters and feature flags.

    Returns:
        EnterpriseTrainingConfig with all flags populated from trial config.
    """
    if not TRAINING_ENGINE_AVAILABLE or EnterpriseTrainingConfig is None:
        raise ImportError("TrainingEngine not available")

    return EnterpriseTrainingConfig(
        # Core training
        max_grad_norm=config.get("max_grad_norm", 1.0),
        max_nan_consecutive=config.get("max_nan_consecutive", 5),
        log_frequency=config.get("log_frequency", 10),
        loss_function=config.get("loss_function", "sparse_categorical_crossentropy"),
        label_smoothing=config.get("label_smoothing", 0.0),
        # Meta-Controller
        use_meta_controller=config.get("use_meta_controller", hn_config.USE_META_CONTROLLER),
        meta_controller_frequency=config.get(
            "meta_controller_frequency", hn_config.META_CONTROLLER_FREQUENCY
        ),
        # GaLore (critical for memory efficiency)
        use_galore=config.get("use_tensor_galore", hn_config.USE_TENSOR_GALORE),
        galore_rank=config.get("galore_rank", hn_config.GALORE_RANK),
        galore_update_proj_gap=config.get(
            "galore_update_proj_gap", hn_config.GALORE_UPDATE_PROJ_GAP
        ),
        galore_scale=config.get("galore_scale", hn_config.GALORE_SCALE),
        galore_vqc_aware=config.get("galore_vqc_aware", hn_config.GALORE_VQC_AWARE),
        # QALRC
        use_qalrc=config.get(
            "use_quantum_lr_controller", getattr(hn_config, "USE_QUANTUM_LR_CONTROLLER", True)
        ),
        qalrc_annealing_power=config.get("qalrc_annealing_power", 2.0),
        qalrc_tunneling_probability=config.get("qalrc_tunneling_probability", 0.05),
        qalrc_entropy_smoothing=config.get("qalrc_entropy_smoothing", 0.9),
        # Barren Plateau Monitor (critical for VQC layer stability)
        use_barren_plateau_detection=config.get(
            "barren_plateau_monitor", hn_config.BARREN_PLATEAU_MONITOR
        ),
        barren_plateau_threshold=config.get(
            "barren_plateau_threshold", hn_config.BARREN_PLATEAU_THRESHOLD
        ),
        barren_plateau_lr_scale=config.get(
            "barren_plateau_recovery_lr_scale", hn_config.BARREN_PLATEAU_RECOVERY_LR_SCALE
        ),
        # QNG (Quantum Natural Gradient)
        use_qng=config.get("use_quantum_natural_gradient", hn_config.USE_QUANTUM_NATURAL_GRADIENT),
        qng_damping=config.get("qng_damping", hn_config.QNG_DAMPING),
        qng_apply_to_quantum_only=config.get(
            "qng_apply_to_quantum_only", hn_config.QNG_APPLY_TO_QUANTUM_ONLY
        ),
        # Entropy Regularization
        use_entropy_regularization=config.get(
            "use_entropy_regularization", hn_config.USE_ENTROPY_REGULARIZATION
        ),
        entropy_reg_weight=config.get("entropy_reg_weight", hn_config.ENTROPY_REG_WEIGHT),
        # QHPM Crystallization
        use_qhpm_crystallization=config.get(
            "use_qhpm_crystallization", hn_config.ENABLE_QHPM_CRYSTALLIZATION
        ),
        crystallization_threshold=config.get(
            "qhpm_crystallization_threshold", hn_config.QHPM_CRYSTALLIZATION_THRESHOLD
        ),
        max_crystallized_directions=config.get(
            "qhpm_max_directions", hn_config.QHPM_MAX_CRYSTALLIZED_DIRECTIONS
        ),
        # SympFlow
        use_sympflow=config.get("use_sympflow_optimizer", hn_config.USE_SYMPFLOW_OPTIMIZER),
        sympflow_mass=config.get("sympflow_mass", hn_config.SYMPFLOW_MASS),
        sympflow_friction=config.get("sympflow_friction", hn_config.SYMPFLOW_FRICTION),
        # Neural ZNE/QEM
        use_neural_zne=config.get("use_neural_zne", hn_config.USE_NEURAL_ZNE),
        use_neural_qem=config.get("use_neural_qem", hn_config.USE_NEURAL_QEM),
        # UnifiedSmartTuner (HPO/WebUI integration)
        use_unified_smart_tuner=config.get(
            "use_unified_smart_tuner",
            getattr(hn_config, "USE_UNIFIED_SMART_TUNER", True),
        ),
        smart_tuner_coordination_mode=config.get(
            "smart_tuner_coordination_mode",
            getattr(hn_config, "SMART_TUNER_MODE", "balanced"),
        ),
        smart_tuner_warmup_steps=config.get(
            "smart_tuner_warmup_steps",
            getattr(hn_config, "SMART_TUNER_WARMUP_STEPS", 1000),
        ),
        smart_tuner_exploration_steps=config.get(
            "smart_tuner_exploration_steps",
            getattr(hn_config, "SMART_TUNER_EXPLORATION_STEPS", 10000),
        ),
        smart_tuner_memory_enabled=config.get(
            "smart_tuner_memory_enabled",
            getattr(hn_config, "SMART_TUNER_MEMORY_ENABLED", True),
        ),
        smart_tuner_webui_reporting=config.get(
            "smart_tuner_webui_reporting",
            getattr(hn_config, "SMART_TUNER_WEBUI_REPORTING", True),
        ),
    )


def train_trial(
    trial_id: str,
    config_path: str,
    epochs: int = 5,
    steps_per_epoch: int = 1500,
) -> float:
    """
    Train a single HPO trial using epoch-based training.

    HPO trials use a fixed epoch budget (default 5) to ensure consistent
    evaluation across different hyperparameter configurations. Early stopping
    is applied based on loss convergence to avoid wasting compute.

    Args:
        trial_id: Trial identifier
        config_path: Path to trial configuration JSON
        epochs: Number of epochs per trial (default: 5 for HPO)
        steps_per_epoch: Steps per epoch for progress calculation

    Returns:
        Best validation loss achieved
    """
    # Load trial configuration
    with open(config_path) as f:
        trial_config = json.load(f)

    logger.info(f"[HPO] Starting trial {trial_id}")
    logger.info(f"[HPO] Configuration: {json.dumps(trial_config, indent=2)}")
    logger.info(f"[HPO] Training for {epochs} epochs ({steps_per_epoch} steps/epoch)")

    # Initialize HPO reporter with sweep_id for real-time logging
    os.environ["HPO_TRIAL_ID"] = trial_id
    sweep_id = trial_config.get("sweep_id") or os.getenv("HPO_SWEEP_ID")
    hpo_reporter = HPOReporter(sweep_id=sweep_id)

    # Initialize enterprise memory manager with dynamic thresholds
    warning_threshold_pct = trial_config.get("memory_warning_pct", 0.95)
    memory_grace_steps = trial_config.get("memory_grace_steps", 10)
    memory_manager = EnterpriseMemoryManager(
        warning_threshold_pct=warning_threshold_pct,
        grace_steps=memory_grace_steps,
    )

    # Initialize quantum control callback for RLS, Hybrid PID, EKF, TNKF
    use_quantum_control = trial_config.get("use_quantum_control", True)
    meta_controller_frequency = trial_config.get(
        "meta_controller_frequency", META_CONTROLLER_FREQUENCY
    )

    _meta_callback = None  # Unused: TrainingEngine handles callbacks internally
    if use_quantum_control and META_CONTROLLER_AVAILABLE:
        _meta_callback = HamiltonianMetaControllerCallback(
            frequency=meta_controller_frequency,
            trigger_sysid_reload=True,  # Reload config on first call
        )
        logger.info(
            f"[HPO] Quantum control enabled: RLS={USE_RLS_SYSID}, "
            f"HybridPID={USE_HYBRID_PID}, frequency={meta_controller_frequency}"
        )
    elif use_quantum_control and not META_CONTROLLER_AVAILABLE:
        logger.warning("[HPO] Quantum control requested but native ops unavailable - skipping")
    else:
        logger.info("[HPO] Quantum control disabled for this trial")

    # QSG evaluation config
    use_qsg_evaluation = trial_config.get("use_qsg_evaluation", USE_QSG_GENERATION)

    # Get dimensions from config
    batch_size = trial_config.get("batch_size", 8)
    sequence_length = trial_config.get("sequence_length", 256)
    # Phase 1 Tokenizer Fix: Use target_vocab_size (tokenizer target) instead of vocab_size
    # Model vocab_size is ALWAYS derived from tokenizer.vocab_size after learning
    target_vocab_size = trial_config.get("target_vocab_size", 32000)
    hidden_dim = trial_config.get("hidden_dim") or 256  # Default if None or missing

    # Get optional HuggingFace dataset name (set from curriculum or manually)
    hf_dataset_name = trial_config.get("hf_dataset_name") or trial_config.get("dataset_name")
    curriculum_id = trial_config.get("curriculum_id")

    if hf_dataset_name:
        logger.info(f"[HPO] Using curriculum dataset: {hf_dataset_name}")
        if curriculum_id:
            logger.info(f"[HPO] Loaded from curriculum: {curriculum_id}")
    else:
        logger.warning("[HPO] No dataset specified, will use built-in sample texts")

    # Create tokenizer and load tokenized dataset
    try:
        # Configurable prefetch buffer for memory management (None = auto-tune)
        prefetch_buffer_size = trial_config.get("prefetch_buffer_size", None)

        # Quantum Tokenization Pipeline (Phase 48+)
        # When USE_INTELLIGENT_VOCAB_CONTROLLER is enabled, the tokenizer auto-trains
        # on the curriculum data to learn frequent n-grams and expand vocabulary.
        use_adaptive_tokenizer = trial_config.get(
            "use_adaptive_tokenizer", USE_INTELLIGENT_VOCAB_CONTROLLER  # Use config flag as default
        )
        auto_train_enabled = trial_config.get(
            "vocab_controller_auto_train", VOCAB_CONTROLLER_AUTO_TRAIN  # Use config flag as default
        )
        # Get vocab tuning hyperparameters from HPO search space
        # These control how the tokenizer learns from the curriculum
        adaptive_min_freq = trial_config.get("vocab_min_ngram_freq", 10)
        vocab_max_ngram_size = trial_config.get("vocab_max_ngram_size", 5)
        vocab_sample_size = trial_config.get("vocab_sample_size", 10000)

        logger.info(
            f"[HPO] Quantum Tokenization Pipeline: "
            f"auto_vocab={use_adaptive_tokenizer}, auto_train={auto_train_enabled}, "
            f"min_freq={adaptive_min_freq}, max_ngram={vocab_max_ngram_size}"
        )

        # Phase 200+: HD Streaming Mode (Quantum-Enhanced Memory Optimization)
        # Uses HolographicCorpus for amplitude-based reservoir sampling
        use_hd_streaming = trial_config.get("use_hd_streaming", True)  # Opt-in for now
        hd_reservoir_size = trial_config.get("hd_reservoir_size", 2000)
        hd_dim = trial_config.get("hd_dim")
        if hd_dim is None:
            # Phase 1.1 Fix: Respect SmartTuner reduction (hd = hidden * 8)
            hd_dim = hidden_dim * 8
        hd_sample_length = trial_config.get("hd_sample_length", 512)  # Tunable HD sample length

        if use_hd_streaming:
            logger.info(
                f"[HPO] HD Streaming Mode: reservoir={hd_reservoir_size}, hd_dim={hd_dim}, "
                f"sample_length={hd_sample_length}"
            )

        train_dataset, tokenizer, merger = load_training_dataset(
            batch_size=batch_size,
            sequence_length=sequence_length,
            vocab_size=target_vocab_size,  # Phase 1: Pass target_vocab_size to tokenizer
            hf_dataset_name=hf_dataset_name,
            prefetch_buffer_size=prefetch_buffer_size,
            use_adaptive_tokenizer=use_adaptive_tokenizer,
            adaptive_min_freq=adaptive_min_freq,
            max_samples=vocab_sample_size,  # Control corpus sample size for tokenizer learning
            # HD Streaming parameters
            use_hd_streaming=use_hd_streaming,
            hd_reservoir_size=hd_reservoir_size,
            hd_dim=hd_dim,
            hd_sample_length=hd_sample_length,  # Use tunable value from QAHPO
        )

        # Phase 1 Tokenizer Fix: ALWAYS derive model vocab from tokenizer
        # This ensures zero dead embeddings and eliminates loss floors at log(vocab_size)
        vocab_size = tokenizer.vocab_size
        logger.info(f"[HPO] Vocabulary aligned: target={target_vocab_size}, actual={vocab_size}")

        if merger is not None:
            logger.info(f"[HPO] SuperwordMerger active: {merger.superword_count} superwords")
            # CRITICAL: Extend vocab_size to include superword tokens
            # SuperwordMerger assigns tokens with indices >= base vocab_size
            vocab_size = vocab_size + merger.superword_count
            logger.info(f"[HPO] Extended vocab_size to {vocab_size} (includes superwords)")
    except Exception as e:
        logger.error(f"[HPO] Failed to load dataset: {e}")
        raise

    # Build model with embedding layer (or HD streaming adapter if enabled)
    model = build_hsmn_model(
        trial_config,
        vocab_size,
        hidden_dim_override=hidden_dim,
        is_hd_mode=use_hd_streaming,
        hd_dim=hd_dim,
    )

    # Create optimizer (pass model for SophiaG)
    optimizer = create_optimizer(trial_config, model=model)

    # Estimate parameter count for logging
    param_count = model.count_params()

    # ========================================================================
    # PARAM BUDGET ENFORCEMENT
    # Skip trials that exceed the user-specified parameter budget.
    # This prevents wasting compute on oversized models.
    # ========================================================================
    param_budget = trial_config.get("param_budget")
    if param_budget is not None and param_count > param_budget:
        budget_msg = (
            f"Model exceeds parameter budget: {param_count / 1e6:.1f}M > {param_budget / 1e6:.1f}M. "
            f"Skipping trial without counting as attempt."
        )
        logger.warning(f"[HPO] {budget_msg}")

        # Clean up model to free memory
        del model
        del optimizer
        gc.collect()

        # Report skip to WebUI and write status file
        hpo_reporter.log(
            f"BUDGET_EXCEEDED: {budget_msg}",
            level="WARNING",
            param_count=param_count,
            param_budget=param_budget,
        )

        # Write status.json with budget exceeded error so executor can detect it
        hpo_reporter.complete(
            success=False,
            error=f"BUDGET_EXCEEDED: {budget_msg}",
            param_count=param_count,
        )

        # Return special sentinel value indicating budget skip (not a failure)
        # The orchestrator should recognize this and not count it as a trial
        # Using float("nan") to distinguish from inf (training failure)
        return float("nan")

    # Log trial configuration to WebUI console for visibility
    # Enrich config with actual values used
    trial_config_for_log = {
        **trial_config,
        "vocab_size": vocab_size,
        "param_count": param_count,
    }
    hpo_reporter.log_trial_start(trial_config_for_log, param_count=param_count)

    # ========================================================================
    # Use TrainingEngine for enterprise-grade training with all enhancements
    # This replaces the manual training loop to enable: GaLore, QNG, Barren
    # Plateau Monitor, Meta-Controller, QHPM Crystallization, Neural ZNE
    # ========================================================================

    if not TRAINING_ENGINE_AVAILABLE:
        error_msg = "TrainingEngine not available - cannot run HPO trial"
        logger.error(f"[HPO] {error_msg}")
        hpo_reporter.complete(success=False, error=error_msg)
        return float("inf")

    # Create TrainingEngine config from trial config
    engine_config = create_engine_config(trial_config)

    # Log which enhancements are enabled
    logger.info("[HPO] TrainingEngine configuration:")
    logger.info(f"  GaLore: {engine_config.use_galore} (rank={engine_config.galore_rank})")
    logger.info(f"  QNG: {engine_config.use_qng} (damping={engine_config.qng_damping})")
    logger.info(f"  Barren Plateau: {engine_config.use_barren_plateau_detection}")
    logger.info(f"  Meta-Controller: {engine_config.use_meta_controller}")
    logger.info(f"  QHPM Crystallization: {engine_config.use_qhpm_crystallization}")

    # Create TrainingEngine
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        config=engine_config,
    )

    # Link engine to HPO sweep for WebUI status reporting
    total_training_steps = epochs * steps_per_epoch
    if sweep_id:
        engine.set_sweep_id(sweep_id)
    engine.set_total_steps(total_training_steps)

    # Run training with TrainingEngine
    logger.info(
        f"[HPO] Starting training: {epochs} epochs x {steps_per_epoch} steps = {total_training_steps} total steps"
    )
    logger.info(f"  UnifiedSmartTuner: {engine_config.use_unified_smart_tuner}")

    try:
        # Create HPO reporter callback for per-batch logging to WebUI
        epoch_tracker = {"epoch": 0}
        reporter_callback = HPOReporterCallback(hpo_reporter, memory_manager, epoch_tracker)

        result: TrainingResult = engine.run(
            epochs=epochs,
            dataset=train_dataset,
            callbacks=[reporter_callback],  # Wire per-batch metrics to WebUI
            steps_per_epoch=steps_per_epoch,  # CRITICAL: Must pass to limit infinite generator
        )

        best_loss = result.final_loss if result.final_loss is not None else float("inf")
        epochs_completed = result.epochs_completed

        if not result.success:
            logger.warning(f"[HPO] Training completed with issues: {result.error}")

    except Exception as e:
        logger.error(f"[HPO] Training failed: {e}")
        memory_manager.cleanup()
        hpo_reporter.complete(success=False, error=str(e))
        return float("inf")

    # QSG evaluation after training
    use_qsg_evaluation = trial_config.get("use_qsg_evaluation", USE_QSG_GENERATION)
    throughput_tokens_per_sec = 0.0
    if use_qsg_evaluation:
        qsg_metrics = evaluate_qsg_generation(model, tokenizer)
        if qsg_metrics:
            throughput_tokens_per_sec = qsg_metrics.get("tokens_per_second", 0)
            logger.info(f"[HPO] QSG metrics: {throughput_tokens_per_sec:.1f} tok/s")

    # Early stopping logging (TrainingEngine may have stopped early)
    # The TrainingEngine handles early stopping internally.
    # The `epochs_completed` from `result` will reflect if it stopped early.

    # Compute efficiency score for ranking
    param_count = estimate_model_params(trial_config)
    param_budget = trial_config.get("param_budget", param_count)  # Default to own size if no budget
    lambda_efficiency = trial_config.get("lambda_efficiency", DEFAULT_LAMBDA_EFFICIENCY)
    efficiency_score = compute_efficiency_score(
        best_loss, param_count, param_budget, lambda_efficiency
    )

    # ===== Multi-Objective Quality Metrics Evaluation =====
    quality_metrics: dict[str, float] = {}
    evaluate_quality = trial_config.get("evaluate_quality_metrics", True)

    if evaluate_quality:
        quality_eval_samples = trial_config.get("quality_eval_samples", 50)
        quality_eval_dataset = trial_config.get("quality_eval_dataset", "synthetic")

        try:
            quality_metrics = evaluate_quality_metrics(
                model,
                vocab_size,
                seq_length=sequence_length,
                num_samples=quality_eval_samples,
                dataset_name=quality_eval_dataset,
            )
        except Exception as e:
            logger.warning(f"[HPO] Quality evaluation failed: {e}")
            quality_metrics = {}

    # Compute composite score for multi-objective ranking
    # Guard against None or inf best_loss from failed trials
    if best_loss is None or not math.isfinite(best_loss):
        logger.warning(
            f"[HPO] Trial {trial_id}: best_loss={best_loss} is not finite, "
            "using inf for composite score"
        )
        composite_score = float("inf")
    else:
        composite_score = compute_composite_score(
            loss=best_loss,
            perplexity=quality_metrics.get("perplexity"),
            ece=quality_metrics.get("expected_calibration_error"),
            param_count=param_count,
            param_budget=param_budget,
            alpha_loss=trial_config.get("alpha_loss", DEFAULT_ALPHA_LOSS),
            beta_perplexity=trial_config.get("beta_perplexity", DEFAULT_BETA_PERPLEXITY),
            gamma_calibration=trial_config.get("gamma_calibration", DEFAULT_GAMMA_CALIBRATION),
            lambda_efficiency=lambda_efficiency,
        )

    # Get final memory stats for multi-objective optimization
    final_mem_stats = memory_manager.get_stats()
    peak_memory_mb = final_mem_stats.get("peak_mb", 0.0)

    # Report completion with all metrics including memory and throughput
    hpo_reporter.complete(
        success=True,
        final_loss=best_loss,
        param_count=param_count,
        efficiency_score=efficiency_score,
        perplexity=quality_metrics.get("perplexity"),
        mean_confidence=quality_metrics.get("mean_confidence"),
        expected_calibration_error=quality_metrics.get("expected_calibration_error"),
        composite_score=composite_score,
        memory_peak_mb=peak_memory_mb,
        epochs_completed=epochs_completed,  # From TrainingResult
        throughput_tokens_per_sec=throughput_tokens_per_sec,
    )

    # Log comprehensive completion summary with memory
    ppl_str = (
        f"{quality_metrics.get('perplexity', 0):.2f}"
        if quality_metrics.get("perplexity")
        else "N/A"
    )
    mem_str = f", peak_mem={peak_memory_mb:.0f}MB" if peak_memory_mb > 0 else ""
    logger.info(
        f"[HPO] Trial {trial_id} completed: loss={best_loss:.6f}, "
        f"composite={composite_score:.6f}, ppl={ppl_str}, "
        f"~{param_count / 1e6:.1f}M params{mem_str}"
    )

    return best_loss


def main():
    """Main entry point for HPO trial runner."""
    import signal

    # Global flag to track if we received a termination signal
    _shutdown_requested = False
    _shutdown_status_file = None
    _shutdown_trial_id = None

    def signal_handler(signum, frame):
        """Handle SIGINT/SIGTERM gracefully to avoid GIL corruption."""
        nonlocal _shutdown_requested
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.warning(f"[HPO] Received {sig_name}, initiating graceful shutdown...")
        _shutdown_requested = True

        # Write interrupted status if possible
        if _shutdown_status_file and _shutdown_trial_id:
            try:
                status = {
                    "trial_id": _shutdown_trial_id,
                    "loss": None,
                    "error": f"Interrupted by {sig_name}",
                    "status": "interrupted",
                }
                with open(_shutdown_status_file, "w") as f:
                    json.dump(status, f, indent=2)
            except Exception:
                pass

        # Raise KeyboardInterrupt to allow graceful cleanup
        raise KeyboardInterrupt(f"Received {sig_name}")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="HPO Trial Runner")
    parser.add_argument("--trial_id", type=str, required=True, help="Trial identifier")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to trial configuration JSON"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs per trial (default: 5)"
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=100,
        help="Steps per epoch (default: 100)",
    )

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"[HPO] Config file not found: {args.config}")
        sys.exit(1)

    # Determine status output path (in same directory as config for SweepExecutor)
    config_dir = Path(args.config).parent
    status_file = config_dir / "status.json"

    # Store for signal handler access
    _shutdown_status_file = status_file
    _shutdown_trial_id = args.trial_id

    # Train trial with epoch-based training
    try:
        best_loss = train_trial(
            args.trial_id,
            args.config,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
        )
        logger.info(f"[HPO] Trial completed successfully with loss={best_loss:.6f}")

        # Write status file for SweepExecutor to read
        # CRITICAL: Use math.isfinite() to check for both NaN AND Inf
        # NaN is used as a sentinel for budget-exceeded trials
        # Inf is used as a sentinel for failed training
        # Both are invalid JSON and must be converted to None
        if math.isfinite(best_loss):
            status = {
                "trial_id": args.trial_id,
                "loss": best_loss,
                "epochs_completed": args.epochs,
                "status": "completed",
            }
            with open(status_file, "w") as f:
                json.dump(status, f, indent=2)
        else:
            logger.info(
                f"[HPO] Trial {args.trial_id} finished with non-finite loss {best_loss}, status.json preserved"
            )

        sys.exit(0)

    except KeyboardInterrupt:
        # Graceful shutdown requested
        logger.warning("[HPO] Trial interrupted by user/system signal")
        status = {
            "trial_id": args.trial_id,
            "loss": None,
            "error": "Interrupted by signal",
            "status": "interrupted",
        }
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"[HPO] Trial failed: {e}", exc_info=True)

        # Write failure status
        status = {
            "trial_id": args.trial_id,
            "loss": None,
            "error": str(e),
            "status": "failed",
        }
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)

        sys.exit(1)


if __name__ == "__main__":
    main()
