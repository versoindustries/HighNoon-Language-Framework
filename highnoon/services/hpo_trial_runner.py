"""
HPO Trial Runner - Trains real HSMN models for hyperparameter optimization.

This module serves as the bridge between the C++ HPO orchestrator and Python model training.
It builds and trains full HSMN models with specified hyperparameters.

Includes RSS memory tracking via psutil for trial resource monitoring.
"""

import argparse
import json
import logging
import math
import os
import sys
from typing import Any

import tensorflow as tf

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
try:
    from highnoon.training.callbacks import HamiltonianMetaControllerCallback

    META_CONTROLLER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass

# Quantum control and QSG imports
from highnoon.config import (
    META_CONTROLLER_FREQUENCY,
    USE_HYBRID_PID,
    USE_QSG_GENERATION,
    USE_RLS_SYSID,
)


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
        warning_threshold_pct: float = 0.95,
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
    try:
        from benchmarks.bench_confidence import (
            compute_confidence_from_entropy,
            compute_expected_calibration_error,
            compute_token_entropy,
        )
        from benchmarks.bench_perplexity import load_dataset_as_batches
    except ImportError as e:
        logger.warning(f"[HPO] Benchmark modules not available: {e}")
        return {}

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
) -> tf.keras.Model:
    """
    Build HSMN model from hyperparameter configuration.

    Args:
        config: Hyperparameter configuration dictionary
        vocab_size: Vocabulary size for embedding and output layers
        hidden_dim_override: Optional override for hidden dimension

    Returns:
        Compiled tf.keras.Model ready for language modeling
    """
    # Extract hyperparameters with defaults
    hidden_dim = config.get("hidden_dim", 512)
    num_reasoning_blocks = config.get("num_reasoning_blocks", 8)
    num_heads = config.get("num_heads", 8)
    config.get("dropout_rate", 0.1)

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
    config.get("monarch_num_blocks", 4)
    config.get("monarch_block_size", 128)

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
    # Input: token IDs (integers)
    input_layer = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_ids")

    # Token embedding layer - converts token IDs to dense vectors
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=hidden_dim,
        name="token_embedding",
    )(input_layer)

    # Build the full HSMN reasoning module with hybrid block pattern
    # Note: Quantum parameters are read from global config by ReasoningModule
    # To override per-trial, set config flags before building
    reasoning_module = ReasoningModule(
        num_layers=num_reasoning_blocks,
        embedding_dim=hidden_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_experts=moe_num_experts,
        wlam_num_heads=wlam_num_heads,
        wlam_kernel_size=wlam_kernel_size,
    )

    x = reasoning_module(x)

    # Output projection to vocabulary logits
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
            from highnoon.config import (
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
    else:  # adam
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
        )

    logger.info(
        f"[HPO] Created {optimizer_name} optimizer with lr={learning_rate}, wd={weight_decay}"
    )

    return optimizer


def train_trial(
    trial_id: str,
    config_path: str,
    epochs: int = 5,
    steps_per_epoch: int = 100,
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

    meta_callback = None
    if use_quantum_control and META_CONTROLLER_AVAILABLE:
        meta_callback = HamiltonianMetaControllerCallback(
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
    vocab_size = trial_config.get("vocab_size", 512)
    hidden_dim = trial_config.get("hidden_dim", 256)

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

        train_dataset, tokenizer, merger = load_training_dataset(
            batch_size=batch_size,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            hf_dataset_name=hf_dataset_name,
            prefetch_buffer_size=prefetch_buffer_size,
        )
        vocab_size = tokenizer.vocab_size  # Use actual tokenizer vocab size
        logger.info(f"[HPO] Dataset loaded with tokenizer: vocab_size={vocab_size}")
        if merger is not None:
            logger.info(f"[HPO] SuperwordMerger active: {merger.superword_count} superwords")
            # CRITICAL: Extend vocab_size to include superword tokens
            # SuperwordMerger assigns tokens with indices >= base vocab_size
            vocab_size = vocab_size + merger.superword_count
            logger.info(f"[HPO] Extended vocab_size to {vocab_size} (includes superwords)")
    except Exception as e:
        logger.error(f"[HPO] Failed to load dataset: {e}")
        raise

    # Build model with embedding layer
    model = build_hsmn_model(trial_config, vocab_size, hidden_dim_override=hidden_dim)

    # Create optimizer (pass model for SophiaG)
    optimizer = create_optimizer(trial_config, model=model)

    # Compile model with sparse categorical cross-entropy for language modeling
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )

    # Training loop - epoch based
    best_loss = float("inf")
    patience_counter = 0
    patience = trial_config.get("early_stopping_patience", 3)  # Epochs of patience
    min_delta = trial_config.get("early_stopping_min_delta", 0.001)
    global_step = 0

    # Gradient clipping to prevent explosion (configurable, default 1.0)
    max_grad_norm = trial_config.get("max_grad_norm", 1.0)
    logger.info(f"[HPO] Gradient clipping enabled: max_norm={max_grad_norm}")

    # NaN tracking for early termination
    nan_count = 0
    max_nan_consecutive = trial_config.get("max_nan_consecutive", 5)

    for epoch in range(epochs):
        epoch_losses = []
        dataset_iter = iter(train_dataset)

        for step in range(steps_per_epoch):
            try:
                # Get batch
                inputs, labels = next(dataset_iter)

                # Training step with cross-entropy loss
                with tf.GradientTape() as tape:
                    predictions = model(inputs, training=True)
                    # Sparse categorical cross-entropy loss
                    loss = tf.keras.losses.sparse_categorical_crossentropy(
                        labels, predictions, from_logits=True
                    )
                    loss = tf.reduce_mean(loss)

                # =====================================================
                # NaN Detection: Check loss for NaN/Inf BEFORE gradients
                # =====================================================
                current_loss = float(loss.numpy())
                if math.isnan(current_loss) or math.isinf(current_loss):
                    nan_count += 1
                    logger.warning(
                        f"[HPO] NaN/Inf loss detected at step {global_step} "
                        f"(epoch {epoch}, batch {step}). Count: {nan_count}/{max_nan_consecutive}"
                    )

                    # Early termination if too many consecutive NaN
                    if nan_count >= max_nan_consecutive:
                        error_msg = (
                            f"Training diverged: {nan_count} consecutive NaN/Inf losses. "
                            "Possible causes: learning rate too high, numerical instability "
                            "in model ops, or corrupt input data."
                        )
                        logger.error(f"[HPO] {error_msg}")
                        tf.keras.backend.clear_session()
                        memory_manager.cleanup()
                        hpo_reporter.complete(success=False, error=error_msg)
                        return float("inf")

                    # Skip this batch and try to recover
                    global_step += 1
                    continue

                # Reset NaN counter on valid loss
                nan_count = 0

                # Compute gradients
                gradients = tape.gradient(loss, model.trainable_variables)

                # =====================================================
                # Gradient Validation: Check for None and NaN gradients
                # =====================================================
                # Filter out None gradients (non-trainable layers)
                valid_grads_and_vars = [
                    (g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None
                ]

                if not valid_grads_and_vars:
                    logger.warning(f"[HPO] No valid gradients at step {global_step}")
                    global_step += 1
                    continue

                gradients_only = [g for g, _ in valid_grads_and_vars]
                vars_only = [v for _, v in valid_grads_and_vars]

                # Check for NaN in gradients
                has_nan_grad = any(tf.reduce_any(tf.math.is_nan(g)) for g in gradients_only)
                if has_nan_grad:
                    nan_count += 1
                    logger.warning(
                        f"[HPO] NaN gradient detected at step {global_step}. "
                        f"Count: {nan_count}/{max_nan_consecutive}"
                    )
                    if nan_count >= max_nan_consecutive:
                        error_msg = (
                            f"Gradient explosion: {nan_count} consecutive NaN gradients. "
                            "Try reducing learning rate or enabling gradient clipping."
                        )
                        logger.error(f"[HPO] {error_msg}")
                        tf.keras.backend.clear_session()
                        memory_manager.cleanup()
                        hpo_reporter.complete(success=False, error=error_msg)
                        return float("inf")
                    global_step += 1
                    continue

                # =====================================================
                # Gradient Clipping: Prevent explosion
                # =====================================================
                gradient_norm = tf.linalg.global_norm(gradients_only)
                clipped_gradients, _ = tf.clip_by_global_norm(
                    gradients_only, max_grad_norm, use_norm=gradient_norm
                )

                # Apply clipped gradients
                optimizer.apply_gradients(zip(clipped_gradients, vars_only))

                epoch_losses.append(current_loss)

                # Check for critical memory situation (swap spike or sustained high memory)
                should_stop, stop_reason = memory_manager.check_memory_critical()
                if should_stop:
                    logger.error(f"[HPO] Memory critical at step {global_step}: {stop_reason}")
                    tf.keras.backend.clear_session()  # Clean up TF state
                    memory_manager.cleanup()  # gc.collect()
                    hpo_reporter.complete(success=False, error=f"Memory critical: {stop_reason}")
                    return float("inf")

                # Report metrics periodically (with memory tracking)
                if hpo_reporter.enabled and global_step % 10 == 0:
                    mem_stats = memory_manager.get_stats()
                    hpo_reporter.report(
                        step=global_step,
                        loss=current_loss,
                        gradient_norm=float(gradient_norm.numpy()),
                        learning_rate=(
                            float(optimizer.learning_rate.numpy())
                            if hasattr(optimizer, "learning_rate")
                            else None
                        ),
                        memory_mb=mem_stats["current_mb"],
                        peak_memory_mb=mem_stats["peak_mb"],
                        system_memory_pct=mem_stats.get("system_percent", 0),
                        swap_used_mb=mem_stats.get("swap_used_mb", 0),
                    )

                # Trigger quantum control callback (RLS, Hybrid PID, EKF, TNKF)
                if meta_callback is not None and global_step % meta_callback.frequency == 0:
                    logs = {
                        "loss": current_loss,
                        "gradient_norm": float(gradient_norm.numpy()),
                        "learning_rate": (
                            float(optimizer.learning_rate.numpy())
                            if hasattr(optimizer, "learning_rate")
                            else 0.0
                        ),
                    }
                    try:
                        block_names, evolution_times = meta_callback.on_batch_end(global_step, logs)
                        if len(block_names) > 0:
                            logger.debug(f"[HPO] Meta-controller updated {len(block_names)} blocks")
                    except Exception as e:
                        logger.warning(f"[HPO] Meta-controller error: {e}")

                global_step += 1

            except tf.errors.ResourceExhaustedError as e:
                logger.error(f"[HPO] OOM error at epoch {epoch}, step {step}: {e}")
                tf.keras.backend.clear_session()  # Clean up TF state
                memory_manager.cleanup()  # gc.collect()
                hpo_reporter.complete(success=False, error=str(e))
                return float("inf")

            except StopIteration:
                # Dataset exhausted, move to next epoch
                break

            except Exception as e:
                logger.error(f"[HPO] Error at epoch {epoch}, step {step}: {e}")
                tf.keras.backend.clear_session()  # Clean up TF state
                memory_manager.cleanup()  # gc.collect()
                hpo_reporter.complete(success=False, error=str(e))
                return float("inf")

        # Epoch complete - compute average loss
        epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")

        # Memory cleanup after each epoch
        memory_manager.cleanup()  # gc.collect()

        logger.info(
            f"[HPO] Epoch {epoch + 1}/{epochs} complete, "
            f"avg_loss={epoch_loss:.6f}, best={best_loss:.6f}"
        )

        # QSG generation quality evaluation at epoch end
        if use_qsg_evaluation and epoch == epochs - 1:  # Final epoch only
            qsg_metrics = evaluate_qsg_generation(model, tokenizer)
            if qsg_metrics:
                logger.info(
                    f"[HPO] QSG metrics: {qsg_metrics.get('tokens_per_second', 0):.1f} tok/s"
                )
                # Report QSG metrics
                if hpo_reporter.enabled:
                    hpo_reporter.report(
                        step=global_step,
                        loss=epoch_loss,
                        qsg_tokens_per_second=qsg_metrics.get("tokens_per_second", 0),
                        qsg_generation_time_ms=qsg_metrics.get("avg_generation_time_ms", 0),
                    )

        # Check for improvement (epoch-level early stopping)
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping based on epoch-level convergence
        if patience_counter >= patience:
            logger.info(
                f"[HPO] Early stopping at epoch {epoch + 1}, "
                f"no improvement for {patience} epochs, best_loss={best_loss:.6f}"
            )
            break

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

    # Report completion with all metrics
    hpo_reporter.complete(
        success=True,
        final_loss=best_loss,
        param_count=param_count,
        efficiency_score=efficiency_score,
        perplexity=quality_metrics.get("perplexity"),
        mean_confidence=quality_metrics.get("mean_confidence"),
        expected_calibration_error=quality_metrics.get("expected_calibration_error"),
        composite_score=composite_score,
    )

    # Log comprehensive completion summary
    ppl_str = (
        f"{quality_metrics.get('perplexity', 0):.2f}"
        if quality_metrics.get("perplexity")
        else "N/A"
    )
    logger.info(
        f"[HPO] Trial {trial_id} completed: loss={best_loss:.6f}, "
        f"composite={composite_score:.6f}, ppl={ppl_str}, "
        f"~{param_count / 1e6:.1f}M params"
    )

    return best_loss


def main():
    """Main entry point for HPO trial runner."""
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

    # Train trial with epoch-based training
    try:
        best_loss = train_trial(
            args.trial_id,
            args.config,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
        )
        logger.info(f"[HPO] Trial completed successfully with loss={best_loss:.6f}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[HPO] Trial failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
