#!/usr/bin/env python3
"""HPO NaN Debug Script - Terminal-based HPO for debugging NaN issues.

This script replicates the WebUI HPO sweep functionality but runs entirely
in the terminal with comprehensive debug logging to trace the source of
NaN/Inf values in training.

Usage:
    python scripts/debug_hpo_nan.py --trials 1 --epochs 2 --optimizer sympflowqng

Debug Features:
    - Verbose loss tracking at every step
    - Gradient norm monitoring with NaN detection
    - Per-layer gradient analysis
    - Memory tracking
    - Early NaN detection with stack traces
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging BEFORE any imports
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "debug_hpo_nan.log", mode="w"),
    ],
)
logger = logging.getLogger("HPO_NAN_DEBUG")

# Suppress noisy TensorFlow logs but keep warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf

# Import global config
import highnoon.config as hn_config

# ============================================================================
# Debug Configuration
# ============================================================================


@dataclass
class NaNDebugConfig:
    """Configuration for NaN debugging."""

    # Model architecture
    hidden_dim: int = 256
    num_reasoning_blocks: int = 4
    num_moe_experts: int = 4
    vocab_size: int = 32000
    sequence_length: int = 128
    batch_size: int = 8

    # Training
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    epochs: int = 2
    steps_per_epoch: int = 20
    max_grad_norm: float = 1.0

    # HPO
    num_trials: int = 1
    param_budget: int = 100_000_000

    # Debug options
    log_every_step: bool = True
    check_gradients_per_layer: bool = True
    trace_nan_source: bool = True
    dump_model_summary: bool = True
    use_synthetic_data: bool = True  # Use synthetic for faster debugging


# ============================================================================
# NaN Detection Utilities
# ============================================================================


def check_tensor_health(tensor: tf.Tensor, name: str) -> dict[str, Any]:
    """Check a tensor for NaN, Inf, and other issues.

    Args:
        tensor: Tensor to check
        name: Name for logging

    Returns:
        Health report dictionary
    """
    if tensor is None:
        return {"name": name, "status": "None", "is_healthy": False}

    try:
        # Check if tensor is a float type (NaN/Inf only apply to floats)
        is_float_tensor = tensor.dtype in [tf.float16, tf.float32, tf.float64, tf.bfloat16]

        if isinstance(tensor, tf.IndexedSlices):
            check_tensor = tensor.values
        else:
            check_tensor = tensor

        # Only check NaN/Inf for float tensors
        if is_float_tensor:
            has_nan = bool(tf.reduce_any(tf.math.is_nan(check_tensor)).numpy())
            has_inf = bool(tf.reduce_any(tf.math.is_inf(check_tensor)).numpy())
            is_healthy = not has_nan and not has_inf
        else:
            # Integer tensors can't have NaN/Inf
            has_nan = False
            has_inf = False
            is_healthy = True

        # Stats work for all numeric types
        min_val = float(tf.reduce_min(check_tensor).numpy())
        max_val = float(tf.reduce_max(check_tensor).numpy())
        mean_val = float(tf.reduce_mean(tf.cast(check_tensor, tf.float32)).numpy())

        if not is_healthy:
            logger.error(
                f"🚨 UNHEALTHY TENSOR: {name}\n"
                f"   has_nan={has_nan}, has_inf={has_inf}\n"
                f"   min={min_val}, max={max_val}, mean={mean_val}"
            )

        return {
            "name": name,
            "dtype": str(tensor.dtype),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "is_healthy": is_healthy,
        }
    except Exception as e:
        logger.warning(f"Could not check tensor {name}: {e}")
        # Assume healthy if we can't check (non-critical for debug)
        return {"name": name, "status": "unchecked", "is_healthy": True}


def check_model_weights(model: tf.keras.Model) -> list[dict]:
    """Check all model weights for NaN/Inf values.

    Args:
        model: Keras model to check

    Returns:
        List of health reports for each weight
    """
    reports = []
    for weight in model.trainable_weights:
        report = check_tensor_health(weight, weight.name)
        reports.append(report)
        if not report["is_healthy"]:
            logger.error(f"🚨 Bad weight found: {weight.name}")
    return reports


def check_gradients(gradients: list, variables: list) -> tuple[bool, list[dict]]:
    """Check all gradients for NaN/Inf values.

    Args:
        gradients: List of gradient tensors
        variables: List of corresponding variables

    Returns:
        Tuple of (all_healthy, reports)
    """
    reports = []
    all_healthy = True

    for grad, var in zip(gradients, variables):
        if grad is None:
            reports.append({"name": var.name, "status": "None gradient", "is_healthy": True})
            continue

        report = check_tensor_health(grad, f"grad:{var.name}")
        reports.append(report)

        if not report["is_healthy"]:
            all_healthy = False
            logger.error(
                f"🚨 Bad gradient for {var.name}:\n"
                f"   Shape: {var.shape}\n"
                f"   Grad stats: min={report.get('min')}, max={report.get('max')}"
            )

    return all_healthy, reports


# ============================================================================
# Custom Training Step with Full Debug Logging
# ============================================================================


class NaNDebugTrainer:
    """Custom trainer with comprehensive NaN debugging."""

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        config: NaNDebugConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.global_step = 0
        self.nan_count = 0
        self.first_nan_step = None

        # Loss function
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
        )

        logger.info(f"NaNDebugTrainer initialized with optimizer: {type(optimizer).__name__}")

    def train_step(self, inputs: tf.Tensor, labels: tf.Tensor) -> dict[str, Any]:
        """Execute a single training step with full debugging.

        Args:
            inputs: Input tensor [batch, seq_len]
            labels: Label tensor [batch, seq_len]

        Returns:
            Dictionary with step results and debug info
        """
        step_start = time.time()

        # Check inputs before forward pass
        if self.config.trace_nan_source:
            input_health = check_tensor_health(inputs, "inputs")
            label_health = check_tensor_health(labels, "labels")
            if not input_health["is_healthy"] or not label_health["is_healthy"]:
                logger.error(f"Step {self.global_step}: Input data contains NaN/Inf!")
                return {"loss": float("nan"), "error": "bad_inputs", "is_valid": False}

        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(inputs, training=True)

            # Check predictions
            if self.config.trace_nan_source:
                pred_health = check_tensor_health(predictions, "predictions")
                if not pred_health["is_healthy"]:
                    logger.error(
                        f"Step {self.global_step}: Forward pass produced NaN/Inf!\n"
                        f"   Predictions: min={pred_health.get('min')}, max={pred_health.get('max')}"
                    )
                    return {"loss": float("nan"), "error": "nan_forward", "is_valid": False}

            # Compute loss (per-token)
            per_token_loss = self.loss_fn(labels, predictions)

            # Check per-token loss
            if self.config.trace_nan_source:
                per_token_health = check_tensor_health(per_token_loss, "per_token_loss")
                if not per_token_health["is_healthy"]:
                    logger.error(
                        f"Step {self.global_step}: Per-token loss contains NaN/Inf!\n"
                        f"   Stats: min={per_token_health.get('min')}, max={per_token_health.get('max')}\n"
                        f"   This usually means:\n"
                        f"   - Label indices out of vocab range\n"
                        f"   - Extremely large logits causing softmax overflow"
                    )

            # Reduce to scalar
            loss = tf.reduce_mean(per_token_loss)

        loss_val = float(loss.numpy())

        # Check loss value
        if not math.isfinite(loss_val):
            self.nan_count += 1
            if self.first_nan_step is None:
                self.first_nan_step = self.global_step
                logger.error(
                    f"🚨🚨🚨 FIRST NaN LOSS at step {self.global_step}! 🚨🚨🚨\n"
                    f"   Loss value: {loss_val}\n"
                    f"   Predictions: shape={predictions.shape}\n"
                    f"   This is the source of the NaN issue!\n"
                    f"   Stack trace:\n{traceback.format_stack()[-5:]}"
                )
            return {
                "loss": loss_val,
                "is_valid": False,
                "gradient_norm": 0.0,
                "error": "nan_loss",
                "nan_count": self.nan_count,
            }

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Check gradients
        if self.config.check_gradients_per_layer:
            grads_healthy, grad_reports = check_gradients(gradients, self.model.trainable_variables)
            if not grads_healthy:
                self.nan_count += 1
                logger.error(f"Step {self.global_step}: NaN/Inf in gradients!")
                # Find which layers have bad gradients
                bad_layers = [r["name"] for r in grad_reports if not r["is_healthy"]]
                logger.error(f"   Bad gradient layers: {bad_layers[:5]}...")  # First 5
                return {
                    "loss": loss_val,
                    "is_valid": False,
                    "error": "nan_gradients",
                    "bad_layers": bad_layers,
                }

        # Compute gradient norm
        valid_grads = [g for g in gradients if g is not None]
        if valid_grads:
            grad_norm = float(tf.linalg.global_norm(valid_grads).numpy())
        else:
            grad_norm = 0.0

        # Check gradient norm
        if not math.isfinite(grad_norm):
            self.nan_count += 1
            logger.error(f"Step {self.global_step}: Gradient norm is {grad_norm}!")
            return {"loss": loss_val, "is_valid": False, "error": "inf_gradient_norm"}

        # Clip gradients
        clipped_grads, _ = tf.clip_by_global_norm(valid_grads, self.config.max_grad_norm)

        # Apply gradients
        valid_vars = [v for g, v in zip(gradients, self.model.trainable_variables) if g is not None]
        self.optimizer.apply_gradients(zip(clipped_grads, valid_vars))

        # Check weights after update
        if self.config.trace_nan_source and self.global_step % 10 == 0:
            weight_reports = check_model_weights(self.model)
            bad_weights = [r["name"] for r in weight_reports if not r["is_healthy"]]
            if bad_weights:
                logger.error(
                    f"Step {self.global_step}: Bad weights after update: {bad_weights[:5]}..."
                )

        step_time = time.time() - step_start
        self.global_step += 1

        # Log step results
        if self.config.log_every_step:
            logger.debug(
                f"Step {self.global_step}: loss={loss_val:.6f}, "
                f"grad_norm={grad_norm:.4f}, time={step_time*1000:.1f}ms"
            )

        return {
            "loss": loss_val,
            "gradient_norm": grad_norm,
            "step_time": step_time,
            "is_valid": True,
        }

    def run_epoch(
        self,
        dataset: tf.data.Dataset,
        steps: int | None = None,
    ) -> dict[str, Any]:
        """Run a single training epoch.

        Args:
            dataset: Training dataset
            steps: Max steps per epoch (None = full dataset)

        Returns:
            Epoch results dictionary
        """
        losses = []
        grad_norms = []
        step_count = 0

        for inputs, labels in dataset:
            if steps is not None and step_count >= steps:
                break

            result = self.train_step(inputs, labels)

            if result["is_valid"]:
                losses.append(result["loss"])
                grad_norms.append(result["gradient_norm"])
            else:
                # Stop on first NaN if debugging
                if self.config.trace_nan_source:
                    logger.error(f"Stopping epoch early due to invalid step at {self.global_step}")
                    break

            step_count += 1

        if not losses:
            return {
                "mean_loss": float("nan"),
                "steps": step_count,
                "success": False,
                "error": "no_valid_steps",
            }

        return {
            "mean_loss": float(sum(losses) / len(losses)),
            "mean_grad_norm": float(sum(grad_norms) / len(grad_norms)) if grad_norms else 0.0,
            "steps": step_count,
            "valid_steps": len(losses),
            "success": True,
        }


# ============================================================================
# Model Building
# ============================================================================


def build_debug_model(config: NaNDebugConfig) -> tf.keras.Model:
    """Build a simplified HSMN model for debugging.

    Args:
        config: Debug configuration

    Returns:
        Compiled Keras model
    """
    from highnoon.models.reasoning import ReasoningModule

    logger.info(
        f"\nBuilding model:\n"
        f"   vocab_size={config.vocab_size}\n"
        f"   hidden_dim={config.hidden_dim}\n"
        f"   num_blocks={config.num_reasoning_blocks}\n"
        f"   num_experts={config.num_moe_experts}"
    )

    # Input layer
    input_layer = tf.keras.layers.Input(
        shape=(config.sequence_length,), dtype=tf.int32, name="input_ids"
    )

    # Embedding
    x = tf.keras.layers.Embedding(
        input_dim=config.vocab_size,
        output_dim=config.hidden_dim,
        name="token_embeddings",
    )(input_layer)

    # Reasoning module
    reasoning = ReasoningModule(
        num_layers=config.num_reasoning_blocks,
        embedding_dim=config.hidden_dim,
        num_heads=4,
        ff_dim=config.hidden_dim * 4,
        num_experts=config.num_moe_experts,
    )
    x = reasoning(x)

    # Output projection
    output = tf.keras.layers.Dense(config.vocab_size, name="lm_head")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output, name="DebugHSMN")

    param_count = model.count_params()
    logger.info(f"Model built: {param_count:,} parameters ({param_count/1e6:.1f}M)")

    if config.dump_model_summary:
        model.summary(print_fn=logger.info)

    return model


def create_optimizer(
    config: NaNDebugConfig, model: tf.keras.Model | None = None
) -> tf.keras.optimizers.Optimizer:
    """Create optimizer based on config.

    Args:
        config: Debug configuration
        model: Model (required for SophiaG/QIAO)

    Returns:
        Keras optimizer
    """
    lr = config.learning_rate
    optimizer_name = config.optimizer.lower()

    logger.info(f"Creating optimizer: {optimizer_name} with lr={lr}")

    if optimizer_name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=0.01,
        )
    elif optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "sophiag":
        try:
            from highnoon.training.optimizers import SophiaG

            if model is None:
                logger.warning("SophiaG requires model, falling back to AdamW")
                return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.01)
            return SophiaG(model=model, learning_rate=lr)
        except ImportError:
            logger.warning("SophiaG not available, using AdamW")
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.01)
    elif optimizer_name == "sympflowqng":
        try:
            from highnoon.training.optimizers import SympFlowQNGOptimizer

            return SympFlowQNGOptimizer(learning_rate=lr)
        except ImportError:
            logger.warning("SympFlowQNG not available, using AdamW")
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.01)
    elif optimizer_name == "qiao":
        try:
            from highnoon.training.optimizers import QIAO

            if model is None:
                logger.warning("QIAO requires model, falling back to AdamW")
                return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.01)
            return QIAO(model=model, learning_rate=lr)
        except ImportError:
            logger.warning("QIAO not available, using AdamW")
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.01)
    else:
        logger.warning(f"Unknown optimizer {optimizer_name}, using Adam")
        return tf.keras.optimizers.Adam(learning_rate=lr)


def create_synthetic_dataset(config: NaNDebugConfig) -> tf.data.Dataset:
    """Create synthetic dataset for debugging.

    Args:
        config: Debug configuration

    Returns:
        TensorFlow dataset
    """
    logger.info(
        f"Creating synthetic dataset:\n"
        f"   batch_size={config.batch_size}\n"
        f"   seq_length={config.sequence_length}\n"
        f"   vocab_size={config.vocab_size}"
    )

    def generator():
        while True:
            # Generate random token IDs within vocab range
            inputs = tf.random.uniform(
                (config.batch_size, config.sequence_length),
                minval=0,
                maxval=config.vocab_size,
                dtype=tf.int32,
            )
            # Labels are shifted inputs (next token prediction)
            labels = tf.concat(
                [
                    inputs[:, 1:],
                    tf.random.uniform((config.batch_size, 1), 0, config.vocab_size, tf.int32),
                ],
                axis=1,
            )
            yield inputs, labels

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((config.batch_size, config.sequence_length), tf.int32),
            tf.TensorSpec((config.batch_size, config.sequence_length), tf.int32),
        ),
    )

    return dataset.prefetch(2)


# ============================================================================
# Trial Runner
# ============================================================================


def run_trial(
    trial_id: int,
    config: NaNDebugConfig,
) -> dict[str, Any]:
    """Run a single HPO trial with full debugging.

    Args:
        trial_id: Trial identifier
        config: Debug configuration

    Returns:
        Trial results dictionary
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING TRIAL {trial_id}")
    logger.info(f"{'='*60}")
    logger.info(f"Config: {config}")

    trial_start = time.time()
    trainer = None  # Initialize to avoid UnboundLocalError in exception handler
    nan_count = 0
    first_nan_step = None

    try:
        # Build model
        model = build_debug_model(config)

        # Check initial weights
        logger.info("\nChecking initial model weights...")
        weight_reports = check_model_weights(model)
        bad_weights = [r["name"] for r in weight_reports if not r["is_healthy"]]
        if bad_weights:
            logger.error(f"Bad initial weights: {bad_weights}")
            return {"trial_id": trial_id, "loss": None, "error": "bad_initial_weights"}
        logger.info("✓ All initial weights are healthy")

        # Create optimizer (pass model for SophiaG/QIAO)
        optimizer = create_optimizer(config, model=model)

        # Create dataset
        dataset = create_synthetic_dataset(config)

        # Create trainer
        trainer = NaNDebugTrainer(model, optimizer, config)

        # Run training epochs
        best_loss = float("inf")
        for epoch in range(config.epochs):
            logger.info(f"\n--- Epoch {epoch + 1}/{config.epochs} ---")

            result = trainer.run_epoch(dataset, steps=config.steps_per_epoch)

            logger.info(
                f"Epoch {epoch + 1} complete: "
                f"loss={result['mean_loss']:.6f}, "
                f"steps={result['steps']}, "
                f"valid={result.get('valid_steps', 0)}"
            )

            if not result["success"]:
                logger.error(f"Epoch failed: {result.get('error', 'unknown')}")
                break

            if result["mean_loss"] < best_loss:
                best_loss = result["mean_loss"]

            # Force garbage collection between epochs
            gc.collect()

        trial_time = time.time() - trial_start
        nan_count = trainer.nan_count
        first_nan_step = trainer.first_nan_step

        logger.info(f"\nTrial {trial_id} completed in {trial_time:.1f}s")
        logger.info(f"Best loss: {best_loss:.6f}")
        logger.info(f"NaN count: {nan_count}")
        if first_nan_step is not None:
            logger.warning(f"First NaN at step: {first_nan_step}")

        # Clean up
        del model
        del optimizer
        del trainer
        gc.collect()

        return {
            "trial_id": trial_id,
            "loss": best_loss if math.isfinite(best_loss) else None,
            "nan_count": nan_count,
            "first_nan_step": first_nan_step,
            "wall_time": trial_time,
            "success": nan_count == 0 and math.isfinite(best_loss),
        }

    except Exception as e:
        trial_time = time.time() - trial_start
        logger.error(f"Trial {trial_id} failed with exception: {e}")
        logger.error(traceback.format_exc())
        return {
            "trial_id": trial_id,
            "loss": None,
            "error": str(e),
            "nan_count": nan_count,
            "first_nan_step": first_nan_step,
            "wall_time": trial_time,
            "traceback": traceback.format_exc(),
        }


# ============================================================================
# Main
# ============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HPO NaN Debug Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--blocks", type=int, default=4, help="Number of reasoning blocks")
    parser.add_argument("--experts", type=int, default=4, help="Number of MoE experts")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sophiag", "sympflowqng", "qiao"],
        help="Optimizer to use",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per trial")
    parser.add_argument("--steps", type=int, default=20, help="Steps per epoch")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clip norm")

    # HPO
    parser.add_argument("--trials", type=int, default=1, help="Number of trials to run")

    # Debug options
    parser.add_argument("--no-log-every-step", action="store_true", help="Disable per-step logging")
    parser.add_argument("--no-trace-nan", action="store_true", help="Disable NaN tracing")
    parser.add_argument(
        "--no-check-grads", action="store_true", help="Disable per-layer gradient checks"
    )

    args = parser.parse_args()

    # Build config
    config = NaNDebugConfig(
        hidden_dim=args.hidden_dim,
        num_reasoning_blocks=args.blocks,
        num_moe_experts=args.experts,
        vocab_size=args.vocab_size,
        sequence_length=args.seq_length,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        max_grad_norm=args.max_grad_norm,
        num_trials=args.trials,
        log_every_step=not args.no_log_every_step,
        trace_nan_source=not args.no_trace_nan,
        check_gradients_per_layer=not args.no_check_grads,
    )

    logger.info("=" * 60)
    logger.info("HPO NaN DEBUG SCRIPT")
    logger.info("=" * 60)
    logger.info(f"\nConfiguration:\n{json.dumps(vars(config), indent=2, default=str)}")

    # Run trials
    results = []
    for trial_id in range(config.num_trials):
        result = run_trial(trial_id, config)
        results.append(result)

        # Log result
        status = "✓" if result.get("success") else "✗"
        loss_str = (
            f"{result.get('loss', float('nan')):.6f}" if result.get("loss") is not None else "NaN"
        )
        logger.info(f"Trial {trial_id}: {status} loss={loss_str}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    successful = sum(1 for r in results if r.get("success"))
    logger.info(f"Trials: {len(results)}, Successful: {successful}")

    for r in results:
        status = "SUCCESS" if r.get("success") else "FAILED"
        loss_str = f"{r.get('loss', float('nan')):.6f}" if r.get("loss") is not None else "NaN"
        error = r.get("error", "")
        nan_info = ""
        if r.get("nan_count", 0) > 0:
            nan_info = f", nan_count={r['nan_count']}, first_nan_step={r.get('first_nan_step')}"
        logger.info(f"  Trial {r['trial_id']}: {status}, loss={loss_str}{nan_info} {error}")

    # Save results to file
    results_file = PROJECT_ROOT / "debug_hpo_nan_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Debug log saved to: {PROJECT_ROOT / 'debug_hpo_nan.log'}")


if __name__ == "__main__":
    main()
