#!/usr/bin/env python3
"""
Validate PID and State-Space Configuration Functionality.

This script launches a training loop that specifically validates the functionality
of pid.conf and state_space.conf by:
1. Loading the config files through the C++ HamiltonianMetaController
2. Running a training loop that triggers the meta-controller callback
3. Verifying that evolution times are updated for model blocks
4. Checking that the control system responds to training metrics

Usage:
    python scripts/validate_control_configs.py
    python scripts/validate_control_configs.py --epochs 3 --steps 50 --frequency 5
    python scripts/validate_control_configs.py --verbose  # Show detailed control outputs

Copyright 2025 Verso Industries
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def validate_config_files_exist() -> dict[str, bool]:
    """Validate that pid.conf and state_space.conf exist and are readable.

    Returns:
        Dictionary with validation results for each config file
    """
    results = {}

    # Check pid.conf
    pid_conf_path = PROJECT_ROOT / "pid.conf"
    results["pid_conf_exists"] = pid_conf_path.exists()
    if results["pid_conf_exists"]:
        try:
            with open(pid_conf_path) as f:
                content = f.read()
            results["pid_conf_readable"] = True
            results["pid_conf_sections"] = []
            for line in content.split("\n"):
                if line.startswith("[") and line.endswith("]"):
                    results["pid_conf_sections"].append(line.strip("[]"))
            logger.info(f"pid.conf exists with sections: {results['pid_conf_sections']}")
        except Exception as e:
            results["pid_conf_readable"] = False
            logger.error(f"Failed to read pid.conf: {e}")
    else:
        logger.error("pid.conf not found at project root")

    # Check state_space.conf
    state_space_path = PROJECT_ROOT / "state_space.conf"
    results["state_space_conf_exists"] = state_space_path.exists()
    if results["state_space_conf_exists"]:
        try:
            with open(state_space_path) as f:
                content = f.read()
            results["state_space_conf_readable"] = True
            results["state_space_conf_sections"] = []
            for line in content.split("\n"):
                if line.startswith("[") and line.endswith("]"):
                    results["state_space_conf_sections"].append(line.strip("[]"))
            logger.info(
                f"state_space.conf exists with sections: {results['state_space_conf_sections']}"
            )
        except Exception as e:
            results["state_space_conf_readable"] = False
            logger.error(f"Failed to read state_space.conf: {e}")
    else:
        logger.error("state_space.conf not found at project root")

    return results


def validate_native_ops_available() -> bool:
    """Check if native ops (including meta controller) are available.

    Returns:
        True if native ops are available, False otherwise
    """
    try:
        from highnoon._native.ops.meta_controller_op import trigger_meta_controller

        logger.info("Native meta_controller_op is available")
        return True
    except ImportError as e:
        logger.error(f"Native ops not available: {e}")
        logger.error("Run 'cd highnoon/_native && ./build_secure.sh' to compile")
        return False


def validate_callback_instantiation() -> bool:
    """Check if HamiltonianMetaControllerCallback can be instantiated.

    Returns:
        True if callback can be created, False otherwise
    """
    try:
        from highnoon.training.callbacks import HamiltonianMetaControllerCallback

        callback = HamiltonianMetaControllerCallback(
            frequency=10,
            trigger_sysid_reload=True,
        )
        logger.info(
            f"HamiltonianMetaControllerCallback created with frequency={callback.frequency}"
        )
        return True
    except ImportError as e:
        logger.error(f"Failed to import callback: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to instantiate callback: {e}")
        return False


def run_training_with_meta_controller(
    epochs: int = 2,
    steps_per_epoch: int = 20,
    controller_frequency: int = 5,
    hidden_dim: int = 256,
    num_reasoning_blocks: int = 4,
    verbose: bool = False,
) -> dict:
    """Run a training loop with the HamiltonianMetaController enabled.

    This function creates a minimal model and runs training while triggering
    the meta-controller callback to verify pid.conf and state_space.conf are
    being used correctly.

    Args:
        epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        controller_frequency: How often to trigger the meta-controller (in steps)
        hidden_dim: Model hidden dimension
        num_reasoning_blocks: Number of reasoning blocks
        verbose: Enable verbose output from controller

    Returns:
        Dictionary with validation results
    """
    import tensorflow as tf

    from highnoon.data.loaders import load_training_dataset
    from highnoon.services.hpo_trial_runner import build_hsmn_model, create_optimizer
    from highnoon.training.callbacks import HamiltonianMetaControllerCallback

    results = {
        "training_completed": False,
        "controller_triggers": 0,
        "blocks_updated": [],
        "evolution_times_seen": [],
        "errors": [],
        "final_loss": float("inf"),
    }

    # Configuration
    vocab_size = 512
    batch_size = 4
    sequence_length = 128
    learning_rate = 1e-4

    config = {
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "num_reasoning_blocks": num_reasoning_blocks,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "learning_rate": learning_rate,
        "optimizer": "adamw",
        "num_heads": 4,
        "ff_dim": hidden_dim * 4,
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "total_training_steps": epochs * steps_per_epoch,
    }

    logger.info("=" * 60)
    logger.info("Running Training with Meta-Controller Validation")
    logger.info("=" * 60)
    logger.info(f"Epochs: {epochs}, Steps/Epoch: {steps_per_epoch}")
    logger.info(f"Controller Frequency: {controller_frequency} batches")

    try:
        # Load dataset (using built-in samples for speed)
        logger.info("Loading dataset...")
        train_dataset, tokenizer, merger = load_training_dataset(
            batch_size=batch_size,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            hf_dataset_name=None,  # Use built-in samples
        )
        actual_vocab_size = tokenizer.vocab_size
        if merger is not None:
            actual_vocab_size += merger.superword_count

        # Build model
        logger.info("Building HSMN model...")
        model = build_hsmn_model(config, actual_vocab_size, hidden_dim_override=hidden_dim)
        logger.info(f"Model parameters: {model.count_params():,}")

        # Create optimizer
        optimizer = create_optimizer(config, model=model)

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["sparse_categorical_accuracy"],
        )

        # Create meta-controller callback
        meta_callback = HamiltonianMetaControllerCallback(
            frequency=controller_frequency,
            trigger_sysid_reload=True,  # Force reload of pid.conf and state_space.conf
        )
        logger.info("Meta-controller callback created with config reload trigger")

        # Training loop
        global_step = 0
        epoch_losses = []

        for epoch in range(epochs):
            dataset_iter = iter(train_dataset)
            batch_losses = []

            for step in range(steps_per_epoch):
                try:
                    inputs, labels = next(dataset_iter)

                    # Training step
                    with tf.GradientTape() as tape:
                        predictions = model(inputs, training=True)
                        loss = tf.keras.losses.sparse_categorical_crossentropy(
                            labels, predictions, from_logits=True
                        )
                        loss = tf.reduce_mean(loss)

                    # Check for NaN
                    current_loss = float(loss.numpy())
                    if math.isnan(current_loss) or math.isinf(current_loss):
                        logger.warning(f"NaN/Inf loss at step {global_step}, skipping")
                        global_step += 1
                        continue

                    # Compute gradients
                    gradients = tape.gradient(loss, model.trainable_variables)
                    valid_grads = [
                        (g, v)
                        for g, v in zip(gradients, model.trainable_variables)
                        if g is not None
                    ]

                    if not valid_grads:
                        logger.warning(f"No valid gradients at step {global_step}")
                        global_step += 1
                        continue

                    grads, vars_ = zip(*valid_grads)
                    gradient_norm = tf.linalg.global_norm(list(grads))
                    clipped_grads, _ = tf.clip_by_global_norm(
                        list(grads), 1.0, use_norm=gradient_norm
                    )
                    optimizer.apply_gradients(zip(clipped_grads, vars_))

                    batch_losses.append(current_loss)

                    # ====== KEY: Trigger Meta-Controller ======
                    # This is where pid.conf and state_space.conf get exercised
                    logs = {
                        "loss": current_loss,
                        "gradient_norm": float(gradient_norm.numpy()),
                        "learning_rate": (
                            float(optimizer.learning_rate.numpy())
                            if hasattr(optimizer, "learning_rate")
                            else learning_rate
                        ),
                    }

                    try:
                        block_names, evolution_times = meta_callback.on_batch_end(
                            global_step,
                            logs,
                            force_reload=False,
                            trial_dir=str(PROJECT_ROOT),  # Use project root for configs
                            control_input_names=["loss", "learning_rate"],
                        )

                        # Check if controller returned updates
                        num_blocks = len(block_names.numpy())
                        if num_blocks > 0:
                            results["controller_triggers"] += 1
                            block_names_list = [b.decode() for b in block_names.numpy()]
                            evolution_times_list = evolution_times.numpy().tolist()

                            results["blocks_updated"].extend(block_names_list)
                            results["evolution_times_seen"].extend(evolution_times_list)

                            if verbose:
                                logger.info(
                                    f"[Step {global_step}] Controller updated {num_blocks} blocks: "
                                    f"{block_names_list[:3]}... times={evolution_times_list[:3]}"
                                )
                        elif verbose and global_step % controller_frequency == 0:
                            logger.debug(f"[Step {global_step}] Controller returned no updates")

                    except Exception as e:
                        results["errors"].append(f"Step {global_step}: {str(e)}")
                        if verbose:
                            logger.warning(f"Meta-controller error at step {global_step}: {e}")

                    # Progress logging
                    if step % 10 == 0 or step == steps_per_epoch - 1:
                        logger.info(
                            f"Epoch {epoch+1}/{epochs}, Step {step+1}/{steps_per_epoch}: "
                            f"loss={current_loss:.6f}, grad_norm={float(gradient_norm.numpy()):.4f}"
                        )

                    global_step += 1

                except StopIteration:
                    logger.warning("Dataset exhausted early")
                    break
                except Exception as e:
                    results["errors"].append(f"Training error at step {global_step}: {str(e)}")
                    logger.error(f"Training error: {e}")
                    break

            if batch_losses:
                epoch_loss = sum(batch_losses) / len(batch_losses)
                epoch_losses.append(epoch_loss)
                logger.info(f"Epoch {epoch+1} complete: avg_loss={epoch_loss:.6f}")

        # Final results
        results["training_completed"] = True
        results["final_loss"] = epoch_losses[-1] if epoch_losses else float("inf")
        results["blocks_updated"] = list(set(results["blocks_updated"]))  # Unique blocks

    except Exception as e:
        results["errors"].append(f"Training loop error: {str(e)}")
        logger.error(f"Training loop failed: {e}", exc_info=True)

    return results


def print_validation_summary(
    config_results: dict,
    native_ops_ok: bool,
    callback_ok: bool,
    training_results: dict,
) -> bool:
    """Print a summary of all validation results.

    Args:
        config_results: Results from validate_config_files_exist()
        native_ops_ok: Result from validate_native_ops_available()
        callback_ok: Result from validate_callback_instantiation()
        training_results: Results from run_training_with_meta_controller()

    Returns:
        True if all validations passed, False otherwise
    """
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    all_passed = True

    # Config file checks
    pid_ok = config_results.get("pid_conf_exists", False) and config_results.get(
        "pid_conf_readable", False
    )
    ss_ok = config_results.get("state_space_conf_exists", False) and config_results.get(
        "state_space_conf_readable", False
    )

    logger.info(f"  pid.conf exists and readable: {'✓' if pid_ok else '✗'}")
    if pid_ok:
        logger.info(f"    Sections: {config_results.get('pid_conf_sections', [])}")

    logger.info(f"  state_space.conf exists and readable: {'✓' if ss_ok else '✗'}")
    if ss_ok:
        logger.info(f"    Sections: {config_results.get('state_space_conf_sections', [])}")

    logger.info(f"  Native ops available: {'✓' if native_ops_ok else '✗'}")
    logger.info(f"  Meta-controller callback instantiable: {'✓' if callback_ok else '✗'}")

    # Training results
    training_ok = training_results.get("training_completed", False)
    logger.info(f"  Training loop completed: {'✓' if training_ok else '✗'}")

    if training_ok:
        triggers = training_results.get("controller_triggers", 0)
        blocks = len(training_results.get("blocks_updated", []))
        times = len(training_results.get("evolution_times_seen", []))

        controller_ok = triggers > 0
        logger.info(
            f"  Meta-controller triggered: {'✓' if controller_ok else '✗'} ({triggers} times)"
        )
        logger.info(f"  Blocks updated: {blocks} unique")
        logger.info(f"  Evolution times generated: {times}")
        logger.info(f"  Final loss: {training_results.get('final_loss', 'N/A'):.6f}")

        if not controller_ok:
            all_passed = False
    else:
        all_passed = False

    # Errors
    errors = training_results.get("errors", [])
    if errors:
        logger.warning(f"  Errors encountered: {len(errors)}")
        for err in errors[:5]:  # Show first 5
            logger.warning(f"    - {err}")
        if len(errors) > 5:
            logger.warning(f"    ... and {len(errors) - 5} more")

    # Overall verdict
    overall = all([pid_ok, ss_ok, native_ops_ok, callback_ok, training_ok])
    all_passed = all_passed and overall

    logger.info("\n" + "-" * 60)
    if all_passed:
        logger.info("✓ ALL VALIDATIONS PASSED")
        logger.info("pid.conf and state_space.conf are functioning correctly!")
    else:
        logger.error("✗ SOME VALIDATIONS FAILED")
        logger.error("Review the errors above to diagnose issues.")
    logger.info("-" * 60)

    return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate PID and State-Space Configuration Functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation (quick)
    python scripts/validate_control_configs.py

    # Extended training validation
    python scripts/validate_control_configs.py --epochs 5 --steps 100

    # Verbose output (show controller updates)
    python scripts/validate_control_configs.py --verbose

    # Custom controller frequency
    python scripts/validate_control_configs.py --frequency 3
        """,
    )

    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs (default: 2)"
    )
    parser.add_argument("--steps", type=int, default=20, help="Steps per epoch (default: 20)")
    parser.add_argument(
        "--frequency",
        type=int,
        default=5,
        help="Meta-controller trigger frequency in batches (default: 5)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Model hidden dimension (default: 256)"
    )
    parser.add_argument(
        "--reasoning-blocks", type=int, default=4, help="Number of reasoning blocks (default: 4)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose controller output")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training loop (only validate config files and imports)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("HighNoon Control Configuration Validator")
    logger.info("=" * 60)
    logger.info(f"Project root: {PROJECT_ROOT}")

    # Step 1: Validate config files exist
    logger.info("\n[1/4] Validating configuration files...")
    config_results = validate_config_files_exist()

    # Step 2: Validate native ops
    logger.info("\n[2/4] Validating native ops availability...")
    native_ops_ok = validate_native_ops_available()

    # Step 3: Validate callback instantiation
    logger.info("\n[3/4] Validating meta-controller callback...")
    callback_ok = validate_callback_instantiation()

    # Step 4: Run training with meta-controller
    training_results = {"training_completed": False, "controller_triggers": 0, "errors": []}

    if not args.skip_training:
        if native_ops_ok and callback_ok:
            logger.info("\n[4/4] Running training with meta-controller...")
            training_results = run_training_with_meta_controller(
                epochs=args.epochs,
                steps_per_epoch=args.steps,
                controller_frequency=args.frequency,
                hidden_dim=args.hidden_dim,
                num_reasoning_blocks=args.reasoning_blocks,
                verbose=args.verbose,
            )
        else:
            logger.warning("\n[4/4] Skipping training (prerequisites not met)")
            training_results["errors"].append("Prerequisites not met for training")
    else:
        logger.info("\n[4/4] Training skipped (--skip-training flag)")

    # Print summary
    all_passed = print_validation_summary(
        config_results,
        native_ops_ok,
        callback_ok,
        training_results,
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
