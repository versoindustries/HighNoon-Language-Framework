#!/usr/bin/env python3
"""
HPO Debug Script - Test training pipeline without WebUI/backend overhead.

Usage:
    python scripts/debug_hpo_training.py --help
    python scripts/debug_hpo_training.py --use-samples  # Fast test with built-in texts
    python scripts/debug_hpo_training.py --dataset databricks/dolly-15k --max-samples 100

This script replicates HPO trial training for debugging purposes.
"""

import argparse
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from highnoon.data.loaders import load_training_dataset
from highnoon.services.hpo_trial_runner import build_hsmn_model, create_optimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_debug_training(
    use_samples: bool = True,
    dataset: str | None = None,
    max_samples: int = 100,
    vocab_size: int = 50000,
    hidden_dim: int = 256,
    num_reasoning_blocks: int = 4,
    batch_size: int = 4,
    sequence_length: int = 128,
    learning_rate: float = 1e-4,
    optimizer: str = "adamw",
    epochs: int = 2,
    steps_per_epoch: int = 20,
):
    """Run debug training with specified configuration."""

    logger.info("=" * 60)
    logger.info("HighNoon HPO Debug Training Script")
    logger.info("=" * 60)

    # Build config
    config = {
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "num_reasoning_blocks": num_reasoning_blocks,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "num_heads": 4,
        "ff_dim": hidden_dim * 4,
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "total_training_steps": epochs * steps_per_epoch,
    }

    logger.info("Configuration:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    # Determine dataset
    hf_dataset_name = None
    if not use_samples and dataset:
        hf_dataset_name = dataset
        logger.info(f"Using HuggingFace dataset: {dataset} (max_samples={max_samples})")
    else:
        logger.info("Using built-in sample texts (fast mode)")

    # Load dataset
    logger.info("Loading dataset...")
    start_time = time.time()

    train_dataset, tokenizer, merger = load_training_dataset(
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        hf_dataset_name=hf_dataset_name,
        max_samples=max_samples,
    )

    load_time = time.time() - start_time
    logger.info(f"Dataset loaded in {load_time:.2f}s")
    if merger is not None:
        logger.info(f"SuperwordMerger: {merger.superword_count} superwords learned")

    # Build model
    logger.info("Building HSMN model...")
    start_time = time.time()

    model = build_hsmn_model(config, tokenizer.vocab_size, hidden_dim_override=hidden_dim)

    build_time = time.time() - start_time
    logger.info(f"Model built in {build_time:.2f}s")
    logger.info(f"Model parameters: {model.count_params():,}")

    # Create optimizer
    optimizer_obj = create_optimizer(config, model=model)

    # Compile model
    model.compile(
        optimizer=optimizer_obj,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )

    # Training loop
    logger.info(f"Starting training: {epochs} epochs, {steps_per_epoch} steps/epoch")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_losses = []
        dataset_iter = iter(train_dataset)

        for step in range(steps_per_epoch):
            try:
                inputs, labels = next(dataset_iter)

                with tf.GradientTape() as tape:
                    predictions = model(inputs, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(
                        labels, predictions, from_logits=True
                    )
                    loss = tf.reduce_mean(loss)

                gradients = tape.gradient(loss, model.trainable_variables)
                gradient_norm = tf.linalg.global_norm(gradients)
                optimizer_obj.apply_gradients(zip(gradients, model.trainable_variables))

                current_loss = float(loss.numpy())
                epoch_losses.append(current_loss)

                if step % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}, Step {step+1}/{steps_per_epoch}: "
                        f"loss={current_loss:.6f}, grad_norm={gradient_norm:.4f}"
                    )

                global_step += 1

            except StopIteration:
                logger.warning(f"Dataset exhausted at step {step}")
                break
            except Exception as e:
                logger.error(f"Training error at step {step}: {e}")
                raise

        epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")
        logger.info(f"Epoch {epoch+1} complete: avg_loss={epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss

    logger.info("=" * 60)
    logger.info(f"Training complete! Best loss: {best_loss:.6f}")
    logger.info("=" * 60)

    return best_loss


def main():
    parser = argparse.ArgumentParser(
        description="HPO Debug Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast test with built-in sample texts (no download)
  python scripts/debug_hpo_training.py --use-samples

  # Test with small HuggingFace dataset
  python scripts/debug_hpo_training.py --dataset databricks/dolly-15k --max-samples 50

  # Test with larger model
  python scripts/debug_hpo_training.py --use-samples --hidden-dim 512 --reasoning-blocks 8
        """,
    )

    # Dataset options
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--use-samples",
        action="store_true",
        default=True,
        help="Use built-in sample texts (fast, no download)",
    )
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., 'databricks/dolly-15k')",
    )
    parser.add_argument(
        "--max-samples", type=int, default=100, help="Max samples from HuggingFace dataset"
    )

    # Model options
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--reasoning-blocks", type=int, default=4)

    # Training options
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sophiag", "grover"]
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=20)

    args = parser.parse_args()

    try:
        run_debug_training(
            use_samples=args.dataset is None,
            dataset=args.dataset,
            max_samples=args.max_samples,
            vocab_size=args.vocab_size,
            hidden_dim=args.hidden_dim,
            num_reasoning_blocks=args.reasoning_blocks,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
        )
    except Exception as e:
        logger.error(f"Debug training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
