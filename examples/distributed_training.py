#!/usr/bin/env python3
# examples/distributed_training.py
# Copyright 2025 Verso Industries
#
# Complete example of multi-node CPU distributed training with HighNoon.
#
# Usage:
#   1. Set TF_CONFIG on each node (see docs/distributed_training.md)
#   2. Run: python examples/distributed_training.py
#
# Example TF_CONFIG for worker 0:
#   export TF_CONFIG='{"cluster":{"worker":["node1:12345","node2:12345"]},"task":{"type":"worker","index":0}}'

"""Multi-Node Distributed Training Example.

This script demonstrates how to train a HighNoon model across multiple
CPU nodes using TensorFlow's MultiWorkerMirroredStrategy.

Prerequisites:
    - TF_CONFIG environment variable set on each node
    - Shared filesystem for checkpoints (/shared/checkpoints)
    - HighNoon installed on all nodes

See Also:
    - docs/distributed_training.md
    - docs/cluster_setup.md
"""

import logging
import os
import sys

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def main():
    """Run distributed training."""
    # Import after logging setup
    import highnoon as hn
    from highnoon.training.distributed import (
        create_cpu_strategy,
        get_worker_info,
        setup_cpu_threading,
        validate_tf_config,
    )

    # ============================================================
    # 1. Validate Configuration
    # ============================================================
    log.info("=" * 60)
    log.info("HighNoon Distributed Training Example")
    log.info("=" * 60)

    valid, msg = validate_tf_config()
    if not valid:
        log.error(f"Invalid TF_CONFIG: {msg}")
        sys.exit(1)
    log.info(f"TF_CONFIG: {msg}")

    worker_info = get_worker_info()
    log.info(f"Worker info: {worker_info}")

    # ============================================================
    # 2. Configure CPU Threading
    # ============================================================
    setup_cpu_threading(
        intra_op_threads=16,  # Adjust based on your CPU cores
        inter_op_threads=2,
    )

    # ============================================================
    # 3. Create Distributed Strategy
    # ============================================================
    strategy = create_cpu_strategy(
        strategy_type="auto",  # Auto-detect from TF_CONFIG
        communication="ring",  # Ring all-reduce (CPU optimized)
    )

    log.info(f"Strategy: {type(strategy).__name__}")
    log.info(f"Number of replicas: {strategy.num_replicas_in_sync}")

    # ============================================================
    # 4. Create Model Inside Strategy Scope
    # ============================================================
    with strategy.scope():
        # Create model - adjust size based on your resources
        model = hn.create_model(
            size="3b",  # Options: "1b", "3b", "7b", "13b", "20b"
            config=hn.Config(
                model=hn.ModelConfig(
                    vocab_size=32000,
                    embedding_dim=768,
                    num_reasoning_blocks=8,
                    num_moe_experts=8,
                    max_seq_length=2048,
                )
            ),
        )

        log.info(f"Model created: {model.name}")

        # Scale learning rate with worker count
        base_lr = 1e-4
        scaled_lr = base_lr * strategy.num_replicas_in_sync
        log.info(f"Learning rate: {base_lr} -> {scaled_lr} (scaled)")

        # Create trainer
        trainer = hn.Trainer(
            model,
            learning_rate=scaled_lr,
            batch_size=8,  # Per-replica batch size
            max_seq_length=2048,
            gradient_accumulation_steps=4,
        )

    # ============================================================
    # 5. Add Curriculum Stages
    # ============================================================
    # Replace with your actual datasets
    trainer.add_curriculum_stage(
        name="foundation",
        datasets=["your_foundation_dataset"],
        epochs=5,
        weight=1.0,
    )

    trainer.add_curriculum_stage(
        name="instruction",
        datasets=["your_instruction_dataset"],
        epochs=3,
        weight=1.5,
    )

    # ============================================================
    # 6. Configure Checkpointing
    # ============================================================
    # IMPORTANT: Must be a shared filesystem accessible by all nodes
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "/shared/checkpoints/highnoon_distributed")

    # Only chief worker should create directory
    if worker_info["is_chief"]:
        os.makedirs(checkpoint_dir, exist_ok=True)
        log.info(f"Checkpoint directory: {checkpoint_dir}")

    # ============================================================
    # 7. Train
    # ============================================================
    log.info("Starting distributed training...")

    try:
        trainer.train(
            epochs_per_stage=5,
            save_checkpoints=True,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=500,
            log_interval=50,
            eval_interval=200,
            resume_from=None,  # Set to checkpoint path to resume
        )
    except KeyboardInterrupt:
        log.warning("Training interrupted by user")
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise

    log.info("Training complete!")

    # ============================================================
    # 8. Save Final Model (Chief Only)
    # ============================================================
    if worker_info["is_chief"]:
        final_model_path = os.path.join(checkpoint_dir, "final_model")
        trainer.save(final_model_path)
        log.info(f"Model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
