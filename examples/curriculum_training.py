#!/usr/bin/env python3
# examples/curriculum_training.py
# Copyright 2025 Verso Industries
#
# Curriculum training example for HighNoon Language Framework.

"""Curriculum Training Example.

Demonstrates training with curriculum learning stages.

Usage:
    python examples/curriculum_training.py

Requirements:
    - HighNoon Language Framework installed
    - TensorFlow 2.x
"""

import highnoon as hn


def training_callback(event: dict):
    """Callback function invoked during training."""
    event_type = event.get("event")

    if event_type == "stage_complete":
        stage = event.get("stage")
        metrics = event.get("metrics", {})
        print(
            f"    [CALLBACK] Stage '{stage}' complete: "
            f"final_loss={metrics.get('final_loss', 'N/A')}"
        )
    elif event_type == "epoch_end":
        stage = event.get("stage")
        epoch = event.get("epoch")
        loss = event.get("loss")
        # Print every 2 epochs to reduce noise
        if epoch % 2 == 0:
            print(f"    [CALLBACK] {stage} - Epoch {epoch}: loss={loss:.4f}")


def main():
    """Run curriculum training example."""
    print("HighNoon Language Framework - Curriculum Training Example")
    print(f"Version: {hn.__version__}")
    print(f"Edition: {hn.__edition__}")
    print("-" * 50)

    # Create model
    print("\n[1] Creating model from 'highnoon-small' preset...")
    model = hn.create_model("highnoon-small")
    print(f"    Model created: {model.num_blocks} reasoning blocks")

    # Create trainer
    print("\n[2] Initializing Trainer...")
    trainer = hn.Trainer(
        model,
        learning_rate=1e-4,
        batch_size=8,
        max_seq_length=2048,
    )

    # Define curriculum stages
    print("\n[3] Defining curriculum stages...")

    trainer.add_curriculum_stage(
        "code_foundation",
        datasets=["the_stack_v2"],
        epochs=3,
    )
    print("    Added: code_foundation (3 epochs)")

    trainer.add_curriculum_stage(
        "code_instruction",
        datasets=["commitpackft"],
        epochs=3,
    )
    print("    Added: code_instruction (3 epochs)")

    trainer.add_curriculum_stage(
        "tool_use",
        datasets=["toolbench"],
        epochs=2,
        weight=1.5,  # Higher weight for tool learning
    )
    print("    Added: tool_use (2 epochs, weight=1.5)")

    # Add training callback
    trainer.add_callback(training_callback)

    # Train with curriculum
    print("\n[4] Starting curriculum training...")
    print("    (This is a simulation - actual training requires datasets)")

    summary = trainer.train(
        epochs_per_stage=2,  # Override: fewer epochs for demo
        save_checkpoints=False,  # Disable for demo
    )

    # Print training summary
    print("\n[5] Training Summary:")
    print(f"    Total steps: {summary['total_steps']}")
    print(f"    Stages completed: {len(summary['stages_completed'])}")
    for stage_info in summary["stages_completed"]:
        print(
            f"      - {stage_info['name']}: "
            f"epochs={stage_info['epochs']}, "
            f"final_loss={stage_info['final_loss']:.4f}"
        )

    print("\n" + "-" * 50)
    print("Curriculum training example complete!")
    print("\nNote: This example simulates training. For actual training,")
    print("      provide real datasets and enable checkpointing.")


if __name__ == "__main__":
    main()
