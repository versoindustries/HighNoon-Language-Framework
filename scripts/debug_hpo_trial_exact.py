#!/usr/bin/env python3
"""Debug HPO Trial Runner - EXACT REPLICA of WebUI HPO Execution.

This script executes an HPO trial EXACTLY as the WebUI does:
1. Uses the same config structure as WebUI creates
2. Calls hpo_trial_runner.train_trial() directly
3. Produces identical logging and results

This ensures that if this script works, the WebUI will work.

Usage:
    python scripts/debug_hpo_trial_exact.py
    python scripts/debug_hpo_trial_exact.py --optimizer sympflowqng
    python scripts/debug_hpo_trial_exact.py --learning-rate 0.0001
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure verbose logging
logging.basicConfig(
    level=logging.DEBUG,  # Maximum verbosity
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_webui_config(
    optimizer: str = "sympflowqng",
    learning_rate: float = 0.0001,
    batch_size: int = 16,
    hidden_dim: int = 512,
    num_reasoning_blocks: int = 8,
    num_moe_experts: int = 8,
    sequence_length: int = 512,
    hf_dataset_name: str = "HuggingFaceFW/fineweb",
    epochs: int = 3,
    steps_per_epoch: int = 50,
) -> dict:
    """Create config EXACTLY as WebUI does in app.py.

    This replicates the model_config dict created in the /api/hpo/sweep/start endpoint.
    """
    config = {
        "sweep_id": "debug_exact",
        # Core architecture - EXACTLY as WebUI creates
        # vocab_size is NOT included - determined by tokenizer
        "hidden_dim": hidden_dim,
        "num_reasoning_blocks": num_reasoning_blocks,
        "num_moe_experts": num_moe_experts,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "param_budget": 1_000_000_000,
        # Mamba2 SSM parameters
        "mamba_state_dim": 64,
        "mamba_conv_dim": 4,
        "mamba_expand": 2,
        # WLAM parameters
        "wlam_num_heads": 8,
        "wlam_kernel_size": 3,
        "wlam_num_landmarks": 32,
        # MoE parameters
        "moe_top_k": 2,
        "moe_capacity_factor": 1.25,
        # FFN and TT
        "ff_expansion": 4,
        "tt_rank_middle": 16,
        # Quantum/superposition
        "superposition_dim": 2,
        "hamiltonian_hidden_dim": 256,
        # Regularization
        "dropout_rate": 0.1,
        "weight_decay": 0.01,
        # Dataset configuration
        "hf_dataset_name": hf_dataset_name,
        "curriculum_id": "debug",
        # Quantum Enhancement Parameters - all enabled like WebUI
        "use_quantum_embedding": True,
        "use_floquet_position": True,
        "use_quantum_feature_maps": True,
        "use_unitary_expert": True,
        "neumann_cayley_terms": 6,
        "use_quantum_norm": True,
        "use_superposition_bpe": True,
        "use_grover_qsg": True,
        "qsg_quality_threshold": 0.7,
        "use_quantum_lm_head": True,
        "use_unitary_residual": True,
        "unitary_residual_init_angle": 0.7854,
        "use_quantum_state_bus": True,
        # Trial info
        "_trial_id": "debug_exact_trial",
        "trial_id": "debug_exact_trial",
        "budget": epochs,
        "max_grad_norm": 1.0,
        # Training params (set by sweep_executor)
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
    }
    return config


def main():
    parser = argparse.ArgumentParser(description="Debug HPO Trial - Exact WebUI Replica")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sympflowqng",
        help="Optimizer to use (default: sympflowqng)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs (default: 3)",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=50,
        help="Steps per epoch (default: 50)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb",
        help="HuggingFace dataset name (default: HuggingFaceFW/fineweb)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("DEBUG HPO TRIAL - EXACT WEBUI REPLICA")
    print("=" * 80)
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Steps/Epoch: {args.steps_per_epoch}")
    print(f"Dataset: {args.dataset}")
    print("=" * 80)

    # Create config exactly as WebUI does
    config = create_webui_config(
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        hf_dataset_name=args.dataset,
    )

    # Save config to temp file (like sweep_executor does)
    config_dir = PROJECT_ROOT / "artifacts" / "hpo_trials" / "debug_exact"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to: {config_file}")

    # Set environment variables like sweep_executor does
    os.environ["HPO_SWEEP_ID"] = "debug_exact"
    os.environ["HPO_TRIAL_ID"] = "debug_exact_trial"
    os.environ["HPO_TRIAL_DIR"] = str(config_dir)
    os.environ["HPO_ROOT"] = str(config_dir.parent)
    os.environ["HPO_API_HOST"] = "127.0.0.1:8000"

    print("\nCalling hpo_trial_runner.train_trial() - EXACTLY as WebUI does...")
    print("-" * 80)

    # Import and call train_trial EXACTLY as hpo_trial_runner.__main__ does
    from highnoon.services.hpo_trial_runner import train_trial

    try:
        loss = train_trial(
            trial_id="debug_exact_trial",
            config_path=str(config_file),
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
        )
        print("-" * 80)
        print(f"\n{'='*80}")
        print("TRIAL COMPLETED SUCCESSFULLY!")
        print(f"Final Loss: {loss}")
        print(f"{'='*80}")
    except Exception as e:
        print("-" * 80)
        print(f"\n{'='*80}")
        print("TRIAL FAILED!")
        print(f"Error: {e}")
        print(f"{'='*80}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
