#!/usr/bin/env python3
"""HPO Sweep Debug Script - Mimics React WebUI HPO Sweep Page.

This script replicates the exact workflow from the React WebUI HPO page
for debugging the HPO sweep and training loop setup, now using the unified
TrainingEngine and HPOTrainingConfig infrastructure.

Configuration (matching WebUI):
- Parameter Budget: 100M (100,000,000)
- Vocabulary Size: 128K (128,000)
- Curriculum: Verso Baseline
- Optimizer: SympFlowQNG (default)

Usage:
    python scripts/debug_hpo_sweep_webui.py
    python scripts/debug_hpo_sweep_webui.py --trials 3 --epochs 2
    python scripts/debug_hpo_sweep_webui.py --param-budget 250000000 --vocab-size 64000
    python scripts/debug_hpo_sweep_webui.py --dry-run  # Config only, no training
    python scripts/debug_hpo_sweep_webui.py --test-serialization  # Test save/load
"""

import argparse
import gc
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Imports (after path setup)
# =============================================================================

import tensorflow as tf  # noqa: E402

# Import global config
# Serialization API
from highnoon.serialization import load_model, save_model  # noqa: E402
from highnoon.services.hpo_training_bridge import HPOTrainingConfig  # noqa: E402

# Vocabulary Controller for quantum tokenization pipeline
from highnoon.tokenization.vocab_controller import (  # noqa: E402
    IntelligentVocabController,
    VocabControllerConfig,
)

# TrainingEngine and HPO Bridge (new unified infrastructure)
from highnoon.training.training_engine import (  # noqa: E402
    TrainingCallback,
    TrainingEngine,
    TrainingResult,
)

# =============================================================================
# WebUI HPO Configuration (from HPO.tsx)
# =============================================================================

LITE_TIER_LIMITS = {
    "maxVocabSize": 256_000,
    "maxContextWindow": 5_000_000,
    "maxTrials": 20,
    "allowedOptimizers": ["sophiag", "qiao", "grover", "sympflow", "sympflowqng"],
}

OPTIMIZER_LR_RANGES = {
    "adam": (1e-5, 1e-3),
    "adamw": (1e-5, 3e-4),
    "sophiag": (1e-5, 1e-4),
    "qiao": (1e-5, 2e-4),
    "grover": (1e-5, 5e-4),
    "sympflow": (1e-5, 3e-4),
    "sympflowqng": (1e-5, 2e-4),
}

# Curriculum presets are loaded from artifacts/curriculum_presets.json
# The expanded Verso Baseline includes 14 datasets across 5 training stages


class DebugCallback(TrainingCallback):
    """Callback for detailed training logging."""

    def __init__(self, log_every: int = 5):
        self.log_every = log_every
        self.step_count = 0

    def on_batch_end(self, step: int, result) -> bool:
        """Called after each training step. Return False to stop training."""
        self.step_count += 1
        if self.step_count % self.log_every == 0:
            logger.info(
                f"    Step {self.step_count}: loss={result.loss:.4f}, "
                f"grad_norm={result.gradient_norm:.4f}, "
                f"lr={result.effective_learning_rate:.2e}"
            )
        return True

    def on_epoch_end(self, epoch: int, result) -> bool:
        """Called after each epoch. Return False to stop training."""
        logger.info(
            f"  Epoch {epoch + 1} complete: "
            f"mean_loss={result.mean_loss:.4f}, "
            f"steps={result.steps_completed}"
        )
        return True


# =============================================================================
# HPO Configuration and Model Building
# =============================================================================


def create_webui_sweep_config(
    param_budget: int = 250_000_000,  # 250M to match WebUI
    vocab_size: int = 128_000,
    context_window: int = 1_000_000,  # 1M context to match WebUI
    optimizer: str = "sympflowqng",
    learning_rate: float = 1e-4,
) -> HPOTrainingConfig:
    """Create HPOTrainingConfig matching WebUI parameters.

    Args:
        param_budget: Maximum model parameters
        vocab_size: Vocabulary size
        context_window: Context window size
        optimizer: Optimizer name
        learning_rate: Learning rate

    Returns:
        HPOTrainingConfig instance
    """
    # Validate against Lite limits
    if vocab_size > LITE_TIER_LIMITS["maxVocabSize"]:
        raise ValueError(f"vocab_size {vocab_size} exceeds Lite limit")
    if optimizer not in LITE_TIER_LIMITS["allowedOptimizers"]:
        raise ValueError(f"optimizer '{optimizer}' not in allowed list")

    return HPOTrainingConfig(
        sweep_id="debug-sweep",
        vocab_size=vocab_size,
        context_window=context_window,
        param_budget=param_budget,
        embedding_dim=512,
        num_reasoning_blocks=6,
        num_moe_experts=4,
        learning_rate=learning_rate,
        batch_size=16,
        optimizer=optimizer,
        epochs=10,
        loss_function="sparse_categorical_crossentropy",
        feature_flags={
            # Training Enhancement Flags (Part 10 cleanup.md)
            "use_tensor_galore": True,
            "galore_rank": 32,
            "galore_vqc_aware": True,
            "use_quantum_natural_gradient": True,
            "qng_damping": 1e-4,
            "use_sympflow": True,
            "barren_plateau_monitor": True,
            "barren_plateau_threshold": 1e-6,
            "use_neural_zne": True,
            "use_qhpm_crystallization": True,
            "use_entropy_regularization": True,
            "use_meta_controller": True,
            # Synergy Flags (S1-S23)
            "s1_unified_qssm_gating": True,
            "s2_coconut_qmamba_amplitudes": True,
            "s11_alphaqubit_decoder": True,
            "s18_cayley_dense": True,
            "s19_coconut_crystallize": True,
            "s20_galore_bp_aware": True,
            "s21_qalrc_quls_entropy": True,
            "s22_bp_qalrc_escape": True,
            "s23_zne_lr_feedback": True,
            # Architecture Flags
            "use_gradient_checkpointing": True,
            "use_unified_quantum_bus": True,
            "use_state_bus": True,
        },
    )


def estimate_model_params(config: HPOTrainingConfig) -> int:
    """Estimate total model parameters from architecture config."""
    vocab_size = config.vocab_size
    embedding_dim = config.embedding_dim
    num_blocks = config.num_reasoning_blocks
    num_experts = config.num_moe_experts

    embed_params = vocab_size * embedding_dim
    ff_dim = embedding_dim * 4
    d_inner = embedding_dim * 2

    spatial_params = 4 * embedding_dim * d_inner + 64 * embedding_dim * 2
    timecrystal_params = 8 * embedding_dim * embedding_dim
    latent_params = 3 * embedding_dim * ff_dim
    wlam_params = 4 * embedding_dim * embedding_dim
    moe_params = num_experts * (2 * embedding_dim * ff_dim) + embedding_dim * num_experts

    blocks_per_pattern = 6
    num_patterns = (num_blocks + blocks_per_pattern - 1) // blocks_per_pattern
    params_per_pattern = (
        spatial_params * 2 + timecrystal_params + latent_params + wlam_params + moe_params
    )
    output_params = embedding_dim * vocab_size

    return int(embed_params + params_per_pattern * num_patterns + output_params)


def sample_hyperparameters(
    base_config: HPOTrainingConfig,
    trial_id: int,
) -> HPOTrainingConfig:
    """Sample hyperparameters within budget constraint.

    Args:
        base_config: Base configuration with budget constraints
        trial_id: Trial ID for deterministic sampling

    Returns:
        HPOTrainingConfig with sampled parameters
    """
    import random

    random.seed(trial_id)

    # Get optimizer-specific LR range
    lr_range = OPTIMIZER_LR_RANGES.get(base_config.optimizer, (1e-5, 3e-4))
    lr_log_min = math.log10(lr_range[0])
    lr_log_max = math.log10(lr_range[1])

    # Sample learning rate
    learning_rate = 10 ** random.uniform(lr_log_min, lr_log_max)
    batch_size = random.choice([8, 16, 32])

    # Try different architectures within budget
    embedding_options = [256, 512, 768]
    block_options = [4, 6, 8]

    best_dim, best_blocks = 256, 4
    best_params = 0

    for dim in embedding_options:
        for blocks in block_options:
            test_config = HPOTrainingConfig(
                vocab_size=base_config.vocab_size,
                embedding_dim=dim,
                num_reasoning_blocks=blocks,
                num_moe_experts=base_config.num_moe_experts,
            )
            estimated = estimate_model_params(test_config)
            if estimated <= base_config.param_budget and estimated > best_params:
                best_dim = dim
                best_blocks = blocks
                best_params = estimated

    return HPOTrainingConfig(
        sweep_id=f"debug-trial-{trial_id}",
        vocab_size=base_config.vocab_size,
        context_window=base_config.context_window,
        param_budget=base_config.param_budget,
        embedding_dim=best_dim,
        num_reasoning_blocks=best_blocks,
        num_moe_experts=base_config.num_moe_experts,
        learning_rate=learning_rate,
        batch_size=batch_size,
        optimizer=base_config.optimizer,
        epochs=base_config.epochs,
        loss_function=base_config.loss_function,
        feature_flags=base_config.feature_flags,
    )


# =============================================================================
# Model Building and Training
# =============================================================================


def build_model(config: HPOTrainingConfig) -> tf.keras.Model:
    """Build HSMN model from HPOTrainingConfig using quantum tokenization pipeline.

    Uses IntelligentVocabController to determine effective vocab size instead of
    the WebUI-configured target. This ensures the embedding and output layers
    match the actual tokenizer vocabulary.
    """
    from highnoon.models.layers.hyperdimensional_layer import HyperdimensionalEmbedding
    from highnoon.models.reasoning.block_factory import QuantumLMHead
    from highnoon.models.reasoning.reasoning_module import ReasoningModule

    # Create vocab controller with auto-learning disabled for debug
    # (effective vocab = base vocab without corpus training)
    vocab_config = VocabControllerConfig(
        model_max_length=config.context_window,
    )
    vocab_controller = IntelligentVocabController(config=vocab_config)

    # Get effective vocab size (base vocab size without corpus training)
    effective_vocab = vocab_controller.effective_vocab_size

    logger.info(
        f"[VocabController] target_vocab={config.vocab_size}, "
        f"effective_vocab={effective_vocab}, base_vocab={vocab_controller.base_vocab_size}"
    )

    # Build model using effective vocabulary size
    hidden_dim = config.embedding_dim

    # Input layer
    input_layer = tf.keras.layers.Input(
        shape=(None,),
        dtype=tf.int32,
        name="token_ids",
    )

    # Use HyperdimensionalEmbedding for quantum-enhanced embedding
    use_hde = config.feature_flags.get("use_hyperdimensional_embedding", True)

    if use_hde:
        # hd_dim must be divisible by model_dim - compute dynamically
        hd_dim = hidden_dim * 8  # Ensures divisibility (e.g., 768*8=6144, 512*8=4096)
        x = HyperdimensionalEmbedding(
            vocab_size=effective_vocab,
            model_dim=hidden_dim,
            hd_dim=hd_dim,
            use_ctqw=True,
            ctqw_steps=3,
            name="hde_embedding",
        )(input_layer)
        logger.info(
            f"[Model] Using HyperdimensionalEmbedding: vocab={effective_vocab}, hd_dim={hd_dim}"
        )
    else:
        x = tf.keras.layers.Embedding(
            input_dim=effective_vocab,
            output_dim=hidden_dim,
            name="token_embedding",
        )(input_layer)
        logger.info(f"[Model] Using standard Embedding: vocab={effective_vocab}")

    # Reasoning module
    reasoning_module = ReasoningModule(
        num_layers=config.num_reasoning_blocks,
        embedding_dim=hidden_dim,
        num_heads=8,
        ff_dim=hidden_dim * 4,
        num_experts=config.num_moe_experts,
    )
    x = reasoning_module(x)

    # Use QuantumLMHead for VQC-based output
    use_quantum_lm_head = config.feature_flags.get("use_quantum_lm_head", True)

    if use_quantum_lm_head:
        output_layer = QuantumLMHead(
            vocab_size=effective_vocab,
            hidden_dim=hidden_dim,
            vqc_layers=2,
            vqc_qubits=8,
            name="quantum_lm_head",
        )(x)
        logger.info(f"[Model] Using QuantumLMHead: vocab={effective_vocab}, vqc_qubits=8")
    else:
        output_layer = tf.keras.layers.Dense(
            effective_vocab,
            name="lm_head",
        )(x)
        logger.info(f"[Model] Using standard Dense LM head: vocab={effective_vocab}")

    model = tf.keras.Model(
        inputs=input_layer,
        outputs=output_layer,
        name="HSMN_QuantumTokenizer",
    )

    # Store vocab controller on model for later access
    model._vocab_controller = vocab_controller

    return model


def create_optimizer(
    config: HPOTrainingConfig, model: tf.keras.Model
) -> tf.keras.optimizers.Optimizer:
    """Create optimizer from HPOTrainingConfig."""
    from highnoon.services.hpo_trial_runner import create_optimizer as hpo_create_optimizer

    return hpo_create_optimizer(config.to_optimizer_config(), model=model)


# =============================================================================
# Verso Baseline Curriculum - Real Dataset Loading
# =============================================================================

# Lightweight datasets from Verso Baseline for debug/testing
# These are small enough for quick iteration but represent production data
VERSO_BASELINE_DEBUG_DATASETS = [
    # Stage 2: Conversational (lightweight)
    "databricks/databricks-dolly-15k",
    # Stage 4: Math (small)
    "gsm8k",
    # Stage 5: Reasoning (curated)
    "garage-bAInd/Open-Platypus",
]

# Full Verso Baseline curriculum (from artifacts/curriculum_presets.json)
VERSO_BASELINE_FULL_DATASETS = [
    # Stage 1: Foundation (subset - these are massive)
    "HuggingFaceTB/cosmopedia",
    # Stage 2: Conversational
    "databricks/databricks-dolly-15k",
    "OpenAssistant/oasst1",
    "tatsu-lab/alpaca",
    # Stage 3: Code
    "sahil2801/CodeAlpaca-20k",
    # Stage 4: Math
    "gsm8k",
    "meta-math/MetaMathQA",
    # Stage 5: Reasoning
    "garage-bAInd/Open-Platypus",
    "Open-Orca/SlimOrca",
]


def create_verso_baseline_dataset(
    tokenizer,
    batch_size: int,
    seq_length: int = 128,
    num_samples: int = 2000,
    use_full_curriculum: bool = False,
    streaming: bool = True,
) -> tf.data.Dataset | None:
    """Create dataset from Verso Baseline curriculum using HuggingFace.

    Args:
        tokenizer: Tokenizer with encode method
        batch_size: Samples per batch
        seq_length: Maximum sequence length
        num_samples: Total samples to load across all datasets
        use_full_curriculum: If True, use full curriculum; else lightweight debug set
        streaming: If True, stream datasets to minimize memory usage

    Returns:
        TensorFlow dataset yielding (inputs, labels) tuples, or None if loading fails.
    """
    try:
        from datasets import (  # noqa: F401 - IterableDataset for type hint
            IterableDataset,
            load_dataset,
        )
    except ImportError:
        logger.warning("[VersoBaseline] HuggingFace datasets not installed")
        return None

    datasets_to_load = (
        VERSO_BASELINE_FULL_DATASETS if use_full_curriculum else VERSO_BASELINE_DEBUG_DATASETS
    )
    samples_per_dataset = num_samples // len(datasets_to_load)

    logger.info(
        f"[VersoBaseline] Loading {len(datasets_to_load)} datasets, "
        f"~{samples_per_dataset} samples each, streaming={streaming}"
    )

    all_texts = []

    for dataset_name in datasets_to_load:
        try:
            logger.info(f"  Loading: {dataset_name}...")

            # Load dataset with streaming for memory efficiency
            ds = load_dataset(
                dataset_name,
                split="train",
                streaming=streaming,
                trust_remote_code=True,
            )

            # Extract text from various dataset formats
            count = 0
            for example in ds:
                text = None

                # Try common text field names
                for field in ["text", "instruction", "question", "prompt", "input", "context"]:
                    if field in example and example[field]:
                        text = str(example[field])
                        # Also append response/output if available
                        for resp_field in ["response", "output", "answer", "completion"]:
                            if resp_field in example and example[resp_field]:
                                text += " " + str(example[resp_field])
                        break

                # Handle chat/conversation format
                if text is None and "messages" in example:
                    messages = example["messages"]
                    if isinstance(messages, list) and len(messages) > 0:
                        text = " ".join(
                            m.get("content", "") for m in messages if isinstance(m, dict)
                        )

                # Handle instruction-input-output format
                if text is None:
                    parts = []
                    for f in ["instruction", "input", "output"]:
                        if f in example and example[f]:
                            parts.append(str(example[f]))
                    if parts:
                        text = " ".join(parts)

                if text and len(text.strip()) > 20:  # Filter very short texts
                    all_texts.append(text.strip())
                    count += 1
                    if count >= samples_per_dataset:
                        break

            logger.info(f"    Loaded {count} samples from {dataset_name}")

        except Exception as e:
            logger.warning(f"  Failed to load {dataset_name}: {e}")
            continue

    if not all_texts:
        logger.warning("[VersoBaseline] No texts loaded from any dataset")
        return None

    logger.info(f"[VersoBaseline] Total samples loaded: {len(all_texts)}")

    # Tokenize texts
    def tokenize_text(text: str) -> tuple[list[int], list[int]]:
        """Tokenize text and create input/label pairs for causal LM."""
        try:
            # Use tokenizer's encode method
            if hasattr(tokenizer, "encode"):
                tokens = tokenizer.encode(text)
            elif hasattr(tokenizer, "tokenize"):
                tokens = tokenizer.tokenize(text)
            else:
                # Fallback: simple byte encoding
                tokens = list(text.encode("utf-8")[:seq_length])

            # Ensure we have enough tokens
            if len(tokens) < 2:
                return None, None

            # Truncate or pad to seq_length
            if len(tokens) > seq_length:
                tokens = tokens[:seq_length]
            else:
                tokens = tokens + [0] * (seq_length - len(tokens))

            # Create causal LM pairs: input[:-1] -> labels[1:]
            input_ids = tokens[:-1] + [0]  # Shift left, pad
            labels = tokens[1:] + [0]  # Shift right, pad

            return input_ids, labels

        except Exception:
            return None, None

    # Create tokenized pairs
    tokenized_inputs = []
    tokenized_labels = []

    for text in all_texts:
        input_ids, labels = tokenize_text(text)
        if input_ids is not None:
            tokenized_inputs.append(input_ids)
            tokenized_labels.append(labels)

    if not tokenized_inputs:
        logger.warning("[VersoBaseline] Tokenization produced no valid samples")
        return None

    logger.info(f"[VersoBaseline] Tokenized {len(tokenized_inputs)} samples, seq_len={seq_length}")

    # Convert to TensorFlow dataset
    import numpy as np

    inputs_array = np.array(tokenized_inputs, dtype=np.int32)
    labels_array = np.array(tokenized_labels, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((inputs_array, labels_array))
    dataset = dataset.shuffle(buffer_size=min(10000, len(tokenized_inputs)))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_dummy_dataset(
    vocab_size: int,
    batch_size: int,
    seq_length: int = 128,
    num_batches: int = 100,
    effective_vocab_size: int | None = None,
):
    """Create dummy dataset for debugging (no HuggingFace download).

    Args:
        vocab_size: Target vocab size (used if effective_vocab_size not provided)
        batch_size: Samples per batch
        seq_length: Sequence length
        num_batches: Number of batches to generate
        effective_vocab_size: Actual vocab size to use for token generation.
            This should be the tokenizer's effective vocab size.

    Returns:
        TensorFlow dataset yielding (inputs, labels) tuples.
    """
    # Use effective vocab size if provided (fixes the vocab mismatch issue)
    actual_vocab = effective_vocab_size if effective_vocab_size else vocab_size

    logger.info(
        f"[DummyDataset] Creating with vocab={actual_vocab} "
        f"(target={vocab_size}), batch={batch_size}, seq={seq_length}"
    )

    def generator():
        for _ in range(num_batches):
            inputs = tf.random.uniform(
                (batch_size, seq_length),
                minval=0,
                maxval=actual_vocab,  # Use ACTUAL vocab size
                dtype=tf.int32,
            )
            labels = tf.roll(inputs, shift=-1, axis=1)
            yield inputs, labels

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, seq_length), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, seq_length), dtype=tf.int32),
        ),
    )
    return dataset.prefetch(2)


def run_training_trial(
    config: HPOTrainingConfig,
    trial_id: int,
    epochs: int = 2,
    steps_per_epoch: int = 20,
    use_real_data: bool = False,
    use_full_curriculum: bool = False,
) -> dict[str, Any]:
    """Run a single training trial using TrainingEngine.

    Args:
        config: HPO training configuration
        trial_id: Trial identifier
        epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        use_real_data: If True, load real Verso Baseline curriculum data
        use_full_curriculum: If True, use full curriculum (more datasets)

    Returns:
        Trial results dictionary
    """
    logger.info("=" * 60)
    logger.info(f"TRIAL {trial_id}")
    logger.info("=" * 60)

    # Log configuration
    estimated_params = estimate_model_params(config)
    logger.info(f"  Estimated params: {estimated_params / 1e6:.1f}M")
    logger.info(f"  Learning rate: {config.learning_rate:.2e}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Embedding dim: {config.embedding_dim}")
    logger.info(f"  Reasoning blocks: {config.num_reasoning_blocks}")
    logger.info(f"  MoE experts: {config.num_moe_experts}")
    logger.info(f"  Optimizer: {config.optimizer}")

    start_time = time.time()

    try:
        # Build model
        logger.info("Building model...")
        model = build_model(config)
        logger.info(f"  Model parameters: {model.count_params():,}")

        # Create optimizer
        logger.info("Creating optimizer...")
        optimizer = create_optimizer(config, model)
        logger.info(f"  Optimizer: {type(optimizer).__name__}")

        # Create EnterpriseTrainingConfig from HPO config
        logger.info("Creating TrainingEngine config...")
        engine_config = config.to_enterprise_training_config()

        # Create TrainingEngine
        engine = TrainingEngine(
            model=model,
            optimizer=optimizer,
            config=engine_config,
        )
        logger.info("  TrainingEngine initialized")
        logger.info(f"    GaLore: {engine_config.use_galore}")
        logger.info(f"    QNG: {engine_config.use_qng}")
        logger.info(f"    Barren Plateau: {engine_config.use_barren_plateau_detection}")

        # Create dataset - use EFFECTIVE vocab size from the model's vocab controller
        logger.info("Creating dataset...")
        effective_vocab = model._vocab_controller.effective_vocab_size
        tokenizer = model._vocab_controller.tokenizer

        dataset = None
        if use_real_data:
            # Load real Verso Baseline curriculum data
            num_samples = steps_per_epoch * epochs * config.batch_size + 500  # Extra buffer
            dataset = create_verso_baseline_dataset(
                tokenizer=tokenizer,
                batch_size=config.batch_size,
                seq_length=128,
                num_samples=num_samples,
                use_full_curriculum=use_full_curriculum,
                streaming=True,
            )
            if dataset is not None:
                logger.info("[Dataset] Using Verso Baseline curriculum data")

        # Fallback to dummy data if real data loading failed or not requested
        if dataset is None:
            logger.info("[Dataset] Using synthetic dummy data")
            dataset = create_dummy_dataset(
                vocab_size=config.vocab_size,  # Target (for logging)
                batch_size=config.batch_size,
                num_batches=steps_per_epoch * epochs,
                effective_vocab_size=effective_vocab,  # Actual vocab for token generation
            )

        # Run training with TrainingEngine
        logger.info(f"Training: {epochs} epochs × {steps_per_epoch} steps")

        result: TrainingResult = engine.run(
            epochs=epochs,
            dataset=dataset,
            callbacks=[DebugCallback(log_every=5)],
            steps_per_epoch=steps_per_epoch,  # Limit iterations for finite dataset control
        )

        wall_time = time.time() - start_time

        trial_result = {
            "trial_id": trial_id,
            "status": "completed" if result.success else "failed",
            "final_loss": result.final_loss,
            "epochs_completed": result.epochs_completed,
            "error": result.error,
            "wall_time_seconds": wall_time,
            "config": config.to_dict(),
        }

        logger.info(
            f"  Trial {trial_id} COMPLETED: loss={result.final_loss:.4f}, time={wall_time:.1f}s"
        )

    except Exception as e:
        wall_time = time.time() - start_time
        trial_result = {
            "trial_id": trial_id,
            "status": "failed",
            "error": str(e),
            "wall_time_seconds": wall_time,
            "config": config.to_dict(),
        }
        logger.error(f"  Trial {trial_id} FAILED: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    gc.collect()
    tf.keras.backend.clear_session()

    return trial_result


# =============================================================================
# Serialization Testing
# =============================================================================


def test_serialization(config: HPOTrainingConfig) -> bool:
    """Test save_model and load_model API."""
    import tempfile

    logger.info("=" * 60)
    logger.info("SERIALIZATION TEST")
    logger.info("=" * 60)

    try:
        # Build a small model
        logger.info("Building test model...")
        model = build_model(config)
        logger.info(f"  Model: {model.name}, params: {model.count_params():,}")

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"

            logger.info(f"Saving model to {save_path}...")
            save_model(
                model=model,
                path=save_path,
                config=config.to_enterprise_training_config(),
            )

            # Verify files exist
            assert (save_path / "model.keras").exists(), "model.keras not found"
            assert (save_path / "config.json").exists(), "config.json not found"
            assert (save_path / "metadata.json").exists(), "metadata.json not found"
            logger.info("  ✓ All expected files created")

            # Load model
            logger.info("Loading model...")
            loaded_model, loaded_config = load_model(save_path)

            assert loaded_model is not None, "Model not loaded"
            assert loaded_config is not None, "Config not loaded"
            assert loaded_model.count_params() == model.count_params(), "Param count mismatch"
            logger.info(f"  ✓ Model loaded: {loaded_model.count_params():,} params")
            logger.info(f"  ✓ Config loaded: {list(loaded_config.keys())}")

        logger.info("\n✅ Serialization test PASSED")
        return True

    except Exception as e:
        logger.error(f"\n❌ Serialization test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Main Sweep Execution
# =============================================================================


def run_hpo_sweep(
    param_budget: int = 100_000_000,
    vocab_size: int = 128_000,
    optimizer: str = "sympflowqng",
    num_trials: int = 3,
    epochs: int = 2,
    steps_per_epoch: int = 20,
    dry_run: bool = False,
    test_save_load: bool = False,
    use_real_data: bool = False,
    use_full_curriculum: bool = False,
) -> list[dict[str, Any]]:
    """Run HPO sweep mimicking WebUI configuration.

    Args:
        param_budget: Maximum model parameters (default: 100M)
        vocab_size: Vocabulary size (default: 128K)
        optimizer: Optimizer name (default: sympflowqng)
        num_trials: Number of trials to run
        epochs: Epochs per trial
        steps_per_epoch: Steps per epoch
        dry_run: If True, only show config without training
        test_save_load: If True, run serialization tests
        use_real_data: If True, load real Verso Baseline curriculum data
        use_full_curriculum: If True, use full curriculum (more datasets)

    Returns:
        List of trial results
    """
    logger.info("=" * 70)
    logger.info("HPO SWEEP DEBUG - TrainingEngine + HPOTrainingConfig Integration")
    logger.info("=" * 70)

    # Create base sweep config
    base_config = create_webui_sweep_config(
        param_budget=param_budget,
        vocab_size=vocab_size,
        optimizer=optimizer,
    )

    logger.info("Sweep Configuration:")
    logger.info(f"  Parameter Budget: {param_budget / 1e6:.0f}M")
    logger.info(f"  Vocabulary Size: {vocab_size / 1e3:.0f}K")
    logger.info(f"  Optimizer: {optimizer}")
    logger.info(f"  Trials: {num_trials}")
    logger.info(f"  Epochs/Trial: {epochs}")
    logger.info(f"  Steps/Epoch: {steps_per_epoch}")
    logger.info(f"  Use Real Data: {use_real_data}")
    if use_real_data:
        logger.info(f"  Full Curriculum: {use_full_curriculum}")
    logger.info("")
    logger.info("Feature Flags (from HPOTrainingConfig):")
    for key, value in base_config.feature_flags.items():
        logger.info(f"  {key}: {value}")

    # Test serialization if requested
    if test_save_load:
        if not test_serialization(base_config):
            return []
        logger.info("")

    if dry_run:
        logger.info("\n[DRY RUN] Showing sampled configurations only:")
        for trial_id in range(num_trials):
            config = sample_hyperparameters(base_config, trial_id)
            estimated = estimate_model_params(config)
            logger.info(f"\nTrial {trial_id}:")
            logger.info(f"  params={estimated / 1e6:.1f}M, lr={config.learning_rate:.2e}")
            logger.info(f"  dim={config.embedding_dim}, blocks={config.num_reasoning_blocks}")
            logger.info(f"  batch={config.batch_size}")
        return []

    # Run trials
    results = []

    for trial_id in range(num_trials):
        logger.info("")

        # Sample hyperparameters
        config = sample_hyperparameters(base_config, trial_id)

        # Run trial with TrainingEngine
        result = run_training_trial(
            config=config,
            trial_id=trial_id,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            use_real_data=use_real_data,
            use_full_curriculum=use_full_curriculum,
        )
        results.append(result)

        # Save training config after each trial (mimics WebUI)
        try:
            output_dir = PROJECT_ROOT / "artifacts" / "debug_hpo"
            output_dir.mkdir(parents=True, exist_ok=True)
            config.save(output_dir / f"trial_{trial_id}_config.json")
            logger.info(f"  Saved config to {output_dir / f'trial_{trial_id}_config.json'}")
        except Exception as e:
            logger.warning(f"  Could not save config: {e}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SWEEP SUMMARY")
    logger.info("=" * 70)

    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] == "failed"]

    logger.info(f"Completed: {len(completed)}/{len(results)}")
    logger.info(f"Failed: {len(failed)}/{len(results)}")

    if completed:
        best = min(completed, key=lambda r: r["final_loss"])
        logger.info(f"\nBest Trial: {best['trial_id']}")
        logger.info(f"  Final Loss: {best['final_loss']:.4f}")

    if failed:
        logger.info("\nFailed Trials:")
        for r in failed:
            logger.info(f"  Trial {r['trial_id']}: {r['error']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="HPO Sweep Debug Script - Tests TrainingEngine + HPOTrainingConfig",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 250M param budget, 1M context, SympFlowQNG, Verso Baseline data
  python scripts/debug_hpo_sweep_webui.py

  # Quick test (synthetic data for speed)
  python scripts/debug_hpo_sweep_webui.py --no-real-data --trials 1 --epochs 1 --steps 10

  # Full curriculum (more datasets, slower but more realistic)
  python scripts/debug_hpo_sweep_webui.py --full-curriculum

  # Test serialization API
  python scripts/debug_hpo_sweep_webui.py --test-serialization --dry-run

  # Dry run (config only, no training)
  python scripts/debug_hpo_sweep_webui.py --dry-run

  # Different optimizer
  python scripts/debug_hpo_sweep_webui.py --optimizer qiao
        """,
    )

    parser.add_argument(
        "--param-budget",
        type=int,
        default=250_000_000,
        help="Parameter budget in total params (default: 250M to match WebUI)",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=128_000, help="Vocabulary size (default: 128K)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sympflowqng",
        choices=["sophiag", "qiao", "grover", "sympflow", "sympflowqng"],
        help="Optimizer (default: sympflowqng)",
    )
    parser.add_argument("--trials", type=int, default=3, help="Number of HPO trials (default: 3)")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per trial (default: 2)")
    parser.add_argument("--steps", type=int, default=20, help="Steps per epoch (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="Show config only, no training")
    parser.add_argument(
        "--test-serialization", action="store_true", help="Run save_model/load_model tests"
    )
    parser.add_argument(
        "--no-real-data",
        action="store_true",
        help="Use synthetic data instead of Verso Baseline curriculum",
    )
    parser.add_argument(
        "--full-curriculum", action="store_true", help="Use full curriculum (more datasets, slower)"
    )

    args = parser.parse_args()

    try:
        results = run_hpo_sweep(
            param_budget=args.param_budget,
            vocab_size=args.vocab_size,
            optimizer=args.optimizer,
            num_trials=args.trials,
            epochs=args.epochs,
            steps_per_epoch=args.steps,
            dry_run=args.dry_run,
            test_save_load=args.test_serialization,
            use_real_data=not args.no_real_data,  # Default: True, load Verso Baseline
            use_full_curriculum=args.full_curriculum,
        )

        if results:
            logger.info("\n✅ HPO sweep completed successfully")

    except Exception as e:
        logger.error(f"Sweep failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
