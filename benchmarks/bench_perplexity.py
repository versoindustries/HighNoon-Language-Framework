# benchmarks/bench_perplexity.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Perplexity benchmarks for HSMN architecture.

Evaluates model quality via cross-entropy perplexity on real HuggingFace
datasets (WikiText-2, etc.) and synthetic text datasets.

Supports streaming for large datasets to minimize memory usage.

Example:
    >>> from benchmarks.bench_perplexity import run_perplexity_benchmark
    >>> results = run_perplexity_benchmark(dataset="wikitext", quick=True)
    >>> print(f"Perplexity: {results['overall_perplexity']:.2f}")

Command-line usage:
    python benchmarks/bench_perplexity.py --dataset wikitext --quick
    python benchmarks/bench_perplexity.py --dataset synthetic
"""

import argparse
import json
import logging
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from benchmarks.benchmark_config import BenchmarkConfig
from benchmarks.benchmark_harness import BenchmarkHarness
from highnoon.tokenization import QWTTextTokenizer

logger = logging.getLogger(__name__)

# Cached tokenizer instance
_tokenizer: QWTTextTokenizer | None = None


def get_tokenizer(vocab_size: int, max_length: int) -> QWTTextTokenizer:
    """Get or create the QWTTextTokenizer instance.

    Uses the actual HSMN tokenizer for proper UTF-8 byte encoding
    compatible with the full model pipeline.

    Args:
        vocab_size: Vocabulary size.
        max_length: Maximum sequence length.

    Returns:
        QWTTextTokenizer instance.
    """
    global _tokenizer
    if _tokenizer is None or _tokenizer.vocab_size != vocab_size:
        _tokenizer = QWTTextTokenizer(
            vocab_size=vocab_size,
            model_max_length=max_length,
            enable_thinking_tokens=False,  # Disable for benchmarks
        )
        logger.info(f"Created QWTTextTokenizer: vocab={vocab_size}, max_len={max_length}")
    return _tokenizer


# HuggingFace dataset configurations
DATASET_CONFIGS = {
    "wikitext": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "text_column": "text",
        "description": "WikiText-2 test set - standard language modeling benchmark",
    },
    "wikitext-103": {
        "path": "wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "test",
        "text_column": "text",
        "description": "WikiText-103 test set - larger language modeling benchmark",
    },
    "ptb": {
        "path": "ptb_text_only",
        "name": None,
        "split": "test",
        "text_column": "sentence",
        "description": "Penn Treebank - classic language modeling benchmark",
    },
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "split": "validation",
        "text_column": "text",
        "description": "C4 validation set - web text benchmark",
    },
}


def load_huggingface_dataset(
    dataset_name: str,
    max_samples: int = 100,
    streaming: bool = True,
) -> Iterator[str]:
    """Load a HuggingFace dataset with optional streaming.

    Args:
        dataset_name: Name of dataset (e.g., "wikitext", "wikitext-103", "c4").
        max_samples: Maximum number of samples to yield.
        streaming: Whether to stream the dataset (recommended for large datasets).

    Yields:
        Text strings from the dataset.

    Raises:
        ImportError: If datasets library is not installed.
        ValueError: If dataset_name is not recognized.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required. Install with: pip install datasets"
        )

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. " f"Available: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    logger.info(f"Loading {dataset_name}: {config['description']}")

    try:
        dataset = load_dataset(
            config["path"],
            config["name"],
            split=config["split"],
            streaming=streaming,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning(f"Failed to load {dataset_name}: {e}. Falling back to synthetic.")
        raise

    text_column = config["text_column"]
    samples_yielded = 0

    for item in dataset:
        text = item.get(text_column, "")
        # Skip empty or very short texts
        if text and len(text) > 50:
            yield text
            samples_yielded += 1
            if samples_yielded >= max_samples:
                break

    logger.info(f"Loaded {samples_yielded} samples from {dataset_name}")


def tokenize_text(
    text: str,
    vocab_size: int,
    max_length: int,
) -> np.ndarray:
    """Tokenize text using the actual QWTTextTokenizer.

    Uses the HSMN UTF-8 byte-level tokenizer for proper encoding
    that matches the full model training pipeline.

    Args:
        text: Input text.
        vocab_size: Vocabulary size.
        max_length: Maximum sequence length.

    Returns:
        Token IDs array of shape [max_length].
    """
    tokenizer = get_tokenizer(vocab_size, max_length)

    # Tokenize with truncation and padding
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
    )

    return np.array(encoded["input_ids"], dtype=np.int32)


def generate_synthetic_dataset(
    num_samples: int,
    vocab_size: int,
    seq_length: int,
    seed: int = 42,
) -> tf.data.Dataset:
    """Generate synthetic token sequences for perplexity evaluation.

    Args:
        num_samples: Number of samples to generate.
        vocab_size: Vocabulary size.
        seq_length: Sequence length per sample.
        seed: Random seed.

    Returns:
        TensorFlow dataset of (input_ids, target_ids) pairs.
    """
    np.random.seed(seed)

    samples = []
    for _ in range(num_samples):
        # Mix of uniform and Zipf-distributed tokens
        if np.random.random() < 0.3:
            tokens = np.random.randint(1, vocab_size, size=seq_length)
        else:
            tokens = np.random.zipf(1.5, size=seq_length).astype(np.int32)
            tokens = np.clip(tokens, 1, vocab_size - 1)
        samples.append(tokens)

    input_ids = np.array(samples, dtype=np.int32)
    target_ids = np.roll(input_ids, -1, axis=1)

    dataset = tf.data.Dataset.from_tensor_slices((input_ids, target_ids))
    return dataset


def load_dataset_as_batches(
    dataset_name: str,
    vocab_size: int,
    seq_length: int,
    batch_size: int,
    max_samples: int,
    seed: int = 42,
) -> Iterator[tuple[tf.Tensor, tf.Tensor]]:
    """Load dataset and yield batches of (input_ids, target_ids).

    Args:
        dataset_name: Dataset name ("synthetic", "wikitext", etc.).
        vocab_size: Vocabulary size.
        seq_length: Sequence length.
        batch_size: Batch size.
        max_samples: Maximum samples.
        seed: Random seed.

    Yields:
        Tuples of (input_ids, target_ids) tensors.
    """
    if dataset_name == "synthetic":
        dataset = generate_synthetic_dataset(max_samples, vocab_size, seq_length, seed)
        for batch in dataset.batch(batch_size):
            yield batch
    else:
        # Load from HuggingFace
        try:
            texts = list(load_huggingface_dataset(dataset_name, max_samples))
        except (ImportError, Exception) as e:
            logger.warning(f"Falling back to synthetic: {e}")
            dataset = generate_synthetic_dataset(max_samples, vocab_size, seq_length, seed)
            for batch in dataset.batch(batch_size):
                yield batch
            return

        # Tokenize and batch
        batch_inputs = []
        batch_targets = []

        for text in texts:
            tokens = tokenize_text(text, vocab_size, seq_length)
            target = np.roll(tokens, -1)
            batch_inputs.append(tokens)
            batch_targets.append(target)

            if len(batch_inputs) >= batch_size:
                yield (
                    tf.constant(np.array(batch_inputs), dtype=tf.int32),
                    tf.constant(np.array(batch_targets), dtype=tf.int32),
                )
                batch_inputs = []
                batch_targets = []

        # Yield remaining
        if batch_inputs:
            yield (
                tf.constant(np.array(batch_inputs), dtype=tf.int32),
                tf.constant(np.array(batch_targets), dtype=tf.int32),
            )


def compute_perplexity(
    model: tf.keras.Model,
    input_ids: tf.Tensor,
    target_ids: tf.Tensor,
) -> tuple[float, float]:
    """Compute perplexity for a batch.

    Args:
        model: HSMN model.
        input_ids: Input token IDs [batch, seq_len].
        target_ids: Target token IDs [batch, seq_len].

    Returns:
        Tuple of (mean_perplexity, cross_entropy_loss).
    """
    outputs = model(input_ids, training=False)
    logits = outputs["logits"]

    loss = tf.keras.losses.sparse_categorical_crossentropy(target_ids, logits, from_logits=True)
    mean_loss = tf.reduce_mean(loss).numpy()
    perplexity = math.exp(min(mean_loss, 100))

    return perplexity, mean_loss


def compute_per_position_perplexity(
    model: tf.keras.Model,
    input_ids: tf.Tensor,
    target_ids: tf.Tensor,
) -> np.ndarray:
    """Compute perplexity at each sequence position.

    Args:
        model: HSMN model.
        input_ids: Input token IDs [batch, seq_len].
        target_ids: Target token IDs [batch, seq_len].

    Returns:
        Array of per-position perplexities [seq_len].
    """
    outputs = model(input_ids, training=False)
    logits = outputs["logits"]

    loss = tf.keras.losses.sparse_categorical_crossentropy(target_ids, logits, from_logits=True)
    mean_loss_per_pos = tf.reduce_mean(loss, axis=0).numpy()
    perplexity_per_pos = np.exp(np.clip(mean_loss_per_pos, 0, 100))

    return perplexity_per_pos


def run_perplexity_benchmark(
    config: BenchmarkConfig | None = None,
    quick: bool = False,
    dataset: str = "wikitext",
) -> dict[str, Any]:
    """Run complete perplexity benchmark suite.

    Args:
        config: Benchmark configuration.
        quick: Use quick configuration.
        dataset: Dataset to use ("synthetic", "wikitext", "wikitext-103", etc.).

    Returns:
        Dictionary with perplexity results.
    """
    if config is None:
        config = BenchmarkConfig.quick() if quick else BenchmarkConfig()

    harness = BenchmarkHarness(config)
    model = harness.get_model()

    logger.info("=" * 60)
    logger.info("HSMN Perplexity Benchmark")
    logger.info(f"Pattern: {config.model.block_pattern}")
    logger.info(f"Dataset: {dataset}")
    logger.info("=" * 60)

    max_samples = config.perplexity.max_samples
    seq_length = min(512, config.model.max_seq_length)
    batch_size = 8

    logger.info(f"Evaluating on {max_samples} samples (seq_len={seq_length})")

    perplexities = []
    losses = []
    position_perplexities = []
    batch_idx = 0

    for input_ids, target_ids in load_dataset_as_batches(
        dataset, config.model.vocab_size, seq_length, batch_size, max_samples, config.seed
    ):
        ppl, loss = compute_perplexity(model, input_ids, target_ids)
        perplexities.append(ppl)
        losses.append(loss)

        if batch_idx < 5:
            pos_ppl = compute_per_position_perplexity(model, input_ids, target_ids)
            position_perplexities.append(pos_ppl)

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx+1}: PPL={ppl:.2f}, Loss={loss:.4f}")

        batch_idx += 1

    overall_ppl = float(np.mean(perplexities))
    overall_loss = float(np.mean(losses))

    results = {
        "overall_perplexity": overall_ppl,
        "overall_cross_entropy": overall_loss,
        "perplexity_std": float(np.std(perplexities)),
        "perplexity_min": float(np.min(perplexities)),
        "perplexity_max": float(np.max(perplexities)),
        "num_batches": batch_idx,
        "sequence_length": seq_length,
        "dataset": dataset,
        "model_config": config.model.to_dict(),
        "block_pattern": config.model.block_pattern,
    }

    if position_perplexities:
        avg_position_ppl = np.mean(position_perplexities, axis=0)
        results["per_position_perplexity"] = {
            "early_context": float(np.mean(avg_position_ppl[: seq_length // 4])),
            "mid_context": float(np.mean(avg_position_ppl[seq_length // 4 : 3 * seq_length // 4])),
            "late_context": float(np.mean(avg_position_ppl[3 * seq_length // 4 :])),
        }

    logger.info(f"\nOverall Perplexity: {overall_ppl:.2f}")
    logger.info(f"Cross-Entropy Loss: {overall_loss:.4f}")

    return results


def format_perplexity_markdown(results: dict[str, Any]) -> str:
    """Format perplexity results as markdown."""
    lines = [
        "# HSMN Perplexity Benchmark",
        "",
        f"**Block Pattern**: `{results['block_pattern']}`",
        f"**Dataset**: {results['dataset']}",
        "",
        "## Overall Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Perplexity** | {results['overall_perplexity']:.2f} |",
        f"| **Cross-Entropy Loss** | {results['overall_cross_entropy']:.4f} |",
        f"| **Std Dev** | {results['perplexity_std']:.2f} |",
        f"| **Min** | {results['perplexity_min']:.2f} |",
        f"| **Max** | {results['perplexity_max']:.2f} |",
        f"| **Batches** | {results['num_batches']} |",
        f"| **Sequence Length** | {results['sequence_length']} |",
        "",
    ]

    if "per_position_perplexity" in results:
        pos = results["per_position_perplexity"]
        lines.extend(
            [
                "## Per-Position Analysis",
                "",
                "| Context Region | Perplexity |",
                "|----------------|------------|",
                f"| Early (0-25%) | {pos['early_context']:.2f} |",
                f"| Mid (25-75%) | {pos['mid_context']:.2f} |",
                f"| Late (75-100%) | {pos['late_context']:.2f} |",
                "",
                "> Lower perplexity in later positions indicates effective context utilization.",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="HSMN Perplexity Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick validation mode")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "wikitext", "wikitext-103", "ptb", "c4"],
        default="wikitext",
        help="Dataset for evaluation (default: wikitext)",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/reports"), help="Output directory"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    results = run_perplexity_benchmark(quick=args.quick, dataset=args.dataset)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_format in ("json", "both"):
        json_path = args.output_dir / "perplexity_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {json_path}")

    if args.output_format in ("markdown", "both"):
        md_path = args.output_dir / "perplexity_results.md"
        with open(md_path, "w") as f:
            f.write(format_perplexity_markdown(results))
        print(f"Saved Markdown: {md_path}")

    print(f"\nPerplexity ({args.dataset}): {results['overall_perplexity']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
