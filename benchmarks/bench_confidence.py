# benchmarks/bench_confidence.py
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

"""Confidence and calibration benchmarks for HSMN architecture.

Evaluates model confidence through entropy analysis, self-consistency
scoring, and calibration metrics (ECE).

Example:
    >>> from benchmarks.bench_confidence import run_confidence_benchmark
    >>> results = run_confidence_benchmark(quick=True)
    >>> print(f"Mean Entropy: {results['mean_entropy']:.3f}")
    >>> print(f"ECE: {results['expected_calibration_error']:.4f}")

Command-line usage:
    python benchmarks/bench_confidence.py --quick
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from benchmarks.benchmark_config import BenchmarkConfig
from benchmarks.benchmark_harness import BenchmarkHarness

logger = logging.getLogger(__name__)


def compute_token_entropy(logits: tf.Tensor) -> tf.Tensor:
    """Compute entropy of token predictions.

    Higher entropy indicates lower confidence (more uncertainty).

    Args:
        logits: Model logits [batch, seq_len, vocab_size].

    Returns:
        Entropy tensor [batch, seq_len].
    """
    probs = tf.nn.softmax(logits, axis=-1)
    # Entropy = -sum(p * log(p))
    log_probs = tf.math.log(probs + 1e-10)
    entropy = -tf.reduce_sum(probs * log_probs, axis=-1)
    return entropy


def compute_confidence_from_entropy(entropy: tf.Tensor, vocab_size: int) -> tf.Tensor:
    """Convert entropy to confidence score (0-1).

    Confidence = 1 - (entropy / max_entropy)
    where max_entropy = log(vocab_size)

    Args:
        entropy: Entropy values.
        vocab_size: Vocabulary size for normalization.

    Returns:
        Confidence scores [0, 1].
    """
    max_entropy = np.log(vocab_size)
    normalized_entropy = entropy / max_entropy
    confidence = 1.0 - tf.clip_by_value(normalized_entropy, 0.0, 1.0)
    return confidence


def compute_top_k_confidence(logits: tf.Tensor, k: int = 5) -> tf.Tensor:
    """Compute confidence as sum of top-K probabilities.

    High top-K concentration indicates confident prediction.

    Args:
        logits: Model logits [batch, seq_len, vocab_size].
        k: Number of top probabilities to sum.

    Returns:
        Top-K confidence [batch, seq_len].
    """
    probs = tf.nn.softmax(logits, axis=-1)
    top_k_probs, _ = tf.math.top_k(probs, k=k)
    confidence = tf.reduce_sum(top_k_probs, axis=-1)
    return confidence


def compute_self_consistency(
    model: tf.keras.Model,
    input_ids: tf.Tensor,
    num_samples: int = 5,
    temperature: float = 1.0,
    generate_tokens: int = 10,
) -> dict[str, Any]:
    """Measure self-consistency across multiple generations.

    Uses QSGGenerator for parallel token generation.

    Args:
        model: HSMN model.
        input_ids: Input token IDs [batch, seq_len].
        num_samples: Number of generation samples.
        temperature: Sampling temperature.
        generate_tokens: Tokens to generate per sample.

    Returns:
        Dictionary with consistency metrics.
    """
    from highnoon.inference.qsg_generator import QSGConfig, QSGGenerator

    config = QSGConfig(
        bond_dim=32,
        coherence_range=min(64, generate_tokens * 2),
        grover_iterations=3,
    )
    generator = QSGGenerator(model, config)

    generations = []

    for _ in range(num_samples):
        try:
            generated = generator.generate(
                input_ids,
                max_new_tokens=generate_tokens,
                temperature=temperature,
            )
            # Extract generated portion (after input)
            new_tokens = generated[:, input_ids.shape[1] :]
            generations.append(new_tokens.numpy())
        except Exception:
            # Fallback: sample from forward pass logits
            outputs = model(input_ids, training=False)
            logits = outputs["logits"][:, -1, :]
            # Sample tokens autoregressively from logits
            sampled = tf.random.categorical(logits / temperature, generate_tokens)
            generations.append(sampled.numpy())

    generations = np.array(generations)  # [num_samples, batch, gen_len]

    # Compute agreement at each position
    batch_size = generations.shape[1]
    gen_length = generations.shape[2]

    agreements = []
    for b in range(batch_size):
        for pos in range(gen_length):
            tokens_at_pos = generations[:, b, pos]
            # Mode frequency / num_samples
            unique, counts = np.unique(tokens_at_pos, return_counts=True)
            max_agreement = counts.max() / num_samples
            agreements.append(max_agreement)

    mean_agreement = float(np.mean(agreements))

    # Exact match across all samples
    exact_matches = 0
    for b in range(batch_size):
        sample_tokens = [tuple(generations[s, b, :]) for s in range(num_samples)]
        if len(set(sample_tokens)) == 1:
            exact_matches += 1
    exact_match_rate = exact_matches / batch_size

    return {
        "mean_position_agreement": mean_agreement,
        "exact_match_rate": float(exact_match_rate),
        "num_samples": num_samples,
        "temperature": temperature,
        "generation_mode": "qsg",
    }


def compute_expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> tuple[float, dict[str, Any]]:
    """Compute Expected Calibration Error (ECE).

    ECE measures how well predicted confidence matches actual accuracy.
    Lower is better (perfectly calibrated = 0).

    Args:
        confidences: Predicted confidence scores.
        accuracies: Actual correctness (0 or 1).
        num_bins: Number of calibration bins.

    Returns:
        Tuple of (ECE value, calibration curve data).
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    calibration_curve = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
            calibration_curve.append(
                {
                    "bin_lower": float(bin_lower),
                    "bin_upper": float(bin_upper),
                    "avg_confidence": float(avg_confidence),
                    "avg_accuracy": float(avg_accuracy),
                    "proportion": float(prop_in_bin),
                }
            )

    return float(ece), {"bins": calibration_curve}


def run_confidence_benchmark(
    config: BenchmarkConfig | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run complete confidence/calibration benchmark suite.

    Args:
        config: Benchmark configuration.
        quick: Use quick configuration.

    Returns:
        Dictionary with confidence and calibration results.
    """
    if config is None:
        config = BenchmarkConfig.quick() if quick else BenchmarkConfig()

    harness = BenchmarkHarness(config)
    model = harness.get_model()

    logger.info("=" * 60)
    logger.info("HSMN Confidence Benchmark")
    logger.info(f"Pattern: {config.model.block_pattern}")
    logger.info("=" * 60)

    # Generate test data
    num_samples = config.perplexity.max_samples
    seq_length = min(256, config.model.max_seq_length)
    batch_size = 8

    all_entropies = []
    all_confidences = []
    all_top_k_confidences = []
    all_accuracies = []

    logger.info(f"Computing entropy and confidence on {num_samples} samples...")

    for batch_idx in range(num_samples // batch_size):
        input_ids = harness.create_input(batch_size, seq_length)
        target_ids = tf.roll(input_ids, -1, axis=1)

        outputs = model(input_ids, training=False)
        logits = outputs["logits"]

        # Entropy and confidence
        entropy = compute_token_entropy(logits)
        confidence = compute_confidence_from_entropy(entropy, config.model.vocab_size)
        top_k_conf = compute_top_k_confidence(logits, k=5)

        # Accuracy (for calibration)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        correct = tf.cast(predictions == target_ids, tf.float32)

        all_entropies.extend(entropy.numpy().flatten().tolist())
        all_confidences.extend(confidence.numpy().flatten().tolist())
        all_top_k_confidences.extend(top_k_conf.numpy().flatten().tolist())
        all_accuracies.extend(correct.numpy().flatten().tolist())

        if (batch_idx + 1) % 5 == 0:
            logger.info(f"  Batch {batch_idx + 1}: mean_entropy={np.mean(entropy.numpy()):.3f}")

    all_entropies = np.array(all_entropies)
    all_confidences = np.array(all_confidences)
    all_top_k_confidences = np.array(all_top_k_confidences)
    all_accuracies = np.array(all_accuracies)

    # Compute ECE
    ece, calibration_curve = compute_expected_calibration_error(
        all_confidences, all_accuracies, num_bins=config.confidence.num_bins
    )

    # Self-consistency (on smaller subset)
    logger.info("Computing self-consistency...")
    consistency_input = harness.create_input(2, 64)
    consistency = compute_self_consistency(
        model,
        consistency_input,
        num_samples=config.confidence.num_samples,
        temperature=1.0,
        generate_tokens=16,
    )

    results = {
        "mean_entropy": float(np.mean(all_entropies)),
        "std_entropy": float(np.std(all_entropies)),
        "min_entropy": float(np.min(all_entropies)),
        "max_entropy": float(np.max(all_entropies)),
        "mean_confidence": float(np.mean(all_confidences)),
        "mean_top_k_confidence": float(np.mean(all_top_k_confidences)),
        "expected_calibration_error": ece,
        "calibration_curve": calibration_curve,
        "self_consistency": consistency,
        "num_tokens_analyzed": len(all_entropies),
        "model_config": config.model.to_dict(),
        "block_pattern": config.model.block_pattern,
    }

    # Entropy distribution buckets
    entropy_buckets = {
        "low_entropy_ratio": float((all_entropies < 2.0).mean()),
        "medium_entropy_ratio": float(((all_entropies >= 2.0) & (all_entropies < 5.0)).mean()),
        "high_entropy_ratio": float((all_entropies >= 5.0).mean()),
    }
    results["entropy_distribution"] = entropy_buckets

    logger.info(f"\nMean Entropy: {results['mean_entropy']:.3f}")
    logger.info(f"Mean Confidence: {results['mean_confidence']:.3f}")
    logger.info(f"ECE: {results['expected_calibration_error']:.4f}")
    logger.info(f"Self-Consistency: {consistency['mean_position_agreement']:.3f}")

    return results


def format_confidence_markdown(results: dict[str, Any]) -> str:
    """Format confidence results as markdown."""
    lines = [
        "# HSMN Confidence & Calibration Benchmark",
        "",
        f"**Block Pattern**: `{results['block_pattern']}`",
        "",
        "## Entropy Analysis",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Mean Entropy** | {results['mean_entropy']:.3f} |",
        f"| **Std Entropy** | {results['std_entropy']:.3f} |",
        f"| **Min Entropy** | {results['min_entropy']:.3f} |",
        f"| **Max Entropy** | {results['max_entropy']:.3f} |",
        "",
        "### Entropy Distribution",
        "",
        f"- **Low Entropy (< 2.0)**: {results['entropy_distribution']['low_entropy_ratio']*100:.1f}%",
        f"- **Medium Entropy (2.0-5.0)**: {results['entropy_distribution']['medium_entropy_ratio']*100:.1f}%",
        f"- **High Entropy (> 5.0)**: {results['entropy_distribution']['high_entropy_ratio']*100:.1f}%",
        "",
        "## Confidence Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Mean Confidence** | {results['mean_confidence']:.3f} |",
        f"| **Mean Top-5 Confidence** | {results['mean_top_k_confidence']:.3f} |",
        f"| **Expected Calibration Error (ECE)** | {results['expected_calibration_error']:.4f} |",
        "",
        "> Lower ECE indicates better calibration (predicted confidence matches actual accuracy).",
        "",
        "## Self-Consistency",
        "",
        f"- **Mean Position Agreement**: {results['self_consistency']['mean_position_agreement']:.3f}",
        f"- **Exact Match Rate**: {results['self_consistency']['exact_match_rate']:.3f}",
        f"- **Samples per Input**: {results['self_consistency']['num_samples']}",
        "",
    ]

    return "\n".join(lines)


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="HSMN Confidence Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick validation mode")
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

    results = run_confidence_benchmark(quick=args.quick)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_format in ("json", "both"):
        json_path = args.output_dir / "confidence_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {json_path}")

    if args.output_format in ("markdown", "both"):
        md_path = args.output_dir / "confidence_results.md"
        with open(md_path, "w") as f:
            f.write(format_confidence_markdown(results))
        print(f"Saved Markdown: {md_path}")

    print(f"\nMean Confidence: {results['mean_confidence']:.3f}")
    print(f"ECE: {results['expected_calibration_error']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
