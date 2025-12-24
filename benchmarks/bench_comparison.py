# benchmarks/bench_comparison.py
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

"""Architecture comparison benchmarks for HSMN.

Compares HSMN's mamba_timecrystal_wlam_moe_hybrid pattern against
baseline architectures to demonstrate O(n) scaling advantage.

Example:
    >>> from benchmarks.bench_comparison import run_comparison_benchmark
    >>> results = run_comparison_benchmark(quick=True)
    >>> print(results['hsmn_vs_transformer_speedup'])

Command-line usage:
    python benchmarks/bench_comparison.py --quick
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from benchmarks.benchmark_config import BenchmarkConfig
from benchmarks.benchmark_harness import BenchmarkHarness

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureProfile:
    """Profile for an architecture being compared."""

    name: str
    complexity: str
    description: str
    timings_ms: list[float]
    sequence_lengths: list[int]

    @property
    def mean_time_per_length(self) -> dict[int, float]:
        """Map of seq_len to mean time."""
        return dict(zip(self.sequence_lengths, self.timings_ms))


def create_transformer_baseline(
    embedding_dim: int,
    num_layers: int,
    num_heads: int,
    vocab_size: int,
    max_seq_length: int,
) -> tf.keras.Model:
    """Create a simple transformer baseline for comparison.

    This is a standard quadratic attention transformer to demonstrate
    the O(n²) vs O(n) scaling difference.

    Args:
        embedding_dim: Model dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        vocab_size: Vocabulary size.
        max_seq_length: Maximum sequence length.

    Returns:
        Simple transformer model.
    """
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")

    # Embedding
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = x + tf.keras.layers.Embedding(max_seq_length, embedding_dim)(tf.range(tf.shape(inputs)[1]))

    # Transformer layers with standard attention
    for i in range(num_layers):
        # Self-attention (O(n²))
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            name=f"attention_{i}",
        )(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn_output)

        # FFN
        ffn = tf.keras.layers.Dense(embedding_dim * 4, activation="gelu")(x)
        ffn = tf.keras.layers.Dense(embedding_dim)(ffn)
        x = tf.keras.layers.LayerNormalization()(x + ffn)

    # Output projection
    outputs = tf.keras.layers.Dense(vocab_size, name="lm_head")(x)

    return tf.keras.Model(inputs, outputs, name="TransformerBaseline")


def create_linear_attention_baseline(
    embedding_dim: int,
    num_layers: int,
    num_heads: int,
    vocab_size: int,
    max_seq_length: int,
) -> tf.keras.Model:
    """Create a linear attention baseline for comparison.

    Uses ELU feature-map based linear attention (O(n)).

    Args:
        embedding_dim: Model dimension.
        num_layers: Number of layers.
        num_heads: Number of attention heads.
        vocab_size: Vocabulary size.
        max_seq_length: Maximum sequence length.

    Returns:
        Linear attention baseline model.
    """
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")

    # Embedding
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)

    # Linear attention layers
    for i in range(num_layers):
        # Linear attention approximation via feature maps
        q = tf.keras.layers.Dense(embedding_dim, name=f"q_{i}")(x)
        k = tf.keras.layers.Dense(embedding_dim, name=f"k_{i}")(x)
        v = tf.keras.layers.Dense(embedding_dim, name=f"v_{i}")(x)

        # ELU feature map
        q = tf.nn.elu(q) + 1.0
        k = tf.nn.elu(k) + 1.0

        # Linear attention: O(n) via (Q * cumsum(K^T V)) / (Q * cumsum(K))
        kv = tf.einsum("bld,ble->bde", k, v)
        qkv = tf.einsum("bld,de->ble", q, kv[0])  # Simplified

        x = tf.keras.layers.LayerNormalization()(x + qkv)

        # FFN
        ffn = tf.keras.layers.Dense(embedding_dim * 4, activation="gelu")(x)
        ffn = tf.keras.layers.Dense(embedding_dim)(ffn)
        x = tf.keras.layers.LayerNormalization()(x + ffn)

    outputs = tf.keras.layers.Dense(vocab_size, name="lm_head")(x)

    return tf.keras.Model(inputs, outputs, name="LinearAttentionBaseline")


def benchmark_model_scaling(
    model: tf.keras.Model,
    sequence_lengths: list[int],
    vocab_size: int,
    batch_size: int = 1,
    warmup: int = 3,
    iterations: int = 10,
) -> list[float]:
    """Benchmark model across sequence lengths.

    Args:
        model: Model to benchmark.
        sequence_lengths: Sequence lengths to test.
        vocab_size: Vocabulary size for input generation.
        batch_size: Batch size.
        warmup: Warmup iterations.
        iterations: Benchmark iterations.

    Returns:
        List of mean times (ms) for each sequence length.
    """
    timings = []

    @tf.function(reduce_retracing=True)
    def forward_fn(x):
        return model(x, training=False)

    for seq_len in sequence_lengths:
        input_ids = tf.random.uniform(
            [batch_size, seq_len], minval=0, maxval=vocab_size, dtype=tf.int32
        )

        # Warmup
        for _ in range(warmup):
            try:
                _ = forward_fn(input_ids)
            except Exception:
                timings.append(float("inf"))
                continue

        # Benchmark
        times = []
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                _ = forward_fn(input_ids)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            except Exception as e:
                logger.warning(f"seq_len={seq_len} failed: {e}")
                times.append(float("inf"))
                break

        mean_time = np.mean(times)
        timings.append(float(mean_time))
        logger.info(f"  seq_len={seq_len}: {mean_time:.1f} ms")

    return timings


def analyze_complexity(
    profile: ArchitectureProfile,
) -> dict[str, Any]:
    """Analyze computational complexity from timing data.

    Args:
        profile: Architecture profile with timings.

    Returns:
        Dictionary with complexity analysis.
    """
    seq_lens = np.array(profile.sequence_lengths, dtype=float)
    times = np.array(profile.timings_ms)

    # Filter out infinite times
    valid = np.isfinite(times)
    if np.sum(valid) < 2:
        return {"complexity": "insufficient_data"}

    seq_lens = seq_lens[valid]
    times = times[valid]

    # Normalize for fitting
    log_seq = np.log(seq_lens)
    log_time = np.log(times)

    # Fit power law: time ~ seq^alpha
    # log(time) = alpha * log(seq) + c
    alpha, c = np.polyfit(log_seq, log_time, 1)

    # Classify complexity
    if alpha < 1.2:
        complexity = "O(n)"
    elif alpha < 1.8:
        complexity = "O(n log n)"
    elif alpha < 2.2:
        complexity = "O(n²)"
    else:
        complexity = f"O(n^{alpha:.1f})"

    return {
        "fitted_exponent": float(alpha),
        "complexity_class": complexity,
        "intercept": float(c),
    }


def run_comparison_benchmark(
    config: BenchmarkConfig | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run architecture comparison benchmarks.

    Args:
        config: Benchmark configuration.
        quick: Use quick configuration.

    Returns:
        Dictionary with comparison results.
    """
    if config is None:
        config = BenchmarkConfig.quick() if quick else BenchmarkConfig()

    harness = BenchmarkHarness(config)

    logger.info("=" * 60)
    logger.info("HSMN Architecture Comparison Benchmark")
    logger.info(f"Pattern: {config.model.block_pattern}")
    logger.info("=" * 60)

    sequence_lengths = config.comparison.sequence_lengths
    vocab_size = config.model.vocab_size
    embedding_dim = config.model.embedding_dim

    # Limit sequence lengths for baselines to prevent OOM
    baseline_seq_lengths = [s for s in sequence_lengths if s <= 2048]

    profiles = {}

    # Benchmark HSMN
    logger.info("\nBenchmarking HSMN (mamba_timecrystal_wlam_moe_hybrid)...")
    hsmn_model = harness.get_model()
    hsmn_timings = benchmark_model_scaling(
        hsmn_model, sequence_lengths, vocab_size, iterations=5 if quick else 10
    )
    profiles["hsmn"] = ArchitectureProfile(
        name="HSMN (Hybrid)",
        complexity="O(n)",
        description="Mamba + TimeCrystal + WLAM + MoE",
        timings_ms=hsmn_timings,
        sequence_lengths=sequence_lengths,
    )

    # Benchmark Transformer baseline
    if config.comparison.include_transformer and baseline_seq_lengths:
        logger.info("\nBenchmarking Transformer baseline (O(n²))...")
        try:
            transformer = create_transformer_baseline(
                embedding_dim=embedding_dim,
                num_layers=4,  # Fewer layers for comparison
                num_heads=8,
                vocab_size=vocab_size,
                max_seq_length=max(baseline_seq_lengths),
            )
            transformer_timings = benchmark_model_scaling(
                transformer, baseline_seq_lengths, vocab_size, iterations=5 if quick else 10
            )
            profiles["transformer"] = ArchitectureProfile(
                name="Transformer",
                complexity="O(n²)",
                description="Standard multi-head attention",
                timings_ms=transformer_timings,
                sequence_lengths=baseline_seq_lengths,
            )
        except Exception as e:
            logger.warning(f"Transformer baseline failed: {e}")

    # Benchmark Linear Attention baseline
    if config.comparison.include_linear_attention and baseline_seq_lengths:
        logger.info("\nBenchmarking Linear Attention baseline (O(n))...")
        try:
            linear_attn = create_linear_attention_baseline(
                embedding_dim=embedding_dim,
                num_layers=4,
                num_heads=8,
                vocab_size=vocab_size,
                max_seq_length=max(baseline_seq_lengths),
            )
            linear_timings = benchmark_model_scaling(
                linear_attn, baseline_seq_lengths, vocab_size, iterations=5 if quick else 10
            )
            profiles["linear_attention"] = ArchitectureProfile(
                name="Linear Attention",
                complexity="O(n)",
                description="ELU feature-map attention",
                timings_ms=linear_timings,
                sequence_lengths=baseline_seq_lengths,
            )
        except Exception as e:
            logger.warning(f"Linear attention baseline failed: {e}")

    # Analyze complexity for each architecture
    complexity_analysis = {}
    for name, profile in profiles.items():
        complexity_analysis[name] = analyze_complexity(profile)

    # Compute speedups vs transformer at common sequence lengths
    speedups = {}
    if "transformer" in profiles and "hsmn" in profiles:
        common_lengths = set(profiles["transformer"].sequence_lengths) & set(
            profiles["hsmn"].sequence_lengths
        )
        for seq_len in sorted(common_lengths):
            hsmn_idx = profiles["hsmn"].sequence_lengths.index(seq_len)
            trans_idx = profiles["transformer"].sequence_lengths.index(seq_len)
            hsmn_time = profiles["hsmn"].timings_ms[hsmn_idx]
            trans_time = profiles["transformer"].timings_ms[trans_idx]
            if hsmn_time > 0 and np.isfinite(hsmn_time) and np.isfinite(trans_time):
                speedups[seq_len] = trans_time / hsmn_time

    results = {
        "profiles": {
            name: {
                "name": p.name,
                "complexity": p.complexity,
                "description": p.description,
                "sequence_lengths": p.sequence_lengths,
                "timings_ms": p.timings_ms,
            }
            for name, p in profiles.items()
        },
        "complexity_analysis": complexity_analysis,
        "hsmn_vs_transformer_speedup": speedups,
        "model_config": config.model.to_dict(),
        "block_pattern": config.model.block_pattern,
    }

    # Summary
    if speedups:
        max_speedup = max(speedups.values())
        max_seq = max(speedups.keys(), key=lambda k: speedups[k])
        logger.info(f"\nMax HSMN speedup over Transformer: {max_speedup:.1f}x at seq_len={max_seq}")

    return results


def format_comparison_markdown(results: dict[str, Any]) -> str:
    """Format comparison results as markdown."""
    lines = [
        "# HSMN Architecture Comparison",
        "",
        f"**Block Pattern**: `{results['block_pattern']}`",
        "",
        "## Architecture Profiles",
        "",
        "| Architecture | Complexity | Description |",
        "|--------------|------------|-------------|",
    ]

    for name, profile in results["profiles"].items():
        lines.append(f"| {profile['name']} | {profile['complexity']} | {profile['description']} |")

    lines.extend(["", "## Timing Results (ms)", ""])

    # Create timing table
    all_seq_lengths = sorted(
        {seq for p in results["profiles"].values() for seq in p["sequence_lengths"]}
    )

    header = (
        "| Seq Length |" + "|".join(f" {p['name']} " for p in results["profiles"].values()) + "|"
    )
    separator = "|" + "|".join(["----------"] * (len(results["profiles"]) + 1)) + "|"

    lines.extend([header, separator])

    for seq_len in all_seq_lengths:
        row = f"| {seq_len} |"
        for profile in results["profiles"].values():
            if seq_len in profile["sequence_lengths"]:
                idx = profile["sequence_lengths"].index(seq_len)
                time_ms = profile["timings_ms"][idx]
                row += f" {time_ms:.1f} |" if np.isfinite(time_ms) else " OOM |"
            else:
                row += " - |"
        lines.append(row)

    lines.extend(["", "## Complexity Analysis", ""])

    for name, analysis in results["complexity_analysis"].items():
        if "fitted_exponent" in analysis:
            lines.append(
                f"- **{name}**: {analysis['complexity_class']} (exponent: {analysis['fitted_exponent']:.2f})"
            )

    if results["hsmn_vs_transformer_speedup"]:
        lines.extend(
            [
                "",
                "## HSMN Speedup vs Transformer",
                "",
                "| Seq Length | Speedup |",
                "|------------|---------|",
            ]
        )
        for seq_len, speedup in sorted(results["hsmn_vs_transformer_speedup"].items()):
            lines.append(f"| {seq_len} | {speedup:.1f}x |")

    return "\n".join(lines)


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="HSMN Architecture Comparison")
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

    results = run_comparison_benchmark(quick=args.quick)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_format in ("json", "both"):
        json_path = args.output_dir / "comparison_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {json_path}")

    if args.output_format in ("markdown", "both"):
        md_path = args.output_dir / "comparison_results.md"
        with open(md_path, "w") as f:
            f.write(format_comparison_markdown(results))
        print(f"Saved Markdown: {md_path}")

    if results["hsmn_vs_transformer_speedup"]:
        max_speedup = max(results["hsmn_vs_transformer_speedup"].values())
        print(f"\nMax HSMN speedup: {max_speedup:.1f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
