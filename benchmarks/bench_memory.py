# benchmarks/bench_memory.py
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

"""Memory profiling benchmarks for HSMN architecture (Production Mode).

Measures memory consumption using the full production architecture:
- Streaming Inference: O(1) memory via StreamingInferenceWrapper
- Training: O(log n) memory via Quantum Memory Replay (QMR)

Memory consumption should remain constant per chunk regardless of
total context length for streaming inference.

Example:
    >>> from benchmarks.bench_memory import run_memory_benchmark
    >>> results = run_memory_benchmark(quick=True)
    >>> print(f"Peak Memory: {results['peak_memory_mb']:.1f} MB")

Command-line usage:
    python benchmarks/bench_memory.py --quick
"""

import argparse
import gc
import json
import logging
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from benchmarks.benchmark_config import BenchmarkConfig
from benchmarks.benchmark_harness import BenchmarkHarness

logger = logging.getLogger(__name__)


def measure_model_memory(model: tf.keras.Model) -> dict[str, float]:
    """Measure memory consumed by model parameters.

    Args:
        model: HSMN model.

    Returns:
        Dictionary with memory breakdown.
    """
    total_params = 0
    trainable_params = 0

    for var in model.trainable_variables:
        total_params += np.prod(var.shape.as_list())
        trainable_params += np.prod(var.shape.as_list())

    for var in model.non_trainable_variables:
        total_params += np.prod(var.shape.as_list())

    # Assuming float32 (4 bytes per param)
    bytes_per_param = 4
    model_size_mb = (total_params * bytes_per_param) / (1024 * 1024)
    trainable_size_mb = (trainable_params * bytes_per_param) / (1024 * 1024)

    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "model_size_mb": float(model_size_mb),
        "trainable_size_mb": float(trainable_size_mb),
    }


def measure_streaming_memory(
    model: tf.keras.Model,
    batch_size: int,
    total_context_length: int,
    vocab_size: int,
    chunk_size: int = 1024,
) -> dict[str, Any]:
    """Measure memory during streaming inference.

    Uses StreamingInferenceWrapper to process long contexts in chunks,
    demonstrating O(1) memory consumption per chunk.

    Args:
        model: HSMN model.
        batch_size: Batch size.
        total_context_length: Total context length to process.
        vocab_size: Vocabulary size.
        chunk_size: Size of each processing chunk.

    Returns:
        Dictionary with memory measurements including per-chunk metrics.
    """
    from highnoon.inference.streaming import StreamingInferenceWrapper

    gc.collect()

    # Create streaming wrapper
    wrapper = StreamingInferenceWrapper(
        model,
        chunk_size=chunk_size,
        compression_ratio=1.0,  # No compression for benchmarking
        compress_every_n_chunks=0,  # Disable compression
    )

    num_chunks = (total_context_length + chunk_size - 1) // chunk_size
    chunk_memories = []
    chunk_times = []

    # Start memory tracking
    tracemalloc.start()
    initial_current, _ = tracemalloc.get_traced_memory()

    for chunk_idx in range(num_chunks):
        # Generate random input for this chunk
        actual_chunk_size = min(chunk_size, total_context_length - chunk_idx * chunk_size)
        input_ids = tf.random.uniform(
            [batch_size, actual_chunk_size], minval=0, maxval=vocab_size, dtype=tf.int32
        )

        # Reset memory tracking for this chunk
        gc.collect()
        tracemalloc.get_traced_memory()[0]
        chunk_start_time = time.perf_counter()

        # Process chunk through streaming wrapper
        _ = wrapper.process_chunk(input_ids, return_logits=True)

        chunk_end_time = time.perf_counter()
        chunk_end_mem = tracemalloc.get_traced_memory()[1]

        chunk_memories.append((chunk_end_mem - initial_current) / (1024 * 1024))
        chunk_times.append((chunk_end_time - chunk_start_time) * 1000)

        # Clean up input
        del input_ids

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Get streaming state memory
    state_memory_bytes = wrapper.get_memory_usage()
    tokens_processed = wrapper.get_context_length()

    gc.collect()

    return {
        "batch_size": batch_size,
        "total_context_length": total_context_length,
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "peak_memory_mb": float(peak / (1024 * 1024)),
        "state_memory_mb": float(state_memory_bytes / (1024 * 1024)),
        "tokens_processed": tokens_processed,
        "chunk_memories_mb": chunk_memories,
        "mean_chunk_memory_mb": float(np.mean(chunk_memories)),
        "max_chunk_memory_mb": float(np.max(chunk_memories)),
        "std_chunk_memory_mb": float(np.std(chunk_memories)),
        "chunk_times_ms": chunk_times,
        "mean_chunk_time_ms": float(np.mean(chunk_times)),
        "total_time_ms": float(sum(chunk_times)),
    }


def analyze_memory_scaling(
    model: tf.keras.Model,
    sequence_lengths: list[int],
    batch_size: int,
    vocab_size: int,
    chunk_size: int = 1024,
) -> dict[str, Any]:
    """Analyze how streaming memory scales with sequence length.

    For O(1) streaming models, memory should remain constant regardless
    of total context length (only dependent on chunk size).

    Args:
        model: HSMN model.
        sequence_lengths: Sequence lengths to test.
        batch_size: Batch size.
        vocab_size: Vocabulary size.
        chunk_size: Size of each processing chunk.

    Returns:
        Dictionary with scaling analysis.
    """
    measurements = []

    for seq_len in sequence_lengths:
        try:
            result = measure_streaming_memory(model, batch_size, seq_len, vocab_size, chunk_size)
            measurements.append(result)
            logger.info(
                f"  ctx_len={seq_len}: mean_chunk={result['mean_chunk_memory_mb']:.1f} MB, "
                f"state={result['state_memory_mb']:.3f} MB"
            )
        except Exception as e:
            logger.error(f"  ctx_len={seq_len}: FAILED - {e}")
            break

    if len(measurements) < 3:
        return {
            "scaling_type": "insufficient_data (need 3+ points)",
            "measurements": measurements,
        }

    # Analyze scaling using mean chunk memory
    seq_lens = np.array([m["total_context_length"] for m in measurements])
    chunk_mems = np.array([m["mean_chunk_memory_mb"] for m in measurements])
    state_mems = np.array([m["state_memory_mb"] for m in measurements])

    # For O(1) memory, chunk memory should be constant regardless of context length
    # Calculate coefficient of variation (CV) to detect constant behavior
    chunk_cv = np.std(chunk_mems) / np.mean(chunk_mems) if np.mean(chunk_mems) > 0 else 0
    state_cv = np.std(state_mems) / np.mean(state_mems) if np.mean(state_mems) > 0 else 0

    # Linear regression to check for growth
    if len(seq_lens) >= 3:
        log_seqs = np.log(seq_lens)
        log_chunk_mems = np.log(np.clip(chunk_mems, 1e-6, None))
        slope, intercept = np.polyfit(log_seqs, log_chunk_mems, 1)
    else:
        slope = 0.0

    # Classify scaling
    # O(1) constant: slope near 0 (< 0.1) and low CV (< 0.2)
    # O(n) linear: slope near 1
    # O(n²) quadratic: slope near 2
    if abs(slope) < 0.15 and chunk_cv < 0.25:
        scaling_type = "O(1) - Constant (Streaming)"
    elif slope < 0.5:
        scaling_type = "O(log n) - Logarithmic"
    elif slope < 1.2:
        scaling_type = "O(n) - Linear"
    elif slope < 1.7:
        scaling_type = "O(n log n) - Quasi-linear"
    else:
        scaling_type = "O(n²) - Quadratic"

    return {
        "scaling_type": scaling_type,
        "power_exponent": float(slope),
        "chunk_memory_cv": float(chunk_cv),
        "state_memory_cv": float(state_cv),
        "mean_chunk_memory_mb": float(np.mean(chunk_mems)),
        "mean_state_memory_mb": float(np.mean(state_mems)),
        "chunk_size": chunk_size,
        "is_constant_memory": abs(slope) < 0.15 and chunk_cv < 0.25,
        "measurements": [
            {
                "total_context_length": m["total_context_length"],
                "num_chunks": m["num_chunks"],
                "mean_chunk_memory_mb": m["mean_chunk_memory_mb"],
                "state_memory_mb": m["state_memory_mb"],
                "peak_memory_mb": m["peak_memory_mb"],
                "tokens_processed": m["tokens_processed"],
            }
            for m in measurements
        ],
    }


def compute_memory_efficiency(results: dict[str, Any]) -> dict[str, float]:
    """Compute memory efficiency metrics.

    Args:
        results: Memory benchmark results.

    Returns:
        Dictionary with efficiency metrics.
    """
    measurements = results.get("scaling_analysis", {}).get("measurements", [])

    if not measurements:
        return {}

    efficiencies = []
    for m in measurements:
        # Tokens per MB using mean chunk memory
        if m["mean_chunk_memory_mb"] > 0:
            tokens_per_mb = m["tokens_processed"] / m["mean_chunk_memory_mb"]
            efficiencies.append(
                {
                    "total_context_length": m["total_context_length"],
                    "tokens_per_mb": tokens_per_mb,
                    "tokens_per_gb": tokens_per_mb * 1024,
                }
            )

    if efficiencies:
        mean_tokens_per_gb = np.mean([e["tokens_per_gb"] for e in efficiencies])
        return {
            "mean_tokens_per_gb": float(mean_tokens_per_gb),
            "per_sequence_efficiency": efficiencies,
        }

    return {}


def run_memory_benchmark(
    config: BenchmarkConfig | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run complete memory benchmark suite using streaming inference.

    Args:
        config: Benchmark configuration.
        quick: Use quick configuration.

    Returns:
        Dictionary with all memory results.
    """
    if config is None:
        config = BenchmarkConfig.quick() if quick else BenchmarkConfig()

    harness = BenchmarkHarness(config)
    model = harness.get_model()

    logger.info("=" * 60)
    logger.info("HSMN Streaming Memory Benchmark")
    logger.info(f"Pattern: {config.model.block_pattern}")
    logger.info("Using: StreamingInferenceWrapper (O(1) memory)")
    logger.info("=" * 60)

    # Model parameter memory
    logger.info("Measuring model parameters...")
    model_memory = measure_model_memory(model)
    logger.info(f"  Model size: {model_memory['model_size_mb']:.1f} MB")
    logger.info(f"  Parameters: {model_memory['total_parameters']:,}")

    # Streaming memory scaling analysis
    logger.info("Measuring streaming memory scaling...")
    chunk_size = 1024
    scaling_analysis = analyze_memory_scaling(
        model,
        config.memory.sequence_lengths,
        config.memory.batch_size,
        config.model.vocab_size,
        chunk_size=chunk_size,
    )
    logger.info(f"  Scaling type: {scaling_analysis['scaling_type']}")
    logger.info(f"  Mean chunk memory: {scaling_analysis['mean_chunk_memory_mb']:.1f} MB")
    logger.info(f"  Mean state memory: {scaling_analysis['mean_state_memory_mb']:.3f} MB")

    if scaling_analysis.get("is_constant_memory"):
        logger.info("  ✅ Confirmed O(1) constant memory per chunk!")

    results = {
        "model_memory": model_memory,
        "scaling_analysis": scaling_analysis,
        "model_config": config.model.to_dict(),
        "block_pattern": config.model.block_pattern,
        "inference_mode": "streaming",
        "chunk_size": chunk_size,
    }

    # Compute efficiency
    efficiency = compute_memory_efficiency(results)
    if efficiency:
        results["efficiency"] = efficiency
        logger.info(f"  Efficiency: {efficiency['mean_tokens_per_gb']:.0f} tokens/GB")

    # Peak memory across all tests
    if scaling_analysis.get("measurements"):
        peak = max(m["peak_memory_mb"] for m in scaling_analysis["measurements"])
        results["peak_memory_mb"] = peak
        logger.info(f"\nPeak Memory: {peak:.1f} MB")

    return results


def format_memory_markdown(results: dict[str, Any]) -> str:
    """Format memory results as markdown."""
    lines = [
        "# HSMN Streaming Memory Benchmark",
        "",
        f"**Block Pattern**: `{results['block_pattern']}`",
        f"**Inference Mode**: `{results.get('inference_mode', 'streaming')}`",
        f"**Chunk Size**: {results.get('chunk_size', 1024)} tokens",
        "",
        "## Model Size",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Total Parameters** | {results['model_memory']['total_parameters']:,} |",
        f"| **Model Size** | {results['model_memory']['model_size_mb']:.1f} MB |",
        f"| **Trainable Size** | {results['model_memory']['trainable_size_mb']:.1f} MB |",
        "",
    ]

    if "scaling_analysis" in results:
        sa = results["scaling_analysis"]
        is_constant = sa.get("is_constant_memory", False)
        status = "✅ Constant" if is_constant else "⚠️ Variable"

        lines.extend(
            [
                "## Streaming Memory Scaling",
                "",
                f"**Scaling Type**: {sa['scaling_type']}",
                f"**Memory Behavior**: {status}",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Power Exponent | {sa.get('power_exponent', 0):.3f} |",
                f"| Chunk Memory CV | {sa.get('chunk_memory_cv', 0):.3f} |",
                f"| Mean Chunk Memory | {sa.get('mean_chunk_memory_mb', 0):.1f} MB |",
                f"| Mean State Memory | {sa.get('mean_state_memory_mb', 0):.3f} MB |",
                "",
            ]
        )

        if sa.get("measurements"):
            lines.extend(
                [
                    "### Measurements",
                    "",
                    "| Context Length | Chunks | Chunk Memory (MB) | State Memory (MB) | Peak (MB) |",
                    "|----------------|--------|-------------------|-------------------|-----------|",
                ]
            )
            for m in sa["measurements"]:
                lines.append(
                    f"| {m['total_context_length']:,} | {m['num_chunks']} | "
                    f"{m['mean_chunk_memory_mb']:.1f} | {m['state_memory_mb']:.3f} | "
                    f"{m['peak_memory_mb']:.1f} |"
                )
            lines.append("")

    if "efficiency" in results:
        eff = results["efficiency"]
        lines.extend(
            [
                "## Memory Efficiency",
                "",
                f"**Mean Efficiency**: {eff['mean_tokens_per_gb']:.0f} tokens/GB",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="HSMN Streaming Memory Benchmark")
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

    results = run_memory_benchmark(quick=args.quick)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_format in ("json", "both"):
        json_path = args.output_dir / "memory_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {json_path}")

    if args.output_format in ("markdown", "both"):
        md_path = args.output_dir / "memory_results.md"
        with open(md_path, "w") as f:
            f.write(format_memory_markdown(results))
        print(f"Saved Markdown: {md_path}")

    if "peak_memory_mb" in results:
        print(f"\nPeak Memory: {results['peak_memory_mb']:.1f} MB")

    scaling_type = results.get("scaling_analysis", {}).get("scaling_type", "Unknown")
    print(f"Scaling: {scaling_type}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
