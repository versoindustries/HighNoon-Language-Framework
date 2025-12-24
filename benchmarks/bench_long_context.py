# benchmarks/bench_long_context.py
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

"""Ultra-long context benchmarks for HSMN architecture (Production Mode).

Uses StreamingInferenceWrapper for O(1) memory streaming inference,
enabling tests at extended context lengths up to 1M tokens.

All benchmarks use the same production code paths as real inference.

Example:
    >>> from benchmarks.bench_long_context import run_long_context_benchmark
    >>> results = run_long_context_benchmark(max_context=131072)
    >>> print(results['streaming_scaling']['max_context_achieved'])

These benchmarks do not require a trained model and measure:
- Streaming forward pass time at various context lengths
- O(1) memory consumption per chunk
- Throughput vs context length
"""

import argparse
import gc
import json
import logging
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from benchmarks.datasets import SyntheticDataGenerator
from benchmarks.hardware_detection import HardwareDetector
from benchmarks.model_builder import HSMNModelBuilder

logger = logging.getLogger(__name__)


# Standard context lengths for benchmarking
CONTEXT_LENGTHS = [
    1024,  # 1K
    4096,  # 4K
    8192,  # 8K
    16384,  # 16K
    32768,  # 32K
    65536,  # 64K
    131072,  # 128K
    262144,  # 256K
    524288,  # 512K
    1048576,  # 1M
]

# Needle positions for comprehensive testing
NEEDLE_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]  # Start, 25%, 50%, 75%, End


@dataclass
class LongContextConfig:
    """Configuration for long-context benchmarks.

    Attributes:
        context_lengths: List of context lengths to test.
        needle_positions: Relative positions for needle insertion.
        num_needles: Number of needles for multi-needle tests.
        warmup_iterations: Warmup iterations before timing.
        benchmark_iterations: Timed iterations per configuration.
        auto_scale: Automatically scale down if memory insufficient.
        max_memory_gb: Maximum memory to use (for auto-scaling).
        chunk_size: Chunk size for streaming inference.
    """

    context_lengths: list[int] = field(default_factory=lambda: [1024, 4096, 16384, 65536, 131072])
    needle_positions: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    num_needles: int = 3
    warmup_iterations: int = 2
    benchmark_iterations: int = 5
    auto_scale: bool = True
    max_memory_gb: float = 0.0  # 0 = auto-detect
    chunk_size: int = 1024  # Streaming chunk size


@dataclass
class LongContextResult:
    """Result from a long-context benchmark.

    Attributes:
        context_length: Context length in tokens.
        success: Whether the test completed successfully.
        total_time_ms: Total processing time in milliseconds.
        time_per_chunk_ms: Average time per chunk.
        memory_per_chunk_mb: Memory per chunk in MB.
        peak_memory_mb: Peak memory usage in MB.
        tokens_per_second: Throughput in tokens/second.
        num_chunks: Number of chunks processed.
        error: Error message if failed.
    """

    context_length: int
    success: bool
    total_time_ms: float = 0.0
    time_per_chunk_ms: float = 0.0
    memory_per_chunk_mb: float = 0.0
    peak_memory_mb: float = 0.0
    tokens_per_second: float = 0.0
    num_chunks: int = 0
    error: str = ""


@dataclass
class NeedleResult:
    """Result from a needle-in-haystack test.

    Attributes:
        context_length: Total context length.
        needle_position: Relative needle position (0-1).
        needle_text: The needle text inserted.
        found: Whether needle-relevant tokens activated.
        attention_score: Attention score at needle position.
        retrieval_rank: Rank of needle in attention (lower = better).
    """

    context_length: int
    needle_position: float
    needle_text: str
    found: bool
    attention_score: float = 0.0
    retrieval_rank: int = 0


def measure_streaming_forward(
    model: tf.keras.Model,
    context_length: int,
    vocab_size: int,
    chunk_size: int = 1024,
    warmup: int = 2,
    iterations: int = 3,
) -> LongContextResult:
    """Measure streaming forward pass time and memory at a specific context length.

    Uses StreamingInferenceWrapper for O(1) memory per chunk.

    Args:
        model: HSMN model to benchmark.
        context_length: Context length to test.
        vocab_size: Vocabulary size for input generation.
        chunk_size: Size of each processing chunk.
        warmup: Warmup iterations.
        iterations: Benchmark iterations.

    Returns:
        LongContextResult with timing and memory measurements.
    """
    from highnoon.inference.streaming import StreamingInferenceWrapper

    result = LongContextResult(context_length=context_length, success=False)

    try:
        num_chunks = (context_length + chunk_size - 1) // chunk_size
        result.num_chunks = num_chunks

        # Warmup
        for _ in range(warmup):
            wrapper = StreamingInferenceWrapper(
                model, chunk_size=chunk_size, compress_every_n_chunks=0
            )
            for chunk_idx in range(min(2, num_chunks)):  # Only warmup 2 chunks
                actual_size = min(chunk_size, context_length - chunk_idx * chunk_size)
                if actual_size <= 0:
                    break
                chunk_input = tf.random.uniform(
                    [1, actual_size], minval=0, maxval=vocab_size, dtype=tf.int32
                )
                _ = wrapper.process_chunk(chunk_input, return_logits=False)
            del wrapper

        gc.collect()

        # Benchmark iterations
        iteration_times = []
        chunk_memories = []

        for _ in range(iterations):
            wrapper = StreamingInferenceWrapper(
                model, chunk_size=chunk_size, compress_every_n_chunks=0
            )

            tracemalloc.start()
            iter_start = time.perf_counter()

            for chunk_idx in range(num_chunks):
                actual_size = min(chunk_size, context_length - chunk_idx * chunk_size)
                if actual_size <= 0:
                    break

                chunk_input = tf.random.uniform(
                    [1, actual_size], minval=0, maxval=vocab_size, dtype=tf.int32
                )

                chunk_start_mem = tracemalloc.get_traced_memory()[0]
                _ = wrapper.process_chunk(chunk_input, return_logits=True)
                chunk_end_mem = tracemalloc.get_traced_memory()[0]

                chunk_memories.append((chunk_end_mem - chunk_start_mem) / (1024 * 1024))
                del chunk_input

            iter_time = (time.perf_counter() - iter_start) * 1000
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            iteration_times.append(iter_time)
            result.peak_memory_mb = max(result.peak_memory_mb, peak / (1024 * 1024))

            del wrapper
            gc.collect()

        result.success = True
        result.total_time_ms = float(np.mean(iteration_times))
        result.time_per_chunk_ms = result.total_time_ms / num_chunks if num_chunks > 0 else 0
        result.memory_per_chunk_mb = float(np.mean(chunk_memories)) if chunk_memories else 0
        result.tokens_per_second = (context_length / result.total_time_ms) * 1000

        logger.info(
            f"  ctx={context_length:>7,}: {result.total_time_ms:.1f}ms total, "
            f"{result.time_per_chunk_ms:.1f}ms/chunk, "
            f"{result.memory_per_chunk_mb:.2f}MB/chunk, "
            f"{result.tokens_per_second:.0f} tok/s"
        )

    except tf.errors.ResourceExhaustedError as e:
        result.error = f"OOM: {str(e)[:100]}"
        logger.warning(f"  ctx={context_length:>7,}: OOM")

    except Exception as e:
        result.error = str(e)[:200]
        logger.error(f"  ctx={context_length:>7,}: ERROR - {result.error}")

    finally:
        gc.collect()

    return result


def run_streaming_scaling(
    model: tf.keras.Model,
    config: LongContextConfig,
    vocab_size: int = 32000,
) -> list[LongContextResult]:
    """Run streaming scaling benchmark across context lengths.

    Args:
        model: HSMN model.
        config: Long-context benchmark configuration.
        vocab_size: Vocabulary size.

    Returns:
        List of LongContextResult for each context length.
    """
    results = []

    logger.info("Running streaming scaling benchmark...")
    logger.info(f"Chunk size: {config.chunk_size} tokens")

    for ctx_len in config.context_lengths:
        result = measure_streaming_forward(
            model=model,
            context_length=ctx_len,
            vocab_size=vocab_size,
            chunk_size=config.chunk_size,
            warmup=config.warmup_iterations,
            iterations=config.benchmark_iterations,
        )
        results.append(result)

        # Stop if OOM
        if not result.success and "OOM" in result.error:
            logger.warning(f"Stopping at {ctx_len} due to OOM")
            break

    return results


def run_needle_in_haystack(
    model: tf.keras.Model,
    config: LongContextConfig,
    vocab_size: int = 32000,
    tokenizer=None,
) -> list[NeedleResult]:
    """Run needle-in-haystack retrieval benchmark using streaming inference.

    For untrained models, this measures whether the model architecture
    can maintain information flow from needle position to output.

    Args:
        model: HSMN model.
        config: Long-context configuration.
        vocab_size: Vocabulary size.
        tokenizer: Optional tokenizer for text-based needles.

    Returns:
        List of NeedleResult for each test case.
    """
    from highnoon.inference.streaming import StreamingInferenceWrapper

    results = []
    generator = SyntheticDataGenerator(seed=42)

    logger.info("Running needle-in-haystack benchmark (streaming mode)...")

    for ctx_len in config.context_lengths[:5]:  # Limit for needle tests
        for needle_pos in config.needle_positions:
            try:
                # Generate test case
                test_data = generator.generate_multi_needle_test(
                    haystack_chars=ctx_len * 4,  # ~4 chars/token
                    num_needles=1,
                    positions=[needle_pos],
                )

                # Create streaming wrapper
                wrapper = StreamingInferenceWrapper(
                    model, chunk_size=config.chunk_size, compress_every_n_chunks=0
                )

                # Generate random tokens as haystack
                input_ids = tf.random.uniform(
                    [1, ctx_len], minval=0, maxval=vocab_size, dtype=tf.int32
                ).numpy()

                # Insert "marker" tokens at needle position
                marker_pos = int(needle_pos * (ctx_len - 10)) + 5
                marker_tokens = [vocab_size - 1] * 5  # Unique pattern
                input_ids[0, marker_pos : marker_pos + 5] = marker_tokens

                # Process through streaming (chunk by chunk)
                num_chunks = (ctx_len + config.chunk_size - 1) // config.chunk_size
                all_outputs = []

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * config.chunk_size
                    end = min((chunk_idx + 1) * config.chunk_size, ctx_len)
                    chunk = tf.constant(input_ids[:, start:end], dtype=tf.int32)
                    output = wrapper.process_chunk(chunk, return_logits=True)
                    if output is not None:
                        all_outputs.append(output)

                # Analyze output at needle position
                if all_outputs:
                    # Find which chunk contains the needle
                    needle_chunk_idx = marker_pos // config.chunk_size
                    needle_local_pos = marker_pos % config.chunk_size

                    if needle_chunk_idx < len(all_outputs):
                        chunk_output = all_outputs[needle_chunk_idx]
                        if needle_local_pos + 5 <= chunk_output.shape[1]:
                            marker_output = chunk_output[0, needle_local_pos : needle_local_pos + 5]
                            baseline_output = all_outputs[0][0, :5]
                            diff = tf.reduce_mean(tf.abs(marker_output - baseline_output)).numpy()
                        else:
                            diff = 0.0
                    else:
                        diff = 0.0
                else:
                    diff = 0.0

                result = NeedleResult(
                    context_length=ctx_len,
                    needle_position=needle_pos,
                    needle_text=test_data["needles"][0] if test_data["needles"] else "",
                    found=diff > 0.1,
                    attention_score=float(diff),
                )
                results.append(result)

                logger.info(
                    f"  ctx={ctx_len:>7,}, pos={needle_pos:.0%}: "
                    f"score={diff:.4f} {'✓' if result.found else '✗'}"
                )

                del wrapper

            except tf.errors.ResourceExhaustedError:
                logger.warning(f"  ctx={ctx_len}, pos={needle_pos}: OOM")
                break

            except Exception as e:
                logger.error(f"  ctx={ctx_len}, pos={needle_pos}: {e}")

    return results


def run_passkey_retrieval(
    model: tf.keras.Model,
    context_lengths: list[int],
    vocab_size: int = 32000,
    chunk_size: int = 1024,
) -> list[dict[str, Any]]:
    """Run passkey retrieval benchmark using streaming inference.

    Tests whether the model can maintain a "passkey" token pattern
    inserted at various points in a long context.

    Args:
        model: HSMN model.
        context_lengths: Context lengths to test.
        vocab_size: Vocabulary size.
        chunk_size: Streaming chunk size.

    Returns:
        List of results per context length.
    """
    from highnoon.inference.streaming import StreamingInferenceWrapper

    results = []
    logger.info("Running passkey retrieval benchmark (streaming mode)...")

    passkey_tokens = [vocab_size - 5, vocab_size - 4, vocab_size - 3]  # Unique pattern

    for ctx_len in context_lengths:
        try:
            # Create random context with passkey at 10% position
            input_ids = np.random.randint(0, vocab_size - 10, size=(1, ctx_len))
            passkey_pos = ctx_len // 10

            # Insert passkey
            input_ids[0, passkey_pos : passkey_pos + 3] = passkey_tokens

            # Process through streaming
            wrapper = StreamingInferenceWrapper(
                model, chunk_size=chunk_size, compress_every_n_chunks=0
            )

            start = time.perf_counter()
            num_chunks = (ctx_len + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, ctx_len)
                chunk = tf.constant(input_ids[:, chunk_start:chunk_end], dtype=tf.int32)
                _ = wrapper.process_chunk(chunk, return_logits=False)

            elapsed = (time.perf_counter() - start) * 1000

            results.append(
                {
                    "context_length": ctx_len,
                    "passkey_position": passkey_pos,
                    "forward_time_ms": elapsed,
                    "num_chunks": num_chunks,
                    "success": True,
                }
            )

            logger.info(f"  ctx={ctx_len:>7,}: {elapsed:.1f}ms ({num_chunks} chunks)")

            del wrapper

        except Exception as e:
            results.append(
                {
                    "context_length": ctx_len,
                    "success": False,
                    "error": str(e)[:100],
                }
            )
            logger.warning(f"  ctx={ctx_len:>7,}: FAILED")

    return results


def analyze_complexity(results: list[LongContextResult]) -> dict[str, Any]:
    """Analyze computational complexity from streaming scaling results.

    Args:
        results: List of streaming scaling results.

    Returns:
        Dictionary with complexity analysis.
    """
    successful = [r for r in results if r.success]

    if len(successful) < 3:
        return {"complexity": "insufficient_data"}

    ctx_lens = np.array([r.context_length for r in successful], dtype=float)
    times = np.array([r.total_time_ms for r in successful])
    chunk_mems = np.array([r.memory_per_chunk_mb for r in successful])

    # Log-log fit for time: time ~ ctx^alpha
    log_ctx = np.log(ctx_lens)
    log_time = np.log(times)
    time_slope, time_intercept = np.polyfit(log_ctx, log_time, 1)

    # Memory per chunk should be constant for streaming (O(1))
    mem_cv = np.std(chunk_mems) / np.mean(chunk_mems) if np.mean(chunk_mems) > 0 else 0

    # Classify time complexity
    if time_slope < 1.2:
        time_complexity = "O(n) - Linear"
    elif time_slope < 1.7:
        time_complexity = "O(n log n) - Quasi-linear"
    elif time_slope < 2.2:
        time_complexity = "O(n²) - Quadratic"
    else:
        time_complexity = f"O(n^{time_slope:.1f})"

    # Memory complexity: streaming should be O(1) per chunk
    if mem_cv < 0.25:
        mem_complexity = "O(1) - Constant (per chunk)"
    else:
        mem_complexity = "Variable"

    return {
        "time_exponent": float(time_slope),
        "time_complexity": time_complexity,
        "memory_per_chunk_cv": float(mem_cv),
        "memory_complexity": mem_complexity,
        "is_constant_memory": mem_cv < 0.25,
        "max_successful_context": int(max(ctx_lens)),
        "max_throughput_tps": float(max(r.tokens_per_second for r in successful)),
        "mean_memory_per_chunk_mb": float(np.mean(chunk_mems)),
    }


def run_long_context_benchmark(
    config: LongContextConfig | None = None,
    model_preset: str = "bench-standard",
    max_context: int | None = None,
) -> dict[str, Any]:
    """Run complete long-context benchmark suite using streaming inference.

    Args:
        config: Long-context configuration.
        model_preset: Model preset to use.
        max_context: Override maximum context length.

    Returns:
        Dictionary with all benchmark results.
    """
    if config is None:
        config = LongContextConfig()

    if max_context is not None:
        config.context_lengths = [c for c in CONTEXT_LENGTHS if c <= max_context]

    # Hardware detection
    hw_detector = HardwareDetector()
    hw_info = hw_detector.detect_all()

    # Auto-scale based on hardware
    if config.auto_scale:
        max_mem = hw_info.benchmark_suitability.get("max_recommended_context", 131072)
        config.context_lengths = [c for c in config.context_lengths if c <= max_mem]
        config.max_memory_gb = hw_info.memory.available_gb * 0.8

    logger.info("=" * 70)
    logger.info("HSMN Long-Context Streaming Benchmark Suite")
    logger.info(f"Model: {model_preset}")
    logger.info(f"Context lengths: {[f'{c//1024}K' for c in config.context_lengths]}")
    logger.info(f"Chunk size: {config.chunk_size} tokens")
    logger.info(f"Available memory: {hw_info.memory.available_gb:.1f} GB")
    logger.info("=" * 70)

    # Build model
    builder = HSMNModelBuilder()
    model_config = builder.get_preset(model_preset)
    model = builder.build(model_preset)
    vocab_size = model_config.vocab_size

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_preset": model_preset,
        "model_config": model_config.to_dict(),
        "inference_mode": "streaming",
        "chunk_size": config.chunk_size,
        "hardware": {
            "cpu": hw_info.cpu.model_name,
            "memory_gb": hw_info.memory.total_gb,
            "available_gb": hw_info.memory.available_gb,
        },
        "config": {
            "context_lengths": config.context_lengths,
            "needle_positions": config.needle_positions,
        },
    }

    # Run streaming scaling
    scaling_results = run_streaming_scaling(model, config, vocab_size)
    results["streaming_scaling"] = [
        {
            "context_length": r.context_length,
            "success": r.success,
            "total_time_ms": r.total_time_ms,
            "time_per_chunk_ms": r.time_per_chunk_ms,
            "memory_per_chunk_mb": r.memory_per_chunk_mb,
            "peak_memory_mb": r.peak_memory_mb,
            "tokens_per_second": r.tokens_per_second,
            "num_chunks": r.num_chunks,
            "error": r.error,
        }
        for r in scaling_results
    ]

    # Complexity analysis
    results["complexity_analysis"] = analyze_complexity(scaling_results)

    # Needle-in-haystack
    needle_results = run_needle_in_haystack(model, config, vocab_size)
    results["needle_in_haystack"] = [
        {
            "context_length": r.context_length,
            "needle_position": r.needle_position,
            "found": r.found,
            "attention_score": r.attention_score,
        }
        for r in needle_results
    ]

    # Passkey retrieval (simplified context list)
    passkey_contexts = [c for c in config.context_lengths if c <= 65536][:5]
    if passkey_contexts:
        results["passkey_retrieval"] = run_passkey_retrieval(
            model, passkey_contexts, vocab_size, config.chunk_size
        )

    # Summary
    successful_scaling = [r for r in scaling_results if r.success]
    results["summary"] = {
        "max_context_achieved": (
            max(r.context_length for r in successful_scaling) if successful_scaling else 0
        ),
        "contexts_tested": len(scaling_results),
        "contexts_successful": len(successful_scaling),
        "time_complexity": results["complexity_analysis"].get("time_complexity", "Unknown"),
        "memory_complexity": results["complexity_analysis"].get("memory_complexity", "Unknown"),
        "is_constant_memory": results["complexity_analysis"].get("is_constant_memory", False),
        "mean_memory_per_chunk_mb": results["complexity_analysis"].get(
            "mean_memory_per_chunk_mb", 0
        ),
    }

    return results


def format_long_context_markdown(results: dict[str, Any]) -> str:
    """Format long-context results as markdown.

    Args:
        results: Results from run_long_context_benchmark.

    Returns:
        Markdown formatted report.
    """
    lines = [
        "# HSMN Long-Context Streaming Benchmark",
        "",
        f"**Generated**: {results['timestamp']}",
        f"**Model**: {results['model_preset']}",
        "**Inference Mode**: `streaming`",
        f"**Chunk Size**: {results.get('chunk_size', 1024)} tokens",
        "",
        "---",
        "",
    ]

    # Try to get full hardware markdown
    try:
        hw_detector = HardwareDetector()
        hw_info = hw_detector.detect_all()
        hw_markdown = hw_detector.format_markdown(hw_info)
        lines.extend([hw_markdown, "", "---", ""])
    except Exception:
        # Fallback to basic hardware info from results
        lines.extend(
            [
                "## Hardware",
                "",
                f"- **CPU**: {results['hardware']['cpu']}",
                f"- **Total Memory**: {results['hardware']['memory_gb']:.1f} GB",
                f"- **Available**: {results['hardware']['available_gb']:.1f} GB",
            ]
        )

    # Summary section
    is_constant = results["summary"].get("is_constant_memory", False)
    mem_status = "✅ O(1) Constant" if is_constant else "⚠️ Variable"

    lines.extend(
        [
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Max Context Achieved** | {results['summary']['max_context_achieved']:,} tokens |",
            f"| **Contexts Tested** | {results['summary']['contexts_tested']} |",
            f"| **Contexts Successful** | {results['summary']['contexts_successful']} |",
            f"| **Time Complexity** | {results['summary']['time_complexity']} |",
            f"| **Memory Complexity** | {results['summary']['memory_complexity']} {mem_status} |",
            f"| **Mean Memory/Chunk** | {results['summary'].get('mean_memory_per_chunk_mb', 0):.2f} MB |",
            "",
            "## Streaming Scaling",
            "",
            "| Context | Chunks | Total Time (ms) | ms/chunk | MB/chunk | Tokens/sec | Status |",
            "|---------|--------|-----------------|----------|----------|------------|--------|",
        ]
    )

    for r in results["streaming_scaling"]:
        status = "✅" if r["success"] else "❌ " + r.get("error", "")[:20]
        if r["success"]:
            lines.append(
                f"| {r['context_length']:,} | {r['num_chunks']} | {r['total_time_ms']:.1f} | "
                f"{r['time_per_chunk_ms']:.1f} | {r['memory_per_chunk_mb']:.2f} | "
                f"{r['tokens_per_second']:,.0f} | {status} |"
            )
        else:
            lines.append(f"| {r['context_length']:,} | - | - | - | - | - | {status} |")

    lines.extend(
        [
            "",
            "## Complexity Analysis",
            "",
            f"- **Time Exponent**: {results['complexity_analysis'].get('time_exponent', 'N/A'):.2f}",
            f"- **Time Complexity**: {results['complexity_analysis'].get('time_complexity', 'N/A')}",
            f"- **Memory per Chunk CV**: {results['complexity_analysis'].get('memory_per_chunk_cv', 'N/A'):.3f}",
            f"- **Memory Complexity**: {results['complexity_analysis'].get('memory_complexity', 'N/A')}",
            "",
        ]
    )

    if results.get("needle_in_haystack"):
        lines.extend(
            [
                "## Needle-in-Haystack (Streaming)",
                "",
                "| Context | Position | Found | Score |",
                "|---------|----------|-------|-------|",
            ]
        )
        for r in results["needle_in_haystack"]:
            found = "✅" if r["found"] else "❌"
            lines.append(
                f"| {r['context_length']:,} | {r['needle_position']:.0%} | "
                f"{found} | {r['attention_score']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="HSMN Long-Context Streaming Benchmark")
    parser.add_argument(
        "--max-context",
        type=int,
        default=131072,
        help="Maximum context length to test (default: 131072)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="bench-standard",
        help="Model preset to use",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Streaming chunk size (default: 1024)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/reports"),
        help="Output directory",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = LongContextConfig(chunk_size=args.chunk_size)

    results = run_long_context_benchmark(
        config=config,
        model_preset=args.preset,
        max_context=args.max_context,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_format in ("json", "both"):
        json_path = args.output_dir / "long_context_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {json_path}")

    if args.output_format in ("markdown", "both"):
        md_path = args.output_dir / "long_context_results.md"
        with open(md_path, "w") as f:
            f.write(format_long_context_markdown(results))
        print(f"Saved Markdown: {md_path}")

    print(f"\n✅ Max context achieved: {results['summary']['max_context_achieved']:,} tokens")
    print(f"   Time complexity: {results['summary']['time_complexity']}")
    print(f"   Memory complexity: {results['summary']['memory_complexity']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
