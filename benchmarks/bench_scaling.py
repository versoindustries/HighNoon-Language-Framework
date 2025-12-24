# benchmarks/bench_scaling.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Scaling Analysis Benchmark.

Tests model efficiency across different scales:
- Batch size scaling (throughput vs batch size)
- Sequence length scaling (throughput vs sequence length)
- Warmup latency (first token vs subsequent)
- Memory scaling
"""

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class ScalingResult:
    """Result from scaling analysis."""

    batch_scaling: dict[int, float]  # batch_size -> tokens/sec
    sequence_scaling: dict[int, float]  # seq_len -> tokens/sec
    warmup_latency_ms: float
    steady_state_latency_ms: float
    memory_scaling: dict[int, float]  # seq_len -> MB
    scaling_efficiency: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "scaling",
            "batch_scaling": self.batch_scaling,
            "sequence_scaling": self.sequence_scaling,
            "warmup_latency_ms": self.warmup_latency_ms,
            "steady_state_latency_ms": self.steady_state_latency_ms,
            "memory_scaling": self.memory_scaling,
            "scaling_efficiency": self.scaling_efficiency,
            "timestamp": self.timestamp,
        }


class ScalingAnalyzer:
    """Analyze model scaling properties."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size

    def measure_throughput(
        self,
        model: tf.keras.Model,
        batch_size: int,
        seq_length: int,
        warmup: int = 2,
        iterations: int = 5,
    ) -> float:
        """Measure throughput in tokens/second."""
        input_ids = tf.random.uniform(
            (batch_size, seq_length),
            minval=0,
            maxval=self.vocab_size,
            dtype=tf.int32,
        )

        # Warmup
        for _ in range(warmup):
            model(input_ids, training=False)

        # Measure
        start = time.perf_counter()
        for _ in range(iterations):
            model(input_ids, training=False)
        elapsed = time.perf_counter() - start

        total_tokens = batch_size * seq_length * iterations
        return total_tokens / elapsed if elapsed > 0 else 0.0

    def measure_batch_scaling(
        self,
        model: tf.keras.Model,
        batch_sizes: list[int],
        seq_length: int = 128,
    ) -> dict[int, float]:
        """Measure throughput vs batch size."""
        results = {}
        for bs in batch_sizes:
            try:
                throughput = self.measure_throughput(model, bs, seq_length)
                results[bs] = throughput
            except Exception as e:
                logger.warning(f"Batch size {bs} failed: {e}")
                break
        return results

    def measure_sequence_scaling(
        self,
        model: tf.keras.Model,
        seq_lengths: list[int],
        batch_size: int = 1,
    ) -> dict[int, float]:
        """Measure throughput vs sequence length."""
        results = {}
        for sl in seq_lengths:
            try:
                throughput = self.measure_throughput(model, batch_size, sl)
                results[sl] = throughput
            except Exception as e:
                logger.warning(f"Sequence length {sl} failed: {e}")
                break
        return results

    def measure_warmup_latency(
        self,
        model: tf.keras.Model,
        batch_size: int = 1,
        seq_length: int = 128,
    ) -> tuple[float, float]:
        """Measure first-call vs subsequent latency."""
        input_ids = tf.random.uniform(
            (batch_size, seq_length),
            minval=0,
            maxval=self.vocab_size,
            dtype=tf.int32,
        )

        # Clear cache
        gc.collect()

        # First call (warmup)
        start = time.perf_counter()
        model(input_ids, training=False)
        warmup_ms = (time.perf_counter() - start) * 1000

        # Subsequent calls
        times = []
        for _ in range(5):
            start = time.perf_counter()
            model(input_ids, training=False)
            times.append((time.perf_counter() - start) * 1000)

        steady_ms = float(np.mean(times))
        return warmup_ms, steady_ms

    def measure_memory_scaling(
        self,
        model: tf.keras.Model,
        seq_lengths: list[int],
        batch_size: int = 1,
    ) -> dict[int, float]:
        """Measure memory usage vs sequence length."""
        import tracemalloc

        results = {}

        for sl in seq_lengths:
            try:
                gc.collect()
                tracemalloc.start()

                input_ids = tf.random.uniform(
                    (batch_size, sl),
                    minval=0,
                    maxval=self.vocab_size,
                    dtype=tf.int32,
                )
                model(input_ids, training=False)

                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                results[sl] = peak / (1024 * 1024)  # MB
            except Exception as e:
                logger.warning(f"Memory measurement at {sl} failed: {e}")
                tracemalloc.stop()
                break

        return results

    def analyze_model(
        self,
        model: tf.keras.Model,
        quick: bool = False,
    ) -> ScalingResult:
        """Full scaling analysis."""
        batch_sizes = [1, 2, 4] if quick else [1, 2, 4, 8, 16]
        seq_lengths = [64, 128, 256] if quick else [64, 128, 256, 512, 1024]

        batch_scaling = self.measure_batch_scaling(model, batch_sizes)
        sequence_scaling = self.measure_sequence_scaling(model, seq_lengths)
        warmup_ms, steady_ms = self.measure_warmup_latency(model)
        memory_scaling = self.measure_memory_scaling(
            model, seq_lengths[:3] if quick else seq_lengths[:5]
        )

        # Calculate scaling efficiency (ideal = linear)
        if len(batch_scaling) >= 2:
            batch_vals = list(batch_scaling.values())
            efficiency = (
                batch_vals[-1]
                / batch_vals[0]
                / (list(batch_scaling.keys())[-1] / list(batch_scaling.keys())[0])
            )
        else:
            efficiency = 1.0

        return ScalingResult(
            batch_scaling=batch_scaling,
            sequence_scaling=sequence_scaling,
            warmup_latency_ms=warmup_ms,
            steady_state_latency_ms=steady_ms,
            memory_scaling=memory_scaling,
            scaling_efficiency=efficiency,
        )


def run_scaling_benchmark(
    model=None,
    quick: bool = False,
    verbose: bool = False,
    output_dir: str | Path | None = None,
) -> ScalingResult:
    """Run scaling analysis benchmark."""
    if verbose:
        print("=" * 60)
        print("Scaling Analysis")
        print("=" * 60)

    if model is None:
        from benchmarks.model_builder import build_benchmark_model

        model = build_benchmark_model(preset="small" if quick else "base")

    analyzer = ScalingAnalyzer()
    result = analyzer.analyze_model(model, quick=quick)

    if verbose:
        print("\nBatch Size Scaling (tok/s):")
        for bs, tps in result.batch_scaling.items():
            print(f"  {bs}: {tps:,.0f}")

        print("\nSequence Length Scaling (tok/s):")
        for sl, tps in result.sequence_scaling.items():
            print(f"  {sl}: {tps:,.0f}")

        print("\nLatency:")
        print(f"  Warmup: {result.warmup_latency_ms:.1f} ms")
        print(f"  Steady-state: {result.steady_state_latency_ms:.1f} ms")

        print("\nMemory Scaling (MB):")
        for sl, mb in result.memory_scaling.items():
            print(f"  {sl} tokens: {mb:.1f} MB")

        print(f"\nScaling Efficiency: {result.scaling_efficiency:.1%}")

    if output_dir:
        output_path = Path(output_dir) / "scaling_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Scaling Analysis")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--output-dir", type=str, default="benchmarks/reports")
    args = parser.parse_args()

    run_scaling_benchmark(quick=args.quick, verbose=True, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
