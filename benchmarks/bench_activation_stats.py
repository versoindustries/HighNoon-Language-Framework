# benchmarks/bench_activation_stats.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Activation Statistics Benchmark.

Monitors activation distributions per layer to detect:
- Dead neurons (always zero)
- Saturated activations (always at bounds)
- Abnormal distributions (high variance, biased mean)
- Layer-wise activation flow

Works on untrained models to validate initialization.
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class ActivationStatsResult:
    """Result from activation statistics analysis."""

    health_score: float
    layer_stats: dict[str, dict[str, float]]
    dead_neuron_ratio: float
    saturated_ratio: float
    total_activations: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "activation_stats",
            "health_score": self.health_score,
            "layer_stats": self.layer_stats,
            "dead_neuron_ratio": self.dead_neuron_ratio,
            "saturated_ratio": self.saturated_ratio,
            "total_activations": self.total_activations,
            "timestamp": self.timestamp,
        }


class ActivationAnalyzer:
    """Analyze activation statistics through model layers."""

    def __init__(self):
        self.activation_records = {}

    def record_activations(self, name: str, tensor: tf.Tensor):
        """Record activation statistics for a layer."""
        arr = tensor.numpy().flatten()
        self.activation_records[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "dead_ratio": float(np.mean(arr == 0)),
            "saturated_ratio": float(np.mean(np.abs(arr) > 0.99 * np.max(np.abs(arr)))),
            "size": len(arr),
        }

    def analyze_model(
        self,
        model: tf.keras.Model,
        batch_size: int = 2,
        seq_length: int = 128,
        vocab_size: int = 32000,
    ) -> ActivationStatsResult:
        """Analyze activation statistics through model.

        Args:
            model: TensorFlow/Keras model to analyze.
            batch_size: Batch size for test input.
            seq_length: Sequence length for test input.
            vocab_size: Vocabulary size for random input.

        Returns:
            ActivationStatsResult with activation statistics.
        """
        # Create random input
        input_ids = tf.random.uniform(
            (batch_size, seq_length),
            minval=0,
            maxval=vocab_size,
            dtype=tf.int32,
        )

        # Forward pass with activation recording
        self.activation_records = {}

        # Get intermediate activations using submodels
        x = input_ids

        # Track through model layers
        for layer in model.layers:
            try:
                x = layer(x, training=False)
                if isinstance(x, tf.Tensor):
                    self.record_activations(layer.name, x)
            except Exception:
                continue

        # Calculate overall statistics
        total_dead = 0
        total_saturated = 0
        total_size = 0

        for _name, stats in self.activation_records.items():
            total_dead += stats["dead_ratio"] * stats["size"]
            total_saturated += stats["saturated_ratio"] * stats["size"]
            total_size += stats["size"]

        dead_ratio = total_dead / total_size if total_size > 0 else 0.0
        saturated_ratio = total_saturated / total_size if total_size > 0 else 0.0

        # Health score (penalize dead and saturated)
        health_score = max(0.0, 1.0 - dead_ratio - saturated_ratio)

        return ActivationStatsResult(
            health_score=health_score,
            layer_stats=self.activation_records,
            dead_neuron_ratio=dead_ratio,
            saturated_ratio=saturated_ratio,
            total_activations=total_size,
        )


def run_activation_stats_benchmark(
    model=None,
    quick: bool = False,
    verbose: bool = False,
    output_dir: str | Path | None = None,
) -> ActivationStatsResult:
    """Run activation statistics benchmark."""
    if verbose:
        print("=" * 60)
        print("Activation Statistics Analysis")
        print("=" * 60)

    if model is None:
        from benchmarks.model_builder import build_benchmark_model

        model = build_benchmark_model(preset="small" if quick else "base")

    analyzer = ActivationAnalyzer()
    result = analyzer.analyze_model(
        model,
        batch_size=2,
        seq_length=64 if quick else 256,
    )

    if verbose:
        print(f"\nHealth Score: {result.health_score:.1%}")
        print(f"Dead Neuron Ratio: {result.dead_neuron_ratio:.2%}")
        print(f"Saturated Ratio: {result.saturated_ratio:.2%}")
        print(f"Total Activations: {result.total_activations:,}")

        print("\nPer-Layer Statistics (sample):")
        for i, (name, stats) in enumerate(result.layer_stats.items()):
            if i >= 5:
                print(f"  ... and {len(result.layer_stats) - 5} more layers")
                break
            print(f"  {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    if output_dir:
        output_path = Path(output_dir) / "activation_stats_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Activation Statistics Analysis")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--output-dir", type=str, default="benchmarks/reports")
    args = parser.parse_args()

    run_activation_stats_benchmark(quick=args.quick, verbose=True, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
