# benchmarks/bench_gradient_flow.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Gradient Flow Analysis Benchmark.

Tests gradient propagation through the model architecture to detect:
- Vanishing gradients (gradient norms → 0)
- Exploding gradients (gradient norms → ∞)
- Dead layers (zero gradients)
- Gradient distribution per layer

Works on untrained models to validate architecture choices.
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
class GradientFlowResult:
    """Result from gradient flow analysis."""

    health_score: float  # 0-1, higher is better
    layer_gradient_norms: dict[str, float]
    vanishing_layers: list[str]
    exploding_layers: list[str]
    dead_layers: list[str]
    gradient_stats: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "gradient_flow",
            "health_score": self.health_score,
            "layer_gradient_norms": self.layer_gradient_norms,
            "vanishing_layers": self.vanishing_layers,
            "exploding_layers": self.exploding_layers,
            "dead_layers": self.dead_layers,
            "gradient_stats": self.gradient_stats,
            "timestamp": self.timestamp,
        }


class GradientFlowAnalyzer:
    """Analyze gradient flow through model layers."""

    def __init__(
        self,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 1e3,
    ):
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold

    def analyze_model(
        self,
        model: tf.keras.Model,
        batch_size: int = 2,
        seq_length: int = 128,
        vocab_size: int = 32000,
    ) -> GradientFlowResult:
        """Analyze gradient flow through model.

        Args:
            model: TensorFlow/Keras model to analyze.
            batch_size: Batch size for test input.
            seq_length: Sequence length for test input.
            vocab_size: Vocabulary size for random input.

        Returns:
            GradientFlowResult with gradient statistics.
        """
        # Create random input
        input_ids = tf.random.uniform(
            (batch_size, seq_length),
            minval=0,
            maxval=vocab_size,
            dtype=tf.int32,
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            outputs = model(input_ids, training=True)
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("output", list(outputs.values())[0]))
            else:
                logits = outputs

            # Simple loss: mean of outputs
            loss = tf.reduce_mean(tf.cast(logits, tf.float32))

        gradients = tape.gradient(loss, model.trainable_variables)

        # Analyze gradients per layer
        layer_norms = {}
        vanishing = []
        exploding = []
        dead = []

        all_norms = []

        for var, grad in zip(model.trainable_variables, gradients):
            if grad is None:
                dead.append(var.name)
                layer_norms[var.name] = 0.0
                continue

            grad_norm = float(tf.norm(grad).numpy())
            layer_norms[var.name] = grad_norm
            all_norms.append(grad_norm)

            if grad_norm < self.vanishing_threshold:
                vanishing.append(var.name)
            elif grad_norm > self.exploding_threshold:
                exploding.append(var.name)

        # Calculate health score
        healthy_layers = (
            len(model.trainable_variables) - len(vanishing) - len(exploding) - len(dead)
        )
        total_layers = len(model.trainable_variables)
        health_score = healthy_layers / total_layers if total_layers > 0 else 0.0

        # Gradient statistics
        stats = {
            "mean_norm": float(np.mean(all_norms)) if all_norms else 0.0,
            "std_norm": float(np.std(all_norms)) if all_norms else 0.0,
            "min_norm": float(np.min(all_norms)) if all_norms else 0.0,
            "max_norm": float(np.max(all_norms)) if all_norms else 0.0,
            "total_layers": total_layers,
            "healthy_layers": healthy_layers,
        }

        return GradientFlowResult(
            health_score=health_score,
            layer_gradient_norms=layer_norms,
            vanishing_layers=vanishing,
            exploding_layers=exploding,
            dead_layers=dead,
            gradient_stats=stats,
        )


def run_gradient_flow_benchmark(
    model=None,
    quick: bool = False,
    verbose: bool = False,
    output_dir: str | Path | None = None,
) -> GradientFlowResult:
    """Run gradient flow analysis benchmark."""
    if verbose:
        print("=" * 60)
        print("Gradient Flow Analysis")
        print("=" * 60)

    if model is None:
        # Create a simple test model
        from benchmarks.model_builder import build_benchmark_model

        model = build_benchmark_model(preset="small" if quick else "base")

    analyzer = GradientFlowAnalyzer()
    result = analyzer.analyze_model(
        model,
        batch_size=2,
        seq_length=64 if quick else 256,
    )

    if verbose:
        print(f"\nHealth Score: {result.health_score:.1%}")
        print("Gradient Statistics:")
        for key, val in result.gradient_stats.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.6f}")
            else:
                print(f"  {key}: {val}")

        if result.vanishing_layers:
            print(f"\n⚠️  Vanishing gradients in {len(result.vanishing_layers)} layers")
        if result.exploding_layers:
            print(f"⚠️  Exploding gradients in {len(result.exploding_layers)} layers")
        if result.dead_layers:
            print(f"⚠️  Dead layers: {len(result.dead_layers)}")

    if output_dir:
        output_path = Path(output_dir) / "gradient_flow_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Gradient Flow Analysis")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--output-dir", type=str, default="benchmarks/reports")
    args = parser.parse_args()

    result = run_gradient_flow_benchmark(quick=args.quick, verbose=True, output_dir=args.output_dir)
    return 0 if result.health_score > 0.8 else 1


if __name__ == "__main__":
    exit(main())
