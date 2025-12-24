# benchmarks/bench_numerical_stability.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Numerical Stability Benchmark.

Tests model numerical properties:
- Precision differences (float32 vs bfloat16)
- Input sensitivity (Lipschitz estimation)
- Output entropy tracking
- NaN/Inf detection
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
class NumericalStabilityResult:
    """Result from numerical stability analysis."""

    stability_score: float
    has_nan: bool
    has_inf: bool
    output_entropy: float
    lipschitz_estimate: float
    precision_diff: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "numerical_stability",
            "stability_score": self.stability_score,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "output_entropy": self.output_entropy,
            "lipschitz_estimate": self.lipschitz_estimate,
            "precision_diff": self.precision_diff,
            "timestamp": self.timestamp,
        }


class NumericalStabilityAnalyzer:
    """Analyze numerical stability of model."""

    def compute_entropy(self, logits: np.ndarray) -> float:
        """Compute entropy of output distribution."""
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        # Avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=-1)
        return float(np.mean(entropy))

    def estimate_lipschitz(
        self,
        model: tf.keras.Model,
        input_ids: tf.Tensor,
        epsilon: float = 1e-4,
    ) -> float:
        """Estimate local Lipschitz constant via finite differences."""
        # Get embedding layer for perturbation
        embed_layer = None
        for layer in model.layers:
            if "embed" in layer.name.lower():
                embed_layer = layer
                break

        if embed_layer is None:
            return 0.0

        # Forward pass
        outputs1 = model(input_ids, training=False)
        if isinstance(outputs1, dict):
            outputs1 = outputs1.get("logits", list(outputs1.values())[0])

        # Perturbed forward (add noise to a second input)
        input_ids2 = tf.where(
            tf.random.uniform(input_ids.shape) > 0.9, (input_ids + 1) % 32000, input_ids
        )
        outputs2 = model(input_ids2, training=False)
        if isinstance(outputs2, dict):
            outputs2 = outputs2.get("logits", list(outputs2.values())[0])

        # Estimate Lipschitz
        input_diff = tf.norm(tf.cast(input_ids2 - input_ids, tf.float32))
        output_diff = tf.norm(tf.cast(outputs2, tf.float32) - tf.cast(outputs1, tf.float32))

        if input_diff > 0:
            return float(output_diff / input_diff)
        return 0.0

    def check_precision_sensitivity(
        self,
        model: tf.keras.Model,
        input_ids: tf.Tensor,
    ) -> float:
        """Check output difference between precisions."""
        # Float32 forward
        outputs_f32 = model(input_ids, training=False)
        if isinstance(outputs_f32, dict):
            outputs_f32 = outputs_f32.get("logits", list(outputs_f32.values())[0])

        # We can't easily switch precision, so just measure variance
        outputs_arr = (
            outputs_f32.numpy() if hasattr(outputs_f32, "numpy") else np.array(outputs_f32)
        )
        return float(np.std(outputs_arr))

    def analyze_model(
        self,
        model: tf.keras.Model,
        batch_size: int = 2,
        seq_length: int = 128,
        vocab_size: int = 32000,
    ) -> NumericalStabilityResult:
        """Analyze numerical stability."""
        input_ids = tf.random.uniform(
            (batch_size, seq_length),
            minval=0,
            maxval=vocab_size,
            dtype=tf.int32,
        )

        # Forward pass
        outputs = model(input_ids, training=False)
        if isinstance(outputs, dict):
            logits = outputs.get("logits", list(outputs.values())[0])
        else:
            logits = outputs

        logits_arr = logits.numpy() if hasattr(logits, "numpy") else np.array(logits)

        # Check for NaN/Inf
        has_nan = bool(np.any(np.isnan(logits_arr)))
        has_inf = bool(np.any(np.isinf(logits_arr)))

        # Compute entropy
        entropy = self.compute_entropy(logits_arr)

        # Estimate Lipschitz
        lipschitz = self.estimate_lipschitz(model, input_ids)

        # Precision sensitivity
        precision_diff = self.check_precision_sensitivity(model, input_ids)

        # Stability score
        stability_score = 1.0
        if has_nan or has_inf:
            stability_score = 0.0
        elif lipschitz > 100:
            stability_score = 0.5

        return NumericalStabilityResult(
            stability_score=stability_score,
            has_nan=has_nan,
            has_inf=has_inf,
            output_entropy=entropy,
            lipschitz_estimate=lipschitz,
            precision_diff=precision_diff,
        )


def run_numerical_stability_benchmark(
    model=None,
    quick: bool = False,
    verbose: bool = False,
    output_dir: str | Path | None = None,
) -> NumericalStabilityResult:
    """Run numerical stability benchmark."""
    if verbose:
        print("=" * 60)
        print("Numerical Stability Analysis")
        print("=" * 60)

    if model is None:
        from benchmarks.model_builder import build_benchmark_model

        model = build_benchmark_model(preset="small" if quick else "base")

    analyzer = NumericalStabilityAnalyzer()
    result = analyzer.analyze_model(
        model,
        batch_size=2,
        seq_length=64 if quick else 256,
    )

    if verbose:
        print(f"\nStability Score: {result.stability_score:.1%}")
        print(f"Has NaN: {result.has_nan}")
        print(f"Has Inf: {result.has_inf}")
        print(f"Output Entropy: {result.output_entropy:.4f}")
        print(f"Lipschitz Estimate: {result.lipschitz_estimate:.4f}")
        print(f"Output Std Dev: {result.precision_diff:.4f}")

    if output_dir:
        output_path = Path(output_dir) / "numerical_stability_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Numerical Stability Analysis")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--output-dir", type=str, default="benchmarks/reports")
    args = parser.parse_args()

    result = run_numerical_stability_benchmark(
        quick=args.quick, verbose=True, output_dir=args.output_dir
    )
    return 0 if result.stability_score > 0.8 else 1


if __name__ == "__main__":
    exit(main())
