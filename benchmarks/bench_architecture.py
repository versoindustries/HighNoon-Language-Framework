# benchmarks/bench_architecture.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Architecture Analysis Benchmark.

Comprehensive architecture profiling including:
- Parameter count breakdown by component
- FLOPs estimation
- Memory footprint
- Layer composition analysis
- Initialization quality verification
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
class ArchitectureResult:
    """Result from architecture analysis."""

    total_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int
    parameter_breakdown: dict[str, int]
    estimated_flops: int
    memory_mb: float
    layer_types: dict[str, int]
    init_quality: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "architecture",
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "non_trainable_parameters": self.non_trainable_parameters,
            "parameter_breakdown": self.parameter_breakdown,
            "estimated_flops": self.estimated_flops,
            "memory_mb": self.memory_mb,
            "layer_types": self.layer_types,
            "init_quality": self.init_quality,
            "timestamp": self.timestamp,
        }


class ArchitectureAnalyzer:
    """Analyze model architecture properties."""

    def analyze_parameters(self, model: tf.keras.Model) -> dict[str, int]:
        """Breakdown parameters by component type."""
        breakdown = {}

        for var in model.trainable_variables:
            # Categorize by layer type
            name_lower = var.name.lower()

            if "embed" in name_lower:
                category = "embedding"
            elif (
                "attention" in name_lower
                or "query" in name_lower
                or "key" in name_lower
                or "value" in name_lower
            ):
                category = "attention"
            elif "moe" in name_lower or "expert" in name_lower:
                category = "moe"
            elif "mamba" in name_lower or "ssm" in name_lower:
                category = "mamba_ssm"
            elif "wlam" in name_lower or "wavelet" in name_lower:
                category = "wlam"
            elif "dense" in name_lower or "ffn" in name_lower:
                category = "dense_ffn"
            elif "norm" in name_lower or "layer_norm" in name_lower:
                category = "normalization"
            elif "lm_head" in name_lower or "output" in name_lower:
                category = "output_head"
            else:
                category = "other"

            param_count = int(np.prod(var.shape))
            breakdown[category] = breakdown.get(category, 0) + param_count

        return breakdown

    def estimate_flops(
        self,
        model: tf.keras.Model,
        batch_size: int = 1,
        seq_length: int = 512,
    ) -> int:
        """Estimate FLOPs for forward pass."""
        # Rough estimation based on parameter count and operations
        total_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)

        # Approximate: 2 FLOPs per parameter per token (multiply-add)
        estimated_flops = total_params * seq_length * batch_size * 2

        return estimated_flops

    def check_init_quality(self, model: tf.keras.Model) -> dict[str, Any]:
        """Check weight initialization quality."""
        results = {
            "mean_check": True,
            "std_check": True,
            "details": {},
        }

        for var in model.trainable_variables[:10]:  # Sample first 10
            arr = var.numpy()
            mean = float(np.mean(arr))
            std = float(np.std(arr))

            # Check for reasonable initialization
            mean_ok = abs(mean) < 0.1
            std_ok = 0.01 < std < 2.0

            results["details"][var.name] = {
                "mean": mean,
                "std": std,
                "mean_ok": mean_ok,
                "std_ok": std_ok,
            }

            if not mean_ok:
                results["mean_check"] = False
            if not std_ok:
                results["std_check"] = False

        return results

    def count_layer_types(self, model: tf.keras.Model) -> dict[str, int]:
        """Count layers by type."""
        type_counts = {}

        for layer in model.layers:
            layer_type = type(layer).__name__
            type_counts[layer_type] = type_counts.get(layer_type, 0) + 1

        return type_counts

    def analyze_model(self, model: tf.keras.Model) -> ArchitectureResult:
        """Full architecture analysis."""
        total_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
        non_trainable = sum(int(np.prod(v.shape)) for v in model.non_trainable_variables)

        # Memory estimation (float32 = 4 bytes per param)
        memory_mb = (total_params + non_trainable) * 4 / (1024 * 1024)

        return ArchitectureResult(
            total_parameters=total_params + non_trainable,
            trainable_parameters=total_params,
            non_trainable_parameters=non_trainable,
            parameter_breakdown=self.analyze_parameters(model),
            estimated_flops=self.estimate_flops(model),
            memory_mb=memory_mb,
            layer_types=self.count_layer_types(model),
            init_quality=self.check_init_quality(model),
        )


def run_architecture_benchmark(
    model=None,
    quick: bool = False,
    verbose: bool = False,
    output_dir: str | Path | None = None,
) -> ArchitectureResult:
    """Run architecture analysis benchmark."""
    if verbose:
        print("=" * 60)
        print("Architecture Analysis")
        print("=" * 60)

    if model is None:
        from benchmarks.model_builder import build_benchmark_model

        model = build_benchmark_model(preset="small" if quick else "base")

    analyzer = ArchitectureAnalyzer()
    result = analyzer.analyze_model(model)

    if verbose:
        print("\nParameter Count:")
        print(f"  Total: {result.total_parameters:,}")
        print(f"  Trainable: {result.trainable_parameters:,}")
        print(f"  Non-trainable: {result.non_trainable_parameters:,}")
        print(f"  Memory: {result.memory_mb:.2f} MB")

        print("\nParameter Breakdown:")
        for category, count in sorted(result.parameter_breakdown.items(), key=lambda x: -x[1]):
            pct = (
                count / result.trainable_parameters * 100 if result.trainable_parameters > 0 else 0
            )
            print(f"  {category}: {count:,} ({pct:.1f}%)")

        print(f"\nEstimated FLOPs: {result.estimated_flops:,}")

        print("\nLayer Types:")
        for ltype, count in sorted(result.layer_types.items()):
            print(f"  {ltype}: {count}")

        init_status = (
            "✓" if result.init_quality["mean_check"] and result.init_quality["std_check"] else "⚠️"
        )
        print(f"\nInitialization Quality: {init_status}")

    if output_dir:
        output_path = Path(output_dir) / "architecture_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Architecture Analysis")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--output-dir", type=str, default="benchmarks/reports")
    args = parser.parse_args()

    run_architecture_benchmark(quick=args.quick, verbose=True, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
