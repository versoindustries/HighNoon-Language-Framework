# benchmarks/bench_leaderboard.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Open LLM Leaderboard Combined Evaluation.

Runs all Open LLM Leaderboard benchmarks and computes combined score.
Follows HuggingFace Open LLM Leaderboard methodology.

Benchmarks included:
- MMLU (5-shot)
- HellaSwag (10-shot)
- ARC-Challenge (25-shot)
- TruthfulQA (0-shot, MC2)
- WinoGrande (5-shot)
- GSM8K (5-shot, CoT)
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.bench_arc import run_arc_benchmark
from benchmarks.bench_gsm8k import run_gsm8k_benchmark
from benchmarks.bench_hellaswag import run_hellaswag_benchmark
from benchmarks.bench_mmlu import run_mmlu_benchmark
from benchmarks.bench_truthfulqa import run_truthfulqa_benchmark
from benchmarks.bench_winogrande import run_winogrande_benchmark

logger = logging.getLogger(__name__)


@dataclass
class LeaderboardResult:
    """Combined Open LLM Leaderboard results."""

    average_score: float
    mmlu: float
    hellaswag: float
    arc_challenge: float
    truthfulqa_mc2: float
    winogrande: float
    gsm8k: float
    total_time_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "open_llm_leaderboard",
            "average_score": self.average_score,
            "scores": {
                "mmlu": self.mmlu,
                "hellaswag": self.hellaswag,
                "arc_challenge": self.arc_challenge,
                "truthfulqa_mc2": self.truthfulqa_mc2,
                "winogrande": self.winogrande,
                "gsm8k": self.gsm8k,
            },
            "total_time_seconds": self.total_time_seconds,
            "timestamp": self.timestamp,
        }


def run_leaderboard_evaluation(
    model=None,
    tokenizer=None,
    quick: bool = False,
    verbose: bool = True,
    output_dir: str | Path | None = None,
) -> LeaderboardResult:
    """Run all Open LLM Leaderboard benchmarks.

    Args:
        model: Language model instance.
        tokenizer: Tokenizer instance.
        quick: Quick mode for validation.
        verbose: Verbose output.
        output_dir: Output directory for results.

    Returns:
        LeaderboardResult with all scores and average.
    """
    import time

    start_time = time.time()

    if verbose:
        print("=" * 70)
        print("   Open LLM Leaderboard Evaluation")
        print("=" * 70)
        print()

    results = {}

    # 1. MMLU
    if verbose:
        print("▸ Running MMLU (5-shot)...")
    mmlu_result = run_mmlu_benchmark(
        model=model, tokenizer=tokenizer, quick=quick, verbose=verbose, output_dir=output_dir
    )
    results["mmlu"] = mmlu_result.overall_accuracy

    # 2. HellaSwag
    if verbose:
        print("\n▸ Running HellaSwag (10-shot)...")
    hellaswag_result = run_hellaswag_benchmark(
        model=model, tokenizer=tokenizer, quick=quick, verbose=verbose, output_dir=output_dir
    )
    results["hellaswag"] = hellaswag_result.accuracy

    # 3. ARC-Challenge
    if verbose:
        print("\n▸ Running ARC-Challenge (25-shot)...")
    arc_result = run_arc_benchmark(
        model=model, tokenizer=tokenizer, quick=quick, verbose=verbose, output_dir=output_dir
    )
    results["arc_challenge"] = arc_result.challenge_accuracy

    # 4. TruthfulQA
    if verbose:
        print("\n▸ Running TruthfulQA (MC2)...")
    truthfulqa_result = run_truthfulqa_benchmark(
        model=model, tokenizer=tokenizer, quick=quick, verbose=verbose, output_dir=output_dir
    )
    results["truthfulqa_mc2"] = truthfulqa_result.mc2_accuracy

    # 5. WinoGrande
    if verbose:
        print("\n▸ Running WinoGrande (5-shot)...")
    winogrande_result = run_winogrande_benchmark(
        model=model, tokenizer=tokenizer, quick=quick, verbose=verbose, output_dir=output_dir
    )
    results["winogrande"] = winogrande_result.accuracy

    # 6. GSM8K
    if verbose:
        print("\n▸ Running GSM8K (5-shot, CoT)...")
    gsm8k_result = run_gsm8k_benchmark(
        model=model, tokenizer=tokenizer, quick=quick, verbose=verbose, output_dir=output_dir
    )
    results["gsm8k"] = gsm8k_result.accuracy

    # Calculate average
    average = sum(results.values()) / len(results)

    total_time = time.time() - start_time

    if verbose:
        print()
        print("=" * 70)
        print("   FINAL RESULTS")
        print("=" * 70)
        for name, score in results.items():
            print(f"  {name:20s}: {score:6.2%}")
        print("-" * 70)
        print(f"  {'AVERAGE':20s}: {average:6.2%}")
        print(f"\n  Total time: {total_time:.1f} seconds")

    result = LeaderboardResult(
        average_score=average,
        mmlu=results["mmlu"],
        hellaswag=results["hellaswag"],
        arc_challenge=results["arc_challenge"],
        truthfulqa_mc2=results["truthfulqa_mc2"],
        winogrande=results["winogrande"],
        gsm8k=results["gsm8k"],
        total_time_seconds=total_time,
    )

    if output_dir:
        output_path = Path(output_dir) / "leaderboard_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        if verbose:
            print(f"\n  Results saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Open LLM Leaderboard Evaluation")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--quick", action="store_true", help="Quick validation mode")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="benchmarks/reports")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # Load model if specified
    model = None
    tokenizer = None
    if args.model:
        try:
            from highnoon import load_model

            model, tokenizer = load_model(args.model)
        except Exception as e:
            logger.warning(f"Could not load model: {e}")

    result = run_leaderboard_evaluation(
        model=model,
        tokenizer=tokenizer,
        quick=args.quick,
        verbose=args.verbose,
        output_dir=args.output_dir,
    )

    return 0 if result.average_score > 0 else 1


if __name__ == "__main__":
    exit(main())
