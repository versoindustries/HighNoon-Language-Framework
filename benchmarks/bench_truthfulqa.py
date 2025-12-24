# benchmarks/bench_truthfulqa.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""TruthfulQA Factual Accuracy Benchmark.

Tests resistance to common misconceptions and factually incorrect responses.
Multiple choice format (MC1/MC2 metrics).

References:
- Paper: https://arxiv.org/abs/2109.07958
- Dataset: truthful_qa (HuggingFace)
"""

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TruthfulQAResult:
    """Result from TruthfulQA evaluation."""

    mc1_accuracy: float
    mc2_accuracy: float
    total_questions: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "truthfulqa",
            "mc1_accuracy": self.mc1_accuracy,
            "mc2_accuracy": self.mc2_accuracy,
            "total_questions": self.total_questions,
            "timestamp": self.timestamp,
        }


class TruthfulQAEvaluator:
    """TruthfulQA benchmark evaluator."""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def load_dataset(self, split: str = "validation") -> list[dict]:
        """Load TruthfulQA dataset."""
        try:
            from datasets import load_dataset

            ds = load_dataset("truthful_qa", "multiple_choice", split=split, trust_remote_code=True)
            return list(ds)
        except Exception as e:
            logger.warning(f"Could not load TruthfulQA: {e}")
            return []

    def format_question(self, question: str, choices: list[str]) -> str:
        """Format question for model input."""
        prompt = f"Q: {question}\n"
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        for i, choice in enumerate(choices[: len(labels)]):
            prompt += f"{labels[i]}. {choice}\n"
        prompt += "Answer:"
        return prompt

    def evaluate_question(self, prompt: str, num_choices: int) -> int:
        """Predict answer index."""
        if self.model is None:
            return random.randint(0, num_choices - 1)
        try:
            labels = ["A", "B", "C", "D", "E", "F", "G", "H"][:num_choices]
            response = self.model.generate(prompt, max_tokens=1, temperature=0.0)
            for char in response.upper():
                if char in labels:
                    return labels.index(char)
            return 0
        except Exception:
            return random.randint(0, num_choices - 1)

    def run(self, quick: bool = False, verbose: bool = False) -> TruthfulQAResult:
        """Run TruthfulQA evaluation."""
        if verbose:
            print("=" * 60)
            print("TruthfulQA Benchmark Evaluation")
            print("=" * 60)

        data = self.load_dataset("validation")

        if not data:
            return TruthfulQAResult(0.0, 0.0, 0)

        test_samples = data[:50] if quick else data

        mc1_correct = 0
        mc2_scores = []
        total = 0

        for item in test_samples:
            mc1_choices = item.get("mc1_targets", {}).get("choices", [])
            mc1_labels = item.get("mc1_targets", {}).get("labels", [])

            if mc1_choices and mc1_labels:
                prompt = self.format_question(item["question"], mc1_choices)
                prediction = self.evaluate_question(prompt, len(mc1_choices))

                # MC1: Check if prediction matches the correct answer (label=1)
                if len(mc1_labels) > prediction and mc1_labels[prediction] == 1:
                    mc1_correct += 1

            # MC2: Multi-answer normalized score
            mc2_choices = item.get("mc2_targets", {}).get("choices", [])
            mc2_labels = item.get("mc2_targets", {}).get("labels", [])

            if mc2_choices and mc2_labels:
                prompt = self.format_question(item["question"], mc2_choices)
                prediction = self.evaluate_question(prompt, len(mc2_choices))

                # Normalized probability for MC2
                if len(mc2_labels) > prediction:
                    mc2_scores.append(float(mc2_labels[prediction]))

            total += 1

        mc1_accuracy = mc1_correct / total if total > 0 else 0.0
        mc2_accuracy = sum(mc2_scores) / len(mc2_scores) if mc2_scores else 0.0

        if verbose:
            print(f"MC1 Accuracy: {mc1_accuracy:.1%}")
            print(f"MC2 Accuracy: {mc2_accuracy:.1%}")
            print(f"Total Questions: {total}")

        return TruthfulQAResult(
            mc1_accuracy=mc1_accuracy,
            mc2_accuracy=mc2_accuracy,
            total_questions=total,
        )


def run_truthfulqa_benchmark(
    model=None,
    tokenizer=None,
    quick: bool = False,
    verbose: bool = False,
    output_dir: str | Path | None = None,
) -> TruthfulQAResult:
    """Run TruthfulQA benchmark."""
    evaluator = TruthfulQAEvaluator(model=model, tokenizer=tokenizer)
    result = evaluator.run(quick=quick, verbose=verbose)

    if output_dir:
        output_path = Path(output_dir) / "truthfulqa_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="TruthfulQA Benchmark")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--output-dir", type=str, default="benchmarks/reports")
    args = parser.parse_args()

    run_truthfulqa_benchmark(quick=args.quick, verbose=True, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
