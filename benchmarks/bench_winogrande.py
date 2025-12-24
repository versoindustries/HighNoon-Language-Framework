# benchmarks/bench_winogrande.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""WinoGrande Coreference Resolution Benchmark.

Tests commonsense reasoning through pronoun disambiguation.
5-shot evaluation with binary choice (1 or 2).

References:
- Paper: https://arxiv.org/abs/1907.10641
- Dataset: winogrande (HuggingFace)
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
class WinoGrandeResult:
    """Result from WinoGrande evaluation."""

    accuracy: float
    total_questions: int
    correct_answers: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "winogrande",
            "accuracy": self.accuracy,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "timestamp": self.timestamp,
        }


class WinoGrandeEvaluator:
    """WinoGrande benchmark with 5-shot protocol."""

    def __init__(self, model=None, tokenizer=None, num_shots: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.num_shots = num_shots

    def load_dataset(self, split: str = "validation") -> list[dict]:
        """Load WinoGrande dataset."""
        try:
            from datasets import load_dataset

            ds = load_dataset("winogrande", "winogrande_xl", split=split, trust_remote_code=True)
            return list(ds)
        except Exception as e:
            logger.warning(f"Could not load WinoGrande: {e}")
            return []

    def format_question(
        self, sentence: str, opt1: str, opt2: str, answer: str | None = None
    ) -> str:
        """Format question for model input."""
        prompt = f"Sentence: {sentence}\n"
        prompt += f"Option 1: {opt1}\n"
        prompt += f"Option 2: {opt2}\n"
        prompt += "Which option correctly fills the blank? Answer with 1 or 2:"
        if answer is not None:
            prompt += f" {answer}"
        return prompt

    def evaluate_question(self, prompt: str) -> str:
        """Predict answer (1 or 2)."""
        if self.model is None:
            return random.choice(["1", "2"])
        try:
            response = self.model.generate(prompt, max_tokens=1, temperature=0.0)
            for char in response:
                if char in "12":
                    return char
            return "1"
        except Exception:
            return random.choice(["1", "2"])

    def run(self, quick: bool = False, verbose: bool = False) -> WinoGrandeResult:
        """Run WinoGrande evaluation."""
        if verbose:
            print("=" * 60)
            print("WinoGrande Benchmark Evaluation")
            print("=" * 60)

        train_data = self.load_dataset("train")[: self.num_shots]
        val_data = self.load_dataset("validation")

        if not val_data:
            return WinoGrandeResult(0.0, 0, 0)

        test_samples = val_data[:100] if quick else val_data

        correct = 0
        total = 0

        for item in test_samples:
            prompt = "Choose the correct option to complete the sentence.\n\n"
            for ex in train_data[: self.num_shots]:
                prompt += (
                    self.format_question(ex["sentence"], ex["option1"], ex["option2"], ex["answer"])
                    + "\n\n"
                )

            prompt += self.format_question(item["sentence"], item["option1"], item["option2"])

            prediction = self.evaluate_question(prompt)
            if prediction == item["answer"]:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        if verbose:
            print(f"Accuracy: {accuracy:.1%} ({correct}/{total})")

        return WinoGrandeResult(
            accuracy=accuracy,
            total_questions=total,
            correct_answers=correct,
        )


def run_winogrande_benchmark(
    model=None,
    tokenizer=None,
    quick: bool = False,
    verbose: bool = False,
    output_dir: str | Path | None = None,
) -> WinoGrandeResult:
    """Run WinoGrande benchmark."""
    evaluator = WinoGrandeEvaluator(model=model, tokenizer=tokenizer)
    result = evaluator.run(quick=quick, verbose=verbose)

    if output_dir:
        output_path = Path(output_dir) / "winogrande_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="WinoGrande Benchmark")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--output-dir", type=str, default="benchmarks/reports")
    args = parser.parse_args()

    run_winogrande_benchmark(quick=args.quick, verbose=True, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
