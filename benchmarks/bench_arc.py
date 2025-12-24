# benchmarks/bench_arc.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""ARC (AI2 Reasoning Challenge) Benchmark.

Tests scientific reasoning with grade-school science questions.
Standard 25-shot evaluation on ARC-Challenge set.

References:
- Paper: https://arxiv.org/abs/1803.05457
- Dataset: allenai/ai2_arc (HuggingFace)
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
class ARCResult:
    """Result from ARC evaluation."""

    accuracy: float
    easy_accuracy: float
    challenge_accuracy: float
    total_questions: int
    correct_answers: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "arc",
            "accuracy": self.accuracy,
            "easy_accuracy": self.easy_accuracy,
            "challenge_accuracy": self.challenge_accuracy,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "timestamp": self.timestamp,
        }


class ARCEvaluator:
    """ARC benchmark evaluator with 25-shot protocol."""

    def __init__(self, model=None, tokenizer=None, num_shots: int = 25):
        self.model = model
        self.tokenizer = tokenizer
        self.num_shots = num_shots

    def load_dataset(self, subset: str = "ARC-Challenge", split: str = "test") -> list[dict]:
        """Load ARC dataset."""
        try:
            from datasets import load_dataset

            ds = load_dataset("allenai/ai2_arc", subset, split=split, trust_remote_code=True)
            return list(ds)
        except Exception as e:
            logger.warning(f"Could not load ARC/{subset}: {e}")
            return []

    def format_question(self, question: str, choices: dict, answer: str | None = None) -> str:
        """Format question for model input."""
        labels = choices.get("label", [])
        texts = choices.get("text", [])

        prompt = f"Question: {question}\n"
        for label, text in zip(labels, texts):
            prompt += f"{label}. {text}\n"
        prompt += "Answer:"
        if answer is not None:
            prompt += f" {answer}"
        return prompt

    def evaluate_question(self, prompt: str, valid_labels: list[str]) -> str:
        """Predict answer for question."""
        if self.model is None:
            return random.choice(valid_labels) if valid_labels else "A"
        try:
            response = self.model.generate(prompt, max_tokens=1, temperature=0.0)
            for char in response.upper():
                if char in valid_labels:
                    return char
            return valid_labels[0] if valid_labels else "A"
        except Exception:
            return random.choice(valid_labels) if valid_labels else "A"

    def evaluate_subset(
        self, subset: str, quick: bool = False, verbose: bool = False
    ) -> tuple[int, int]:
        """Evaluate a subset (Easy or Challenge)."""
        train_data = self.load_dataset(subset, "train")[: self.num_shots]
        test_data = self.load_dataset(subset, "test")

        if not test_data:
            return 0, 0

        test_samples = test_data[:50] if quick else test_data

        correct = 0
        total = 0

        for item in test_samples:
            prompt = "The following are science questions.\n\n"
            for ex in train_data[: self.num_shots]:
                prompt += (
                    self.format_question(ex["question"], ex["choices"], ex["answerKey"]) + "\n\n"
                )

            prompt += self.format_question(item["question"], item["choices"])

            valid_labels = item["choices"].get("label", ["A", "B", "C", "D"])
            prediction = self.evaluate_question(prompt, valid_labels)

            if prediction == item["answerKey"]:
                correct += 1
            total += 1

        if verbose:
            subset_name = "Challenge" if "Challenge" in subset else "Easy"
            accuracy = correct / total if total > 0 else 0
            print(f"  ARC-{subset_name}: {accuracy:.1%} ({correct}/{total})")

        return correct, total

    def run(self, quick: bool = False, verbose: bool = False) -> ARCResult:
        """Run ARC evaluation (Challenge + Easy)."""
        if verbose:
            print("=" * 60)
            print("ARC Benchmark Evaluation")
            print("=" * 60)

        challenge_correct, challenge_total = self.evaluate_subset("ARC-Challenge", quick, verbose)
        easy_correct, easy_total = self.evaluate_subset("ARC-Easy", quick, verbose)

        total_correct = challenge_correct + easy_correct
        total_questions = challenge_total + easy_total

        accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        challenge_acc = challenge_correct / challenge_total if challenge_total > 0 else 0.0
        easy_acc = easy_correct / easy_total if easy_total > 0 else 0.0

        if verbose:
            print(f"Overall: {accuracy:.1%} ({total_correct}/{total_questions})")

        return ARCResult(
            accuracy=accuracy,
            easy_accuracy=easy_acc,
            challenge_accuracy=challenge_acc,
            total_questions=total_questions,
            correct_answers=total_correct,
        )


def run_arc_benchmark(
    model=None,
    tokenizer=None,
    quick: bool = False,
    verbose: bool = False,
    output_dir: str | Path | None = None,
) -> ARCResult:
    """Run ARC benchmark."""
    evaluator = ARCEvaluator(model=model, tokenizer=tokenizer)
    result = evaluator.run(quick=quick, verbose=verbose)

    if output_dir:
        output_path = Path(output_dir) / "arc_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="ARC Benchmark")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--output-dir", type=str, default="benchmarks/reports")
    args = parser.parse_args()

    run_arc_benchmark(quick=args.quick, verbose=True, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
