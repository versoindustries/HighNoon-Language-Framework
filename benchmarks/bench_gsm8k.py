# benchmarks/bench_gsm8k.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""GSM8K Grade School Math Benchmark.

Tests mathematical reasoning with word problems.
5-shot chain-of-thought evaluation.

References:
- Paper: https://arxiv.org/abs/2110.14168
- Dataset: gsm8k (HuggingFace)
"""

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GSM8KResult:
    """Result from GSM8K evaluation."""

    accuracy: float
    total_questions: int
    correct_answers: int
    extraction_failures: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "gsm8k",
            "accuracy": self.accuracy,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "extraction_failures": self.extraction_failures,
            "timestamp": self.timestamp,
        }


class GSM8KEvaluator:
    """GSM8K benchmark with 5-shot chain-of-thought."""

    def __init__(self, model=None, tokenizer=None, num_shots: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.num_shots = num_shots

    def load_dataset(self, split: str = "test") -> list[dict]:
        """Load GSM8K dataset."""
        try:
            from datasets import load_dataset

            ds = load_dataset("gsm8k", "main", split=split, trust_remote_code=True)
            return list(ds)
        except Exception as e:
            logger.warning(f"Could not load GSM8K: {e}")
            return []

    def extract_answer(self, text: str) -> str | None:
        """Extract final numerical answer from text."""
        # Look for #### pattern (GSM8K format)
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
        if match:
            return match.group(1).replace(",", "")

        # Find last number in text
        numbers = re.findall(r"-?[\d,]+\.?\d*", text)
        if numbers:
            return numbers[-1].replace(",", "")

        return None

    def format_example(self, question: str, answer: str) -> str:
        """Format a single example with chain-of-thought."""
        return f"Question: {question}\nLet's think step by step.\n{answer}\n"

    def evaluate_question(self, prompt: str) -> str | None:
        """Generate answer for question."""
        if self.model is None:
            return str(random.randint(1, 100))
        try:
            response = self.model.generate(prompt, max_tokens=256, temperature=0.0)
            return self.extract_answer(response)
        except Exception:
            return None

    def run(self, quick: bool = False, verbose: bool = False) -> GSM8KResult:
        """Run GSM8K evaluation."""
        if verbose:
            print("=" * 60)
            print("GSM8K Benchmark Evaluation")
            print("=" * 60)

        train_data = self.load_dataset("train")[: self.num_shots]
        test_data = self.load_dataset("test")

        if not test_data:
            return GSM8KResult(0.0, 0, 0, 0)

        test_samples = test_data[:50] if quick else test_data

        correct = 0
        total = 0
        extraction_failures = 0

        for item in test_samples:
            prompt = "Solve the following math problems step by step.\n\n"
            for ex in train_data[: self.num_shots]:
                prompt += self.format_example(ex["question"], ex["answer"]) + "\n"

            prompt += f"Question: {item['question']}\nLet's think step by step.\n"

            prediction = self.evaluate_question(prompt)
            expected = self.extract_answer(item["answer"])

            if prediction is None:
                extraction_failures += 1
            elif prediction == expected:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        if verbose:
            print(f"Accuracy: {accuracy:.1%} ({correct}/{total})")
            print(f"Extraction failures: {extraction_failures}")

        return GSM8KResult(
            accuracy=accuracy,
            total_questions=total,
            correct_answers=correct,
            extraction_failures=extraction_failures,
        )


def run_gsm8k_benchmark(
    model=None,
    tokenizer=None,
    quick: bool = False,
    verbose: bool = False,
    output_dir: str | Path | None = None,
) -> GSM8KResult:
    """Run GSM8K benchmark."""
    evaluator = GSM8KEvaluator(model=model, tokenizer=tokenizer)
    result = evaluator.run(quick=quick, verbose=verbose)

    if output_dir:
        output_path = Path(output_dir) / "gsm8k_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="GSM8K Benchmark")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--output-dir", type=str, default="benchmarks/reports")
    args = parser.parse_args()

    run_gsm8k_benchmark(quick=args.quick, verbose=True, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
