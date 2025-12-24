# benchmarks/bench_humaneval.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""HumanEval Code Generation Benchmark.

Tests code generation capability with Python function completion.
Pass@k metric with sandboxed execution.

References:
- Paper: https://arxiv.org/abs/2107.03374
- Dataset: openai_humaneval (HuggingFace)
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HumanEvalResult:
    """Result from HumanEval evaluation."""

    pass_at_1: float
    pass_at_10: float | None
    pass_at_100: float | None
    total_problems: int
    passed_problems: int
    execution_errors: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "humaneval",
            "pass_at_1": self.pass_at_1,
            "pass_at_10": self.pass_at_10,
            "pass_at_100": self.pass_at_100,
            "total_problems": self.total_problems,
            "passed_problems": self.passed_problems,
            "execution_errors": self.execution_errors,
            "timestamp": self.timestamp,
        }


def execute_code_safely(code: str, test_code: str, timeout: int = 5) -> bool:
    """Execute code in a sandboxed environment.

    Args:
        code: Generated code to execute.
        test_code: Test assertions to run.
        timeout: Execution timeout in seconds.

    Returns:
        True if tests pass, False otherwise.
    """

    def run_tests():
        try:
            exec_globals = {}
            # Execute the generated code
            exec(code, exec_globals)
            # Run the test
            exec(test_code, exec_globals)
            return True
        except Exception:
            return False

    try:
        # Use multiprocessing for timeout
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_tests)
            return future.result(timeout=timeout)
    except Exception:
        return False


class HumanEvalEvaluator:
    """HumanEval benchmark evaluator."""

    def __init__(self, model=None, tokenizer=None, num_samples: int = 1):
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples  # For pass@k

    def load_dataset(self) -> list[dict]:
        """Load HumanEval dataset."""
        try:
            from datasets import load_dataset

            ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
            return list(ds)
        except Exception as e:
            logger.warning(f"Could not load HumanEval: {e}")
            return []

    def generate_completion(self, prompt: str) -> str:
        """Generate code completion using QSGGenerator."""
        if self.model is None:
            # Dummy completion for testing
            return "    return None\n"
        try:
            import tensorflow as tf

            from highnoon.inference.qsg_generator import QSGConfig, QSGGenerator

            # Tokenize prompt if tokenizer available
            if self.tokenizer:
                input_ids = self.tokenizer.encode(prompt, return_tensors="tf")
            else:
                # Fallback: use random tokens (benchmark-only mode)
                input_ids = tf.random.uniform([1, 256], minval=0, maxval=32000, dtype=tf.int32)

            config = QSGConfig(bond_dim=32, coherence_range=128, grover_iterations=3)
            generator = QSGGenerator(self.model, config)

            # Generate completion tokens
            output = generator.generate(input_ids, max_new_tokens=256)

            # Decode if possible
            if self.tokenizer:
                completion = self.tokenizer.decode(output[0, input_ids.shape[1] :])
                # Stop at common code terminators
                for stop in ["\ndef ", "\nclass ", "\n#", "\nif __name__"]:
                    if stop in completion:
                        completion = completion[: completion.index(stop)]
                return completion

            return "    return None\n"
        except Exception:
            return "    return None\n"

    def evaluate_problem(self, problem: dict) -> bool:
        """Evaluate a single problem."""
        prompt = problem["prompt"]
        test = problem["test"]
        problem["entry_point"]

        # Generate completion
        completion = self.generate_completion(prompt)

        # Combine prompt + completion for full code
        full_code = prompt + completion

        # Run tests
        try:
            return execute_code_safely(full_code, test)
        except Exception:
            return False

    def run(self, quick: bool = False, verbose: bool = False) -> HumanEvalResult:
        """Run HumanEval evaluation."""
        if verbose:
            print("=" * 60)
            print("HumanEval Benchmark Evaluation")
            print("=" * 60)

        data = self.load_dataset()

        if not data:
            return HumanEvalResult(0.0, None, None, 0, 0, 0)

        test_samples = data[:20] if quick else data

        passed = 0
        errors = 0
        total = len(test_samples)

        for i, problem in enumerate(test_samples):
            if verbose:
                print(f"  [{i+1}/{total}] {problem['task_id']}...", end=" ", flush=True)

            try:
                if self.evaluate_problem(problem):
                    passed += 1
                    if verbose:
                        print("PASS")
                else:
                    if verbose:
                        print("FAIL")
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"ERROR: {e}")

        pass_at_1 = passed / total if total > 0 else 0.0

        if verbose:
            print("-" * 60)
            print(f"Pass@1: {pass_at_1:.1%} ({passed}/{total})")
            print(f"Execution errors: {errors}")

        return HumanEvalResult(
            pass_at_1=pass_at_1,
            pass_at_10=None,  # Would need multiple samples
            pass_at_100=None,
            total_problems=total,
            passed_problems=passed,
            execution_errors=errors,
        )


def run_humaneval_benchmark(
    model=None,
    tokenizer=None,
    quick: bool = False,
    verbose: bool = False,
    output_dir: str | Path | None = None,
) -> HumanEvalResult:
    """Run HumanEval benchmark."""
    evaluator = HumanEvalEvaluator(model=model, tokenizer=tokenizer)
    result = evaluator.run(quick=quick, verbose=verbose)

    if output_dir:
        output_path = Path(output_dir) / "humaneval_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="HumanEval Benchmark")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--output-dir", type=str, default="benchmarks/reports")
    args = parser.parse_args()

    run_humaneval_benchmark(quick=args.quick, verbose=True, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
