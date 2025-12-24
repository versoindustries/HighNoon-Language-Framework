# benchmarks/__main__.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified CLI entry point for HSMN benchmark suite.

Provides a single command to run the complete benchmark suite using
the full HSMN architecture with real HuggingFace datasets.

Usage:
    # Run full benchmark with WikiText-2
    python -m benchmarks

    # Quick validation run
    python -m benchmarks --quick

    # Ultra mode with 1M token context
    python -m benchmarks --ultra

    # Hardware detection only
    python -m benchmarks --hardware

    # Specify dataset
    python -m benchmarks --dataset wikitext-103

    # Run specific benchmark only
    python -m benchmarks --only throughput
    python -m benchmarks --only long-context
"""

import argparse
import logging
import sys
from pathlib import Path

from benchmarks.bench_comparison import run_comparison_benchmark
from benchmarks.bench_confidence import run_confidence_benchmark
from benchmarks.bench_memory import run_memory_benchmark
from benchmarks.bench_perplexity import run_perplexity_benchmark
from benchmarks.bench_quantum import run_quantum_benchmarks
from benchmarks.bench_throughput import run_throughput_benchmark
from benchmarks.benchmark_config import BenchmarkConfig
from benchmarks.generate_report import run_full_benchmark


def main() -> int:
    """Main CLI entry point for HSMN benchmarks.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="HSMN Enterprise Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m benchmarks --quick              # Quick validation
    python -m benchmarks --enterprise         # Enterprise (128K context)
    python -m benchmarks --ultra              # Ultra (1M context)
    python -m benchmarks --hardware           # Hardware info only
    python -m benchmarks --only long-context  # Long-context only
    python -m benchmarks --dataset wikitext   # Full run with WikiText-2

Output:
    Reports are saved to benchmarks/reports/ by default.
    JSON and Markdown formats are generated.
        """,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (smaller model, fewer iterations)",
    )
    parser.add_argument(
        "--enterprise",
        action="store_true",
        help="Run enterprise-grade benchmarks (128k context, comprehensive)",
    )
    parser.add_argument(
        "--ultra",
        action="store_true",
        help="Run ultra benchmarks (up to 1M token context, requires 64GB+ RAM)",
    )
    parser.add_argument("--hardware", action="store_true", help="Display hardware information only")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "wikitext", "wikitext-103", "ptb", "c4"],
        default="wikitext",
        help="Dataset for perplexity evaluation (default: wikitext)",
    )
    parser.add_argument(
        "--only",
        choices=[
            "throughput",
            "perplexity",
            "confidence",
            "memory",
            "comparison",
            "quantum",
            "long-context",
            "all",
        ],
        default="all",
        help="Run only a specific benchmark (default: all)",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Maximum context length for long-context benchmarks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/reports"),
        help="Output directory for reports",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Hardware info only
    if args.hardware:
        from benchmarks.hardware_detection import HardwareDetector, print_system_info

        print_system_info()

        # Save to file
        args.output_dir.mkdir(parents=True, exist_ok=True)
        detector = HardwareDetector()
        info = detector.detect_all()

        md_path = args.output_dir / "hardware_info.md"
        with open(md_path, "w") as f:
            f.write(detector.format_markdown(info))
        print(f"\nSaved to: {md_path}")
        return 0

    # Determine mode
    if args.ultra:
        mode = "Ultra (1M context)"
        config = BenchmarkConfig.ultra()
    elif args.enterprise:
        mode = "Enterprise (128K)"
        config = BenchmarkConfig.enterprise()
    elif args.quick:
        mode = "Quick"
        config = BenchmarkConfig.quick()
    else:
        mode = "Full"
        config = BenchmarkConfig()

    print("=" * 70)
    print("HSMN Enterprise Benchmark Suite")
    print(f"Mode: {mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Benchmark: {args.only}")
    print("=" * 70)

    # Run selected benchmark(s)
    if args.only == "all":
        report_path = run_full_benchmark(
            config=config,
            quick=args.quick,
            output_dir=args.output_dir,
            dataset=args.dataset,
        )
        print(f"\n✅ Full benchmark complete: {report_path}")

    elif args.only == "throughput":
        results = run_throughput_benchmark(config=config, quick=args.quick)
        print(f"\n✅ Throughput: {results['forward_summary']['mean_tokens_per_second']:,.0f} tok/s")

    elif args.only == "perplexity":
        results = run_perplexity_benchmark(config=config, quick=args.quick, dataset=args.dataset)
        print(f"\n✅ Perplexity: {results['overall_perplexity']:.2f}")

    elif args.only == "confidence":
        results = run_confidence_benchmark(config=config, quick=args.quick)
        print(f"\n✅ ECE: {results['expected_calibration_error']:.4f}")

    elif args.only == "memory":
        results = run_memory_benchmark(config=config, quick=args.quick)
        print(f"\n✅ Peak Memory: {results.get('peak_memory_mb', 0):.1f} MB")

    elif args.only == "comparison":
        results = run_comparison_benchmark(config=config, quick=args.quick)
        speedups = results.get("hsmn_vs_transformer_speedup", {})
        if speedups:
            max_speedup = max(speedups.values())
            print(f"\n✅ Max speedup vs Transformer: {max_speedup:.1f}x")

    elif args.only == "quantum":
        results = run_quantum_benchmarks(verbose=True)
        passed = results["summary"]["passed"]
        total = results["summary"]["total"]
        print(f"\n✅ Quantum Benchmarks: {passed}/{total} passed ({100*passed/total:.1f}%)")

    elif args.only == "long-context":
        from benchmarks.bench_long_context import (
            format_long_context_markdown,
            run_long_context_benchmark,
        )

        max_ctx = args.max_context or (1048576 if args.ultra else 131072)
        preset = "bench-enterprise" if args.ultra else "bench-standard"

        results = run_long_context_benchmark(
            model_preset=preset,
            max_context=max_ctx,
        )

        # Save results
        args.output_dir.mkdir(parents=True, exist_ok=True)

        import json

        json_path = args.output_dir / "long_context_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        md_path = args.output_dir / "long_context_results.md"
        with open(md_path, "w") as f:
            f.write(format_long_context_markdown(results))

        print(f"\n✅ Max context: {results['summary']['max_context_achieved']:,} tokens")
        print(f"   Complexity: {results['summary']['time_complexity']}")
        print(f"   Results: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
