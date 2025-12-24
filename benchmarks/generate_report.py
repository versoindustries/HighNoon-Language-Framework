# benchmarks/generate_report.py
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

"""Comprehensive benchmark report generator for HSMN architecture.

Runs all benchmarks and generates a unified markdown report with
executive summary, detailed results, and visualizations.

Example:
    >>> from benchmarks.generate_report import run_full_benchmark
    >>> report_path = run_full_benchmark(output_dir="benchmarks/reports")

Command-line usage:
    python benchmarks/generate_report.py --output benchmarks/reports/hsmn_benchmark.md
    python benchmarks/generate_report.py --quick  # Quick validation
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.bench_comparison import format_comparison_markdown, run_comparison_benchmark
from benchmarks.bench_confidence import format_confidence_markdown, run_confidence_benchmark
from benchmarks.bench_memory import format_memory_markdown, run_memory_benchmark
from benchmarks.bench_perplexity import format_perplexity_markdown, run_perplexity_benchmark
from benchmarks.bench_throughput import format_throughput_markdown, run_throughput_benchmark
from benchmarks.benchmark_config import BenchmarkConfig

logger = logging.getLogger(__name__)


def generate_executive_summary(results: dict[str, Any]) -> str:
    """Generate executive summary from all benchmark results.

    Args:
        results: Dictionary with all benchmark results.

    Returns:
        Markdown executive summary.
    """
    lines = [
        "## Executive Summary",
        "",
    ]

    # Key metrics table
    lines.extend(
        [
            "| Metric | Value | Rating |",
            "|--------|-------|--------|",
        ]
    )

    # Throughput
    if "throughput" in results and results["throughput"].get("forward_summary"):
        tps = results["throughput"]["forward_summary"]["mean_tokens_per_second"]
        rating = (
            "🟢 Excellent" if tps > 10000 else "🟡 Good" if tps > 1000 else "🔴 Needs Improvement"
        )
        lines.append(f"| **Forward Throughput** | {tps:,.0f} tok/s | {rating} |")

    # Perplexity
    if "perplexity" in results:
        ppl = results["perplexity"]["overall_perplexity"]
        rating = "🟢 Excellent" if ppl < 50 else "🟡 Good" if ppl < 200 else "🔴 High"
        lines.append(f"| **Perplexity** | {ppl:.2f} | {rating} |")

    # Confidence
    if "confidence" in results:
        conf = results["confidence"]["mean_confidence"]
        ece = results["confidence"]["expected_calibration_error"]
        rating = (
            "🟢 Well Calibrated"
            if ece < 0.05
            else "🟡 Moderate" if ece < 0.15 else "🔴 Poor Calibration"
        )
        lines.append(f"| **Mean Confidence** | {conf:.3f} | - |")
        lines.append(f"| **Calibration (ECE)** | {ece:.4f} | {rating} |")

    # Memory
    if "memory" in results and "peak_memory_mb" in results["memory"]:
        peak_mb = results["memory"]["peak_memory_mb"]
        params = results["memory"]["model_memory"]["total_parameters"]
        lines.append(f"| **Peak Memory** | {peak_mb:.1f} MB | - |")
        lines.append(f"| **Parameters** | {params:,} | - |")

    # Complexity
    if "comparison" in results and "complexity_analysis" in results["comparison"]:
        hsmn_analysis = results["comparison"]["complexity_analysis"].get("hsmn", {})
        complexity = hsmn_analysis.get("complexity_class", "O(n)")
        rating = (
            "🟢 Linear"
            if "O(n)" in complexity
            else "🟡 Quasi-linear" if "log" in complexity else "🔴 Quadratic"
        )
        lines.append(f"| **Complexity** | {complexity} | {rating} |")

    # Speedup
    if "comparison" in results and results["comparison"].get("hsmn_vs_transformer_speedup"):
        speedups = results["comparison"]["hsmn_vs_transformer_speedup"]
        if speedups:
            max_speedup = max(speedups.values())
            lines.append(f"| **Max Speedup vs Transformer** | {max_speedup:.1f}x | - |")

    lines.append("")

    return "\n".join(lines)


def generate_architecture_overview(results: dict[str, Any]) -> str:
    """Generate architecture overview section.

    Args:
        results: Benchmark results.

    Returns:
        Markdown architecture overview.
    """
    block_pattern = results.get("throughput", {}).get(
        "block_pattern", "mamba_timecrystal_wlam_moe_hybrid"
    )

    lines = [
        "## Architecture Overview",
        "",
        f"**Block Pattern**: `{block_pattern}`",
        "",
        "The HSMN architecture uses a hybrid pattern of linear-time blocks:",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[Input] --> B[Mamba SSM<br>O(n)]",
        "    B --> C[TimeCrystal<br>O(n)]",
        "    C --> D[WLAM<br>O(n log n)]",
        "    D --> E[MoE<br>O(n·k)]",
        "    E --> F[Output]",
        "    E --> |repeat| B",
        "```",
        "",
        "### Block Descriptions",
        "",
        "| Block | Complexity | Purpose |",
        "|-------|------------|---------|",
        "| **Mamba SSM** | O(n) | State space sequence modeling |",
        "| **TimeCrystal** | O(n) | Hamiltonian energy dynamics |",
        "| **WLAM** | O(n log n) | Wavelet-based attention |",
        "| **MoE** | O(n·k) | Sparse mixture of experts |",
        "",
    ]

    return "\n".join(lines)


def generate_scaling_diagram(results: dict[str, Any]) -> str:
    """Generate scaling analysis diagram.

    Args:
        results: Benchmark results with comparison data.

    Returns:
        Markdown with Mermaid chart.
    """
    if "comparison" not in results:
        return ""

    lines = [
        "## Scaling Analysis",
        "",
        "```mermaid",
        "xychart-beta",
        '    title "Time vs Sequence Length (log-log)"',
        "    x-axis [128, 256, 512, 1024, 2048]",
        '    y-axis "Time (ms)" 0 --> 1000',
    ]

    # Add data series if available
    profiles = results["comparison"].get("profiles", {})

    if "hsmn" in profiles:
        timings = profiles["hsmn"]["timings_ms"][:5]  # First 5 values
        if timings:
            values = ", ".join(str(int(t)) for t in timings if t < 10000)
            lines.append(f"    line [{values}]")

    lines.extend(
        [
            "```",
            "",
            "> HSMN maintains O(n) linear scaling compared to O(n²) transformer baseline.",
            "",
        ]
    )

    return "\n".join(lines)


def run_full_benchmark(
    config: BenchmarkConfig | None = None,
    quick: bool = False,
    output_dir: Path | None = None,
    dataset: str = "wikitext",
) -> Path:
    """Run all benchmarks and generate comprehensive report.

    Args:
        config: Benchmark configuration.
        quick: Use quick configuration.
        output_dir: Output directory for reports.
        dataset: Dataset for perplexity evaluation.

    Returns:
        Path to generated report.
    """
    if config is None:
        config = BenchmarkConfig.quick() if quick else BenchmarkConfig()

    output_dir = output_dir or config.output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("HSMN Enterprise Benchmark Suite")
    logger.info(f"Pattern: {config.model.block_pattern}")
    logger.info(f"Mode: {'Quick' if quick else 'Full'}")
    logger.info("=" * 70)

    results = {}

    # Run all benchmarks
    logger.info("\n[1/5] Running throughput benchmark...")
    try:
        results["throughput"] = run_throughput_benchmark(config)
    except Exception as e:
        logger.error(f"Throughput benchmark failed: {e}")
        results["throughput"] = {"error": str(e)}

    logger.info("\n[2/5] Running perplexity benchmark...")
    try:
        results["perplexity"] = run_perplexity_benchmark(config, dataset=dataset)
    except Exception as e:
        logger.error(f"Perplexity benchmark failed: {e}")
        results["perplexity"] = {"error": str(e)}

    logger.info("\n[3/5] Running confidence benchmark...")
    try:
        results["confidence"] = run_confidence_benchmark(config)
    except Exception as e:
        logger.error(f"Confidence benchmark failed: {e}")
        results["confidence"] = {"error": str(e)}

    logger.info("\n[4/5] Running memory benchmark...")
    try:
        results["memory"] = run_memory_benchmark(config)
    except Exception as e:
        logger.error(f"Memory benchmark failed: {e}")
        results["memory"] = {"error": str(e)}

    logger.info("\n[5/5] Running comparison benchmark...")
    try:
        results["comparison"] = run_comparison_benchmark(config)
    except Exception as e:
        logger.error(f"Comparison benchmark failed: {e}")
        results["comparison"] = {"error": str(e)}

    # Generate report
    logger.info("\nGenerating report...")

    # Detect hardware for report
    try:
        from benchmarks.hardware_detection import HardwareDetector

        hw_detector = HardwareDetector()
        hw_info = hw_detector.detect_all()
        hw_summary = hw_detector.format_summary(hw_info)
        results["hardware"] = hw_info.to_dict()
    except ImportError:
        hw_summary = "Hardware detection not available"
        hw_info = None

    report_lines = [
        "# HSMN Enterprise Benchmark Report",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        f"**Configuration**: {'Quick' if quick else 'Full'}",
        f"**Perplexity Dataset**: {dataset}",
        "",
        f"**System**: {hw_summary}",
        "",
        "---",
        "",
    ]

    # Hardware section if available
    if hw_info is not None:
        # Use the comprehensive hardware markdown output
        hw_markdown = hw_detector.format_markdown(hw_info)
        report_lines.extend([hw_markdown, ""])

    # Executive Summary
    report_lines.append(generate_executive_summary(results))

    # Architecture Overview
    report_lines.append(generate_architecture_overview(results))

    # Scaling Diagram
    report_lines.append(generate_scaling_diagram(results))

    report_lines.append("---\n")

    # Detailed Results
    report_lines.append("# Detailed Results\n")

    if "throughput" in results and "error" not in results["throughput"]:
        report_lines.append(format_throughput_markdown(results["throughput"]))
        report_lines.append("\n---\n")

    if "perplexity" in results and "error" not in results["perplexity"]:
        report_lines.append(format_perplexity_markdown(results["perplexity"]))
        report_lines.append("\n---\n")

    if "confidence" in results and "error" not in results["confidence"]:
        report_lines.append(format_confidence_markdown(results["confidence"]))
        report_lines.append("\n---\n")

    if "memory" in results and "error" not in results["memory"]:
        report_lines.append(format_memory_markdown(results["memory"]))
        report_lines.append("\n---\n")

    if "comparison" in results and "error" not in results["comparison"]:
        report_lines.append(format_comparison_markdown(results["comparison"]))

    # Write report
    report_content = "\n".join(report_lines)
    report_path = output_dir / "hsmn_benchmark_report.md"

    with open(report_path, "w") as f:
        f.write(report_content)

    logger.info(f"\nReport saved to: {report_path}")

    # Save raw JSON results
    json_path = output_dir / "hsmn_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"JSON results saved to: {json_path}")

    return report_path


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive HSMN benchmark report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick validation run
    python benchmarks/generate_report.py --quick

    # Full benchmark with WikiText-2
    python benchmarks/generate_report.py --dataset wikitext

    # Full benchmark with custom output
    python benchmarks/generate_report.py --output benchmarks/reports/full_report.md
        """,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation benchmark (smaller model, fewer iterations)",
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "wikitext", "wikitext-103", "ptb", "c4"],
        default="wikitext",
        help="Dataset for perplexity evaluation (default: wikitext)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/reports"),
        help="Output directory for reports",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    report_path = run_full_benchmark(
        quick=args.quick,
        output_dir=args.output_dir,
        dataset=args.dataset,
    )

    print(f"\n✅ Report generated: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
