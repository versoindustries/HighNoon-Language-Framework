# benchmarks/__init__.py
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

"""Enterprise Benchmark Suite for HSMN Architecture.

This package provides comprehensive benchmarking tools for evaluating the
'mamba_timecrystal_wlam_moe_hybrid' reasoning block pattern.

All benchmarks use the full HSMN model architecture and support real
HuggingFace datasets (WikiText-2, WikiText-103, PTB, C4).

Usage:
    # Run full benchmark suite
    python -m benchmarks --quick

    # Run specific benchmark
    python -m benchmarks.bench_throughput --quick

Modules:
    benchmark_config: Configuration dataclasses for benchmark parameters
    benchmark_harness: Core timing, memory profiling, and statistics
    bench_throughput: Tokens/second and scaling measurements
    bench_perplexity: Cross-entropy perplexity evaluation
    bench_confidence: Entropy and calibration metrics
    bench_memory: Memory usage analysis
    bench_comparison: Architecture comparison against baselines
    generate_report: Markdown report generation

Example:
    >>> from benchmarks import run_full_benchmark
    >>> report_path = run_full_benchmark(output_dir="benchmarks/reports")
"""

from benchmarks.bench_comparison import run_comparison_benchmark
from benchmarks.bench_confidence import run_confidence_benchmark
from benchmarks.bench_memory import run_memory_benchmark
from benchmarks.bench_perplexity import run_perplexity_benchmark
from benchmarks.bench_throughput import run_throughput_benchmark
from benchmarks.benchmark_config import BenchmarkConfig
from benchmarks.benchmark_harness import BenchmarkHarness, BenchmarkResult
from benchmarks.generate_report import run_full_benchmark

__all__ = [
    "BenchmarkConfig",
    "BenchmarkHarness",
    "BenchmarkResult",
    "run_throughput_benchmark",
    "run_perplexity_benchmark",
    "run_confidence_benchmark",
    "run_memory_benchmark",
    "run_comparison_benchmark",
    "run_full_benchmark",
]
