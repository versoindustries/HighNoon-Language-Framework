# benchmarks/benchmark_cache.py
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

"""Enterprise-grade cache profiling benchmark harness.

This module provides comprehensive cache profiling infrastructure for HSMN
native operations, integrating Linux `perf` for hardware counter collection
and generating actionable optimization reports.

Target Architecture: x86-64 with AVX2/AVX-512 (primary), ARM64 NEON (secondary)

Example:
    >>> from benchmarks.benchmark_cache import CacheBenchmark
    >>> bench = CacheBenchmark()
    >>> results = bench.run_suite(ops=["moe", "ssm", "galore"])
    >>> bench.generate_report(results, format="markdown")

CLI Usage:
    python -m benchmarks.benchmark_cache --ops moe,ssm --sizes 128,512,2048 --report both
"""

import argparse
import json
import logging
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class OpCategory(Enum):
    """Categories of operations to benchmark."""

    MOE = "moe"
    SSM = "ssm"
    GALORE = "galore"
    ATTENTION = "attention"
    EMBEDDING = "embedding"
    ALL = "all"


@dataclass
class CacheBenchmarkConfig:
    """Configuration for cache benchmarking.

    Attributes:
        ops: Operations to benchmark.
        sizes: Input sizes (sequence lengths) to test.
        batch_size: Batch size for benchmarks.
        warmup_iterations: Warmup iterations.
        benchmark_iterations: Timed iterations.
        use_perf: Whether to use Linux perf for hardware counters.
        perf_events: Hardware counter events to collect.
        output_dir: Output directory for reports.
        verbose: Enable verbose logging.
    """

    ops: list[str] = field(default_factory=lambda: ["moe", "ssm", "galore"])
    sizes: list[int] = field(default_factory=lambda: [128, 512, 2048, 8192])
    batch_size: int = 1
    warmup_iterations: int = 5
    benchmark_iterations: int = 20
    use_perf: bool = True
    perf_events: list[str] = field(
        default_factory=lambda: [
            "cache-references",
            "cache-misses",
            "L1-dcache-load-misses",
            "L1-dcache-store-misses",
            "LLC-load-misses",
            "LLC-store-misses",
            "dTLB-load-misses",
            "dTLB-store-misses",
        ]
    )
    output_dir: Path = field(default_factory=lambda: Path("benchmarks/reports"))
    verbose: bool = False

    @classmethod
    def quick(cls) -> "CacheBenchmarkConfig":
        """Quick config for validation."""
        return cls(
            ops=["moe"],
            sizes=[128, 512],
            warmup_iterations=2,
            benchmark_iterations=5,
        )

    @classmethod
    def full(cls) -> "CacheBenchmarkConfig":
        """Full config for comprehensive analysis."""
        return cls(
            ops=["moe", "ssm", "galore", "attention"],
            sizes=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
            warmup_iterations=10,
            benchmark_iterations=50,
        )

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> "CacheBenchmarkConfig":
        """Create config from CLI arguments."""
        return cls(
            ops=args.ops.split(",") if args.ops else ["moe", "ssm", "galore"],
            sizes=[int(s) for s in args.sizes.split(",")] if args.sizes else [128, 512, 2048],
            batch_size=args.batch_size,
            warmup_iterations=args.warmup,
            benchmark_iterations=args.iterations,
            use_perf=not args.dry_run,
            output_dir=Path(args.output_dir),
            verbose=args.verbose,
        )


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PerfCounters:
    """Hardware performance counter results.

    Attributes:
        cache_references: Total cache references.
        cache_misses: Total cache misses.
        l1_dcache_load_misses: L1 data cache load misses.
        l1_dcache_store_misses: L1 data cache store misses.
        llc_load_misses: Last-level cache load misses.
        llc_store_misses: Last-level cache store misses.
        dtlb_load_misses: Data TLB load misses.
        dtlb_store_misses: Data TLB store misses.
        cache_miss_rate: Calculated cache miss rate (0.0-1.0).
        l1_miss_rate: L1 cache miss rate.
        llc_miss_rate: LLC miss rate relative to L1 misses.
    """

    cache_references: int = 0
    cache_misses: int = 0
    l1_dcache_load_misses: int = 0
    l1_dcache_store_misses: int = 0
    llc_load_misses: int = 0
    llc_store_misses: int = 0
    dtlb_load_misses: int = 0
    dtlb_store_misses: int = 0
    cache_miss_rate: float = 0.0
    l1_miss_rate: float = 0.0
    llc_miss_rate: float = 0.0

    def __post_init__(self) -> None:
        """Calculate derived metrics."""
        if self.cache_references > 0:
            self.cache_miss_rate = self.cache_misses / self.cache_references
        total_l1_misses = self.l1_dcache_load_misses + self.l1_dcache_store_misses
        if self.cache_references > 0:
            self.l1_miss_rate = total_l1_misses / self.cache_references
        if total_l1_misses > 0:
            total_llc_misses = self.llc_load_misses + self.llc_store_misses
            self.llc_miss_rate = total_llc_misses / total_l1_misses


@dataclass
class KernelResult:
    """Result for a single kernel benchmark.

    Attributes:
        kernel_name: Name of the kernel benchmarked.
        input_size: Input size (sequence length).
        batch_size: Batch size used.
        times_ms: List of timing measurements in milliseconds.
        mean_ms: Mean execution time.
        std_ms: Standard deviation.
        min_ms: Minimum time.
        max_ms: Maximum time.
        p95_ms: 95th percentile time.
        perf_counters: Hardware counter results (if available).
        throughput_elements_per_sec: Throughput in elements/second.
    """

    kernel_name: str
    input_size: int
    batch_size: int
    times_ms: list[float] = field(default_factory=list)
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p95_ms: float = 0.0
    perf_counters: PerfCounters | None = None
    throughput_elements_per_sec: float = 0.0

    def __post_init__(self) -> None:
        """Calculate statistics."""
        if self.times_ms:
            self.mean_ms = statistics.mean(self.times_ms)
            self.std_ms = statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0
            self.min_ms = min(self.times_ms)
            self.max_ms = max(self.times_ms)
            sorted_times = sorted(self.times_ms)
            self.p95_ms = sorted_times[int(len(sorted_times) * 0.95)]

            # Calculate throughput
            total_elements = self.input_size * self.batch_size
            if self.mean_ms > 0:
                self.throughput_elements_per_sec = total_elements / (self.mean_ms / 1000)


@dataclass
class BenchmarkSuiteResult:
    """Complete benchmark suite results.

    Attributes:
        timestamp: ISO timestamp of benchmark run.
        config: Configuration used.
        system_info: System information.
        kernel_results: Results for each kernel.
        recommendations: Generated optimization recommendations.
    """

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict[str, Any] = field(default_factory=dict)
    system_info: dict[str, Any] = field(default_factory=dict)
    kernel_results: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "config": self.config,
            "system_info": self.system_info,
            "kernel_results": self.kernel_results,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Perf Integration
# =============================================================================


class PerfRunner:
    """Linux perf integration for hardware counter collection.

    Supports running Python code with perf stat and parsing results.
    Falls back to timing-only mode if perf is unavailable.
    """

    def __init__(self, events: list[str] | None = None, verbose: bool = False):
        """Initialize PerfRunner.

        Args:
            events: List of perf events to collect.
            verbose: Enable verbose output.
        """
        self.events = events or [
            "cache-references",
            "cache-misses",
            "L1-dcache-load-misses",
            "LLC-load-misses",
        ]
        self.verbose = verbose
        self._perf_available = self._check_perf_available()

    def _check_perf_available(self) -> bool:
        """Check if perf is available and accessible."""
        if platform.system() != "Linux":
            logger.warning("perf is only available on Linux")
            return False

        if shutil.which("perf") is None:
            logger.warning("perf command not found in PATH")
            return False

        # Check perf_event_paranoid
        try:
            with open("/proc/sys/kernel/perf_event_paranoid") as f:
                paranoid = int(f.read().strip())
                if paranoid > 1:
                    logger.warning(
                        f"perf_event_paranoid is {paranoid}. "
                        "You may need to run with sudo or set: "
                        "sudo sysctl -w kernel.perf_event_paranoid=1"
                    )
        except (OSError, ValueError):
            pass

        # Test if we can actually collect events
        try:
            result = subprocess.run(
                ["perf", "stat", "-e", "cycles", "sleep", "0.001"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                logger.warning("perf stat test failed, may lack permissions")
                return False
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"perf test failed: {e}")
            return False

        return True

    @property
    def is_available(self) -> bool:
        """Check if perf is available."""
        return self._perf_available

    def run_with_perf(
        self,
        python_code: str,
        iterations: int = 1,
    ) -> tuple[list[float], PerfCounters | None]:
        """Run Python code with perf stat.

        Args:
            python_code: Python code to execute.
            iterations: Number of iterations to run.

        Returns:
            Tuple of (timing list in ms, perf counters or None).
        """
        if not self._perf_available:
            return self._run_without_perf(python_code, iterations), None

        # Create temporary file with the code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Wrap code with timing
            wrapped_code = f"""
import time
import json

times = []
for _ in range({iterations}):
    start = time.perf_counter()
    {python_code}
    end = time.perf_counter()
    times.append((end - start) * 1000)

print("TIMES:" + json.dumps(times))
"""
            f.write(wrapped_code)
            f.flush()
            temp_path = f.name

        try:
            # Build perf command
            events_str = ",".join(self.events)
            cmd = [
                "perf",
                "stat",
                "-e",
                events_str,
                "-x",
                ",",  # CSV output
                sys.executable,
                temp_path,
            ]

            if self.verbose:
                logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Parse timing from stdout
            times = []
            for line in result.stdout.split("\n"):
                if line.startswith("TIMES:"):
                    times = json.loads(line[6:])
                    break

            # Parse perf counters from stderr
            counters = self._parse_perf_output(result.stderr)

            return times, counters

        except subprocess.TimeoutExpired:
            logger.error("Benchmark timed out")
            return [], None
        except Exception as e:
            logger.error(f"Perf run failed: {e}")
            return self._run_without_perf(python_code, iterations), None
        finally:
            os.unlink(temp_path)

    def _run_without_perf(self, python_code: str, iterations: int) -> list[float]:
        """Run code without perf, timing only."""
        times = []
        # Create a local namespace for exec
        local_ns: dict[str, Any] = {}
        exec("import tensorflow as tf; import numpy as np", local_ns)

        for _ in range(iterations):
            start = time.perf_counter()
            exec(python_code, local_ns)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        return times

    def _parse_perf_output(self, stderr: str) -> PerfCounters:
        """Parse perf stat CSV output.

        Args:
            stderr: stderr output from perf stat -x ,

        Returns:
            PerfCounters with parsed values.
        """
        counters = PerfCounters()

        for line in stderr.split("\n"):
            parts = line.split(",")
            if len(parts) < 3:
                continue

            try:
                value_str = (
                    parts[0].strip().replace("<not counted>", "0").replace("<not supported>", "0")
                )
                if not value_str:
                    continue
                value = int(value_str)
                event = parts[2].strip()

                if "cache-references" in event:
                    counters.cache_references = value
                elif "cache-misses" in event and "L1" not in event and "LLC" not in event:
                    counters.cache_misses = value
                elif "L1-dcache-load-misses" in event:
                    counters.l1_dcache_load_misses = value
                elif "L1-dcache-store-misses" in event:
                    counters.l1_dcache_store_misses = value
                elif "LLC-load-misses" in event:
                    counters.llc_load_misses = value
                elif "LLC-store-misses" in event:
                    counters.llc_store_misses = value
                elif "dTLB-load-misses" in event:
                    counters.dtlb_load_misses = value
                elif "dTLB-store-misses" in event:
                    counters.dtlb_store_misses = value
            except (ValueError, IndexError):
                continue

        # Recalculate derived metrics
        counters.__post_init__()
        return counters


# =============================================================================
# Kernel Benchmarks
# =============================================================================


class KernelBenchmarks:
    """Individual kernel benchmarks for cache profiling.

    Each benchmark isolates a specific operation to measure its cache behavior.
    """

    def __init__(self, config: CacheBenchmarkConfig):
        """Initialize kernel benchmarks.

        Args:
            config: Benchmark configuration.
        """
        self.config = config
        self.perf = PerfRunner(events=config.perf_events, verbose=config.verbose)

    def _generate_moe_benchmark_code(self, size: int, batch: int) -> str:
        """Generate MoE dispatch benchmark code.

        Tests expert routing and dispatch cache patterns.
        """
        return f"""
import tensorflow as tf
import numpy as np

# Simulate MoE dispatch with expert selection
hidden_dim = 128
num_experts = 8
top_k = 2

# Input hidden states
hidden = tf.random.normal([{batch}, {size}, hidden_dim])

# Router logits (simulated)
router_logits = tf.random.normal([{batch}, {size}, num_experts])

# Top-k expert selection
topk_gates, topk_indices = tf.math.top_k(router_logits, k=top_k)
topk_gates = tf.nn.softmax(topk_gates, axis=-1)

# Dispatch: gather input for each expert
dispatch_mask = tf.one_hot(topk_indices, num_experts)
dispatch_mask = tf.reduce_sum(dispatch_mask, axis=-2)

# Expert computation (simulated dense)
for e in range(num_experts):
    expert_mask = dispatch_mask[:, :, e:e+1]
    expert_input = hidden * expert_mask
    _ = tf.matmul(expert_input, tf.random.normal([hidden_dim, hidden_dim * 4]))
"""

    def _generate_ssm_benchmark_code(self, size: int, batch: int) -> str:
        """Generate SSM state update benchmark code.

        Tests sequential state update cache patterns.
        """
        return f"""
import tensorflow as tf
import numpy as np

# SSM parameters
state_dim = 64
hidden_dim = 128

# Input sequence
x = tf.random.normal([{batch}, {size}, hidden_dim])

# SSM matrices
A = tf.random.normal([state_dim, state_dim]) * 0.1
B = tf.random.normal([hidden_dim, state_dim])
C = tf.random.normal([state_dim, hidden_dim])

# Initial state
h = tf.zeros([{batch}, state_dim])

# Sequential state update (main cache pressure point)
outputs = []
for t in range({size}):
    x_t = x[:, t, :]  # [batch, hidden_dim]
    # State update: h = A @ h + B @ x
    h = tf.tanh(tf.matmul(h, A) + tf.matmul(x_t, B))
    # Output: y = C @ h
    y = tf.matmul(h, C)
    outputs.append(y)

output = tf.stack(outputs, axis=1)
"""

    def _generate_galore_benchmark_code(self, size: int, batch: int) -> str:
        """Generate Quantum GaLore projection benchmark code.

        Tests random projection matrix cache patterns.
        """
        return f"""
import tensorflow as tf
import numpy as np

# Gradient dimensions
rows = {size}
cols = 128
rank = 32

# Gradient matrix
gradient = tf.random.normal([{batch}, rows, cols])

# Random projection matrix (quantum-inspired)
rotation = tf.random.normal([rank, rows]) / np.sqrt(rows)
bias = tf.random.uniform([rank], 0, 2 * np.pi)

# Project gradient to low-rank space
# G_c = φ(R)^T @ G where φ(x) = cos(Rx + b)
projected = []
for b in range({batch}):
    g = gradient[b]  # [rows, cols]
    # Quantum feature projection
    features = tf.cos(tf.matmul(rotation, g) + bias[:, tf.newaxis])  # [rank, cols]
    projected.append(features)

output = tf.stack(projected, axis=0)
"""

    def _generate_attention_benchmark_code(self, size: int, batch: int) -> str:
        """Generate attention benchmark code.

        Tests QK^T and attention weight cache patterns.
        """
        return f"""
import tensorflow as tf
import numpy as np

# Attention parameters
hidden_dim = 128
num_heads = 8
head_dim = hidden_dim // num_heads

# Input
x = tf.random.normal([{batch}, {size}, hidden_dim])

# QKV projections
Wq = tf.random.normal([hidden_dim, hidden_dim])
Wk = tf.random.normal([hidden_dim, hidden_dim])
Wv = tf.random.normal([hidden_dim, hidden_dim])

Q = tf.matmul(x, Wq)
K = tf.matmul(x, Wk)
V = tf.matmul(x, Wv)

# Reshape for multi-head
Q = tf.reshape(Q, [{batch}, {size}, num_heads, head_dim])
K = tf.reshape(K, [{batch}, {size}, num_heads, head_dim])
V = tf.reshape(V, [{batch}, {size}, num_heads, head_dim])

Q = tf.transpose(Q, [0, 2, 1, 3])  # [batch, heads, seq, head_dim]
K = tf.transpose(K, [0, 2, 1, 3])
V = tf.transpose(V, [0, 2, 1, 3])

# Attention scores
scores = tf.matmul(Q, K, transpose_b=True) / np.sqrt(head_dim)

# Causal mask
mask = tf.linalg.band_part(tf.ones([{size}, {size}]), -1, 0)
scores = scores * mask - 1e9 * (1 - mask)

# Softmax and output
attn_weights = tf.nn.softmax(scores, axis=-1)
output = tf.matmul(attn_weights, V)
"""

    def _generate_embedding_benchmark_code(self, size: int, batch: int) -> str:
        """Generate embedding lookup benchmark code.

        Tests scattered memory access patterns.
        """
        return f"""
import tensorflow as tf
import numpy as np

# Embedding parameters
vocab_size = 32000
embedding_dim = 128

# Token IDs (random access pattern)
token_ids = tf.random.uniform([{batch}, {size}], 0, vocab_size, dtype=tf.int32)

# Embedding table
embedding_table = tf.random.normal([vocab_size, embedding_dim])

# Embedding lookup (scattered access)
embeddings = tf.nn.embedding_lookup(embedding_table, token_ids)

# Add positional embeddings
positions = tf.range({size})
pos_table = tf.random.normal([{size * 4}, embedding_dim])  # Larger for random access
pos_embeddings = tf.nn.embedding_lookup(pos_table, positions)

output = embeddings + pos_embeddings
"""

    def run_kernel_benchmark(
        self,
        kernel_name: str,
        size: int,
        batch: int,
    ) -> KernelResult:
        """Run a single kernel benchmark.

        Args:
            kernel_name: Name of the kernel to benchmark.
            size: Input size.
            batch: Batch size.

        Returns:
            KernelResult with timing and cache data.
        """
        # Generate benchmark code
        code_generators = {
            "moe": self._generate_moe_benchmark_code,
            "ssm": self._generate_ssm_benchmark_code,
            "galore": self._generate_galore_benchmark_code,
            "attention": self._generate_attention_benchmark_code,
            "embedding": self._generate_embedding_benchmark_code,
        }

        generator = code_generators.get(kernel_name)
        if generator is None:
            raise ValueError(f"Unknown kernel: {kernel_name}")

        code = generator(size, batch)

        # Warmup
        logger.info(f"Warming up {kernel_name} (size={size}, batch={batch})...")
        self.perf._run_without_perf(code, self.config.warmup_iterations)

        # Run with perf
        logger.info(f"Benchmarking {kernel_name}...")
        times, counters = self.perf.run_with_perf(code, self.config.benchmark_iterations)

        return KernelResult(
            kernel_name=kernel_name,
            input_size=size,
            batch_size=batch,
            times_ms=times,
            perf_counters=counters,
        )


# =============================================================================
# Main Benchmark Runner
# =============================================================================


class CacheBenchmark:
    """Main cache benchmark runner.

    Orchestrates kernel benchmarks and generates reports.
    """

    def __init__(self, config: CacheBenchmarkConfig | None = None):
        """Initialize cache benchmark.

        Args:
            config: Benchmark configuration.
        """
        self.config = config or CacheBenchmarkConfig()
        self.kernels = KernelBenchmarks(self.config)

        if self.config.verbose:
            logging.basicConfig(level=logging.INFO)

    def get_system_info(self) -> dict[str, Any]:
        """Collect system information."""
        info: dict[str, Any] = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "perf_available": self.kernels.perf.is_available,
        }

        # Try to get CPU cache info on Linux
        try:
            with open("/sys/devices/system/cpu/cpu0/cache/index0/size") as f:
                info["l1d_cache"] = f.read().strip()
            with open("/sys/devices/system/cpu/cpu0/cache/index2/size") as f:
                info["l2_cache"] = f.read().strip()
            with open("/sys/devices/system/cpu/cpu0/cache/index3/size") as f:
                info["l3_cache"] = f.read().strip()
        except (FileNotFoundError, OSError):
            pass

        # Memory info
        try:
            import psutil

            mem = psutil.virtual_memory()
            info["total_ram_gb"] = round(mem.total / (1024**3), 2)
            info["available_ram_gb"] = round(mem.available / (1024**3), 2)
        except ImportError:
            pass

        return info

    def run_suite(self, ops: list[str] | None = None) -> BenchmarkSuiteResult:
        """Run the full benchmark suite.

        Args:
            ops: Operations to benchmark (uses config if None).

        Returns:
            BenchmarkSuiteResult with all results.
        """
        ops = ops or self.config.ops

        result = BenchmarkSuiteResult(
            config=asdict(self.config),
            system_info=self.get_system_info(),
        )

        for op in ops:
            for size in self.config.sizes:
                logger.info(f"Running {op} benchmark with size={size}...")
                try:
                    kernel_result = self.kernels.run_kernel_benchmark(
                        op, size, self.config.batch_size
                    )
                    result.kernel_results.append(asdict(kernel_result))
                except Exception as e:
                    logger.error(f"Benchmark {op} (size={size}) failed: {e}")
                    continue

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result.kernel_results)

        return result

    def _generate_recommendations(self, kernel_results: list[dict[str, Any]]) -> list[str]:
        """Generate optimization recommendations based on results.

        Args:
            kernel_results: List of kernel result dictionaries.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        for kr in kernel_results:
            counters = kr.get("perf_counters")
            if counters is None:
                continue

            kernel = kr["kernel_name"]
            size = kr["input_size"]
            miss_rate = counters.get("cache_miss_rate", 0)
            _l1_miss_rate = counters.get(
                "l1_miss_rate", 0
            )  # noqa: F841 - unused but kept for future
            llc_miss_rate = counters.get("llc_miss_rate", 0)

            if miss_rate > 0.10:  # >10% cache miss rate
                recommendations.append(
                    f"🔴 **{kernel}** (size={size}): High cache miss rate ({miss_rate:.1%}). "
                    "Consider adding prefetch hints or cache blocking."
                )
            elif miss_rate > 0.05:  # 5-10% miss rate
                recommendations.append(
                    f"🟡 **{kernel}** (size={size}): Moderate cache miss rate ({miss_rate:.1%}). "
                    "May benefit from prefetch optimization."
                )
            else:
                recommendations.append(
                    f"🟢 **{kernel}** (size={size}): Good cache efficiency ({miss_rate:.1%})."
                )

            if llc_miss_rate > 0.50:  # >50% of L1 misses go to main memory
                recommendations.append(
                    f"   ⚠️ High LLC miss rate ({llc_miss_rate:.1%}). "
                    "Data working set may exceed cache. Consider blocking."
                )

        return recommendations

    def generate_report(
        self,
        result: BenchmarkSuiteResult,
        format: str = "both",
    ) -> tuple[Path | None, Path | None]:
        """Generate benchmark report.

        Args:
            result: Benchmark suite result.
            format: Output format ('json', 'markdown', 'both').

        Returns:
            Tuple of (json_path, markdown_path).
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = None
        md_path = None

        if format in ("json", "both"):
            json_path = self.config.output_dir / f"cache_benchmark_{timestamp}.json"
            with open(json_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            logger.info(f"Saved JSON report to {json_path}")

        if format in ("markdown", "both"):
            md_path = self.config.output_dir / f"cache_benchmark_{timestamp}.md"
            md_content = self._generate_markdown_report(result)
            with open(md_path, "w") as f:
                f.write(md_content)
            logger.info(f"Saved Markdown report to {md_path}")

        return json_path, md_path

    def _generate_markdown_report(self, result: BenchmarkSuiteResult) -> str:
        """Generate Markdown report content."""
        lines = [
            "# Cache Benchmark Report",
            "",
            f"> **Generated**: {result.timestamp}",
            "",
            "---",
            "",
            "## System Information",
            "",
        ]

        for key, value in result.system_info.items():
            lines.append(f"- **{key}**: {value}")

        lines.extend(
            [
                "",
                "---",
                "",
                "## Results Summary",
                "",
                "| Kernel | Size | Mean (ms) | P95 (ms) | Cache Miss % | L1 Miss % | Throughput |",
                "|--------|------|-----------|----------|--------------|-----------|------------|",
            ]
        )

        for kr in result.kernel_results:
            counters = kr.get("perf_counters") or {}
            miss_rate = counters.get("cache_miss_rate", 0) * 100
            l1_miss = counters.get("l1_miss_rate", 0) * 100
            throughput = kr.get("throughput_elements_per_sec", 0)

            lines.append(
                f"| {kr['kernel_name']} | {kr['input_size']} | "
                f"{kr['mean_ms']:.2f} | {kr['p95_ms']:.2f} | "
                f"{miss_rate:.1f}% | {l1_miss:.1f}% | "
                f"{throughput/1e6:.2f}M/s |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## Recommendations",
                "",
            ]
        )

        for rec in result.recommendations:
            lines.append(f"- {rec}")

        lines.extend(
            [
                "",
                "---",
                "",
                "## Configuration",
                "",
                "```json",
                json.dumps(result.config, indent=2, default=str),
                "```",
            ]
        )

        return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Enterprise cache profiling for HSMN native operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation
  python -m benchmarks.benchmark_cache --ops moe --sizes 128 --dry-run

  # Full MoE and SSM analysis
  python -m benchmarks.benchmark_cache --ops moe,ssm --sizes 128,512,2048

  # Comprehensive suite with report
  python -m benchmarks.benchmark_cache --ops moe,ssm,galore,attention \\
      --sizes 64,128,256,512,1024,2048,4096 --report both
        """,
    )

    parser.add_argument(
        "--ops",
        type=str,
        default="moe,ssm,galore",
        help="Comma-separated list of operations to benchmark (default: moe,ssm,galore)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="128,512,2048",
        help="Comma-separated list of input sizes (default: 128,512,2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for benchmarks (default: 1)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Benchmark iterations (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/reports",
        help="Output directory for reports (default: benchmarks/reports)",
    )
    parser.add_argument(
        "--report",
        type=str,
        choices=["json", "markdown", "both"],
        default="both",
        help="Report format (default: both)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without perf (timing only)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = CacheBenchmarkConfig.from_cli(args)

    print("🔬 HSMN Cache Benchmark")
    print(f"   Operations: {config.ops}")
    print(f"   Sizes: {config.sizes}")
    print(f"   Perf enabled: {config.use_perf}")
    print()

    benchmark = CacheBenchmark(config)

    if not benchmark.kernels.perf.is_available and config.use_perf:
        print("⚠️  Linux perf not available, falling back to timing-only mode")
        print("   For hardware counters, ensure perf is installed and accessible.")
        print()

    print("Running benchmarks...")
    result = benchmark.run_suite()

    print()
    print("Generating reports...")
    json_path, md_path = benchmark.generate_report(result, format=args.report)

    print()
    print("=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)

    for rec in result.recommendations:
        print(f"  {rec}")

    if json_path:
        print(f"\n📄 JSON report: {json_path}")
    if md_path:
        print(f"📝 Markdown report: {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
