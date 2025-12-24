# benchmarks/hardware_detection.py
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

"""Comprehensive hardware detection and specification reporting.

Detects CPU, GPU, memory, storage, and OS specifications for benchmark
reproducibility and performance analysis.

Example:
    >>> from benchmarks.hardware_detection import HardwareDetector
    >>> detector = HardwareDetector()
    >>> info = detector.detect_all()
    >>> print(info.cpu.model_name)
    'AMD Ryzen 9 7950X 16-Core Processor'

Command-line usage:
    python -m benchmarks.hardware_detection
"""

import json
import logging
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CPUInfo:
    """CPU specification details.

    Attributes:
        model_name: CPU model name string.
        vendor: CPU vendor (Intel, AMD, ARM).
        architecture: CPU architecture (x86_64, aarch64).
        physical_cores: Number of physical CPU cores.
        logical_cores: Number of logical cores (with hyperthreading).
        base_frequency_mhz: Base clock frequency in MHz.
        max_frequency_mhz: Maximum boost frequency in MHz.
        cache_l1d_kb: L1 data cache size in KB per core.
        cache_l1i_kb: L1 instruction cache size in KB per core.
        cache_l2_kb: L2 cache size in KB per core.
        cache_l3_kb: L3 cache size in KB (shared).
        simd_extensions: List of SIMD extensions (AVX, AVX2, AVX-512, NEON).
        numa_nodes: Number of NUMA nodes.
        governor: Current CPU frequency governor.
    """

    model_name: str = "Unknown"
    vendor: str = "Unknown"
    architecture: str = "Unknown"
    physical_cores: int = 0
    logical_cores: int = 0
    base_frequency_mhz: float = 0.0
    max_frequency_mhz: float = 0.0
    cache_l1d_kb: int = 0
    cache_l1i_kb: int = 0
    cache_l2_kb: int = 0
    cache_l3_kb: int = 0
    simd_extensions: list[str] = field(default_factory=list)
    numa_nodes: int = 1
    governor: str = "Unknown"


@dataclass
class MemoryInfo:
    """Memory specification details.

    Attributes:
        total_gb: Total physical memory in GB.
        available_gb: Available memory in GB.
        used_gb: Used memory in GB.
        swap_total_gb: Total swap space in GB.
        swap_used_gb: Used swap space in GB.
        memory_type: Memory type (DDR4, DDR5).
        speed_mhz: Memory speed in MHz.
        channels: Number of memory channels.
    """

    total_gb: float = 0.0
    available_gb: float = 0.0
    used_gb: float = 0.0
    swap_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    memory_type: str = "Unknown"
    speed_mhz: int = 0
    channels: int = 0


@dataclass
class GPUInfo:
    """GPU specification details.

    Attributes:
        name: GPU model name.
        vendor: GPU vendor (NVIDIA, AMD, Intel).
        memory_total_gb: Total GPU memory in GB.
        memory_used_gb: Used GPU memory in GB.
        compute_capability: CUDA compute capability (for NVIDIA).
        driver_version: GPU driver version.
        cuda_version: CUDA version (if available).
        tensorflow_device: TensorFlow device string.
    """

    name: str = "No GPU detected"
    vendor: str = "None"
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    compute_capability: str = "N/A"
    driver_version: str = "N/A"
    cuda_version: str = "N/A"
    tensorflow_device: str = "CPU"


@dataclass
class StorageInfo:
    """Storage specification details.

    Attributes:
        device_type: Storage type (SSD, HDD, NVMe).
        total_gb: Total storage in GB.
        free_gb: Free storage in GB.
        mount_point: Primary mount point.
        filesystem: Filesystem type.
    """

    device_type: str = "Unknown"
    total_gb: float = 0.0
    free_gb: float = 0.0
    mount_point: str = "/"
    filesystem: str = "Unknown"


@dataclass
class OSInfo:
    """Operating system specification details.

    Attributes:
        system: OS name (Linux, Darwin, Windows).
        release: OS release/kernel version.
        version: OS version string.
        machine: Machine type.
        distribution: Linux distribution name and version.
        python_version: Python interpreter version.
        tensorflow_version: TensorFlow version.
    """

    system: str = "Unknown"
    release: str = "Unknown"
    version: str = "Unknown"
    machine: str = "Unknown"
    distribution: str = "Unknown"
    python_version: str = "Unknown"
    tensorflow_version: str = "Unknown"


@dataclass
class HardwareInfo:
    """Complete hardware specification container.

    Attributes:
        cpu: CPU specifications.
        memory: Memory specifications.
        gpu: GPU specifications.
        storage: Storage specifications.
        os: Operating system specifications.
        timestamp: Detection timestamp.
        benchmark_suitability: Assessment of hardware for benchmarking.
    """

    cpu: CPUInfo = field(default_factory=CPUInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    gpu: GPUInfo = field(default_factory=GPUInfo)
    storage: StorageInfo = field(default_factory=StorageInfo)
    os: OSInfo = field(default_factory=OSInfo)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    benchmark_suitability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class HardwareDetector:
    """Comprehensive hardware detection for benchmarking.

    Detects CPU, GPU, memory, storage, and OS specifications using
    system interfaces and command-line tools.

    Example:
        >>> detector = HardwareDetector()
        >>> info = detector.detect_all()
        >>> print(detector.format_markdown(info))
    """

    def __init__(self) -> None:
        """Initialize hardware detector."""
        self._cache: HardwareInfo | None = None

    def detect_all(self, use_cache: bool = True) -> HardwareInfo:
        """Detect all hardware specifications.

        Args:
            use_cache: Use cached results if available.

        Returns:
            Complete HardwareInfo with all specifications.
        """
        if use_cache and self._cache is not None:
            return self._cache

        info = HardwareInfo(
            cpu=self._detect_cpu(),
            memory=self._detect_memory(),
            gpu=self._detect_gpu(),
            storage=self._detect_storage(),
            os=self._detect_os(),
        )

        # Assess benchmark suitability
        info.benchmark_suitability = self._assess_suitability(info)

        self._cache = info
        return info

    def _detect_cpu(self) -> CPUInfo:
        """Detect CPU specifications."""
        cpu = CPUInfo()
        cpu.architecture = platform.machine()

        try:
            import psutil

            cpu.logical_cores = psutil.cpu_count(logical=True) or 0
            cpu.physical_cores = psutil.cpu_count(logical=False) or 0

            # Get CPU frequency
            freq = psutil.cpu_freq()
            if freq:
                cpu.base_frequency_mhz = freq.current
                cpu.max_frequency_mhz = freq.max if freq.max else freq.current
        except ImportError:
            cpu.logical_cores = os.cpu_count() or 0
            cpu.physical_cores = cpu.logical_cores

        # Parse /proc/cpuinfo for detailed info (Linux)
        if platform.system() == "Linux":
            cpu = self._parse_cpuinfo(cpu)
            cpu = self._detect_simd_extensions(cpu)
            cpu = self._detect_cpu_governor(cpu)
            cpu = self._detect_cache_sizes(cpu)

        return cpu

    def _parse_cpuinfo(self, cpu: CPUInfo) -> CPUInfo:
        """Parse /proc/cpuinfo for CPU details."""
        try:
            with open("/proc/cpuinfo") as f:
                content = f.read()

            # Model name
            match = re.search(r"model name\s*:\s*(.+)", content)
            if match:
                cpu.model_name = match.group(1).strip()

            # Vendor
            match = re.search(r"vendor_id\s*:\s*(.+)", content)
            if match:
                vendor = match.group(1).strip()
                if "AMD" in vendor:
                    cpu.vendor = "AMD"
                elif "Intel" in vendor:
                    cpu.vendor = "Intel"
                else:
                    cpu.vendor = vendor

        except FileNotFoundError:
            pass

        return cpu

    def _detect_simd_extensions(self, cpu: CPUInfo) -> CPUInfo:
        """Detect SIMD instruction set extensions."""
        simd = []

        try:
            with open("/proc/cpuinfo") as f:
                content = f.read()

            flags_match = re.search(r"flags\s*:\s*(.+)", content)
            if flags_match:
                flags = flags_match.group(1).split()

                if "avx512f" in flags:
                    simd.append("AVX-512")
                if "avx2" in flags:
                    simd.append("AVX2")
                if "avx" in flags:
                    simd.append("AVX")
                if "sse4_2" in flags:
                    simd.append("SSE4.2")
                if "sse4_1" in flags:
                    simd.append("SSE4.1")
                if "fma" in flags:
                    simd.append("FMA")
                if "f16c" in flags:
                    simd.append("F16C")

        except FileNotFoundError:
            pass

        # ARM NEON detection
        if cpu.architecture in ("aarch64", "arm64"):
            simd.append("NEON")

        cpu.simd_extensions = simd
        return cpu

    def _detect_cpu_governor(self, cpu: CPUInfo) -> CPUInfo:
        """Detect CPU frequency governor."""
        try:
            gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            if gov_path.exists():
                cpu.governor = gov_path.read_text().strip()
        except Exception:
            pass

        return cpu

    def _detect_cache_sizes(self, cpu: CPUInfo) -> CPUInfo:
        """Detect CPU cache sizes."""
        try:
            result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                output = result.stdout

                # L1d cache
                match = re.search(r"L1d cache:\s*(\d+)", output)
                if match:
                    cpu.cache_l1d_kb = int(match.group(1))

                # L1i cache
                match = re.search(r"L1i cache:\s*(\d+)", output)
                if match:
                    cpu.cache_l1i_kb = int(match.group(1))

                # L2 cache
                match = re.search(r"L2 cache:\s*(\d+)", output)
                if match:
                    cpu.cache_l2_kb = int(match.group(1))

                # L3 cache
                match = re.search(r"L3 cache:\s*(\d+)", output)
                if match:
                    cpu.cache_l3_kb = int(match.group(1))

                # NUMA nodes
                match = re.search(r"NUMA node\(s\):\s*(\d+)", output)
                if match:
                    cpu.numa_nodes = int(match.group(1))

        except Exception:
            pass

        return cpu

    def _detect_memory(self) -> MemoryInfo:
        """Detect memory specifications."""
        mem = MemoryInfo()

        try:
            import psutil

            vm = psutil.virtual_memory()
            mem.total_gb = vm.total / (1024**3)
            mem.available_gb = vm.available / (1024**3)
            mem.used_gb = vm.used / (1024**3)

            swap = psutil.swap_memory()
            mem.swap_total_gb = swap.total / (1024**3)
            mem.swap_used_gb = swap.used / (1024**3)

        except ImportError:
            # Fallback: parse /proc/meminfo
            try:
                with open("/proc/meminfo") as f:
                    content = f.read()

                match = re.search(r"MemTotal:\s*(\d+)", content)
                if match:
                    mem.total_gb = int(match.group(1)) / (1024**2)

                match = re.search(r"MemAvailable:\s*(\d+)", content)
                if match:
                    mem.available_gb = int(match.group(1)) / (1024**2)

            except FileNotFoundError:
                pass

        # Try to detect memory type and speed via dmidecode (requires root)
        mem = self._detect_memory_details(mem)

        return mem

    def _detect_memory_details(self, mem: MemoryInfo) -> MemoryInfo:
        """Detect memory type and speed (requires dmidecode, may need root)."""
        try:
            # Try without sudo first (may fail on non-root)
            result = subprocess.run(
                ["dmidecode", "-t", "memory"],
                capture_output=True,
                text=True,
                timeout=2,  # Short timeout to fail fast
            )
            if result.returncode == 0:
                output = result.stdout

                # Memory type (DDR4, DDR5)
                match = re.search(r"Type:\s*(DDR\d+)", output)
                if match:
                    mem.memory_type = match.group(1)

                # Speed
                match = re.search(r"Speed:\s*(\d+)\s*MT/s", output)
                if match:
                    mem.speed_mhz = int(match.group(1))

        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception:
            pass

        return mem

    def _detect_gpu(self) -> GPUInfo:
        """Detect GPU specifications."""
        gpu = GPUInfo()

        # nvidia-smi for NVIDIA GPUs (fast, no Python import)
        gpu = self._detect_nvidia_gpu(gpu)

        # Only try TensorFlow if nvidia-smi succeeded and we want device confirmation
        # Skip if HIGHNOON_SKIP_TF env var is set (for fast startup)
        if os.environ.get("HIGHNOON_SKIP_TF") != "1" and gpu.vendor == "NVIDIA":
            try:
                import tensorflow as tf

                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    gpu.tensorflow_device = str(gpus[0])
            except Exception:
                pass

        return gpu

    def _detect_nvidia_gpu(self, gpu: GPUInfo) -> GPUInfo:
        """Detect NVIDIA GPU via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines:
                    parts = lines[0].split(",")
                    if len(parts) >= 5:
                        gpu.name = parts[0].strip()
                        gpu.vendor = "NVIDIA"
                        gpu.memory_total_gb = float(parts[1].strip()) / 1024
                        gpu.memory_used_gb = float(parts[2].strip()) / 1024
                        gpu.driver_version = parts[3].strip()
                        gpu.compute_capability = parts[4].strip()

            # CUDA version
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu.cuda_version = result.stdout.strip()

        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"nvidia-smi detection failed: {e}")

        return gpu

    def _detect_storage(self) -> StorageInfo:
        """Detect storage specifications."""
        storage = StorageInfo()

        try:
            import psutil

            # Get root partition info
            partitions = psutil.disk_partitions()
            for part in partitions:
                if part.mountpoint == "/":
                    storage.mount_point = part.mountpoint
                    storage.filesystem = part.fstype

                    usage = psutil.disk_usage(part.mountpoint)
                    storage.total_gb = usage.total / (1024**3)
                    storage.free_gb = usage.free / (1024**3)
                    break

        except ImportError:
            pass

        # Detect SSD vs HDD
        storage = self._detect_storage_type(storage)

        return storage

    def _detect_storage_type(self, storage: StorageInfo) -> StorageInfo:
        """Detect if storage is SSD or HDD."""
        try:
            # Check for NVMe
            nvme_path = Path("/sys/class/nvme")
            if nvme_path.exists() and list(nvme_path.iterdir()):
                storage.device_type = "NVMe SSD"
                return storage

            # Check rotational flag for SATA devices
            block_path = Path("/sys/block")
            if block_path.exists():
                for device in block_path.iterdir():
                    rot_path = device / "queue" / "rotational"
                    if rot_path.exists():
                        is_rotational = rot_path.read_text().strip() == "1"
                        if not is_rotational:
                            storage.device_type = "SATA SSD"
                        else:
                            storage.device_type = "HDD"
                        break

        except Exception:
            pass

        return storage

    def _detect_os(self) -> OSInfo:
        """Detect operating system specifications."""
        os_info = OSInfo(
            system=platform.system(),
            release=platform.release(),
            version=platform.version(),
            machine=platform.machine(),
            python_version=sys.version.split()[0],
        )

        # Linux distribution
        if platform.system() == "Linux":
            try:
                result = subprocess.run(
                    ["lsb_release", "-d"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    os_info.distribution = result.stdout.split(":")[-1].strip()
            except Exception:
                try:
                    with open("/etc/os-release") as f:
                        for line in f:
                            if line.startswith("PRETTY_NAME="):
                                os_info.distribution = line.split("=")[1].strip().strip('"')
                                break
                except FileNotFoundError:
                    pass

        # TensorFlow version (skip if fast startup requested)
        if os.environ.get("HIGHNOON_SKIP_TF") != "1":
            try:
                import tensorflow as tf

                os_info.tensorflow_version = tf.__version__
            except ImportError:
                os_info.tensorflow_version = "Not installed"
        else:
            os_info.tensorflow_version = "Skipped (fast mode)"

        return os_info

    def _assess_suitability(self, info: HardwareInfo) -> dict[str, Any]:
        """Assess hardware suitability for benchmarking.

        Args:
            info: Complete hardware information.

        Returns:
            Dictionary with suitability assessment.
        """
        assessment = {
            "max_recommended_context": 4096,
            "max_recommended_batch_size": 1,
            "estimated_1m_context_memory_gb": 0.0,
            "can_run_1m_context": False,
            "simd_optimized": False,
            "warnings": [],
            "recommendations": [],
        }

        # Context length estimation based on memory
        # Rough estimate: embedding + attention = 2 * seq_len * hidden_dim * batch * 4 bytes
        available_gb = info.memory.available_gb
        hidden_dim = 512  # Default HSMN hidden dim

        # Memory per 1K tokens (rough estimate)
        mem_per_1k = (1024 * hidden_dim * 4 * 2) / (1024**3)  # GB

        # Max context that fits in 80% of available memory
        max_context = int((available_gb * 0.8) / mem_per_1k * 1024)
        assessment["max_recommended_context"] = min(max_context, 1_000_000)

        # 1M context memory estimation
        assessment["estimated_1m_context_memory_gb"] = 1_000_000 * mem_per_1k / 1024
        assessment["can_run_1m_context"] = (
            assessment["estimated_1m_context_memory_gb"] < available_gb * 0.9
        )

        # SIMD optimization check
        if any(ext in info.cpu.simd_extensions for ext in ["AVX2", "AVX-512"]):
            assessment["simd_optimized"] = True
        else:
            assessment["warnings"].append(
                "No AVX2/AVX-512 detected - performance may be suboptimal"
            )

        # Governor check
        if info.cpu.governor not in ("performance", "schedutil"):
            assessment["warnings"].append(
                f"CPU governor is '{info.cpu.governor}' - consider 'performance' for benchmarks"
            )

        # Memory recommendations
        if info.memory.total_gb < 16:
            assessment["warnings"].append("Less than 16GB RAM - long-context tests will be limited")
            assessment["max_recommended_context"] = min(assessment["max_recommended_context"], 8192)
        elif info.memory.total_gb >= 64:
            assessment["recommendations"].append(
                "64GB+ RAM available - can run full 1M context tests"
            )

        # Batch size recommendation based on memory
        if info.memory.available_gb >= 32:
            assessment["max_recommended_batch_size"] = 8
        elif info.memory.available_gb >= 16:
            assessment["max_recommended_batch_size"] = 4
        elif info.memory.available_gb >= 8:
            assessment["max_recommended_batch_size"] = 2

        return assessment

    def format_markdown(self, info: HardwareInfo | None = None) -> str:
        """Format hardware info as markdown.

        Args:
            info: Hardware info to format (detects if None).

        Returns:
            Markdown formatted hardware report.
        """
        if info is None:
            info = self.detect_all()

        lines = [
            "# Hardware Specification Report",
            "",
            f"**Generated**: {info.timestamp}",
            "",
            "---",
            "",
            "## CPU",
            "",
            "| Specification | Value |",
            "|---------------|-------|",
            f"| **Model** | {info.cpu.model_name} |",
            f"| **Vendor** | {info.cpu.vendor} |",
            f"| **Architecture** | {info.cpu.architecture} |",
            f"| **Physical Cores** | {info.cpu.physical_cores} |",
            f"| **Logical Cores** | {info.cpu.logical_cores} |",
            f"| **Base Frequency** | {info.cpu.base_frequency_mhz:.0f} MHz |",
            f"| **Max Frequency** | {info.cpu.max_frequency_mhz:.0f} MHz |",
            f"| **L1d Cache** | {info.cpu.cache_l1d_kb} KB/core |",
            f"| **L2 Cache** | {info.cpu.cache_l2_kb} KB/core |",
            f"| **L3 Cache** | {info.cpu.cache_l3_kb} KB |",
            f"| **SIMD Extensions** | {', '.join(info.cpu.simd_extensions) or 'None detected'} |",
            f"| **Governor** | {info.cpu.governor} |",
            f"| **NUMA Nodes** | {info.cpu.numa_nodes} |",
            "",
            "## Memory",
            "",
            "| Specification | Value |",
            "|---------------|-------|",
            f"| **Total RAM** | {info.memory.total_gb:.1f} GB |",
            f"| **Available** | {info.memory.available_gb:.1f} GB |",
            f"| **Used** | {info.memory.used_gb:.1f} GB |",
            f"| **Type** | {info.memory.memory_type} |",
            f"| **Speed** | {info.memory.speed_mhz} MT/s |",
            f"| **Swap Total** | {info.memory.swap_total_gb:.1f} GB |",
            "",
            "## GPU",
            "",
            "| Specification | Value |",
            "|---------------|-------|",
            f"| **Model** | {info.gpu.name} |",
            f"| **Vendor** | {info.gpu.vendor} |",
            f"| **VRAM** | {info.gpu.memory_total_gb:.1f} GB |",
            f"| **Compute Capability** | {info.gpu.compute_capability} |",
            f"| **Driver** | {info.gpu.driver_version} |",
            f"| **CUDA** | {info.gpu.cuda_version} |",
            "",
            "## Storage",
            "",
            "| Specification | Value |",
            "|---------------|-------|",
            f"| **Type** | {info.storage.device_type} |",
            f"| **Total** | {info.storage.total_gb:.0f} GB |",
            f"| **Free** | {info.storage.free_gb:.0f} GB |",
            f"| **Filesystem** | {info.storage.filesystem} |",
            "",
            "## Operating System",
            "",
            "| Specification | Value |",
            "|---------------|-------|",
            f"| **OS** | {info.os.system} |",
            f"| **Distribution** | {info.os.distribution} |",
            f"| **Kernel** | {info.os.release} |",
            f"| **Python** | {info.os.python_version} |",
            f"| **TensorFlow** | {info.os.tensorflow_version} |",
            "",
        ]

        # Benchmark suitability
        suit = info.benchmark_suitability
        lines.extend(
            [
                "## Benchmark Suitability",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| **Max Recommended Context** | {suit.get('max_recommended_context', 4096):,} tokens |",
                f"| **Max Recommended Batch Size** | {suit.get('max_recommended_batch_size', 1)} |",
                f"| **Can Run 1M Context** | {'✅ Yes' if suit.get('can_run_1m_context') else '❌ No'} |",
                f"| **Est. 1M Context Memory** | {suit.get('estimated_1m_context_memory_gb', 0):.1f} GB |",
                f"| **SIMD Optimized** | {'✅ Yes' if suit.get('simd_optimized') else '⚠️ Limited'} |",
                "",
            ]
        )

        # Warnings
        warnings = suit.get("warnings", [])
        if warnings:
            lines.extend(
                [
                    "> [!WARNING]",
                    "> **Performance Warnings:**",
                ]
            )
            for warning in warnings:
                lines.append(f"> - {warning}")
            lines.append("")

        # Recommendations
        recommendations = suit.get("recommendations", [])
        if recommendations:
            lines.extend(
                [
                    "> [!TIP]",
                    "> **Recommendations:**",
                ]
            )
            for rec in recommendations:
                lines.append(f"> - {rec}")
            lines.append("")

        return "\n".join(lines)

    def format_summary(self, info: HardwareInfo | None = None) -> str:
        """Format a brief hardware summary.

        Args:
            info: Hardware info to format (detects if None).

        Returns:
            Brief summary string.
        """
        if info is None:
            info = self.detect_all()

        return (
            f"CPU: {info.cpu.model_name} ({info.cpu.physical_cores}C/{info.cpu.logical_cores}T) | "
            f"RAM: {info.memory.total_gb:.0f}GB | "
            f"GPU: {info.gpu.name} | "
            f"Max Context: {info.benchmark_suitability.get('max_recommended_context', 4096):,}"
        )


def print_system_info() -> None:
    """Print system information to stdout."""
    detector = HardwareDetector()
    info = detector.detect_all()

    print("=" * 70)
    print("HSMN Benchmark Suite - Hardware Detection")
    print("=" * 70)
    print()
    print(f"CPU: {info.cpu.model_name}")
    print(f"     {info.cpu.physical_cores} cores / {info.cpu.logical_cores} threads")
    print(
        f"     {info.cpu.base_frequency_mhz:.0f} MHz (boost: {info.cpu.max_frequency_mhz:.0f} MHz)"
    )
    print(f"     SIMD: {', '.join(info.cpu.simd_extensions) or 'None'}")
    print()
    print(
        f"Memory: {info.memory.total_gb:.1f} GB total, {info.memory.available_gb:.1f} GB available"
    )
    print()
    print(f"GPU: {info.gpu.name}")
    if info.gpu.vendor != "None":
        print(f"     VRAM: {info.gpu.memory_total_gb:.1f} GB")
    print()
    print(f"OS: {info.os.distribution or info.os.system} ({info.os.release})")
    print(f"Python: {info.os.python_version}")
    print(f"TensorFlow: {info.os.tensorflow_version}")
    print()
    print(
        f"Max Recommended Context: {info.benchmark_suitability.get('max_recommended_context', 4096):,} tokens"
    )
    print(
        f"Can Run 1M Context: {'Yes' if info.benchmark_suitability.get('can_run_1m_context') else 'No'}"
    )
    print("=" * 70)


def main() -> int:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="HSMN Hardware Detection")
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path",
    )

    args = parser.parse_args()

    detector = HardwareDetector()
    info = detector.detect_all()

    if args.format == "text":
        output = detector.format_summary(info)
        print_system_info()
    elif args.format == "markdown":
        output = detector.format_markdown(info)
        print(output)
    else:  # json
        output = info.to_json()
        print(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            if args.format == "json":
                f.write(output)
            elif args.format == "markdown":
                f.write(output)
            else:
                f.write(detector.format_markdown(info))
        print(f"\nSaved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
