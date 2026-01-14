# benchmarks/benchmark_config.py
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

"""Configuration dataclasses for HSMN benchmark suite.

This module defines configuration parameters for all benchmark types,
including model settings, sequence ranges, and output formats.

Example:
    >>> from benchmarks.benchmark_config import BenchmarkConfig
    >>> config = BenchmarkConfig.quick()
    >>> print(config.sequence_lengths)
    [128, 256, 512]
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import architecture parameters from central config for consistency with training/HPO
from highnoon.config import (
    ENABLE_QUANTUM_MEMORY_REPLAY,
    ENTANGLEMENT_REGULARIZATION,
    HD_DIM_EMBEDDING,
    HD_DIM_MOE,
    HD_DIM_SPATIAL,
    HD_DIM_TIMECRYSTAL,
    MAX_CONTEXT_LEN,
    NUM_EXPERTS,
    REASONING_BLOCK_PATTERN,
    REASONING_LAYERS,
    TOP_K,
    USE_QUANTUM_HOLOGRAPHIC_MEMORY,
    VOCAB_SIZE,
)


def get_available_ram_gb() -> float:
    """Get available system RAM in gigabytes.

    Returns:
        Available RAM in GB.
    """
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        # Fallback: assume 16GB available
        return 16.0


@dataclass
class BenchmarkMode:
    """Configuration for benchmark execution mode.

    Controls whether benchmarks use production-optimized paths (streaming,
    QMR, fused ops) or baseline unoptimized paths for comparison.

    Attributes:
        mode: Execution mode - 'production', 'baseline', or 'comparison'.
        use_streaming_inference: Use StreamingInferenceWrapper for O(1) memory.
        use_qsg_generation: Use QSGGenerator for parallel generation.
        use_fused_training: Use fused C++ train_step_op with QMR.
        use_quantum_memory_replay: Enable QMR for O(log n) memory.
        streaming_chunk_size: Chunk size for streaming inference.
        auto_memory_scaling: Auto-adjust batch/seq based on available RAM.

    Example:
        >>> mode = BenchmarkMode.production()
        >>> mode.use_streaming_inference
        True
        >>> mode = BenchmarkMode.baseline()
        >>> mode.use_streaming_inference
        False
    """

    mode: str = "production"
    use_streaming_inference: bool = True
    use_qsg_generation: bool = True
    use_fused_training: bool = True
    use_quantum_memory_replay: bool = True
    streaming_chunk_size: int = 4096
    auto_memory_scaling: bool = True

    @classmethod
    def production(cls) -> "BenchmarkMode":
        """Create production mode with all optimizations enabled.

        Uses streaming inference, QSG, fused training ops, and QMR.
        This reflects actual production performance.

        Returns:
            BenchmarkMode with production settings.
        """
        return cls(
            mode="production",
            use_streaming_inference=True,
            use_qsg_generation=True,
            use_fused_training=True,
            use_quantum_memory_replay=True,
            streaming_chunk_size=4096,
            auto_memory_scaling=True,
        )

    @classmethod
    def baseline(cls) -> "BenchmarkMode":
        """Create baseline mode without optimizations.

        Uses direct model calls without streaming, QMR, or fused ops.
        This measures theoretical peak throughput (but may OOM on long sequences).

        Returns:
            BenchmarkMode with baseline settings.
        """
        return cls(
            mode="baseline",
            use_streaming_inference=False,
            use_qsg_generation=False,
            use_fused_training=False,
            use_quantum_memory_replay=False,
            streaming_chunk_size=4096,
            auto_memory_scaling=True,
        )

    @classmethod
    def comparison(cls) -> "BenchmarkMode":
        """Create comparison mode that runs both production and baseline.

        Useful for measuring the impact of quantum optimizations.

        Returns:
            BenchmarkMode with comparison settings.
        """
        return cls(
            mode="comparison",
            use_streaming_inference=True,
            use_qsg_generation=True,
            use_fused_training=True,
            use_quantum_memory_replay=True,
            streaming_chunk_size=4096,
            auto_memory_scaling=True,
        )

    def get_max_sequence_length(self, available_ram_gb: float | None = None) -> int:
        """Get maximum safe sequence length based on available RAM and mode.

        Args:
            available_ram_gb: Available RAM in GB. Auto-detected if None.

        Returns:
            Maximum sequence length that won't OOM.
        """
        if available_ram_gb is None:
            available_ram_gb = get_available_ram_gb()

        # Reserve some RAM for system (20%)
        usable_ram_gb = available_ram_gb * 0.8

        if self.use_streaming_inference:
            # Streaming is O(1) memory, limited only by chunk processing
            return 1_048_576  # 1M tokens (Lite limit)

        # For baseline mode, estimate based on activation memory
        # Rough estimate: 1GB per 32K tokens for typical model
        tokens_per_gb = 32768
        max_tokens = int(usable_ram_gb * tokens_per_gb)

        # Cap at reasonable values
        return min(max_tokens, 131072)

    def get_safe_batch_sizes(
        self,
        seq_length: int,
        available_ram_gb: float | None = None,
    ) -> list[int]:
        """Get safe batch sizes for a given sequence length.

        Args:
            seq_length: Sequence length to test.
            available_ram_gb: Available RAM in GB. Auto-detected if None.

        Returns:
            List of safe batch sizes.
        """
        if available_ram_gb is None:
            available_ram_gb = get_available_ram_gb()

        usable_ram_gb = available_ram_gb * 0.8

        if self.use_streaming_inference:
            # Streaming allows larger batches
            return [1, 2, 4, 8, 16]

        # Estimate memory per token per batch
        # ~4 bytes * embedding_dim * num_blocks * 2 (forward + grad)
        bytes_per_token_per_batch = 4 * 128 * 6 * 2  # Conservative estimate

        max_batch = int(usable_ram_gb * (1024**3) / (seq_length * bytes_per_token_per_batch))
        max_batch = max(1, min(max_batch, 16))

        # Return powers of 2 up to max
        batch_sizes = []
        b = 1
        while b <= max_batch:
            batch_sizes.append(b)
            b *= 2

        return batch_sizes if batch_sizes else [1]


@dataclass
class ModelConfig:
    """Configuration for HSMN model instantiation.

    Imports default values from highnoon/config.py for consistency with
    training and HPO sweeps. Override individual values as needed.

    Attributes:
        vocab_size: Vocabulary size for embeddings.
        embedding_dim: Token embedding dimension.
        num_reasoning_blocks: Number of reasoning blocks (max 24 Lite).
        num_experts: Number of MoE experts (max 12 Lite).
        top_k: Top-K experts per token.
        block_pattern: Reasoning block pattern string.
        max_seq_length: Maximum sequence length supported.
    """

    # Import defaults from central config for consistency
    vocab_size: int = VOCAB_SIZE
    # CRITICAL: embedding_dim MUST match HD_DIM_MOE for SuperposedExpert C++ ops
    # See enterprise_training_engine_audit.py lines 1018-1025 for validation
    embedding_dim: int = HD_DIM_MOE  # 512 - matches validated audit script config
    num_reasoning_blocks: int = REASONING_LAYERS
    num_experts: int = NUM_EXPERTS
    top_k: int = TOP_K
    block_pattern: str = REASONING_BLOCK_PATTERN
    max_seq_length: int = MAX_CONTEXT_LEN
    # Quantum features - use values from config.py (same as training/HPO)
    use_quantum_memory_replay: bool = ENABLE_QUANTUM_MEMORY_REPLAY
    use_entanglement_loss: bool = ENTANGLEMENT_REGULARIZATION > 0
    use_quantum_holographic_memory: bool = USE_QUANTUM_HOLOGRAPHIC_MEMORY
    # Per-layer HD dimensions for proper C++ op compatibility
    hd_dim_embedding: int = HD_DIM_EMBEDDING
    hd_dim_spatial: int = HD_DIM_SPATIAL
    hd_dim_timecrystal: int = HD_DIM_TIMECRYSTAL
    hd_dim_moe: int = HD_DIM_MOE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for model initialization."""
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_reasoning_blocks": self.num_reasoning_blocks,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "block_pattern": self.block_pattern,
            "max_seq_length": self.max_seq_length,
            "use_quantum_memory_replay": self.use_quantum_memory_replay,
            "use_entanglement_loss": self.use_entanglement_loss,
            "use_quantum_holographic_memory": self.use_quantum_holographic_memory,
        }


@dataclass
class ThroughputConfig:
    """Configuration for throughput benchmarks.

    Attributes:
        batch_sizes: Batch sizes to test for scaling.
        sequence_lengths: Sequence lengths to test.
        warmup_iterations: Warmup iterations before timing.
        benchmark_iterations: Number of timed iterations.
        measure_generation: Whether to measure autoregressive generation.
        generation_tokens: Tokens to generate for generation benchmark.
    """

    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    sequence_lengths: list[int] = field(
        default_factory=lambda: [128, 512, 2048, 8192, 32768, 131072]
    )
    warmup_iterations: int = 5
    benchmark_iterations: int = 20
    measure_generation: bool = True
    generation_tokens: int = 64


@dataclass
class PerplexityConfig:
    """Configuration for perplexity benchmarks.

    Attributes:
        dataset_name: Name of dataset to evaluate ("synthetic", "wikitext", "custom").
        custom_texts: Custom texts for evaluation if dataset_name is "custom".
        stride: Stride for sliding window evaluation.
        max_samples: Maximum samples to evaluate.
    """

    dataset_name: str = "synthetic"
    custom_texts: list[str] | None = None
    stride: int = 512
    max_samples: int = 500  # ~60+ batches at batch_size=8


@dataclass
class ConfidenceConfig:
    """Configuration for confidence/calibration benchmarks.

    Attributes:
        num_samples: Number of samples for self-consistency.
        temperatures: Temperatures for sampling.
        num_bins: Number of bins for calibration curves.
    """

    num_samples: int = 5
    temperatures: list[float] = field(default_factory=lambda: [0.7, 1.0, 1.3])
    num_bins: int = 10


@dataclass
class MemoryConfig:
    """Configuration for memory benchmarks.

    Attributes:
        sequence_lengths: Sequence lengths to profile.
        batch_size: Batch size for memory tests.
        measure_peak: Whether to measure peak memory.
        measure_per_component: Whether to break down by component.
    """

    sequence_lengths: list[int] = field(
        default_factory=lambda: [128, 512, 2048, 8192, 32768, 65536, 131072]
    )
    batch_size: int = 1
    measure_peak: bool = True
    measure_per_component: bool = True


@dataclass
class ComparisonConfig:
    """Configuration for architecture comparison benchmarks.

    Attributes:
        include_transformer: Include transformer baseline.
        include_mamba_only: Include Mamba-only baseline.
        include_linear_attention: Include linear attention baseline.
        sequence_lengths: Sequence lengths for scaling comparison.
    """

    include_transformer: bool = True
    include_mamba_only: bool = True
    include_linear_attention: bool = True
    sequence_lengths: list[int] = field(
        default_factory=lambda: [128, 256, 512, 1024, 2048, 4096, 8192]
    )


@dataclass
class BenchmarkConfig:
    """Master configuration for all benchmark types.

    Attributes:
        name: Benchmark run name/identifier.
        output_dir: Directory for output files.
        output_format: Output format ("json", "markdown", "both").
        model: Model configuration.
        throughput: Throughput benchmark configuration.
        perplexity: Perplexity benchmark configuration.
        confidence: Confidence benchmark configuration.
        memory: Memory benchmark configuration.
        comparison: Comparison benchmark configuration.
        verbose: Enable verbose logging.
        seed: Random seed for reproducibility.

    Example:
        >>> config = BenchmarkConfig.quick()
        >>> harness = BenchmarkHarness(config)
    """

    name: str = "hsmn_benchmark"
    output_dir: Path = field(default_factory=lambda: Path("benchmarks/reports"))
    output_format: str = "both"
    model: ModelConfig = field(default_factory=ModelConfig)
    benchmark_mode: BenchmarkMode = field(default_factory=BenchmarkMode.production)
    throughput: ThroughputConfig = field(default_factory=ThroughputConfig)
    perplexity: PerplexityConfig = field(default_factory=PerplexityConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    verbose: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        """Ensure output_dir is a Path object."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @classmethod
    def quick(cls) -> "BenchmarkConfig":
        """Create a quick benchmark configuration for validation.

        Uses smaller model, fewer iterations, and shorter sequences
        for rapid testing of benchmark functionality.

        Returns:
            BenchmarkConfig with minimal settings.
        """
        return cls(
            name="quick_validation",
            model=ModelConfig(
                embedding_dim=128,  # Fast benchmarks
                num_reasoning_blocks=REASONING_LAYERS,
                num_experts=NUM_EXPERTS,
                max_seq_length=8192,
                # Quantum features use config.py defaults (enabled)
            ),
            throughput=ThroughputConfig(
                batch_sizes=[1, 2],
                sequence_lengths=[128, 512, 1024, 2048],
                warmup_iterations=2,
                benchmark_iterations=5,
                generation_tokens=16,
            ),
            perplexity=PerplexityConfig(
                max_samples=100,
            ),
            confidence=ConfidenceConfig(
                num_samples=3,
            ),
            memory=MemoryConfig(
                sequence_lengths=[128, 512, 1024, 2048],
            ),
            comparison=ComparisonConfig(
                include_transformer=False,
                sequence_lengths=[128, 512, 1024, 2048],
            ),
        )

    @classmethod
    def micro(cls) -> "BenchmarkConfig":
        """Create a micro benchmark configuration for minimum latency.

        Uses ~1M parameter model for fastest possible iteration.
        Ideal for quick sanity checks and development testing.

        Returns:
            BenchmarkConfig with minimal model size (~1M params).
        """
        return cls(
            name="micro_benchmark",
            model=ModelConfig(
                embedding_dim=64,  # Very small for ~1M params
                num_reasoning_blocks=2,
                num_experts=2,
                max_seq_length=1024,
                # Quantum features use config.py defaults (enabled)
            ),
            throughput=ThroughputConfig(
                batch_sizes=[1],
                sequence_lengths=[32, 64, 128],
                warmup_iterations=1,
                benchmark_iterations=2,
                measure_generation=True,
                generation_tokens=8,
            ),
            perplexity=PerplexityConfig(
                max_samples=20,
            ),
            confidence=ConfidenceConfig(
                num_samples=1,
            ),
            memory=MemoryConfig(
                sequence_lengths=[32, 64, 128],
            ),
            comparison=ComparisonConfig(
                include_transformer=False,
                include_mamba_only=False,
                include_linear_attention=False,
                sequence_lengths=[32, 64, 128],
            ),
        )

    @classmethod
    def speed(cls) -> "BenchmarkConfig":
        """Create a speed-focused benchmark configuration.

        Optimized for fast iteration with stateful generation enabled.
        Uses O(1) cached generation to demonstrate true throughput potential.
        Model size ~5M parameters for quick but meaningful benchmarks.

        Returns:
            BenchmarkConfig optimized for speed testing.
        """
        return cls(
            name="speed_benchmark",
            model=ModelConfig(
                embedding_dim=128,
                num_reasoning_blocks=REASONING_LAYERS,
                num_experts=NUM_EXPERTS,
                max_seq_length=2048,
                # Quantum features use config.py defaults (enabled)
            ),
            throughput=ThroughputConfig(
                batch_sizes=[1],
                sequence_lengths=[64, 128, 256],
                warmup_iterations=1,
                benchmark_iterations=2,
                measure_generation=True,
                generation_tokens=16,  # Fewer tokens, fast iteration
            ),
            perplexity=PerplexityConfig(
                max_samples=30,
            ),
            confidence=ConfidenceConfig(
                num_samples=2,
            ),
            memory=MemoryConfig(
                sequence_lengths=[64, 128, 256],
            ),
            comparison=ComparisonConfig(
                include_transformer=False,
                include_mamba_only=False,
                include_linear_attention=False,
                sequence_lengths=[64, 128, 256],
            ),
        )

    @classmethod
    def full(cls) -> "BenchmarkConfig":
        """Create a full benchmark configuration for comprehensive evaluation.

        Uses production-scale settings for thorough benchmarking.

        Returns:
            BenchmarkConfig with full settings.
        """
        return cls(
            name="full_benchmark",
            model=ModelConfig(
                embedding_dim=128,  # Fast benchmarks
                num_reasoning_blocks=REASONING_LAYERS,
                num_experts=NUM_EXPERTS,
                # Quantum features use config.py defaults (enabled)
            ),
            throughput=ThroughputConfig(
                batch_sizes=[1, 2, 4, 8, 16],
                sequence_lengths=[128, 256, 512, 1024, 2048, 4096],
                warmup_iterations=10,
                benchmark_iterations=50,
                generation_tokens=128,
            ),
            perplexity=PerplexityConfig(
                max_samples=500,
            ),
            confidence=ConfidenceConfig(
                num_samples=10,
            ),
            memory=MemoryConfig(
                sequence_lengths=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
            ),
            comparison=ComparisonConfig(
                sequence_lengths=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
            ),
        )

    @classmethod
    def long_context(cls) -> "BenchmarkConfig":
        """Create a configuration focused on long-context evaluation.

        Tests the O(n) advantage of HSMN at extended sequence lengths.

        Returns:
            BenchmarkConfig optimized for long-context testing.
        """
        return cls(
            name="long_context_benchmark",
            model=ModelConfig(
                embedding_dim=128,  # Fast benchmarks
                num_reasoning_blocks=REASONING_LAYERS,
                num_experts=NUM_EXPERTS,
                max_seq_length=65536,
                # Quantum features use config.py defaults (enabled)
            ),
            throughput=ThroughputConfig(
                batch_sizes=[1],
                sequence_lengths=[1024, 4096, 16384, 32768, 65536],
                warmup_iterations=3,
                benchmark_iterations=10,
                measure_generation=False,
            ),
            memory=MemoryConfig(
                sequence_lengths=[1024, 4096, 16384, 32768, 65536],
            ),
            comparison=ComparisonConfig(
                sequence_lengths=[1024, 4096, 16384, 32768],
            ),
        )

    @classmethod
    def enterprise(cls) -> "BenchmarkConfig":
        """Create enterprise-grade benchmark configuration.

        Comprehensive 128k context testing with full statistical significance.
        Designed for production quality validation.

        Returns:
            BenchmarkConfig for enterprise benchmarking.
        """
        return cls(
            name="enterprise_benchmark",
            model=ModelConfig(
                embedding_dim=128,  # Fast benchmarks
                num_reasoning_blocks=REASONING_LAYERS,
                num_experts=NUM_EXPERTS,
                max_seq_length=131072,
                # Quantum features use config.py defaults (enabled)
            ),
            throughput=ThroughputConfig(
                batch_sizes=[1, 2, 4],
                sequence_lengths=[128, 512, 2048, 8192, 32768, 65536, 131072],
                warmup_iterations=5,
                benchmark_iterations=20,
                measure_generation=False,  # Skip generation for long context
            ),
            perplexity=PerplexityConfig(
                max_samples=1000,  # ~125+ batches
            ),
            confidence=ConfidenceConfig(
                num_samples=10,
            ),
            memory=MemoryConfig(
                sequence_lengths=[128, 512, 2048, 8192, 32768, 65536, 131072],
            ),
            comparison=ComparisonConfig(
                include_transformer=False,  # Transformer OOMs at 128k
                sequence_lengths=[128, 512, 2048, 8192, 32768, 65536, 131072],
            ),
        )

    @classmethod
    def ultra(cls) -> "BenchmarkConfig":
        """Create ultra-long context benchmark configuration.

        Tests up to 1M token context length for maximum scaling analysis.
        Uses 100M parameter model for comprehensive evaluation.
        Requires 64GB+ RAM for full suite. Auto-scales on lower memory systems.

        Returns:
            BenchmarkConfig for 1M token context benchmarking with 100M model.
        """
        return cls(
            name="ultra_benchmark",
            model=ModelConfig(
                embedding_dim=128,  # Fast benchmarks
                num_reasoning_blocks=REASONING_LAYERS,
                num_experts=NUM_EXPERTS,
                max_seq_length=1048576,  # 1M tokens
                # Quantum features use config.py defaults (enabled)
            ),
            throughput=ThroughputConfig(
                batch_sizes=[1],  # Batch=1 for ultra-long context
                sequence_lengths=[
                    1024,
                    4096,
                    16384,
                    65536,
                    131072,
                    262144,
                    524288,
                    1048576,  # 256K, 512K, 1M
                ],
                warmup_iterations=2,
                benchmark_iterations=5,
                measure_generation=False,
            ),
            perplexity=PerplexityConfig(
                max_samples=100,  # Reduced for memory
            ),
            confidence=ConfidenceConfig(
                num_samples=5,
            ),
            memory=MemoryConfig(
                sequence_lengths=[
                    1024,
                    4096,
                    16384,
                    65536,
                    131072,
                    262144,
                    524288,
                    1048576,
                ],
            ),
            comparison=ComparisonConfig(
                include_transformer=False,
                include_linear_attention=False,  # Only HSMN at ultra-long
                sequence_lengths=[1024, 4096, 16384, 65536, 131072],
            ),
        )
