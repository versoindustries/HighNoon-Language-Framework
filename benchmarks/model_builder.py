# benchmarks/model_builder.py
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

"""HSMN model builder for benchmarks - aligned with HPO trial runner pattern.

Builds full HSMN models with all quantum enhancement phases enabled,
matching the architecture used in HPO sweeps and training.

Example:
    >>> from benchmarks.model_builder import build_benchmark_model
    >>> model = build_benchmark_model("3b")
    >>> print(model.count_params())

The model builder ensures benchmark models exactly match the production
training architecture for accurate performance measurement.
"""

import logging
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class HSMNModelConfig:
    """Configuration for HSMN model construction.

    Matches the hyperparameter structure from hpo_trial_runner.py to ensure
    benchmark models are identical to trained models.

    Attributes:
        hidden_dim: Hidden dimension / embedding size.
        num_reasoning_blocks: Number of hybrid reasoning blocks (max 24 Lite).
        num_heads: Number of attention heads.
        ff_dim: Feed-forward dimension.
        ff_expansion: FFN expansion factor (used if ff_dim not set).
        vocab_size: Vocabulary size.
        max_seq_length: Maximum sequence length supported.

        # Mamba2 SSM parameters
        mamba_state_dim: SSM state dimension.
        mamba_conv_dim: Convolution kernel size.
        mamba_expand: Expansion factor for SSM.

        # MoE parameters
        num_moe_experts: Number of MoE experts (max 12 Lite).
        moe_top_k: Top-K experts per token.
        moe_capacity_factor: Capacity factor for load balancing.

        # WLAM parameters
        wlam_num_heads: Number of WLAM attention heads.
        wlam_kernel_size: Wavelet kernel size.
        wlam_num_landmarks: Number of landmarks for linear attention.

        # TensorTrain decomposition
        tt_rank_middle: TT-rank for middle layers.

        # Quantum parameters
        superposition_dim: Superposition basis dimension.
        hamiltonian_hidden_dim: Hamiltonian controller hidden dim.

        # Quantum enhancement flags (Phases 1-84)
        use_quantum_memory_replay: Phase 6 - O(log n) checkpointing.
        use_entanglement_loss: Phase 7 - Entropy regularization.
        use_quantum_holographic_memory: Phase 25 - Associative memory.
        use_unitary_residual: Phase 26 - Unitary residual connections.
        use_quantum_norm: Phase 27 - Quantum normalization.
        use_unitary_expert: Phase 28 - Unitary MoE experts.
        use_qhpm_crystallization: Phase 25+ - Memory crystallization.
    """

    # Core architecture
    hidden_dim: int = 512
    num_reasoning_blocks: int = 6
    num_heads: int = 8
    ff_dim: int | None = None
    ff_expansion: int = 4
    vocab_size: int = 32000
    max_seq_length: int = 131072
    dropout_rate: float = 0.1

    # Mamba2 SSM
    mamba_state_dim: int = 64
    mamba_conv_dim: int = 4
    mamba_expand: int = 2

    # MoE
    num_moe_experts: int = 8
    moe_top_k: int = 2
    moe_capacity_factor: float = 1.25

    # WLAM
    wlam_num_heads: int = 8
    wlam_kernel_size: int = 3
    wlam_num_landmarks: int = 32

    # TensorTrain
    tt_rank_middle: int = 16

    # Quantum
    superposition_dim: int = 2
    hamiltonian_hidden_dim: int = 256

    # Quantum enhancement flags
    use_quantum_memory_replay: bool = True
    use_entanglement_loss: bool = True
    use_quantum_holographic_memory: bool = True
    use_unitary_residual: bool = True
    use_quantum_norm: bool = True
    use_unitary_expert: bool = True
    use_qhpm_crystallization: bool = True

    # Quantum v5.0 phases (47-84)
    use_quantum_measurement_dropout: bool = True
    qmd_drop_rate: float = 0.1
    use_q_ssm_gating: bool = True
    use_quantum_coherence_bus: bool = True
    use_hyperdimensional_embedding: bool = True
    use_hypertokens: bool = True
    use_majorana_position: bool = True
    use_born_rule_loss: bool = True
    use_qasa_attention: bool = True
    use_mpqr_reasoning: bool = True
    mpqr_num_paths: int = 8
    use_td_moe: bool = True
    use_vqem: bool = True
    use_random_natural_gradient: bool = True
    use_quantum_crystallization: bool = True
    use_multi_stage_hamiltonian: bool = True
    use_spini_integrator: bool = True
    use_qcot_reasoning: bool = True

    def __post_init__(self) -> None:
        """Compute derived values."""
        if self.ff_dim is None:
            self.ff_dim = self.hidden_dim * self.ff_expansion

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hidden_dim": self.hidden_dim,
            "num_reasoning_blocks": self.num_reasoning_blocks,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "vocab_size": self.vocab_size,
            "max_seq_length": self.max_seq_length,
            "num_moe_experts": self.num_moe_experts,
            "moe_top_k": self.moe_top_k,
            "mamba_state_dim": self.mamba_state_dim,
            "wlam_num_heads": self.wlam_num_heads,
            "use_quantum_holographic_memory": self.use_quantum_holographic_memory,
            "use_unitary_residual": self.use_unitary_residual,
        }

    @property
    def estimated_params(self) -> int:
        """Estimate total parameter count.

        Returns rough estimate based on main components:
        - Embeddings: vocab_size * hidden_dim
        - Blocks: num_blocks * (attention + ffn + mamba + moe)
        - Output head: hidden_dim * vocab_size
        """
        embed_params = self.vocab_size * self.hidden_dim

        # Per block estimate
        attn_params = 4 * self.hidden_dim * self.hidden_dim  # QKV + out
        ffn_params = 2 * self.hidden_dim * self.ff_dim
        mamba_params = 4 * self.hidden_dim * self.mamba_state_dim
        moe_params = (
            self.num_moe_experts * 2 * self.hidden_dim * self.ff_dim * 0.3
        )  # ~30% due to sparsity

        block_params = attn_params + ffn_params + mamba_params + moe_params
        total_block_params = self.num_reasoning_blocks * block_params

        head_params = self.hidden_dim * self.vocab_size

        return int(embed_params + total_block_params + head_params)

    @property
    def estimated_memory_gb(self) -> float:
        """Estimate peak memory usage in GB.

        Includes model parameters (4 bytes/param) plus activations
        for batch_size=1 at max_seq_length.
        """
        param_bytes = self.estimated_params * 4
        # Activation memory scales with seq_length * hidden_dim * num_blocks
        activation_bytes = self.max_seq_length * self.hidden_dim * self.num_reasoning_blocks * 4 * 2
        return (param_bytes + activation_bytes) / (1024**3)


# Model size presets matching common configurations
MODEL_PRESETS: dict[str, HSMNModelConfig] = {
    "100m": HSMNModelConfig(
        hidden_dim=256,
        num_reasoning_blocks=4,
        num_heads=4,
        num_moe_experts=4,
        vocab_size=32000,
        max_seq_length=8192,
    ),
    "500m": HSMNModelConfig(
        hidden_dim=512,
        num_reasoning_blocks=6,
        num_heads=8,
        num_moe_experts=8,
        vocab_size=32000,
        max_seq_length=32768,
    ),
    "1b": HSMNModelConfig(
        hidden_dim=768,
        num_reasoning_blocks=8,
        num_heads=8,
        num_moe_experts=8,
        vocab_size=32000,
        max_seq_length=65536,
    ),
    "3b": HSMNModelConfig(
        hidden_dim=1024,
        num_reasoning_blocks=12,
        num_heads=16,
        num_moe_experts=12,
        vocab_size=32000,
        max_seq_length=131072,
    ),
    "7b": HSMNModelConfig(
        hidden_dim=1536,
        num_reasoning_blocks=16,
        num_heads=16,
        num_moe_experts=12,
        vocab_size=32000,
        max_seq_length=131072,
    ),
    "13b": HSMNModelConfig(
        hidden_dim=2048,
        num_reasoning_blocks=20,
        num_heads=32,
        num_moe_experts=12,
        vocab_size=32000,
        max_seq_length=262144,
    ),
    "20b": HSMNModelConfig(
        hidden_dim=2560,
        num_reasoning_blocks=24,
        num_heads=32,
        num_moe_experts=12,
        vocab_size=32000,
        max_seq_length=524288,
    ),
    # Benchmark-optimized presets
    "bench-quick": HSMNModelConfig(
        hidden_dim=256,
        num_reasoning_blocks=3,
        num_heads=4,
        num_moe_experts=4,
        vocab_size=32000,
        max_seq_length=4096,
    ),
    "bench-standard": HSMNModelConfig(
        hidden_dim=512,
        num_reasoning_blocks=6,
        num_heads=8,
        num_moe_experts=8,
        vocab_size=32000,
        max_seq_length=131072,
    ),
    "bench-enterprise": HSMNModelConfig(
        hidden_dim=768,
        num_reasoning_blocks=12,
        num_heads=8,
        num_moe_experts=8,
        vocab_size=32000,
        max_seq_length=1048576,  # 1M tokens
    ),
    # 100M parameter benchmark - FAST VERSION for quick iteration
    # Reduced to ~5M params for fast throughput testing
    "100m-benchmark": HSMNModelConfig(
        hidden_dim=128,  # Reduced from 512
        num_reasoning_blocks=8,  # Keep 8 blocks for meaningful benchmarks
        num_heads=4,  # Reduced from 8
        num_moe_experts=2,  # Reduced from 8
        vocab_size=32000,
        max_seq_length=1048576,  # 1M tokens
        # Quantum features use defaults (all enabled)
    ),
    # Full 100M param version for comprehensive benchmarking
    "100m-full": HSMNModelConfig(
        hidden_dim=512,
        num_reasoning_blocks=8,
        num_heads=8,
        num_moe_experts=8,
        vocab_size=32000,
        max_seq_length=1048576,
    ),
}


class HSMNModelBuilder:
    """Builder for HSMN benchmark models.

    Constructs models matching the HPO trial runner pattern with all
    quantum enhancement phases enabled.

    Example:
        >>> builder = HSMNModelBuilder()
        >>> model = builder.build("3b")
        >>> info = builder.get_model_info(model)
    """

    def __init__(self) -> None:
        """Initialize model builder."""
        self._cache: dict[str, tf.keras.Model] = {}

    def list_presets(self) -> list[str]:
        """List available model presets."""
        return list(MODEL_PRESETS.keys())

    def get_preset(self, name: str) -> HSMNModelConfig:
        """Get a model preset configuration.

        Args:
            name: Preset name (100m, 500m, 1b, 3b, 7b, 13b, 20b, or bench-*).

        Returns:
            HSMNModelConfig for the preset.

        Raises:
            ValueError: If preset not found.
        """
        name = name.lower()
        if name not in MODEL_PRESETS:
            available = ", ".join(MODEL_PRESETS.keys())
            raise ValueError(f"Unknown preset: {name}. Available: {available}")
        return MODEL_PRESETS[name]

    def build(
        self,
        config: HSMNModelConfig | str,
        use_cache: bool = True,
        force_rebuild: bool = False,
    ) -> tf.keras.Model:
        """Build HSMN model from configuration.

        Args:
            config: Model configuration or preset name.
            use_cache: Cache and reuse built models.
            force_rebuild: Force rebuild even if cached.

        Returns:
            Built and compiled tf.keras.Model.
        """
        if isinstance(config, str):
            config = self.get_preset(config)

        cache_key = f"hsmn_{config.hidden_dim}_{config.num_reasoning_blocks}"

        if use_cache and not force_rebuild and cache_key in self._cache:
            logger.debug(f"Returning cached model: {cache_key}")
            return self._cache[cache_key]

        logger.info(
            f"Building HSMN model: {config.hidden_dim}d, {config.num_reasoning_blocks} blocks"
        )
        logger.info(f"  Estimated parameters: {config.estimated_params:,}")
        logger.info(f"  Estimated memory: {config.estimated_memory_gb:.1f} GB")

        model = self._build_model(config)

        if use_cache:
            self._cache[cache_key] = model

        return model

    def _build_model(self, config: HSMNModelConfig) -> tf.keras.Model:
        """Internal model building logic matching HPO pattern."""
        from highnoon.models.reasoning.reasoning_module import ReasoningModule

        # Input: token IDs
        input_layer = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_ids")

        # Token embedding
        x = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.hidden_dim,
            name="token_embedding",
        )(input_layer)

        # Full HSMN reasoning module with hybrid block pattern
        # Note: Quantum parameters are read from global config by ReasoningModule
        # The block pattern is: mamba_timecrystal_wlam_moe_hybrid
        reasoning_module = ReasoningModule(
            num_layers=config.num_reasoning_blocks,
            embedding_dim=config.hidden_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            num_experts=config.num_moe_experts,
            wlam_num_heads=config.wlam_num_heads,
            wlam_kernel_size=config.wlam_kernel_size,
        )

        x = reasoning_module(x)

        # Output projection to vocabulary logits
        output_layer = tf.keras.layers.Dense(config.vocab_size, name="lm_head")(x)

        model = tf.keras.Model(
            inputs=input_layer,
            outputs=output_layer,
            name=f"HSMN_{config.hidden_dim}d_{config.num_reasoning_blocks}b",
        )

        # Build with sample input to initialize weights
        sample_input = tf.random.uniform(
            [1, 128], minval=0, maxval=config.vocab_size, dtype=tf.int32
        )
        _ = model(sample_input, training=False)

        logger.info(f"Built HSMN model: {model.count_params():,} parameters")

        return model

    def get_model_info(self, model: tf.keras.Model) -> dict[str, Any]:
        """Get model information.

        Args:
            model: HSMN model.

        Returns:
            Dictionary with model information.
        """
        trainable = sum(tf.reduce_prod(var.shape).numpy() for var in model.trainable_variables)
        non_trainable = sum(
            tf.reduce_prod(var.shape).numpy() for var in model.non_trainable_variables
        )

        return {
            "name": model.name,
            "total_parameters": trainable + non_trainable,
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable,
            "model_size_mb": (trainable + non_trainable) * 4 / (1024**2),
            "num_layers": len(model.layers),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
        }

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._cache.clear()
        logger.debug("Model cache cleared")


def build_benchmark_model(
    preset: str = "bench-standard",
    max_seq_length: int | None = None,
) -> tf.keras.Model:
    """Convenience function to build a benchmark model.

    Args:
        preset: Model preset name.
        max_seq_length: Override max sequence length.

    Returns:
        Built HSMN model.
    """
    builder = HSMNModelBuilder()
    config = builder.get_preset(preset)

    if max_seq_length is not None:
        config.max_seq_length = max_seq_length

    return builder.build(config)


def estimate_context_memory(
    hidden_dim: int,
    num_blocks: int,
    context_length: int,
    batch_size: int = 1,
) -> float:
    """Estimate memory required for a given context length.

    Args:
        hidden_dim: Model hidden dimension.
        num_blocks: Number of reasoning blocks.
        context_length: Context length in tokens.
        batch_size: Batch size.

    Returns:
        Estimated memory in GB.
    """
    # Activation memory per block: seq_len * hidden_dim * batch * 4 bytes * 2 (fwd+grad)
    activation_bytes = context_length * hidden_dim * batch_size * 4 * 2 * num_blocks

    # KV cache for attention
    kv_bytes = context_length * hidden_dim * batch_size * 4 * 2  # K and V

    return (activation_bytes + kv_bytes) / (1024**3)


def main() -> int:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="HSMN Model Builder")
    parser.add_argument(
        "--preset",
        "-p",
        type=str,
        default="bench-standard",
        help="Model preset to build",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available presets",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show preset information without building",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Actually build the model",
    )

    args = parser.parse_args()

    builder = HSMNModelBuilder()

    if args.list:
        print("Available model presets:")
        print()
        for name in builder.list_presets():
            config = builder.get_preset(name)
            print(
                f"  {name:15} {config.estimated_params/1e9:.2f}B params, "
                f"{config.hidden_dim}d, {config.num_reasoning_blocks} blocks, "
                f"max {config.max_seq_length:,} ctx"
            )
        return 0

    if args.info:
        config = builder.get_preset(args.preset)
        print(f"Preset: {args.preset}")
        print(f"  Hidden Dim: {config.hidden_dim}")
        print(f"  Blocks: {config.num_reasoning_blocks}")
        print(f"  Heads: {config.num_heads}")
        print(f"  FFN Dim: {config.ff_dim}")
        print(f"  MoE Experts: {config.num_moe_experts}")
        print(f"  Max Context: {config.max_seq_length:,}")
        print(f"  Est. Parameters: {config.estimated_params:,}")
        print(f"  Est. Memory: {config.estimated_memory_gb:.1f} GB")
        return 0

    if args.build:
        print(f"Building {args.preset}...")
        model = builder.build(args.preset)
        info = builder.get_model_info(model)
        print(f"Model built: {info['name']}")
        print(f"  Total Parameters: {info['total_parameters']:,}")
        print(f"  Model Size: {info['model_size_mb']:.1f} MB")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
