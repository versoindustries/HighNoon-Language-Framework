#!/usr/bin/env python3
# benchmarks/optimizer_benchmark.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Benchmark script to compare optimizer performance:
# - SophiaG (2nd-order Hessian)
# - QIAO (Quantum-Inspired Alternating)
# - Adam/AdamW (baseline)
# - SympFlow (symplectic Hamiltonian)

"""
Optimizer Benchmark Suite

Compares convergence speed, final loss, memory usage, and wall-clock time
for all available optimizers on a representative training task.

Usage:
    python benchmarks/optimizer_benchmark.py --epochs 10 --batch-size 8
"""

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from highnoon.training.optimizers import QIAO, SophiaG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single optimizer benchmark run."""

    optimizer_name: str
    final_loss: float
    best_loss: float
    convergence_step: int  # Step where 90% of improvement achieved
    total_steps: int
    wall_time_seconds: float
    time_per_step_ms: float
    peak_memory_mb: float
    loss_history: list[float] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    epochs: int = 10
    batch_size: int = 8
    seq_len: int = 128
    vocab_size: int = 1000
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    seed: int = 42


def create_simple_model(cfg: BenchmarkConfig) -> tf.keras.Model:
    """Create a simple transformer-like model for benchmarking."""
    inputs = tf.keras.Input(shape=(cfg.seq_len,), dtype=tf.int32, name="input_ids")

    # Embedding
    x = tf.keras.layers.Embedding(cfg.vocab_size, cfg.embedding_dim)(inputs)

    # Simple transformer blocks
    for i in range(cfg.num_layers):
        # Self-attention approximation (linear for speed)
        attn = tf.keras.layers.Dense(cfg.embedding_dim, name=f"attn_{i}")(x)
        x = tf.keras.layers.LayerNormalization()(x + attn)

        # FFN
        ffn = tf.keras.layers.Dense(cfg.hidden_dim, activation="gelu", name=f"ffn1_{i}")(x)
        ffn = tf.keras.layers.Dense(cfg.embedding_dim, name=f"ffn2_{i}")(ffn)
        x = tf.keras.layers.LayerNormalization()(x + ffn)

    # Output head
    outputs = tf.keras.layers.Dense(cfg.vocab_size, name="lm_head")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="benchmark_model")
    return model


def create_synthetic_data(cfg: BenchmarkConfig, num_batches: int = 100):
    """Create synthetic training data with learnable patterns.

    Uses realistic text-like sequences with:
    - Bigram patterns (next token depends on previous)
    - Common "word" token sequences
    - Positional patterns
    """
    np.random.seed(cfg.seed)

    # Create a simple bigram transition matrix for more learnable patterns
    # Token i is more likely to be followed by (i+1) mod vocab, (i+2) mod vocab, etc.
    def generate_sequence(length):
        seq = [np.random.randint(0, cfg.vocab_size)]
        for _ in range(length - 1):
            prev = seq[-1]
            # Next token has 60% chance to be "related" to previous
            if np.random.random() < 0.6:
                # Related tokens: prev+1, prev+2, or prev+vocab//2
                offset = np.random.choice([1, 2, cfg.vocab_size // 2])
                next_tok = (prev + offset) % cfg.vocab_size
            else:
                next_tok = np.random.randint(0, cfg.vocab_size)
            seq.append(next_tok)
        return seq

    def data_generator():
        for _ in range(num_batches):
            batch = [generate_sequence(cfg.seq_len) for _ in range(cfg.batch_size)]
            x = np.array(batch, dtype=np.int32)
            # Target is next token prediction
            y = np.roll(x, -1, axis=1)
            yield x, y

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(cfg.batch_size, cfg.seq_len), dtype=tf.int32),
            tf.TensorSpec(shape=(cfg.batch_size, cfg.seq_len), dtype=tf.int32),
        ),
    )
    return dataset


def get_optimizer(name: str, model: tf.keras.Model, cfg: BenchmarkConfig):
    """Create optimizer by name."""
    lr = cfg.learning_rate

    if name == "sophiag":
        return SophiaG(model=model, learning_rate=lr)
    elif name == "qiao":
        return QIAO(model=model, learning_rate=lr)
    elif name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.01)
    elif name == "sympflow":
        # SympFlow wrapper using native op
        return create_sympflow_optimizer(lr, cfg)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def create_sympflow_optimizer(learning_rate: float, cfg: BenchmarkConfig):
    """Create SympFlow optimizer wrapper with automatic warmup.

    SympFlow uses Hamiltonian dynamics (symplectic integration) for optimization.
    Key insight: Momentum-based methods need warmup to build up gradient statistics.

    This implementation automatically:
    1. Uses Adam-like behavior for first N steps (warmup)
    2. Transitions to full Hamiltonian dynamics after warmup
    3. Anneals mass and friction for stable convergence
    """

    class SympFlowOptimizer(tf.keras.optimizers.Optimizer):
        """SympFlow optimizer with automatic warmup.

        No user tuning required - warmup is handled internally.
        """

        def __init__(
            self,
            learning_rate=0.001,
            warmup_steps=100,
            initial_mass=1.0,
            target_mass=0.1,
            initial_friction=0.9,
            target_friction=0.01,
            name="SympFlow",
            **kwargs,
        ):
            super().__init__(learning_rate=learning_rate, name=name, **kwargs)
            self._warmup_steps = warmup_steps
            self._initial_mass = initial_mass
            self._target_mass = target_mass
            self._initial_friction = initial_friction
            self._target_friction = target_friction
            # Use TF Variable so it persists properly
            self._iterations_var = None

        def build(self, var_list):
            super().build(var_list)
            self._momentum = []
            self._var_refs = []
            self._built_var_list = list(var_list)
            self._v = []
            self._num_vars = len(list(var_list))
            for var in var_list:
                self._var_refs.append(id(var))
                self._momentum.append(self.add_variable_from_reference(var, "momentum"))
                self._v.append(self.add_variable_from_reference(var, "velocity"))

        def _get_warmup_factor(self):
            """Returns warmup progress (0.0 = start, 1.0 = done)."""
            # Use optimizer's built-in iterations counter
            step = int(self.iterations.numpy()) // max(self._num_vars, 1)
            return min(1.0, float(step) / float(self._warmup_steps))

        def _get_current_mass(self):
            t = self._get_warmup_factor()
            return self._initial_mass * (1 - t) + self._target_mass * t

        def _get_current_friction(self):
            t = self._get_warmup_factor()
            return self._initial_friction * (1 - t) + self._target_friction * t

        def update_step(self, gradient, variable, learning_rate):
            if isinstance(gradient, tf.IndexedSlices):
                gradient = tf.convert_to_tensor(gradient)

            var_id = id(variable)
            try:
                idx = self._var_refs.index(var_id)
            except ValueError:
                return

            momentum = self._momentum[idx]
            v = self._v[idx]

            # Scale learning rate higher for this optimizer
            h = tf.cast(learning_rate, variable.dtype) * 10.0
            warmup_factor = self._get_warmup_factor()
            m = self._get_current_mass()
            gamma = self._get_current_friction()

            # Update second moment
            beta2 = 0.999
            v.assign(beta2 * v + (1 - beta2) * tf.square(gradient))

            if warmup_factor < 1.0:
                # WARMUP: Adam-like update
                beta1 = 0.9
                momentum.assign(beta1 * momentum + (1 - beta1) * gradient)
                denom = tf.sqrt(v) + 1e-8
                update = h * momentum / denom
                variable.assign_sub(update)
            else:
                # HAMILTONIAN: Störmer-Verlet
                p_half = momentum - (h / 2.0) * gradient
                p_half = p_half * (1.0 - gamma * h)
                variable.assign_add(h * p_half / m)
                momentum.assign(p_half - (h / 2.0) * gradient)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "warmup_steps": self._warmup_steps,
                    "initial_mass": self._initial_mass,
                    "target_mass": self._target_mass,
                    "initial_friction": self._initial_friction,
                    "target_friction": self._target_friction,
                }
            )
            return config

    warmup_steps = 50

    return SympFlowOptimizer(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        initial_mass=1.0,
        target_mass=0.1,
        initial_friction=0.9,
        target_friction=0.01,
    )


def benchmark_optimizer(
    optimizer_name: str,
    cfg: BenchmarkConfig,
) -> BenchmarkResult:
    """Run benchmark for a single optimizer."""
    logger.info(f"Benchmarking {optimizer_name}...")

    # Clear memory
    tf.keras.backend.clear_session()
    gc.collect()

    # Set seed for reproducibility
    tf.random.set_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create model and optimizer
    model = create_simple_model(cfg)
    optimizer = get_optimizer(optimizer_name, model, cfg)

    # Create data
    num_batches = cfg.epochs * 100  # 100 batches per epoch
    create_synthetic_data(cfg, num_batches)

    # Loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Track metrics
    loss_history = []
    step = 0
    start_time = time.time()

    # Training loop
    for epoch in range(cfg.epochs):
        epoch_losses = []

        for x_batch, y_batch in create_synthetic_data(cfg, 100):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = loss_fn(y_batch, logits)

            gradients = tape.gradient(loss, model.trainable_variables)

            # Convert IndexedSlices to dense tensors for QIAO compatibility
            if optimizer_name == "qiao":
                gradients = [
                    tf.convert_to_tensor(g) if isinstance(g, tf.IndexedSlices) else g
                    for g in gradients
                ]

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            loss_val = float(loss.numpy())
            epoch_losses.append(loss_val)
            loss_history.append(loss_val)
            step += 1

            # Update Hessian for SophiaG/QIAO
            if optimizer_name in ["sophiag", "qiao"] and step % 10 == 0:
                try:

                    def loss_fn_for_hessian(x=x_batch, y=y_batch):
                        return loss_fn(y, model(x, training=True))

                    optimizer.update_hessian(loss_fn_for_hessian, (x_batch, y_batch))
                except Exception:
                    pass  # Skip if update_hessian not available

        avg_loss = np.mean(epoch_losses)
        logger.info(f"  Epoch {epoch+1}/{cfg.epochs}: loss={avg_loss:.4f}")

    end_time = time.time()
    wall_time = end_time - start_time

    # Calculate metrics
    final_loss = loss_history[-1] if loss_history else float("inf")
    best_loss = min(loss_history) if loss_history else float("inf")

    # Find convergence step (90% of improvement)
    if len(loss_history) > 1:
        improvement = loss_history[0] - best_loss
        target = loss_history[0] - 0.9 * improvement
        convergence_step = next(
            (i for i, loss_val in enumerate(loss_history) if loss_val <= target), len(loss_history)
        )
    else:
        convergence_step = 0

    # Get memory info (approximate)
    try:
        peak_memory = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e6
    except Exception:
        peak_memory = 0.0  # CPU only

    result = BenchmarkResult(
        optimizer_name=optimizer_name,
        final_loss=final_loss,
        best_loss=best_loss,
        convergence_step=convergence_step,
        total_steps=step,
        wall_time_seconds=wall_time,
        time_per_step_ms=(wall_time / max(step, 1)) * 1000,
        peak_memory_mb=peak_memory,
        loss_history=loss_history[:100],  # Keep first 100 for plotting
    )

    logger.info(
        f"  -> Final: {final_loss:.4f}, Best: {best_loss:.4f}, "
        f"Time: {wall_time:.1f}s ({result.time_per_step_ms:.2f}ms/step)"
    )

    return result


def run_benchmark(cfg: BenchmarkConfig, optimizers: list[str]) -> list[BenchmarkResult]:
    """Run benchmark for all optimizers."""
    results = []

    for opt_name in optimizers:
        try:
            result = benchmark_optimizer(opt_name, cfg)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to benchmark {opt_name}: {e}")

    return results


def print_results_table(results: list[BenchmarkResult]):
    """Print results as formatted table."""
    print("\n" + "=" * 80)
    print("OPTIMIZER BENCHMARK RESULTS")
    print("=" * 80)
    print(
        f"{'Optimizer':<12} {'Final Loss':>12} {'Best Loss':>12} {'Conv. Step':>12} "
        f"{'Time (s)':>10} {'ms/step':>10}"
    )
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.best_loss):
        print(
            f"{r.optimizer_name:<12} {r.final_loss:>12.4f} {r.best_loss:>12.4f} "
            f"{r.convergence_step:>12} {r.wall_time_seconds:>10.1f} "
            f"{r.time_per_step_ms:>10.2f}"
        )

    print("=" * 80)

    # Determine winner
    best = min(results, key=lambda x: x.best_loss)
    fastest = min(results, key=lambda x: x.time_per_step_ms)

    print(f"\n🏆 Best Loss: {best.optimizer_name} ({best.best_loss:.4f})")
    print(f"⚡ Fastest: {fastest.optimizer_name} ({fastest.time_per_step_ms:.2f}ms/step)")


def save_results(results: list[BenchmarkResult], output_path: str):
    """Save results to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Optimizer Benchmark Suite")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["sophiag", "qiao", "adam", "adamw", "sympflow"],
        help="Optimizers to benchmark",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json", help="Output JSON file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    cfg = BenchmarkConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    logger.info("=" * 60)
    logger.info("OPTIMIZER BENCHMARK")
    logger.info("=" * 60)
    logger.info(
        f"Config: epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
        f"seq_len={cfg.seq_len}, lr={cfg.learning_rate}"
    )
    logger.info(f"Optimizers: {args.optimizers}")
    logger.info("=" * 60)

    results = run_benchmark(cfg, args.optimizers)

    print_results_table(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
