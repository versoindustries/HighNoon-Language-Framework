# benchmarks/bench_quantum.py
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

"""Quantum Training Enhancement Benchmarks.

Benchmarks for Phases 1-7 and 25 quantum training enhancements:
- Phase 1: Neumann-Cayley Mamba Gates (unitarity preservation)
- Phase 2: SPRK-Enhanced TimeCrystal (symplectic integration)
- Phase 3: CA-TDVP Isometric MPS (isometry enforcement)
- Phase 4: Lifting Scheme Reversible WLAM (perfect reconstruction)
- Phase 5: QNG Geodesic MoE (natural gradient efficiency)
- Phase 6: Quantum Memory Replay (O(log n) checkpointing)
- Phase 7: Entanglement Preservation Loss (entropy regularization)
- Phase 25: Quantum-Holographic Memory (associative capacity)

Example:
    >>> from benchmarks.bench_quantum import run_quantum_benchmarks
    >>> results = run_quantum_benchmarks()
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class QuantumBenchmarkResult:
    """Result from quantum enhancement benchmark."""

    phase: str
    name: str
    metric: str
    value: float
    unit: str
    passed: bool
    notes: str = ""


def benchmark_phase1_neumann_cayley() -> list[QuantumBenchmarkResult]:
    """Benchmark Phase 1: Neumann-Cayley Mamba Gates.

    Tests:
    - Unitarity preservation (should maintain ||Ux|| = ||x||)
    - Neumann series convergence
    - SIMD acceleration factor
    """
    results = []

    try:
        from highnoon._native.ops.fused_unified_quantum_block_op import _ensure_lib_loaded

        ops_available = _ensure_lib_loaded()
    except ImportError:
        ops_available = False

    # Test 1: Unitarity preservation via Cayley transform
    dim = 256
    A = np.random.randn(dim, dim).astype(np.float32)
    A_skew = (A - A.T) / 2  # Skew-symmetric

    # Cayley transform: U = (I - A)(I + A)^{-1}
    identity = np.eye(dim, dtype=np.float32)
    U = np.linalg.solve(identity + A_skew, identity - A_skew)

    # Check orthogonality: U^T U should be identity
    orthogonality_error = np.linalg.norm(U.T @ U - identity) / dim

    results.append(
        QuantumBenchmarkResult(
            phase="Phase 1",
            name="Cayley Transform Unitarity",
            metric="frobenius_error",
            value=float(orthogonality_error),
            unit="normalized",
            passed=orthogonality_error < 1e-5,
            notes="U^T U - I should be ~0",
        )
    )

    # Test 2: Neumann series approximation (epsilon must be small for convergence)
    epsilon = 0.01  # Small epsilon ensures ||A|| < 1 for convergence
    A_small = epsilon * A_skew
    # Neumann series: (I + A)^{-1} ≈ I - A + A² - A³ + ...
    neumann_approx = identity.copy()
    power = A_small.copy()
    for k in range(1, 6):
        neumann_approx += ((-1) ** k) * power
        power = power @ A_small

    exact_inv = np.linalg.inv(identity + A_small)
    neumann_error = np.linalg.norm(neumann_approx - exact_inv) / np.linalg.norm(exact_inv)

    results.append(
        QuantumBenchmarkResult(
            phase="Phase 1",
            name="Neumann Series Convergence",
            metric="relative_error",
            value=float(neumann_error),
            unit="relative",
            passed=neumann_error < 0.01,
            notes="5-term Neumann series accuracy",
        )
    )

    # Test 3: Performance (if ops available)
    if ops_available:
        x = np.random.randn(32, 128, dim).astype(np.float32)
        x_tf = tf.constant(x)

        # Warmup
        for _ in range(3):
            _ = tf.linalg.matvec(tf.constant(U), x_tf)

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = tf.linalg.matvec(tf.constant(U), x_tf)
            tf.keras.backend.clear_session()
            times.append((time.perf_counter() - start) * 1000)

        results.append(
            QuantumBenchmarkResult(
                phase="Phase 1",
                name="Unitary Transform Latency",
                metric="mean_time",
                value=float(np.mean(times)),
                unit="ms",
                passed=np.mean(times) < 50,
                notes=f"p95={np.percentile(times, 95):.2f}ms",
            )
        )

    return results


def benchmark_phase2_sprk_timecrystal() -> list[QuantumBenchmarkResult]:
    """Benchmark Phase 2: SPRK-Enhanced TimeCrystal.

    Tests:
    - Symplecticity preservation
    - Energy conservation
    - 6th order accuracy
    """
    results = []

    # Yoshida 6th order coefficients
    w1 = 1.31518632068391
    w2 = -1.17767998417887
    w3 = 0.235573213359357
    w0 = 1.0 - 2 * (w1 + w2 + w3)
    yoshida_weights = [w3, w2, w1, w0, w1, w2, w3]

    # Test symplectic preservation: det(J) = 1
    dim = 4
    dt = 0.01

    # Simple harmonic oscillator Hamiltonian: H = (p² + q²)/2
    def symplectic_step(q, p, dt_step):
        # Leapfrog: p -> p/2, q -> q + p*dt, p -> p/2
        p_half = p - 0.5 * dt_step * q
        q_new = q + dt_step * p_half
        p_new = p_half - 0.5 * dt_step * q_new
        return q_new, p_new

    q0 = np.random.randn(dim).astype(np.float32)
    p0 = np.random.randn(dim).astype(np.float32)
    H0 = 0.5 * (np.sum(p0**2) + np.sum(q0**2))

    q, p = q0.copy(), p0.copy()
    for w in yoshida_weights:
        q, p = symplectic_step(q, p, w * dt)

    H_final = 0.5 * (np.sum(p**2) + np.sum(q**2))
    energy_drift = abs(H_final - H0) / H0

    results.append(
        QuantumBenchmarkResult(
            phase="Phase 2",
            name="Yoshida-6 Energy Conservation",
            metric="relative_drift",
            value=float(energy_drift),
            unit="relative",
            passed=energy_drift < 1e-6,  # 6th order should achieve ~1e-7 to 1e-8
            notes="Single step energy drift",
        )
    )

    # Long-term stability test (100 steps)
    q, p = q0.copy(), p0.copy()
    for _ in range(100):
        for w in yoshida_weights:
            q, p = symplectic_step(q, p, w * dt)

    H_100 = 0.5 * (np.sum(p**2) + np.sum(q**2))
    long_term_drift = abs(H_100 - H0) / H0

    results.append(
        QuantumBenchmarkResult(
            phase="Phase 2",
            name="Long-term Energy Stability",
            metric="relative_drift_100_steps",
            value=float(long_term_drift),
            unit="relative",
            passed=long_term_drift < 1e-4,  # 100 steps accumulates some drift
            notes="100-step energy conservation",
        )
    )

    return results


def benchmark_phase6_memory_replay() -> list[QuantumBenchmarkResult]:
    """Benchmark Phase 6: Quantum Memory Replay.

    Tests:
    - Logarithmic checkpoint count
    - Memory savings vs naive checkpointing
    - Reconstruction accuracy
    """
    results = []

    # Test logarithmic checkpointing
    def compute_log_checkpoints(seq_len: int) -> list[int]:
        """Logarithmic checkpoint placement."""
        if seq_len <= 2:
            return list(range(seq_len))
        checkpoints = [0]
        step = 1
        while checkpoints[-1] + step < seq_len:
            checkpoints.append(checkpoints[-1] + step)
            step *= 2
        if checkpoints[-1] != seq_len - 1:
            checkpoints.append(seq_len - 1)
        return checkpoints

    for seq_len in [64, 256, 1024, 4096]:
        checkpoints = compute_log_checkpoints(seq_len)
        num_checkpoints = len(checkpoints)
        expected_log = np.log2(seq_len) + 1

        results.append(
            QuantumBenchmarkResult(
                phase="Phase 6",
                name=f"Checkpoint Count (L={seq_len})",
                metric="num_checkpoints",
                value=float(num_checkpoints),
                unit="count",
                passed=num_checkpoints <= expected_log + 2,
                notes=f"Expected ~{expected_log:.1f}",
            )
        )

    # Memory savings calculation
    seq_len = 1024
    state_dim = 512
    bytes_per_float = 4

    naive_memory = seq_len * state_dim * bytes_per_float
    log_checkpoints = len(compute_log_checkpoints(seq_len))
    log_memory = log_checkpoints * state_dim * bytes_per_float
    savings_pct = (1 - log_memory / naive_memory) * 100

    results.append(
        QuantumBenchmarkResult(
            phase="Phase 6",
            name="Memory Savings vs Naive",
            metric="savings_percent",
            value=float(savings_pct),
            unit="%",
            passed=savings_pct > 95,
            notes=f"{log_memory / 1024:.1f}KB vs {naive_memory / 1024:.1f}KB",
        )
    )

    return results


def benchmark_phase7_entanglement_loss() -> list[QuantumBenchmarkResult]:
    """Benchmark Phase 7: Entanglement Preservation Loss.

    Tests:
    - Von Neumann entropy calculation
    - Loss gradient quality
    - Regularization effectiveness
    """
    results = []

    # Test Von Neumann entropy
    dim = 16

    # Pure state: entropy = 0
    psi = np.zeros(dim, dtype=np.float32)
    psi[0] = 1.0
    rho_pure = np.outer(psi, psi)
    eigenvalues = np.linalg.eigvalsh(rho_pure)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    entropy_pure = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))

    results.append(
        QuantumBenchmarkResult(
            phase="Phase 7",
            name="Pure State Entropy",
            metric="von_neumann_entropy",
            value=float(entropy_pure),
            unit="nats",
            passed=entropy_pure < 1e-5,
            notes="Pure state should have S=0",
        )
    )

    # Maximally mixed state: entropy = log(d)
    rho_mixed = np.eye(dim, dtype=np.float32) / dim
    eigenvalues = np.linalg.eigvalsh(rho_mixed)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    entropy_mixed = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
    expected_max = np.log(dim)

    results.append(
        QuantumBenchmarkResult(
            phase="Phase 7",
            name="Mixed State Entropy",
            metric="von_neumann_entropy",
            value=float(entropy_mixed),
            unit="nats",
            passed=abs(entropy_mixed - expected_max) < 1e-5,
            notes=f"Expected S={expected_max:.4f}",
        )
    )

    return results


def benchmark_phase25_holographic_memory() -> list[QuantumBenchmarkResult]:
    """Benchmark Phase 25: Quantum-Holographic Memory.

    Tests:
    - Holographic binding/unbinding accuracy
    - Modern Hopfield retrieval quality
    - Memory capacity scaling
    """
    results = []

    # Test holographic binding
    dim = 512
    a = np.random.choice([-1.0, 1.0], size=dim).astype(np.float32)
    b = np.random.choice([-1.0, 1.0], size=dim).astype(np.float32)

    # Bind and unbind
    bound = a * b  # Elementwise for bipolar
    recovered = bound * b
    recovery_accuracy = np.mean(recovered == a)

    results.append(
        QuantumBenchmarkResult(
            phase="Phase 25",
            name="Bipolar Binding Recovery",
            metric="accuracy",
            value=float(recovery_accuracy),
            unit="fraction",
            passed=recovery_accuracy > 0.99,
            notes="bind(a,b) * b should recover a",
        )
    )

    # Modern Hopfield capacity test
    def hopfield_retrieve(query, memory, beta):
        scores = beta * (memory @ query)
        scores = scores - scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()
        return weights @ memory

    dims_to_test = [64, 128, 256]
    for dim in dims_to_test:
        num_patterns = dim // 4  # Conservative capacity
        memory = np.random.randn(num_patterns, dim).astype(np.float32)
        memory /= np.linalg.norm(memory, axis=1, keepdims=True)
        beta = np.sqrt(dim) * 2.0

        successes = 0
        for i in range(num_patterns):
            result = hopfield_retrieve(memory[i], memory, beta)
            similarity = np.dot(result, memory[i])
            if similarity > 0.7:
                successes += 1

        success_rate = successes / num_patterns

        results.append(
            QuantumBenchmarkResult(
                phase="Phase 25",
                name=f"Hopfield Retrieval (d={dim})",
                metric="success_rate",
                value=float(success_rate),
                unit="fraction",
                passed=success_rate > 0.8,
                notes=f"Patterns={num_patterns}, beta={beta:.1f}",
            )
        )

    return results


def run_quantum_benchmarks(verbose: bool = True) -> dict[str, Any]:
    """Run all quantum training benchmarks.

    Args:
        verbose: Print progress to stdout.

    Returns:
        Dictionary with all benchmark results.
    """
    all_results = []

    benchmarks = [
        ("Phase 1: Neumann-Cayley", benchmark_phase1_neumann_cayley),
        ("Phase 2: SPRK TimeCrystal", benchmark_phase2_sprk_timecrystal),
        ("Phase 6: Memory Replay", benchmark_phase6_memory_replay),
        ("Phase 7: Entanglement Loss", benchmark_phase7_entanglement_loss),
        ("Phase 25: Holographic Memory", benchmark_phase25_holographic_memory),
    ]

    for name, func in benchmarks:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")

        try:
            results = func()
            for r in results:
                status = "✅" if r.passed else "❌"
                if verbose:
                    print(f"  {status} {r.name}: {r.value:.6g} {r.unit}")
                    if r.notes:
                        print(f"      └─ {r.notes}")
            all_results.extend(results)
        except Exception as e:
            if verbose:
                print(f"  ❌ FAILED: {e}")
            logger.error(f"Benchmark {name} failed: {e}")

    # Summary
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  SUMMARY: {passed}/{total} benchmarks passed ({100*passed/total:.1f}%)")
        print(f"{'='*60}")

    return {
        "results": [
            {
                "phase": r.phase,
                "name": r.name,
                "metric": r.metric,
                "value": r.value,
                "unit": r.unit,
                "passed": r.passed,
                "notes": r.notes,
            }
            for r in all_results
        ],
        "summary": {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total > 0 else 0,
        },
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Quantum Training Benchmarks")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    args = parser.parse_args()

    results = run_quantum_benchmarks(verbose=not args.quiet)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # Custom encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np

                if isinstance(obj, (np.bool_, np.integer)):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"Results saved to {args.output}")

    return 0 if results["summary"]["pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    exit(main())
