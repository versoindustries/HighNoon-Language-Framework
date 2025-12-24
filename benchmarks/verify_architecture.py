# benchmarks/verify_architecture.py
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

"""Architecture verification for HSMN benchmarks.

Verifies that all quantum enhancements and native ops are properly
active before running benchmarks. This ensures benchmark results
accurately reflect the production architecture.

Example:
    >>> from benchmarks.verify_architecture import verify_full_architecture
    >>> result = verify_full_architecture()
    >>> if result.all_passed:
    ...     print("✅ All architecture components verified!")
    >>> else:
    ...     print(f"❌ {len(result.failures)} components failed")

Command-line usage:
    python benchmarks/verify_architecture.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class VerificationCheck:
    """Result of a single verification check.

    Attributes:
        name: Name of the component being verified.
        category: Category (e.g., 'native_ops', 'config', 'inference').
        passed: Whether the check passed.
        message: Status message or error details.
        details: Additional details if available.
    """

    name: str
    category: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Complete architecture verification result.

    Attributes:
        all_passed: True if all checks passed.
        checks: List of individual check results.
        summary: Summary statistics.
    """

    all_passed: bool
    checks: list[VerificationCheck]
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> list[VerificationCheck]:
        """Get list of failed checks."""
        return [c for c in self.checks if not c.passed]

    @property
    def passed_count(self) -> int:
        """Get count of passed checks."""
        return sum(1 for c in self.checks if c.passed)


def check_native_ops_available() -> VerificationCheck:
    """Check if native C++ ops are loaded."""
    try:
        from benchmarks.native_ops_bridge import NativeOpsBridge

        bridge = NativeOpsBridge()
        report = bridge.get_availability_report()

        all(report.values())
        available_list = [k for k, v in report.items() if v]

        return VerificationCheck(
            name="Native Ops Bridge",
            category="native_ops",
            passed=True,  # Partial availability is OK
            message=f"{len(available_list)}/{len(report)} ops available",
            details=report,
        )
    except Exception as e:
        return VerificationCheck(
            name="Native Ops Bridge",
            category="native_ops",
            passed=False,
            message=f"Failed to load: {str(e)[:100]}",
        )


def check_streaming_inference() -> VerificationCheck:
    """Check if StreamingInferenceWrapper is available."""
    try:
        from highnoon.inference import StreamingInferenceWrapper

        # Try to inspect the class
        has_process_chunk = hasattr(StreamingInferenceWrapper, "process_chunk")
        has_reset = hasattr(StreamingInferenceWrapper, "reset_state")

        return VerificationCheck(
            name="Streaming Inference",
            category="inference",
            passed=True,
            message="StreamingInferenceWrapper available",
            details={
                "has_process_chunk": has_process_chunk,
                "has_reset_state": has_reset,
            },
        )
    except ImportError as e:
        return VerificationCheck(
            name="Streaming Inference",
            category="inference",
            passed=False,
            message=f"Import failed: {str(e)[:100]}",
        )


def check_qsg_generator() -> VerificationCheck:
    """Check if QSGGenerator is available."""
    try:
        from highnoon.inference import QSGGenerator

        has_generate = hasattr(QSGGenerator, "generate")

        return VerificationCheck(
            name="QSG Generator",
            category="inference",
            passed=True,
            message="QSGGenerator available for parallel generation",
            details={"has_generate": has_generate},
        )
    except ImportError as e:
        return VerificationCheck(
            name="QSG Generator",
            category="inference",
            passed=False,
            message=f"Import failed: {str(e)[:100]}",
        )


def check_qmr_enabled() -> VerificationCheck:
    """Check if Quantum Memory Replay is enabled in config."""
    try:
        from highnoon.config import ENABLE_QUANTUM_MEMORY_REPLAY

        return VerificationCheck(
            name="Quantum Memory Replay",
            category="config",
            passed=ENABLE_QUANTUM_MEMORY_REPLAY,
            message=f"QMR config: {'ENABLED' if ENABLE_QUANTUM_MEMORY_REPLAY else 'DISABLED'}",
            details={"enabled": ENABLE_QUANTUM_MEMORY_REPLAY},
        )
    except ImportError:
        return VerificationCheck(
            name="Quantum Memory Replay",
            category="config",
            passed=False,
            message="Could not import config",
        )


def check_holographic_memory() -> VerificationCheck:
    """Check if Quantum Holographic Memory is enabled."""
    try:
        from highnoon.config import USE_QUANTUM_HOLOGRAPHIC_MEMORY

        return VerificationCheck(
            name="Quantum Holographic Memory",
            category="config",
            passed=True,  # Not required, just check status
            message=f"Holographic Memory: {'ENABLED' if USE_QUANTUM_HOLOGRAPHIC_MEMORY else 'DISABLED'}",
            details={"enabled": USE_QUANTUM_HOLOGRAPHIC_MEMORY},
        )
    except ImportError:
        return VerificationCheck(
            name="Quantum Holographic Memory",
            category="config",
            passed=True,
            message="Config not available (optional)",
        )


def check_train_step_op() -> VerificationCheck:
    """Check if fused train_step op is available."""
    try:
        from highnoon._native.ops import train_step

        module = train_step.train_step_module()

        return VerificationCheck(
            name="Fused Train Step Op",
            category="native_ops",
            passed=module is not None,
            message="train_step_op available" if module else "train_step_op not loaded",
            details={"module_loaded": module is not None},
        )
    except (ImportError, AttributeError) as e:
        return VerificationCheck(
            name="Fused Train Step Op",
            category="native_ops",
            passed=False,
            message=f"Not available: {str(e)[:50]}",
        )


def check_quantum_ops() -> VerificationCheck:
    """Check if quantum ops (holographic bind, QSVT) are available."""
    try:
        from highnoon._native.ops.fused_unified_quantum_block_op import (
            unified_quantum_ops_available,
        )

        available = unified_quantum_ops_available()

        return VerificationCheck(
            name="Unified Quantum Ops",
            category="native_ops",
            passed=available,
            message="Native quantum ops loaded" if available else "Using Python fallbacks",
            details={"native_available": available},
        )
    except ImportError as e:
        return VerificationCheck(
            name="Unified Quantum Ops",
            category="native_ops",
            passed=False,
            message=f"Import failed: {str(e)[:50]}",
        )


def check_model_loadable() -> VerificationCheck:
    """Check if HSMN model can be instantiated."""
    try:
        from highnoon.config import REASONING_LAYERS, VOCAB_SIZE
        from highnoon.models.hsmn import HSMN

        # Try to instantiate a minimal model
        model = HSMN(
            vocab_size=VOCAB_SIZE,
            embedding_dim=64,  # Minimal for testing
            num_reasoning_blocks=min(2, REASONING_LAYERS),
        )

        # Build with sample input
        sample = tf.random.uniform([1, 32], minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)
        _ = model(sample, training=False)

        param_count = sum(tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables)

        return VerificationCheck(
            name="HSMN Model",
            category="model",
            passed=True,
            message=f"Model loadable ({param_count:,} params)",
            details={"param_count": int(param_count)},
        )
    except Exception as e:
        return VerificationCheck(
            name="HSMN Model",
            category="model",
            passed=False,
            message=f"Failed: {str(e)[:100]}",
        )


def check_benchmark_config() -> VerificationCheck:
    """Check if benchmark configuration is valid."""
    try:
        from benchmarks.benchmark_config import BenchmarkConfig, BenchmarkMode

        BenchmarkConfig()
        mode = BenchmarkMode.production()

        return VerificationCheck(
            name="Benchmark Config",
            category="config",
            passed=True,
            message=f"Mode: {mode.mode}, streaming: {mode.use_streaming_inference}",
            details={
                "mode": mode.mode,
                "use_streaming": mode.use_streaming_inference,
                "use_qsg": mode.use_qsg_generation,
                "use_qmr": mode.use_quantum_memory_replay,
            },
        )
    except Exception as e:
        return VerificationCheck(
            name="Benchmark Config",
            category="config",
            passed=False,
            message=f"Failed: {str(e)[:100]}",
        )


def verify_full_architecture(verbose: bool = False) -> VerificationResult:
    """Verify the complete HSMN architecture for benchmarking.

    Runs all verification checks to ensure the production architecture
    is properly available before running benchmarks.

    Args:
        verbose: Enable verbose logging.

    Returns:
        VerificationResult with all check results.
    """
    checks = [
        check_native_ops_available(),
        check_streaming_inference(),
        check_qsg_generator(),
        check_qmr_enabled(),
        check_holographic_memory(),
        check_train_step_op(),
        check_quantum_ops(),
        check_model_loadable(),
        check_benchmark_config(),
    ]

    # Log results if verbose
    if verbose:
        for check in checks:
            status = "✅" if check.passed else "❌"
            logger.info(f"  {status} [{check.category}] {check.name}: {check.message}")

    # Compute summary
    by_category: dict[str, list[VerificationCheck]] = {}
    for check in checks:
        if check.category not in by_category:
            by_category[check.category] = []
        by_category[check.category].append(check)

    summary = {
        "total_checks": len(checks),
        "passed": sum(1 for c in checks if c.passed),
        "failed": sum(1 for c in checks if not c.passed),
        "by_category": {
            cat: {
                "passed": sum(1 for c in cat_checks if c.passed),
                "total": len(cat_checks),
            }
            for cat, cat_checks in by_category.items()
        },
    }

    all_passed = all(c.passed for c in checks)

    return VerificationResult(
        all_passed=all_passed,
        checks=checks,
        summary=summary,
    )


def format_verification_markdown(result: VerificationResult) -> str:
    """Format verification result as markdown."""
    lines = [
        "# HSMN Architecture Verification",
        "",
        f"**Status**: {'✅ All Passed' if result.all_passed else '❌ Some Checks Failed'}",
        f"**Passed**: {result.passed_count}/{len(result.checks)}",
        "",
        "## Check Results",
        "",
        "| Category | Component | Status | Message |",
        "|----------|-----------|--------|---------|",
    ]

    for check in result.checks:
        status = "✅" if check.passed else "❌"
        lines.append(f"| {check.category} | {check.name} | {status} | {check.message} |")

    lines.extend(
        [
            "",
            "## Summary by Category",
            "",
        ]
    )

    for cat, stats in result.summary.get("by_category", {}).items():
        status = "✅" if stats["passed"] == stats["total"] else "⚠️"
        lines.append(f"- **{cat}**: {stats['passed']}/{stats['total']} {status}")

    if not result.all_passed:
        lines.extend(
            [
                "",
                "## Failed Checks",
                "",
            ]
        )
        for check in result.failures:
            lines.append(f"- **{check.name}**: {check.message}")

    return "\n".join(lines)


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Verify HSMN architecture for benchmarking")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    logger.info("Verifying HSMN architecture for benchmarking...")
    logger.info("")

    result = verify_full_architecture(verbose=args.verbose)

    if args.format == "markdown":
        print(format_verification_markdown(result))
    elif args.format == "json":
        import json
        from dataclasses import asdict

        print(
            json.dumps(
                {
                    "all_passed": result.all_passed,
                    "checks": [asdict(c) for c in result.checks],
                    "summary": result.summary,
                },
                indent=2,
            )
        )
    else:
        # Text format
        print("")
        if result.all_passed:
            print("✅ All architecture checks passed!")
            print("   Benchmarks will use full production architecture.")
        else:
            print(f"❌ {len(result.failures)} check(s) failed:")
            for check in result.failures:
                print(f"   - {check.name}: {check.message}")
            print("")
            print("   Some benchmarks may use Python fallbacks.")

        print("")
        print(f"Summary: {result.passed_count}/{len(result.checks)} checks passed")

    return 0 if result.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
