"""
OpenTelemetry distributed tracing for HSMN Architecture.

Provides instrumentation for multi-op execution paths with critical path analysis.
Integrates with Prometheus metrics for unified observability.

Phase 9.2 Deliverable: Distributed tracing per technical_roadmap.md section 9.2.

Usage:
    from highnoon.telemetry.opentelemetry_tracer import get_tracer, trace_op_execution

    tracer = get_tracer()

    with tracer.start_as_current_span("fused_reasoning_stack_forward"):
        # ... op execution code ...
        pass

Reference:
    - OpenTelemetry Python SDK: https://opentelemetry.io/docs/instrumentation/python/
    - Jaeger backend: https://www.jaegertracing.io/
"""

from __future__ import annotations

import functools
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

import tensorflow as tf

# OpenTelemetry is optional for environments without tracing infrastructure
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None  # type: ignore
    TracerProvider = None  # type: ignore
    OTLPSpanExporter = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    ConsoleSpanExporter = None  # type: ignore
    Resource = None  # type: ignore


F = TypeVar("F", bound=Callable[..., Any])

# Global tracer instance (singleton)
_tracer_instance: Any | None = None
_tracer_lock = threading.Lock()


def initialize_tracing(
    service_name: str = "hsmn_training",
    otlp_endpoint: str | None = None,
    console_debug: bool = False,
) -> None:
    """
    Initialize OpenTelemetry tracing with OTLP exporter.

    Args:
        service_name: Name of the service for trace identification
        otlp_endpoint: OTLP gRPC endpoint (e.g., "localhost:4317" for Jaeger)
                       If None, uses OTEL_EXPORTER_OTLP_ENDPOINT env var
        console_debug: If True, also export spans to console for debugging

    Environment Variables:
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (default: http://localhost:4317)
        HSMN_ENABLE_TRACING: Set to "1" to enable tracing (default: "0")

    Example:
        # In training script main():
        if os.getenv("HSMN_ENABLE_TRACING", "0") == "1":
            initialize_tracing(
                service_name="hsmn_training",
                otlp_endpoint="localhost:4317",  # Jaeger OTLP gRPC receiver
            )
    """
    if not OPENTELEMETRY_AVAILABLE:
        return

    global _tracer_instance

    with _tracer_lock:
        if _tracer_instance is not None:
            return  # Already initialized

        # Create resource with service metadata
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": "9.2.0",  # Phase 9.2
                "deployment.environment": os.getenv("HSMN_ENV", "development"),
            }
        )

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add OTLP exporter if endpoint configured
        endpoint = otlp_endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        if endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            except Exception as e:
                print(f"[WARNING] Failed to initialize OTLP exporter: {e}")

        # Add console exporter for debugging
        if console_debug:
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))

        # Set global tracer provider
        trace.set_tracer_provider(provider)
        _tracer_instance = trace.get_tracer(__name__)


def get_tracer() -> Any:
    """
    Get the global OpenTelemetry tracer instance.

    Returns:
        Tracer instance if OpenTelemetry is available and initialized, else NoopTracer

    Example:
        tracer = get_tracer()
        with tracer.start_as_current_span("my_operation"):
            # ... traced code ...
            pass
    """
    global _tracer_instance

    if not OPENTELEMETRY_AVAILABLE:
        return NoopTracer()

    with _tracer_lock:
        if _tracer_instance is None:
            # Auto-initialize with defaults if not explicitly initialized
            initialize_tracing()

        return _tracer_instance or NoopTracer()


def trace_op_execution(op_name: str, include_shapes: bool = True) -> Callable[[F], F]:
    """
    Decorator to trace TensorFlow custom op execution with automatic span creation.

    Args:
        op_name: Name of the op (e.g., "fused_hnn_step")
        include_shapes: Include input/output tensor shapes in span attributes

    Returns:
        Decorated function with tracing instrumentation

    Example:
        @trace_op_execution("fused_hnn_step")
        def fused_hnn_step_forward(q, p, ...):
            # ... op implementation ...
            return q_next, p_next

    Span Attributes:
        - op.name: Op name
        - op.device: Execution device (cpu/gpu)
        - op.input_shapes: Input tensor shapes (if include_shapes=True)
        - op.output_shapes: Output tensor shapes (if include_shapes=True)
        - op.latency_ms: Execution time in milliseconds
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()

            with tracer.start_as_current_span(f"{op_name}_forward") as span:
                start_time = time.perf_counter()

                # Record op metadata
                if span and span.is_recording():
                    span.set_attribute("op.name", op_name)
                    span.set_attribute("op.device", _get_device_from_args(args))

                    if include_shapes:
                        input_shapes = _extract_tensor_shapes(args)
                        if input_shapes:
                            span.set_attribute("op.input_shapes", str(input_shapes))

                # Execute op
                try:
                    result = func(*args, **kwargs)

                    # Record output shapes
                    if span and span.is_recording() and include_shapes:
                        output_shapes = _extract_tensor_shapes([result])
                        if output_shapes:
                            span.set_attribute("op.output_shapes", str(output_shapes))

                    return result

                finally:
                    # Record latency
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    if span and span.is_recording():
                        span.set_attribute("op.latency_ms", latency_ms)

        return wrapper  # type: ignore

    return decorator


@contextmanager
def trace_training_step(step: int, batch_size: int) -> Iterator[None]:
    """
    Context manager for tracing a full training step with metadata.

    Args:
        step: Global training step number
        batch_size: Batch size for this step

    Yields:
        None

    Example:
        for step, (inputs, labels) in enumerate(dataset):
            with trace_training_step(step, batch_size=inputs.shape[0]):
                loss = train_step(inputs, labels)
    """
    tracer = get_tracer()

    with tracer.start_as_current_span("training_step") as span:
        if span and span.is_recording():
            span.set_attribute("training.step", step)
            span.set_attribute("training.batch_size", batch_size)

        yield


@contextmanager
def trace_reasoning_stack(
    sequence_length: int,
    num_blocks: int,
    moe_active: bool = False,
) -> Iterator[None]:
    """
    Context manager for tracing FusedReasoningStack execution.

    Args:
        sequence_length: Input sequence length
        num_blocks: Number of reasoning blocks
        moe_active: Whether MoE routing is active

    Yields:
        None

    Example:
        with trace_reasoning_stack(seq_len=128, num_blocks=4, moe_active=True):
            output = reasoning_stack(inputs)
    """
    tracer = get_tracer()

    with tracer.start_as_current_span("fused_reasoning_stack") as span:
        if span and span.is_recording():
            span.set_attribute("reasoning.sequence_length", sequence_length)
            span.set_attribute("reasoning.num_blocks", num_blocks)
            span.set_attribute("reasoning.moe_active", moe_active)

        yield


def _get_device_from_args(args: tuple) -> str:
    """Extract device placement from tensor arguments."""
    for arg in args:
        if isinstance(arg, tf.Tensor):
            return arg.device.split("/")[-1]  # Extract 'cpu:0' or 'gpu:0'
    return "cpu"


def _extract_tensor_shapes(tensors: Any) -> list:
    """Extract shapes from tensor or list of tensors."""
    shapes = []

    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            if isinstance(t, tf.Tensor):
                shapes.append(t.shape.as_list())
    elif isinstance(tensors, tf.Tensor):
        shapes.append(tensors.shape.as_list())

    return shapes


class NoopTracer:
    """
    No-op tracer for environments without OpenTelemetry.

    Provides same API as real tracer but performs no instrumentation.
    """

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Iterator[NoopSpan]:
        """Create a no-op span context."""
        yield NoopSpan()


class NoopSpan:
    """No-op span for environments without OpenTelemetry."""

    def is_recording(self) -> bool:
        """Always returns False for no-op span."""
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op attribute setting."""
        pass


# Convenience function for critical path analysis
def get_critical_path_ops() -> list[str]:
    """
    Returns list of ops on the critical path for FusedReasoningStack.

    Use this to prioritize optimization efforts based on trace analysis.

    Returns:
        List of op names typically on the critical execution path

    Example:
        critical_ops = get_critical_path_ops()
        for op in critical_ops:
            # ... profile or optimize this op ...
            pass
    """
    return [
        "selective_scan",  # Mamba SSM (often bottleneck for long sequences)
        "fused_hnn_step",  # Hamiltonian integration (iterative, accumulates latency)
        "fused_superposition_moe",  # MoE routing (top-k + dispatch overhead)
        "time_crystal_step",  # Limit cycle enforcement (frequency-domain FFT)
        "vqc_expectation",  # Variational quantum circuits (exponential state vector)
        "fused_quantum_gnn_step",  # Graph message passing (edge-centric, large graphs)
    ]


# Export public API
__all__ = [
    "initialize_tracing",
    "get_tracer",
    "trace_op_execution",
    "trace_training_step",
    "trace_reasoning_stack",
    "get_critical_path_ops",
    "OPENTELEMETRY_AVAILABLE",
]
