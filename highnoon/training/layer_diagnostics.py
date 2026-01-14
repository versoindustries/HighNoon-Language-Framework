"""Layer-by-layer diagnostics for training debugging.

Provides detailed per-layer activation and gradient statistics during training,
enabling identification of the exact layer where NaN/Inf issues originate.

Example:
    >>> config = LayerDiagnosticsConfig(capture_frequency=1, max_layers=50)
    >>> diagnostics = LayerDiagnostics(config)
    >>> snapshot = diagnostics.capture_activations(model, inputs, training=True)
    >>> print(snapshot.format_table())
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf


@dataclass
class LayerDiagnosticsConfig:
    """Configuration for layer diagnostics capture.

    Attributes:
        capture_frequency: Capture diagnostics every N steps. Default 1 (every step).
        max_layers: Maximum number of layers to trace. 0 = unlimited.
        include_sublayers: Whether to trace sublayers within blocks.
        activation_percentiles: Percentiles to compute for activations.
        gradient_percentiles: Percentiles to compute for gradients.
        layer_name_patterns: Patterns to prioritize for tracing.
        skip_patterns: Patterns to skip when tracing.
        compute_histograms: Whether to compute value histograms.
        histogram_bins: Number of bins for histograms.
    """

    capture_frequency: int = 1
    max_layers: int = 50
    include_sublayers: bool = True
    activation_percentiles: tuple[float, ...] = (0.0, 25.0, 50.0, 75.0, 100.0)
    gradient_percentiles: tuple[float, ...] = (0.0, 50.0, 100.0)
    layer_name_patterns: tuple[str, ...] = (
        "qhd_spatial",
        "hd_timecrystal",
        "continuous_thought",
        "latent_reasoning",
        "coconut",
        "moe",
        "quantum_lm_head",
        "lm_head",
        "embedding",
    )
    skip_patterns: tuple[str, ...] = ("dropout", "batch_norm_", "_bn")
    compute_histograms: bool = False
    histogram_bins: int = 50


@dataclass
class LayerActivationStats:
    """Statistics for a single layer's activations.

    Attributes:
        name: Layer name.
        layer_class: Layer class name.
        shape: Output tensor shape.
        dtype: Output tensor dtype.
        min_val: Minimum activation value.
        max_val: Maximum activation value.
        mean_val: Mean activation value.
        std_val: Standard deviation of activations.
        norm: L2 norm of activations.
        has_nan: Whether activations contain NaN.
        has_inf: Whether activations contain Inf.
        nan_count: Number of NaN values.
        inf_count: Number of Inf values.
        percentiles: Percentile values if computed.
        histogram: Histogram bins and counts if computed.
    """

    name: str
    layer_class: str
    shape: list[int | None]
    dtype: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    norm: float
    has_nan: bool
    has_inf: bool
    nan_count: int = 0
    inf_count: int = 0
    percentiles: dict[str, float] = field(default_factory=dict)
    histogram: dict[str, Any] | None = None

    @property
    def status(self) -> str:
        """Return status string for display."""
        if self.has_nan and self.has_inf:
            return "⚠ NaN+Inf"
        if self.has_nan:
            return "⚠ NaN"
        if self.has_inf:
            return "⚠ Inf"
        if math.isinf(self.max_val) or math.isinf(self.min_val):
            return "⚠ Overflow"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "layer_class": self.layer_class,
            "shape": self.shape,
            "dtype": self.dtype,
            "min": self.min_val,
            "max": self.max_val,
            "mean": self.mean_val,
            "std": self.std_val,
            "norm": self.norm,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "status": self.status,
            "percentiles": self.percentiles,
        }


@dataclass
class LayerGradientStats:
    """Statistics for a single layer's gradients.

    Attributes:
        name: Variable/layer name.
        layer_class: Associated layer class name if known.
        shape: Gradient tensor shape.
        dtype: Gradient tensor dtype.
        max_abs: Maximum absolute gradient value.
        mean_abs: Mean absolute gradient value.
        norm: L2 norm of gradients.
        has_nan: Whether gradients contain NaN.
        has_inf: Whether gradients contain Inf.
        is_zero: Whether gradient is effectively zero.
        percentiles: Percentile values if computed.
    """

    name: str
    layer_class: str
    shape: list[int | None]
    dtype: str
    max_abs: float
    mean_abs: float
    norm: float
    has_nan: bool
    has_inf: bool
    is_zero: bool
    percentiles: dict[str, float] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Return status string for display."""
        if self.has_nan and self.has_inf:
            return "⚠ NaN+Inf"
        if self.has_nan:
            return "⚠ NaN"
        if self.has_inf:
            return "⚠ Inf"
        if self.is_zero:
            return "zero"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "layer_class": self.layer_class,
            "shape": self.shape,
            "dtype": self.dtype,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "norm": self.norm,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "is_zero": self.is_zero,
            "status": self.status,
            "percentiles": self.percentiles,
        }


@dataclass
class LayerDiagnosticsSnapshot:
    """Complete diagnostics snapshot for a training step.

    Attributes:
        step: Training step number.
        activations: Per-layer activation statistics.
        gradients: Per-layer gradient statistics.
        first_nan_activation: Name of first layer with NaN activations.
        first_inf_activation: Name of first layer with Inf activations.
        first_nan_gradient: Name of first variable with NaN gradients.
        loss_value: Loss value at this step.
        grad_norm: Global gradient norm.
        is_valid: Whether the step was valid (no NaN/Inf in loss).
    """

    step: int
    activations: list[LayerActivationStats]
    gradients: list[LayerGradientStats]
    first_nan_activation: str | None = None
    first_inf_activation: str | None = None
    first_nan_gradient: str | None = None
    loss_value: float | None = None
    grad_norm: float | None = None
    is_valid: bool = True

    def format_table(self, max_name_len: int = 40) -> str:
        """Format diagnostics as a readable table.

        Args:
            max_name_len: Maximum length for layer names before truncation.

        Returns:
            Formatted table string for console output.
        """
        lines = []

        # Header
        header = (
            f"{'Layer':<{max_name_len}} │ {'Type':<20} │ "
            f"{'Activation':<12} │ {'Act Mean':>12} │ "
            f"{'Gradient':<10} │ {'Grad Norm':>12}"
        )
        sep_line = "─" * len(header)
        lines.append("┌" + sep_line + "┐")
        lines.append("│ " + header + " │")
        lines.append("├" + sep_line + "┤")

        # Build a map of variable names to gradient stats
        grad_by_prefix: dict[str, LayerGradientStats] = {}
        for g in self.gradients:
            # Extract layer name prefix from variable name
            parts = g.name.split("/")
            if len(parts) >= 2:
                prefix = "/".join(parts[:-1])
            else:
                prefix = g.name.rstrip(":0")
            if prefix not in grad_by_prefix or g.norm > grad_by_prefix[prefix].norm:
                grad_by_prefix[prefix] = g

        # Rows
        for act in self.activations:
            # Truncate long names
            name = act.name
            if len(name) > max_name_len:
                name = "..." + name[-(max_name_len - 3) :]

            # Find matching gradient
            grad_status = "-"
            grad_norm_str = "-"
            for prefix, grad in grad_by_prefix.items():
                if act.name in prefix or prefix in act.name:
                    grad_status = grad.status
                    if grad.has_nan:
                        grad_norm_str = "nan"
                    elif grad.has_inf:
                        grad_norm_str = "inf"
                    elif grad.is_zero:
                        grad_norm_str = "0.00e+00"
                    else:
                        grad_norm_str = f"{grad.norm:.2e}"
                    break

            # Format activation mean
            if act.has_nan:
                act_mean_str = "nan"
            elif act.has_inf:
                act_mean_str = "inf"
            else:
                act_mean_str = f"{act.mean_val:.4e}"

            row = (
                f"{name:<{max_name_len}} │ {act.layer_class:<20} │ "
                f"{act.status:<12} │ {act_mean_str:>12} │ "
                f"{grad_status:<10} │ {grad_norm_str:>12}"
            )
            lines.append("│ " + row + " │")

        lines.append("└" + sep_line + "┘")

        # Summary
        if self.first_nan_activation:
            lines.append(f"⚠ First NaN activation: {self.first_nan_activation}")
        if self.first_inf_activation:
            lines.append(f"⚠ First Inf activation: {self.first_inf_activation}")
        if self.first_nan_gradient:
            lines.append(f"⚠ First NaN gradient: {self.first_nan_gradient}")
        if self.loss_value is not None:
            lines.append(f"Loss: {self.loss_value:.6f}")
        if self.grad_norm is not None:
            lines.append(f"Global grad norm: {self.grad_norm:.6e}")

        return "\n".join(lines)

    def format_compact(self) -> str:
        """Format diagnostics as compact single line per layer.

        Returns:
            Compact multi-line string with one line per layer.
        """
        lines = []
        for act in self.activations:
            status = act.status
            if act.has_nan:
                lines.append(
                    f"  {act.name} ({act.layer_class}): {status} "
                    f"mean={act.mean_val:.4e} nan_count={act.nan_count}"
                )
            elif act.has_inf:
                lines.append(
                    f"  {act.name} ({act.layer_class}): {status} "
                    f"mean={act.mean_val:.4e} inf_count={act.inf_count}"
                )
            else:
                lines.append(
                    f"  {act.name} ({act.layer_class}): {status} "
                    f"mean={act.mean_val:.4e} min={act.min_val:.4e} max={act.max_val:.4e}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step": self.step,
            "activations": [a.to_dict() for a in self.activations],
            "gradients": [g.to_dict() for g in self.gradients],
            "first_nan_activation": self.first_nan_activation,
            "first_inf_activation": self.first_inf_activation,
            "first_nan_gradient": self.first_nan_gradient,
            "loss_value": self.loss_value,
            "grad_norm": self.grad_norm,
            "is_valid": self.is_valid,
        }


class LayerDiagnostics:
    """Capture and analyze per-layer diagnostics during training.

    This class provides detailed layer-by-layer statistics for debugging
    NaN/Inf issues in deep neural networks. It can:

    - Capture activation statistics from key layers during forward pass
    - Aggregate gradient statistics per layer from training gradients
    - Identify the first layer where NaN/Inf values appear
    - Generate formatted reports for console and JSON output

    Example:
        >>> diagnostics = LayerDiagnostics()
        >>> # After forward pass and gradient computation
        >>> snapshot = diagnostics.capture_step(
        ...     model=model,
        ...     inputs=inputs,
        ...     gradients=gradients,
        ...     variables=variables,
        ...     step=0,
        ...     loss=loss_value,
        ...     grad_norm=grad_norm,
        ... )
        >>> print(snapshot.format_table())
    """

    def __init__(self, config: LayerDiagnosticsConfig | None = None) -> None:
        """Initialize layer diagnostics.

        Args:
            config: Configuration for diagnostics capture.
        """
        self.config = config or LayerDiagnosticsConfig()
        self._probe_model: tf.keras.Model | None = None
        self._probe_outputs: list[tuple[str, str, tf.Tensor]] = []
        self._step_count = 0

    def _should_trace_layer(self, layer_name: str) -> bool:
        """Check if a layer should be traced based on config patterns.

        Args:
            layer_name: Name of the layer.

        Returns:
            True if layer should be traced.
        """
        name_lower = layer_name.lower()

        # Skip if matches skip patterns
        for pattern in self.config.skip_patterns:
            if pattern.lower() in name_lower:
                return False

        # Include if matches priority patterns
        for pattern in self.config.layer_name_patterns:
            if pattern.lower() in name_lower:
                return True

        # Include by default
        return True

    def _get_traceable_layers(
        self, model: tf.keras.Model
    ) -> list[tuple[tf.keras.layers.Layer, str]]:
        """Get list of layers to trace from model.

        Args:
            model: Keras model to inspect.

        Returns:
            List of (layer, output_name) tuples.
        """
        layers_to_trace: list[tuple[tf.keras.layers.Layer, str]] = []
        seen_names: set[str] = set()

        def add_layer(layer: tf.keras.layers.Layer, prefix: str = "") -> None:
            if self.config.max_layers > 0 and len(layers_to_trace) >= self.config.max_layers:
                return

            name = f"{prefix}/{layer.name}" if prefix else layer.name
            if name in seen_names:
                return

            if self._should_trace_layer(name):
                seen_names.add(name)
                layers_to_trace.append((layer, name))

            if self.config.include_sublayers:
                if hasattr(layer, "layers"):
                    for sublayer in layer.layers:
                        add_layer(sublayer, name)
                if hasattr(layer, "reasoning_blocks"):
                    for block in layer.reasoning_blocks:
                        add_layer(block, name)

        for layer in model.layers:
            add_layer(layer)
            if self.config.max_layers > 0 and len(layers_to_trace) >= self.config.max_layers:
                break

        return layers_to_trace

    def _compute_tensor_stats(
        self,
        tensor: tf.Tensor,
        name: str,
        layer_class: str,
        is_activation: bool = True,
    ) -> LayerActivationStats | LayerGradientStats:
        """Compute statistics for a tensor.

        Args:
            tensor: Tensor to analyze.
            name: Name for the tensor.
            layer_class: Class name of the associated layer.
            is_activation: Whether this is an activation (vs gradient).

        Returns:
            Statistics object for the tensor.
        """
        shape = tensor.shape.as_list() if hasattr(tensor.shape, "as_list") else list(tensor.shape)
        dtype = tensor.dtype.name if hasattr(tensor.dtype, "name") else str(tensor.dtype)

        # Convert to float for stats
        if tensor.dtype == tf.complex64 or tensor.dtype == tf.complex128:
            tensor_float = tf.abs(tensor)
        else:
            tensor_float = tf.cast(tensor, tf.float32)

        # Check for NaN/Inf
        nan_mask = tf.math.is_nan(tensor_float)
        inf_mask = tf.math.is_inf(tensor_float)
        has_nan = bool(tf.reduce_any(nan_mask).numpy())
        has_inf = bool(tf.reduce_any(inf_mask).numpy())
        nan_count = int(tf.reduce_sum(tf.cast(nan_mask, tf.int32)).numpy())
        inf_count = int(tf.reduce_sum(tf.cast(inf_mask, tf.int32)).numpy())

        # Replace NaN/Inf for stat computation
        clean_tensor = tf.where(
            nan_mask | inf_mask,
            tf.zeros_like(tensor_float),
            tensor_float,
        )

        # Compute stats
        min_val = float(tf.reduce_min(clean_tensor).numpy())
        max_val = float(tf.reduce_max(clean_tensor).numpy())
        mean_val = float(tf.reduce_mean(clean_tensor).numpy())

        # Variance and std
        variance = tf.reduce_mean(tf.square(clean_tensor - mean_val))
        std_val = float(tf.sqrt(variance).numpy())

        # Norm
        norm = float(tf.linalg.global_norm([clean_tensor]).numpy())

        # Percentiles (optional)
        percentiles: dict[str, float] = {}
        if is_activation and self.config.activation_percentiles:
            flat = tf.reshape(clean_tensor, [-1])
            for p in self.config.activation_percentiles:
                try:
                    pct_val = float(
                        tf.numpy_function(
                            lambda x, pct: float(__import__("numpy").percentile(x, pct)),
                            [flat, p],
                            tf.float32,
                        ).numpy()
                    )
                    percentiles[f"p{int(p)}"] = pct_val
                except Exception:
                    pass

        if is_activation:
            return LayerActivationStats(
                name=name,
                layer_class=layer_class,
                shape=shape,
                dtype=dtype,
                min_val=min_val,
                max_val=max_val,
                mean_val=mean_val,
                std_val=std_val,
                norm=norm,
                has_nan=has_nan,
                has_inf=has_inf,
                nan_count=nan_count,
                inf_count=inf_count,
                percentiles=percentiles,
            )
        else:
            abs_tensor = tf.abs(clean_tensor)
            max_abs = float(tf.reduce_max(abs_tensor).numpy())
            mean_abs = float(tf.reduce_mean(abs_tensor).numpy())
            is_zero = max_abs < 1e-12

            return LayerGradientStats(
                name=name,
                layer_class=layer_class,
                shape=shape,
                dtype=dtype,
                max_abs=max_abs,
                mean_abs=mean_abs,
                norm=norm,
                has_nan=has_nan,
                has_inf=has_inf,
                is_zero=is_zero,
                percentiles=percentiles,
            )

    def capture_activations(
        self,
        model: tf.keras.Model,
        inputs: tf.Tensor,
        training: bool = True,
    ) -> list[LayerActivationStats]:
        """Capture activation statistics from model forward pass.

        Args:
            model: Keras model to trace.
            inputs: Input tensor for forward pass.
            training: Whether to run in training mode.

        Returns:
            List of activation statistics per layer.
        """
        layers_to_trace = self._get_traceable_layers(model)
        activations: list[LayerActivationStats] = []

        # Try to build probe model for efficient multi-output capture
        outputs: list[tf.Tensor] = []
        output_info: list[tuple[str, str]] = []

        for layer, name in layers_to_trace:
            try:
                output = layer.output
                if output is not None:
                    outputs.append(output)
                    output_info.append((name, layer.__class__.__name__))
            except Exception:
                # Layer may not have output yet
                continue

        if outputs:
            try:
                probe_model = tf.keras.Model(inputs=model.inputs, outputs=outputs)
                layer_outputs = probe_model(inputs, training=training)

                if not isinstance(layer_outputs, (list, tuple)):
                    layer_outputs = [layer_outputs]

                for (name, layer_class), output in zip(output_info, layer_outputs):
                    if not hasattr(output, "dtype") or output.dtype.is_bool:
                        continue
                    stats = self._compute_tensor_stats(
                        output, name, layer_class, is_activation=True
                    )
                    if isinstance(stats, LayerActivationStats):
                        activations.append(stats)
            except Exception:
                # Fall back to per-layer calls
                pass

        # Fallback: run individual forward passes per layer
        if not activations:
            for layer, name in layers_to_trace:
                try:
                    # This is slower but more robust
                    output = layer(inputs, training=training)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    if not hasattr(output, "dtype") or output.dtype.is_bool:
                        continue
                    stats = self._compute_tensor_stats(
                        output, name, layer.__class__.__name__, is_activation=True
                    )
                    if isinstance(stats, LayerActivationStats):
                        activations.append(stats)
                except Exception:
                    continue

        return activations

    def capture_gradients(
        self,
        gradients: Sequence[tf.Tensor | tf.IndexedSlices | None],
        variables: Sequence[tf.Variable],
    ) -> list[LayerGradientStats]:
        """Capture gradient statistics from training step.

        Args:
            gradients: List of gradient tensors.
            variables: List of corresponding variables.

        Returns:
            List of gradient statistics per variable.
        """
        gradient_stats: list[LayerGradientStats] = []

        for grad, var in zip(gradients, variables):
            if grad is None:
                continue

            # Handle IndexedSlices
            values = grad.values if isinstance(grad, tf.IndexedSlices) else grad
            if values is None:
                continue

            # Extract layer class from variable name
            name = var.name
            parts = name.split("/")
            layer_class = parts[-2] if len(parts) >= 2 else "unknown"

            stats = self._compute_tensor_stats(values, name, layer_class, is_activation=False)
            if isinstance(stats, LayerGradientStats):
                gradient_stats.append(stats)

        return gradient_stats

    def capture_step(
        self,
        model: tf.keras.Model,
        inputs: tf.Tensor,
        gradients: Sequence[tf.Tensor | tf.IndexedSlices | None] | None = None,
        variables: Sequence[tf.Variable] | None = None,
        step: int = 0,
        loss: float | None = None,
        grad_norm: float | None = None,
        training: bool = True,
    ) -> LayerDiagnosticsSnapshot:
        """Capture complete diagnostics snapshot for a training step.

        Args:
            model: Keras model.
            inputs: Input tensor.
            gradients: Optional gradient tensors from training step.
            variables: Optional variable tensors from training step.
            step: Current training step.
            loss: Loss value at this step.
            grad_norm: Global gradient norm.
            training: Whether running in training mode.

        Returns:
            Complete diagnostics snapshot.
        """
        self._step_count += 1

        # Check if we should capture based on frequency
        if self._step_count % self.config.capture_frequency != 0:
            return LayerDiagnosticsSnapshot(
                step=step,
                activations=[],
                gradients=[],
                loss_value=loss,
                grad_norm=grad_norm,
            )

        # Capture activations
        activations = self.capture_activations(model, inputs, training=training)

        # Capture gradients if provided
        gradient_stats: list[LayerGradientStats] = []
        if gradients is not None and variables is not None:
            gradient_stats = self.capture_gradients(gradients, variables)

        # Find first NaN/Inf layers
        first_nan_activation = None
        first_inf_activation = None
        first_nan_gradient = None

        for act in activations:
            if act.has_nan and first_nan_activation is None:
                first_nan_activation = act.name
            if act.has_inf and first_inf_activation is None:
                first_inf_activation = act.name

        for grad in gradient_stats:
            if grad.has_nan and first_nan_gradient is None:
                first_nan_gradient = grad.name

        is_valid = loss is None or (math.isfinite(loss))

        return LayerDiagnosticsSnapshot(
            step=step,
            activations=activations,
            gradients=gradient_stats,
            first_nan_activation=first_nan_activation,
            first_inf_activation=first_inf_activation,
            first_nan_gradient=first_nan_gradient,
            loss_value=loss,
            grad_norm=grad_norm,
            is_valid=is_valid,
        )


__all__ = [
    "LayerDiagnosticsConfig",
    "LayerActivationStats",
    "LayerGradientStats",
    "LayerDiagnosticsSnapshot",
    "LayerDiagnostics",
]
