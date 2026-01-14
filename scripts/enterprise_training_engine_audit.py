#!/usr/bin/env python3
"""Enterprise training engine audit for C++ ops and gradient connectivity.

This script exercises the full TrainingEngine stack using block_factory and
all feature flags from config.py. It validates native ops availability and
audits gradient connectivity across key blocks.

Enhanced with detailed layer-by-layer diagnostics showing:
- Per-layer activation statistics (min/max/mean/norm, NaN/Inf detection)
- Per-layer gradient statistics (norm, max_abs, zero/non-finite flags)
- First NaN/Inf layer identification for debugging
- Formatted table output and JSON report generation

Usage:
    python scripts/enterprise_training_engine_audit.py \
        --steps-per-epoch 1 --epochs 1 --hd-dim 512 \
        --layer-diagnostics --verbose-gradients \
        --report artifacts/audit_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Phase 4.2 (GRADIENT_CONNECTIVITY_ROADMAP): Suppress TensorFlow complex-to-float casting warnings.
# Must be before TensorFlow import to catch all warnings.
warnings.filterwarnings("ignore", message=".*casting.*complex.*float.*")
warnings.filterwarnings("ignore", message=".*incompatible dtype float.*imaginary part.*")

# Also filter TensorFlow's internal logging system which bypasses Python warnings
import logging

import tensorflow as tf

tf_logger = logging.getLogger("tensorflow")


class ComplexCastingFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "casting" in msg and "complex" in msg and ("float" in msg or "imaginary" in msg):
            return False
        return True


tf_logger.addFilter(ComplexCastingFilter())


# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def _extract_config_flags() -> dict[str, Any]:
    from highnoon import config as hn_config

    flags: dict[str, Any] = {}
    for name, value in vars(hn_config).items():
        if not name.isupper() or name.startswith("_"):
            continue
        if callable(value) or isinstance(value, type):
            continue
        flags[name] = value
        flags[name.lower()] = value
        if name.startswith("ENABLE_"):
            flags[f"use_{name[len('ENABLE_'):].lower()}"] = value
        if name.endswith("_ENABLED"):
            base = name[: -len("_ENABLED")]
            flags[f"use_{base.lower()}"] = value

    if "use_q_ssm_gating" not in flags and hasattr(hn_config, "USE_UNIFIED_QSSM_GATING"):
        flags["use_q_ssm_gating"] = hn_config.USE_UNIFIED_QSSM_GATING

    return flags


ARCH_FLAG_NAMES: tuple[str, ...] = (
    "QUANTUM_ENABLE_SAMPLING",
    "NEURAL_KALMAN_PROPAGATE_COV",
    "TENSOR_STREAM_DEBUG",
    "USE_NUMA_ALLOCATION",
    "ENABLE_KERNEL_TIMING",
    "USE_WORK_STEALING_MOE",
    "QHD_WALK_LEARN_TIME",
    "USE_CHUNKED_FORWARD",
    "STATE_BUS_GUMBEL_HARD",
    "USE_SPECULATIVE",
    "USE_QAT",
    "COLLAPSE_HARD_SAMPLES",
    "MPS_COMPUTE_ENTROPY",
    "MPS_UNIFORM_MODE",
)


def _force_enable_arch_flags(config_module: Any, config: dict[str, Any] | None = None) -> None:
    for name in ARCH_FLAG_NAMES:
        if hasattr(config_module, name):
            setattr(config_module, name, True)
        if config is not None:
            config[name] = True
            config[name.lower()] = True
            if name.startswith("ENABLE_"):
                config[f"use_{name[len('ENABLE_'):].lower()}"] = True


def _load_curriculum_presets() -> dict[str, dict[str, Any]]:
    presets_path = PROJECT_ROOT / "artifacts" / "curriculum_presets.json"
    if not presets_path.exists():
        return {}
    try:
        data = json.loads(presets_path.read_text())
    except Exception:
        return {}
    presets = data.get("presets", {})
    return presets if isinstance(presets, dict) else {}


def _resolve_curriculum_dataset(
    curriculum_id: str | None,
) -> tuple[str | None, list[str]]:
    if not curriculum_id:
        return None, []
    normalized = curriculum_id.replace("_", "-")
    presets = _load_curriculum_presets()
    preset = presets.get(normalized)
    if not isinstance(preset, dict):
        return None, []

    datasets: list[str] = []

    # Prioritize stages data - this contains ALL datasets (180+ for verso-baseline)
    # The hf_datasets field is just a summary list and incomplete
    if isinstance(preset.get("stages"), list):
        for stage in preset["stages"]:
            if not isinstance(stage, dict):
                continue
            for ds_entry in stage.get("datasets", []):
                ds_id = ds_entry.get("dataset_id") if isinstance(ds_entry, dict) else ds_entry
                if isinstance(ds_id, str) and ds_id not in datasets:
                    datasets.append(ds_id)

    # Fallback to hf_datasets if stages not available
    if not datasets and isinstance(preset.get("hf_datasets"), list):
        datasets.extend([ds for ds in preset["hf_datasets"] if isinstance(ds, str)])

    primary = datasets[0] if datasets else None
    return primary, datasets


def _memory_snapshot(process=None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    if PSUTIL_AVAILABLE:
        proc = process or psutil.Process()
        mem_info = proc.memory_info()
        snapshot["rss_mb"] = mem_info.rss / (1024 * 1024)
        snapshot["vms_mb"] = mem_info.vms / (1024 * 1024)
        snapshot["shared_mb"] = getattr(mem_info, "shared", 0) / (1024 * 1024)
        sys_mem = psutil.virtual_memory()
        snapshot["system_total_mb"] = sys_mem.total / (1024 * 1024)
        snapshot["system_available_mb"] = sys_mem.available / (1024 * 1024)
        snapshot["system_percent"] = sys_mem.percent
    else:
        try:
            import resource

            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            snapshot["rss_mb"] = rss_kb / 1024.0
        except Exception:
            snapshot["rss_mb"] = None
    return snapshot


def _count_tokens(tensor) -> int | None:
    if tensor is None:
        return None
    try:
        shape = tensor.shape
        if shape.rank is not None and shape.rank >= 2 and shape[0] and shape[1]:
            return int(shape[0]) * int(shape[1])
        size = tf.size(tensor)
        if hasattr(size, "numpy"):
            return int(size.numpy())
    except Exception:
        return None
    return None


def _profile_flops(
    model: tf.keras.Model,
    inputs: tf.Tensor,
    training: bool = True,
) -> dict[str, Any]:
    """Profile FLOPS for a single forward pass using TensorFlow profiler.

    Uses tf.profiler to measure floating point operations during model inference.
    Falls back to architecture-based estimation if profiler is unavailable.

    Args:
        model: The Keras model to profile.
        inputs: Input tensor for the forward pass.
        training: Whether to run in training mode.

    Returns:
        Dictionary containing FLOPS statistics:
        - total_flops: Total floating point operations
        - gflops: FLOPS in billions (GFLOPS)
        - profiler_available: Whether profiling succeeded
        - estimation_method: 'profiler', 'keras_flops', or 'architecture'
        - error: Error message if profiling failed
    """
    result: dict[str, Any] = {
        "total_flops": None,
        "gflops": None,
        "profiler_available": False,
        "estimation_method": None,
        "error": None,
    }

    batch_size = inputs.shape[0] if inputs.shape[0] is not None else 1
    seq_len = inputs.shape[1] if len(inputs.shape) > 1 and inputs.shape[1] is not None else 1

    # Method 1: Try TensorFlow Profiler
    try:

        @tf.function
        def forward_pass(x):
            return model(x, training=training)

        concrete_func = forward_pass.get_concrete_function(
            tf.TensorSpec(shape=inputs.shape, dtype=inputs.dtype)
        )

        try:
            from tensorflow.python.profiler.model_analyzer import profile
            from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

            opts = ProfileOptionBuilder.float_operation()
            opts["output"] = "none"

            graph = concrete_func.graph
            flops_stats = profile(graph, options=opts)

            if flops_stats is not None and flops_stats.total_float_ops > 0:
                total_flops = flops_stats.total_float_ops
                result["total_flops"] = total_flops
                result["gflops"] = total_flops / 1e9
                result["profiler_available"] = True
                result["estimation_method"] = "profiler"
                return result
        except (ImportError, Exception) as e:
            result["error"] = f"TF Profiler failed: {e}"
    except Exception as e:
        result["error"] = f"Concrete function creation failed: {e}"

    # Method 2: Try keras_flops package (if installed)
    try:
        from keras_flops import get_flops

        total_flops = get_flops(model, batch_size=batch_size)
        if total_flops > 0:
            result["total_flops"] = total_flops
            result["gflops"] = total_flops / 1e9
            result["profiler_available"] = True
            result["estimation_method"] = "keras_flops"
            return result
    except ImportError:
        pass
    except Exception as e:
        result["error"] = f"keras_flops failed: {e}"

    # Method 3: Architecture-based estimation
    try:
        total_flops = 0

        for layer in model.layers:
            layer_flops = 0
            layer_name = layer.__class__.__name__

            # Dense layers: 2 * M * N per sample (matmul + bias)
            if hasattr(layer, "units") and hasattr(layer, "kernel"):
                kernel_shape = layer.kernel.shape
                if len(kernel_shape) >= 2:
                    in_features = int(kernel_shape[0])
                    out_features = int(kernel_shape[1])
                    # For sequence models: batch * seq_len * (2 * in * out)
                    layer_flops = batch_size * seq_len * 2 * in_features * out_features

            # Conv1D layers: 2 * kernel_size * in_channels * out_channels * seq_len
            elif hasattr(layer, "filters") and hasattr(layer, "kernel_size"):
                kernel_size = (
                    layer.kernel_size[0]
                    if isinstance(layer.kernel_size, tuple)
                    else layer.kernel_size
                )
                in_channels = getattr(layer, "input_spec", None)
                if in_channels and hasattr(in_channels, "shape") and in_channels.shape:
                    in_dim = in_channels.shape[-1] or 1
                else:
                    in_dim = 1
                layer_flops = batch_size * seq_len * 2 * kernel_size * in_dim * layer.filters

            # MultiHeadAttention: 4 * seq_len^2 * hidden_dim (Q/K/V projections + attention)
            elif "Attention" in layer_name or "attention" in layer.name.lower():
                # Estimate based on typical attention cost
                hidden_dim = 512  # Default estimate
                if hasattr(layer, "key_dim"):
                    hidden_dim = layer.key_dim * (getattr(layer, "num_heads", 8))
                layer_flops = batch_size * 4 * seq_len * seq_len * hidden_dim

            # LayerNormalization: 5 * elements (mean, var, normalize, scale, shift)
            elif "LayerNorm" in layer_name or "layer_norm" in layer.name.lower():
                # Count trainable params to estimate dimension
                if layer.trainable_variables:
                    dim = sum(v.shape.num_elements() for v in layer.trainable_variables) // 2
                    layer_flops = batch_size * seq_len * 5 * dim

            # Embedding: lookup only, minimal FLOPS
            elif "Embedding" in layer_name:
                if hasattr(layer, "output_dim"):
                    layer_flops = batch_size * seq_len * layer.output_dim  # Just the copy

            total_flops += layer_flops

        if total_flops > 0:
            result["total_flops"] = total_flops
            result["gflops"] = total_flops / 1e9
            result["profiler_available"] = True
            result["estimation_method"] = "architecture"
    except Exception as e:
        if result["error"]:
            result["error"] += f"; Architecture estimation failed: {e}"
        else:
            result["error"] = f"Architecture estimation failed: {e}"

    return result


def _estimate_model_flops(
    model: tf.keras.Model,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> dict[str, Any]:
    """Estimate theoretical FLOPS for model based on architecture.

    Provides a static estimate of FLOPS based on layer types and dimensions.
    This is faster than profiling but less accurate.

    Args:
        model: The Keras model to analyze.
        batch_size: Batch size for estimation.
        seq_len: Sequence length for estimation.
        vocab_size: Vocabulary size.

    Returns:
        Dictionary with FLOPS estimates per component.
    """
    total_flops = 0
    component_flops: dict[str, int] = {}

    for layer in model.layers:
        layer_flops = 0
        layer_name = layer.__class__.__name__

        # Estimate based on layer type
        if hasattr(layer, "units"):
            # Dense layers: 2 * input_dim * output_dim per sample
            input_dim = getattr(layer, "input_dim", 0)
            if input_dim == 0 and hasattr(layer, "input_spec") and layer.input_spec:
                spec = (
                    layer.input_spec[0] if isinstance(layer.input_spec, list) else layer.input_spec
                )
                if hasattr(spec, "shape") and spec.shape:
                    input_dim = spec.shape[-1] or 0
            layer_flops = 2 * batch_size * seq_len * input_dim * layer.units

        elif hasattr(layer, "filters"):
            # Conv layers approximation
            kernel_size = getattr(layer, "kernel_size", (3,))
            k = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
            in_channels = getattr(layer, "input_dim", 1)
            layer_flops = 2 * batch_size * seq_len * k * in_channels * layer.filters

        if layer_flops > 0:
            component_flops[layer.name] = layer_flops
            total_flops += layer_flops

    return {
        "estimated_total_flops": total_flops,
        "estimated_gflops": total_flops / 1e9,
        "component_flops": component_flops,
        "estimation_method": "architecture_based",
    }


def _trace_zero_gradient_variables(
    model: tf.keras.Model,
    gradients: list[tf.Tensor | None],
    variables: list[tf.Variable],
    verbose: int = 1,
) -> dict[str, Any]:
    """Trace zero-gradient variables to their parent layers and identify root causes.

    This enterprise-grade diagnostic function helps identify gradient blocking patterns
    including:
    - Variables not connected to the computation graph
    - tf.stop_gradient barriers
    - Conditional paths not taken
    - C++ ops with missing backward implementations

    Args:
        model: The Keras model to analyze.
        gradients: List of gradients from tape.gradient().
        variables: List of trainable variables.
        verbose: Verbosity level (0=silent, 1=summary, 2=detailed).

    Returns:
        Diagnostic report with categorized zero-gradient variables.
    """
    report = {
        "total_vars": len(variables),
        "zero_grad_count": 0,
        "none_grad_count": 0,
        "by_layer": {},
        "by_variable_type": {},
        "suspected_causes": [],
    }

    # Build layer name mapping for variable tracing
    layer_by_var = {}
    for layer in model.layers:
        for var in layer.trainable_variables:
            layer_by_var[var.ref()] = layer.name

    zero_vars = []
    none_vars = []

    for var, grad in zip(variables, gradients):
        if grad is None:
            none_vars.append(var)
            report["none_grad_count"] += 1
        elif isinstance(grad, tf.IndexedSlices):
            # IndexedSlices are sparse gradients - check if actually zero
            if tf.reduce_sum(tf.abs(grad.values)) < 1e-12:
                zero_vars.append((var, "indexed_slices_zero"))
                report["zero_grad_count"] += 1
        elif tf.reduce_max(tf.abs(grad)) < 1e-12:
            zero_vars.append((var, "zero_tensor"))
            report["zero_grad_count"] += 1

    # Group by layer
    for var in none_vars:
        layer_name = layer_by_var.get(var.ref(), "unknown")
        var_base = var.name.split("/")[-1].rstrip(":0")
        if layer_name not in report["by_layer"]:
            report["by_layer"][layer_name] = {"none": [], "zero": []}
        report["by_layer"][layer_name]["none"].append(var_base)
        # Track by variable type
        if var_base not in report["by_variable_type"]:
            report["by_variable_type"][var_base] = {"count": 0, "layers": []}
        report["by_variable_type"][var_base]["count"] += 1
        report["by_variable_type"][var_base]["layers"].append(layer_name)

    for var, cause in zero_vars:
        layer_name = layer_by_var.get(var.ref(), "unknown")
        var_base = var.name.split("/")[-1].rstrip(":0")
        if layer_name not in report["by_layer"]:
            report["by_layer"][layer_name] = {"none": [], "zero": []}
        report["by_layer"][layer_name]["zero"].append(var_base)
        if var_base not in report["by_variable_type"]:
            report["by_variable_type"][var_base] = {"count": 0, "layers": []}
        report["by_variable_type"][var_base]["count"] += 1
        report["by_variable_type"][var_base]["layers"].append(layer_name)

    # Identify suspected causes
    for var_type, info in report["by_variable_type"].items():
        if var_type in ("kernel", "bias") and info["count"] > 5:
            report["suspected_causes"].append(
                f"Multiple Dense '{var_type}' vars ({info['count']}) missing gradients - "
                "likely C++ native op backward issue"
            )
        elif var_type in ("j_upper", "r_diag", "h_proj", "qsvt_coefficients"):
            report["suspected_causes"].append(
                f"QuantumEnhancedBlock '{var_type}' missing gradients - "
                "check USE_PORT_HAMILTONIAN/USE_QSVT_ACTIVATIONS flags and forward path"
            )
        elif var_type.startswith("ring_core"):
            report["suspected_causes"].append(
                f"TensorRingLayer '{var_type}' missing gradients - "
                "check fused_tensor_ring_forward custom gradient variables kwarg"
            )
        elif var_type.startswith(("shared_basis", "expert_coeffs", "expert_bias")):
            report["suspected_causes"].append(
                f"HDSharedExpertBasis '{var_type}' missing gradients - "
                "layer may not be used in current MoE expert configuration"
            )
        elif var_type in ("vqc_params", "u1_weights", "u2_weights"):
            report["suspected_causes"].append(
                f"Quantum VQC '{var_type}' missing gradients - "
                "check @tf.custom_gradient wrapper around VQC forward"
            )

    # Print diagnostic output
    if verbose >= 1:
        print("\n" + "=" * 70)
        print("GRADIENT DIAGNOSTIC REPORT")
        print("=" * 70)
        print(f"Total trainable variables: {report['total_vars']}")
        print(f"Variables with None gradient: {report['none_grad_count']}")
        print(f"Variables with zero gradient: {report['zero_grad_count']}")
        coverage = (
            report["total_vars"] - report["none_grad_count"] - report["zero_grad_count"]
        ) / max(1, report["total_vars"])
        print(f"Effective coverage: {coverage:.1%}")

        if verbose >= 2 and report["by_layer"]:
            print("\nBy Layer:")
            for layer_name, vars_info in sorted(report["by_layer"].items()):
                none_list = vars_info.get("none", [])
                zero_list = vars_info.get("zero", [])
                if none_list or zero_list:
                    print(f"  {layer_name}:")
                    if none_list:
                        print(
                            f"    None: {', '.join(none_list[:5])}"
                            + (f" (+{len(none_list)-5} more)" if len(none_list) > 5 else "")
                        )
                    if zero_list:
                        print(
                            f"    Zero: {', '.join(zero_list[:5])}"
                            + (f" (+{len(zero_list)-5} more)" if len(zero_list) > 5 else "")
                        )

        if report["suspected_causes"]:
            print("\nSuspected Root Causes:")
            for i, cause in enumerate(report["suspected_causes"][:5], 1):
                print(f"  {i}. {cause}")

        print("=" * 70 + "\n")

    return report


def _iter_layer_outputs(output):
    if output is None:
        return []
    if isinstance(output, dict):
        return [(str(key), value) for key, value in output.items()]
    if isinstance(output, (list, tuple)):
        return [(str(idx), value) for idx, value in enumerate(output)]
    return [("", output)]


def _scan_nonfinite_layers(
    model: tf.keras.Model,
    inputs: Any,
    training: bool = True,
    labels: Any | None = None,
    loss_fn: Any | None = None,
) -> dict[str, Any]:
    outputs: list[tf.Tensor] = []
    meta: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    reasoning_layer = None
    try:
        from highnoon.models.reasoning.reasoning_module import ReasoningModule

        reasoning_layer = next(
            (layer for layer in model.layers if isinstance(layer, ReasoningModule)),
            None,
        )
    except Exception:
        reasoning_layer = None

    def _source_layer(tensor: Any):
        history = getattr(tensor, "_keras_history", None)
        if history is None:
            return None
        if isinstance(history, tuple):
            return history[0]
        return getattr(history, "layer", None)

    def _add_output(
        tensor: Any,
        layer_name: str,
        layer_class: str,
        output_label: str,
        role: str = "layer",
    ) -> None:
        if tensor is None or not hasattr(tensor, "dtype"):
            return
        tensor_id = id(tensor)
        if tensor_id in seen_ids:
            return
        seen_ids.add(tensor_id)
        outputs.append(tensor)
        meta.append(
            {
                "layer_name": layer_name,
                "layer_class": layer_class,
                "output_label": output_label,
                "role": role,
            }
        )

    if reasoning_layer is not None:
        try:
            reasoning_input = reasoning_layer.input
        except Exception:
            reasoning_input = None
        if reasoning_input is not None:
            source = _source_layer(reasoning_input)
            source_name = source.name if source is not None else "reasoning_input"
            source_class = source.__class__.__name__ if source is not None else "KerasTensor"
            _add_output(reasoning_input, source_name, source_class, "0", role="reasoning_input")

        if hasattr(reasoning_layer, "reasoning_blocks"):
            for block in reasoning_layer.reasoning_blocks:
                try:
                    block_output = block.output
                except Exception:
                    continue
                for out_idx, value in _iter_layer_outputs(block_output):
                    label = out_idx if out_idx else "0"
                    _add_output(value, block.name, block.__class__.__name__, label, role="block")

        try:
            reasoning_output = reasoning_layer.output
        except Exception:
            reasoning_output = None
        if reasoning_output is not None:
            _add_output(
                reasoning_output,
                reasoning_layer.name,
                reasoning_layer.__class__.__name__,
                "0",
                role="reasoning_output",
            )

    head_layer = None
    for name in ("quantum_lm_head", "lm_head"):
        try:
            head_layer = model.get_layer(name)
            break
        except Exception:
            continue
    if head_layer is not None:
        try:
            head_output = head_layer.output
        except Exception:
            head_output = None
        if head_output is not None:
            _add_output(
                head_output, head_layer.name, head_layer.__class__.__name__, "0", role="head"
            )

    if not outputs:
        try:
            _add_output(model.output, model.name, model.__class__.__name__, "0", role="head")
        except Exception:
            return {"status": "no_outputs", "nonfinite_layers": []}

    try:
        probe_model = tf.keras.Model(inputs=model.inputs, outputs=outputs)
    except Exception as exc:
        return {"status": "build_failed", "error": str(exc), "nonfinite_layers": []}

    try:
        layer_values = probe_model(inputs, training=training)
    except Exception as exc:
        return {"status": "run_failed", "error": str(exc), "nonfinite_layers": []}

    if not isinstance(layer_values, (list, tuple)):
        layer_values = [layer_values]

    nonfinite_layers: list[dict[str, Any]] = []
    head_shape = None
    head_dtype = None
    head_has_nan = False
    head_has_inf = False
    head_loss_val = None
    for info, value in zip(meta, layer_values):
        if not hasattr(value, "dtype"):
            continue
        if info.get("role") == "head":
            head_shape = (
                value.shape.as_list() if hasattr(value.shape, "as_list") else list(value.shape)
            )
            head_dtype = value.dtype.name if hasattr(value.dtype, "name") else str(value.dtype)
            if value.dtype.is_floating or value.dtype.is_complex:
                head_has_nan = bool(tf.reduce_any(tf.math.is_nan(value)).numpy())
                head_has_inf = bool(tf.reduce_any(tf.math.is_inf(value)).numpy())
                if labels is not None and loss_fn is not None:
                    try:
                        basic_loss = loss_fn(labels, value)
                        if isinstance(basic_loss, tf.Tensor) and len(basic_loss.shape) > 0:
                            basic_loss = tf.reduce_mean(basic_loss)
                        head_loss_val = float(basic_loss.numpy())
                    except Exception:
                        head_loss_val = None
        if not (value.dtype.is_floating or value.dtype.is_complex):
            continue
        has_nan = bool(tf.reduce_any(tf.math.is_nan(value)).numpy())
        has_inf = bool(tf.reduce_any(tf.math.is_inf(value)).numpy())
        if not (has_nan or has_inf):
            continue
        shape = value.shape.as_list() if hasattr(value.shape, "as_list") else list(value.shape)
        nonfinite_layers.append(
            {
                **info,
                "shape": shape,
                "dtype": value.dtype.name if hasattr(value.dtype, "name") else str(value.dtype),
                "has_nan": has_nan,
                "has_inf": has_inf,
            }
        )

    return {
        "status": "ok",
        "nonfinite_layers": nonfinite_layers,
        "head_shape": head_shape,
        "head_dtype": head_dtype,
        "logits_has_nan": head_has_nan,
        "logits_has_inf": head_has_inf,
        "basic_loss": head_loss_val,
    }


def _capture_reasoning_block_diagnostics(
    model: tf.keras.Model,
    input_ids: tf.Tensor,
    training: bool = True,
) -> dict[str, Any]:
    reasoning_layer = getattr(model, "reasoning_module", None)
    if reasoning_layer is None and hasattr(model, "reasoning"):
        reasoning_layer = getattr(model, "reasoning", None)
    if reasoning_layer is None:
        try:
            from highnoon.models.reasoning.reasoning_module import ReasoningModule

            reasoning_layer = next(
                (layer for layer in model.layers if isinstance(layer, ReasoningModule)),
                None,
            )
        except Exception:
            reasoning_layer = None
    if reasoning_layer is None or not hasattr(reasoning_layer, "trace_block_stats"):
        return {"status": "unavailable", "blocks": []}

    try:
        if hasattr(model, "token_embedding"):
            hidden_states = model.token_embedding(input_ids)
            if hasattr(model, "embedding_dropout"):
                hidden_states = model.embedding_dropout(hidden_states, training=training)
        elif reasoning_layer.input is not None:
            probe = tf.keras.Model(inputs=model.inputs, outputs=reasoning_layer.input)
            hidden_states = probe(input_ids, training=training)
        else:
            return {"status": "missing_embedding", "blocks": []}

        _, block_stats = reasoning_layer.trace_block_stats(
            hidden_states, training=training, mask=None
        )
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "blocks": []}

    first_nonfinite = None
    for entry in block_stats:
        for stage in ("input", "normalized", "block_output", "output"):
            stats = entry.get(stage)
            if stats and (stats.get("has_nan") or stats.get("has_inf")):
                first_nonfinite = {
                    "block_index": entry.get("block_index"),
                    "block_name": entry.get("block_name"),
                    "block_class": entry.get("block_class"),
                    "stage": stage,
                    "stats": stats,
                }
                break
        if first_nonfinite:
            break

    return {
        "status": "ok",
        "first_nonfinite": first_nonfinite,
        "blocks": block_stats,
    }


def _check_native_ops(strict: bool = True) -> dict[str, Any]:
    from highnoon._native import _load_consolidated_binary, get_op, is_native_available
    from highnoon._native.ops.fused_coconut_ops import fused_coconut_bfs_available
    from highnoon._native.ops.fused_continuous_thought_op import fused_continuous_thought_available
    from highnoon._native.ops.fused_latent_reasoning_op import _fused_latent_reasoning_op

    report: dict[str, Any] = {
        "native_available": is_native_available(),
        "core_binary_loaded": False,
        "missing_symbols": [],
        "status": "ok",
    }

    core = _load_consolidated_binary()
    report["core_binary_loaded"] = core is not None

    required_symbols = {
        "QHDSpatialBlockForward": ("qhd_spatial_block", "QHDSpatialBlockForward"),
        "QHDSpatialBlockBackward": ("qhd_spatial_block", "QHDSpatialBlockBackward"),
        "HDTimeCrystalForward": ("hd_timecrystal", "HDTimeCrystalForward"),
        "HDTimeCrystalBackward": ("hd_timecrystal", "HDTimeCrystalBackward"),
        "FusedContinuousThought": ("fused_continuous_thought", "FusedContinuousThought"),
        "FusedContinuousThoughtGrad": ("fused_continuous_thought", "FusedContinuousThoughtGrad"),
        "FusedLatentReasoning": ("fused_latent_reasoning", "FusedLatentReasoning"),
        "FusedLatentReasoningGrad": ("fused_latent_reasoning", "FusedLatentReasoningGrad"),
        "fused_coconut_bfs": ("fused_coconut_bfs", "fused_coconut_bfs"),
    }

    if core is not None:
        for symbol in required_symbols:
            if not hasattr(core, symbol):
                report["missing_symbols"].append(symbol)
    else:
        for _, (op_name, symbol) in required_symbols.items():
            op = get_op(op_name)
            if op is None or not hasattr(op, symbol):
                report["missing_symbols"].append(symbol)

    if not fused_continuous_thought_available():
        report["missing_symbols"].append("fused_continuous_thought_available")
    if not fused_coconut_bfs_available():
        report["missing_symbols"].append("fused_coconut_bfs_available")
    if _fused_latent_reasoning_op is None:
        report["missing_symbols"].append("fused_latent_reasoning_available")

    if report["missing_symbols"]:
        report["status"] = "missing_ops"
        if strict:
            report["status"] = "failed"

    return report


def _prepare_trial_config(args: argparse.Namespace) -> dict[str, Any]:
    from highnoon import config as hn_config

    config = _extract_config_flags()
    if args.enable_arch_flags:
        _force_enable_arch_flags(hn_config, config)
    config.update(
        {
            "batch_size": args.batch_size,
            "context_window": args.context_window,
            "target_vocab_size": args.vocab_size,
            "hidden_dim": args.embedding_dim,
            "num_reasoning_blocks": args.num_reasoning_blocks,
            "num_heads": hn_config.REASONING_HEADS,
            "num_moe_experts": hn_config.NUM_EXPERTS,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        }
    )

    if args.hd_dim:
        config["hd_dim"] = args.hd_dim
    if args.hd_dim_embedding:
        config["hd_dim_embedding"] = args.hd_dim_embedding
    if args.hd_dim_spatial:
        config["hd_dim_spatial"] = args.hd_dim_spatial
    if args.hd_dim_timecrystal:
        config["hd_dim_timecrystal"] = args.hd_dim_timecrystal
    if args.hd_dim_moe:
        config["hd_dim_moe"] = args.hd_dim_moe

    return config


def _align_vocab_size(tokenizer, target_vocab_size: int) -> int:
    if hasattr(tokenizer, "ensure_trained_or_fallback"):
        vocab_size = tokenizer.ensure_trained_or_fallback(target_vocab_size)
    elif hasattr(tokenizer, "is_trained") and not tokenizer.is_trained:
        vocab_size = target_vocab_size
    else:
        vocab_size = tokenizer.vocab_size

    if vocab_size < 256:
        vocab_size = target_vocab_size

    return min(vocab_size, target_vocab_size)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enterprise TrainingEngine audit (C++ ops + gradients)"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--num-reasoning-blocks", type=int, default=6)
    parser.add_argument("--optimizer", type=str, default="sympflowqng")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--steps-per-epoch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no-strict", action="store_false", dest="strict")
    parser.add_argument("--hf-dataset-name", type=str, default="")
    parser.add_argument("--curriculum-id", type=str, default="")
    parser.add_argument("--hd-dim", type=int, default=0)
    parser.add_argument("--hd-dim-embedding", type=int, default=0)
    parser.add_argument("--hd-dim-spatial", type=int, default=0)
    parser.add_argument("--hd-dim-timecrystal", type=int, default=0)
    parser.add_argument("--hd-dim-moe", type=int, default=0)
    parser.add_argument("--report", type=str, default="")
    parser.add_argument(
        "--forward-only",
        action="store_true",
        default=False,
        help="Run a single forward pass with block diagnostics (skip training)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for forward-only diagnostics",
    )
    parser.add_argument(
        "--forward-training",
        action="store_true",
        default=False,
        help="Use training=True in forward-only diagnostics",
    )
    parser.add_argument(
        "--enable-arch-flags",
        action="store_true",
        default=False,
        help="Force-enable advanced architecture flags for audit runs",
    )
    # Layer diagnostics flags
    parser.add_argument(
        "--layer-diagnostics",
        action="store_true",
        default=True,
        help="Enable per-layer activation/gradient diagnostics (default: True)",
    )
    parser.add_argument(
        "--no-layer-diagnostics",
        action="store_false",
        dest="layer_diagnostics",
        help="Disable per-layer diagnostics",
    )
    parser.add_argument(
        "--diagnostics-frequency",
        type=int,
        default=1,
        help="Capture diagnostics every N steps (default: 1)",
    )
    parser.add_argument(
        "--max-layers-trace",
        type=int,
        default=50,
        help="Maximum layers to trace for diagnostics (default: 50, 0=unlimited)",
    )
    parser.add_argument(
        "--verbose-gradients",
        action="store_true",
        default=False,
        help="Show detailed per-variable gradient statistics",
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        default=False,
        help="Use compact single-line-per-layer output format",
    )
    parser.add_argument(
        "--profile-flops",
        action="store_true",
        default=True,
        help="Enable per-step FLOPS profiling (default: True, uses architecture estimation if profiler unavailable)",
    )
    parser.add_argument(
        "--flops-frequency",
        type=int,
        default=1,
        help="Profile FLOPS every N steps to reduce overhead (default: 1)",
    )
    parser.add_argument(
        "--no-profile-flops",
        action="store_false",
        dest="profile_flops",
        help="Disable FLOPS profiling",
    )
    parser.add_argument(
        "--export-params",
        type=str,
        default="",
        help="Export all trainable parameters with stats to a separate JSON file for HPO analysis",
    )
    args = parser.parse_args()

    op_report = _check_native_ops(strict=args.strict)
    if op_report["status"] == "failed":
        print("[Audit] Native op validation failed.")
        print(json.dumps(op_report, indent=2))
        return 1

    from highnoon import config as hn_config
    from highnoon.services.hpo_trial_runner import (
        build_hsmn_model,
        create_engine_config,
        create_optimizer,
    )
    from highnoon.training.gradient_audit import GradientAudit, GradientAuditConfig
    from highnoon.training.layer_diagnostics import (
        LayerDiagnostics,
        LayerDiagnosticsConfig,
        LayerDiagnosticsSnapshot,
    )
    from highnoon.training.training_engine import TrainingEngine

    if args.embedding_dim != hn_config.HD_DIM_MOE:
        msg = (
            f"[Audit] embedding_dim ({args.embedding_dim}) must match "
            f"HD_DIM_MOE ({hn_config.HD_DIM_MOE}) for fused_superposition_moe."
        )
        if args.strict:
            raise RuntimeError(msg)
        print(msg)

    trial_config = _prepare_trial_config(args)
    process = psutil.Process() if PSUTIL_AVAILABLE else None
    runtime_snapshot = {
        "start_time": time.time(),
        "start_memory": _memory_snapshot(process),
    }

    vocab_size = trial_config["target_vocab_size"]

    hd_dim = trial_config.get("hd_dim")
    hd_dim_embedding = (
        trial_config.get("hd_dim_embedding") or hd_dim or (trial_config["hidden_dim"] * 8)
    )
    hd_dim_spatial = (
        trial_config.get("hd_dim_spatial") or hd_dim or (trial_config["hidden_dim"] * 8)
    )
    hd_dim_timecrystal = (
        trial_config.get("hd_dim_timecrystal") or hd_dim or (trial_config["hidden_dim"] * 8)
    )
    hd_dim_moe = trial_config.get("hd_dim_moe") or hd_dim or (trial_config["hidden_dim"] * 8)

    model = build_hsmn_model(
        trial_config,
        vocab_size,
        hidden_dim_override=trial_config["hidden_dim"],
        hd_dim=hd_dim,
        hd_dim_embedding=hd_dim_embedding,
        hd_dim_spatial=hd_dim_spatial,
        hd_dim_timecrystal=hd_dim_timecrystal,
        hd_dim_moe=hd_dim_moe,
    )

    model_stats = {
        "total_params": int(model.count_params()),
        "trainable_params": int(
            sum(v.shape.num_elements() or 0 for v in model.trainable_variables)
        ),
        "non_trainable_params": int(
            sum(v.shape.num_elements() or 0 for v in model.non_trainable_variables)
        ),
        "layer_count": len(model.layers),
        "layers": [
            {
                "name": layer.name,
                "class": layer.__class__.__name__,
                "trainable": layer.trainable,
                "params": int(layer.count_params()),
            }
            for layer in model.layers
        ],
        "trainable_variables": [
            {
                "name": var.name,
                "shape": var.shape.as_list(),
                "dtype": var.dtype.name if hasattr(var.dtype, "name") else str(var.dtype),
            }
            for var in model.trainable_variables
        ],
    }

    # Export detailed parameter dump if requested
    if args.export_params.strip():
        print("[Audit] Exporting parameters to:", args.export_params)
        param_export: dict[str, Any] = {
            "export_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_trainable_params": model_stats["trainable_params"],
            "total_non_trainable_params": model_stats["non_trainable_params"],
            "trainable_variables": [],
            "non_trainable_variables": [],
        }

        def _var_stats(var: tf.Variable) -> dict[str, Any]:
            """Compute statistics for a variable."""
            try:
                values = var.numpy().flatten().astype(float)
                return {
                    "name": var.name,
                    "shape": var.shape.as_list(),
                    "dtype": var.dtype.name if hasattr(var.dtype, "name") else str(var.dtype),
                    "num_elements": int(var.shape.num_elements() or 0),
                    "trainable": var.trainable,
                    "stats": {
                        "min": float(values.min()) if len(values) > 0 else None,
                        "max": float(values.max()) if len(values) > 0 else None,
                        "mean": float(values.mean()) if len(values) > 0 else None,
                        "std": float(values.std()) if len(values) > 0 else None,
                        "has_nan": bool(np.isnan(values).any()),
                        "has_inf": bool(np.isinf(values).any()),
                        "zero_fraction": (
                            float((values == 0).sum() / len(values)) if len(values) > 0 else None
                        ),
                    },
                }
            except Exception as e:
                return {
                    "name": var.name,
                    "shape": var.shape.as_list(),
                    "dtype": var.dtype.name if hasattr(var.dtype, "name") else str(var.dtype),
                    "num_elements": int(var.shape.num_elements() or 0),
                    "trainable": var.trainable,
                    "stats": {"error": str(e)},
                }

        import numpy as np  # Ensure numpy is available for stats

        for var in model.trainable_variables:
            param_export["trainable_variables"].append(_var_stats(var))

        for var in model.non_trainable_variables:
            param_export["non_trainable_variables"].append(_var_stats(var))

        # Group by layer/component for easier analysis
        grouped_params: dict[str, list[dict[str, Any]]] = {}
        for entry in param_export["trainable_variables"]:
            # Extract component name from variable name (e.g., "block_0/dense/kernel:0" -> "block_0")
            name_parts = entry["name"].split("/")
            component = name_parts[0] if len(name_parts) > 1 else "root"
            if component not in grouped_params:
                grouped_params[component] = []
            grouped_params[component].append(entry)

        param_export["grouped_by_component"] = grouped_params

        # Summary statistics
        param_export["summary"] = {
            "num_trainable_vars": len(param_export["trainable_variables"]),
            "num_non_trainable_vars": len(param_export["non_trainable_variables"]),
            "num_components": len(grouped_params),
            "component_names": list(grouped_params.keys()),
            "vars_with_nan": [
                e["name"]
                for e in param_export["trainable_variables"]
                if e.get("stats", {}).get("has_nan")
            ],
            "vars_with_inf": [
                e["name"]
                for e in param_export["trainable_variables"]
                if e.get("stats", {}).get("has_inf")
            ],
            "vars_all_zero": [
                e["name"]
                for e in param_export["trainable_variables"]
                if e.get("stats", {}).get("zero_fraction") == 1.0
            ],
        }

        Path(args.export_params).write_text(json.dumps(param_export, indent=2))
        print(
            f"[Audit] Exported {len(param_export['trainable_variables'])} trainable vars, "
            f"{len(param_export['non_trainable_variables'])} non-trainable vars"
        )

    if args.forward_only:
        seq_len = max(1, args.seq_len)
        forward_training = args.forward_training
        input_ids = tf.random.uniform(
            shape=(trial_config["batch_size"], seq_len),
            minval=0,
            maxval=vocab_size,
            dtype=tf.int32,
        )
        block_diag = _capture_reasoning_block_diagnostics(
            model, input_ids, training=forward_training
        )
        if block_diag["status"] == "ok":
            print("[Audit] Reasoning block diagnostics (forward-only):")

            def _stage_status(stats: dict[str, Any] | None) -> str:
                if not stats:
                    return "n/a"
                if stats.get("has_nan") and stats.get("has_inf"):
                    return "NaN+Inf"
                if stats.get("has_nan"):
                    return "NaN"
                if stats.get("has_inf"):
                    return "Inf"
                return "ok"

            for entry in block_diag.get("blocks", []):
                print(
                    f"  - {entry.get('block_name')} ({entry.get('block_class')}): "
                    f"in={_stage_status(entry.get('input'))} "
                    f"norm={_stage_status(entry.get('normalized'))} "
                    f"block={_stage_status(entry.get('block_output'))} "
                    f"out={_stage_status(entry.get('output'))}"
                )
            first_block = block_diag.get("first_nonfinite")
            if first_block:
                print(
                    "[Audit] First nonfinite reasoning block: "
                    f"{first_block['block_name']} ({first_block['block_class']}) "
                    f"stage={first_block['stage']}"
                )
        else:
            detail = block_diag.get("error", "")
            detail = f": {detail}" if detail else ""
            print("[Audit] Reasoning block diagnostics failed: " f"{block_diag['status']}{detail}")

        raw_output = model(input_ids, training=forward_training)
        if isinstance(raw_output, dict):
            predictions = raw_output.get("logits", raw_output.get("output"))
        else:
            predictions = raw_output
        if predictions is not None:
            has_nan = bool(tf.reduce_any(tf.math.is_nan(predictions)).numpy())
            has_inf = bool(tf.reduce_any(tf.math.is_inf(predictions)).numpy())
            print(f"[Audit] Logits NaN={has_nan} Inf={has_inf}")

        return 0

    from highnoon.data.loaders import load_training_dataset

    curriculum_id = args.curriculum_id.strip()
    dataset_name = args.hf_dataset_name.strip()
    resolved_dataset = None
    curriculum_datasets: list[str] = []
    if dataset_name:
        resolved_dataset = dataset_name
    elif curriculum_id:
        resolved_dataset, curriculum_datasets = _resolve_curriculum_dataset(curriculum_id)
        if resolved_dataset is None:
            print(f"[Audit] Curriculum '{curriculum_id}' not resolved to a dataset.")

    if resolved_dataset:
        trial_config["hf_dataset_name"] = resolved_dataset
    if curriculum_id:
        trial_config["curriculum_id"] = curriculum_id
    if curriculum_datasets:
        trial_config["curriculum_datasets"] = curriculum_datasets

    train_dataset, tokenizer, merger = load_training_dataset(
        batch_size=trial_config["batch_size"],
        context_window=trial_config["context_window"],
        streaming_mode="packed",
        vocab_size=trial_config["target_vocab_size"],
        hf_dataset_name=trial_config.get("hf_dataset_name"),
        hf_dataset_names=trial_config.get("curriculum_datasets"),  # Multi-dataset curriculum
        prefetch_buffer_size=trial_config.get("prefetch_buffer_size"),
        use_adaptive_tokenizer=trial_config.get(
            "use_adaptive_tokenizer",
            trial_config.get("use_intelligent_vocab_controller", True),
        ),
        adaptive_min_freq=trial_config.get(
            "vocab_controller_min_ngram_freq",
            trial_config.get("vocab_min_ngram_freq", 10),
        ),
        max_samples=trial_config.get(
            "vocab_controller_sample_size",
            trial_config.get("vocab_sample_size"),
        ),
    )

    vocab_size = _align_vocab_size(tokenizer, trial_config["target_vocab_size"])
    if merger is not None:
        vocab_size += merger.superword_count

    optimizer = create_optimizer(trial_config, model=model)
    engine_config = create_engine_config(trial_config)

    audit_config = GradientAuditConfig(
        # Variables NOT trained by SGD - these are expected to have zero gradients
        expected_no_grad_patterns=(
            "evolution_time",  # Meta-Controller controlled
            "last_stable",  # State tracking
            "continuous_thought_gate",
            "blend_gate",
            "hd_char_basis",  # Frozen HD basis
            "hd_position_basis",  # Frozen HD basis
            "hd_projection",  # Rare token path (conditional)
            "dense_fallback",  # VQC fallback path (conditional blend)
            "fallback_weight",  # Blend weight for fallback
            "state_transition",  # Kalman state matrix
            "process_noise",  # Kalman noise model
            "observation_noise",  # Kalman noise model
            "floquet_step_counter",  # Step counter (non-trainable)
            "floquet_phase",  # Phase tracker (non-trainable)
            "mps_core",  # MPS cores are auxiliary state
            "coherence",  # Coherence metric (non-trainable)
            "gru_w_",  # Kalman GRU weights (trainable=False)
            "gru_b_",  # Kalman GRU biases (trainable=False)
        ),
        group_patterns={
            # Updated to match actual layer naming with "quantum_" prefix
            "QHDSpatialBlock": ("quantum_qhd_spatial", "qhd_spatial_block"),
            "HDTimeCrystalBlock": ("quantum_hd_timecrystal", "hd_timecrystal"),
            "ContinuousThoughtBlock": ("quantum_continuous_thought", "continuous_thought"),
            "LatentReasoningBlock": ("quantum_latent_reasoning", "latent_reasoning"),
            "QuantumLMHead": ("quantum_lm_head", "lm_head"),
            "KalmanBlock": ("quantum_kalman", "kalman_block"),
            "SelfConsistencyBlock": ("quantum_self_consistency", "self_consistency"),
            "WLAMBlock": ("quantum_wlam", "wlam_block"),
            "MoELayer": ("quantum_moe", "moe_layer"),
        },
        zero_tol=1e-12,
        sample_limit=len(model.trainable_variables),
        include_per_variable_stats=True,
        per_variable_limit=0,
    )

    # Initialize layer diagnostics if enabled
    layer_diagnostics: LayerDiagnostics | None = None
    if args.layer_diagnostics:
        layer_diag_config = LayerDiagnosticsConfig(
            capture_frequency=args.diagnostics_frequency,
            max_layers=args.max_layers_trace,
            include_sublayers=True,
        )
        layer_diagnostics = LayerDiagnostics(layer_diag_config)

    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        config=engine_config,
        gradient_audit=GradientAudit(audit_config),
    )
    engine.set_total_steps(args.steps_per_epoch * args.epochs)

    audit_records: list[dict[str, Any]] = []
    last_result = {"step": 0, "result": None}
    last_batch: dict[str, Any] = {
        "inputs": None,
        "labels": None,
        "gradients": None,
        "variables": None,
    }
    perf_records: list[dict[str, Any]] = []
    layer_diagnostics_records: list[dict[str, Any]] = []
    nan_diagnostics: dict[str, Any] | None = None
    reasoning_block_diagnostics: dict[str, Any] | None = None

    def _format_float(val: float | None, precision: int = 4) -> str:
        """Format float for display, handling special values."""
        if val is None:
            return "-"
        if math.isnan(val):
            return "nan"
        if math.isinf(val):
            return "inf" if val > 0 else "-inf"
        if abs(val) < 1e-10:
            return "0.00e+00"
        return f"{val:.{precision}e}"

    def _print_layer_diagnostics_table(
        snapshot: LayerDiagnosticsSnapshot,
        compact: bool = False,
        verbose_grads: bool = False,
    ) -> None:
        """Print formatted layer diagnostics table to console."""
        if not snapshot.activations:
            print(f"[Step {snapshot.step}] No layer activations captured.")
            return

        if compact:
            print(f"\n[Step {snapshot.step}] Layer Diagnostics (compact):")
            print(snapshot.format_compact())
        else:
            print(f"\n[Step {snapshot.step}] Layer-by-Layer Diagnostics:")
            print(snapshot.format_table())

        # Print gradient details if verbose
        if verbose_grads and snapshot.gradients:
            print(f"\n[Step {snapshot.step}] Per-Variable Gradient Details:")
            print("-" * 90)
            for g in snapshot.gradients[:30]:  # Limit output
                status_icon = "⚠" if g.has_nan or g.has_inf or g.is_zero else "✓"
                print(
                    f"  {status_icon} {g.name[:60]:<60} "
                    f"norm={_format_float(g.norm)} "
                    f"max={_format_float(g.max_abs)} "
                    f"status={g.status}"
                )
            if len(snapshot.gradients) > 30:
                print(f"  ... and {len(snapshot.gradients) - 30} more variables")

    class AuditCallback:
        """Enhanced callback for training audit with layer diagnostics."""

        def __init__(self) -> None:
            self.current_epoch = 0
            self._step_start = 0.0
            self._cpu_start = 0.0
            self._step_count = 0

        def on_batch_start(self, step: int) -> bool:
            self._step_start = time.perf_counter()
            self._cpu_start = time.process_time()
            return True

        def on_epoch_start(self, epoch: int) -> bool:
            self.current_epoch = epoch
            return True

        def on_epoch_end(self, epoch: int, result) -> bool:
            return True

        def on_batch_end(self, step: int, result) -> bool:
            step_time = time.perf_counter() - self._step_start
            cpu_time = time.process_time() - self._cpu_start
            tokens = _count_tokens(last_batch["inputs"])
            tokens_per_sec = None
            if tokens is not None and step_time > 0:
                tokens_per_sec = tokens / step_time
            memory = _memory_snapshot(process)
            audit = engine.get_last_gradient_audit()
            audit_data = asdict(audit) if audit is not None else {}

            # Profile FLOPS if enabled
            flops_data: dict[str, Any] | None = None
            if args.profile_flops and last_batch["inputs"] is not None:
                if self._step_count % args.flops_frequency == 0:
                    flops_data = _profile_flops(model, last_batch["inputs"], training=True)
                    if flops_data.get("profiler_available") and flops_data.get("gflops"):
                        gflops = flops_data["gflops"]
                        flops_per_sec = None
                        if step_time > 0:
                            flops_per_sec = flops_data["total_flops"] / step_time
                        flops_data["flops_per_sec"] = flops_per_sec
                        flops_data["tflops_per_sec"] = (
                            flops_per_sec / 1e12 if flops_per_sec else None
                        )

            # Capture layer diagnostics if enabled
            layer_diag_snapshot: LayerDiagnosticsSnapshot | None = None
            if layer_diagnostics is not None and last_batch["inputs"] is not None:
                # Get gradients from the gradient audit if available
                gradients = last_batch.get("gradients")
                variables = last_batch.get("variables")

                layer_diag_snapshot = layer_diagnostics.capture_step(
                    model=model,
                    inputs=last_batch["inputs"],
                    gradients=gradients,
                    variables=variables,
                    step=step,
                    loss=result.loss,
                    grad_norm=result.gradient_norm,
                    training=True,
                )

                # Print diagnostics to console
                _print_layer_diagnostics_table(
                    layer_diag_snapshot,
                    compact=args.compact_output,
                    verbose_grads=args.verbose_gradients,
                )

                # Store for report
                layer_diagnostics_records.append(layer_diag_snapshot.to_dict())

            # Print step summary
            loss_display = _format_float(result.loss, 6)
            grad_norm_display = _format_float(result.gradient_norm, 4)
            valid_str = "✓" if result.is_valid else "✗"
            print(
                f"\n[Step {step}] Summary: loss={loss_display} "
                f"grad_norm={grad_norm_display} lr={result.effective_learning_rate:.2e} "
                f"valid={valid_str}"
            )

            # Show warning if NaN/Inf detected
            if not result.is_valid:
                print(f"⚠ [Step {step}] NaN/Inf detected in loss or gradients!")
                if layer_diag_snapshot:
                    if layer_diag_snapshot.first_nan_activation:
                        print(
                            f"  → First NaN activation layer: "
                            f"{layer_diag_snapshot.first_nan_activation}"
                        )
                    if layer_diag_snapshot.first_inf_activation:
                        print(
                            f"  → First Inf activation layer: "
                            f"{layer_diag_snapshot.first_inf_activation}"
                        )
                    if layer_diag_snapshot.first_nan_gradient:
                        print(
                            f"  → First NaN gradient variable: "
                            f"{layer_diag_snapshot.first_nan_gradient}"
                        )

            # Show gradient audit summary
            if audit and args.verbose_gradients:
                print(
                    f"[Step {step}] Gradient Coverage: "
                    f"{audit.nonzero_vars}/{audit.total_vars} "
                    f"({audit.coverage_ratio:.1%})"
                )
                if audit.unexpected_samples:
                    print("  ⚠ Unexpected zero/missing gradients:")
                    for name in audit.unexpected_samples[:5]:
                        print(f"    - {name}")

                    # Run enhanced gradient diagnostic on step 0 for detailed analysis
                    # TODO: Implement _print_gradient_diagnostic_from_audit function
                    # if step == 0 and hasattr(audit, "variable_stats") and audit.variable_stats:
                    #     _print_gradient_diagnostic_from_audit(
                    #         audit.variable_stats,
                    #         audit.unexpected_samples,
                    #         verbose=2 if args.verbose_gradients else 1,
                    #     )

            audit_records.append(
                {
                    "epoch": self.current_epoch,
                    "step": step,
                    "loss": result.loss,
                    "grad_norm": result.gradient_norm,
                    "lr": result.effective_learning_rate,
                    "is_valid": result.is_valid,
                    "audit": audit_data,
                    "timing": {
                        "step_time_s": step_time,
                        "cpu_time_s": cpu_time,
                        "tokens": tokens,
                        "tokens_per_sec": tokens_per_sec,
                    },
                    "memory": memory,
                    "flops": flops_data,
                }
            )
            perf_records.append(
                {
                    "step_time_s": step_time,
                    "cpu_time_s": cpu_time,
                    "tokens": tokens,
                    "tokens_per_sec": tokens_per_sec,
                    "rss_mb": memory.get("rss_mb"),
                    "gflops": flops_data.get("gflops") if flops_data else None,
                    "tflops_per_sec": flops_data.get("tflops_per_sec") if flops_data else None,
                }
            )
            last_result["step"] = step
            last_result["result"] = result
            self._step_count += 1
            return True

    def _dataset_with_capture():
        for inputs, labels in train_dataset:
            last_batch["inputs"] = inputs
            last_batch["labels"] = labels
            yield inputs, labels

    engine.run(
        epochs=args.epochs,
        dataset=_dataset_with_capture(),
        callbacks=[AuditCallback()],
        steps_per_epoch=args.steps_per_epoch,
    )

    print("\n[Audit] Native ops status:", op_report["status"])
    if op_report["missing_symbols"]:
        print("[Audit] Missing symbols:", ", ".join(op_report["missing_symbols"]))
    print(
        "[Audit] Config: "
        f"batch={trial_config['batch_size']} context={trial_config['context_window']} "
        f"dim={trial_config['hidden_dim']} blocks={trial_config['num_reasoning_blocks']} "
        f"vocab={vocab_size} optimizer={trial_config['optimizer']}"
    )
    print(
        "[Audit] Model: "
        f"layers={model_stats['layer_count']} "
        f"params={model_stats['total_params']} "
        f"trainable={model_stats['trainable_params']} "
        f"non_trainable={model_stats['non_trainable_params']}"
    )

    if last_result["result"] is not None:
        result = last_result["result"]
        print(
            f"[Audit] Last step: loss={result.loss:.6f} "
            f"grad_norm={result.gradient_norm:.6f} "
            f"lr={result.effective_learning_rate:.6e} "
            f"valid={result.is_valid}"
        )
        if result.metrics:
            quls_keys = [k for k in result.metrics if k.startswith("quls/")]
            if quls_keys:
                print("[Audit] QULS metrics:")
                for key in sorted(quls_keys):
                    print(f"  - {key}={result.metrics[key]:.6f}")

        if not result.is_valid and last_batch["inputs"] is not None:
            print("[Audit] NaN/Inf detected. Running forward-pass diagnostics...")
            raw_output = model(last_batch["inputs"], training=True)
            if isinstance(raw_output, dict):
                predictions = raw_output.get("logits", raw_output.get("output"))
            else:
                predictions = raw_output
            if predictions is not None:
                has_nan = bool(tf.reduce_any(tf.math.is_nan(predictions)).numpy())
                has_inf = bool(tf.reduce_any(tf.math.is_inf(predictions)).numpy())
                print(f"[Audit] Logits NaN={has_nan} Inf={has_inf}")
                basic_loss = engine._loss_fn(last_batch["labels"], predictions)
                if hasattr(basic_loss, "numpy"):
                    basic_loss_val = float(tf.reduce_mean(basic_loss).numpy())
                else:
                    basic_loss_val = float(basic_loss)
                print(f"[Audit] Basic loss (non-QULS)={basic_loss_val:.6f}")
            nan_diagnostics = _scan_nonfinite_layers(model, last_batch["inputs"], training=True)
            if nan_diagnostics["status"] == "ok":
                layers = nan_diagnostics.get("nonfinite_layers", [])
                if layers:
                    first = layers[0]
                    print(
                        "[Audit] First nonfinite layer: "
                        f"{first['layer_name']} ({first['layer_class']}) "
                        f"output={first['output_label']} "
                        f"shape={first['shape']} "
                        f"nan={first['has_nan']} inf={first['has_inf']}"
                    )
                    if len(layers) > 1:
                        print(
                            "[Audit] Additional nonfinite layers "
                            f"({len(layers) - 1} more, showing up to 10):"
                        )
                        for entry in layers[1:11]:
                            print(
                                f"  - {entry['layer_name']} ({entry['layer_class']}) "
                                f"output={entry['output_label']} "
                                f"shape={entry['shape']} "
                                f"nan={entry['has_nan']} inf={entry['has_inf']}"
                            )
                else:
                    print("[Audit] No nonfinite activations detected in layer outputs.")
            else:
                error_msg = nan_diagnostics.get("error", "")
                detail = f": {error_msg}" if error_msg else ""
                print(
                    "[Audit] Nonfinite layer scan failed: " f"{nan_diagnostics['status']}{detail}"
                )
            reasoning_block_diagnostics = _capture_reasoning_block_diagnostics(
                model, last_batch["inputs"], training=True
            )
            if reasoning_block_diagnostics["status"] == "ok":
                block_entries = reasoning_block_diagnostics.get("blocks", [])
                if block_entries:
                    print("[Audit] Reasoning block diagnostics:")

                    def _stage_status(stats: dict[str, Any] | None) -> str:
                        if not stats:
                            return "n/a"
                        if stats.get("has_nan") and stats.get("has_inf"):
                            return "NaN+Inf"
                        if stats.get("has_nan"):
                            return "NaN"
                        if stats.get("has_inf"):
                            return "Inf"
                        return "ok"

                    for entry in block_entries:
                        print(
                            f"  - {entry.get('block_name')} ({entry.get('block_class')}): "
                            f"in={_stage_status(entry.get('input'))} "
                            f"norm={_stage_status(entry.get('normalized'))} "
                            f"block={_stage_status(entry.get('block_output'))} "
                            f"out={_stage_status(entry.get('output'))}"
                        )
                first_block = reasoning_block_diagnostics.get("first_nonfinite")
                if first_block:
                    print(
                        "[Audit] First nonfinite reasoning block: "
                        f"{first_block['block_name']} ({first_block['block_class']}) "
                        f"stage={first_block['stage']}"
                    )
            else:
                detail = reasoning_block_diagnostics.get("error", "")
                detail = f": {detail}" if detail else ""
                print(
                    "[Audit] Reasoning block diagnostics failed: "
                    f"{reasoning_block_diagnostics['status']}{detail}"
                )
            if engine._quls is not None and predictions is not None:
                quls_loss, quls_components = engine._quls.compute_loss(
                    predictions=predictions,
                    targets=last_batch["labels"],
                    hidden_states=None,
                    vqc_outputs=None,
                    coherence_metrics=None,
                    hamiltonian_energies=None,
                    bond_entropies=None,
                    gradient_entropy=None,
                    in_barren_plateau=False,
                    zne_statistics=None,
                )
                quls_loss_val = float(quls_loss.numpy())
                print(f"[Audit] QULS loss={quls_loss_val:.6f}")
                for key, value in quls_components.items():
                    if hasattr(value, "numpy"):
                        value = float(value.numpy())
                    print(f"  - quls/{key}={value:.6f}")

    if audit_records:
        final_audit = audit_records[-1]["audit"]
        if final_audit:
            print(
                "[Audit] Gradient coverage: "
                f"{final_audit.get('nonzero_vars', 0)}/"
                f"{final_audit.get('total_vars', 0)} "
                f"({final_audit.get('coverage_ratio', 0.0):.3f})"
            )
            unexpected = final_audit.get("unexpected_samples", [])
            if unexpected:
                print("[Audit] Unexpected missing/zero gradients (sample):")
                for name in unexpected:
                    print(f"  - {name}")
            group_stats = final_audit.get("group_stats", {})
            if group_stats:
                print("[Audit] Group coverage:")
                for group, stats in group_stats.items():
                    print(
                        f"  - {group}: {stats.get('nonzero', 0)}/"
                        f"{stats.get('total', 0)} "
                        f"({stats.get('coverage_ratio', 0.0):.3f})"
                    )

    perf_summary: dict[str, Any] = {}
    if perf_records:
        step_times = [
            rec["step_time_s"] for rec in perf_records if rec.get("step_time_s") is not None
        ]
        cpu_times = [rec["cpu_time_s"] for rec in perf_records if rec.get("cpu_time_s") is not None]
        tokens_per_sec = [
            rec["tokens_per_sec"] for rec in perf_records if rec.get("tokens_per_sec") is not None
        ]
        rss_values = [rec["rss_mb"] for rec in perf_records if rec.get("rss_mb") is not None]
        if step_times:
            perf_summary["step_time_avg_s"] = sum(step_times) / len(step_times)
            perf_summary["step_time_min_s"] = min(step_times)
            perf_summary["step_time_max_s"] = max(step_times)
        if cpu_times:
            perf_summary["cpu_time_avg_s"] = sum(cpu_times) / len(cpu_times)
        if tokens_per_sec:
            perf_summary["tokens_per_sec_avg"] = sum(tokens_per_sec) / len(tokens_per_sec)
            perf_summary["tokens_per_sec_max"] = max(tokens_per_sec)
        if rss_values:
            perf_summary["rss_peak_mb"] = max(rss_values)
            perf_summary["rss_last_mb"] = rss_values[-1]
        # FLOPS summary
        gflops_values = [rec["gflops"] for rec in perf_records if rec.get("gflops") is not None]
        tflops_per_sec_values = [
            rec["tflops_per_sec"] for rec in perf_records if rec.get("tflops_per_sec") is not None
        ]
        if gflops_values:
            perf_summary["gflops_avg"] = sum(gflops_values) / len(gflops_values)
            perf_summary["gflops_per_step"] = gflops_values[-1] if gflops_values else None
        if tflops_per_sec_values:
            perf_summary["tflops_per_sec_avg"] = sum(tflops_per_sec_values) / len(
                tflops_per_sec_values
            )
            perf_summary["tflops_per_sec_max"] = max(tflops_per_sec_values)

    if perf_summary:
        step_avg = perf_summary.get("step_time_avg_s")
        step_max = perf_summary.get("step_time_max_s")
        tps_avg = perf_summary.get("tokens_per_sec_avg")
        rss_peak = perf_summary.get("rss_peak_mb")
        tflops_avg = perf_summary.get("tflops_per_sec_avg")
        tps_display = f"{tps_avg:.2f}" if tps_avg is not None else "n/a"
        rss_display = f"{rss_peak:.2f}" if rss_peak is not None else "n/a"
        tflops_display = f"{tflops_avg:.4f}" if tflops_avg is not None else "n/a"
        print(
            "[Audit] Performance: "
            f"step_avg={step_avg:.4f}s step_max={step_max:.4f}s "
            f"tokens/s={tps_display} rss_peak_mb={rss_display}"
        )
        if tflops_avg is not None:
            gflops_per_step = perf_summary.get("gflops_avg", 0)
            print(
                f"[Audit] FLOPS: "
                f"gflops/step={gflops_per_step:.2f} "
                f"tflops/s_avg={tflops_display}"
            )

    report_path = args.report.strip()
    if report_path:
        runtime_snapshot["end_time"] = time.time()
        runtime_snapshot["end_memory"] = _memory_snapshot(process)
        runtime_snapshot["duration_s"] = (
            runtime_snapshot["end_time"] - runtime_snapshot["start_time"]
        )
        report = {
            "op_report": op_report,
            "trial_config": trial_config,
            "audit_records": audit_records,
            "model_stats": model_stats,
            "performance_summary": perf_summary,
            "runtime_snapshot": runtime_snapshot,
            "arch_flags_enabled": args.enable_arch_flags,
            "engine_config": {
                "loss_function": engine_config.loss_function,
                "use_qng": engine_config.use_qng,
                "use_galore": engine_config.use_galore,
                "use_quls": engine_config.use_quls,
                "use_unified_smart_tuner": engine_config.use_unified_smart_tuner,
            },
            "config_snapshot": _extract_config_flags(),
            "nan_diagnostics": nan_diagnostics,
            "reasoning_block_diagnostics": reasoning_block_diagnostics,
            "layer_diagnostics": layer_diagnostics_records if args.layer_diagnostics else [],
        }
        Path(report_path).write_text(json.dumps(report, indent=2))
        print(f"[Audit] Wrote report to {report_path}")

    if op_report["status"] == "failed":
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
