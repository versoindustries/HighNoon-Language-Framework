# src/training/training_loop.py
import json  # Import for structured JSON logging
import logging
import math  # Import for is_finite check
import os
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import psutil
import tensorflow as tf

# --- START: DEFINITIVE FIX for Keras 3 Variable Type ---
# Import the Keras backend variable type to handle modern TensorFlow versions
# where Keras layers create their own variable objects.
from keras.src.backend import Variable as KerasVariable

# Import config module for global feature flags
# Using `hn_config` for consistency with hpo_trial_runner.py
import highnoon.config as hn_config
from highnoon._native.ops import (
    train_step as train_step_loader,  # Ensure TrainStep custom op is registered
)
from highnoon.analysis.memory_profiler import MemoryProfiler
from highnoon.analysis.system_identification import parse_log_file, prepare_data_from_logs
from highnoon.config import (  # Lite Edition Limits (enforced by C++ binaries); EWC (deprecated but kept for compatibility); GaLore gradient compression; Quantum training flags; Neural error mitigation; Meta controller
    BARREN_PLATEAU_MONITOR,
    BARREN_PLATEAU_RECOVERY_LR_SCALE,
    BARREN_PLATEAU_THRESHOLD,
    ENABLE_EWC,
    EWC_LAMBDA,
    GALORE_RANK,
    GALORE_SCALE,
    GALORE_UPDATE_PROJ_GAP,
    LITE_MAX_CONTEXT_LENGTH,
    LITE_MAX_MOE_EXPERTS,
    LITE_MAX_PARAMS,
    LITE_MAX_REASONING_BLOCKS,
    META_CONTROLLER_FREQUENCY,
    QNG_DAMPING,
    USE_META_CONTROLLER,
    USE_NEURAL_QEM,
    USE_NEURAL_ZNE,
    USE_QUANTUM_NATURAL_GRADIENT,
    USE_TENSOR_GALORE,
)

# Note: Full config module available via `config.*` for any additional flags
from highnoon.models.hsmn import HSMN
from highnoon.runtime.control_config import ensure_control_configs
from highnoon.training.callbacks import HamiltonianMetaControllerCallback
from highnoon.training.control_bridge import EvolutionTimeControlBridge
from highnoon.training.data_utils import ensure_path

# GaLore gradient compression
from highnoon.training.gradient_compression import GaLoreOptimizerWrapper, TensorGaLoreCompressor

# HPO integration
from highnoon.training.hpo_bridge import HPOReporter, load_trial_config
from highnoon.training.memory_budget import AdaptiveMemoryController, BatchPlan
from highnoon.training.optimizers import SophiaG
from highnoon.training.runtime_control import should_stop, wait_if_paused

# Unified Smart Tuner
from highnoon.training.unified_smart_tuner import UnifiedSmartTuner, UnifiedSmartTunerConfig

# --- END: DEFINITIVE FIX ---

# --- Logger Setup ---
logger = logging.getLogger(__name__)
NUM_N4SID_MATRICES = 4
DEFAULT_CONTROLLER_METRICS = (
    "loss",
    "gradient_norm",
    "memory_bandwidth",
    "activation_sparsity",
    "moe_load_balancing_loss",
    "cpu_temperature",
)

_TRAIN_STEP_TENSOR_ARG_ORDER = (
    "context_tokens",
    "target_tokens",
    "batch_size_val",
    "seq_len_val",
    "input_dim_val",
    "label_dim_val",
    "model_weights",
    "optimizer_state",
    "ewc_fisher",
    "ewc_optimal_weights",
    "ewc_lambda",
    "k_val",
    "global_step",
    "n4sid_order",
    "block_types",
    "block_weight_counts",
    "block_descriptors",
    "initial_float_states",
    "initial_int_states",
    "n4sid_u",
    "n4sid_y",
    "lr",
    "beta_1",
    "rho",
    "epsilon",
    "beta_2",
    "loss_mask",
    # Quantum Training Config (Phase T1-T6)
    "enable_qng",
    "qng_damping",
    "qng_ema_decay",
    "enable_barren_plateau",
    "barren_plateau_threshold",
    "barren_plateau_lr_scale",
)

_TRAIN_STEP_MODULE = train_step_loader.train_step_module()
_TRAIN_STEP_CALLABLE = None
_TRAIN_STEP_STATE_CACHE: dict[object, tuple[tf.Variable, tf.Variable]] = {}
if _TRAIN_STEP_MODULE is not None:
    _TRAIN_STEP_CALLABLE = getattr(_TRAIN_STEP_MODULE, "train_step", None) or getattr(
        _TRAIN_STEP_MODULE, "TrainStep", None
    )
    if _TRAIN_STEP_CALLABLE is None:
        logger.warning(
            "[TRAIN STEP] Loaded custom op but no callable named train_step/TrainStep was found."
        )
else:
    logger.warning("[TRAIN STEP] TrainStep custom op is unavailable; fused training path disabled.")


def _call_train_step_with_eager_bridge(
    tensor_arg_items: list[tuple[str, Any]],
    attr_kwargs: dict[str, Any],
    output_structure: Any,
    output_dtypes: list[tf.dtypes.DType],
):
    """Run the fused TrainStep op eagerly even when the surrounding code is traced."""

    if tf.executing_eagerly():
        call_kwargs = dict(tensor_arg_items)
        call_kwargs.update(attr_kwargs)
        return _TRAIN_STEP_CALLABLE(**call_kwargs)

    def _materialize(element: Any) -> tf.Tensor:
        if isinstance(element, tf.Variable):
            return element.read_value()
        return tf.convert_to_tensor(element)

    arg_specs: list[tuple[str, Any, int]] = []
    flat_inputs: list[tf.Tensor] = []

    for name, value in tensor_arg_items:
        structure = tf.nest.map_structure(lambda _: None, value)
        flat_values = [_materialize(elem) for elem in tf.nest.flatten(value)]
        arg_specs.append((name, structure, len(flat_values)))
        flat_inputs.extend(flat_values)

    if not flat_inputs:
        raise RuntimeError("TrainStep wrapper expected tensor inputs but received none.")

    def _py_func(*flat_args: tf.Tensor):
        iterator = iter(flat_args)
        call_kwargs: dict[str, Any] = {}
        for name, structure, count in arg_specs:
            values = [next(iterator) for _ in range(count)]
            call_kwargs[name] = tf.nest.pack_sequence_as(structure, values)
        call_kwargs.update(attr_kwargs)
        outputs = _TRAIN_STEP_CALLABLE(**call_kwargs)
        return tf.nest.flatten(outputs)

    flat_outputs = tf.py_function(
        func=_py_func,
        inp=flat_inputs,
        Tout=output_dtypes,
        name="train_step_eager_wrapper",
    )

    return tf.nest.pack_sequence_as(output_structure, flat_outputs)


def _train_step_diagnostics_blob(max_symbols: int = 10) -> str:
    diagnostics = train_step_loader.train_step_diagnostics()
    diagnostics = dict(diagnostics) if diagnostics is not None else {}
    symbols = diagnostics.get("available_symbols")
    if isinstance(symbols, list):
        diagnostics["available_symbol_count"] = len(symbols)
        if max_symbols >= 0 and len(symbols) > max_symbols:
            diagnostics["available_symbols"] = symbols[:max_symbols]
        diagnostics.setdefault("available_symbols", symbols)
    try:
        return json.dumps(diagnostics, indent=2, sort_keys=True, default=str)
    except TypeError:
        safe_diag = {k: str(v) for k, v in diagnostics.items()}
        return json.dumps(safe_diag, indent=2, sort_keys=True)


@dataclass
class PlanDatasetUpdate:
    dist_train: tf.distribute.DistributedDataset | None
    dist_val: tf.distribute.DistributedDataset | None
    val_dataset: tf.data.Dataset | None
    num_train_examples: int
    metadata: dict[str, object]


@dataclass
class PlanUpdateHooks:
    controller: AdaptiveMemoryController
    current_plan: BatchPlan
    max_microbatch: int
    max_seq_len: int
    dataset_refresh: Callable[[BatchPlan], PlanDatasetUpdate]


def _profiler_mark(profiler: MemoryProfiler | None, label: str, **extra: object) -> None:
    if profiler is None:
        return
    payload = {k: v for k, v in extra.items() if v is not None}
    profiler.mark(label, extra=payload or None)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _embedding_variable(model: HSMN) -> tf.Variable | None:
    def _first_matrix(layer: tf.Module | None) -> tf.Variable | None:
        if layer is None:
            return None
        candidates = list(getattr(layer, "weights", []) or getattr(layer, "variables", []) or [])
        for candidate in candidates:
            if candidate is None:
                continue
            base_candidate = candidate
            if isinstance(candidate, KerasVariable):
                base_candidate = getattr(candidate, "value", candidate)
            if (
                isinstance(base_candidate, tf.Variable)
                and getattr(base_candidate.shape, "rank", None) == 2
            ):
                return candidate
        return None

    reasoning = getattr(model, "reasoning", None)
    if reasoning is not None:
        token_embedding = getattr(reasoning, "token_embedding", None)
        token_candidate = _first_matrix(token_embedding)
        if token_candidate is not None:
            return token_candidate

    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        embedding_layer = getattr(encoder, "embedding", None)
        encoder_candidate = _first_matrix(embedding_layer)
        if encoder_candidate is not None:
            return encoder_candidate

    return None


def _reasoning_block_weights(model: HSMN) -> list[tf.Variable]:
    reasoning = getattr(model, "reasoning", None)
    if reasoning is None:
        return []
    fused_block_weights = list(getattr(reasoning, "_fused_block_weights", []) or [])

    def _dedup(sequence: list[tf.Variable]) -> list[tf.Variable]:
        ordered: list[tf.Variable] = []
        seen: set[int] = set()
        for var in sequence:
            if var is None:
                continue
            var_id = id(var)
            if var_id in seen:
                continue
            seen.add(var_id)
            ordered.append(var)
        return ordered

    if fused_block_weights:
        return _dedup(fused_block_weights)

    ordered: list[tf.Variable] = []
    for block in getattr(reasoning, "reasoning_blocks", []) or []:
        block_vars: list[tf.Variable] = []
        if hasattr(block, "get_weights_for_fused_op"):
            try:
                block_vars = list(block.get_weights_for_fused_op())
            except TypeError:
                block_vars = list(block.weights)
        else:
            block_vars = list(block.weights)
        ordered.extend(block_vars)
    return _dedup(ordered)


def _sanitize_block_descriptors(model: HSMN) -> None:
    """Ensure block descriptor strings are JSON-parseable by the fused op."""
    reasoning = getattr(model, "reasoning", None)
    if reasoning is None or not hasattr(reasoning, "block_descriptors"):
        return
    try:
        raw = reasoning.block_descriptors.numpy().tolist()
    except Exception:
        logger.warning("[TRAIN STEP] Could not read block_descriptors; skipping sanitization.")
        return
    cleaned = []
    touched = False
    for entry in raw:
        s = entry.decode("utf-8") if isinstance(entry, (bytes, bytearray)) else str(entry)
        try:
            json.loads(s)
            cleaned.append(s)
        except Exception:
            touched = True
            fallback = json.dumps(
                {"type": "Unknown", "metadata": {}, "stateful": False, "weight_count": 0}
            )
            cleaned.append(fallback)
    if touched:
        logger.warning(
            "[TRAIN STEP] Replaced malformed block_descriptors entries with safe fallbacks."
        )
    try:
        reasoning.block_descriptors = tf.constant(cleaned, dtype=tf.string)
    except Exception as exc:
        logger.warning("[TRAIN STEP] Failed to set sanitized block_descriptors: %s", exc)


def _validate_reasoning_weight_layout(model: HSMN, ordered_weights: list[tf.Variable]) -> None:
    reasoning = getattr(model, "reasoning", None)
    if reasoning is None:
        return
    block_counts = reasoning.block_weight_counts.numpy().tolist()
    block_types = reasoning.block_types.numpy().tolist()
    decoded_types = [
        bt.decode("utf-8") if isinstance(bt, (bytes, bytearray)) else str(bt) for bt in block_types
    ]
    total_reasoning = sum(block_counts)
    if total_reasoning > len(ordered_weights):
        raise ValueError(
            f"Reasoning stack expects {total_reasoning} weights but only {len(ordered_weights)} are available."
        )
    reasoning_blocks: list[tf.keras.layers.Layer] = list(
        getattr(reasoning, "reasoning_blocks", []) or []
    )
    tail = ordered_weights[-total_reasoning:] if total_reasoning else []
    offset = 0
    for idx, (block_type, count) in enumerate(zip(decoded_types, block_counts)):
        block_slice = tail[offset : offset + count]
        if len(block_slice) != count:
            raise ValueError(
                f"[TRAIN STEP] Weight layout mismatch for block '{block_type}': "
                f"expected {count} tensors but found {len(block_slice)}."
            )
        if block_type == "TimeCrystalSequenceBlock" and count >= 7:
            block_obj = reasoning_blocks[idx] if idx < len(reasoning_blocks) else None
            cell = getattr(block_obj, "cell", None)

            b3_tensor = getattr(cell, "b3", None)
            epsilon_tensor = getattr(cell, "evolution_time_bias", None)

            # Fall back to the ordered slice if the direct attribute lookup fails.
            if b3_tensor is None and len(block_slice) > 5:
                b3_tensor = block_slice[5]
            if epsilon_tensor is None and len(block_slice) > 6:
                epsilon_tensor = block_slice[6]

            if b3_tensor is None or epsilon_tensor is None:
                raise ValueError(
                    "[TRAIN STEP] Could not locate TimeCrystalSequenceBlock control scalars for validation."
                )

            def _ensure_scalar(tensor: tf.Variable, label: str) -> None:
                if int(tf.size(tensor)) != 1:
                    raise ValueError(label)

            _ensure_scalar(
                b3_tensor,
                "[TRAIN STEP] TimeCrystalSequenceBlock b3 parameter must be a scalar tensor.",
            )
            _ensure_scalar(
                epsilon_tensor,
                "[TRAIN STEP] TimeCrystalSequenceBlock epsilon parameter must be a scalar tensor.",
            )
        offset += count


def _variable_ref_key(variable: tf.Variable) -> object:
    """
    Obtain a stable reference key for tf.Variable or new-style KerasVariable
    objects. Keras 3 variables wrap a backend-specific variable that does expose
    ref()/experimental_ref(), so we attempt to unwrap them before falling back
    to an object id.
    """
    candidates: list[object] = [variable]
    if isinstance(variable, KerasVariable):
        for attr in ("value", "variable", "base_variable"):
            inner = getattr(variable, attr, None)
            if inner is not None:
                candidates.append(inner)
    for candidate in candidates:
        if candidate is None:
            continue
        ref_fn = getattr(candidate, "ref", None)
        if callable(ref_fn):
            try:
                return ref_fn()
            except TypeError:
                pass
        exp_ref_fn = getattr(candidate, "experimental_ref", None)
        if callable(exp_ref_fn):
            try:
                return exp_ref_fn()
            except TypeError:
                pass
    return id(variable)


def _prepare_trainable_lists(
    model: HSMN, optimizer: tf.keras.optimizers.Optimizer
) -> tuple[list[tf.Variable], list[tf.Variable]]:
    reasoning_weights = _reasoning_block_weights(model)
    base_weights = list(model.trainable_variables)
    base_ids = {id(var) for var in base_weights}
    missing_reasoning = [var for var in reasoning_weights if id(var) not in base_ids]
    if missing_reasoning:
        base_weights = base_weights + missing_reasoning

    # Filter out non-float32 variables (e.g., int32 counters) that shouldn't be trained
    float32_weights = []
    for var in base_weights:
        if var.dtype == tf.float32:
            float32_weights.append(var)
        else:
            logger.warning(
                "[TRAIN STEP] Excluding variable %s with dtype %s from training (only float32 supported)",
                var.name,
                var.dtype,
            )
    base_weights = float32_weights

    reasoning_ids = {id(var) for var in reasoning_weights}
    non_reasoning_weights: list[tf.Variable] = []
    seen_non_reasoning: set[int] = set()
    for var in base_weights:
        var_id = id(var)
        if var_id in reasoning_ids or var_id in seen_non_reasoning:
            continue
        seen_non_reasoning.add(var_id)
        non_reasoning_weights.append(var)
    embedding_var = _embedding_variable(model)
    if embedding_var is not None:
        non_reasoning_weights = [embedding_var] + [
            var for var in non_reasoning_weights if var is not embedding_var
        ]

    ordered_weights: list[tf.Variable] = non_reasoning_weights + reasoning_weights

    momentum_by_ref: dict[object, tf.Variable] = {}
    hessian_by_ref: dict[object, tf.Variable] = {}

    if hasattr(optimizer, "momentums"):
        for var, state in zip(base_weights, optimizer.momentums):
            momentum_by_ref[_variable_ref_key(var)] = state
    if hasattr(optimizer, "hessians"):
        for var, state in zip(base_weights, optimizer.hessians):
            hessian_by_ref[_variable_ref_key(var)] = state

    optimizer_state_list: list[tf.Variable] = []

    # Cache ensures that newly created slot variables persist across train steps.
    # Keyed by variable ref so custom ops always see a stable ordering.
    global _TRAIN_STEP_STATE_CACHE
    try:
        cache = _TRAIN_STEP_STATE_CACHE
    except NameError:
        cache = {}
        _TRAIN_STEP_STATE_CACHE = cache

    def _maybe_create_slot(
        ref: object,
        existing: tf.Variable | None,
        cached: tuple[tf.Variable, tf.Variable] | None,
        index: int,
        base_var: tf.Variable,
        suffix: str,
    ) -> tf.Variable:
        if existing is not None:
            return existing
        if cached is not None and cached[index] is not None:
            return cached[index]
        with tf.device("/CPU:0"):
            # Ensure optimizer state is always float32, regardless of base_var dtype
            initial_value = tf.zeros_like(base_var)
            if initial_value.dtype != tf.float32:
                initial_value = tf.cast(initial_value, tf.float32)
            return tf.Variable(
                initial_value,
                trainable=False,
                name=f"{base_var.name.split(':')[0]}_{suffix}",
                dtype=tf.float32,
            )

    for var in ordered_weights:
        ref = _variable_ref_key(var)
        cached_pair = cache.get(ref)

        momentum_var = _maybe_create_slot(
            ref,
            momentum_by_ref.get(ref),
            cached_pair,
            0,
            var,
            "sophia_m",
        )
        hessian_var = _maybe_create_slot(
            ref,
            hessian_by_ref.get(ref),
            cached_pair,
            1,
            var,
            "sophia_h",
        )
        cache[ref] = (momentum_var, hessian_var)
        optimizer_state_list.extend([momentum_var, hessian_var])

    return ordered_weights, optimizer_state_list


def _controller_metric_inventory(evolution_time_vars: dict[str, tf.Variable]) -> list[str]:
    metrics: list[str] = list(hn_config.CONTROL_METRIC_EMA_KEYS or [])
    if not metrics:
        metrics = list(DEFAULT_CONTROLLER_METRICS)
    else:
        metrics.extend(DEFAULT_CONTROLLER_METRICS)
    metrics.extend(sorted(evolution_time_vars.keys()))
    seen: set[str] = set()
    ordered: list[str] = []
    for name in metrics:
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _resolve_control_config_dir(trial_dir: str | None) -> str:
    override = os.getenv("HSMN_CONTROL_CONFIG_DIR")
    target = trial_dir or override or "."
    return ensure_path(target)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that supports NumPy scalars/arrays on NumPy>=2.0."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):  # bool_ no longer aliases bool once serialized
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_memory_embeddings(
    model: HSMN,
    dataset: tf.data.Dataset,
    output_dir: str,
    dataset_name: str,
    max_samples: int = 1000,
) -> None:
    """
    Extracts and saves memory embeddings from the model for a given dataset.
    This function is designed to be called after training on a specific task.
    """
    logger.info(f"--- Starting Memory Embedding Extraction for {dataset_name} ---")
    output_path = ensure_path(output_dir)
    embedding_file_path = os.path.join(output_path, f"{dataset_name}_memory_embeddings.npz")

    all_embeddings = []
    all_levels = []
    sample_count = 0

    @tf.function
    def get_memory(context_tokens):
        return model.compute_memory_levels(context_tokens, training=False)

    num_batches_to_take = (
        max(1, max_samples // model.optimizer.batch_size)
        if hasattr(model.optimizer, "batch_size") and model.optimizer.batch_size > 0
        else 1
    )

    for context_tokens, _ in dataset.take(num_batches_to_take):
        if sample_count >= max_samples:
            break

        memory_ta, num_levels = get_memory(context_tokens)
        num_levels_py = num_levels.numpy()

        for level_idx in range(num_levels_py):
            level_embeddings = memory_ta.read(level_idx).numpy()
            for i in range(level_embeddings.shape[0]):
                if sample_count >= max_samples:
                    break
                all_embeddings.extend(level_embeddings[i])
                all_levels.extend([level_idx] * level_embeddings.shape[1])
                sample_count += 1

    if all_embeddings:
        np.savez_compressed(
            embedding_file_path, embeddings=np.array(all_embeddings), levels=np.array(all_levels)
        )
        logger.info(f"Saved {len(all_embeddings)} memory embeddings to {embedding_file_path}")
    else:
        logger.warning("No embeddings were extracted to save.")


def find_control_time_vars(model: tf.keras.Model) -> dict[str, tf.Variable | KerasVariable]:
    """
    Finds controllable variables (like 'evolution_time') using a robust, recursive,
    and type-agnostic deep search.

    This function addresses two primary issues:
    1.  **Type Incompatibility:** It correctly identifies both `tf.Variable` and the
        newer `keras.src.backend.Variable` objects used in Keras 3.
    2.  **Deep Search:** It recursively traverses the entire model hierarchy through
        the `layers` attribute, ensuring no nested variables are missed.

    Args:
        model: The Keras model to search within.

    Returns:
        A dictionary mapping unique, clean variable names to their variable objects.
    """
    logger.info("--- [find_control_time_vars] Starting search for control variables. ---")
    found_vars: OrderedDict[str, tf.Variable | KerasVariable] = OrderedDict()
    non_scalar_summaries: OrderedDict[str, tf.Tensor] = OrderedDict()
    seen_paths = set()
    # Include quantum-layer control tags so the logging pipeline can surface VQC parameters.
    CONTROL_KEYS = (
        "evolution_time",
        "evolution_time_vqc",
        "vqc_thetas",
        "inverse_temperature",
        "epsilon_param",
        "evolution_time_param",
    )

    def _normalized_name(var) -> str:
        candidate = getattr(var, "path", None) or getattr(var, "name", "")
        candidate = candidate.split(":")[0]
        if candidate.startswith("model/"):
            candidate = candidate[6:]
        return candidate or f"control_var_{len(found_vars)}"

    def _register(var):
        if not isinstance(var, (tf.Variable, KerasVariable)):
            return
        identifier = getattr(var, "path", None) or getattr(var, "name", "")
        if not identifier:
            return
        identifier_lower = identifier.lower()
        if "last_stable" in identifier_lower:
            return
        if not any(token in identifier_lower for token in CONTROL_KEYS):
            return
        try:
            rank = var.shape.rank
        except AttributeError:
            rank = None
        if rank not in (None, 0):
            name = _normalized_name(var)
            try:
                summary_value = tf.reduce_mean(tf.abs(var))
                non_scalar_summaries[name] = summary_value
            except Exception as exc:
                logger.debug(
                    "Skipping summary for non-scalar control var '%s': %s",
                    name,
                    exc,
                )
            # Do not return; still register the variable itself.
        if identifier in seen_paths:
            return
        name = _normalized_name(var)
        original = name
        suffix = 1
        while name in found_vars and found_vars[name] is not var:
            name = f"{original}__{suffix}"
            suffix += 1
        found_vars[name] = var
        seen_paths.add(identifier)

    visited_layers = set()

    def find_vars_recursive(layer):
        if layer is None or id(layer) in visited_layers:
            return
        visited_layers.add(id(layer))

        if hasattr(layer, "control_vars") and layer.control_vars:
            for tag, var_list in layer.control_vars.items():
                if tag in CONTROL_KEYS:
                    for var in var_list:
                        _register(var)

        for var in getattr(layer, "variables", []):
            _register(var)

        sub_layers = []
        if hasattr(layer, "_flatten_layers"):
            try:
                sub_layers = layer._flatten_layers(include_self=False, recursive=True)
            except TypeError:
                sub_layers = layer._flatten_layers(include_self=False)
        elif hasattr(layer, "layers"):
            sub_layers = layer.layers

        for sub_layer in sub_layers:
            if sub_layer is not layer:
                find_vars_recursive(sub_layer)

    find_vars_recursive(model)

    for var in getattr(model, "variables", []):
        _register(var)

    logger.info(
        f"--- [find_control_time_vars] Found {len(found_vars)} unique control variable(s). ---"
    )
    if non_scalar_summaries:
        logger.info(
            "--- [find_control_time_vars] Reporting summary statistics for non-scalar control parameters ---"
        )
        for name, summary_tensor in non_scalar_summaries.items():
            try:
                summary_val = float(summary_tensor.numpy())
                logger.info("  • %s | mean(|value|) = %.6f", name, summary_val)
            except Exception as exc:
                logger.debug(
                    "Could not summarize non-scalar control var '%s': %s",
                    name,
                    exc,
                )

    # Log the found variables with a robust type check.
    for name, var in found_vars.items():
        # Check if it's a variable-like object before accessing attributes.
        if hasattr(var, "shape") and hasattr(var, "dtype") and hasattr(var, "numpy"):
            # Keras 3 `var.dtype` can be a string ('float32'), while TensorFlow's is an
            # object (`tf.float32`) with a `.name` attribute. `getattr` handles both.
            dtype_name = getattr(var.dtype, "name", str(var.dtype))
            try:
                # Add the initial value to the log for immediate inspection.
                value_str = np.array2string(var.numpy(), precision=6, suppress_small=True)
                logger.info(
                    f"  ✅ FOUND: {name} | Value: {value_str} | Shape: {var.shape} | DType: {dtype_name}"
                )
            except Exception as e:
                logger.error(f"  - FAILED to read value for '{name}': {e}")
        else:
            logger.error(
                f"  - SKIPPING LOG: Expected a variable-like object for key '{name}', but got type {type(var)}."
            )
    return dict(found_vars)


def train_on_dataset_dist(
    strategy: tf.distribute.Strategy,
    model: HSMN,
    dataset_name: str,
    dist_train_dataset: tf.distribute.DistributedDataset,
    dist_val_dataset: tf.distribute.DistributedDataset | None,
    val_dataset: tf.data.Dataset | None,
    num_train_examples: int,
    epochs: int,
    batch_size: int,
    accum_steps: int,
    model_path: str,
    checkpoint_dir: str,
    cumulative_fisher: dict[str, tf.Tensor],
    cumulative_optimal_weights: dict[str, tf.Tensor],
    tokenizer,
    trigger_sysid_reload: bool,
    online_sysid_freq=850,
    use_adapters: bool = False,
    save_embeddings_for_analysis: bool = False,
    trial_dir: str = None,
    skip_hessian_update: bool = False,
    plan_hooks: PlanUpdateHooks | None = None,
    profiler: MemoryProfiler | None = None,
    n4sid_order: int = 5,  # New parameter for N4SID system order
) -> tuple[HSMN, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """
    Trains the model on a dataset using a distributed strategy.
    """
    if trial_dir is None:
        trial_dir = os.getenv("HSMN_TRIAL_DIR")
    log_file_path = os.path.join(trial_dir, "training_log.log") if trial_dir else "training_log.log"
    _profiler_mark(
        profiler,
        "train_loop_enter",
        dataset=dataset_name,
        epochs=epochs,
        train_examples=num_train_examples,
    )

    with strategy.scope():
        optimizer = model.optimizer  # Get the optimizer from the compiled model
        if skip_hessian_update and isinstance(optimizer, SophiaG):
            logger.info("[EWC] Hessian updates disabled for this run (skip_hessian_update=True).")

        # Initialize GaLore gradient compressor if enabled
        # NOTE: GaLore compression is applied during gradient accumulation phases.
        # The fused C++ TrainStep op handles optimizer updates internally, so
        # GaLore compression is applied before gradients are passed to the op.
        galore_compressor: TensorGaLoreCompressor | None = None
        if USE_TENSOR_GALORE:
            galore_compressor = TensorGaLoreCompressor(
                rank=GALORE_RANK,
                update_proj_gap=GALORE_UPDATE_PROJ_GAP,
                scale=GALORE_SCALE,
                enabled=True,
            )
            logger.info(
                "[GALORE] Initialized TensorGaLoreCompressor: rank=%d, update_gap=%d, scale=%.2f",
                GALORE_RANK,
                GALORE_UPDATE_PROJ_GAP,
                GALORE_SCALE,
            )

        # Initialize Unified Smart Tuner for coordinated training parameter control
        # The smart tuner replaces independent operation of QALRC, BarrenPlateauMonitor,
        # GaLore, and Meta-Controller with a single orchestrated system.
        smart_tuner: UnifiedSmartTuner | None = None
        use_smart_tuner = (
            hn_config.USE_UNIFIED_SMART_TUNER
            if hasattr(hn_config, "USE_UNIFIED_SMART_TUNER")
            else True
        )
        if use_smart_tuner:
            # Estimate total steps for the tuner
            total_training_steps = epochs * max(1, num_train_examples // max(1, batch_size))
            smart_tuner_config = UnifiedSmartTunerConfig(
                enabled=True,
                memory_enabled=True,
                memory_path=ensure_path(os.path.join(trial_dir or ".", "tuner_memory")),
                coordination_mode=getattr(hn_config, "SMART_TUNER_MODE", "balanced"),
                lr_initial=(
                    float(optimizer.learning_rate.numpy())
                    if hasattr(optimizer, "learning_rate")
                    else 3e-4
                ),
                lr_min=1e-7,
                lr_max=1e-2,
                galore_rank=GALORE_RANK,
                barren_plateau_threshold=BARREN_PLATEAU_THRESHOLD,
                meta_controller_frequency=META_CONTROLLER_FREQUENCY,
                max_grad_norm=1.0,
                warmup_steps=getattr(hn_config, "SMART_TUNER_WARMUP_STEPS", 1000),
                exploration_steps=getattr(hn_config, "SMART_TUNER_EXPLORATION_STEPS", 10000),
            )
            smart_tuner = UnifiedSmartTuner(
                model=model,
                optimizer=optimizer,
                config=smart_tuner_config,
                total_steps=total_training_steps,
            )
            logger.info(
                "[SmartTuner] Initialized UnifiedSmartTuner: mode=%s, lr_initial=%.2e, total_steps=%d",
                smart_tuner_config.coordination_mode,
                smart_tuner_config.lr_initial,
                total_training_steps,
            )

        model_weights_list, optimizer_state_list = _prepare_trainable_lists(model, optimizer)
        try:
            _validate_reasoning_weight_layout(model, model_weights_list)
        except ValueError as layout_err:
            logger.error("[TRAIN STEP] %s", layout_err)
            raise
        global_step = tf.Variable(0, dtype=tf.int64, name="global_step", trainable=False)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

    if trial_dir:
        ensure_path(trial_dir)

    logging.getLogger()
    file_handler = logging.FileHandler(log_file_path, mode="a")
    json_log_path = (
        os.path.join(trial_dir, "metrics_log.jsonl") if trial_dir else "metrics_log.jsonl"
    )
    control_log_path = (
        os.path.join(trial_dir, "control_updates_log.jsonl")
        if trial_dir
        else "control_updates_log.jsonl"
    )
    control_bridge: EvolutionTimeControlBridge | None = None
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    # root_logger.addHandler(file_handler) # Avoid duplicate handlers if already added
    initial_rss_mb = psutil.Process().memory_info().rss / 1024**2
    logger.info(f"[MEMORY] Before training {dataset_name}: {initial_rss_mb:.2f} MB")
    _profiler_mark(
        profiler,
        "train_loop_pre_memory",
        dataset=dataset_name,
        rss_mb=initial_rss_mb,
    )
    current_plan = plan_hooks.current_plan if plan_hooks else None
    current_microbatch = current_plan.microbatch_size if current_plan else batch_size
    max(1, current_plan.accum_steps if current_plan else accum_steps)
    global_batch_size = max(1, current_microbatch) * strategy.num_replicas_in_sync

    # Initialize HPO reporter if in HPO mode
    hpo_reporter = HPOReporter()
    if hpo_reporter.enabled:
        logger.info(f"[HPO] HPO mode enabled for trial {hpo_reporter.trial_id}")
        # Apply trial config if available
        trial_config = load_trial_config()
        if trial_config:
            logger.info(f"[HPO] Applying trial configuration: {trial_config}")
            # Note: Trial config application happens at model creation time,
            # but we log it here for visibility

    with strategy.scope():

        def _warmup_model_for_control_scan():
            warmup_batch = max(1, min(current_microbatch or 1, 2))
            warmup_seq_len = getattr(model, "max_seq_len", 256)
            dummy_context = tf.zeros([warmup_batch, warmup_seq_len], dtype=tf.int32)
            dummy_target = tf.zeros([warmup_batch, warmup_seq_len], dtype=tf.int32)
            _ = model((dummy_context, dummy_target), training=False)

        if not getattr(model, "built", False):
            logger.info(
                "[VARS] Model not yet built; running warmup forward pass for control-var discovery."
            )
            try:
                _warmup_model_for_control_scan()
                logger.info("[VARS] Warmup build complete.")
            except Exception as exc:
                logger.warning(
                    "[VARS] Warmup build before control-var scan failed: %s",
                    exc,
                )

        # Ensure block descriptors are parsable before the fused op consumes them.
        _sanitize_block_descriptors(model)

        logger.info("[VARS] Searching for controllable 'evolution_time' variables...")
        evolution_time_vars = find_control_time_vars(model)
        if not evolution_time_vars:
            logger.info(
                "[VARS] Control-var scan returned empty; forcing additional warmup and retry."
            )
            try:
                _warmup_model_for_control_scan()
            except Exception as exc:
                logger.warning(
                    "[VARS] Retry warmup for control-var scan failed: %s",
                    exc,
                )
            else:
                evolution_time_vars = find_control_time_vars(model)

        control_config_dir = _resolve_control_config_dir(trial_dir)

        if not evolution_time_vars:
            logger.warning(
                "No controllable 'evolution_time' tf.Variables found. The Hamiltonian Meta-Controller will be disabled. "
                "This is expected if the current block_pattern does not include 'timecrystal' blocks. "
                "If you expected this to work, verify the REASONING_BLOCK_PATTERN."
            )
            evolution_time_vars = {}  # Use an empty dict to safely disable the controller
        else:
            control_names = list(evolution_time_vars.keys())
            verbose_controls = _env_flag("HSMN_META_CTL_VERBOSE", False)
            preview_cap = max(1, int(os.getenv("HSMN_META_CTL_PREVIEW", "8")))
            logger.info(
                "[META-CTL] Found %d control vars%s",
                len(control_names),
                (
                    ""
                    if verbose_controls or len(control_names) <= preview_cap
                    else f" (previewing first {preview_cap})"
                ),
            )
            names_to_log = control_names if verbose_controls else control_names[:preview_cap]
            for k in names_to_log:
                v = evolution_time_vars[k]
                try:
                    logger.info(f"[META-CTL]   -> {k} | {v.name} = {float(v.numpy()):.6f}")
                except Exception as e:
                    logger.warning(f"[META-CTL] Could not read initial value for {v.name}: {e}")
            if not verbose_controls and len(control_names) > preview_cap:
                logger.info(
                    "[META-CTL] ... %d additional controls not shown; set HSMN_META_CTL_VERBOSE=1 to log all initial values.",
                    len(control_names) - preview_cap,
                )

            controller_metrics = _controller_metric_inventory(evolution_time_vars)
            ensure_control_configs(
                metrics=controller_metrics,
                control_inputs=evolution_time_vars.keys(),
                directory=control_config_dir,
                force_pid=_env_flag("HSMN_FORCE_PID_REFRESH", False),
                force_state_space=_env_flag("HSMN_FORCE_STATE_SPACE_REFRESH", False),
            )
            logger.info("[META-CTL] Control config root: %s", control_config_dir)

        meta_controller_callback = HamiltonianMetaControllerCallback(
            frequency=hn_config.META_CONTROLLER_FREQUENCY,
            trigger_sysid_reload=trigger_sysid_reload,
        )
        control_bridge = EvolutionTimeControlBridge(
            evolution_time_vars,
            control_log_path,
            control_config_path=control_config_dir,
        )

    def _compute_validation_loss(
        logits: tf.Tensor, target_tokens: tf.Tensor, loss_mask: tf.Tensor | None
    ) -> tf.Tensor:
        labels = target_tokens[:, 1:]
        logits_for_loss = logits[:, : tf.shape(labels)[1], :]
        per_token_loss = loss_fn(labels, logits_for_loss)
        if loss_mask is not None:
            mask = loss_mask[:, : tf.shape(labels)[1]]
        else:
            pad_id = getattr(model.tokenizer, "pad_token_id", 0)
            mask = tf.cast(labels != pad_id, tf.float32)
        denom = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(per_token_loss * mask) / denom

    @tf.function(reduce_retracing=True)
    def _val_step(inputs):
        if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            context_tokens, target_tokens, loss_mask = inputs
        else:
            context_tokens, target_tokens = inputs
            loss_mask = None
        logits, _, _ = model((context_tokens, target_tokens), training=False)
        return _compute_validation_loss(logits, target_tokens, loss_mask)

    @tf.function(reduce_retracing=True)
    def distributed_train_step(dist_inputs, n4sid_u_tensor, n4sid_y_tensor, n4sid_order_tensor):
        """
        Performs a single distributed training step, including the forward pass,
        loss calculation, gradient computation, and optimizer update, all
        encapsulated within a single C++ TensorFlow operation (`TrainStepOp`).
        This function is designed to be compiled into a static graph by
        `tf.function` for high performance.

        This step can also optionally perform system identification (N4SID) if
        the appropriate `n4sid_*` tensors are provided with data.

        Args:
            dist_inputs: A tuple of (context_tokens, target_tokens) distributed
                         across replicas.
            n4sid_u_tensor: A tensor containing the input data (u) for N4SID.
                            Should be an empty tensor if N4SID is not to be run.
            n4sid_y_tensor: A tensor containing the output data (y) for N4SID.
                            Should be an empty tensor if N4SID is not to be run.
            n4sid_order_tensor: A scalar tensor specifying the desired order for
                                the N4SID algorithm. A value of 0 disables the
                                N4SID calculation in the C++ op.

        Returns:
            A tuple containing various metrics and state information from the
            training step, such as losses, gradients, and updated states.
        """
        if isinstance(dist_inputs, (tuple, list)) and len(dist_inputs) == 3:
            context_tokens, target_tokens, loss_mask = dist_inputs
        else:
            context_tokens, target_tokens = dist_inputs
            loss_mask = tf.ones_like(tf.cast(target_tokens, tf.float32))

        # Gather EWC state - ensure float32 dtype for all tensors
        ewc_fisher_list = [
            cumulative_fisher.get(var.name, tf.cast(tf.zeros_like(var), tf.float32))
            for var in model_weights_list
        ]
        ewc_optimal_weights_list = [
            cumulative_optimal_weights.get(var.name, tf.cast(tf.zeros_like(var), tf.float32))
            for var in model_weights_list
        ]
        # EWC is deprecated - only apply penalty if explicitly enabled
        ewc_lambda_value = EWC_LAMBDA if ENABLE_EWC else 0.0
        ewc_lambda_tensor = tf.constant(ewc_lambda_value, dtype=tf.float32)

        # Gather k_val from optimizer
        k_val_tensor = tf.constant(optimizer.k, dtype=tf.int32)

        # Gather block_types and block_weight_counts from ReasoningModule
        block_types_tensor = model.reasoning.block_types
        block_weight_counts_tensor = model.reasoning.block_weight_counts
        block_descriptors_tensor = model.reasoning.block_descriptors

        # Gather initial_float_states and initial_int_states from ReasoningModule
        batch_size_replica = tf.shape(context_tokens)[0]
        initial_float_states_list, initial_int_states_list = model.reasoning.get_initial_states(
            batch_size_replica
        )

        # The n4sid_u_tensor and n4sid_y_tensor are now passed in as arguments.
        # The n4sid_order is also passed in as n4sid_order_tensor.

        # Call the fused C++ TrainStepOp
        # Ensure tensors reside on CPU for the custom op (supports CPU fallback)
        with tf.device("/CPU:0"):
            context_tokens_cpu = tf.cast(tf.identity(context_tokens), tf.int32)
            target_tokens_cpu = tf.cast(tf.identity(target_tokens), tf.int32)
            loss_mask_cpu = tf.cast(tf.identity(loss_mask), tf.float32)

        # Get actual dimensions
        input_dim_val = tf.cast(
            model.encoder.embedding_dim, tf.int64
        )  # Assuming input_dim is the embedding dimension
        label_dim_val = tf.cast(model.tokenizer.vocab_size, tf.int64)

        # Get N4SID input/output dimensions from the provided tensors
        tf.cast(tf.shape(n4sid_u_tensor)[1], tf.int32)
        tf.cast(tf.shape(n4sid_y_tensor)[1], tf.int32)
        num_n4sid_inputs_attr = n4sid_u_tensor.shape[1]
        num_n4sid_outputs_attr = n4sid_y_tensor.shape[1]
        num_n4sid_inputs_attr = (
            int(num_n4sid_inputs_attr) if num_n4sid_inputs_attr is not None else 0
        )
        num_n4sid_outputs_attr = (
            int(num_n4sid_outputs_attr) if num_n4sid_outputs_attr is not None else 0
        )

        if _TRAIN_STEP_CALLABLE is None:
            diagnostics_blob = _train_step_diagnostics_blob()
            logger.error(
                "[TRAIN STEP] Custom op is unavailable; diagnostics follow:\n%s",
                diagnostics_blob,
            )
            raise RuntimeError(
                "TrainStep custom op is not available. Rebuild/load the operator.\n"
                f"Diagnostics: {diagnostics_blob}"
            )

        logger.debug(
            "[TrainStep] inputs -> weights=%d optimizer_state=%d fisher=%d optimal=%d float_states=%d int_states=%d",
            len(model_weights_list),
            len(optimizer_state_list),
            len(ewc_fisher_list),
            len(ewc_optimal_weights_list),
            len(initial_float_states_list),
            len(initial_int_states_list),
        )

        # Comprehensive dtype validation and casting for TrainStep op
        # Ensure all weight/state tensors are float32
        model_weights_float32 = [
            tf.cast(w, tf.float32) if w.dtype != tf.float32 else w for w in model_weights_list
        ]
        optimizer_state_float32 = [
            tf.cast(s, tf.float32) if s.dtype != tf.float32 else s for s in optimizer_state_list
        ]
        ewc_fisher_float32 = [
            tf.cast(f, tf.float32) if f.dtype != tf.float32 else f for f in ewc_fisher_list
        ]
        ewc_optimal_weights_float32 = [
            tf.cast(o, tf.float32) if o.dtype != tf.float32 else o for o in ewc_optimal_weights_list
        ]

        tensor_kwargs = {
            "context_tokens": context_tokens_cpu,
            "target_tokens": target_tokens_cpu,
            "batch_size_val": tf.cast(batch_size_replica, tf.int64),
            "seq_len_val": tf.cast(tf.shape(context_tokens)[1], tf.int64),
            "input_dim_val": input_dim_val,
            "label_dim_val": label_dim_val,
            "model_weights": model_weights_float32,
            "optimizer_state": optimizer_state_float32,
            "ewc_fisher": ewc_fisher_float32,
            "ewc_optimal_weights": ewc_optimal_weights_float32,
            "ewc_lambda": ewc_lambda_tensor,
            "k_val": k_val_tensor,
            "global_step": global_step,
            "n4sid_order": n4sid_order_tensor,
            "block_types": block_types_tensor,
            "block_weight_counts": block_weight_counts_tensor,
            "block_descriptors": block_descriptors_tensor,
            "initial_float_states": initial_float_states_list,
            "initial_int_states": initial_int_states_list,
            "n4sid_u": n4sid_u_tensor,
            "n4sid_y": n4sid_y_tensor,
            "lr": tf.cast(optimizer.learning_rate, tf.float32),
            "beta_1": tf.cast(optimizer.beta_1, tf.float32),
            "rho": tf.cast(optimizer.rho, tf.float32),
            "epsilon": tf.cast(optimizer.epsilon, tf.float32),
            "beta_2": tf.cast(optimizer.beta_2, tf.float32),
            "loss_mask": loss_mask_cpu,
            # Quantum Training Config (Phase T1-T6)
            "enable_qng": tf.constant(hn_config.USE_QUANTUM_NATURAL_GRADIENT, dtype=tf.bool),
            "qng_damping": tf.constant(hn_config.QNG_DAMPING, dtype=tf.float32),
            "qng_ema_decay": tf.constant(0.99, dtype=tf.float32),  # Default EMA decay
            "enable_barren_plateau": tf.constant(hn_config.BARREN_PLATEAU_MONITOR, dtype=tf.bool),
            "barren_plateau_threshold": tf.constant(
                hn_config.BARREN_PLATEAU_THRESHOLD, dtype=tf.float32
            ),
            "barren_plateau_lr_scale": tf.constant(
                hn_config.BARREN_PLATEAU_RECOVERY_LR_SCALE, dtype=tf.float32
            ),
        }

        attr_kwargs = {
            "num_n4sid_inputs": num_n4sid_inputs_attr,
            "num_n4sid_outputs": num_n4sid_outputs_attr,
            "NumN4SIDMatrices": NUM_N4SID_MATRICES,
        }

        tensor_arg_items = [(name, tensor_kwargs[name]) for name in _TRAIN_STEP_TENSOR_ARG_ORDER]

        output_structure = (
            [None] * len(model_weights_float32),
            [None] * len(optimizer_state_float32),
            None,
            None,
            None,
            None,
            None,
            [None] * NUM_N4SID_MATRICES,
            [None] * len(initial_float_states_list),
            [None] * len(initial_int_states_list),
            [None] * len(initial_float_states_list),
            [None] * len(initial_int_states_list),
        )

        output_dtypes = (
            [var.dtype for var in model_weights_float32]
            + [tf.float32] * len(optimizer_state_float32)  # Force float32 for optimizer states
            + [tf.float32, tf.float32, tf.float32, tf.float32]
            + [tf.float32]
            + [tf.float32] * NUM_N4SID_MATRICES
            + [tensor.dtype for tensor in initial_float_states_list]
            + [tensor.dtype for tensor in initial_int_states_list]
            + [tensor.dtype for tensor in initial_float_states_list]
            + [tensor.dtype for tensor in initial_int_states_list]
        )

        (
            new_model_weights,
            new_optimizer_state,
            total_loss,
            std_loss,
            ewc_loss,
            gradient_norm,
            logits,
            n4sid_matrices,
            final_float_states_out,
            final_int_states_out,
            grad_initial_float_states_out,
            grad_initial_int_states_out,
        ) = _call_train_step_with_eager_bridge(
            tensor_arg_items,
            attr_kwargs,
            output_structure,
            list(output_dtypes),
        )

        if len(n4sid_matrices) >= 4:
            n4sid_A, n4sid_B, n4sid_C, n4sid_D = n4sid_matrices[:4]
        else:
            n4sid_A = n4sid_B = n4sid_C = n4sid_D = (
                tf.zeros_like(n4sid_matrices[0])
                if n4sid_matrices
                else tf.zeros([0, 0], dtype=tf.float32)
            )

        # Update model weights and optimizer state with the outputs from the C++ op
        for i, new_weight in enumerate(new_model_weights):
            new_weight.set_shape(model_weights_list[i].shape)
            model_weights_list[i].assign(new_weight)

        # Update optimizer state
        for i, new_opt_state in enumerate(new_optimizer_state):
            new_opt_state.set_shape(optimizer_state_list[i].shape)
            optimizer_state_list[i].assign(new_opt_state)

        # The C++ op now handles gradient application internally.
        # We return dummy grads and aux_metrics to match the expected signature.
        grads = [tf.zeros_like(v) for v in model_weights_list]  # Dummy grads
        aux_metrics = {}  # Dummy aux_metrics

        global_step.assign_add(1)  # Increment global step

        return (
            total_loss,
            std_loss,
            ewc_loss,
            grads,
            aux_metrics,
            gradient_norm,
            logits,
            final_float_states_out,
            final_int_states_out,
            grad_initial_float_states_out,
            grad_initial_int_states_out,
            n4sid_A,
            n4sid_B,
            n4sid_C,
            n4sid_D,
        )

    @tf.function(reduce_retracing=True)
    def distributed_val_step(dist_inputs):
        per_replica_loss = strategy.run(_val_step, args=(dist_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    float("inf")

    train_iterator = iter(dist_train_dataset)

    steps_per_epoch = max(1, num_train_examples // global_batch_size)
    max_batches_env = os.getenv("HSMN_MAX_TRAINING_BATCHES")
    effective_steps_per_epoch = steps_per_epoch
    if max_batches_env:
        try:
            parsed = max(1, int(max_batches_env))
            effective_steps_per_epoch = min(steps_per_epoch, parsed)
            logger.info(
                "[TRAINING] Limiting to %d batch(es) per epoch via HSMN_MAX_TRAINING_BATCHES=%s.",
                effective_steps_per_epoch,
                max_batches_env,
            )
        except ValueError:
            logger.warning(
                "[TRAINING] Ignoring invalid HSMN_MAX_TRAINING_BATCHES=%s.", max_batches_env
            )
    # --- END: FIX ---
    batch_checkpoint_interval = 0
    if profiler is not None and profiler.config.batch_event_frequency > 0:
        target_marks = max(1, profiler.config.batch_event_frequency)
        batch_checkpoint_interval = max(1, math.ceil(effective_steps_per_epoch / target_marks))

    for epoch in range(epochs):
        try:
            wait_if_paused()
            if should_stop():
                logger.info(
                    "[TRAINING] Stop requested before epoch %d on '%s'. Exiting training loop.",
                    epoch + 1,
                    dataset_name,
                )
                _profiler_mark(
                    profiler,
                    "train_stop_requested",
                    dataset=dataset_name,
                    epoch=epoch + 1,
                    phase="pre_epoch",
                )
                break

            logger.info(f"--- Starting Epoch {epoch + 1}/{epochs} on {dataset_name} ---")
            _profiler_mark(
                profiler,
                "train_epoch_start",
                dataset=dataset_name,
                epoch=epoch + 1,
                steps_per_epoch=effective_steps_per_epoch,
            )
            epoch_train_loss, num_train_batches = 0.0, 0

            stop_training = False
            for batch_idx in range(effective_steps_per_epoch):
                wait_if_paused()
                if should_stop():
                    logger.info(
                        "[TRAINING] Stop requested during epoch %d batch %d on '%s'.",
                        epoch + 1,
                        batch_idx + 1,
                        dataset_name,
                    )
                    stop_training = True
                    _profiler_mark(
                        profiler,
                        "train_stop_requested",
                        dataset=dataset_name,
                        epoch=epoch + 1,
                        batch=batch_idx + 1,
                        phase="mid_epoch",
                    )
                    break
                # --- START: NATIVE SYS-ID INTEGRATION ---
                # The Python-based system identification is now replaced by feeding
                # the required data directly into the C++ TrainStepOp.
                n4sid_u_tensor = tf.zeros([0, 0], dtype=tf.float32)
                n4sid_y_tensor = tf.zeros([0, 0], dtype=tf.float32)
                n4sid_order_val = 0
                trigger_sysid_reload_now = False

                if batch_idx > 0 and batch_idx % online_sysid_freq == 0:
                    logger.info(
                        f"[SYS-ID] Preparing data for Online System Identification (Batch {batch_idx+1})..."
                    )
                    id_log_path = (
                        os.path.join(trial_dir, "metrics_log.jsonl")
                        if trial_dir
                        else "metrics_log.jsonl"
                    )
                    metric_list_path = (
                        os.path.join(trial_dir, "metric_list.conf")
                        if trial_dir
                        else "metric_list.conf"
                    )
                    input_list_path = (
                        os.path.join(trial_dir, "input_list.conf")
                        if trial_dir
                        else "input_list.conf"
                    )

                    log_data = parse_log_file(id_log_path)
                    u, y, metric_keys, _ = prepare_data_from_logs(
                        log_data, metric_list_path, input_list_path
                    )

                    if u is not None and y is not None and u.shape[0] > 0 and y.shape[0] > 0:
                        n4sid_u_tensor = tf.convert_to_tensor(u, dtype=tf.float32)
                        n4sid_y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
                        n4sid_order_val = n4sid_order
                        trigger_sysid_reload_now = True
                        logger.info(
                            f"[SYS-ID] Data prepared for C++ TrainStepOp (u_shape={u.shape}, y_shape={y.shape})."
                        )
                    else:
                        logger.warning(
                            "[SYS-ID] Skipping online system ID: Not enough data in log file to create a model."
                        )
                # --- END: NATIVE SYS-ID INTEGRATION ---

                (
                    total_loss,
                    std_loss,
                    ewc_loss,
                    grads,
                    aux_metrics,
                    gradient_norm,
                    per_replica_logits,
                    final_float_states_out,
                    final_int_states_out,
                    grad_initial_float_states_out,
                    grad_initial_int_states_out,
                    n4sid_A,
                    n4sid_B,
                    n4sid_C,
                    n4sid_D,
                ) = distributed_train_step(
                    next(train_iterator),
                    n4sid_u_tensor,
                    n4sid_y_tensor,
                    tf.constant(n4sid_order_val, dtype=tf.int32),
                )
                total_loss_val = (
                    strategy.reduce(tf.distribute.ReduceOp.SUM, total_loss, axis=None)
                    / strategy.num_replicas_in_sync
                )

                if tf.math.is_finite(total_loss_val):
                    epoch_train_loss += total_loss_val.numpy()
                    num_train_batches += 1

                    logs = {
                        "loss": total_loss_val.numpy(),
                        "gradient_norm": strategy.reduce(
                            tf.distribute.ReduceOp.MEAN, gradient_norm, axis=None
                        ).numpy(),
                        "memory_bandwidth": psutil.virtual_memory().percent,  # System RAM usage percentage
                    }

                    try:
                        temps = psutil.sensors_temperatures()
                        if "coretemp" in temps and temps["coretemp"]:
                            # Take the first core temperature reading as a representative value.
                            cpu_temp = temps["coretemp"][0].current
                        elif temps:
                            # Fallback: take the first sensor reading available.
                            first_sensor = next(iter(temps.values()))
                            cpu_temp = first_sensor[0].current
                        else:
                            cpu_temp = 70.0  # Default fallback value if no sensors are found
                        logs["cpu_temperature"] = cpu_temp
                    except (AttributeError, IndexError, StopIteration):
                        logs["cpu_temperature"] = 70.0  # Default fallback on error
                    if aux_metrics:
                        if (
                            "graph_learner_keys" in aux_metrics
                            and "graph_learner_values" in aux_metrics
                        ):
                            keys = aux_metrics.pop("graph_learner_keys").numpy()
                            values = aux_metrics.pop("graph_learner_values").numpy()
                            # The keys were flattened inside the while_loop, e.g., b'graph_learner_level_0/some_metric'
                            for k_bytes, v in zip(keys, values):
                                key_str = k_bytes.decode("utf-8")
                                logs[key_str] = v

                        for component_name, component_metrics in aux_metrics.items():
                            if isinstance(component_metrics, dict):
                                for (
                                    metric_name,
                                    per_replica_metric_tensor,
                                ) in component_metrics.items():
                                    flat_key = f"{str(component_name)}/{metric_name}"
                                    reduced_metric = strategy.reduce(
                                        tf.distribute.ReduceOp.MEAN,
                                        per_replica_metric_tensor,
                                        axis=None,
                                    )
                                    logs[flat_key] = reduced_metric.numpy()
                            else:  # Handle cases where the metric is not in a nested dict
                                reduced_metric = strategy.reduce(
                                    tf.distribute.ReduceOp.MEAN, component_metrics, axis=None
                                )
                                logs[component_name] = reduced_metric.numpy()

                    for var_name, var in evolution_time_vars.items():
                        logs[var_name] = var.numpy()

                    if "gradient_norm" in logs:
                        logs["gradient_norm"] = float(np.clip(logs["gradient_norm"], 0.0, 1e6))

                    # Unified Smart Tuner orchestration
                    # Coordinates LR, GaLore rank, BP mitigation, and meta-controller decisions
                    if smart_tuner is not None:
                        try:
                            # Get gradients for tuner analysis (already computed in C++ op)
                            # We use the gradient_norm from the op output
                            grad_norm_for_tuner = logs.get("gradient_norm", 1.0)
                            loss_for_tuner = logs.get("loss", 1.0)

                            # Call orchestrator with step-level metrics
                            tuning_decisions = smart_tuner.orchestrate(
                                step=int(global_step.numpy()),
                                gradients=grads,  # From distributed_train_step
                                variables=model_weights_list,
                                loss=float(loss_for_tuner),
                                gradient_norm=float(grad_norm_for_tuner),
                            )

                            # Apply tuning decisions
                            if tuning_decisions.learning_rate != float(
                                optimizer.learning_rate.numpy()
                            ):
                                optimizer.learning_rate.assign(tuning_decisions.learning_rate)

                            # Log smart tuner state
                            logs["smart_tuner_phase"] = tuning_decisions.phase
                            logs["smart_tuner_lr"] = tuning_decisions.learning_rate
                            logs["smart_tuner_lr_scale"] = tuning_decisions.lr_scale_factor
                            logs["smart_tuner_exploration_factor"] = (
                                tuning_decisions.exploration_factor
                            )
                            logs["smart_tuner_emergency"] = tuning_decisions.emergency_mode
                            logs["smart_tuner_bp_active"] = tuning_decisions.barren_plateau_active

                            # Log additional metrics from tuner
                            for key, value in tuning_decisions.additional_metrics.items():
                                logs[f"smart_tuner_{key}"] = value

                            # Periodic verbose logging for debugging
                            if batch_idx % 100 == 0:
                                logger.info(
                                    "[SmartTuner] Step %d: phase=%s, lr=%.2e, exploration=%.2f, "
                                    "emergency=%s, bp_active=%s",
                                    int(global_step.numpy()),
                                    tuning_decisions.phase,
                                    tuning_decisions.learning_rate,
                                    tuning_decisions.exploration_factor,
                                    tuning_decisions.emergency_mode,
                                    tuning_decisions.barren_plateau_active,
                                )
                        except Exception as tuner_exc:
                            logger.warning(
                                "[SmartTuner] Orchestration failed at step %d: %s",
                                int(global_step.numpy()),
                                tuner_exc,
                            )

                    # Report metrics to HPO orchestrator if enabled
                    if hpo_reporter.enabled and hpo_reporter.should_report():
                        hpo_reporter.report(
                            step=int(global_step.numpy()),
                            loss=float(logs.get("loss", 0.0)),
                            gradient_norm=float(logs.get("gradient_norm", 0.0)),
                            learning_rate=(
                                float(optimizer.learning_rate.numpy())
                                if hasattr(optimizer, "learning_rate")
                                else None
                            ),
                            **{
                                k: float(v) if isinstance(v, (int, float, np.number)) else v
                                for k, v in logs.items()
                                if k not in ["loss", "gradient_norm"]
                            },
                        )

                    if plan_hooks and plan_hooks.controller and current_plan:
                        rss_bytes = psutil.Process().memory_info().rss
                        logs["memory_rss_bytes"] = int(rss_bytes)
                        try:
                            candidate_plan = plan_hooks.controller.ingest_sample(
                                rss_bytes=rss_bytes,
                                max_microbatch=max(plan_hooks.max_microbatch, current_microbatch),
                                max_seq_len=plan_hooks.max_seq_len,
                            )
                        except Exception as exc:  # pragma: no cover - defensive logging only
                            logger.debug("[MEMORY] Adaptive controller ingest failed: %s", exc)
                            candidate_plan = None
                        if candidate_plan and (
                            candidate_plan.microbatch_size != current_plan.microbatch_size
                            or candidate_plan.accum_steps != current_plan.accum_steps
                            or candidate_plan.effective_seq_len != current_plan.effective_seq_len
                        ):
                            logs["memory_plan_pending"] = True
                            logs["memory_plan_microbatch"] = candidate_plan.microbatch_size
                            logs["memory_plan_accum_steps"] = candidate_plan.accum_steps
                            logs["memory_plan_seq_len"] = candidate_plan.effective_seq_len
                    elif plan_hooks and current_plan:
                        logs.setdefault("memory_rss_bytes", int(psutil.Process().memory_info().rss))
                        logs.setdefault("memory_plan_microbatch", current_plan.microbatch_size)
                        logs.setdefault("memory_plan_accum_steps", current_plan.accum_steps)
                        logs.setdefault("memory_plan_seq_len", current_plan.effective_seq_len)

                    controller_logs = (
                        control_bridge.prepare_metrics(logs) if control_bridge else logs
                    )

                    with open(json_log_path, "a") as f:
                        f.write(json.dumps(logs, cls=NumpyJSONEncoder) + "\n")

                    if (
                        profiler is not None
                        and batch_checkpoint_interval
                        and (
                            (batch_idx + 1) % batch_checkpoint_interval == 0
                            or batch_idx + 1 == effective_steps_per_epoch
                        )
                    ):
                        try:
                            loss_value = float(total_loss_val.numpy())
                        except Exception:
                            loss_value = None
                        _profiler_mark(
                            profiler,
                            "train_batch_checkpoint",
                            dataset=dataset_name,
                            epoch=epoch + 1,
                            batch=batch_idx + 1,
                            steps_per_epoch=effective_steps_per_epoch,
                            loss=round(loss_value, 4) if loss_value is not None else None,
                        )

                    grad_norm_scalar = None
                    if gradient_norm is not None:
                        try:
                            grad_norm_scalar = float(
                                strategy.reduce(
                                    tf.distribute.ReduceOp.MEAN, gradient_norm, axis=None
                                )
                            )
                        except Exception:
                            grad_norm_scalar = None

                    controller_safe = (
                        evolution_time_vars
                        and control_bridge
                        and grad_norm_scalar is not None
                        and np.isfinite(grad_norm_scalar)
                        and np.isfinite(total_loss_val)
                    )

                    if evolution_time_vars and control_bridge and not controller_safe:
                        logger.warning(
                            "[META-CTL] Skipping controller update at batch %d due to non-finite metrics "
                            "(grad_norm=%s, loss=%s)",
                            batch_idx + 1,
                            grad_norm_scalar,
                            (
                                float(total_loss_val.numpy())
                                if hasattr(total_loss_val, "numpy")
                                else total_loss_val
                            ),
                        )

                    if controller_safe:
                        block_names_tensor, new_times_tensor = (
                            meta_controller_callback.on_batch_end(
                                batch_idx,
                                logs=controller_logs,
                                force_reload=trigger_sysid_reload_now,
                                trial_dir=control_config_dir,
                                control_input_names=list(evolution_time_vars.keys()),
                            )
                        )

                        if tf.shape(block_names_tensor)[0] > 0:
                            control_bridge.apply_updates(
                                batch_number=batch_idx + 1,
                                block_names=block_names_tensor.numpy(),
                                proposed_values=new_times_tensor.numpy(),
                                force_allow=trigger_sysid_reload_now,
                                telemetry=logs,
                            )

                    # --- Logging logic moved inside the accumulation block ---
                    std_loss_val = (
                        strategy.reduce(tf.distribute.ReduceOp.SUM, std_loss, axis=None)
                        / strategy.num_replicas_in_sync
                        if std_loss is not None
                        else 0.0
                    )
                    ewc_loss_val = (
                        strategy.reduce(tf.distribute.ReduceOp.SUM, ewc_loss, axis=None)
                        / strategy.num_replicas_in_sync
                        if ewc_loss is not None
                        else 0.0
                    )
                    grad_norm_val = (
                        strategy.reduce(tf.distribute.ReduceOp.MEAN, gradient_norm, axis=None)
                        if gradient_norm is not None
                        else 0.0
                    )
                    # Build a structured log message for better readability in the terminal.
                    log_parts = [
                        f"[METRICS] E:{epoch+1} B:{batch_idx+1}/{effective_steps_per_epoch}",
                        f"Loss(Tot): {total_loss_val.numpy():.4f}",
                        f"Loss(Std): {std_loss_val.numpy():.4f}",
                        f"Loss(EWC): {ewc_loss_val.numpy():.4f}",
                        f"GradNorm: {grad_norm_val.numpy():.3f}",
                    ]
                    moe_metrics = []
                    hamiltonian_metrics = []
                    for key, numpy_val in logs.items():
                        if "/" in key:  # This identifies our block-specific metrics
                            # --- START: DEFINITIVE FIX for Logging Block IDs ---
                            # The old logic only extracted digits from the prefix, failing for keys like 'quantum_graph_learner/...'
                            # This new logic is more robust.
                            prefix, metric_name = key.split("/", 1)
                            block_id_digits = "".join(filter(str.isdigit, prefix))

                            if (
                                "quantum_graph_learner" in prefix
                                and "evolution_time_level_" in metric_name
                            ):
                                # Specifically handle quantum graph learner levels
                                block_id = "L" + metric_name.split("_")[-1]
                            else:
                                # Fallback for timecrystal blocks or others
                                block_id = block_id_digits or "?"
                            # --- END: DEFINITIVE FIX ---

                            if "energy_drift" in metric_name:
                                hamiltonian_metrics.append(f"H-Drift({block_id}): {numpy_val:.4e}")
                            elif "evolution_time" in metric_name:
                                hamiltonian_metrics.append(f"H-Time({block_id}): {numpy_val:.4f}")
                            elif "load_balancing_loss" in metric_name:
                                moe_metrics.append(f"MoE-Load({block_id}): {numpy_val:.3f}")
                            elif "router_z_loss" in metric_name:
                                moe_metrics.append(f"MoE-Z({block_id}): {numpy_val:.3f}")

                    if moe_metrics:
                        log_parts.append(f"[{' | '.join(moe_metrics)}]")
                    if hamiltonian_metrics:
                        log_parts.append(f"[{' | '.join(hamiltonian_metrics)}]")

                    logger.info(" | ".join(log_parts))

            if stop_training:
                break
        except Exception as e:
            error_message = f"[{datetime.now().isoformat()}] Error during training epoch {epoch + 1} on dataset '{dataset_name}': {e}\n"
            with open("error.md", "a") as error_file:
                error_file.write(error_message)
            logger.error(f"Caught exception during training. Details logged to error.md: {e}")
            # Report failure to HPO orchestrator if enabled
            if hpo_reporter.enabled:
                hpo_reporter.report_completion(success=False, error_message=str(e))
            # Optionally re-raise the exception if you want training to stop
            # raise

    # Report successful completion to HPO orchestrator if enabled
    if hpo_reporter.enabled:
        hpo_reporter.report_completion(success=True)

    # Log GaLore compression statistics if enabled
    if galore_compressor is not None:
        stats = galore_compressor.get_statistics()
        logger.info(
            "[GALORE] Training complete. Compression stats: "
            "variables=%d, overall_compression=%.2fx",
            stats.get("num_variables", 0),
            stats.get("overall_compression_ratio", 1.0),
        )

    # Log Smart Tuner statistics and record trial for cross-trial memory
    if smart_tuner is not None:
        tuner_stats = smart_tuner.get_statistics()
        logger.info(
            "[SmartTuner] Training complete. Final state: phase=%s, exploration_factor=%.2f, "
            "emergency_mode=%s",
            tuner_stats.get("current_phase", "unknown"),
            tuner_stats.get("exploration_factor", 0.0),
            tuner_stats.get("emergency_mode", False),
        )

        # Record trial for cross-trial memory learning
        if hpo_reporter.enabled and hpo_reporter.trial_id:
            try:
                # Get architecture config from model
                architecture_config = {
                    "embedding_dim": getattr(model.encoder, "embedding_dim", 512),
                    "num_reasoning_blocks": len(getattr(model.reasoning, "reasoning_blocks", [])),
                    "num_moe_experts": getattr(hn_config, "MOE_NUM_EXPERTS", 8),
                    "vocab_size": getattr(model.tokenizer, "vocab_size", 32000),
                }

                # Get hyperparameters
                hyperparameters = {
                    "learning_rate": float(optimizer.learning_rate.numpy()),
                    "batch_size": batch_size,
                    "epochs": epochs,
                }

                # Compute final loss (use last epoch average)
                final_loss = (
                    epoch_train_loss / max(1, num_train_batches)
                    if num_train_batches > 0
                    else float("inf")
                )

                smart_tuner.record_trial(
                    trial_id=hpo_reporter.trial_id,
                    architecture_config=architecture_config,
                    hyperparameters=hyperparameters,
                    final_loss=final_loss,
                    best_epoch=epochs,  # Simplified - could track best epoch separately
                )
                logger.info(
                    "[SmartTuner] Recorded trial %s for cross-trial memory (loss=%.4f)",
                    hpo_reporter.trial_id,
                    final_loss,
                )
            except Exception as record_exc:
                logger.warning(
                    "[SmartTuner] Failed to record trial for cross-trial memory: %s",
                    record_exc,
                )

    return model, cumulative_fisher, cumulative_optimal_weights
