# highnoon/_native/ops/train_step.py
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

"""Python wrapper for the fused TrainStep custom C++ operation.

Provides a high-performance fused training step that combines:
- Forward pass
- Loss computation
- Gradient computation
- Optimizer update

All in a single C++ kernel to minimize Python overhead and memory bandwidth.
"""

import logging
import os
import platform
from copy import deepcopy

import tensorflow as tf

from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

_train_step_module: object | None = None
_train_step_diagnostics: dict[str, object] = {
    "resolved_path": None,
    "exists": False,
    "file_size_bytes": None,
    "mtime": None,
    "target_arch": os.getenv("VERSO_TARGET_ARCH"),
    "python_version": platform.python_version(),
    "tensorflow_version": tf.__version__,
    "load_error": None,
    "loaded": False,
    "available_symbols": [],
}

try:
    # lib_loader.resolve_op_library now returns _highnoon_core.so path
    _op_lib_path = resolve_op_library(__file__, "_train_step_op.so")
    _train_step_diagnostics["resolved_path"] = _op_lib_path
    _train_step_diagnostics["exists"] = os.path.exists(_op_lib_path)

    if _train_step_diagnostics["exists"]:
        try:
            stat_result = os.stat(_op_lib_path)
            _train_step_diagnostics["file_size_bytes"] = stat_result.st_size
            _train_step_diagnostics["mtime"] = stat_result.st_mtime
        except OSError as stat_exc:  # pragma: no cover - informational only
            logger.debug("Unable to stat TrainStep op: %s", stat_exc)

        _train_step_module = tf.load_op_library(_op_lib_path)
        # Check if the specific op exists in the consolidated binary
        if hasattr(_train_step_module, "train_step"):
            logger.info("Successfully loaded TrainStep custom C++ op from %s.", _op_lib_path)
            _train_step_diagnostics["loaded"] = True
            _train_step_diagnostics["available_symbols"] = sorted(dir(_train_step_module))
        else:
            raise AttributeError("train_step op not found in library")
    else:
        raise FileNotFoundError(f"TrainStep op library not found at {_op_lib_path}")
except Exception as exc:  # noqa: BLE001
    logger.error("Unable to load TrainStep custom op: %s", exc)
    _train_step_diagnostics["load_error"] = str(exc)


def train_step_module() -> object | None:
    """Expose the loaded module reference for downstream diagnostics if needed."""
    return _train_step_module


def train_step_diagnostics() -> dict[str, object]:
    """Return a shallow copy of the latest loader diagnostics for reporting."""
    return deepcopy(_train_step_diagnostics)


def train_step_available() -> bool:
    """Check if the fused train step operation is available."""
    return _train_step_module is not None


def fused_train_step(
    model_weights: tf.Tensor,
    gradients: tf.Tensor,
    optimizer_state: tf.Tensor,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.0,
) -> tf.Tensor:
    """Execute a fused training step using the native C++ operation.

    Args:
        model_weights: Current model weights tensor.
        gradients: Computed gradients tensor.
        optimizer_state: Current optimizer state (momentum, variance, etc.).
        learning_rate: Learning rate for the update.
        beta1: First moment decay rate (default: 0.9).
        beta2: Second moment decay rate (default: 0.999).
        epsilon: Numerical stability constant (default: 1e-8).
        weight_decay: Weight decay coefficient (default: 0.0).

    Returns:
        Updated model weights tensor.

    Raises:
        NotImplementedError: If the native operation is not available.
    """
    if _train_step_module is None:
        raise NotImplementedError(
            "The fused_train_step custom op is unavailable. Rebuild the C++ ops via build_ops.sh."
        )

    return _train_step_module.train_step(
        model_weights,
        gradients,
        optimizer_state,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
    )


__all__ = [
    "train_step_module",
    "train_step_diagnostics",
    "train_step_available",
    "fused_train_step",
]
