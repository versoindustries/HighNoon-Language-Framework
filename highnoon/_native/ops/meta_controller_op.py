# highnoon/_native/ops/meta_controller_op.py
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

"""Python wrapper for the Hamiltonian Meta-Controller TensorFlow op.

This module provides a Python interface to the C++ TriggerMetaController op,
which orchestrates the quantum-enhanced control system including:
- RLS System Identifier (fast online system identification)
- Hybrid PID Tuner (Adam + relay-based tuning)
- Extended Kalman Filter (nonlinear state estimation)
- Tensor Network Kalman Filter (memory-efficient high-dimensional states)

Example:
    >>> from highnoon._native.ops.meta_controller_op import trigger_meta_controller
    >>> block_names, evolution_times = trigger_meta_controller(
    ...     metric_values=tf.constant([0.5, 0.1]),
    ...     metric_names=tf.constant(["loss", "grad_norm"]),
    ... )
"""

from __future__ import annotations

import logging

import tensorflow as tf

from highnoon._native.ops.lib_loader import get_highnoon_core_path

logger = logging.getLogger(__name__)

# Load the compiled library and get the op
_ops = None


def _get_native_ops():
    """Lazy-load the native ops library."""
    global _ops
    if _ops is None:
        lib_path = get_highnoon_core_path()
        if lib_path is None:
            raise ImportError(
                "highnoon_core library not found. "
                "Run 'cd highnoon/_native && ./build_secure.sh' to compile."
            )
        try:
            _ops = tf.load_op_library(lib_path)
            logger.debug(f"Loaded meta_controller_op from {lib_path}")
        except Exception as e:
            raise ImportError(f"Failed to load highnoon_core: {e}") from e
    return _ops


def trigger_meta_controller(
    metric_values: tf.Tensor,
    metric_names: tf.Tensor,
    control_input_names: tf.Tensor | None = None,
    trigger_autotune: tf.Tensor | bool = False,
    trigger_system_id: tf.Tensor | bool = False,
    config_path: tf.Tensor | str = "",
) -> tuple[tf.Tensor, tf.Tensor]:
    """Trigger the Hamiltonian Meta-Controller with current training metrics.

    The meta-controller uses quantum-enhanced algorithms to dynamically adjust
    model evolution times during training. It includes:
    - RLS for fast online system identification (every batch vs every 850)
    - Hybrid PID tuning with Adam optimizer
    - Extended Kalman filtering for nonlinear dynamics
    - Tensor Network Kalman for memory-efficient state estimation

    Args:
        metric_values: 1D tensor of metric values [N]
        metric_names: 1D string tensor of metric names [N]
        control_input_names: Optional string tensor of controllable variable names
        trigger_autotune: If True, trigger automatic PID tuning
        trigger_system_id: If True, trigger system identification reload
        config_path: Path to trial-specific config directory (for HPO)

    Returns:
        Tuple of (block_names, evolution_times):
        - block_names: String tensor of reasoning block names
        - evolution_times: Float tensor of new evolution times for each block

    Example:
        >>> metrics = tf.constant([0.5, 0.1, 1e-4], dtype=tf.float32)
        >>> names = tf.constant(["loss", "gradient_norm", "learning_rate"])
        >>> blocks, times = trigger_meta_controller(metrics, names)
        >>> print(f"Updated {len(blocks)} blocks")
    """
    ops = _get_native_ops()

    # Ensure inputs are tensors with correct types
    if not isinstance(metric_values, tf.Tensor):
        metric_values = tf.constant(metric_values, dtype=tf.float32)

    if not isinstance(metric_names, tf.Tensor):
        metric_names = tf.constant(metric_names, dtype=tf.string)

    if control_input_names is None:
        control_input_names = tf.constant([], dtype=tf.string)
    elif not isinstance(control_input_names, tf.Tensor):
        control_input_names = tf.constant(control_input_names, dtype=tf.string)

    if not isinstance(trigger_autotune, tf.Tensor):
        trigger_autotune = tf.constant(trigger_autotune, dtype=tf.bool)

    if not isinstance(trigger_system_id, tf.Tensor):
        trigger_system_id = tf.constant(trigger_system_id, dtype=tf.bool)

    if not isinstance(config_path, tf.Tensor):
        config_path = tf.constant(config_path, dtype=tf.string)

    # Call the C++ op
    block_names, evolution_times = ops.trigger_meta_controller(
        metric_values=metric_values,
        metric_names=metric_names,
        control_input_names=control_input_names,
        trigger_autotune=trigger_autotune,
        trigger_system_id=trigger_system_id,
        config_path=config_path,
    )

    return block_names, evolution_times


__all__ = ["trigger_meta_controller"]
