# highnoon/_native/ops/fused_state_bus_op.py
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

"""Python wrapper for fused State Bus C++ op."""

from __future__ import annotations

import tensorflow as tf

from highnoon._native import get_op

_lib = get_op("fused_state_bus")
_fused_state_bus_op = getattr(_lib, "FusedStateBus", None) if _lib else None
_fused_state_bus_grad_op = getattr(_lib, "FusedStateBusGrad", None) if _lib else None


def fused_state_bus_available() -> bool:
    """Check if the fused State Bus op is available."""
    return _fused_state_bus_op is not None


def fused_state_bus(
    query: tf.Tensor,
    write_value: tf.Tensor,
    slots: tf.Tensor,
    read_query_weight: tf.Tensor,
    read_query_bias: tf.Tensor,
    write_gate_weight: tf.Tensor,
    write_gate_bias: tf.Tensor,
    write_value_weight: tf.Tensor,
    write_value_bias: tf.Tensor,
    num_slots: int,
    bus_dim: int,
    write_enabled: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Fused State Bus read + (optional) write.

    Args:
        query: Query tensor [batch, dim].
        write_value: Value tensor for write [batch, dim].
        slots: Current slots [batch, num_slots, bus_dim].
        read_query_weight: [dim, bus_dim].
        read_query_bias: [bus_dim].
        write_gate_weight: [dim, num_slots].
        write_gate_bias: [num_slots].
        write_value_weight: [dim, bus_dim].
        write_value_bias: [bus_dim].
        num_slots: Number of slots.
        bus_dim: Bus dimension.
        write_enabled: Whether to apply slot updates.

    Returns:
        Tuple of (context, slots_new).
    """
    if _fused_state_bus_op is None:
        raise RuntimeError(
            "FusedStateBus C++ op not available. Build with: "
            "cd highnoon/_native && ./build_ops.sh fused_state_bus"
        )

    query = tf.cast(query, tf.float32)
    write_value = tf.cast(write_value, tf.float32)
    slots = tf.cast(slots, tf.float32)
    read_query_weight = tf.cast(read_query_weight, tf.float32)
    read_query_bias = tf.cast(read_query_bias, tf.float32)
    write_gate_weight = tf.cast(write_gate_weight, tf.float32)
    write_gate_bias = tf.cast(write_gate_bias, tf.float32)
    write_value_weight = tf.cast(write_value_weight, tf.float32)
    write_value_bias = tf.cast(write_value_bias, tf.float32)

    @tf.custom_gradient
    def _fused_state_bus_inner(
        q_in,
        w_in,
        slots_in,
        rq_w,
        rq_b,
        wg_w,
        wg_b,
        wv_w,
        wv_b,
    ):
        context, slots_new = _fused_state_bus_op(
            query=q_in,
            write_value=w_in,
            slots=slots_in,
            read_query_weight=rq_w,
            read_query_bias=rq_b,
            write_gate_weight=wg_w,
            write_gate_bias=wg_b,
            write_value_weight=wv_w,
            write_value_bias=wv_b,
            num_slots=num_slots,
            bus_dim=bus_dim,
            write_enabled=write_enabled,
        )

        def grad(grad_context, grad_slots_new):
            if _fused_state_bus_grad_op is not None:
                grads = _fused_state_bus_grad_op(
                    grad_context=grad_context,
                    grad_slots_new=grad_slots_new,
                    query=q_in,
                    write_value=w_in,
                    slots=slots_in,
                    read_query_weight=rq_w,
                    read_query_bias=rq_b,
                    write_gate_weight=wg_w,
                    write_gate_bias=wg_b,
                    write_value_weight=wv_w,
                    write_value_bias=wv_b,
                    num_slots=num_slots,
                    bus_dim=bus_dim,
                    write_enabled=write_enabled,
                )
                return grads
            return (
                tf.zeros_like(q_in),
                tf.zeros_like(w_in),
                tf.zeros_like(slots_in),
                tf.zeros_like(rq_w),
                tf.zeros_like(rq_b),
                tf.zeros_like(wg_w),
                tf.zeros_like(wg_b),
                tf.zeros_like(wv_w),
                tf.zeros_like(wv_b),
            )

        return (context, slots_new), grad

    return _fused_state_bus_inner(
        query,
        write_value,
        slots,
        read_query_weight,
        read_query_bias,
        write_gate_weight,
        write_gate_bias,
        write_value_weight,
        write_value_bias,
    )


__all__ = ["fused_state_bus", "fused_state_bus_available"]
