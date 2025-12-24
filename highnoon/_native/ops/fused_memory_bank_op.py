# highnoon/_native/ops/fused_memory_bank_op.py
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

"""Python wrapper for fused Memory Bank C++ op.

This module provides a Python interface to the C++ FusedMemoryBank kernel with
automatic gradient support via tf.custom_gradient.
"""

from __future__ import annotations

import tensorflow as tf

from highnoon._native import get_op

# Load the C++ op library
_lib = get_op("fused_memory_bank")
_fused_memory_bank_op = getattr(_lib, "FusedMemoryBank", None) if _lib else None
_fused_memory_bank_grad_op = getattr(_lib, "FusedMemoryBankGrad", None) if _lib else None


def fused_memory_bank(
    x: tf.Tensor,
    memory: tf.Tensor,
    query_proj: tf.Tensor,
    key_proj: tf.Tensor,
    write_proj: tf.Tensor,
    gate_proj: tf.Tensor,
    num_slots: int,
    slot_dim: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Fused Gated External Memory Bank.

    C++-accelerated memory bank that fuses content-based read, gated write,
    and memory update operations.

    Args:
        x: Input tensor [batch, seq_len, embed_dim].
        memory: Current memory [batch, num_slots, slot_dim].
        query_proj: Query projection weights [embed_dim, slot_dim].
        key_proj: Key projection weights [slot_dim, slot_dim].
        write_proj: Write content projection [embed_dim, slot_dim].
        gate_proj: Gate projection [embed_dim, 1].
        num_slots: Number of memory slots.
        slot_dim: Dimension per memory slot.

    Returns:
        Tuple of (output, updated_memory):
        - output: Augmented output [batch, seq_len, embed_dim]
        - memory_new: Updated memory [batch, num_slots, slot_dim]

    Raises:
        RuntimeError: If C++ op library is not available.
    """
    if _fused_memory_bank_op is None:
        raise RuntimeError(
            "FusedMemoryBank C++ op not available. Build with: "
            "cd highnoon/_native && ./build_ops.sh fused_memory_bank"
        )

    # Ensure float32
    x = tf.cast(x, tf.float32)
    memory = tf.cast(memory, tf.float32)
    query_proj = tf.cast(query_proj, tf.float32)
    key_proj = tf.cast(key_proj, tf.float32)
    write_proj = tf.cast(write_proj, tf.float32)
    gate_proj = tf.cast(gate_proj, tf.float32)

    @tf.custom_gradient
    def _fused_memory_bank_inner(x_in, mem, qp, kp, wp, gp):
        """Inner function with tensor-only signature."""
        output, memory_new = _fused_memory_bank_op(
            x=x_in,
            memory=mem,
            query_proj=qp,
            key_proj=kp,
            write_proj=wp,
            gate_proj=gp,
            num_slots=num_slots,
            slot_dim=slot_dim,
        )

        def grad(grad_output, grad_memory_new):
            """Compute gradients."""
            if _fused_memory_bank_grad_op is not None:
                grads = _fused_memory_bank_grad_op(
                    grad_output=grad_output,
                    grad_memory_new=grad_memory_new,
                    x=x_in,
                    memory=mem,
                    query_proj=qp,
                    key_proj=kp,
                    write_proj=wp,
                    gate_proj=gp,
                    num_slots=num_slots,
                    slot_dim=slot_dim,
                )
                return grads
            else:
                return (
                    tf.zeros_like(x_in),
                    tf.zeros_like(mem),
                    tf.zeros_like(qp),
                    tf.zeros_like(kp),
                    tf.zeros_like(wp),
                    tf.zeros_like(gp),
                )

        return (output, memory_new), grad

    return _fused_memory_bank_inner(x, memory, query_proj, key_proj, write_proj, gate_proj)


__all__ = ["fused_memory_bank"]
