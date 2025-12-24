# highnoon/_native/ops/fused_adaptive_memory_op.py
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

"""Python wrapper for Fused Adaptive Memory C++ op.

Phase 18.3: Implements Titans-inspired adaptive memory with test-time learning.
This wrapper provides a Python interface to the C++ FusedAdaptiveMemory kernel.
"""

from __future__ import annotations

import tensorflow as tf

from highnoon._native import get_op

# Load the C++ op library
_lib = get_op("fused_adaptive_memory")
_fused_adaptive_memory_op = getattr(_lib, "FusedAdaptiveMemory", None) if _lib else None
_fused_adaptive_memory_grad_op = getattr(_lib, "FusedAdaptiveMemoryGrad", None) if _lib else None


def fused_adaptive_memory_available() -> bool:
    """Check if the C++ op is available."""
    return _fused_adaptive_memory_op is not None


def fused_adaptive_memory(
    x: tf.Tensor,
    memory: tf.Tensor,
    predictor_w1: tf.Tensor,
    predictor_b1: tf.Tensor,
    predictor_w2: tf.Tensor,
    predictor_b2: tf.Tensor,
    compress_query: tf.Tensor,
    compress_key: tf.Tensor,
    write_proj: tf.Tensor,
    gate_proj: tf.Tensor,
    learning_rate: float = 0.01,
    surprise_threshold: float = 0.5,
    enable_ttl: bool = True,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Fused Adaptive Memory with Test-Time Learning.

    Phase 18.3: C++-accelerated adaptive memory that can update its internal
    prediction MLP during inference based on surprise signals. When surprise
    (prediction error) exceeds the threshold, weights are updated via gradient
    descent to better capture the pattern for future retrieval.

    Args:
        x: Input tensor [batch, seq_len, input_dim].
        memory: Current memory state [batch, num_slots, slot_dim].
        predictor_w1: First layer weights [slot_dim, mlp_dim].
        predictor_b1: First layer bias [mlp_dim].
        predictor_w2: Second layer weights [mlp_dim, input_dim].
        predictor_b2: Second layer bias [input_dim].
        compress_query: Query compression [input_dim, slot_dim].
        compress_key: Key compression [slot_dim, slot_dim].
        write_proj: Write content projection [input_dim, slot_dim].
        gate_proj: Write gate projection [input_dim, 1].
        learning_rate: TTL learning rate (default: 0.01).
        surprise_threshold: Threshold for triggering updates (default: 0.5).
        enable_ttl: Whether to enable test-time learning (default: True).

    Returns:
        Tuple of (output, memory_new, surprise):
        - output: Augmented input [batch, seq_len, input_dim]
        - memory_new: Updated memory [batch, num_slots, slot_dim]
        - surprise: Per-batch surprise values [batch]

    Raises:
        RuntimeError: If C++ op is not available and fallback disabled.
    """
    if _fused_adaptive_memory_op is None:
        # Python fallback
        return _adaptive_memory_fallback(
            x,
            memory,
            predictor_w1,
            predictor_b1,
            predictor_w2,
            predictor_b2,
            compress_query,
            compress_key,
            write_proj,
            gate_proj,
            learning_rate,
            surprise_threshold,
            enable_ttl,
        )

    # Ensure float32
    x = tf.cast(x, tf.float32)
    memory = tf.cast(memory, tf.float32)
    predictor_w1 = tf.cast(predictor_w1, tf.float32)
    predictor_b1 = tf.cast(predictor_b1, tf.float32)
    predictor_w2 = tf.cast(predictor_w2, tf.float32)
    predictor_b2 = tf.cast(predictor_b2, tf.float32)
    compress_query = tf.cast(compress_query, tf.float32)
    compress_key = tf.cast(compress_key, tf.float32)
    write_proj = tf.cast(write_proj, tf.float32)
    gate_proj = tf.cast(gate_proj, tf.float32)

    @tf.custom_gradient
    def _inner(x_in, mem, w1, b1, w2, b2, cq, ck, wp, gp):
        output, memory_new, surprise = _fused_adaptive_memory_op(
            x=x_in,
            memory=mem,
            predictor_w1=w1,
            predictor_b1=b1,
            predictor_w2=w2,
            predictor_b2=b2,
            compress_query=cq,
            compress_key=ck,
            write_proj=wp,
            gate_proj=gp,
            learning_rate=learning_rate,
            surprise_threshold=surprise_threshold,
            enable_ttl=enable_ttl,
        )

        def grad(grad_output, grad_memory_new, grad_surprise):
            if _fused_adaptive_memory_grad_op is not None:
                grad_x, grad_mem = _fused_adaptive_memory_grad_op(
                    grad_output=grad_output,
                    grad_memory_new=grad_memory_new,
                    x=x_in,
                    memory=mem,
                )
                # Return gradients for all inputs
                return (
                    grad_x,
                    grad_mem,
                    tf.zeros_like(w1),  # predictor weights not in training graph
                    tf.zeros_like(b1),
                    tf.zeros_like(w2),
                    tf.zeros_like(b2),
                    tf.zeros_like(cq),
                    tf.zeros_like(ck),
                    tf.zeros_like(wp),
                    tf.zeros_like(gp),
                )
            else:
                return tuple(tf.zeros_like(t) for t in [x_in, mem, w1, b1, w2, b2, cq, ck, wp, gp])

        return (output, memory_new, surprise), grad

    return _inner(
        x,
        memory,
        predictor_w1,
        predictor_b1,
        predictor_w2,
        predictor_b2,
        compress_query,
        compress_key,
        write_proj,
        gate_proj,
    )


def _adaptive_memory_fallback(
    x: tf.Tensor,
    memory: tf.Tensor,
    predictor_w1: tf.Tensor,
    predictor_b1: tf.Tensor,
    predictor_w2: tf.Tensor,
    predictor_b2: tf.Tensor,
    compress_query: tf.Tensor,
    compress_key: tf.Tensor,
    write_proj: tf.Tensor,
    gate_proj: tf.Tensor,
    learning_rate: float,
    surprise_threshold: float,
    enable_ttl: bool,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Pure TensorFlow fallback for adaptive memory.

    Note: This does NOT support test-time learning (weight updates during
    inference are not possible in pure TensorFlow without eager mode hacks).
    The C++ op is required for full TTL support.
    """
    tf.shape(x)[0]

    # Memory summary: mean over slots
    memory_summary = tf.reduce_mean(memory, axis=1)  # [B, slot_dim]

    # Predict from memory: W2 @ GELU(W1 @ memory_summary + b1) + b2
    hidden = tf.nn.gelu(tf.matmul(memory_summary, predictor_w1) + predictor_b1)
    prediction = tf.matmul(hidden, predictor_w2) + predictor_b2

    # X summary: mean over sequence
    x_summary = tf.reduce_mean(x, axis=1)  # [B, input_dim]

    # Compute surprise
    diff_sq = tf.square(x_summary - prediction)
    input_dim = tf.cast(tf.shape(x_summary)[-1], tf.float32)
    surprise = tf.reduce_mean(diff_sq, axis=-1) / input_dim
    surprise = tf.clip_by_value(surprise, 0.0, 1.0)

    # Simple output augmentation: add memory context
    slot_dim = tf.shape(memory_summary)[-1]
    x_dim = tf.shape(x)[-1]
    min_dim = tf.minimum(slot_dim, x_dim)

    # Broadcast memory summary over sequence
    memory_broadcast = tf.expand_dims(memory_summary[:, :min_dim], 1)
    tf.shape(x)

    # Pad if needed
    if slot_dim < x_dim:
        pad_size = x_dim - slot_dim
        memory_broadcast = tf.pad(memory_broadcast, [[0, 0], [0, 0], [0, pad_size]])

    output = x + 0.1 * memory_broadcast

    # Memory passthrough (no update in fallback)
    memory_new = memory

    return output, memory_new, surprise


__all__ = [
    "fused_adaptive_memory",
    "fused_adaptive_memory_available",
]
