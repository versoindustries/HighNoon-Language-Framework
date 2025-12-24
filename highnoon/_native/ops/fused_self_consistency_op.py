# highnoon/_native/ops/fused_self_consistency_op.py
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

"""Python wrapper for fused Self-Consistency Verification C++ op.

This module provides a Python interface to the C++ FusedSelfConsistency
kernel with automatic gradient support via tf.custom_gradient.

The SelfConsistencyVerifier implements DeepSeek-R1 style multi-path
verification for reasoning confidence.
"""

from __future__ import annotations

import tensorflow as tf

from highnoon._native import get_op

# Load the C++ op library
_lib = get_op("fused_self_consistency")
_fused_self_consistency_op = getattr(_lib, "FusedSelfConsistency", None) if _lib else None
_fused_self_consistency_grad_op = getattr(_lib, "FusedSelfConsistencyGrad", None) if _lib else None


def fused_self_consistency(
    paths: tf.Tensor,
    verification_weights: tf.Tensor,
    aggregation_weight: tf.Tensor,
    aggregation_bias: tf.Tensor,
    norm_gamma: tf.Tensor,
    norm_beta: tf.Tensor,
    num_verification_heads: int = 4,
    threshold: float = 0.5,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Fused Self-Consistency Verification.

    C++-accelerated self-consistency verification that fuses:
    - L2 normalization of reasoning paths
    - Pairwise cosine similarity computation
    - Consistency score aggregation
    - Softmax-weighted path combination
    - Aggregation projection
    - Layer normalization
    - Threshold-based confidence gating

    Args:
        paths: Input reasoning paths [batch, seq_len, num_paths, dim].
        verification_weights: Verification head weights [num_heads, dim, head_dim].
        aggregation_weight: Aggregation projection [dim, dim].
        aggregation_bias: Aggregation bias [dim].
        norm_gamma: LayerNorm scale [dim].
        norm_beta: LayerNorm bias [dim].
        num_verification_heads: Number of parallel verification heads.
        threshold: Consistency threshold for confidence gating.

    Returns:
        Tuple of (output, confidence):
        - output: Verified output [batch, seq_len, dim]
        - confidence: Consistency confidence [batch, seq_len]

    Raises:
        RuntimeError: If C++ op library is not available.
    """
    if _fused_self_consistency_op is None:
        raise RuntimeError(
            "FusedSelfConsistency C++ op not available. Build with: "
            "cd highnoon/_native && ./build_ops.sh fused_self_consistency"
        )

    # Ensure float32
    paths = tf.cast(paths, tf.float32)
    verification_weights = tf.cast(verification_weights, tf.float32)
    aggregation_weight = tf.cast(aggregation_weight, tf.float32)
    aggregation_bias = tf.cast(aggregation_bias, tf.float32)
    norm_gamma = tf.cast(norm_gamma, tf.float32)
    norm_beta = tf.cast(norm_beta, tf.float32)

    @tf.custom_gradient
    def _fused_self_consistency_inner(paths_in, vw, agg_w, agg_b, ng, nb):
        """Inner function with tensor-only signature."""
        output, confidence = _fused_self_consistency_op(
            paths=paths_in,
            verification_weights=vw,
            aggregation_weight=agg_w,
            aggregation_bias=agg_b,
            norm_gamma=ng,
            norm_beta=nb,
            num_verification_heads=num_verification_heads,
            threshold=threshold,
        )

        def grad(grad_output, grad_confidence):
            """Compute gradients."""
            if _fused_self_consistency_grad_op is not None:
                grads = _fused_self_consistency_grad_op(
                    grad_output=grad_output,
                    grad_confidence=grad_confidence,
                    paths=paths_in,
                    verification_weights=vw,
                    aggregation_weight=agg_w,
                    num_verification_heads=num_verification_heads,
                    threshold=threshold,
                )
                return grads
            else:
                return (
                    tf.zeros_like(paths_in),
                    tf.zeros_like(vw),
                    tf.zeros_like(agg_w),
                    tf.zeros_like(agg_b),
                    tf.zeros_like(ng),
                    tf.zeros_like(nb),
                )

        return (output, confidence), grad

    return _fused_self_consistency_inner(
        paths,
        verification_weights,
        aggregation_weight,
        aggregation_bias,
        norm_gamma,
        norm_beta,
    )


__all__ = ["fused_self_consistency"]
