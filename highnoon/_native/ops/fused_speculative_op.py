# highnoon/_native/ops/fused_speculative_op.py
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

"""Python wrapper for fused Speculative Decoding C++ ops.

Provides accelerated speculative decoding verification and sampling.
"""

from __future__ import annotations

import tensorflow as tf

from highnoon._native import get_op

_lib = get_op("fused_speculative")
_verify_op = getattr(_lib, "FusedSpeculativeVerify", None) if _lib else None
_sample_op = getattr(_lib, "FusedSpeculativeSample", None) if _lib else None


def fused_speculative_verify(
    target_logits: tf.Tensor,
    draft_probs: tf.Tensor,
    draft_tokens: tf.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    use_sampling: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Verify draft tokens against target distribution.

    Args:
        target_logits: Target model logits [batch, num_spec, vocab].
        draft_probs: Draft model probabilities [batch, num_spec, vocab].
        draft_tokens: Draft token indices [batch, num_spec].
        temperature: Sampling temperature.
        top_k: Top-k filtering.
        use_sampling: Use sampling vs greedy for bonus token.

    Returns:
        Tuple of (num_accepted, bonus_token):
        - num_accepted: Number of accepted tokens per batch [batch]
        - bonus_token: Resampled bonus token [batch]
    """
    if _verify_op is None:
        raise RuntimeError(
            "FusedSpeculativeVerify C++ op not available. "
            "Build with: cd highnoon/_native && ./build_secure.sh"
        )

    target_logits = tf.cast(target_logits, tf.float32)
    draft_probs = tf.cast(draft_probs, tf.float32)
    draft_tokens = tf.cast(draft_tokens, tf.int32)

    return _verify_op(
        target_logits=target_logits,
        draft_probs=draft_probs,
        draft_tokens=draft_tokens,
        temperature=temperature,
        top_k=top_k,
        use_sampling=use_sampling,
    )


def fused_speculative_sample(
    logits: tf.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    use_sampling: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Sample tokens from logits with temperature and top-k.

    Args:
        logits: Input logits [batch, vocab].
        temperature: Sampling temperature.
        top_k: Top-k filtering.
        use_sampling: Use sampling vs greedy.

    Returns:
        Tuple of (tokens, probs):
        - tokens: Sampled token indices [batch]
        - probs: Probability distribution [batch, vocab]
    """
    if _sample_op is None:
        raise RuntimeError(
            "FusedSpeculativeSample C++ op not available. "
            "Build with: cd highnoon/_native && ./build_secure.sh"
        )

    logits = tf.cast(logits, tf.float32)

    return _sample_op(
        logits=logits,
        temperature=temperature,
        top_k=top_k,
        use_sampling=use_sampling,
    )


__all__ = ["fused_speculative_verify", "fused_speculative_sample"]
