# highnoon/_native/ops/fused_wavelet_encoder.py
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

"""Python wrapper for the fused Wavelet Encoder custom C++ operation.

Provides a high-performance wavelet transformation for sequence encoding,
returning approximation and detail coefficients.
"""

import logging

import tensorflow as tf

from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

# --- Load the Custom Operator ---
_fused_wavelet_module = None
fused_wavelet_encoder_chunk_op = None
fused_wavelet_encoder_chunk_grad_op = None

try:
    # lib_loader.resolve_op_library now returns _highnoon_core.so path
    _op_lib_path = resolve_op_library(__file__, "_fused_wavelet_encoder_op.so")
    _fused_wavelet_module = tf.load_op_library(_op_lib_path)
    # Try to get the ops - names may differ in consolidated binary
    if hasattr(_fused_wavelet_module, "fused_wavelet_encoder_chunk"):
        fused_wavelet_encoder_chunk_op = _fused_wavelet_module.fused_wavelet_encoder_chunk
        fused_wavelet_encoder_chunk_grad_op = getattr(
            _fused_wavelet_module, "fused_wavelet_encoder_chunk_grad", None
        )
        logger.info("Successfully loaded custom C++ FusedWaveletEncoderChunk operator.")
    else:
        raise AttributeError("fused_wavelet_encoder_chunk op not found in library")
except (tf.errors.NotFoundError, OSError, AttributeError) as e:
    logger.error(f"Could not load the custom C++ FusedWaveletEncoderChunk op. Error: {e}")
    fused_wavelet_encoder_chunk_op = None


def fused_wavelet_encoder_available() -> bool:
    """Check if the fused wavelet encoder operation is available."""
    return fused_wavelet_encoder_chunk_op is not None


@tf.custom_gradient
def fused_wavelet_encoder_chunk(
    input_data: tf.Tensor,
    low_pass_filter: tf.Tensor,
    high_pass_filter: tf.Tensor,
    mask: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Python wrapper for the FusedWaveletEncoderChunk custom C++ operator.

    Performs a 1D discrete wavelet transform on the input sequence data,
    returning approximation and detail coefficients.

    Args:
        input_data: Input tensor of shape [batch, seq_len, hidden_dim].
        low_pass_filter: Low-pass filter for wavelet transform.
        high_pass_filter: High-pass filter for wavelet transform.
        mask: Boolean mask tensor of shape [batch, seq_len].

    Returns:
        Tuple of (approx_coeffs, detail_coeffs) tensors.

    Raises:
        NotImplementedError: If the C++ operation is not available.
    """
    if fused_wavelet_encoder_chunk_op is None:
        raise NotImplementedError(
            "The C++ FusedWaveletEncoderChunk operator could not be loaded. "
            "Rebuild ops via build_ops.sh."
        )

    # The C++ op returns two tensors.
    approx_coeffs, detail_coeffs = fused_wavelet_encoder_chunk_op(
        input_data, low_pass_filter, high_pass_filter, mask
    )

    def grad_fn(
        grad_approx: tf.Tensor,
        grad_detail: tf.Tensor,
        variables: list[tf.Variable] | None = None,
    ) -> tuple[tuple[tf.Tensor, ...], list[tf.Tensor | None]]:
        """Gradient function that calls the custom C++ backward kernel."""
        if fused_wavelet_encoder_chunk_grad_op is None:
            raise NotImplementedError(
                "The C++ FusedWaveletEncoderChunkGrad operator is unavailable."
            )

        # Call the C++ gradient op with all required inputs.
        grads = fused_wavelet_encoder_chunk_grad_op(
            grad_approx_coeffs=grad_approx,
            grad_detail_coeffs=grad_detail,
            input_data=input_data,
            low_pass_filter=low_pass_filter,
            high_pass_filter=high_pass_filter,
            mask=mask,
        )

        # The C++ op returns 3 gradients for the differentiable inputs.
        input_grads = (grads[0], grads[1], grads[2], None)
        variable_grads_list = [None] * len(variables) if variables is not None else []

        return input_grads, variable_grads_list

    return (approx_coeffs, detail_coeffs), grad_fn


__all__ = [
    "fused_wavelet_encoder_chunk",
    "fused_wavelet_encoder_available",
]
