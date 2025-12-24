# highnoon/_native/ops/fused_qwt_tokenizer.py
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

"""Python wrapper for the fused Quantum Wavelet Tokenizer custom op.

This operation performs quantum-inspired wavelet transformation and
continuous-time quantum walk (CTQW) evolution on input token embeddings.
"""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.python.framework import ops

logger = logging.getLogger(__name__)

_qwt_module = None
_fused_qwt_tokenizer = None
_fused_qwt_tokenizer_grad = None
_OP_LIBRARY_PATH: str | None = None

try:
    from highnoon._native import fused_qwt_tokenizer_op_path, load_qwt_tokenizer

    _qwt_module = load_qwt_tokenizer()
    if _qwt_module is not None:
        _fused_qwt_tokenizer = _qwt_module.fused_qwt_tokenizer
        _fused_qwt_tokenizer_grad = _qwt_module.fused_qwt_tokenizer_grad
        _OP_LIBRARY_PATH = fused_qwt_tokenizer_op_path()
        logger.info("Loaded fused_qwt_tokenizer op from consolidated binary")
    else:
        logger.error("fused_qwt_tokenizer op not found in consolidated binary")
except ImportError as err:  # pragma: no cover - load-time
    logger.error("Unable to load fused_qwt_tokenizer op: %s", err)


def fused_qwt_tokenizer(
    input_data: tf.Tensor,
    low_pass_filter: tf.Tensor,
    high_pass_filter: tf.Tensor,
    mask: tf.Tensor,
    evolution_time: tf.Tensor,
    ctqw_steps: tf.Tensor,
    epsilon: float = 1e-5,
    num_levels: int = 1,
    # Phase 17: Enhancement parameters
    use_lifting_scheme: bool = True,
    pade_order: int = 4,
    use_jacobi_preconditioner: bool = True,
    skip_stride: int = 0,
    max_skips_per_node: int = 2,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Invokes the fused Quantum Wavelet Tokenizer custom op.

    This op performs a quantum-inspired wavelet transformation and subsequent
    continuous-time quantum walk (CTQW) evolution on the input data. It
    integrates 1D DWT chunking, wavelet-informed sparse Hamiltonian construction,
    and Cayley-integrated CTQW evolution.

    Phase 10.3: Added multi-scale wavelet support via num_levels parameter.
    Phase 17: Added 5 enhancements for improved performance and accuracy.

    Args:
        input_data: A `tf.Tensor` of type `float32` with shape `[batch, seq_len, hidden_dim]`.
            The input sequence data. `seq_len` must be even.
        low_pass_filter: A `tf.Tensor` of type `float32` with shape `[filter_width, 1, hidden_dim]`.
            The low-pass filter for the wavelet transform.
        high_pass_filter: A `tf.Tensor` of type `float32` with shape `[filter_width, 1, hidden_dim]`.
            The high-pass filter for the wavelet transform.
        mask: A `tf.Tensor` of type `bool` with shape `[batch, seq_len]`.
            A boolean mask indicating valid elements in the input sequence.
        evolution_time: A `tf.Tensor` of type `float32`. Can be a scalar or a 1D tensor
            of shape `[batch]`. The total evolution time for the quantum walk.
        ctqw_steps: A `tf.Tensor` of type `int32`. Can be a scalar or a 1D tensor
            of shape `[batch]`. The number of steps for the continuous-time quantum walk.
        epsilon: A `float` (default 1e-5). A small value added for numerical stability,
            e.g., in energy calculations to prevent division by zero.
        num_levels: Number of cascaded DWT levels for multi-scale decomposition (default: 1).
            Higher values produce more coefficient bands: level 1 = seq_len/2,
            level 2 = seq_len/2 + seq_len/4, level 3 = seq_len/2 + seq_len/4 + seq_len/8, etc.
        use_lifting_scheme: If True (default), use lifting scheme DWT which is ~50% faster
            than standard FIR convolution. The lifting scheme computes identical coefficients.
        pade_order: Order of Padé approximation for matrix exponential (1-4, default: 4).
            Higher orders provide better accuracy for larger evolution times.
            Order 1 is equivalent to the original Cayley approximation.
        use_jacobi_preconditioner: If True (default), apply Jacobi (diagonal) preconditioning
            to the BiCGSTAB solver, reducing iterations by 2-4x.
        skip_stride: Stride for skip connections in the Hamiltonian (0 = disabled, default).
            When > 0, adds long-range edges at this stride for improved global context.
        max_skips_per_node: Maximum skip connections per node (default: 2).
            Only used when skip_stride > 0.

    Returns:
        A tuple of three `tf.Tensor` objects:
        - approx_coeffs: Approximation coefficients from the wavelet transform,
            shape `[batch, output_nodes, hidden_dim]` where output_nodes depends on num_levels.
        - detail_coeffs: Detail coefficients from the wavelet transform,
            shape `[batch, output_nodes, hidden_dim]`.
        - qwt_embeddings: The evolved, globally contextualized embeddings after
            the quantum walk, shape `[batch, output_nodes, hidden_dim]`.
    """
    if _fused_qwt_tokenizer is None:
        raise NotImplementedError(
            "The fused_qwt_tokenizer custom op is unavailable. "
            "Rebuild the C++ ops via build_ops.sh."
        )

    return _fused_qwt_tokenizer(
        input_data,
        low_pass_filter,
        high_pass_filter,
        mask,
        evolution_time,
        ctqw_steps,
        epsilon=epsilon,
        num_wavelet_levels=num_levels,
        # Phase 17: Enhancement attributes
        use_lifting_scheme=use_lifting_scheme,
        pade_order=pade_order,
        use_jacobi_preconditioner=use_jacobi_preconditioner,
        skip_stride=skip_stride,
        max_skips_per_node=max_skips_per_node,
    )


@ops.RegisterGradient("FusedQwtTokenizer")
def _fused_qwt_tokenizer_gradients(op, grad_approx, grad_detail, grad_qwt):
    if _fused_qwt_tokenizer_grad is None:
        raise NotImplementedError("The fused_qwt_tokenizer_grad custom op is unavailable.")
    grad_inputs = _fused_qwt_tokenizer_grad(
        grad_approx,
        grad_detail,
        grad_qwt,
        op.inputs[0],  # input_data
        op.inputs[1],  # low_pass_filter
        op.inputs[2],  # high_pass_filter
        op.inputs[3],  # mask
        op.inputs[4],  # evolution_time
        op.inputs[5],  # ctqw_steps
        op.outputs[0],  # approx_coeffs
        op.outputs[1],  # detail_coeffs
        epsilon=op.get_attr("epsilon"),
        num_wavelet_levels=op.get_attr("num_wavelet_levels"),  # Phase 10.3
        # Phase 17: Forward enhancement attributes to gradient op
        use_lifting_scheme=op.get_attr("use_lifting_scheme"),
        pade_order=op.get_attr("pade_order"),
        use_jacobi_preconditioner=op.get_attr("use_jacobi_preconditioner"),
        skip_stride=op.get_attr("skip_stride"),
        max_skips_per_node=op.get_attr("max_skips_per_node"),
    )
    return (
        grad_inputs[0],
        grad_inputs[1],
        grad_inputs[2],
        None,
        grad_inputs[3],
        None,
    )


def fused_qwt_tokenizer_available() -> bool:
    """Check if the fused QWT tokenizer operation is available."""
    return _fused_qwt_tokenizer is not None


def fused_qwt_tokenizer_grad_available() -> bool:
    """Check if the fused QWT tokenizer gradient operation is available."""
    return _fused_qwt_tokenizer_grad is not None


def fused_qwt_tokenizer_op_path() -> str | None:
    """Get the path to the loaded operation library."""
    return _OP_LIBRARY_PATH


__all__ = [
    "fused_qwt_tokenizer",
    "fused_qwt_tokenizer_available",
    "fused_qwt_tokenizer_grad_available",
    "fused_qwt_tokenizer_op_path",
]
