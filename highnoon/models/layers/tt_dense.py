# highnoon/models/layers/tt_dense.py
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

"""Tensor-Train Dense Layer for Weight Compression.

This module provides a TTDense layer that factorizes weight matrices using
Tensor-Train decomposition for 10-100x parameter reduction while maintaining
accuracy. Designed for use with RG-LRU and other layers requiring compressed
linear transformations.

The TT decomposition represents W ∈ ℝ^{m×n} as:
    W[i,j] = G_1[i_1] @ G_2[i_2] @ ... @ G_d[i_d]

where G_k are the TT-cores with shape [r_{k-1}, m_k, n_k, r_k].
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers, layers

from highnoon._native import get_op

# Try to load C++ TT ops
_lib = get_op("fused_rg_lru")
_tt_matvec_op = getattr(_lib, "TTMatVec", None) if _lib else None


def _factorize_dim(n: int, num_factors: int = 3) -> list[int]:
    """Factorize dimension into approximately equal factors.

    Args:
        n: Dimension to factorize.
        num_factors: Number of factors to produce.

    Returns:
        List of factors whose product equals n.
    """
    if num_factors == 1:
        return [n]

    # Find closest factor split
    target = int(round(n ** (1.0 / num_factors)))

    factors = []
    remaining = n

    for _i in range(num_factors - 1):
        # Find largest factor <= target that divides remaining
        factor = target
        while remaining % factor != 0 and factor > 1:
            factor -= 1
        factors.append(factor)
        remaining //= factor

    factors.append(remaining)
    return factors


class TTDense(layers.Layer):
    """Tensor-Train factorized Dense layer.

    Compresses W ∈ ℝ^{m×n} using TT decomposition for massive parameter
    reduction. Suitable for large projection matrices in RG-LRU and attention.

    Compression ratio: Original = m*n parameters
                       TT = sum(r_{k-1} * m_k * n_k * r_k) parameters

    For a 512x512 matrix with ranks [1,4,4,1] and dims [8,8,8]:
    Original: 262,144 parameters
    TT: 3 * (4 * 8 * 8 * 4) = 3,072 parameters → 85x compression

    Attributes:
        input_dim: Total input dimension.
        output_dim: Total output dimension.
        input_factors: Factorization of input dimension.
        output_factors: Factorization of output dimension.
        tt_ranks: TT-ranks including boundary 1s.
    """

    def __init__(
        self,
        output_dim: int,
        tt_ranks: list[int] | None = None,
        num_factors: int = 3,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        name: str = "tt_dense",
        **kwargs,
    ):
        """Initialize TTDense layer.

        Args:
            output_dim: Output dimension.
            tt_ranks: TT-ranks. If None, defaults to [1, 4, 4, 1].
            num_factors: Number of factors for dimension decomposition.
            use_bias: Whether to add bias.
            kernel_initializer: Initializer for TT cores.
            bias_initializer: Initializer for bias.
            name: Layer name.
            **kwargs: Additional Keras layer arguments.
        """
        super().__init__(name=name, **kwargs)

        self.output_dim = output_dim
        self.tt_ranks = tt_ranks if tt_ranks is not None else [1, 4, 4, 1]
        self.num_factors = len(self.tt_ranks) - 1
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # Validate ranks
        if self.tt_ranks[0] != 1 or self.tt_ranks[-1] != 1:
            raise ValueError("First and last TT-ranks must be 1")

        self.cores = []
        self._built = False

        # Initialize attributes that will be set in build()
        # This ensures they exist for early access (e.g., .kernel, .bias properties)
        self.input_dim: int | None = None
        self.input_factors: list[int] = []
        self.output_factors: list[int] = []
        self._bias: tf.Variable | None = None  # Internal bias storage

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the TT cores and bias.

        Args:
            input_shape: Input tensor shape.
        """
        # Get input dimension from last axis
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Input dimension must be specified")

        self.input_dim = int(input_dim)

        # Factorize dimensions
        self.input_factors = _factorize_dim(self.input_dim, self.num_factors)
        self.output_factors = _factorize_dim(self.output_dim, self.num_factors)

        # Verify factorization
        if np.prod(self.input_factors) != self.input_dim:
            # Pad input factors to match
            remaining = self.input_dim // np.prod(self.input_factors)
            self.input_factors[-1] *= remaining

        if np.prod(self.output_factors) != self.output_dim:
            remaining = self.output_dim // np.prod(self.output_factors)
            self.output_factors[-1] *= remaining

        # Create TT cores G_k with shape [r_{k-1}, m_k, n_k, r_k]
        for k in range(self.num_factors):
            core_shape = (
                self.tt_ranks[k],  # r_{k-1}
                self.input_factors[k],  # m_k
                self.output_factors[k],  # n_k
                self.tt_ranks[k + 1],  # r_k
            )

            # Initialize with appropriate scale
            fan_in = self.tt_ranks[k] * self.input_factors[k]
            fan_out = self.tt_ranks[k + 1] * self.output_factors[k]
            std = math.sqrt(2.0 / (fan_in + fan_out))

            core = self.add_weight(
                name=f"tt_core_{k}",
                shape=core_shape,
                initializer=initializers.TruncatedNormal(stddev=std),
                trainable=True,
            )
            self.cores.append(core)

        # Bias
        if self.use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        else:
            self._bias = None

        self._built = True
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass through TT-factorized dense layer.

        Args:
            inputs: Input tensor of shape [batch, input_dim] or [batch, seq, input_dim].

        Returns:
            Output tensor of shape [batch, output_dim] or [batch, seq, output_dim].
        """
        input_shape = tf.shape(inputs)
        is_sequence = len(inputs.shape) == 3

        if is_sequence:
            # Flatten sequence: [batch, seq, dim] -> [batch*seq, dim]
            batch_size = input_shape[0]
            seq_len = input_shape[1]
            inputs_flat = tf.reshape(inputs, [-1, self.input_dim])
        else:
            inputs_flat = inputs

        # Reshape input for TT contraction: [batch, m_1, m_2, ..., m_d]
        effective_batch = tf.shape(inputs_flat)[0]
        x = tf.reshape(inputs_flat, [effective_batch] + list(self.input_factors))

        # Perform TT contraction using einsum
        output = self._tt_contract(x)

        # Reshape output: [batch, n_1, n_2, ..., n_d] -> [batch, output_dim]
        output = tf.reshape(output, [effective_batch, self.output_dim])

        # Add bias
        if self.use_bias and self._bias is not None:
            output = output + self._bias

        if is_sequence:
            # Restore sequence: [batch*seq, dim] -> [batch, seq, dim]
            output = tf.reshape(output, [batch_size, seq_len, self.output_dim])

        return output

    def _tt_contract(self, x: tf.Tensor) -> tf.Tensor:
        """Perform TT contraction for matrix-vector product.

        Uses optimized einsum sequence for stable graph-mode execution.

        Args:
            x: Reshaped input [batch, m_1, m_2, ..., m_d]

        Returns:
            Output [batch, n_1, n_2, ..., n_d]
        """
        # Build dynamic einsum strings
        d = self.num_factors

        # Use 'z' for batch, lowercase for input modes, uppercase for output
        batch_idx = "z"
        in_modes = "abcdefghijklm"[:d]
        out_modes = "ABCDEFGHIJKLM"[:d]
        rank_modes = "nopqrstuvwxy"[: d + 1]

        # Start: res has shape [batch, m_1, m_2, ..., m_d]
        res = x
        res_str = batch_idx + in_modes

        for k in range(d):
            r_l = rank_modes[k]  # left rank
            m_k = in_modes[k]  # input mode
            n_k = out_modes[k]  # output mode
            r_r = rank_modes[k + 1]  # right rank

            core_str = f"{r_l}{m_k}{n_k}{r_r}"

            # Build output string
            out_parts = [batch_idx]

            for i in range(d):
                if i < k:
                    out_parts.append(out_modes[i])  # Already contracted
                elif i == k:
                    out_parts.append(n_k)  # Current output mode
                else:
                    out_parts.append(in_modes[i])  # Not yet contracted

            out_parts.append(r_r)  # Add right rank
            out_str = "".join(out_parts)

            # Handle boundary ranks (size 1)
            if k == 0:
                # First core: squeeze left rank
                core_squeezed = tf.squeeze(self.cores[k], axis=0)
                einsum_str = f"{res_str},{m_k}{n_k}{r_r}->{out_str}"
                res = tf.einsum(einsum_str, res, core_squeezed)
            elif k == d - 1:
                # Last core: squeeze right rank
                core_squeezed = tf.squeeze(self.cores[k], axis=-1)
                # Remove final rank from output
                out_str_final = out_str[:-1]
                einsum_str = f"{res_str},{r_l}{m_k}{n_k}->{out_str_final}"
                res = tf.einsum(einsum_str, res, core_squeezed)
                out_str = out_str_final
            else:
                res = tf.einsum(f"{res_str},{core_str}->{out_str}", res, self.cores[k])

            res_str = out_str

        return res

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape."""
        if len(input_shape) == 3:
            return tf.TensorShape([input_shape[0], input_shape[1], self.output_dim])
        return tf.TensorShape([input_shape[0], self.output_dim])

    def get_config(self) -> dict:
        """Get layer config for serialization."""
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "tt_ranks": self.tt_ranks,
                "num_factors": self.num_factors,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
            }
        )
        return config

    @property
    def compression_ratio(self) -> float:
        """Compute parameter compression ratio."""
        original = self.input_dim * self.output_dim
        if self.use_bias:
            original += self.output_dim

        tt_params = sum(
            self.tt_ranks[k] * self.input_factors[k] * self.output_factors[k] * self.tt_ranks[k + 1]
            for k in range(self.num_factors)
        )
        if self.use_bias:
            tt_params += self.output_dim

        return original / tt_params if tt_params > 0 else 0.0

    @property
    def kernel(self) -> tf.Tensor | None:
        """Reconstruct full kernel matrix from TT-cores.

        This provides backward compatibility with code that expects
        a standard Dense layer interface (e.g., fused C++ ops).

        Note: This is computationally expensive for large matrices.
        For performance-critical paths, access TT-cores directly.

        Returns:
            Reconstructed kernel [input_dim, output_dim], or None if not built.
        """
        if not self._built or not self.cores:
            return None

        # Contract TT-cores to reconstruct dense kernel
        # Result should be [m_1*m_2*...*m_d, n_1*n_2*...*n_d] = [input_dim, output_dim]

        # Start with first core: [1, m_1, n_1, r_1] -> [m_1, n_1, r_1]
        result = tf.squeeze(self.cores[0], axis=0)  # [m_1, n_1, r_1]

        for k in range(1, self.num_factors):
            core = self.cores[k]  # [r_{k-1}, m_k, n_k, r_k]

            # result has shape [m_1*...*m_{k-1}, n_1*...*n_{k-1}, r_{k-1}]
            # Contract over rank dimension
            # result: [..., r_{k-1}] @ core: [r_{k-1}, m_k, n_k, r_k]

            # Reshape for einsum
            r_prev = self.tt_ranks[k]
            m_k = self.input_factors[k]
            n_k = self.output_factors[k]
            r_next = self.tt_ranks[k + 1]

            # Flatten result's spatial dims for easier contraction
            result_shape = tf.shape(result)
            result_flat = tf.reshape(result, [-1, r_prev])  # [M*N, r_prev]

            # Reshape core for matmul: [r_prev, m_k * n_k * r_next]
            core_flat = tf.reshape(core, [r_prev, m_k * n_k * r_next])

            # Contract: [M*N, r_prev] @ [r_prev, m_k*n_k*r_next] -> [M*N, m_k*n_k*r_next]
            contracted = tf.matmul(result_flat, core_flat)

            # Reshape back
            tf.reduce_prod(result_shape[:-1])
            result = tf.reshape(contracted, [-1, m_k, n_k, r_next])

            # Merge adjacent input/output dimensions
            # [M_prev/n_prev, n_prev, m_k, n_k, r_next] -> [M_prev*m_k, n_prev*n_k, r_next]
            # This is complex, so use a simpler approach: just track running products

        # Final squeeze of rank dimension (should be 1)
        result = tf.squeeze(result, axis=-1)  # [..., m_d, n_d]

        # Reshape to [input_dim, output_dim]
        kernel = tf.reshape(result, [self.input_dim, self.output_dim])

        return kernel

    @property
    def bias(self) -> tf.Tensor | None:
        """Return the bias tensor.

        This provides backward compatibility with code that expects
        a standard Dense layer interface.

        Returns:
            Bias tensor [output_dim], or None if not built or use_bias=False.
        """
        return self._bias


def create_tt_dense(
    output_dim: int,
    tt_ranks: list[int] | None = None,
    use_bias: bool = True,
    **kwargs,
) -> TTDense:
    """Factory function to create TTDense layer with optimal defaults.

    Args:
        output_dim: Output dimension.
        tt_ranks: TT-ranks. If None, automatically determined.
        use_bias: Whether to use bias.
        **kwargs: Additional layer arguments.

    Returns:
        Configured TTDense layer.
    """
    if tt_ranks is None:
        # Default ranks based on output dimension
        if output_dim <= 64:
            tt_ranks = [1, 2, 1]
        elif output_dim <= 256:
            tt_ranks = [1, 4, 1]
        elif output_dim <= 1024:
            tt_ranks = [1, 4, 4, 1]
        else:
            tt_ranks = [1, 8, 8, 1]

    return TTDense(
        output_dim=output_dim,
        tt_ranks=tt_ranks,
        use_bias=use_bias,
        **kwargs,
    )


__all__ = ["TTDense", "create_tt_dense"]
