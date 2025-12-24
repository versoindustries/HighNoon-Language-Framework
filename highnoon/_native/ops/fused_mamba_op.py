# highnoon/_native/ops/fused_mamba_op.py
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

"""Python wrapper for FusedMambaCore C++ kernel with Phase 17 Enhancements.

This module provides the Python interface for the Mamba SSM C++ kernel
with SIMD optimizations and the following enhancements:
    1. VQC-Gated Selective Scan
    2. AVX2/AVX512 Parallel Scan
    3. SSD Chunk Processing (Mamba-2)
    4. MPS-Factorized State (via separate kernel)
    5. Dynamic State Evolution (RWKV-7)
    6. Quantum Superposition Paths

Usage:
    from highnoon._native.ops.fused_mamba_op import fused_mamba_core

    output, h_final = fused_mamba_core(
        x_c, z, conv_filter, conv_bias,
        dt, a_log, b_proj, c_proj, d_skip,
        conv_dim=4,
        use_parallel_scan=True,  # Enable SIMD parallel scan
    )
"""

from __future__ import annotations

import logging

import tensorflow as tf

from highnoon._native import get_op

logger = logging.getLogger(__name__)

# Load the C++ library
_lib = get_op("fused_mamba")
_fused_mamba_core_op = _lib.fused_mamba_core if _lib else None
_fused_mamba_core_grad_op = _lib.fused_mamba_core_grad if _lib else None


def fused_mamba_core_available() -> bool:
    """Check if C++ kernel is available."""
    return _fused_mamba_core_op is not None


def fused_mamba_core(
    x_c: tf.Tensor,
    z: tf.Tensor,
    conv_filter: tf.Tensor,
    conv_bias: tf.Tensor,
    dt: tf.Tensor,
    a_log: tf.Tensor,
    b_proj: tf.Tensor,
    c_proj: tf.Tensor,
    d_skip: tf.Tensor,
    conv_dim: int = 4,
    # Enhancement 1: VQC-Gated Selective Scan
    use_vqc_gate: bool = False,
    vqc_angles: tf.Tensor | None = None,
    vqc_num_layers: int = 2,
    # Enhancement 2: Parallel SIMD Scan
    use_parallel_scan: bool = True,
    parallel_chunk_size: int = 256,
    # Enhancement 3: SSD Chunk Processing
    use_ssd_chunks: bool = False,
    ssd_chunk_size: int = 128,
    # Enhancement 5: Dynamic State Evolution
    use_dynamic_state: bool = False,
    dse_rank: int = 32,
    # Enhancement 6: Superposition Paths
    use_superposition: bool = False,
    superposition_dim: int = 4,
    superposition_temperature: float = 1.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """C++-accelerated Mamba SSM core operation with enhancements.

    Implements the full Mamba forward pass as a single fused kernel:
        x_conv = silu(conv1d(x_c, filter, bias))
        y = ssm_scan(x_conv, dt, a_log, b, c, d)  # Optionally parallel
        output = y * silu(z)

    Args:
        x_c: Conv input [batch, seq_len, d_inner].
        z: Gate input [batch, seq_len, d_inner].
        conv_filter: Depthwise conv filter [conv_dim, 1, d_inner].
        conv_bias: Conv bias [d_inner].
        dt: Discretization timesteps [batch, seq_len, d_inner].
        a_log: Log decay rates [d_inner, state_dim].
        b_proj: B projection [batch, seq_len, state_dim].
        c_proj: C projection [batch, seq_len, state_dim].
        d_skip: Skip connection [d_inner].
        conv_dim: Convolution kernel size.
        use_vqc_gate: Enable VQC-gated delta (Enhancement 1).
        vqc_angles: VQC rotation angles [num_layers, 2].
        vqc_num_layers: Number of VQC layers.
        use_parallel_scan: Enable AVX parallel scan (Enhancement 2).
        parallel_chunk_size: Chunk size for parallel processing.
        use_ssd_chunks: Enable SSD chunk processing (Enhancement 3).
        ssd_chunk_size: SSD chunk size.
        use_dynamic_state: Enable dynamic A matrix (Enhancement 5).
        dse_rank: Low-rank dimension for dynamic state.
        use_superposition: Enable superposition paths (Enhancement 6).
        superposition_dim: Number of parallel paths.
        superposition_temperature: Collapse temperature.

    Returns:
        Tuple of:
            output: SSM output [batch, seq_len, d_inner]
            h_final: Final hidden state [batch, d_inner, state_dim]

    Raises:
        RuntimeError: If C++ kernel is not available.
    """
    if _fused_mamba_core_op is None:
        raise RuntimeError(
            "FusedMambaCore C++ kernel not available. "
            "Build the native library with: cd highnoon/_native && ./build_secure.sh"
        )

    # Ensure float32 for C++ kernel
    x_c = tf.cast(x_c, tf.float32)
    z = tf.cast(z, tf.float32)
    conv_filter = tf.cast(conv_filter, tf.float32)
    conv_bias = tf.cast(conv_bias, tf.float32)
    dt = tf.cast(dt, tf.float32)
    a_log = tf.cast(a_log, tf.float32)
    b_proj = tf.cast(b_proj, tf.float32)
    c_proj = tf.cast(c_proj, tf.float32)
    d_skip = tf.cast(d_skip, tf.float32)

    # VQC angles - provide empty tensor if not using VQC gate
    if vqc_angles is None:
        vqc_angles = tf.zeros([vqc_num_layers, 2], dtype=tf.float32)
    else:
        vqc_angles = tf.cast(vqc_angles, tf.float32)

    @tf.custom_gradient
    def _fused_mamba_core_inner(x_c_t, z_t, conv_f, conv_b, dt_t, a_log_t, b_p, c_p, d_s, vqc_a):
        output, h_final, conv_cache = _fused_mamba_core_op(
            x_c_t,
            z_t,
            conv_f,
            conv_b,
            dt_t,
            a_log_t,
            b_p,
            c_p,
            d_s,
            vqc_a,
            conv_dim=conv_dim,
            use_vqc_gate=use_vqc_gate,
            vqc_num_layers=vqc_num_layers,
            use_parallel_scan=use_parallel_scan,
            parallel_chunk_size=parallel_chunk_size,
            use_ssd_chunks=use_ssd_chunks,
            ssd_chunk_size=ssd_chunk_size,
            use_dynamic_state=use_dynamic_state,
            dse_rank=dse_rank,
            use_superposition=use_superposition,
            superposition_dim=superposition_dim,
            superposition_temperature=superposition_temperature,
        )

        def grad(grad_output, grad_h_final):
            if _fused_mamba_core_grad_op is None:
                raise RuntimeError("FusedMambaCoreGrad C++ kernel not available.")
            grads = _fused_mamba_core_grad_op(
                grad_output,
                grad_h_final,
                x_c_t,
                z_t,
                conv_f,
                conv_b,
                dt_t,
                a_log_t,
                b_p,
                c_p,
                d_s,
                conv_cache,
                conv_dim=conv_dim,
            )
            # Returns: (grad_x_c, grad_z, grad_conv_filter, grad_conv_bias,
            #           grad_dt, grad_a_log, grad_b_proj, grad_c_proj, grad_d_skip)
            # Add None for vqc_angles gradient (not trained through this path)
            return tuple(grads) + (None,)

        return (output, h_final), grad

    return _fused_mamba_core_inner(
        x_c, z, conv_filter, conv_bias, dt, a_log, b_proj, c_proj, d_skip, vqc_angles
    )


__all__ = ["fused_mamba_core", "fused_mamba_core_available"]
