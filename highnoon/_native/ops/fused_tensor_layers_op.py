# highnoon/_native/ops/fused_tensor_layers_op.py
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

"""Python wrapper for Fused Tensor Layer C++ kernels.

Provides access to optimized Tucker and Tensor-Ring operations.
"""

from __future__ import annotations

import logging

import tensorflow as tf

from highnoon._native import get_op

logger = logging.getLogger(__name__)

_tensor_layers_module = None


def _get_ops():
    global _tensor_layers_module
    if _tensor_layers_module is None:
        _tensor_layers_module = get_op("fused_tensor_layers")
        if _tensor_layers_module is not None:
            logger.info(f"Loaded fused_tensor_layers. Available ops: {dir(_tensor_layers_module)}")
            print(f"--- Fused Tensor Layers Ops: {dir(_tensor_layers_module)} ---")
    return _tensor_layers_module


def fused_tensor_layers_available() -> bool:
    """Check if Tensor Layer C++ ops are available."""
    return _get_ops() is not None


def _tucker_forward_reference(
    x: tf.Tensor,
    u_in: tf.Tensor,
    core: tf.Tensor,
    u_out: tf.Tensor,
    bias: tf.Tensor | None,
) -> tf.Tensor:
    x_proj = tf.matmul(x, u_in)
    core_proj = tf.matmul(x_proj, tf.transpose(core))
    y = tf.matmul(core_proj, tf.transpose(u_out))
    if bias is not None:
        y = tf.nn.bias_add(y, bias)
    return y


def _tensor_ring_forward_reference(
    inputs: list[tf.Tensor],
    cores: list[tf.Tensor],
    bias: tf.Tensor | None,
    ring_rank: int,
    local_dims_in: list[int],
    local_dims_out: list[int],
) -> tf.Tensor:
    batch_size = tf.shape(inputs[0])[0]
    num_cores = len(cores)

    batch_matrices = []
    for i in range(num_cores):
        d_out_i = local_dims_out[i]
        d_in_i = local_dims_in[i]
        core = tf.reshape(cores[i], [ring_rank, d_out_i, d_in_i, ring_rank])
        site = tf.einsum("aoib,ci->caob", core, inputs[i])
        batch_matrices.append(site)

    b_sums = [tf.reduce_sum(m, axis=2) for m in batch_matrices]

    eye = tf.eye(ring_rank, batch_shape=[batch_size])
    prefixes = [eye]
    for m in range(num_cores):
        prefixes.append(tf.matmul(prefixes[-1], b_sums[m]))

    suffixes = [eye]
    for m in range(num_cores - 1, -1, -1):
        suffixes.insert(0, tf.matmul(b_sums[m], suffixes[0]))

    output_parts = []
    for m in range(num_cores):
        part = tf.einsum("bij,bjok,bki->bo", prefixes[m], batch_matrices[m], suffixes[m + 1])
        output_parts.append(part)

    output = tf.concat(output_parts, axis=-1)
    if bias is not None:
        output = tf.nn.bias_add(output, bias)

    # Gradient stabilization: clip to prevent explosion that causes NaN gradients
    output = tf.clip_by_value(output, -1e6, 1e6)

    # GRADIENT FIX: Removed tf.where(is_finite, x, zeros) which blocked gradient flow
    # for NaN elements (gradient of tf.where for false branch is zero).
    # Let the clip handle stabilization; if NaNs still occur, fix the root cause.

    return output


def fused_tucker_forward(
    x: tf.Tensor,
    u_in: tf.Tensor,
    core: tf.Tensor,
    u_out: tf.Tensor,
    bias: tf.Tensor | None = None,
) -> tf.Tensor:
    """Fused Tucker decomposition forward pass.

    y = ((x @ U_in) @ G^T) @ U_out^T + b

    Args:
        x: Input tensor [batch, D_in]
        u_in: Input factor [D_in, R_in]
        core: Core tensor [R_out, R_in]
        u_out: Output factor [D_out, R_out]
        bias: Optional bias [D_out]

    Returns:
        Output tensor [batch, D_out]
    """
    ops = _get_ops()
    if ops is None:
        return _tucker_forward_reference(x, u_in, core, u_out, bias)

    bias_provided = bias is not None
    if bias is None:
        bias = tf.zeros([tf.shape(u_out)[0]], dtype=x.dtype)

    @tf.custom_gradient
    def _fused_tucker_inner(x_in, u_in_in, core_in, u_out_in, bias_in):
        output = ops.fused_tucker_forward(x_in, u_in_in, core_in, u_out_in, bias_in)

        def grad(dy, variables=None):
            """Gradient function with variables kwarg for tf.custom_gradient compliance."""
            with tf.GradientTape() as tape:
                tape.watch([x_in, u_in_in, core_in, u_out_in])
                if bias_provided:
                    tape.watch(bias_in)
                # Also watch any passed variables
                if variables:
                    tape.watch(variables)
                ref = _tucker_forward_reference(
                    x_in, u_in_in, core_in, u_out_in, bias_in if bias_provided else None
                )
            watched = [x_in, u_in_in, core_in, u_out_in]
            if bias_provided:
                watched.append(bias_in)
            grads = tape.gradient(ref, watched, output_gradients=dy)
            if bias_provided:
                grad_x, grad_u_in, grad_core, grad_u_out, grad_bias = grads
            else:
                grad_x, grad_u_in, grad_core, grad_u_out = grads
                grad_bias = None

            input_grads = (grad_x, grad_u_in, grad_core, grad_u_out, grad_bias)

            # Compute variable gradients if variables were passed
            if variables:
                var_grads = tape.gradient(ref, variables, output_gradients=dy)
                return input_grads, var_grads
            return input_grads

        return output, grad

    return _fused_tucker_inner(x, u_in, core, u_out, bias)


def fused_tensor_ring_forward(
    inputs: list[tf.Tensor],
    cores: list[tf.Tensor],
    bias: tf.Tensor | None,
    ring_rank: int,
    local_dims_in: list[int],
    local_dims_out: list[int],
) -> tf.Tensor:
    """Fused Tensor-Ring decomposition forward pass with proper trace contraction.

    Args:
        inputs: List of input tensors [batch, local_d_in]
        cores: List of ring cores [ring_rank, local_d_out*local_d_in, ring_rank]
        bias: Optional bias [total_D_out]
        ring_rank: Bond dimension R
        local_dims_in: List of d_in_i
        local_dims_out: List of d_out_i

    Returns:
        Output tensor [batch, total_D_out]
    """
    ops = _get_ops()
    if ops is None:
        raise RuntimeError("FusedTensorRingForward requires C++ ops for proper trace contraction.")

    bias_provided = bias is not None
    if bias is None:
        total_D_out = sum(local_dims_out)
        bias = tf.zeros([total_D_out], dtype=inputs[0].dtype)

    @tf.custom_gradient
    def _fused_tensor_ring_inner(inputs_in, cores_in, bias_in):
        output = ops.fused_tensor_ring_forward(
            inputs=inputs_in,
            cores=cores_in,
            bias=bias_in,
            local_dims_in=local_dims_in,
            local_dims_out=local_dims_out,
            ring_rank=ring_rank,
        )

        def grad(dy, variables=None):
            """Gradient function with variables kwarg for tf.custom_gradient compliance.

            When TensorFlow detects that the function uses tf.Variables (like the
            ring cores), it passes them via the 'variables' kwarg. We must compute
            and return gradients for these variables as a second return value.

            GRADIENT FIX: The cores_in list contains tensor values read from variables.
            When TensorFlow calls this grad function, 'variables' contains the actual
            tf.Variables from which cores_in was read. We need to map grad_cores
            (computed for cores_in tensors) to the corresponding variables.
            """
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(inputs_in)
                tape.watch(cores_in)
                if bias_provided:
                    tape.watch(bias_in)
                # Also watch any passed variables (the actual tf.Variables)
                if variables:
                    tape.watch(variables)
                ref = _tensor_ring_forward_reference(
                    inputs_in,
                    cores_in,
                    bias_in if bias_provided else None,
                    ring_rank,
                    local_dims_in,
                    local_dims_out,
                )
            watched = [inputs_in, cores_in]
            if bias_provided:
                watched.append(bias_in)
            grads = tape.gradient(ref, watched, output_gradients=dy)
            grad_inputs = grads[0]
            grad_cores = grads[1]  # This is a list of gradients for cores_in
            grad_bias = grads[2] if bias_provided else None

            input_grads = (grad_inputs, grad_cores, grad_bias)

            # Compute variable gradients if variables were passed
            if variables:
                # GRADIENT FIX: Map grad_cores to variables by matching ring_core names
                # The grad_cores list corresponds to cores_in which came from variables
                var_grads = []
                for v in variables:
                    v_name = v.name.lower()
                    found_grad = None

                    # Check if this is a ring_core variable
                    if "ring_core" in v_name:
                        # Extract index from variable name (e.g., "ring_core_0:0" -> 0)
                        for idx, _c in enumerate(cores_in):
                            if f"ring_core_{idx}" in v_name:
                                # This variable corresponds to cores_in[idx]
                                if isinstance(grad_cores, list):
                                    found_grad = grad_cores[idx]
                                else:
                                    # grad_cores is a single tensor - shouldn't happen for lists
                                    found_grad = tape.gradient(ref, v, output_gradients=dy)
                                break
                    elif "bias" in v_name and bias_provided:
                        found_grad = grad_bias

                    if found_grad is None:
                        # Fallback: compute gradient directly for this variable
                        found_grad = tape.gradient(ref, v, output_gradients=dy)

                    var_grads.append(found_grad)

                del tape  # Release persistent tape
                return input_grads, var_grads

            del tape  # Release persistent tape
            return input_grads

        return output, grad

    return _fused_tensor_ring_inner(inputs, cores, bias)
