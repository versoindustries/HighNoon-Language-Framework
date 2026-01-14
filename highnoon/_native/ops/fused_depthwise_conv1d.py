# src/ops/fused_depthwise_conv1d.py
# Copyright 2025 Verso Industries
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

import logging
import os

import tensorflow as tf

from highnoon._native.ops.lib_loader import resolve_op_library

# --- Setup ---
logger = logging.getLogger(__name__)

# --- Load the Custom Operator ---
_op_module = None
fused_depthwise_conv1d_op = None
fused_depthwise_conv1d_grad_op = None


def _load_op():
    """Load the FusedDepthwiseConv1D operation from the consolidated library."""
    global _op_module, fused_depthwise_conv1d_op, fused_depthwise_conv1d_grad_op

    # First try to load from the consolidated _highnoon_core.so library
    consolidated_lib_path = resolve_op_library(__file__, "_highnoon_core.so")
    if os.path.exists(consolidated_lib_path):
        try:
            _op_module = tf.load_op_library(consolidated_lib_path)
            # Access the ops directly from the loaded module
            fused_depthwise_conv1d_op = _op_module.FusedDepthwiseConv1D
            fused_depthwise_conv1d_grad_op = _op_module.FusedDepthwiseConv1DGrad
            logger.info("Loaded FusedDepthwiseConv1D from consolidated _highnoon_core.so")
            return
        except (tf.errors.NotFoundError, OSError, AttributeError) as e:
            logger.debug(f"Could not load from consolidated library: {e}")

    # Fallback: try individual .so file (legacy path)
    _op_lib_path = resolve_op_library(__file__, "_fused_depthwise_conv1d.so")
    if os.path.exists(_op_lib_path):
        try:
            _op_module = tf.load_op_library(_op_lib_path)
            fused_depthwise_conv1d_op = _op_module.FusedDepthwiseConv1D
            fused_depthwise_conv1d_grad_op = _op_module.FusedDepthwiseConv1DGrad
            logger.info("Loaded FusedDepthwiseConv1D from individual .so file")
            return
        except (tf.errors.NotFoundError, OSError, AttributeError) as e:
            logger.error(f"Failed to load FusedDepthwiseConv1D: {e}")
    else:
        logger.warning(f"Custom C++ op files not found. Looked for: {consolidated_lib_path}")


_load_op()


@tf.autograph.experimental.do_not_convert
def fused_depthwise_conv1d(input_tensor, filter_tensor, bias_tensor, stride, padding):
    """
    Python wrapper for the FusedDepthwiseConv1D custom C++ operator.
    This function mimics a 1D depthwise convolution with 'causal' padding.

    Note: This function is decorated with @tf.autograph.experimental.do_not_convert
    because it wraps a custom C++ operator whose source code is not accessible to AutoGraph.

    Args:
        input_tensor: Input tensor of shape [batch, width, channels]
        filter_tensor: Filter tensor of shape [filter_width, 1, channels]
        bias_tensor: Bias tensor of shape [channels]
        stride: Stride value (must be a concrete integer or int tensor)
        padding: Padding mode string ("SAME" or "VALID" or string tensor)

    Returns:
        Output tensor after depthwise convolution
    """
    if fused_depthwise_conv1d_op is None:
        raise NotImplementedError(
            "The C++ FusedDepthwiseConv1D operator could not be loaded. "
            "Please ensure it has been compiled correctly."
        )

    # Convert stride to int (handle both Python int and tf.Tensor)
    if isinstance(stride, tf.Tensor):
        stride_val = tf.get_static_value(stride)
        if stride_val is None:
            # In graph mode, try to extract the value if it's a constant
            if hasattr(stride, "numpy"):
                try:
                    stride_val = int(stride.numpy())
                except (AttributeError, TypeError, RuntimeError) as err:
                    raise ValueError(
                        "'stride' must be a concrete integer for fused_depthwise_conv1d"
                    ) from err
            else:
                raise ValueError("'stride' must be a concrete integer for fused_depthwise_conv1d")
    else:
        stride_val = int(stride)

    # Convert padding to string (handle both Python str and tf.Tensor)
    if isinstance(padding, tf.Tensor):
        padding_val = tf.get_static_value(padding)
        if padding_val is None:
            # In graph mode, try to extract the value if it's a constant
            if hasattr(padding, "numpy"):
                try:
                    padding_val = padding.numpy()
                    # Handle numpy array wrapping (0-dimensional array)
                    if hasattr(padding_val, "item"):
                        padding_val = padding_val.item()
                    if isinstance(padding_val, bytes):
                        padding_val = padding_val.decode("utf-8")
                    else:
                        padding_val = str(padding_val)
                except (AttributeError, TypeError, RuntimeError) as err:
                    raise ValueError(
                        "'padding' must be a concrete string for fused_depthwise_conv1d"
                    ) from err
            else:
                raise ValueError("'padding' must be a concrete string for fused_depthwise_conv1d")
        else:
            # tf.get_static_value returned a value, handle it
            # It might still be a numpy array wrapper
            if hasattr(padding_val, "item"):
                padding_val = padding_val.item()
            if isinstance(padding_val, bytes):
                padding_val = padding_val.decode("utf-8")
            else:
                padding_val = str(padding_val)
    else:
        padding_val = str(padding)

    # Use a closure to pass stride_val and padding_val to the custom gradient function
    # This avoids counting them as differentiable parameters
    @tf.custom_gradient
    def _compute_with_grad(input_t, filter_t, bias_t):
        """
        Internal function with custom gradient that closes over stride_val and padding_val.
        Only the 3 tensor inputs (input, filter, bias) are tracked for differentiation.
        """
        output = fused_depthwise_conv1d_op(
            input=input_t,
            filter=filter_t,
            bias=bias_t,
            stride=stride_val,  # Closed over from outer scope
            padding=padding_val,  # Closed over from outer scope
        )

        def grad_fn(grad_output, variables=None):
            """
            Gradient function that calls the custom C++ backward kernel.

            Returns gradients for the 3 tensor inputs: input_tensor, filter_tensor, bias_tensor.
            stride_val and padding_val are closed over from the outer scope and are not
            differentiable parameters.
            """
            if fused_depthwise_conv1d_grad_op is None:
                raise NotImplementedError(
                    "The C++ FusedDepthwiseConv1DGrad operator could not be loaded."
                )

            grad_input, grad_filter, grad_bias = fused_depthwise_conv1d_grad_op(
                grad_output=grad_output,
                input=input_t,
                filter=filter_t,
                stride=tf.convert_to_tensor(stride_val, dtype=tf.int32),
                padding=tf.convert_to_tensor(padding_val),
            )

            # Return gradients for the 3 differentiable tensor inputs
            input_grads = (grad_input, grad_filter, grad_bias)

            # GRADIENT FIX: Map C++ gradient outputs to captured tf.Variables by name
            # Instead of returning [None] * len(variables) which zeros out gradients
            if variables:
                var_grads = []
                for v in variables:
                    name = v.name.lower()
                    if "filter" in name or "kernel" in name:
                        var_grads.append(grad_filter)
                    elif "bias" in name:
                        var_grads.append(grad_bias)
                    else:
                        # Unknown variable - return None to let TF handle
                        var_grads.append(None)
                return input_grads, var_grads
            else:
                return input_grads

        return output, grad_fn

    # Call the closure with only the 3 tensor inputs
    return _compute_with_grad(input_tensor, filter_tensor, bias_tensor)
