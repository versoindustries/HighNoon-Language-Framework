# src/models/spatial/mamba.py
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

import logging
import math
import sys
from typing import Any

import tensorflow as tf

from highnoon._native.ops.fused_depthwise_conv1d import fused_depthwise_conv1d
from highnoon._native.ops.selective_scan_op import selective_scan
from highnoon.config import DEBUG_MODE
from highnoon.models.reasoning.fused_contract import FusedReasoningBlockMixin
from highnoon.models.tensor_layers import TTLayer

logger = logging.getLogger(__name__)


def apply_rope(x: tf.Tensor, dim: int | None = None) -> tf.Tensor:
    """Apply Rotary Position Embeddings (RoPE) to input tensor.

    Phase 12.7: Adds position-awareness to SSM blocks without absolute
    position embeddings. RoPE rotates pairs of dimensions based on position.

    Complexity: O(L) - element-wise operation.

    Args:
        x: Input tensor [batch, seq_len, dim].
        dim: Dimension for rotation (default: last dimension of x).

    Returns:
        Position-encoded tensor with same shape as input.
    """
    if dim is None:
        dim = x.shape[-1] or tf.shape(x)[-1]

    seq_len = tf.shape(x)[1]
    position_ids = tf.cast(tf.range(seq_len), tf.float32)

    # Compute rotation frequencies
    dim_float = tf.cast(dim, tf.float32)
    half_dim = dim // 2
    freq_indices = tf.cast(tf.range(0, half_dim * 2, 2), tf.float32)
    freqs = 1.0 / tf.pow(10000.0, freq_indices / dim_float)

    # Compute angles: [seq_len, half_dim]
    angles = tf.einsum("s,d->sd", position_ids, freqs)
    cos = tf.cos(angles)  # [seq_len, half_dim]
    sin = tf.sin(angles)  # [seq_len, half_dim]

    # Apply rotation
    x1 = x[..., ::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices

    # Rotate: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    rotated_x1 = x1 * cos[tf.newaxis, :, :] - x2 * sin[tf.newaxis, :, :]
    rotated_x2 = x1 * sin[tf.newaxis, :, :] + x2 * cos[tf.newaxis, :, :]

    # Interleave back
    batch_size = tf.shape(x)[0]
    rotated = tf.reshape(tf.stack([rotated_x1, rotated_x2], axis=-1), [batch_size, seq_len, -1])

    # Handle odd dimensions
    if dim % 2 == 1:
        rotated = tf.concat([rotated, x[..., -1:]], axis=-1)

    return rotated


def create_band_diagonal_mask(rows, cols, band_size):
    """Creates a band-diagonal binary mask tensor."""
    if band_size < 0:
        return tf.ones((rows, cols), dtype=tf.float32)

    row_indices = tf.range(rows, dtype=tf.int32)[:, tf.newaxis]
    col_indices = tf.range(cols, dtype=tf.int32)[tf.newaxis, :]
    distance_from_diagonal = tf.abs(row_indices - col_indices)
    mask = tf.cast(distance_from_diagonal <= band_size, dtype=tf.float32)
    return mask


class SpatialBlock(FusedReasoningBlockMixin, tf.keras.layers.Layer):
    """
    A Keras implementation of the Mamba-1 State-Space Model block.
    MODIFIED: All projection layers (Dense) have been replaced with TTLayer
    to resolve graph-mode build errors and improve parameter efficiency.

    NOTE: Uses TT decomposition - the C++ fused kernel has been extended to support TT layers.
    """

    fused_block_type = "SpatialBlock"
    fused_block_stateful = False

    def __init__(
        self,
        embedding_dim: int,
        state_dim: int = 16,
        conv_dim: int = 4,
        expand_factor: int = 2,
        **kwargs,
    ):
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.conv_dim = conv_dim
        self.expand_factor = expand_factor
        self.d_inner = self.embedding_dim * self.expand_factor
        self.dt_rank = math.ceil(self.embedding_dim / 16)
        # For hybrid compatibility, set data format attribute
        self.data_format = "channels_last"

        def factorize(dim):
            if dim <= 0:
                return [1, dim]
            s = int(math.sqrt(dim))
            while s > 1 and dim % s != 0:
                s -= 1
            if s == 1:
                for i in range(2, int(dim**0.5) + 2):
                    if dim % i == 0:
                        s = i
                        break
            return [s, dim // s] if s > 1 else [1, dim]

        self.in_proj = TTLayer(
            input_dims=factorize(self.embedding_dim),
            output_dims=factorize(2 * self.d_inner),
            tt_ranks=[1, 16, 1],
            name="in_proj_tt",
        )
        self.x_proj = TTLayer(
            input_dims=factorize(self.d_inner),
            output_dims=factorize(self.dt_rank + 2 * self.state_dim),
            tt_ranks=[1, 16, 1],
            name="x_proj_tt",
        )
        self.dt_proj = TTLayer(
            input_dims=factorize(self.dt_rank),
            output_dims=factorize(self.d_inner),
            tt_ranks=[1, 16, 1],
            name="dt_proj_tt",
        )
        self.out_proj = TTLayer(
            input_dims=factorize(self.d_inner),
            output_dims=factorize(self.embedding_dim),
            tt_ranks=[1, 16, 1],
            name="out_proj_tt",
        )

        self.conv1d_filter = None
        self.conv1d_bias = None

        self.A_log = self.add_weight(
            shape=(self.d_inner, self.state_dim),
            name="A_log",
            trainable=True,
            initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
        )
        self.D = self.add_weight(
            shape=(self.d_inner,), name="D", trainable=True, initializer="zeros"
        )

        # --- START: FIX ---
        # This will hold the compiled graph function to prevent retracing.
        # It is initialized in the `build` method.
        self._call_3d_impl = None
        # --- END: FIX ---

    def build(self, input_shape):
        """
        Explicitly builds the layer's weights.
        MODIFIED: This method now uses statically-known dimensions from the constructor
        to build sub-layers, making it robust to partially-defined shapes in graph mode.
        """
        # --- START: DEFINITIVE FIX ---
        # Instead of relying on `input_shape`, which can be `(None, None, None)`
        # in graph mode, we use the static dimensions defined in `__init__`.
        if not self.in_proj.built:
            self.in_proj.build(tf.TensorShape([None, self.embedding_dim]))

        if self.conv1d_filter is None:
            self.conv1d_filter = self.add_weight(
                name="conv1d_filter",
                shape=(self.conv_dim, 1, self.d_inner),
                initializer="glorot_uniform",
                trainable=True,
            )
        if self.conv1d_bias is None:
            self.conv1d_bias = self.add_weight(
                name="conv1d_bias", shape=(self.d_inner,), initializer="zeros", trainable=True
            )

        if not self.x_proj.built:
            self.x_proj.build(tf.TensorShape([None, self.d_inner]))

        if not self.dt_proj.built:
            self.dt_proj.build(tf.TensorShape([None, self.dt_rank]))

        if not self.out_proj.built:
            self.out_proj.build(tf.TensorShape([None, self.d_inner]))

        # --- START: FIX ---
        # Define the internal, compiled function with a fixed input signature.
        # This prevents Keras from retracing the function for every new input shape.
        if self._call_3d_impl is None:
            # Compile `_call_3d` once with a concrete signature so sequences of different
            # lengths reuse the same trace instead of triggering retracing.
            input_dtype = tf.dtypes.as_dtype(self.compute_dtype or tf.float32)
            call_signature = tf.TensorSpec(
                shape=[None, None, self.embedding_dim], dtype=input_dtype, name="x_3d"
            )
            compiled = tf.function(
                self._call_3d,
                reduce_retracing=True,
            )
            self._call_3d_impl = compiled.get_concrete_function(call_signature)
        # --- END: FIX ---
        super().build(input_shape)
        # --- END: DEFINITIVE FIX ---

    def _call_3d(self, x_3d: tf.Tensor) -> tf.Tensor:
        """Internal logic for the forward pass, designed to be compiled by tf.function."""
        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] input shape:",
                tf.shape(x_3d),
                "rank:",
                tf.rank(x_3d),
                output_stream=sys.stderr,
            )

        original_3d_shape = tf.shape(x_3d)
        x_2d = tf.reshape(x_3d, [-1, self.embedding_dim])

        xz = self.in_proj(x_2d)
        xz = tf.reshape(xz, [original_3d_shape[0], original_3d_shape[1], 2 * self.d_inner])

        # --- START: DEFINITIVE FIX for Retracing ---
        # The root cause of the retracing is using Python integers in `tf.split`.
        # By converting `self.d_inner` to a `tf.constant`, we provide a stable
        # graph constant, which prevents TensorFlow from retracing the function.
        # The `num_or_size_splits` argument must be a list of Python integers or a 1-D integer tensor.
        x_conv, z = tf.split(xz, num_or_size_splits=2, axis=-1)

        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] x_conv shape before conv1d:",
                tf.shape(x_conv),
                "rank:",
                tf.rank(x_conv),
                output_stream=sys.stderr,
            )

        stride = tf.constant(1, dtype=tf.int32)
        padding = tf.constant("SAME")
        x_conv_activated = tf.nn.silu(
            fused_depthwise_conv1d(x_conv, self.conv1d_filter, self.conv1d_bias, stride, padding)
        )
        y = x_conv_activated * z

        y_2d = tf.reshape(y, [-1, self.d_inner])

        dt_B_C = self.x_proj(y_2d)
        dt_B_C = tf.reshape(dt_B_C, [original_3d_shape[0], original_3d_shape[1], -1])

        # Apply the same fix here for the second `tf.split` call.
        # Using Python integers here is the source of the retracing.
        dt_unscaled, B, C = tf.split(
            dt_B_C,
            num_or_size_splits=[self.dt_rank, self.state_dim, self.state_dim],
            axis=-1,
        )

        dt_unscaled_2d = tf.reshape(dt_unscaled, [-1, self.dt_rank])
        dt = tf.nn.softplus(self.dt_proj(dt_unscaled_2d))
        dt = tf.reshape(dt, [original_3d_shape[0], original_3d_shape[1], self.d_inner])

        ssm_out, hidden_states = selective_scan(y, dt, self.A_log, B, C, self.D)

        # Ensure hidden_states is used in the graph to allow gradients to flow.
        # This is a trick to make autograd track the tensor.
        hidden_states_sum = tf.reduce_sum(hidden_states)
        ssm_out = ssm_out + (hidden_states_sum * tf.zeros_like(ssm_out))

        ssm_out_2d = tf.reshape(ssm_out, [-1, self.d_inner])
        output = self.out_proj(ssm_out_2d)
        output = tf.reshape(
            output, [original_3d_shape[0], original_3d_shape[1], self.embedding_dim]
        )
        # --- END: DEFINITIVE FIX for Retracing ---
        return output

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Public-facing call method. It handles rank polymorphism before dispatching
        to the compiled `_call_3d` method to prevent tf.function retracing.
        """
        target_dtype = tf.dtypes.as_dtype(self.compute_dtype or tf.float32)
        x = tf.cast(x, target_dtype)
        tf.rank(x)
        original_shape = tf.shape(x)

        # --- START: DEFINITIVE FIX for Gradient Shape Mismatch ---
        # The tf.cond was causing issues with gradient calculation in graph mode.
        # This unconditional reshape is more robust and creates a single, stable
        # graph path for both 2D and 3D inputs.
        # If input is 2D (B, D), reshape to (B, 1, D).
        # If input is 3D (B, S, D), reshape to (B, S, D) (no-op).
        x_3d = tf.reshape(x, [original_shape[0], -1, self.embedding_dim])

        output = self._call_3d_impl(x_3d)

        # Reshape the output back to the original rank.
        # If the original was 2D, the output shape will be (B, 1, D), which we
        # reshape back to (B, D).
        output = tf.reshape(output, original_shape)
        # --- END: DEFINITIVE FIX ---

        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] output shape:",
                tf.shape(output),
                "rank:",
                tf.rank(output),
                output_stream=sys.stderr,
            )

        return output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "state_dim": self.state_dim,
                "conv_dim": self.conv_dim,
                "expand_factor": self.expand_factor,
                "d_inner": self.d_inner,  # Include d_inner for compatibility
                "data_format": self.data_format,  # Include data_format for compatibility
            }
        )
        return config

    def fused_metadata(self) -> dict[str, int]:
        return {
            "embedding_dim": int(self.embedding_dim),
            "state_dim": int(self.state_dim),
            "conv_dim": int(self.conv_dim),
            "expand_factor": int(self.expand_factor),
            "d_inner": int(self.d_inner),
        }

    def get_fused_op_descriptor(self) -> dict[str, Any]:
        """Returns descriptor including TT layer metadata for C++ fused kernel."""
        from ..tensor_layers import SuperpositionTTLayer, TTLayer

        # Extract TT layer info
        tt_layers = []
        weight_idx = 0
        for attr_name in ["in_proj", "x_proj", "dt_proj", "out_proj"]:
            if hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                if isinstance(attr, (TTLayer, SuperpositionTTLayer)):
                    tt_layer_info = {
                        "name": attr_name,
                        "input_dims": list(attr.input_dims),
                        "output_dims": list(attr.output_dims),
                        "tt_ranks": list(attr.tt_ranks),
                        "num_cores": int(attr.d),
                        "core_indices": list(range(weight_idx, weight_idx + attr.d)),
                    }
                    if isinstance(attr, SuperpositionTTLayer):
                        tt_layer_info["superposition_dim"] = int(attr.superposition_dim)
                        tt_layer_info["is_superposition"] = True
                    else:
                        tt_layer_info["is_superposition"] = False
                    tt_layers.append(tt_layer_info)
                    weight_idx += attr.d
                else:
                    weight_idx += 1  # Regular dense layer has 1 weight

        metadata = self.fused_metadata()
        if tt_layers:
            metadata["tt_layers"] = tt_layers

        return {
            "type": self.__class__.__name__,
            "stateful": True,
            "metadata": metadata,
            "weight_count": len(self.get_weights_for_fused_op()),
        }


class ReasoningMamba2Block(FusedReasoningBlockMixin, tf.keras.layers.Layer):
    """
    A stateful Mamba-2 block implementation for the reasoning core.
    MODIFIED: Replaced the standard Conv1D layer with the custom C++ operator
    `fused_depthwise_conv1d` to enable gradient computation on CPU.

    NOTE: Uses TT decomposition - the C++ fused kernel has been extended to support TT layers.
    """

    fused_block_type = "ReasoningMamba2Block"
    fused_block_stateful = True

    def __init__(
        self,
        embedding_dim: int,
        state_dim: int,
        head_dim: int,
        conv_dim: int,
        expand_factor: int = 2,
        lrf_rank: int | None = None,
        sparsity_band: int | None = None,
        d_inner: int | None = None,  # ACCEPT NEW HP
        **kwargs,
    ):
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.head_dim = head_dim
        self.conv_dim = conv_dim
        self.expand_factor = expand_factor
        self.d_inner = d_inner or self.embedding_dim * self.expand_factor
        self.dt_rank = math.ceil(self.embedding_dim / 16)
        self.lrf_rank = lrf_rank
        self.sparsity_band = sparsity_band

        if self.d_inner % self.head_dim != 0:
            raise ValueError("d_inner must be divisible by head_dim")
        self.d_head = self.d_inner // self.head_dim

        self.data_format = "channels_last"
        # logger.info(f"[{self.name}] Using data_format: '{self.data_format}' for custom Conv1D.")

        def factorize(dim):
            if dim <= 0:
                return [1, dim]
            s = int(math.sqrt(dim))
            while s > 1 and dim % s != 0:
                s -= 1
            if s == 1:
                for i in range(2, int(dim**0.5) + 2):
                    if dim % i == 0:
                        s = i
                        break
            return [s, dim // s] if s > 1 else [1, dim]

        in_proj_input_dims = factorize(self.embedding_dim)
        in_proj_output_dims = factorize(2 * self.d_inner)
        self.in_proj = TTLayer(
            input_dims=in_proj_input_dims,
            output_dims=in_proj_output_dims,
            tt_ranks=[1, 16, 1],
            name="in_proj",
        )

        # FIX: Replace Dense with TTLayer for consistency and graph-mode stability.
        x_proj_input_dims = factorize(self.d_inner)
        x_proj_output_dims = factorize(self.dt_rank + 2 * self.state_dim)
        self.x_proj = TTLayer(
            input_dims=x_proj_input_dims,
            output_dims=x_proj_output_dims,
            tt_ranks=[1, 16, 1],
            name="x_proj",
        )

        dt_proj_input_dims = factorize(self.dt_rank)
        dt_proj_output_dims = factorize(self.d_inner)
        self.dt_proj = TTLayer(
            input_dims=dt_proj_input_dims,
            output_dims=dt_proj_output_dims,
            tt_ranks=[1, 16, 1],
            name="dt_proj",
        )

        out_proj_input_dims = factorize(self.d_inner)
        out_proj_output_dims = factorize(self.embedding_dim)
        self.out_proj = TTLayer(
            input_dims=out_proj_input_dims,
            output_dims=out_proj_output_dims,
            tt_ranks=[1, 16, 1],
            name="out_proj",
        )

        self.conv1d_filter = None
        self.conv1d_bias = None

        self.A_log = self.add_weight(
            shape=(self.d_inner, self.state_dim), name="A_log", trainable=True
        )
        self.D = self.add_weight(
            shape=(self.d_inner,), name="D", trainable=True, initializer="zeros"
        )

        if self.sparsity_band is not None:
            self.A_mask = create_band_diagonal_mask(
                self.d_inner, self.state_dim, self.sparsity_band
            )
        else:
            self.A_mask = None

    def build(self, input_shape):
        x_shape, _ = input_shape
        if not self.in_proj.built:
            self.in_proj.build(tf.TensorShape([None, self.embedding_dim]))

        if self.conv1d_filter is None:
            self.conv1d_filter = self.add_weight(
                name="conv1d_filter",
                shape=(self.conv_dim, 1, self.d_inner),
                initializer="glorot_uniform",
                trainable=True,
            )
        if self.conv1d_bias is None:
            self.conv1d_bias = self.add_weight(
                name="conv1d_bias", shape=(self.d_inner,), initializer="zeros", trainable=True
            )

        x_activated_shape = tf.TensorShape([None, self.d_inner])
        if not self.x_proj.built:
            self.x_proj.build(x_activated_shape)

        dt_unscaled_shape = tf.TensorShape([None, self.dt_rank])
        if not self.dt_proj.built:
            self.dt_proj.build(dt_unscaled_shape)

        y_gated_shape = tf.TensorShape([None, self.d_inner])
        if not self.out_proj.built:
            self.out_proj.build(y_gated_shape)

        super().build(input_shape)

    @tf.function(reduce_retracing=True)
    def call(
        self, inputs: tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]], training: bool = False
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """
        Performs a single stateful step of the Mamba-2 block.
        """
        x, (h_padded, conv_state) = inputs

        if len(x.shape) > 2:
            x_squeezed = tf.squeeze(x, axis=1)
        else:
            x_squeezed = x

        # --- START: DEBUG LOGGING ---
        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] --- ReasoningMamba2Block.call START ---", output_stream=sys.stderr
            )
            tf.print(
                f"[{self.name}] Input x shape (1-step):",
                tf.shape(x_squeezed),
                output_stream=sys.stderr,
            )
            tf.print(
                f"[{self.name}] Input h_padded shape:", tf.shape(h_padded), output_stream=sys.stderr
            )
            tf.print(
                f"[{self.name}] Input conv_state shape:",
                tf.shape(conv_state),
                output_stream=sys.stderr,
            )
            tf.print(
                f"[{self.name}] Static h_padded shape (for debugging None):",
                h_padded.shape,
                output_stream=sys.stderr,
            )
            tf.print(
                f"[{self.name}] Static conv_state shape (for debugging None):",
                conv_state.shape,
                output_stream=sys.stderr,
            )
        # --- END: DEBUG LOGGING ---

        xz = self.in_proj(x_squeezed)
        # Use constant python values for split that are guaranteed to be static
        x_c, z = tf.split(xz, num_or_size_splits=[self.d_inner, self.d_inner], axis=-1)

        conv_input = tf.concat([conv_state, x_c[:, tf.newaxis, :]], axis=1)

        stride = tf.constant(1, dtype=tf.int32)
        padding = tf.constant("SAME")
        conv_out_full = fused_depthwise_conv1d(
            conv_input, self.conv1d_filter, self.conv1d_bias, stride, padding
        )

        conv_out = conv_out_full[:, -1, :]
        x_activated = tf.nn.silu(conv_out)

        # Slice the conv state using tensor operations to determine the slice size
        conv_dim_tensor = tf.constant(self.conv_dim, dtype=tf.int32)
        new_conv_state = tf.slice(
            conv_input,
            begin=tf.stack([0, 1, 0]),
            size=tf.stack([tf.shape(conv_input)[0], conv_dim_tensor - 1, tf.shape(conv_input)[2]]),
        )

        dt_B_C_x = self.x_proj(x_activated)

        # Use tf.stack to define split sizes for safety
        dt_rank_tensor = tf.constant(self.dt_rank, dtype=tf.int32)
        state_dim_tensor = tf.constant(self.state_dim, dtype=tf.int32)

        dt_unscaled, B, C = tf.split(
            dt_B_C_x, tf.stack([dt_rank_tensor, state_dim_tensor, state_dim_tensor]), axis=-1
        )

        dt = tf.nn.softplus(self.dt_proj(dt_unscaled))

        A_log_param = self.A_log
        if self.A_mask is not None:
            A_mask_casted = tf.cast(self.A_mask, dtype=A_log_param.dtype)
            A_log_param = A_log_param * A_mask_casted

        A = -tf.exp(tf.cast(A_log_param, tf.float32))
        dt_broadcast = tf.expand_dims(dt, axis=-1)
        A_disc = tf.exp(dt_broadcast * A)

        x_activated_expanded = tf.expand_dims(x_activated, axis=1)
        B_expanded = tf.expand_dims(B, axis=1)

        Bx = dt_broadcast * (x_activated_expanded * B_expanded)

        # The core state recurrence
        new_h = A_disc * h_padded + Bx

        C_expanded = tf.expand_dims(C, axis=1)
        ssm_out = tf.reduce_sum(new_h * C_expanded, axis=-1)

        y = ssm_out + x_activated * self.D
        y_gated = y * tf.nn.silu(z)

        output = self.out_proj(y_gated)
        new_state = (new_h, new_conv_state)

        output = tf.expand_dims(output, axis=1)

        if DEBUG_MODE:
            tf.print(f"[{self.name}] Output shape:", tf.shape(output), output_stream=sys.stderr)
            tf.print(
                f"[{self.name}] Output state h shape:", tf.shape(new_h), output_stream=sys.stderr
            )
            tf.print(
                f"[{self.name}] --- ReasoningMamba2Block.call END ---", output_stream=sys.stderr
            )

        return output, new_state

    def get_config(self) -> dict:
        """Enables serialization of the layer."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "state_dim": self.state_dim,
                "head_dim": self.head_dim,
                "conv_dim": self.conv_dim,
                "expand_factor": self.expand_factor,
                "lrf_rank": self.lrf_rank,
                "sparsity_band": self.sparsity_band,
                "d_inner": self.d_inner,  # Include d_inner in config
                "data_format": self.data_format,  # Include data_format in config
            }
        )
        return config

    def fused_metadata(self) -> dict[str, int]:
        return {
            "embedding_dim": int(self.embedding_dim),
            "state_dim": int(self.state_dim),
            "conv_dim": int(self.conv_dim),
            "expand_factor": int(self.expand_factor),
            "head_dim": int(self.head_dim),
            "dt_rank": int(self.dt_rank),
            "d_inner": int(self.d_inner),
        }

    def get_fused_op_descriptor(self) -> dict[str, Any]:
        """Returns descriptor including TT layer metadata for C++ fused kernel."""
        from ..tensor_layers import SuperpositionTTLayer, TTLayer

        # Extract TT layer info
        tt_layers = []
        weight_idx = 0
        for attr_name in ["in_proj", "x_proj", "dt_proj", "out_proj"]:
            if hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                if isinstance(attr, (TTLayer, SuperpositionTTLayer)):
                    tt_layer_info = {
                        "name": attr_name,
                        "input_dims": list(attr.input_dims),
                        "output_dims": list(attr.output_dims),
                        "tt_ranks": list(attr.tt_ranks),
                        "num_cores": int(attr.d),
                        "core_indices": list(range(weight_idx, weight_idx + attr.d)),
                    }
                    if isinstance(attr, SuperpositionTTLayer):
                        tt_layer_info["superposition_dim"] = int(attr.superposition_dim)
                        tt_layer_info["is_superposition"] = True
                    else:
                        tt_layer_info["is_superposition"] = False
                    tt_layers.append(tt_layer_info)
                    weight_idx += attr.d
                else:
                    weight_idx += 1  # Regular dense layer has 1 weight

        metadata = self.fused_metadata()
        if tt_layers:
            metadata["tt_layers"] = tt_layers

        return {
            "type": self.__class__.__name__,
            "stateful": True,
            "metadata": metadata,
            "weight_count": len(self.get_weights_for_fused_op()),
        }
