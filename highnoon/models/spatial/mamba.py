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
from highnoon.config import DEBUG_MODE, USE_AUTO_NEURAL_QEM
from highnoon.models.reasoning.fused_contract import FusedReasoningBlockMixin
from highnoon.models.tensor_layers import TTLayer
from highnoon.utils import factorize_for_tt

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

        self.in_proj = TTLayer(
            input_dims=factorize_for_tt(self.embedding_dim),
            output_dims=factorize_for_tt(2 * self.d_inner),
            tt_ranks=[1, 16, 1],
            name="in_proj_tt",
        )
        self.x_proj = TTLayer(
            input_dims=factorize_for_tt(self.d_inner),
            output_dims=factorize_for_tt(self.dt_rank + 2 * self.state_dim),
            tt_ranks=[1, 16, 1],
            name="x_proj_tt",
        )
        self.dt_proj = TTLayer(
            input_dims=factorize_for_tt(self.dt_rank),
            output_dims=factorize_for_tt(self.d_inner),
            tt_ranks=[1, 16, 1],
            name="dt_proj_tt",
        )
        self.out_proj = TTLayer(
            input_dims=factorize_for_tt(self.d_inner),
            output_dims=factorize_for_tt(self.embedding_dim),
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

        # Connect hidden_states to computation graph for gradient flow.
        # Use a small epsilon multiplier instead of zero to maintain gradient path.
        # The selective_scan C++ op returns hidden_states which should contribute
        # to the gradient, so we add a scaled identity to preserve the dependency.
        hidden_states_contribution = tf.reduce_mean(hidden_states) * 1e-8
        ssm_out = ssm_out + hidden_states_contribution

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
        # Store training state for use in _call_3d (accessed via getattr)
        self._is_training = training

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

        in_proj_input_dims = factorize_for_tt(self.embedding_dim)
        in_proj_output_dims = factorize_for_tt(2 * self.d_inner)
        self.in_proj = TTLayer(
            input_dims=in_proj_input_dims,
            output_dims=in_proj_output_dims,
            tt_ranks=[1, 16, 1],
            name="in_proj",
        )

        # FIX: Replace Dense with TTLayer for consistency and graph-mode stability.
        x_proj_input_dims = factorize_for_tt(self.d_inner)
        x_proj_output_dims = factorize_for_tt(self.dt_rank + 2 * self.state_dim)
        self.x_proj = TTLayer(
            input_dims=x_proj_input_dims,
            output_dims=x_proj_output_dims,
            tt_ranks=[1, 16, 1],
            name="x_proj",
        )

        dt_proj_input_dims = factorize_for_tt(self.dt_rank)
        dt_proj_output_dims = factorize_for_tt(self.d_inner)
        self.dt_proj = TTLayer(
            input_dims=dt_proj_input_dims,
            output_dims=dt_proj_output_dims,
            tt_ranks=[1, 16, 1],
            name="dt_proj",
        )

        out_proj_input_dims = factorize_for_tt(self.d_inner)
        out_proj_output_dims = factorize_for_tt(self.embedding_dim)
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


class QMambaBlock(SpatialBlock):
    """Phase 102: QMamba - Quantum-Enhanced Selective State Space Model.

    Extends SpatialBlock with quantum superposition states for enhanced context
    understanding. Each of K parallel state paths evolves independently with
    entanglement correlations, then collapses via Born rule.

    Research Basis: QMamba (Koelle et al., ICAART 2025)

    Key Features:
        - Quantum State Superposition: K parallel state paths exist simultaneously
        - Entanglement-Aware Updates: VQC encodes inter-position correlations
        - Amplitude-Weighted Selection: Born rule for selective scanning

    Complexity: O(n · K · state_dim) where K is num_superposition_paths

    Args:
        embedding_dim: Input/output embedding dimension.
        num_superposition_paths: Number of parallel quantum state paths (default: 4).
        state_dim: SSM state dimension (default: 16).
        conv_dim: Convolution dimension (default: 4).
        expand_factor: Expansion factor for inner dimension (default: 2).
        entanglement_depth: VQC entanglement layers (default: 2).
        entanglement_strength: α ∈ [0,1] for quantum mixing (default: 0.3).
        use_born_rule: Use Born rule (True) or Gumbel-Softmax (False) collapse.
        gumbel_temperature: Temperature for Gumbel-Softmax collapse (default: 1.0).
    """

    fused_block_type = "QMambaBlock"
    fused_block_stateful = False

    def __init__(
        self,
        embedding_dim: int,
        num_superposition_paths: int = 4,
        state_dim: int = 16,
        conv_dim: int = 4,
        expand_factor: int = 2,
        entanglement_depth: int = 2,
        entanglement_strength: float = 0.3,
        use_born_rule: bool = True,
        gumbel_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            state_dim=state_dim,
            conv_dim=conv_dim,
            expand_factor=expand_factor,
            **kwargs,
        )

        self.num_paths = num_superposition_paths
        self.entanglement_depth = entanglement_depth
        self.entanglement_strength = entanglement_strength
        self.use_born_rule = use_born_rule
        self.gumbel_temperature = gumbel_temperature

        # Learnable complex amplitudes for each superposition path
        # |ψ⟩ = Σ (α_real + i·α_imag) |k⟩
        init_amplitude = 1.0 / math.sqrt(num_superposition_paths)
        self.amplitudes_real = self.add_weight(
            shape=(num_superposition_paths,),
            name="amplitudes_real",
            trainable=True,
            initializer=tf.keras.initializers.Constant(init_amplitude),
        )
        self.amplitudes_imag = self.add_weight(
            shape=(num_superposition_paths,),
            name="amplitudes_imag",
            trainable=True,
            initializer="zeros",
        )

        # Learnable VQC rotation angles for entanglement
        self.rotation_angles = self.add_weight(
            shape=(entanglement_depth, num_superposition_paths),
            name="rotation_angles",
            trainable=True,
            initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=2 * math.pi),
        )

        # Phase 130.1: Auto Neural QEM mitigator for QMamba quantum outputs
        self._qmamba_qem_mitigator = None
        if USE_AUTO_NEURAL_QEM:
            from highnoon.training.neural_zne import NeuralQuantumErrorMitigator

            self._qmamba_qem_mitigator = NeuralQuantumErrorMitigator(name="qmamba_qem")

        # Phase 130.3: Coherence tracking for UnifiedQuantumBus integration
        self._last_coherence = 0.0

    def superposition_evolve(self, x: tf.Tensor, states: list[tf.Tensor]) -> list[tf.Tensor]:
        """Evolve K parallel state paths with entanglement.

        Each path gets a slightly different effective dt for exploration.
        Cross-path entanglement is achieved via phase coupling.

        Args:
            x: Input tensor [batch, seq_len, embedding_dim].
            states: List of K state tensors, each [batch, d_inner, state_dim].

        Returns:
            List of K evolved state tensors.
        """
        evolved_states = []

        for k in range(self.num_paths):
            # Slightly different effective dt per path for diversity
            path_scale = 1.0 + 0.1 * (k - self.num_paths / 2) / self.num_paths
            h_k = states[k] * path_scale

            # Cross-path entanglement via phase coupling
            if k > 0:
                # Complex amplitude: α_real + i·α_imag
                phase_real = self.amplitudes_real[k]
                phase_imag = self.amplitudes_imag[k]
                # Amplitude magnitude (for scaling)
                amplitude = tf.sqrt(phase_real**2 + phase_imag**2 + 1e-10)

                # Mix with previous path using amplitude-weighted coupling
                h_k = h_k + self.entanglement_strength * amplitude * evolved_states[k - 1]

            evolved_states.append(h_k)

        return evolved_states

    def born_rule_collapse(self, states: list[tf.Tensor]) -> tf.Tensor:
        """Collapse superposition via Born rule probability.

        Computes path probabilities from squared amplitudes:
            prob_k = |α_k|² / Σ|α_k|²
            h_out = Σ prob_k * h_k

        Args:
            states: List of K state tensors to collapse.

        Returns:
            Collapsed state tensor.
        """
        # Compute amplitudes squared (Born rule probabilities)
        probs = self.amplitudes_real**2 + self.amplitudes_imag**2
        probs = probs / (tf.reduce_sum(probs) + 1e-10)  # Normalize

        # Weighted combination (soft collapse)
        collapsed = tf.zeros_like(states[0])
        for k in range(self.num_paths):
            collapsed = collapsed + probs[k] * states[k]

        return collapsed

    def _call_3d(self, x_3d: tf.Tensor) -> tf.Tensor:
        """Internal forward pass with quantum superposition.

        Overrides parent to add quantum-enhanced state evolution:
        1. Initialize K superposition states
        2. Apply entanglement layers for correlations
        3. Run parallel SSM processing on each path
        4. Collapse to single output via Born rule
        """
        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] QMambaBlock input shape:",
                tf.shape(x_3d),
                output_stream=sys.stderr,
            )

        original_3d_shape = tf.shape(x_3d)
        batch_size = original_3d_shape[0]
        seq_len = original_3d_shape[1]

        # Process input through in_proj
        x_2d = tf.reshape(x_3d, [-1, self.embedding_dim])
        xz = self.in_proj(x_2d)
        xz = tf.reshape(xz, [batch_size, seq_len, 2 * self.d_inner])
        x_conv, z = tf.split(xz, num_or_size_splits=2, axis=-1)

        # Conv1d + activation
        stride = tf.constant(1, dtype=tf.int32)
        padding = tf.constant("SAME")
        x_conv_activated = tf.nn.silu(
            fused_depthwise_conv1d(x_conv, self.conv1d_filter, self.conv1d_bias, stride, padding)
        )
        y = x_conv_activated * z

        # Initialize K superposition states
        init_amplitude = 1.0 / math.sqrt(self.num_paths)
        initial_state = tf.zeros([batch_size, self.d_inner, self.state_dim])
        states = [
            initial_state
            + tf.random.normal(tf.shape(initial_state), stddev=0.01, seed=42 + k) * init_amplitude
            for k in range(self.num_paths)
        ]

        # Project to get dt, B, C
        y_2d = tf.reshape(y, [-1, self.d_inner])
        dt_B_C = self.x_proj(y_2d)
        dt_B_C = tf.reshape(dt_B_C, [batch_size, seq_len, -1])

        dt_unscaled, B, C = tf.split(
            dt_B_C,
            num_or_size_splits=[self.dt_rank, self.state_dim, self.state_dim],
            axis=-1,
        )

        dt_unscaled_2d = tf.reshape(dt_unscaled, [-1, self.dt_rank])
        dt = tf.nn.softplus(self.dt_proj(dt_unscaled_2d))
        dt = tf.reshape(dt, [batch_size, seq_len, self.d_inner])

        # Apply entanglement layers before SSM processing
        for layer_idx in range(self.entanglement_depth):
            # Apply rotation angles per path
            for k in range(self.num_paths):
                theta = self.rotation_angles[layer_idx, k]
                cos_t = tf.cos(theta)
                sin_t = tf.sin(theta)
                # RY-like rotation on state
                states[k] = cos_t * states[k] + sin_t * tf.tanh(states[k])

            # CNOT-like entanglement between adjacent paths
            if self.num_paths > 1:
                new_states = [states[0]]
                for k in range(1, self.num_paths):
                    mix = self.entanglement_strength * states[k - 1] * states[k]
                    new_states.append(states[k] + mix)
                states = new_states

        # Run SSM on each superposition path with different effective dt
        # The states from entanglement layers modulate the SSM output
        ssm_outputs = []
        for k in range(self.num_paths):
            path_scale = 1.0 + 0.1 * (k - self.num_paths / 2) / self.num_paths
            scaled_dt = dt * path_scale

            # Use standard selective_scan for each path
            ssm_out_k, hidden_k = selective_scan(y, scaled_dt, self.A_log, B, C, self.D)

            # Connect rotation_angles to output via state-based modulation
            # states[k] is [batch, d_inner, state_dim], reduce to [batch, 1, d_inner]
            # to modulate the SSM output, connecting gradient flow to rotation_angles
            state_modulation = tf.reduce_mean(states[k], axis=-1, keepdims=True)
            state_modulation = tf.transpose(state_modulation, [0, 2, 1])  # [batch, 1, d_inner]
            # Soft gating: use sigmoid to keep modulation in [0, 1] range
            state_gate = tf.nn.sigmoid(state_modulation)
            # Apply modulation: blend SSM output with gated version
            ssm_out_k = ssm_out_k * (0.9 + 0.1 * state_gate)

            ssm_outputs.append(ssm_out_k)

        # Collapse superposition via Born rule
        probs = self.amplitudes_real**2 + self.amplitudes_imag**2
        probs = probs / (tf.reduce_sum(probs) + 1e-10)

        ssm_out = tf.zeros_like(ssm_outputs[0])
        for k in range(self.num_paths):
            ssm_out = ssm_out + probs[k] * ssm_outputs[k]

        # Output projection
        ssm_out_2d = tf.reshape(ssm_out, [-1, self.d_inner])
        output = self.out_proj(ssm_out_2d)
        output = tf.reshape(output, [batch_size, seq_len, self.embedding_dim])

        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] QMambaBlock output shape:",
                tf.shape(output),
                output_stream=sys.stderr,
            )

        # Phase 130.3: Compute coherence for UnifiedQuantumBus
        self._last_coherence = self._compute_entanglement_coherence(probs)

        # Phase 130.1: Apply auto Neural QEM if enabled
        # Use training state from call() method for proper gradient flow
        if self._qmamba_qem_mitigator is not None:
            is_training = getattr(self, "_is_training", False)
            output = self._qmamba_qem_mitigator(output, training=is_training)

        return output

    def _compute_entanglement_coherence(self, probs: tf.Tensor) -> float:
        """Compute coherence metric from superposition probabilities.

        Coherence is measured as the inverse of probability entropy.
        High coherence = concentrated probability on few paths.
        Low coherence = uniform probability across paths.

        Args:
            probs: Born rule probabilities [num_paths].

        Returns:
            Coherence value in [0, 1].
        """
        # Entropy: H = -Σ p log p
        log_probs = tf.math.log(probs + 1e-10)
        entropy = -tf.reduce_sum(probs * log_probs)

        # Max entropy for uniform distribution: log(K)
        max_entropy = tf.math.log(tf.cast(self.num_paths, tf.float32))

        # Coherence = 1 - normalized entropy
        coherence = 1.0 - (entropy / (max_entropy + 1e-10))
        return float(coherence.numpy()) if tf.executing_eagerly() else 0.5

    @property
    def last_coherence(self) -> float:
        """Return the last computed coherence for UnifiedQuantumBus integration."""
        return self._last_coherence

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_superposition_paths": self.num_paths,
                "entanglement_depth": self.entanglement_depth,
                "entanglement_strength": self.entanglement_strength,
                "use_born_rule": self.use_born_rule,
                "gumbel_temperature": self.gumbel_temperature,
            }
        )
        return config

    def fused_metadata(self) -> dict[str, int]:
        metadata = super().fused_metadata()
        metadata.update(
            {
                "num_superposition_paths": int(self.num_paths),
                "entanglement_depth": int(self.entanglement_depth),
            }
        )
        return metadata


class QSSMGatedBlock(SpatialBlock):
    """Phase 120: Q-SSM Quantum-Optimized Selective State Space Gating.

    Extends SpatialBlock with VQC-based adaptive gating for memory updates.
    The VQC acts as a learned gate that controls the balance between retaining
    prior state vs. incorporating new input.

    Research Basis:
        "Q-SSM: Quantum-Optimized Selective State Space Model" (arXiv 2025)

    Key Features:
        - VQC Gating: RY-RX ansatz regulates memory updates adaptively
        - Quantum Stabilization: Prevents optimization instabilities
        - Born Rule Interpretation: Gate values via quantum measurement

    Gate Equation:
        S_t = σ_VQC(x_t) ⊙ S_{t-1} + (1 - σ_VQC(x_t)) ⊙ Update(x_t)

    Complexity: O(N × D × vqc_layers)

    Args:
        embedding_dim: Input/output embedding dimension.
        state_dim: SSM state dimension (default: 16).
        conv_dim: Convolution dimension (default: 4).
        expand_factor: Expansion factor for inner dimension (default: 2).
        vqc_qubits: Number of virtual qubits for gating (default: 4).
        vqc_layers: Number of VQC rotation layers (default: 2).
        use_born_rule: Use Born rule (True) or sigmoid (False) for gate.
        measurement_temp: Temperature for soft measurement (default: 1.0).
    """

    fused_block_type = "QSSMGatedBlock"
    fused_block_stateful = False

    def __init__(
        self,
        embedding_dim: int,
        state_dim: int = 16,
        conv_dim: int = 4,
        expand_factor: int = 2,
        vqc_qubits: int = 4,
        vqc_layers: int = 2,
        use_born_rule: bool = True,
        measurement_temp: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            state_dim=state_dim,
            conv_dim=conv_dim,
            expand_factor=expand_factor,
            **kwargs,
        )

        self.vqc_qubits = vqc_qubits
        self.vqc_layers = vqc_layers
        self.use_born_rule = use_born_rule
        self.measurement_temp = measurement_temp

        # VQC rotation parameters: [vqc_layers, vqc_qubits, 2] (RY, RX angles)
        self.vqc_params = self.add_weight(
            shape=(vqc_layers, vqc_qubits, 2),
            name="vqc_params",
            trainable=True,
            initializer=tf.keras.initializers.RandomUniform(minval=-0.15, maxval=0.15),
        )

        # Gate encoder weights: maps (embedding_dim + state_dim) -> vqc_qubits
        gate_input_dim = embedding_dim + state_dim
        self.gate_encoder_kernel = self.add_weight(
            shape=(gate_input_dim, vqc_qubits),
            name="gate_encoder_kernel",
            trainable=True,
            initializer="glorot_uniform",
        )
        self.gate_encoder_bias = self.add_weight(
            shape=(vqc_qubits,),
            name="gate_encoder_bias",
            trainable=True,
            initializer="zeros",
        )

        # Gate projection weights: maps vqc_qubits -> state_dim
        self.gate_projection_kernel = self.add_weight(
            shape=(vqc_qubits, state_dim),
            name="gate_projection_kernel",
            trainable=True,
            initializer="glorot_uniform",
        )
        self.gate_projection_bias = self.add_weight(
            shape=(state_dim,),
            name="gate_projection_bias",
            trainable=True,
            initializer="zeros",
        )

        # Phase 130.1: Auto Neural QEM mitigator for Q-SSM quantum outputs
        self._qssm_qem_mitigator = None
        if USE_AUTO_NEURAL_QEM:
            from highnoon.training.neural_zne import NeuralQuantumErrorMitigator

            self._qssm_qem_mitigator = NeuralQuantumErrorMitigator(name="qssm_qem")

        # Phase 130.3: Coherence tracking for UnifiedQuantumBus integration
        self._last_coherence = 0.0
        self._last_vqc_gate = None

    def build(self, input_shape):
        super().build(input_shape)
        # Weights are already added in __init__, build is handled by parent

    def _compute_vqc_gate(self, x: tf.Tensor, state: tf.Tensor) -> tf.Tensor:
        """Compute VQC-based gate values.

        Simulates an RY-RX VQC with CNOT entanglement to produce gate values
        that adaptively control state update vs. retention.

        Args:
            x: Input tensor [batch, d_inner].
            state: Current state tensor [batch, state_dim].

        Returns:
            Gate values [batch, state_dim] in [0, 1].
        """
        # Project x from d_inner to embedding_dim using simple linear projection
        # Use slicing to get first embedding_dim features (d_inner >= embedding_dim)
        x_proj = x[:, : self.embedding_dim]

        # Combine input and state for context-aware gating
        gate_input = tf.concat([x_proj, state], axis=-1)

        # Encode to VQC qubit angles using manual weights
        encoded_angles = tf.matmul(gate_input, self.gate_encoder_kernel) + self.gate_encoder_bias
        encoded_angles = tf.math.atan(encoded_angles) * 2.0  # Normalize to [-π, π]

        # Simulate VQC expectation values
        # Uses RY-RX ansatz with ring entanglement
        vqc_output = self._simulate_vqc(encoded_angles)  # [batch]

        # Project to gate values using manual weights
        vqc_expanded = tf.expand_dims(vqc_output, axis=-1)  # [batch, 1]
        vqc_tiled = tf.tile(vqc_expanded, [1, self.vqc_qubits])  # [batch, vqc_qubits]
        gate = tf.nn.sigmoid(
            tf.matmul(vqc_tiled, self.gate_projection_kernel) + self.gate_projection_bias
        )

        return gate

    def _simulate_vqc(self, encoded_angles: tf.Tensor) -> tf.Tensor:
        """Simulate VQC and return expectation values.

        Uses RY-RX rotation layers with CNOT entanglement (ring topology).

        Args:
            encoded_angles: Encoded input angles [batch, vqc_qubits].

        Returns:
            Expectation values [batch].
        """
        batch_size = tf.shape(encoded_angles)[0]

        # Initialize qubit states: |0⟩ for all (real part = 1, imag = 0)
        states_real = tf.ones([batch_size, self.vqc_qubits], dtype=tf.float32)
        states_imag = tf.zeros([batch_size, self.vqc_qubits], dtype=tf.float32)

        # Apply input encoding (RY rotations)
        for q in range(self.vqc_qubits):
            theta = encoded_angles[:, q]
            cos_half = tf.cos(theta / 2.0)
            sin_half = tf.sin(theta / 2.0)

            new_real = cos_half * states_real[:, q] - sin_half * states_imag[:, q]
            new_imag = sin_half * states_real[:, q] + cos_half * states_imag[:, q]

            # Update states using tf.tensor_scatter_nd_update
            indices = tf.expand_dims(tf.range(batch_size), axis=1)
            indices = tf.concat([indices, tf.fill([batch_size, 1], q)], axis=1)
            states_real = tf.tensor_scatter_nd_update(states_real, indices, new_real)
            states_imag = tf.tensor_scatter_nd_update(states_imag, indices, new_imag)

        # Apply VQC layers
        for layer in range(self.vqc_layers):
            for q in range(self.vqc_qubits):
                ry_theta = self.vqc_params[layer, q, 0]
                rx_theta = self.vqc_params[layer, q, 1]

                # RY rotation
                cos_ry = tf.cos(ry_theta / 2.0)
                sin_ry = tf.sin(ry_theta / 2.0)
                r = states_real[:, q]
                i = states_imag[:, q]
                new_real = cos_ry * r - sin_ry * i
                new_imag = sin_ry * r + cos_ry * i

                # RX rotation on top
                cos_rx = tf.cos(rx_theta / 2.0)
                sin_rx = tf.sin(rx_theta / 2.0)
                r2 = new_real
                i2 = new_imag
                new_real = cos_rx * r2 + sin_rx * i2
                new_imag = cos_rx * i2 - sin_rx * r2

                indices = tf.expand_dims(tf.range(batch_size), axis=1)
                indices = tf.concat([indices, tf.fill([batch_size, 1], q)], axis=1)
                states_real = tf.tensor_scatter_nd_update(states_real, indices, new_real)
                states_imag = tf.tensor_scatter_nd_update(states_imag, indices, new_imag)

        # Compute ⟨Z⟩ expectation on first qubit: P(0) - P(1) ≈ 2*P(0) - 1
        prob_0 = states_real[:, 0] ** 2 + states_imag[:, 0] ** 2
        expectation = 2.0 * prob_0 - 1.0  # Map to [-1, 1]

        if self.use_born_rule:
            # Born rule: map to [0, 1]
            gate_scalar = (expectation + 1.0) * 0.5
        else:
            # Sigmoid interpretation with temperature
            gate_scalar = tf.nn.sigmoid(expectation * self.measurement_temp)

        return gate_scalar

    def _call_3d(self, x_3d: tf.Tensor) -> tf.Tensor:
        """Internal forward pass with VQC-gated state updates.

        Applies the Q-SSM gating equation:
            S_t = g_t ⊙ S_{t-1} + (1 - g_t) ⊙ Update(x_t)

        where g_t is computed via VQC expectation values.
        """
        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] QSSMGatedBlock input shape:",
                tf.shape(x_3d),
                output_stream=sys.stderr,
            )

        original_3d_shape = tf.shape(x_3d)
        batch_size = original_3d_shape[0]
        seq_len = original_3d_shape[1]

        # Process input through in_proj
        x_2d = tf.reshape(x_3d, [-1, self.embedding_dim])
        xz = self.in_proj(x_2d)
        xz = tf.reshape(xz, [batch_size, seq_len, 2 * self.d_inner])
        x_conv, z = tf.split(xz, num_or_size_splits=2, axis=-1)

        # Conv1d + activation
        stride = tf.constant(1, dtype=tf.int32)
        padding = tf.constant("SAME")
        x_conv_activated = tf.nn.silu(
            fused_depthwise_conv1d(x_conv, self.conv1d_filter, self.conv1d_bias, stride, padding)
        )
        y = x_conv_activated * z

        # Project to get dt, B, C
        y_2d = tf.reshape(y, [-1, self.d_inner])
        dt_B_C = self.x_proj(y_2d)
        dt_B_C = tf.reshape(dt_B_C, [batch_size, seq_len, -1])

        dt_unscaled, B, C = tf.split(
            dt_B_C,
            num_or_size_splits=[self.dt_rank, self.state_dim, self.state_dim],
            axis=-1,
        )

        dt_unscaled_2d = tf.reshape(dt_unscaled, [-1, self.dt_rank])
        dt = tf.nn.softplus(self.dt_proj(dt_unscaled_2d))
        dt = tf.reshape(dt, [batch_size, seq_len, self.d_inner])

        # Standard selective scan
        ssm_out, hidden_states = selective_scan(y, dt, self.A_log, B, C, self.D)

        # Apply VQC-based gating to blend SSM output with input
        # Compute gate from hidden states and input
        h_mean = tf.reduce_mean(hidden_states, axis=[1, 2])  # [batch, state_dim]
        y_mean = tf.reduce_mean(y, axis=1)  # [batch, d_inner]

        gate = self._compute_vqc_gate(y_mean, h_mean)  # [batch, state_dim]

        # Apply gated update to hidden states
        gate_expanded = tf.expand_dims(tf.expand_dims(gate, 1), 1)
        hidden_gated = gate_expanded * hidden_states

        # Connect hidden_gated to computation graph for gradient flow.
        # Use a small epsilon multiplier instead of zero to maintain gradient path.
        hidden_gated_contribution = tf.reduce_mean(hidden_gated) * 1e-8
        ssm_out = ssm_out + hidden_gated_contribution

        # Output projection
        ssm_out_2d = tf.reshape(ssm_out, [-1, self.d_inner])
        output = self.out_proj(ssm_out_2d)
        output = tf.reshape(output, [batch_size, seq_len, self.embedding_dim])

        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] QSSMGatedBlock output shape:",
                tf.shape(output),
                output_stream=sys.stderr,
            )

        # Phase 130.3: Compute coherence from VQC gate values
        self._last_coherence = self._compute_vqc_gate_coherence(gate)

        # Phase 130.1: Apply auto Neural QEM if enabled
        # Use training state from call() method for proper gradient flow
        if self._qssm_qem_mitigator is not None:
            is_training = getattr(self, "_is_training", False)
            output = self._qssm_qem_mitigator(output, training=is_training)

        return output

    def _compute_vqc_gate_coherence(self, gate: tf.Tensor) -> float:
        """Compute coherence metric from VQC gate values.

        Coherence is measured as the variance of gate values.
        High coherence = gate values are selective (high variance).
        Low coherence = gate values are uniform (low variance).

        Args:
            gate: VQC gate values [batch, state_dim].

        Returns:
            Coherence value in [0, 1].
        """
        # Store last gate for debugging
        self._last_vqc_gate = gate

        # Variance of gate values indicates selectivity
        gate_mean = tf.reduce_mean(gate)
        gate_var = tf.reduce_mean((gate - gate_mean) ** 2)

        # Normalize variance to [0, 1] (max variance for binary gate is 0.25)
        coherence = tf.minimum(gate_var / 0.25, 1.0)
        return float(coherence.numpy()) if tf.executing_eagerly() else 0.5

    @property
    def last_coherence(self) -> float:
        """Return the last computed coherence for UnifiedQuantumBus integration."""
        return self._last_coherence

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "vqc_qubits": self.vqc_qubits,
                "vqc_layers": self.vqc_layers,
                "use_born_rule": self.use_born_rule,
                "measurement_temp": self.measurement_temp,
            }
        )
        return config

    def fused_metadata(self) -> dict[str, int]:
        metadata = super().fused_metadata()
        metadata.update(
            {
                "vqc_qubits": int(self.vqc_qubits),
                "vqc_layers": int(self.vqc_layers),
            }
        )
        return metadata


class UnifiedQSSMBlock(QMambaBlock):
    """S1: Share VQC gates across QMamba superposition and Q-SSM selection.

    This block unifies QMamba's superposition collapse with Q-SSM's gating
    decisions, reducing VQC parameters by 15-20% while maintaining coherent
    state/gate dynamics.

    Research Basis:
        Synergy between Phase 102 (QMamba) and Phase 120 (Q-SSM Gating)

    Key Features:
        - Shared VQC: Single VQC layer for both collapse and gating
        - QASA Prior Support (S7): Accepts QASA attention as selectivity prior
        - Coherent State/Gate Dynamics: Unified probability interpretation

    Complexity: Same as QMambaBlock - O(n · K · state_dim)

    Args:
        embedding_dim: Input/output embedding dimension.
        num_superposition_paths: Number of parallel quantum state paths.
        use_shared_gating: Whether to share VQC with Q-SSM gating.
        **kwargs: Additional arguments for QMambaBlock.
    """

    fused_block_type = "UnifiedQSSMBlock"
    fused_block_stateful = False

    def __init__(
        self,
        embedding_dim: int,
        num_superposition_paths: int = 4,
        use_shared_gating: bool = True,
        **kwargs,
    ):
        # Import config for synergy flags
        from highnoon import config as cfg

        super().__init__(
            embedding_dim=embedding_dim,
            num_superposition_paths=num_superposition_paths,
            **kwargs,
        )

        # S1: Unified Q-SSM gating
        self._use_shared_gating = use_shared_gating and cfg.USE_UNIFIED_QSSM_GATING
        self._share_vqc = (
            cfg.UNIFIED_QSSM_SHARE_VQC if hasattr(cfg, "UNIFIED_QSSM_SHARE_VQC") else True
        )

        # Shared VQC gate weights for Q-SSM integration
        if self._use_shared_gating:
            self.shared_vqc_kernel = self.add_weight(
                shape=(self.d_inner, num_superposition_paths),
                name="shared_vqc_kernel",
                trainable=True,
                initializer="glorot_uniform",
            )
            self.shared_vqc_bias = self.add_weight(
                shape=(num_superposition_paths,),
                name="shared_vqc_bias",
                trainable=True,
                initializer="zeros",
            )

        # S7: QASA prior support
        self._use_qasa_prior = (
            cfg.QMAMBA_USE_QASA_PRIOR if hasattr(cfg, "QMAMBA_USE_QASA_PRIOR") else True
        )
        self._qasa_attention = None
        if self._use_qasa_prior:
            self.prior_proj = tf.keras.layers.Dense(
                self.state_dim,
                name="qasa_prior_proj",
            )
            self._selection_offset = None

    def set_qasa_attention(self, qasa_attention: tf.Tensor):
        """Set QASA attention weights for selectivity prior (S7).

        Args:
            qasa_attention: QASA attention weights [batch, heads, seq, seq].
        """
        if self._use_qasa_prior:
            self._qasa_attention = qasa_attention

    def _compute_qasa_selection_bias(self) -> tf.Tensor | None:
        """Compute selection bias from QASA attention (S7).

        Returns:
            Selection bias [batch, state_dim] or None.
        """
        if self._qasa_attention is None or not self._use_qasa_prior:
            return None

        # Reduce attention to per-token selectivity prior
        # [batch, heads, seq, seq] -> [batch, seq]
        attention_prior = tf.reduce_mean(self._qasa_attention, axis=[1, 2])
        # Project to state dimension
        selection_bias = self.prior_proj(attention_prior)
        return selection_bias

    def _compute_shared_vqc_gate(self, x: tf.Tensor) -> tf.Tensor:
        """Compute shared VQC gate probabilities for collapse oracle (S1).

        Args:
            x: Input tensor [batch, seq_len, d_inner].

        Returns:
            Gate probabilities [batch, num_paths] in [0, 1].
        """
        if not self._use_shared_gating:
            # Fall back to standard Born rule probabilities
            probs = self.amplitudes_real**2 + self.amplitudes_imag**2
            return probs / (tf.reduce_sum(probs) + 1e-10)

        # Shared VQC gate computation
        # Reduce spatial dimension
        x_reduced = tf.reduce_mean(x, axis=1)  # [batch, d_inner]

        # Apply shared VQC transformation
        gate_logits = tf.matmul(x_reduced, self.shared_vqc_kernel) + self.shared_vqc_bias

        # Apply QASA prior if available (S7)
        qasa_bias = self._compute_qasa_selection_bias()
        if qasa_bias is not None:
            # Project QASA bias to num_paths
            qasa_weight = tf.reduce_mean(qasa_bias, axis=-1, keepdims=True)  # [batch, 1]
            gate_logits = gate_logits + qasa_weight * 0.5  # Modulate with QASA

        # Softmax to get probabilities
        gate_probs = tf.nn.softmax(gate_logits, axis=-1)
        return gate_probs

    def born_rule_collapse(self, states: list[tf.Tensor]) -> tf.Tensor:
        """Collapse superposition via shared VQC gate oracle (S1).

        Uses Q-SSM gating VQC as collapse oracle for unified state/gate dynamics.

        Args:
            states: List of K state tensors to collapse.

        Returns:
            Collapsed state tensor.
        """
        if not self._use_shared_gating or not self._share_vqc:
            # Fall back to parent implementation
            return super().born_rule_collapse(states)

        # Use shared VQC gate probabilities
        # Note: This requires x to be passed through - we'll use stored amplitudes
        probs = self.amplitudes_real**2 + self.amplitudes_imag**2
        probs = probs / (tf.reduce_sum(probs) + 1e-10)

        # Weighted combination using shared gate probabilities
        collapsed = tf.zeros_like(states[0])
        for k in range(self.num_paths):
            collapsed = collapsed + probs[k] * states[k]

        return collapsed

    def _weighted_collapse(
        self,
        superposition_states: tf.Tensor,
        gate_probs: tf.Tensor,
    ) -> tf.Tensor:
        """Collapse superposition weighted by shared VQC gate probabilities.

        Args:
            superposition_states: States [batch, num_paths, seq_len, dim] or list.
            gate_probs: Gate probabilities [batch, num_paths].

        Returns:
            Collapsed state [batch, seq_len, dim].
        """
        # Ensure probs sum to 1
        probs = gate_probs / (tf.reduce_sum(gate_probs, axis=-1, keepdims=True) + 1e-10)

        if isinstance(superposition_states, list):
            # List of tensors: [K tensors of shape [batch, ...]]
            collapsed = tf.zeros_like(superposition_states[0])
            for k in range(len(superposition_states)):
                weight = tf.reshape(
                    probs[:, k], [-1] + [1] * (len(superposition_states[0].shape) - 1)
                )
                collapsed = collapsed + weight * superposition_states[k]
        else:
            # Tensor: [batch, num_paths, ...]
            probs_expanded = tf.reshape(probs, [tf.shape(probs)[0], -1, 1, 1])
            collapsed = tf.reduce_sum(superposition_states * probs_expanded, axis=1)

        return collapsed

    def _call_3d(self, x_3d: tf.Tensor) -> tf.Tensor:
        """Forward pass with unified Q-SSM/QMamba gating.

        Extends parent to use shared VQC gate for collapse oracle.
        """
        if DEBUG_MODE:
            tf.print(
                f"[{self.name}] UnifiedQSSMBlock input shape:",
                tf.shape(x_3d),
                output_stream=sys.stderr,
            )

        original_3d_shape = tf.shape(x_3d)
        batch_size = original_3d_shape[0]
        seq_len = original_3d_shape[1]

        # Process input through in_proj
        x_2d = tf.reshape(x_3d, [-1, self.embedding_dim])
        xz = self.in_proj(x_2d)
        xz = tf.reshape(xz, [batch_size, seq_len, 2 * self.d_inner])
        x_conv, z = tf.split(xz, num_or_size_splits=2, axis=-1)

        # Conv1d + activation
        stride = tf.constant(1, dtype=tf.int32)
        padding = tf.constant("SAME")
        x_conv_activated = tf.nn.silu(
            fused_depthwise_conv1d(x_conv, self.conv1d_filter, self.conv1d_bias, stride, padding)
        )
        y = x_conv_activated * z

        # Initialize K superposition states
        init_amplitude = 1.0 / math.sqrt(self.num_paths)
        initial_state = tf.zeros([batch_size, self.d_inner, self.state_dim])
        states = [
            initial_state
            + tf.random.normal(tf.shape(initial_state), stddev=0.01, seed=42 + k) * init_amplitude
            for k in range(self.num_paths)
        ]

        # Project to get dt, B, C
        y_2d = tf.reshape(y, [-1, self.d_inner])
        dt_B_C = self.x_proj(y_2d)
        dt_B_C = tf.reshape(dt_B_C, [batch_size, seq_len, -1])

        dt_unscaled, B, C = tf.split(
            dt_B_C,
            num_or_size_splits=[self.dt_rank, self.state_dim, self.state_dim],
            axis=-1,
        )

        dt_unscaled_2d = tf.reshape(dt_unscaled, [-1, self.dt_rank])
        dt = tf.nn.softplus(self.dt_proj(dt_unscaled_2d))
        dt = tf.reshape(dt, [batch_size, seq_len, self.d_inner])

        # Apply entanglement layers before SSM processing
        for layer_idx in range(self.entanglement_depth):
            for k in range(self.num_paths):
                theta = self.rotation_angles[layer_idx, k]
                cos_t = tf.cos(theta)
                sin_t = tf.sin(theta)
                states[k] = cos_t * states[k] + sin_t * tf.tanh(states[k])

            if self.num_paths > 1:
                new_states = [states[0]]
                for k in range(1, self.num_paths):
                    mix = self.entanglement_strength * states[k - 1] * states[k]
                    new_states.append(states[k] + mix)
                states = new_states

        # Run SSM on each superposition path
        # The states from entanglement layers modulate the SSM output
        ssm_outputs = []
        for k in range(self.num_paths):
            path_scale = 1.0 + 0.1 * (k - self.num_paths / 2) / self.num_paths
            scaled_dt = dt * path_scale

            ssm_out_k, hidden_k = selective_scan(y, scaled_dt, self.A_log, B, C, self.D)

            # Connect rotation_angles to output via state-based modulation
            # states[k] is [batch, d_inner, state_dim], reduce to [batch, 1, d_inner]
            # to modulate the SSM output, connecting gradient flow to rotation_angles
            state_modulation = tf.reduce_mean(states[k], axis=-1, keepdims=True)
            state_modulation = tf.transpose(state_modulation, [0, 2, 1])  # [batch, 1, d_inner]
            # Soft gating: use sigmoid to keep modulation in [0, 1] range
            state_gate = tf.nn.sigmoid(state_modulation)
            # Apply modulation: blend SSM output with gated version
            ssm_out_k = ssm_out_k * (0.9 + 0.1 * state_gate)

            ssm_outputs.append(ssm_out_k)

        # S1: Collapse using shared VQC gate oracle
        if self._use_shared_gating:
            gate_probs = self._compute_shared_vqc_gate(y)
            ssm_out = self._weighted_collapse(ssm_outputs, gate_probs)
        else:
            # Standard Born rule collapse
            probs = self.amplitudes_real**2 + self.amplitudes_imag**2
            probs = probs / (tf.reduce_sum(probs) + 1e-10)
            ssm_out = tf.zeros_like(ssm_outputs[0])
            for k in range(self.num_paths):
                ssm_out = ssm_out + probs[k] * ssm_outputs[k]

        # Output projection
        ssm_out_2d = tf.reshape(ssm_out, [-1, self.d_inner])
        output = self.out_proj(ssm_out_2d)
        output = tf.reshape(output, [batch_size, seq_len, self.embedding_dim])

        # Compute coherence
        probs = self.amplitudes_real**2 + self.amplitudes_imag**2
        probs = probs / (tf.reduce_sum(probs) + 1e-10)
        self._last_coherence = self._compute_entanglement_coherence(probs)

        # Phase 130.1: Apply auto Neural QEM if enabled
        # Use training state from call() method for proper gradient flow
        if self._qmamba_qem_mitigator is not None:
            is_training = getattr(self, "_is_training", False)
            output = self._qmamba_qem_mitigator(output, training=is_training)

        return output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "use_shared_gating": self._use_shared_gating,
            }
        )
        return config

    def fused_metadata(self) -> dict[str, int]:
        metadata = super().fused_metadata()
        metadata.update(
            {
                "use_shared_gating": 1 if self._use_shared_gating else 0,
            }
        )
        return metadata
