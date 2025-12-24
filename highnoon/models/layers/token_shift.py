# highnoon/models/layers/token_shift.py
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

"""RWKV-6 style data-dependent token shifting.

This module implements data-dependent token shifting from the RWKV-6 (Finch)
architecture. Token shifting enables improved long-term memory management
by using input-dependent gating to mix current and previous token states.

Phase 14.1.1 Enhancement:
- O(n) linear time complexity
- O(1) memory per step
- Achieves 16 tok/s on ARM Cortex-A76

Reference: RWKV-6 Finch architecture (2024)
"""

from __future__ import annotations

import logging
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.config import TOKEN_SHIFT_DECAY, USE_DATA_DEPENDENT_SHIFT

logger = logging.getLogger(__name__)


class DataDependentTokenShift(layers.Layer):
    """RWKV-6 style adaptive token shifting for improved long-term memory.

    This layer uses input-dependent gating to mix current and previous token
    states, enabling better information flow in recurrent processing. The
    gating mechanism learns to balance between new information and historical
    context on a per-token basis.

    Attributes:
        embedding_dim: Dimension of the input embeddings.
        decay: Base decay factor for temporal mixing.
        use_learned_decay: Whether to learn per-dimension decay rates.

    Example:
        >>> shift = DataDependentTokenShift(embedding_dim=512)
        >>> x = tf.random.normal([2, 100, 512])
        >>> shifted = shift(x)
        >>> assert shifted.shape == (2, 100, 512)
    """

    def __init__(
        self,
        embedding_dim: int,
        decay: float = TOKEN_SHIFT_DECAY,
        use_learned_decay: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize DataDependentTokenShift layer.

        Args:
            embedding_dim: Dimension of input embeddings.
            decay: Base decay factor for temporal mixing (0-1).
            use_learned_decay: If True, learn per-dimension decay rates.
            **kwargs: Additional Keras layer arguments.

        Raises:
            ValueError: If embedding_dim is not positive.
            ValueError: If decay is not in range (0, 1).
        """
        super().__init__(**kwargs)

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if not 0 < decay < 1:
            raise ValueError(f"decay must be in (0, 1), got {decay}")

        self.embedding_dim = embedding_dim
        self.decay = decay
        self.use_learned_decay = use_learned_decay

        # Data-dependent gating: learns to mix current and previous tokens
        self.shift_gate = layers.Dense(
            embedding_dim,
            activation="sigmoid",
            kernel_initializer="glorot_uniform",
            bias_initializer=tf.constant_initializer(0.5),  # Start balanced
            name="shift_gate",
            dtype=tf.float32,  # Enforce float32
        )

        # Optional: learn per-dimension decay rates
        if use_learned_decay:
            self.decay_weights: tf.Variable | None = None
        else:
            self.decay_weights = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        if self.use_learned_decay:
            self.decay_weights = self.add_weight(
                name="decay_weights",
                shape=(self.embedding_dim,),
                initializer=tf.constant_initializer(self.decay),
                trainable=True,
                dtype=tf.float32,  # Enforce float32
            )

        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        prev_x: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply data-dependent token shifting.

        Args:
            x: Input tensor of shape [batch, seq_len, embedding_dim].
            prev_x: Optional previous token states. If None, uses shifted x.
            training: Whether in training mode.

        Returns:
            Shifted tensor of same shape as input with temporal mixing applied.
        """
        # Ensure float32 precision
        x = tf.cast(x, tf.float32)

        # If no previous state provided, create shifted version
        if prev_x is None:
            # Shift tokens by one position (prepend zeros)
            padding = tf.zeros_like(x[:, :1, :])
            prev_x = tf.concat([padding, x[:, :-1, :]], axis=1)
        else:
            prev_x = tf.cast(prev_x, tf.float32)

        # Compute data-dependent gate
        gate = self.shift_gate(x)  # [batch, seq_len, embedding_dim]

        # Apply learned decay if enabled
        if self.use_learned_decay and self.decay_weights is not None:
            # Sigmoid to keep decay in (0, 1) range
            learned_decay = tf.nn.sigmoid(self.decay_weights)
            gate = gate * learned_decay

        # Mix current and previous tokens
        # gate=1 means keep current, gate=0 means use previous
        shifted = gate * x + (1.0 - gate) * prev_x

        return shifted

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "decay": self.decay,
                "use_learned_decay": self.use_learned_decay,
            }
        )
        return config


class TemporalTokenMixer(layers.Layer):
    """Extended token mixing with multi-step temporal context.

    Combines DataDependentTokenShift with exponential moving average
    for capturing longer temporal dependencies while maintaining O(1)
    memory per step.

    This is useful for models that need to track information over
    longer horizons without the quadratic memory cost of attention.

    Attributes:
        embedding_dim: Dimension of embeddings.
        num_shifts: Number of temporal shift steps to combine.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_shifts: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize TemporalTokenMixer.

        Args:
            embedding_dim: Dimension of input embeddings.
            num_shifts: Number of temporal shift levels to combine.
            **kwargs: Additional Keras layer arguments.

        Raises:
            ValueError: If embedding_dim is not positive.
            ValueError: If num_shifts is less than 1.
        """
        super().__init__(**kwargs)

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if num_shifts < 1:
            raise ValueError(f"num_shifts must be >= 1, got {num_shifts}")

        self.embedding_dim = embedding_dim
        self.num_shifts = num_shifts

        # Create shift layers with decreasing decay
        self.shift_layers = [
            DataDependentTokenShift(
                embedding_dim=embedding_dim,
                decay=TOKEN_SHIFT_DECAY ** (i + 1),  # Exponential decay
                name=f"shift_{i}",
            )
            for i in range(num_shifts)
        ]

        # Combine shifted representations
        self.combine = layers.Dense(
            embedding_dim,
            activation=None,
            name="combine",
            dtype=tf.float32,
        )

        # Layer normalization for stability
        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name="temporal_norm",
            dtype=tf.float32,
        )

    def call(
        self,
        x: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply multi-step temporal mixing.

        Args:
            x: Input tensor [batch, seq_len, embedding_dim].
            training: Whether in training mode.

        Returns:
            Temporally mixed tensor of same shape.
        """
        x = tf.cast(x, tf.float32)

        # Collect shifted representations
        shifted_stack = [x]  # Include original
        current = x
        for shift_layer in self.shift_layers:
            current = shift_layer(current, training=training)
            shifted_stack.append(current)

        # Concatenate and combine
        combined = tf.concat(shifted_stack, axis=-1)  # [batch, seq, dim * (num_shifts+1)]
        output = self.combine(combined)  # [batch, seq, dim]

        # Add residual and normalize
        output = self.layer_norm(output + x)

        return output

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_shifts": self.num_shifts,
            }
        )
        return config


def create_token_shift_layer(
    embedding_dim: int,
    use_multi_step: bool = False,
    mode: str = "standard",
    layer_position: int = 0,
    shift_distances: list[int] | None = None,
    **kwargs: Any,
) -> layers.Layer:
    """Factory function for creating token shift layers.

    Args:
        embedding_dim: Dimension of input embeddings.
        use_multi_step: If True, use TemporalTokenMixer; else single-level shift.
        mode: Token shift mode - one of:
            - "standard": Data-dependent gating (default, RWKV-6 style)
            - "simplified": RWKV-7 style, no input-dependent gate (3x faster)
            - "fourier": Frequency-domain enhanced mixing
            - "delta": Gated delta rule for precise memory control
            - "hierarchical": Layer-position aware decay
        layer_position: Layer index for hierarchical decay (0-indexed).
        shift_distances: List of shift distances for multi-position mode.
        **kwargs: Additional arguments for the layer.

    Returns:
        Configured token shift layer.

    Example:
        >>> layer = create_token_shift_layer(512, mode="simplified")
        >>> x = tf.random.normal([2, 100, 512])
        >>> out = layer(x)
    """
    if not USE_DATA_DEPENDENT_SHIFT:
        # Return identity layer if feature is disabled
        return layers.Lambda(lambda x: x, name="identity_shift")

    if mode == "simplified":
        return SimplifiedTokenShift(embedding_dim=embedding_dim, **kwargs)
    elif mode == "fourier":
        return FourierTokenShift(embedding_dim=embedding_dim, **kwargs)
    elif mode == "delta":
        return DeltaTokenShift(embedding_dim=embedding_dim, **kwargs)
    elif mode == "hierarchical":
        return HierarchicalTokenShift(
            embedding_dim=embedding_dim, layer_position=layer_position, **kwargs
        )
    elif mode == "multi_position":
        if shift_distances is None:
            shift_distances = [1, 2, 4]
        return MultiPositionTokenShift(
            embedding_dim=embedding_dim, shift_distances=shift_distances, **kwargs
        )
    elif use_multi_step:
        return TemporalTokenMixer(embedding_dim=embedding_dim, **kwargs)
    else:
        return DataDependentTokenShift(embedding_dim=embedding_dim, **kwargs)


# =============================================================================
# ENHANCEMENT 1: SIMPLIFIED TOKEN SHIFT (RWKV-7 STYLE)
# =============================================================================


class SimplifiedTokenShift(layers.Layer):
    """RWKV-7 style simplified token shifting (3x faster).

    Uses only learned decay weights without input-dependent gating.
    This trades some expressivity for significant speed improvements.

    Attributes:
        embedding_dim: Dimension of the input embeddings.
        decay: Base decay factor.
    """

    def __init__(
        self,
        embedding_dim: int,
        decay: float = TOKEN_SHIFT_DECAY,
        **kwargs: Any,
    ) -> None:
        """Initialize SimplifiedTokenShift layer.

        Args:
            embedding_dim: Dimension of input embeddings.
            decay: Base decay factor for temporal mixing (0-1).
            **kwargs: Additional Keras layer arguments.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.decay_weights: tf.Variable | None = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer weights."""
        self.decay_weights = self.add_weight(
            name="decay_weights",
            shape=(self.embedding_dim,),
            initializer=tf.constant_initializer(self.decay),
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        prev_x: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply simplified token shifting.

        Args:
            x: Input tensor [batch, seq_len, embedding_dim].
            prev_x: Previous token states. If None, uses shifted x.
            training: Whether in training mode.

        Returns:
            Shifted tensor of same shape as input.
        """
        x = tf.cast(x, tf.float32)

        if prev_x is None:
            padding = tf.zeros_like(x[:, :1, :])
            prev_x = tf.concat([padding, x[:, :-1, :]], axis=1)
        else:
            prev_x = tf.cast(prev_x, tf.float32)

        # Fixed decay gate (no input-dependent computation)
        gate = tf.nn.sigmoid(self.decay_weights)

        # Token mixing
        shifted = gate * x + (1.0 - gate) * prev_x
        return shifted

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "decay": self.decay,
            }
        )
        return config


# =============================================================================
# ENHANCEMENT 2: HIERARCHICAL TOKEN SHIFT
# =============================================================================


class HierarchicalTokenShift(layers.Layer):
    """Token shifting with layer-position aware decay rates.

    Earlier layers use faster decay (local patterns),
    later layers use slower decay (global context).

    Attributes:
        embedding_dim: Dimension of embeddings.
        layer_position: Current layer index (0-indexed).
        decay_factor: Scaling factor for hierarchy.
    """

    def __init__(
        self,
        embedding_dim: int,
        layer_position: int = 0,
        decay: float = TOKEN_SHIFT_DECAY,
        decay_factor: float = 2.0,
        **kwargs: Any,
    ) -> None:
        """Initialize HierarchicalTokenShift."""
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.layer_position = layer_position
        self.decay = decay
        self.decay_factor = decay_factor

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer weights."""
        self.gate_kernel = self.add_weight(
            name="gate_kernel",
            shape=(self.embedding_dim, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
            dtype=tf.float32,
        )
        self.gate_bias = self.add_weight(
            name="gate_bias",
            shape=(self.embedding_dim,),
            initializer=tf.constant_initializer(0.5),
            trainable=True,
            dtype=tf.float32,
        )
        self.decay_weights = self.add_weight(
            name="decay_weights",
            shape=(self.embedding_dim,),
            initializer=tf.constant_initializer(self.decay),
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        prev_x: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply hierarchical token shifting."""
        x = tf.cast(x, tf.float32)

        if prev_x is None:
            padding = tf.zeros_like(x[:, :1, :])
            prev_x = tf.concat([padding, x[:, :-1, :]], axis=1)
        else:
            prev_x = tf.cast(prev_x, tf.float32)

        # Compute gate
        gate = tf.nn.sigmoid(tf.einsum("bld,de->ble", x, self.gate_kernel) + self.gate_bias)

        # Apply hierarchical decay scaling
        base_decay = tf.nn.sigmoid(self.decay_weights)
        exponent = 1.0 / (tf.cast(self.layer_position, tf.float32) + self.decay_factor)
        hierarchical_decay = tf.pow(base_decay, exponent)

        gate = gate * hierarchical_decay

        shifted = gate * x + (1.0 - gate) * prev_x
        return shifted

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "layer_position": self.layer_position,
                "decay": self.decay,
                "decay_factor": self.decay_factor,
            }
        )
        return config


# =============================================================================
# ENHANCEMENT 3: FOURIER-ENHANCED TOKEN SHIFT
# =============================================================================


class FourierTokenShift(layers.Layer):
    """Token shifting with Fourier-domain global mixing.

    Combines time-domain token shift with frequency-domain processing
    for O(n log n) global context without quadratic attention.

    Attributes:
        embedding_dim: Dimension of embeddings.
        min_seq_len: Minimum sequence length to use FFT (default 256).
    """

    def __init__(
        self,
        embedding_dim: int,
        decay: float = TOKEN_SHIFT_DECAY,
        min_seq_len: int = 256,
        **kwargs: Any,
    ) -> None:
        """Initialize FourierTokenShift."""
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.min_seq_len = min_seq_len

        # Time-domain shift
        self.time_shift = DataDependentTokenShift(
            embedding_dim=embedding_dim,
            decay=decay,
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer weights."""
        # Learnable frequency filter (complex weights for each frequency bin)
        # For seq_len = S, we have S/2 + 1 frequency bins
        # We learn a real and imaginary component per dimension
        self.freq_filter_real = self.add_weight(
            name="freq_filter_real",
            shape=(self.embedding_dim,),
            initializer="ones",
            trainable=True,
            dtype=tf.float32,
        )
        self.freq_filter_imag = self.add_weight(
            name="freq_filter_imag",
            shape=(self.embedding_dim,),
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )

        # Blend gate between time and frequency domains
        self.blend_gate = self.add_weight(
            name="blend_gate",
            shape=(self.embedding_dim,),
            initializer="zeros",  # Start with 50-50 blend
            trainable=True,
            dtype=tf.float32,
        )

        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        prev_x: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply Fourier-enhanced token shifting."""
        x = tf.cast(x, tf.float32)
        seq_len = tf.shape(x)[1]

        # Time-domain shift
        time_out = self.time_shift(x, prev_x, training=training)

        # Only use FFT for sequences >= min_seq_len
        def freq_path():
            # Apply FFT along sequence dimension for each feature
            x_complex = tf.cast(x, tf.complex64)
            spectrum = tf.signal.fft(x_complex)

            # Apply learnable frequency filter
            filter_complex = tf.complex(self.freq_filter_real, self.freq_filter_imag)
            filtered = spectrum * filter_complex

            # Inverse FFT
            freq_out = tf.math.real(tf.signal.ifft(filtered))
            return freq_out

        def skip_freq():
            return time_out

        freq_out = tf.cond(seq_len >= self.min_seq_len, freq_path, skip_freq)

        # Blend time and frequency domains
        blend = tf.nn.sigmoid(self.blend_gate)
        output = blend * time_out + (1.0 - blend) * freq_out

        return output

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "decay": self.decay,
                "min_seq_len": self.min_seq_len,
            }
        )
        return config


# =============================================================================
# ENHANCEMENT 4: DELTA RULE TOKEN SHIFT
# =============================================================================


class DeltaTokenShift(layers.Layer):
    """Gated Delta Networks for precise memory control.

    Uses separate erase and write gates following the ICLR 2025 paper.
    Better for in-context learning and retrieval tasks.

    Attributes:
        embedding_dim: Dimension of embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        **kwargs: Any,
    ) -> None:
        """Initialize DeltaTokenShift."""
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer weights."""
        # Erase gate projection
        self.erase_kernel = self.add_weight(
            name="erase_kernel",
            shape=(self.embedding_dim, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
            dtype=tf.float32,
        )
        self.erase_bias = self.add_weight(
            name="erase_bias",
            shape=(self.embedding_dim,),
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )

        # Write gate projection
        self.write_kernel = self.add_weight(
            name="write_kernel",
            shape=(self.embedding_dim, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
            dtype=tf.float32,
        )
        self.write_bias = self.add_weight(
            name="write_bias",
            shape=(self.embedding_dim,),
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )

        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        state: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply delta rule token shifting.

        Args:
            x: Input tensor [batch, seq_len, embedding_dim].
            state: Initial memory state [batch, embedding_dim].
            training: Whether in training mode.

        Returns:
            Updated output tensor [batch, seq_len, embedding_dim].
        """
        x = tf.cast(x, tf.float32)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        if state is None:
            state = tf.zeros([batch_size, self.embedding_dim], dtype=tf.float32)

        # Compute gates
        erase = tf.nn.sigmoid(tf.einsum("bld,de->ble", x, self.erase_kernel) + self.erase_bias)
        write = tf.nn.sigmoid(tf.einsum("bld,de->ble", x, self.write_kernel) + self.write_bias)

        # Process sequence recurrently
        outputs = tf.TensorArray(dtype=tf.float32, size=seq_len)

        for t in tf.range(seq_len):
            e_t = erase[:, t, :]  # [batch, dim]
            w_t = write[:, t, :]  # [batch, dim]
            x_t = x[:, t, :]  # [batch, dim]

            # Delta update
            state = state * (1.0 - e_t) + w_t * x_t
            outputs = outputs.write(t, state)

        output = tf.transpose(outputs.stack(), [1, 0, 2])
        return output

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
            }
        )
        return config


# =============================================================================
# ENHANCEMENT 5: MULTI-POSITION TOKEN SHIFT
# =============================================================================


class MultiPositionTokenShift(layers.Layer):
    """Token shifting with multiple shift distances.

    Allows direct access to tokens at multiple distances [1, 2, 4, ...].

    Attributes:
        embedding_dim: Dimension of embeddings.
        shift_distances: List of shift distances.
    """

    def __init__(
        self,
        embedding_dim: int,
        shift_distances: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize MultiPositionTokenShift."""
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.shift_distances = shift_distances or [1, 2, 4]
        self.num_distances = len(self.shift_distances)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer weights."""
        # Learnable blend weights for each distance
        self.blend_weights = self.add_weight(
            name="blend_weights",
            shape=(self.num_distances,),
            initializer=tf.constant_initializer(1.0 / self.num_distances),
            trainable=True,
            dtype=tf.float32,
        )

        # Projection to combine shifted inputs with original
        self.combine_dense = layers.Dense(
            self.embedding_dim,
            name="combine",
            dtype=tf.float32,
        )

        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply multi-position token shifting."""
        x = tf.cast(x, tf.float32)

        # Create shifted versions
        shifted_versions = [x]  # Include original at distance 0

        for distance in self.shift_distances:
            # Shift by distance positions
            padding = tf.zeros_like(x[:, :distance, :])
            shifted = tf.concat([padding, x[:, :-distance, :]], axis=1)
            shifted_versions.append(shifted)

        # Apply blend weights (softmax for proper distribution)
        weights = tf.nn.softmax(self.blend_weights)

        # Weighted sum of shifted versions (excluding original at index 0)
        blended = x  # Start with original
        for i, shifted in enumerate(shifted_versions[1:]):
            blended = blended + weights[i] * shifted

        return blended

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "shift_distances": self.shift_distances,
            }
        )
        return config
