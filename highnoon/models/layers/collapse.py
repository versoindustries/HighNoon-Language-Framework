# highnoon/models/layers/collapse.py
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

"""Contextual Gating Collapse layer for superposition state collapse.

This layer uses a cross-attention mechanism to collapse the superposition
dimension of a tensor, conditioned on a context vector. It takes the original
token embedding as the context to generate a Query vector and the superposed
states to generate Key and Value vectors.

Phase 16 Enhancements:
- Gumbel-Softmax unified training/inference path
- C++ SIMD-optimized fused kernel support
- Optional kernel attention linearization (ELU+1, ReLU²)
- Temperature annealing for gradual sharpening
"""

import logging

import tensorflow as tf
from tensorflow.keras import layers

# C++ kernel import
from highnoon._native.ops.fused_collapse_op import (
    fused_collapse,
    fused_collapse_available,
    gumbel_softmax_tf,
)

# Phase 16 configuration imports
from highnoon.config import (
    COLLAPSE_ADAPTIVE_STRATEGY,
    COLLAPSE_ENTROPY_THRESHOLD,
    COLLAPSE_GUMBEL_ANNEAL_STEPS,
    COLLAPSE_GUMBEL_FINAL_TEMP,
    COLLAPSE_GUMBEL_INITIAL_TEMP,
    COLLAPSE_HARD_SAMPLES,
    COLLAPSE_KERNEL_FEATURE_MAP,
    COLLAPSE_KERNEL_THRESHOLD,
    COLLAPSE_USE_GUMBEL_SOFTMAX,
    COLLAPSE_USE_KERNEL_ATTENTION,
    FUSED_COLLAPSE_SIMD_THRESHOLD,
    USE_FUSED_COLLAPSE_OP,
)

logger = logging.getLogger(__name__)


class ContextualGatingCollapse(layers.Layer):
    """Implements the Contextual Gating Collapse mechanism.

    This layer uses a cross-attention mechanism to collapse the superposition
    dimension of a tensor, conditioned on a context vector. It takes the
    original token embedding as the context to generate a Query vector and the
    superposed states to generate Key and Value vectors.

    Phase 16 Features:
    - Gumbel-Softmax: Unified differentiable sampling for train/inference
    - C++ Kernel: SIMD-optimized fused forward/backward pass
    - Kernel Attention: Optional ELU+1/ReLU² feature maps for O(S) complexity
    - Adaptive Collapse: Entropy-based soft/hard blending

    Attributes:
        d_in: Dimension of the context input.
        d_out: Dimension of the superposed input and output.
        num_heads: Number of attention heads.
        use_gumbel_softmax: Enable Gumbel-Softmax unified path.
        use_fused_op: Enable C++ fused kernel.
        use_kernel_attention: Enable kernel attention feature maps.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int = 4,
        use_gumbel_softmax: bool | None = None,
        use_fused_op: bool | None = None,
        use_kernel_attention: bool | None = None,
        initial_temperature: float | None = None,
        **kwargs,
    ):
        """Initialize the ContextualGatingCollapse layer.

        Args:
            d_in: The dimension of the context input (x_context).
            d_out: The dimension of the superposed input (y_superposed).
            num_heads: The number of attention heads.
            use_gumbel_softmax: Enable Gumbel-Softmax. Defaults to config.
            use_fused_op: Enable C++ kernel. Defaults to config.
            use_kernel_attention: Enable kernel attention. Defaults to config.
            initial_temperature: Initial temperature. Defaults to config.
            **kwargs: Standard Keras layer arguments.
        """
        super().__init__(**kwargs)
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads

        if self.d_out % self.num_heads != 0:
            raise ValueError(
                f"'d_out' ({self.d_out}) must be divisible by 'num_heads' ({self.num_heads})."
            )
        self.head_dim = self.d_out // self.num_heads

        # Phase 16 configuration
        self.use_gumbel_softmax = (
            use_gumbel_softmax if use_gumbel_softmax is not None else COLLAPSE_USE_GUMBEL_SOFTMAX
        )
        self.use_fused_op = use_fused_op if use_fused_op is not None else USE_FUSED_COLLAPSE_OP
        self.use_kernel_attention = (
            use_kernel_attention
            if use_kernel_attention is not None
            else COLLAPSE_USE_KERNEL_ATTENTION
        )
        self.initial_temperature = (
            initial_temperature if initial_temperature is not None else COLLAPSE_GUMBEL_INITIAL_TEMP
        )
        self.final_temperature = COLLAPSE_GUMBEL_FINAL_TEMP
        self.anneal_steps = COLLAPSE_GUMBEL_ANNEAL_STEPS
        self.use_hard_samples = COLLAPSE_HARD_SAMPLES
        self.use_adaptive_strategy = COLLAPSE_ADAPTIVE_STRATEGY
        self.entropy_threshold = COLLAPSE_ENTROPY_THRESHOLD

        # Feature map mapping
        feature_map_str = COLLAPSE_KERNEL_FEATURE_MAP
        self.feature_map_id = {
            "softmax": 0,
            "elu_plus_one": 1,
            "relu_squared": 2,
        }.get(feature_map_str, 0)

        # Temperature variable (non-trainable, can be adjusted externally)
        self.temperature = tf.Variable(
            self.initial_temperature,
            trainable=False,
            name="collapse_temperature",
            dtype=tf.float32,
        )

        # Step counter for temperature annealing
        self.step_counter = tf.Variable(0, trainable=False, name="collapse_step", dtype=tf.int64)

        # Projections for Query, Key, Value
        self.query_proj = layers.Dense(self.d_out, name="collapse_query_proj")
        self.key_proj = layers.Dense(self.d_out, name="collapse_key_proj")
        self.value_proj = layers.Dense(self.d_out, name="collapse_value_proj")

        # Final output projection
        self.output_proj = layers.Dense(self.d_out, name="collapse_output_proj")

        # Check C++ kernel availability
        self._fused_op_available = self.use_fused_op and fused_collapse_available()
        if self.use_fused_op and not self._fused_op_available:
            logger.warning(
                "C++ fused_collapse op not available, falling back to Python implementation. "
                "Rebuild with ./build_secure.sh to enable."
            )

    def build(self, input_shape):
        """Explicitly builds the internal Dense layers.

        This enhanced version is more strict about its expected input shape
        to prevent silent errors during graph construction.
        """
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            superposed_shape, context_shape = input_shape
        else:
            raise ValueError(
                "ContextualGatingCollapse expects a list or tuple of two input shapes: "
                f"(superposed_shape, context_shape). Received: {input_shape}"
            )

        if not self.query_proj.built:
            self.query_proj.build(context_shape)
        if not self.key_proj.built:
            self.key_proj.build(superposed_shape)
        if not self.value_proj.built:
            self.value_proj.build(superposed_shape)

        output_proj_input_shape = tf.TensorShape([None, self.d_out])
        if not self.output_proj.built:
            self.output_proj.build(output_proj_input_shape)

        super().build(input_shape)

    def _update_temperature(self):
        """Update temperature based on step counter (annealing)."""
        if self.anneal_steps > 0:
            progress = tf.cast(self.step_counter, tf.float32) / float(self.anneal_steps)
            progress = tf.minimum(progress, 1.0)
            # Linear annealing from initial to final temperature
            new_temp = self.initial_temperature + progress * (
                self.final_temperature - self.initial_temperature
            )
            self.temperature.assign(new_temp)
        self.step_counter.assign_add(1)

    def _gumbel_softmax(
        self,
        logits: tf.Tensor,
        temperature: float,
        hard: bool = False,
    ) -> tf.Tensor:
        """Differentiable Gumbel-Softmax sampling.

        Args:
            logits: Input logits [batch, num_classes].
            temperature: Softmax temperature.
            hard: Use straight-through hard samples.

        Returns:
            Soft or hard samples.
        """
        return gumbel_softmax_tf(logits, temperature, hard)

    def _kernel_attention_feature_map(self, x: tf.Tensor) -> tf.Tensor:
        """Apply kernel attention feature map.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor with feature map applied.
        """
        if self.feature_map_id == 1:
            # ELU+1: φ(x) = elu(x) + 1
            return tf.nn.elu(x) + 1.0
        elif self.feature_map_id == 2:
            # ReLU²: φ(x) = relu(x)²
            return tf.square(tf.nn.relu(x))
        else:
            # Standard softmax (no feature map)
            return x

    def _call_fused(self, y_superposed, x_context, training):
        """Forward pass using C++ fused kernel.

        Args:
            y_superposed: Superposed states [batch, S, d_out].
            x_context: Context tensor [batch, d_in].
            training: Whether in training mode.

        Returns:
            Collapsed output [batch, d_out].
        """
        # Update temperature during training
        if training:
            self._update_temperature()

        output = fused_collapse(
            context=x_context,
            superposed=y_superposed,
            q_weights=self.query_proj.kernel,
            k_weights=self.key_proj.kernel,
            v_weights=self.value_proj.kernel,
            o_weights=self.output_proj.kernel,
            q_bias=self.query_proj.bias,
            k_bias=self.key_proj.bias,
            v_bias=self.value_proj.bias,
            o_bias=self.output_proj.bias,
            num_heads=self.num_heads,
            temperature=float(self.temperature),
            training=training if training is not None else True,
            use_kernel_attention=self.use_kernel_attention,
            feature_map=self.feature_map_id,
        )

        return output

    def _call_python(self, y_superposed, x_context, training):
        """Forward pass using Python implementation.

        Args:
            y_superposed: Superposed states [batch, S, d_out].
            x_context: Context tensor [batch, d_in].
            training: Whether in training mode.

        Returns:
            Collapsed output [batch, d_out].
        """
        batch_size = tf.shape(y_superposed)[0]
        superposition_dim = tf.shape(y_superposed)[1]

        # Update temperature during training
        if training:
            self._update_temperature()

        # 1. Project inputs to Q, K, V
        query = self.query_proj(x_context)
        key = self.key_proj(y_superposed)
        value = self.value_proj(y_superposed)

        # 2. Reshape for multi-head attention
        query = tf.reshape(query, (batch_size, 1, self.num_heads, self.head_dim))
        query = tf.transpose(query, perm=[0, 2, 1, 3])  # [B, H, 1, d_h]

        key = tf.reshape(key, (batch_size, -1, self.num_heads, self.head_dim))
        key = tf.transpose(key, perm=[0, 2, 1, 3])  # [B, H, S, d_h]

        value = tf.reshape(value, (batch_size, -1, self.num_heads, self.head_dim))
        value = tf.transpose(value, perm=[0, 2, 1, 3])  # [B, H, S, d_h]

        # 3. Compute attention scores
        if self.use_kernel_attention and superposition_dim > COLLAPSE_KERNEL_THRESHOLD:
            # Kernel attention path (O(S) linear complexity)
            phi_q = self._kernel_attention_feature_map(query)  # [B, H, 1, d_h]
            phi_k = self._kernel_attention_feature_map(key)  # [B, H, S, d_h]

            # Compute attention using kernel trick
            # numerator = φ(Q) @ (φ(K)^T @ V)
            kv = tf.einsum("bhsd,bhsv->bhdv", phi_k, value)  # [B, H, d_h, d_h]
            numerator = tf.einsum("bhqd,bhdv->bhqv", phi_q, kv)  # [B, H, 1, d_h]

            # denominator = φ(Q) @ sum(φ(K))
            k_sum = tf.reduce_sum(phi_k, axis=2, keepdims=True)  # [B, H, 1, d_h]
            denominator = tf.reduce_sum(phi_q * k_sum, axis=-1, keepdims=True)  # [B, H, 1, 1]

            context_vector = numerator / (denominator + 1e-6)  # [B, H, 1, d_h]
        else:
            # Standard attention path
            attention_scores = tf.matmul(query, key, transpose_b=True)  # [B, H, 1, S]
            dk = tf.cast(tf.shape(key)[-1], tf.float32)
            scaled_attention_scores = attention_scores / tf.math.sqrt(dk)

            if self.use_gumbel_softmax:
                # Gumbel-Softmax unified path
                # Reshape to [B*H, S] for Gumbel-Softmax
                logits_flat = tf.reshape(
                    scaled_attention_scores, [batch_size * self.num_heads, superposition_dim]
                )

                # Use hard samples at inference if configured
                use_hard = self.use_hard_samples and not training
                attention_weights_flat = self._gumbel_softmax(
                    logits_flat,
                    float(self.temperature),
                    hard=use_hard,
                )

                # Reshape back to [B, H, 1, S]
                attention_weights = tf.reshape(
                    attention_weights_flat,
                    [batch_size, self.num_heads, 1, superposition_dim],
                )

                # Add diversity loss during training
                if training:
                    entropy_val = -tf.reduce_sum(
                        attention_weights * tf.math.log(attention_weights + 1e-9),
                        axis=-1,
                    )
                    diversity_loss = -tf.reduce_mean(entropy_val)
                    self.add_loss(diversity_loss * 0.01)  # Scaled down

                context_vector = tf.matmul(attention_weights, value)  # [B, H, 1, d_h]

            elif self.use_adaptive_strategy:
                # Adaptive collapse: blend soft and hard based on entropy
                attention_weights = tf.nn.softmax(scaled_attention_scores, axis=-1)

                # Compute entropy per head
                entropy = -tf.reduce_sum(
                    attention_weights * tf.math.log(attention_weights + 1e-9),
                    axis=-1,
                )  # [B, H, 1]

                # Blend factor: high entropy → soft, low entropy → hard
                blend_factor = tf.sigmoid(self.entropy_threshold - entropy)
                blend_factor = tf.expand_dims(blend_factor, -1)  # [B, H, 1, 1]

                # Soft output
                soft_output = tf.matmul(attention_weights, value)

                # Hard output (argmax selection)
                hard_indices = tf.argmax(attention_weights, axis=-1)  # [B, H, 1]
                hard_output = tf.gather(value, hard_indices, batch_dims=2)  # [B, H, 1, d_h]

                context_vector = blend_factor * soft_output + (1 - blend_factor) * hard_output

            else:
                # Original behavior: train/infer branching
                if training:
                    attention_weights = tf.nn.softmax(scaled_attention_scores, axis=-1)

                    # Diversity loss
                    entropy_val = -tf.reduce_sum(
                        attention_weights * tf.math.log(attention_weights + 1e-9),
                        axis=-1,
                    )
                    diversity_loss = -tf.reduce_mean(entropy_val)
                    self.add_loss(diversity_loss)

                    context_vector = tf.matmul(attention_weights, value)
                else:
                    # Inference: probabilistic sampling
                    logits_with_temp = scaled_attention_scores / self.temperature
                    reshaped_logits = tf.reshape(
                        logits_with_temp,
                        [batch_size * self.num_heads, superposition_dim],
                    )
                    sampled_indices = tf.random.categorical(
                        reshaped_logits, num_samples=1, dtype=tf.int32
                    )
                    reshaped_values = tf.reshape(
                        value,
                        [batch_size * self.num_heads, superposition_dim, self.head_dim],
                    )
                    context_vector = tf.gather(reshaped_values, sampled_indices, batch_dims=1)
                    context_vector = tf.reshape(
                        context_vector,
                        [batch_size, self.num_heads, 1, self.head_dim],
                    )

        # 4. Concatenate heads and project
        context_vector = tf.transpose(context_vector, perm=[0, 2, 1, 3])
        context_vector = tf.reshape(context_vector, (batch_size, -1, self.d_out))
        collapsed_vector = tf.squeeze(context_vector, axis=1)
        output = self.output_proj(collapsed_vector)

        return output

    def call(self, inputs, training=None):
        """Forward pass for the Contextual Gating Collapse mechanism.

        Args:
            inputs: Must be a tuple of two tensors:
                - y_superposed: Tensor with superposition dimension [B, S, d_out].
                - x_context: Context tensor [B, d_in].
            training: Standard Keras training flag.

        Returns:
            Collapsed output tensor [B, d_out].
        """
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            y_superposed, x_context = inputs
        else:
            raise ValueError(
                "ContextualGatingCollapse expects a list or tuple of two input tensors: "
                f"(y_superposed, x_context). Received: {inputs}"
            )

        # Dispatch to C++ kernel or Python implementation
        superposition_dim = tf.shape(y_superposed)[1]

        if self._fused_op_available and superposition_dim >= FUSED_COLLAPSE_SIMD_THRESHOLD:
            return self._call_fused(y_superposed, x_context, training)
        else:
            return self._call_python(y_superposed, x_context, training)

    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "d_in": self.d_in,
                "d_out": self.d_out,
                "num_heads": self.num_heads,
                "use_gumbel_softmax": self.use_gumbel_softmax,
                "use_fused_op": self.use_fused_op,
                "use_kernel_attention": self.use_kernel_attention,
                "initial_temperature": self.initial_temperature,
            }
        )
        return config
