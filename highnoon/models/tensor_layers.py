# src/models/tensor_layers.py
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

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers, layers


class TTLayer(layers.Layer):
    """
    A Tensor Train (TT) layer for efficient linear transformations.

    This layer implements a linear transformation where the weight matrix W
    is represented in a Tensor Train decomposition format. This allows for
    a dramatic reduction in the number of parameters compared to a standard
    Dense layer, especially for large input and output dimensions.

    The layer "tensorizes" the input and output dimensions, meaning it reshapes
    the conceptual M x N weight matrix into a higher-order tensor. This tensor
    is then decomposed into a chain of smaller, 4D "core" tensors.

    For example, an input dimension of 512 could be factorized into
    input_dims=[8, 8, 8], and an output of 2048 into output_dims=[8, 8, 32].
    The layer then learns the TT-cores that represent this high-dimensional mapping.
    """

    def __init__(self, input_dims: list, output_dims: list, tt_ranks: list, **kwargs):
        """
        Initializes the TTLayer.

        Args:
            input_dims (list): A list of integers representing the factorization
                of the input dimension. The product of these integers must equal
                the total input size. E.g., for 512, this could be [8, 8, 8].
            output_dims (list): A list of integers representing the factorization
                of the output dimension. Must have the same length as input_dims.
            tt_ranks (list): A list of integers specifying the TT-ranks of the
                decomposition. Must have length len(input_dims) + 1. The first
                and last ranks must be 1. E.g., [1, 16, 16, 1].
            **kwargs: Standard Keras layer arguments.
        """
        super().__init__(**kwargs)

        # --- Validate Inputs ---
        if len(input_dims) != len(output_dims):
            raise ValueError("input_dims and output_dims must have the same length.")
        if len(tt_ranks) != len(input_dims) + 1:
            raise ValueError("tt_ranks must have length len(input_dims) + 1.")
        if tt_ranks[0] != 1 or tt_ranks[-1] != 1:
            raise ValueError("The first and last TT-ranks must be 1.")

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.tt_ranks = tt_ranks
        self.d = len(input_dims)  # Number of cores

        # Calculate total input and output dimensions
        self.input_dim = np.prod(input_dims)
        self.output_dim = np.prod(output_dims)

        self.cores = []

    def build(self, input_shape):
        """
        Creates the trainable weights of the layer (the TT-cores).

        This method is called by Keras automatically the first time the layer is
        used. It creates a list of 4D tensors, one for each core in the TT chain.
        """
        if input_shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.input_dim}, "
                f"but received input with shape {input_shape}."
            )

        for k in range(self.d):
            # Each core G_k has shape (r_{k-1}, m_k, n_k, r_k)
            core_shape = (
                self.tt_ranks[k],
                self.input_dims[k],
                self.output_dims[k],
                self.tt_ranks[k + 1],
            )
            # Use a standard Glorot uniform initializer
            initializer = initializers.GlorotUniform()
            self.cores.append(
                self.add_weight(
                    name=f"tt_core_{k}", shape=core_shape, initializer=initializer, trainable=True
                )
            )
        super().build(input_shape)

    def call(self, inputs):
        """
        Implements the forward pass using a stable sequence of einsum
        operations, which is robust for graph-mode execution.
        """
        batch_size = tf.shape(inputs)[0]
        # Reshape input to [B, m_1, m_2, ..., m_d]
        res = tf.reshape(inputs, [batch_size] + self.input_dims)

        # Define indices for einsum
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        in_indices = alphabet[1 : self.d + 1]
        out_indices = "BCDEFGHIJKLMNOPQRSTUVWXYZ"[: self.d]
        rank_indices = alphabet[self.d + 1 : self.d + 1 + self.d + 1]
        res_indices = alphabet[0] + in_indices

        # Sequentially contract with each core
        for i in range(self.d):
            core = self.cores[i]  # Shape (r_in, m_in, n_out, r_out)

            # Define indices for this core
            rank_in_idx = rank_indices[i]
            input_idx = in_indices[i]
            output_idx = out_indices[i]
            rank_out_idx = rank_indices[i + 1]
            core_indices = f"{rank_in_idx}{input_idx}{output_idx}{rank_out_idx}"

            # Build the einsum string dynamically for the output
            res_indices_list = list(res_indices)
            next_indices_list = [res_indices_list[0]]  # keep batch dim
            next_indices_list.extend(list(out_indices[:i]))  # keep previous output dims
            next_indices_list.append(output_idx)  # add new output dim
            next_indices_list.extend(list(in_indices[i + 1 :]))  # keep remaining input dims
            next_indices_list.append(rank_out_idx)  # add new rank dim
            out_indices_str = "".join(next_indices_list)

            if i == 0:
                # First core has rank 1 on the input side, so we squeeze it.
                core_squeezed = tf.squeeze(core, axis=0)
                einsum_str = (
                    f"{res_indices},{input_idx}{output_idx}{rank_out_idx}->{out_indices_str}"
                )
                res = tf.einsum(einsum_str, res, core_squeezed)
            else:
                einsum_str = f"{res_indices},{core_indices}->{out_indices_str}"
                res = tf.einsum(einsum_str, res, core)

            res_indices = out_indices_str

        # After the loop, res has shape [B, n1, ..., nd, 1]
        res = tf.squeeze(res, axis=-1)

        # Final reshape to the dense output vector: [B, D_out]
        output = tf.reshape(res, [batch_size, self.output_dim])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dims": self.input_dims,
                "output_dims": self.output_dims,
                "tt_ranks": self.tt_ranks,
            }
        )
        return config


# --- START: SuperpositionTTLayer Implementation ---
class SuperpositionTTLayer(layers.Layer):
    """
    A Tensor Train (TT) layer augmented with a superposition dimension.

    This layer extends the TTLayer to represent multiple parameter sets in a
    single tensor structure. The additional `superposition_dim` is treated
    like a batch dimension during the forward pass, allowing for parallel
    computation across different parameter "universes".
    """

    def __init__(
        self, input_dims: list, output_dims: list, tt_ranks: list, superposition_dim: int, **kwargs
    ):
        """
        Initializes the SuperpositionTTLayer.

        Args:
            input_dims (list): Factorization of the input dimension.
            output_dims (list): Factorization of the output dimension.
            tt_ranks (list): TT-ranks for the decomposition.
            superposition_dim (int): The number of parameter sets to hold in superposition.
            **kwargs: Standard Keras layer arguments.
        """
        super().__init__(**kwargs)

        if len(input_dims) != len(output_dims):
            raise ValueError("input_dims and output_dims must have the same length.")
        if len(tt_ranks) != len(input_dims) + 1:
            raise ValueError("tt_ranks must have length len(input_dims) + 1.")
        if tt_ranks[0] != 1 or tt_ranks[-1] != 1:
            raise ValueError("The first and last TT-ranks must be 1.")

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.tt_ranks = tt_ranks
        self.superposition_dim = superposition_dim
        self.d = len(input_dims)

        self.input_dim = np.prod(input_dims)
        self.output_dim = np.prod(output_dims)

        self.cores = []

    def build(self, input_shape):
        """
        Creates the trainable weights (the TT-cores), augmenting their shape
        with the superposition dimension.
        """
        if input_shape[-1] != self.input_dim:
            # Allow for sequence inputs [B, T, D]
            if len(input_shape) != 3 or input_shape[-1] != self.input_dim:
                raise ValueError(
                    f"Input dimension mismatch. Expected last dim to be {self.input_dim}, "
                    f"but received input with shape {input_shape}."
                )

        for k in range(self.d):
            # Each core G_k now has shape (r_k, m_k, n_k, r_{k+1}, superposition_dim)
            core_shape = (
                self.tt_ranks[k],
                self.input_dims[k],
                self.output_dims[k],
                self.tt_ranks[k + 1],
                self.superposition_dim,
            )
            initializer = initializers.GlorotUniform()
            self.cores.append(
                self.add_weight(
                    name=f"superposition_tt_core_{k}",
                    shape=core_shape,
                    initializer=initializer,
                    trainable=True,
                )
            )
        super().build(input_shape)

    def call(self, inputs):
        """
        Implements the forward pass, preserving the superposition dimension.
        Uses tf.einsum for explicit control over tensor contractions.
        """
        input_shape = tf.shape(inputs)
        is_sequence_input = len(inputs.shape) == 3

        if is_sequence_input:
            batch_size, seq_len, _ = inputs.shape
            # Reshape [B, T, D] -> [B*T, D] to process as a batch
            reshaped_inputs = tf.reshape(inputs, [-1, self.input_dim])
            effective_batch_size = input_shape[0] * input_shape[1]
        else:
            reshaped_inputs = inputs
            effective_batch_size = input_shape[0]
        res = tf.reshape(reshaped_inputs, [effective_batch_size] + self.input_dims)

        # einsum alphabet
        alphabet = "abcdefghijklmnopqrstuvwxyz"

        # Initial input indices: 'a' for batch, 'b,c,d...' for input modes
        in_indices = alphabet[1 : self.d + 1]
        out_indices = "BCDEFGHIJKLMNOPQRSTUVWXYZ"[: self.d]
        rank_indices = alphabet[self.d + 1 : self.d + 1 + self.d + 1]

        res_indices = alphabet[0] + in_indices

        # Sequentially contract with each core
        for i in range(self.d):
            core = self.cores[i]  # Shape (r_in, m_in, n_out, r_out, S)

            # Define indices for this core
            rank_in_idx = rank_indices[i]
            input_idx = in_indices[i]
            output_idx = out_indices[i]
            rank_out_idx = rank_indices[i + 1]

            core_indices = f"{rank_in_idx}{input_idx}{output_idx}{rank_out_idx}s"

            # Build the einsum string dynamically
            res_indices_list = list(res_indices)
            next_indices_list = [res_indices_list[0]]  # keep batch

            # Add existing output modes
            next_indices_list.extend(list(out_indices[:i]))
            # Add new output mode
            next_indices_list.append(output_idx)
            # Add remaining input modes
            next_indices_list.extend(list(in_indices[i + 1 :]))
            # Add new rank and superposition dim
            next_indices_list.extend([rank_out_idx, "s"])

            out_indices_str = "".join(next_indices_list)

            # Handle first core where rank_in is 1
            if i == 0:
                core_squeezed = tf.squeeze(core, axis=0)
                einsum_str = (
                    f"{res_indices},{input_idx}{output_idx}{rank_out_idx}s->{out_indices_str}"
                )
                res = tf.einsum(einsum_str, res, core_squeezed)

            else:
                einsum_str = f"{res_indices},{core_indices}->{out_indices_str}"
                res = tf.einsum(einsum_str, res, core)

            res_indices = out_indices_str

        # After loop, res has shape [B, n1, ..., nd, 1, S]
        res = tf.squeeze(res, axis=-2)

        # Final reshape to the dense output vector: [B, D_out, S]
        output = tf.reshape(res, [effective_batch_size, self.output_dim, self.superposition_dim])

        if is_sequence_input:
            # Reshape back to [B, T, D_out, S]
            output = tf.reshape(
                output, [input_shape[0], input_shape[1], self.output_dim, self.superposition_dim]
            )

        return output

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:  # [B, T, D_in]
            return (input_shape[0], input_shape[1], self.output_dim, self.superposition_dim)
        # [B, D_in]
        return (input_shape[0], self.output_dim, self.superposition_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dims": self.input_dims,
                "output_dims": self.output_dims,
                "tt_ranks": self.tt_ranks,
                "superposition_dim": self.superposition_dim,
            }
        )
        return config


# --- END: SuperpositionTTLayer Implementation ---

# Phase 14.4.3: Import config for hybrid TN-NN
from highnoon.config import USE_HYBRID_TN  # noqa: E402


class HybridTTLayer(layers.Layer):
    """Hybrid Tensor Network-Neural Network layer.

    Combines TT decomposition with neural pre/post processing for Phase 14.4.3.

    CONSTRAINT: All computations use float32 precision. No quantization.

    Attributes:
        input_dims: Factorization of input dimension.
        output_dims: Factorization of output dimension.
        tt_ranks: TT-ranks for decomposition.
    """

    def __init__(
        self,
        input_dims: list,
        output_dims: list,
        tt_ranks: list,
        use_preprocessing: bool = True,
        use_postprocessing: bool = True,
        dropout_rate: float = 0.1,
        name: str = "hybrid_tt",
        **kwargs,
    ):
        """Initialize HybridTTLayer."""
        super().__init__(name=name, **kwargs)

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.tt_ranks = tt_ranks
        self.use_preprocessing = use_preprocessing and USE_HYBRID_TN
        self.use_postprocessing = use_postprocessing and USE_HYBRID_TN
        self.dropout_rate = dropout_rate

        self.input_dim = int(np.prod(input_dims))
        self.output_dim = int(np.prod(output_dims))

        # Core TT layer
        self.tt_core = TTLayer(
            input_dims=input_dims,
            output_dims=output_dims,
            tt_ranks=tt_ranks,
            name=f"{name}_tt_core",
        )

        # Neural preprocessing
        if self.use_preprocessing:
            self.pre_norm = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_pre_norm")
            self.pre_dense = layers.Dense(
                self.input_dim, activation="gelu", name=f"{name}_pre_dense"
            )
            self.pre_gate = layers.Dense(
                self.input_dim, activation="sigmoid", name=f"{name}_pre_gate"
            )
            self.pre_dropout = layers.Dropout(dropout_rate)

        # Neural postprocessing
        if self.use_postprocessing:
            self.post_norm = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_post_norm")
            self.post_dense = layers.Dense(
                self.output_dim, activation="gelu", name=f"{name}_post_dense"
            )
            self.post_gate = layers.Dense(
                self.output_dim, activation="sigmoid", name=f"{name}_post_gate"
            )
            self.post_dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        """Forward pass through hybrid TT layer."""
        x = tf.cast(inputs, tf.float32)
        original_shape = tf.shape(x)
        is_sequence = len(x.shape) == 3

        if is_sequence:
            x = tf.reshape(x, [-1, self.input_dim])

        # Neural preprocessing
        if self.use_preprocessing:
            residual = x
            x = self.pre_norm(x)
            x = self.pre_dense(x)
            x = self.pre_dropout(x, training=training)
            gate = self.pre_gate(residual)
            x = gate * x + (1 - gate) * residual

        # TT contraction
        x = self.tt_core(x)

        # Neural postprocessing
        if self.use_postprocessing:
            residual = x
            x = self.post_norm(x)
            x = self.post_dense(x)
            x = self.post_dropout(x, training=training)
            gate = self.post_gate(residual)
            x = gate * x + (1 - gate) * residual

        if is_sequence:
            x = tf.reshape(x, [original_shape[0], original_shape[1], self.output_dim])

        return x

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        if len(input_shape) == 3:
            return (input_shape[0], input_shape[1], self.output_dim)
        return (input_shape[0], self.output_dim)

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "input_dims": self.input_dims,
                "output_dims": self.output_dims,
                "tt_ranks": self.tt_ranks,
                "use_preprocessing": self.use_preprocessing,
                "use_postprocessing": self.use_postprocessing,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


def create_hybrid_tt_layer(input_dims: list, output_dims: list, tt_ranks: list, **kwargs):
    """Factory function for creating hybrid or standard TT layer."""
    if not USE_HYBRID_TN:
        return TTLayer(input_dims=input_dims, output_dims=output_dims, tt_ranks=tt_ranks)
    return HybridTTLayer(
        input_dims=input_dims, output_dims=output_dims, tt_ranks=tt_ranks, **kwargs
    )
