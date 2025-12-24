# highnoon/_native/ops/structured_sparse_matmul.py
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

"""Python wrapper for the Structured Sparse Matmul custom C++ operation.

Provides efficient band-diagonal sparse matrix-vector multiplication,
useful for local attention patterns and structured sparsity.
"""

import logging

import tensorflow as tf

from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

# --- Load the Custom Operator ---
_sparse_matmul_module = None
structured_sparse_matmul_op = None
structured_sparse_matmul_grad_op = None

try:
    _op_lib_path = resolve_op_library(__file__, "_structured_sparse_matmul_op.so")
    _sparse_matmul_module = tf.load_op_library(_op_lib_path)

    if hasattr(_sparse_matmul_module, "structured_sparse_matmul"):
        structured_sparse_matmul_op = _sparse_matmul_module.structured_sparse_matmul
        structured_sparse_matmul_grad_op = getattr(
            _sparse_matmul_module, "structured_sparse_matmul_grad", None
        )
        logger.info("Successfully loaded custom C++ StructuredSparseMatmul operator.")
    else:
        raise AttributeError("structured_sparse_matmul op not found in library")
except (tf.errors.NotFoundError, OSError, AttributeError) as e:
    logger.warning(f"Could not load the custom C++ StructuredSparseMatmul op: {e}")
    structured_sparse_matmul_op = None


def structured_sparse_matmul_available() -> bool:
    """Check if the structured sparse matmul operation is available."""
    return structured_sparse_matmul_op is not None


@tf.custom_gradient
def structured_sparse_matmul(
    matrix_diagonals: tf.Tensor,
    vector: tf.Tensor,
    structure: str = "band_diagonal",
    lower_bands: int = 64,
    upper_bands: int = 64,
) -> tf.Tensor:
    """Python wrapper for the StructuredSparseMatmul custom C++ operator.

    Performs efficient band-diagonal sparse matrix-vector multiplication.
    This is useful for implementing local attention patterns where each
    token only attends to a fixed window of neighbors.

    Args:
        matrix_diagonals: Sparse matrix in diagonal format.
            Shape: [num_rows, num_diagonals] where num_diagonals = lower_bands + upper_bands + 1.
            The diagonals are stored as: [d_{-kl}, ..., d_0, ..., d_{ku}]
        vector: Input vector to multiply.
            Shape: [batch_size, num_cols].
        structure: Sparse structure type. Currently only "band_diagonal" is supported.
        lower_bands: Number of lower diagonals (kl). Default: 64.
        upper_bands: Number of upper diagonals (ku). Default: 64.

    Returns:
        Result of the sparse matrix-vector multiplication.
        Shape: [batch_size, num_rows].

    Raises:
        NotImplementedError: If the C++ operation is not available.
    """
    if structured_sparse_matmul_op is None:
        raise NotImplementedError(
            "The C++ StructuredSparseMatmul operator could not be loaded. "
            "Rebuild ops via build_secure.sh."
        )

    # Convert string and int arguments to tensors for the C++ op
    structure_tensor = tf.constant(structure, dtype=tf.string)
    lower_bands_tensor = tf.constant(lower_bands, dtype=tf.int32)
    upper_bands_tensor = tf.constant(upper_bands, dtype=tf.int32)

    result = structured_sparse_matmul_op(
        matrix_diagonals,
        vector,
        structure_tensor,
        lower_bands_tensor,
        upper_bands_tensor,
    )

    def grad_fn(
        grad_output: tf.Tensor,
        variables: list[tf.Variable] | None = None,
    ) -> tuple[tuple[tf.Tensor, ...], list[tf.Tensor | None]]:
        """Gradient function that calls the custom C++ backward kernel."""
        if structured_sparse_matmul_grad_op is None:
            raise NotImplementedError("The C++ StructuredSparseMatmulGrad operator is unavailable.")

        grads = structured_sparse_matmul_grad_op(
            grad_output,
            matrix_diagonals,
            vector,
            structure_tensor,
            lower_bands_tensor,
            upper_bands_tensor,
        )

        # Returns gradients for: matrix_diagonals, vector (None for structure, lower_bands, upper_bands)
        input_grads = (grads[0], grads[1], None, None, None)
        variable_grads_list = [None] * len(variables) if variables is not None else []

        return input_grads, variable_grads_list

    return result, grad_fn


__all__ = [
    "structured_sparse_matmul",
    "structured_sparse_matmul_available",
]
