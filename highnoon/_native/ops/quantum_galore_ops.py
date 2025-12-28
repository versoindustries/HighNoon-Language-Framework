# highnoon/_native/ops/quantum_galore_ops.py
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

"""Phase 91: Python wrappers for Quantum GaLore C++ operations.

This module provides Python interfaces to the C++ Quantum GaLore operations
for entropy-based dynamic rank selection and quantum random feature projection.

Note: This module requires the native C++ ops. No Python fallbacks are provided.

Example:
    >>> from highnoon._native.ops.quantum_galore_ops import quantum_galore_project
    >>> compressed, actual_rank = quantum_galore_project(
    ...     gradient, eigenvalues, rotation_matrix, bias, max_rank=32
    ... )
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# Load native ops
_NATIVE_OPS_AVAILABLE = False
_native_lib = None

try:
    from highnoon._native import _load_consolidated_binary, is_consolidated_available

    if is_consolidated_available():
        # Load the consolidated binary to register ops with TensorFlow
        _native_lib = _load_consolidated_binary()
        if _native_lib is not None:
            # Check if the ops are accessible via the library module
            _NATIVE_OPS_AVAILABLE = hasattr(_native_lib, "quantum_galore_project") or \
                                    hasattr(_native_lib, "compute_effective_rank")
            if not _NATIVE_OPS_AVAILABLE:
                # TensorFlow registers ops with snake_case names on the loaded module
                # Try to detect by looking for any GaLore-related attribute
                lib_attrs = [a for a in dir(_native_lib) if 'galore' in a.lower() or 'rank' in a.lower()]
                _NATIVE_OPS_AVAILABLE = len(lib_attrs) > 0
            if _NATIVE_OPS_AVAILABLE:
                logger.debug("[QUANTUM_GALORE] Native ops loaded successfully")
            else:
                # Ops may still be available via gen_* module pattern
                _NATIVE_OPS_AVAILABLE = True  # Assume available, will fail gracefully
                logger.debug("[QUANTUM_GALORE] Assuming native ops available")
except ImportError as e:
    logger.warning("[QUANTUM_GALORE] Native library not available: %s", e)
except Exception as e:
    logger.warning("[QUANTUM_GALORE] Failed to load native ops: %s", e)


def is_native_available() -> bool:
    """Check if native Quantum GaLore ops are available."""
    return _NATIVE_OPS_AVAILABLE


def _require_native():
    """Raise error if native ops not available."""
    if not _NATIVE_OPS_AVAILABLE or _native_lib is None:
        raise RuntimeError(
            "[QUANTUM_GALORE] Native C++ ops required but not available. "
            "Rebuild with: ./build_secure.sh --debug --lite"
        )


def _get_op(op_name: str):
    """Get an op from the native library.
    
    TensorFlow registers ops with both snake_case and PascalCase names.
    Special handling for 'GaLore' capitalization.
    """
    _require_native()
    
    # Try exact name first
    if hasattr(_native_lib, op_name):
        return getattr(_native_lib, op_name)
    
    # Try PascalCase (convert from snake_case)
    # Special handling: 'galore' -> 'GaLore' (not 'Galore')
    pascal_name = ''.join(word.capitalize() for word in op_name.split('_'))
    pascal_name = pascal_name.replace('Galore', 'GaLore')  # Fix GaLore capitalization
    if hasattr(_native_lib, pascal_name):
        return getattr(_native_lib, pascal_name)
    
    # List available relevant ops for debugging
    available = [a for a in dir(_native_lib) if op_name.replace('_', '').lower() in a.lower()]
    raise RuntimeError(
        f"[QUANTUM_GALORE] Op '{op_name}' not found in native library. "
        f"Tried: {pascal_name}. Available similar: {available}"
    )


# =============================================================================
# Core Operations (C++ only, no fallbacks)
# =============================================================================


def compute_effective_rank(
    eigenvalues: tf.Tensor,
    max_rank: int = 32,
    min_rank: int = 4,
) -> tf.Tensor:
    """Compute effective rank from eigenvalue spectrum entropy.

    Uses Shannon entropy: effective_rank = exp(-Σ p_i log(p_i))
    where p_i = λ_i / Σλ_j are normalized eigenvalues.

    Args:
        eigenvalues: Sorted eigenvalues (descending order) [N].
        max_rank: Maximum rank cap.
        min_rank: Minimum rank floor.

    Returns:
        Scalar int32 tensor with effective rank.

    Raises:
        RuntimeError: If native ops not available.
    """
    _require_native()
    op = _get_op("compute_effective_rank")
    return op(eigenvalues=eigenvalues, max_rank=max_rank, min_rank=min_rank)


def compute_block_influence(
    gradient_norms: tf.Tensor,
    weight_norms: tf.Tensor,
) -> tf.Tensor:
    """Compute Taylor expansion influence scores for block-wise allocation.

    Influence = ||∇W||² / ||W||² approximates Fisher information diagonal.

    Args:
        gradient_norms: L2 norms of gradients per block [num_blocks].
        weight_norms: L2 norms of weights per block [num_blocks].

    Returns:
        Normalized influence scores (sum to 1) [num_blocks].

    Raises:
        RuntimeError: If native ops not available.
    """
    _require_native()
    op = _get_op("compute_block_influence")
    return op(gradient_norms=gradient_norms, weight_norms=weight_norms)


def allocate_block_ranks(
    influence_scores: tf.Tensor,
    total_rank_budget: int = 256,
    min_rank_per_block: int = 4,
    critical_block_ids: list[int] | None = None,
) -> tf.Tensor:
    """Allocate rank budget across blocks based on influence scores.

    Critical blocks (first/last layers) receive minimum 1.5x average allocation.

    Args:
        influence_scores: Normalized influence scores per block [num_blocks].
        total_rank_budget: Total rank budget to distribute.
        min_rank_per_block: Minimum rank per block.
        critical_block_ids: Indices of critical blocks (e.g., [0, N-1]).

    Returns:
        Rank allocation per block [num_blocks] as int32.

    Raises:
        RuntimeError: If native ops not available.
    """
    _require_native()
    if critical_block_ids is None:
        critical_block_ids = []
    op = _get_op("allocate_block_ranks")
    return op(
        influence_scores=influence_scores,
        total_rank_budget=total_rank_budget,
        min_rank_per_block=min_rank_per_block,
        critical_block_ids=critical_block_ids,
    )


def quantum_galore_project(
    gradient: tf.Tensor,
    eigenvalues: tf.Tensor,
    rotation_matrix: tf.Tensor,
    bias: tf.Tensor,
    max_rank: int = 32,
    min_rank: int = 4,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Project gradient to low-rank space using quantum random features.

    Uses entropy-based dynamic rank selection and quantum feature map
    projection for stable, memory-efficient gradient compression.

    Args:
        gradient: Input gradient tensor [rows, cols].
        eigenvalues: Pre-computed singular values (descending order).
        rotation_matrix: Quantum random rotation parameters.
        bias: Quantum random bias values.
        max_rank: Maximum allowable projection rank.
        min_rank: Minimum projection rank.

    Returns:
        Tuple of:
        - compressed: Compressed gradient [effective_rank, cols] or [rows, effective_rank]
        - actual_rank: Scalar int32 with the rank used

    Raises:
        RuntimeError: If native ops not available.
    """
    _require_native()
    op = _get_op("quantum_galore_project")
    return op(
        gradient=gradient,
        eigenvalues=eigenvalues,
        rotation_matrix=rotation_matrix,
        bias=bias,
        max_rank=max_rank,
        min_rank=min_rank,
    )


def quantum_galore_deproject(
    compressed: tf.Tensor,
    rotation_matrix: tf.Tensor,
    bias: tf.Tensor,
    original_shape: Tuple[int, int],
    row_projection: bool = True,
) -> tf.Tensor:
    """Reconstruct full gradient from low-rank compressed representation.

    Uses quantum random feature adjoint mapping for reconstruction.

    Args:
        compressed: Compressed gradient [rank, cols] or [rows, rank].
        rotation_matrix: Same rotation parameters used in projection.
        bias: Same bias values used in projection.
        original_shape: Original gradient shape (rows, cols).
        row_projection: Whether original projection was along rows.

    Returns:
        Reconstructed gradient [rows, cols].

    Raises:
        RuntimeError: If native ops not available.
    """
    _require_native()
    op = _get_op("quantum_galore_deproject")
    return op(
        compressed=compressed,
        rotation_matrix=rotation_matrix,
        bias=bias,
        original_shape=tf.constant(list(original_shape), dtype=tf.int32),
        row_projection=row_projection,
    )


def init_quantum_random_features(
    rank: int,
    dim: int,
    seed: int = 42,
    scale: float | None = None,
) -> Tuple[tf.Variable, tf.Variable]:
    """Initialize quantum random feature parameters.

    Creates rotation matrix and bias values for quantum feature map.

    Args:
        rank: Target projection rank.
        dim: Dimension of rotation (rows or cols of gradient).
        seed: Random seed for reproducibility.
        scale: Scaling factor (default: 1/sqrt(dim)).

    Returns:
        Tuple of:
        - rotation_matrix: Variable [rank, dim]
        - bias: Variable [rank]
    """
    if scale is None:
        scale = 1.0 / np.sqrt(dim)

    tf.random.set_seed(seed)

    rotation_matrix = tf.Variable(
        tf.random.normal([rank, dim], stddev=scale),
        trainable=False,
        name="quantum_galore_rotation",
    )

    bias = tf.Variable(
        tf.random.uniform([rank], 0, 2 * np.pi),
        trainable=False,
        name="quantum_galore_bias",
    )

    return rotation_matrix, bias


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "is_native_available",
    "compute_effective_rank",
    "compute_block_influence",
    "allocate_block_ranks",
    "quantum_galore_project",
    "quantum_galore_deproject",
    "init_quantum_random_features",
]
