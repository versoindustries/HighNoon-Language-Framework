# highnoon/_native/ops/quantum_coherence_bus_ops.py
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

"""Python wrappers for Quantum Coherence Bus C++ operations (Phase 76/127).

Provides cross-block entanglement coordination, gradient teleportation,
and global phase synchronization for the HSMN architecture.

NO PYTHON FALLBACKS: RuntimeError raised if native ops unavailable.

Ops:
    Phase 76 (QCB):
        - qcb_initialize: Create GHZ-like entanglement mesh
        - qcb_coherent_transfer: State transfer between blocks
        - qcb_teleport_gradient: Aggregate gradients via entanglement
        - qcb_synchronize_phase: Global phase synchronization
        - qcb_update_mesh: Update mesh with new block states

    Phase 127 (Unified Quantum Bus):
        - unified_bus_propagate_entanglement: Propagate correlations
        - unified_bus_update_strength: Update entanglement matrix
"""

import logging

import tensorflow as tf

from highnoon import config
from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

_module = None
_available = False


def _load_ops():
    """Load ops from consolidated binary."""
    global _module, _available
    if _module is not None:
        return _available

    try:
        lib_path = resolve_op_library(__file__, "_highnoon_core.so")
        if lib_path is None:
            raise RuntimeError("Could not find _highnoon_core.so")
        _module = tf.load_op_library(lib_path)
        _available = True
        logger.info(f"Quantum Coherence Bus ops loaded from {lib_path}")
    except Exception as e:
        _available = False
        logger.warning(f"Failed to load Quantum Coherence Bus ops: {e}")
        raise RuntimeError(
            "Quantum Coherence Bus native ops not available. " "Run ./build_secure.sh to compile."
        ) from e
    return _available


def ops_available() -> bool:
    """Check if native ops are available."""
    try:
        _load_ops()
        return _available
    except RuntimeError:
        return False


# =============================================================================
# Phase 76: Quantum Coherence Bus (QCB)
# =============================================================================


def qcb_initialize(
    num_blocks: int | None = None,
    entanglement_dim: int = 64,
    bus_slots: int = 8,
    bidirectional: bool = True,
    coherence_threshold: float | None = None,
    seed: int = 42,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Initialize Quantum Coherence Bus with GHZ-like entanglement.

    Creates a maximally entangled state spanning all blocks in the
    HSMN architecture for cross-block coherent communication.

    Args:
        num_blocks: Number of reasoning blocks (default from config).
        entanglement_dim: Dimension of entangled states.
        bus_slots: Number of bus slots for parallel communication.
        bidirectional: If True, enable bidirectional communication.
        coherence_threshold: Minimum coherence threshold (default from config).
        seed: Random seed for initialization.

    Returns:
        Tuple of (entangled_state [num_blocks, entanglement_dim],
                  initial_fidelity scalar).

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    num_blocks = num_blocks or config.QCB_NUM_NODES
    coherence_threshold = coherence_threshold or config.QCB_FIDELITY_THRESHOLD

    return _module.qcb_initialize(
        num_blocks=num_blocks,
        entanglement_dim=entanglement_dim,
        bus_slots=bus_slots,
        bidirectional=bidirectional,
        coherence_threshold=coherence_threshold,
        seed=seed,
    )


def qcb_coherent_transfer(
    source_state: tf.Tensor,
    entangled_state: tf.Tensor,
    source_block: int,
    target_block: int,
    num_blocks: int | None = None,
    entanglement_dim: int = 64,
) -> tf.Tensor:
    """Coherent state transfer between blocks via QCB entanglement.

    Transfers quantum state from source block to target block using
    the shared entanglement mesh without destroying coherence.

    Args:
        source_state: State to transfer [batch, dim].
        entangled_state: Global entanglement mesh [num_blocks, entanglement_dim].
        source_block: Source block index.
        target_block: Target block index.
        num_blocks: Number of blocks (default from config).
        entanglement_dim: Entanglement dimension.

    Returns:
        Teleported state at target block [batch, dim].

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    num_blocks = num_blocks or config.QCB_NUM_NODES

    return _module.qcb_coherent_transfer(
        source_state,
        entangled_state,
        source_block=source_block,
        target_block=target_block,
        num_blocks=num_blocks,
        entanglement_dim=entanglement_dim,
    )


def qcb_teleport_gradient(
    block_gradients: tf.Tensor,
    entangled_state: tf.Tensor,
    num_blocks: int | None = None,
    entanglement_dim: int = 64,
) -> tf.Tensor:
    """Teleport and aggregate gradients from all blocks via QCB.

    Uses quantum entanglement to coherently aggregate gradients from
    all blocks, providing a gradient signal that preserves quantum
    correlations across the architecture.

    Args:
        block_gradients: Gradients from each block [num_blocks, num_params].
        entangled_state: Global entanglement mesh [num_blocks, entanglement_dim].
        num_blocks: Number of blocks (default from config).
        entanglement_dim: Entanglement dimension.

    Returns:
        Aggregated gradient for optimizer [num_params].

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    num_blocks = num_blocks or config.QCB_NUM_NODES

    return _module.qcb_teleport_gradient(
        block_gradients,
        entangled_state,
        num_blocks=num_blocks,
        entanglement_dim=entanglement_dim,
    )


def qcb_synchronize_phase(
    entangled_state: tf.Tensor,
    num_blocks: int | None = None,
    entanglement_dim: int = 64,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Synchronize quantum phase across all blocks in QCB.

    Performs global phase alignment to maintain coherent evolution
    across the distributed quantum state.

    Args:
        entangled_state: Input entanglement mesh [num_blocks, entanglement_dim].
        num_blocks: Number of blocks (default from config).
        entanglement_dim: Entanglement dimension.

    Returns:
        Tuple of (synchronized_state [num_blocks, entanglement_dim],
                  fidelity scalar).

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    num_blocks = num_blocks or config.QCB_NUM_NODES

    return _module.qcb_synchronize_phase(
        entangled_state,
        num_blocks=num_blocks,
        entanglement_dim=entanglement_dim,
    )


def qcb_update_mesh(
    entangled_state: tf.Tensor,
    block_states: tf.Tensor,
    num_blocks: int | None = None,
    entanglement_dim: int = 64,
    learning_rate: float = 0.01,
) -> tf.Tensor:
    """Update QCB mesh with new block states while preserving coherence.

    Incrementally updates the entanglement mesh based on new block
    outputs while maintaining global coherence properties.

    Args:
        entangled_state: Current mesh [num_blocks, entanglement_dim].
        block_states: New block states [num_blocks, state_dim].
        num_blocks: Number of blocks (default from config).
        entanglement_dim: Entanglement dimension.
        learning_rate: Update rate for mesh adaptation.

    Returns:
        Updated mesh [num_blocks, entanglement_dim].

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    num_blocks = num_blocks or config.QCB_NUM_NODES

    return _module.qcb_update_mesh(
        entangled_state,
        block_states,
        num_blocks=num_blocks,
        entanglement_dim=entanglement_dim,
        learning_rate=learning_rate,
    )


# =============================================================================
# Phase 127: Unified Quantum Entanglement Bus
# =============================================================================


def unified_bus_propagate_entanglement(
    block_states: tf.Tensor,
    entanglement_strength: tf.Tensor,
    num_blocks: int | None = None,
    bus_dim: int = 64,
    mps_bond_dim: int | None = None,
    coherence_threshold: float | None = None,
    propagation_rate: float = 0.1,
    use_adaptive: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Unified Quantum Bus - Propagate entanglement across blocks.

    Propagates quantum correlations across blocks with O(n·d) complexity
    using SIMD-optimized entanglement-weighted state mixing.

    Args:
        block_states: Input block states [batch, num_blocks, dim].
        entanglement_strength: Learnable entanglement matrix [num_blocks, num_blocks].
        num_blocks: Number of blocks (default from config).
        bus_dim: Bus channel dimension.
        mps_bond_dim: MPS bond dimension (default from config).
        coherence_threshold: Coherence threshold (default from config).
        propagation_rate: Rate of entanglement propagation.
        use_adaptive: If True, use adaptive propagation.

    Returns:
        Tuple of (entangled_states [batch, num_blocks, dim],
                  coherence [num_blocks, num_blocks]).

    Raises:
        RuntimeError: If native op not available.
    """
    if not config.USE_UNIFIED_QUANTUM_BUS:
        # Pass through if disabled
        nb = block_states.shape[1] or num_blocks or config.QCB_NUM_NODES
        coherence = tf.eye(nb, dtype=tf.float32)
        return block_states, coherence

    _load_ops()
    num_blocks = num_blocks or config.QCB_NUM_NODES
    mps_bond_dim = mps_bond_dim or config.UNIFIED_BUS_MPS_BOND_DIM
    coherence_threshold = coherence_threshold or config.UNIFIED_BUS_COHERENCE_THRESHOLD

    return _module.unified_quantum_bus_propagate_entanglement(
        block_states,
        entanglement_strength,
        num_blocks=num_blocks,
        bus_dim=bus_dim,
        mps_bond_dim=mps_bond_dim,
        coherence_threshold=coherence_threshold,
        propagation_rate=propagation_rate,
        use_adaptive=use_adaptive,
    )


def unified_bus_update_strength(
    entanglement_strength: tf.Tensor,
    coherence: tf.Tensor,
    num_blocks: int | None = None,
    coherence_threshold: float | None = None,
    propagation_rate: float = 0.1,
    use_adaptive: bool = True,
) -> tf.Tensor:
    """Update entanglement strength based on coherence feedback.

    Adaptively adjusts the entanglement matrix based on measured
    coherence to maintain optimal cross-block communication.

    Args:
        entanglement_strength: Current entanglement matrix [num_blocks, num_blocks].
        coherence: Measured coherence matrix [num_blocks, num_blocks].
        num_blocks: Number of blocks (default from config).
        coherence_threshold: Coherence threshold (default from config).
        propagation_rate: Update rate.
        use_adaptive: If True, use adaptive updating.

    Returns:
        Updated entanglement matrix [num_blocks, num_blocks].

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    num_blocks = num_blocks or config.QCB_NUM_NODES
    coherence_threshold = coherence_threshold or config.UNIFIED_BUS_COHERENCE_THRESHOLD

    return _module.unified_quantum_bus_update_strength(
        entanglement_strength,
        coherence,
        num_blocks=num_blocks,
        coherence_threshold=coherence_threshold,
        propagation_rate=propagation_rate,
        use_adaptive=use_adaptive,
    )


__all__ = [
    # Phase 76: QCB
    "qcb_initialize",
    "qcb_coherent_transfer",
    "qcb_teleport_gradient",
    "qcb_synchronize_phase",
    "qcb_update_mesh",
    # Phase 127: Unified Bus
    "unified_bus_propagate_entanglement",
    "unified_bus_update_strength",
    # Utility
    "ops_available",
]
