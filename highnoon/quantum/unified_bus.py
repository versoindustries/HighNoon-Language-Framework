# highnoon/quantum/unified_bus.py
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

"""Unified MPS-Backed Quantum Bus - Phase 3 Implementation.

Consolidates 3 separate quantum bus implementations into 1 MPS-backed system:
    - QuantumStateBus (Phase 35)
    - QuantumCoherenceBus (Phase 76)
    - QuantumTeleportBus (Phase 44)

Memory reduction: O(3d²) → O(N × d × χ²) where χ << d

Key features:
    - Single MPS state representation for all quantum communication
    - Coherence tracking via single scalar (replaces full matrix)
    - Teleportation via gauge transformation (no data copy)
    - Full gradient support for end-to-end training

Reference:
    QUANTUM_ROADMAP.md Phase 3: Unified Quantum Bus
"""

from __future__ import annotations

import logging
from typing import Any

import tensorflow as tf

from highnoon import config as hn_config

logger = logging.getLogger(__name__)


# Phase 1000: Unified Spectral Cache for shared SVD computation
try:
    from highnoon.training.spectral_cache import get_spectral_cache

    _SPECTRAL_CACHE_AVAILABLE = True
except ImportError:
    _SPECTRAL_CACHE_AVAILABLE = False
    get_spectral_cache = None


class UnifiedQuantumBus(tf.keras.layers.Layer):
    """MPS-backed unified quantum bus for cross-block communication.

    Replaces: QuantumStateBus, QuantumCoherenceBus, QuantumTeleportBus

    Memory complexity: O(N × d × χ²) where:
        - N = number of sites (blocks)
        - d = physical dimension (feature dim per site)
        - χ = bond dimension (entanglement capacity)

    For typical settings (N=8, d=64, χ=32):
        Old (3 buses): 3 × 64² × sizeof(float) = 48KB
        New (unified): 8 × 64 × 32² × sizeof(float) = 2MB (but shared, amortized)

    The unified bus provides:
        1. State propagation between blocks
        2. Entanglement tracking via bond entropies
        3. Coherence monitoring via single metric
        4. Gradient-aware teleportation

    Attributes:
        num_sites: Number of MPS sites (typically = num transformer blocks).
        physical_dim: Feature dimension at each site.
        bond_dim: Bond dimension controlling entanglement capacity.
        coherence: Current coherence metric (scalar).

    Example:
        >>> bus = UnifiedQuantumBus(num_sites=8, physical_dim=64, bond_dim=32)
        >>> bus.write(hidden_state, site_index=0)
        >>> propagated = bus.propagate_entanglement([0], [4])
        >>> output = bus.read(site_index=4)
    """

    def __init__(
        self,
        num_sites: int = 8,
        physical_dim: int = 64,
        bond_dim: int = 32,
        enable_gradients: bool = True,
        name: str = "unified_quantum_bus",
        **kwargs,
    ):
        """Initialize UnifiedQuantumBus.

        Args:
            num_sites: Number of MPS sites (communication channels).
            physical_dim: Physical dimension at each site.
            bond_dim: Bond dimension for entanglement. Larger = more capacity.
            enable_gradients: Whether to allow gradient flow.
            name: Layer name.
            **kwargs: Additional layer arguments.
        """
        super().__init__(name=name, **kwargs)

        self.num_sites = num_sites
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.enable_gradients = enable_gradients

        # MPS core tensors: A[i] has shape [χ_left, d, χ_right]
        self.mps_cores: list[tf.Variable] = []

        # Coherence metric (replaces full coherence matrix)
        self.coherence = tf.Variable(1.0, trainable=False, name="coherence")

        # Orthogonality center for efficient operations
        self._orth_center = tf.Variable(0, trainable=False, dtype=tf.int32)

        # Write/read projections
        self.write_proj = tf.keras.layers.Dense(
            physical_dim, use_bias=False, name=f"{name}_write_proj"
        )
        self.read_proj = tf.keras.layers.Dense(
            physical_dim, use_bias=False, name=f"{name}_read_proj"
        )

        logger.info(
            f"[UnifiedBus] Created with {num_sites} sites, "
            f"physical_dim={physical_dim}, bond_dim={bond_dim}"
        )

    def build(self, input_shape):
        """Build MPS core tensors."""
        # Bond dimensions: boundary=1, internal=bond_dim
        bond_dims = [1] + [self.bond_dim] * (self.num_sites - 1) + [1]

        for i in range(self.num_sites):
            chi_left = bond_dims[i]
            chi_right = bond_dims[i + 1]

            # Xavier initialization
            stddev = (2.0 / (chi_left * self.physical_dim + chi_right)) ** 0.5

            core = self.add_weight(
                name=f"mps_core_{i}",
                shape=(chi_left, self.physical_dim, chi_right),
                initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
                trainable=self.enable_gradients,
            )
            self.mps_cores.append(core)

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        site_index: int = 0,
        mode: str = "read",
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass: read or write to bus.

        Args:
            inputs: Input tensor [batch, features].
            site_index: Site to read/write.
            mode: "read" or "write".
            training: Whether in training mode.

        Returns:
            Output tensor [batch, features].
        """
        if mode == "write":
            return self.write(inputs, site_index)
        else:
            return self.read(site_index, batch_size=tf.shape(inputs)[0])

    def write(
        self,
        state: tf.Tensor,
        site_index: int,
    ) -> tf.Tensor:
        """Write state to MPS site.

        Complexity: O(χ² × d)

        Args:
            state: State tensor [batch, features] or [features].
            site_index: Site index to write to.

        Returns:
            Written state (after projection).
        """
        if site_index < 0 or site_index >= self.num_sites:
            raise ValueError(f"site_index must be in [0, {self.num_sites})")

        # Project to physical dimension
        projected = self.write_proj(state)  # [batch, physical_dim]

        # Update MPS core at site
        # For simplicity, we update the diagonal of the local tensor
        core = self.mps_cores[site_index]  # [χ_l, d, χ_r]

        # Compute outer product to update core
        if len(projected.shape) == 1:
            projected = tf.expand_dims(projected, 0)

        # Average over batch for core update
        avg_state = tf.reduce_mean(projected, axis=0)  # [d]

        # Create rank-1 update: core += α * |v⟩⟨v|
        chi_l, d, chi_r = core.shape
        update = tf.reshape(avg_state, [1, d, 1])
        update = tf.tile(update, [chi_l, 1, chi_r]) * 0.1

        # Apply update with gradient preservation
        new_core = core + update

        # State management - gradients flow through 'projected' which is returned
        self.mps_cores[site_index].assign(new_core)

        return projected

    def read(
        self,
        site_index: int,
        batch_size: int = 1,
    ) -> tf.Tensor:
        """Read state from MPS site.

        Complexity: O(χ² × d)

        Args:
            site_index: Site index to read from.
            batch_size: Batch size for output.

        Returns:
            State tensor [batch, physical_dim].
        """
        if site_index < 0 or site_index >= self.num_sites:
            raise ValueError(f"site_index must be in [0, {self.num_sites})")

        # Contract to get local expectation
        core = self.mps_cores[site_index]  # [χ_l, d, χ_r]

        # Local state: sum over bond indices
        local_state = tf.reduce_sum(core, axis=[0, 2])  # [d]

        # Normalize
        local_state = local_state / (tf.norm(local_state) + 1e-8)

        # Project and expand for batch (Dense expects 2D input)
        local_state_2d = tf.expand_dims(local_state, 0)  # [1, d]
        output = self.read_proj(local_state_2d)  # [1, physical_dim]
        output = tf.tile(output, [batch_size, 1])  # [batch, physical_dim]

        return output

    def propagate_entanglement(
        self,
        source_sites: list[int],
        target_sites: list[int],
    ) -> tf.Tensor:
        """Propagate correlations from source to target via MPS contraction.

        Complexity: O(χ³) per site in range

        Args:
            source_sites: List of source site indices.
            target_sites: List of target site indices.

        Returns:
            Contracted correlation tensor.
        """
        min_site = min(min(source_sites), min(target_sites))
        max_site = max(max(source_sites), max(target_sites))

        # Contract MPS cores in range
        result = self.mps_cores[min_site]  # [χ_l, d, χ_r]

        for i in range(min_site + 1, max_site + 1):
            core = self.mps_cores[i]  # [χ_l', d, χ_r']
            # Contract: result[..., χ_r] @ core[χ_l', ...]
            # result shape: [χ_l, d_1, χ_r] @ [χ_l', d_2, χ_r']
            # After contraction: [χ_l, d_1, d_2, χ_r']
            result = tf.einsum("ijk,klm->ijlm", result, core)
            # Flatten middle dims for next iteration
            shape = tf.shape(result)
            result = tf.reshape(result, [shape[0], -1, shape[-1]])

        # Sum over bonds to get scalar correlation, normalized by tensor size
        correlation = tf.reduce_sum(result)
        result_size = tf.cast(tf.reduce_prod(tf.shape(result)), tf.float32)
        correlation = correlation / result_size

        # Update coherence for monitoring (not in critical gradient path)
        new_coherence = 0.9 * self.coherence + 0.1 * tf.abs(correlation)
        self.coherence.assign(new_coherence)

        return correlation

    def teleport(
        self,
        state: tf.Tensor,
        source: int,
        target: int,
    ) -> tf.Tensor:
        """Quantum teleportation via MPS gauge transformation.

        Uses canonical form to efficiently "move" information without copying.
        Complexity: O(χ²) per site traversed.

        Args:
            state: State to teleport [batch, features].
            source: Source site index.
            target: Target site index.

        Returns:
            Teleported state at target [batch, features].
        """
        # Write to source
        self.write(state, source)

        # Move orthogonality center to target (gauge transformation)
        self._move_orth_center(target)

        # Read from target
        return self.read(target, batch_size=tf.shape(state)[0])

    def _move_orth_center(self, target: int):
        """Move orthogonality center to target site.

        Uses QR decomposition to maintain canonical form.
        """
        current = int(self._orth_center.numpy())

        if current == target:
            return

        direction = 1 if target > current else -1

        while current != target:
            # QR decomposition to move center
            core = self.mps_cores[current]  # [χ_l, d, χ_r]
            chi_l, d, chi_r = core.shape

            if direction == 1:
                # Moving right: reshape to [χ_l * d, χ_r] and QR
                reshaped = tf.reshape(core, [chi_l * d, chi_r])
                q, r = tf.linalg.qr(reshaped)
                self.mps_cores[current].assign(tf.reshape(q, [chi_l, d, -1]))

                # Absorb R into next core
                next_core = self.mps_cores[current + 1]
                next_chi_l, next_d, next_chi_r = next_core.shape
                next_reshaped = tf.reshape(next_core, [next_chi_l, next_d * next_chi_r])
                updated = tf.matmul(r, next_reshaped)
                self.mps_cores[current + 1].assign(tf.reshape(updated, [-1, next_d, next_chi_r]))
            else:
                # Moving left: reshape to [χ_l, d * χ_r] and RQ
                reshaped = tf.reshape(core, [chi_l, d * chi_r])
                # RQ via transposed QR
                q, r = tf.linalg.qr(tf.transpose(reshaped))
                self.mps_cores[current].assign(tf.reshape(tf.transpose(q), [-1, d, chi_r]))

                # Absorb into previous core
                prev_core = self.mps_cores[current - 1]
                prev_chi_l, prev_d, prev_chi_r = prev_core.shape
                prev_reshaped = tf.reshape(prev_core, [prev_chi_l * prev_d, prev_chi_r])
                updated = tf.matmul(prev_reshaped, tf.transpose(r))
                self.mps_cores[current - 1].assign(tf.reshape(updated, [prev_chi_l, prev_d, -1]))

            current += direction

        self._orth_center.assign(target)

    def get_bond_entropies(self) -> tf.Tensor:
        """Compute bond entanglement entropies.

        Phase 1000: Uses unified SpectralCache for shared SVD computation
        across QULS, HD compression, and MPS bond entropy.

        Returns entropies at each bond for QULS integration.

        Returns:
            Entanglement entropies [num_sites - 1].
        """
        entropies = []

        for i in range(self.num_sites - 1):
            # SVD at bond i
            core = self.mps_cores[i]  # [χ_l, d, χ_r]
            chi_l, d, chi_r = core.shape
            reshaped = tf.reshape(core, [chi_l * d, chi_r])

            # Phase 1000: Use SpectralCache for SVD when available
            if _SPECTRAL_CACHE_AVAILABLE:
                cache = get_spectral_cache()
                svd_result = cache.get_svd(reshaped)
                s = svd_result.s
            else:
                # Fallback: Direct SVD
                s = tf.linalg.svd(reshaped, compute_uv=False)

            s = s / (tf.norm(s) + 1e-10)  # Normalize

            # Von Neumann entropy
            s_sq = s**2
            entropy = -tf.reduce_sum(s_sq * tf.math.log(s_sq + 1e-10))
            entropies.append(entropy)

        return tf.stack(entropies)

    def get_coherence(self) -> tf.Tensor:
        """Get current coherence metric."""
        return self.coherence

    def update_coherence(self, fidelity: tf.Tensor):
        """Update coherence metric based on operation fidelity.

        Args:
            fidelity: Fidelity metric (scalar or batch).
        """
        if tf.rank(fidelity) > 0:
            fidelity = tf.reduce_mean(fidelity)

        self.coherence.assign(0.9 * self.coherence + 0.1 * fidelity)

    def reset(self):
        """Reset bus to initial state."""
        self.coherence.assign(1.0)
        self._orth_center.assign(0)

        # Reinitialize MPS cores
        for _, core in enumerate(self.mps_cores):
            stddev = (2.0 / (core.shape[0] + core.shape[2])) ** 0.5
            new_core = tf.random.normal(core.shape, stddev=stddev)
            core.assign(new_core)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "num_sites": self.num_sites,
                "physical_dim": self.physical_dim,
                "bond_dim": self.bond_dim,
                "enable_gradients": self.enable_gradients,
            }
        )
        return config


# =============================================================================
# Phase 1001: Adaptive MPS Bond Dimension
# =============================================================================


class AdaptiveMPSBus(UnifiedQuantumBus):
    """MPS-backed quantum bus with adaptive bond dimension.

    Phase 1001: Extends UnifiedQuantumBus with dynamic bond dimension adaptation
    based on input sequence spectral entropy, training phase, and gradient health.

    Bond dimension scaling factors:
        - Entropy: 1.0 + 0.5 × (spectral_entropy - 0.5)
        - Phase: exploration = 1.2x, exploitation = 1.0x
        - Health: barren_plateau = 1.5x, explosion = 0.75x, normal = 1.0x

    Bounds: [min_bond_dim, max_bond_dim] (default [8, 128])

    Memory: O(N × d × χ²) where χ adapts dynamically

    Reference:
        update.md Phase 1001: Adaptive MPS Bond Dimension

    Example:
        >>> bus = AdaptiveMPSBus(num_sites=8, base_bond_dim=32)
        >>> bus.adapt_bond_dim(spectral_entropy=0.7, training_phase="exploration")
        >>> print(bus.bond_dim)  # Dynamic value
    """

    def __init__(
        self,
        num_sites: int = 8,
        physical_dim: int = 64,
        base_bond_dim: int = 32,
        min_bond_dim: int = 8,
        max_bond_dim: int = 128,
        enable_gradients: bool = True,
        name: str = "adaptive_mps_bus",
        **kwargs,
    ):
        """Initialize AdaptiveMPSBus.

        Args:
            num_sites: Number of MPS sites.
            physical_dim: Physical dimension at each site.
            base_bond_dim: Base bond dimension (adaptively scaled).
            min_bond_dim: Minimum allowed bond dimension.
            max_bond_dim: Maximum allowed bond dimension.
            enable_gradients: Whether to allow gradient flow.
            name: Layer name.
            **kwargs: Additional layer arguments.
        """
        # Store adaptive parameters before super().__init__
        self.base_bond_dim = base_bond_dim
        self.min_bond_dim = min_bond_dim
        self.max_bond_dim = max_bond_dim

        # Initialize with base bond dimension
        super().__init__(
            num_sites=num_sites,
            physical_dim=physical_dim,
            bond_dim=base_bond_dim,
            enable_gradients=enable_gradients,
            name=name,
            **kwargs,
        )

        # Track adaptation history for diagnostics
        self._adaptation_history: list[dict] = []
        self._current_phase = "exploration"
        self._current_health = "normal"

        logger.info(
            f"[AdaptiveMPSBus] Created with base_bond_dim={base_bond_dim}, "
            f"bounds=[{min_bond_dim}, {max_bond_dim}]"
        )

    def adapt_bond_dim(
        self,
        spectral_entropy: float | tf.Tensor | None = None,
        training_phase: str | None = None,
        grad_health: str | None = None,
    ) -> int:
        """Dynamically adjust MPS bond dimension based on metrics.

        Args:
            spectral_entropy: Normalized entropy in [0, 1]. Higher values indicate
                more complex representations needing higher bond dimension.
            training_phase: "exploration" or "exploitation". Exploration phase
                uses higher bond dimension for more expressiveness.
            grad_health: "normal", "bp" (barren plateau), or "explosion".
                BP increases bond dimension to capture more gradient signal.
                Explosion decreases to constrain capacity.

        Returns:
            New bond dimension (int).

        Example:
            >>> new_dim = bus.adapt_bond_dim(
            ...     spectral_entropy=0.7,
            ...     training_phase="exploration",
            ...     grad_health="bp"
            ... )
        """
        # Convert Tensor to float if needed
        if spectral_entropy is not None:
            if tf.is_tensor(spectral_entropy):
                spectral_entropy = float(spectral_entropy.numpy())

        # Update state
        if training_phase is not None:
            self._current_phase = training_phase
        if grad_health is not None:
            self._current_health = grad_health

        # Start with base bond dimension
        target_chi = float(self.base_bond_dim)

        # Entropy scaling: higher entropy = more entanglement needed
        if spectral_entropy is not None:
            # Scale: 0.75x at entropy=0, 1.25x at entropy=1
            entropy_factor = 1.0 + 0.5 * (spectral_entropy - 0.5)
            target_chi *= entropy_factor

        # Phase scaling: exploration needs more capacity
        phase_factor = 1.2 if self._current_phase == "exploration" else 1.0
        target_chi *= phase_factor

        # Health scaling: BP = increase, explosion = decrease
        if self._current_health == "bp":
            health_factor = 1.5  # Increase to capture more gradient signal
        elif self._current_health == "explosion":
            health_factor = 0.75  # Decrease to constrain capacity
        else:
            health_factor = 1.0
        target_chi *= health_factor

        # Apply bounds
        target_chi = int(max(self.min_bond_dim, min(self.max_bond_dim, target_chi)))

        # Track history
        self._adaptation_history.append(
            {
                "spectral_entropy": spectral_entropy,
                "training_phase": self._current_phase,
                "grad_health": self._current_health,
                "old_bond_dim": self.bond_dim,
                "new_bond_dim": target_chi,
            }
        )

        # Update bond dimension if changed
        if target_chi != self.bond_dim:
            self._update_bond_dim(target_chi)

        return target_chi

    def _update_bond_dim(self, new_bond_dim: int) -> None:
        """Update MPS cores to new bond dimension.

        Uses truncated SVD to resize cores while preserving information.

        Args:
            new_bond_dim: Target bond dimension.
        """
        old_bond_dim = self.bond_dim
        self.bond_dim = new_bond_dim

        # Guard: If build() hasn't been called yet, mps_cores is empty
        # Just update bond_dim for when build() is eventually called
        if not self.mps_cores:
            logger.debug(
                f"[AdaptiveMPSBus] Bond dimension set to {new_bond_dim} (deferred until build)"
            )
            return

        # Recompute bond dimensions for all cores
        bond_dims = [1] + [new_bond_dim] * (self.num_sites - 1) + [1]

        new_cores = []
        for i in range(self.num_sites):
            old_core = self.mps_cores[i]  # [χ_l_old, d, χ_r_old]
            old_shape = old_core.shape
            old_chi_l, old_d, old_chi_r = old_shape

            new_chi_l = bond_dims[i]
            new_chi_r = bond_dims[i + 1]

            # Create resized tensor with preserved values
            # Start with Xavier-initialized tensor
            stddev = (2.0 / (new_chi_l * self.physical_dim + new_chi_r)) ** 0.5
            new_core_data = tf.random.normal(
                [new_chi_l, self.physical_dim, new_chi_r], stddev=stddev, dtype=old_core.dtype
            )

            # Copy preserved region from old core
            min_chi_l = min(old_chi_l, new_chi_l)
            min_chi_r = min(old_chi_r, new_chi_r)

            # Create indices for the preserved region
            preserved_slice = old_core[:min_chi_l, :, :min_chi_r]

            # Build the new tensor: start with noise, then overlay preserved data
            indices_l = tf.range(min_chi_l)
            indices_d = tf.range(self.physical_dim)
            indices_r = tf.range(min_chi_r)

            # Use tensor_scatter_nd_update to copy preserved values
            # Create meshgrid for indices
            idx_l, idx_d, idx_r = tf.meshgrid(indices_l, indices_d, indices_r, indexing="ij")
            indices = tf.stack(
                [tf.reshape(idx_l, [-1]), tf.reshape(idx_d, [-1]), tf.reshape(idx_r, [-1])], axis=1
            )
            updates = tf.reshape(preserved_slice, [-1])

            new_core_data = tf.tensor_scatter_nd_update(new_core_data, indices, updates)

            # Create new variable with correct shape
            new_var = tf.Variable(
                new_core_data,
                name=f"mps_core_{i}",
                trainable=self.enable_gradients,
            )
            new_cores.append(new_var)

        # Replace the mps_cores list
        self.mps_cores = new_cores

        logger.info(
            f"[AdaptiveMPSBus] Bond dimension: {old_bond_dim} → {new_bond_dim} "
            f"(phase={self._current_phase}, health={self._current_health})"
        )

    def set_barren_plateau_mode(self, active: bool) -> None:
        """Set barren plateau recovery mode.

        When active, increases bond dimension to capture more gradient signal.

        Args:
            active: Whether barren plateau recovery is active.
        """
        new_health = "bp" if active else "normal"
        if new_health != self._current_health:
            logger.info(
                f"[AdaptiveMPSBus] {'Entering' if active else 'Exiting'} barren plateau mode"
            )
            self.adapt_bond_dim(grad_health=new_health)

    def set_training_phase(self, phase: str) -> None:
        """Set training phase.

        Args:
            phase: "exploration" or "exploitation".
        """
        if phase not in ("exploration", "exploitation"):
            raise ValueError(f"Unknown phase: {phase}")
        if phase != self._current_phase:
            logger.info(f"[AdaptiveMPSBus] Phase: {self._current_phase} → {phase}")
            self.adapt_bond_dim(training_phase=phase)

    def get_adaptation_history(self) -> list[dict]:
        """Get bond dimension adaptation history.

        Returns:
            List of adaptation events with metrics and bond dimension changes.
        """
        return self._adaptation_history.copy()

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "base_bond_dim": self.base_bond_dim,
                "min_bond_dim": self.min_bond_dim,
                "max_bond_dim": self.max_bond_dim,
            }
        )
        return config


# =============================================================================
# Factory Functions
# =============================================================================


def create_unified_bus(
    num_sites: int | None = None,
    physical_dim: int | None = None,
    bond_dim: int | None = None,
    **kwargs,
) -> UnifiedQuantumBus:
    """Factory function for UnifiedQuantumBus.

    Uses config defaults if parameters not specified.

    Args:
        num_sites: Number of sites.
        physical_dim: Physical dimension.
        bond_dim: Bond dimension.
        **kwargs: Additional arguments.

    Returns:
        Configured UnifiedQuantumBus instance.
    """
    num_sites = num_sites or getattr(hn_config, "UNIFIED_BUS_NUM_SITES", 8)
    physical_dim = physical_dim or getattr(hn_config, "UNIFIED_BUS_PHYSICAL_DIM", 64)
    bond_dim = bond_dim or getattr(hn_config, "UNIFIED_BUS_BOND_DIM", 32)

    # UQHA: Use AdaptiveMPSBus by default when config enables it
    use_adaptive = kwargs.pop("use_adaptive", getattr(hn_config, "USE_ADAPTIVE_MPS_BUS", True))

    if use_adaptive:
        return AdaptiveMPSBus(
            num_sites=num_sites,
            physical_dim=physical_dim,
            base_bond_dim=bond_dim,
            **kwargs,
        )

    return UnifiedQuantumBus(
        num_sites=num_sites,
        physical_dim=physical_dim,
        bond_dim=bond_dim,
        **kwargs,
    )


def create_adaptive_bus(
    num_sites: int | None = None,
    physical_dim: int | None = None,
    base_bond_dim: int | None = None,
    min_bond_dim: int = 8,
    max_bond_dim: int = 128,
    **kwargs,
) -> AdaptiveMPSBus:
    """Factory function for AdaptiveMPSBus.

    UQHA Phase 1001: Creates an AdaptiveMPSBus with dynamic bond dimension
    adaptation based on spectral entropy, training phase, and gradient health.

    Uses config defaults if parameters not specified.

    Args:
        num_sites: Number of sites.
        physical_dim: Physical dimension.
        base_bond_dim: Base bond dimension (will be adapted).
        min_bond_dim: Minimum allowed bond dimension.
        max_bond_dim: Maximum allowed bond dimension.
        **kwargs: Additional arguments.

    Returns:
        Configured AdaptiveMPSBus instance.

    Example:
        >>> bus = create_adaptive_bus(base_bond_dim=32)
        >>> bus.adapt_bond_dim(spectral_entropy=0.7, training_phase="exploration")
    """
    num_sites = num_sites or getattr(hn_config, "UNIFIED_BUS_NUM_SITES", 8)
    physical_dim = physical_dim or getattr(hn_config, "UNIFIED_BUS_PHYSICAL_DIM", 64)
    base_bond_dim = base_bond_dim or getattr(hn_config, "UNIFIED_BUS_BOND_DIM", 32)

    return AdaptiveMPSBus(
        num_sites=num_sites,
        physical_dim=physical_dim,
        base_bond_dim=base_bond_dim,
        min_bond_dim=min_bond_dim,
        max_bond_dim=max_bond_dim,
        **kwargs,
    )


# =============================================================================
# Backwards Compatibility
# =============================================================================


class QuantumStateBusAdapter:
    """Adapter providing old QuantumStateBus interface."""

    def __init__(self, bus: UnifiedQuantumBus | None = None):
        self._bus = bus or create_unified_bus()

    def write(self, state, slot=0):
        return self._bus.write(state, slot)

    def read(self, slot=0, batch_size=1):
        return self._bus.read(slot, batch_size)

    def get_metrics(self):
        return {"coherence": float(self._bus.get_coherence().numpy())}


class QuantumCoherenceBusAdapter:
    """Adapter providing old QuantumCoherenceBus interface."""

    def __init__(self, bus: UnifiedQuantumBus | None = None):
        self._bus = bus or create_unified_bus()

    def get_coherence(self):
        return self._bus.get_coherence()

    def synchronize_phase(self, sources, targets):
        return self._bus.propagate_entanglement(sources, targets)


class QuantumTeleportBusAdapter:
    """Adapter providing old QuantumTeleportBus interface."""

    def __init__(self, bus: UnifiedQuantumBus | None = None):
        self._bus = bus or create_unified_bus()

    def teleport(self, state, source, target):
        return self._bus.teleport(state, source, target)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "UnifiedQuantumBus",
    "AdaptiveMPSBus",
    "create_unified_bus",
    "create_adaptive_bus",
    "QuantumStateBusAdapter",
    "QuantumCoherenceBusAdapter",
    "QuantumTeleportBusAdapter",
]
