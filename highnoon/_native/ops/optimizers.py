# highnoon/_native/ops/optimizers.py
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

"""Python wrappers for native optimizer operations (Phases 46, 59, 60).

Provides access to high-performance C++ optimizer implementations:
- SophiaG: Second-order optimizer with Hessian diagonal approximation
- Lion: Memory-efficient sign-based optimizer
- Adiabatic (Phase 59): Quantum-inspired optimizer with tunneling
- Geodesic (Phase 60): Manifold-aware optimization with natural gradients
- SympFlow (Phase 46): Symplectic Hamiltonian dynamics optimizer

NO PYTHON FALLBACKS: RuntimeError raised if native ops unavailable.
"""

import logging
from typing import Tuple

import tensorflow as tf

from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

# Module-level state
_module = None
_available = False


def _load_ops():
    """Load optimizer ops from consolidated binary."""
    global _module, _available
    if _module is not None:
        return _available

    try:
        lib_path = resolve_op_library(__file__, "_highnoon_core.so")
        if lib_path is None:
            raise RuntimeError("Could not find _highnoon_core.so")
        _module = tf.load_op_library(lib_path)
        _available = True
        logger.info(f"Optimizer ops loaded from {lib_path}")
    except Exception as e:
        _available = False
        logger.warning(f"Failed to load optimizer ops: {e}")
        raise RuntimeError(
            "Native optimizer ops not available. Run ./build_secure.sh to compile."
        ) from e
    return _available


def ops_available() -> bool:
    """Check if native optimizer ops are available."""
    try:
        _load_ops()
        return _available
    except RuntimeError:
        return False


# Legacy compatibility aliases
_optimizers_module = None
_sophia_update_op = None
_lion_update_op = None

try:
    _load_ops()
    _optimizers_module = _module
    if hasattr(_module, "sophia_update"):
        _sophia_update_op = _module.sophia_update
    if hasattr(_module, "lion_update"):
        _lion_update_op = _module.lion_update
except RuntimeError:
    pass


def native_optimizers_available() -> bool:
    """Check if native optimizer operations are available."""
    return _optimizers_module is not None


def sophia_update_available() -> bool:
    """Check if native SophiaG update operation is available."""
    return _sophia_update_op is not None


def lion_update_available() -> bool:
    """Check if native Lion update operation is available."""
    return _lion_update_op is not None


def sophia_update(
    param: tf.Tensor,
    grad: tf.Tensor,
    exp_avg: tf.Tensor,
    hessian: tf.Tensor,
    learning_rate: float,
    beta1: float = 0.965,
    beta2: float = 0.99,
    rho: float = 0.04,
    weight_decay: float = 0.0,
) -> tf.Tensor:
    """Apply SophiaG optimizer update using native C++ operation.

    SophiaG is a second-order optimizer that uses Hessian diagonal
    approximation for adaptive learning rates.

    Args:
        param: Parameter tensor to update.
        grad: Gradient tensor.
        exp_avg: Exponential moving average of gradients.
        hessian: Hessian diagonal estimate.
        learning_rate: Learning rate.
        beta1: Momentum coefficient (default: 0.965).
        beta2: Hessian EMA coefficient (default: 0.99).
        rho: Clipping threshold (default: 0.04).
        weight_decay: Weight decay coefficient (default: 0.0).

    Returns:
        Updated parameter tensor.

    Raises:
        NotImplementedError: If native operation is not available.
    """
    if _sophia_update_op is None:
        raise NotImplementedError(
            "Native SophiaG update operation is unavailable. "
            "Use the Python SophiaG optimizer instead."
        )

    return _sophia_update_op(
        param,
        grad,
        exp_avg,
        hessian,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        rho=rho,
        weight_decay=weight_decay,
    )


def lion_update(
    param: tf.Tensor,
    grad: tf.Tensor,
    exp_avg: tf.Tensor,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
) -> tf.Tensor:
    """Apply Lion optimizer update using native C++ operation.

    Lion is a memory-efficient optimizer using only sign-based updates.

    Args:
        param: Parameter tensor to update.
        grad: Gradient tensor.
        exp_avg: Exponential moving average of gradients.
        learning_rate: Learning rate.
        beta1: Momentum coefficient for update (default: 0.9).
        beta2: Momentum coefficient for EMA (default: 0.99).
        weight_decay: Weight decay coefficient (default: 0.0).

    Returns:
        Updated parameter tensor.

    Raises:
        NotImplementedError: If native operation is not available.
    """
    if _lion_update_op is None:
        raise NotImplementedError(
            "Native Lion update operation is unavailable. " "Use the Python Lion optimizer instead."
        )

    return _lion_update_op(
        param,
        grad,
        exp_avg,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
    )


# =============================================================================
# Phase 59: Adiabatic Optimizer
# =============================================================================


def adiabatic_optimizer_step(
    params: tf.Tensor,
    gradients: tf.Tensor,
    velocity: tf.Tensor,
    schedule_s: float = 0.5,
    initial_temp: float = 10.0,
    final_temp: float = 0.01,
    seed: int = 42,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Quantum adiabatic optimizer step with tunneling.

    This optimizer uses quantum-inspired dynamics to escape local minima
    via simulated quantum tunneling. The temperature schedule controls
    the transition from quantum (high temp) to classical (low temp) regime.

    Args:
        params: Parameter tensor to update.
        gradients: Gradient tensor.
        velocity: Velocity (momentum) tensor.
        schedule_s: Annealing schedule parameter [0, 1] where 0=start, 1=end.
        initial_temp: Initial temperature (quantum regime).
        final_temp: Final temperature (classical regime).
        seed: Random seed for stochastic tunneling.

    Returns:
        Tuple of (updated_params, updated_velocity).

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    return _module.adiabatic_optimizer_step(
        params,
        gradients,
        velocity,
        schedule_s=schedule_s,
        initial_temp=initial_temp,
        final_temp=final_temp,
        seed=seed,
    )


def adiabatic_optimizer_available() -> bool:
    """Check if native adiabatic optimizer is available."""
    try:
        _load_ops()
        return hasattr(_module, "adiabatic_optimizer_step")
    except RuntimeError:
        return False


# =============================================================================
# Phase 60: Geodesic Optimizer
# =============================================================================


def geodesic_optimizer_step(
    params: tf.Tensor,
    gradients: tf.Tensor,
    velocity: tf.Tensor,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Geodesic optimizer step with natural gradients on parameter manifold.

    This optimizer follows geodesic paths on the parameter manifold,
    accounting for the local curvature via the quantum geometric tensor.
    This leads to more efficient optimization, especially for quantum layers.

    Args:
        params: Parameter tensor to update.
        gradients: Gradient tensor.
        velocity: Velocity (momentum) tensor.
        learning_rate: Learning rate.
        momentum: Geodesic momentum coefficient.

    Returns:
        Tuple of (updated_params, updated_velocity).

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    return _module.geodesic_optimizer_step(
        params,
        gradients,
        velocity,
        learning_rate=learning_rate,
        momentum=momentum,
    )


def geodesic_optimizer_available() -> bool:
    """Check if native geodesic optimizer is available."""
    try:
        _load_ops()
        return hasattr(_module, "geodesic_optimizer_step")
    except RuntimeError:
        return False


# =============================================================================
# Phase 46: SympFlow Optimizer
# =============================================================================


def sympflow_step(
    params: tf.Tensor,
    gradients: tf.Tensor,
    velocity: tf.Tensor,
    learning_rate: float = 0.001,
    mass: float = 1.0,
    friction: float = 0.1,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Symplectic flow optimizer step with Hamiltonian dynamics.

    SympFlow uses symplectic integration to preserve the Hamiltonian
    structure of the optimization landscape, leading to better
    long-term convergence properties.

    Args:
        params: Parameter tensor (position in phase space).
        gradients: Gradient tensor (force).
        velocity: Velocity tensor (momentum / mass).
        learning_rate: Step size for symplectic integration.
        mass: Effective mass for momentum dynamics.
        friction: Friction coefficient for energy dissipation.

    Returns:
        Tuple of (updated_params, updated_velocity).

    Raises:
        RuntimeError: If native op not available.
    """
    _load_ops()
    return _module.sympflow_step(
        params,
        gradients,
        velocity,
        learning_rate=learning_rate,
        mass=mass,
        friction=friction,
    )


def sympflow_kinetic_energy(
    velocity: tf.Tensor,
    mass: float = 1.0,
) -> tf.Tensor:
    """Compute kinetic energy for SympFlow optimizer.

    E_k = 0.5 * mass * ||velocity||^2

    Args:
        velocity: Velocity tensor.
        mass: Effective mass.

    Returns:
        Scalar kinetic energy.
    """
    _load_ops()
    return _module.sympflow_kinetic_energy(velocity, mass=mass)


def sympflow_available() -> bool:
    """Check if native SympFlow optimizer is available."""
    try:
        _load_ops()
        return hasattr(_module, "sympflow_step")
    except RuntimeError:
        return False


__all__ = [
    # Legacy
    "native_optimizers_available",
    "sophia_update_available",
    "lion_update_available",
    "sophia_update",
    "lion_update",
    # Phase 59: Adiabatic
    "adiabatic_optimizer_step",
    "adiabatic_optimizer_available",
    # Phase 60: Geodesic
    "geodesic_optimizer_step",
    "geodesic_optimizer_available",
    # Phase 46: SympFlow
    "sympflow_step",
    "sympflow_kinetic_energy",
    "sympflow_available",
    # General
    "ops_available",
]
