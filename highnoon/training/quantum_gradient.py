# highnoon/training/quantum_gradient.py
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

"""Quantum Natural Gradient (QNG) for Neural Network Training.

This module implements Quantum Natural Gradient descent, which uses the
Quantum Fisher Information Matrix (QFIM) to precondition gradients. This
accounts for the curved geometry of quantum state space and typically leads
to faster convergence compared to vanilla gradient descent.

Key concepts:
- The QFIM captures how sensitive quantum states are to parameter changes
- QNG preconditions gradients: θ_{t+1} = θ_t - η * F⁻¹(θ_t) * ∇L(θ_t)
- For efficiency, we use diagonal QFIM approximation

Mathematical Framework:
    The Quantum Fisher Information Matrix F_ij is defined as:
    F_ij = Re[⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩]

    For computational efficiency, we approximate with the diagonal:
    F_ii ≈ Var[∂_i log p(x|θ)]

    This is equivalent to the classical Fisher Information under
    certain conditions and provides similar optimization benefits.

Reference:
    - Stokes et al., "Quantum Natural Gradient" (2020)
    - Amari, "Natural Gradient Works Efficiently in Learning" (1998)

Example:
    >>> qng = QuantumNaturalGradient(damping=1e-4)
    >>> preconditioned_grad = qng.apply(gradient, variable)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf

from highnoon import config

logger = logging.getLogger(__name__)


@dataclass
class QFIMState:
    """Stores QFIM diagonal estimate for a variable.

    Attributes:
        variable_name: Name of the weight variable.
        shape: Shape of the weight tensor.
        qfim_diagonal: Diagonal approximation of QFIM.
        ema_decay: Decay factor for exponential moving average.
        last_update_step: Step at which QFIM was last updated.
    """

    variable_name: str
    shape: tuple[int, ...]
    qfim_diagonal: tf.Variable
    ema_decay: float = 0.99
    last_update_step: int = 0


@dataclass
class QuantumNaturalGradient:
    """Applies Quantum Natural Gradient preconditioning to gradients.

    Uses a diagonal approximation of the Quantum Fisher Information Matrix
    (QFIM) to precondition gradients. The QFIM is estimated via exponential
    moving average of squared gradients (similar to Adam's second moment).

    Attributes:
        damping: Regularization to prevent division by zero (ε in F⁻¹).
        ema_decay: Decay for QFIM exponential moving average.
        apply_to_quantum_only: Only apply QNG to quantum-enhanced layers.
        enabled: Whether QNG is active.
    """

    damping: float = field(default_factory=lambda: config.QNG_DAMPING)
    ema_decay: float = 0.99
    apply_to_quantum_only: bool = field(default_factory=lambda: config.QNG_APPLY_TO_QUANTUM_ONLY)
    enabled: bool = field(default_factory=lambda: config.USE_QUANTUM_NATURAL_GRADIENT)

    # Internal state
    _qfim_states: dict[str, QFIMState] = field(default_factory=dict)
    _step_counter: int = 0

    # Patterns for identifying quantum layers
    _quantum_patterns: tuple[str, ...] = (
        "quantum",
        "vqc",
        "hamiltonian",
        "timecrystal",
        "port_hamiltonian",
        "mps",
        "evolution",
        "qsvt",
        "holographic",
    )

    def __post_init__(self) -> None:
        """Initialize internal state after dataclass construction."""
        if not hasattr(self, "_qfim_states") or self._qfim_states is None:
            self._qfim_states = {}
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0

    def _is_quantum_layer(self, variable_name: str) -> bool:
        """Check if a variable belongs to a quantum-enhanced layer."""
        name_lower = variable_name.lower()
        return any(pattern in name_lower for pattern in self._quantum_patterns)

    def _should_apply_qng(self, variable_name: str) -> bool:
        """Determine if QNG should be applied to this variable."""
        if not self.enabled:
            return False
        if self.apply_to_quantum_only:
            return self._is_quantum_layer(variable_name)
        return True

    def _initialize_qfim_state(
        self,
        variable_name: str,
        gradient: tf.Tensor,
    ) -> QFIMState:
        """Initialize QFIM state for a new variable.

        Args:
            variable_name: Name of the weight variable.
            gradient: Gradient tensor to initialize from.

        Returns:
            Initialized QFIMState with QFIM diagonal.
        """
        shape = gradient.shape.as_list()

        # Initialize QFIM diagonal with squared gradient
        # This provides a reasonable starting estimate
        initial_qfim = tf.square(gradient) + self.damping

        qfim_var = tf.Variable(
            initial_qfim,
            trainable=False,
            name=f"qfim_diag/{variable_name.replace(':', '_')}",
        )

        state = QFIMState(
            variable_name=variable_name,
            shape=tuple(shape),
            qfim_diagonal=qfim_var,
            ema_decay=self.ema_decay,
            last_update_step=self._step_counter,
        )

        logger.debug(
            "[QNG] Initialized QFIM for '%s': shape=%s",
            variable_name,
            shape,
        )

        return state

    def apply(
        self,
        gradient: tf.Tensor,
        variable: tf.Variable,
    ) -> tf.Tensor:
        """Apply Quantum Natural Gradient preconditioning.

        Preconditions the gradient using the inverse of the QFIM diagonal
        approximation: g_precond = F⁻¹ * g

        Args:
            gradient: Raw gradient tensor.
            variable: Associated weight variable.

        Returns:
            Preconditioned gradient tensor.
        """
        var_name = variable.name

        if not self._should_apply_qng(var_name):
            return gradient

        # Initialize or retrieve QFIM state
        if var_name not in self._qfim_states:
            self._qfim_states[var_name] = self._initialize_qfim_state(var_name, gradient)

        state = self._qfim_states[var_name]

        # Update QFIM diagonal estimate via EMA of squared gradients
        # This approximates the diagonal of the Fisher Information Matrix
        grad_squared = tf.square(gradient)
        state.qfim_diagonal.assign(
            state.ema_decay * state.qfim_diagonal + (1 - state.ema_decay) * grad_squared
        )

        # Precondition gradient: g_precond = g / (sqrt(F_diag) + damping)
        # Using sqrt for numerical stability (similar to Adam)
        preconditioner = tf.sqrt(state.qfim_diagonal) + self.damping
        preconditioned_grad = gradient / preconditioner

        state.last_update_step = self._step_counter

        return preconditioned_grad

    def step(self) -> None:
        """Increment step counter. Call once per training step."""
        self._step_counter += 1

    def get_statistics(self) -> dict[str, Any]:
        """Get QNG statistics for all tracked variables."""
        stats = {
            "enabled": self.enabled,
            "damping": self.damping,
            "apply_to_quantum_only": self.apply_to_quantum_only,
            "step_counter": self._step_counter,
            "num_variables": len(self._qfim_states),
            "variables": {},
        }

        for name, state in self._qfim_states.items():
            qfim_diag = state.qfim_diagonal.numpy()
            stats["variables"][name] = {
                "shape": state.shape,
                "qfim_mean": float(np.mean(qfim_diag)),
                "qfim_max": float(np.max(qfim_diag)),
                "qfim_min": float(np.min(qfim_diag)),
                "last_update_step": state.last_update_step,
            }

        return stats

    def reset(self) -> None:
        """Reset all QNG state."""
        self._qfim_states.clear()
        self._step_counter = 0
        logger.info("[QNG] State reset")


class QNGOptimizerWrapper:
    """Wrapper that adds QNG preconditioning to any optimizer.

    Example:
        >>> base_optimizer = tf.keras.optimizers.Adam()
        >>> qng_optimizer = QNGOptimizerWrapper(base_optimizer)
        >>> qng_optimizer.apply_gradients(zip(grads, vars))
    """

    def __init__(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        damping: float = 1e-4,
        apply_to_quantum_only: bool = True,
    ):
        """Initialize QNG wrapper.

        Args:
            optimizer: Base optimizer to wrap.
            damping: QFIM regularization damping.
            apply_to_quantum_only: Only apply to quantum layers.
        """
        self.optimizer = optimizer
        self.qng = QuantumNaturalGradient(
            damping=damping,
            apply_to_quantum_only=apply_to_quantum_only,
            enabled=True,
        )

    @property
    def learning_rate(self):
        """Passthrough to base optimizer learning rate."""
        return self.optimizer.learning_rate

    def apply_gradients(
        self,
        grads_and_vars: list[tuple[tf.Tensor, tf.Variable]],
    ) -> None:
        """Apply QNG-preconditioned gradients.

        Args:
            grads_and_vars: List of (gradient, variable) pairs.
        """
        preconditioned_grads_and_vars = []

        for grad, var in grads_and_vars:
            if grad is None:
                preconditioned_grads_and_vars.append((None, var))
                continue

            precond_grad = self.qng.apply(grad, var)
            preconditioned_grads_and_vars.append((precond_grad, var))

        self.optimizer.apply_gradients(preconditioned_grads_and_vars)
        self.qng.step()


__all__ = [
    "QuantumNaturalGradient",
    "QNGOptimizerWrapper",
    "QFIMState",
]
