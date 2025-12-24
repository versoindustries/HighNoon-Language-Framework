# highnoon/training/evolution_time_optimizer.py
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

"""Evolution Time QNG Optimizer for Hamiltonian Layers.

This module provides specialized optimization for Hamiltonian `evolution_time`
parameters using Riemannian gradient descent on the manifold of unitary
evolutions. This accounts for the non-Euclidean geometry of time parameters
in quantum dynamics.

Key Concepts:
- Evolution time t defines unitary U(t) = exp(-iHt) dynamics
- The parameter space is naturally a positive real line (t > 0)
- We use log-space optimization with adaptive curvature estimation

Mathematical Framework:
    For Hamiltonian H with evolution time t:
    U(t) = exp(-iHt)

    The Riemannian metric on R+ is: g(t) = 1/t²
    This leads to the natural gradient: ∇_R f = t² * ∇f

    Log-space equivalent: log(t) ← log(t) - η * ∂L/∂log(t)

Reference:
    - Port-Hamiltonian Neural Networks (2022)
    - Natural Gradient on Lie Groups (Bonnabel, 2013)

Example:
    >>> optimizer = EvolutionTimeOptimizer()
    >>> new_time = optimizer.update(evolution_time, gradient)
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
class EvolutionTimeState:
    """Stores optimizer state for an evolution time parameter.

    Attributes:
        variable_name: Name of the evolution time variable.
        log_time: Log of current evolution time (for log-space optimization).
        momentum: Momentum for the log-space update.
        curvature_estimate: Adaptive curvature estimate.
    """

    variable_name: str
    log_time: tf.Variable
    momentum: tf.Variable
    curvature_estimate: tf.Variable


@dataclass
class EvolutionTimeOptimizer:
    """Optimizes Hamiltonian evolution_time parameters using Riemannian gradients.

    Uses log-space optimization with momentum and adaptive curvature estimation
    to efficiently navigate the positive real line geometry of time parameters.

    Attributes:
        learning_rate: Base learning rate for log-space updates.
        momentum_decay: Momentum coefficient (β₁).
        curvature_decay: Curvature EMA coefficient (β₂).
        min_time: Minimum allowed evolution time.
        max_time: Maximum allowed evolution time.
        enabled: Whether optimizer is active.
    """

    learning_rate: float = 0.01
    momentum_decay: float = 0.9
    curvature_decay: float = 0.999
    min_time: float = 1e-6
    max_time: float = 1e3
    enabled: bool = field(default_factory=lambda: config.USE_EVOLUTION_TIME_QNG)

    # Internal state
    _states: dict[str, EvolutionTimeState] = field(default_factory=dict)
    _step_counter: int = 0

    def __post_init__(self) -> None:
        """Initialize internal state."""
        if not hasattr(self, "_states") or self._states is None:
            self._states = {}
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0

    def _is_evolution_time_var(self, variable_name: str) -> bool:
        """Check if variable is an evolution time parameter."""
        name_lower = variable_name.lower()
        patterns = ("evolution_time", "evo_time", "dt", "time_step")
        return any(p in name_lower for p in patterns)

    def _initialize_state(
        self,
        variable: tf.Variable,
    ) -> EvolutionTimeState:
        """Initialize optimizer state for an evolution time variable."""
        var_name = variable.name
        initial_time = tf.maximum(variable, self.min_time)

        state = EvolutionTimeState(
            variable_name=var_name,
            log_time=tf.Variable(
                tf.math.log(initial_time),
                trainable=False,
                name=f"evo_log_time/{var_name.replace(':', '_')}",
            ),
            momentum=tf.Variable(
                tf.zeros_like(variable),
                trainable=False,
                name=f"evo_momentum/{var_name.replace(':', '_')}",
            ),
            curvature_estimate=tf.Variable(
                tf.ones_like(variable) * 0.1,
                trainable=False,
                name=f"evo_curvature/{var_name.replace(':', '_')}",
            ),
        )

        logger.debug("[EVO-TIME] Initialized state for '%s'", var_name)
        return state

    def update(
        self,
        variable: tf.Variable,
        gradient: tf.Tensor,
    ) -> tf.Tensor:
        """Update evolution time using Riemannian gradient.

        Args:
            variable: Evolution time variable to update.
            gradient: Gradient ∂L/∂t.

        Returns:
            New evolution time value.
        """
        if not self.enabled:
            return variable

        var_name = variable.name

        if not self._is_evolution_time_var(var_name):
            return variable

        # Initialize state if needed
        if var_name not in self._states:
            self._states[var_name] = self._initialize_state(variable)

        state = self._states[var_name]

        # Convert gradient to log-space: ∂L/∂log(t) = t * ∂L/∂t
        current_time = tf.maximum(variable, self.min_time)
        log_gradient = current_time * gradient

        # Update momentum
        state.momentum.assign(
            self.momentum_decay * state.momentum + (1 - self.momentum_decay) * log_gradient
        )

        # Update curvature estimate (squared gradient EMA)
        state.curvature_estimate.assign(
            self.curvature_decay * state.curvature_estimate
            + (1 - self.curvature_decay) * tf.square(log_gradient)
        )

        # Adaptive step size based on curvature
        adaptive_lr = self.learning_rate / (tf.sqrt(state.curvature_estimate) + 1e-8)

        # Update log-time
        state.log_time.assign_sub(adaptive_lr * state.momentum)

        # Convert back to time-space with bounds
        new_time = tf.exp(state.log_time)
        new_time = tf.clip_by_value(new_time, self.min_time, self.max_time)

        # Update the original variable
        variable.assign(new_time)

        self._step_counter += 1
        return new_time

    def get_statistics(self) -> dict[str, Any]:
        """Get optimizer statistics."""
        stats = {
            "enabled": self.enabled,
            "learning_rate": self.learning_rate,
            "step_counter": self._step_counter,
            "num_variables": len(self._states),
            "variables": {},
        }

        for name, state in self._states.items():
            stats["variables"][name] = {
                "log_time": float(state.log_time.numpy().mean()),
                "current_time": float(np.exp(state.log_time.numpy()).mean()),
                "momentum_norm": float(np.linalg.norm(state.momentum.numpy().flatten())),
            }

        return stats

    def reset(self) -> None:
        """Reset all optimizer state."""
        self._states.clear()
        self._step_counter = 0
        logger.info("[EVO-TIME] Optimizer reset")


__all__ = [
    "EvolutionTimeOptimizer",
    "EvolutionTimeState",
]
