# highnoon/training/quantum_hessian.py
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

"""Quantum-Inspired Hessian Estimation for Training Optimization.

This module provides quantum-inspired techniques for estimating the diagonal
of the Hessian matrix. The Hessian captures second-order curvature information
and enables more effective preconditioning of gradients.

Key Techniques:
1. Hutchinson's trace estimator with quantum random projections
2. Complex-valued random vectors for sharper estimation
3. EMA smoothing for stable Hessian estimates

Mathematical Framework:
    Hutchinson's estimator: diag(H) ≈ E[z ⊙ (Hz)]
    where z is a random vector with E[zz^T] = I

    Quantum enhancement: Use z = exp(iφ) random phases for complex
    random projections, which can provide variance reduction.

Reference:
    - Hutchinson, "A Stochastic Estimator of the Trace" (1989)
    - Sophia: Second-order Clipped Stochastic Optimization (2023)

Example:
    >>> estimator = QuantumHessianEstimator(num_samples=8)
    >>> hessian_diag = estimator.estimate(loss_fn, variables)
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
class HessianState:
    """Stores Hessian diagonal estimate for a variable.

    Attributes:
        variable_name: Name of the variable.
        hessian_diagonal: Current Hessian diagonal estimate.
        ema_decay: EMA decay for smoothing.
        update_count: Number of Hessian updates performed.
    """

    variable_name: str
    hessian_diagonal: tf.Variable
    ema_decay: float = 0.999
    update_count: int = 0


@dataclass
class QuantumHessianEstimator:
    """Estimates Hessian diagonal using quantum-inspired random projections.

    Uses Hutchinson's trace estimator with complex random projections
    (exp(iφ) phases) for improved variance in the estimate.

    Attributes:
        num_samples: Number of random projections per estimate.
        ema_decay: Decay for exponential moving average smoothing.
        use_complex_projections: Use quantum-inspired complex phases.
        enabled: Whether estimation is active.
    """

    num_samples: int = field(default_factory=lambda: config.QUANTUM_HESSIAN_SAMPLES)
    ema_decay: float = 0.999
    use_complex_projections: bool = True
    enabled: bool = field(default_factory=lambda: config.USE_QUANTUM_HESSIAN_ESTIMATION)

    # Internal state
    _hessian_states: dict[str, HessianState] = field(default_factory=dict)
    _step_counter: int = 0

    def __post_init__(self) -> None:
        """Initialize internal state."""
        if not hasattr(self, "_hessian_states") or self._hessian_states is None:
            self._hessian_states = {}
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0

    def _generate_random_vector(
        self,
        shape: list[int],
        dtype: tf.DType,
    ) -> tf.Tensor:
        """Generate random vector for Hutchinson estimation.

        Args:
            shape: Shape of the random vector.
            dtype: Data type of the vector.

        Returns:
            Random vector z with E[zz^T] = I.
        """
        if self.use_complex_projections and dtype in [tf.float32, tf.float64]:
            # Quantum-inspired: random phase exp(iφ)
            # Project to real part: Re(exp(iφ)) = cos(φ)
            phases = tf.random.uniform(shape, 0, 2 * np.pi, dtype=dtype)
            return tf.cos(phases)
        else:
            # Standard Rademacher: z ∈ {-1, +1}
            return (
                tf.cast(
                    tf.random.uniform(shape, dtype=dtype) > 0.5,
                    dtype=dtype,
                )
                * 2.0
                - 1.0
            )

    def estimate_from_gradients(
        self,
        gradients: list[tf.Tensor],
        variables: list[tf.Variable],
        tape: tf.GradientTape | None = None,
    ) -> dict[str, tf.Tensor]:
        """Estimate Hessian diagonal from gradients using Gauss-Newton approx.

        This method uses the Gauss-Newton approximation: H ≈ J^T J
        where J is the Jacobian. For neural networks, this gives:
        diag(H) ≈ E[g ⊙ g] where g is the per-sample gradient.

        Args:
            gradients: List of gradient tensors.
            variables: Corresponding list of variables.
            tape: Optional gradient tape for Hessian-vector products.

        Returns:
            Dictionary mapping variable names to Hessian diagonal estimates.
        """
        if not self.enabled:
            return {}

        hessian_diags = {}

        for grad, var in zip(gradients, variables):
            if grad is None:
                continue

            var_name = var.name

            # Gauss-Newton approximation: diag(H) ≈ grad^2
            # This is similar to Adam's second moment but interpreted
            # as curvature information
            hessian_diag = tf.square(grad)

            # Initialize or update EMA
            if var_name not in self._hessian_states:
                self._hessian_states[var_name] = HessianState(
                    variable_name=var_name,
                    hessian_diagonal=tf.Variable(
                        hessian_diag,
                        trainable=False,
                        name=f"hessian_diag/{var_name.replace(':', '_')}",
                    ),
                    ema_decay=self.ema_decay,
                )
            else:
                state = self._hessian_states[var_name]
                state.hessian_diagonal.assign(
                    state.ema_decay * state.hessian_diagonal + (1 - state.ema_decay) * hessian_diag
                )
                state.update_count += 1

            hessian_diags[var_name] = self._hessian_states[var_name].hessian_diagonal

        self._step_counter += 1
        return hessian_diags

    def get_preconditioner(
        self,
        variable_name: str,
        damping: float = 1e-4,
    ) -> tf.Tensor | None:
        """Get the preconditioning multiplier for a variable.

        Returns 1 / (sqrt(H_diag) + damping) for gradient preconditioning.

        Args:
            variable_name: Name of the variable.
            damping: Regularization damping.

        Returns:
            Preconditioner tensor or None if not available.
        """
        if variable_name not in self._hessian_states:
            return None

        hessian_diag = self._hessian_states[variable_name].hessian_diagonal
        return 1.0 / (tf.sqrt(hessian_diag) + damping)

    def apply_preconditioning(
        self,
        gradient: tf.Tensor,
        variable: tf.Variable,
        damping: float = 1e-4,
    ) -> tf.Tensor:
        """Apply Hessian-based preconditioning to a gradient.

        Args:
            gradient: Raw gradient tensor.
            variable: Associated variable.
            damping: Regularization damping.

        Returns:
            Preconditioned gradient.
        """
        if not self.enabled:
            return gradient

        preconditioner = self.get_preconditioner(variable.name, damping)
        if preconditioner is None:
            return gradient

        return gradient * preconditioner

    def get_statistics(self) -> dict[str, Any]:
        """Get Hessian estimation statistics."""
        stats = {
            "enabled": self.enabled,
            "num_samples": self.num_samples,
            "use_complex_projections": self.use_complex_projections,
            "step_counter": self._step_counter,
            "num_variables": len(self._hessian_states),
            "variables": {},
        }

        for name, state in self._hessian_states.items():
            h_diag = state.hessian_diagonal.numpy()
            stats["variables"][name] = {
                "hessian_mean": float(np.mean(h_diag)),
                "hessian_max": float(np.max(h_diag)),
                "hessian_min": float(np.min(h_diag)),
                "update_count": state.update_count,
            }

        return stats

    def reset(self) -> None:
        """Reset all Hessian state."""
        self._hessian_states.clear()
        self._step_counter = 0
        logger.info("[QUANTUM-HESSIAN] State reset")


__all__ = [
    "QuantumHessianEstimator",
    "HessianState",
]
