# highnoon/learning/crystallized_optimizer.py
# Copyright 2025-2026 Verso Industries (Author: Michael B. Zimmerman)
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

"""Crystallized Optimizer for QHPM-Protected Weight Updates.

Wraps any base optimizer to project gradients orthogonal to crystallized
parameter directions, preventing catastrophic forgetting of protected knowledge.

The crystallization mechanism is based on QHPM (Quantum Holographic Persistent
Memory) from quantum_holographic_memory.h, which stores "crystallized" parameter
directions that should be protected from updates.

Mathematical foundation:
    For crystallized direction c (unit vector), gradient g is projected:
        g_protected = g - (g · c) * c

    For multiple directions C = [c1, ..., ck]:
        g_protected = g - C @ (C^T @ C)^(-1) @ C^T @ g
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from highnoon import config

logger = logging.getLogger(__name__)


# =============================================================================
# CRYSTALLIZATION STATE
# =============================================================================


@dataclass
class CrystallizationState:
    """State for gradient crystallization protection.

    Attributes:
        directions: Dictionary mapping variable names to crystallized directions.
        confidences: Confidence scores for each direction.
        decay_factor: Decay rate for crystallization strength.
        max_directions: Maximum number of protected directions per variable.
    """

    directions: dict[str, np.ndarray] = field(default_factory=dict)
    confidences: dict[str, np.ndarray] = field(default_factory=dict)
    decay_factor: float = 0.99
    max_directions: int = 256

    def add_direction(
        self,
        var_name: str,
        direction: np.ndarray,
        confidence: float = 1.0,
    ) -> None:
        """Add a crystallized direction for a variable.

        Args:
            var_name: Name of the variable.
            direction: Direction vector to protect.
            confidence: Initial confidence (0-1).
        """
        # Normalize direction
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        if var_name not in self.directions:
            self.directions[var_name] = direction.reshape(1, -1)
            self.confidences[var_name] = np.array([confidence])
        else:
            # Check if similar direction already exists
            existing = self.directions[var_name]
            similarities = np.abs(existing @ direction)

            if np.max(similarities) > 0.95:
                # Update existing instead of adding
                idx = np.argmax(similarities)
                self.confidences[var_name][idx] = max(self.confidences[var_name][idx], confidence)
            else:
                # Add new direction
                if len(existing) < self.max_directions:
                    self.directions[var_name] = np.vstack([existing, direction])
                    self.confidences[var_name] = np.append(self.confidences[var_name], confidence)
                else:
                    # Replace lowest confidence direction
                    min_idx = np.argmin(self.confidences[var_name])
                    if self.confidences[var_name][min_idx] < confidence:
                        self.directions[var_name][min_idx] = direction
                        self.confidences[var_name][min_idx] = confidence

    def decay(self) -> None:
        """Apply decay to all crystallization confidences."""
        for var_name in self.confidences:
            self.confidences[var_name] *= self.decay_factor
            # Remove directions with very low confidence
            mask = self.confidences[var_name] > 0.01
            if not np.all(mask):
                self.directions[var_name] = self.directions[var_name][mask]
                self.confidences[var_name] = self.confidences[var_name][mask]

    def save(self, path: Path) -> None:
        """Save crystallization state to disk.

        Args:
            path: Directory path for saving.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save directions
        np.savez(
            path / "crystallization_directions.npz",
            **dict(self.directions.items()),
        )

        # Save confidences
        np.savez(
            path / "crystallization_confidences.npz",
            **dict(self.confidences.items()),
        )

        logger.info(f"Saved crystallization state to {path}")

    @classmethod
    def load(cls, path: Path) -> CrystallizationState:
        """Load crystallization state from disk.

        Args:
            path: Directory path to load from.

        Returns:
            Loaded CrystallizationState.
        """
        path = Path(path)
        state = cls()

        directions_path = path / "crystallization_directions.npz"
        confidences_path = path / "crystallization_confidences.npz"

        if directions_path.exists():
            with np.load(directions_path) as data:
                state.directions = {k: data[k] for k in data.files}

        if confidences_path.exists():
            with np.load(confidences_path) as data:
                state.confidences = {k: data[k] for k in data.files}

        logger.info(f"Loaded crystallization state from {path}")
        return state


# =============================================================================
# CRYSTALLIZED OPTIMIZER
# =============================================================================


class CrystallizedOptimizer(tf.keras.optimizers.Optimizer):
    """Optimizer wrapper with QHPM crystallization protection.

    This optimizer wraps any base optimizer and projects gradients
    orthogonal to crystallized parameter directions before applying updates.
    This prevents catastrophic forgetting of protected knowledge.

    Attributes:
        base_optimizer: The wrapped optimizer.
        crystallization_state: State tracking protected directions.
        enabled: Whether crystallization is active.

    Example:
        >>> base_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        >>> opt = CrystallizedOptimizer(base_opt)
        >>> opt.crystallize_direction("dense/kernel", direction_vector)
        >>> opt.apply_gradients(grads_and_vars)  # Protected apply
    """

    def __init__(
        self,
        base_optimizer: tf.keras.optimizers.Optimizer,
        crystallization_state: CrystallizationState | None = None,
        enabled: bool = True,
        name: str = "CrystallizedOptimizer",
        **kwargs: Any,
    ) -> None:
        """Initialize CrystallizedOptimizer.

        Args:
            base_optimizer: The optimizer to wrap.
            crystallization_state: Existing state (creates new if None).
            enabled: Whether to enable crystallization protection.
            name: Optimizer name.
            **kwargs: Additional optimizer arguments.
        """
        super().__init__(name=name, **kwargs)
        self.base_optimizer = base_optimizer
        self.crystallization_state = crystallization_state or CrystallizationState(
            decay_factor=config.MIU_CRYSTALLIZATION_DECAY,
            max_directions=config.MIU_MAX_CRYSTAL_DIRECTIONS,
        )
        self.enabled = enabled
        self._step_count = 0

        logger.info(
            f"CrystallizedOptimizer wrapping {type(base_optimizer).__name__}, enabled={enabled}"
        )

    @property
    def learning_rate(self):
        """Get learning rate from base optimizer."""
        return self.base_optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        """Set learning rate on base optimizer."""
        self.base_optimizer.learning_rate = value

    def crystallize_direction(
        self,
        var_name: str,
        direction: np.ndarray,
        confidence: float = 1.0,
    ) -> None:
        """Add a crystallized direction for protection.

        Args:
            var_name: Name of the variable to protect.
            direction: Direction vector in parameter space.
            confidence: Confidence level (0-1).
        """
        self.crystallization_state.add_direction(var_name, direction, confidence)
        logger.debug(f"Crystallized direction for {var_name}, confidence={confidence}")

    def crystallize_from_embeddings(
        self,
        embeddings: list[np.ndarray],
        variable: tf.Variable,
    ) -> None:
        """Crystallize directions from HD embeddings.

        Projects embeddings into parameter space and protects those directions.

        Args:
            embeddings: List of HD embeddings to protect.
            variable: Variable to protect.
        """
        var_name = variable.name
        var_shape = variable.shape.as_list()

        for emb in embeddings:
            # Project embedding to variable shape
            # This is a simplification; real projection depends on architecture
            if len(var_shape) == 2:
                # Dense layer: project to rows or columns
                if var_shape[0] == len(emb):
                    direction = emb
                elif var_shape[1] == len(emb):
                    direction = emb
                else:
                    # Pad or truncate
                    flat_size = np.prod(var_shape)
                    direction = np.zeros(flat_size)
                    direction[: len(emb)] = emb[:flat_size]
            else:
                # Flatten approach
                flat_size = np.prod(var_shape)
                direction = np.zeros(flat_size)
                direction[: len(emb)] = emb[:flat_size]

            self.crystallize_direction(var_name, direction)

    def project_gradient(
        self,
        gradient: tf.Tensor,
        variable: tf.Variable,
    ) -> tf.Tensor:
        """Project gradient orthogonal to crystallized directions.

        Args:
            gradient: Original gradient tensor.
            variable: Variable the gradient is for.

        Returns:
            Projected gradient tensor.
        """
        if not self.enabled:
            return gradient

        var_name = variable.name
        if var_name not in self.crystallization_state.directions:
            return gradient

        # Get crystallized directions
        C = self.crystallization_state.directions[var_name]  # [k, dim]
        confidences = self.crystallization_state.confidences[var_name]  # [k]

        # Flatten gradient
        grad_flat = tf.reshape(gradient, [-1])

        # Ensure dimensions match
        if C.shape[1] != grad_flat.shape[0]:
            # Dimensions don't match, skip projection
            logger.warning(
                f"Dimension mismatch for {var_name}: "
                f"crystal={C.shape[1]}, grad={grad_flat.shape[0]}"
            )
            return gradient

        # Project: g_protected = g - C @ (C^T @ C)^(-1) @ C^T @ g
        # Weighted by confidences
        C_weighted = C * confidences[:, np.newaxis]

        C_tf = tf.constant(C_weighted, dtype=gradient.dtype)
        CtC = tf.matmul(C_tf, C_tf, transpose_b=True)
        CtC_inv = tf.linalg.pinv(CtC + 1e-6 * tf.eye(C_tf.shape[0]))
        Ctg = tf.matmul(C_tf, tf.expand_dims(grad_flat, 1))
        projection = tf.matmul(C_tf, tf.matmul(CtC_inv, Ctg), transpose_a=True)

        grad_projected = grad_flat - tf.squeeze(projection)

        return tf.reshape(grad_projected, gradient.shape)

    def apply_gradients(
        self,
        grads_and_vars,
        name: str | None = None,
        **kwargs: Any,
    ):
        """Apply gradients with crystallization protection.

        Args:
            grads_and_vars: List of (gradient, variable) tuples.
            name: Name scope for the operation.
            **kwargs: Additional arguments passed to base optimizer.

        Returns:
            Operation that applies gradients.
        """
        # Project gradients
        protected_grads_and_vars = []
        for grad, var in grads_and_vars:
            if grad is not None:
                protected_grad = self.project_gradient(grad, var)
                protected_grads_and_vars.append((protected_grad, var))
            else:
                protected_grads_and_vars.append((grad, var))

        # Apply via base optimizer
        result = self.base_optimizer.apply_gradients(protected_grads_and_vars, name=name, **kwargs)

        # Periodic decay
        self._step_count += 1
        if self._step_count % 100 == 0:
            self.crystallization_state.decay()

        return result

    def get_config(self) -> dict:
        """Get optimizer configuration."""
        config_dict = super().get_config()
        config_dict.update(
            {
                "base_optimizer": tf.keras.optimizers.serialize(self.base_optimizer),
                "enabled": self.enabled,
            }
        )
        return config_dict

    @classmethod
    def from_config(cls, config_dict: dict) -> CrystallizedOptimizer:
        """Create optimizer from configuration."""
        base_optimizer = tf.keras.optimizers.deserialize(config_dict.pop("base_optimizer"))
        return cls(base_optimizer=base_optimizer, **config_dict)


__all__ = ["CrystallizedOptimizer", "CrystallizationState"]
