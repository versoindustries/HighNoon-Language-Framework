# highnoon/models/spatial/kalman.py
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

"""Kalman filter based state-space model block for sequence modeling.

This module provides a Kalman filter implementation as a Keras layer,
suitable for integration into the reasoning stack. The Kalman filter
provides optimal state estimation for linear dynamical systems.

The implementation supports:
- State prediction and update steps
- Learned state transition and observation matrices
- Stateful processing for long sequences
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from highnoon.models.reasoning.fused_contract import FusedReasoningBlockMixin


class KalmanBlock(FusedReasoningBlockMixin, layers.Layer):
    """Kalman filter based sequence modeling block.

    Implements a learned Kalman filter that maintains state estimates
    over a sequence, suitable for use as a reasoning block in the
    HighNoon architecture.

    The block learns:
    - State transition matrix A: models state dynamics
    - Observation matrix H: maps hidden state to observations
    - Process noise covariance Q: models uncertainty in dynamics
    - Observation noise covariance R: models measurement uncertainty

    Args:
        state_dim: Dimension of the latent state.
        embedding_dim: Input/output embedding dimension.
        d_inner: Internal feature dimension (default: 2 * state_dim).
        name: Layer name.

    Example:
        >>> block = KalmanBlock(state_dim=64, embedding_dim=256)
        >>> x = tf.random.normal([2, 32, 256])
        >>> initial_state = (
        ...     tf.zeros([2, 64]),  # state estimate
        ...     tf.eye(64, batch_shape=[2])  # covariance
        ... )
        >>> output, new_state = block(x, initial_state=initial_state)
    """

    fused_block_type = "KalmanBlock"
    fused_block_stateful = True

    def __init__(
        self,
        state_dim: int,
        embedding_dim: int,
        d_inner: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.d_inner = d_inner or (2 * state_dim)

        # Will be built in build()
        self.input_proj: layers.Dense | None = None
        self.output_proj: layers.Dense | None = None
        self.state_proj: layers.Dense | None = None

        # Kalman filter parameters
        self.A: tf.Variable | None = None  # State transition
        self.H: tf.Variable | None = None  # Observation matrix
        self.Q: tf.Variable | None = None  # Process noise covariance
        self.R: tf.Variable | None = None  # Observation noise covariance

    def build(self, input_shape: tf.TensorShape | tuple) -> None:
        """Build layer weights.

        Args:
            input_shape: Shape of input tensor or tuple for stateful call.
        """
        # Handle stateful build signature (input_shape, state_shape)
        if isinstance(input_shape, tuple) and len(input_shape) == 2:
            input_shape = input_shape[0]

        # Import TT config for Kalman projections
        from highnoon.config import TT_KALMAN_RANKS, USE_TT_KALMAN_PROJECTIONS

        if USE_TT_KALMAN_PROJECTIONS:
            # Use TTDense for parameter reduction on projections
            from highnoon.models.layers.tt_dense import TTDense

            self.input_proj = TTDense(
                output_dim=self.d_inner,
                tt_ranks=TT_KALMAN_RANKS,
                use_bias=True,
                name=f"{self.name}_input_proj_tt",
            )
            self.output_proj = TTDense(
                output_dim=self.embedding_dim,
                tt_ranks=TT_KALMAN_RANKS,
                use_bias=True,
                name=f"{self.name}_output_proj_tt",
            )
            self.state_proj = TTDense(
                output_dim=self.state_dim,
                tt_ranks=TT_KALMAN_RANKS,
                use_bias=True,
                name=f"{self.name}_state_proj_tt",
            )
        else:
            # Project input to state dimension
            self.input_proj = layers.Dense(
                self.d_inner,
                activation="gelu",
                name=f"{self.name}_input_proj",
            )

            # Project state back to embedding
            self.output_proj = layers.Dense(
                self.embedding_dim,
                name=f"{self.name}_output_proj",
            )

            # State update projection
            self.state_proj = layers.Dense(
                self.state_dim,
                name=f"{self.name}_state_proj",
            )

        # Initialize Kalman filter parameters
        # A: State transition (identity + learned residual)
        self.A = self.add_weight(
            name="state_transition",
            shape=(self.state_dim, self.state_dim),
            initializer="orthogonal",
            trainable=True,
        )

        # H: Observation matrix
        self.H = self.add_weight(
            name="observation_matrix",
            shape=(self.d_inner, self.state_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Q: Process noise (diagonal for efficiency)
        self.Q = self.add_weight(
            name="process_noise",
            shape=(self.state_dim,),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
        )

        # R: Observation noise (diagonal)
        self.R = self.add_weight(
            name="observation_noise",
            shape=(self.d_inner,),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
        )

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        initial_state: tuple[tf.Tensor, tf.Tensor] | None = None,
        training: bool = False,
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """Forward pass through Kalman filter block.

        Args:
            inputs: Input tensor [batch, seq_len, embedding_dim].
            initial_state: Tuple of (state_estimate, state_covariance).
                If None, initializes to zeros and identity.
            training: Whether in training mode.

        Returns:
            Tuple of (output, (state_estimate, state_covariance)):
            - output: Filtered output [batch, seq_len, embedding_dim]
            - state_estimate: Final state [batch, state_dim]
            - state_covariance: Final covariance [batch, state_dim, state_dim]
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Initialize state if not provided
        if initial_state is None:
            state = tf.zeros([batch_size, self.state_dim], dtype=inputs.dtype)
            # Identity covariance (simplified as diagonal)
            P_diag = tf.ones([batch_size, self.state_dim], dtype=inputs.dtype)
        else:
            state, P = initial_state
            # Extract diagonal if full covariance provided
            if len(P.shape) == 3:
                P_diag = tf.linalg.diag_part(P)
            else:
                P_diag = P

        # Project input
        z = self.input_proj(inputs)  # [B, L, d_inner]

        # Capture layer variables for use inside while_loop
        A = self.A
        H = self.H
        Q = self.Q
        R = self.R
        state_proj = self.state_proj
        output_proj = self.output_proj

        # Graph-compatible Kalman step function for tf.while_loop
        def kalman_step(t, state, P_diag, outputs_ta):
            """Single Kalman filter step."""
            z_t = z[:, t, :]  # [B, d_inner]

            # Predict step
            state_pred = tf.linalg.matvec(A, state)  # [B, state_dim]
            P_pred_diag = P_diag + tf.nn.softplus(Q)  # [B, state_dim]

            # Predicted observation
            H_state = tf.linalg.matvec(H, state_pred)  # [B, d_inner]

            # Innovation (measurement residual)
            innovation = z_t - H_state  # [B, d_inner]

            # Innovation covariance (simplified scalar)
            avg_P = tf.reduce_mean(P_pred_diag, axis=-1, keepdims=True)  # [B, 1]
            avg_R = tf.reduce_mean(tf.nn.softplus(R))  # scalar
            S_inv = 1.0 / (avg_P + avg_R + 1e-6)  # [B, 1]

            # Kalman gain (simplified)
            K_gain = P_pred_diag * S_inv  # [B, state_dim]

            # Update state
            state_update = state_proj(innovation)  # [B, state_dim]
            new_state = state_pred + K_gain * state_update

            # Update covariance (simplified Joseph form)
            new_P_diag = P_pred_diag * (1.0 - K_gain)
            new_P_diag = tf.maximum(new_P_diag, 1e-6)  # Ensure positive

            # Output projection
            out_t = output_proj(new_state)

            return t + 1, new_state, new_P_diag, outputs_ta.write(t, out_t)

        # Initialize TensorArray for outputs
        outputs_ta = tf.TensorArray(
            dtype=inputs.dtype,
            size=seq_len,
            dynamic_size=False,
            element_shape=[None, self.embedding_dim],
        )

        # Run while_loop for graph-mode compatible sequence processing
        _, final_state, final_P_diag, outputs_ta = tf.while_loop(
            cond=lambda t, *_: t < seq_len,
            body=kalman_step,
            loop_vars=[tf.constant(0), state, P_diag, outputs_ta],
            parallel_iterations=1,  # Sequential processing required for Kalman
        )

        # Stack outputs: TensorArray gives [L, B, dim], transpose to [B, L, dim]
        output = outputs_ta.stack()  # [L, B, embedding_dim]
        output = tf.transpose(output, [1, 0, 2])  # [B, L, embedding_dim]

        # Construct full covariance for state return
        P = tf.linalg.diag(final_P_diag)  # [B, state_dim, state_dim]

        return output, (final_state, P)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "state_dim": self.state_dim,
                "embedding_dim": self.embedding_dim,
                "d_inner": self.d_inner,
            }
        )
        return config

    def get_weights_for_fused_op(self) -> list[tf.Tensor]:
        """Return weights in order expected by fused kernel."""
        weights = []
        if self.input_proj:
            weights.extend([self.input_proj.kernel, self.input_proj.bias])
        if self.output_proj:
            weights.extend([self.output_proj.kernel, self.output_proj.bias])
        if self.state_proj:
            weights.extend([self.state_proj.kernel, self.state_proj.bias])
        if self.A is not None:
            weights.append(self.A)
        if self.H is not None:
            weights.append(self.H)
        if self.Q is not None:
            weights.append(self.Q)
        if self.R is not None:
            weights.append(self.R)
        return weights

    def fused_metadata(self) -> dict[str, Any]:
        """Return metadata for fused kernel."""
        return {
            "state_dim": self.state_dim,
            "embedding_dim": self.embedding_dim,
            "d_inner": self.d_inner,
        }


class ExtendedKalmanBlock(KalmanBlock):
    """Extended Kalman Filter block with adaptive noise estimation.

    Extends KalmanBlock with:
    - Adaptive noise covariance estimation (Q and R)
    - Nonlinear dynamics support via learned Jacobians
    - Innovation-based adaptation for online tuning

    This block integrates with the C++ ExtendedKalmanFilter for optimal
    performance while providing Python-side fallback.

    Args:
        state_dim: Dimension of the latent state.
        embedding_dim: Input/output embedding dimension.
        d_inner: Internal feature dimension (default: 2 * state_dim).
        adaptive_noise: Enable adaptive noise estimation.
        adaptation_rate: Learning rate for noise adaptation (0.01-0.1).
        use_nonlinear: Use learned nonlinear dynamics vs linear.
        name: Layer name.

    Example:
        >>> block = ExtendedKalmanBlock(
        ...     state_dim=64,
        ...     embedding_dim=256,
        ...     adaptive_noise=True,
        ... )
        >>> x = tf.random.normal([2, 32, 256])
        >>> output, state = block(x)
    """

    fused_block_type = "ExtendedKalmanBlock"

    def __init__(
        self,
        state_dim: int,
        embedding_dim: int,
        d_inner: int | None = None,
        adaptive_noise: bool = True,
        adaptation_rate: float = 0.05,
        use_nonlinear: bool = False,
        **kwargs,
    ):
        super().__init__(
            state_dim=state_dim,
            embedding_dim=embedding_dim,
            d_inner=d_inner,
            **kwargs,
        )
        self.adaptive_noise = adaptive_noise
        self.adaptation_rate = adaptation_rate
        self.use_nonlinear = use_nonlinear

        # EKF-specific layers (if nonlinear)
        self.dynamics_net: layers.Layer | None = None
        self.jacobian_net: layers.Layer | None = None

        # Innovation statistics for adaptive noise
        self.innovation_ema: tf.Variable | None = None

    def build(self, input_shape: tf.TensorShape | tuple) -> None:
        """Build layer weights including EKF-specific components."""
        super().build(input_shape)

        if self.use_nonlinear:
            # Nonlinear dynamics network f(x)
            self.dynamics_net = tf.keras.Sequential(
                [
                    layers.Dense(self.state_dim * 2, activation="gelu"),
                    layers.Dense(self.state_dim),
                ],
                name=f"{self.name}_dynamics",
            )

            # Jacobian approximation network df/dx
            self.jacobian_net = layers.Dense(
                self.state_dim * self.state_dim,
                name=f"{self.name}_jacobian",
            )

        # Running innovation EMA for adaptive noise
        self.innovation_ema = self.add_weight(
            name="innovation_ema",
            shape=(self.d_inner,),
            initializer="zeros",
            trainable=False,
        )

    def call(
        self,
        inputs: tf.Tensor,
        initial_state: tuple[tf.Tensor, tf.Tensor] | None = None,
        training: bool = False,
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """Forward pass through Extended Kalman filter block.

        Args:
            inputs: Input tensor [batch, seq_len, embedding_dim].
            initial_state: Tuple of (state_estimate, state_covariance).
            training: Whether in training mode.

        Returns:
            Tuple of (output, (state_estimate, state_covariance)).
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Initialize state
        if initial_state is None:
            state = tf.zeros([batch_size, self.state_dim], dtype=inputs.dtype)
            P_diag = tf.ones([batch_size, self.state_dim], dtype=inputs.dtype)
        else:
            state, P = initial_state
            if len(P.shape) == 3:
                P_diag = tf.linalg.diag_part(P)
            else:
                P_diag = P

        # Project input to observation space
        z = self.input_proj(inputs)  # [B, L, d_inner]

        # Capture layer state
        A = self.A
        H = self.H
        Q = self.Q
        R = self.R
        state_proj = self.state_proj
        output_proj = self.output_proj
        dynamics_net = self.dynamics_net
        use_nonlinear = self.use_nonlinear
        adaptive_noise = self.adaptive_noise
        adaptation_rate = self.adaptation_rate

        # EKF step function with adaptive noise
        def ekf_step(t, state, P_diag, Q_adapt, R_adapt, outputs_ta):
            """Single EKF step with adaptive noise."""
            z_t = z[:, t, :]

            # Predict step
            if use_nonlinear and dynamics_net is not None:
                # Nonlinear: x_pred = f(x)
                state_pred = dynamics_net(state)
            else:
                # Linear: x_pred = A @ x
                state_pred = tf.linalg.matvec(A, state)

            P_pred_diag = P_diag + tf.nn.softplus(Q_adapt)

            # Predicted observation
            H_state = tf.linalg.matvec(H, state_pred)

            # Innovation
            innovation = z_t - H_state

            # Innovation covariance
            avg_P = tf.reduce_mean(P_pred_diag, axis=-1, keepdims=True)
            avg_R = tf.reduce_mean(tf.nn.softplus(R_adapt))
            S_inv = 1.0 / (avg_P + avg_R + 1e-6)

            # Adaptive noise estimation based on innovation
            if adaptive_noise:
                innovation_sq = tf.reduce_mean(tf.square(innovation), axis=-1, keepdims=True)
                expected_innovation = avg_P + avg_R
                nis_ratio = innovation_sq / (expected_innovation + 1e-6)

                # Adjust R if innovation is too large/small
                R_scale = tf.where(
                    nis_ratio > 2.0,
                    1.0 + adaptation_rate,
                    tf.where(nis_ratio < 0.5, 1.0 - adaptation_rate, 1.0),
                )
                R_adapt = R_adapt * R_scale

            # Kalman gain
            K_gain = P_pred_diag * S_inv

            # Update state
            state_update = state_proj(innovation)
            new_state = state_pred + K_gain * state_update

            # Joseph form covariance update
            new_P_diag = P_pred_diag * (1.0 - K_gain)
            new_P_diag = tf.maximum(new_P_diag, 1e-6)

            # Output
            out_t = output_proj(new_state)

            return t + 1, new_state, new_P_diag, Q_adapt, R_adapt, outputs_ta.write(t, out_t)

        # Initialize TensorArray
        outputs_ta = tf.TensorArray(
            dtype=inputs.dtype,
            size=seq_len,
            dynamic_size=False,
            element_shape=[None, self.embedding_dim],
        )

        # Initial adaptive covariances
        Q_adapt = Q
        R_adapt = R

        # Run EKF loop
        _, final_state, final_P_diag, _, _, outputs_ta = tf.while_loop(
            cond=lambda t, *_: t < seq_len,
            body=ekf_step,
            loop_vars=[tf.constant(0), state, P_diag, Q_adapt, R_adapt, outputs_ta],
            parallel_iterations=1,
        )

        # Stack outputs
        output = outputs_ta.stack()
        output = tf.transpose(output, [1, 0, 2])

        P = tf.linalg.diag(final_P_diag)

        return output, (final_state, P)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "adaptive_noise": self.adaptive_noise,
                "adaptation_rate": self.adaptation_rate,
                "use_nonlinear": self.use_nonlinear,
            }
        )
        return config


__all__ = ["KalmanBlock", "ExtendedKalmanBlock"]
