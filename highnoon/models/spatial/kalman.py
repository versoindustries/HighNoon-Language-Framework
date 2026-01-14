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

import logging
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from highnoon import config
from highnoon.models.reasoning.fused_contract import FusedReasoningBlockMixin

logger = logging.getLogger(__name__)

# Lazy-load C++ ops
_neural_kalman_ops = None


def _get_neural_kalman_ops():
    """Lazy-load NeuralKalman C++ ops."""
    global _neural_kalman_ops
    if _neural_kalman_ops is None:
        try:
            from highnoon._native import get_op

            _neural_kalman_ops = get_op("highnoon_core")
            if _neural_kalman_ops is not None and hasattr(_neural_kalman_ops, "NeuralKalmanStep"):
                logger.info("[KalmanBlock] C++ NeuralKalmanStep op loaded successfully.")
            else:
                _neural_kalman_ops = None
        except Exception as e:
            logger.debug(f"[KalmanBlock] C++ ops unavailable: {e}")
            _neural_kalman_ops = None
    return _neural_kalman_ops


def _neural_kalman_step_with_gradient(
    x_prior: tf.Tensor,
    z: tf.Tensor,
    gru_hidden: tf.Tensor,
    W_z: tf.Tensor,
    W_r: tf.Tensor,
    W_h: tf.Tensor,
    b_z: tf.Tensor,
    b_r: tf.Tensor,
    b_h: tf.Tensor,
    W_out: tf.Tensor,
    hidden_dim: int,
    state_dim: int,
    ops,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Wrap NeuralKalmanStep with custom gradient for training support.

    The C++ op does not have registered gradients, so we provide them here
    using the C++ NeuralKalmanStepBackward op.

    Args:
        x_prior: Prior state estimate [batch, state_dim].
        z: Measurement [batch, state_dim].
        gru_hidden: GRU hidden state [batch, hidden_dim].
        W_z, W_r, W_h: GRU weight matrices.
        b_z, b_r, b_h: GRU biases.
        W_out: Output projection [state_dim, hidden_dim].
        hidden_dim: GRU hidden dimension.
        state_dim: State dimension.
        ops: Loaded C++ ops module.

    Returns:
        Tuple of (x_posterior, gru_hidden_new).
    """
    if not hasattr(ops, "NeuralKalmanStepBackward"):
        raise RuntimeError(
            "NeuralKalmanStepBackward op is required for KalmanBlock training. "
            "Rebuild native ops to enable full C++ gradients."
        )

    @tf.custom_gradient
    def _forward(x_prior, z, gru_hidden, W_z, W_r, W_h, b_z, b_r, b_h, W_out):
        # Call C++ forward pass
        x_posterior, gru_hidden_new = ops.NeuralKalmanStep(
            x_prior=x_prior,
            z=z,
            gru_hidden=gru_hidden,
            w_z=W_z,
            w_r=W_r,
            w_h=W_h,
            b_z=b_z,
            b_r=b_r,
            b_h=b_h,
            w_out=W_out,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            propagate_covariance=config.NEURAL_KALMAN_PROPAGATE_COV,
            max_innovation=config.NEURAL_KALMAN_MAX_INNOVATION,
            epsilon=config.NEURAL_KALMAN_EPSILON,
            grad_clip_norm=config.NEURAL_KALMAN_GRAD_CLIP,
            enable_adaptive_scaling=config.NEURAL_KALMAN_ADAPTIVE_SCALE,
            enable_diagnostics=config.NEURAL_KALMAN_DEBUG,
        )

        # Compute K_gain for backward pass cache using C++ op
        k_gain = ops.LearnedKalmanGain(
            gru_hidden=gru_hidden_new,
            w_out=W_out,
        )

        def grad_fn(grad_x_posterior, grad_gru_hidden_new, variables=None):
            """
            Gradient computation using C++ NeuralKalmanStepBackward op.
            """
            # Check if backward op is available
            # Use C++ backward op for proper analytic gradients
            grads = ops.NeuralKalmanStepBackward(
                grad_x_posterior=grad_x_posterior,
                grad_gru_hidden_new=grad_gru_hidden_new,
                x_prior=x_prior,
                z=z,
                gru_hidden=gru_hidden,
                w_z=W_z,
                w_r=W_r,
                w_h=W_h,
                b_z=b_z,
                b_r=b_r,
                b_h=b_h,
                w_out=W_out,
                gru_hidden_new_saved=gru_hidden_new,
                k_gain_saved=k_gain,
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                max_innovation=config.NEURAL_KALMAN_MAX_INNOVATION,
                epsilon=config.NEURAL_KALMAN_EPSILON,
                grad_clip_norm=config.NEURAL_KALMAN_GRAD_CLIP,
                enable_adaptive_scaling=config.NEURAL_KALMAN_ADAPTIVE_SCALE,
                enable_diagnostics=config.NEURAL_KALMAN_DEBUG,
            )
            # Returns: (grad_x_prior, grad_z, grad_gru_hidden,
            #           grad_w_z, grad_w_r, grad_w_h,
            #           grad_b_z, grad_b_r, grad_b_h, grad_w_out)
            input_grads = (
                grads[0],  # grad_x_prior
                grads[1],  # grad_z
                grads[2],  # grad_gru_hidden
                grads[3],  # grad_W_z
                grads[4],  # grad_W_r
                grads[5],  # grad_W_h
                grads[6],  # grad_b_z
                grads[7],  # grad_b_r
                grads[8],  # grad_b_h
                grads[9],  # grad_W_out
            )

            if variables is None:
                return input_grads

            # Return separate input grads and variable grads
            variable_grads = [tf.zeros_like(v) for v in variables]
            return input_grads, variable_grads

        return (x_posterior, gru_hidden_new), grad_fn

    return _forward(x_prior, z, gru_hidden, W_z, W_r, W_h, b_z, b_r, b_h, W_out)


def _neural_kalman_step_full_with_gradient(
    x_prior: tf.Tensor,
    P_prior: tf.Tensor,
    z: tf.Tensor,
    A: tf.Tensor,
    H: tf.Tensor,
    Q: tf.Tensor,
    R: tf.Tensor,
    gru_hidden: tf.Tensor,
    W_z: tf.Tensor,
    W_r: tf.Tensor,
    W_h: tf.Tensor,
    b_z: tf.Tensor,
    b_r: tf.Tensor,
    b_h: tf.Tensor,
    W_out: tf.Tensor,
    hidden_dim: int,
    state_dim: int,
    obs_dim: int,
    ops,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Full Neural Kalman step with proper Kalman filter equations.

    Implements the complete Kalman filter with:
    1. State prediction: x_pred = A @ x_prior (orthogonal A keeps ||x|| bounded)
    2. Covariance prediction: P_pred = P_prior + softplus(Q)
    3. Innovation: y = z - H @ x_pred
    4. Kalman gain: K = P_pred / (P_pred + R) (bounded in [0, 1])
    5. State update: x_post = x_pred + K * (H^T @ innovation)
    6. Covariance update: P_post = P_pred * (1 - K) (stabilizes over time)

    Args:
        x_prior: Prior state estimate [batch, state_dim].
        P_prior: Prior covariance diagonal [batch, state_dim].
        z: Measurement [batch, obs_dim].
        A: State transition matrix [state_dim, state_dim] (orthogonal).
        H: Observation matrix [obs_dim, state_dim].
        Q: Process noise [state_dim].
        R: Measurement noise [obs_dim].
        gru_hidden: GRU hidden state [batch, hidden_dim].
        W_z, W_r, W_h: GRU weights (auxiliary, for adaptive noise).
        b_z, b_r, b_h: GRU biases.
        W_out: GRU output projection.
        hidden_dim: GRU hidden dimension.
        state_dim: Kalman state dimension.
        obs_dim: Observation dimension.
        ops: Loaded C++ ops module.

    Returns:
        Tuple of (x_posterior, P_posterior, gru_hidden_new).
    """
    # Phase 43.2: Require C++ NeuralKalmanStepFull op - no Python fallback
    if not hasattr(ops, "NeuralKalmanStepFull"):
        raise RuntimeError(
            "C++ NeuralKalmanStepFull op is required. "
            "Rebuild native ops: @build_secure.sh --lite --debug"
        )

    # Use C++ implementation with proper backward pass
    @tf.custom_gradient
    def _forward_full(
        x_prior, P_prior, z, A, H, Q, R, gru_hidden, W_z, W_r, W_h, b_z, b_r, b_h, W_out
    ):
        # Call C++ forward pass with full Kalman equations
        x_posterior, P_posterior, gru_hidden_new = ops.NeuralKalmanStepFull(
            x_prior=x_prior,
            p_prior=P_prior,
            z=z,
            state_trans=A,
            obs_mat=H,
            proc_noise=Q,
            meas_noise=R,
            gru_hidden=gru_hidden,
            w_z=W_z,
            w_r=W_r,
            w_h=W_h,
            b_z=b_z,
            b_r=b_r,
            b_h=b_h,
            w_out=W_out,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            obs_dim=obs_dim,
            use_full_kalman=True,
            p_min=config.NEURAL_KALMAN_EPSILON,
            p_max=10.0,
            k_max=1.0,
            max_innovation=config.NEURAL_KALMAN_MAX_INNOVATION,
            epsilon=config.NEURAL_KALMAN_EPSILON,
            grad_clip_norm=config.NEURAL_KALMAN_GRAD_CLIP,
            enable_adaptive_scaling=config.NEURAL_KALMAN_ADAPTIVE_SCALE,
            enable_diagnostics=config.NEURAL_KALMAN_DEBUG,
        )

        def grad_fn(grad_x_posterior, grad_P_posterior, grad_gru_hidden_new, variables=None):
            """Gradient using C++ NeuralKalmanStepFullBackward."""
            grads = ops.NeuralKalmanStepFullBackward(
                grad_x_posterior=grad_x_posterior,
                grad_p_posterior=grad_P_posterior,
                grad_gru_hidden_new=grad_gru_hidden_new,
                x_prior=x_prior,
                p_prior=P_prior,
                z=z,
                state_trans=A,
                obs_mat=H,
                proc_noise=Q,
                meas_noise=R,
                gru_hidden=gru_hidden,
                w_z=W_z,
                w_r=W_r,
                w_h=W_h,
                b_z=b_z,
                b_r=b_r,
                b_h=b_h,
                w_out=W_out,
                x_posterior=x_posterior,
                p_posterior=P_posterior,
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                obs_dim=obs_dim,
            )
            # Returns: (grad_x_prior, grad_P_prior, grad_z, grad_A, grad_H,
            #           grad_Q, grad_R, grad_gru_hidden, grad_w_z, ...)
            input_grads = tuple(grads[:15])

            if variables is None:
                return input_grads
            variable_grads = [tf.zeros_like(v) for v in variables]
            return input_grads, variable_grads

        return (x_posterior, P_posterior, gru_hidden_new), grad_fn

    return _forward_full(
        x_prior, P_prior, z, A, H, Q, R, gru_hidden, W_z, W_r, W_h, b_z, b_r, b_h, W_out
    )


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

    UQHA Integration:
    - energy_drift from HDTimeCrystal modulates process noise Q
    - High drift → increased Q → Kalman filter trusts predictions less
    - This provides adaptive uncertainty tracking for HNN dynamics

    Args:
        state_dim: Dimension of the latent state.
        embedding_dim: Input/output embedding dimension.
        d_inner: Internal feature dimension (default: 2 * state_dim).
        energy_drift_gain: Scaling factor for energy drift → Q modulation.
        name: Layer name.

    Example:
        >>> block = KalmanBlock(state_dim=64, embedding_dim=256)
        >>> x = tf.random.normal([2, 32, 256])
        >>> initial_state = (
        ...     tf.zeros([2, 64]),  # state estimate
        ...     tf.eye(64, batch_shape=[2])  # covariance
        ... )
        >>> output, new_state = block(x, initial_state=initial_state)
        >>> # Wire energy drift from HDTimeCrystal
        >>> block.set_energy_drift(0.005)  # High drift = more uncertainty
    """

    fused_block_type = "KalmanBlock"
    fused_block_stateful = True

    def __init__(
        self,
        state_dim: int,
        embedding_dim: int,
        d_inner: int | None = None,
        energy_drift_gain: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.d_inner = d_inner or (2 * state_dim)
        self.energy_drift_gain = energy_drift_gain

        # UQHA: Energy drift modulation for process noise
        self._energy_drift: float | None = None

        # Will be built in build()
        self.input_proj: layers.Dense | None = None
        self.output_proj: layers.Dense | None = None
        self.state_proj: layers.Dense | None = None

        # Kalman filter parameters
        self.A: tf.Variable | None = None  # State transition
        self.H: tf.Variable | None = None  # Observation matrix
        self.Q: tf.Variable | None = None  # Process noise covariance
        self.R: tf.Variable | None = None  # Observation noise covariance

        # GRU weights for Neural Kalman (C++ op)
        self.gru_hidden_dim = getattr(config, "NEURAL_KALMAN_HIDDEN_DIM", 128)
        self._prenorm = None
        self.W_z: tf.Variable | None = None  # GRU update gate weights
        self.W_r: tf.Variable | None = None  # GRU reset gate weights
        self.W_h: tf.Variable | None = None  # GRU candidate weights
        self.b_z: tf.Variable | None = None  # GRU update gate bias
        self.b_r: tf.Variable | None = None  # GRU reset gate bias
        self.b_h: tf.Variable | None = None  # GRU candidate bias
        self.W_out: tf.Variable | None = None  # Output projection to Kalman gain

    def build(self, input_shape: tf.TensorShape | tuple) -> None:
        """Build layer weights.

        Args:
            input_shape: Shape of input tensor or tuple for stateful call.
        """
        # Guard for idempotent build - prevent sublayer recreation
        if self.built:
            return

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

        # GRU weights for Neural Kalman C++ op
        # NOTE: GRU is now AUXILIARY after Phase 43.2 refactor.
        # Kalman gain is computed via K = P_pred / (P_pred + R), not GRU.
        # GRU is only used for adaptive noise estimation - set trainable=False.
        if getattr(config, "USE_NATIVE_NEURAL_KALMAN", True):
            gru_input_dim = self.state_dim  # Innovation dimension
            gru_concat_dim = self.gru_hidden_dim + gru_input_dim

            self.W_z = self.add_weight(
                name="gru_w_z",
                shape=(self.gru_hidden_dim, gru_concat_dim),
                initializer="glorot_uniform",
                trainable=False,
            )
            self.W_r = self.add_weight(
                name="gru_w_r",
                shape=(self.gru_hidden_dim, gru_concat_dim),
                initializer="glorot_uniform",
                trainable=False,
            )
            self.W_h = self.add_weight(
                name="gru_w_h",
                shape=(self.gru_hidden_dim, gru_concat_dim),
                initializer="glorot_uniform",
                trainable=False,
            )
            self.b_z = self.add_weight(
                name="gru_b_z",
                shape=(self.gru_hidden_dim,),
                initializer="zeros",
                trainable=False,
            )
            self.b_r = self.add_weight(
                name="gru_b_r",
                shape=(self.gru_hidden_dim,),
                initializer="zeros",
                trainable=False,
            )
            self.b_h = self.add_weight(
                name="gru_b_h",
                shape=(self.gru_hidden_dim,),
                initializer="zeros",
                trainable=False,
            )
            self.W_out = self.add_weight(
                name="gru_w_out",
                shape=(self.state_dim, self.gru_hidden_dim),
                initializer="glorot_uniform",
                trainable=False,
            )

        if config.NEURAL_KALMAN_PRENORM:
            # GRADIENT FIX: Increased epsilon from 1e-5 to 1e-4 to prevent inf gradients
            # when state_dim=16 features have very low variance
            self._prenorm = layers.LayerNormalization(
                axis=-1,
                epsilon=1e-4,
                name=f"{self.name}_prenorm",
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
        # Require C++ NeuralKalmanStep op - no Python fallback
        if self.W_z is None:
            raise RuntimeError(
                "KalmanBlock requires C++ NeuralKalmanStep op. "
                "Set USE_NATIVE_NEURAL_KALMAN=True and rebuild native library."
            )

        ops = _get_neural_kalman_ops()
        if ops is None:
            raise RuntimeError(
                "C++ NeuralKalmanStep op not available. "
                "Build the native library: cd highnoon/_native/build && cmake .. && make -j$(nproc)"
            )

        return self._call_native(inputs, initial_state, training)

    def _call_native(
        self,
        inputs: tf.Tensor,
        initial_state: tuple[tf.Tensor, tf.Tensor] | None = None,
        training: bool = False,
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """Forward pass using full Kalman filter with proper stability guarantees.

        Phase 43.2: Uses proper Kalman filter equations with:
        - Orthogonal A matrix for bounded state prediction
        - Covariance tracking for bounded Kalman gain (K ∈ [0, 1])
        - P update that stabilizes gains over time

        Args:
            inputs: Input tensor [batch, seq_len, embedding_dim].
            initial_state: Tuple of (state_estimate, state_covariance).
            training: Whether in training mode.

        Returns:
            Tuple of (output, (state_estimate, state_covariance)).
        """
        ops = _get_neural_kalman_ops()
        if ops is None or self.W_z is None:
            raise RuntimeError(
                "KalmanBlock requires C++ NeuralKalmanStep op. Python fallback is disabled."
            )

        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Initialize state and covariance diagonal
        if initial_state is None:
            state = tf.zeros([batch_size, self.state_dim], dtype=inputs.dtype)
            P_diag = (
                tf.ones([batch_size, self.state_dim], dtype=inputs.dtype)
                * config.NEURAL_KALMAN_INITIAL_P
            )
            gru_hidden = tf.zeros([batch_size, self.gru_hidden_dim], dtype=inputs.dtype)
        else:
            state, P = initial_state
            # Extract diagonal if full covariance provided
            if len(P.shape) == 3:
                P_diag = tf.linalg.diag_part(P)
            else:
                P_diag = P
            gru_hidden = tf.zeros([batch_size, self.gru_hidden_dim], dtype=inputs.dtype)

        # Project input to observation space
        z = self.input_proj(inputs)  # [B, L, d_inner]
        if self._prenorm is not None:
            z = self._prenorm(z)

        # Sequential processing using full Kalman filter
        outputs_ta = tf.TensorArray(
            dtype=inputs.dtype,
            size=seq_len,
            dynamic_size=False,
            element_shape=[None, self.embedding_dim],
        )

        # Capture weights for while_loop closure
        A = self.A  # Orthogonal state transition [state_dim, state_dim]
        H = self.H  # Observation matrix [d_inner, state_dim]
        Q = self.get_effective_Q()  # UQHA: Process noise with energy_drift modulation
        R = self.R  # Measurement noise [d_inner]
        W_z = self.W_z
        W_r = self.W_r
        W_h = self.W_h
        b_z = self.b_z
        b_r = self.b_r
        b_h = self.b_h
        W_out = self.W_out
        output_proj = self.output_proj
        gru_hidden_dim = self.gru_hidden_dim
        state_dim = self.state_dim
        d_inner = self.d_inner

        def full_kalman_step(t, state, P_diag, gru_hidden, outputs_ta):
            """Single full Kalman step with proper stability guarantees."""
            z_t = z[:, t, :]  # [B, d_inner] - observation at time t

            # Call full Kalman step with A, H, Q, R, P
            new_state, new_P_diag, new_gru_hidden = _neural_kalman_step_full_with_gradient(
                x_prior=state,
                P_prior=P_diag,
                z=z_t,
                A=A,
                H=H,
                Q=Q,
                R=R,
                gru_hidden=gru_hidden,
                W_z=W_z,
                W_r=W_r,
                W_h=W_h,
                b_z=b_z,
                b_r=b_r,
                b_h=b_h,
                W_out=W_out,
                hidden_dim=gru_hidden_dim,
                state_dim=state_dim,
                obs_dim=d_inner,
                ops=ops,
            )

            # Output projection
            out_t = output_proj(new_state)

            return t + 1, new_state, new_P_diag, new_gru_hidden, outputs_ta.write(t, out_t)

        # Run while_loop for sequential processing with covariance tracking
        _, final_state, final_P_diag, _, outputs_ta = tf.while_loop(
            cond=lambda t, *_: t < seq_len,
            body=full_kalman_step,
            loop_vars=[tf.constant(0), state, P_diag, gru_hidden, outputs_ta],
            parallel_iterations=1,
        )

        # Stack outputs
        output = outputs_ta.stack()  # [L, B, embedding_dim]
        output = tf.transpose(output, [1, 0, 2])  # [B, L, embedding_dim]

        # Return final covariance as diagonal matrix
        P = tf.linalg.diag(final_P_diag)

        return output, (final_state, P)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "state_dim": self.state_dim,
                "embedding_dim": self.embedding_dim,
                "d_inner": self.d_inner,
                "energy_drift_gain": self.energy_drift_gain,
            }
        )
        return config

    def set_energy_drift(self, drift: float) -> None:
        """Set energy drift from HDTimeCrystal for Q modulation.

        UQHA Integration: This method is called by the training engine
        or block_factory to pass energy_drift metrics from TimeCrystal
        blocks into the Kalman filter. High drift increases process noise Q,
        causing the filter to trust predictions less.

        Args:
            drift: Energy drift value |H_final - H_initial| from TimeCrystal.
        """
        self._energy_drift = float(drift)
        logger.debug(f"[KalmanBlock] Set energy_drift={drift:.6f} for Q modulation")

    def get_energy_drift(self) -> float | None:
        """Get current energy drift value."""
        return self._energy_drift

    def get_effective_Q(self) -> tf.Tensor:
        """Get effective process noise with energy drift modulation.

        Returns:
            Q modulated by energy drift: Q_eff = Q * (1 + gain * drift)
        """
        if self.Q is None:
            raise ValueError("KalmanBlock not built yet")

        if self._energy_drift is None or self._energy_drift == 0.0:
            return tf.nn.softplus(self.Q)

        # Scale Q by energy drift: higher drift = more process noise
        # Q_eff = Q * (1 + energy_drift_gain * drift)
        drift_scale = 1.0 + self.energy_drift_gain * self._energy_drift
        return tf.nn.softplus(self.Q) * tf.cast(drift_scale, self.Q.dtype)

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


# UQHA Phase 1001: TT-compressed Kalman filter for high-dimensional states
# UQHA Priority 3: Symplectic Kalman (Port-Hamiltonian filter dynamics)
from highnoon.models.spatial.symplectic_kalman import (
    SymplecticGNNKalmanLayer,
    SymplecticKalmanBlock,
    symplectic_gnn_kalman,
)
from highnoon.models.spatial.tensor_network_kalman import (
    TensorNetworkKalmanBlock,
    TensorNetworkKalmanFilter,
)

__all__ = [
    "KalmanBlock",
    "ExtendedKalmanBlock",
    "TensorNetworkKalmanFilter",
    "TensorNetworkKalmanBlock",
    "symplectic_gnn_kalman",
    "SymplecticGNNKalmanLayer",
    "SymplecticKalmanBlock",
]
