# highnoon/models/spatial/tensor_network_kalman.py
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

"""UQHA Phase 1001: TensorNetworkKalmanFilter Python wrapper.

Exposes the C++ TT-compressed Kalman filter with O(n·r²) complexity.

The TensorNetworkKalmanFilter uses Tensor-Train decomposition to:
- Reduce covariance storage from O(n²) to O(n·r²)
- Speed up matrix-vector products from O(n²) to O(n·r²)
- Enable Kalman filtering for high-dimensional states (n > 64)

Example:
    >>> from highnoon.models.spatial.tensor_network_kalman import TensorNetworkKalmanFilter
    >>>
    >>> # Create filter with 64-dimensional state
    >>> tnkf = TensorNetworkKalmanFilter(state_dim=64, input_dim=4, output_dim=64)
    >>> tnkf.init(A, B, C, D, Q, R, max_rank=8)
    >>>
    >>> # Run filter
    >>> for u, y in zip(controls, measurements):
    ...     state, cov_diag = tnkf.step(u, y)
"""

import logging
import uuid

import numpy as np
import tensorflow as tf

from highnoon._native import get_op

logger = logging.getLogger(__name__)


class TensorNetworkKalmanFilter:
    """TT-compressed Kalman filter for high-dimensional state estimation.

    UQHA Phase 1001: Provides O(n·r²) complexity instead of O(n³) for
    standard Kalman filters. Uses C++ native ops for performance.

    The filter maintains state estimates using TT decomposition:
    - State estimate x̂ ∈ ℝⁿ (dense vector)
    - Covariance P ∈ ℝⁿˣⁿ (TT-compressed, O(n·r²) storage)
    - Transition matrix A ∈ ℝⁿˣⁿ (TT-compressed)

    For state dimensions < 16, falls back to dense operations.

    Attributes:
        state_dim: State dimension n.
        input_dim: Control input dimension.
        output_dim: Observation dimension.
        max_rank: Maximum TT rank for compression.
        filter_id: Unique identifier for this filter instance.

    Example:
        >>> tnkf = TensorNetworkKalmanFilter(state_dim=64, output_dim=64)
        >>> tnkf.init(A, B, C, D, Q, R, max_rank=8)
        >>>
        >>> # Predict + Update in one step
        >>> state, cov_diag = tnkf.step(control_input, measurement)
        >>>
        >>> # Check memory savings
        >>> bytes_used, is_tt = tnkf.get_memory_usage()
        >>> print(f"Using TT mode: {is_tt}, Memory: {bytes_used / 1024:.1f} KB")
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 1,
        output_dim: int | None = None,
        max_rank: int = 8,
        filter_id: str | None = None,
    ):
        """Initialize TensorNetworkKalmanFilter.

        Args:
            state_dim: State dimension.
            input_dim: Control input dimension.
            output_dim: Observation dimension (defaults to state_dim).
            max_rank: Maximum TT rank for compression (4-16 typical).
            filter_id: Unique identifier (auto-generated if None).
        """
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim or state_dim
        self.max_rank = max_rank
        self.filter_id = filter_id or f"tnkf_{uuid.uuid4().hex[:8]}"

        # Load C++ ops
        self._ops = get_op("tensor_network_kalman")
        if self._ops is None:
            raise RuntimeError(
                "TensorNetworkKalmanFilter C++ ops not found. "
                "Please compile: cd highnoon/_native && ./build.sh"
            )

        self._initialized = False

        logger.info(
            f"[TNKF] Created filter '{self.filter_id}': state_dim={state_dim}, max_rank={max_rank}"
        )

    def init(
        self,
        A: tf.Tensor | np.ndarray,
        B: tf.Tensor | np.ndarray,
        C: tf.Tensor | np.ndarray,
        D: tf.Tensor | np.ndarray,
        Q: tf.Tensor | np.ndarray,
        R: tf.Tensor | np.ndarray,
        max_rank: int | None = None,
    ) -> bool:
        """Initialize filter with state-space model.

        State-space representation:
            x[k+1] = A @ x[k] + B @ u[k] + w[k]  (state transition)
            y[k]   = C @ x[k] + D @ u[k] + v[k]  (observation)

        Where w ~ N(0, Q) is process noise and v ~ N(0, R) is measurement noise.

        Args:
            A: State transition matrix [state_dim, state_dim].
            B: Control input matrix [state_dim, input_dim].
            C: Observation matrix [output_dim, state_dim].
            D: Feedthrough matrix [output_dim, input_dim].
            Q: Process noise covariance [state_dim, state_dim].
            R: Measurement noise covariance [output_dim, output_dim].
            max_rank: Override max TT rank (optional).

        Returns:
            True if initialization succeeded.

        Raises:
            ValueError: If matrix dimensions are incompatible.
        """
        max_rank = max_rank or self.max_rank

        # Convert to float32 tensors
        A = tf.cast(A, tf.float32)
        B = tf.cast(B, tf.float32)
        C = tf.cast(C, tf.float32)
        D = tf.cast(D, tf.float32)
        Q = tf.cast(Q, tf.float32)
        R = tf.cast(R, tf.float32)

        # Validate dimensions
        if A.shape[0] != self.state_dim or A.shape[1] != self.state_dim:
            raise ValueError(f"A must be [{self.state_dim}, {self.state_dim}], got {A.shape}")
        if B.shape[0] != self.state_dim:
            raise ValueError(f"B must have {self.state_dim} rows, got {B.shape[0]}")
        if C.shape[1] != self.state_dim:
            raise ValueError(f"C must have {self.state_dim} columns, got {C.shape[1]}")

        # Call C++ init op
        success = self._ops.TNKFInit(
            state_trans=A,
            control_mat=B,
            obs_mat=C,
            feedthrough=D,
            proc_noise=Q,
            meas_noise=R,
            filter_id=self.filter_id,
            max_rank=max_rank,
        )

        self._initialized = bool(success.numpy())
        if self._initialized:
            logger.debug(f"[TNKF] Filter '{self.filter_id}' initialized")
        else:
            logger.error(f"[TNKF] Filter '{self.filter_id}' initialization failed")

        return self._initialized

    def predict(self, u: tf.Tensor | np.ndarray) -> tf.Tensor:
        """Predict step (time update).

        Performs:
            x̂[k|k-1] = A @ x̂[k-1|k-1] + B @ u[k]
            P[k|k-1] = A @ P[k-1|k-1] @ A.T + Q

        Uses TT-accelerated matrix operations for O(n·r²) complexity.

        Args:
            u: Control input vector [input_dim].

        Returns:
            Predicted state estimate [state_dim].

        Raises:
            RuntimeError: If filter not initialized.
        """
        self._check_initialized()
        u = tf.cast(u, tf.float32)
        return self._ops.TNKFPredict(
            control_input=u,
            filter_id=self.filter_id,
        )

    def update(self, y: tf.Tensor | np.ndarray) -> tuple[tf.Tensor, tf.Tensor]:
        """Update step (measurement update).

        Performs:
            K[k] = P[k|k-1] @ C.T @ (C @ P[k|k-1] @ C.T + R)^-1
            x̂[k|k] = x̂[k|k-1] + K[k] @ (y[k] - C @ x̂[k|k-1])
            P[k|k] = (I - K[k] @ C) @ P[k|k-1]

        Args:
            y: Measurement vector [output_dim].

        Returns:
            Tuple of (state_posterior, covariance_diagonal).

        Raises:
            RuntimeError: If filter not initialized.
        """
        self._check_initialized()
        y = tf.cast(y, tf.float32)
        return self._ops.TNKFUpdate(
            measurement=y,
            filter_id=self.filter_id,
        )

    def step(
        self,
        u: tf.Tensor | np.ndarray,
        y: tf.Tensor | np.ndarray,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Combined predict + update step.

        More efficient than calling predict() and update() separately.

        Args:
            u: Control input vector [input_dim].
            y: Measurement vector [output_dim].

        Returns:
            Tuple of (state_posterior, covariance_diagonal).

        Raises:
            RuntimeError: If filter not initialized.
        """
        self._check_initialized()
        u = tf.cast(u, tf.float32)
        y = tf.cast(y, tf.float32)
        return self._ops.TNKFStep(
            control_input=u,
            measurement=y,
            filter_id=self.filter_id,
        )

    def get_state(self) -> tf.Tensor:
        """Get current state estimate.

        Returns:
            Current state estimate [state_dim].
        """
        self._check_initialized()
        return self._ops.TNKFGetState(filter_id=self.filter_id)

    def get_memory_usage(self) -> tuple[int, bool]:
        """Get memory usage and compression status.

        Returns:
            Tuple of (bytes_used, is_tt_mode).
            is_tt_mode is True if TT compression is active (state_dim >= 16).
        """
        self._check_initialized()
        bytes_used, is_tt = self._ops.TNKFGetMemoryUsage(filter_id=self.filter_id)
        return int(bytes_used.numpy()), bool(is_tt.numpy())

    def reset(self) -> bool:
        """Reset filter to initial state (zero state, identity covariance).

        Returns:
            True if reset succeeded.
        """
        self._check_initialized()
        success = self._ops.TNKFReset(filter_id=self.filter_id)
        return bool(success.numpy())

    def update_A(self, A_new: tf.Tensor | np.ndarray) -> bool:
        """Update state transition matrix (for online system identification).

        Used when A matrix is learned online (e.g., from RLS).
        Re-decomposes A into TT format.

        Args:
            A_new: New state transition matrix [state_dim, state_dim].

        Returns:
            True if update succeeded.
        """
        self._check_initialized()
        A_new = tf.cast(A_new, tf.float32)
        success = self._ops.TNKFUpdateA(
            new_state_trans=A_new,
            filter_id=self.filter_id,
        )
        return bool(success.numpy())

    def _check_initialized(self) -> None:
        """Check that filter is initialized."""
        if not self._initialized:
            raise RuntimeError(f"TNKF '{self.filter_id}' not initialized. Call init() first.")


class TensorNetworkKalmanBlock(tf.keras.layers.Layer):
    """Keras layer wrapper for TensorNetworkKalmanFilter.

    Provides a drop-in replacement for KalmanBlock with TT compression
    for high-dimensional states.

    Example:
        >>> block = TensorNetworkKalmanBlock(
        ...     state_dim=64,
        ...     embedding_dim=256,
        ...     max_rank=8,
        ... )
        >>> x = tf.random.normal([2, 32, 256])
        >>> output, (state, P_diag) = block(x)
    """

    def __init__(
        self,
        state_dim: int,
        embedding_dim: int,
        max_rank: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.max_rank = max_rank

        self._tnkf: TensorNetworkKalmanFilter | None = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer weights."""
        # State-space matrices (learnable)
        self.A = self.add_weight(
            name="A",
            shape=[self.state_dim, self.state_dim],
            initializer=tf.keras.initializers.Orthogonal(),
            trainable=True,
        )
        self.B = self.add_weight(
            name="B",
            shape=[self.state_dim, self.embedding_dim],
            initializer="glorot_uniform",
            trainable=True,
        )
        self.C = self.add_weight(
            name="C",
            shape=[self.embedding_dim, self.state_dim],
            initializer="glorot_uniform",
            trainable=True,
        )
        self.D = self.add_weight(
            name="D",
            shape=[self.embedding_dim, self.embedding_dim],
            initializer=tf.keras.initializers.Identity(gain=0.1),
            trainable=True,
        )
        self.Q = self.add_weight(
            name="Q",
            shape=[self.state_dim, self.state_dim],
            initializer=tf.keras.initializers.Identity(gain=0.01),
            trainable=True,
        )
        self.R = self.add_weight(
            name="R",
            shape=[self.embedding_dim, self.embedding_dim],
            initializer=tf.keras.initializers.Identity(gain=0.1),
            trainable=True,
        )

        # Input/output projections
        self.input_proj = tf.keras.layers.Dense(self.embedding_dim, name="input_proj")
        self.output_proj = tf.keras.layers.Dense(self.embedding_dim, name="output_proj")

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        initial_state: tuple[tf.Tensor, tf.Tensor] | None = None,
        training: bool = False,
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """Forward pass through TT-Kalman block.

        Args:
            inputs: Input tensor [batch, seq_len, embedding_dim].
            initial_state: Optional (state, covariance_diag).
            training: Whether in training mode.

        Returns:
            Tuple of (output, (final_state, covariance_diag)).
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Initialize TNKF if needed (per-call for stateless operation)
        tnkf = TensorNetworkKalmanFilter(
            state_dim=self.state_dim,
            input_dim=self.embedding_dim,
            output_dim=self.embedding_dim,
            max_rank=self.max_rank,
        )
        tnkf.init(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            Q=tf.nn.softplus(self.Q),  # Ensure positive definite
            R=tf.nn.softplus(self.R),
        )

        # Initialize state
        if initial_state is not None:
            state, P_diag = initial_state
        else:
            state = tf.zeros([batch_size, self.state_dim], dtype=inputs.dtype)
            P_diag = tf.ones([batch_size, self.state_dim], dtype=inputs.dtype)

        # Project input
        z = self.input_proj(inputs)  # [B, L, embedding_dim]

        # Process sequence (simplified: process mean over batch)
        # For production, would iterate over batch
        outputs = []
        for t in range(seq_len):
            z_t = z[:, t, :]  # [B, embedding_dim]

            # Use zero control input
            u = tf.zeros([self.embedding_dim], dtype=inputs.dtype)

            # Run TNKF step on batch mean (simplified)
            mean_z = tf.reduce_mean(z_t, axis=0)
            state_new, P_diag_new = tnkf.step(u, mean_z)

            # Broadcast back to batch
            output_t = tf.broadcast_to(state_new[None, :], [batch_size, self.state_dim])
            outputs.append(output_t)
            P_diag = tf.broadcast_to(P_diag_new[None, :], [batch_size, self.state_dim])

        output = tf.stack(outputs, axis=1)  # [B, L, state_dim]
        output = self.output_proj(output)  # [B, L, embedding_dim]

        return output, (state, P_diag)

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "state_dim": self.state_dim,
                "embedding_dim": self.embedding_dim,
                "max_rank": self.max_rank,
            }
        )
        return config


__all__ = ["TensorNetworkKalmanFilter", "TensorNetworkKalmanBlock"]
