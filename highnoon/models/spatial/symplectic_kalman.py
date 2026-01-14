# highnoon/models/spatial/symplectic_kalman.py
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

"""UQHA Priority 3: Symplectic GNN-Kalman Filter Python wrapper.

Exposes the C++ SymplecticGNNKalman op for Port-Hamiltonian filter dynamics.

The Symplectic GNN-Kalman preserves Hamiltonian structure in:
- Message passing (symplectic coupling: dq = ∂H/∂p, dp = -∂H/∂q)
- Time integration (Verlet-like symplectic integrator)
- Kalman correction (structure-preserving update)

Benefits:
- Energy conservation in long-term dynamics
- Stable predictions without exponential error growth
- Graph-structured state spaces

Example:
    >>> from highnoon.models.spatial.symplectic_kalman import SymplecticGNNKalmanLayer
    >>>
    >>> layer = SymplecticGNNKalmanLayer(node_dim=64, dt=0.01)
    >>> q_out, p_out = layer(q, p, edges, observations, kalman_gain)
"""

import logging

import tensorflow as tf

from highnoon._native import get_op

logger = logging.getLogger(__name__)


def _get_symplectic_gnn_kalman_op():
    """Get the SymplecticGNNKalman C++ op."""
    ops = get_op("symplectic_gnn_kalman")
    if ops is None:
        return None
    return getattr(ops, "SymplecticGNNKalman", None)


def symplectic_gnn_kalman(
    node_q: tf.Tensor,
    node_p: tf.Tensor,
    edges: tf.Tensor,
    observations: tf.Tensor,
    kalman_gain: tf.Tensor,
    dt: float = 0.01,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Symplectic GNN-Kalman step.

    Performs one step of symplectic message passing + Kalman correction
    on a graph-structured state space.

    The state is represented as (q, p) pairs (position, momentum) for
    each node, preserving Hamiltonian structure.

    Args:
        node_q: Position states [batch, num_nodes, node_dim].
        node_p: Momentum states [batch, num_nodes, node_dim].
        edges: Edge list [num_edges, 2] mapping src -> dst.
        observations: Observation vector [batch, node_dim].
        kalman_gain: Kalman gain matrix [2 * node_dim] (for q and p).
        dt: Time step for symplectic integration.

    Returns:
        Tuple of (output_q, output_p) with same shapes as inputs.
    """
    op_fn = _get_symplectic_gnn_kalman_op()
    if op_fn is None:
        # Fallback to TensorFlow implementation
        return _symplectic_gnn_kalman_tf(node_q, node_p, edges, observations, kalman_gain, dt)

    return op_fn(
        node_q=node_q,
        node_p=node_p,
        edges=edges,
        observations=observations,
        kalman_gain=kalman_gain,
        dt=dt,
    )


def _symplectic_gnn_kalman_tf(
    node_q: tf.Tensor,
    node_p: tf.Tensor,
    edges: tf.Tensor,
    observations: tf.Tensor,
    kalman_gain: tf.Tensor,
    dt: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """TensorFlow fallback for symplectic GNN-Kalman."""
    tf.shape(node_q)[0]
    tf.shape(node_q)[1]
    dim = node_q.shape[-1]

    # Message passing: symplectic coupling
    src_idx = edges[:, 0]
    dst_idx = edges[:, 1]

    def batch_step(inputs):
        q, p = inputs

        # Gather source and destination states
        q_src = tf.gather(q, src_idx)
        q_dst = tf.gather(q, dst_idx)
        p_src = tf.gather(p, src_idx)
        p_dst = tf.gather(p, dst_idx)

        # Symplectic coupling: dq = ∂H/∂p, dp = -∂H/∂q
        dq = p_src - p_dst
        dp = -(q_src - q_dst)

        # Aggregate messages to destination nodes
        msg_q = tf.zeros_like(q)
        msg_p = tf.zeros_like(p)
        msg_q = tf.tensor_scatter_nd_add(msg_q, dst_idx[:, None], dq)
        msg_p = tf.tensor_scatter_nd_add(msg_p, dst_idx[:, None], dp)

        # Symplectic integrator (Verlet)
        p_half = p + 0.5 * dt * msg_p
        q_new = q + dt * p_half
        p_new = p_half + 0.5 * dt * msg_p

        # Kalman correction on first node
        if observations is not None and kalman_gain is not None:
            innovation = observations - q_new[0, :]
            q_new = tf.tensor_scatter_nd_update(
                q_new,
                [[0]],
                [q_new[0, :] + kalman_gain[:dim] * innovation],
            )
            p_new = tf.tensor_scatter_nd_update(
                p_new,
                [[0]],
                [p_new[0, :] + kalman_gain[dim:] * innovation],
            )

        return q_new, p_new

    out_q, out_p = tf.map_fn(
        batch_step, (node_q, node_p), fn_output_signature=(node_q.dtype, node_p.dtype)
    )
    return out_q, out_p


class SymplecticGNNKalmanLayer(tf.keras.layers.Layer):
    """Keras layer for Symplectic GNN-Kalman dynamics.

    Implements Port-Hamiltonian filter dynamics where the state
    evolves according to Hamiltonian mechanics with Kalman corrections.

    This preserves energy (up to numerical precision) and provides
    stable long-term dynamics prediction.

    Example:
        >>> layer = SymplecticGNNKalmanLayer(node_dim=64)
        >>> q_out, p_out = layer(q, p, edges)  # During training
    """

    def __init__(
        self,
        node_dim: int,
        dt: float = 0.01,
        learn_kalman_gain: bool = True,
        **kwargs,
    ):
        """Initialize SymplecticGNNKalmanLayer.

        Args:
            node_dim: Dimension of each node's state.
            dt: Time step for symplectic integration.
            learn_kalman_gain: Whether to learn the Kalman gain.
        """
        super().__init__(**kwargs)
        self.node_dim = node_dim
        self.dt = dt
        self.learn_kalman_gain = learn_kalman_gain

    def build(self, input_shape):
        """Build layer weights."""
        if self.learn_kalman_gain:
            # Kalman gain for both q and p
            self.kalman_gain = self.add_weight(
                name="kalman_gain",
                shape=[2 * self.node_dim],
                initializer=tf.keras.initializers.Constant(0.1),
                trainable=True,
            )
        super().build(input_shape)

    def call(
        self,
        node_q: tf.Tensor,
        node_p: tf.Tensor,
        edges: tf.Tensor,
        observations: tf.Tensor | None = None,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Forward pass.

        Args:
            node_q: Position states [batch, num_nodes, node_dim].
            node_p: Momentum states [batch, num_nodes, node_dim].
            edges: Edge list [num_edges, 2].
            observations: Optional observations [batch, node_dim].
            training: Whether in training mode.

        Returns:
            Tuple of (output_q, output_p).
        """
        kalman_gain = self.kalman_gain if self.learn_kalman_gain else None

        # Use zero observations if not provided
        if observations is None:
            observations = tf.zeros([tf.shape(node_q)[0], self.node_dim])

        return symplectic_gnn_kalman(
            node_q=node_q,
            node_p=node_p,
            edges=edges,
            observations=observations,
            kalman_gain=kalman_gain,
            dt=self.dt,
        )

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "node_dim": self.node_dim,
                "dt": self.dt,
                "learn_kalman_gain": self.learn_kalman_gain,
            }
        )
        return config


class SymplecticKalmanBlock(tf.keras.layers.Layer):
    """Port-Hamiltonian Kalman block for sequence processing.

    Wraps SymplecticGNNKalmanLayer for use in transformer-like architectures.
    Converts embedding sequences to graph structure for symplectic filtering.

    Example:
        >>> block = SymplecticKalmanBlock(embed_dim=256)
        >>> output = block(x)  # [B, L, D] -> [B, L, D]
    """

    def __init__(
        self,
        embed_dim: int,
        state_dim: int = 64,
        dt: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.dt = dt

    def build(self, input_shape):
        """Build layer weights."""
        # Project to state space
        self.to_qp = tf.keras.layers.Dense(2 * self.state_dim, name="to_qp")
        # Project back
        self.from_qp = tf.keras.layers.Dense(self.embed_dim, name="from_qp")
        # Create symplectic layer
        self.sgkf = SymplecticGNNKalmanLayer(node_dim=self.state_dim, dt=self.dt)
        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, embed_dim].
            training: Whether in training mode.

        Returns:
            Output tensor [batch, seq_len, embed_dim].
        """
        tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Project to (q, p) space
        qp = self.to_qp(x)  # [B, L, 2*state_dim]
        q = qp[..., : self.state_dim]  # [B, L, state_dim]
        p = qp[..., self.state_dim :]  # [B, L, state_dim]

        # Create sequential edges (each position connected to next)
        edges_list = []
        for i in range(seq_len - 1):
            edges_list.append([i, i + 1])
            edges_list.append([i + 1, i])  # Bidirectional
        edges = (
            tf.constant(edges_list, dtype=tf.int32)
            if edges_list
            else tf.zeros([0, 2], dtype=tf.int32)
        )

        # Apply symplectic filtering
        q_out, p_out = self.sgkf(q, p, edges, training=training)

        # Combine and project back
        qp_out = tf.concat([q_out, p_out], axis=-1)
        output = self.from_qp(qp_out)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "state_dim": self.state_dim,
                "dt": self.dt,
            }
        )
        return config


__all__ = [
    "symplectic_gnn_kalman",
    "SymplecticGNNKalmanLayer",
    "SymplecticKalmanBlock",
]
