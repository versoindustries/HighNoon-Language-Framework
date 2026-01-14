# highnoon/qsg/mps_context.py
import logging

import tensorflow as tf
from tensorflow.keras import layers

from highnoon._native import get_op

logger = logging.getLogger(__name__)

# Register gradient for QSGMPSContextEntangle (Phase A1 training support)
_gradient_registered = False


def _register_mps_gradient():
    """Register gradient function for QSGMPSContextEntangle native op."""
    global _gradient_registered
    if _gradient_registered:
        return

    qsg_ops = get_op("fused_qsg_op")
    if qsg_ops is None:
        logger.warning("Cannot register gradient: native ops not available")
        return

    @tf.RegisterGradient("QSGMPSContextEntangle")
    def _qsgmps_context_entangle_grad(op, grad_context, grad_entropy):
        """Gradient for QSGMPSContextEntangle op."""
        embeddings = op.inputs[0]
        site_weights = op.inputs[1]

        # Call the C++ gradient kernel
        grad_embeddings, grad_site_weights = qsg_ops.qsgmps_context_entangle_grad(
            grad_context,
            grad_entropy,
            embeddings,
            site_weights,
        )
        return grad_embeddings, grad_site_weights

    _gradient_registered = True
    logger.debug("Registered gradient for QSGMPSContextEntangle")


# Try to register gradient at module load (if ops are available)
try:
    _register_mps_gradient()
except Exception as e:
    logger.debug(f"Deferred gradient registration: {e}")


class MPSContextEntangler(layers.Layer):
    """MPS-based context encoding for QSG Phase 1.

    Instead of running the full model, we:
    1. Embed tokens via shared embeddings
    2. Run QSGMPSContextEntangle (native O(n·chi²) op)
    3. Output entangled context with bond dimension chi

    Complexity: O(n·chi^2) where chi = bond_dim (default 32)
    """

    def __init__(
        self,
        embedding_dim: int,
        bond_dim: int = 32,
        spatial_bond_dim: int = 4,
        num_mps_layers: int = 1,  # Typically 1 for context entanglement
        name: str = "mps_context_entangler",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.embedding_dim = embedding_dim
        self.bond_dim = bond_dim
        self.spatial_bond_dim = spatial_bond_dim
        self.num_mps_layers = num_mps_layers

        # Site tensor projection: maps input embedding to [bond, phys, bond] site tensor
        # This allows the site tensor to be dynamic (context-dependent)
        self.site_proj = layers.Dense(
            bond_dim * spatial_bond_dim * bond_dim, name=f"{name}_site_proj"
        )

        # Load native ops from fused_qsg_op
        self._qsg_ops = get_op("fused_qsg_op")

    def call(self, embeddings: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute entangled context representation.

        Args:
            embeddings: Input tensor [batch, seq, dim]

        Returns:
            context: Entangled hidden states [batch, seq, dim]
            entanglement_entropy: Per-bond entropy [batch, seq-1]
        """
        # Ensure native op is available
        if self._qsg_ops is None:
            raise RuntimeError("Native QSG ops not found. Please compile native modules.")

        batch_size = tf.shape(embeddings)[0]
        seq_len = tf.shape(embeddings)[1]

        # 1. Project embeddings to site tensors
        # [batch, seq, bond * phys * bond]
        site_flat = self.site_proj(embeddings)

        # Reshape to [batch, seq, bond, phys, bond] for the native op
        site_weights = tf.reshape(
            site_flat, [batch_size, seq_len, self.bond_dim, self.spatial_bond_dim, self.bond_dim]
        )

        # 2. Call Native MPS Context Entanglement Op
        # TF converts CamelCase to snake_case: QSGMPSContextEntangle -> qsgmps_context_entangle
        context, entropy = self._qsg_ops.qsgmps_context_entangle(embeddings, site_weights)

        return context, entropy

    def entangle(self, embeddings: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Alias for call() for backward compatibility."""
        return self.call(embeddings)


def compute_position_importance(entropy: tf.Tensor) -> tf.Tensor:
    """High entropy bonds indicate important boundaries.

    Uses entanglement entropy to determine which positions are
    most critical for generation coherence.

    Args:
        entropy: Bond entropy tensor [batch, seq-1]

    Returns:
        importance: [batch, seq] weights for each position, normalized to sum to 1
    """
    # Entropy at bond (i, i+1) indicates correlation between positions
    # Average entropy on either side of each position

    # Pad entropy to align with positions
    # Left pad for bond (i-1, i) -> bond 0 maps to pos 1
    # Right pad for bond (i, i+1) -> bond 0 maps to pos 0

    B = tf.shape(entropy)[0]

    # Prepend/Append zeros for boundary conditions
    zeros = tf.zeros([B, 1], dtype=entropy.dtype)

    left_entropy = tf.concat([zeros, entropy], axis=1)  # Entropy to the left of pos i
    right_entropy = tf.concat([entropy, zeros], axis=1)  # Entropy to the right of pos i

    importance = (left_entropy + right_entropy) / 2.0

    # Softmax to normalize probability distribution
    importance = tf.nn.softmax(importance, axis=-1)

    return importance
