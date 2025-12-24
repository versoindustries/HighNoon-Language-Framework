#!/usr/bin/env python3
"""
src/models/quantum/tensor_layers.py

Tensor decomposition layers for parameter compression and expressivity enhancement.

Implements Tucker and Tensor-Ring decompositions for efficient weight representation:
- TuckerLayer: Tucker decomposition W = G ×₁ U⁽¹⁾ ×₂ U⁽²⁾ ×₃ U⁽³⁾
- TensorRingLayer: Ring-structured MPS with cyclic boundary conditions

These layers achieve significant parameter compression (target: 10× reduction)
while maintaining accuracy (< 2% loss) per Phase 7.4 roadmap.
"""


import numpy as np
import tensorflow as tf

from highnoon._native.ops.fused_tensor_layers_op import (
    fused_tensor_layers_available,
    fused_tensor_ring_forward,
    fused_tucker_forward,
)
from highnoon.models.utils.control_vars import ControlVarMixin


class TuckerLayer(ControlVarMixin, tf.keras.layers.Layer):
    """
    Tucker decomposition layer for parameter compression.

    Decomposes a weight tensor W into:
        W = G ×₁ U⁽¹⁾ ×₂ U⁽²⁾ ×₃ U⁽³⁾

    where:
    - G: Core tensor (small, shape [R1, R2, R3])
    - U⁽ⁱ⁾: Factor matrices (tall-skinny, shape [Dᵢ, Rᵢ])
    - Rᵢ: Tucker ranks (compression parameters)

    Memory: O(R₁R₂R₃ + Σᵢ DᵢRᵢ) vs O(D₁D₂D₃) for full tensor
    Typical compression ratio: 5-20× with < 2% accuracy loss
    """

    def __init__(
        self,
        output_dim: int,
        input_dim: int,
        tucker_ranks: int | list[int] | None = None,
        compression_ratio: float = 10.0,
        use_bias: bool = True,
        activation: str | None = None,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        adaptive_rank: bool = True,
        ard_threshold: float = 1e-3,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initialize Tucker decomposition layer.

        Args:
            output_dim: Output dimension (D_out)
            input_dim: Input dimension (D_in)
            tucker_ranks: Tucker ranks [R1, R2] or single value for both.
                         If None, computed from compression_ratio.
            compression_ratio: Target compression ratio (used if tucker_ranks is None)
            use_bias: Whether to include bias term
            activation: Activation function ('relu', 'tanh', 'gelu', etc.)
            kernel_initializer: Initializer for factor matrices
            bias_initializer: Initializer for bias
            adaptive_rank: Whether to use Bayesian ARD for adaptive rank tuning
            ard_threshold: Precision threshold for rank pruning
            name: Layer name
        """
        super().__init__(name=name, **kwargs)

        if output_dim < 1 or input_dim < 1:
            raise ValueError(
                f"output_dim and input_dim must be >= 1, got {output_dim}, {input_dim}"
            )

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.use_bias = use_bias
        self.compression_ratio = compression_ratio

        # Determine Tucker ranks
        if tucker_ranks is None:
            # Compute ranks from compression ratio
            # Full params: D_out * D_in
            # Tucker params: R1*R2 + D_out*R1 + D_in*R2
            # Solve for R1=R2=R to achieve target compression
            full_params = output_dim * input_dim
            target_params = full_params / compression_ratio

            # Quadratic: R² + R(D_out + D_in) - target_params = 0
            # R = (-b + sqrt(b²+4ac)) / 2a with a=1, b=(D_out+D_in), c=-target_params
            b = output_dim + input_dim
            discriminant = b**2 + 4 * target_params
            rank = int((-b + np.sqrt(discriminant)) / 2)
            rank = max(1, min(rank, min(output_dim, input_dim) // 2))

            self.tucker_ranks = [rank, rank]
        elif isinstance(tucker_ranks, int):
            self.tucker_ranks = [tucker_ranks, tucker_ranks]
        else:
            self.tucker_ranks = list(tucker_ranks)

        # Validate ranks
        if len(self.tucker_ranks) != 2:
            raise ValueError(f"tucker_ranks must have 2 elements, got {len(self.tucker_ranks)}")

        self.rank_out, self.rank_in = self.tucker_ranks

        if self.rank_out > output_dim or self.rank_in > input_dim:
            raise ValueError(
                f"Tucker ranks {self.tucker_ranks} exceed dimensions [{output_dim}, {input_dim}]"
            )

        # Activation
        self.activation_fn = tf.keras.activations.get(activation) if activation else None

        # Initializers
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        # Adaptive Rank Selection (ARD)
        self.adaptive_rank = adaptive_rank
        self.ard_threshold = ard_threshold
        self.log_alpha_in = None
        self.log_alpha_out = None

        # Weights (initialized in build)
        self.core_tensor = None
        self.factor_out = None
        self.factor_in = None
        self.bias = None

        # Masking for ARD
        self.rank_mask_in = None
        self.rank_mask_out = None

    def build(self, input_shape):
        """Build Tucker decomposition weights."""
        # Core tensor G: [R_out, R_in]
        self.core_tensor = self.add_weight(
            name="core_tensor",
            shape=(self.rank_out, self.rank_in),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=tf.float32,
        )

        # Factor matrix U_out: [D_out, R_out]
        self.factor_out = self.add_weight(
            name="factor_out",
            shape=(self.output_dim, self.rank_out),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=tf.float32,
        )

        # Factor matrix U_in: [D_in, R_in]
        self.factor_in = self.add_weight(
            name="factor_in",
            shape=(self.input_dim, self.rank_in),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=tf.float32,
        )

        # Bias
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=tf.float32,
            )

        # ARD Precision parameters (Log-Alpha)
        if self.adaptive_rank:
            self.log_alpha_in = self.add_weight(
                name="log_alpha_in",
                shape=(self.rank_in,),
                initializer=tf.keras.initializers.Constant(
                    2.0
                ),  # Start with low precision (wide prior)
                trainable=True,
            )
            self.log_alpha_out = self.add_weight(
                name="log_alpha_out",
                shape=(self.rank_out,),
                initializer=tf.keras.initializers.Constant(2.0),
                trainable=True,
            )

            # Binary masks for pruning (not trainable)
            self.rank_mask_in = self.add_weight(
                name="rank_mask_in",
                shape=(self.rank_in,),
                initializer="ones",
                trainable=False,
            )
            self.rank_mask_out = self.add_weight(
                name="rank_mask_out",
                shape=(self.rank_out,),
                initializer="ones",
                trainable=False,
            )

        # Register control variables
        self.register_control_var("core_tensor", self.core_tensor)
        self.register_control_var("factor_out", self.factor_out)
        self.register_control_var("factor_in", self.factor_in)
        if self.use_bias:
            self.register_control_var("bias", self.bias)

        super().build(input_shape)

    def call(self, inputs, training=False):
        """
        Forward pass: y = W·x + b where W is Tucker-decomposed.

        Efficient computation:
            1. x' = U_in^T · x  (project input)
            2. z = G · x'       (core tensor multiply)
            3. y = U_out · z    (project output)

        Args:
            inputs: Input tensor, shape [..., input_dim]
            training: Training mode flag

        Returns:
            Output tensor, shape [..., output_dim]
        """
        # Store input shape for broadcasting
        input_shape = tf.shape(inputs)
        batch_dims = input_shape[:-1]

        # Flatten batch dimensions: [..., D_in] -> [batch_flat, D_in]
        inputs_flat = tf.reshape(inputs, [-1, self.input_dim])

        # 1. Apply ARD scaling if enabled
        # Masked factors
        u_in = self.factor_in
        u_out = self.factor_out

        if self.adaptive_rank:
            # Apply pruning masks
            u_in = u_in * self.rank_mask_in
            u_out = u_out * self.rank_mask_out

            # ARD Scaling: Factorized weight follows N(0, α⁻¹)
            # We scale by sqrt(exp(-log_alpha))
            scale_in = tf.exp(-0.5 * self.log_alpha_in)
            scale_out = tf.exp(-0.5 * self.log_alpha_out)
            u_in = u_in * scale_in
            u_out = u_out * scale_out

        # Use Fused C++ kernel if available
        if fused_tensor_layers_available():
            output_flat = fused_tucker_forward(
                inputs_flat, u_in, self.core_tensor, u_out, self.bias
            )
        else:
            raise RuntimeError(
                "TuckerLayer requires C++ fused ops (fused_tucker_forward) in strict mode."
            )

        # Apply activation
        if self.activation_fn is not None:
            output_flat = self.activation_fn(output_flat)

        # Restore batch dimensions
        output_shape = tf.concat([batch_dims, [self.output_dim]], axis=0)
        output = tf.reshape(output_flat, output_shape)

        # Add ARD regularization loss: KL divergence with the sparse prior
        if self.adaptive_rank and training:
            # KL loss for ARD weights (simplified version of evidence lower bound)
            # L = Σ [ α_i * ||u_i||² - log(α_i) ]
            alpha_in = tf.exp(self.log_alpha_in)
            alpha_out = tf.exp(self.log_alpha_out)

            kl_in = 0.5 * (
                alpha_in * tf.reduce_sum(tf.square(self.factor_in), axis=0) - self.log_alpha_in
            )
            kl_out = 0.5 * (
                alpha_out * tf.reduce_sum(tf.square(self.factor_out), axis=0) - self.log_alpha_out
            )

            self.add_loss(1e-5 * (tf.reduce_sum(kl_in) + tf.reduce_sum(kl_out)))

        return output

    def reconstruct_full_weight(self) -> tf.Tensor:
        """
        Reconstruct full weight matrix from Tucker decomposition.

        Returns:
            W: Full weight matrix, shape [output_dim, input_dim]
        """
        # W = U_out · G · U_in^T
        # Step 1: G · U_in^T -> [R_out, D_in]
        temp = tf.matmul(self.core_tensor, self.factor_in, transpose_b=True)

        # Step 2: U_out · temp -> [D_out, D_in]
        full_weight = tf.matmul(self.factor_out, temp)

        return full_weight

    def compute_compression_ratio(self) -> float:
        """
        Compute actual compression ratio.

        Returns:
            ratio: Compression ratio (full_params / compressed_params)
        """
        full_params = self.output_dim * self.input_dim
        tucker_params = (
            self.rank_out * self.rank_in  # Core tensor
            + self.output_dim * self.rank_out  # Factor out
            + self.input_dim * self.rank_in  # Factor in
        )

        if self.use_bias:
            full_params += self.output_dim
            tucker_params += self.output_dim

        return full_params / tucker_params

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "input_dim": self.input_dim,
                "tucker_ranks": self.tucker_ranks,
                "compression_ratio": self.compression_ratio,
                "use_bias": self.use_bias,
                "activation": (
                    tf.keras.activations.serialize(self.activation_fn)
                    if self.activation_fn
                    else None
                ),
                "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
                "adaptive_rank": self.adaptive_rank,
                "ard_threshold": self.ard_threshold,
            }
        )
        return config

    def prune_ranks(self):
        """Prune ranks based on ARD precision parameters."""
        if not self.adaptive_rank:
            return

        # τ = exp(-log_alpha)
        prec_in = tf.exp(self.log_alpha_in).numpy()
        prec_out = tf.exp(self.log_alpha_out).numpy()

        # mask = precision < 1/threshold
        # If precision alpha is large, it means variance is small -> prune
        mask_in = prec_in < (1.0 / self.ard_threshold)
        mask_out = prec_out < (1.0 / self.ard_threshold)

        # Update binary masks
        self.rank_mask_in.assign(mask_in.astype(np.float32))
        self.rank_mask_out.assign(mask_out.astype(np.float32))

        active_in = np.sum(mask_in)
        active_out = np.sum(mask_out)
        return active_out, active_in


class TensorRingLayer(ControlVarMixin, tf.keras.layers.Layer):
    """
    Tensor-Ring decomposition layer with cyclic boundary conditions.

    Ring-structured tensor network similar to MPS but with periodic boundary:
        W = Σ A⁽¹⁾ · A⁽²⁾ · ... · A⁽ᴺ⁾ (with A⁽ᴺ⁾ connecting back to A⁽¹⁾)

    Key differences from standard Tensor-Train (TT):
    - Cyclic structure: No boundary cores with bond_dim=1
    - All bonds have same dimension: more uniform expressivity
    - Better for periodic systems (molecular rings, cyclic graphs)

    Memory: O(N * R² * D) where N=#cores, R=ring rank, D=local dim
    """

    def __init__(
        self,
        output_dim: int,
        input_dim: int,
        num_cores: int = 4,
        ring_rank: int = 8,
        use_bias: bool = True,
        activation: str | None = None,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        enable_cpp: bool = True,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initialize Tensor-Ring layer.

        Args:
            output_dim: Output dimension
            input_dim: Input dimension
            num_cores: Number of tensor cores in ring (N)
            ring_rank: Bond dimension for all connections (R)
            use_bias: Whether to include bias term
            activation: Activation function
            kernel_initializer: Initializer for core tensors
            bias_initializer: Initializer for bias
            name: Layer name
        """
        super().__init__(name=name, **kwargs)

        if output_dim < 1 or input_dim < 1:
            raise ValueError("output_dim and input_dim must be >= 1")
        if num_cores < 2:
            raise ValueError(f"num_cores must be >= 2 for ring structure, got {num_cores}")
        if ring_rank < 1:
            raise ValueError(f"ring_rank must be >= 1, got {ring_rank}")

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_cores = num_cores
        self.ring_rank = ring_rank
        self.use_bias = use_bias
        self.enable_cpp = enable_cpp

        # Compute local dimensions for each core
        # Distribute dimensions across cores as evenly as possible
        self.local_dim_out = self._compute_local_dims(output_dim, num_cores)
        self.local_dim_in = self._compute_local_dims(input_dim, num_cores)

        # Activation
        self.activation_fn = tf.keras.activations.get(activation) if activation else None

        # Initializers
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        # Ring cores (initialized in build)
        self.ring_cores: list[tf.Variable] = []
        self.bias = None

    def _compute_local_dims(self, total_dim: int, num_cores: int) -> list[int]:
        """
        Distribute total dimension across cores as evenly as possible.

        Args:
            total_dim: Total dimension to distribute
            num_cores: Number of cores

        Returns:
            local_dims: List of local dimensions for each core
        """
        base_dim = total_dim // num_cores
        remainder = total_dim % num_cores

        local_dims = [base_dim + (1 if i < remainder else 0) for i in range(num_cores)]
        return local_dims

    def build(self, input_shape):
        """Build Tensor-Ring cores."""
        # Create ring cores: A[i] has shape [R, D_out_i * D_in_i, R]
        # where R is ring_rank, and D_out_i, D_in_i are local output/input dimensions

        for i in range(self.num_cores):
            # Local dimensions for this core
            d_out_i = self.local_dim_out[i]
            d_in_i = self.local_dim_in[i]
            local_features = d_out_i * d_in_i

            # Core tensor shape: [R, local_features, R]
            # For cyclic structure, all bonds have rank R
            core_shape = (self.ring_rank, local_features, self.ring_rank)

            # Xavier initialization with scaling for stability
            stddev = np.sqrt(2.0 / (self.ring_rank + local_features))

            core = self.add_weight(
                name=f"ring_core_{i}",
                shape=core_shape,
                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev),
                trainable=True,
                dtype=tf.float32,
            )

            self.ring_cores.append(core)
            self.register_control_var(f"ring_core_{i}", core)

        # Bias
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=tf.float32,
            )
            self.register_control_var("bias", self.bias)

        super().build(input_shape)

    def _call_cpp(self, input_splits, batch_dims):
        """Use C++ kernel for proper TR Trace contraction."""
        output_flat = fused_tensor_ring_forward(
            inputs=input_splits,
            cores=self.ring_cores,
            bias=self.bias,
            ring_rank=self.ring_rank,
            local_dims_in=self.local_dim_in,
            local_dims_out=self.local_dim_out,
        )
        output_shape = tf.concat([batch_dims, [self.output_dim]], axis=0)
        return tf.reshape(output_flat, output_shape)

    def call(self, inputs, training=False):
        """Forward pass through Tensor-Ring layer."""
        input_shape = tf.shape(inputs)
        batch_dims = input_shape[:-1]
        batch_size = tf.reduce_prod(batch_dims)
        inputs_flat = tf.reshape(inputs, [batch_size, self.input_dim])

        # Split input across cores
        input_splits = []
        start_idx = 0
        for d_in_i in self.local_dim_in:
            input_splits.append(inputs_flat[:, start_idx : start_idx + d_in_i])
            start_idx += d_in_i

        if self.enable_cpp and fused_tensor_layers_available():
            return self._call_cpp(input_splits, batch_dims)
        else:
            raise RuntimeError(
                "TensorRingLayer requires C++ fused ops (fused_tensor_ring_forward) in strict mode."
            )

    def compute_compression_ratio(self) -> float:
        """
        Compute actual compression ratio.

        Returns:
            ratio: Compression ratio (full_params / compressed_params)
        """
        full_params = self.output_dim * self.input_dim

        # Ring params: N cores × R × (D_out_i * D_in_i) × R
        ring_params = 0
        for i in range(self.num_cores):
            d_out_i = self.local_dim_out[i]
            d_in_i = self.local_dim_in[i]
            ring_params += self.ring_rank * (d_out_i * d_in_i) * self.ring_rank

        if self.use_bias:
            full_params += self.output_dim
            ring_params += self.output_dim

        return full_params / ring_params

    def canonicalize(self, mode="left"):
        """Perform DMRG-style canonicalization of the ring cores.

        Args:
            mode: "left" for left-canonical or "right" for right-canonical.
        """
        for i in range(self.num_cores - 1):
            if mode == "left":
                # Left canonical: A[i] = Q * R, then A[i+1] = R * A[i+1]
                core = self.ring_cores[i]
                r1, d, r2 = core.shape
                core_reshaped = tf.reshape(core, [r1 * d, r2])
                q, r = tf.linalg.qr(core_reshaped)
                self.ring_cores[i].assign(tf.reshape(q, [r1, d, r2]))

                next_core = self.ring_cores[i + 1]
                nr1, nd, nr2 = next_core.shape
                next_core_reshaped = tf.reshape(next_core, [nr1, nd * nr2])
                new_next = tf.matmul(r, next_core_reshaped)
                self.ring_cores[i + 1].assign(tf.reshape(new_next, [nr1, nd, nr2]))
            else:
                idx = self.num_cores - 1 - i
                core = self.ring_cores[idx]
                r1, d, r2 = core.shape
                core_reshaped = tf.reshape(core, [r1, d * r2])
                q, r = tf.linalg.qr(tf.transpose(core_reshaped))
                self.ring_cores[idx].assign(tf.transpose(tf.reshape(q, [d * r2, r1])))

                prev_core = self.ring_cores[idx - 1]
                pr1, pd, pr2 = prev_core.shape
                prev_core_reshaped = tf.reshape(prev_core, [pr1 * pd, pr2])
                new_prev = tf.matmul(prev_core_reshaped, tf.transpose(r))
                self.ring_cores[idx - 1].assign(tf.reshape(new_prev, [pr1, pd, pr2]))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "input_dim": self.input_dim,
                "num_cores": self.num_cores,
                "ring_rank": self.ring_rank,
                "use_bias": self.use_bias,
                "enable_cpp": self.enable_cpp,
                "activation": (
                    tf.keras.activations.serialize(self.activation_fn)
                    if self.activation_fn
                    else None
                ),
                "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            }
        )
        return config


# =============================================================================
# UNIFIED FACTORY
# =============================================================================


def create_tensor_network_layer(
    type: str = "tucker", output_dim: int = 1024, input_dim: int = 1024, rank: int = 64, **kwargs
) -> tf.keras.layers.Layer:
    """Unified factory for tensor network layers.

    Args:
        type: "tucker", "ring", or "tt" (Tensor-Train)
        output_dim: Output dimension
        input_dim: Input dimension
        rank: Target rank/bond dimension
        **kwargs: Additional layer arguments

    Returns:
        Tensor network layer instance.
    """
    if type.lower() == "tucker":
        return TuckerLayer(
            output_dim, input_dim, tucker_ranks=[rank, rank] if rank else None, **kwargs
        )
    elif type.lower() in ["ring", "tensor-ring", "tr"]:
        return TensorRingLayer(output_dim, input_dim, ring_rank=rank if rank else 8, **kwargs)
    elif type.lower() == "tt":
        # TT decomposition - use TuckerLayer as fallback since TTLayer would self-import
        # For dedicated TT support, use highnoon.models.layers.tt_dense.TTDense
        from highnoon.models.layers.tt_dense import TTDense

        return TTDense(output_dim=output_dim, tt_rank=rank if rank else 8, **kwargs)
    else:
        raise ValueError(f"Unknown tensor network type: {type}")


# Export
__all__ = ["TuckerLayer", "TensorRingLayer", "create_tensor_network_layer"]
