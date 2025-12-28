# src/training/optimizers.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import tensorflow as tf

# --- Logger Setup ---
logger = logging.getLogger(__name__)


class SophiaG(tf.keras.optimizers.Optimizer):
    """
    Implementation of the SophiaG optimizer.
    Sophia is a second-order optimizer that uses a lightweight, stochastic
    estimate of the diagonal Hessian to precondition gradients.
    """

    def __init__(
        self,
        model,
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.99,
        rho=0.03,
        weight_decay=0.0,
        k=10,
        name="SophiaG",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate, name=name, weight_decay=weight_decay, **kwargs
        )
        self.model = model

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.rho = rho
        self.epsilon = 1e-12
        self.k = k
        # FIX: Store slot variables in internal lists for robust access across replicas.
        self.momentums = []
        self.hessians = []

    def build(self, var_list):
        """Create slots for the first moment (m) and diagonal Hessian estimate (h)."""
        # --- START: DEFINITIVE FIX for HPO ---
        # The original build was not fully idempotent. If called on a new model
        # instance (as in HPO), it would fail. This check ensures that slots are
        # only created if they don't exist *or* if the optimizer is being built
        # for a new set of variables, which clears the old state.
        super().build(var_list)
        self.momentums = []
        self.hessians = []
        for var in var_list:
            self.momentums.append(self.add_variable_from_reference(var, "m"))
            # CRITICAL FIX: Initialize Hessian to 1.0 instead of 0.0
            # When update_hessian() is not called (as in molecular property prediction),
            # the Hessian remains at its initial value. A zero Hessian causes division
            # by epsilon (1e-12), amplifying gradients by 1e12 and causing immediate NaN.
            # Initializing to 1.0 gives Adam-like behavior until update_hessian() is called.
            h_slot = self.add_variable_from_reference(var, "h")
            h_slot.assign(tf.ones_like(var))  # Initialize to 1.0 for stability
            self.hessians.append(h_slot)
        # --- END: DEFINITIVE FIX ---

    def update_step(self, grad, var, learning_rate):
        """
        Update step for SophiaG optimizer. This method implements the core logic for
        updating a single variable and is required by the Keras Optimizer base class.
        """
        if grad is None:
            return

        lr = tf.cast(learning_rate, var.dtype.base_dtype)
        beta_1_t = tf.cast(self.beta_1, var.dtype.base_dtype)
        rho_t = tf.cast(self.rho, var.dtype.base_dtype)

        # FIX: Access slots via the optimizer's internal list using the variable's
        # unique index. This is robust to TensorFlow's distributed strategies.
        var_index = self._get_variable_index(var)
        m = self.momentums[var_index]
        h = self.hessians[var_index]

        if isinstance(grad, tf.IndexedSlices):
            # Sparse update logic for embeddings
            m_slices = tf.gather(m, grad.indices)
            h_slices = tf.gather(h, grad.indices)

            m_t_slices = m_slices * beta_1_t + (1.0 - beta_1_t) * grad.values

            update_slices = m_t_slices / tf.maximum(h_slices, self.epsilon)
            clipped_update_slices = tf.clip_by_value(update_slices, -rho_t, rho_t)

            # --- START: DEFINITIVE FIX ---
            # Use tf.tensor_scatter_nd_update and assign the result back.
            # This is the universally compatible way to perform sparse updates.
            m_updated = tf.tensor_scatter_nd_update(
                m, tf.expand_dims(grad.indices, axis=-1), m_t_slices
            )
            m.assign(m_updated)
            # --- END: DEFINITIVE FIX ---
            var.scatter_add(tf.IndexedSlices(-lr * clipped_update_slices, grad.indices))
        else:
            # Dense update logic
            m.assign(beta_1_t * m + (1.0 - beta_1_t) * grad)
            update = m / tf.maximum(h, self.epsilon)
            clipped_update = tf.clip_by_value(update, -rho_t, rho_t)
            var.assign_sub(lr * clipped_update)

    def update_hessian(self, loss_fn, data_batch):
        """Update the diagonal Hessian estimate 'h' periodically."""
        beta_2_t = tf.cast(self.beta_2, "float32")

        with tf.GradientTape() as h_tape:
            ctx, tgt = data_batch
            # --- START: FIX ---
            # The self.model's call method now returns three values (logits, moe_info, aux_metrics).
            # We need to unpack all three, even if we only use the logits here.
            logits, _, _ = self.model((ctx, tgt), training=True)
            # --- END: FIX ---

            probs = tf.nn.softmax(logits, axis=-1)
            safe_probs = tf.clip_by_value(probs[:, -1, :], 1e-9, 1.0)
            sampled_labels = tf.random.categorical(tf.math.log(safe_probs), 1)
            sampled_labels = tf.squeeze(sampled_labels, axis=-1)

            sampled_loss = tf.reduce_mean(loss_fn(tgt[:, 1:], logits[:, : tf.shape(tgt)[1] - 1, :]))

        # --- START: DEFINITIVE FIX for untracked variables ---
        # The tape must compute gradients with respect to the variables the optimizer
        # was built with, which are stored in `self.variables`. Using `self.model.trainable_variables`
        # can lead to mismatches, especially during HPO where the model object is replaced.
        hessian_grads = h_tape.gradient(sampled_loss, self.variables)

        # Now, we iterate through the optimizer's `self.variables` and the computed
        # gradients. This ensures a one-to-one correspondence and prevents KeyErrors
        # from `_get_variable_index` because we are guaranteed to be iterating over
        # variables that the optimizer is aware of.
        for h_grad, var in zip(hessian_grads, self.variables):
            if h_grad is not None:
                # The `try...except` block is no longer strictly necessary with this fix,
                # but it remains as a safeguard against other potential edge cases.
                try:
                    var_index = self._get_variable_index(var)
                    h = self.hessians[var_index]
                    h_hat = tf.square(h_grad)
                    h.assign(beta_2_t * h + (1.0 - beta_2_t) * h_hat)
                except KeyError:
                    # This can happen if the Hessian tape tracks a variable the optimizer doesn't,
                    # This is common during hyperparameter tuning where models are rebuilt.
                    # It is safe to ignore this variable for the Hessian update.
                    logger.warning(f"Skipping Hessian update for untracked variable: {var.name}")
        # --- END: DEFINITIVE FIX for untracked variables ---

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "rho": self.rho,
                "k": self.k,
            }
        )
        return config


class QIAO(tf.keras.optimizers.Optimizer):
    """
    Implementation of the Quantum-Inspired Alternating Optimizer (QIAO).

    This optimizer alternates between a standard "cost" update step (using
    SophiaG logic) and an exploratory "mixer" step that adds gradient-orthogonal
    noise to escape local minima, inspired by the structure of QAOA.
    """

    def __init__(
        self,
        model,
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.99,
        rho=0.03,
        weight_decay=0.0,
        mixer_frequency=10,  # Corresponds to M in the proposal
        mixer_strength=0.1,  # Controls the magnitude of the noise
        name="QIAO",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate, name=name, weight_decay=weight_decay, **kwargs
        )
        self.model = model
        # SophiaG parameters
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.rho = rho
        self.epsilon = 1e-12

        # QIAO parameters
        self.mixer_frequency = mixer_frequency
        self.mixer_strength = mixer_strength

        # SophiaG slots
        self.momentums = []
        self.hessians = []

    def build(self, var_list):
        """Create slots for the SophiaG momentums and Hessian estimates."""
        super().build(var_list)
        if not hasattr(self, "momentums") or not self.momentums:
            self.momentums = []
            for var in var_list:
                self.momentums.append(self.add_variable_from_reference(var, "m"))
        if not hasattr(self, "hessians") or not self.hessians:
            self.hessians = []
            for var in var_list:
                # CRITICAL FIX: Initialize Hessian to 1.0 instead of 0.0
                # When update_hessian() is not called, the Hessian remains at its
                # initial value. A zero Hessian causes division by epsilon (1e-12),
                # amplifying gradients by 1e12 and causing immediate NaN.
                # Initializing to 1.0 gives Adam-like behavior until update_hessian() is called.
                h_slot = self.add_variable_from_reference(var, "h")
                h_slot.assign(tf.ones_like(var))  # Initialize to 1.0 for stability
                self.hessians.append(h_slot)

    def update_step(self, grad, var, learning_rate):
        """
        Performs a single update step, alternating between cost and mixer steps.
        """
        if grad is None:
            return

        lr = tf.cast(learning_rate, var.dtype.base_dtype)

        # Check if it's time for a mixer step using the built-in iteration counter
        is_mixer_step = tf.equal(self.iterations % self.mixer_frequency, 0)

        def cost_step():
            """Performs a SophiaG update (exploitation)."""
            beta_1_t = tf.cast(self.beta_1, var.dtype.base_dtype)
            rho_t = tf.cast(self.rho, var.dtype.base_dtype)
            var_index = self._get_variable_index(var)
            m = self.momentums[var_index]
            h = self.hessians[var_index]

            if isinstance(grad, tf.IndexedSlices):
                m_slices = tf.gather(m, grad.indices)
                h_slices = tf.gather(h, grad.indices)
                m_t_slices = m_slices * beta_1_t + (1.0 - beta_1_t) * grad.values
                update_slices = m_t_slices / tf.maximum(h_slices, self.epsilon)
                clipped_update_slices = tf.clip_by_value(update_slices, -rho_t, rho_t)
                # Apply the same sparse update fix here
                m_updated = tf.tensor_scatter_nd_update(
                    m, tf.expand_dims(grad.indices, axis=-1), m_t_slices
                )
                m.assign(m_updated)
                var.scatter_add(tf.IndexedSlices(-lr * clipped_update_slices, grad.indices))
            else:
                m.assign(beta_1_t * m + (1.0 - beta_1_t) * grad)
                update = m / tf.maximum(h, self.epsilon)
                clipped_update = tf.clip_by_value(update, -rho_t, rho_t)
                var.assign_sub(lr * clipped_update)
            return

        def mixer_step():
            """Performs an exploration step with gradient-orthogonal noise.

            Note: For sparse gradients (IndexedSlices), we fall back to cost_step
            since orthogonal noise projection is not well-defined for sparse tensors.
            """
            # For sparse gradients, mixer step is skipped - use cost step
            if isinstance(grad, tf.IndexedSlices):
                cost_step()
                return

            # Generate random noise from a normal distribution
            noise = tf.random.normal(shape=tf.shape(var), dtype=var.dtype)

            # Project noise onto the gradient to find the parallel component
            grad_norm_sq = tf.reduce_sum(grad * grad)
            # Avoid division by zero if gradient is zero
            grad_norm_sq = tf.where(tf.equal(grad_norm_sq, 0.0), 1.0, grad_norm_sq)

            dot_product = tf.reduce_sum(noise * grad)
            noise_parallel_to_grad = (dot_product / grad_norm_sq) * grad

            # Subtract the parallel component to get the orthogonal component
            noise_orthogonal = noise - noise_parallel_to_grad

            # Normalize the orthogonal noise
            noise_ortho_norm = tf.sqrt(tf.reduce_sum(noise_orthogonal * noise_orthogonal))
            # Avoid division by zero if orthogonal noise is zero
            noise_ortho_norm = tf.where(tf.equal(noise_ortho_norm, 0.0), 1.0, noise_ortho_norm)

            normalized_noise = noise_orthogonal / noise_ortho_norm

            # Apply the update
            var.assign_add(lr * self.mixer_strength * normalized_noise)
            return

        # For sparse gradients use cost_step directly, else tf.cond for graph mode
        if isinstance(grad, tf.IndexedSlices):
            # Sparse grads can't use mixer step's noise projection
            cost_step()
        else:
            # Use tf.cond to choose the step, ensuring graph compatibility
            tf.cond(is_mixer_step, mixer_step, cost_step)

    def update_hessian(self, loss_fn, data_batch):
        """
        Update the diagonal Hessian estimate 'h' periodically. This is required
        for the SophiaG 'cost' step.
        """
        beta_2_t = tf.cast(self.beta_2, "float32")

        with tf.GradientTape() as h_tape:
            ctx, tgt = data_batch
            # --- START: FIX ---
            # The self.model's call method now returns three values (logits, moe_info, aux_metrics).
            # We need to unpack all three, even if we only use the logits here.
            logits, _, _ = self.model((ctx, tgt), training=True)
            # --- END: FIX ---

            probs = tf.nn.softmax(logits, axis=-1)
            safe_probs = tf.clip_by_value(probs[:, -1, :], 1e-9, 1.0)
            sampled_labels = tf.random.categorical(tf.math.log(safe_probs), 1)
            sampled_labels = tf.squeeze(sampled_labels, axis=-1)

            # Using the full loss for Hessian estimation as SophiaG does
            sampled_loss = tf.reduce_mean(loss_fn(tgt[:, 1:], logits[:, : tf.shape(tgt)[1] - 1, :]))

        # Apply the same fix as in SophiaG: compute gradients w.r.t. self.variables.
        hessian_grads = h_tape.gradient(sampled_loss, self.variables)

        # Iterate over self.variables to ensure alignment.
        for h_grad, var in zip(hessian_grads, self.variables):
            if h_grad is not None:
                try:
                    var_index = self._get_variable_index(var)
                    h = self.hessians[var_index]
                    h_hat = tf.square(h_grad)
                    h.assign(beta_2_t * h + (1.0 - beta_2_t) * h_hat)
                except KeyError:
                    # This can happen if the Hessian tape tracks a variable the optimizer doesn't.
                    # This is common during hyperparameter tuning where models are rebuilt.
                    # It is safe to ignore this variable for the Hessian update.
                    logger.warning(f"Skipping Hessian update for untracked variable: {var.name}")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "rho": self.rho,
                "mixer_frequency": self.mixer_frequency,
                "mixer_strength": self.mixer_strength,
            }
        )
        return config


class Lion(tf.keras.optimizers.Optimizer):
    """
    Implementation of the Lion (EvoLved Sign Momentum) optimizer.

    Lion uses the sign of the momentum to update weights, achieving memory
    efficiency comparable to SGD while matching or exceeding AdamW performance.

    Reference: https://arxiv.org/abs/2302.06675 (Google Brain, 2023)

    Key properties:
    - Memory efficient: no second moment (like SGD)
    - Uses sign operation for updates (magnitude-invariant)
    - EMA interpolation between gradient and momentum
    """

    def __init__(
        self,
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.99,
        weight_decay=0.0,
        name="Lion",
        **kwargs,
    ):
        """Initialize Lion optimizer.

        Args:
            learning_rate: Learning rate (default 1e-4, typically lower than Adam)
            beta_1: Coefficient for computing running average of gradient (default 0.9)
            beta_2: Coefficient for computing EMA of gradient for next step (default 0.99)
            weight_decay: Weight decay coefficient (default 0.0)
            name: Optimizer name
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            learning_rate=learning_rate, name=name, weight_decay=weight_decay, **kwargs
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        # Momentum storage
        self.momentums = []

    def build(self, var_list):
        """Create slots for first moment (momentum)."""
        super().build(var_list)
        self.momentums = []
        for var in var_list:
            self.momentums.append(self.add_variable_from_reference(var, "m"))

    def update_step(self, grad, var, learning_rate):
        """
        Lion update step using sign of interpolated momentum.

        Update rule:
        1. c_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t  (interpolation for update)
        2. theta_t = theta_{t-1} - lr * sign(c_t)       (sign-based update)
        3. m_t = beta_2 * m_{t-1} + (1 - beta_2) * g_t  (momentum update for next step)
        """
        if grad is None:
            return

        lr = tf.cast(learning_rate, var.dtype.base_dtype)
        beta_1_t = tf.cast(self.beta_1, var.dtype.base_dtype)
        beta_2_t = tf.cast(self.beta_2, var.dtype.base_dtype)

        var_index = self._get_variable_index(var)
        m = self.momentums[var_index]

        if isinstance(grad, tf.IndexedSlices):
            # Sparse update for embeddings
            m_slices = tf.gather(m, grad.indices)

            # Compute interpolation for update direction
            c_t = beta_1_t * m_slices + (1.0 - beta_1_t) * grad.values

            # Sign-based update
            update = lr * tf.sign(c_t)
            var.scatter_add(tf.IndexedSlices(-update, grad.indices))

            # Update momentum for next step
            m_t_slices = beta_2_t * m_slices + (1.0 - beta_2_t) * grad.values
            m_updated = tf.tensor_scatter_nd_update(
                m, tf.expand_dims(grad.indices, axis=-1), m_t_slices
            )
            m.assign(m_updated)
        else:
            # Dense update
            # Compute interpolation for update direction
            c_t = beta_1_t * m + (1.0 - beta_1_t) * grad

            # Sign-based update
            var.assign_sub(lr * tf.sign(c_t))

            # Update momentum for next step (different beta for state update)
            m.assign(beta_2_t * m + (1.0 - beta_2_t) * grad)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
            }
        )
        return config


class SympFlowQNGOptimizer(tf.keras.optimizers.Optimizer):
    """S12: Symplectic momentum with quantum-natural geodesic corrections.

    Combines the stability of symplectic integration with the efficiency
    of quantum natural gradient for superior optimization landscape
    navigation.

    Research Basis:
        Synergy between Phase 91 (SympFlow) and Phase 100 (QNG)

    Key Features:
        - Symplectic Integration: Preserves Hamiltonian structure
        - Quantum Natural Gradient: Adapts to parameter manifold curvature
        - Geodesic Corrections: Follows natural geodesics on parameter space
        - NaN/Inf Safety: Skips updates when non-finite gradients detected
        - Gradient Clipping: Optional per-variable gradient clipping

    Update Rule:
        1. p_half = p - (lr/2) * g                    (half-step momentum)
        2. p_half += w * geodesic_correction(p, QFIM) (geodesic correction)
        3. θ = θ + lr * p_half / mass                 (full-step position)
        4. p = p_half - (lr/2) * g' - friction * p    (second half-step with friction)

    Args:
        learning_rate: Learning rate (step size).
        mass: Effective mass for momentum dynamics.
        friction: Dissipation rate for momentum.
        geodesic_weight: Weight for QNG geodesic corrections.
        clipnorm: Maximum gradient norm for clipping (None to disable).
        **kwargs: Additional arguments for base optimizer.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        mass: float = 1.0,
        friction: float = 0.01,
        geodesic_weight: float = 0.1,
        clipnorm: float | None = 1.0,
        name: str = "SympFlowQNG",
        **kwargs,
    ):
        # Pass clipnorm to base class if supported, otherwise handle manually
        if "clipnorm" not in kwargs:
            kwargs["clipnorm"] = clipnorm
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)

        # Import config for defaults
        from highnoon import config as cfg

        self.mass = mass or getattr(cfg, "SYMPFLOW_MASS", 1.0)
        self.friction = friction or getattr(cfg, "SYMPFLOW_FRICTION", 0.01)
        self.geodesic_weight = geodesic_weight or getattr(cfg, "SYMPFLOW_GEODESIC_WEIGHT", 0.1)
        self._use_geodesic = getattr(cfg, "SYMPFLOW_USE_QNG_GEODESIC", True)
        self._clipnorm = clipnorm

        # Momentum slots
        self.momentums = []
        # QFIM diagonal estimates
        self.qfim_diags = []
        # NaN recovery: track consecutive NaN updates per variable
        self._nan_counts = {}

    def build(self, var_list):
        """Create momentum and QFIM slots."""
        super().build(var_list)

        self.momentums = []
        self.qfim_diags = []
        for var in var_list:
            # Momentum slot (initialized to zero)
            self.momentums.append(self.add_variable_from_reference(var, "momentum"))
            # QFIM diagonal estimate (initialized to 1.0 for stability)
            qfim_slot = self.add_variable_from_reference(var, "qfim_diag")
            qfim_slot.assign(tf.ones_like(var))
            self.qfim_diags.append(qfim_slot)

    def _compute_qfim_approx(
        self,
        var: tf.Variable,
        grad: tf.Tensor,
        var_index: int,
    ) -> tf.Tensor:
        """Approximate diagonal of Quantum Fisher Information Matrix.

        Uses EMA of squared gradients as efficient approximation.
        This is analogous to the empirical Fisher information.

        Args:
            var: Variable being optimized.
            grad: Current gradient.
            var_index: Index into QFIM slots.

        Returns:
            QFIM diagonal estimate [same shape as var].
        """
        if not self._use_geodesic:
            return tf.ones_like(grad)

        qfim = self.qfim_diags[var_index]

        # EMA update of squared gradients (empirical Fisher)
        ema_decay = 0.99
        grad_sq = tf.square(grad) + 1e-8
        qfim.assign(ema_decay * qfim + (1.0 - ema_decay) * grad_sq)

        return qfim

    def _geodesic_correction(
        self,
        momentum: tf.Tensor,
        qfim_diag: tf.Tensor,
    ) -> tf.Tensor:
        """Compute geodesic correction for manifold-aware updates.

        Uses approximate Christoffel symbols from QFIM to follow
        natural geodesics on the parameter manifold.

        The correction term is: -0.5 * (∂QFIM/∂θ) / QFIM * p²

        For diagonal approximation, this simplifies to correcting
        the momentum in regions of high curvature.

        Args:
            momentum: Current momentum tensor.
            qfim_diag: Diagonal QFIM estimate.

        Returns:
            Geodesic correction tensor.
        """
        if not self._use_geodesic:
            return tf.zeros_like(momentum)

        # Approximate Christoffel symbol (diagonal case)
        # Γ ≈ 0.5 * grad(log(F)) = 0.5 * grad(F) / F
        # We approximate grad(F) using central differences would be expensive,
        # so we use a simplified local curvature correction based on QFIM variance
        qfim_mean = tf.reduce_mean(qfim_diag) + 1e-8
        qfim_normalized = qfim_diag / qfim_mean

        # High QFIM values indicate high curvature - reduce momentum there
        # Low QFIM values indicate flat regions - allow larger steps
        curvature_factor = tf.math.log(qfim_normalized + 1e-8)

        # Geodesic correction: dampens momentum in high-curvature regions
        correction = -0.5 * curvature_factor * tf.square(momentum) / (qfim_diag + 1e-8)

        return correction

    def update_step(self, grad, var, learning_rate):
        """Symplectic update with optional geodesic correction.

        Implements leapfrog integration with geodesic modifications.
        Includes NaN/Inf safety checks and optional gradient clipping.
        """
        if grad is None:
            return

        # NaN/Inf safety check for gradients
        if isinstance(grad, tf.IndexedSlices):
            grad_finite = tf.reduce_all(tf.math.is_finite(grad.values))
        else:
            grad_finite = tf.reduce_all(tf.math.is_finite(grad))

        # Skip update if gradient is non-finite
        if not grad_finite:
            var_name = getattr(var, "name", "unknown")
            logger.warning(f"[SympFlowQNG] Skipping update for {var_name}: non-finite gradient")
            return

        lr = tf.cast(learning_rate, var.dtype.base_dtype)
        var_index = self._get_variable_index(var)
        momentum = self.momentums[var_index]

        # Handle sparse gradients
        if isinstance(grad, tf.IndexedSlices):
            # For sparse gradients, use simplified update with clipping
            grad_values = grad.values
            if self._clipnorm is not None:
                grad_norm = tf.linalg.norm(grad_values)
                grad_values = tf.cond(
                    grad_norm > self._clipnorm,
                    lambda: grad_values * (self._clipnorm / (grad_norm + 1e-8)),
                    lambda: grad_values,
                )
            m_slices = tf.gather(momentum, grad.indices)
            new_m = m_slices - lr * grad_values
            var.scatter_add(tf.IndexedSlices(lr * new_m / self.mass, grad.indices))
            m_updated = tf.tensor_scatter_nd_update(
                momentum, tf.expand_dims(grad.indices, -1), new_m * (1.0 - self.friction)
            )
            momentum.assign(m_updated)
            return

        # Dense update with symplectic integration

        # Per-variable gradient clipping if enabled
        clipped_grad = grad
        if self._clipnorm is not None:
            grad_norm = tf.linalg.norm(grad)
            clipped_grad = tf.cond(
                grad_norm > self._clipnorm,
                lambda: grad * (self._clipnorm / (grad_norm + 1e-8)),
                lambda: grad,
            )

        # Step 1: First half-step for momentum
        p_half = momentum - 0.5 * lr * clipped_grad

        # Step 2: Apply geodesic correction if enabled
        if self._use_geodesic:
            qfim = self._compute_qfim_approx(var, clipped_grad, var_index)
            geodesic_corr = self._geodesic_correction(p_half, qfim)
            p_half = p_half + self.geodesic_weight * geodesic_corr

        # Step 3: Full step for position (parameters)
        update = lr * p_half / self.mass

        # NaN safety: check update before applying
        if tf.reduce_all(tf.math.is_finite(update)):
            var.assign_add(update)
        else:
            var_name = getattr(var, "name", "unknown")
            logger.warning(f"[SympFlowQNG] Skipping update for {var_name}: non-finite update value")
            return

        # Step 4: Second half-step for momentum with friction
        # Note: We use the same gradient for simplicity (Störmer-Verlet)
        # A more accurate (but expensive) approach would recompute the gradient
        new_momentum = p_half - 0.5 * lr * clipped_grad
        new_momentum = new_momentum * (1.0 - self.friction * lr)

        # NaN safety for momentum update
        if tf.reduce_all(tf.math.is_finite(new_momentum)):
            momentum.assign(new_momentum)
        else:
            # Reset momentum to zero on NaN
            momentum.assign(tf.zeros_like(momentum))

    def get_config(self):
        """Get optimizer configuration."""
        config = super().get_config()
        config.update(
            {
                "mass": self.mass,
                "friction": self.friction,
                "geodesic_weight": self.geodesic_weight,
                "clipnorm": self._clipnorm,
            }
        )
        return config
