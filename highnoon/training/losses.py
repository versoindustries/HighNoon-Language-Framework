# src/training/losses.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Physics-informed loss functions for HSMN-Architecture.

This module provides loss functions that enforce physical conservation laws
and constraints during neural network training. These losses are designed
to work with EHD thruster optimization, thermal field modeling, and other
physics-constrained learning tasks.

Key Features:
- Thermal conservation loss: enforces heat equation residual bounds
- Tunable weighting for multi-objective optimization
- Compatible with TensorFlow's automatic differentiation
- Float32 precision for consistency with HSMN architecture
- HPO-compatible hyperparameters for automated tuning

References:
    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
    Physics-informed neural networks: A deep learning framework for solving
    forward and inverse problems involving nonlinear partial differential equations.
    Journal of Computational Physics, 378, 686-707.
"""

import logging

import tensorflow as tf
import tensorflow_probability as tfp

logger = logging.getLogger(__name__)


def thermal_conservation_loss(
    T_pred: tf.Tensor,
    Q_joule: tf.Tensor,
    k_th: tf.Tensor | float,
    grid_spacing: tuple[float, float, float],
    lambda_thermal: float = 0.1,
    epsilon_th: float = 1e-3,
    reduction: str = "mean",
) -> tf.Tensor:
    """Compute thermal conservation loss for heat equation physics.

    This loss enforces the steady-state heat equation with Joule heating source:
        ∇²T + Q_joule/k_th ≈ 0

    The loss penalizes the residual of this equation, encouraging the predicted
    temperature field to satisfy thermal conservation laws.

    Mathematical Formulation:
        residual = |∇²T + Q_joule/k_th|
        loss = λ_thermal * mean(max(0, residual - ε_th)²)

    The loss uses a soft threshold ε_th to allow small numerical errors without
    penalty, then applies a quadratic penalty for violations beyond this threshold.

    Args:
        T_pred: Predicted temperature field [batch, nx, ny, nz] in Kelvin.
                Must have at least 3 grid points in each spatial dimension to
                compute second-order finite differences.
        Q_joule: Joule heating source term [batch, nx, ny, nz] in W/m³.
                Typically computed as Q_joule = σ|E|² where σ is electrical
                conductivity and E is electric field.
        k_th: Thermal conductivity in W/(m·K). Can be:
              - Scalar float: uniform material property
              - Tensor [batch, nx, ny, nz]: spatially-varying material
        grid_spacing: Tuple of (dx, dy, dz) spatial resolution in meters.
                     Used for finite difference computation of Laplacian.
        lambda_thermal: Loss weight coefficient (default: 0.1).
                       This hyperparameter is HPO-compatible and can be tuned
                       to balance thermal conservation against other objectives.
        epsilon_th: Residual tolerance threshold in K/m² (default: 1e-3).
                   Residuals below this value are not penalized, allowing for
                   numerical discretization errors.
        reduction: Reduction method ('mean', 'sum', 'none'). Default: 'mean'.

    Returns:
        Scalar loss tensor (float32) if reduction='mean' or 'sum',
        or tensor [batch, nx-2, ny-2, nz-2] if reduction='none'.

    Raises:
        ValueError: If T_pred or Q_joule have incompatible shapes, or if
                   grid dimensions are too small (<3 in any direction).
        TypeError: If k_th type is not float or Tensor.

    Example:
        >>> import tensorflow as tf
        >>> # Setup: 64³ grid with 1mm spacing
        >>> T_pred = tf.random.normal([4, 64, 64, 64])  # batch=4
        >>> E = tf.random.normal([4, 64, 64, 64, 3])   # electric field
        >>> sigma = 1e6  # copper conductivity (S/m)
        >>> Q_joule = sigma * tf.reduce_sum(tf.square(E), axis=-1)
        >>> k_th = 400.0  # copper thermal conductivity (W/m·K)
        >>> loss = thermal_conservation_loss(
        ...     T_pred, Q_joule, k_th,
        ...     grid_spacing=(1e-3, 1e-3, 1e-3),
        ...     lambda_thermal=0.1
        ... )
        >>> print(f"Thermal loss: {loss.numpy():.6f}")

    Notes:
        - This loss uses second-order central finite differences for Laplacian.
        - Boundary points are excluded (interior domain only) to avoid BC issues.
        - All computations are float32 for consistency with HSMN architecture.
        - Gradients flow through both T_pred and Q_joule for full differentiability.
    """
    # Input validation - accept tf.Tensor or tf.Variable
    if not (tf.is_tensor(T_pred) or isinstance(T_pred, tf.Variable)):
        raise TypeError(f"T_pred must be a TensorFlow tensor, got {type(T_pred)}")
    if not (tf.is_tensor(Q_joule) or isinstance(Q_joule, tf.Variable)):
        raise TypeError(f"Q_joule must be a TensorFlow tensor, got {type(Q_joule)}")

    if T_pred.shape.rank != 4 or Q_joule.shape.rank != 4:
        raise ValueError(
            f"T_pred and Q_joule must be 4D tensors [batch, nx, ny, nz], "
            f"got T_pred.shape={T_pred.shape}, Q_joule.shape={Q_joule.shape}"
        )

    if T_pred.shape != Q_joule.shape:
        raise ValueError(
            f"T_pred and Q_joule must have same shape, "
            f"got T_pred.shape={T_pred.shape}, Q_joule.shape={Q_joule.shape}"
        )

    batch_size, nx, ny, nz = T_pred.shape
    if nx < 3 or ny < 3 or nz < 3:
        raise ValueError(
            f"Grid dimensions must be at least 3 in each direction for finite differences, "
            f"got shape=[{batch_size}, {nx}, {ny}, {nz}]"
        )

    if not isinstance(k_th, (float, int, tf.Tensor)):
        raise TypeError(f"k_th must be float or Tensor, got {type(k_th)}")

    if isinstance(k_th, tf.Tensor):
        if k_th.shape.rank != 0 and k_th.shape != T_pred.shape:
            raise ValueError(
                f"If k_th is a tensor, it must be scalar or match T_pred.shape, "
                f"got k_th.shape={k_th.shape}"
            )

    if reduction not in ["mean", "sum", "none"]:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

    # Ensure float32 precision
    T_pred = tf.cast(T_pred, tf.float32)
    Q_joule = tf.cast(Q_joule, tf.float32)
    if isinstance(k_th, tf.Tensor):
        k_th = tf.cast(k_th, tf.float32)
    else:
        k_th = tf.constant(k_th, dtype=tf.float32)

    # Extract grid spacing
    dx, dy, dz = grid_spacing
    dx = tf.constant(dx, dtype=tf.float32)
    dy = tf.constant(dy, dtype=tf.float32)
    dz = tf.constant(dz, dtype=tf.float32)

    # Compute Laplacian: ∇²T using second-order central finite differences
    # ∇²T = ∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²
    # Interior points only (exclude boundaries)

    # Second derivatives via 3-point stencil: (T[i+1] - 2*T[i] + T[i-1]) / h²
    d2T_dx2 = (
        T_pred[:, 2:, 1:-1, 1:-1] - 2.0 * T_pred[:, 1:-1, 1:-1, 1:-1] + T_pred[:, :-2, 1:-1, 1:-1]
    ) / (dx**2)

    d2T_dy2 = (
        T_pred[:, 1:-1, 2:, 1:-1] - 2.0 * T_pred[:, 1:-1, 1:-1, 1:-1] + T_pred[:, 1:-1, :-2, 1:-1]
    ) / (dy**2)

    d2T_dz2 = (
        T_pred[:, 1:-1, 1:-1, 2:] - 2.0 * T_pred[:, 1:-1, 1:-1, 1:-1] + T_pred[:, 1:-1, 1:-1, :-2]
    ) / (dz**2)

    laplacian_T = d2T_dx2 + d2T_dy2 + d2T_dz2

    # Extract interior points of Q_joule to match Laplacian shape
    Q_interior = Q_joule[:, 1:-1, 1:-1, 1:-1]

    # Extract interior k_th if it's a field
    if k_th.shape.rank > 0:
        k_th_interior = k_th[:, 1:-1, 1:-1, 1:-1]
    else:
        k_th_interior = k_th

    # Compute heat equation residual: ∇²T + Q_joule/k_th
    # Ideal physics: this should be ≈ 0
    residual = laplacian_T + Q_interior / k_th_interior

    # Apply soft threshold: only penalize residuals above epsilon_th
    # This allows small numerical errors without penalty
    residual_magnitude = tf.abs(residual)
    excess_residual = tf.nn.relu(residual_magnitude - epsilon_th)

    # Quadratic penalty on violations
    pointwise_loss = tf.square(excess_residual)

    # Apply reduction
    if reduction == "none":
        loss = pointwise_loss
    elif reduction == "sum":
        loss = tf.reduce_sum(pointwise_loss)
    else:  # reduction == 'mean'
        loss = tf.reduce_mean(pointwise_loss)

    # Apply thermal loss weight
    loss = lambda_thermal * loss

    return loss


def thermal_boundary_loss(
    T_pred: tf.Tensor,
    T_boundary: tf.Tensor,
    boundary_mask: tf.Tensor,
    lambda_bc: float = 1.0,
    reduction: str = "mean",
) -> tf.Tensor:
    """Compute loss for enforcing Dirichlet boundary conditions on temperature.

    This loss penalizes deviations from specified boundary conditions, which is
    critical for well-posed thermal problems (e.g., fixed temperature electrodes).

    Args:
        T_pred: Predicted temperature field [batch, nx, ny, nz] in Kelvin.
        T_boundary: Target boundary temperatures [batch, nx, ny, nz] in Kelvin.
                   Only enforced at locations where boundary_mask=1.
        boundary_mask: Binary mask [batch, nx, ny, nz] indicating boundary points.
                      1 = boundary (enforce BC), 0 = interior (ignore).
        lambda_bc: Boundary condition loss weight (default: 1.0).
        reduction: Reduction method ('mean', 'sum', 'none'). Default: 'mean'.

    Returns:
        Scalar loss tensor (float32) if reduction='mean' or 'sum',
        or tensor [batch, nx, ny, nz] if reduction='none'.

    Example:
        >>> # Fixed temperature at z=0 (cold plate) and z=nz-1 (hot electrode)
        >>> boundary_mask = tf.zeros([4, 64, 64, 64])
        >>> boundary_mask = boundary_mask[:, :, :, 0].assign(1.0)    # cold plate
        >>> boundary_mask = boundary_mask[:, :, :, -1].assign(1.0)   # hot electrode
        >>> T_boundary = tf.zeros_like(T_pred)
        >>> T_boundary = T_boundary[:, :, :, 0].assign(300.0)   # 300K cold
        >>> T_boundary = T_boundary[:, :, :, -1].assign(500.0)  # 500K hot
        >>> bc_loss = thermal_boundary_loss(T_pred, T_boundary, boundary_mask)
    """
    # Input validation
    if T_pred.shape != T_boundary.shape or T_pred.shape != boundary_mask.shape:
        raise ValueError(
            f"T_pred, T_boundary, and boundary_mask must have same shape, "
            f"got {T_pred.shape}, {T_boundary.shape}, {boundary_mask.shape}"
        )

    # Ensure float32
    T_pred = tf.cast(T_pred, tf.float32)
    T_boundary = tf.cast(T_boundary, tf.float32)
    boundary_mask = tf.cast(boundary_mask, tf.float32)

    # Compute squared error at boundary points
    error = tf.square(T_pred - T_boundary)
    masked_error = error * boundary_mask

    # Apply reduction
    if reduction == "none":
        loss = masked_error
    elif reduction == "sum":
        loss = tf.reduce_sum(masked_error)
    else:  # reduction == 'mean'
        # Mean over boundary points only (avoid division by zero)
        num_boundary_points = tf.reduce_sum(boundary_mask) + 1e-12
        loss = tf.reduce_sum(masked_error) / num_boundary_points

    # Apply boundary loss weight
    loss = lambda_bc * loss

    return loss


def electromagnetic_energy_conservation_loss(
    E_pred: tf.Tensor,
    B_pred: tf.Tensor,
    J_current: tf.Tensor,
    grid_spacing: tuple[float, float, float],
    dt: float,
    epsilon_0: float = 8.854e-12,
    mu_0: float = 1.257e-6,
    lambda_em: float = 0.1,
    reduction: str = "mean",
) -> tf.Tensor:
    """Compute electromagnetic energy conservation loss (Poynting theorem).

    Enforces energy conservation in electromagnetic fields via Poynting theorem:
        ∂u_em/∂t + ∇·S + J·E = 0

    where:
        u_em = (ε₀|E|² + |B|²/μ₀)/2  (EM energy density)
        S = (E × B)/μ₀                (Poynting vector)
        J·E                            (Ohmic dissipation)

    Args:
        E_pred: Electric field [batch, nx, ny, nz, 3] in V/m.
        B_pred: Magnetic field [batch, nx, ny, nz, 3] in Tesla.
        J_current: Current density [batch, nx, ny, nz, 3] in A/m².
        grid_spacing: Tuple of (dx, dy, dz) spatial resolution in meters.
        dt: Time step in seconds.
        epsilon_0: Vacuum permittivity (F/m). Default: 8.854e-12.
        mu_0: Vacuum permeability (H/m). Default: 1.257e-6.
        lambda_em: Loss weight coefficient (default: 0.1).
        reduction: Reduction method ('mean', 'sum', 'none'). Default: 'mean'.

    Returns:
        Scalar loss tensor (float32).

    Notes:
        - This is a more advanced loss for full Maxwell solver integration.
        - Currently implements a simplified form; full Poynting theorem requires
          time derivatives which need temporal state tracking.
        - Useful for EHD problems with significant magnetic field effects.
    """
    # Input validation
    if E_pred.shape.rank != 5 or B_pred.shape.rank != 5 or J_current.shape.rank != 5:
        raise ValueError("E_pred, B_pred, J_current must be 5D tensors [batch, nx, ny, nz, 3]")

    if E_pred.shape != B_pred.shape or E_pred.shape != J_current.shape:
        raise ValueError(
            f"E_pred, B_pred, J_current must have same shape, "
            f"got {E_pred.shape}, {B_pred.shape}, {J_current.shape}"
        )

    # Ensure float32 (note: epsilon_0 and mu_0 are small, may need careful scaling)
    E_pred = tf.cast(E_pred, tf.float32)
    B_pred = tf.cast(B_pred, tf.float32)
    J_current = tf.cast(J_current, tf.float32)

    # Compute EM energy density: u_em = (ε₀|E|² + |B|²/μ₀)/2
    E_sq = tf.reduce_sum(tf.square(E_pred), axis=-1)
    B_sq = tf.reduce_sum(tf.square(B_pred), axis=-1)
    0.5 * (epsilon_0 * E_sq + B_sq / mu_0)

    # Compute Poynting vector: S = (E × B)/μ₀
    # Cross product in 3D: (E × B)_i = ε_ijk E_j B_k
    S_x = (E_pred[..., 1] * B_pred[..., 2] - E_pred[..., 2] * B_pred[..., 1]) / mu_0
    S_y = (E_pred[..., 2] * B_pred[..., 0] - E_pred[..., 0] * B_pred[..., 2]) / mu_0
    S_z = (E_pred[..., 0] * B_pred[..., 1] - E_pred[..., 1] * B_pred[..., 0]) / mu_0
    S = tf.stack([S_x, S_y, S_z], axis=-1)

    # Compute divergence of Poynting vector: ∇·S
    dx, dy, dz = grid_spacing
    dx = tf.constant(dx, dtype=tf.float32)
    dy = tf.constant(dy, dtype=tf.float32)
    dz = tf.constant(dz, dtype=tf.float32)

    dSx_dx = (S[..., 0][:, 2:, 1:-1, 1:-1] - S[..., 0][:, :-2, 1:-1, 1:-1]) / (2.0 * dx)
    dSy_dy = (S[..., 1][:, 1:-1, 2:, 1:-1] - S[..., 1][:, 1:-1, :-2, 1:-1]) / (2.0 * dy)
    dSz_dz = (S[..., 2][:, 1:-1, 1:-1, 2:] - S[..., 2][:, 1:-1, 1:-1, :-2]) / (2.0 * dz)
    div_S = dSx_dx + dSy_dy + dSz_dz

    # Compute Ohmic dissipation: J·E
    J_interior = J_current[:, 1:-1, 1:-1, 1:-1, :]
    E_interior = E_pred[:, 1:-1, 1:-1, 1:-1, :]
    ohmic_loss = tf.reduce_sum(J_interior * E_interior, axis=-1)

    # Simplified energy conservation residual (steady-state approximation):
    # ∇·S + J·E ≈ 0
    # (Full time-dependent form requires ∂u_em/∂t tracking)
    residual = div_S + ohmic_loss

    # Penalize residual
    pointwise_loss = tf.square(residual)

    # Apply reduction
    if reduction == "none":
        loss = pointwise_loss
    elif reduction == "sum":
        loss = tf.reduce_sum(pointwise_loss)
    else:  # reduction == 'mean'
        loss = tf.reduce_mean(pointwise_loss)

    # Apply weight
    loss = lambda_em * loss

    return loss


def trajectory_prediction_loss(
    y_pred: tf.Tensor,
    y_true: tf.Tensor,
    physics_constraints: bool = True,
    lambda_physics: float = 10.0,
    reduction: str = "mean",
) -> tf.Tensor:
    """Compute trajectory prediction loss for electrode degradation modeling.

    This loss function combines MSE on predicted trajectories with physics-informed
    constraints ensuring physically plausible erosion dynamics.

    Physics Constraints:
        - Monotonic erosion: dr_e/dt ≤ 0 (electrodes erode, never grow)
        - Applied as penalty term: penalize positive dr_e values

    Args:
        y_pred: Predicted trajectories [batch, horizon, 3] where features = [r_e, J, T].
        y_true: Ground truth trajectories [batch, horizon, 3].
        physics_constraints: If True, add physics-informed penalty terms.
        lambda_physics: Weight for physics constraint penalty (default: 10.0).
        reduction: Loss reduction mode ('mean', 'sum', 'none').

    Returns:
        loss: Scalar tensor (if reduction='mean' or 'sum') or tensor [batch, horizon, 3]
              (if reduction='none').

    Example:
        >>> y_pred = model(inputs)  # [32, 10, 3]
        >>> y_true = targets  # [32, 10, 3]
        >>> loss = trajectory_prediction_loss(y_pred, y_true, physics_constraints=True)
    """
    # Cast to float32
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    # MSE loss on predictions
    mse_loss = tf.reduce_mean(tf.square(y_pred - y_true))

    total_loss = mse_loss

    if physics_constraints:
        # Extract radius predictions: [batch, horizon, 3] → [batch, horizon]
        r_e_pred = y_pred[..., 0]  # First feature is electrode radius

        # Compute radius changes: dr_e = r_e(t+1) - r_e(t)
        # [batch, horizon] → [batch, horizon-1]
        dr_e = r_e_pred[:, 1:] - r_e_pred[:, :-1]

        # Penalty for growing electrodes (dr_e > 0)
        # Use ReLU to penalize only positive changes
        growth_penalty = tf.reduce_mean(tf.nn.relu(dr_e))

        # Add weighted physics penalty
        total_loss = mse_loss + lambda_physics * growth_penalty

    # Apply reduction (already handled by reduce_mean above, but maintain interface)
    if reduction == "none":
        # Return per-sample losses
        sample_losses = tf.reduce_mean(tf.square(y_pred - y_true), axis=[1, 2])
        return sample_losses
    elif reduction == "sum":
        return total_loss * tf.cast(tf.shape(y_pred)[0], tf.float32)
    else:  # reduction == 'mean'
        return total_loss


def erosion_constraint_loss(
    degradation_state: tf.Tensor,
    dt: float = 1.0,
    lambda_erosion: float = 1.0,
    reduction: str = "mean",
) -> tf.Tensor:
    """Compute erosion constraint loss for physically plausible degradation.

    This loss enforces the physics constraint that electrode radius must only
    decrease over time (electrodes erode, never grow) when using HNN sequence
    integrator for degradation trajectories.

    Mathematical Formulation:
        dr_e/dt ≤ 0  (monotonic erosion)
        loss = λ_erosion * mean(ReLU(dr_e/dt))

    The loss penalizes any positive radius changes, ensuring the HNN trajectory
    remains in the physically plausible subspace of phase space.

    Args:
        degradation_state: Degradation trajectory [batch, timesteps, state_dim] where
                          state_dim includes [r_e, T, J, oxide_thickness, ...].
                          First dimension (index 0) is electrode radius r_e.
        dt: Time step duration in seconds (default: 1.0).
           Used to compute erosion rate dr_e/dt.
        lambda_erosion: Loss weight coefficient (default: 1.0).
                       HPO-compatible hyperparameter for tuning.
        reduction: Reduction method ('mean', 'sum', 'none'). Default: 'mean'.

    Returns:
        Scalar loss tensor (float32) if reduction='mean' or 'sum',
        or tensor [batch, timesteps-1] if reduction='none'.

    Raises:
        ValueError: If degradation_state has incorrect rank (<3D) or
                   timesteps dimension is too small (<2).

    Example:
        >>> import tensorflow as tf
        >>> from highnoon._native.ops.fused_hnn_sequence import fused_hnn_sequence
        >>> # Initial degradation state: [r_e, T, J, oxide_thickness]
        >>> initial_state = tf.constant([[1.0e-3, 500.0, 1e6, 0.0]], dtype=tf.float32)
        >>> # Evolve degradation trajectory via HNN
        >>> degradation_trajectory, _, _, _, _ = fused_hnn_sequence(
        ...     sequence_input=voltage_sequence,  # [batch, timesteps, control_dim]
        ...     initial_q=initial_state[:, :2],   # [r_e, T]
        ...     initial_p=initial_state[:, 2:],   # [J, oxide_thickness]
        ...     w1=w1, b1=b1, w2=w2, b2=b2,       # HNN parameters
        ...     w3=w3, b3=b3, w_out=w_out, b_out=b_out,
        ...     evolution_time_param=tf.constant([dt])
        ... )
        >>> # Enforce erosion constraint
        >>> erosion_loss = erosion_constraint_loss(
        ...     degradation_state=degradation_trajectory,
        ...     dt=1.0,
        ...     lambda_erosion=1.0
        ... )
        >>> print(f"Erosion constraint loss: {erosion_loss.numpy():.6f}")

    Notes:
        - This loss is compatible with fused_hnn_sequence for degradation modeling.
        - HNN provides 10^-6 energy drift stability, preventing unphysical oscillations.
        - Erosion rate r_e = K * J² * exp(-E_a / (k_B * T)) for Joule heating erosion.
        - This constraint ensures trajectories stay in the physical regime.
        - Gradients flow through the HNN integrator for end-to-end training.

    References:
        Anders, A., & Anders, S. (2005). Erosion of cathode spots in vacuum arcs.
        IEEE Transactions on Plasma Science, 33(5), 1456-1464.
    """
    # Input validation
    if not (tf.is_tensor(degradation_state) or isinstance(degradation_state, tf.Variable)):
        raise TypeError(
            f"degradation_state must be a TensorFlow tensor, got {type(degradation_state)}"
        )

    if degradation_state.shape.rank != 3:
        raise ValueError(
            f"degradation_state must be 3D tensor [batch, timesteps, state_dim], "
            f"got shape={degradation_state.shape}"
        )

    batch_size, timesteps, state_dim = degradation_state.shape
    if timesteps < 2:
        raise ValueError(f"timesteps must be at least 2 to compute erosion rate, got {timesteps}")

    if reduction not in ["mean", "sum", "none"]:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

    # Ensure float32 precision
    degradation_state = tf.cast(degradation_state, tf.float32)
    dt = tf.constant(dt, dtype=tf.float32)

    # Extract electrode radius trajectory: [batch, timesteps, state_dim] → [batch, timesteps]
    r_e = degradation_state[:, :, 0]

    # Compute radius changes: dr_e = r_e(t+1) - r_e(t)
    # [batch, timesteps] → [batch, timesteps-1]
    dr_e = r_e[:, 1:] - r_e[:, :-1]

    # Compute erosion rate: dr_e/dt (should be ≤ 0)
    erosion_rate = dr_e / dt

    # Penalty for positive erosion rate (electrode growing)
    # ReLU: max(0, erosion_rate) → penalizes only violations
    violation = tf.nn.relu(erosion_rate)

    # Quadratic penalty on violations
    pointwise_loss = tf.square(violation)

    # Apply reduction
    if reduction == "none":
        loss = pointwise_loss
    elif reduction == "sum":
        loss = tf.reduce_sum(pointwise_loss)
    else:  # reduction == 'mean'
        loss = tf.reduce_mean(pointwise_loss)

    # Apply erosion loss weight
    loss = lambda_erosion * loss

    return loss


# ============================================================================
# VAE Loss Functions for Generative Emitter Design (Sprint 4)
# ============================================================================


def reconstruction_loss(
    node_features_true: tf.Tensor,
    node_features_pred: tf.Tensor,
    validity_mask: tf.Tensor,
    reduction: str = "mean",
) -> tf.Tensor:
    """Compute reconstruction loss for VAE emitter configurations.

    Measures how well the decoder reconstructs the input graph from latent space.
    Only considers valid nodes (validity_mask=1) to handle variable-length graphs.

    Args:
        node_features_true: Ground truth node features [batch, max_N, 7].
        node_features_pred: Predicted node features [batch, max_N, 7].
        validity_mask: Binary mask indicating valid nodes [batch, max_N].
        reduction: Loss reduction mode ('mean', 'sum', 'none').

    Returns:
        loss: Reconstruction loss (MSE on valid nodes only).
    """
    # Cast to float32
    node_features_true = tf.cast(node_features_true, tf.float32)
    node_features_pred = tf.cast(node_features_pred, tf.float32)
    validity_mask = tf.cast(validity_mask, tf.float32)

    # Expand mask to match node features: [batch, max_N] → [batch, max_N, 1]
    mask_expanded = tf.expand_dims(validity_mask, axis=-1)

    # Compute squared error
    squared_error = tf.square(node_features_true - node_features_pred)

    # Apply mask (only consider valid nodes)
    masked_error = squared_error * mask_expanded

    # Compute mean per sample
    num_valid_per_sample = tf.reduce_sum(validity_mask, axis=1, keepdims=True)  # [batch, 1]
    # Avoid division by zero
    num_valid_per_sample = tf.maximum(num_valid_per_sample, 1.0)

    # Sum over nodes and features, normalize by num valid nodes
    sample_loss = tf.reduce_sum(masked_error, axis=[1, 2]) / num_valid_per_sample[:, 0]

    # Apply reduction
    if reduction == "none":
        return sample_loss
    elif reduction == "sum":
        return tf.reduce_sum(sample_loss)
    else:  # reduction == 'mean'
        return tf.reduce_mean(sample_loss)


def kl_divergence_loss(mu: tf.Tensor, logvar: tf.Tensor, reduction: str = "mean") -> tf.Tensor:
    """Compute KL divergence loss for VAE latent space regularization.

    KL(q(z|x) || p(z)) where:
        q(z|x) = N(μ, σ²) is the encoder distribution
        p(z) = N(0, I) is the prior

    Formula: KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

    Args:
        mu: Latent mean [batch, latent_dim].
        logvar: Log variance log(σ²) [batch, latent_dim].
        reduction: Loss reduction mode ('mean', 'sum', 'none').

    Returns:
        loss: KL divergence loss (non-negative).
    """
    # Cast to float32
    mu = tf.cast(mu, tf.float32)
    logvar = tf.cast(logvar, tf.float32)

    # KL divergence formula
    # KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_per_dim = -0.5 * (1.0 + logvar - tf.square(mu) - tf.exp(logvar))

    # Sum over latent dimensions
    kl_per_sample = tf.reduce_sum(kl_per_dim, axis=1)

    # Apply reduction
    if reduction == "none":
        return kl_per_sample
    elif reduction == "sum":
        return tf.reduce_sum(kl_per_sample)
    else:  # reduction == 'mean'
        return tf.reduce_mean(kl_per_sample)


def physics_penalty_loss(
    node_features: tf.Tensor, validity_mask: tf.Tensor, d_min: float = 5e-3, reduction: str = "mean"
) -> tf.Tensor:
    """Compute physics constraint penalty for emitter configurations.

    Penalizes violations of physical constraints:
        1. Minimum spacing: d_ij >= d_min for all valid emitter pairs
        2. (Future) Collision detection, boundary violations, etc.

    Args:
        node_features: Node features [batch, max_N, 7] where first 3 are [x, y, z].
        validity_mask: Binary mask [batch, max_N].
        d_min: Minimum spacing between emitters (meters, default: 5mm).
        reduction: Loss reduction mode ('mean', 'sum', 'none').

    Returns:
        loss: Physics penalty (0 if no violations).
    """
    # Cast to float32
    node_features = tf.cast(node_features, tf.float32)
    validity_mask = tf.cast(validity_mask, tf.float32)

    tf.shape(node_features)[0]
    max_N = tf.shape(node_features)[1]

    # Extract positions [batch, max_N, 3]
    positions = node_features[:, :, :3]

    # Compute pairwise distances for each sample in batch
    # positions: [batch, max_N, 3] → [batch, max_N, 1, 3] and [batch, 1, max_N, 3]
    pos_i = tf.expand_dims(positions, axis=2)  # [batch, max_N, 1, 3]
    pos_j = tf.expand_dims(positions, axis=1)  # [batch, 1, max_N, 3]
    diff = pos_i - pos_j  # [batch, max_N, max_N, 3]
    distances = tf.norm(diff, axis=-1)  # [batch, max_N, max_N]

    # Mask: only consider valid pairs (i≠j, both valid)
    mask_i = tf.expand_dims(validity_mask, axis=2)  # [batch, max_N, 1]
    mask_j = tf.expand_dims(validity_mask, axis=1)  # [batch, 1, max_N]
    pair_mask = mask_i * mask_j  # [batch, max_N, max_N]

    # Exclude diagonal (self-distances)
    eye = tf.eye(max_N, dtype=tf.float32)
    eye = tf.expand_dims(eye, axis=0)  # [1, max_N, max_N]
    pair_mask = pair_mask * (1.0 - eye)

    # Compute spacing violations: max(0, d_min - distance)
    violations = tf.nn.relu(d_min - distances)  # [batch, max_N, max_N]

    # Apply mask and sum over pairs
    masked_violations = violations * pair_mask
    total_violation_per_sample = tf.reduce_sum(masked_violations, axis=[1, 2])  # [batch]

    # Normalize by number of valid pairs
    num_pairs_per_sample = tf.reduce_sum(pair_mask, axis=[1, 2])  # [batch]
    # Avoid division by zero
    num_pairs_per_sample = tf.maximum(num_pairs_per_sample, 1.0)

    penalty_per_sample = total_violation_per_sample / num_pairs_per_sample

    # Apply reduction
    if reduction == "none":
        return penalty_per_sample
    elif reduction == "sum":
        return tf.reduce_sum(penalty_per_sample)
    else:  # reduction == 'mean'
        return tf.reduce_mean(penalty_per_sample)


def vae_total_loss(
    node_features_true: tf.Tensor,
    node_features_pred: tf.Tensor,
    validity_mask: tf.Tensor,
    mu: tf.Tensor,
    logvar: tf.Tensor,
    beta: float = 1.0,
    lambda_physics: float = 10.0,
    d_min: float = 5e-3,
    reduction: str = "mean",
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Compute total VAE loss with all components.

    Total loss = reconstruction_loss + β·KL_divergence + λ_physics·physics_penalty

    Args:
        node_features_true: Ground truth [batch, max_N, 7].
        node_features_pred: Predicted [batch, max_N, 7].
        validity_mask: Binary mask [batch, max_N].
        mu: Latent mean [batch, latent_dim].
        logvar: Latent log variance [batch, latent_dim].
        beta: Weight for KL divergence (β-VAE, default: 1.0).
        lambda_physics: Weight for physics penalty (default: 10.0).
        d_min: Minimum spacing constraint (meters, default: 5mm).
        reduction: Loss reduction mode ('mean', 'sum', 'none').

    Returns:
        total_loss: Combined loss scalar.
        loss_components: Dict with individual loss terms for logging.
    """
    # Compute individual loss components
    recon_loss = reconstruction_loss(
        node_features_true, node_features_pred, validity_mask, reduction=reduction
    )
    kl_loss = kl_divergence_loss(mu, logvar, reduction=reduction)
    phys_penalty = physics_penalty_loss(
        node_features_pred, validity_mask, d_min=d_min, reduction=reduction
    )

    # Combine with weights
    total_loss = recon_loss + beta * kl_loss + lambda_physics * phys_penalty

    # Return components for logging/monitoring
    loss_components = {
        "reconstruction": recon_loss,
        "kl_divergence": kl_loss,
        "physics_penalty": phys_penalty,
        "total": total_loss,
    }

    return total_loss, loss_components


# =============================================================================
# Domain Adaptation Losses (Phase 3: Transfer Learning)
# =============================================================================


def mmd_loss(
    source_features: tf.Tensor,
    target_features: tf.Tensor,
    kernel: str = "rbf",
    bandwidth: float | None = None,
    reduction: str = "mean",
) -> tf.Tensor:
    """Compute Maximum Mean Discrepancy (MMD) for domain adaptation.

    MMD is a kernel-based distance metric between two distributions that measures
    the difference in mean embeddings in a reproducing kernel Hilbert space (RKHS).
    It's used in transfer learning to minimize distribution shift between source
    (pre-training) and target (fine-tuning) domains.

    Mathematical Formulation:
        MMD²(P, Q) = E[k(x, x')] - 2E[k(x, y)] + E[k(y, y')]
        where x, x' ~ P (source), y, y' ~ Q (target), k is the kernel

    For RBF kernel: k(x, y) = exp(-||x - y||² / (2σ²))

    Args:
        source_features: Source domain features [batch_source, feature_dim].
                        Typically extracted from pre-trained model on standard geometries.
        target_features: Target domain features [batch_target, feature_dim].
                        Extracted from same model on novel/fine-tuning geometries.
        kernel: Kernel type ('rbf', 'linear'). Default: 'rbf' (Gaussian kernel).
               RBF kernel is more flexible and commonly used for domain adaptation.
        bandwidth: RBF kernel bandwidth σ (default: None, uses median heuristic).
                  If None, automatically computed as median pairwise distance.
                  This heuristic adapts to the scale of the feature space.
        reduction: Reduction method ('mean', 'sum', 'none'). Default: 'mean'.

    Returns:
        Scalar MMD² distance (float32) if reduction='mean' or 'sum',
        or full kernel matrix if reduction='none'.

    Raises:
        ValueError: If feature dimensions don't match or batch sizes are too small.

    Example:
        >>> import tensorflow as tf
        >>> # Pre-training features (standard geometries)
        >>> source_feats = tf.random.normal([64, 128])
        >>> # Fine-tuning features (novel fractal emitters)
        >>> target_feats = tf.random.normal([32, 128])
        >>> # Compute MMD loss for domain adaptation
        >>> domain_loss = mmd_loss(source_feats, target_feats)
        >>> # Typical values: MMD² ∈ [0, 1], closer to 0 = better alignment
        >>> print(f"Domain shift: {domain_loss:.4f}")

    References:
        Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
        A kernel two-sample test. Journal of Machine Learning Research, 13(1), 723-773.

        Tzeng, E., Hoffman, J., Saenko, K., & Darrell, T. (2017).
        Adversarial discriminative domain adaptation. CVPR 2017.
    """
    # Input validation
    if source_features.shape[-1] != target_features.shape[-1]:
        raise ValueError(
            f"Feature dimensions must match: source={source_features.shape[-1]}, "
            f"target={target_features.shape[-1]}"
        )

    batch_source = tf.shape(source_features)[0]
    batch_target = tf.shape(target_features)[0]
    source_features.shape[-1]

    # Ensure float32 precision
    source_features = tf.cast(source_features, tf.float32)
    target_features = tf.cast(target_features, tf.float32)

    if kernel == "rbf":
        # Automatic bandwidth selection via median heuristic (if not provided)
        if bandwidth is None:
            # Compute pairwise distances for bandwidth estimation
            # Use subsample if batch is large to save memory
            max_samples = 1000
            if batch_source + batch_target > max_samples:
                indices_s = tf.random.shuffle(tf.range(batch_source))[: max_samples // 2]
                indices_t = tf.random.shuffle(tf.range(batch_target))[: max_samples // 2]
                sub_source = tf.gather(source_features, indices_s)
                sub_target = tf.gather(target_features, indices_t)
            else:
                sub_source = source_features
                sub_target = target_features

            # Compute all pairwise distances
            combined = tf.concat([sub_source, sub_target], axis=0)
            dists = tf.norm(tf.expand_dims(combined, 1) - tf.expand_dims(combined, 0), axis=-1)
            # Median heuristic: bandwidth = median(pairwise distances)
            bandwidth = tf.sqrt(0.5 * tfp.stats.percentile(dists, 50.0))
            # Avoid zero bandwidth
            bandwidth = tf.maximum(bandwidth, 1e-5)

        # RBF kernel: k(x, y) = exp(-||x - y||² / (2σ²))
        def rbf_kernel(x, y):
            """Compute RBF kernel matrix between x and y."""
            # x: [n, d], y: [m, d] → output: [n, m]
            dists_sq = tf.reduce_sum((tf.expand_dims(x, 1) - tf.expand_dims(y, 0)) ** 2, axis=-1)
            return tf.exp(-dists_sq / (2.0 * bandwidth**2))

        # Compute kernel matrices
        k_ss = rbf_kernel(source_features, source_features)  # [batch_s, batch_s]
        k_tt = rbf_kernel(target_features, target_features)  # [batch_t, batch_t]
        k_st = rbf_kernel(source_features, target_features)  # [batch_s, batch_t]

    elif kernel == "linear":
        # Linear kernel: k(x, y) = x^T y
        k_ss = tf.matmul(source_features, source_features, transpose_b=True)
        k_tt = tf.matmul(target_features, target_features, transpose_b=True)
        k_st = tf.matmul(source_features, target_features, transpose_b=True)

    else:
        raise ValueError(f"Unknown kernel type: {kernel}. Supported: 'rbf', 'linear'")

    # MMD² = E[k(x, x')] - 2E[k(x, y)] + E[k(y, y')]
    # Exclude diagonal for unbiased estimate (x ≠ x', y ≠ y')
    batch_s_f = tf.cast(batch_source, tf.float32)
    batch_t_f = tf.cast(batch_target, tf.float32)

    # E[k(x, x')] with x ≠ x'
    term1 = (tf.reduce_sum(k_ss) - tf.linalg.trace(k_ss)) / (batch_s_f * (batch_s_f - 1.0))

    # E[k(y, y')] with y ≠ y'
    term2 = (tf.reduce_sum(k_tt) - tf.linalg.trace(k_tt)) / (batch_t_f * (batch_t_f - 1.0))

    # -2E[k(x, y)]
    term3 = -2.0 * tf.reduce_mean(k_st)

    mmd_squared = term1 + term2 + term3

    # MMD² can be slightly negative due to finite sample bias - clip to zero
    mmd_squared = tf.maximum(mmd_squared, 0.0)

    if reduction == "mean":
        return mmd_squared
    elif reduction == "sum":
        return mmd_squared
    elif reduction == "none":
        return mmd_squared
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
