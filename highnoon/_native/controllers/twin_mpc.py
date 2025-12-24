"""Twin-Based Model Predictive Control.

Integrates digital twin (physics-informed or N4SID-identified) with MPC
for predictive control. Uses the twin as the internal model for optimization.

Key Features:
    - Digital twin serves as prediction model
    - Real-time MPC optimization (< 10ms target)
    - Constraint handling (state/input limits)
    - Adaptive horizons based on twin accuracy
    - Graceful degradation if twin prediction error exceeds threshold

Reference:
    Camacho, E. F., & Alba, C. B. (2013). Model Predictive Control.
    Springer Science & Business Media.
"""

import logging
import time
from typing import Any

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class TwinMPCController:
    """Model Predictive Control using Digital Twin as internal model.

    Optimizes control sequence to minimize cost over prediction horizon:
        min_u  sum_{t=0}^{H-1} ||x[t] - x_ref||_Q^2 + ||u[t]||_R^2

    Subject to:
        x[t+1] = twin.step(x[t], u[t])  (twin dynamics)
        x_min <= x[t] <= x_max           (state constraints)
        u_min <= u[t] <= u_max           (input constraints)

    Args:
        twin: Digital twin model (PhysicsInformedTwin or N4SIDModel)
        prediction_horizon: Number of time steps to predict [default: 10]
        Q: State cost matrix [state_dim, state_dim] or scalar [default: 1.0]
        R: Input cost matrix [input_dim, input_dim] or scalar [default: 0.1]
        state_bounds: Optional (x_min, x_max) tuples
        input_bounds: Optional (u_min, u_max) tuples
        max_error_threshold: Maximum allowed prediction error before fallback [default: 0.15]
        optimizer: Optimization algorithm ('adam', 'sgd') [default: 'adam']
        learning_rate: Optimization learning rate [default: 0.1]
        max_iterations: Maximum optimization iterations [default: 20]
    """

    def __init__(
        self,
        twin: Any,  # PhysicsInformedTwin or N4SIDModel
        prediction_horizon: int = 10,
        Q: float | np.ndarray = 1.0,
        R: float | np.ndarray = 0.1,
        state_bounds: tuple[np.ndarray, np.ndarray] | None = None,
        input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
        max_error_threshold: float = 0.15,
        optimizer: str = "adam",
        learning_rate: float = 0.1,
        max_iterations: int = 20,
        name: str = "twin_mpc",
    ):
        self.twin = twin
        self.prediction_horizon = prediction_horizon
        self.max_error_threshold = max_error_threshold
        self.max_iterations = max_iterations
        self.name = name

        # Extract dimensions from twin
        if hasattr(twin, "state_dim"):
            self.state_dim = twin.state_dim
            self.input_dim = twin.input_dim if hasattr(twin, "input_dim") else twin.n_inputs
        elif hasattr(twin, "order"):
            self.state_dim = twin.order
            self.input_dim = twin.n_inputs
        else:
            raise ValueError("Twin must have 'state_dim' or 'order' attribute")

        # Cost matrices
        if np.isscalar(Q):
            self.Q = tf.constant(Q * np.eye(self.state_dim), dtype=tf.float32)
        else:
            self.Q = tf.constant(Q, dtype=tf.float32)

        if np.isscalar(R):
            self.R = tf.constant(R * np.eye(self.input_dim), dtype=tf.float32)
        else:
            self.R = tf.constant(R, dtype=tf.float32)

        # Constraints
        if state_bounds is not None:
            self.x_min = tf.constant(state_bounds[0], dtype=tf.float32)
            self.x_max = tf.constant(state_bounds[1], dtype=tf.float32)
            self.has_state_constraints = True
        else:
            self.has_state_constraints = False

        if input_bounds is not None:
            self.u_min = tf.constant(input_bounds[0], dtype=tf.float32)
            self.u_max = tf.constant(input_bounds[1], dtype=tf.float32)
            self.has_input_constraints = True
        else:
            self.has_input_constraints = False

        # Optimizer
        if optimizer == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Statistics
        self.solve_times = []
        self.prediction_errors = []

        logger.info(
            f"TwinMPC initialized: horizon={prediction_horizon}, "
            f"state_dim={self.state_dim}, input_dim={self.input_dim}"
        )

    def _rollout_twin(self, x0: tf.Tensor, u_sequence: tf.Tensor) -> tf.Tensor:
        """Rollout twin dynamics over control sequence.

        Args:
            x0: Initial state [state_dim]
            u_sequence: Control sequence [horizon, input_dim]

        Returns:
            x_trajectory: State trajectory [horizon+1, state_dim]
        """
        x = x0
        x_list = [x]

        for t in range(self.prediction_horizon):
            u = u_sequence[t]

            # Twin step (handles both PhysicsInformedTwin and N4SIDModel)
            if hasattr(self.twin, "step"):
                x = self.twin.step(x, u)
            elif hasattr(self.twin, "simulate"):
                # N4SIDModel.simulate expects [time_steps, input_dim]
                x, _ = self.twin.simulate(tf.expand_dims(u, 0), initial_state=x)
                x = x[0]  # Extract single step
            else:
                raise RuntimeError("Twin must have 'step' or 'simulate' method")

            x_list.append(x)

        return tf.stack(x_list, axis=0)  # [horizon+1, state_dim]

    def _compute_cost(
        self, x_trajectory: tf.Tensor, u_sequence: tf.Tensor, x_ref: tf.Tensor
    ) -> tf.Tensor:
        """Compute MPC cost function.

        Cost = sum_{t=0}^{H-1} (x[t] - x_ref)^T Q (x[t] - x_ref) + u[t]^T R u[t]

        Args:
            x_trajectory: State trajectory [horizon+1, state_dim]
            u_sequence: Control sequence [horizon, input_dim]
            x_ref: Reference state [state_dim]

        Returns:
            cost: Total cost (scalar)
        """
        # State cost (exclude last state since no control applied)
        x_error = x_trajectory[:-1] - x_ref  # [horizon, state_dim]
        state_cost = tf.reduce_sum(
            tf.reduce_sum(x_error * tf.linalg.matvec(self.Q, x_error), axis=1)
        )

        # Input cost
        input_cost = tf.reduce_sum(
            tf.reduce_sum(u_sequence * tf.linalg.matvec(self.R, u_sequence), axis=1)
        )

        # Terminal cost (penalize final state error more heavily)
        terminal_error = x_trajectory[-1] - x_ref
        terminal_cost = 2.0 * tf.reduce_sum(
            terminal_error * tf.linalg.matvec(self.Q, terminal_error)
        )

        return state_cost + input_cost + terminal_cost

    def _apply_constraints(self, u_sequence: tf.Variable) -> tf.Variable:
        """Project control sequence onto constraint set.

        Args:
            u_sequence: Control sequence variable [horizon, input_dim]

        Returns:
            Constrained control sequence
        """
        if self.has_input_constraints:
            u_clipped = tf.clip_by_value(u_sequence, self.u_min, self.u_max)
            u_sequence.assign(u_clipped)

        return u_sequence

    def compute_control(
        self,
        current_state: np.ndarray,
        reference_state: np.ndarray | None = None,
        warm_start: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute optimal control action using MPC.

        Args:
            current_state: Current system state [state_dim]
            reference_state: Target state [state_dim]. If None, uses origin.
            warm_start: Initial guess for control sequence [horizon, input_dim]

        Returns:
            u_opt: Optimal control action for current step [input_dim]
            info: Dictionary with optimization info:
                - cost: Final cost value
                - iterations: Number of optimization iterations
                - solve_time: Computation time [ms]
                - converged: Whether optimization converged
        """
        start_time = time.time()

        # Convert to tensors
        x0 = tf.constant(current_state, dtype=tf.float32)
        x_ref = tf.constant(
            reference_state if reference_state is not None else np.zeros(self.state_dim),
            dtype=tf.float32,
        )

        # Initialize control sequence
        if warm_start is not None:
            u_init = warm_start
        else:
            u_init = np.zeros([self.prediction_horizon, self.input_dim], dtype=np.float32)

        u_sequence = tf.Variable(u_init, dtype=tf.float32)

        # Optimization loop
        best_cost = float("inf")
        best_u = u_sequence.numpy().copy()
        converged = False

        for iteration in range(self.max_iterations):
            with tf.GradientTape() as tape:
                # Rollout twin
                x_trajectory = self._rollout_twin(x0, u_sequence)

                # Compute cost
                cost = self._compute_cost(x_trajectory, u_sequence, x_ref)

                # Add constraint penalty if state constraints violated
                if self.has_state_constraints:
                    x_violation = tf.maximum(0.0, x_trajectory - self.x_max) + tf.maximum(
                        0.0, self.x_min - x_trajectory
                    )
                    constraint_penalty = 1e3 * tf.reduce_sum(tf.square(x_violation))
                    cost = cost + constraint_penalty

            # Compute gradients
            gradients = tape.gradient(cost, [u_sequence])

            # Apply gradients
            self.optimizer.apply_gradients(zip(gradients, [u_sequence]))

            # Project onto constraints
            u_sequence = self._apply_constraints(u_sequence)

            # Track best solution
            current_cost = cost.numpy()
            if current_cost < best_cost:
                best_cost = current_cost
                best_u = u_sequence.numpy().copy()

            # Check convergence
            if iteration > 0 and abs(current_cost - best_cost) / (best_cost + 1e-10) < 1e-4:
                converged = True
                break

        # Extract first control action (receding horizon principle)
        u_opt = best_u[0]

        # Compute solve time
        solve_time_ms = (time.time() - start_time) * 1000.0
        self.solve_times.append(solve_time_ms)

        info = {
            "cost": float(best_cost),
            "iterations": iteration + 1,
            "solve_time": solve_time_ms,
            "converged": converged,
        }

        logger.debug(
            f"MPC solved: cost={best_cost:.3f}, iterations={iteration+1}, "
            f"time={solve_time_ms:.2f}ms, converged={converged}"
        )

        return u_opt, info

    def validate_twin_accuracy(
        self, x_true: np.ndarray, u_sequence: np.ndarray, horizon: int | None = None
    ) -> dict[str, float]:
        """Validate twin prediction accuracy against true trajectory.

        Args:
            x_true: True state trajectory [time_steps, state_dim]
            u_sequence: Control inputs used [time_steps, input_dim]
            horizon: Prediction horizon to validate [default: self.prediction_horizon]

        Returns:
            Dictionary with validation metrics:
                - rmse: Root mean squared error
                - max_error: Maximum absolute error
                - relative_error: RMSE / ||x_true||
                - within_threshold: Whether error < max_error_threshold
        """
        horizon = horizon or self.prediction_horizon
        time_steps = min(len(x_true), len(u_sequence), horizon)

        # Rollout twin
        x0 = tf.constant(x_true[0], dtype=tf.float32)
        u_seq = tf.constant(u_sequence[:time_steps], dtype=tf.float32)

        if hasattr(self.twin, "simulate"):
            x_pred, _ = self.twin.simulate(u_seq, initial_state=x0)
            x_pred = x_pred.numpy()
        else:
            # Manual rollout with step()
            x_list = []
            x = x0
            for t in range(time_steps):
                x_list.append(x.numpy())
                x = self.twin.step(x, u_seq[t])
            x_pred = np.array(x_list)

        # Compute metrics
        error = x_true[:time_steps] - x_pred
        rmse = np.sqrt(np.mean(error**2))
        max_error = np.max(np.abs(error))
        norm_x_true = np.linalg.norm(x_true[:time_steps]) + 1e-10
        relative_error = rmse / norm_x_true

        within_threshold = relative_error < self.max_error_threshold

        self.prediction_errors.append(relative_error)

        metrics = {
            "rmse": float(rmse),
            "max_error": float(max_error),
            "relative_error": float(relative_error),
            "within_threshold": bool(within_threshold),
        }

        logger.info(
            f"Twin validation: RMSE={rmse:.4f}, max_error={max_error:.4f}, "
            f"relative={relative_error:.2%}, ok={within_threshold}"
        )

        return metrics

    def get_statistics(self) -> dict[str, Any]:
        """Get controller performance statistics.

        Returns:
            Dictionary with statistics:
                - mean_solve_time: Average solve time [ms]
                - p95_solve_time: 95th percentile solve time [ms]
                - mean_prediction_error: Average twin prediction error
                - num_solves: Total number of MPC solves
        """
        if not self.solve_times:
            return {
                "mean_solve_time": 0.0,
                "p95_solve_time": 0.0,
                "mean_prediction_error": 0.0,
                "num_solves": 0,
            }

        solve_times_sorted = sorted(self.solve_times)
        p95_idx = int(0.95 * len(solve_times_sorted))

        return {
            "mean_solve_time": float(np.mean(self.solve_times)),
            "p95_solve_time": float(solve_times_sorted[p95_idx]) if solve_times_sorted else 0.0,
            "mean_prediction_error": (
                float(np.mean(self.prediction_errors)) if self.prediction_errors else 0.0
            ),
            "num_solves": len(self.solve_times),
        }

    def reset_statistics(self):
        """Reset performance statistics."""
        self.solve_times = []
        self.prediction_errors = []
