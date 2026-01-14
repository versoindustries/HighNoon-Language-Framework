# highnoon/services/hpo_surrogate_gp.py
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

"""Deep Kernel Gaussian Process Surrogate for QAHPO.

Enterprise Enhancement Phase 1.1: Advanced surrogate models using
Gaussian Processes with neural network feature extraction (Deep Kernels).

This module provides uncertainty-aware predictions for hyperparameter
configurations, enabling information-theoretic acquisition functions
like MES (Max-value Entropy Search) and GIBBON.

Key Features:
    1. **Deep Kernel GP**: Uses MLP to map hyperparameters to latent space
       where RBF kernel operates, capturing complex correlations.

    2. **Uncertainty Quantification**: Provides mean + variance for
       acquisition functions beyond point estimates.

    3. **Ensemble Acquisition Portfolio**: Combines EI, UCB, PI with
       Hedge-based (EXP3) selection for robust optimization.

References:
    - Wilson et al., "Deep Kernel Learning" (ICML 2016)
    - Snoek et al., "Practical Bayesian Optimization" (NeurIPS 2012)
    - Hoffman et al., "Portfolio Allocation for Bayesian Optimization" (UAI 2011)

Example:
    >>> surrogate = DeepKernelGPSurrogate(input_dim=10, hidden_dims=[32, 16])
    >>> surrogate.add_observation(config, loss)
    >>> surrogate.fit()
    >>> mean, var = surrogate.predict(new_config)
    >>> acq = surrogate.acquisition(new_config, kind="ei")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# DEPENDENCY CHECK: gpytorch and botorch are optional enterprise dependencies
# =============================================================================
try:
    import gpytorch
    import torch
    import torch.nn as nn
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import ConstantMean
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.models import ExactGP

    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False
    logger.info("[HPO-GP] gpytorch not available, Deep Kernel GP disabled")

try:
    from botorch.acquisition import (
        ExpectedImprovement,
        ProbabilityOfImprovement,
        UpperConfidenceBound,
    )
    from botorch.optim import optimize_acqf

    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    logger.info("[HPO-GP] botorch not available, advanced acquisitions disabled")


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class DeepKernelGPConfig:
    """Configuration for Deep Kernel GP surrogate.

    Attributes:
        hidden_dims: Hidden layer dimensions for feature extractor.
        dropout: Dropout rate for regularization.
        learning_rate: Learning rate for GP hyperparameter optimization.
        n_epochs: Number of training epochs for GP fitting.
        min_observations: Minimum observations before fitting.
        normalize_y: Whether to normalize target values.
        use_ard: Use Automatic Relevance Determination (per-dimension lengthscales).
        noise_constraint: Lower bound on observation noise.
    """

    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    learning_rate: float = 0.01
    n_epochs: int = 100
    min_observations: int = 10
    normalize_y: bool = True
    use_ard: bool = True
    noise_constraint: float = 1e-4


# =============================================================================
# NUMPY-ONLY FALLBACK FOR WHEN GPYTORCH IS UNAVAILABLE
# =============================================================================
class SimpleGPSurrogate:
    """Simple GP surrogate using NumPy (fallback when gpytorch unavailable).

    Implements basic Gaussian Process regression with RBF kernel.
    No deep kernels, but provides uncertainty estimates.

    Attributes:
        observations: List of (config, loss) pairs.
        length_scale: RBF kernel length scale.
        noise_var: Observation noise variance.
        alpha: Regularization for kernel matrix inversion.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        noise_var: float = 0.01,
        alpha: float = 1e-6,
        min_observations: int = 5,
    ) -> None:
        """Initialize simple GP surrogate.

        Args:
            length_scale: RBF kernel length scale.
            noise_var: Observation noise variance.
            alpha: Regularization for matrix inversion.
            min_observations: Minimum observations before fitting.
        """
        self.length_scale = length_scale
        self.noise_var = noise_var
        self.alpha = alpha
        self.min_observations = min_observations

        self.observations: list[tuple[dict[str, Any], float]] = []
        self._param_names: list[str] = []
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._K_inv: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._is_fitted: bool = False

    def _encode_config(self, config: dict[str, Any]) -> np.ndarray:
        """Encode configuration to numeric vector.

        Args:
            config: Hyperparameter configuration.

        Returns:
            Numeric feature vector.
        """
        if not self._param_names:
            # Infer param names from first config
            self._param_names = [
                k
                for k, v in config.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and not k.startswith("_")
            ]

        features = []
        for name in self._param_names:
            val = config.get(name, 0)
            if isinstance(val, (int, float)):
                features.append(float(val))
            else:
                features.append(0.0)

        return np.array(features)

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix.

        Args:
            X1: First input matrix [N1, D].
            X2: Second input matrix [N2, D].

        Returns:
            Kernel matrix [N1, N2].
        """
        # Squared Euclidean distance
        dist_sq = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-0.5 * dist_sq / (self.length_scale**2))

    def add_observation(self, config: dict[str, Any], loss: float) -> None:
        """Add observation to the surrogate.

        Args:
            config: Hyperparameter configuration.
            loss: Achieved loss value.
        """
        if loss is None or not np.isfinite(loss):
            return
        self.observations.append((config.copy(), loss))
        self._is_fitted = False

    def fit(self) -> bool:
        """Fit the GP surrogate on accumulated observations.

        Returns:
            True if fitting succeeded, False if insufficient data.
        """
        if len(self.observations) < self.min_observations:
            return False

        # Encode observations
        X_list = [self._encode_config(c) for c, _ in self.observations]
        y_list = [loss for _, loss in self.observations]

        self._X = np.array(X_list)
        self._y = np.array(y_list)

        # Normalize targets
        self._y_mean = np.mean(self._y)
        self._y_std = np.std(self._y) + 1e-8
        y_norm = (self._y - self._y_mean) / self._y_std

        # Compute kernel matrix
        K = self._rbf_kernel(self._X, self._X)
        K += (self.noise_var + self.alpha) * np.eye(len(K))

        # Invert kernel matrix
        try:
            self._K_inv = np.linalg.inv(K)
            self._alpha_weights = self._K_inv @ y_norm
            self._is_fitted = True
            logger.debug(f"[SimpleGP] Fitted on {len(self.observations)} observations")
            return True
        except np.linalg.LinAlgError:
            logger.warning("[SimpleGP] Kernel matrix inversion failed")
            return False

    def predict(self, config: dict[str, Any]) -> tuple[float, float]:
        """Predict mean and variance for a configuration.

        Args:
            config: Configuration to evaluate.

        Returns:
            Tuple of (mean, variance) predictions.
        """
        if not self._is_fitted or self._X is None:
            return 0.0, 1.0

        x = self._encode_config(config).reshape(1, -1)
        k_star = self._rbf_kernel(x, self._X)
        k_star_star = self._rbf_kernel(x, x)

        # Posterior mean
        mean_norm = k_star @ self._alpha_weights
        mean = mean_norm[0] * self._y_std + self._y_mean

        # Posterior variance
        var = k_star_star[0, 0] - k_star @ self._K_inv @ k_star.T
        var = max(1e-8, var[0, 0])

        return float(mean), float(var)

    def acquisition(
        self,
        config: dict[str, Any],
        kind: str = "ei",
        xi: float = 0.01,
        kappa: float = 2.0,
    ) -> float:
        """Compute acquisition function value.

        Args:
            config: Configuration to evaluate.
            kind: Acquisition function type ("ei", "ucb", "pi").
            xi: Exploration parameter for EI/PI.
            kappa: Exploration parameter for UCB.

        Returns:
            Acquisition value (higher is better).
        """
        if not self._is_fitted:
            return 1.0  # Encourage exploration

        mean, var = self.predict(config)
        std = np.sqrt(var)

        if std < 1e-8:
            return 0.0

        # Best observed so far
        y_best = min(loss for _, loss in self.observations)

        if kind == "ei":
            # Expected Improvement (minimization)
            z = (y_best - mean - xi) / std
            ei = (y_best - mean - xi) * self._norm_cdf(z) + std * self._norm_pdf(z)
            return float(max(0, ei))
        elif kind == "ucb":
            # Lower Confidence Bound (for minimization)
            return float(-(mean - kappa * std))
        elif kind == "pi":
            # Probability of Improvement
            z = (y_best - mean - xi) / std
            return float(self._norm_cdf(z))
        else:
            return float(-mean)  # Default: greedy

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def get_statistics(self) -> dict[str, Any]:
        """Get surrogate model statistics.

        Returns:
            Dictionary with model statistics.
        """
        return {
            "type": "SimpleGP",
            "n_observations": len(self.observations),
            "is_fitted": self._is_fitted,
            "n_params": len(self._param_names),
            "length_scale": self.length_scale,
            "noise_var": self.noise_var,
        }


# =============================================================================
# DEEP KERNEL GP (REQUIRES GPYTORCH)
# =============================================================================
if GPYTORCH_AVAILABLE:

    class FeatureExtractor(nn.Module):
        """Neural network feature extractor for Deep Kernel GP.

        Maps raw hyperparameter vectors to a learned latent space where
        the RBF kernel can capture complex correlations.

        Attributes:
            layers: Sequential MLP layers with ReLU and dropout.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, ...] = (64, 32),
            dropout: float = 0.1,
        ) -> None:
            """Initialize feature extractor.

            Args:
                input_dim: Dimension of input hyperparameter vectors.
                hidden_dims: Hidden layer dimensions.
                dropout: Dropout rate between layers.
            """
            super().__init__()

            layers = []
            prev_dim = input_dim

            for h_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, h_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                prev_dim = h_dim

            self.layers = nn.Sequential(*layers)
            self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through feature extractor.

            Args:
                x: Input tensor [batch, input_dim].

            Returns:
                Latent features [batch, output_dim].
            """
            return self.layers(x)

    class DeepKernelGP(ExactGP):
        """Gaussian Process with Deep Kernel (learned feature extractor).

        Combines a neural network feature extractor with a GP to capture
        complex hyperparameter correlations while maintaining uncertainty.

        Architecture:
            Input → MLP Feature Extractor → Latent Space → RBF Kernel → GP

        Attributes:
            feature_extractor: Neural network for feature extraction.
            mean_module: GP mean function (constant).
            covar_module: GP covariance (scaled RBF).
        """

        def __init__(
            self,
            train_x: torch.Tensor,
            train_y: torch.Tensor,
            likelihood: GaussianLikelihood,
            input_dim: int,
            hidden_dims: tuple[int, ...] = (64, 32),
            dropout: float = 0.1,
            use_ard: bool = True,
        ) -> None:
            """Initialize Deep Kernel GP.

            Args:
                train_x: Training inputs [N, D].
                train_y: Training targets [N].
                likelihood: GP likelihood.
                input_dim: Input dimension.
                hidden_dims: Feature extractor hidden dimensions.
                dropout: Dropout rate.
                use_ard: Use automatic relevance determination.
            """
            super().__init__(train_x, train_y, likelihood)

            self.feature_extractor = FeatureExtractor(input_dim, hidden_dims, dropout)
            latent_dim = self.feature_extractor.output_dim

            self.mean_module = ConstantMean()

            # Use ARD (per-dimension lengthscales) if enabled
            if use_ard:
                base_kernel = RBFKernel(ard_num_dims=latent_dim)
            else:
                base_kernel = RBFKernel()

            self.covar_module = ScaleKernel(base_kernel)

        def forward(self, x: torch.Tensor) -> MultivariateNormal:
            """Forward pass computing GP posterior.

            Args:
                x: Input tensor [batch, input_dim].

            Returns:
                MultivariateNormal posterior distribution.
            """
            # Extract features
            features = self.feature_extractor(x)

            # GP components
            mean = self.mean_module(features)
            covar = self.covar_module(features)

            return MultivariateNormal(mean, covar)


class DeepKernelGPSurrogate:
    """Deep Kernel GP Surrogate wrapper for QAHPO integration.

    Provides a unified interface for adding observations, fitting,
    and computing acquisition functions compatible with the existing
    QAHPO scheduler.

    When gpytorch is available, uses DeepKernelGP. Otherwise, falls
    back to SimpleGPSurrogate (NumPy-based).

    Attributes:
        config: Deep Kernel GP configuration.
        observations: List of (config, loss) pairs.
        model: The underlying GP model.
        likelihood: GP likelihood.
    """

    def __init__(
        self,
        config: DeepKernelGPConfig | None = None,
    ) -> None:
        """Initialize Deep Kernel GP surrogate.

        Args:
            config: Configuration for the GP. Uses defaults if None.
        """
        self.config = config or DeepKernelGPConfig()
        self.observations: list[tuple[dict[str, Any], float]] = []
        self._param_names: list[str] = []
        self._is_fitted: bool = False

        # PyTorch/GPyTorch components (initialized on first fit)
        self._model: Any = None
        self._likelihood: Any = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

        # Fallback for when gpytorch unavailable
        self._simple_gp: SimpleGPSurrogate | None = None
        if not GPYTORCH_AVAILABLE:
            self._simple_gp = SimpleGPSurrogate(min_observations=self.config.min_observations)
            logger.info("[DeepKernelGP] Using SimpleGP fallback (gpytorch unavailable)")

    def _infer_param_names(self, config: dict[str, Any]) -> list[str]:
        """Infer tunable parameter names from a config.

        Args:
            config: Sample configuration.

        Returns:
            List of numeric parameter names.
        """
        return [
            k
            for k, v in config.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not k.startswith("_")
        ]

    def _encode_config(self, config: dict[str, Any]) -> np.ndarray:
        """Encode configuration to numeric vector.

        Args:
            config: Hyperparameter configuration.

        Returns:
            Numeric feature vector.
        """
        features = []
        for name in self._param_names:
            val = config.get(name, 0)
            if isinstance(val, (int, float)):
                features.append(float(val))
            else:
                features.append(0.0)
        return np.array(features, dtype=np.float32)

    def add_observation(self, config: dict[str, Any], loss: float) -> None:
        """Add observation to the surrogate.

        Args:
            config: Hyperparameter configuration.
            loss: Achieved loss value.
        """
        if loss is None or not np.isfinite(loss):
            return

        self.observations.append((config.copy(), loss))
        self._is_fitted = False

        # Also add to fallback if using it
        if self._simple_gp is not None:
            self._simple_gp.add_observation(config, loss)

    def fit(self) -> bool:
        """Fit the GP surrogate on accumulated observations.

        Returns:
            True if fitting succeeded, False if insufficient data.
        """
        if len(self.observations) < self.config.min_observations:
            return False

        # Use fallback if gpytorch unavailable
        if self._simple_gp is not None:
            return self._simple_gp.fit()

        # Infer param names from first config
        if not self._param_names:
            self._param_names = self._infer_param_names(self.observations[0][0])

        if not self._param_names:
            logger.warning("[DeepKernelGP] No numeric parameters found")
            return False

        # Encode observations
        X = np.array([self._encode_config(c) for c, _ in self.observations])
        y = np.array([loss for _, loss in self.observations])

        # Normalize targets
        self._y_mean = y.mean()
        self._y_std = y.std() + 1e-8
        y_norm = (y - self._y_mean) / self._y_std

        # Convert to PyTorch tensors
        train_x = torch.tensor(X, dtype=torch.float32)
        train_y = torch.tensor(y_norm, dtype=torch.float32)

        # Initialize GP
        self._likelihood = GaussianLikelihood()
        self._likelihood.noise_covar.register_constraint(
            "raw_noise",
            gpytorch.constraints.GreaterThan(self.config.noise_constraint),
        )

        self._model = DeepKernelGP(
            train_x,
            train_y,
            self._likelihood,
            input_dim=len(self._param_names),
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
            use_ard=self.config.use_ard,
        )

        # Training mode
        self._model.train()
        self._likelihood.train()

        # Optimizer
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.config.learning_rate,
        )
        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)

        # Training loop
        for epoch in range(self.config.n_epochs):
            optimizer.zero_grad()
            output = self._model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                logger.debug(f"[DeepKernelGP] Epoch {epoch}, MLL: {-loss.item():.4f}")

        self._is_fitted = True
        logger.info(
            f"[DeepKernelGP] Fitted on {len(self.observations)} observations, "
            f"{len(self._param_names)} parameters"
        )
        return True

    def predict(self, config: dict[str, Any]) -> tuple[float, float]:
        """Predict mean and variance for a configuration.

        Args:
            config: Configuration to evaluate.

        Returns:
            Tuple of (mean, variance) predictions.
        """
        if self._simple_gp is not None:
            return self._simple_gp.predict(config)

        if not self._is_fitted or self._model is None:
            return 0.0, 1.0

        # Encode and predict
        x = self._encode_config(config)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self._model(x_tensor)
            mean_norm = posterior.mean.item()
            var_norm = posterior.variance.item()

        # Denormalize
        mean = mean_norm * self._y_std + self._y_mean
        var = var_norm * (self._y_std**2)

        return float(mean), float(max(1e-8, var))

    def acquisition(
        self,
        config: dict[str, Any],
        kind: str = "ei",
        xi: float = 0.01,
        kappa: float = 2.0,
    ) -> float:
        """Compute acquisition function value.

        Args:
            config: Configuration to evaluate.
            kind: Acquisition type ("ei", "ucb", "pi").
            xi: Exploration parameter for EI/PI.
            kappa: Exploration parameter for UCB.

        Returns:
            Acquisition value (higher is better).
        """
        if self._simple_gp is not None:
            return self._simple_gp.acquisition(config, kind, xi, kappa)

        if not self._is_fitted:
            return 1.0

        mean, var = self.predict(config)
        std = np.sqrt(var)

        if std < 1e-8:
            return 0.0

        y_best = min(loss for _, loss in self.observations)

        if kind == "ei":
            z = (y_best - mean - xi) / std
            ei = (y_best - mean - xi) * SimpleGPSurrogate._norm_cdf(
                z
            ) + std * SimpleGPSurrogate._norm_pdf(z)
            return float(max(0, ei))
        elif kind == "ucb":
            return float(-(mean - kappa * std))
        elif kind == "pi":
            z = (y_best - mean - xi) / std
            return float(SimpleGPSurrogate._norm_cdf(z))
        else:
            return float(-mean)

    def get_statistics(self) -> dict[str, Any]:
        """Get surrogate model statistics.

        Returns:
            Dictionary with model statistics.
        """
        if self._simple_gp is not None:
            return self._simple_gp.get_statistics()

        return {
            "type": "DeepKernelGP",
            "n_observations": len(self.observations),
            "is_fitted": self._is_fitted,
            "n_params": len(self._param_names),
            "hidden_dims": self.config.hidden_dims,
            "use_ard": self.config.use_ard,
            "gpytorch_available": GPYTORCH_AVAILABLE,
            "botorch_available": BOTORCH_AVAILABLE,
        }


# =============================================================================
# ENSEMBLE ACQUISITION PORTFOLIO
# =============================================================================
@dataclass
class AcquisitionPortfolio:
    """Ensemble of acquisition functions with Hedge-based selection.

    Maintains a portfolio of acquisition functions and uses
    EXP3 (Exponential-weight algorithm for Exploration and Exploitation)
    to adaptively select the best performer.

    Attributes:
        acquisitions: List of acquisition function names.
        weights: Current weights for each acquisition.
        eta: Learning rate for weight updates.
        gamma: Exploration probability for EXP3.
        history: History of selections and rewards.
    """

    acquisitions: list[str] = field(default_factory=lambda: ["ei", "ucb", "pi"])
    weights: list[float] = field(default_factory=list)
    eta: float = 0.1
    gamma: float = 0.1
    history: list[tuple[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize weights if not provided."""
        if not self.weights:
            self.weights = [1.0 / len(self.acquisitions)] * len(self.acquisitions)

    def select_acquisition(self, rng: np.random.Generator | None = None) -> str:
        """Select acquisition function using EXP3.

        Args:
            rng: NumPy random generator.

        Returns:
            Name of selected acquisition function.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Compute probabilities with exploration
        probs = np.array(self.weights)
        probs = (1 - self.gamma) * probs / probs.sum() + self.gamma / len(probs)

        # Sample
        idx = rng.choice(len(self.acquisitions), p=probs)
        return self.acquisitions[idx]

    def update(self, selected: str, reward: float) -> None:
        """Update weights based on observed reward.

        Args:
            selected: The acquisition function that was used.
            reward: Reward signal (higher is better, e.g., -loss).
        """
        if selected not in self.acquisitions:
            return

        idx = self.acquisitions.index(selected)

        # EXP3 update
        probs = np.array(self.weights)
        probs = probs / probs.sum()

        estimated_reward = reward / max(1e-8, probs[idx])
        self.weights[idx] *= np.exp(self.eta * estimated_reward / len(self.acquisitions))

        # Normalize
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        self.history.append((selected, reward))

    def get_statistics(self) -> dict[str, Any]:
        """Get portfolio statistics.

        Returns:
            Dictionary with portfolio state.
        """
        return {
            "acquisitions": self.acquisitions,
            "weights": self.weights.copy(),
            "n_selections": len(self.history),
            "selection_counts": {
                acq: sum(1 for s, _ in self.history if s == acq) for acq in self.acquisitions
            },
        }


__all__ = [
    "DeepKernelGPConfig",
    "DeepKernelGPSurrogate",
    "SimpleGPSurrogate",
    "AcquisitionPortfolio",
    "GPYTORCH_AVAILABLE",
    "BOTORCH_AVAILABLE",
]
