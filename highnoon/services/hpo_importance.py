# highnoon/services/hpo_importance.py
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

"""Hyperparameter Importance Analysis using fANOVA.

This module implements functional ANOVA (fANOVA) for analyzing which
hyperparameters contribute most to variance in model performance.

Key Features:
1. Individual hyperparameter importance scores
2. Pairwise interaction importance
3. Marginal effect curves
4. Random Forest-based variance decomposition

References:
- Hutter et al., "An Efficient Approach for Assessing Hyperparameter
  Importance" (ICML 2014)
- https://github.com/automl/fanova

Example:
    >>> analyzer = HyperparameterImportanceAnalyzer()
    >>> analyzer.fit(trial_history)
    >>> importance = analyzer.get_importance()
    >>> marginal = analyzer.get_marginal_curve("learning_rate")
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Try to import sklearn (optional dependency)
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, fANOVA analysis disabled")


@dataclass
class ParameterImportance:
    """Importance score for a single hyperparameter.

    Attributes:
        name: Hyperparameter name
        importance: Fraction of variance explained (0-1)
        rank: Rank among all hyperparameters (1 = most important)
        std: Standard deviation of importance estimate
        is_categorical: Whether this is a categorical parameter
    """

    name: str
    importance: float
    rank: int
    std: float = 0.0
    is_categorical: bool = False


@dataclass
class InteractionImportance:
    """Importance score for a pair of hyperparameters.

    Attributes:
        param1: First hyperparameter name
        param2: Second hyperparameter name
        importance: Fraction of variance explained by interaction
        is_significant: Whether interaction explains > 1% variance
    """

    param1: str
    param2: str
    importance: float
    is_significant: bool = False


@dataclass
class MarginalCurve:
    """Marginal effect curve for a hyperparameter.

    Shows the expected loss as a function of one hyperparameter,
    with all others marginalized (averaged).

    Attributes:
        param_name: Hyperparameter name
        x_values: Parameter values on x-axis
        y_mean: Mean expected loss at each x value
        y_std: Standard deviation at each x value
        is_categorical: Whether parameter is categorical
    """

    param_name: str
    x_values: list[float | str]
    y_mean: list[float]
    y_std: list[float]
    is_categorical: bool = False


@dataclass
class ImportanceResult:
    """Complete importance analysis result.

    Attributes:
        individual: Per-parameter importance scores
        interactions: Pairwise interaction importance
        total_variance: Total variance in trial losses
        explained_variance: Fraction explained by fitted model
        n_trials: Number of trials used for analysis
        param_names: List of analyzed parameter names
    """

    individual: list[ParameterImportance]
    interactions: list[InteractionImportance]
    total_variance: float
    explained_variance: float
    n_trials: int
    param_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "individual": [
                {
                    "name": p.name,
                    "importance": p.importance,
                    "rank": p.rank,
                    "std": p.std,
                    "is_categorical": p.is_categorical,
                }
                for p in self.individual
            ],
            "interactions": [
                {
                    "param1": i.param1,
                    "param2": i.param2,
                    "importance": i.importance,
                    "is_significant": i.is_significant,
                }
                for i in self.interactions
            ],
            "total_variance": self.total_variance,
            "explained_variance": self.explained_variance,
            "n_trials": self.n_trials,
            "param_names": self.param_names,
        }


class HyperparameterImportanceAnalyzer:
    """Analyzes hyperparameter importance using fANOVA.

    Uses Random Forest variance decomposition to estimate the
    contribution of each hyperparameter to variance in loss.

    Attributes:
        n_trees: Number of trees in Random Forest
        min_trials: Minimum trials needed for analysis
        max_interactions: Maximum interaction pairs to compute
    """

    def __init__(
        self,
        n_trees: int = 100,
        min_trials: int = 10,
        max_interactions: int = 10,
        seed: int = 42,
    ):
        """Initialize importance analyzer.

        Args:
            n_trees: Number of trees in Random Forest
            min_trials: Minimum trials needed for analysis
            max_interactions: Maximum interaction pairs to report
            seed: Random seed for reproducibility
        """
        self.n_trees = n_trees
        self.min_trials = min_trials
        self.max_interactions = max_interactions
        self.seed = seed

        self._forest: RandomForestRegressor | None = None
        self._encoders: dict[str, LabelEncoder] = {}
        self._param_names: list[str] = []
        self._categorical_mask: list[bool] = []
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._is_fitted: bool = False

    def fit(
        self,
        trial_configs: list[dict[str, Any]],
        trial_losses: list[float],
        param_names: list[str] | None = None,
    ) -> bool:
        """Fit the importance analyzer on trial data.

        Args:
            trial_configs: List of hyperparameter configurations
            trial_losses: Corresponding loss values (lower is better)
            param_names: Optional list of parameters to analyze
                        (default: all numeric/string params)

        Returns:
            True if fitting succeeded, False otherwise
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, cannot fit fANOVA")
            return False

        if len(trial_configs) < self.min_trials:
            logger.info(
                f"fANOVA requires at least {self.min_trials} trials, " f"got {len(trial_configs)}"
            )
            return False

        # Filter out invalid losses
        valid_indices = [
            i
            for i, loss in enumerate(trial_losses)
            if loss is not None and not np.isinf(loss) and not np.isnan(loss)
        ]

        if len(valid_indices) < self.min_trials:
            logger.info(f"Only {len(valid_indices)} valid trials, need {self.min_trials}")
            return False

        configs = [trial_configs[i] for i in valid_indices]
        losses = [trial_losses[i] for i in valid_indices]

        # Determine parameters to analyze
        if param_names is None:
            param_names = self._infer_param_names(configs)

        if not param_names:
            logger.warning("No analyzable parameters found")
            return False

        self._param_names = param_names
        logger.info(f"Analyzing importance for {len(param_names)} parameters")

        # Encode configurations to numeric matrix
        X, categorical_mask = self._encode_configs(configs, param_names)
        y = np.array(losses)

        self._X = X
        self._y = y
        self._categorical_mask = categorical_mask

        # Fit Random Forest
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._forest = RandomForestRegressor(
                n_estimators=self.n_trees,
                min_samples_leaf=3,
                max_depth=min(10, len(param_names) + 2),
                random_state=self.seed,
                n_jobs=-1,
            )
            self._forest.fit(X, y)

        self._is_fitted = True
        logger.info(f"fANOVA fitted on {len(configs)} trials")
        return True

    def _infer_param_names(self, configs: list[dict[str, Any]]) -> list[str]:
        """Infer analyzable parameter names from configs."""
        if not configs:
            return []

        # Collect all keys that appear in most configs
        key_counts: dict[str, int] = {}
        for config in configs:
            for key, value in config.items():
                # Skip internal keys
                if key.startswith("_"):
                    continue
                # Skip non-analyzable types
                if not isinstance(value, (int, float, str, bool)):
                    continue
                key_counts[key] = key_counts.get(key, 0) + 1

        # Keep keys that appear in at least 80% of configs
        threshold = len(configs) * 0.8
        return [k for k, count in key_counts.items() if count >= threshold]

    def _encode_configs(
        self,
        configs: list[dict[str, Any]],
        param_names: list[str],
    ) -> tuple[np.ndarray, list[bool]]:
        """Encode configurations to numeric matrix."""
        n_samples = len(configs)
        n_features = len(param_names)
        X = np.zeros((n_samples, n_features))
        categorical_mask = []

        for j, param in enumerate(param_names):
            values = [config.get(param) for config in configs]

            # Check if categorical
            if any(isinstance(v, str) for v in values if v is not None):
                # Encode categorical
                encoder = LabelEncoder()
                str_values = [str(v) if v is not None else "__MISSING__" for v in values]
                X[:, j] = encoder.fit_transform(str_values)
                self._encoders[param] = encoder
                categorical_mask.append(True)
            else:
                # Numeric - fill missing with median
                numeric_values = [float(v) for v in values if v is not None]
                median = np.median(numeric_values) if numeric_values else 0.0
                X[:, j] = [float(v) if v is not None else median for v in values]
                categorical_mask.append(False)

        return X, categorical_mask

    def get_importance(self) -> ImportanceResult | None:
        """Get importance scores for all parameters.

        Returns:
            ImportanceResult or None if not fitted
        """
        if not self._is_fitted or self._forest is None:
            logger.warning("Analyzer not fitted, call fit() first")
            return None

        # Get feature importances from Random Forest
        rf_importances = self._forest.feature_importances_

        # Compute individual importance using permutation-like variance decomposition
        individual = self._compute_individual_importance(rf_importances)

        # Compute pairwise interactions
        interactions = self._compute_interaction_importance()

        # Total variance in losses
        total_variance = float(np.var(self._y)) if self._y is not None else 0.0

        # Explained variance (R² of forest)
        if self._X is not None and self._y is not None:
            predictions = self._forest.predict(self._X)
            ss_res = np.sum((self._y - predictions) ** 2)
            ss_tot = np.sum((self._y - np.mean(self._y)) ** 2)
            explained_variance = 1 - (ss_res / (ss_tot + 1e-10))
        else:
            explained_variance = 0.0

        return ImportanceResult(
            individual=individual,
            interactions=interactions,
            total_variance=total_variance,
            explained_variance=float(explained_variance),
            n_trials=len(self._y) if self._y is not None else 0,
            param_names=self._param_names,
        )

    def _compute_individual_importance(
        self,
        rf_importances: np.ndarray,
    ) -> list[ParameterImportance]:
        """Compute individual parameter importance from RF importances."""
        # Normalize importances to sum to 1
        total = rf_importances.sum()
        if total > 0:
            normalized = rf_importances / total
        else:
            normalized = rf_importances

        # Create importance objects
        importances = []
        for i, param in enumerate(self._param_names):
            imp = ParameterImportance(
                name=param,
                importance=float(normalized[i]),
                rank=0,  # Set below
                std=0.0,  # Could compute via bootstrap
                is_categorical=self._categorical_mask[i],
            )
            importances.append(imp)

        # Sort by importance and set ranks
        importances.sort(key=lambda x: x.importance, reverse=True)
        for rank, imp in enumerate(importances, 1):
            imp.rank = rank

        return importances

    def _compute_interaction_importance(self) -> list[InteractionImportance]:
        """Compute pairwise interaction importance.

        Uses variance of predictions when varying two parameters together
        vs individually.
        """
        if self._forest is None or self._X is None:
            return []

        n_params = len(self._param_names)
        if n_params < 2:
            return []

        interactions = []

        # Compute all pairwise interactions
        for i in range(n_params):
            for j in range(i + 1, n_params):
                interaction_score = self._estimate_interaction(i, j)
                interaction = InteractionImportance(
                    param1=self._param_names[i],
                    param2=self._param_names[j],
                    importance=interaction_score,
                    is_significant=interaction_score > 0.01,  # > 1%
                )
                interactions.append(interaction)

        # Sort by importance and limit
        interactions.sort(key=lambda x: x.importance, reverse=True)
        return interactions[: self.max_interactions]

    def _estimate_interaction(self, idx1: int, idx2: int) -> float:
        """Estimate interaction importance between two parameters.

        Uses variance decomposition: interaction = joint variance - sum of marginal.
        This is a simplified approximation of true fANOVA.
        """
        if self._X is None or self._y is None or self._forest is None:
            return 0.0

        # Get unique values for each parameter
        values1 = np.unique(self._X[:, idx1])
        values2 = np.unique(self._X[:, idx2])

        # Limit to reasonable number of evaluations
        if len(values1) > 10:
            values1 = np.linspace(values1.min(), values1.max(), 10)
        if len(values2) > 10:
            values2 = np.linspace(values2.min(), values2.max(), 10)

        # Template: mean of all other features
        template = self._X.mean(axis=0)

        # Compute marginal variances
        marginal1 = []
        for v1 in values1:
            x = template.copy()
            x[idx1] = v1
            marginal1.append(self._forest.predict([x])[0])
        var1 = np.var(marginal1)

        marginal2 = []
        for v2 in values2:
            x = template.copy()
            x[idx2] = v2
            marginal2.append(self._forest.predict([x])[0])
        var2 = np.var(marginal2)

        # Compute joint variance
        joint = []
        for v1 in values1:
            for v2 in values2:
                x = template.copy()
                x[idx1] = v1
                x[idx2] = v2
                joint.append(self._forest.predict([x])[0])
        var_joint = np.var(joint)

        # Interaction = joint - (marginal1 + marginal2)
        # Normalize by total prediction variance
        total_var = np.var(self._forest.predict(self._X))
        if total_var < 1e-10:
            return 0.0

        interaction = max(0, var_joint - var1 - var2) / total_var
        return float(interaction)

    def get_marginal_curve(
        self,
        param_name: str,
        n_points: int = 20,
    ) -> MarginalCurve | None:
        """Get marginal effect curve for a parameter.

        Shows expected loss as function of one parameter,
        with all others marginalized.

        Args:
            param_name: Parameter to analyze
            n_points: Number of points on curve (for continuous)

        Returns:
            MarginalCurve or None if not available
        """
        if not self._is_fitted or self._forest is None:
            return None

        if param_name not in self._param_names:
            logger.warning(f"Parameter {param_name} not in analyzed parameters")
            return None

        idx = self._param_names.index(param_name)
        is_categorical = self._categorical_mask[idx]

        if is_categorical:
            return self._marginal_categorical(param_name, idx)
        else:
            return self._marginal_continuous(param_name, idx, n_points)

    def _marginal_categorical(self, param_name: str, idx: int) -> MarginalCurve:
        """Compute marginal curve for categorical parameter."""
        encoder = self._encoders.get(param_name)
        if encoder is None or self._X is None:
            return MarginalCurve(param_name, [], [], [], is_categorical=True)

        _template = self._X.mean(axis=0)  # noqa: F841 - kept for future marginal computation
        x_values = list(encoder.classes_)
        y_means = []
        y_stds = []

        for encoded_val in range(len(encoder.classes_)):
            preds = []
            for _ in range(10):  # Average over multiple templates
                x = self._X[np.random.randint(len(self._X))].copy()
                x[idx] = encoded_val
                preds.append(self._forest.predict([x])[0])
            y_means.append(float(np.mean(preds)))
            y_stds.append(float(np.std(preds)))

        return MarginalCurve(
            param_name=param_name,
            x_values=x_values,
            y_mean=y_means,
            y_std=y_stds,
            is_categorical=True,
        )

    def _marginal_continuous(
        self,
        param_name: str,
        idx: int,
        n_points: int,
    ) -> MarginalCurve:
        """Compute marginal curve for continuous parameter."""
        if self._X is None or self._forest is None:
            return MarginalCurve(param_name, [], [], [], is_categorical=False)

        # Get range from observed data
        col = self._X[:, idx]
        x_values = np.linspace(col.min(), col.max(), n_points)

        y_means = []
        y_stds = []

        for x_val in x_values:
            preds = []
            # Average over all other configurations
            for row in self._X:
                x = row.copy()
                x[idx] = x_val
                preds.append(self._forest.predict([x])[0])
            y_means.append(float(np.mean(preds)))
            y_stds.append(float(np.std(preds)))

        return MarginalCurve(
            param_name=param_name,
            x_values=[float(v) for v in x_values],
            y_mean=y_means,
            y_std=y_stds,
            is_categorical=False,
        )


__all__ = [
    "HyperparameterImportanceAnalyzer",
    "ImportanceResult",
    "ParameterImportance",
    "InteractionImportance",
    "MarginalCurve",
]
