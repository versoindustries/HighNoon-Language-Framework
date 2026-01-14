# highnoon/_native/ops/lorentzian_feature_transform.py
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

"""Python wrapper for the Lorentzian Feature Transform custom C++ operation.

Provides Lie algebra matrix exponential transformation for hyperbolic
embeddings, useful for hierarchical language structure representation.

The transformation uses the Lorentz group SO(1, D_spatial) to apply
boosts and rotations in hyperbolic space.
"""

import logging

import tensorflow as tf

from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

# --- Load the Custom Operator ---
_lorentzian_module = None
lorentzian_feature_transform_op = None
lorentzian_feature_transform_grad_op = None

try:
    _op_lib_path = resolve_op_library(__file__, "_lorentzian_feature_transform_op.so")
    _lorentzian_module = tf.load_op_library(_op_lib_path)

    if hasattr(_lorentzian_module, "lorentzian_feature_transform"):
        lorentzian_feature_transform_op = _lorentzian_module.lorentzian_feature_transform
        lorentzian_feature_transform_grad_op = getattr(
            _lorentzian_module, "lorentzian_feature_transform_grad", None
        )
        logger.info("Successfully loaded custom C++ LorentzianFeatureTransform operator.")
    else:
        raise AttributeError("lorentzian_feature_transform op not found in library")
except (tf.errors.NotFoundError, OSError, AttributeError) as e:
    logger.warning(f"Could not load the custom C++ LorentzianFeatureTransform op: {e}")
    lorentzian_feature_transform_op = None


def lorentzian_feature_transform_available() -> bool:
    """Check if the Lorentzian feature transform operation is available."""
    return lorentzian_feature_transform_op is not None


@tf.custom_gradient
def lorentzian_feature_transform(
    node_features: tf.Tensor,
    boost_vector: tf.Tensor,
    rotation_matrix_param: tf.Tensor,
) -> tf.Tensor:
    """Python wrapper for the LorentzianFeatureTransform custom C++ operator.

    Applies a Lorentz transformation (boosts and rotations) to node features
    using Lie algebra matrix exponential. This is useful for representing
    hierarchical structures in hyperbolic space (e.g., Poincaré embeddings).

    The transformation computes:
        X = [[0, a^T], [a, S]]  where S = R - R^T (skew-symmetric)
        M = exp(X)  (Lorentz transformation matrix)
        output = features @ M

    Args:
        node_features: Input tensor of shape [batch, num_nodes, D_hyp].
            D_hyp = D_spatial + 1 (hyperbolic dimension).
        boost_vector: Boost vector of shape [D_spatial].
            Controls the "velocity" of the hyperbolic transformation.
        rotation_matrix_param: Rotation parameter matrix of shape [D_spatial, D_spatial].
            Used to construct the skew-symmetric part of the Lie algebra.

    Returns:
        Transformed features tensor of shape [batch, num_nodes, D_hyp].

    Raises:
        NotImplementedError: If the C++ operation is not available.
    """
    if lorentzian_feature_transform_op is None:
        raise NotImplementedError(
            "The C++ LorentzianFeatureTransform operator could not be loaded. "
            "Rebuild ops via build_secure.sh."
        )

    transformed_features = lorentzian_feature_transform_op(
        node_features, boost_vector, rotation_matrix_param
    )

    def grad_fn(
        grad_output: tf.Tensor,
        variables: list[tf.Variable] | None = None,
    ) -> tuple[tuple[tf.Tensor, ...], list[tf.Tensor | None]]:
        """Gradient function that calls the custom C++ backward kernel."""
        if lorentzian_feature_transform_grad_op is None:
            raise NotImplementedError(
                "The C++ LorentzianFeatureTransformGrad operator is unavailable."
            )

        grads = lorentzian_feature_transform_grad_op(
            grad_output,
            node_features,
            boost_vector,
            rotation_matrix_param,
        )

        # Returns gradients for: node_features, boost_vector, rotation_matrix_param
        input_grads = (grads[0], grads[1], grads[2])

        # GRADIENT FIX: Map C++ gradient outputs to tf.Variables by name pattern
        # Instead of returning [None] * len(variables) which zeros out all gradients
        if variables is not None and len(variables) > 0:
            # grads[0]=grad_node_features, grads[1]=grad_boost_vector, grads[2]=grad_rotation_matrix
            variable_grads_list = []
            for v in variables:
                name = v.name.lower()
                if "boost" in name:
                    variable_grads_list.append(grads[1])
                elif "rotation" in name or "matrix" in name:
                    variable_grads_list.append(grads[2])
                elif "feature" in name or "node" in name:
                    variable_grads_list.append(grads[0])
                else:
                    variable_grads_list.append(None)
        else:
            variable_grads_list = []

        return input_grads, variable_grads_list

    return transformed_features, grad_fn


__all__ = [
    "lorentzian_feature_transform",
    "lorentzian_feature_transform_available",
]
