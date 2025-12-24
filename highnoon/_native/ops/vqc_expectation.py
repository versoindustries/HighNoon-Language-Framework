# src/ops/vqc_expectation.py

import logging
import os
import sys

import tensorflow as tf

from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

_VQC_MODULE = None
_RUN_VQC_EXPECTATION = None
_RUN_VQC_EXPECTATION_GRAD = None
_VQC_AVAILABLE = False
_VQC_DEBUG = os.getenv("HSMN_VQC_DEBUG", "0").strip().lower() not in ("0", "false", "")

try:
    # lib_loader.resolve_op_library now returns _highnoon_core.so path
    _op_path = resolve_op_library(__file__, "_vqc_expectation_op.so")
    _VQC_MODULE = tf.load_op_library(_op_path)
    # Check if the ops exist in the consolidated binary
    if hasattr(_VQC_MODULE, "run_vqc_expectation"):
        _RUN_VQC_EXPECTATION = _VQC_MODULE.run_vqc_expectation
        _RUN_VQC_EXPECTATION_GRAD = getattr(_VQC_MODULE, "run_vqc_expectation_grad", None)
        _VQC_AVAILABLE = True
        logger.info("Successfully loaded VQC expectation C++ operator.")
    else:
        raise AttributeError("run_vqc_expectation op not found in library")
except (tf.errors.NotFoundError, OSError, AttributeError) as err:
    logger.error(
        "VQC expectation operator is unavailable: %s",
        err,
    )


def vqc_expectation_available() -> bool:
    """Returns True when the compiled VQC operator has been loaded."""
    return _VQC_AVAILABLE


@tf.custom_gradient
def run_vqc_expectation(
    data_angles: tf.Tensor,
    circuit_params: tf.Tensor,
    entangler_pairs: tf.Tensor,
    measurement_paulis: tf.Tensor,
    measurement_coeffs: tf.Tensor,
) -> tf.Tensor:
    """
    Invokes the fused VQC expectation operator. Raises NotImplementedError if the
    shared object failed to load, allowing callers to fallback to the Python
    simulator seamlessly.
    """
    if not _VQC_AVAILABLE or _RUN_VQC_EXPECTATION is None:
        raise NotImplementedError("VQC expectation operator is not available.")

    if _VQC_DEBUG:
        tf.print("[VQC PY] Invoking run_vqc_expectation raw op", output_stream=sys.stderr)
    # Directly call the C++ op in both eager and graph mode.
    # The tf.custom_gradient decorator handles the graph tracing.
    expectations = _RUN_VQC_EXPECTATION(
        data_angles=data_angles,
        circuit_params=circuit_params,
        entangler_pairs=entangler_pairs,
        measurement_paulis=measurement_paulis,
        measurement_coeffs=measurement_coeffs,
    )

    def grad_fn(grad_expectations, variables=None):
        """
        Gradient function that calls the custom C++ backward kernel.
        This signature is compatible with both standard and recompute_grad contexts.
        """
        if not _VQC_AVAILABLE or _RUN_VQC_EXPECTATION_GRAD is None:
            raise NotImplementedError("VQC expectation gradient operator is not available.")

        # Directly call the C++ grad op in both eager and graph mode.
        grad_angles, grad_params = _RUN_VQC_EXPECTATION_GRAD(
            grad_expectations=grad_expectations,
            data_angles=data_angles,
            circuit_params=circuit_params,
            entangler_pairs=entangler_pairs,
            measurement_paulis=measurement_paulis,
            measurement_coeffs=measurement_coeffs,
        )

        # Gradients for the inputs of the forward function:
        # 1. data_angles (differentiable)
        # 2. circuit_params (differentiable)
        # 3. entangler_pairs (int32, non-differentiable)
        # 4. measurement_paulis (int32, non-differentiable)
        # 5. measurement_coeffs (float, treated as constant during backprop)
        input_grads = (grad_angles, grad_params, None, None, None)
        variable_grads = [None] * len(variables) if variables is not None else []
        return input_grads, variable_grads

    return expectations, grad_fn
