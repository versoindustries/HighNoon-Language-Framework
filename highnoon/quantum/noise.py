# src/quantum/noise.py
#
# Noise models and mitigation utilities for hybrid quantum-classical layers.

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf


class NoiseModel:
    """
    Base interface for quantum noise models.
    """

    def apply(self, expectation: tf.Tensor, scale: float = 1.0) -> tf.Tensor:
        raise NotImplementedError


class DepolarizingNoiseModel(NoiseModel):
    """
    Simple depolarizing noise channel. The expectation value shrinks towards zero
    as the depolarizing probability increases.
    """

    def __init__(self, base_probability: float = 0.01):
        if base_probability < 0.0:
            raise ValueError("Depolarizing probability must be non-negative.")
        self.base_probability = base_probability

    def apply(self, expectation: tf.Tensor, scale: float = 1.0) -> tf.Tensor:
        probability = tf.clip_by_value(self.base_probability * scale, 0.0, 1.0)
        return (1.0 - probability) * expectation


class AmplitudeDampingNoiseModel(NoiseModel):
    """
    Amplitude damping channel that biases the state towards |0>.
    """

    def __init__(self, base_damping: float = 0.01):
        if base_damping < 0.0:
            raise ValueError("Amplitude damping parameter must be non-negative.")
        self.base_damping = base_damping

    def apply(self, expectation: tf.Tensor, scale: float = 1.0) -> tf.Tensor:
        gamma = tf.clip_by_value(self.base_damping * scale, 0.0, 1.0)
        # The expectation of Pauli-Z drifts towards +1 under amplitude damping.
        return (1.0 - gamma) * expectation + gamma


class PhaseDampingNoiseModel(NoiseModel):
    """
    Pure dephasing channel that exponentially suppresses coherences without
    introducing amplitude damping.
    """

    def __init__(self, base_dephasing: float = 0.01):
        if base_dephasing < 0.0:
            raise ValueError("Phase damping parameter must be non-negative.")
        self.base_dephasing = base_dephasing

    def apply(self, expectation: tf.Tensor, scale: float = 1.0) -> tf.Tensor:
        gamma = tf.clip_by_value(self.base_dephasing * scale, 0.0, 5.0)
        return tf.exp(-gamma) * expectation


class MeasurementErrorMitigator(NoiseModel):
    """
    Simple readout-error mitigation that rescales expectations by an effective
    assignment fidelity factor.
    """

    def __init__(self, assignment_error: float = 0.02):
        if assignment_error < 0.0 or assignment_error >= 0.5:
            raise ValueError("Assignment error must lie in [0, 0.5).")
        self.assignment_error = assignment_error

    def apply(self, expectation: tf.Tensor, scale: float = 1.0) -> tf.Tensor:
        error = tf.clip_by_value(self.assignment_error * scale, 0.0, 0.49)
        correction = 1.0 - 2.0 * error
        safe_correction = tf.where(
            tf.less(correction, 1e-3),
            tf.ones_like(correction),
            correction,
        )
        return tf.clip_by_value(expectation / safe_correction, -1.0, 1.0)


class ZeroNoiseExtrapolator:
    """
    Implements a lightweight Richardson extrapolation to estimate the zero-noise
    expectation value given observations at multiple scaled noise levels.
    """

    def __init__(self, scales: Sequence[float]):
        if not scales:
            raise ValueError("At least one scale must be provided for extrapolation.")
        self.scales = tuple(float(s) for s in scales)

    def extrapolate(self, noisy_values: tf.Tensor) -> tf.Tensor:
        """
        Args:
            noisy_values: Tensor of shape [num_scales, batch] containing the measured
                          expectation values at the configured noise scales.

        Returns:
            Tensor of shape [batch] with the extrapolated zero-noise estimate.
        """
        if len(self.scales) == 1:
            return noisy_values[0]

        scales_tensor = tf.convert_to_tensor(self.scales, dtype=tf.float32)
        y = tf.convert_to_tensor(noisy_values, dtype=tf.float32)

        available_scales = tf.shape(scales_tensor)[0]
        y_shape = tf.shape(y)
        y_rank = tf.rank(y)

        matching_axes = tf.where(tf.equal(y_shape, available_scales))
        has_matching_axis = tf.greater(tf.shape(matching_axes)[0], 0)

        def _move_axis_to_front(axis, tensor):
            axis = tf.cast(axis, tf.int32)

            def _transpose():
                before = tf.range(axis)
                after = tf.range(axis + 1, y_rank)
                perm = tf.concat([[axis], before, after], axis=0)
                return tf.transpose(tensor, perm=perm)

            return tf.cond(tf.equal(axis, 0), lambda: tensor, _transpose)

        def _align_with_matching_axis():
            axis = matching_axes[0, 0]
            return _move_axis_to_front(axis, y)

        y_aligned = tf.cond(
            has_matching_axis,
            _align_with_matching_axis,
            lambda: y,
        )

        effective_num_scales = tf.shape(y_aligned)[0]
        can_extrapolate = tf.logical_and(
            tf.greater_equal(effective_num_scales, 2),
            tf.less_equal(effective_num_scales, available_scales),
        )

        def _return_noisy_baseline():
            return tf.gather(y_aligned, 0, axis=0)

        def _perform_extrapolation():
            scales_effective = scales_tensor[:effective_num_scales]
            ones = tf.ones_like(scales_effective)
            design_matrix = tf.stack([ones, scales_effective], axis=1)

            reshape_shape = tf.concat(
                [tf.reshape(effective_num_scales, [1]), tf.constant([-1], dtype=tf.int32)],
                axis=0,
            )
            y_flat = tf.reshape(y_aligned, reshape_shape)

            solution = tf.linalg.lstsq(design_matrix, y_flat, fast=True)
            intercept = solution[0]
            return tf.reshape(intercept, tf.shape(y_aligned)[1:])

        return tf.cond(
            can_extrapolate,
            _perform_extrapolation,
            _return_noisy_baseline,
        )
