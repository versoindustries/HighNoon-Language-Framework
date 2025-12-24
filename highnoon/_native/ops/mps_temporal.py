# highnoon/_native/ops/mps_temporal.py
# Python wrapper for MPSTemporalScan C++ Op

import tensorflow as tf

from highnoon._native.ops.lib_loader import resolve_op_library

_LIB_PATH = resolve_op_library(__file__, "_highnoon_core")
_mps_temporal_module = tf.load_op_library(_LIB_PATH)


def _mps_temporal_scan_fallback(inputs, site_weights, initial_state):
    """TensorFlow fallback for MPSTemporalScan (used for gradients)."""
    site_weights = tf.cast(site_weights, tf.float32)
    initial_state = tf.cast(initial_state, tf.float32)

    time_major_weights = tf.transpose(site_weights, [1, 0, 2, 3, 4])
    batch_size = tf.shape(site_weights)[0]
    phys_dim = tf.shape(site_weights)[3]

    def step(state, site_t):
        left_env, _, _ = state
        left = left_env[:, 0, :]
        result = tf.einsum("bc,bcdk->bdk", left, site_t)

        output = tf.reduce_mean(result, axis=2)
        norm_sq = tf.reduce_sum(tf.square(result), axis=[1, 2])
        log_prob = 0.5 * tf.math.log(norm_sq + 1e-12)

        denom = tf.sqrt(tf.maximum(norm_sq, 1e-12))
        result_normed = tf.where(
            norm_sq[:, None, None] > 1e-12,
            result / denom[:, None, None],
            result,
        )
        left_env_next = tf.reduce_mean(result_normed, axis=1, keepdims=True)
        return left_env_next, output, log_prob

    init_output = tf.zeros([batch_size, phys_dim], dtype=tf.float32)
    init_log_prob = tf.zeros([batch_size], dtype=tf.float32)
    init_state = (initial_state, init_output, init_log_prob)
    _, outputs, log_probs = tf.scan(step, time_major_weights, initializer=init_state)
    outputs = tf.transpose(outputs, [1, 0, 2])
    log_probs = tf.transpose(log_probs, [1, 0])
    return outputs, log_probs


@tf.custom_gradient
def _mps_temporal_scan_with_gradient(
    inputs,
    site_weights,
    initial_state,
    max_bond_dim,
    use_tdvp_bool,
):
    """Inner function with custom gradient for MPSTemporalScan."""
    outputs, log_probs = _mps_temporal_module.mps_temporal_scan(
        inputs,
        site_weights,
        initial_state,
        max_bond_dim,
        use_tdvp=use_tdvp_bool,
    )

    def grad_fn(grad_outputs, grad_log_probs, variables=None):
        if grad_outputs is None:
            return [None, tf.zeros_like(site_weights), tf.zeros_like(initial_state), None, None]

        with tf.GradientTape() as tape:
            tape.watch([site_weights, initial_state])
            fallback_outputs, fallback_log_probs = _mps_temporal_scan_fallback(
                inputs, site_weights, initial_state
            )
            loss = tf.reduce_sum(fallback_outputs * grad_outputs)
            if grad_log_probs is not None:
                loss += tf.reduce_sum(fallback_log_probs * grad_log_probs)

        grad_site_weights, grad_initial_state = tape.gradient(loss, [site_weights, initial_state])
        if grad_site_weights is None:
            grad_site_weights = tf.zeros_like(site_weights)
        if grad_initial_state is None:
            grad_initial_state = tf.zeros_like(initial_state)

        return [None, grad_site_weights, grad_initial_state, None, None]

    return (outputs, log_probs), grad_fn


def mps_temporal_scan(
    inputs,
    site_weights,
    initial_state,
    max_bond_dim,
    use_tdvp=False,
):
    """
    Perform efficient O(n·χ²) MPS temporal scan.

    Args:
        inputs: tf.Tensor [B, L, D]
        site_weights: tf.Tensor [B, L, chi, d, chi]
        initial_state: tf.Tensor [B, 1, chi]
        max_bond_dim: int
        use_tdvp: bool

    Returns:
        outputs: tf.Tensor [B, L, d]
        log_probs: tf.Tensor [B, L]
    """
    use_tdvp_bool = bool(use_tdvp)

    return _mps_temporal_scan_with_gradient(
        inputs,
        site_weights,
        initial_state,
        max_bond_dim,
        use_tdvp_bool,
    )
