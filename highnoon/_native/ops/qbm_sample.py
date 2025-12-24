# src/ops/qbm_sample.py
# Copyright 2025 Verso Industries
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
#
# Python wrapper for QBM sampling operator with policy gradient (REINFORCE) support.

import logging

import tensorflow as tf
from tensorflow.python.framework import ops

from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

# ==================== Load Custom C++ Operator ====================

_qbm_sample_module = None
qbm_sample_op = None
qbm_sample_grad_op = None

try:
    # lib_loader.resolve_op_library now returns _highnoon_core.so path
    _op_lib_path = resolve_op_library(__file__, "_qbm_sample_op.so")
    _qbm_sample_module = tf.load_op_library(_op_lib_path)
    # Check if the ops exist in the consolidated binary (TF converts to snake_case)
    if hasattr(_qbm_sample_module, "qbm_sample"):
        qbm_sample_op = _qbm_sample_module.qbm_sample
        qbm_sample_grad_op = getattr(_qbm_sample_module, "qbm_sample_grad", None)
        logger.info("Successfully loaded QBMSample C++ operator and gradient.")
    else:
        raise AttributeError("qbm_sample op not found in library")
except (tf.errors.NotFoundError, OSError, AttributeError) as e:
    logger.error(f"Error loading QBMSample operator: {e}")
    qbm_sample_op = None
    qbm_sample_grad_op = None


# ==================== Custom Gradient for REINFORCE ====================


@ops.RegisterGradient("QBMSample")
def _qbm_sample_gradient(
    op: tf.Operation,
    grad_expert_assignments: tf.Tensor,
    grad_sample_log_probs: tf.Tensor,
    grad_annealing_energies: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    """
    Custom gradient for QBMSample using REINFORCE (policy gradient).

    REINFORCE gradient: ∇_θ J = E[∇_θ log π(a|s) * (R - baseline)]

    Args:
        op: The forward operation
        grad_expert_assignments: Gradient w.r.t. expert_assignments (None, integer output)
        grad_sample_log_probs: Gradient w.r.t. sample_log_probs (from downstream loss)
        grad_annealing_energies: Gradient w.r.t. annealing_energies (None, diagnostic)

    Returns:
        Tuple of gradients for: energy_matrix, temperature_init, temperature_final,
        num_annealing_steps, seed
    """
    energy_matrix = op.inputs[0]
    temp_init = op.inputs[1]
    temp_final = op.inputs[2]
    _num_steps = op.inputs[3]  # noqa: F841 - reserved for future use
    _seed = op.inputs[4]  # noqa: F841 - reserved for future use

    expert_assignments = op.outputs[0]
    sample_log_probs = op.outputs[1]

    # Baseline for variance reduction: moving average of rewards
    # In practice, this would be a trainable variable updated during training
    # For now, use the mean of the incoming gradient as a simple baseline
    baseline = tf.reduce_mean(tf.stop_gradient(grad_sample_log_probs))

    if qbm_sample_grad_op is None:
        # Fallback: return zero gradients (shouldn't happen if op loaded successfully)
        return (
            tf.zeros_like(energy_matrix),
            tf.zeros_like(temp_init),
            tf.zeros_like(temp_final),
            None,  # num_steps (integer, no gradient)
            None,  # seed (integer, no gradient)
        )

    # Call custom gradient kernel
    grad_energy_matrix, grad_temp_init, grad_temp_final = qbm_sample_grad_op(
        grad_sample_log_probs=grad_sample_log_probs,
        energy_matrix=energy_matrix,
        expert_assignments=expert_assignments,
        sample_log_probs=sample_log_probs,
        baseline=baseline,
    )

    return (
        grad_energy_matrix,
        grad_temp_init,
        grad_temp_final,
        None,  # num_annealing_steps
        None,  # seed
    )


# ==================== Python Wrapper Function ====================


def qbm_sample(
    energy_matrix: tf.Tensor,
    temperature_init: float = 1.0,
    temperature_final: float = 0.1,
    num_annealing_steps: int = 100,
    seed: int = 42,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Quantum Boltzmann Machine sampling with simulated quantum annealing.

    Implements the annealing schedule:
        H(t) = (1-s(t)) * H_0 + s(t) * H_f
    where:
        - H_0: Transverse field (promotes exploration via high temperature)
        - H_f: Problem Hamiltonian (expert affinities)
        - s(t) = t / T_anneal (linear annealing schedule)

    Uses Metropolis-Hastings sampling at each annealing step.

    Args:
        energy_matrix: [batch, num_experts] tensor of expert affinity energies.
                      Lower energy = higher affinity for that expert.
        temperature_init: Initial temperature T_init (default: 1.0).
                         Higher temperature = more exploration.
        temperature_final: Final temperature T_final (default: 0.1).
                          Lower temperature = more exploitation.
        num_annealing_steps: Number of annealing iterations (default: 100).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        expert_assignments: [batch] int32 tensor of sampled expert indices.
        sample_log_probs: [batch] float32 tensor of log probabilities for REINFORCE.
        annealing_energies: [batch, num_annealing_steps] float32 tensor of energy
                            trajectory during annealing (for telemetry).

    Example:
        >>> energy_matrix = tf.constant([[1.0, 2.0, 0.5], [0.2, 1.5, 0.8]])
        >>> experts, log_probs, energies = qbm_sample(energy_matrix)
        >>> print(experts)  # e.g., [2, 0] (sampled stochastically)
    """
    if qbm_sample_op is None:
        raise RuntimeError(
            "QBMSample operator could not be loaded. "
            "Ensure the .so file is compiled via build_op.sh."
        )

    # Convert scalar inputs to tensors
    temp_init_tensor = tf.constant(temperature_init, dtype=tf.float32)
    temp_final_tensor = tf.constant(temperature_final, dtype=tf.float32)
    num_steps_tensor = tf.constant(num_annealing_steps, dtype=tf.int32)

    # Handle seed: if it's already a tensor (e.g., tf.Variable in graph mode), cast it
    # Otherwise convert Python value to tensor
    if isinstance(seed, (tf.Tensor, tf.Variable)):
        seed_tensor = tf.cast(seed, dtype=tf.int32)
    else:
        seed_tensor = tf.constant(seed, dtype=tf.int32)

    # Call forward op
    expert_assignments, sample_log_probs, annealing_energies = qbm_sample_op(
        energy_matrix=energy_matrix,
        temperature_init=temp_init_tensor,
        temperature_final=temp_final_tensor,
        num_annealing_steps=num_steps_tensor,
        seed=seed_tensor,
    )

    return expert_assignments, sample_log_probs, annealing_energies


# ==================== Utility Functions ====================


def compute_expert_entropy(expert_assignments: tf.Tensor, num_experts: int) -> tf.Tensor:
    """
    Compute Shannon entropy of expert distribution for monitoring exploration.

    Entropy = -sum_e p(e) * log p(e)

    High entropy (close to log(num_experts)) indicates good exploration.
    Low entropy indicates expert collapse.

    Args:
        expert_assignments: [batch] int32 tensor of expert indices.
        num_experts: Total number of experts.

    Returns:
        entropy: Scalar entropy value.
    """
    # Compute empirical distribution
    counts = tf.cast(tf.math.bincount(expert_assignments, minlength=num_experts), tf.float32)
    probs = counts / tf.reduce_sum(counts)

    # Entropy: -sum p * log(p)
    # Add epsilon to avoid log(0)
    log_probs = tf.math.log(probs + 1e-10)
    entropy = -tf.reduce_sum(probs * log_probs)

    return entropy


def compute_exploration_ratio(expert_assignments: tf.Tensor, num_experts: int) -> tf.Tensor:
    """
    Compute the fraction of experts that receive at least one token.

    This is a simpler metric than entropy for monitoring expert collapse.

    Args:
        expert_assignments: [batch] int32 tensor of expert indices.
        num_experts: Total number of experts.

    Returns:
        exploration_ratio: Scalar in [0, 1] indicating fraction of active experts.
    """
    unique_experts = tf.unique(expert_assignments)[0]
    num_active_experts = tf.cast(tf.size(unique_experts), tf.float32)
    total_experts = tf.cast(num_experts, tf.float32)

    return num_active_experts / total_experts
