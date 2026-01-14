# highnoon/_native/ops/hd_hierarchical_block_ops.py
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

"""Python gradient registration for HDHierarchicalBlockForward C++ op.

This module registers the gradient function that connects HDHierarchicalBlockForward
to HDHierarchicalBlockBackward for automatic differentiation during training.

IMPORTANT: This module must be imported BEFORE the op is used in training,
otherwise TensorFlow will raise "gradient registry has no entry" error.
"""

import logging

import tensorflow as tf

from highnoon._native.ops.lib_loader import resolve_op_library

logger = logging.getLogger(__name__)

# Module-level op handle (singleton)
_hd_hierarchical_ops_module = None
_hd_hierarchical_ops_loaded = False


def _load_hd_hierarchical_ops():
    """Load HD hierarchical ops from consolidated binary."""
    global _hd_hierarchical_ops_module, _hd_hierarchical_ops_loaded

    if _hd_hierarchical_ops_loaded:
        return _hd_hierarchical_ops_module

    try:
        lib_path = resolve_op_library(__file__, "_highnoon_core.so")
        if lib_path is None:
            raise RuntimeError("Could not find _highnoon_core.so")

        _hd_hierarchical_ops_module = tf.load_op_library(lib_path)
        _hd_hierarchical_ops_loaded = True
        logger.info(f"HD hierarchical ops loaded from {lib_path}")

    except Exception as e:
        _hd_hierarchical_ops_loaded = True  # Mark as attempted
        logger.warning(f"Failed to load HD hierarchical ops: {e}")
        raise RuntimeError(
            "HD hierarchical native ops not available. Run ./build_secure.sh to compile."
        ) from e

    return _hd_hierarchical_ops_module


def hd_hierarchical_ops_available() -> bool:
    """Check if HD hierarchical native ops are available."""
    try:
        _load_hd_hierarchical_ops()
        return _hd_hierarchical_ops_module is not None
    except RuntimeError:
        return False


# =============================================================================
# Gradient Registration for HDHierarchicalBlockForward
# =============================================================================


@tf.RegisterGradient("HDHierarchicalBlockForward")
def _hd_hierarchical_block_forward_grad(
    op, grad_output, grad_h_final, grad_coherence, grad_next_state
):
    """Gradient registration for HDHierarchicalBlockForward.

    Connects the C++ forward op to its backward op for automatic differentiation.
    Forward op outputs: (hd_output, h_final, coherence, next_state)
    We receive gradients for all 4 outputs but only propagate grad_output.

    Forward op inputs (18 total):
        0: hd_input            [batch, seq_len, hd_dim]
        1: a_log               [state_dim]
        2: b_proj              [hd_dim, state_dim]
        3: c_proj              [hd_dim, state_dim]
        4: dt                  [seq_len, hd_dim]
        5: skip_proj           [hd_dim, hd_dim]
        6: amplitudes_real     [num_paths]
        7: amplitudes_imag     [num_paths]
        8: rotation_angles     [entanglement_depth, num_paths]
        9: level_embeddings    [hierarchical_levels + 1, hd_dim]
        10: cross_q_proj       [hd_dim, hd_dim]
        11: cross_k_proj       [hd_dim, hd_dim]
        12: cross_v_proj       [hd_dim, hd_dim]
        13: cross_o_proj       [hd_dim, hd_dim]
        14: uncertainty_trace  [batch] - not trainable
        15: prev_state         [batch, state_size] - not trainable
        16: qfm_rotation       [qfm_depth, hd_dim]
        17: qfm_bias           [qfm_depth, hd_dim]

    Returns gradients for all 18 inputs (None for non-trainable).
    """
    # Extract forward inputs
    hd_input = op.inputs[0]
    a_log = op.inputs[1]
    b_proj = op.inputs[2]
    c_proj = op.inputs[3]
    dt = op.inputs[4]
    skip_proj = op.inputs[5]
    amplitudes_real = op.inputs[6]
    amplitudes_imag = op.inputs[7]
    rotation_angles = op.inputs[8]
    level_embeddings = op.inputs[9]
    cross_q_proj = op.inputs[10]
    cross_k_proj = op.inputs[11]
    cross_v_proj = op.inputs[12]
    cross_o_proj = op.inputs[13]
    # uncertainty_trace = op.inputs[14]  # Not trainable
    # prev_state = op.inputs[15]         # Not trainable
    op.inputs[16]
    op.inputs[17]

    # Get attributes for backward op
    hd_dim = op.get_attr("hd_dim")
    hidden_dim = op.get_attr("hidden_dim")
    state_dim = op.get_attr("state_dim")
    num_paths = op.get_attr("num_paths")
    entanglement_depth = op.get_attr("entanglement_depth")
    entanglement_strength = op.get_attr("entanglement_strength")
    hierarchical_levels = op.get_attr("hierarchical_levels")
    pooling_ratio = op.get_attr("pooling_ratio")
    use_ctqw = op.get_attr("use_ctqw")
    use_cross_attention = op.get_attr("use_cross_attention")
    ctqw_time = op.get_attr("ctqw_time")
    use_quantum_cross_attention = op.get_attr("use_quantum_cross_attention")
    cross_attn_qfm_depth = op.get_attr("cross_attn_qfm_depth")
    min_chunk_size = op.get_attr("min_chunk_size")
    max_chunk_size = op.get_attr("max_chunk_size")
    boundary_threshold = op.get_attr("boundary_threshold")

    # Load ops module
    ops_module = _load_hd_hierarchical_ops()

    # Call backward op
    # Inputs to backward: grad_output + forward inputs (except state-related)
    grads = ops_module.hd_hierarchical_block_backward(
        grad_output=grad_output,
        hd_input=hd_input,
        a_log=a_log,
        b_proj=b_proj,
        c_proj=c_proj,
        dt=dt,
        skip_proj=skip_proj,
        amplitudes_real=amplitudes_real,
        amplitudes_imag=amplitudes_imag,
        rotation_angles=rotation_angles,
        level_embeddings=level_embeddings,
        cross_q_proj=cross_q_proj,
        cross_k_proj=cross_k_proj,
        cross_v_proj=cross_v_proj,
        cross_o_proj=cross_o_proj,
        hd_dim=hd_dim,
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        num_paths=num_paths,
        entanglement_depth=entanglement_depth,
        entanglement_strength=entanglement_strength,
        hierarchical_levels=hierarchical_levels,
        pooling_ratio=pooling_ratio,
        use_ctqw=use_ctqw,
        use_cross_attention=use_cross_attention,
        ctqw_time=ctqw_time,
        use_quantum_cross_attention=use_quantum_cross_attention,
        cross_attn_qfm_depth=cross_attn_qfm_depth,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        boundary_threshold=boundary_threshold,
    )

    # Backward op outputs 14 gradients:
    # 0: grad_input, 1: grad_a_log, 2: grad_b_proj, 3: grad_c_proj,
    # 4: grad_dt, 5: grad_skip, 6: grad_amp_real, 7: grad_amp_imag,
    # 8: grad_rot_angles, 9: grad_level_embed, 10: grad_cross_q,
    # 11: grad_cross_k, 12: grad_cross_v, 13: grad_cross_o

    # Return gradients for all 18 forward inputs
    # (None for non-trainable inputs: uncertainty_trace, prev_state)
    # Note: qfm_rotation and qfm_bias need gradients too but backward doesn't compute them yet
    return [
        grads[0],  # grad_hd_input
        grads[1],  # grad_a_log
        grads[2],  # grad_b_proj
        grads[3],  # grad_c_proj
        grads[4],  # grad_dt
        grads[5],  # grad_skip_proj
        grads[6],  # grad_amplitudes_real
        grads[7],  # grad_amplitudes_imag
        grads[8],  # grad_rotation_angles
        grads[9],  # grad_level_embeddings
        grads[10],  # grad_cross_q_proj
        grads[11],  # grad_cross_k_proj
        grads[12],  # grad_cross_v_proj
        grads[13],  # grad_cross_o_proj
        None,  # uncertainty_trace - not trainable
        None,  # prev_state - not trainable
        None,  # qfm_rotation - TODO: add to backward op
        None,  # qfm_bias - TODO: add to backward op
    ]


# Ensure gradient is registered when this module is imported
def register_hd_hierarchical_gradient():
    """Explicitly register the gradient (already done by decorator on import)."""
    # The decorator @tf.RegisterGradient has already registered it
    # This function is for explicit calls if needed
    pass


logger.debug("HDHierarchicalBlockForward gradient registered")
