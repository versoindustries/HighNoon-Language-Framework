# highnoon/_native/ops/__init__.py
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

"""Native C++ TensorFlow operations for HighNoon Language Framework.

This module provides Python wrappers for compiled C++ TensorFlow ops.
Each wrapper loads the corresponding .so file and exposes the op for use.

NO PYTHON FALLBACKS: If a .so file cannot be loaded, the wrapper raises
an error. This is intentional to protect proprietary implementations.

Available Operations:
    - train_step: Fused C++ training step with optimizer
    - fused_qwt_tokenizer: Quantum Wavelet Tokenizer
    - fused_wavelet_encoder: Wavelet encoding for sequences
    - selective_scan: Mamba-style selective scan
    - optimizers: Native optimizer implementations (SophiaG, Lion, Adiabatic, Geodesic, SympFlow)
    - quantum_ops: Quantum architecture operations (Phases 26-36, 51-52)
    - quantum_coherence_bus_ops: Cross-block entanglement (Phases 76, 127)
    - quantum_teleport_bus_ops: State teleportation (Phase 44)
    - quantum_dropout_ops: Quantum measurement dropout (Phase 47)
    - vqem_ops: Variational quantum error mitigation (Phase 62)
    - alphaqubit_ops: Neural syndrome decoder (Phase 61)
    - entropy_regularization_ops: Von Neumann entropy (Phase 45)
    - intrinsic_plasticity_ops: Stiefel manifold ops (Phase 71)
"""

# Phase 61: AlphaQubit Decoder
from highnoon._native.ops.alphaqubit_ops import alphaqubit_decode, create_alphaqubit_weights
from highnoon._native.ops.alphaqubit_ops import ops_available as alphaqubit_ops_available

# Phase 45: Entropy Regularization
from highnoon._native.ops.entropy_regularization_ops import (
    compute_activation_covariance,
    von_neumann_entropy_loss,
)
from highnoon._native.ops.entropy_regularization_ops import ops_available as entropy_ops_available

# QWT tokenizer operation
from highnoon._native.ops.fused_qwt_tokenizer import (
    fused_qwt_tokenizer,
    fused_qwt_tokenizer_available,
    fused_qwt_tokenizer_grad_available,
    fused_qwt_tokenizer_op_path,
)

# Wavelet encoder operation
from highnoon._native.ops.fused_wavelet_encoder import (
    fused_wavelet_encoder_available,
    fused_wavelet_encoder_chunk,
)

# Phase 200+: HD Streaming Adapter
from highnoon._native.ops.hd_streaming_adapter import (
    HDStreamingAdapter,
    hd_streaming_project,
    hd_streaming_project_grad,
)

# Phase 71: Intrinsic Plasticity
from highnoon._native.ops.intrinsic_plasticity_ops import (
    cayley_parameterization,
    compute_plasticity_metric,
    enforce_unitary_constraint,
    measure_layer_plasticity,
    project_gradient_tangent,
    retract_to_manifold,
)
from highnoon._native.ops.intrinsic_plasticity_ops import ops_available as plasticity_ops_available
from highnoon._native.ops.lib_loader import resolve_op_library

# MPS operations
from highnoon._native.ops.mps_contract import mps_contract
from highnoon._native.ops.mps_temporal import mps_temporal_scan

# Native optimizers (including Phase 46, 59, 60)
from highnoon._native.ops.optimizers import (  # Phase 59: Adiabatic; Phase 60: Geodesic; Phase 46: SympFlow
    adiabatic_optimizer_available,
    adiabatic_optimizer_step,
    geodesic_optimizer_available,
    geodesic_optimizer_step,
    lion_update,
    lion_update_available,
    native_optimizers_available,
    sophia_update,
    sophia_update_available,
    sympflow_available,
    sympflow_kinetic_energy,
    sympflow_step,
)

# Phases 73/79/80/84: Quantum Advanced Ops
from highnoon._native.ops.quantum_advanced_ops import (
    compute_coherence,
    nqs_decoder,
    qcot_reason,
    waveform_attention,
)
from highnoon._native.ops.quantum_advanced_ops import ops_available as advanced_ops_available

# Phase 76/127: Quantum Coherence Bus
from highnoon._native.ops.quantum_coherence_bus_ops import (
    ops_available as coherence_bus_ops_available,
)
from highnoon._native.ops.quantum_coherence_bus_ops import (
    qcb_coherent_transfer,
    qcb_initialize,
    qcb_synchronize_phase,
    qcb_teleport_gradient,
    qcb_update_mesh,
    unified_bus_propagate_entanglement,
    unified_bus_update_strength,
)

# Phase 65/83: Quantum Crystallization
from highnoon._native.ops.quantum_crystallization_ops import (
    crystallize_memory,
    retrieve_from_crystal,
)
from highnoon._native.ops.quantum_crystallization_ops import (
    ops_available as crystallization_ops_available,
)

# Phase 47: Quantum Measurement Dropout
from highnoon._native.ops.quantum_dropout_ops import (
    entangling_dropout,
    quantum_measurement_dropout,
    soft_quantum_dropout,
)
from highnoon._native.ops.quantum_dropout_ops import ops_available as dropout_ops_available

# Quantum Architecture ops (Phases 26-36, 51-52)
from highnoon._native.ops.quantum_ops import (  # Phase 34: Unitary Residual; Phase 30: Quantum Norm; Phase 29: Unitary Expert; Phase 26: Quantum Embedding; Phase 27: Floquet Position; Phase 33: Quantum LM Head; Phase 32: Grover QSG; Phase 51: Born Rule Loss; Phase 52: Quantum Fidelity Loss
    born_rule_loss,
    born_rule_loss_available,
    floquet_position_encoding_forward,
    grover_guided_qsg,
    grover_single_iteration,
    haar_random_key_init,
    init_floquet_angles,
    quantum_activation,
    quantum_embedding_forward,
    quantum_fidelity_loss,
    quantum_fidelity_loss_available,
    quantum_lm_head_forward,
    quantum_ops_available,
    rms_norm_forward,
    unitary_expert_forward,
    unitary_norm_forward,
    unitary_residual_backward,
    unitary_residual_forward,
)

# Phase 44: Quantum Teleport Bus
from highnoon._native.ops.quantum_teleport_bus_ops import bell_measurement, quantum_teleport_state
from highnoon._native.ops.quantum_teleport_bus_ops import (
    ops_available as teleport_bus_ops_available,
)

# Selective scan (Mamba) operation
from highnoon._native.ops.selective_scan_op import (
    SELECTIVE_SCAN_CACHE_MAX_SEQ_LEN,
    selective_scan,
    selective_scan_available,
)

# Specialized Quantum Ops (Phases 50, 55-58, 64, 68, 70, 72, 78)
from highnoon._native.ops.specialized_quantum_ops import (
    majorana_position_encode,
    mpqr_reasoning,
    multi_stage_hamiltonian,
    random_natural_gradient,
    spiking_quantum_neuron,
    spini_optimizer,
    symplectic_gnn_kalman,
    td_moe_forward,
    teleport_gradients,
    topological_wavelet_attention,
)
from highnoon._native.ops.specialized_quantum_ops import ops_available as specialized_ops_available

# Train step operation
from highnoon._native.ops.train_step import (
    fused_train_step,
    train_step_available,
    train_step_diagnostics,
    train_step_module,
)

# Phase 62: VQEM Error Mitigation
from highnoon._native.ops.vqem_ops import create_vqem_params, vqem_forward, vqem_train_step
from highnoon._native.ops.vqem_ops import ops_available as vqem_ops_available

__all__ = [
    # Utility
    "resolve_op_library",
    # Train step
    "train_step_module",
    "train_step_diagnostics",
    "train_step_available",
    "fused_train_step",
    # QWT tokenizer
    "fused_qwt_tokenizer",
    "fused_qwt_tokenizer_available",
    "fused_qwt_tokenizer_grad_available",
    "fused_qwt_tokenizer_op_path",
    # Wavelet encoder
    "fused_wavelet_encoder_chunk",
    "fused_wavelet_encoder_available",
    # Selective scan
    "selective_scan",
    "selective_scan_available",
    "SELECTIVE_SCAN_CACHE_MAX_SEQ_LEN",
    # Optimizers (Legacy + Phase 46, 59, 60)
    "native_optimizers_available",
    "sophia_update_available",
    "lion_update_available",
    "sophia_update",
    "lion_update",
    "adiabatic_optimizer_step",
    "adiabatic_optimizer_available",
    "geodesic_optimizer_step",
    "geodesic_optimizer_available",
    "sympflow_step",
    "sympflow_kinetic_energy",
    "sympflow_available",
    # Quantum ops (Phases 26-36, 51-52)
    "quantum_ops_available",
    "unitary_residual_forward",
    "unitary_residual_backward",
    "unitary_norm_forward",
    "rms_norm_forward",
    "unitary_expert_forward",
    "quantum_activation",
    "quantum_embedding_forward",
    "haar_random_key_init",
    "floquet_position_encoding_forward",
    "init_floquet_angles",
    "quantum_lm_head_forward",
    "grover_guided_qsg",
    "grover_single_iteration",
    "born_rule_loss",
    "born_rule_loss_available",
    "quantum_fidelity_loss",
    "quantum_fidelity_loss_available",
    # Phase 44: Teleport Bus
    "quantum_teleport_state",
    "bell_measurement",
    "teleport_bus_ops_available",
    # Phase 45: Entropy Regularization
    "von_neumann_entropy_loss",
    "compute_activation_covariance",
    "entropy_ops_available",
    # Phase 47: Quantum Dropout
    "quantum_measurement_dropout",
    "soft_quantum_dropout",
    "entangling_dropout",
    "dropout_ops_available",
    # Phase 61: AlphaQubit
    "alphaqubit_decode",
    "create_alphaqubit_weights",
    "alphaqubit_ops_available",
    # Phase 62: VQEM
    "vqem_forward",
    "vqem_train_step",
    "create_vqem_params",
    "vqem_ops_available",
    # Phase 71: Intrinsic Plasticity
    "cayley_parameterization",
    "enforce_unitary_constraint",
    "project_gradient_tangent",
    "retract_to_manifold",
    "compute_plasticity_metric",
    "measure_layer_plasticity",
    "plasticity_ops_available",
    # Phase 76/127: Coherence Bus
    "qcb_initialize",
    "qcb_coherent_transfer",
    "qcb_teleport_gradient",
    "qcb_synchronize_phase",
    "qcb_update_mesh",
    "unified_bus_propagate_entanglement",
    "unified_bus_update_strength",
    "coherence_bus_ops_available",
    # Phase 65/83: Crystallization
    "crystallize_memory",
    "retrieve_from_crystal",
    "crystallization_ops_available",
    # Phases 73/79/80/84: Advanced
    "nqs_decoder",
    "qcot_reason",
    "waveform_attention",
    "compute_coherence",
    "advanced_ops_available",
    # Specialized (Phases 50, 55-58, 64, 68, 70, 72, 78)
    "teleport_gradients",
    "spiking_quantum_neuron",
    "majorana_position_encode",
    "td_moe_forward",
    "topological_wavelet_attention",
    "mpqr_reasoning",
    "symplectic_gnn_kalman",
    "spini_optimizer",
    "multi_stage_hamiltonian",
    "random_natural_gradient",
    "specialized_ops_available",
    # Phase 200+: HD Streaming Adapter
    "HDStreamingAdapter",
    "hd_streaming_project",
    "hd_streaming_project_grad",
]
