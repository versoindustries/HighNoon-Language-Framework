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
    - optimizers: Native optimizer implementations (SophiaG, Lion)
"""

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
from highnoon._native.ops.lib_loader import resolve_op_library

# MPS operations
from highnoon._native.ops.mps_contract import mps_contract
from highnoon._native.ops.mps_temporal import mps_temporal_scan

# Native optimizers
from highnoon._native.ops.optimizers import (
    lion_update,
    lion_update_available,
    native_optimizers_available,
    sophia_update,
    sophia_update_available,
)

# Quantum Architecture ops (Phases 26-36)
from highnoon._native.ops.quantum_ops import (  # Phase 34: Unitary Residual; Phase 30: Quantum Norm; Phase 29: Unitary Expert; Phase 26: Quantum Embedding; Phase 27: Floquet Position; Phase 33: Quantum LM Head; Phase 32: Grover QSG
    floquet_position_encoding_forward,
    grover_guided_qsg,
    grover_single_iteration,
    haar_random_key_init,
    init_floquet_angles,
    quantum_activation,
    quantum_embedding_forward,
    quantum_lm_head_forward,
    quantum_ops_available,
    rms_norm_forward,
    unitary_expert_forward,
    unitary_norm_forward,
    unitary_residual_backward,
    unitary_residual_forward,
)

# Selective scan (Mamba) operation
from highnoon._native.ops.selective_scan_op import (
    SELECTIVE_SCAN_CACHE_MAX_SEQ_LEN,
    selective_scan,
    selective_scan_available,
)

# Train step operation
from highnoon._native.ops.train_step import (
    fused_train_step,
    train_step_available,
    train_step_diagnostics,
    train_step_module,
)

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
    # Optimizers
    "native_optimizers_available",
    "sophia_update_available",
    "lion_update_available",
    "sophia_update",
    "lion_update",
    # Quantum ops (Phases 26-36)
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
]
