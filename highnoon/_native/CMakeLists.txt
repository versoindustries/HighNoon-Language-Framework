# highnoon/_native/CMakeLists.txt
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# CMake build configuration for HighNoon Language Framework native operations.
# This builds a single consolidated binary (_highnoon_core.so) with all ops
# statically linked, enabling LTO optimizations and security hardening.
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release -DPRODUCTION_BUILD=ON
#   cmake --build . --parallel
#
# Or use the build_secure.sh script for fully hardened builds.

cmake_minimum_required(VERSION 3.18)
project(highnoon_core VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# C++ Standard Configuration
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# Build Options
# =============================================================================
option(ENABLE_LTO "Enable Link-Time Optimization for better inlining" OFF)
option(ENABLE_OBFUSCATION "Enable symbol and control flow obfuscation" ON)
option(STRIP_SYMBOLS "Strip all symbols from release binary" ON)
option(PRODUCTION_BUILD "Enable production hardening (anti-debug, etc.)" OFF)
option(ENABLE_ANTIDEBUG "Enable anti-debugging measures" OFF)
option(ENABLE_OPENMP "Enable OpenMP for parallelization" ON)
option(BUILD_TESTS "Build native op tests" OFF)

# =============================================================================
# Chain Secret Configuration
# =============================================================================
# Generate or use provided chain secrets for binary authentication.
# These are embedded at compile time and used to detect tampering.

if(NOT DEFINED CHAIN_SECRET_HIGH)
    # Generate random secret if not provided
    string(RANDOM LENGTH 16 ALPHABET "0123456789ABCDEF" CHAIN_SECRET_RANDOM)
    set(CHAIN_SECRET_HIGH "0x${CHAIN_SECRET_RANDOM}ULL")
    message(STATUS "Generated CHAIN_SECRET_HIGH: ${CHAIN_SECRET_HIGH}")
endif()

if(NOT DEFINED CHAIN_SECRET_LOW)
    string(RANDOM LENGTH 16 ALPHABET "0123456789ABCDEF" CHAIN_SECRET_RANDOM)
    set(CHAIN_SECRET_LOW "0x${CHAIN_SECRET_RANDOM}ULL")
    message(STATUS "Generated CHAIN_SECRET_LOW: ${CHAIN_SECRET_LOW}")
endif()

# Crypto key for string encryption
if(NOT DEFINED HN_CRYPTO_KEY)
    set(HN_CRYPTO_KEY "V3RS0_1NDUSTR13S_H1GHN00N_2025_K3Y")
endif()

# =============================================================================
# Edition Configuration (Lite, Pro, Enterprise)
# =============================================================================
# HN_EDITION values:
#   0 = LITE       (default) - Free tier with scale limits enforced
#   1 = PRO        - Paid tier with no scale limits, pre-compiled binary
#   2 = ENTERPRISE - Source code access + no limits + dedicated support

if(NOT DEFINED HN_EDITION)
    set(HN_EDITION 0)
endif()

if(NOT DEFINED HN_EDITION_NAME)
    if(HN_EDITION EQUAL 0)
        set(HN_EDITION_NAME "LITE")
    elseif(HN_EDITION EQUAL 1)
        set(HN_EDITION_NAME "PRO")
    elseif(HN_EDITION EQUAL 2)
        set(HN_EDITION_NAME "ENTERPRISE")
    else()
        set(HN_EDITION_NAME "UNKNOWN")
    endif()
endif()

message(STATUS "Edition: ${HN_EDITION_NAME} (HN_EDITION=${HN_EDITION})")

# =============================================================================
# Find TensorFlow
# =============================================================================
message(STATUS "Detecting TensorFlow installation...")

# Find Python - prefer PYTHON_EXEC env var for venv compatibility
if(DEFINED ENV{PYTHON_EXEC})
    set(Python3_EXECUTABLE "$ENV{PYTHON_EXEC}")
    message(STATUS "Using Python from PYTHON_EXEC: ${Python3_EXECUTABLE}")
else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

# Get TensorFlow compile flags
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import tensorflow as tf; print(tf.sysconfig.get_include())"
    OUTPUT_VARIABLE TF_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TF_RESULT
)

if(NOT TF_RESULT EQUAL 0)
    message(FATAL_ERROR "TensorFlow not found. Install with: pip install tensorflow>=2.15.0")
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE TF_CFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE TF_LFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "TensorFlow include: ${TF_INCLUDE_DIR}")
message(STATUS "TensorFlow cflags: ${TF_CFLAGS}")
message(STATUS "TensorFlow lflags: ${TF_LFLAGS}")

# =============================================================================
# Detect Architecture and SIMD Support
# =============================================================================
include(CheckCXXCompilerFlag)

# Detect target architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(TARGET_ARCH "x86_64")

    # Use CPU_OPT_FLAGS from environment if provided by build_secure.sh
    # This ensures we use CPU detection rather than compiler capability detection
    if(DEFINED ENV{CPU_OPT_FLAGS})
        # Parse space-separated flags from environment
        separate_arguments(SIMD_FLAGS UNIX_COMMAND "$ENV{CPU_OPT_FLAGS}")
        message(STATUS "SIMD: Using CPU-detected flags from environment: $ENV{CPU_OPT_FLAGS}")
    else()
        # Fallback: Check actual CPU capabilities (not just compiler support)
        # Read /proc/cpuinfo to detect actual CPU features
        execute_process(
            COMMAND bash -c "grep -q 'avx512f' /proc/cpuinfo && echo 'yes' || echo 'no'"
            OUTPUT_VARIABLE CPU_HAS_AVX512
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(
            COMMAND bash -c "grep -q 'avx2' /proc/cpuinfo && echo 'yes' || echo 'no'"
            OUTPUT_VARIABLE CPU_HAS_AVX2
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(CPU_HAS_AVX512 STREQUAL "yes")
            set(SIMD_FLAGS -O3 -mavx512f -mavx512bw -mfma)
            message(STATUS "SIMD: AVX-512 detected on CPU")
        elseif(CPU_HAS_AVX2 STREQUAL "yes")
            set(SIMD_FLAGS -O3 -mavx2 -mfma)
            message(STATUS "SIMD: AVX2 detected on CPU")
        else()
            set(SIMD_FLAGS -O3 -msse4.2)
            message(STATUS "SIMD: SSE4.2 fallback")
        endif()
    endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(TARGET_ARCH "arm64")
    if(DEFINED ENV{CPU_OPT_FLAGS})
        separate_arguments(SIMD_FLAGS UNIX_COMMAND "$ENV{CPU_OPT_FLAGS}")
        message(STATUS "SIMD: Using CPU-detected flags from environment: $ENV{CPU_OPT_FLAGS}")
    else()
        set(SIMD_FLAGS -O3 -march=armv8.2-a+simd+fp16)
        message(STATUS "SIMD: ARM NEON")
    endif()
else()
    set(TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    set(SIMD_FLAGS "")
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# =============================================================================
# Source Files - ALL ops consolidated into single binary
# =============================================================================
set(HIGHNOON_OPS_SOURCES
    # Core Language Model Ops
    ops/fused_moe_dispatch_op.cc
    ops/fused_superposition_moe_op.cc
    ops/fused_reasoning_stack_op.cc
    ops/fused_reasoning_stack/fused_reasoning_stack_kernel.cc
    ops/fused_reasoning_stack/helpers.cc
    ops/fused_reasoning_stack/tt_helpers.cc
    ops/fused_hnn_step_op.cc
    ops/fused_hnn_sequence_op.cc
    ops/fused_qwt_tokenizer_op.cc
    ops/selective_scan_op.cc
    ops/fused_norm_proj_act_op.cc
    ops/fused_add_op.cc
    ops/fused_graph_pad_op.cc
    ops/fused_depthwise_conv1d.cc
    ops/mps_contract_op.cc
    ops/mps_evolution_op.cc
    ops/fused_mps_temporal_op.cc
    ops/meta_controller_op.cc
    ops/time_crystal_step_op.cc
    # Phase 1 Migration: Flash Linear Attention
    ops/fused_linear_attention_op.cc
    # Phase 1 Migration: Token Shift (RWKV-6 DataDependentTokenShift)
    ops/fused_token_shift_op.cc
    # Phase 3 Migration: Local Attention (Griffin-style windowed attention)
    ops/fused_local_attention_op.cc
    # Phase 3 Migration: Mamba SSM (State-Space Model)
    ops/fused_mamba_op.cc
    # Phase 4 Migration: WLAM (Wavelet-Enhanced Linear Attention)
    ops/fused_wlam_op.cc
    # Phase 4 Migration: Memory Bank (GatedExternalMemory)
    ops/fused_memory_bank_op.cc
    # Phase 4 Migration: Memory Builder Enhancements (CTQW, MultiRate, CrossLevel, Adaptive, QGAN)
    ops/fused_memory_builder_enhancements_op.cc
    # Phase 12.11: Cross-Block State Bus
    ops/fused_state_bus_op.cc
    # Phase 12.11: Enhancement 4: Quantum-Inspired Slot Superposition
    ops/fused_superposition_slots_op.cc
    # Phase 4 Migration: Latent Reasoning Block
    ops/fused_latent_reasoning_op.cc
    # Phase 5 Migration: Continuous Thought (COCONUT)
    ops/fused_continuous_thought_op.cc
    # Phase 5 Migration: Self-Consistency Verification
    ops/fused_self_consistency_op.cc
    # Phase 5 Migration: Speculative Decoding
    ops/fused_speculative_op.cc
    # Phase 5 Migration: Streaming Inference
    ops/fused_streaming_op.cc
    # Phase 5 Migration: Quantum Layers
    ops/fused_quantum_layers_op.cc
    # Phase 5 Migration: Tensor Layers
    ops/fused_tensor_layers_op.cc
    # Phase 5 Migration: Custom Attention
    ops/fused_custom_attention_op.cc
    # Phase 5 Migration: Optimizers
    ops/fused_optimizers_op.cc
    # Phase 5 Migration: QBM
    ops/fused_qbm_op.cc
    # Phase 5 Migration: QGAN
    ops/fused_qgan_op.cc
    # Phase 5 Migration: Stateful Wrapper
    ops/fused_stateful_wrapper_op.cc
    # Phase 15: Flash Attention with enhanced features
    ops/fused_flash_attention_op.cc  # GLA, RALA, Hybrid, Chunkwise, QuantumInspired
    # Phase 16: Contextual Gating Collapse
    ops/fused_collapse_op.cc         # Multi-head cross-attention collapse with Gumbel-Softmax
    # Phase 17: MoE Layer Enhancements
    ops/mla_collapse_op.cc            # Multi-Head Latent Attention collapse (DeepSeek-V2 style)
    ops/quantum_moe_ops.cc            # Quantum-Inspired MoE: QIR, Hamiltonian, MPO, BornRule
    # Attribution Enforcement (tamper-proof, no Python wrapper)
    ops/get_attribution_op.cc         # Framework attribution, trigger detection, metadata
    ops/set_attribution_op.cc         # Custom attribution (Pro/Enterprise only)
    # Phase 18: Final Upgrade Enhancements
    ops/fused_differential_attention_op.cc  # Differential Transformer (ICLR 2025)
    ops/fused_mod_routing_op.cc       # Mixture-of-Depths dynamic layer routing
    ops/fused_native_sparse_attention_op.cc # Native Sparse Attention (DeepSeek ACL 2025)
    # New ops for full native coverage
    ops/qbm_sample_op.cc              # QBM Sampling with annealing
    ops/vqc_expectation_op.cc         # VQC expectation values
    ops/train_step_op.cc              # Full TrainStep with N4SID/EWC/reasoning stack
    ops/fused_wavelet_encoder_op.cc   # Wavelet encoder for sequence chunking
    # N4SID System Identification (online learning for meta-controller)
    ops/n4sid_solver.cc               # N4SID Subspace System Identification
    # Lorentzian/Hyperbolic Geometry (hierarchical language structures)
    ops/lorentzian_feature_transform_op.cc  # Lie algebra matrix exponential
    ops/fused_lorentzian_gat_op.cc    # Lorentzian Graph Attention
    # Efficient Sparse Operations
    ops/structured_sparse_matmul_op.cc # Band-diagonal sparse matmul
    # Phase 18.1: Latent KV Attention (MLA-style KV compression)
    ops/fused_latent_kv_attention_op.cc
    # Phase 18.3: Adaptive Memory with Test-Time Learning (Titans-inspired)
    ops/fused_adaptive_memory_op.cc
    # Phase 19-24: Unified Quantum Block (all quantum enhancements)
    ops/fused_unified_quantum_block_op.cc
    # Quantum Superposition Generation (QSG) - parallel non-autoregressive generation
    ops/fused_qsg_op.cc
    # Phases 26-36: Unified Quantum Architecture Enhancements
    ops/quantum_residual_op.cc        # Phase 34: Unitary residual connections
    ops/quantum_norm_op.cc            # Phase 30: Quantum normalization (Stiefel)
    ops/quantum_expert_op.cc          # Phase 29: Unitary expert networks (Cayley)
    ops/quantum_embedding_op.cc       # Phase 26: Holographic embeddings (FFT-bind)
    ops/quantum_position_encoding_op.cc  # Phase 27: Floquet position encoding (SU(2))
    ops/quantum_lm_head_op.cc         # Phase 33: VQC-based LM head (Born rule)
    ops/qsg_grover_op.cc              # Phase 32: Grover-guided QSG enhancement
    # Phases 37-46: Quantum Enhancement Integration (final_enhancements.md)
    ops/qmamba_op.cc                  # Phase 37: QMamba quantum-enhanced SSM
    ops/discrete_time_crystal_op.cc   # Phase 38: DTC state protection
    ops/coconut_reservoir_op.cc       # Phase 39: Coconut continuous latent reasoning
    ops/lmwt_attention_op.cc          # Phase 41: Learnable wavelet transformer
    ops/qmoe_routing_op.cc            # Phase 42: Quantum MoE routing
    ops/neural_kalman_op.cc           # Phase 43: Neural Kalman with learned gain
    ops/quantum_teleport_bus_op.cc    # Phase 44: Quantum teleport state bus
    ops/entropy_regularization_op.cc  # Phase 45: Von Neumann entropy regularization
    ops/sympflow_optimizer_op.cc      # Phase 46: SympFlow Hamiltonian optimizer
    # Phases 47-84: Quantum Enhancement Integration v5.0 (final_enhancements.md)
    # Pillar 1: Foundation (Critical)
    ops/quantum_coherence_bus_op.cc   # Phase 76: Unified Quantum Coherence Bus
    ops/q_ssm_gating_op.cc            # Phase 69: Q-SSM quantum VQC gating
    ops/intrinsic_plasticity_op.cc    # Phase 71: Intrinsic plasticity preservation
    ops/quantum_measurement_dropout_op.cc  # Phase 47: Quantum measurement dropout
    # Pillar 2: Input/Output Enhancement
    ops/hyperdimensional_embedding_op.cc  # Phase 48: Hyperdimensional embeddings
    ops/hypertoken_op.cc              # Phase 49: Holographic hypertokens
    ops/majorana_position_op.cc       # Phase 50: Majorana position encoding
    ops/born_rule_loss_op.cc          # Phase 51: Born rule loss
    ops/quantum_fidelity_loss_op.cc   # Phase 52: Quantum fidelity regularization
    # Pillar 3: Topological Reasoning
    ops/qasa_attention_op.cc          # Phase 53: QASA attention
    ops/mpqr_reasoning_op.cc          # Phase 55: Multi-path quantum reasoning
    ops/topological_wavelet_op.cc     # Phase 56: Topological wavelet attention
    ops/td_moe_op.cc                  # Phase 57: TD-MoE Tucker decomposition
    ops/symplectic_gnn_kalman_op.cc   # Phase 58: Symplectic GNN Kalman
    # Pillar 4: Training & Optimization
    ops/adiabatic_optimizer_op.cc     # Phase 59: Quantum adiabatic optimizer
    ops/geodesic_optimizer_op.cc      # Phase 60: Geodesic optimizer
    ops/alphaqubit_decoder_op.cc      # Phase 61: AlphaQubit-2 decoder
    ops/vqem_op.cc                    # Phase 62: VQEM error mitigation
    ops/gradient_teleportation_op.cc  # Phase 64: Gradient teleportation
    # Pillar 5: Memory & Continual Learning
    ops/quantum_crystallization_op.cc # Phase 65/83: Quantum crystallization
    ops/quantum_neuromorphic_op.cc    # Phase 68: Quantum neuromorphic memory
    # Pillar 6: Coherence Mesh
    ops/multi_stage_hamiltonian_op.cc # Phase 70: Multi-stage Hamiltonian
    ops/random_natural_gradient_op.cc # Phase 72: Random natural gradient
    # Pillar 7: Advanced Quantum Intelligence
    ops/spini_integrator_op.cc        # Phase 78: SPINI integrator
    ops/quantum_advanced_ops.cc       # Phases 73-84: Consolidated advanced ops
)

# Controller sources (requires spdlog: apt install libspdlog-dev)
set(HIGHNOON_CONTROLLER_SOURCES "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/controllers/hamiltonian_meta_controller.cc")
    list(APPEND HIGHNOON_CONTROLLER_SOURCES
        controllers/hamiltonian_meta_controller.cc
        controllers/hierarchical_controller.cc
        controllers/mpc_controller.cc
        controllers/adaptive_phase_controller.cc
        controllers/kalman_filter.cc
        controllers/sensitivity_tracker.cc
        controllers/config/state_space_config_loader.cc
        controllers/utils/matrix_utils.cc
        # Phase 2.1: Quantum-Enhanced Control System (QUANTUM_CONTROL_ENHANCEMENT_ROADMAP.md)
        controllers/rls_system_identifier.cc      # O(n²) recursive system ID (170x faster than N4SID)
        controllers/hybrid_pid_tuner.cc           # Relay/Z-N + Adam gradient descent hybrid
        controllers/extended_kalman_filter.cc     # EKF for nonlinear dynamics
        controllers/tensor_network_kalman.cc      # TT-decomposed Kalman (O(n×r²) memory)
    )
    message(STATUS "Controllers: Enabled (spdlog required)")
else()
    message(STATUS "Controllers: Disabled (source files not found)")
endif()

# Combine all sources
set(HIGHNOON_ALL_SOURCES
    ${HIGHNOON_OPS_SOURCES}
    ${HIGHNOON_CONTROLLER_SOURCES}
)

# =============================================================================
# Compiler Flags
# =============================================================================

# Base optimization and security flags
set(BASE_CXX_FLAGS
    -O3                         # Maximum optimization
    -fPIC                       # Position independent code
    -fvisibility=hidden         # Hide all symbols by default
    -fstack-protector-strong    # Stack smashing protection
    -fno-strict-aliasing        # Safe type punning
    -Wall -Wextra               # Warnings
    -Wno-unused-parameter       # TensorFlow has many of these
)

# Hardening flags for security
set(HARDENING_FLAGS
    -D_FORTIFY_SOURCE=2         # Buffer overflow detection
    -fno-delete-null-pointer-checks
)

# TensorFlow required flags
set(TF_REQUIRED_FLAGS
    -DEIGEN_USE_THREADS
    ${TF_CFLAGS}
)

# LTO flags
if(ENABLE_LTO)
    set(LTO_FLAGS -flto)
    message(STATUS "LTO: Enabled")
else()
    set(LTO_FLAGS "")
    message(STATUS "LTO: Disabled")
endif()

# Anti-debug flags
if(PRODUCTION_BUILD AND ENABLE_ANTIDEBUG)
    set(ANTIDEBUG_FLAGS -DPRODUCTION_BUILD=1)
    message(STATUS "Anti-Debug: Enabled")
else()
    set(ANTIDEBUG_FLAGS "")
    message(STATUS "Anti-Debug: Disabled")
endif()

# =============================================================================
# Build Consolidated Library
# =============================================================================
add_library(highnoon_core SHARED ${HIGHNOON_ALL_SOURCES})

# Set output properties
set_target_properties(highnoon_core PROPERTIES
    OUTPUT_NAME "_highnoon_core"
    PREFIX ""
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Include directories (including spdlog for controllers)
target_include_directories(highnoon_core PRIVATE
    ${TF_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/common
    ${CMAKE_CURRENT_SOURCE_DIR}/controllers
    /usr/include  # System includes including spdlog
)

# Compile definitions
target_compile_definitions(highnoon_core PRIVATE
    CHAIN_SECRET_HIGH=${CHAIN_SECRET_HIGH}
    CHAIN_SECRET_LOW=${CHAIN_SECRET_LOW}
    HN_CRYPTO_KEY="${HN_CRYPTO_KEY}"
    HIGHNOON_VERSION="${PROJECT_VERSION}"
    HIGHNOON_EDITION="${HN_EDITION_NAME}"
    HN_EDITION=${HN_EDITION}
)

# Compile options
target_compile_options(highnoon_core PRIVATE
    ${BASE_CXX_FLAGS}
    ${HARDENING_FLAGS}
    ${TF_REQUIRED_FLAGS}
    ${SIMD_FLAGS}
    ${LTO_FLAGS}
    ${ANTIDEBUG_FLAGS}
)

# OpenMP support
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(highnoon_core PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP: Enabled")
    else()
        message(WARNING "OpenMP: Not found, parallelization disabled")
    endif()
endif()

# spdlog and fmt linkage (required for controllers)
if(HIGHNOON_CONTROLLER_SOURCES)
    find_package(spdlog QUIET)
    if(spdlog_FOUND)
        target_link_libraries(highnoon_core PRIVATE spdlog::spdlog)
        message(STATUS "spdlog: Found via CMake package")
    else()
        # Fallback to system library
        target_link_libraries(highnoon_core PRIVATE spdlog fmt)
        message(STATUS "spdlog: Using system library")
    endif()
endif()

# Link options
target_link_options(highnoon_core PRIVATE
    -Wl,-z,relro          # Read-only relocation
    -Wl,-z,now            # Immediate binding
    -Wl,--no-as-needed    # TensorFlow requires this
    ${LTO_FLAGS}
)

# TensorFlow link flags (parse and add)
separate_arguments(TF_LFLAGS_LIST UNIX_COMMAND "${TF_LFLAGS}")
target_link_options(highnoon_core PRIVATE ${TF_LFLAGS_LIST})

# NOTE: Symbol stripping moved to after copy step below (line ~615)

# =============================================================================
# Installation Paths (define early for post-build commands)
# =============================================================================
set(INSTALL_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin/${TARGET_ARCH}")

# =============================================================================
# HPO Orchestrator Executable (hpo_main)
# =============================================================================
# The HPO binary is a standalone executable for coordinating hyperparameter
# optimization trials. It communicates via file-based IPC with the Python layer.

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ops/hpo_main.cc")
    add_executable(hpo_main ops/hpo_main.cc)

    set_target_properties(hpo_main PROPERTIES
        OUTPUT_NAME "hpo_main"
        CXX_VISIBILITY_PRESET hidden
    )

    target_include_directories(hpo_main PRIVATE
        ${TF_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ops
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/common
    )

    target_compile_options(hpo_main PRIVATE
        ${BASE_CXX_FLAGS}
        ${TF_REQUIRED_FLAGS}
        ${SIMD_FLAGS}
        ${LTO_FLAGS}
    )

    target_link_options(hpo_main PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        ${LTO_FLAGS}
        ${TF_LFLAGS_LIST}
    )

    # Also build as shared library for Python ctypes
    add_library(hpo_main_lib SHARED ops/hpo_main.cc)

    set_target_properties(hpo_main_lib PROPERTIES
        OUTPUT_NAME "_hpo_main"
        PREFIX ""
        SUFFIX ".so"
        CXX_VISIBILITY_PRESET hidden
    )

    target_include_directories(hpo_main_lib PRIVATE
        ${TF_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ops
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/common
    )

    target_compile_options(hpo_main_lib PRIVATE
        ${BASE_CXX_FLAGS}
        ${TF_REQUIRED_FLAGS}
        ${SIMD_FLAGS}
        ${LTO_FLAGS}
    )

    target_link_options(hpo_main_lib PRIVATE
        ${LTO_FLAGS}
        ${TF_LFLAGS_LIST}
    )

    # Copy HPO binaries to bin directory
    add_custom_command(TARGET hpo_main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:hpo_main> ${INSTALL_BIN_DIR}/
        COMMENT "Copying hpo_main to ${INSTALL_BIN_DIR}/"
    )

    add_custom_command(TARGET hpo_main_lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:hpo_main_lib> ${INSTALL_BIN_DIR}/
        COMMENT "Copying _hpo_main.so to ${INSTALL_BIN_DIR}/"
    )

    message(STATUS "HPO Orchestrator: Enabled")
else()
    message(STATUS "HPO Orchestrator: Disabled (source not found)")
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS highnoon_core
    LIBRARY DESTINATION ${INSTALL_BIN_DIR}
)

# Copy to bin directory on build, then strip
add_custom_command(TARGET highnoon_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTALL_BIN_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:highnoon_core> ${INSTALL_BIN_DIR}/
    COMMENT "Copying _highnoon_core.so to ${INSTALL_BIN_DIR}/"
)

# =============================================================================
# Post-Build: Symbol Stripping (AFTER copy to ensure stripped binary deployed)
# =============================================================================
if(STRIP_SYMBOLS AND CMAKE_BUILD_TYPE STREQUAL "Release")
    # Find strip command
    find_program(STRIP_EXECUTABLE strip)
    if(STRIP_EXECUTABLE)
        add_custom_command(TARGET highnoon_core POST_BUILD
            COMMAND ${STRIP_EXECUTABLE} --strip-all ${INSTALL_BIN_DIR}/_highnoon_core.so
            COMMENT "Stripping symbols from ${INSTALL_BIN_DIR}/_highnoon_core.so"
        )
        message(STATUS "Symbol Stripping: Enabled (Release build)")
    else()
        message(WARNING "Symbol Stripping: strip command not found!")
    endif()
else()
    message(STATUS "Symbol Stripping: Disabled")
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== HighNoon Build Configuration ===")
message(STATUS "Version:         ${PROJECT_VERSION}")
message(STATUS "Edition:         ${HN_EDITION_NAME} (code=${HN_EDITION})")
message(STATUS "Architecture:    ${TARGET_ARCH}")
message(STATUS "Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "LTO:             ${ENABLE_LTO}")
message(STATUS "Production:      ${PRODUCTION_BUILD}")
message(STATUS "Anti-Debug:      ${ENABLE_ANTIDEBUG}")
message(STATUS "Strip Symbols:   ${STRIP_SYMBOLS}")
message(STATUS "Output:          ${INSTALL_BIN_DIR}/_highnoon_core.so")
message(STATUS "")
