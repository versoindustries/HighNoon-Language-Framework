#!/bin/bash

# ==============================================================================
# HighNoon Language Framework - Build Script for Custom TensorFlow Operators
#
# This script builds the custom C++ TensorFlow operators for HighNoon.
# It supports x86_64 and arm64 architectures.
#
# Usage:
#   ./build_ops.sh              # Build all ops for current architecture
#   ./build_ops.sh <op_name>    # Build specific op
#   ./build_ops.sh clean        # Clean build artifacts
#
# Environment Variables:
#   PYTHON_EXEC     - Python interpreter to use (default: ./venv/bin/python)
#   CXX_COMPILER    - C++ compiler (default: g++)
#   CPU_OPT_FLAGS   - CPU optimization flags (auto-detected if not set)
# ==============================================================================

set -u
set -o pipefail

# --- Script Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="${SCRIPT_DIR}"
OPS_DIR="${NATIVE_DIR}/ops"
BIN_DIR="${NATIVE_DIR}/bin"
LOG_DIR="${NATIVE_DIR}/logs"

CXX_COMPILER="${CXX_COMPILER:-g++}"
PYTHON_EXEC="${PYTHON_EXEC:-}"

# --- Helper Functions ---
log_info() {
    echo "--- $1 ---"
}

log_error() {
    echo >&2 "Error: $1"
}

log_success() {
    echo "✅ $1"
}

# --- Detect Python and TensorFlow ---
detect_environment() {
    log_info "Detecting Python and TensorFlow environment"

    if [ -z "${PYTHON_EXEC}" ]; then
        if [ -n "${VIRTUAL_ENV:-}" ]; then
            PYTHON_EXEC="${VIRTUAL_ENV}/bin/python"
        elif [ -f "./venv/bin/python" ]; then
            PYTHON_EXEC="./venv/bin/python"
        else
            PYTHON_EXEC="python3"
        fi
    fi

    echo "Using Python: ${PYTHON_EXEC}"

    if ! command -v "${PYTHON_EXEC}" >/dev/null 2>&1; then
        log_error "Python interpreter not found: ${PYTHON_EXEC}"
        return 1
    fi

    if ! ${PYTHON_EXEC} -c "import tensorflow" 2>/dev/null; then
        log_error "TensorFlow not found. Please install: pip install tensorflow"
        return 1
    fi

    # Get TensorFlow compile and link flags
    TF_CFLAGS=$(${PYTHON_EXEC} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
    TF_LFLAGS=$(${PYTHON_EXEC} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
    TF_INCLUDE=$(${PYTHON_EXEC} -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

    echo "TensorFlow include: ${TF_INCLUDE}"
}

# --- Detect CPU Architecture ---
detect_architecture() {
    local arch
    arch=$(uname -m)

    case "${arch}" in
        x86_64|amd64)
            ARCH_NAME="x86_64"
            if [ -z "${CPU_OPT_FLAGS:-}" ]; then
                # Auto-detect SIMD support
                local flags=""
                if [ -f /proc/cpuinfo ]; then
                    flags=$(grep -m1 "flags" /proc/cpuinfo | tr ' ' '\n')
                fi

                if echo "${flags}" | grep -q "avx512f"; then
                    CPU_OPT_FLAGS="-O3 -mavx512f -mavx512bw -mfma"
                elif echo "${flags}" | grep -q "avx2"; then
                    CPU_OPT_FLAGS="-O3 -mavx2 -mfma"
                elif echo "${flags}" | grep -q "avx"; then
                    CPU_OPT_FLAGS="-O3 -mavx"
                else
                    CPU_OPT_FLAGS="-O3 -msse4.2"
                fi
            fi
            ;;
        aarch64|arm64)
            ARCH_NAME="arm64"
            CPU_OPT_FLAGS="${CPU_OPT_FLAGS:--O3 -march=armv8.2-a+simd+fp16}"
            ;;
        *)
            log_error "Unsupported architecture: ${arch}"
            return 1
            ;;
    esac

    echo "Architecture: ${ARCH_NAME}"
    echo "CPU flags: ${CPU_OPT_FLAGS}"
}

# --- Build a single op ---
build_op() {
    local op_name=$1
    local source_file=""
    local output_file=""

    # Find source file
    if [ -f "${OPS_DIR}/${op_name}.cc" ]; then
        source_file="${OPS_DIR}/${op_name}.cc"
    elif [ -f "${OPS_DIR}/${op_name}_op.cc" ]; then
        source_file="${OPS_DIR}/${op_name}_op.cc"
    else
        log_error "Source file not found for '${op_name}'"
        return 1
    fi

    output_file="${BIN_DIR}/${ARCH_NAME}/_${op_name}.${ARCH_NAME}.so"
    mkdir -p "$(dirname "${output_file}")"

    echo "Building: ${op_name}"
    echo "  Source: ${source_file}"
    echo "  Output: ${output_file}"

    local include_flags="-I${OPS_DIR} -I${TF_INCLUDE}"
    local openmp_flags="-fopenmp"

    ${CXX_COMPILER} -std=c++17 -shared -o "${output_file}" "${source_file}" \
        -fPIC ${TF_CFLAGS} ${TF_LFLAGS} ${CPU_OPT_FLAGS} ${openmp_flags} \
        ${include_flags} \
        -DEIGEN_USE_THREADS \
        -Wl,--no-as-needed 2>&1

    if [ $? -ne 0 ]; then
        log_error "Build failed for ${op_name}"
        return 1
    fi

    log_success "Built ${op_name}"
    return 0
}

# --- Clean ---
clean() {
    log_info "Cleaning build artifacts"
    rm -rf "${LOG_DIR}"
    find "${BIN_DIR}" -name "*.so" -type f -delete 2>/dev/null || true
    log_success "Clean complete"
}

# --- Main ---
main() {
    if [ "$#" -gt 0 ] && [ "$1" == "clean" ]; then
        clean
        exit 0
    fi

    detect_environment || exit 1
    detect_architecture || exit 1

    echo "----------------------------------------"

    # List of core ops for HighNoon
    local ops=(
        "fused_moe_dispatch"
        "fused_superposition_moe"
        "fused_reasoning_stack"
        "fused_add"
        "fused_norm_proj_act"
        "fused_qwt_tokenizer"
        "fused_graph_pad"
        "selective_scan"
    )

    if [ "$#" -gt 0 ]; then
        ops=("$@")
    fi

    mkdir -p "${LOG_DIR}"

    local success=0
    local failed=0

    for op in "${ops[@]}"; do
        if build_op "${op}"; then
            ((success++))
        else
            ((failed++))
        fi
    done

    echo "----------------------------------------"
    log_info "Build Summary"
    echo "Successful: ${success}"
    echo "Failed: ${failed}"

    if [ "${failed}" -gt 0 ]; then
        log_error "Some ops failed to build"
        exit 1
    fi

    log_success "All ops built successfully"
}

main "$@"
