# highnoon/_native/__init__.py
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

"""Native binary loader and license validation for HighNoon Language Framework.

This module handles loading of compiled C++ TensorFlow operations and validates
the binary chain authentication to prevent tampering.

The Lite edition includes compiled binaries with hard-coded scale limits that
cannot be bypassed from Python. Enterprise binaries remove these limits.

Binary Structure (Consolidated Build):
    _native/
    ├── bin/
    │   ├── x86_64/              # Linux x86_64 binaries
    │   │   └── _highnoon_core.so  # Single consolidated binary
    │   └── arm64/               # Linux arm64 binaries
    │       └── _highnoon_core.so
    └── ops/                     # C++ source files (for reference)
        └── ...

Security Features:
    - Scale limits enforced in C++ (cannot be bypassed from Python)
    - Chain authentication validates binary integrity
    - Anti-debugging measures in production builds
    - String encryption hides sensitive values
"""

import logging
import os
import platform
import sys
import threading
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# Determine platform-specific binary directory
_PLATFORM = sys.platform
_ARCH = platform.machine()

# Map architecture names to standardized directory names
_ARCH_MAP = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
}

_ARCH_DIR = _ARCH_MAP.get(_ARCH, _ARCH)

# Native directory path
_NATIVE_DIR = Path(__file__).parent.absolute()
_BIN_DIR = _NATIVE_DIR / "bin" / _ARCH_DIR

# Core binary name (consolidated binary)
_CORE_BINARY_NAME = "_highnoon_core.so"

# Thread-safe singleton for consolidated binary
_consolidated_binary_lock = threading.Lock()
_consolidated_binary: Any | None = None
_consolidated_binary_loaded = False


def _get_binary_path() -> Path:
    """Get the path to platform-specific binaries."""
    return _BIN_DIR


def _find_core_binary() -> Path | None:
    """Find the consolidated core binary.

    Returns:
        Path to _highnoon_core.so if found, None otherwise.
    """
    core_path = _BIN_DIR / _CORE_BINARY_NAME
    if core_path.exists():
        return core_path
    return None


def _find_op_binary(op_name: str) -> Path | None:
    """Find the binary for an operation.

    First checks for consolidated binary, then falls back to individual binaries.

    Searches in order:
    1. _highnoon_core.so (consolidated binary with all ops)
    2. _<op_name>.so (legacy individual binary)
    3. _<op_name>.<arch>.so (arch-specific)
    4. _<op_name>_op.so (op suffix format)
    5. _<op_name>_op.<arch>.so

    Args:
        op_name: Name of the operation.

    Returns:
        Path to binary if found, None otherwise.
    """
    # Prefer consolidated binary
    core_path = _find_core_binary()
    if core_path is not None:
        return core_path

    # Fall back to individual binaries (legacy support)
    patterns = [
        f"_{op_name}.so",
        f"_{op_name}.{_ARCH_DIR}.so",
        f"_{op_name}_op.so",
        f"_{op_name}_op.{_ARCH_DIR}.so",
    ]

    for pattern in patterns:
        path = _BIN_DIR / pattern
        if path.exists():
            return path

    return None


def _load_consolidated_binary() -> Any | None:
    """Load the consolidated core binary (singleton).

    Thread-safe loading of the single _highnoon_core.so binary
    containing all native ops.

    Returns:
        Loaded TensorFlow operation module, or None if unavailable.
    """
    global _consolidated_binary, _consolidated_binary_loaded

    # Double-checked locking pattern
    if _consolidated_binary_loaded:
        return _consolidated_binary

    with _consolidated_binary_lock:
        # Check again under lock
        if _consolidated_binary_loaded:
            return _consolidated_binary

        try:
            import tensorflow as tf
        except ImportError:
            log.warning(
                "TensorFlow is required to load native operations. "
                "Install with: pip install tensorflow>=2.15.0"
            )
            _consolidated_binary_loaded = True
            return None

        core_path = _find_core_binary()
        if core_path is None:
            log.debug(
                f"Consolidated binary '{_CORE_BINARY_NAME}' not found in {_BIN_DIR}. "
                f"Will attempt individual binary loading."
            )
            _consolidated_binary_loaded = True
            return None

        try:
            _consolidated_binary = tf.load_op_library(str(core_path))
            log.info(f"Loaded consolidated binary: {core_path}")
            _consolidated_binary_loaded = True
            return _consolidated_binary
        except Exception as e:
            log.error(f"Failed to load consolidated binary {core_path}: {e}")
            _consolidated_binary_loaded = True
            return None


def _load_op(op_name: str) -> Any:
    """Load a compiled TensorFlow operation.

    Attempts to use consolidated binary first, falls back to individual binaries.

    Args:
        op_name: Name of the operation (without .so extension).

    Returns:
        Loaded TensorFlow operation module.

    Raises:
        ImportError: If the operation cannot be loaded.
    """
    # Try consolidated binary first
    consolidated = _load_consolidated_binary()
    if consolidated is not None:
        return consolidated

    # Fall back to individual binary loading
    try:
        import tensorflow as tf
    except ImportError as err:
        raise ImportError(
            "TensorFlow is required to load native operations. "
            "Install with: pip install tensorflow>=2.15.0"
        ) from err

    op_path = _find_op_binary(op_name)

    if op_path is None:
        log.debug(
            f"Native operation '{op_name}' not found in {_BIN_DIR}. "
            f"Using Python fallback if available."
        )
        return None

    # Skip if it's the core binary (already tried above)
    if op_path.name == _CORE_BINARY_NAME:
        return None

    try:
        op_module = tf.load_op_library(str(op_path))
        log.debug(f"Loaded native operation: {op_name} from {op_path}")
        return op_module
    except Exception as e:
        log.warning(f"Failed to load native operation {op_name}: {e}")
        return None


# Cache for loaded operations
_loaded_ops: dict[str, Any] = {}


def get_op(op_name: str) -> Any | None:
    """Get a compiled TensorFlow operation, loading it if necessary.

    Args:
        op_name: Name of the operation.

    Returns:
        Loaded operation module, or None if not available.
    """
    if op_name not in _loaded_ops:
        _loaded_ops[op_name] = _load_op(op_name)
    return _loaded_ops[op_name]


def list_available_ops() -> list[str]:
    """List all available native operations.

    Returns:
        List of available operation names.
    """
    # Standard ops available in consolidated binary
    consolidated_ops = [
        "fused_moe_dispatch",
        "fused_superposition_moe",
        "fused_reasoning_stack",
        "fused_hnn_step",
        "fused_hnn_sequence",
        "fused_qwt_tokenizer",
        "selective_scan",
        "fused_norm_proj_act",
        "fused_add",
        "fused_graph_pad",
        "mps_contract",
        "meta_controller",
        "time_crystal_step",
        "fused_linear_attention",  # Phase 1 C++ Migration
        "fused_min_gru",  # Phase 1 C++ Migration - MinGRU
        "fused_token_shift",  # Phase 1 C++ Migration - Token Shift
        "fused_rg_lru",  # Phase 2 C++ Migration - RG-LRU
        "fused_ssd",  # Phase 2 C++ Migration - SSD
        "fused_local_attention",  # Phase 3 C++ Migration - Local Attention
        "fused_mamba",  # Phase 3 C++ Migration - Mamba SSM
        "fused_continuous_thought",  # Phase 5 C++ Migration - Continuous Thought
        "fused_flash_attention",  # Phase 16 Flash Linear Attention Enhancements (GLA, RALA, Hybrid)
    ]

    # Check if consolidated binary is available
    if _find_core_binary() is not None:
        return sorted(consolidated_ops)

    # Fall back to checking individual binaries
    ops = []
    if _BIN_DIR.exists():
        for path in _BIN_DIR.glob("*.so"):
            name = path.stem
            # Remove leading underscore and trailing arch/format
            if name.startswith("_"):
                name = name[1:]
            # Remove architecture suffix
            for suffix in [f".{_ARCH_DIR}", "_op"]:
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
            if name not in ops and name != "highnoon_core":
                ops.append(name)
    return sorted(ops)


def is_native_available() -> bool:
    """Check if any native operations are available.

    Returns:
        True if native binaries are present.
    """
    return _BIN_DIR.exists() and any(_BIN_DIR.glob("*.so"))


def is_consolidated_available() -> bool:
    """Check if consolidated binary is available.

    Returns:
        True if _highnoon_core.so is present.
    """
    return _find_core_binary() is not None


def is_enterprise() -> bool:
    """Check if enterprise binaries are installed.

    Returns:
        True if enterprise binaries are available, False for Lite edition.
    """
    # In Lite edition, this always returns False
    # Enterprise binaries override this function
    return False


def get_edition() -> str:
    """Get the current edition of the framework.

    Returns:
        'lite' for Lite edition, 'enterprise' for Enterprise edition.
    """
    return "enterprise" if is_enterprise() else "lite"


def get_binary_info() -> dict[str, Any]:
    """Get information about loaded binaries.

    Returns:
        Dictionary with binary information.
    """
    core_path = _find_core_binary()
    info = {
        "edition": get_edition(),
        "arch": _ARCH_DIR,
        "platform": _PLATFORM,
        "bin_dir": str(_BIN_DIR),
        "consolidated_available": core_path is not None,
        "consolidated_path": str(core_path) if core_path else None,
        "native_available": is_native_available(),
        "available_ops": list_available_ops(),
    }

    if core_path is not None:
        info["consolidated_size_bytes"] = core_path.stat().st_size

    return info


def check_enterprise_license(domain: str) -> bool:
    """Check if enterprise license is available for a domain module.

    Args:
        domain: Domain module to check ('chemistry', 'physics', etc.).

    Returns:
        True if the domain is accessible, False if locked.
    """
    from highnoon._native._limits import check_enterprise_license as _check

    return _check(domain)


# Commonly used ops - can be imported directly
def load_moe_dispatch():
    """Load the fused MoE dispatch operation."""
    return get_op("fused_moe_dispatch")


def load_superposition_moe():
    """Load the fused superposition MoE operation."""
    return get_op("fused_superposition_moe")


def load_reasoning_stack():
    """Load the fused reasoning stack operation."""
    return get_op("fused_reasoning_stack")


def load_selective_scan():
    """Load the selective scan (Mamba) operation."""
    return get_op("selective_scan")


def load_qwt_tokenizer():
    """Load the QWT tokenizer operation."""
    return get_op("fused_qwt_tokenizer")


def fused_qwt_tokenizer_op_path() -> str | None:
    """Return the resolved shared object path for the QWT tokenizer op."""
    op_path = _find_op_binary("fused_qwt_tokenizer")
    return str(op_path) if op_path is not None else None


def load_hnn_step():
    """Load the HNN step operation."""
    return get_op("fused_hnn_step")


def load_hnn_sequence():
    """Load the HNN sequence operation."""
    return get_op("fused_hnn_sequence")


def load_meta_controller():
    """Load the meta controller operation."""
    return get_op("meta_controller")


def load_linear_attention():
    """Load the fused linear attention operation (Phase 1 C++ Migration)."""
    return get_op("fused_linear_attention")


def load_min_gru():
    """Load the fused MinGRU operation (Phase 1 C++ Migration)."""
    return get_op("fused_min_gru")


def load_token_shift():
    """Load the fused Token Shift operation (Phase 1 C++ Migration)."""
    return get_op("fused_token_shift")


def load_rg_lru():
    """Load the fused RG-LRU operation (Phase 2 C++ Migration)."""
    return get_op("fused_rg_lru")


def load_ssd():
    """Load the fused SSD operation (Phase 2 C++ Migration)."""
    return get_op("fused_ssd")


def load_continuous_thought():
    """Load the fused continuous thought operation (Phase 5 C++ Migration)."""
    return get_op("fused_continuous_thought")


# Log edition and native availability on import
_edition = get_edition().upper()
_native_avail = "available" if is_native_available() else "not found"
_consolidated_avail = "consolidated" if is_consolidated_available() else "individual"
log.info(f"HighNoon Language Framework - {_edition} Edition")
log.info(f"Platform: {_PLATFORM}/{_ARCH}, Native ops: {_native_avail} ({_consolidated_avail})")

__all__ = [
    "get_op",
    "list_available_ops",
    "is_native_available",
    "is_consolidated_available",
    "is_enterprise",
    "get_edition",
    "get_binary_info",
    "check_enterprise_license",
    "load_moe_dispatch",
    "load_superposition_moe",
    "load_reasoning_stack",
    "load_selective_scan",
    "load_qwt_tokenizer",
    "fused_qwt_tokenizer_op_path",
    "load_hnn_step",
    "load_hnn_sequence",
    "load_meta_controller",
    "load_linear_attention",
    "load_min_gru",
    "load_token_shift",
    "load_rg_lru",
    "load_ssd",
    "load_continuous_thought",
]
