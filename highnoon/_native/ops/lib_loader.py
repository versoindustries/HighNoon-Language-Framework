# highnoon/_native/ops/lib_loader.py
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

"""Library loader for native C++ TensorFlow operations.

Resolves and loads compiled .so files based on target architecture.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from highnoon._native.runtime.arch import DEFAULT_VERSO_TARGET_ARCH, canonicalize_target_arch

logger = logging.getLogger(__name__)

# The bin directory is relative to _native/
_NATIVE_DIR = Path(__file__).resolve().parent.parent
_BIN_DIR = _NATIVE_DIR / "bin"


def _base_path(module_file: str, lib_basename: str) -> Path:
    """Get base path for a library, handling the bin directory structure."""
    raw_path = Path(lib_basename)
    if raw_path.suffix == ".so":
        # Strip the suffix so we can append architecture tags in a consistent way.
        return raw_path.with_suffix("")
    return raw_path


def get_consolidated_library(target_arch: str | None = None) -> str | None:
    """Get the consolidated _highnoon_core.so binary path if it exists.

    Args:
        target_arch: Optional target architecture override.

    Returns:
        Path to consolidated .so file if found, None otherwise.
    """
    arch = canonicalize_target_arch(target_arch or os.getenv("VERSO_TARGET_ARCH"))
    arch_bin_dir = _BIN_DIR / arch
    consolidated_path = arch_bin_dir / "_highnoon_core.so"
    if consolidated_path.exists():
        return str(consolidated_path)

    # Fallback: Check local build directory (developer convenience)
    build_path = _NATIVE_DIR / "build" / "_highnoon_core.so"
    if build_path.exists():
        return str(build_path)

    return None


def get_highnoon_core_path(target_arch: str | None = None) -> str:
    """Get the path to the consolidated _highnoon_core.so binary.

    This is the primary function for loading unified quantum ops.
    Unlike get_consolidated_library, this raises RuntimeError if
    the binary is not found.

    Args:
        target_arch: Optional target architecture override.

    Returns:
        Path to the _highnoon_core.so file.

    Raises:
        RuntimeError: If the consolidated binary is not found.
    """
    lib_path = get_consolidated_library(target_arch)
    if lib_path is None:
        arch = canonicalize_target_arch(target_arch or os.getenv("VERSO_TARGET_ARCH"))
        arch_bin_dir = _BIN_DIR / arch
        raise RuntimeError(
            f"HighNoon core binary not found at {arch_bin_dir / '_highnoon_core.so'}. "
            "Run build_secure.sh to compile native ops."
        )
    return lib_path


def resolve_op_library(
    module_file: str,
    lib_basename: str,
    *,
    target_arch: str | None = None,
    prefer_consolidated: bool = True,
) -> str:
    """
    Resolve the shared object path for a Verso custom op.

    PRIORITY ORDER (v2.0 consolidated build):
    1. _highnoon_core.so (consolidated binary containing ALL ops)
    2. *.{arch}.so (architecture-specific individual binary)
    3. *.so (legacy individual binary)

    Args:
        module_file: The __file__ of the calling module (used for context).
        lib_basename: Base name of the library (e.g., "_fused_moe_dispatch_op.so").
        target_arch: Optional target architecture override.
        prefer_consolidated: If True (default), prefer the consolidated binary.

    Returns:
        Path to the resolved .so file.
    """
    arch = canonicalize_target_arch(target_arch or os.getenv("VERSO_TARGET_ARCH"))
    arch_bin_dir = _BIN_DIR / arch

    # PRIORITY 1: Consolidated binary (contains all ops)
    if prefer_consolidated:
        consolidated = get_consolidated_library(target_arch)
        if consolidated:
            return consolidated

    # PRIORITY 2-4: Individual binaries (legacy fallback)
    base_path = _base_path(module_file, lib_basename)
    base_name = base_path.name

    candidates = [
        arch_bin_dir / f"{base_name}.{arch}.so",
        arch_bin_dir / f"{base_name}.so",
        # Fallback to legacy location (in ops dir itself)
        _NATIVE_DIR / "ops" / f"{base_name}.{arch}.so",
        _NATIVE_DIR / "ops" / f"{base_name}.so",
    ]

    resolved = candidates[0]
    fell_back = False
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate
            if candidate != candidates[0]:
                fell_back = True
            break

    if fell_back and arch != DEFAULT_VERSO_TARGET_ARCH:
        logger.warning(
            "[OPS] Falling back to alternative shared object %s for arch=%s. "
            "Rebuild ops for architecture-specific binaries.",
            resolved,
            arch,
        )
    return str(resolved)
