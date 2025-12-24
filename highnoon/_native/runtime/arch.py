# highnoon/_native/runtime/arch.py
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

"""Architecture detection and canonicalization for native op loading."""

from __future__ import annotations

import os

DEFAULT_VERSO_TARGET_ARCH = "x86_64"
_ARCH_ALIASES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "x64": "x86_64",
    "arm64": "arm64",
    "aarch64": "arm64",
    "armv8": "arm64",
}
SUPPORTED_VERSO_ARCHES = tuple(sorted(set(_ARCH_ALIASES.values())))
_PLATFORM_ARCH_HINTS = {
    "marlin_arm": "arm64",
}


class UnsupportedTargetArch(ValueError):
    """Raised when an unsupported `VERSO_TARGET_ARCH` value is requested."""


def _normalize_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return cleaned.replace("-", "_")


def canonicalize_target_arch(value: str | None, *, strict: bool = False) -> str:
    """
    Return the canonical Verso target architecture string.

    Parameters
    ----------
    value:
        Raw string that may include aliases (e.g., ``amd64`` → ``x86_64``).
        ``None`` falls back to :data:`DEFAULT_VERSO_TARGET_ARCH`.
    strict:
        When ``True``, invalid values raise :class:`UnsupportedTargetArch`.
        When ``False`` (default) invalid values fall back to the default arch.
    """

    normalized = _normalize_value(value)
    if normalized is None:
        return DEFAULT_VERSO_TARGET_ARCH
    canonical = _ARCH_ALIASES.get(normalized)
    if canonical:
        return canonical
    if strict:
        raise UnsupportedTargetArch(
            f"Unsupported VERSO target architecture '{value}'. "
            f"Supported: {', '.join(SUPPORTED_VERSO_ARCHES)}"
        )
    return DEFAULT_VERSO_TARGET_ARCH


def _platform_hint() -> str | None:
    platform_value = os.getenv("HSMN_TARGET_PLATFORM")
    normalized = _normalize_value(platform_value)
    if not normalized:
        return None
    hinted = _PLATFORM_ARCH_HINTS.get(normalized)
    if not hinted:
        return None
    return canonicalize_target_arch(hinted)


def get_verso_target_arch() -> str:
    """
    Resolve the effective target architecture using environment hints.

    Precedence:
    1. ``VERSO_TARGET_ARCH`` (if set)
    2. ``HSMN_TARGET_PLATFORM`` mapped via :data:`_PLATFORM_ARCH_HINTS`
    3. :data:`DEFAULT_VERSO_TARGET_ARCH`
    """

    env_value = os.getenv("VERSO_TARGET_ARCH")
    if env_value:
        return canonicalize_target_arch(env_value)
    platform_value = _platform_hint()
    if platform_value:
        return platform_value
    return DEFAULT_VERSO_TARGET_ARCH


def is_arm64(value: str | None = None) -> bool:
    """Convenience helper to check whether the requested arch resolves to ARM64."""

    return canonicalize_target_arch(value) == "arm64"
