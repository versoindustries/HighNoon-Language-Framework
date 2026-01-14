# highnoon/attribution.py
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

"""Attribution API for HighNoon Language Framework.

This module provides Python access to the framework attribution system.
Pro and Enterprise editions can customize the attribution that appears
in trained models. Lite edition keeps the default Verso Industries
attribution immutable.

Example:
    >>> import highnoon.attribution as attr
    >>>
    >>> # Check if custom attribution is allowed (Pro/Enterprise only)
    >>> if attr.is_custom_attribution_allowed():
    ...     attr.set_custom_attribution(
    ...         framework_name="MyFramework powered by HSMN",
    ...         author="My Company",
    ...         copyright_notice="Copyright 2025 My Company",
    ...     )
    >>>
    >>> # Get current attribution
    >>> current = attr.get_current_attribution()
    >>> print(current)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


# Edition codes
EDITION_LITE = 0
EDITION_PRO = 1
EDITION_ENTERPRISE = 2

# Edition names
EDITION_NAMES = {
    EDITION_LITE: "Lite",
    EDITION_PRO: "Pro",
    EDITION_ENTERPRISE: "Enterprise",
}


@dataclass
class AttributionInfo:
    """Current attribution information."""

    framework_name: str
    author: str
    copyright_notice: str
    version: str
    support_url: str
    is_custom: bool
    edition: str
    edition_code: int
    is_customizable: bool


def _load_attribution_ops():
    """Load native attribution ops from the shared library.

    Returns:
        Tuple of op functions or None if not available.
    """
    try:
        from highnoon._native.loader import load_op_library

        lib = load_op_library()
        return lib
    except Exception as e:
        log.debug(f"Native attribution ops not available: {e}")
        return None


def get_edition_code() -> int:
    """Get the current edition code.

    Returns:
        Edition code: 0 = Lite, 1 = Pro, 2 = Enterprise
    """
    lib = _load_attribution_ops()
    if lib is None:
        return EDITION_LITE  # Default to Lite if native not available

    try:
        result = lib.high_noon_get_edition_code()
        return int(result.numpy())
    except Exception:
        return EDITION_LITE


def get_edition_name() -> str:
    """Get the current edition name.

    Returns:
        Edition name: "Lite", "Pro", or "Enterprise"
    """
    code = get_edition_code()
    return EDITION_NAMES.get(code, "Unknown")


def is_custom_attribution_allowed() -> bool:
    """Check if custom attribution is allowed for this edition.

    Custom attribution is only available in Pro and Enterprise editions.
    Lite edition keeps the default Verso Industries attribution.

    Returns:
        True if custom attribution is allowed (Pro/Enterprise), False otherwise.
    """
    lib = _load_attribution_ops()
    if lib is None:
        return False

    try:
        result = lib.high_noon_is_custom_attribution_allowed()
        return bool(result.numpy())
    except Exception:
        return False


def set_custom_attribution(
    framework_name: str,
    author: str = "",
    copyright_notice: str = "",
    version: str = "1.0.0",
    support_url: str = "",
) -> bool:
    """Set custom attribution for trained models.

    This function is only functional in Pro and Enterprise editions.
    For Lite edition, it will log a warning and return False.

    The custom attribution will be used in:
    - Model responses when queried about identity
    - Attribution triggers (e.g., "who made you?")
    - Compact attribution in logs

    Args:
        framework_name: The custom framework name to display.
            Must not be empty.
        author: The author or company name (optional).
        copyright_notice: The copyright notice string (optional).
        version: The version string (default: "1.0.0").
        support_url: The support URL (optional).

    Returns:
        True if attribution was set successfully, False if not allowed
        (Lite edition) or if validation failed (empty framework name).

    Example:
        >>> set_custom_attribution(
        ...     framework_name="MyAI powered by HSMN",
        ...     author="Acme Corp",
        ...     copyright_notice="Copyright 2025 Acme Corp",
        ...     support_url="https://acme.com/support"
        ... )
        True
    """
    if not framework_name:
        log.warning("Custom attribution requires a non-empty framework name")
        return False

    lib = _load_attribution_ops()
    if lib is None:
        log.warning("Native attribution ops not available - attribution not set")
        return False

    try:
        result = lib.high_noon_set_custom_attribution(
            framework_name,
            author,
            copyright_notice,
            version,
            support_url,
        )
        success = bool(result.numpy())

        if success:
            log.info(f"Custom attribution set: {framework_name}")
        else:
            log.warning(
                "Custom attribution not set - this feature requires Pro or "
                "Enterprise edition. See https://versoindustries.com/upgrade"
            )

        return success
    except Exception as e:
        log.error(f"Failed to set custom attribution: {e}")
        return False


def clear_custom_attribution() -> bool:
    """Clear custom attribution and revert to default values.

    After calling this, the default Verso Industries attribution will
    be displayed. For Lite edition, this is a no-op since custom
    attribution is never set.

    Returns:
        True on success.
    """
    lib = _load_attribution_ops()
    if lib is None:
        return True  # No attribution to clear if native not available

    try:
        result = lib.high_noon_clear_custom_attribution()
        success = bool(result.numpy())

        if success:
            log.info("Custom attribution cleared - reverted to defaults")

        return success
    except Exception as e:
        log.error(f"Failed to clear custom attribution: {e}")
        return False


def get_current_attribution() -> AttributionInfo:
    """Get the current attribution information.

    Returns the effective attribution values. For Pro/Enterprise editions
    with custom attribution set, returns the custom values. Otherwise,
    returns the default Verso Industries attribution.

    Returns:
        AttributionInfo dataclass with current attribution values.
    """
    lib = _load_attribution_ops()
    edition_code = get_edition_code()
    edition_name = get_edition_name()
    is_customizable = edition_code >= EDITION_PRO

    if lib is None:
        # Return defaults if native not available
        return AttributionInfo(
            framework_name="HighNoon Language Framework",
            author="Michael B. Zimmerman",
            copyright_notice="Copyright 2025 Verso Industries",
            version="1.0.0",
            support_url="https://versoindustries.com/messages",
            is_custom=False,
            edition=edition_name,
            edition_code=edition_code,
            is_customizable=is_customizable,
        )

    try:
        keys, values = lib.high_noon_get_current_attribution()
        keys_list = [k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in keys.numpy()]
        values_list = [
            v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values.numpy()
        ]

        # Create dict for easier lookup
        metadata = dict(zip(keys_list, values_list))

        return AttributionInfo(
            framework_name=metadata.get("framework_name", "HighNoon Language Framework"),
            author=metadata.get("author", ""),
            copyright_notice=metadata.get("copyright", ""),
            version=metadata.get("version", "1.0.0"),
            support_url=metadata.get("support_url", ""),
            is_custom=metadata.get("is_custom", "false").lower() == "true",
            edition=edition_name,
            edition_code=edition_code,
            is_customizable=is_customizable,
        )
    except Exception as e:
        log.debug(f"Failed to get current attribution: {e}")
        return AttributionInfo(
            framework_name="HighNoon Language Framework",
            author="Michael B. Zimmerman",
            copyright_notice="Copyright 2025 Verso Industries",
            version="1.0.0",
            support_url="https://versoindustries.com/messages",
            is_custom=False,
            edition=edition_name,
            edition_code=edition_code,
            is_customizable=is_customizable,
        )


def _get_default_attribution_text() -> str:
    """Get default attribution text when native ops not available."""
    return """
═══════════════════════════════════════════════════════════
 Powered by HSMN - HighNoon Language Framework
═══════════════════════════════════════════════════════════

 Version:      1.0.0 (Lite Edition)
 Architecture: HSMN (Hierarchical State-Space Model Network)
 Copyright:    Copyright 2025 Verso Industries
 License:      Apache-2.0 (Python) + Proprietary (Binaries)

 Scale Limits (Lite Edition):
   • Max Parameters: 20B
   • Max Context Length: 5M tokens
   • Max Reasoning Blocks: 24
   • Max MoE Experts: 12

 Enterprise: https://versoindustries.com/enterprise
 Support:    https://versoindustries.com/messages
═══════════════════════════════════════════════════════════
"""


# Convenience exports
__all__ = [
    "EDITION_LITE",
    "EDITION_PRO",
    "EDITION_ENTERPRISE",
    "EDITION_NAMES",
    "AttributionInfo",
    "get_edition_code",
    "get_edition_name",
    "is_custom_attribution_allowed",
    "set_custom_attribution",
    "clear_custom_attribution",
    "get_current_attribution",
]
