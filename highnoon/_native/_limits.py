# highnoon/_native/_limits.py
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

"""Python-side limit enforcement for HighNoon Language Framework.

This module provides edition-aware limit validation. Limits are enforced
based on the build-time edition:

  - LITE (0):       Free tier with scale limits enforced
  - PRO (1):        Paid tier with no scale limits
  - ENTERPRISE (2): Source code access + no limits

IMPORTANT: These limits are also enforced in the compiled C++ binaries.
For Lite edition, limits cannot be bypassed by modifying this file.
Pro and Enterprise editions have no limits enforced.
"""

import logging
import sys
from typing import Any

log = logging.getLogger(__name__)


# =============================================================================
# EDITION CONSTANTS
# =============================================================================

EDITION_LITE = 0
EDITION_PRO = 1
EDITION_ENTERPRISE = 2

EDITION_NAMES = {
    EDITION_LITE: "LITE",
    EDITION_PRO: "PRO",
    EDITION_ENTERPRISE: "ENTERPRISE",
}

EDITION_DESCRIPTIONS = {
    EDITION_LITE: "Free tier with scale limits (20B params, 5M context)",
    EDITION_PRO: "Pro tier - no scale limits, unlimited performance",
    EDITION_ENTERPRISE: "Enterprise tier - source code access + unlimited",
}


def _detect_edition() -> int:
    """Detect the current edition from the compiled binary.

    Returns the edition code (0=Lite, 1=Pro, 2=Enterprise).
    Defaults to Lite if detection fails.
    """
    try:
        # Try to get edition from compiled binary
        from highnoon._native import _highnoon_core

        edition = getattr(_highnoon_core, "_hn_edition", None)
        if edition is not None:
            return int(edition)
    except ImportError:
        log.debug("Native binary not loaded, defaulting to Lite edition")
    except Exception as e:
        log.debug(f"Edition detection failed: {e}")

    # Default to Lite edition
    return EDITION_LITE


# Cached edition value
_CURRENT_EDITION: int | None = None


def get_edition() -> int:
    """Get the current edition code.

    Returns:
        Edition code: 0=LITE, 1=PRO, 2=ENTERPRISE
    """
    global _CURRENT_EDITION
    if _CURRENT_EDITION is None:
        _CURRENT_EDITION = _detect_edition()
    return _CURRENT_EDITION


def get_edition_name() -> str:
    """Get the current edition name as a string."""
    return EDITION_NAMES.get(get_edition(), "UNKNOWN")


def get_edition_description() -> str:
    """Get a description of the current edition."""
    return EDITION_DESCRIPTIONS.get(get_edition(), "Unknown edition")


def is_unlimited() -> bool:
    """Check if running with unlimited scale (Pro or Enterprise)."""
    return get_edition() >= EDITION_PRO


def is_lite() -> bool:
    """Check if running Lite edition with scale limits."""
    return get_edition() == EDITION_LITE


def is_pro() -> bool:
    """Check if running Pro edition."""
    return get_edition() == EDITION_PRO


def is_enterprise() -> bool:
    """Check if running Enterprise edition."""
    return get_edition() == EDITION_ENTERPRISE


# =============================================================================
# EXCEPTION CLASSES
# =============================================================================


class LimitExceededError(Exception):
    """Raised when a configuration exceeds Lite edition limits.

    This error indicates that the requested model configuration exceeds
    the scale limits of the Lite edition. To use larger configurations,
    upgrade to Pro or Enterprise edition.

    Attributes:
        violations: List of specific limit violations.
    """

    def __init__(self, message: str, violations: list[str] | None = None):
        super().__init__(message)
        self.violations = violations or []


class LicenseError(Exception):
    """Raised when attempting to access enterprise-only features.

    This error indicates that the requested feature or domain module
    requires a Pro or Enterprise license.

    Attributes:
        domain: The domain module that was attempted to access.
    """

    def __init__(self, message: str, domain: str | None = None):
        super().__init__(message)
        self.domain = domain


# =============================================================================
# EDITION-AWARE LIMITS
# =============================================================================

# Lite edition limits (used when is_lite() returns True)
_LITE_MAX_TOTAL_PARAMS = 20_000_000_000
_LITE_MAX_REASONING_BLOCKS = 24
_LITE_MAX_MOE_EXPERTS = 12
_LITE_MAX_CONTEXT_LENGTH = 5_000_000
_LITE_MAX_EMBEDDING_DIM = 4096

# Unlimited values for Pro/Enterprise
_UNLIMITED = sys.maxsize


def _get_limit(lite_value: int) -> int:
    """Get the effective limit based on current edition."""
    if is_unlimited():
        return _UNLIMITED
    return lite_value


# Module-level constants represent the Lite edition limits.
# Use get_effective_limit() for edition-aware limits that respect Pro/Enterprise unlocking.
MAX_TOTAL_PARAMS = _LITE_MAX_TOTAL_PARAMS
MAX_REASONING_BLOCKS = _LITE_MAX_REASONING_BLOCKS
MAX_MOE_EXPERTS = _LITE_MAX_MOE_EXPERTS
MAX_CONTEXT_LENGTH = _LITE_MAX_CONTEXT_LENGTH
MAX_EMBEDDING_DIM = _LITE_MAX_EMBEDDING_DIM


def get_effective_limit(limit_name: str) -> int:
    """Get the effective limit for the current edition.

    Args:
        limit_name: One of 'max_total_params', 'max_reasoning_blocks',
                    'max_moe_experts', 'max_context_length', 'max_embedding_dim'

    Returns:
        The limit value for the current edition. Returns sys.maxsize
        for Pro/Enterprise editions.
    """
    limits = {
        "max_total_params": _LITE_MAX_TOTAL_PARAMS,
        "max_reasoning_blocks": _LITE_MAX_REASONING_BLOCKS,
        "max_moe_experts": _LITE_MAX_MOE_EXPERTS,
        "max_context_length": _LITE_MAX_CONTEXT_LENGTH,
        "max_embedding_dim": _LITE_MAX_EMBEDDING_DIM,
    }
    lite_value = limits.get(limit_name.lower())
    if lite_value is None:
        raise ValueError(f"Unknown limit: {limit_name}")
    return _get_limit(lite_value)


# Domain modules locked in Lite edition (require Pro/Enterprise license)
LOCKED_DOMAINS = frozenset(
    {
        "chemistry",  # Molecular/material science modules
        "physics",  # Physics simulation modules
        "inverse_design",  # Inverse design/optimization modules
        "hardware_control",  # Hardware control/MPC modules
        "graph_learning",  # Advanced graph learning modules
    }
)


def is_domain_unlocked(domain: str) -> bool:
    """Check if a domain module is unlocked for the current edition.

    Args:
        domain: Domain module name to check.

    Returns:
        True if the domain is accessible, False if locked.
    """
    if is_unlimited():
        return True  # Pro/Enterprise have all domains unlocked
    return domain.lower() not in LOCKED_DOMAINS


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_model_config(config: Any) -> None:
    """Validate model configuration against edition limits.

    For Lite edition, checks the configuration object for any values that
    exceed the scale limits. For Pro and Enterprise editions, this function
    is a no-op (no limits enforced).

    Args:
        config: Configuration object with model parameters.

    Raises:
        LimitExceededError: If Lite edition and configuration exceeds limits.
    """
    # Pro and Enterprise have no limits - skip validation entirely
    if is_unlimited():
        log.debug(f"Running {get_edition_name()} edition - no limit validation")
        return

    violations = []

    # Check reasoning blocks
    num_blocks = getattr(config, "num_reasoning_blocks", None)
    if num_blocks is not None and num_blocks > _LITE_MAX_REASONING_BLOCKS:
        violations.append(
            f"num_reasoning_blocks: {num_blocks} > {_LITE_MAX_REASONING_BLOCKS} (max for Lite)"
        )

    # Check MoE experts
    num_experts = getattr(config, "num_moe_experts", None)
    if num_experts is not None and num_experts > _LITE_MAX_MOE_EXPERTS:
        violations.append(
            f"num_moe_experts: {num_experts} > {_LITE_MAX_MOE_EXPERTS} (max for Lite)"
        )

    # Check embedding dimension
    embedding_dim = getattr(config, "embedding_dim", None)
    if embedding_dim is not None and embedding_dim > _LITE_MAX_EMBEDDING_DIM:
        violations.append(
            f"embedding_dim: {embedding_dim} > {_LITE_MAX_EMBEDDING_DIM} (max for Lite)"
        )

    # Check context length / max sequence length
    max_seq = getattr(config, "max_seq_length", None)
    if max_seq is not None and max_seq > _LITE_MAX_CONTEXT_LENGTH:
        violations.append(f"max_seq_length: {max_seq} > {_LITE_MAX_CONTEXT_LENGTH} (max for Lite)")

    # Raise error if any violations found
    if violations:
        message = (
            "Configuration exceeds Lite edition limits:\n"
            "  - " + "\n  - ".join(violations) + "\n\n"
            "To use larger configurations, upgrade to Pro or Enterprise:\n"
            "  https://versoindustries.com/upgrade\n\n"
            "For questions, visit: https://www.versoindustries.com/messages"
        )
        log.error(message)
        raise LimitExceededError(message, violations)


def warn_near_limits(config: Any, threshold: float = 0.8) -> list[str]:
    """Log warnings for configurations that are approaching limits.

    This function checks if any configuration values are at or above
    the warning threshold (default 80% of limit) and logs warnings.

    Args:
        config: Configuration object with model parameters.
        threshold: Fraction of limit at which to warn (default: 0.8).

    Returns:
        List of warning messages generated.
    """
    warnings = []

    # Check reasoning blocks
    num_blocks = getattr(config, "num_reasoning_blocks", None)
    if num_blocks is not None:
        usage = num_blocks / MAX_REASONING_BLOCKS
        if usage >= threshold and num_blocks <= MAX_REASONING_BLOCKS:
            msg = (
                f"num_reasoning_blocks={num_blocks} is at {usage:.0%} of "
                f"Lite limit ({MAX_REASONING_BLOCKS})"
            )
            warnings.append(msg)
            log.warning(f"[NEAR LIMIT] {msg}")

    # Check MoE experts
    num_experts = getattr(config, "num_moe_experts", None)
    if num_experts is not None:
        usage = num_experts / MAX_MOE_EXPERTS
        if usage >= threshold and num_experts <= MAX_MOE_EXPERTS:
            msg = (
                f"num_moe_experts={num_experts} is at {usage:.0%} of "
                f"Lite limit ({MAX_MOE_EXPERTS})"
            )
            warnings.append(msg)
            log.warning(f"[NEAR LIMIT] {msg}")

    # Check embedding dimension
    embedding_dim = getattr(config, "embedding_dim", None)
    if embedding_dim is not None:
        usage = embedding_dim / MAX_EMBEDDING_DIM
        if usage >= threshold and embedding_dim <= MAX_EMBEDDING_DIM:
            msg = (
                f"embedding_dim={embedding_dim} is at {usage:.0%} of "
                f"Lite limit ({MAX_EMBEDDING_DIM})"
            )
            warnings.append(msg)
            log.warning(f"[NEAR LIMIT] {msg}")

    # Check context length
    max_seq = getattr(config, "max_seq_length", None)
    if max_seq is not None:
        usage = max_seq / MAX_CONTEXT_LENGTH
        if usage >= threshold and max_seq <= MAX_CONTEXT_LENGTH:
            msg = (
                f"max_seq_length={max_seq} is at {usage:.0%} of "
                f"Lite limit ({MAX_CONTEXT_LENGTH:,})"
            )
            warnings.append(msg)
            log.warning(f"[NEAR LIMIT] {msg}")

    return warnings


def estimate_param_count(config: Any) -> int:
    """Estimate the total parameter count for a model configuration.

    This is a rough estimate based on the configuration parameters.
    The actual count may vary based on the specific architecture.

    Args:
        config: Configuration object with model parameters.

    Returns:
        Estimated total parameter count.
    """
    vocab_size = getattr(config, "vocab_size", 32000)
    embedding_dim = getattr(config, "embedding_dim", 768)
    num_blocks = getattr(config, "num_reasoning_blocks", 4)
    num_experts = getattr(config, "num_moe_experts", 8)

    # Embedding parameters
    embed_params = vocab_size * embedding_dim

    # Per-block parameters (rough estimate)
    # Attention: 4 * d^2 (Q, K, V, O projections)
    # FFN: 8 * d^2 (standard 4x expansion)
    block_params = (4 + 8) * embedding_dim * embedding_dim

    # MoE adds expert parameters
    if num_experts > 1:
        block_params += num_experts * 8 * embedding_dim * embedding_dim

    # Total
    total = embed_params + (num_blocks * block_params)

    return total


def validate_param_count(config: Any) -> None:
    """Validate that estimated parameter count is within limits.

    Args:
        config: Configuration object with model parameters.

    Raises:
        LimitExceededError: If estimated parameters exceed limit.
    """
    estimated = estimate_param_count(config)

    if estimated > MAX_TOTAL_PARAMS:
        message = (
            f"Estimated parameter count ({estimated:,}) exceeds Lite edition "
            f"limit ({MAX_TOTAL_PARAMS:,} / 20B parameters).\n\n"
            "Consider reducing:\n"
            "  - embedding_dim\n"
            "  - num_reasoning_blocks\n"
            "  - num_moe_experts\n\n"
            "Or upgrade to Enterprise: https://versoindustries.com/enterprise"
        )
        raise LimitExceededError(message)


def check_enterprise_license(domain: str) -> bool:
    """Check if enterprise license is available for a domain module.

    In the Lite edition, all locked domains return False.
    Enterprise binaries override this function to check actual licenses.

    Args:
        domain: Domain module name to check.

    Returns:
        True if the domain is accessible, False if locked.
    """
    # Pro/Enterprise have all domains unlocked
    if is_unlimited():
        return True
    if domain.lower() in LOCKED_DOMAINS:
        log.debug(f"Domain '{domain}' is locked in Lite edition")
        return False
    return True


def require_enterprise(domain: str) -> None:
    """Require Pro/Enterprise license for a domain, raising if not available.

    Args:
        domain: Domain module that requires Pro or Enterprise license.

    Raises:
        LicenseError: If Pro/Enterprise license is not available.
    """
    if not check_enterprise_license(domain):
        raise LicenseError(
            f"The '{domain}' module requires a Pro or Enterprise license.\n\n"
            f"This module is part of the full HSMN-Architecture and is not\n"
            f"available in the Lite edition of HighNoon Language Framework.\n\n"
            f"Upgrade to Pro or Enterprise: https://versoindustries.com/upgrade\n"
            f"For questions, visit: https://www.versoindustries.com/messages",
            domain=domain,
        )


def get_limits() -> dict:
    """Get current edition limits as a dictionary.

    Returns:
        Dictionary with edition info and effective limits.
        For Pro/Enterprise, limits are sys.maxsize (effectively unlimited).
    """
    return {
        "edition": get_edition_name(),
        "edition_code": get_edition(),
        "unlimited": is_unlimited(),
        "max_total_params": get_effective_limit("max_total_params"),
        "max_reasoning_blocks": get_effective_limit("max_reasoning_blocks"),
        "max_moe_experts": get_effective_limit("max_moe_experts"),
        "max_context_length": get_effective_limit("max_context_length"),
        "max_embedding_dim": get_effective_limit("max_embedding_dim"),
        "locked_domains": [] if is_unlimited() else list(LOCKED_DOMAINS),
    }
