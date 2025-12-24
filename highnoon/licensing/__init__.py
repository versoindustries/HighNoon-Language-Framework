# highnoon/licensing/__init__.py
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

"""HighNoon Licensing Module.

This module provides license validation and edition information for the
HighNoon Language Framework.

EDITION TIERS:
    LITE (0):       Free tier with scale limits enforced
                    - 20B max parameters
                    - 5M max context length
                    - 24 max reasoning blocks
                    - 12 max MoE experts
                    - Language module only

    PRO (1):        Paid tier with no scale limits
                    - Unlimited parameters
                    - Unlimited context length
                    - Unlimited reasoning blocks
                    - Unlimited MoE experts
                    - All domain modules unlocked
                    - Pre-compiled binary

    ENTERPRISE (2): Full source code access + no limits
                    - All Pro features
                    - Full source code access
                    - Custom architecture support
                    - Dedicated support channel
                    - Enterprise license agreement

For licensing inquiries: sales@versoindustries.com
"""

from highnoon._native._limits import (  # Edition constants; Edition detection; Limit functions; Exceptions
    EDITION_DESCRIPTIONS,
    EDITION_ENTERPRISE,
    EDITION_LITE,
    EDITION_NAMES,
    EDITION_PRO,
    LicenseError,
    LimitExceededError,
    get_edition,
    get_edition_description,
    get_edition_name,
    get_effective_limit,
    get_limits,
    is_domain_unlocked,
    is_enterprise,
    is_lite,
    is_pro,
    is_unlimited,
)

__all__ = [
    # Edition constants
    "EDITION_LITE",
    "EDITION_PRO",
    "EDITION_ENTERPRISE",
    "EDITION_NAMES",
    "EDITION_DESCRIPTIONS",
    # Edition detection
    "get_edition",
    "get_edition_name",
    "get_edition_description",
    "is_unlimited",
    "is_lite",
    "is_pro",
    "is_enterprise",
    # Limit functions
    "get_effective_limit",
    "get_limits",
    "is_domain_unlocked",
    # Exceptions
    "LimitExceededError",
    "LicenseError",
    # Info functions
    "get_edition_info",
    "print_edition_banner",
]


def get_edition_info() -> dict:
    """Get comprehensive edition information.

    Returns:
        Dictionary containing edition details, limits, and licensing info.
    """
    edition = get_edition()
    limits = get_limits()

    info = {
        "edition": get_edition_name(),
        "edition_code": edition,
        "description": get_edition_description(),
        "unlimited": is_unlimited(),
        "limits": limits,
        "features": {
            "max_parameters": "Unlimited" if is_unlimited() else "20B",
            "max_context_length": "Unlimited" if is_unlimited() else "5M tokens",
            "max_reasoning_blocks": "Unlimited" if is_unlimited() else "24",
            "max_moe_experts": "Unlimited" if is_unlimited() else "12",
            "domain_modules": "All" if is_unlimited() else "Language only",
            "source_access": is_enterprise(),
        },
    }

    if is_lite():
        info["upgrade_url"] = "https://versoindustries.com/upgrade"
    elif is_pro():
        info["support"] = "priority@versoindustries.com"
    elif is_enterprise():
        info["support"] = "enterprise@versoindustries.com"
        info["source_license"] = "Enterprise Source License Agreement"

    return info


def print_edition_banner() -> None:
    """Print a banner showing current edition information."""
    get_edition_name()

    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║                    HighNoon Language Framework                           ║")
    print("║                         Verso Industries 2025                            ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")

    if is_lite():
        print("║  📦 EDITION: LITE                                                        ║")
        print("║  Scale limits enforced (20B params, 5M context)                          ║")
        print("║  Upgrade: https://versoindustries.com/upgrade                            ║")
    elif is_pro():
        print("║  🚀 EDITION: PRO                                                         ║")
        print("║  Unlimited scale - no limits enforced                                    ║")
        print("║  Priority support: priority@versoindustries.com                          ║")
    elif is_enterprise():
        print("║  🏢 EDITION: ENTERPRISE                                                  ║")
        print("║  Full source code access + unlimited scale                               ║")
        print("║  Enterprise support: enterprise@versoindustries.com                      ║")

    print("╚══════════════════════════════════════════════════════════════════════════╝")
