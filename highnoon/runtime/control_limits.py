# highnoon/runtime/control_limits.py
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

"""Control limits for Hamiltonian evolution dynamics.

This module defines bounds and constraints for the Time Crystal block's
evolution time parameter, ensuring stable and safe Hamiltonian integration.

The limits are designed to:
- Prevent numerical instability from too-large time steps
- Ensure minimum precision with non-zero lower bounds
- Limit step-to-step variation for smooth training dynamics
- Support hardware-specific constraints when running on quantum devices

Example:
    >>> limits = get_evolution_time_limits()
    >>> print(f"Valid range: [{limits.min_value}, {limits.max_value}]")
    Valid range: [1e-06, 0.1]
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class EvolutionTimeLimits:
    """Configuration for evolution time bounds and step constraints.

    Attributes:
        min_value: Minimum allowed evolution time. Must be positive to ensure
            non-trivial dynamics. Default: 1e-6.
        max_value: Maximum allowed evolution time. Larger values increase
            integration error but allow faster state changes. Default: 0.1.
        initial_value: Starting evolution time for newly created blocks.
            Should be between min_value and max_value. Default: 1e-4.
        hardware_relative_step: Maximum relative change per step as a fraction
            of current value (e.g., 0.5 = 50% max change). Default: 0.5.
        hardware_absolute_step: Maximum absolute change per step regardless
            of current value. Default: 0.01.

    Note:
        When running on quantum hardware, these limits may be overridden
        by hardware-specific constraints from the environment.
    """

    min_value: float = 1e-6
    max_value: float = 0.1
    initial_value: float = 1e-4
    hardware_relative_step: float = 0.5
    hardware_absolute_step: float = 0.01

    def __post_init__(self) -> None:
        """Validate limit consistency."""
        if self.min_value <= 0:
            raise ValueError(f"min_value must be positive, got {self.min_value}")
        if self.max_value <= self.min_value:
            raise ValueError(f"max_value ({self.max_value}) must be > min_value ({self.min_value})")
        if not (self.min_value <= self.initial_value <= self.max_value):
            raise ValueError(
                f"initial_value ({self.initial_value}) must be in "
                f"[{self.min_value}, {self.max_value}]"
            )
        if self.hardware_relative_step <= 0:
            raise ValueError(
                f"hardware_relative_step must be positive, got {self.hardware_relative_step}"
            )
        if self.hardware_absolute_step <= 0:
            raise ValueError(
                f"hardware_absolute_step must be positive, got {self.hardware_absolute_step}"
            )


# Cache for default limits
_default_limits: EvolutionTimeLimits | None = None


def get_evolution_time_limits() -> EvolutionTimeLimits:
    """Get evolution time limits, potentially from environment configuration.

    The function checks for environment variable overrides in the format:
    - HIGHNOON_EVOLUTION_TIME_MIN: Override min_value
    - HIGHNOON_EVOLUTION_TIME_MAX: Override max_value
    - HIGHNOON_EVOLUTION_TIME_INIT: Override initial_value

    Returns:
        EvolutionTimeLimits instance with appropriate bounds.

    Example:
        >>> limits = get_evolution_time_limits()
        >>> # Use in TimeCrystalBlock
        >>> evolution_time = tf.clip_by_value(
        ...     raw_time, limits.min_value, limits.max_value
        ... )
    """
    global _default_limits

    if _default_limits is not None:
        return _default_limits

    # Check for environment overrides
    min_val = float(os.getenv("HIGHNOON_EVOLUTION_TIME_MIN", "1e-6"))
    max_val = float(os.getenv("HIGHNOON_EVOLUTION_TIME_MAX", "0.1"))
    init_val = float(os.getenv("HIGHNOON_EVOLUTION_TIME_INIT", "1e-4"))
    rel_step = float(os.getenv("HIGHNOON_EVOLUTION_RELATIVE_STEP", "0.5"))
    abs_step = float(os.getenv("HIGHNOON_EVOLUTION_ABSOLUTE_STEP", "0.01"))

    _default_limits = EvolutionTimeLimits(
        min_value=min_val,
        max_value=max_val,
        initial_value=init_val,
        hardware_relative_step=rel_step,
        hardware_absolute_step=abs_step,
    )

    return _default_limits


def reset_evolution_time_limits() -> None:
    """Reset cached limits (useful for testing with different environments)."""
    global _default_limits
    _default_limits = None


__all__ = [
    "EvolutionTimeLimits",
    "get_evolution_time_limits",
    "reset_evolution_time_limits",
]
