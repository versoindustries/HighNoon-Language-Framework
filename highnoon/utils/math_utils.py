# highnoon/utils/math_utils.py
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

"""Mathematical utilities for HighNoon framework.

This module provides shared mathematical helper functions used across
the codebase, particularly for tensor decomposition and optimization.
"""

import math


def factorize_for_tt(dim: int) -> list[int]:
    """Factorize dimension for Tensor-Train layer shape optimization.

    Finds two factors of dim that are as close to sqrt(dim) as possible,
    which provides optimal TT-rank efficiency for weight matrices.

    Args:
        dim: The dimension to factorize (must be positive integer).

    Returns:
        A list of two integers [a, b] where a * b >= dim:
        - For valid dims: Returns [factor, dim // factor] where factor is
          the largest factor <= sqrt(dim)
        - For dim <= 0: Returns [1, dim] (identity case)
        - For prime dims: Returns [1, dim] (no factorization possible)

    Examples:
        >>> factorize_for_tt(512)
        [16, 32]
        >>> factorize_for_tt(1024)
        [32, 32]
        >>> factorize_for_tt(17)  # Prime
        [1, 17]
        >>> factorize_for_tt(0)
        [1, 0]
    """
    if dim <= 0:
        return [1, dim]

    s = int(math.sqrt(dim))

    # Find largest factor <= sqrt(dim)
    while s > 1 and dim % s != 0:
        s -= 1

    # If no factor found, try finding smallest factor > 1
    if s == 1:
        for i in range(2, int(dim**0.5) + 2):
            if dim % i == 0:
                s = i
                break

    return [s, dim // s] if s > 1 else [1, dim]


__all__ = ["factorize_for_tt"]
