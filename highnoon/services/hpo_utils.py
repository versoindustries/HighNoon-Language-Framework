"""Utility functions for HPO (Hyperparameter Optimization).

This module provides shared utilities used by HPO components.
"""

from typing import Any

import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization.

    This function recursively traverses dictionaries and lists, converting
    numpy integer and floating-point types to Python native int/float types.

    Args:
        obj: Object to convert (can be dict, list, scalar, or numpy type)

    Returns:
        Object with numpy types converted to Python native types

    Examples:
        >>> convert_numpy_types(np.int64(42))
        42
        >>> convert_numpy_types({"a": np.float64(3.14)})
        {'a': 3.14}
        >>> convert_numpy_types([np.int32(1), np.int32(2)])
        [1, 2]
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
