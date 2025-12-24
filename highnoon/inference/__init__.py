# highnoon/inference/__init__.py
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

"""HighNoon Inference Module - Optimized generation utilities.

This module provides inference optimizations including:
- QSGGenerator: 50-100x faster parallel generation via Quantum Superposition
- StatefulInferenceWrapper: O(1) per-token generation with SSM state caching
- SpeculativeGenerator: 2-3x faster generation with draft models
- StreamingInferenceWrapper: Infinite context via state compression
"""

# QSG: Quantum Superposition Generation (default for new code)
from highnoon.inference.qsg_generator import QSGConfig, QSGGenerator
from highnoon.inference.stateful_wrapper import StatefulInferenceWrapper

# Phase 13.3: Speculative Decoding - import conditionally
SpeculativeGenerator = None
SpeculativeDecodingConfig = None

try:
    from highnoon.inference.speculative import SpeculativeDecodingConfig, SpeculativeGenerator
except ImportError:
    pass

# Phase 13.8: KV-Free State Streaming - import conditionally
StreamingInferenceWrapper = None
StreamingState = None

try:
    from highnoon.inference.streaming import StreamingInferenceWrapper, StreamingState
except ImportError:
    pass

__all__ = [
    # QSG (primary)
    "QSGGenerator",
    "QSGConfig",
    # Stateful AR
    "StatefulInferenceWrapper",
    # Speculative
    "SpeculativeGenerator",
    "SpeculativeDecodingConfig",
    # Streaming
    "StreamingInferenceWrapper",
    "StreamingState",
]
