# highnoon/tokenization/__init__.py
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

"""Tokenization module for HighNoon Language Framework.

This module provides tokenization utilities for the HighNoon language models,
including the Quantum Wavelet Text Tokenizer (QWT) which integrates with
the native C++ QWT operations for high-performance text processing.

Phase 10.2: Added SuperwordMerger for semantic n-gram grouping.
Phase 10.5: Added ByteStreamTokenizer for token-free processing.

Example:
    >>> from highnoon.tokenization import QWTTextTokenizer
    >>> tokenizer = QWTTextTokenizer(vocab_size=512, model_max_length=256)
    >>> tokens = tokenizer("Hello, world!", return_tensors="tf")
"""

from highnoon.tokenization.adaptive_qwt_tokenizer import AdaptiveQWTTokenizer
from highnoon.tokenization.byte_stream_tokenizer import ByteStreamTokenizer
from highnoon.tokenization.qwt_text_tokenizer import QWTTextTokenizer
from highnoon.tokenization.superword_merger import (
    SuperwordEntry,
    SuperwordMerger,
    SuperwordMergerConfig,
)

__all__ = [
    "AdaptiveQWTTokenizer",
    "QWTTextTokenizer",
    "ByteStreamTokenizer",
    "SuperwordMerger",
    "SuperwordMergerConfig",
    "SuperwordEntry",
]

