# highnoon/data/__init__.py
# Copyright 2025 Verso Industries
#
# HighNoon Language Framework - Data Module
#
# This module provides data loading, preprocessing, and dataset management
# for language model training.

from highnoon.data.preprocessors import BasePreprocessor, InstructionPreprocessor, TextPreprocessor
from highnoon.data.sequence_packer import (
    MAX_CONTEXT_WINDOW,
    MIN_CONTEXT_WINDOW,
    PackedBatch,
    PackedSequence,
    ParallelTokenizer,
    StreamingSequencePacker,
)
from highnoon.data.tfrecord_cache import TFRecordCache

__all__ = [
    "BasePreprocessor",
    "TextPreprocessor",
    "InstructionPreprocessor",
    # Phase 200: Enterprise streaming
    "StreamingSequencePacker",
    "PackedSequence",
    "PackedBatch",
    "ParallelTokenizer",
    "TFRecordCache",
    "MIN_CONTEXT_WINDOW",
    "MAX_CONTEXT_WINDOW",
]
