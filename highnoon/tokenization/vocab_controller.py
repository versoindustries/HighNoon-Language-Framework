# highnoon/tokenization/vocab_controller.py
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

"""Intelligent Vocabulary Controller for HighNoon Framework.

This module provides the IntelligentVocabController, which manages the
vocabulary sizing flow from dataset → tokenizer → model. It bridges the
gap between static vocabulary configuration and dynamic codebook learning.

Key Features:
    - Automatic codebook training from corpus samples
    - Pre-trained codebook loading for common datasets
    - Consistent vocab_size reporting to model builder
    - Integration with unified smart tuner for HPO

Architecture:
    ┌────────────────────────────────────────────────────────────────────┐
    │                 IntelligentVocabController                         │
    ├────────────────────────────────────────────────────────────────────┤
    │  Dataset/Curriculum ─→ Sample Corpus ─→ Train Codebook            │
    │                             │                                      │
    │  Pre-trained Codebook ─────┴─→ effective_vocab_size ─→ Model      │
    └────────────────────────────────────────────────────────────────────┘

Example:
    >>> controller = IntelligentVocabController(model_max_length=4096)
    >>> controller.train_codebook(corpus_texts, min_freq=10)
    >>> vocab_size = controller.effective_vocab_size  # Use for model
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from highnoon.tokenization.adaptive_qwt_tokenizer import AdaptiveQWTTokenizer

logger = logging.getLogger(__name__)


# Pre-trained codebook registry for common datasets
PRETRAINED_CODEBOOKS = {
    "verso-baseline": "artifacts/codebooks/verso_baseline.json",
    "openwebtext": "artifacts/codebooks/openwebtext.json",
    "starcoder": "artifacts/codebooks/starcoder.json",
}


@dataclass
class VocabControllerConfig:
    """Configuration for IntelligentVocabController.

    Attributes:
        model_max_length: Maximum sequence length for the model.
        min_ngram_freq: Minimum n-gram frequency for codebook learning.
        sample_size: Number of texts to sample for codebook training.
        byte_offset: Byte offset for tokenizer (default: 32).
        enable_thinking_tokens: Enable thinking token injection.
        min_ngram_size: Minimum n-gram size for learning.
        max_ngram_size: Maximum n-gram size for learning.
    """

    model_max_length: int = 4096
    min_ngram_freq: int = 10
    sample_size: int = 10000
    byte_offset: int = 32
    enable_thinking_tokens: bool = True
    min_ngram_size: int = 2
    max_ngram_size: int = 5


class IntelligentVocabController:
    """Manages vocabulary sizing with automatic codebook learning.

    This controller wraps AdaptiveQWTTokenizer and handles:
    1. Automatic codebook training from corpus samples
    2. Pre-trained codebook loading/saving
    3. Consistent effective_vocab_size for model building
    4. Smart tuner integration for HPO

    The effective vocab size is determined by:
    - Base vocabulary (special + byte tokens): ~362 tokens
    - Learned superwords from codebook: variable
    - Total = base + learned

    Attributes:
        tokenizer: The wrapped AdaptiveQWTTokenizer instance.
        config: VocabControllerConfig with settings.

    Example:
        >>> controller = IntelligentVocabController(model_max_length=4096)
        >>> # Option 1: Train from corpus
        >>> controller.train_codebook(texts, min_freq=10)
        >>> # Option 2: Load pre-trained
        >>> controller.load_pretrained("verso-baseline")
        >>> # Use for model building
        >>> model = build_model(vocab_size=controller.effective_vocab_size)
    """

    # Base vocab: special tokens (10) + byte offset (32) + 256 bytes + buffer
    BASE_VOCAB_SIZE = 362

    def __init__(
        self,
        config: VocabControllerConfig | None = None,
        *,
        model_max_length: int | None = None,
    ) -> None:
        """Initialize the IntelligentVocabController.

        Args:
            config: Configuration object. If None, uses defaults.
            model_max_length: Override for max sequence length.
        """
        if config is None:
            config = VocabControllerConfig()
        if model_max_length is not None:
            config.model_max_length = model_max_length

        self.config = config
        self._codebook_path: Path | None = None
        self._codebook_source: str | None = None  # "trained" or "pretrained:name"

        # Initialize tokenizer with large target to allow growth
        # The actual vocab used is determined by codebook training
        target_vocab = 128000  # Maximum practical vocab size
        self._tokenizer = AdaptiveQWTTokenizer(
            vocab_size=target_vocab,
            model_max_length=config.model_max_length,
            byte_offset=config.byte_offset,
            enable_thinking_tokens=config.enable_thinking_tokens,
            min_ngram_size=config.min_ngram_size,
            max_ngram_size=config.max_ngram_size,
        )

        logger.info(
            "[VocabController] Initialized: max_length=%d, base_vocab=%d",
            config.model_max_length,
            self.BASE_VOCAB_SIZE,
        )

    @property
    def tokenizer(self) -> AdaptiveQWTTokenizer:
        """Return the wrapped tokenizer instance."""
        return self._tokenizer

    @property
    def effective_vocab_size(self) -> int:
        """Return the effective vocabulary size for model building.

        This is the actual size to use for embedding and output layers.
        Before codebook training: returns base vocab size (~362).
        After codebook training: returns base + learned tokens.
        """
        return self._tokenizer.vocab_size

    @property
    def base_vocab_size(self) -> int:
        """Return the base vocabulary size (special + byte tokens)."""
        return self.BASE_VOCAB_SIZE

    @property
    def learned_vocab_size(self) -> int:
        """Return the number of learned superword tokens."""
        if self._tokenizer.merger is not None:
            return self._tokenizer.merger.superword_count
        return 0

    @property
    def is_trained(self) -> bool:
        """Return whether the codebook has been trained or loaded."""
        return self._tokenizer.is_trained

    @property
    def codebook_source(self) -> str | None:
        """Return the source of the current codebook (trained/pretrained)."""
        return self._codebook_source

    def train_codebook(
        self,
        texts: Sequence[str],
        min_freq: int | None = None,
        progress_callback: Any = None,
    ) -> int:
        """Train codebook from corpus texts.

        Analyzes the input texts for frequent n-grams and builds a
        codebook of superword tokens for sequence compression.

        Args:
            texts: Corpus texts to learn from.
            min_freq: Minimum frequency for inclusion (default: config.min_ngram_freq).
            progress_callback: Optional progress callback.

        Returns:
            Number of superwords learned.

        Example:
            >>> texts = load_training_data()[:10000]  # Sample
            >>> learned = controller.train_codebook(texts, min_freq=10)
            >>> print(f"Learned {learned} tokens, vocab now {controller.effective_vocab_size}")
        """
        if min_freq is None:
            min_freq = self.config.min_ngram_freq

        # Sample if necessary
        if len(texts) > self.config.sample_size:
            import random

            texts = random.sample(list(texts), self.config.sample_size)

        logger.info(
            "[VocabController] Training codebook from %d texts (min_freq=%d)",
            len(texts),
            min_freq,
        )

        learned = self._tokenizer.learn_from_corpus(
            texts,
            min_freq=min_freq,
            progress_callback=progress_callback,
        )

        self._codebook_source = "trained"
        logger.info(
            "[VocabController] Codebook trained: %d superwords, effective_vocab=%d",
            learned,
            self.effective_vocab_size,
        )

        return learned

    def load_pretrained(self, name: str) -> int:
        """Load a pre-trained codebook by name.

        Pre-trained codebooks are available for common datasets:
        - 'verso-baseline': Verso Baseline curriculum
        - 'openwebtext': OpenWebText corpus
        - 'starcoder': StarCoder code data

        Args:
            name: Name of the pre-trained codebook.

        Returns:
            Number of superwords loaded.

        Raises:
            ValueError: If the codebook name is not recognized.
            FileNotFoundError: If the codebook file is missing.

        Example:
            >>> controller.load_pretrained("verso-baseline")
        """
        if name not in PRETRAINED_CODEBOOKS:
            available = list(PRETRAINED_CODEBOOKS.keys())
            raise ValueError(f"Unknown pretrained codebook '{name}'. " f"Available: {available}")

        # Resolve path relative to project root
        codebook_path = Path(__file__).parent.parent.parent / PRETRAINED_CODEBOOKS[name]

        if not codebook_path.exists():
            logger.warning(
                "[VocabController] Pre-trained codebook '%s' not found at %s. "
                "Using base vocabulary only.",
                name,
                codebook_path,
            )
            return 0

        return self.load_codebook(codebook_path)

    def load_codebook(self, path: str | Path) -> int:
        """Load a codebook from file.

        Args:
            path: Path to the codebook JSON file.

        Returns:
            Number of superwords loaded.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        loaded = self._tokenizer.load_codebook(str(path))
        self._codebook_path = path
        self._codebook_source = f"loaded:{path.name}"

        logger.info(
            "[VocabController] Loaded codebook from %s: %d superwords, effective_vocab=%d",
            path,
            loaded,
            self.effective_vocab_size,
        )

        return loaded

    def save_codebook(self, path: str | Path) -> None:
        """Save the current codebook to file.

        Args:
            path: Path to save the codebook JSON file.

        Raises:
            RuntimeError: If no codebook has been trained.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save_codebook(str(path))
        self._codebook_path = path

        logger.info(
            "[VocabController] Saved codebook to %s",
            path,
        )

    def tokenize(
        self,
        texts: str | list[str],
        *,
        max_length: int | None = None,
        padding: bool | str = "max_length",
        truncation: bool = True,
        return_tensors: str | None = "tf",
    ) -> dict[str, Any]:
        """Tokenize texts using the trained tokenizer.

        Args:
            texts: Text(s) to tokenize.
            max_length: Maximum sequence length.
            padding: Padding strategy.
            truncation: Whether to truncate.
            return_tensors: Output format ('tf' for TensorFlow).

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'.
        """
        return self._tokenizer(
            texts,
            max_length=max_length or self.config.model_max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            ids: Token IDs to decode.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            Decoded text string.
        """
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_stats(self) -> dict[str, Any]:
        """Get controller statistics for reporting.

        Returns:
            Dictionary with vocab sizing and codebook info.
        """
        return {
            "effective_vocab_size": self.effective_vocab_size,
            "base_vocab_size": self.base_vocab_size,
            "learned_vocab_size": self.learned_vocab_size,
            "is_trained": self.is_trained,
            "codebook_source": self._codebook_source,
            "model_max_length": self.config.model_max_length,
            "tokenizer_stats": self._tokenizer.get_stats(),
        }


__all__ = ["IntelligentVocabController", "VocabControllerConfig", "PRETRAINED_CODEBOOKS"]
