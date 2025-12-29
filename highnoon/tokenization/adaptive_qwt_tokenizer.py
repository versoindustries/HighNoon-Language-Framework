# highnoon/tokenization/adaptive_qwt_tokenizer.py
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

"""Adaptive QWT Tokenizer with Learnable Codebook.

This module extends the base QWTTextTokenizer with a learnable codebook
that grows the vocabulary by learning frequent n-grams from the training corpus.
This bridges the gap between user-configured vocab_size and actual token output.

Architecture:
    1. Base UTF-8 byte encoding (QWTTextTokenizer)
    2. + Learned n-gram tokens (via SuperwordMerger)
    = Full vocab utilization with sequence compression

Example:
    >>> tokenizer = AdaptiveQWTTokenizer(vocab_size=8000, model_max_length=256)
    >>> tokenizer.learn_from_corpus(["Hello world", "Machine learning"])
    >>> tokens = tokenizer("Hello world")
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from highnoon.tokenization.qwt_text_tokenizer import QWTTextTokenizer
from highnoon.tokenization.superword_merger import SuperwordMerger, SuperwordMergerConfig

logger = logging.getLogger(__name__)


class AdaptiveQWTTokenizer(QWTTextTokenizer):
    """Adaptive QWT Tokenizer with learnable n-gram codebook.

    Extends QWTTextTokenizer to support vocabulary expansion through
    learned n-gram tokens. The tokenizer maintains a base vocabulary
    from UTF-8 byte encoding and dynamically extends it with frequently
    occurring n-grams learned from the training corpus.

    This allows the WebUI-configured vocab_size to be fully utilized
    rather than being limited to ~288 byte tokens.

    Attributes:
        target_vocab_size: The user-configured target vocabulary size.
        codebook_capacity: Maximum n-grams that can be learned.
        merger: SuperwordMerger instance for n-gram learning.

    Example:
        >>> tokenizer = AdaptiveQWTTokenizer(vocab_size=4096, model_max_length=512)
        >>> # Learn from corpus (can be done during data loading)
        >>> learned = tokenizer.learn_from_corpus(texts, min_freq=10)
        >>> print(f"Learned {learned} n-gram tokens")
        >>> # Now tokenize with compression
        >>> output = tokenizer("Hello world", return_tensors="tf")
    """

    # Base vocab: special tokens (10) + byte offset (32) + 256 bytes + buffer
    _BASE_VOCAB_SIZE = 10 + 32 + 256 + 64  # = 362

    def __init__(
        self,
        *,
        vocab_size: int,
        model_max_length: int,
        max_vocab_size: int | None = None,
        byte_offset: int = 32,
        enable_thinking_tokens: bool = True,
        min_ngram_size: int = 2,
        max_ngram_size: int = 5,
    ) -> None:
        """Initialize the Adaptive QWT Tokenizer.

        Args:
            vocab_size: Target vocabulary size (will grow via learning).
            model_max_length: Maximum sequence length for the model.
            max_vocab_size: Hard cap on vocabulary size from QAHPO budget.
                If set, vocabulary learning will stop at this limit.
            byte_offset: Offset for byte values in vocabulary (default: 32).
            enable_thinking_tokens: Enable thinking token injection.
            min_ngram_size: Minimum n-gram size for codebook learning.
            max_ngram_size: Maximum n-gram size for codebook learning.
        """
        # Initialize base tokenizer with minimum required vocab
        base_vocab = max(self._BASE_VOCAB_SIZE, byte_offset + 256 + 64)
        super().__init__(
            vocab_size=base_vocab,
            model_max_length=model_max_length,
            byte_offset=byte_offset,
            enable_thinking_tokens=enable_thinking_tokens,
        )

        # Store target configuration
        self._target_vocab_size = vocab_size
        self._base_vocab_size = base_vocab
        self._codebook_capacity = max(0, vocab_size - base_vocab)

        # Hard cap from QAHPO budget (enforced during learning)
        self._max_vocab_size = max_vocab_size
        if max_vocab_size is not None and max_vocab_size < vocab_size:
            # Reduce codebook capacity to respect budget
            self._codebook_capacity = max(0, max_vocab_size - base_vocab)
            logger.info(
                "[AdaptiveQWT] Budget cap active: max_vocab_size=%d limits codebook to %d",
                max_vocab_size,
                self._codebook_capacity,
            )

        # N-gram configuration
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size

        # SuperwordMerger for learned tokens (initialized on first training)
        self._merger: SuperwordMerger | None = None
        self._trained = False

        logger.info(
            "[AdaptiveQWT] Initialized: base_vocab=%d, target=%d, codebook_capacity=%d%s",
            self._base_vocab_size,
            self._target_vocab_size,
            self._codebook_capacity,
            f" (capped at {max_vocab_size})" if max_vocab_size else "",
        )

    @property
    def vocab_size(self) -> int:
        """Return the actual utilized vocabulary size.

        Before training: returns base vocab size (bytes + special tokens).
        After training: returns base + learned n-gram tokens.
        """
        if self._merger is not None:
            return self._base_vocab_size + self._merger.superword_count
        return self._base_vocab_size

    @property
    def target_vocab_size(self) -> int:
        """Return the target vocabulary size from config."""
        return self._target_vocab_size

    @property
    def codebook_capacity(self) -> int:
        """Return the remaining capacity for learned tokens."""
        if self._merger is not None:
            return max(0, self._codebook_capacity - self._merger.superword_count)
        return self._codebook_capacity

    @property
    def is_trained(self) -> bool:
        """Return whether the codebook has been trained."""
        return self._trained

    @property
    def merger(self) -> SuperwordMerger | None:
        """Return the SuperwordMerger instance (or None if not trained)."""
        return self._merger

    def learn_from_corpus(
        self,
        texts: Sequence[str],
        min_freq: int = 10,
        progress_callback: Any = None,
    ) -> int:
        """Learn frequent n-grams from corpus to populate codebook.

        This method tokenizes the input texts, counts n-gram frequencies,
        and adds the most frequent n-grams to the codebook. This should
        be called once during data loading before training begins.

        Args:
            texts: Sequence of text strings to learn from.
            min_freq: Minimum frequency for an n-gram to be included.
            progress_callback: Optional callback for progress updates.

        Returns:
            Number of n-gram tokens learned.

        Example:
            >>> texts = ["Hello world", "Machine learning is fun"]
            >>> learned = tokenizer.learn_from_corpus(texts, min_freq=5)
            >>> print(f"Learned {learned} n-gram tokens")
        """
        if self._codebook_capacity <= 0:
            logger.warning(
                "[AdaptiveQWT] No codebook capacity (vocab_size=%d <= base=%d)",
                self._target_vocab_size,
                self._base_vocab_size,
            )
            return 0

        logger.info(
            "[AdaptiveQWT] Learning from %d texts (min_freq=%d, capacity=%d)",
            len(texts),
            min_freq,
            self._codebook_capacity,
        )

        # Tokenize all texts to byte tokens with interrupt handling
        token_sequences: list[list[int]] = []
        interrupted = False
        try:
            for i, text in enumerate(texts):
                encoding = self._encode_one(
                    text,
                    truncation=True,
                    max_length=self.model_max_length,
                    add_special_tokens=False,
                )
                token_sequences.append(encoding.input_ids)
                # Log progress periodically
                if (i + 1) % 5000 == 0:
                    logger.info("[AdaptiveQWT] Tokenized %d/%d texts...", i + 1, len(texts))
        except KeyboardInterrupt:
            logger.warning(
                "[AdaptiveQWT] Tokenization interrupted at %d/%d texts. "
                "Continuing with partial data.",
                len(token_sequences),
                len(texts),
            )
            interrupted = True

        if not token_sequences:
            logger.warning("[AdaptiveQWT] No sequences tokenized, returning 0")
            return 0

        if interrupted:
            logger.info(
                "[AdaptiveQWT] Using %d tokenized sequences (interrupted)",
                len(token_sequences),
            )

        # Create and train SuperwordMerger
        merger_config = SuperwordMergerConfig(
            min_frequency=min_freq,
            max_vocab_size=self._codebook_capacity,
            min_ngram_size=self._min_ngram_size,
            max_ngram_size=self._max_ngram_size,
            byte_offset=self._byte_offset,
        )
        self._merger = SuperwordMerger(
            base_vocab_size=self._base_vocab_size,
            config=merger_config,
        )

        # Train on tokenized sequences
        learned_count = self._merger.train(
            token_sequences,
            progress_callback=progress_callback,
        )

        self._trained = True
        final_vocab = self.vocab_size

        # Check if we hit the budget cap
        if self._max_vocab_size is not None and final_vocab >= self._max_vocab_size:
            logger.info(
                "[AdaptiveQWT] vocab_size capped at %d (budget limit)",
                self._max_vocab_size,
            )
        else:
            logger.info(
                "[AdaptiveQWT] Learned %d n-gram tokens, vocab_size now %d",
                learned_count,
                final_vocab,
            )

        return learned_count

    def __call__(
        self,
        texts: str | list[str],
        *,
        truncation: bool = True,
        max_length: int | None = None,
        padding: bool | str = False,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        """Tokenize text(s) with optional n-gram compression.

        If the codebook has been trained, applies SuperwordMerger
        to compress frequent n-grams into single tokens.

        Args:
            texts: Single string or list of strings to tokenize.
            truncation: Whether to truncate to max_length.
            max_length: Maximum length for truncation/padding.
            padding: Padding strategy ('max_length', True, or False).
            add_special_tokens: Whether to add CLS/EOS tokens.
            return_tensors: Output format ('tf' for TensorFlow tensors).

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' keys.
        """
        # Get base encoding from parent class
        result = super().__call__(
            texts,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            add_special_tokens=add_special_tokens,
            return_tensors=None,  # Apply merger before converting to tensors
        )

        # Apply n-gram merging if trained
        if self._merger is not None and self._merger.superword_count > 0:
            single = isinstance(texts, str)
            input_ids = result["input_ids"]

            if single:
                # Single sequence
                merged_ids = self._merger.apply(input_ids)
                result["input_ids"] = merged_ids
                result["attention_mask"] = [1] * len(merged_ids)
            else:
                # Batch of sequences
                merged_batch = []
                attention_batch = []
                for seq in input_ids:
                    merged = self._merger.apply(seq)
                    merged_batch.append(merged)
                    attention_batch.append([1] * len(merged))
                result["input_ids"] = merged_batch
                result["attention_mask"] = attention_batch

            # Re-pad if needed (merging may have shortened sequences)
            pad_target = None
            if padding == "max_length" or (padding is True and max_length):
                pad_target = max_length or self.model_max_length

            if pad_target is not None:
                if single:
                    result = self._pad_result(result, pad_target, single=True)
                else:
                    result = self._pad_result(result, pad_target, single=False)

        # Convert to tensors if requested
        if return_tensors == "tf":
            import tensorflow as tf

            result = {k: tf.constant(v, dtype=tf.int32) for k, v in result.items()}

        return result

    def _pad_result(
        self,
        result: dict[str, Any],
        target_length: int,
        single: bool,
    ) -> dict[str, Any]:
        """Pad result to target length after merging."""
        if single:
            ids = result["input_ids"]
            mask = result["attention_mask"]
            if len(ids) < target_length:
                pad_amount = target_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_amount
                mask = mask + [0] * pad_amount
            elif len(ids) > target_length:
                ids = ids[:target_length]
                mask = mask[:target_length]
            result["input_ids"] = ids
            result["attention_mask"] = mask
        else:
            new_ids = []
            new_masks = []
            for ids, mask in zip(result["input_ids"], result["attention_mask"]):
                if len(ids) < target_length:
                    pad_amount = target_length - len(ids)
                    ids = list(ids) + [self.pad_token_id] * pad_amount
                    mask = list(mask) + [0] * pad_amount
                elif len(ids) > target_length:
                    ids = ids[:target_length]
                    mask = mask[:target_length]
                new_ids.append(ids)
                new_masks.append(mask)
            result["input_ids"] = new_ids
            result["attention_mask"] = new_masks
        return result

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text.

        If n-gram tokens are present, expands them back to bytes first.

        Args:
            ids: Sequence of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded string.
        """
        # Expand superwords if merger is active
        if self._merger is not None:
            ids = self._merger.reverse_merge(ids)

        return super().decode(ids, skip_special_tokens=skip_special_tokens)

    def get_stats(self) -> dict[str, Any]:
        """Get tokenizer statistics.

        Returns:
            Dictionary containing tokenizer configuration and learning stats.
        """
        stats = {
            "base_vocab_size": self._base_vocab_size,
            "target_vocab_size": self._target_vocab_size,
            "current_vocab_size": self.vocab_size,
            "codebook_capacity": self._codebook_capacity,
            "remaining_capacity": self.codebook_capacity,
            "is_trained": self._trained,
            "min_ngram_size": self._min_ngram_size,
            "max_ngram_size": self._max_ngram_size,
        }
        if self._merger is not None:
            stats["merger_stats"] = self._merger.get_stats()
        return stats

    def save_codebook(self, path: str) -> None:
        """Save the learned codebook to a file.

        Saves the SuperwordMerger state including all learned n-grams.
        Can be loaded later to skip the learning phase.

        Args:
            path: Path to save the codebook (JSON format).

        Raises:
            RuntimeError: If codebook has not been trained.

        Example:
            >>> tokenizer.learn_from_corpus(texts)
            >>> tokenizer.save_codebook("codebook.json")
        """
        if self._merger is None or not self._trained:
            raise RuntimeError(
                "Cannot save codebook: no training has been performed. "
                "Call learn_from_corpus() first."
            )
        from pathlib import Path as PathLib

        save_path = PathLib(path)
        self._merger.save(save_path)
        logger.info(
            "[AdaptiveQWT] Saved codebook to %s (%d superwords)",
            path,
            self._merger.superword_count,
        )

    def load_codebook(self, path: str) -> int:
        """Load a pre-trained codebook from file.

        Loads a previously saved SuperwordMerger state, allowing
        vocabulary expansion without re-learning from corpus.

        Args:
            path: Path to the saved codebook (JSON format).

        Returns:
            Number of superwords loaded.

        Raises:
            FileNotFoundError: If the codebook file does not exist.
            ValueError: If the codebook is incompatible with this tokenizer.

        Example:
            >>> tokenizer.load_codebook("codebook.json")
            >>> print(f"Vocab size: {tokenizer.vocab_size}")
        """
        from pathlib import Path as PathLib

        load_path = PathLib(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Codebook file not found: {path}")

        self._merger = SuperwordMerger.load(load_path)

        # Validate compatibility
        if self._merger.base_vocab_size != self._base_vocab_size:
            raise ValueError(
                f"Codebook base_vocab_size ({self._merger.base_vocab_size}) "
                f"does not match tokenizer ({self._base_vocab_size})"
            )

        self._trained = True
        superword_count = self._merger.superword_count

        logger.info(
            "[AdaptiveQWT] Loaded codebook from %s (%d superwords, vocab_size=%d)",
            path,
            superword_count,
            self.vocab_size,
        )

        return superword_count


__all__ = ["AdaptiveQWTTokenizer"]
