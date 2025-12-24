# highnoon/data/preprocessors/base.py
# Copyright 2025 Verso Industries
#
# Base preprocessor class for HighNoon Language Framework.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import tensorflow as tf


@dataclass
class PreprocessorConfig:
    """Configuration for data preprocessors.

    Attributes:
        max_seq_length: Maximum sequence length (truncate/pad to this).
        pad_token_id: Token ID for padding.
        eos_token_id: End of sequence token ID.
        bos_token_id: Beginning of sequence token ID (optional).
        add_special_tokens: Whether to add BOS/EOS tokens.
        truncation: Truncation strategy ('left', 'right', 'none').
    """

    max_seq_length: int = 2048
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int | None = None
    add_special_tokens: bool = True
    truncation: str = "right"


class BasePreprocessor(ABC):
    """Abstract base class for data preprocessors.

    Preprocessors are responsible for transforming raw text data into
    tokenized tensors suitable for language model training.

    Subclasses must implement the `process` method which handles
    the specific preprocessing logic for different data formats
    (raw text, instruction-response pairs, etc.).
    """

    def __init__(self, tokenizer: Any, config: PreprocessorConfig | None = None):
        """Initialize preprocessor.

        Args:
            tokenizer: Tokenizer instance with encode/decode methods.
            config: Preprocessor configuration. Uses defaults if None.
        """
        self.tokenizer = tokenizer
        self.config = config or PreprocessorConfig()

    @abstractmethod
    def process(self, examples: list[dict[str, Any]]) -> dict[str, tf.Tensor]:
        """Process a batch of examples into tokenized tensors.

        Args:
            examples: List of example dictionaries from the dataset.
                     Format depends on the specific preprocessor.

        Returns:
            Dictionary with tokenized tensors:
                - input_ids: Token IDs [batch, seq_len]
                - attention_mask: Mask for valid tokens [batch, seq_len]
                - labels: Target token IDs for training [batch, seq_len]
        """
        pass

    def tokenize(self, text: str) -> list[int]:
        """Tokenize a single text string.

        Args:
            text: Input text to tokenize.

        Returns:
            List of token IDs.
        """
        return self.tokenizer.encode(text)

    def pad_sequence(
        self,
        token_ids: list[int],
        max_length: int | None = None,
    ) -> list[int]:
        """Pad or truncate a sequence to the target length.

        Args:
            token_ids: Input token ID sequence.
            max_length: Target length (uses config.max_seq_length if None).

        Returns:
            Padded/truncated sequence of token IDs.
        """
        max_length = max_length or self.config.max_seq_length

        if len(token_ids) > max_length:
            # Truncate
            if self.config.truncation == "right":
                token_ids = token_ids[:max_length]
            elif self.config.truncation == "left":
                token_ids = token_ids[-max_length:]
            # 'none' means no truncation - let it overflow

        if len(token_ids) < max_length:
            # Pad
            padding_length = max_length - len(token_ids)
            token_ids = token_ids + [self.config.pad_token_id] * padding_length

        return token_ids

    def create_attention_mask(self, token_ids: list[int]) -> list[int]:
        """Create attention mask (1 for real tokens, 0 for padding).

        Args:
            token_ids: Token ID sequence.

        Returns:
            Attention mask as list of 0s and 1s.
        """
        return [1 if t != self.config.pad_token_id else 0 for t in token_ids]

    def create_dataset(
        self,
        examples: list[dict[str, Any]],
        batch_size: int = 8,
        shuffle: bool = True,
        shuffle_buffer: int = 10000,
    ) -> tf.data.Dataset:
        """Create a TensorFlow dataset from examples.

        Args:
            examples: List of example dictionaries.
            batch_size: Batch size for training.
            shuffle: Whether to shuffle the dataset.
            shuffle_buffer: Buffer size for shuffling.

        Returns:
            tf.data.Dataset ready for training.
        """
        processed = self.process(examples)

        dataset = tf.data.Dataset.from_tensor_slices(processed)

        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer)

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
