# highnoon/tokenization/byte_stream_tokenizer.py
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

"""Phase 10.5: Byte-Stream Tokenizer for token-free processing.

This module implements a minimal tokenizer that converts text to raw UTF-8 bytes,
enabling token-free processing inspired by MambaByte. This mode bypasses learned
tokenization in favor of direct byte-level representation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import tensorflow as tf


@dataclass
class _ByteEncoding:
    """Internal encoding representation for byte-stream mode."""

    input_ids: list[int]
    attention_mask: list[int]


class ByteStreamTokenizer:
    """Byte-level tokenizer for token-free text processing.

    This tokenizer converts text directly to UTF-8 byte sequences without
    learned vocabulary. It provides a 256-token vocabulary (0-255) corresponding
    to all possible byte values.

    Key features:
    - No learned vocabulary or subword merging
    - Direct UTF-8 byte encoding
    - Fixed 256-token vocabulary
    - Compatible with MambaByte-style architectures

    Example:
        >>> tokenizer = ByteStreamTokenizer(model_max_length=512)
        >>> encoded = tokenizer("Hello!")
        >>> encoded["input_ids"]
        [72, 101, 108, 108, 111, 33]
        >>> tokenizer.decode([72, 101, 108, 108, 111, 33])
        'Hello!'

    Attributes:
        model_max_length: Maximum sequence length.
        vocab_size: Always 256 (one token per byte value).
    """

    # Special tokens (using reserved byte values that rarely appear in text)
    PAD_TOKEN_ID = 0  # Null byte as padding
    EOS_TOKEN_ID = 3  # End of text (ETX)
    BOS_TOKEN_ID = 2  # Start of text (STX)

    def __init__(
        self,
        model_max_length: int = 8192,
    ):
        """Initialize the ByteStreamTokenizer.

        Args:
            model_max_length: Maximum sequence length for the model.
        """
        self._model_max_length = model_max_length
        self._vocab_size = 256

        # Token sets
        self.special_tokens_set = frozenset(
            {
                self.PAD_TOKEN_ID,
                self.BOS_TOKEN_ID,
                self.EOS_TOKEN_ID,
            }
        )

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size (always 256)."""
        return self._vocab_size

    @property
    def model_max_length(self) -> int:
        """Return the maximum model length."""
        return self._model_max_length

    @property
    def pad_token_id(self) -> int:
        """Return the padding token ID."""
        return self.PAD_TOKEN_ID

    @property
    def eos_token_id(self) -> int:
        """Return the end-of-sequence token ID."""
        return self.EOS_TOKEN_ID

    @property
    def bos_token_id(self) -> int:
        """Return the beginning-of-sequence token ID."""
        return self.BOS_TOKEN_ID

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
        """Tokenize text(s) into byte IDs and attention masks.

        Args:
            texts: Single string or list of strings to tokenize.
            truncation: Whether to truncate to max_length.
            max_length: Maximum length for truncation/padding.
            padding: Whether/how to pad sequences.
            add_special_tokens: Whether to add BOS/EOS tokens.
            return_tensors: If "tf", return TensorFlow tensors.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'.
        """
        if isinstance(texts, str):
            texts = [texts]

        max_len = max_length or self._model_max_length
        encodings: list[_ByteEncoding] = [
            self._encode_one(
                text,
                truncation=truncation,
                max_length=max_len,
                add_special_tokens=add_special_tokens,
            )
            for text in texts
        ]

        # Handle padding
        if padding:
            target_len = max(len(e.input_ids) for e in encodings)
            if padding == "max_length":
                target_len = max_len
            encodings = [self._pad_encoding(e, target_len) for e in encodings]

        result: dict[str, Any] = {
            "input_ids": [e.input_ids for e in encodings],
            "attention_mask": [e.attention_mask for e in encodings],
        }

        # Flatten single-item batches
        if len(texts) == 1:
            result["input_ids"] = result["input_ids"][0]
            result["attention_mask"] = result["attention_mask"][0]

        # Convert to tensors if requested
        if return_tensors == "tf":
            result["input_ids"] = tf.constant(result["input_ids"], dtype=tf.int32)
            result["attention_mask"] = tf.constant(result["attention_mask"], dtype=tf.int32)

        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode text to byte IDs.

        Args:
            text: Text to encode.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of byte IDs.
        """
        result = self(
            text,
            truncation=False,
            add_special_tokens=add_special_tokens,
        )
        return result["input_ids"]

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode byte IDs back to text.

        Args:
            ids: Sequence of byte IDs to decode.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded string.
        """
        if skip_special_tokens:
            ids = [i for i in ids if i not in self.special_tokens_set]

        # Filter valid byte values (0-255)
        bytes_list = bytes([i for i in ids if 0 <= i <= 255])

        # Decode UTF-8, replacing invalid sequences
        return bytes_list.decode("utf-8", errors="replace")

    def batch_decode(
        self,
        batch_ids: Iterable[Sequence[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode a batch of byte ID sequences.

        Args:
            batch_ids: Iterable of byte ID sequences.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            List of decoded strings.
        """
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def _encode_one(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        add_special_tokens: bool,
    ) -> _ByteEncoding:
        """Encode a single text string to bytes.

        Args:
            text: Text to encode.
            truncation: Whether to truncate.
            max_length: Maximum length.
            add_special_tokens: Whether to add BOS/EOS.

        Returns:
            _ByteEncoding with input_ids and attention_mask.
        """
        # Convert text to UTF-8 bytes
        byte_ids = list(text.encode("utf-8"))

        # Add special tokens if requested
        if add_special_tokens:
            byte_ids = [self.BOS_TOKEN_ID] + byte_ids + [self.EOS_TOKEN_ID]

        # Truncate if needed
        if truncation and len(byte_ids) > max_length:
            byte_ids = byte_ids[:max_length]

        return _ByteEncoding(
            input_ids=byte_ids,
            attention_mask=[1] * len(byte_ids),
        )

    def _pad_encoding(self, encoding: _ByteEncoding, length: int) -> _ByteEncoding:
        """Pad an encoding to specified length.

        Args:
            encoding: Encoding to pad.
            length: Target length.

        Returns:
            Padded encoding.
        """
        pad_len = length - len(encoding.input_ids)
        if pad_len <= 0:
            return encoding

        return _ByteEncoding(
            input_ids=encoding.input_ids + [self.PAD_TOKEN_ID] * pad_len,
            attention_mask=encoding.attention_mask + [0] * pad_len,
        )


__all__ = ["ByteStreamTokenizer"]
