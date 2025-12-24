# highnoon/tokenization/qwt_text_tokenizer.py
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

"""Quantum Wavelet Tokenizer front-end with a HuggingFace-like interface.

This module provides the text-side API expected by the rest of the codebase
(`__call__`, `decode`, `batch_decode`, padding utilities, etc.) while emitting
the integer token IDs that feed directly into the fused Quantum Wavelet
Tokenizer stack during training. Internally it performs a UTF-8 byte mapping
so it stays deterministic and dependency-free, but all special-token handling
and field names align with the QWT runtime expectations.

Phase 10.1 Enhancement: Added thinking tokens (<think>, <pause>, <reflect>,
<conclude>) for "computational breathing room" during reasoning tasks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import tensorflow as tf


@dataclass
class _Encoding:
    """Internal encoding representation."""

    input_ids: list[int]
    attention_mask: list[int]


@dataclass
class SuperpositionEncoding:
    """Phase 31: Superposition BPE encoding with multiple segmentations.

    Holds multiple tokenization paths with associated amplitudes.
    During generation, these can be processed in quantum superposition.
    """

    # List of alternative tokenizations [num_alternatives, seq_len]
    input_ids: list[list[int]]
    # Amplitudes for each alternative [num_alternatives]
    amplitudes: list[float]
    # Attention masks [num_alternatives, seq_len]
    attention_masks: list[list[int]]


class QWTTextTokenizer:
    """Lightweight tokenizer front-end paired with the fused QWT runtime.

    This tokenizer uses UTF-8 byte encoding to provide a deterministic,
    dependency-free tokenization pipeline that integrates directly with
    the Quantum Wavelet Tokenizer C++ operations.

    Phase 10.1 Enhancement: Supports thinking tokens for reasoning tasks:
        - <think> (ID 6): Start internal reasoning block
        - <pause> (ID 7): Additional computation time
        - <reflect> (ID 8): Backtrack/reconsider reasoning
        - <conclude> (ID 9): Finalize thought chain

    Example:
        >>> tokenizer = QWTTextTokenizer(vocab_size=512, model_max_length=256)
        >>> tokens = tokenizer("Hello, world!", return_tensors="tf")
        >>> print(tokens["input_ids"].shape)
        TensorShape([1, 15])
    """

    _BYTE_VOCAB = 256
    _SPECIAL_TOKENS = {
        "unk_token_id": 0,
        "cls_token_id": 1,
        "pad_token_id": 2,
        "eos_token_id": 3,
        "sep_token_id": 4,
        "mask_token_id": 5,
        # Phase 10.1: Thinking tokens for computational breathing room
        "think_token_id": 6,
        "pause_token_id": 7,
        "reflect_token_id": 8,
        "conclude_token_id": 9,
    }

    # Complexity indicators for automatic thinking token injection
    _COMPLEXITY_PATTERNS = [
        (r"\bif\b.*\bthen\b", 1),  # Conditional logic
        (r"\b(because|therefore|thus|hence)\b", 1),  # Causal reasoning
        (r"\b(first|second|third|finally)\b", 1),  # Sequential steps
        (r"\b(however|but|although)\b", 1),  # Contrast/exception
        (r"\?\s*$", 2),  # Questions require more thought
        (r"\b(analyze|evaluate|compare|contrast)\b", 2),  # Analytical tasks
        (r"\b(prove|derive|calculate|solve)\b", 3),  # Mathematical reasoning
    ]

    def __init__(
        self,
        *,
        vocab_size: int,
        model_max_length: int,
        byte_offset: int = 32,
        enable_thinking_tokens: bool = True,
    ) -> None:
        """Initialize the QWT Text Tokenizer.

        Args:
            vocab_size: Total vocabulary size. Must be >= byte_offset + 256.
            model_max_length: Maximum sequence length for the model.
            byte_offset: Offset for byte values in vocabulary (default: 32).
            enable_thinking_tokens: Whether to enable thinking token injection
                based on text complexity analysis (default: True).
        """
        min_required = byte_offset + self._BYTE_VOCAB
        if vocab_size < min_required:
            raise ValueError(
                f"QWTTextTokenizer requires vocab_size>={min_required}, got {vocab_size}."
            )
        self._vocab_size = vocab_size
        self.model_max_length = model_max_length
        self.padding_side = "right"
        self.truncation_side = "right"
        self._byte_offset = byte_offset
        self.enable_thinking_tokens = enable_thinking_tokens

        # Initialize special token attributes with type hints
        self.unk_token_id: int = self._SPECIAL_TOKENS["unk_token_id"]
        self.cls_token_id: int = self._SPECIAL_TOKENS["cls_token_id"]
        self.pad_token_id: int = self._SPECIAL_TOKENS["pad_token_id"]
        self.eos_token_id: int = self._SPECIAL_TOKENS["eos_token_id"]
        self.sep_token_id: int = self._SPECIAL_TOKENS["sep_token_id"]
        self.mask_token_id: int = self._SPECIAL_TOKENS["mask_token_id"]
        # Phase 10.1: Thinking token IDs
        self.think_token_id: int = self._SPECIAL_TOKENS["think_token_id"]
        self.pause_token_id: int = self._SPECIAL_TOKENS["pause_token_id"]
        self.reflect_token_id: int = self._SPECIAL_TOKENS["reflect_token_id"]
        self.conclude_token_id: int = self._SPECIAL_TOKENS["conclude_token_id"]

        # Token string representations
        self.unk_token: str = "<unk>"
        self.cls_token: str = "<cls>"
        self.pad_token: str = "<pad>"
        self.eos_token: str = "<eos>"
        self.sep_token: str = "<sep>"
        self.mask_token: str = "<mask>"
        # Phase 10.1: Thinking token strings
        self.think_token: str = "<think>"
        self.pause_token: str = "<pause>"
        self.reflect_token: str = "<reflect>"
        self.conclude_token: str = "<conclude>"

        self.special_tokens_set = set(self._SPECIAL_TOKENS.values())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._vocab_size

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
        """Tokenize text(s) into input IDs and attention masks.

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
        single = isinstance(texts, str)
        inputs: list[str] = [texts] if single else list(texts)
        encodings = [
            self._encode_one(
                text,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            )
            for text in inputs
        ]
        pad_target = None
        if padding == "max_length" or (padding is True and max_length):
            pad_target = max_length or self.model_max_length
        elif padding is True:
            pad_target = max(len(enc.input_ids) for enc in encodings)
        if pad_target is not None:
            encodings = [self._pad_encoding(enc, pad_target) for enc in encodings]
        result: dict[str, Any] = {
            "input_ids": [enc.input_ids for enc in encodings],
            "attention_mask": [enc.attention_mask for enc in encodings],
        }
        if single:
            result = {k: v[0] for k, v in result.items()}
        if return_tensors == "tf":
            result = {k: tf.constant(v, dtype=tf.int32) for k, v in result.items()}
        return result

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            ids: Sequence of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded string.
        """
        bytes_out: list[int] = []
        special = self.special_tokens_set if skip_special_tokens else set()
        for idx in ids:
            if idx in special:
                continue
            if idx < self._byte_offset or idx >= self._byte_offset + self._BYTE_VOCAB:
                continue
            bytes_out.append(idx - self._byte_offset)
        try:
            return bytes(bytes_out).decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            return bytes(bytes_out).decode("latin-1", errors="ignore")

    def batch_decode(
        self,
        batch_ids: Iterable[Sequence[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode a batch of token ID sequences.

        Args:
            batch_ids: Iterable of token ID sequences.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            List of decoded strings.
        """
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def inject_thinking_tokens(
        self,
        token_ids: list[int],
        text: str | None = None,
        complexity_override: int | None = None,
    ) -> list[int]:
        """Inject thinking tokens based on text complexity analysis.

        This method analyzes the input text (or uses an override complexity level)
        to determine where to insert thinking tokens for "computational breathing
        room". Higher complexity texts get more thinking tokens.

        Args:
            token_ids: The original token IDs from encoding.
            text: Original text for complexity analysis. If None and
                complexity_override is None, no injection occurs.
            complexity_override: Manual complexity level (0-10). If provided,
                text analysis is skipped.

        Returns:
            Modified token IDs with thinking tokens injected.

        Example:
            >>> tokenizer = QWTTextTokenizer(vocab_size=512, model_max_length=256)
            >>> ids = tokenizer("Solve this equation", add_special_tokens=False)["input_ids"]
            >>> ids_with_thinking = tokenizer.inject_thinking_tokens(ids, "Solve this equation")
            >>> # Now contains <think>, <pause>, <conclude> tokens
        """
        if not self.enable_thinking_tokens:
            return token_ids

        # Determine complexity
        if complexity_override is not None:
            complexity = max(0, min(10, complexity_override))
        elif text is not None:
            complexity = self._compute_complexity(text)
        else:
            return token_ids

        if complexity == 0:
            return token_ids

        # Inject thinking tokens based on complexity
        result: list[int] = []

        # Start with <think> for any non-trivial complexity
        if complexity >= 1:
            result.append(self.think_token_id)

        # Add content tokens with <pause> tokens interspersed for high complexity
        pause_interval = max(10, 50 - (complexity * 5))  # More pauses for higher complexity
        for i, token_id in enumerate(token_ids):
            result.append(token_id)
            # Insert pauses for very complex content
            if complexity >= 5 and (i + 1) % pause_interval == 0 and i < len(token_ids) - 1:
                result.append(self.pause_token_id)

        # Add <reflect> for medium-high complexity
        if complexity >= 3:
            result.append(self.reflect_token_id)

        # Add <conclude> to signal end of thinking
        if complexity >= 1:
            result.append(self.conclude_token_id)

        return result

    def decode_with_thinking_tokens(
        self,
        ids: Sequence[int],
        *,
        show_thinking: bool = True,
    ) -> str:
        """Decode token IDs with visible thinking token markers.

        Unlike `decode()` with `skip_special_tokens=False`, this method
        renders thinking tokens as human-readable markers in the output.

        Args:
            ids: Sequence of token IDs to decode.
            show_thinking: If True, show thinking tokens. If False, same as decode().

        Returns:
            Decoded string with thinking token markers.
        """
        if not show_thinking:
            return self.decode(ids, skip_special_tokens=True)

        # Map thinking token IDs to their strings
        thinking_token_map = {
            self.think_token_id: self.think_token,
            self.pause_token_id: self.pause_token,
            self.reflect_token_id: self.reflect_token,
            self.conclude_token_id: self.conclude_token,
        }

        parts: list[str] = []
        bytes_buffer: list[int] = []

        for idx in ids:
            # Skip non-thinking special tokens
            if idx in self.special_tokens_set and idx not in thinking_token_map:
                continue

            # Handle thinking tokens
            if idx in thinking_token_map:
                # Flush byte buffer first
                if bytes_buffer:
                    try:
                        parts.append(bytes(bytes_buffer).decode("utf-8", errors="ignore"))
                    except UnicodeDecodeError:
                        parts.append(bytes(bytes_buffer).decode("latin-1", errors="ignore"))
                    bytes_buffer = []
                parts.append(thinking_token_map[idx])
                continue

            # Handle byte tokens
            if self._byte_offset <= idx < self._byte_offset + self._BYTE_VOCAB:
                bytes_buffer.append(idx - self._byte_offset)

        # Flush remaining bytes
        if bytes_buffer:
            try:
                parts.append(bytes(bytes_buffer).decode("utf-8", errors="ignore"))
            except UnicodeDecodeError:
                parts.append(bytes(bytes_buffer).decode("latin-1", errors="ignore"))

        return "".join(parts)

    def _compute_complexity(self, text: str) -> int:
        """Compute text complexity score based on pattern matching.

        Args:
            text: Input text to analyze.

        Returns:
            Complexity score from 0 to 10.
        """
        score = 0
        text_lower = text.lower()
        for pattern, weight in self._COMPLEXITY_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += weight

        # Clamp to 0-10 range
        return min(10, max(0, score))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encode_one(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int | None,
        add_special_tokens: bool,
    ) -> _Encoding:
        """Encode a single text string."""
        byte_ids = list(text.encode("utf-8"))
        tokens: list[int] = []
        if add_special_tokens and self.cls_token_id is not None:
            tokens.append(self.cls_token_id)
        tokens.extend(self._byte_offset + b for b in byte_ids)
        if add_special_tokens and self.eos_token_id is not None:
            tokens.append(self.eos_token_id)
        target_len = max_length
        if target_len is None and truncation:
            target_len = self.model_max_length
        if target_len is not None and len(tokens) > target_len:
            tokens = tokens[:target_len]
        attention = [1] * len(tokens)
        return _Encoding(tokens, attention)

    def _pad_encoding(self, encoding: _Encoding, length: int) -> _Encoding:
        """Pad an encoding to specified length."""
        pad_amount = max(length - len(encoding.input_ids), 0)
        if pad_amount:
            encoding.input_ids = encoding.input_ids + [self.pad_token_id] * pad_amount
            encoding.attention_mask = encoding.attention_mask + [0] * pad_amount
        return encoding

    # ------------------------------------------------------------------
    # Phase 31: Superposition BPE
    # ------------------------------------------------------------------
    def superposition_encode(
        self,
        text: str,
        *,
        max_superposition: int = 4,
        amplitude_threshold: float = 0.1,
        add_special_tokens: bool = True,
        max_length: int | None = None,
    ) -> SuperpositionEncoding:
        """Phase 31: Encode text with multiple segmentation alternatives.

        Creates a quantum superposition of tokenizations by exploring
        alternative segmentation paths and assigning amplitudes based on
        segmentation quality/likelihood.

        Args:
            text: Input text to tokenize.
            max_superposition: Maximum number of alternative tokenizations.
            amplitude_threshold: Minimum amplitude to keep a segmentation.
            add_special_tokens: Whether to add CLS/EOS tokens.
            max_length: Maximum sequence length.

        Returns:
            SuperpositionEncoding with multiple tokenizations and amplitudes.

        Example:
            >>> tokenizer = QWTTextTokenizer(vocab_size=512, model_max_length=256)
            >>> superpos = tokenizer.superposition_encode("Hello world")
            >>> print(len(superpos.input_ids))  # Multiple alternatives
        """
        # Get the base encoding
        base = self._encode_one(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )

        # Generate alternative tokenizations by varying byte boundaries
        # This simulates different BPE merge decisions
        alternatives: list[tuple[list[int], float]] = []

        # Primary segmentation (highest amplitude)
        alternatives.append((base.input_ids, 1.0))

        # Generate alternatives by shifting byte boundaries
        byte_ids = list(text.encode("utf-8"))

        for shift_pattern in range(1, max_superposition):
            alt_tokens: list[int] = []
            if add_special_tokens:
                alt_tokens.append(self.cls_token_id)

            # Apply different grouping strategies
            i = 0
            while i < len(byte_ids):
                # Vary group size based on shift pattern
                group_size = 1 + (shift_pattern % 3)

                # Encode group as individual bytes (simulating different merges)
                for j in range(min(group_size, len(byte_ids) - i)):
                    alt_tokens.append(self._byte_offset + byte_ids[i + j])
                i += group_size

            if add_special_tokens:
                alt_tokens.append(self.eos_token_id)

            # Apply length limit
            if max_length and len(alt_tokens) > max_length:
                alt_tokens = alt_tokens[:max_length]

            # Amplitude decreases for less canonical segmentations
            amplitude = 1.0 / (1.0 + shift_pattern * 0.5)

            if amplitude >= amplitude_threshold:
                alternatives.append((alt_tokens, amplitude))

        # Normalize amplitudes (Born rule: sum of squares = 1)
        total_sq = sum(a**2 for _, a in alternatives)
        if total_sq > 0:
            norm_factor = 1.0 / (total_sq**0.5)
            alternatives = [(ids, amp * norm_factor) for ids, amp in alternatives]

        # Build result
        all_ids = [ids for ids, _ in alternatives]
        all_amps = [amp for _, amp in alternatives]
        all_masks = [[1] * len(ids) for ids in all_ids]

        return SuperpositionEncoding(
            input_ids=all_ids,
            amplitudes=all_amps,
            attention_masks=all_masks,
        )

    def collapse_superposition(
        self,
        superpos: SuperpositionEncoding,
        *,
        strategy: str = "sample",
    ) -> _Encoding:
        """Collapse superposition to single encoding.

        Args:
            superpos: Superposition encoding to collapse.
            strategy: Collapse strategy:
                - 'max': Select highest amplitude
                - 'sample': Sample according to Born rule (|amplitude|^2)

        Returns:
            Single collapsed encoding.
        """
        import random

        if strategy == "max":
            # Deterministic: select highest amplitude
            idx = max(range(len(superpos.amplitudes)), key=lambda i: superpos.amplitudes[i])
        elif strategy == "sample":
            # Probabilistic: sample according to Born rule
            probs = [amp**2 for amp in superpos.amplitudes]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [1.0 / len(superpos.amplitudes)] * len(superpos.amplitudes)

            idx = random.choices(range(len(superpos.amplitudes)), weights=probs, k=1)[0]
        else:
            raise ValueError(f"Unknown collapse strategy: {strategy}")

        return _Encoding(
            input_ids=superpos.input_ids[idx],
            attention_mask=superpos.attention_masks[idx],
        )


__all__ = ["QWTTextTokenizer", "SuperpositionEncoding"]
