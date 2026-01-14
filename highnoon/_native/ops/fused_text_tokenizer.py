# highnoon/_native/ops/fused_text_tokenizer.py
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

"""Python wrapper for fused text tokenization ops.

This module provides Python bindings for the high-performance C++ text
tokenization operations with SIMD optimization and trie-based n-gram merging.

Operations:
    - `fused_text_tokenize()`: Single text tokenization with optional merging
    - `fused_text_tokenize_batch()`: Parallel batched tokenization
    - `SuperwordTrieHandle`: Resource handle for trie-based superword lookup

Performance Characteristics:
    - Byte tokenization: >500MB/s per core with SIMD (AVX2/AVX-512)
    - N-gram lookup: O(max_ngram_size) via trie, no Python overhead
    - Batch processing: Linear scaling with CPU cores

Example:
    >>> from highnoon._native.ops.fused_text_tokenizer import (
    ...     fused_text_tokenize_batch,
    ...     SuperwordTrieHandle,
    ... )
    >>> trie = SuperwordTrieHandle()
    >>> trie.insert_batch(ngrams, superword_ids)
    >>> tokens, lengths = fused_text_tokenize_batch(texts, trie)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import tensorflow as tf

logger = logging.getLogger(__name__)

# Native op module - loaded from consolidated binary
_text_tok_module = None
_ops_available = False

try:
    from highnoon._native import load_highnoon_core

    _core = load_highnoon_core()
    if _core is not None:
        _text_tok_module = _core
        _ops_available = True
        logger.info("[FusedTextTokenizer] Loaded native ops from consolidated binary")
except ImportError as err:
    logger.warning("[FusedTextTokenizer] Native ops not available: %s", err)


def ops_available() -> bool:
    """Check if native text tokenizer ops are available."""
    return _ops_available


class SuperwordTrieHandle:
    """Resource handle for SuperwordTrie with Python-friendly API.

    Wraps the C++ SuperwordTrie resource for efficient n-gram to superword
    mapping. The trie enables O(max_ngram_size) lookup complexity.

    Example:
        >>> trie = SuperwordTrieHandle()
        >>> # Insert individual n-grams
        >>> trie.insert([65, 66], 1000)  # "AB" -> superword 1000
        >>> # Or batch insert from table
        >>> trie.insert_batch(
        ...     ngrams=[[65, 66], [67, 68, 69]],
        ...     superword_ids=[1000, 1001],
        ... )
    """

    def __init__(self) -> None:
        """Create empty SuperwordTrie resource."""
        if not _ops_available:
            raise RuntimeError(
                "Native text tokenizer ops not available. Rebuild C++ ops with build_ops.sh."
            )
        self._handle = _text_tok_module.superword_trie_create()
        self._num_entries = 0

    @property
    def handle(self) -> tf.Tensor:
        """Get the raw resource handle for use in ops."""
        return self._handle

    @property
    def num_entries(self) -> int:
        """Get number of n-gram entries in the trie."""
        return self._num_entries

    def insert(self, ngram: Sequence[int], superword_id: int) -> None:
        """Insert a single n-gram to superword mapping.

        Args:
            ngram: Sequence of token IDs forming the n-gram.
            superword_id: Superword ID to map to.
        """
        ngram_tensor = tf.constant(list(ngram), dtype=tf.int32)
        id_tensor = tf.constant(superword_id, dtype=tf.int32)
        _text_tok_module.superword_trie_insert(self._handle, ngram_tensor, id_tensor)
        self._num_entries += 1

    def insert_batch(
        self,
        ngrams: Sequence[Sequence[int]],
        superword_ids: Sequence[int],
    ) -> None:
        """Insert multiple n-gram mappings efficiently.

        Uses flat representation for minimal Python overhead.

        Args:
            ngrams: List of n-grams (each is a sequence of token IDs).
            superword_ids: Corresponding superword IDs.
        """
        if len(ngrams) != len(superword_ids):
            raise ValueError(
                f"ngrams and superword_ids must have same length: "
                f"{len(ngrams)} != {len(superword_ids)}"
            )

        if not ngrams:
            return

        # Build flat representation
        offsets = [0]
        tokens = []
        for ngram in ngrams:
            tokens.extend(ngram)
            offsets.append(len(tokens))

        offsets_tensor = tf.constant(offsets, dtype=tf.int32)
        tokens_tensor = tf.constant(tokens, dtype=tf.int32)
        ids_tensor = tf.constant(list(superword_ids), dtype=tf.int32)

        _text_tok_module.superword_trie_build_from_table(
            self._handle, offsets_tensor, tokens_tensor, ids_tensor
        )
        self._num_entries = len(ngrams)

    def clear(self) -> None:
        """Clear and rebuild an empty trie."""
        # Recreate handle (cheaper than modifying existing)
        self._handle = _text_tok_module.superword_trie_create()
        self._num_entries = 0


def fused_text_tokenize(
    text: str | bytes,
    trie: SuperwordTrieHandle | None = None,
    byte_offset: int = 32,
    add_special_tokens: bool = True,
    max_length: int = 131072,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Tokenize single text with SIMD optimization.

    Args:
        text: UTF-8 text string or bytes.
        trie: Optional SuperwordTrieHandle for n-gram merging.
        byte_offset: Offset added to byte values (default: 32).
        add_special_tokens: Add CLS/EOS tokens.
        max_length: Maximum output length.

    Returns:
        Tuple of (tokens, length) where:
        - tokens: int32 tensor [max_length] of token IDs
        - length: scalar int32 actual token count
    """
    if not _ops_available:
        raise RuntimeError("Native text tokenizer ops not available.")

    if isinstance(text, str):
        text = text.encode("utf-8")

    text_tensor = tf.constant(text.decode("utf-8", errors="replace"))
    handle = trie.handle if trie is not None else tf.constant([], dtype=tf.resource)

    return _text_tok_module.fused_text_tokenize(
        text_tensor,
        handle,
        byte_offset=byte_offset,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
    )


def fused_text_tokenize_batch(
    texts: Sequence[str | bytes],
    trie: SuperwordTrieHandle | None = None,
    byte_offset: int = 32,
    add_special_tokens: bool = True,
    max_length: int = 131072,
    num_threads: int = 0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Tokenize batch of texts in parallel with SIMD optimization.

    Args:
        texts: Sequence of UTF-8 text strings or bytes.
        trie: Optional SuperwordTrieHandle for n-gram merging.
        byte_offset: Offset added to byte values (default: 32).
        add_special_tokens: Add CLS/EOS tokens.
        max_length: Maximum output length per text.
        num_threads: Parallel threads (0 = auto-detect).

    Returns:
        Tuple of (tokens, lengths) where:
        - tokens: int32 tensor [batch_size, max_length]
        - lengths: int32 tensor [batch_size] actual token counts
    """
    if not _ops_available:
        raise RuntimeError("Native text tokenizer ops not available.")

    # Convert to strings
    str_texts = []
    for t in texts:
        if isinstance(t, bytes):
            t = t.decode("utf-8", errors="replace")
        str_texts.append(t)

    texts_tensor = tf.constant(str_texts)
    handle = trie.handle if trie is not None else tf.constant([], dtype=tf.resource)

    return _text_tok_module.fused_text_tokenize_batch(
        texts_tensor,
        handle,
        byte_offset=byte_offset,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        num_threads=num_threads,
    )


def trie_apply_merges(
    tokens: Sequence[int],
    trie: SuperwordTrieHandle,
    superword_table: dict[tuple[int, ...], int] | None = None,
    max_ngram: int = 8,
) -> list[int]:
    """Apply superword merges using longest-match algorithm.

    Optimized Python implementation with O(N × max_ngram) complexity.
    Uses greedy longest-match to find superwords.

    Args:
        tokens: Input token IDs from base tokenization.
        trie: SuperwordTrieHandle (used to check if properly built).
        superword_table: Dict mapping n-gram tuples to superword IDs.
        max_ngram: Maximum n-gram size to check.

    Returns:
        Token IDs with superword merges applied.
    """
    if trie is None or trie.num_entries == 0:
        return list(tokens)

    if superword_table is None or len(superword_table) == 0:
        return list(tokens)

    # Optimized longest-match algorithm
    tokens_list = list(tokens)
    n_tokens = len(tokens_list)
    result: list[int] = []
    i = 0

    while i < n_tokens:
        # Try longest match first (greedy)
        matched = False
        for n in range(min(max_ngram, n_tokens - i), 1, -1):  # n >= 2
            ngram = tuple(tokens_list[i : i + n])
            superword_id = superword_table.get(ngram)
            if superword_id is not None:
                result.append(superword_id)
                i += n
                matched = True
                break

        if not matched:
            result.append(tokens_list[i])
            i += 1

    return result


def create_trie_from_superword_table(
    superword_table: dict[tuple[int, ...], int],
) -> SuperwordTrieHandle:
    """Create SuperwordTrieHandle from a superword table.

    Convenience function for migrating from dict-based superword storage
    to the optimized trie structure.

    Args:
        superword_table: Dict mapping n-gram tuples to superword IDs.

    Returns:
        Populated SuperwordTrieHandle.
    """
    trie = SuperwordTrieHandle()

    if not superword_table:
        return trie

    ngrams = list(superword_table.keys())
    ids = [superword_table[ng] for ng in ngrams]
    trie.insert_batch(ngrams, ids)

    return trie


__all__ = [
    "ops_available",
    "SuperwordTrieHandle",
    "fused_text_tokenize",
    "fused_text_tokenize_batch",
    "trie_apply_merges",
    "create_trie_from_superword_table",
]
