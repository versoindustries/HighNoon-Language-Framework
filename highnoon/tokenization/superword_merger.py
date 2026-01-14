# highnoon/tokenization/superword_merger.py
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

"""Enterprise-grade Superword Merger with native C++ trie acceleration.

Phase 1000+ Enhancement: High-performance n-gram merging using SIMD-optimized
C++ trie for O(max_ngram_size) lookup complexity. Streaming n-gram counting
to minimize memory during training.

The merger learns common n-gram patterns from training corpus and merges them
into superword units, reducing sequence length while preserving semantics.

Performance Characteristics:
    - Training: O(T × N) where T=texts, N=avg length (streaming, low memory)
    - Lookup: O(max_ngram_size) per position (constant, not O(vocab))
    - Memory: O(V × avg_ngram_len) where V=superword count

Example:
    >>> merger = SuperwordMerger(base_vocab_size=512)
    >>> # Train on corpus
    >>> merger.train([tokenized_sequence1, tokenized_sequence2])
    >>> # Apply merges using native C++ trie
    >>> merged = merger.apply(original_tokens)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Native C++ trie for O(N) lookup
_native_trie_available = False
_SuperwordTrieHandle = None

try:
    from highnoon._native.ops.fused_text_tokenizer import SuperwordTrieHandle, ops_available
    from highnoon._native.ops.fused_text_tokenizer import (
        trie_apply_merges as native_trie_apply_merges,
    )

    if ops_available():
        _SuperwordTrieHandle = SuperwordTrieHandle
        _native_trie_available = True
        logger.info("[SuperwordMerger] Native C++ trie acceleration enabled")
except ImportError:
    native_trie_apply_merges = None
    logger.debug("[SuperwordMerger] Native trie not available")


@dataclass
class SuperwordEntry:
    """A superword entry in the merge table.

    Attributes:
        token_ids: The original token IDs that form this superword.
        superword_id: The assigned ID for this superword.
        frequency: How often this n-gram appeared in training.
        text_repr: Optional text representation for debugging.
    """

    token_ids: tuple[int, ...]
    superword_id: int
    frequency: int = 0
    text_repr: str = ""


@dataclass
class SuperwordMergerConfig:
    """Configuration for the SuperwordMerger.

    Attributes:
        min_frequency: Minimum n-gram frequency to consider for merging.
        max_vocab_size: Maximum number of superwords to create.
        min_ngram_size: Minimum n-gram size (default: 2).
        max_ngram_size: Maximum n-gram size (default: 5).
        byte_offset: Byte offset used by the tokenizer.
    """

    min_frequency: int = 100
    max_vocab_size: int = 10000
    min_ngram_size: int = 2
    max_ngram_size: int = 5
    byte_offset: int = 32


class SuperwordMerger:
    """Enterprise-grade Superword Merger with native C++ acceleration.

    This class provides learned merging of frequent n-grams into superword
    tokens, reducing sequence length while preserving semantic information.
    Uses SIMD-optimized C++ trie for O(max_ngram_size) lookup complexity.

    Example:
        >>> merger = SuperwordMerger(base_vocab_size=512)
        >>> # Train on corpus
        >>> merger.train([tokenized_sequence1, tokenized_sequence2])
        >>> # Apply merges
        >>> merged = merger.apply(original_tokens)
    """

    def __init__(
        self,
        base_vocab_size: int,
        config: SuperwordMergerConfig | None = None,
    ) -> None:
        """Initialize the SuperwordMerger.

        Args:
            base_vocab_size: The vocabulary size of the base tokenizer.
                Superword IDs will start after this.
            config: Optional configuration. Uses defaults if not provided.
        """
        self.base_vocab_size = base_vocab_size
        self.config = config or SuperwordMergerConfig()

        # Merge table: maps n-gram tuple -> SuperwordEntry (Python side for save/load)
        self._superword_table: dict[tuple[int, ...], SuperwordEntry] = {}

        # Reverse lookup: superword_id -> SuperwordEntry
        self._id_to_entry: dict[int, SuperwordEntry] = {}

        # Next available superword ID
        self._next_superword_id = base_vocab_size

        # Statistics
        self._total_merges_applied = 0

        # Native C++ trie (None until trained, then built for O(N) lookup)
        self._native_trie: Any = None

    @property
    def superword_count(self) -> int:
        """Return the number of learned superwords."""
        return len(self._superword_table)

    @property
    def superword_table(self) -> dict[tuple[int, ...], SuperwordEntry]:
        """Return the superword merge table."""
        return self._superword_table

    @property
    def has_native_trie(self) -> bool:
        """Check if native C++ trie is available and built."""
        return self._native_trie is not None

    def train(
        self,
        token_sequences: Sequence[Sequence[int]],
        progress_callback: Any = None,
    ) -> int:
        """Train the merger by learning frequent n-grams.

        Uses streaming n-gram counting for minimal memory footprint.
        Automatically builds native C++ trie after training.

        Args:
            token_sequences: Iterable of token ID sequences from the corpus.
            progress_callback: Optional callback for progress updates.

        Returns:
            Number of superwords learned.
        """
        logger.info(
            "Training SuperwordMerger on %d sequences (n-gram range: %d-%d)",
            len(token_sequences),
            self.config.min_ngram_size,
            self.config.max_ngram_size,
        )

        # Streaming n-gram counting - process sequences one at a time
        ngram_counts: Counter[tuple[int, ...]] = Counter()
        total_sequences = len(token_sequences)
        interrupted = False

        try:
            for seq_idx, sequence in enumerate(token_sequences):
                seq_list = list(sequence)
                # Stream n-grams directly into counter (no list materialization)
                for n in range(self.config.min_ngram_size, self.config.max_ngram_size + 1):
                    ngram_counts.update(self._extract_ngrams_gen(seq_list, n))

                # Progress callback at intervals
                if progress_callback and (seq_idx + 1) % 1000 == 0:
                    progress_callback(seq_idx + 1, total_sequences)

        except KeyboardInterrupt:
            logger.warning(
                "SuperwordMerger training interrupted at sequence %d/%d. "
                "Using partial n-gram counts.",
                seq_idx,
                total_sequences,
            )
            interrupted = True

        if interrupted:
            logger.info(
                "Continuing with %d n-grams collected before interrupt",
                len(ngram_counts),
            )

        # Filter by minimum frequency and sort by frequency
        qualified = [
            (ngram, count)
            for ngram, count in ngram_counts.items()
            if count >= self.config.min_frequency
        ]
        qualified.sort(key=lambda x: (-x[1], x[0]))  # Descending by frequency

        # Take top vocab_size entries
        top_merges = qualified[: self.config.max_vocab_size]

        # Build merge table
        for ngram, freq in top_merges:
            superword_id = self._next_superword_id
            self._next_superword_id += 1

            entry = SuperwordEntry(
                token_ids=ngram,
                superword_id=superword_id,
                frequency=freq,
            )
            self._superword_table[ngram] = entry
            self._id_to_entry[superword_id] = entry

        # Build native C++ trie for fast lookup
        self._build_native_trie()

        status = "interrupted" if interrupted else "complete"
        logger.info(
            "SuperwordMerger training %s: %d superwords learned (native_trie=%s)",
            status,
            len(self._superword_table),
            self._native_trie is not None,
        )
        return len(self._superword_table)

    def _build_native_trie(self) -> None:
        """Build native C++ trie from superword table."""
        if not _native_trie_available or not self._superword_table:
            self._native_trie = None
            return

        try:
            self._native_trie = _SuperwordTrieHandle()
            ngrams = list(self._superword_table.keys())
            ids = [self._superword_table[ng].superword_id for ng in ngrams]
            self._native_trie.insert_batch(ngrams, ids)
            logger.debug("Built native C++ trie with %d entries", len(ngrams))
        except Exception as e:
            logger.warning("Failed to build native trie: %s", e)
            self._native_trie = None

    def apply(self, token_ids: Sequence[int]) -> list[int]:
        """Apply learned superword merges to a token sequence.

        Uses optimized longest-match algorithm with O(N × max_ngram) complexity.

        Args:
            token_ids: Original token IDs from the base tokenizer.

        Returns:
            Token IDs with superword merges applied.

        Raises:
            RuntimeError: If native trie is not available.
        """
        if not self._superword_table:
            return list(token_ids)

        if self._native_trie is None:
            raise RuntimeError(
                "SuperwordMerger.apply() requires native C++ trie. "
                "Rebuild native ops with build_secure.sh."
            )

        if native_trie_apply_merges is None:
            raise RuntimeError(
                "Native trie_apply_merges not available. Rebuild native ops with build_secure.sh."
            )

        # Build ID-only lookup table for efficiency
        id_table = {ngram: entry.superword_id for ngram, entry in self._superword_table.items()}

        # Use optimized longest-match algorithm
        merged = native_trie_apply_merges(
            list(token_ids),
            self._native_trie,
            superword_table=id_table,
            max_ngram=self.config.max_ngram_size,
        )
        self._total_merges_applied += len(token_ids) - len(merged)
        return merged

    def reverse_merge(self, token_ids: Sequence[int]) -> list[int]:
        """Reverse superword merges back to original tokens.

        Args:
            token_ids: Token IDs potentially containing superword IDs.

        Returns:
            Original token IDs with superwords expanded.
        """
        result: list[int] = []

        for token_id in token_ids:
            if token_id in self._id_to_entry:
                # Expand superword to original tokens
                result.extend(self._id_to_entry[token_id].token_ids)
            else:
                result.append(token_id)

        return result

    def save(self, path: str | Path) -> None:
        """Save the merger state to a file.

        Args:
            path: Path to save the merger state (JSON format).
        """
        state = {
            "base_vocab_size": self.base_vocab_size,
            "config": {
                "min_frequency": self.config.min_frequency,
                "max_vocab_size": self.config.max_vocab_size,
                "min_ngram_size": self.config.min_ngram_size,
                "max_ngram_size": self.config.max_ngram_size,
                "byte_offset": self.config.byte_offset,
            },
            "superwords": [
                {
                    "token_ids": list(entry.token_ids),
                    "superword_id": entry.superword_id,
                    "frequency": entry.frequency,
                    "text_repr": entry.text_repr,
                }
                for entry in self._superword_table.values()
            ],
            "next_id": self._next_superword_id,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info("SuperwordMerger saved to %s (%d superwords)", path, len(self._superword_table))

    @classmethod
    def load(cls, path: str | Path) -> SuperwordMerger:
        """Load a merger from a saved file.

        Automatically rebuilds native C++ trie on load.

        Args:
            path: Path to the saved merger state.

        Returns:
            Loaded SuperwordMerger instance.
        """
        with open(path) as f:
            state = json.load(f)

        config = SuperwordMergerConfig(**state["config"])
        merger = cls(base_vocab_size=state["base_vocab_size"], config=config)

        for entry_data in state["superwords"]:
            entry = SuperwordEntry(
                token_ids=tuple(entry_data["token_ids"]),
                superword_id=entry_data["superword_id"],
                frequency=entry_data["frequency"],
                text_repr=entry_data.get("text_repr", ""),
            )
            merger._superword_table[entry.token_ids] = entry
            merger._id_to_entry[entry.superword_id] = entry

        merger._next_superword_id = state.get("next_id", merger.base_vocab_size)

        # Rebuild native trie
        merger._build_native_trie()

        logger.info(
            "SuperwordMerger loaded from %s (%d superwords, native_trie=%s)",
            path,
            len(merger._superword_table),
            merger._native_trie is not None,
        )
        return merger

    def _extract_ngrams_gen(self, tokens: list[int], n: int):
        """Generator-based n-gram extraction for streaming counting.

        Args:
            tokens: List of token IDs.
            n: N-gram size.

        Yields:
            N-gram tuples.
        """
        for i in range(len(tokens) - n + 1):
            yield tuple(tokens[i : i + n])

    def get_stats(self) -> dict[str, Any]:
        """Get merger statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            "superword_count": len(self._superword_table),
            "total_merges_applied": self._total_merges_applied,
            "base_vocab_size": self.base_vocab_size,
            "next_superword_id": self._next_superword_id,
            "native_trie_available": _native_trie_available,
            "native_trie_built": self._native_trie is not None,
        }


__all__ = ["SuperwordMerger", "SuperwordMergerConfig", "SuperwordEntry"]
