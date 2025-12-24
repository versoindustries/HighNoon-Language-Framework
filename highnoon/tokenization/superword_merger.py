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

"""Superword Merger for semantic n-gram grouping.

Phase 10.2 Enhancement: Provides internal semantic grouping layer for learning
common n-gram patterns and merging them into superword units. This improves
tokenization quality for domain-specific terms and common phrases without
requiring an external tokenizer.

The merger works by:
1. Analyzing training corpus for frequent n-grams (2-5 tokens)
2. Building a merge table that maps frequent n-grams to superword IDs
3. Applying merges during encoding to reduce sequence length
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    """Superword Merger for semantic n-gram grouping.

    This class provides learned merging of frequent n-grams into superword
    tokens, reducing sequence length while preserving semantic information.

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

        # Merge table: maps n-gram tuple -> SuperwordEntry
        self._superword_table: dict[tuple[int, ...], SuperwordEntry] = {}

        # Reverse lookup: superword_id -> SuperwordEntry
        self._id_to_entry: dict[int, SuperwordEntry] = {}

        # Next available superword ID
        self._next_superword_id = base_vocab_size

        # Statistics
        self._total_merges_applied = 0

    @property
    def superword_count(self) -> int:
        """Return the number of learned superwords."""
        return len(self._superword_table)

    @property
    def superword_table(self) -> dict[tuple[int, ...], SuperwordEntry]:
        """Return the superword merge table."""
        return self._superword_table

    def train(
        self,
        token_sequences: Sequence[Sequence[int]],
        progress_callback: Any = None,
    ) -> int:
        """Train the merger by learning frequent n-grams.

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

        # Count n-gram frequencies
        ngram_counts: Counter[tuple[int, ...]] = Counter()

        for seq_idx, sequence in enumerate(token_sequences):
            if progress_callback and seq_idx % 1000 == 0:
                progress_callback(seq_idx, len(token_sequences))

            seq_list = list(sequence)
            for n in range(self.config.min_ngram_size, self.config.max_ngram_size + 1):
                for ngram in self._extract_ngrams(seq_list, n):
                    ngram_counts[ngram] += 1

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

        logger.info(
            "SuperwordMerger training complete: %d superwords learned",
            len(self._superword_table),
        )
        return len(self._superword_table)

    def apply(self, token_ids: Sequence[int]) -> list[int]:
        """Apply learned superword merges to a token sequence.

        Args:
            token_ids: Original token IDs from the base tokenizer.

        Returns:
            Token IDs with superword merges applied.
        """
        if not self._superword_table:
            return list(token_ids)

        result: list[int] = []
        tokens = list(token_ids)
        i = 0

        while i < len(tokens):
            # Try longest match first
            merged = False
            for n in range(self.config.max_ngram_size, self.config.min_ngram_size - 1, -1):
                if i + n <= len(tokens):
                    ngram = tuple(tokens[i : i + n])
                    if ngram in self._superword_table:
                        result.append(self._superword_table[ngram].superword_id)
                        i += n
                        self._total_merges_applied += 1
                        merged = True
                        break

            if not merged:
                result.append(tokens[i])
                i += 1

        return result

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

        logger.info(
            "SuperwordMerger loaded from %s (%d superwords)", path, len(merger._superword_table)
        )
        return merger

    def _extract_ngrams(self, tokens: list[int], n: int) -> Iterator[tuple[int, ...]]:
        """Extract n-grams from a token sequence.

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
        }


__all__ = ["SuperwordMerger", "SuperwordMergerConfig", "SuperwordEntry"]
