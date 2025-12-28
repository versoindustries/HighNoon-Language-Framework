# highnoon/data/streaming_tokenizer.py
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

"""Streaming HD Tokenizer for Memory-Efficient Data Loading.

Phase 200+: Quantum-Enhanced Tokenizer Memory Optimization

This module provides streaming tokenization that compresses samples into
a HolographicCorpus on-the-fly. It processes HuggingFace datasets as true
streams, never materializing the entire dataset in memory.

Key Features:
    - True streaming from HuggingFace datasets
    - On-the-fly HD compression
    - Amplitude-based importance sampling
    - Incremental tokenizer learning (for AdaptiveQWTTokenizer)

Memory: O(reservoir_size × hd_dim) instead of O(n_samples × seq_len)

Usage:
    >>> streamer = StreamingHDTokenizer()
    >>> corpus = streamer.process_stream(
    ...     hf_dataset_name="databricks/dolly-15k",
    ...     tokenizer=AdaptiveQWTTokenizer(),
    ...     reservoir_size=2000,
    ... )
    >>> dataset = corpus.create_dataset(batch_size=16)

References:
    - HighNoon hd_corpus.py
    - HuggingFace datasets streaming
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from highnoon.data.hd_corpus import HDCorpusConfig, HolographicCorpus

logger = logging.getLogger(__name__)


def _load_hf_dataset_with_retry(
    hf_dataset_name: str,
    split: str = "train",
    max_retries: int = 3,
    initial_delay: float = 2.0,
) -> Any:
    """Load HuggingFace dataset with retry logic for network errors.

    Args:
        hf_dataset_name: Name of the HuggingFace dataset.
        split: Dataset split to load.
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries (doubles each retry).

    Returns:
        Loaded streaming dataset.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    from datasets import load_dataset as hf_load_dataset

    retry_delay = initial_delay

    for attempt in range(max_retries):
        try:
            return hf_load_dataset(
                hf_dataset_name,
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
        except (ConnectionError, TimeoutError, OSError) as net_err:
            if attempt < max_retries - 1:
                logger.warning(
                    "[HD Stream] Network error (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    net_err,
                    retry_delay,
                )
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ["connection", "timeout", "network", "socket", "ssl"]):
                if attempt < max_retries - 1:
                    logger.warning(
                        "[HD Stream] Network error (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1,
                        max_retries,
                        e,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
            else:
                raise  # Non-network error, don't retry

    raise RuntimeError(f"Failed to load dataset after {max_retries} attempts")


class StreamingHDTokenizer:
    """Streaming tokenizer that compresses samples into HD corpus.

    Processes HuggingFace datasets as true streams:
    1. Tokenizes each sample
    2. Computes amplitude (information content)
    3. Adds to reservoir with weighted sampling
    4. Never stores raw token sequences

    Also supports incremental vocabulary learning for AdaptiveQWTTokenizer.

    Attributes:
        log_frequency: How often to log progress.
        learn_vocab_samples: Number of samples to use for vocab learning.
    """

    def __init__(
        self,
        log_frequency: int = 1000,
        learn_vocab_samples: int = 5000,
    ) -> None:
        """Initialize StreamingHDTokenizer.

        Args:
            log_frequency: Log progress every N samples.
            learn_vocab_samples: Samples to use for tokenizer vocab learning.
        """
        self.log_frequency = log_frequency
        self.learn_vocab_samples = learn_vocab_samples

    def process_stream(
        self,
        hf_dataset_name: str,
        tokenizer: Any,
        text_column: str | None = None,
        split: str = "train",
        reservoir_size: int = 2000,
        hd_dim: int = 1024,
        max_seq_len: int = 256,
        max_samples: int | None = None,
        vocab_size: int | None = None,
        learn_vocab: bool = True,
    ) -> HolographicCorpus:
        """Process streaming dataset into holographic corpus.

        Args:
            hf_dataset_name: HuggingFace dataset identifier.
            tokenizer: Tokenizer instance (QWTTextTokenizer or AdaptiveQWTTokenizer).
            text_column: Column containing text. Auto-detected if None.
            split: Dataset split to use.
            reservoir_size: Maximum samples in HD corpus.
            hd_dim: Holographic dimension.
            max_seq_len: Maximum sequence length.
            max_samples: Maximum samples to process (None = all).
            vocab_size: Vocabulary size. Uses tokenizer.vocab_size if None.
            learn_vocab: Whether to learn vocabulary from stream first.

        Returns:
            HolographicCorpus with compressed samples.
        """
        logger.info(
            "[HD Stream] Starting: dataset=%s, reservoir=%d, hd_dim=%d",
            hf_dataset_name,
            reservoir_size,
            hd_dim,
        )

        # Determine vocab size
        effective_vocab_size = vocab_size or getattr(tokenizer, "vocab_size", 50000)

        # Create HD corpus
        config = HDCorpusConfig(
            hd_dim=hd_dim,
            reservoir_size=reservoir_size,
            max_seq_len=max_seq_len,
            vocab_size=effective_vocab_size,
        )
        corpus = HolographicCorpus(config=config)

        # Load streaming dataset with retry logic for network errors
        try:
            hf_dataset = _load_hf_dataset_with_retry(hf_dataset_name, split=split)
            logger.info("[HD Stream] Dataset loaded in streaming mode")
        except ImportError:
            logger.error("[HD Stream] HuggingFace datasets not installed")
            raise
        except Exception as e:
            logger.error("[HD Stream] Failed to load dataset: %s", e)
            raise

        # Phase 1: Vocab learning (if using AdaptiveQWTTokenizer)
        if learn_vocab and hasattr(tokenizer, "learn_from_corpus"):
            if not getattr(tokenizer, "is_trained", True):
                logger.info(
                    "[HD Stream] Learning vocabulary from first %d samples...",
                    self.learn_vocab_samples,
                )
                vocab_texts = []
                for i, sample in enumerate(hf_dataset):
                    if i >= self.learn_vocab_samples:
                        break
                    text = self._extract_text(sample, text_column)
                    if text:
                        vocab_texts.append(text)

                if vocab_texts:
                    learned = tokenizer.learn_from_corpus(vocab_texts, min_freq=10)
                    logger.info(
                        "[HD Stream] Learned %d n-grams, vocab_size=%d",
                        learned,
                        tokenizer.vocab_size,
                    )
                    # Update corpus vocab size
                    corpus.config.vocab_size = tokenizer.vocab_size
                    # Reinitialize base vectors with correct size
                    self._reinit_corpus_vectors(corpus, tokenizer.vocab_size)

                # Reload dataset for processing (streaming consumes iterator)
                hf_dataset = _load_hf_dataset_with_retry(hf_dataset_name, split=split)

        # Phase 2: Stream tokenize and compress
        logger.info("[HD Stream] Beginning streaming tokenization...")

        processed = 0
        accepted = 0
        text_col_detected = text_column

        for sample in hf_dataset:
            # Detect text column from first sample
            if text_col_detected is None:
                text_col_detected = self._detect_text_column(sample)
                logger.info("[HD Stream] Using text column: %s", text_col_detected)

            # Extract text
            text = sample.get(text_col_detected, "")
            if not text or not isinstance(text, str):
                continue

            # Tokenize
            try:
                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_seq_len,
                    padding=False,  # No padding for compression
                    add_special_tokens=True,
                )
                token_ids = encoding["input_ids"]

                # Skip very short sequences
                if len(token_ids) < 4:
                    continue

                # Add to corpus (reservoir sampling)
                was_added = corpus.add_sample(np.array(token_ids))
                if was_added:
                    accepted += 1

            except Exception as e:
                logger.debug("[HD Stream] Tokenization failed for sample %d: %s", processed, e)
                continue

            processed += 1

            # Progress logging
            if processed % self.log_frequency == 0:
                logger.info(
                    "[HD Stream] Processed %d samples, reservoir: %d/%d",
                    processed,
                    len(corpus.bundles),
                    reservoir_size,
                )

            # Check max samples limit
            if max_samples is not None and processed >= max_samples:
                logger.info("[HD Stream] Reached max_samples limit: %d", max_samples)
                break

        # Log final statistics
        stats = corpus.get_statistics()
        logger.info(
            "[HD Stream] Complete: processed=%d, in_reservoir=%d, memory=%.2f MB",
            stats["total_samples_seen"],
            stats["samples_in_reservoir"],
            stats["memory_bytes"] / (1024 * 1024),
        )

        return corpus

    def process_texts(
        self,
        texts: list[str],
        tokenizer: Any,
        reservoir_size: int = 2000,
        hd_dim: int = 1024,
        max_seq_len: int = 256,
        vocab_size: int | None = None,
    ) -> HolographicCorpus:
        """Process a list of texts into holographic corpus.

        Convenience method for non-streaming use cases.

        Args:
            texts: List of text strings.
            tokenizer: Tokenizer instance.
            reservoir_size: Maximum samples in HD corpus.
            hd_dim: Holographic dimension.
            max_seq_len: Maximum sequence length.
            vocab_size: Vocabulary size.

        Returns:
            HolographicCorpus with compressed samples.
        """
        effective_vocab_size = vocab_size or getattr(tokenizer, "vocab_size", 50000)

        config = HDCorpusConfig(
            hd_dim=hd_dim,
            reservoir_size=reservoir_size,
            max_seq_len=max_seq_len,
            vocab_size=effective_vocab_size,
        )
        corpus = HolographicCorpus(config=config)

        for i, text in enumerate(texts):
            if not text:
                continue

            try:
                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_seq_len,
                    padding=False,
                    add_special_tokens=True,
                )
                token_ids = encoding["input_ids"]

                if len(token_ids) >= 4:
                    corpus.add_sample(np.array(token_ids))

            except Exception as e:
                logger.debug("[HD Stream] Failed on text %d: %s", i, e)
                continue

            if (i + 1) % self.log_frequency == 0:
                logger.info("[HD Stream] Processed %d/%d texts", i + 1, len(texts))

        return corpus

    def _extract_text(self, sample: dict, text_column: str | None) -> str:
        """Extract text from sample."""
        if text_column:
            return sample.get(text_column, "")

        # Try common column names
        for col in ["text", "content", "context", "instruction", "input", "response"]:
            if col in sample and isinstance(sample[col], str):
                return sample[col]

        return ""

    def _detect_text_column(self, sample: dict) -> str | None:
        """Auto-detect text column from sample."""
        priority_columns = [
            "context",
            "instruction",
            "text",
            "input",
            "response",
            "content",
        ]

        available = list(sample.keys())

        for col in priority_columns:
            if col in available and isinstance(sample.get(col), str):
                return col

        # Fall back to first string column
        for col in available:
            if isinstance(sample.get(col), str):
                return col

        return None

    def _reinit_corpus_vectors(self, corpus: HolographicCorpus, new_vocab_size: int) -> None:
        """Reinitialize corpus base vectors for new vocab size."""
        import tensorflow as tf

        scale = np.sqrt(2.0 / (new_vocab_size + corpus.config.hd_dim))
        corpus.base_vectors = tf.Variable(
            tf.random.normal([new_vocab_size, corpus.config.hd_dim], stddev=scale),
            trainable=False,
            name="hd_base_vectors",
        )
        corpus.config.vocab_size = new_vocab_size
        logger.info("[HD Stream] Reinitialized base vectors for vocab_size=%d", new_vocab_size)


__all__ = ["StreamingHDTokenizer"]
