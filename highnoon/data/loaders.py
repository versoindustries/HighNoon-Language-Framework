"""HighNoon Data Loaders - Unified dataset loading utilities.

This module provides functions for loading training datasets in a consistent format
for both curriculum-based training and HPO trial evaluation.

Phase 11: Integrated QWTTextTokenizer for proper text tokenization.
Phase 10.2: Integrated SuperwordMerger for semantic n-gram grouping.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import tensorflow as tf

from highnoon import config
from highnoon.tokenization import QWTTextTokenizer, SuperwordMerger, SuperwordMergerConfig

logger = logging.getLogger(__name__)

# Sample text data for testing (short phrases for quick HPO validation)
_SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks learn patterns from data.",
    "Transformers revolutionized natural language processing.",
    "Attention mechanisms allow models to focus on relevant information.",
    "Language models generate text token by token.",
    "Training requires large datasets and compute resources.",
    "Embeddings represent words as dense vectors.",
    "Backpropagation updates weights to minimize loss.",
    "Gradient descent optimizes neural network parameters.",
    "Python is commonly used for deep learning research.",
    "TensorFlow and PyTorch are popular frameworks.",
    "Tokenization converts text to numeric sequences.",
    "The model learns to predict the next token.",
    "Transfer learning reuses pretrained model weights.",
    "Fine-tuning adapts models to specific tasks.",
    "Hyperparameter optimization improves model performance.",
    "Batch normalization stabilizes training dynamics.",
    "Dropout prevents overfitting during training.",
    "Learning rate schedules adjust optimization speed.",
]


def create_tokenizer(
    vocab_size: int = 512,
    max_length: int = 256,
) -> QWTTextTokenizer:
    """Create a tokenizer instance.

    Args:
        vocab_size: Vocabulary size for the tokenizer.
        max_length: Maximum sequence length.

    Returns:
        Configured QWTTextTokenizer instance.
    """
    return QWTTextTokenizer(
        vocab_size=vocab_size,
        model_max_length=max_length,
        enable_thinking_tokens=True,
    )


def load_training_dataset(
    batch_size: int = 16,
    sequence_length: int = 256,
    vocab_size: int | None = None,  # Defaults to config.VOCAB_SIZE
    tokenizer: QWTTextTokenizer | None = None,
    use_dummy_embeddings: bool = False,
    curriculum_id: str | None = None,
    dataset_name: str | None = None,
    split: str = "train",
    shuffle: bool = True,
    shuffle_buffer: int = 1000,
    hf_dataset_name: str | None = None,
    text_column: str | None = None,
    max_samples: int | None = None,
    enable_superwords: bool | None = None,
    superword_min_frequency: int | None = None,
    superword_max_vocab: int | None = None,
    prefetch_buffer_size: int | None = None,
    use_adaptive_tokenizer: bool = True,
    adaptive_min_freq: int = 10,
    # Phase 200+: HD Streaming Mode (Quantum-Enhanced Memory Optimization)
    use_hd_streaming: bool = False,
    hd_reservoir_size: int = 2000,
    hd_dim: int = 1024,
    hd_sample_length: int = 512,  # Max tokens per HD sample (affects position keys memory)
) -> tuple[tf.data.Dataset, QWTTextTokenizer, SuperwordMerger | None]:
    """Load a training dataset with tokenization and optional superword merging.

    Supports loading from HuggingFace datasets (for development/testing) or
    falls back to sample texts when no dataset is specified.

    When use_adaptive_tokenizer is enabled (default), uses AdaptiveQWTTokenizer
    which learns frequent n-grams from the corpus to expand vocabulary and
    compress sequences. This fixes the vocab_size mismatch issue.

    When superword merging is enabled, trains a SuperwordMerger on the tokenized
    corpus to learn frequent n-grams, then applies merges to reduce sequence length.

    Args:
        batch_size: Batch size for the dataset
        sequence_length: Sequence length for tokenization
        vocab_size: Vocabulary size for the tokenizer
        tokenizer: Optional pre-configured tokenizer
        use_dummy_embeddings: If True, return embeddings instead of token IDs
        curriculum_id: Optional curriculum ID to load specific data
        dataset_name: Optional dataset name to load
        split: Dataset split to load (train, validation, test)
        shuffle: Whether to shuffle the dataset
        shuffle_buffer: Buffer size for shuffling
        hf_dataset_name: Optional HuggingFace dataset identifier (e.g., "databricks/dolly-15k")
        text_column: Column name containing text in HuggingFace dataset
        max_samples: Maximum number of samples to load from HuggingFace dataset
        enable_superwords: Enable superword merging. Defaults to config.ENABLE_SUPERWORDS.
        superword_min_frequency: Min n-gram frequency. Defaults to config.SUPERWORD_MIN_FREQUENCY.
        superword_max_vocab: Max superwords to learn. Defaults to config.SUPERWORD_MAX_VOCAB_SIZE.
        prefetch_buffer_size: Number of batches to prefetch. If None, uses tf.data.AUTOTUNE.
            For memory-constrained HPO, set to 1-4. For high-throughput, use None (auto).
        use_adaptive_tokenizer: Use AdaptiveQWTTokenizer with learnable codebook (default: True).
        adaptive_min_freq: Minimum n-gram frequency for adaptive tokenizer learning.
        use_hd_streaming: Use HolographicCorpus for memory-efficient streaming (default: False).
            Enables quantum-enhanced compression with 5-20x memory reduction.
        hd_reservoir_size: Maximum samples in HD reservoir (default: 2000).
        hd_dim: Holographic dimension for compression (default: 1024).

    Returns:
        Tuple of (dataset, tokenizer, merger) where:
        - dataset yields (input_ids, labels)
        - tokenizer is the QWTTextTokenizer or AdaptiveQWTTokenizer instance
        - merger is the trained SuperwordMerger (or None if using adaptive tokenizer/HD streaming)
    """

    # Phase 200+: HD Streaming Mode (Quantum-Enhanced Memory Optimization)
    # This mode uses HolographicCorpus to compress samples into fixed-size HD bundles
    # with amplitude-based reservoir sampling, achieving 5-20x memory reduction.
    if use_hd_streaming and hf_dataset_name:
        logger.info(
            "[Data Loaders] Using HD Streaming Mode: reservoir=%d, hd_dim=%d",
            hd_reservoir_size,
            hd_dim,
        )

        from highnoon.data.streaming_tokenizer import StreamingHDTokenizer

        # Create tokenizer if not provided
        if tokenizer is None:
            if use_adaptive_tokenizer:
                from highnoon.tokenization import AdaptiveQWTTokenizer

                tokenizer = AdaptiveQWTTokenizer(
                    vocab_size=vocab_size or config.VOCAB_SIZE,
                    max_vocab_size=vocab_size,  # Hard cap from QAHPO budget
                    model_max_length=sequence_length,
                    min_ngram_size=2,
                    max_ngram_size=5,
                )
            else:
                tokenizer = create_tokenizer(
                    vocab_size=vocab_size or config.VOCAB_SIZE,
                    max_length=sequence_length,
                )

        # Create streaming HD tokenizer and process dataset
        hd_streamer = StreamingHDTokenizer(
            log_frequency=1000,
            learn_vocab_samples=min(5000, max_samples or 5000),
        )

        # PHASE 500+: Use hd_sample_length for HD corpus position keys
        # CRITICAL: Individual HD samples should be bounded (e.g., 128-2048 tokens), NOT full context_window
        # Long context is achieved through bundling across MANY samples in the reservoir,
        # not by allocating [context_window × hd_dim] for position keys (which would be 48GB for 1M × 12288!)
        # The context_window param refers to model's attention window, not HD sample length.
        # hd_sample_length is tunable through QAHPO/WebUI, defaults to 512
        hd_max_seq_len = min(hd_sample_length, sequence_length)  # Don't exceed sequence_length

        corpus = hd_streamer.process_stream(
            hf_dataset_name=hf_dataset_name,
            tokenizer=tokenizer,
            text_column=text_column,
            split=split,
            reservoir_size=hd_reservoir_size,
            hd_dim=hd_dim,
            max_seq_len=hd_max_seq_len,  # Use tunable hd_sample_length, not context_window
            max_samples=max_samples,
            vocab_size=tokenizer.vocab_size,
            learn_vocab=use_adaptive_tokenizer,
        )

        # Create dataset from HD corpus
        dataset = corpus.create_dataset(
            batch_size=batch_size,
            shuffle=shuffle,
            prefetch=prefetch_buffer_size,
        )

        logger.info(
            "[Data Loaders] HD Streaming complete: %d samples compressed, %.2f MB",
            len(corpus.bundles),
            corpus.get_statistics()["memory_bytes"] / (1024 * 1024),
        )

        return dataset, tokenizer, None  # merger is always None in HD mode

    # Resolve vocab_size from config if not provided
    if vocab_size is None:
        vocab_size = config.VOCAB_SIZE
        logger.info("[Data Loaders] Using config.VOCAB_SIZE=%d", vocab_size)

    # Create or use provided tokenizer
    if tokenizer is None:
        if use_adaptive_tokenizer:
            from highnoon.tokenization import AdaptiveQWTTokenizer

            tokenizer = AdaptiveQWTTokenizer(
                vocab_size=vocab_size,
                max_vocab_size=vocab_size,  # Hard cap from QAHPO budget
                model_max_length=sequence_length,
                min_ngram_size=2,
                max_ngram_size=5,
            )
            logger.info(
                "[Data Loaders] Using AdaptiveQWTTokenizer: target_vocab=%d, max_length=%d",
                vocab_size,
                sequence_length,
            )
        else:
            tokenizer = create_tokenizer(vocab_size=vocab_size, max_length=sequence_length)
            logger.info(
                "[Data Loaders] Using QWTTextTokenizer: vocab_size=%d, max_length=%d",
                tokenizer.vocab_size,
                tokenizer.model_max_length,
            )
    else:
        logger.info(
            "[Data Loaders] Using provided tokenizer: vocab_size=%d",
            tokenizer.vocab_size,
        )

    # Collect texts from HuggingFace dataset or fallback to sample texts
    all_token_ids = []
    raw_texts = []  # Collect raw texts for adaptive tokenizer learning

    if hf_dataset_name:
        # Load from HuggingFace dataset with streaming for efficiency
        logger.info("[Data Loaders] Loading HuggingFace dataset (streaming): %s", hf_dataset_name)
        try:
            from datasets import load_dataset as hf_load_dataset

            # Retry logic for transient network errors
            max_retries = 3
            retry_delay = 2.0  # seconds, doubles each retry
            hf_dataset = None

            for attempt in range(max_retries):
                try:
                    # Use streaming=True to avoid downloading entire dataset
                    hf_dataset = hf_load_dataset(
                        hf_dataset_name,
                        split=split,
                        streaming=True,  # Stream data on-the-fly
                        trust_remote_code=True,
                    )
                    break  # Success, exit retry loop
                except (ConnectionError, TimeoutError, OSError) as net_err:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "[Data Loaders] Network error loading dataset (attempt %d/%d): %s. Retrying in %.1fs...",
                            attempt + 1,
                            max_retries,
                            net_err,
                            retry_delay,
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise  # Re-raise on final attempt
                except Exception as e:
                    # Check if it's a requests/urllib network error
                    err_str = str(e).lower()
                    if any(
                        x in err_str for x in ["connection", "timeout", "network", "socket", "ssl"]
                    ):
                        if attempt < max_retries - 1:
                            logger.warning(
                                "[Data Loaders] Network error loading dataset (attempt %d/%d): %s. Retrying in %.1fs...",
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

            if hf_dataset is None:
                raise RuntimeError(f"Failed to load dataset after {max_retries} attempts")

            logger.info("[Data Loaders] Streaming dataset: %s", hf_dataset_name)

            # For streaming datasets, we need to peek at first sample for column detection
            first_sample = None
            text_column_detected = text_column

            # Tokenize samples from streaming dataset
            sample_count = 0
            max_to_load = max_samples or 10000  # Default limit for streaming

            for sample in hf_dataset:
                # Detect text column from first sample
                if first_sample is None:
                    first_sample = sample
                    if text_column_detected is None:
                        available_columns = list(sample.keys())
                        potential_columns = [
                            "context",
                            "instruction",
                            "text",
                            "input",
                            "response",
                            "content",
                        ]
                        for col in potential_columns:
                            if col in available_columns:
                                text_column_detected = col
                                break
                        if text_column_detected is None:
                            # Fall back to first string column
                            for col in available_columns:
                                if isinstance(sample.get(col), str):
                                    text_column_detected = col
                                    break
                        logger.info("[Data Loaders] Using text column: %s", text_column_detected)

                # Extract text (collect raw, tokenize later for adaptive learning)
                text = sample.get(text_column_detected) or ""
                if not text or not isinstance(text, str):
                    continue

                raw_texts.append(text)
                sample_count += 1

                # Respect max_samples limit
                if sample_count >= max_to_load:
                    break

                # Log progress periodically
                if sample_count % 1000 == 0:
                    logger.info("[Data Loaders] Tokenized %d samples...", sample_count)

            logger.info(
                "[Data Loaders] Collected %d text samples from HuggingFace",
                len(raw_texts),
            )

        except Exception as e:
            logger.error("[Data Loaders] Failed to load HuggingFace dataset: %s", e)
            logger.warning("[Data Loaders] Falling back to sample texts")
            hf_dataset_name = None  # Fall back to sample texts

    if not raw_texts:
        # Fallback: use built-in sample texts
        logger.info("[Data Loaders] Using built-in sample texts (%d samples)", len(_SAMPLE_TEXTS))
        raw_texts = list(_SAMPLE_TEXTS)

    # Train AdaptiveQWTTokenizer if enabled (before tokenizing)
    from highnoon.tokenization import AdaptiveQWTTokenizer

    if isinstance(tokenizer, AdaptiveQWTTokenizer) and not tokenizer.is_trained:
        logger.info(
            "[Data Loaders] Training AdaptiveQWTTokenizer on %d texts (min_freq=%d)...",
            len(raw_texts),
            adaptive_min_freq,
        )
        learned_count = tokenizer.learn_from_corpus(raw_texts, min_freq=adaptive_min_freq)
        logger.info(
            "[Data Loaders] AdaptiveQWTTokenizer learned %d n-grams, vocab_size=%d",
            learned_count,
            tokenizer.vocab_size,
        )

    # Now tokenize all collected texts
    logger.info("[Data Loaders] Tokenizing %d texts...", len(raw_texts))
    for text in raw_texts:
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=sequence_length,
            padding="max_length",
            add_special_tokens=True,
        )
        all_token_ids.append(encoding["input_ids"])

    # Convert to numpy array
    token_array = np.array(all_token_ids, dtype=np.int32)
    num_samples = len(token_array)

    # Validate token IDs are within vocab bounds
    actual_vocab_size = tokenizer.vocab_size
    max_token_id = int(np.max(token_array))
    min_token_id = int(np.min(token_array))

    logger.info(
        "[Data Loaders] Tokenized %d samples, shape=%s, " "token_range=[%d, %d], vocab_size=%d",
        num_samples,
        token_array.shape,
        min_token_id,
        max_token_id,
        actual_vocab_size,
    )

    if max_token_id >= actual_vocab_size:
        logger.error(
            "[Data Loaders] CRITICAL: max_token_id=%d >= vocab_size=%d. "
            "This will cause NaN loss! Clamping tokens to valid range.",
            max_token_id,
            actual_vocab_size,
        )
        # Clamp to valid range to prevent NaN
        token_array = np.clip(token_array, 0, actual_vocab_size - 1)

    # Phase 10.2: SuperwordMerger Integration
    # Train and apply superword merging if enabled
    # NOTE: Skip if using AdaptiveQWTTokenizer since it has its own internal merger
    use_superwords = (
        enable_superwords if enable_superwords is not None else config.ENABLE_SUPERWORDS
    )
    merger: SuperwordMerger | None = None

    # AdaptiveQWTTokenizer has its own merger, don't double-apply
    if isinstance(tokenizer, AdaptiveQWTTokenizer):
        use_superwords = False
        if tokenizer.merger is not None:
            logger.info(
                "[Data Loaders] Using AdaptiveQWTTokenizer's internal merger (%d superwords)",
                tokenizer.merger.superword_count,
            )

    if use_superwords and num_samples > 0:
        logger.info("[Data Loaders] Training SuperwordMerger on %d sequences...", num_samples)

        # Get config values with fallbacks
        min_freq = (
            superword_min_frequency
            if superword_min_frequency is not None
            else getattr(config, "SUPERWORD_MIN_FREQUENCY", 100)
        )
        max_vocab = (
            superword_max_vocab
            if superword_max_vocab is not None
            else getattr(config, "SUPERWORD_MAX_VOCAB_SIZE", 10000)
        )
        min_ngram = getattr(config, "SUPERWORD_MIN_NGRAM", 2)
        max_ngram = getattr(config, "SUPERWORD_MAX_NGRAM", 5)

        # Create and train merger
        merger_config = SuperwordMergerConfig(
            min_frequency=min_freq,
            max_vocab_size=max_vocab,
            min_ngram_size=min_ngram,
            max_ngram_size=max_ngram,
        )
        merger = SuperwordMerger(base_vocab_size=vocab_size, config=merger_config)

        # Train on tokenized sequences
        superword_count = merger.train(list(token_array))
        logger.info(
            "[Data Loaders] SuperwordMerger trained: %d superwords learned (min_freq=%d)",
            superword_count,
            min_freq,
        )

        # Apply merges to all sequences
        if superword_count > 0:
            merged_sequences = [merger.apply(seq) for seq in token_array]

            # Pad/truncate merged sequences to uniform length
            # Merged sequences are shorter, so we need to re-pad
            max_merged_len = max(len(seq) for seq in merged_sequences)
            target_len = min(max_merged_len, sequence_length)

            padded_merged = []
            for seq in merged_sequences:
                if len(seq) >= target_len:
                    padded_merged.append(seq[:target_len])
                else:
                    # Pad with 0 (padding token)
                    padded_merged.append(seq + [0] * (target_len - len(seq)))

            token_array = np.array(padded_merged, dtype=np.int32)

            # Calculate compression stats
            original_tokens = sum(len(seq) for seq in all_token_ids)
            merged_tokens = sum(len(seq) for seq in merged_sequences)
            compression_ratio = (
                1.0 - (merged_tokens / original_tokens) if original_tokens > 0 else 0.0
            )

            logger.info(
                "[Data Loaders] SuperwordMerger applied: %d → %d tokens (%.1f%% reduction), new shape=%s",
                original_tokens,
                merged_tokens,
                compression_ratio * 100,
                token_array.shape,
            )
        else:
            logger.info(
                "[Data Loaders] No superwords learned (corpus may be too small or min_freq too high)"
            )

    # Get actual sequence length from token_array (may be shorter after superword merging)
    actual_seq_len = token_array.shape[1] if len(token_array.shape) > 1 else sequence_length

    def data_generator():
        """Generate training data from tokenized texts."""
        indices = list(range(num_samples))

        while True:
            if shuffle:
                np.random.shuffle(indices)

            for start_idx in range(0, num_samples - batch_size + 1, batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]
                batch_tokens = token_array[batch_indices]

                if use_dummy_embeddings:
                    # For models that expect embeddings, convert to embeddings
                    # This is a placeholder - real embedding would come from model
                    inputs = tf.random.normal((batch_size, actual_seq_len, vocab_size))
                    labels = tf.random.normal((batch_size, actual_seq_len, vocab_size))
                else:
                    # Language modeling: input tokens, labels shifted by 1
                    # Input: tokens[:-1], Labels: tokens[1:]
                    inputs = batch_tokens[:, :-1]
                    labels = batch_tokens[:, 1:]

                    # Convert to tensors
                    inputs = tf.constant(inputs, dtype=tf.int32)
                    labels = tf.constant(labels, dtype=tf.int32)

                yield inputs, labels

    if use_dummy_embeddings:
        output_signature = (
            tf.TensorSpec(shape=(batch_size, actual_seq_len, vocab_size), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, actual_seq_len, vocab_size), dtype=tf.float32),
        )
    else:
        # Use actual sequence length minus 1 (for shifted labels)
        output_signature = (
            tf.TensorSpec(shape=(batch_size, actual_seq_len - 1), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, actual_seq_len - 1), dtype=tf.int32),
        )

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature,
    )

    # Prefetch for performance (configurable for memory-constrained environments)
    if prefetch_buffer_size is not None:
        dataset = dataset.prefetch(prefetch_buffer_size)
        logger.info(f"[Data Loaders] Using prefetch buffer size: {prefetch_buffer_size}")
    else:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, tokenizer, merger


def load_training_dataset_legacy(
    batch_size: int = 16,
    sequence_length: int = 512,
    input_dim: int = 256,
    output_dim: int = 1024,
    curriculum_id: str | None = None,
    dataset_name: str | None = None,
    split: str = "train",
    shuffle: bool = True,
    shuffle_buffer: int = 10000,
) -> tf.data.Dataset:
    """Legacy loader returning dummy embeddings (for backward compatibility).

    This function maintains the original API for existing code that expects
    embedding inputs rather than token IDs.
    """
    logger.warning(
        "[Data Loaders] Using legacy dummy embeddings. "
        "Use load_training_dataset() with tokenizer for production."
    )

    def dummy_generator():
        """Generate dummy training data for testing."""
        while True:
            inputs = tf.random.normal((batch_size, sequence_length, input_dim))
            labels = tf.random.normal((batch_size, sequence_length, output_dim))
            yield inputs, labels

    dataset = tf.data.Dataset.from_generator(
        dummy_generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, sequence_length, input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, sequence_length, output_dim), dtype=tf.float32),
        ),
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_curriculum_datasets(curriculum_id: str) -> list[dict[str, Any]]:
    """Get dataset information for a curriculum.

    Args:
        curriculum_id: Curriculum identifier

    Returns:
        List of dataset configurations from the curriculum
    """
    # TODO: Load from backend API or local storage
    logger.info(f"[Data Loaders] Loading curriculum datasets for: {curriculum_id}")
    return []


__all__ = [
    "create_tokenizer",
    "load_training_dataset",
    "load_training_dataset_legacy",
    "get_curriculum_datasets",
]
