"""HighNoon Data Loaders - Enterprise Streaming Pipeline.

Phase 200: Memory-efficient streaming for 100K-5M token context training.

This module provides unified data loading with pure streaming capability:
- StreamingSequencePacker for large context windows (100K-5M tokens)
- Chunked mode for medium contexts with adaptive tokenization
- No legacy padding-based approaches

Key Features:
    - Pure streaming: O(1) memory per sample
    - HD document boundaries for separation
    - Progressive packing for memory efficiency
    - Automatic mode selection based on context_window

Memory Comparison (10K samples, 2M context):
    - Old padding approach: 80 GB
    - New streaming packer: 8 MB per batch
"""

from __future__ import annotations

import logging
from typing import Any

import tensorflow as tf

from highnoon import config
from highnoon.data.sequence_packer import (
    DEFAULT_NUM_WORKERS,
    MAX_CONTEXT_WINDOW,
    MIN_CONTEXT_WINDOW,
    StreamingSequencePacker,
)
from highnoon.data.tfrecord_cache import TFRecordCache
from highnoon.tokenization import QWTTextTokenizer, SuperwordMerger

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
    batch_size: int = 1,
    context_window: int = MIN_CONTEXT_WINDOW,
    vocab_size: int | None = None,
    tokenizer: QWTTextTokenizer | None = None,
    streaming_mode: str = "packed",
    hf_dataset_name: str | None = None,
    hf_dataset_names: list[str] | None = None,  # NEW: Multiple datasets for curriculum
    dataset_weights: list[float] | None = None,  # NEW: Sampling weights per dataset
    text_column: str | None = None,
    split: str = "train",
    max_samples: int | None = None,
    prefetch_buffer_size: int | None = None,
    use_adaptive_tokenizer: bool = True,
    adaptive_min_freq: int = 10,
    # New performance parameters
    use_tfrecord_cache: bool = False,
    tfrecord_cache_dir: str | None = None,
    num_tokenizer_workers: int = DEFAULT_NUM_WORKERS,
    # Deprecated parameters (kept for migration period)
    sequence_length: int | None = None,
    curriculum_id: str | None = None,
    dataset_name: str | None = None,
    shuffle: bool = True,  # Ignored in streaming mode
    shuffle_buffer: int = 1000,  # Ignored in streaming mode
    use_dummy_embeddings: bool = False,  # Ignored
    enable_superwords: bool | None = None,  # Handled by tokenizer
    superword_min_frequency: int | None = None,
    superword_max_vocab: int | None = None,
) -> tuple[tf.data.Dataset, QWTTextTokenizer, SuperwordMerger | None]:
    """Load a training dataset with enterprise streaming support.

    This function provides memory-efficient data loading for large context
    windows (100K-5M tokens) using the StreamingSequencePacker.

    Streaming Modes:
        - "packed": Pure streaming with HD document boundaries (default)
        - "chunked": Chunked streaming with adaptive tokenization

    Args:
        batch_size: Batch size for training (default 1 for large contexts).
        context_window: Target context window size in tokens (min 100K).
        vocab_size: Vocabulary size for tokenizer.
        tokenizer: Pre-configured tokenizer instance.
        streaming_mode: "packed" or "chunked".
        hf_dataset_name: HuggingFace dataset identifier.
        text_column: Column name for text (auto-detected if None).
        split: Dataset split to use.
        max_samples: Maximum samples to process.
        prefetch_buffer_size: Prefetch buffer (None for AUTOTUNE).
        use_adaptive_tokenizer: Use AdaptiveQWTTokenizer.
        adaptive_min_freq: Minimum frequency for adaptive n-grams.

        Deprecated (for backward compatibility):
        sequence_length: Use context_window instead.
        curriculum_id: Ignored.
        dataset_name: Use hf_dataset_name instead.
        shuffle: Ignored (streaming is sequential).
        shuffle_buffer: Ignored.
        use_dummy_embeddings: Ignored.
        enable_superwords: Handled by tokenizer.
        superword_min_frequency: Ignored.
        superword_max_vocab: Ignored.

    Returns:
        Tuple of (dataset, tokenizer, merger) where:
        - dataset yields (input_ids, labels)
        - tokenizer is the configured tokenizer instance
        - merger is None (deprecated, kept for API compatibility)

    Raises:
        LimitExceededError: If context_window exceeds Lite edition limit.
        ValueError: If streaming_mode is invalid.
    """
    # Handle deprecated parameter
    if sequence_length is not None:
        logger.warning("[Data Loaders] 'sequence_length' is deprecated, use 'context_window'")
        context_window = sequence_length

    # Handle deprecated dataset_name
    if hf_dataset_name is None and dataset_name is not None:
        hf_dataset_name = dataset_name

    # =========================================================================
    # LITE EDITION LIMIT VALIDATION
    # =========================================================================
    from highnoon._native._limits import MAX_CONTEXT_LENGTH, LimitExceededError, is_lite

    if is_lite() and context_window > MAX_CONTEXT_LENGTH:
        raise LimitExceededError(
            f"context_window ({context_window:,}) exceeds Lite edition limit "
            f"({MAX_CONTEXT_LENGTH:,} / 5M tokens).\n\n"
            "Upgrade to Pro or Enterprise for unlimited context:\n"
            "  https://versoindustries.com/upgrade",
            violations=[f"context_window: {context_window:,} > {MAX_CONTEXT_LENGTH:,}"],
        )

    # Enforce minimum context window
    if context_window < MIN_CONTEXT_WINDOW:
        logger.warning(
            "[Data Loaders] context_window %d < minimum %d, using minimum",
            context_window,
            MIN_CONTEXT_WINDOW,
        )
        context_window = MIN_CONTEXT_WINDOW

    # Validate streaming mode
    if streaming_mode not in ("packed", "chunked"):
        raise ValueError(f"Invalid streaming_mode: {streaming_mode}. Use 'packed' or 'chunked'.")

    # =========================================================================
    # TOKENIZER SETUP
    # =========================================================================
    if vocab_size is None:
        vocab_size = config.VOCAB_SIZE
        logger.info("[Data Loaders] Using config.VOCAB_SIZE=%d", vocab_size)

    if tokenizer is None:
        if use_adaptive_tokenizer:
            from highnoon.tokenization import AdaptiveQWTTokenizer

            tokenizer = AdaptiveQWTTokenizer(
                vocab_size=vocab_size,
                max_vocab_size=vocab_size,
                model_max_length=context_window,
                min_ngram_size=2,
                max_ngram_size=5,
            )
            logger.info(
                "[Data Loaders] Using AdaptiveQWTTokenizer: target_vocab=%d, context=%d",
                vocab_size,
                context_window,
            )
        else:
            tokenizer = create_tokenizer(vocab_size=vocab_size, max_length=context_window)
            logger.info(
                "[Data Loaders] Using QWTTextTokenizer: vocab=%d, context=%d",
                tokenizer.vocab_size,
                context_window,
            )
    else:
        logger.info("[Data Loaders] Using provided tokenizer: vocab_size=%d", tokenizer.vocab_size)

    # =========================================================================
    # STREAMING MODE: PACKED (Enterprise)
    # =========================================================================
    if streaming_mode == "packed":
        logger.info(
            "[Data Loaders] Using PACKED streaming: context=%d, batch=%d",
            context_window,
            batch_size,
        )

        packer = StreamingSequencePacker(
            target_length=context_window,
            max_length=MAX_CONTEXT_WINDOW,
            tokenizer=tokenizer,
            use_hd_boundaries=True,
        )

        if hf_dataset_names and len(hf_dataset_names) > 1:
            # Multi-dataset curriculum mixing with interleaving
            logger.info(
                "[Data Loaders] Multi-dataset curriculum: interleaving %d datasets",
                len(hf_dataset_names),
            )

            num_workers = num_tokenizer_workers
            datasets_list = []
            valid_names = []

            for ds_name in hf_dataset_names:
                try:
                    ds = packer.create_tf_dataset(
                        hf_dataset_name=ds_name,
                        batch_size=batch_size,
                        split=split,
                        text_column=text_column,
                        max_samples=max_samples,
                        prefetch=None,  # Prefetch after interleaving
                        use_parallel=True,
                        num_workers=num_workers,
                    )
                    datasets_list.append(ds)
                    valid_names.append(ds_name)
                    logger.info("[Data Loaders] Added dataset: %s", ds_name)
                except Exception as e:
                    logger.warning("[Data Loaders] Failed to load dataset %s: %s", ds_name, e)
                    continue

            if not datasets_list:
                raise ValueError(f"No valid datasets found in hf_dataset_names: {hf_dataset_names}")

            # Compute weights (equal if not specified, or normalize provided weights)
            if dataset_weights and len(dataset_weights) >= len(datasets_list):
                # Use provided weights (normalize to sum to 1)
                weights = [w for w, _ in zip(dataset_weights, valid_names)]
                total = sum(weights)
                weights = [w / total for w in weights]
            else:
                # Equal weights
                weights = [1.0 / len(datasets_list)] * len(datasets_list)

            logger.info(
                "[Data Loaders] Interleaving %d datasets with weights: %s",
                len(datasets_list),
                {n: f"{w:.3f}" for n, w in zip(valid_names, weights)},
            )

            # Interleave datasets using sample_from_datasets
            dataset = tf.data.Dataset.sample_from_datasets(
                datasets_list,
                weights=weights,
                seed=42,
                stop_on_empty_dataset=False,  # Continue even if one dataset exhausts
            )

            if prefetch_buffer_size is not None:
                dataset = dataset.prefetch(prefetch_buffer_size)
            else:
                dataset = dataset.prefetch(tf.data.AUTOTUNE)

        elif hf_dataset_name or (hf_dataset_names and len(hf_dataset_names) == 1):
            # Single dataset mode (backward compatible)
            single_name = hf_dataset_name or hf_dataset_names[0]
            # Use parallel tokenization parameters
            num_workers = num_tokenizer_workers

            if use_tfrecord_cache:
                # Use TFRecord caching for maximum throughput
                logger.info("[Data Loaders] Using TFRecord cache for maximum throughput")
                cache = TFRecordCache(
                    cache_dir=tfrecord_cache_dir,
                    num_shards=8,
                    auto_cleanup=True,  # Delete cache on exit per user request
                )
                dataset = cache.create_cached_dataset(
                    packer=packer,
                    hf_dataset_name=single_name,
                    batch_size=batch_size,
                    split=split,
                    text_column=text_column,
                    max_samples=max_samples,
                    use_parallel=True,
                    num_workers=num_workers,
                )
            else:
                # Stream directly with parallel tokenization
                dataset = packer.create_tf_dataset(
                    hf_dataset_name=single_name,
                    batch_size=batch_size,
                    split=split,
                    text_column=text_column,
                    max_samples=max_samples,
                    prefetch=prefetch_buffer_size,
                    use_parallel=True,
                    num_workers=num_workers,
                )
        else:
            # Use sample texts for testing
            logger.info("[Data Loaders] No dataset specified, using sample texts")

            def sample_text_iterator():
                # Repeat sample texts to fill context window
                while True:
                    yield from _SAMPLE_TEXTS

            # Create dataset from text iterator
            def batch_generator():
                for batch in packer.stream_from_texts(sample_text_iterator(), batch_size):
                    yield (batch.input_ids, batch.labels)

            dataset = tf.data.Dataset.from_generator(
                batch_generator,
                output_signature=(
                    tf.TensorSpec(shape=(batch_size, context_window), dtype=tf.int32),
                    tf.TensorSpec(shape=(batch_size, context_window), dtype=tf.int32),
                ),
            )

            if prefetch_buffer_size is not None:
                dataset = dataset.prefetch(prefetch_buffer_size)
            else:
                dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset, tokenizer, None  # merger is None in streaming mode

    # =========================================================================
    # STREAMING MODE: CHUNKED (Adaptive tokenization)
    # =========================================================================
    elif streaming_mode == "chunked":
        logger.info(
            "[Data Loaders] Using CHUNKED streaming: context=%d, batch=%d",
            context_window,
            batch_size,
        )

        # Train adaptive tokenizer if needed
        from highnoon.tokenization import AdaptiveQWTTokenizer

        if isinstance(tokenizer, AdaptiveQWTTokenizer) and not tokenizer.is_trained:
            if hf_dataset_name:
                # Learn from first N samples
                logger.info("[Data Loaders] Training AdaptiveQWTTokenizer from corpus...")
                sample_texts = []
                from datasets import load_dataset as hf_load_dataset

                hf_dataset = hf_load_dataset(
                    hf_dataset_name,
                    split=split,
                    streaming=True,
                    trust_remote_code=True,
                )
                for _i, sample in enumerate(hf_dataset):
                    for col in ["text", "context", "instruction", "input", "content"]:
                        if col in sample and isinstance(sample[col], str):
                            sample_texts.append(sample[col])
                            break
                    if len(sample_texts) >= 5000:
                        break

                learned = tokenizer.learn_from_corpus(sample_texts, min_freq=adaptive_min_freq)
                logger.info(
                    "[Data Loaders] Learned %d n-grams, vocab=%d", learned, tokenizer.vocab_size
                )
            else:
                tokenizer.learn_from_corpus(list(_SAMPLE_TEXTS), min_freq=2)

        # Create chunked streaming dataset
        packer = StreamingSequencePacker(
            target_length=context_window,
            max_length=MAX_CONTEXT_WINDOW,
            tokenizer=tokenizer,
            use_hd_boundaries=True,
        )

        if hf_dataset_name:
            dataset = packer.create_tf_dataset(
                hf_dataset_name=hf_dataset_name,
                batch_size=batch_size,
                split=split,
                text_column=text_column,
                max_samples=max_samples,
                prefetch=prefetch_buffer_size,
            )
        else:
            logger.info("[Data Loaders] No dataset specified, using sample texts")

            def sample_batch_gen():
                for batch in packer.stream_from_texts(iter(_SAMPLE_TEXTS * 100), batch_size):
                    yield (batch.input_ids, batch.labels)

            dataset = tf.data.Dataset.from_generator(
                sample_batch_gen,
                output_signature=(
                    tf.TensorSpec(shape=(batch_size, context_window), dtype=tf.int32),
                    tf.TensorSpec(shape=(batch_size, context_window), dtype=tf.int32),
                ),
            ).prefetch(tf.data.AUTOTUNE)

        return dataset, tokenizer, None

    # Should not reach here due to validation above
    raise ValueError(f"Unhandled streaming_mode: {streaming_mode}")


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
    "get_curriculum_datasets",
    "MIN_CONTEXT_WINDOW",
    "MAX_CONTEXT_WINDOW",
]
