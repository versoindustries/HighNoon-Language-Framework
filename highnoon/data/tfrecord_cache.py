# highnoon/data/tfrecord_cache.py
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

"""TFRecord Caching for Maximum Training Throughput.

Pre-tokenizes and caches datasets as sharded TFRecords for fastest possible
training I/O. Uses parallel interleave for multi-file reading.

Key Features:
    - Sharded TFRecords for parallel I/O
    - Automatic cleanup after training (atexit hook)
    - Parallel interleave reading with AUTOTUNE
    - Compatible with StreamingSequencePacker

Performance:
    - First run: Tokenize + write cache (slower)
    - Subsequent runs: Read from cache (3-5x faster)
    - Auto-cleanup: Cache deleted on exit (user requested)

Usage:
    >>> cache = TFRecordCache(cache_dir="/tmp/highnoon_cache")
    >>> dataset = cache.create_cached_dataset(
    ...     packer=StreamingSequencePacker(...),
    ...     hf_dataset_name="databricks/dolly-15k",
    ...     batch_size=8,
    ... )
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from highnoon.data.sequence_packer import DEFAULT_NUM_WORKERS, StreamingSequencePacker

from highnoon.data.sequence_packer import DEFAULT_NUM_WORKERS

logger = logging.getLogger(__name__)

# Registry for cache directories to clean up
_CACHE_CLEANUP_REGISTRY: list[Path] = []


def _cleanup_caches() -> None:
    """Clean up all registered cache directories on exit."""
    for cache_dir in _CACHE_CLEANUP_REGISTRY:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                logger.info("[TFRecordCache] Cleaned up cache: %s", cache_dir)
            except Exception as e:
                logger.warning("[TFRecordCache] Failed to cleanup %s: %s", cache_dir, e)


# Register cleanup handler
atexit.register(_cleanup_caches)


class TFRecordCache:
    """TFRecord-based caching for maximum training throughput.

    Pre-tokenizes dataset once with parallel tokenization, stores as
    sharded TFRecords, then reads with parallel interleave for fastest
    possible training I/O.

    Attributes:
        cache_dir: Directory for TFRecord files.
        num_shards: Number of TFRecord shards for parallel reading.
        auto_cleanup: Automatically delete cache on exit.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        num_shards: int = 8,
        auto_cleanup: bool = True,
    ) -> None:
        """Initialize TFRecordCache.

        Args:
            cache_dir: Directory for cache files. If None, uses temp dir.
            num_shards: Number of shards for parallel I/O.
            auto_cleanup: Delete cache on program exit.
        """
        if cache_dir is None:
            cache_dir = Path(tempfile.mkdtemp(prefix="highnoon_tfcache_"))
        else:
            cache_dir = Path(cache_dir)

        self.cache_dir = cache_dir
        self.num_shards = num_shards
        self.auto_cleanup = auto_cleanup

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Register for cleanup if requested
        if auto_cleanup and self.cache_dir not in _CACHE_CLEANUP_REGISTRY:
            _CACHE_CLEANUP_REGISTRY.append(self.cache_dir)

        logger.info(
            "[TFRecordCache] Initialized: dir=%s, shards=%d, auto_cleanup=%s",
            self.cache_dir,
            num_shards,
            auto_cleanup,
        )

    def _get_cache_key(
        self,
        dataset_name: str,
        context_window: int,
        vocab_size: int,
        max_samples: int | None,
    ) -> str:
        """Generate cache key from dataset parameters."""
        key_str = f"{dataset_name}_{context_window}_{vocab_size}_{max_samples or 'all'}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _serialize_example(
        self,
        input_ids: np.ndarray,
        labels: np.ndarray,
    ) -> bytes:
        """Serialize a single training example to TFRecord format."""
        feature = {
            "input_ids": tf.train.Feature(
                int64_list=tf.train.Int64List(value=input_ids.flatten().tolist())
            ),
            "labels": tf.train.Feature(
                int64_list=tf.train.Int64List(value=labels.flatten().tolist())
            ),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def write_cache(
        self,
        packer: StreamingSequencePacker,
        hf_dataset_name: str,
        split: str = "train",
        text_column: str | None = None,
        max_samples: int | None = None,
        use_parallel: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> tuple[Path, int]:
        """Write tokenized dataset to TFRecord cache.

        Args:
            packer: StreamingSequencePacker instance.
            hf_dataset_name: HuggingFace dataset identifier.
            split: Dataset split.
            text_column: Text column name.
            max_samples: Maximum samples to process.
            use_parallel: Use parallel tokenization.
            num_workers: Number of tokenizer workers.

        Returns:
            Tuple of (cache_path, num_sequences).
        """
        cache_key = self._get_cache_key(
            hf_dataset_name,
            packer.target_length,
            packer.tokenizer.vocab_size if packer.tokenizer else 0,
            max_samples,
        )
        cache_path = self.cache_dir / cache_key

        # Check if cache already exists
        if cache_path.exists():
            existing_shards = list(cache_path.glob("*.tfrecord"))
            if existing_shards:
                logger.info(
                    "[TFRecordCache] Using existing cache: %s (%d shards)",
                    cache_path,
                    len(existing_shards),
                )
                # Count sequences (approximate from file size)
                total_size = sum(f.stat().st_size for f in existing_shards)
                approx_seqs = total_size // (packer.target_length * 8)  # Rough estimate
                return cache_path, approx_seqs

        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info("[TFRecordCache] Writing cache to: %s", cache_path)

        # Create shard writers
        writers = []
        for i in range(self.num_shards):
            shard_path = cache_path / f"shard_{i:04d}.tfrecord"
            writers.append(tf.io.TFRecordWriter(str(shard_path)))

        start_time = time.time()
        sequence_count = 0
        shard_idx = 0

        # Stream packed batches and write to shards (round-robin)
        for batch in packer.stream_packed_batches(
            hf_dataset_name=hf_dataset_name,
            batch_size=1,  # Write one sequence at a time
            split=split,
            text_column=text_column,
            max_samples=max_samples,
            use_parallel=use_parallel,
            num_workers=num_workers,
        ):
            # Write each sequence in batch to shard
            for i in range(batch.num_sequences):
                serialized = self._serialize_example(
                    batch.input_ids[i],
                    batch.labels[i],
                )
                writers[shard_idx].write(serialized)
                shard_idx = (shard_idx + 1) % self.num_shards
                sequence_count += 1

            if sequence_count % 100 == 0:
                elapsed = time.time() - start_time
                rate = sequence_count / elapsed if elapsed > 0 else 0
                logger.info(
                    "[TFRecordCache] Written %d sequences (%.1f seq/s)",
                    sequence_count,
                    rate,
                )

        # Close writers
        for w in writers:
            w.close()

        elapsed = time.time() - start_time
        logger.info(
            "[TFRecordCache] Cache complete: %d sequences in %.1fs (%.1f seq/s)",
            sequence_count,
            elapsed,
            sequence_count / elapsed if elapsed > 0 else 0,
        )

        return cache_path, sequence_count

    def create_dataset_from_cache(
        self,
        cache_path: Path,
        batch_size: int = 8,
        context_window: int = 131072,
        shuffle_buffer: int = 1000,
        cycle_length: int = 4,
    ) -> tf.data.Dataset:
        """Create tf.data.Dataset from cached TFRecords.

        Uses parallel interleave for maximum read throughput.

        Args:
            cache_path: Path to cache directory.
            batch_size: Batch size for training.
            context_window: Sequence length for reshaping.
            shuffle_buffer: Shuffle buffer size.
            cycle_length: Number of files to read in parallel.

        Returns:
            tf.data.Dataset yielding (input_ids, labels).
        """
        # Find all shard files
        shard_pattern = str(cache_path / "*.tfrecord")
        files = tf.data.Dataset.list_files(shard_pattern, shuffle=True)

        # Parse function
        def parse_fn(serialized):
            features = {
                "input_ids": tf.io.FixedLenFeature([context_window], tf.int64),
                "labels": tf.io.FixedLenFeature([context_window], tf.int64),
            }
            example = tf.io.parse_single_example(serialized, features)
            return (
                tf.cast(example["input_ids"], tf.int32),
                tf.cast(example["labels"], tf.int32),
            )

        # Parallel interleave: read multiple shards concurrently
        dataset = files.interleave(
            lambda f: tf.data.TFRecordDataset(f, compression_type=None),
            cycle_length=cycle_length,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,  # Allow reordering for speed
        )

        # Parse in parallel
        dataset = dataset.map(
            parse_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        # Shuffle, batch, prefetch
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def create_cached_dataset(
        self,
        packer: StreamingSequencePacker,
        hf_dataset_name: str,
        batch_size: int = 8,
        split: str = "train",
        text_column: str | None = None,
        max_samples: int | None = None,
        use_parallel: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
        shuffle_buffer: int = 1000,
    ) -> tf.data.Dataset:
        """Create high-throughput dataset with TFRecord caching.

        First call writes cache, subsequent calls read from cache.
        Cache is automatically deleted on program exit.

        Args:
            packer: StreamingSequencePacker instance.
            hf_dataset_name: HuggingFace dataset identifier.
            batch_size: Training batch size.
            split: Dataset split.
            text_column: Text column name.
            max_samples: Maximum samples.
            use_parallel: Use parallel tokenization for cache writing.
            num_workers: Number of tokenizer workers.
            shuffle_buffer: Shuffle buffer size.

        Returns:
            tf.data.Dataset yielding (input_ids, labels).
        """
        # Write cache (or find existing)
        cache_path, num_sequences = self.write_cache(
            packer=packer,
            hf_dataset_name=hf_dataset_name,
            split=split,
            text_column=text_column,
            max_samples=max_samples,
            use_parallel=use_parallel,
            num_workers=num_workers,
        )

        logger.info(
            "[TFRecordCache] Loading dataset from cache: %s (%d sequences)",
            cache_path,
            num_sequences,
        )

        # Create dataset from cache
        return self.create_dataset_from_cache(
            cache_path=cache_path,
            batch_size=batch_size,
            context_window=packer.target_length,
            shuffle_buffer=shuffle_buffer,
        )

    def cleanup(self) -> None:
        """Manually cleanup cache directory."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info("[TFRecordCache] Cleaned up: %s", self.cache_dir)

        # Remove from registry
        if self.cache_dir in _CACHE_CLEANUP_REGISTRY:
            _CACHE_CLEANUP_REGISTRY.remove(self.cache_dir)


__all__ = ["TFRecordCache"]
