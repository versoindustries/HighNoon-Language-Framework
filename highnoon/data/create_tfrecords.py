# highnoon/data/create_tfrecords.py
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

"""TFRecord creation utilities for HighNoon Language Framework.

This module provides functions for converting HuggingFace datasets to
TFRecord format for efficient training data pipelines.
"""

import json
import logging
import os
from typing import Any

import tensorflow as tf

logger = logging.getLogger(__name__)

# Schema version for TFRecord files - increment when format changes
TFRECORD_SCHEMA_VERSION = "1.0.0"


def build_huggingface_kwargs(dataset_config: dict[str, Any]) -> dict[str, Any]:
    """Build kwargs for HuggingFace load_dataset from config.

    Args:
        dataset_config: Dataset configuration dictionary with optional keys:
            - repo: HuggingFace repo ID (e.g., "openai/webgpt_comparisons")
            - name: Dataset name/config (e.g., "default")
            - split: Data split (e.g., "train")
            - streaming: Whether to stream data
            - trust_remote_code: Whether to trust remote code

    Returns:
        Dictionary of kwargs for datasets.load_dataset()
    """
    kwargs = {}

    # Primary identifier
    repo = dataset_config.get("repo") or dataset_config.get("dataset_id")
    if repo:
        kwargs["path"] = repo

    # Optional config name
    name = dataset_config.get("name")
    if name and name != repo:
        kwargs["name"] = name

    # Split specification
    split = dataset_config.get("split")
    if split:
        kwargs["split"] = split

    # Streaming mode
    if dataset_config.get("streaming"):
        kwargs["streaming"] = True

    # Trust remote code for custom datasets
    if dataset_config.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True

    # Cache directory
    cache_dir = dataset_config.get("cache_dir")
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    return kwargs


def create_tfrecords(
    dataset_config: dict[str, Any],
    output_dir: str,
    tokenizer: Any,
    max_length: int = 512,
    num_workers: int = 4,
    shard_size: int = 10000,
) -> dict[str, Any]:
    """Convert a HuggingFace dataset to TFRecord format.

    Args:
        dataset_config: Dataset configuration with repo, name, split, etc.
        output_dir: Directory to write TFRecord files.
        tokenizer: Tokenizer instance for encoding text.
        max_length: Maximum sequence length for tokenization.
        num_workers: Number of parallel workers for processing.
        shard_size: Number of examples per TFRecord shard.

    Returns:
        Dictionary with metadata about the created TFRecords:
            - num_examples: Total number of examples processed
            - num_shards: Number of TFRecord shards created
            - schema_version: TFRecord schema version
            - output_dir: Output directory path
    """
    from datasets import load_dataset

    logger.info(f"Creating TFRecords in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    hf_kwargs = build_huggingface_kwargs(dataset_config)
    logger.info(f"Loading dataset with kwargs: {hf_kwargs}")

    try:
        dataset = load_dataset(**hf_kwargs)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Get text column
    text_column = dataset_config.get("text_column", "text")

    # Process and write TFRecords
    num_examples = 0
    num_shards = 0
    current_shard_examples = []
    shard_writer = None

    def _write_shard(examples: list, shard_idx: int) -> None:
        """Write a shard of examples to TFRecord."""
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.tfrecord")
        with tf.io.TFRecordWriter(shard_path) as writer:
            for example in examples:
                feature = {
                    "input_ids": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=example["input_ids"])
                    ),
                    "attention_mask": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=example["attention_mask"])
                    ),
                }
                tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())
        logger.info(f"Wrote shard {shard_idx} with {len(examples)} examples")

    # Process dataset
    for item in dataset:
        text = item.get(text_column, "")
        if not text:
            continue

        # Tokenize
        encoded = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        current_shard_examples.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
        num_examples += 1

        # Write shard if full
        if len(current_shard_examples) >= shard_size:
            _write_shard(current_shard_examples, num_shards)
            num_shards += 1
            current_shard_examples = []

    # Write remaining examples
    if current_shard_examples:
        _write_shard(current_shard_examples, num_shards)
        num_shards += 1

    # Write metadata
    metadata = {
        "schema_version": TFRECORD_SCHEMA_VERSION,
        "num_examples": num_examples,
        "num_shards": num_shards,
        "max_length": max_length,
        "dataset_config": dataset_config,
    }

    metadata_path = os.path.join(output_dir, "dataset_info.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Created {num_shards} TFRecord shards with {num_examples} examples " f"in {output_dir}"
    )

    return metadata


def load_tfrecord_dataset(
    tfrecord_dir: str,
    batch_size: int = 32,
    shuffle_buffer: int = 10000,
    max_length: int | None = None,
) -> tf.data.Dataset:
    """Load a TFRecord dataset for training.

    Args:
        tfrecord_dir: Directory containing TFRecord shards.
        batch_size: Batch size for the dataset.
        shuffle_buffer: Size of shuffle buffer.
        max_length: Maximum sequence length (for padding).

    Returns:
        tf.data.Dataset ready for training.
    """
    # Find TFRecord files
    tfrecord_files = sorted(
        [os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith(".tfrecord")]
    )

    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")

    # Load metadata
    metadata_path = os.path.join(tfrecord_dir, "dataset_info.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        if max_length is None:
            max_length = metadata.get("max_length", 512)

    # Define feature description
    feature_description = {
        "input_ids": tf.io.FixedLenFeature([max_length], tf.int64),
        "attention_mask": tf.io.FixedLenFeature([max_length], tf.int64),
    }

    def _parse_example(serialized_example):
        """Parse a single TFRecord example."""
        example = tf.io.parse_single_example(serialized_example, feature_description)
        return {
            "input_ids": tf.cast(example["input_ids"], tf.int32),
            "attention_mask": tf.cast(example["attention_mask"], tf.int32),
        }

    # Create dataset
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=4)
    dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
