# src/training/data_utils.py
import json
import logging
import os
import shutil
import time
import traceback
from collections.abc import Callable
from typing import Any

import tensorflow as tf
from datasets import load_dataset

from highnoon.config import get_tokenizer
from highnoon.data.create_tfrecords import (
    TFRECORD_SCHEMA_VERSION,
    build_huggingface_kwargs,
    create_tfrecords,
)

# --- Logger Setup ---
logger = logging.getLogger(__name__)

_HF_CACHE_ROOT = os.getenv("HSMN_HF_DATA_CACHE") or os.path.join("artifacts", "hf_cache")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_FORCE_HF_REFRESH = _env_flag("HSMN_FORCE_HF_REFRESH", False)

try:
    _PAD_TOKEN_ID: int = int(getattr(get_tokenizer(), "pad_token_id", 0) or 0)
except Exception:  # noqa: BLE001 - tokenizer access is best-effort at import time
    _PAD_TOKEN_ID = 0


def ensure_path(path: str) -> str:
    """Normalizes a path and creates the directory if it doesn't exist."""
    normalized_path = os.path.normpath(path)
    os.makedirs(normalized_path, exist_ok=True)
    return normalized_path


def _safe_dataset_dir(name: str) -> str:
    sanitized = name.replace("/", "__").replace("\\", "__")
    return sanitized or "dataset"


def ensure_huggingface_local_cache(
    dataset_config: dict[str, Any],
    *,
    splits: tuple[str, ...] = ("train", "validation"),
    cache_root: str | None = None,
    force_refresh: bool | None = None,
) -> dict[str, str]:
    """Materializes the requested Hugging Face splits to disk and returns their paths."""

    repo_id = (
        dataset_config.get("repo") or dataset_config.get("dataset_id") or dataset_config.get("name")
    )
    dataset_name = dataset_config.get("name") or repo_id
    if not repo_id or not dataset_name:
        return {}

    local_cache_root = ensure_path(cache_root or _HF_CACHE_ROOT)
    sanitized_name = _safe_dataset_dir(dataset_name)
    should_refresh = _FORCE_HF_REFRESH if force_refresh is None else force_refresh
    hf_kwargs = build_huggingface_kwargs(dataset_config)

    cached_paths: dict[str, str] = {}
    for split in splits:
        split = split or "train"
        target_dir = os.path.join(local_cache_root, sanitized_name, split)
        dataset_info = os.path.join(target_dir, "dataset_info.json")
        if os.path.exists(dataset_info) and not should_refresh:
            cached_paths[split] = target_dir
            continue
        if should_refresh and os.path.exists(target_dir):
            try:
                shutil.rmtree(target_dir)
            except OSError as exc:
                logger.warning(
                    "[DATA] Unable to refresh Hugging Face cache %s: %s", target_dir, exc
                )
                continue
        try:
            dataset = load_dataset(
                repo_id,
                name=dataset_config.get("config"),
                split=split,
                streaming=False,
                **hf_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[DATA] Unable to download Hugging Face split '%s' for '%s': %s",
                split,
                repo_id,
                exc,
            )
            continue
        ensure_path(target_dir)
        try:
            dataset.save_to_disk(target_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[DATA] Failed to persist Hugging Face split '%s' for '%s': %s",
                split,
                repo_id,
                exc,
            )
            continue
        cached_paths[split] = target_dir
        try:
            logger.info(
                "[DATA] Hugging Face split '%s' for '%s' saved to %s (%s rows).",
                split,
                dataset_name,
                target_dir,
                len(dataset),
            )
        except TypeError:
            logger.info(
                "[DATA] Hugging Face split '%s' for '%s' saved to %s.",
                split,
                dataset_name,
                target_dir,
            )

    return cached_paths


def find_tfrecord_shards(dataset_name: str, data_dir: str) -> list[str]:
    """
    Finds and returns a list of all TFRecord file paths for a given dataset.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'sciq').
        data_dir (str): The root directory where TFRecord datasets are stored.

    Returns:
        A sorted list of file paths to the TFRecord shards. Returns an empty
        list if the directory does not exist or contains no .tfrecord files.
    """
    dataset_path = os.path.join(data_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        logger.warning(
            f"TFRecord directory not found for dataset '{dataset_name}' at: {dataset_path}"
        )
        return []

    shards = [
        os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".tfrecord")
    ]
    shards.sort()

    if not shards:
        logger.warning(
            f"No .tfrecord shards found for dataset '{dataset_name}' in directory: {dataset_path}"
        )

    return shards


def delete_tfrecord_dataset(dataset_name: str, data_dir: str) -> bool:
    """
    Removes all TFRecord shards for a dataset within the provided data directory.

    Returns:
        bool: True if any data was removed, False otherwise.
    """
    dataset_path = os.path.join(data_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        return False
    try:
        shutil.rmtree(dataset_path)
    except OSError as exc:
        logger.warning("Failed to delete TFRecord directory %s: %s", dataset_path, exc)
        return False
    logger.info(
        "[DATA] Deleted TFRecord shards for '%s' in %s to reclaim disk space.",
        dataset_name,
        data_dir,
    )
    return True


# --- START: FIX ---
# Metadata helpers ------------------------------------------------------------
def _read_metadata(metadata_dir: str) -> dict | None:
    metadata_path = os.path.join(metadata_dir, "metadata.json")
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _legacy_shards_in_split(split_root: str, dataset_name: str) -> list[str]:
    if not os.path.isdir(split_root):
        return []
    prefix = f"{dataset_name}-"
    return [
        os.path.join(split_root, entry)
        for entry in os.listdir(split_root)
        if entry.startswith(prefix) and entry.endswith(".tfrecord")
    ]


def _archive_legacy_shards(split_root: str, dataset_name: str) -> tuple[int, str | None]:
    legacy_shards = _legacy_shards_in_split(split_root, dataset_name)
    if not legacy_shards:
        return 0, None
    archive_dir = os.path.join(split_root, "_legacy", dataset_name, str(int(time.time())))
    os.makedirs(archive_dir, exist_ok=True)
    for shard_path in legacy_shards:
        try:
            shutil.move(shard_path, os.path.join(archive_dir, os.path.basename(shard_path)))
        except OSError as exc:
            logger.warning("Failed to archive legacy shard %s: %s", shard_path, exc)
    return len(legacy_shards), archive_dir


def _max_sequence_length_in_shards(shards: list[str], probe_files: int = 3) -> int:
    try:
        import tensorflow as tf
    except ImportError:
        return 0

    max_len = 0
    for shard_path in shards[:probe_files]:
        try:
            for record in tf.io.tf_record_iterator(shard_path):
                example = tf.train.Example()
                example.ParseFromString(record)
                feature_map = example.features.feature
                input_ids = feature_map.get("input_ids")
                labels = feature_map.get("labels")
                candidate = 0
                if input_ids is not None:
                    candidate = max(candidate, len(input_ids.int64_list.value))
                if labels is not None:
                    candidate = max(candidate, len(labels.int64_list.value))
                max_len = max(max_len, candidate)
                break
        except Exception as exc:
            logger.debug("Failed to inspect shard %s: %s", shard_path, exc)
    return max_len


def _requires_regeneration(
    dataset_name: str,
    split_root: str,
    expected_meta: dict[str, object],
) -> tuple[bool, str]:
    shards = find_tfrecord_shards(dataset_name, split_root)
    dataset_dir = os.path.join(split_root, dataset_name)

    if not shards:
        legacy_candidates = _legacy_shards_in_split(split_root, dataset_name)
        if legacy_candidates:
            return True, "legacy shard layout detected"
        return True, "missing shards"

    metadata = _read_metadata(dataset_dir)
    if metadata is None:
        return True, "metadata missing"

    if metadata.get("schema_version") != TFRECORD_SCHEMA_VERSION:
        return (
            True,
            f"schema version mismatch ({metadata.get('schema_version')} != {TFRECORD_SCHEMA_VERSION})",
        )

    for key, expected_value in expected_meta.items():
        if metadata.get(key) != expected_value:
            return True, f"metadata mismatch for '{key}' ({metadata.get(key)} != {expected_value})"

    max_observed = _max_sequence_length_in_shards(shards)
    target_max = expected_meta.get("max_seq_len")
    if target_max and max_observed and max_observed > target_max:
        return True, f"sequence length {max_observed} exceeds limit {target_max}"

    return False, ""


# Re-implemented the missing prepare_tokenized_dataset function.
def prepare_tokenized_dataset(
    dataset_name: str,
    dataset_config: dict,
    preprocessor: Callable,
    tokenizer: object,
    max_seq_len: int,
    tokenized_data_dir: str,
    *,
    pad_to_max_length: bool = False,
    train_shards: int = 100,
    validation_shards: int = 10,
):
    """
    Checks if a tokenized TFRecord dataset exists, and creates it if not.
    This function acts as a wrapper around the create_tfrecords script.
    """
    record_format = dataset_config.get("record_format", "text")
    if record_format != "text":
        logger.info(
            "Skipping tokenized dataset preparation for '%s' (record_format=%s).",
            dataset_name,
            record_format,
        )
        return

    logger.info(f"Preparing tokenized dataset for '{dataset_name}'...")

    desired_meta = {
        "max_seq_len": int(max_seq_len),
        "pad_to_max_length": bool(pad_to_max_length),
    }

    splits = (
        ("train", train_shards, logger.info),
        ("validation", validation_shards, logger.warning),
    )

    for split, num_shards, warn_logger in splits:
        split_root = ensure_path(os.path.join(tokenized_data_dir, split))
        needs_regen, reason = _requires_regeneration(
            dataset_name, split_root, expected_meta=desired_meta
        )
        if needs_regen:
            logger.info(
                "Tokenized '%s' split for '%s' requires regeneration (reason: %s).",
                split,
                dataset_name,
                reason,
            )
            archived_count, archive_dir = _archive_legacy_shards(split_root, dataset_name)
            if archived_count > 0:
                logger.info(
                    "Archived %d legacy shard(s) for dataset '%s' split '%s' into %s.",
                    archived_count,
                    dataset_name,
                    split,
                    archive_dir,
                )
            try:
                split_config = dataset_config.copy()
                split_config["split"] = split
                create_tfrecords(
                    dataset_config=split_config,
                    num_shards=num_shards,
                    output_dir=tokenized_data_dir,
                    max_seq_len=max_seq_len,
                    pad_to_max_length=pad_to_max_length,
                )
                metadata_dir = os.path.join(split_root, dataset_name)
                generated_meta = _read_metadata(metadata_dir)
                if generated_meta:
                    logger.info(
                        "Generated tokenized '%s' split for '%s' (max_seq_len=%s, observed_input=%s, observed_target=%s).",
                        split,
                        dataset_name,
                        generated_meta.get("max_seq_len"),
                        generated_meta.get("max_observed_input_tokens"),
                        generated_meta.get("max_observed_target_tokens"),
                    )
            except Exception as exc:
                if split == "validation":
                    warn_logger(
                        "Could not create '%s' split for %s. This may be expected. Error: %s",
                        split,
                        dataset_name,
                        exc,
                    )
                else:
                    logger.error(f"Failed to create '{split}' split for {dataset_name}: {exc}")
                    traceback.print_exc()
        else:
            metadata_dir = os.path.join(split_root, dataset_name)
            metadata = _read_metadata(metadata_dir)
            if metadata:
                logger.info(
                    "Tokenized '%s' split for '%s' is up to date (max_seq_len=%s, observed_input=%s, observed_target=%s).",
                    split,
                    dataset_name,
                    metadata.get("max_seq_len"),
                    metadata.get("max_observed_input_tokens"),
                    metadata.get("max_observed_target_tokens"),
                )
            else:
                logger.info(
                    "Tokenized '%s' split for '%s' is up to date (max_seq_len=%d, pad_to_max_length=%s).",
                    split,
                    dataset_name,
                    max_seq_len,
                    pad_to_max_length,
                )


# --- END: FIX ---


def build_tfrecord_dataset(
    paths: list[str] | tf.data.Dataset,
    feature_description: dict[str, Any],
    parse_fn: Callable,
    *,
    shuffle: bool = False,
    repeat: bool = False,
    bucket_meta: dict | None = None,
    shuffle_buffer_size: int = 10000,
) -> tf.data.Dataset:
    """
    Builds an optimized tf.data.Dataset from TFRecord files.

    Args:
        paths: A list of file paths or a tf.data.Dataset of file paths.
        feature_description: A dictionary describing the features for parsing.
        parse_fn: A function to parse the tf.train.Example proto.
        shuffle: Whether to shuffle the dataset.
        repeat: Whether to repeat the dataset indefinitely.
        bucket_meta: Optional dictionary for bucketing.
        shuffle_buffer_size: The size of the shuffle buffer.

    Returns:
        An optimized tf.data.Dataset instance.
    """
    autotune = tf.data.AUTOTUNE

    if isinstance(paths, list):
        if not paths:
            raise ValueError("Cannot build a dataset from an empty list of paths.")
        shard_source = tf.data.Dataset.from_tensor_slices(paths)
    else:
        shard_source = paths

    dataset = shard_source.interleave(
        lambda path: tf.data.TFRecordDataset(path, num_parallel_reads=autotune),
        cycle_length=autotune,
        num_parallel_calls=autotune,
        deterministic=not shuffle,
    )

    pad_value = tf.constant(_PAD_TOKEN_ID, dtype=tf.int32)

    def _align_lengths(
        context_tokens: tf.Tensor, target_tokens: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Pads the shorter of (context, target) so both share an identical length."""

        context_tokens = tf.cast(context_tokens, tf.int32)
        target_tokens = tf.cast(target_tokens, tf.int32)

        context_len = tf.shape(context_tokens)[0]
        target_len = tf.shape(target_tokens)[0]
        max_len = tf.maximum(context_len, target_len)

        def _pad_to_length(tensor: tf.Tensor, tensor_len: tf.Tensor) -> tf.Tensor:
            pad_amount = max_len - tensor_len
            return tf.pad(tensor, [[0, pad_amount]], constant_values=pad_value)

        context_tokens = tf.cond(
            tf.equal(context_len, max_len),
            lambda: context_tokens,
            lambda: _pad_to_length(context_tokens, context_len),
        )
        target_tokens = tf.cond(
            tf.equal(target_len, max_len),
            lambda: target_tokens,
            lambda: _pad_to_length(target_tokens, target_len),
        )

        return context_tokens, target_tokens

    dataset = dataset.map(parse_fn, num_parallel_calls=autotune)
    dataset = dataset.map(_align_lengths, num_parallel_calls=autotune)

    if repeat:
        dataset = dataset.repeat()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    if bucket_meta:

        def element_length_func(context, *_):
            return tf.shape(context)[0]

        dataset = dataset.bucket_by_sequence_length(
            element_length_func=element_length_func,
            bucket_boundaries=bucket_meta["bucket_boundaries"],
            bucket_batch_sizes=bucket_meta["bucket_batch_sizes"],
            pad_to_bucket_boundary=False,
            drop_remainder=True,
        )

    return dataset.prefetch(autotune)
