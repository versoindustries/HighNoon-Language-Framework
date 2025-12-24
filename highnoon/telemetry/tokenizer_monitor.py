# src/telemetry/tokenizer_monitor.py
"""Lightweight telemetry hook for the fused Quantum Wavelet Tokenizer.

This module captures live token/embedding samples from the training graph and
persists them to ``state/tokenizer_metrics.json`` so the web console can render
real tokenizer graphs. The capture logic is throttled (default: one snapshot
every eight seconds) to minimise overhead during training or HPO sweeps.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "off", "no"}


@dataclass(frozen=True)
class _MonitorConfig:
    enabled: bool = _env_flag("HSMN_TOKENIZER_MONITOR", "1")
    min_interval_sec: float = float(os.getenv("HSMN_TOKENIZER_MONITOR_INTERVAL_SEC", "8.0"))
    max_samples: int = max(1, int(os.getenv("HSMN_TOKENIZER_MONITOR_SAMPLES", "4")))
    max_chunks: int = max(1, int(os.getenv("HSMN_TOKENIZER_MONITOR_CHUNKS", "2")))
    histogram_bins: int = max(8, int(os.getenv("HSMN_TOKENIZER_MONITOR_BINS", "32")))
    histogram_range: float = float(os.getenv("HSMN_TOKENIZER_MONITOR_RANGE", "2.0"))
    telemetry_path: Path = Path("state/tokenizer_metrics.json")
    stale_after_sec: float = float(os.getenv("HSMN_TOKENIZER_MONITOR_STALE_SEC", "300"))


class TokenizerMonitor:
    """Singleton-style helper that records live tokenizer activity."""

    _cfg = _MonitorConfig()
    _tokenizer = None
    _lock = threading.Lock()
    _last_write = 0.0

    @classmethod
    def configure(cls, tokenizer: Any) -> None:
        """Registers the text-side tokenizer used for decoding sample chunks."""

        if tokenizer is None:
            return
        cls._tokenizer = tokenizer

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._cfg.enabled and cls._tokenizer is not None

    @classmethod
    def maybe_record(
        cls,
        *,
        chunk_tokens: tf.Tensor,
        qwt_embeddings: tf.Tensor,
        training: bool,
    ) -> None:
        """Schedules a telemetry snapshot via ``tf.py_function`` if enabled."""

        if not cls.is_enabled():
            return

        def _record(tokens: tf.Tensor, embeddings: tf.Tensor) -> tf.Tensor:
            cls._record_numpy(tokens.numpy(), embeddings.numpy(), training)
            return tf.constant(0, dtype=tf.int32)

        tf.py_function(
            _record,
            inp=[tf.convert_to_tensor(chunk_tokens), tf.convert_to_tensor(qwt_embeddings)],
            Tout=tf.int32,
        )

    @classmethod
    def _record_numpy(cls, tokens: np.ndarray, embeddings: np.ndarray, training: bool) -> None:
        now = time.time()
        with cls._lock:
            if now - cls._last_write < cls._cfg.min_interval_sec:
                return
            cls._last_write = now

        if tokens.size == 0 or embeddings.size == 0:
            return

        payload = cls._build_payload(tokens, embeddings, training)
        cls._write_payload(payload)

    @classmethod
    def _build_payload(
        cls, tokens: np.ndarray, embeddings: np.ndarray, training: bool
    ) -> dict[str, Any]:
        flat_embeddings = embeddings.reshape(-1)
        bin_range = cls._cfg.histogram_range
        hist, edges = np.histogram(
            flat_embeddings,
            bins=cls._cfg.histogram_bins,
            range=(-bin_range, bin_range),
        )

        labels = [
            f"{edges[i]:.2f}–{edges[i + 1]:.2f}" for i in range(min(len(hist), len(edges) - 1))
        ]

        samples: list[dict[str, Any]] = []
        max_batches = min(tokens.shape[0], cls._cfg.max_samples)
        for batch_idx in range(max_batches):
            max_chunks = min(tokens.shape[1], cls._cfg.max_chunks)
            for chunk_idx in range(max_chunks):
                token_ids = tokens[batch_idx, chunk_idx].tolist()
                decoded = cls._tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                if not decoded:
                    decoded = "<pad>"
                samples.append(
                    {
                        "batch": int(batch_idx),
                        "chunk": int(chunk_idx),
                        "text": decoded,
                        "token_count": int(len(token_ids)),
                    }
                )
                if len(samples) >= cls._cfg.max_samples:
                    break
            if len(samples) >= cls._cfg.max_samples:
                break

        generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "labels": labels,
            "values": hist.astype(int).tolist(),
            "type": "bar",
            "title": "QWT Embedding Distribution",
            "generated_at": generated_at,
            "source": "training" if training else "evaluation",
            "samples": samples,
            "meta": {
                "max_abs": float(np.max(np.abs(flat_embeddings))),
                "mean": float(np.mean(flat_embeddings)),
                "std": float(np.std(flat_embeddings)),
                "vocab_size": getattr(cls._tokenizer, "vocab_size", None),
                "pad_token_id": getattr(cls._tokenizer, "pad_token_id", None),
            },
        }
        return payload

    @classmethod
    def _write_payload(cls, payload: dict[str, Any]) -> None:
        path = cls._cfg.telemetry_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.replace(path)

    @classmethod
    def load_snapshot(cls, max_age: float | None = None) -> dict[str, Any] | None:
        """Returns the newest telemetry payload if it is not stale."""

        max_age = cls._cfg.stale_after_sec if max_age is None else max_age
        path = cls._cfg.telemetry_path
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

        if not payload:
            return None

        ts_raw = payload.get("generated_at")
        if ts_raw:
            try:
                timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                timestamp = None
            if timestamp is not None and max_age is not None:
                delta = datetime.now(timezone.utc) - timestamp
                if delta.total_seconds() > max_age:
                    return None
        return payload


__all__ = ["TokenizerMonitor"]
