"""Shared diagnostics for verifying the fused tokenizer runtime during launches."""

from __future__ import annotations

import logging
from typing import Any

from highnoon._native.ops import fused_qwt_tokenizer as qwt_ops
from highnoon.tokenization import QWTTextTokenizer


def log_qwt_runtime(
    logger: logging.Logger,
    tokenizer: QWTTextTokenizer | Any,
    *,
    context: str,
) -> None:
    """Emits a consistent log line confirming the fused tokenizer configuration.

    Args:
        logger: Logger to use for output.
        tokenizer: QWTTextTokenizer or compatible tokenizer instance.
        context: Context string for the log message (e.g., "TRAINING", "INFERENCE").
    """
    # Check if native ops are available
    native_available = False
    try:
        native_available = qwt_ops.fused_qwt_tokenizer_available()
    except (AttributeError, ImportError):
        pass

    if native_available:
        op_path = qwt_ops.fused_qwt_tokenizer_op_path() or "<unknown>"
        grad_status = "enabled" if qwt_ops.fused_qwt_tokenizer_grad_available() else "missing"
        logger.info(
            "[TOKENIZER][%s] fused_qwt_tokenizer loaded from %s (grad=%s).",
            context,
            op_path,
            grad_status,
        )
    else:
        logger.info(
            "[TOKENIZER][%s] Using Python tokenizer (native ops not available).",
            context,
        )

    # Log tokenizer configuration
    logger.info(
        "[TOKENIZER][%s] tokenizer=%s vocab_size=%d max_length=%d",
        context,
        tokenizer.__class__.__name__,
        getattr(tokenizer, "vocab_size", -1),
        getattr(tokenizer, "_max_length", getattr(tokenizer, "model_max_length", -1)),
    )


__all__ = ["log_qwt_runtime"]
