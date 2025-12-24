# src/models/reasoning/fused_contract.py
# Copyright 2025 Verso Industries
#
# Common mixins/utilities for exporting reasoning block metadata to the fused
# C++ operator. The mixin avoids repeating boilerplate across every block while
# keeping the per-block metadata explicit.

from __future__ import annotations

from typing import Any

import tensorflow as tf


class FusedReasoningBlockMixin:
    """Provides canonical export hooks for fused_reasoning_stack."""

    fused_block_type: str = "GenericReasoningBlock"
    fused_block_stateful: bool = False
    fused_block_compatible: bool = True  # NEW: Can this block use the C++ fused kernel?

    def get_weights_for_fused_op(self) -> list[tf.Tensor]:
        """Return weights in the order expected by the fused kernel."""
        return list(getattr(self, "weights", []))

    def fused_metadata(self) -> dict[str, Any]:
        """Override to add block-specific metadata fields."""
        return {}

    def get_fused_op_descriptor(self) -> dict[str, Any]:
        """Return the descriptor consumed by block_factory + fused kernel."""
        weights = self.get_weights_for_fused_op()
        return {
            "type": self.fused_block_type,
            "stateful": bool(self.fused_block_stateful),
            "weight_count": len(weights),
            "metadata": self.fused_metadata(),
        }


class TTBasedReasoningMixin(FusedReasoningBlockMixin):
    """
    Mixin for blocks that use Tensor-Train (TT) decomposition or other non-dense
    weight structures that are incompatible with the dense C++ kernel.

    These blocks will be executed in Python using TensorFlow ops instead of through
    the fused C++ reasoning stack.
    """

    fused_block_compatible: bool = False
