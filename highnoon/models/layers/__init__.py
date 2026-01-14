# highnoon/models/layers/__init__.py
# Copyright 2025 Verso Industries
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Model layer components for HighNoon Language Framework."""

from typing import TYPE_CHECKING

from highnoon.models.layers.adapter import AdapterLayer
from highnoon.models.layers.chunker import StreamingChunker

# [DEPRECATED] ContextualGatingCollapse removed - replaced by holographic routing in SuperposedExpert
from highnoon.models.layers.dense import LowRankDense

# WLAM requires tensorflow_models and fused_contract - import conditionally
_WLAM_AVAILABLE = False
WLAMBlock: type | None = None
WLAM: type | None = None

try:
    from highnoon.models.layers.wlam import WLAMBlock as _WLAMBlock

    WLAMBlock = _WLAMBlock
    WLAM = _WLAMBlock  # Alias for backward compatibility
    _WLAM_AVAILABLE = True
except ImportError:
    pass

# Phase 13.6: Local Attention - import conditionally
LocalAttentionBlock: type | None = None

try:
    from highnoon.models.layers.local_attention import LocalAttentionBlock
except ImportError:
    pass

# Phase 13.7: Flash Linear Attention - import conditionally
FlashLinearAttention: type | None = None

try:
    from highnoon.models.layers.flash_linear_attention import FlashLinearAttention
except ImportError:
    pass

# Phase 15.4+: Quantum GQA with Tensor Decomposition (Primary Attention)
# Combines quantum kernel attention with TPA for O(n) + 10x+ KV cache reduction
QuantumGQA: type | None = None

try:
    from highnoon.models.layers.quantum_gqa import QuantumGQA
except ImportError:
    pass

# Phase 14.1.1: Data-Dependent Token Shift - import conditionally
DataDependentTokenShift: type | None = None
TemporalTokenMixer: type | None = None
create_token_shift_layer = None

try:
    from highnoon.models.layers.token_shift import (
        DataDependentTokenShift,
        TemporalTokenMixer,
        create_token_shift_layer,
    )
except ImportError:
    pass

# Phase 18.1: Latent KV Attention - import conditionally
LatentKVAttention: type | None = None

try:
    from highnoon.models.layers.latent_kv_attention import LatentKVAttention
except ImportError:
    pass

# Phase 48: Hyperdimensional Quantum Embeddings - import conditionally
HyperdimensionalEmbedding: type | None = None
DualPathEmbedding: type | None = None

try:
    from highnoon.models.layers.hyperdimensional_layer import (
        DualPathEmbedding,
        HyperdimensionalEmbedding,
    )
except ImportError:
    pass

# Phase 2 Memory Roadmap: Holographic Token Bundling - import conditionally
HolographicBundle: type | None = None
HolographicUnbundle: type | None = None

try:
    from highnoon.models.layers.holographic_bundle import HolographicBundle, HolographicUnbundle
except ImportError:
    pass

__all__ = [
    # [DEPRECATED] "ContextualGatingCollapse" - removed, replaced by holographic routing
    "AdapterLayer",
    "StreamingChunker",
    "LowRankDense",
    "WLAMBlock",
    "WLAM",
    # Phase 13.6: Local Attention
    "LocalAttentionBlock",
    # Phase 13.7: Flash Linear Attention
    "FlashLinearAttention",
    # Phase 15.4+: Quantum GQA (Primary - with Tensor Decomposition)
    "QuantumGQA",
    # Phase 14.1.1: Token Shift
    "DataDependentTokenShift",
    "TemporalTokenMixer",
    "create_token_shift_layer",
    # Phase 18: Frontier Memory Innovations
    "LatentKVAttention",
    # Phase 48: Hyperdimensional Quantum Embeddings
    "HyperdimensionalEmbedding",
    "DualPathEmbedding",
    # Phase 2 Memory Roadmap: Holographic Token Bundling
    "HolographicBundle",
    "HolographicUnbundle",
]
