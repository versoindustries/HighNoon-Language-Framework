# highnoon/models/__init__.py
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

"""HighNoon Language Framework - Core Models.

This module provides the core neural network models for the HighNoon
Language Framework, including:

- HSMN: Hierarchical State-Space Model Network (full architecture)
- HSMNLanguageModel: Lightweight language model variant
- MoELayer: Mixture of Experts layer with superposition

Factory functions:
- create_model: Create a model from a preset name or configuration
"""

import logging
from typing import Optional, Union

log = logging.getLogger(__name__)


# Lazy imports
def __getattr__(name):
    if name == "HSMN":
        from highnoon.models.hsmn import HSMN

        return HSMN
    elif name == "HSMNLanguageModel":
        # Alias for backward compatibility - now points to HSMN
        from highnoon.models.hsmn import HSMN

        return HSMN
    elif name == "LanguageModel":
        from highnoon.models.hsmn import HSMN

        return HSMN
    elif name == "MoELayer":
        from highnoon.models.moe import MoELayer

        return MoELayer
    elif name == "SuperposedExpert":
        from highnoon.models.moe import SuperposedExpert

        return SuperposedExpert
    elif name == "HamiltonianNN":
        from highnoon.models.hamiltonian import HamiltonianNN

        return HamiltonianNN
    elif name == "TimeCrystalBlock":
        from highnoon.models.hamiltonian import TimeCrystalBlock

        return TimeCrystalBlock
    elif name == "TimeCrystalSequenceBlock":
        from highnoon.models.hamiltonian import TimeCrystalSequenceBlock

        return TimeCrystalSequenceBlock
    elif name == "SpatialBlock":
        from highnoon.models.spatial.mamba import SpatialBlock

        return SpatialBlock
    elif name == "ReasoningMamba2Block":
        from highnoon.models.spatial.mamba import ReasoningMamba2Block

        return ReasoningMamba2Block
    elif name == "FloquetTimeCrystalBlock":
        from highnoon.models.hamiltonian import FloquetTimeCrystalBlock

        return FloquetTimeCrystalBlock
    elif name == "FloquetTimeCrystalSequenceBlock":
        from highnoon.models.hamiltonian import FloquetTimeCrystalSequenceBlock

        return FloquetTimeCrystalSequenceBlock
    raise AttributeError(f"module 'highnoon.models' has no attribute {name!r}")


def create_model(name_or_config, **kwargs):
    """Create a HighNoon language model.

    This factory function creates a model from either a preset name
    or a configuration object.

    Args:
        name_or_config: Either a preset name (e.g., 'highnoon-3b') or
            a Config object with model configuration.
        **kwargs: Additional keyword arguments passed to the model
            constructor. These override values from the config.

    Returns:
        Instantiated HighNoon language model.

    Examples:
        # From preset
        model = create_model("highnoon-3b")

        # From config
        from highnoon.config import Config, ModelConfig
        config = Config(model=ModelConfig(embedding_dim=1024))
        model = create_model(config)

        # With overrides
        model = create_model("highnoon-base", num_reasoning_blocks=8)

    Raises:
        ValueError: If the preset name is not recognized.
        LimitExceededError: If the configuration exceeds Lite limits.
    """
    from highnoon.config import Config, get_preset_config
    from highnoon.models.hsmn import HSMN

    # Get configuration
    if isinstance(name_or_config, str):
        config = get_preset_config(name_or_config)
        log.info(f"Creating model from preset: {name_or_config}")
    elif isinstance(name_or_config, Config):
        config = name_or_config
        log.info("Creating model from provided configuration")
    else:
        raise TypeError(f"Expected str or Config, got {type(name_or_config).__name__}")

    # Extract model config and apply overrides
    model_config = config.model.to_dict()
    model_config.update(kwargs)

    # Create model
    model = HSMN(
        vocab_size=model_config.get("vocab_size", 32000),
        embedding_dim=model_config.get("embedding_dim", 768),
        max_seq_length=model_config.get("max_seq_length", 4096),
        num_reasoning_blocks=model_config.get("num_reasoning_blocks", 4),
        block_pattern=model_config.get("block_pattern", "mamba_timecrystal_wlam_moe_hybrid"),
    )

    log.info(
        f"Created model with {model_config.get('num_reasoning_blocks', 4)} "
        f"reasoning blocks, embedding_dim={model_config.get('embedding_dim', 768)}"
    )

    return model


__all__ = [
    "HSMN",
    "HSMNLanguageModel",  # Alias for backward compatibility
    "LanguageModel",
    "MoELayer",
    "SuperposedExpert",
    "HamiltonianNN",
    "TimeCrystalBlock",
    "TimeCrystalSequenceBlock",
    "FloquetTimeCrystalBlock",
    "FloquetTimeCrystalSequenceBlock",
    "SpatialBlock",
    "ReasoningMamba2Block",
    "create_model",
]
