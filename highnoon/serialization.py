# highnoon/serialization.py
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

"""Model Serialization API for HighNoon.

Provides standardized save/load interface for trained models with full
configuration, tokenizer, and metadata preservation.

Usage:
    import highnoon as hn
    
    # After training
    hn.save_model(model, "my_model", config=training_config)
    
    # For inference
    model, config = hn.load_model("my_model")
    output = model(input_ids)

The HighNoon model format creates a directory structure:
    my_model/
    ├── model.keras          # Keras SavedModel weights
    ├── config.json          # Training/model configuration
    ├── tokenizer/           # Tokenizer files (if provided)
    │   ├── vocab.json
    │   └── merges.txt
    └── metadata.json        # Version, edition, attribution

This format ensures all components needed for inference are bundled
together and can be easily loaded without external dependencies.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import tensorflow as tf

import highnoon

logger = logging.getLogger(__name__)

# Format version for backward compatibility
FORMAT_VERSION = "1.0.0"


def save_model(
    model: tf.keras.Model,
    path: str | Path,
    config: Any | None = None,
    tokenizer: Any = None,
    include_optimizer: bool = False,
    overwrite: bool = False,
) -> Path:
    """Save model with full training configuration.
    
    Creates a standardized HighNoon model directory containing the model
    weights, training configuration, tokenizer (if provided), and metadata.
    
    Args:
        model: Trained tf.keras.Model to save.
        path: Directory path to save the model to.
        config: Training configuration (EnterpriseTrainingConfig, 
                HPOTrainingConfig, or dict). Saved to config.json.
        tokenizer: Optional tokenizer to save with the model.
        include_optimizer: Whether to include optimizer state for resuming.
        overwrite: Whether to overwrite existing model directory.
        
    Returns:
        Path to the saved model directory.
        
    Raises:
        FileExistsError: If path exists and overwrite=False.
        
    Example:
        >>> from highnoon import save_model
        >>> from highnoon.training.training_engine import EnterpriseTrainingConfig
        >>>
        >>> config = EnterpriseTrainingConfig()
        >>> save_model(model, "my_model", config=config, tokenizer=tokenizer)
        PosixPath('my_model')
    """
    path = Path(path)
    
    # Handle existing directory
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
            logger.info(f"[Serialization] Removed existing directory: {path}")
        else:
            raise FileExistsError(
                f"Model directory already exists: {path}. "
                "Use overwrite=True to replace."
            )
    
    path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save model weights
    model_path = path / "model.keras"
    model.save(model_path, include_optimizer=include_optimizer)
    logger.info(f"[Serialization] Saved model weights to {model_path}")
    
    # 2. Save configuration
    config_path = path / "config.json"
    config_dict = _serialize_config(config)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"[Serialization] Saved config to {config_path}")
    
    # 3. Save tokenizer if provided
    if tokenizer is not None:
        tokenizer_dir = path / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        _save_tokenizer(tokenizer, tokenizer_dir)
        logger.info(f"[Serialization] Saved tokenizer to {tokenizer_dir}")
    
    # 4. Save metadata
    metadata_path = path / "metadata.json"
    metadata = {
        "format_version": FORMAT_VERSION,
        "highnoon_version": highnoon.__version__,
        "edition": highnoon.__edition__,
        "author": highnoon.__author__,
        "saved_at": datetime.now().isoformat(),
        "model_name": model.name if hasattr(model, "name") else "unknown",
        "has_optimizer": include_optimizer,
        "has_tokenizer": tokenizer is not None,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"[Serialization] Saved metadata to {metadata_path}")
    
    logger.info(f"[Serialization] Model saved successfully to {path}")
    return path


def load_model(
    path: str | Path,
    compile_model: bool = False,
    custom_objects: dict[str, Any] | None = None,
) -> tuple[tf.keras.Model, dict[str, Any]]:
    """Load model and configuration from HighNoon format.
    
    Loads a model saved with save_model(), returning both the model
    and its configuration dictionary.
    
    Args:
        path: Path to the saved model directory.
        compile_model: Whether to compile the model after loading.
        custom_objects: Optional dictionary of custom layers/objects.
        
    Returns:
        Tuple of (model, config_dict) where config_dict contains:
        - "training_config": The saved training configuration
        - "metadata": Model metadata (version, edition, etc.)
        - "tokenizer_path": Path to tokenizer (if present)
        
    Raises:
        FileNotFoundError: If model directory doesn't exist.
        ValueError: If model format is incompatible.
        
    Example:
        >>> from highnoon import load_model
        >>>
        >>> model, config = load_model("my_model")
        >>> output = model(input_ids)
        >>> training_config = config["training_config"]
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model directory not found: {path}")
    
    # 1. Load metadata first for version checking
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"format_version": "0.0.0"}
    
    # Version compatibility check
    saved_version = metadata.get("format_version", "0.0.0")
    if not _is_compatible_version(saved_version):
        logger.warning(
            f"[Serialization] Model format version {saved_version} may not be "
            f"fully compatible with current version {FORMAT_VERSION}"
        )
    
    # 2. Import HighNoon custom layers for deserialization
    # This ensures all custom Keras layers are registered
    highnoon_custom_objects = _get_highnoon_custom_objects()
    
    # Merge with user-provided custom_objects
    all_custom_objects = {**highnoon_custom_objects}
    if custom_objects:
        all_custom_objects.update(custom_objects)
    
    # 3. Load model
    model_path = path / "model.keras"
    if not model_path.exists():
        # Try legacy format
        model_path = path / "model"
    
    model = tf.keras.models.load_model(
        model_path,
        compile=compile_model,
        custom_objects=all_custom_objects,
    )
    logger.info(f"[Serialization] Loaded model from {model_path}")
    
    # 4. Load configuration
    config_path = path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            training_config = json.load(f)
        logger.info(f"[Serialization] Loaded config from {config_path}")
    else:
        training_config = {}
    
    # 5. Check for tokenizer
    tokenizer_path = path / "tokenizer"
    
    # Build result config
    result_config = {
        "training_config": training_config,
        "metadata": metadata,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path.exists() else None,
        "model_path": str(path),
    }
    
    logger.info(f"[Serialization] Model loaded successfully from {path}")
    return model, result_config


def _get_highnoon_custom_objects() -> dict[str, Any]:
    """Get dictionary of HighNoon custom Keras objects for deserialization.
    
    Imports all custom layers, modules, and components that may be used
    in saved models to ensure proper deserialization.
    
    Returns:
        Dictionary mapping class names to classes.
    """
    custom_objects = {}
    
    # Import custom layers with fallback for missing modules
    layer_imports = [
        ("highnoon.models.reasoning.reasoning_module", "ReasoningModule"),
        ("highnoon.models.moe", "SparseMoE"),
        ("highnoon.models.moe", "FusedMoEBlock"),
        ("highnoon.models.hamiltonian", "TimeCrystalBlock"),
        ("highnoon.models.hamiltonian", "HamiltonianNeuralNetwork"),
        ("highnoon.models.spatial", "SpatialBlock"),
        ("highnoon.models.spatial", "QMambaBlock"),
        ("highnoon.models.layers.wlam", "WLAMBlock"),
        ("highnoon.models.layers.flash_linear_attention", "FlashLinearAttention"),
        ("highnoon.models.layers.latent_kv_attention", "LatentKVAttention"),
        ("highnoon.models.layers.tt_dense", "TTDense"),
        ("highnoon.models.layers.collapse", "SuperpositionCollapseLayer"),
        ("highnoon.models.layers.cayley_weights", "CayleyLinear"),
        ("highnoon.quantum.layers", "HybridVQCLayer"),
        ("highnoon.quantum.layers", "EvolutionTimeVQCLayer"),
    ]
    
    for module_path, class_name in layer_imports:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name, None)
            if cls is not None:
                custom_objects[class_name] = cls
        except (ImportError, AttributeError):
            pass  # Layer not available, skip
    
    return custom_objects


def load_tokenizer(path: str | Path) -> Any:
    """Load tokenizer from HighNoon model directory or tokenizer path.
    
    Args:
        path: Path to model directory or tokenizer subdirectory.
        
    Returns:
        Loaded tokenizer instance.
        
    Raises:
        FileNotFoundError: If tokenizer files not found.
    """
    path = Path(path)
    
    # Check if this is a model directory or tokenizer directory
    if (path / "tokenizer").exists():
        tokenizer_path = path / "tokenizer"
    else:
        tokenizer_path = path
    
    # Try to load QWTTextTokenizer
    try:
        from highnoon.tokenization import QWTTextTokenizer
        
        vocab_path = tokenizer_path / "vocab.json"
        merges_path = tokenizer_path / "merges.txt"
        
        if vocab_path.exists():
            tokenizer = QWTTextTokenizer.from_files(
                vocab_file=str(vocab_path),
                merges_file=str(merges_path) if merges_path.exists() else None,
            )
            logger.info(f"[Serialization] Loaded QWTTextTokenizer from {tokenizer_path}")
            return tokenizer
    except (ImportError, AttributeError, FileNotFoundError):
        pass
    
    # Try to load generic tokenizer config
    config_path = tokenizer_path / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        logger.info(f"[Serialization] Loaded tokenizer config from {config_path}")
        return config
    
    raise FileNotFoundError(f"No tokenizer found at {tokenizer_path}")


def _serialize_config(config: Any) -> dict[str, Any]:
    """Convert configuration to JSON-serializable dictionary.
    
    Handles EnterpriseTrainingConfig, HPOTrainingConfig, and plain dicts.
    """
    if config is None:
        return {}
    
    if isinstance(config, dict):
        return config
    
    # Try dataclass asdict
    try:
        return asdict(config)
    except (TypeError, ValueError):
        pass
    
    # Try to_dict method
    if hasattr(config, "to_dict"):
        return config.to_dict()
    
    # Fallback: extract __dict__
    if hasattr(config, "__dict__"):
        result = {}
        for k, v in config.__dict__.items():
            if not k.startswith("_"):
                try:
                    json.dumps(v)  # Check if serializable
                    result[k] = v
                except (TypeError, ValueError):
                    result[k] = str(v)
        return result
    
    return {"raw": str(config)}


def _save_tokenizer(tokenizer: Any, tokenizer_dir: Path) -> None:
    """Save tokenizer to directory.
    
    Supports QWTTextTokenizer and HuggingFace tokenizers.
    """
    # Try QWTTextTokenizer save method
    if hasattr(tokenizer, "save"):
        tokenizer.save(str(tokenizer_dir))
        return
    
    # Try HuggingFace tokenizer
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(tokenizer_dir))
        return
    
    # Try to save vocabulary
    if hasattr(tokenizer, "vocab"):
        vocab_path = tokenizer_dir / "vocab.json"
        with open(vocab_path, "w") as f:
            json.dump(tokenizer.vocab, f, indent=2)
    
    # Save config if available
    if hasattr(tokenizer, "to_dict"):
        config_path = tokenizer_dir / "tokenizer_config.json"
        with open(config_path, "w") as f:
            json.dump(tokenizer.to_dict(), f, indent=2)


def _is_compatible_version(saved_version: str) -> bool:
    """Check if saved format version is compatible with current."""
    try:
        saved_major = int(saved_version.split(".")[0])
        current_major = int(FORMAT_VERSION.split(".")[0])
        return saved_major == current_major
    except (ValueError, IndexError):
        return False


__all__ = ["save_model", "load_model", "load_tokenizer"]
