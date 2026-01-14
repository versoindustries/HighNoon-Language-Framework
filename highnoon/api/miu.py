# highnoon/api/miu.py
# Copyright 2025-2026 Verso Industries (Author: Michael B. Zimmerman)
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

"""Model Inference Update (MIU) Public API.

High-level API for downstream inference software to integrate user-activated
continual learning. This is the primary interface for teaching the model
new content from chats, codebases, documents, and web searches.

Example:
    >>> from highnoon.api import ModelInferenceUpdate, ContentSource
    >>>
    >>> # Initialize with your model
    >>> miu = ModelInferenceUpdate(model)
    >>>
    >>> # Learn from selected content
    >>> result = miu.learn_from_content([
    ...     ContentSource(source_type="codebase", path="./my_project"),
    ...     ContentSource(source_type="document", path="./docs/*.md"),
    ... ])
    >>>
    >>> if result.success:
    ...     print(f"Learned from {result.chunks_learned} chunks!")
    >>> else:
    ...     # Rollback if something went wrong
    ...     miu.rollback(result.checkpoint_path)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import tensorflow as tf

from highnoon import config

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC DATA TYPES
# =============================================================================


@dataclass
class ContentSource:
    """Source of content for model learning.

    Use this to specify what content the model should learn from.
    Supports codebases, documents, chat exports, and raw text.

    Attributes:
        source_type: Type of content. One of:
            - "text": Raw text content (provide in 'content' field)
            - "codebase": Directory with source code files
            - "document": Text files (supports glob patterns)
            - "chat": JSON chat export
            - "web": Cached web content (not yet implemented)
        path: Path to file or directory (for file-based sources).
        content: Raw text content (for direct text input).
        filters: Optional filters to customize what's included.
            For codebases: {"extensions": [".py", ".md"], "exclude": ["**/test/**"]}
            For documents: {"extensions": [".txt", ".md"]}

    Examples:
        >>> # Learn from a Python project
        >>> ContentSource(
        ...     source_type="codebase",
        ...     path="./my_project",
        ...     filters={"extensions": [".py"], "exclude": ["**/tests/**"]}
        ... )

        >>> # Learn from documentation
        >>> ContentSource(
        ...     source_type="document",
        ...     path="./docs/**/*.md"
        ... )

        >>> # Learn from raw text
        >>> ContentSource(
        ...     source_type="text",
        ...     content="Important information to remember..."
        ... )

        >>> # Learn from chat export (OpenAI format)
        >>> ContentSource(
        ...     source_type="chat",
        ...     path="./chat_export.json"
        ... )
    """

    source_type: Literal["text", "codebase", "document", "chat", "web"]
    path: Path | str | None = None
    content: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.path is None and self.content is None:
            raise ValueError("Either 'path' or 'content' must be provided")
        if self.path is not None:
            self.path = Path(self.path)

    def to_internal(self):
        """Convert to internal ContentSource format."""
        from highnoon.learning.content_indexer import ContentSource as InternalCS

        return InternalCS(
            source_type=self.source_type,
            path=self.path,
            content=self.content,
            filters=self.filters,
        )


@dataclass
class MIUConfig:
    """Configuration for a Model Inference Update session.

    Customize the learning process with these options.
    All fields have sensible defaults from the global config.

    Attributes:
        learning_rate: Learning rate for weight updates.
            Default: 1e-5 (very conservative to prevent forgetting)
        max_steps: Maximum gradient update steps.
            Default: 100
        batch_size: Mini-batch size for updates.
            Default: 4
        crystallize_after: Whether to protect new knowledge.
            Default: True (recommended)
        checkpoint_name: Custom name for the rollback checkpoint.
            Default: auto-generated from session ID
        max_gradient_norm: Gradient clipping threshold.
            Default: 1.0

    Example:
        >>> config = MIUConfig(
        ...     learning_rate=1e-6,  # Extra conservative
        ...     max_steps=50,
        ...     checkpoint_name="project-alpha-v1"
        ... )
    """

    learning_rate: float | None = None
    max_steps: int | None = None
    batch_size: int | None = None
    crystallize_after: bool = True
    checkpoint_name: str | None = None
    max_gradient_norm: float | None = None

    def __post_init__(self) -> None:
        """Apply defaults from config."""
        if self.learning_rate is None:
            self.learning_rate = config.MIU_LEARNING_RATE
        if self.max_steps is None:
            self.max_steps = config.MIU_MAX_STEPS
        if self.batch_size is None:
            self.batch_size = config.MIU_BATCH_SIZE
        if self.max_gradient_norm is None:
            self.max_gradient_norm = config.MIU_MAX_GRADIENT_NORM


@dataclass
class MIUResult:
    """Result of a Model Inference Update operation.

    Check this after learning to verify success and access metrics.

    Attributes:
        success: Whether learning completed successfully.
        chunks_learned: Number of content chunks incorporated.
        num_updates: Number of gradient updates applied.
        checkpoint_path: Path for rollback (if enabled).
        rollback_available: Whether rollback is possible.
        message: Human-readable status message.
        metrics: Detailed metrics dictionary.
        errors: List of any errors encountered.

    Example:
        >>> result = miu.learn_from_content(sources)
        >>> if result.success:
        ...     print(result.message)
        ...     print(f"Duration: {result.metrics['duration_seconds']:.1f}s")
        >>> else:
        ...     print(f"Failed: {result.errors}")
        ...     miu.rollback(result.checkpoint_path)
    """

    success: bool
    chunks_learned: int = 0
    num_updates: int = 0
    checkpoint_path: Path | None = None
    rollback_available: bool = False
    message: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint.

    Returned by list_checkpoints() for browsing available rollback points.

    Attributes:
        name: Checkpoint identifier.
        created_at: When the checkpoint was created.
        path: Path to checkpoint files.
        description: Brief description.
    """

    name: str
    created_at: str
    path: Path
    description: str = ""


# =============================================================================
# PUBLIC API CLASS
# =============================================================================


class ModelInferenceUpdate:
    """High-level API for user-activated continual learning.

    This class provides a simple interface for downstream inference software
    to teach the model new content. It handles:
        - Content indexing and chunking
        - Hyperdimensional encoding
        - Protected weight updates (QHPM crystallization)
        - Automatic checkpointing for rollback safety

    Thread Safety:
        Not thread-safe. Use one instance per model, and don't call
        methods concurrently.

    Example:
        >>> from highnoon.api import ModelInferenceUpdate, ContentSource
        >>>
        >>> # Create MIU instance with your model
        >>> miu = ModelInferenceUpdate(model)
        >>>
        >>> # Define what to learn
        >>> sources = [
        ...     ContentSource(source_type="codebase", path="./my_code"),
        ...     ContentSource(source_type="document", path="./my_docs"),
        ... ]
        >>>
        >>> # Learn with progress reporting
        >>> def on_progress(done, total):
        ...     print(f"Processing: {done} chunks...")
        >>>
        >>> result = miu.learn_from_content(
        ...     sources,
        ...     on_progress=on_progress
        ... )
        >>>
        >>> if result.success:
        ...     print(f"✓ {result.message}")
        >>> else:
        ...     print(f"✗ Failed: {result.errors}")
        ...     miu.rollback(result.checkpoint_path)
    """

    def __init__(
        self,
        model: tf.keras.Model,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        """Initialize ModelInferenceUpdate.

        Args:
            model: The model to teach. Must be a tf.keras.Model with
                trainable weights.
            checkpoint_dir: Directory for rollback checkpoints.
                Defaults to config.MIU_CHECKPOINT_DIR ("checkpoints/miu").

        Raises:
            ValueError: If model has no trainable variables.
        """
        if not model.trainable_variables:
            raise ValueError("Model has no trainable variables")

        # Lazy import to avoid circular dependencies
        from highnoon.learning.miu_controller import MIUController

        self._controller = MIUController(
            model=model,
            checkpoint_dir=checkpoint_dir,
        )
        self._model = model

        logger.info(
            f"ModelInferenceUpdate initialized for model with "
            f"{len(model.trainable_variables)} trainable variables"
        )

    def learn_from_content(
        self,
        sources: list[ContentSource],
        config: MIUConfig | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> MIUResult:
        """Teach the model from selected content.

        This is the main method for continual learning. It:
            1. Creates a safety checkpoint for rollback
            2. Indexes and chunks the content
            3. Encodes chunks into hyperdimensional space
            4. Applies protected gradient updates
            5. Optionally crystallizes new knowledge for protection

        Args:
            sources: List of ContentSource objects defining what to learn.
            config: Optional MIUConfig to customize the learning process.
            on_progress: Optional callback(chunks_done, total) for progress
                updates. Called periodically during indexing.

        Returns:
            MIUResult with success status, metrics, and rollback info.

        Example:
            >>> sources = [ContentSource(source_type="codebase", path="./")]
            >>> result = miu.learn_from_content(sources)
            >>> print(f"Learned: {result.chunks_learned} chunks")
        """
        miu_config = config or MIUConfig()

        # Convert public sources to internal format
        internal_sources = [s.to_internal() for s in sources]

        try:
            # Begin session with checkpoint
            session = self._controller.begin_session(
                sources=internal_sources,
                checkpoint_name=miu_config.checkpoint_name,
                progress_callback=on_progress,
            )

            # Finalize and apply updates
            internal_result = self._controller.finalize_session(
                session=session,
                max_steps=miu_config.max_steps,
            )

            # Convert to public result
            return MIUResult(
                success=internal_result.success,
                chunks_learned=internal_result.chunks_learned,
                num_updates=internal_result.num_updates,
                checkpoint_path=internal_result.checkpoint_path,
                rollback_available=internal_result.rollback_available,
                message=self._format_message(internal_result),
                metrics=internal_result.metrics,
                errors=internal_result.errors,
            )

        except Exception as e:
            logger.exception("MIU learning failed")
            return MIUResult(
                success=False,
                errors=[str(e)],
                message=f"Learning failed: {e}",
            )

    def _format_message(self, result) -> str:
        """Format a human-readable result message."""
        if result.success:
            duration = result.metrics.get("duration_seconds", 0)
            return (
                f"Successfully learned from {result.chunks_learned} chunks "
                f"in {duration:.1f}s ({result.num_updates} updates, "
                f"{result.crystallized_directions} protected)"
            )
        else:
            return f"Learning failed: {', '.join(result.errors)}"

    def rollback(self, checkpoint: Path | str | None = None) -> bool:
        """Rollback to a previous checkpoint.

        Use this to undo learning if something went wrong or if
        the model's behavior degraded after learning.

        Args:
            checkpoint: Path to checkpoint directory. If None, uses
                the most recent checkpoint.

        Returns:
            True if rollback succeeded, False otherwise.

        Example:
            >>> result = miu.learn_from_content(sources)
            >>> if not result.success:
            ...     miu.rollback(result.checkpoint_path)
        """
        if checkpoint is None:
            checkpoints = self._controller.list_checkpoints()
            if not checkpoints:
                logger.error("No checkpoints available for rollback")
                return False
            checkpoint = checkpoints[0].path

        return self._controller.rollback(Path(checkpoint))

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """List available rollback checkpoints.

        Returns:
            List of CheckpointInfo objects, newest first.

        Example:
            >>> for cp in miu.list_checkpoints():
            ...     print(f"{cp.name}: {cp.created_at}")
        """
        internal_list = self._controller.list_checkpoints()
        return [
            CheckpointInfo(
                name=cp.name,
                created_at=cp.created_at.isoformat(),
                path=cp.path,
                description=cp.sources_summary,
            )
            for cp in internal_list
        ]

    @property
    def is_enabled(self) -> bool:
        """Check if MIU is enabled in config."""
        return config.USE_MIU


__all__ = [
    "ModelInferenceUpdate",
    "ContentSource",
    "MIUConfig",
    "MIUResult",
    "CheckpointInfo",
]
