# highnoon/learning/miu_controller.py
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

"""Model Inference Update Controller.

Main orchestrator for user-activated continual learning sessions. Coordinates
content indexing, HD encoding, memory consolidation, and protected weight updates.

The MIU Controller provides a high-level interface for downstream software to:
    1. Begin learning sessions with selected content
    2. Monitor progress and control learning
    3. Apply protected weight updates
    4. Rollback to previous checkpoints if needed

Example:
    >>> controller = MIUController(model)
    >>> session = controller.begin_session([
    ...     ContentSource(source_type="codebase", path="./project"),
    ... ])
    >>> controller.add_content(session, more_sources)
    >>> result = controller.finalize_session(session)
    >>> print(f"Applied {result.num_updates} updates")
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import tensorflow as tf

from highnoon import config
from highnoon.learning.content_indexer import ContentChunk, ContentIndexer, ContentSource
from highnoon.learning.crystallized_optimizer import CrystallizationState, CrystallizedOptimizer
from highnoon.learning.hd_learner import HDLearner

logger = logging.getLogger(__name__)


# =============================================================================
# SESSION AND RESULT TYPES
# =============================================================================


@dataclass
class MIUSession:
    """Active Model Inference Update session.

    Attributes:
        session_id: Unique session identifier.
        sources: Content sources being learned.
        start_time: When session started.
        chunks_processed: Number of content chunks processed.
        status: Current session status.
        checkpoint_path: Path to pre-session checkpoint.
        metadata: Additional session metadata.
    """

    session_id: str
    sources: list[ContentSource]
    start_time: float = field(default_factory=time.time)
    chunks_processed: int = 0
    status: str = "active"
    checkpoint_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        return time.time() - self.start_time


@dataclass
class MIUResult:
    """Result of a completed MIU session.

    Attributes:
        success: Whether the session completed successfully.
        session_id: Session identifier.
        num_updates: Number of gradient updates applied.
        chunks_learned: Number of content chunks incorporated.
        crystallized_directions: Number of new protected directions.
        checkpoint_path: Path to rollback checkpoint.
        rollback_available: Whether rollback is possible.
        metrics: Performance and quality metrics.
        errors: Any errors encountered.
    """

    success: bool
    session_id: str
    num_updates: int = 0
    chunks_learned: int = 0
    crystallized_directions: int = 0
    checkpoint_path: Path | None = None
    rollback_available: bool = True
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class CheckpointInfo:
    """Information about an MIU checkpoint.

    Attributes:
        name: Checkpoint name.
        path: Path to checkpoint files.
        created_at: When checkpoint was created.
        session_id: Associated session ID.
        sources_summary: Summary of learned content.
    """

    name: str
    path: Path
    created_at: datetime
    session_id: str
    sources_summary: str


# =============================================================================
# MIU CONTROLLER
# =============================================================================


class MIUController:
    """Main controller for Model Inference Update sessions.

    Coordinates the full MIU pipeline:
        1. Content indexing and HD encoding
        2. Memory consolidation via Modern Hopfield
        3. Gradient computation with crystallization protection
        4. Checkpoint management for rollback safety

    Attributes:
        model: The model to update.
        indexer: Content indexer for processing sources.
        learner: HD learner for memory and gradients.
        optimizer: Crystallized optimizer for protected updates.

    Example:
        >>> controller = MIUController(model)
        >>> session = controller.begin_session([
        ...     ContentSource(source_type="codebase", path="./project"),
        ...     ContentSource(source_type="document", path="./docs/*.md"),
        ... ])
        >>> # Monitor progress
        >>> print(f"Processed: {session.chunks_processed} chunks")
        >>> # Finalize and apply updates
        >>> result = controller.finalize_session(session)
        >>> # Rollback if needed
        >>> if not result.success:
        ...     controller.rollback(result.checkpoint_path)
    """

    def __init__(
        self,
        model: tf.keras.Model,
        checkpoint_dir: str | Path | None = None,
        learning_rate: float | None = None,
        max_gradient_norm: float | None = None,
    ) -> None:
        """Initialize MIUController.

        Args:
            model: The model to update.
            checkpoint_dir: Directory for checkpoints (defaults to config).
            learning_rate: Learning rate for updates (defaults to config).
            max_gradient_norm: Gradient clipping threshold (defaults to config).
        """
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir or config.MIU_CHECKPOINT_DIR)
        self.learning_rate = learning_rate or config.MIU_LEARNING_RATE
        self.max_gradient_norm = max_gradient_norm or config.MIU_MAX_GRADIENT_NORM

        # Create components
        self.indexer = ContentIndexer()
        self.learner = HDLearner(model=model)

        # Create crystallized optimizer
        base_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.optimizer = CrystallizedOptimizer(
            base_optimizer=base_optimizer,
            enabled=config.MIU_USE_CRYSTALLIZATION,
        )

        # Session tracking
        self._active_sessions: dict[str, MIUSession] = {}
        self._checkpoints: list[CheckpointInfo] = []

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._load_checkpoint_history()

        logger.info(
            f"MIUController initialized: checkpoint_dir={self.checkpoint_dir}, "
            f"lr={self.learning_rate}, max_grad_norm={self.max_gradient_norm}"
        )

    def begin_session(
        self,
        sources: list[ContentSource],
        checkpoint_name: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> MIUSession:
        """Begin a new MIU learning session.

        Creates a pre-session checkpoint for rollback safety, then
        begins indexing and encoding the provided content sources.

        Args:
            sources: Content sources to learn from.
            checkpoint_name: Optional name for the checkpoint.
            progress_callback: Optional callback(chunks_done, total_chunks).

        Returns:
            Active MIUSession for tracking progress.
        """
        session_id = str(uuid4())[:8]
        checkpoint_name = checkpoint_name or f"miu_{session_id}"

        logger.info(f"Beginning MIU session {session_id} with {len(sources)} sources")

        # Create pre-session checkpoint for rollback
        checkpoint_path = None
        if config.MIU_CHECKPOINT_ENABLED:
            checkpoint_path = self._create_checkpoint(checkpoint_name, session_id)

        # Create session
        session = MIUSession(
            session_id=session_id,
            sources=sources,
            checkpoint_path=checkpoint_path,
            metadata={"checkpoint_name": checkpoint_name},
        )
        self._active_sessions[session_id] = session

        # Index content
        self._index_sources(session, progress_callback)

        return session

    def _index_sources(
        self,
        session: MIUSession,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Index all sources in a session.

        Args:
            session: The active session.
            progress_callback: Optional progress callback.
        """
        total_chunks = 0
        all_chunks: list[ContentChunk] = []

        for source in session.sources:
            for chunk in self.indexer.index_source(source):
                # Encode chunk to HD embedding
                self.indexer.encode_chunk(chunk)
                all_chunks.append(chunk)
                total_chunks += 1

                if progress_callback:
                    progress_callback(total_chunks, -1)  # -1 = unknown total

        # Add to HD learner memory
        self.learner.add_embeddings(all_chunks)
        session.chunks_processed = total_chunks

        logger.info(f"Session {session.session_id}: indexed {total_chunks} chunks")

    def add_content(
        self,
        session: MIUSession,
        sources: list[ContentSource],
    ) -> int:
        """Add more content to an active session.

        Args:
            session: The active session.
            sources: Additional content sources.

        Returns:
            Number of new chunks added.
        """
        if session.status != "active":
            raise ValueError(f"Session {session.session_id} is not active")

        initial_count = session.chunks_processed
        session.sources.extend(sources)
        self._index_sources(session)

        added = session.chunks_processed - initial_count
        logger.info(f"Added {added} chunks to session {session.session_id}")
        return added

    def finalize_session(
        self,
        session: MIUSession,
        max_steps: int | None = None,
    ) -> MIUResult:
        """Finalize session and apply weight updates.

        Consolidates memory, computes gradients, applies protected updates,
        and optionally crystallizes new knowledge.

        Args:
            session: The session to finalize.
            max_steps: Maximum update steps (defaults to config).

        Returns:
            MIUResult with outcome details.
        """
        if session.status != "active":
            return MIUResult(
                success=False,
                session_id=session.session_id,
                errors=[f"Session is {session.status}, not active"],
            )

        max_steps = max_steps or config.MIU_MAX_STEPS
        logger.info(f"Finalizing session {session.session_id} with max_steps={max_steps}")

        try:
            # Consolidate memory
            consolidated = self.learner.consolidate_memory()

            # Apply gradient updates
            num_updates = 0

            for _step in range(max_steps):
                # Create learning batch
                batch = self.learner.create_learning_batch()
                if batch.embeddings.shape[0] == 0:
                    break

                # Compute gradients
                with tf.GradientTape() as tape:
                    grads_and_vars = self.learner.compute_gradients(batch, tape)

                if not grads_and_vars:
                    break

                # Clip gradients
                grads_and_vars = self._clip_gradients(grads_and_vars)

                # Apply with crystallization protection
                self.optimizer.apply_gradients(grads_and_vars)
                num_updates += 1

            # Crystallize new knowledge if enabled
            crystallized = 0
            if config.MIU_USE_CRYSTALLIZATION:
                crystallized = self._crystallize_learned_content()

            # Update session status
            session.status = "completed"
            del self._active_sessions[session.session_id]

            result = MIUResult(
                success=True,
                session_id=session.session_id,
                num_updates=num_updates,
                chunks_learned=session.chunks_processed,
                crystallized_directions=crystallized,
                checkpoint_path=session.checkpoint_path,
                rollback_available=session.checkpoint_path is not None,
                metrics={
                    "duration_seconds": session.duration_seconds,
                    "chunks_per_second": session.chunks_processed
                    / max(session.duration_seconds, 1),
                    "consolidated_memories": consolidated,
                },
            )

            logger.info(
                f"Session {session.session_id} completed: "
                f"{num_updates} updates, {crystallized} crystallized"
            )

            return result

        except Exception as e:
            logger.error(f"Session {session.session_id} failed: {e}")
            session.status = "failed"
            return MIUResult(
                success=False,
                session_id=session.session_id,
                checkpoint_path=session.checkpoint_path,
                rollback_available=session.checkpoint_path is not None,
                errors=[str(e)],
            )

    def _clip_gradients(
        self,
        grads_and_vars: list[tuple[tf.Tensor, tf.Variable]],
    ) -> list[tuple[tf.Tensor, tf.Variable]]:
        """Clip gradients by global norm.

        Args:
            grads_and_vars: List of (gradient, variable) tuples.

        Returns:
            Clipped gradients and variables.
        """
        grads = [g for g, _ in grads_and_vars]
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_gradient_norm)
        return list(zip(clipped_grads, [v for _, v in grads_and_vars]))

    def _crystallize_learned_content(self) -> int:
        """Crystallize directions from learned content.

        Returns:
            Number of directions crystallized.
        """
        crystallized_embeddings = self.learner.get_crystallized_embeddings()

        if not crystallized_embeddings:
            return 0

        # Crystallize for relevant variables
        count = 0
        for var in self.model.trainable_variables:
            if "embedding" in var.name.lower() or "dense" in var.name.lower():
                self.optimizer.crystallize_from_embeddings(crystallized_embeddings, var)
                count += len(crystallized_embeddings)

        return count

    def _create_checkpoint(self, name: str, session_id: str) -> Path:
        """Create a model checkpoint.

        Args:
            name: Checkpoint name.
            session_id: Associated session ID.

        Returns:
            Path to checkpoint directory.
        """
        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        self.model.save_weights(str(checkpoint_path / "model_weights"))

        # Save crystallization state
        self.optimizer.crystallization_state.save(checkpoint_path)

        # Save metadata
        metadata = {
            "name": name,
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "model_name": self.model.name if hasattr(self.model, "name") else "unknown",
        }
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Track checkpoint
        info = CheckpointInfo(
            name=name,
            path=checkpoint_path,
            created_at=datetime.now(),
            session_id=session_id,
            sources_summary=f"Pre-session checkpoint for {session_id}",
        )
        self._checkpoints.append(info)
        self._prune_old_checkpoints()

        logger.info(f"Created checkpoint: {checkpoint_path}")
        return checkpoint_path

    def rollback(self, checkpoint_path: Path | str) -> bool:
        """Rollback to a previous checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.

        Returns:
            True if rollback succeeded.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            # Load model weights
            self.model.load_weights(str(checkpoint_path / "model_weights"))

            # Load crystallization state
            self.optimizer.crystallization_state = CrystallizationState.load(checkpoint_path)

            # Clear HD learner memory
            self.learner.clear_memory()

            logger.info(f"Rolled back to checkpoint: {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """List available checkpoints.

        Returns:
            List of CheckpointInfo objects.
        """
        return list(self._checkpoints)

    def _load_checkpoint_history(self) -> None:
        """Load existing checkpoints from disk."""
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                try:
                    with open(path / "metadata.json") as f:
                        metadata = json.load(f)
                    info = CheckpointInfo(
                        name=metadata.get("name", path.name),
                        path=path,
                        created_at=datetime.fromisoformat(metadata.get("created_at", "2000-01-01")),
                        session_id=metadata.get("session_id", "unknown"),
                        sources_summary=metadata.get("sources_summary", ""),
                    )
                    self._checkpoints.append(info)
                except (json.JSONDecodeError, OSError):
                    pass

        # Sort by creation time
        self._checkpoints.sort(key=lambda x: x.created_at, reverse=True)

    def _prune_old_checkpoints(self) -> None:
        """Remove oldest checkpoints if over limit."""
        max_checkpoints = config.MIU_MAX_CHECKPOINTS

        while len(self._checkpoints) > max_checkpoints:
            oldest = self._checkpoints.pop()
            try:
                shutil.rmtree(oldest.path)
                logger.info(f"Pruned old checkpoint: {oldest.name}")
            except OSError as e:
                logger.warning(f"Failed to prune checkpoint {oldest.name}: {e}")


__all__ = [
    "MIUController",
    "MIUSession",
    "MIUResult",
    "CheckpointInfo",
]
