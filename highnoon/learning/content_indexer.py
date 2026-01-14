# highnoon/learning/content_indexer.py
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

"""Content Indexer for Model Inference Update.

Processes various content sources (files, text, codebases) into hyperdimensional
embeddings for memory-augmented learning.

Supported content sources:
    - Text files (.txt, .md, .rst)
    - Code files (.py, .js, .ts, .go, .rs, .cpp, .h, .java, etc.)
    - Chat exports (JSON format)
    - Documents (processed as text)
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tensorflow as tf

from highnoon import config

logger = logging.getLogger(__name__)


# =============================================================================
# CONTENT SOURCE DEFINITION
# =============================================================================


@dataclass
class ContentSource:
    """Source of content for MIU learning.

    Attributes:
        source_type: Type of content source.
        path: Path to file or directory (for file-based sources).
        content: Raw text content (for direct text input).
        filters: Optional filters for content selection.
        metadata: Additional metadata about the source.

    Example:
        >>> source = ContentSource(
        ...     source_type="codebase",
        ...     path="./my_project",
        ...     filters={"extensions": [".py", ".md"]}
        ... )
    """

    source_type: Literal["text", "codebase", "document", "chat", "web"]
    path: Path | str | None = None
    content: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate content source configuration."""
        if self.path is None and self.content is None:
            raise ValueError("Either 'path' or 'content' must be provided")
        if self.path is not None:
            self.path = Path(self.path)


@dataclass
class ContentChunk:
    """A processed chunk of content ready for HD encoding.

    Attributes:
        text: The text content of this chunk.
        source_id: Identifier of the source this chunk came from.
        chunk_idx: Index of this chunk within its source.
        metadata: Additional metadata (file path, line numbers, etc.).
        embedding: Optional pre-computed embedding.
    """

    text: str
    source_id: str
    chunk_idx: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None


# =============================================================================
# CONTENT INDEXER
# =============================================================================


class ContentIndexer:
    """Indexes and processes content for MIU learning.

    This class handles:
        1. File discovery and reading
        2. Content chunking with overlap
        3. HD embedding generation
        4. Memory-efficient streaming

    Attributes:
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        hd_dim: Hyperdimensional embedding dimension.

    Example:
        >>> indexer = ContentIndexer()
        >>> for chunk in indexer.index_source(source):
        ...     print(f"Chunk {chunk.chunk_idx}: {len(chunk.text)} chars")
    """

    # File extensions by category
    CODE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".go",
        ".rs",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".java",
        ".kt",
        ".swift",
        ".rb",
        ".php",
        ".cs",
        ".scala",
        ".hs",
        ".ml",
        ".ex",
        ".exs",
        ".clj",
        ".lisp",
        ".sh",
        ".bash",
        ".zsh",
        ".sql",
        ".r",
        ".R",
        ".jl",
    }
    TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".org", ".adoc", ".tex"}
    CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml"}

    def __init__(
        self,
        chunk_size: int = 2048,
        chunk_overlap: int = 256,
        hd_dim: int | None = None,
    ) -> None:
        """Initialize ContentIndexer.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
            hd_dim: HD embedding dimension (defaults to config.MIU_HD_DIM).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.hd_dim = hd_dim or config.MIU_HD_DIM
        self._tokenizer = None
        logger.info(
            f"ContentIndexer initialized: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, hd_dim={self.hd_dim}"
        )

    @property
    def tokenizer(self):
        """Lazy-load tokenizer to avoid circular imports."""
        if self._tokenizer is None:
            self._tokenizer = config.get_tokenizer()
        return self._tokenizer

    def index_source(self, source: ContentSource) -> Iterator[ContentChunk]:
        """Index a content source into chunks.

        Args:
            source: The content source to index.

        Yields:
            ContentChunk objects ready for HD encoding.
        """
        source_id = self._compute_source_id(source)

        if source.content is not None:
            # Direct text content
            yield from self._chunk_text(source.content, source_id, source.metadata)

        elif source.source_type == "codebase":
            yield from self._index_codebase(source, source_id)

        elif source.source_type == "document":
            yield from self._index_document(source, source_id)

        elif source.source_type == "chat":
            yield from self._index_chat(source, source_id)

        else:
            # Generic file handling
            yield from self._index_file(source, source_id)

    def _compute_source_id(self, source: ContentSource) -> str:
        """Compute a unique ID for a content source."""
        if source.path is not None:
            return hashlib.sha256(str(source.path).encode()).hexdigest()[:16]
        elif source.content is not None:
            return hashlib.sha256(source.content.encode()).hexdigest()[:16]
        return hashlib.sha256(str(source).encode()).hexdigest()[:16]

    def _chunk_text(
        self,
        text: str,
        source_id: str,
        metadata: dict[str, Any],
    ) -> Iterator[ContentChunk]:
        """Split text into overlapping chunks.

        Args:
            text: Text content to chunk.
            source_id: Source identifier.
            metadata: Metadata to attach to chunks.

        Yields:
            ContentChunk objects.
        """
        if not text.strip():
            return

        # Split on paragraph boundaries when possible
        paragraphs = text.split("\n\n")
        current_chunk = ""
        chunk_idx = 0
        start_para = 0

        for para_idx, para in enumerate(paragraphs):
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Emit current chunk
                if current_chunk.strip():
                    yield ContentChunk(
                        text=current_chunk.strip(),
                        source_id=source_id,
                        chunk_idx=chunk_idx,
                        metadata={
                            **metadata,
                            "start_paragraph": start_para,
                            "end_paragraph": para_idx - 1,
                        },
                    )
                    chunk_idx += 1

                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + para + "\n\n"
                start_para = para_idx

        # Emit final chunk
        if current_chunk.strip():
            yield ContentChunk(
                text=current_chunk.strip(),
                source_id=source_id,
                chunk_idx=chunk_idx,
                metadata={
                    **metadata,
                    "start_paragraph": start_para,
                    "end_paragraph": len(paragraphs) - 1,
                },
            )

    def _index_codebase(self, source: ContentSource, source_id: str) -> Iterator[ContentChunk]:
        """Index a codebase directory.

        Args:
            source: Content source with codebase path.
            source_id: Source identifier.

        Yields:
            ContentChunk objects for each code file.
        """
        if source.path is None:
            return

        path = Path(source.path)
        if not path.exists():
            logger.warning(f"Codebase path does not exist: {path}")
            return

        # Get file filters
        extensions = source.filters.get("extensions", self.CODE_EXTENSIONS)
        exclude_patterns = source.filters.get(
            "exclude", ["**/node_modules/**", "**/.git/**", "**/venv/**", "**/__pycache__/**"]
        )

        # Find all matching files
        files = []
        if path.is_file():
            files = [path]
        else:
            for ext in extensions:
                pattern = f"**/*{ext}" if not ext.startswith("*") else ext
                files.extend(path.glob(pattern))

        # Filter out excluded paths
        def is_excluded(file_path: Path) -> bool:
            file_str = str(file_path)
            return any(
                file_path.match(pattern) or pattern.strip("*") in file_str
                for pattern in exclude_patterns
            )

        files = [f for f in files if not is_excluded(f)]
        logger.info(f"Indexing {len(files)} files from {path}")

        for file_path in sorted(files):
            try:
                content = file_path.read_text(encoding="utf-8")
                file_metadata = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "extension": file_path.suffix,
                    "source_type": "code",
                }
                yield from self._chunk_text(content, source_id, file_metadata)
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to read {file_path}: {e}")

    def _index_document(self, source: ContentSource, source_id: str) -> Iterator[ContentChunk]:
        """Index a document file or glob pattern.

        Args:
            source: Content source with document path.
            source_id: Source identifier.

        Yields:
            ContentChunk objects.
        """
        if source.path is None:
            return

        path = Path(source.path)

        # Handle glob patterns
        if "*" in str(path):
            files = list(glob.glob(str(path), recursive=True))
        elif path.is_dir():
            extensions = self.TEXT_EXTENSIONS | self.CONFIG_EXTENSIONS
            files = [str(f) for ext in extensions for f in path.glob(f"**/*{ext}")]
        else:
            files = [str(path)]

        for file_str in sorted(files):
            file_path = Path(file_str)
            try:
                content = file_path.read_text(encoding="utf-8")
                file_metadata = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "source_type": "document",
                }
                yield from self._chunk_text(content, source_id, file_metadata)
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to read {file_path}: {e}")

    def _index_chat(self, source: ContentSource, source_id: str) -> Iterator[ContentChunk]:
        """Index chat export JSON.

        Expected format:
            {"messages": [{"role": "user", "content": "..."}, ...]}
            or [{"role": "user", "content": "..."}, ...]

        Args:
            source: Content source with chat export.
            source_id: Source identifier.

        Yields:
            ContentChunk objects.
        """
        if source.path is None and source.content is None:
            return

        try:
            if source.path is not None:
                with open(source.path, encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(source.content or "{}")

            # Handle different formats
            messages = data.get("messages", data) if isinstance(data, dict) else data

            if not isinstance(messages, list):
                logger.warning("Invalid chat format: expected list of messages")
                return

            # Combine messages into conversation chunks
            conversation_text = ""
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    role = msg.get("role", "unknown")
                    content = msg["content"]
                    conversation_text += f"{role.upper()}: {content}\n\n"

            metadata = {"source_type": "chat"}
            yield from self._chunk_text(conversation_text, source_id, metadata)

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse chat export: {e}")

    def _index_file(self, source: ContentSource, source_id: str) -> Iterator[ContentChunk]:
        """Index a generic file.

        Args:
            source: Content source with file path.
            source_id: Source identifier.

        Yields:
            ContentChunk objects.
        """
        if source.path is None:
            return

        path = Path(source.path)
        if not path.exists() or not path.is_file():
            logger.warning(f"File does not exist: {path}")
            return

        try:
            content = path.read_text(encoding="utf-8")
            metadata = {
                "file_path": str(path),
                "file_name": path.name,
                "source_type": source.source_type,
            }
            yield from self._chunk_text(content, source_id, metadata)
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to read {path}: {e}")

    def encode_chunk(self, chunk: ContentChunk) -> tf.Tensor:
        """Encode a content chunk into HD embedding.

        Uses the tokenizer and HD embedding system to convert text
        into a hyperdimensional vector representation.

        Args:
            chunk: Content chunk to encode.

        Returns:
            HD embedding tensor of shape [hd_dim].
        """
        # Tokenize
        tokens = self.tokenizer(
            chunk.text,
            max_length=self.chunk_size // 4,  # Approximate tokens from chars
            truncation=True,
            return_tensors="tf",
        )

        # Get token embeddings and bundle into HD space
        # This uses the holographic bundling from hyperdimensional_embedding_op.h
        token_ids = tokens["input_ids"]

        # Use simple averaging as fallback, can be replaced with full HQE
        # when integrated with model's embedding layer
        embeddings = tf.nn.embedding_lookup(
            self._get_base_vectors(),
            token_ids,
        )

        # Bundle via mean pooling (simplified holographic bundle)
        hd_embedding = tf.reduce_mean(embeddings, axis=[0, 1])

        # Normalize to unit sphere
        hd_embedding = tf.nn.l2_normalize(hd_embedding, epsilon=1e-6)

        chunk.embedding = hd_embedding.numpy()
        return hd_embedding

    def _get_base_vectors(self) -> tf.Variable:
        """Get or create base HD vectors for token embedding.

        Returns:
            Base vectors of shape [vocab_size, hd_dim].
        """
        if not hasattr(self, "_base_vectors"):
            vocab_size = self.tokenizer.vocab_size
            # Initialize with random orthogonal vectors
            initializer = tf.keras.initializers.Orthogonal()
            self._base_vectors = tf.Variable(
                initializer((vocab_size, self.hd_dim)),
                trainable=False,
                name="miu_base_vectors",
            )
        return self._base_vectors


__all__ = ["ContentIndexer", "ContentSource", "ContentChunk"]
