# highnoon/retrieval/rag_module.py
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

"""Phase 13.11: Retrieval-Augmented Generation (RAG) Integration.

This module provides infrastructure for retrieval-augmented generation,
enabling models to access external knowledge bases during inference.

The RAG module supports:
- Multiple index backends (FAISS, Annoy, or numpy-based fallback)
- Document chunking and embedding
- Query-time retrieval with configurable top-k
- Seamless integration with generation loops

Example:
    >>> rag = RAGModule(embedding_dim=512, top_k=5)
    >>> rag.add_documents(["Doc 1 content", "Doc 2 content"])
    >>> context = rag.retrieve("query about documents")
    >>> # Use context in generation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import tensorflow as tf

from highnoon.config import RAG_TOP_K

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Container for retrieved documents and metadata.

    Attributes:
        documents: List of retrieved document texts.
        scores: Similarity scores for each document.
        ids: Optional document IDs.
        metadata: Optional metadata for each document.
    """

    documents: list[str]
    scores: list[float]
    ids: list[str] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def to_prompt_context(self, max_tokens: int = 2000) -> str:
        """Format retrieved documents as prompt context.

        Args:
            max_tokens: Approximate max tokens (characters / 4) to include.

        Returns:
            Formatted context string for prompt injection.
        """
        context_parts = []
        char_count = 0
        max_chars = max_tokens * 4  # Approximate

        for i, doc in enumerate(self.documents):
            if char_count + len(doc) > max_chars:
                break
            context_parts.append(f"[Document {i+1}] {doc}")
            char_count += len(doc)

        return "\n\n".join(context_parts)

    def __len__(self) -> int:
        return len(self.documents)


class EmbeddingIndex(Protocol):
    """Protocol for embedding index implementations."""

    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        """Add embeddings to the index."""
        ...

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors. Returns (distances, indices)."""
        ...

    def save(self, path: str) -> None:
        """Save index to disk."""
        ...

    def load(self, path: str) -> None:
        """Load index from disk."""
        ...


class NumpyIndex:
    """Simple numpy-based vector index (fallback when FAISS unavailable).

    This is a basic brute-force nearest neighbor search using numpy.
    For production use, prefer FAISS or Annoy.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.embeddings: np.ndarray | None = None
        self.ids: list[str] = []

    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        """Add embeddings to the index."""
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Brute-force nearest neighbor search."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return np.array([]), np.array([])

        # Normalize for cosine similarity
        query_norm = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-8)
        emb_norm = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=-1, keepdims=True) + 1e-8
        )

        # Compute cosine similarity
        if query_norm.ndim == 1:
            query_norm = query_norm[np.newaxis, :]

        similarities = np.dot(query_norm, emb_norm.T).squeeze()

        # Get top-k
        k = min(k, len(self.embeddings))
        top_indices = np.argsort(similarities)[-k:][::-1]
        top_scores = similarities[top_indices]

        return top_scores, top_indices

    def save(self, path: str) -> None:
        """Save index to disk."""
        np.savez(
            path,
            embeddings=self.embeddings,
            ids=np.array(self.ids, dtype=object),
        )

    def load(self, path: str) -> None:
        """Load index from disk."""
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.ids = data["ids"].tolist()

    def __len__(self) -> int:
        return len(self.ids)


class FAISSIndex:
    """FAISS-based vector index for efficient similarity search.

    CPU-optimized vector search. Requires: pip install faiss-cpu
    """

    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        """Initialize FAISS index.

        Args:
            embedding_dim: Dimension of embeddings.
            index_type: Index type - "flat" (brute force), "ivf" (inverted file),
                       or "hnsw" (hierarchical navigable small world).
        """
        try:
            import faiss

            self._faiss = faiss
        except ImportError:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")

        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.ids: list[str] = []

        if index_type == "flat":
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        """Add embeddings to the index."""
        # Normalize for cosine similarity
        embeddings = embeddings.astype(np.float32)
        self._faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        query = query.astype(np.float32)
        if query.ndim == 1:
            query = query[np.newaxis, :]
        self._faiss.normalize_L2(query)

        k = min(k, len(self.ids))
        scores, indices = self.index.search(query, k)
        return scores.squeeze(), indices.squeeze()

    def save(self, path: str) -> None:
        """Save index to disk."""
        self._faiss.write_index(self.index, path)
        # Save IDs separately
        with open(f"{path}.ids", "w") as f:
            json.dump(self.ids, f)

    def load(self, path: str) -> None:
        """Load index from disk."""
        self.index = self._faiss.read_index(path)
        with open(f"{path}.ids") as f:
            self.ids = json.load(f)

    def __len__(self) -> int:
        return len(self.ids)


class DocumentStore:
    """Simple document storage with chunking support.

    Stores documents and their chunks for retrieval.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize document store.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: dict[str, str] = {}
        self.chunks: dict[str, str] = {}  # chunk_id -> content
        self.chunk_to_doc: dict[str, str] = {}  # chunk_id -> doc_id
        self._next_doc_id = 0
        self._next_chunk_id = 0

    def add_document(
        self,
        content: str,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Add a document with chunking.

        Args:
            content: Document text content.
            doc_id: Optional document ID (auto-generated if not provided).
            metadata: Optional metadata to associate with document.

        Returns:
            List of chunk IDs created.
        """
        if doc_id is None:
            doc_id = f"doc_{self._next_doc_id}"
            self._next_doc_id += 1

        self.documents[doc_id] = content

        # Chunk the document
        chunk_ids = []
        start = 0
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end]

            chunk_id = f"{doc_id}_chunk_{self._next_chunk_id}"
            self._next_chunk_id += 1

            self.chunks[chunk_id] = chunk_content
            self.chunk_to_doc[chunk_id] = doc_id
            chunk_ids.append(chunk_id)

            start = end - self.chunk_overlap
            if start >= len(content):
                break

        return chunk_ids

    def get_chunk(self, chunk_id: str) -> str | None:
        """Get chunk content by ID."""
        return self.chunks.get(chunk_id)

    def get_document(self, doc_id: str) -> str | None:
        """Get full document by ID."""
        return self.documents.get(doc_id)

    def list_chunks(self) -> list[str]:
        """List all chunk IDs."""
        return list(self.chunks.keys())

    def __len__(self) -> int:
        return len(self.documents)


class RAGModule(tf.keras.layers.Layer):
    """Retrieval-Augmented Generation module.

    This layer retrieves relevant documents based on query embeddings
    and prepares context for injection into the model.

    Example:
        >>> model = HSMN(vocab_size=32000, embedding_dim=512)
        >>> rag = RAGModule(embedding_dim=512, top_k=5)
        >>> rag.add_documents(documents)
        >>>
        >>> # During inference
        >>> query_embedding = model.embed(input_ids)
        >>> context = rag.retrieve(query_embedding)
        >>> # Prepend context to generation
    """

    def __init__(
        self,
        embedding_dim: int,
        top_k: int = RAG_TOP_K,
        index_backend: str = "numpy",
        chunk_size: int = 512,
        **kwargs,
    ):
        """Initialize RAG module.

        Args:
            embedding_dim: Dimension of embeddings.
            top_k: Number of documents to retrieve.
            index_backend: "numpy", "faiss", or "annoy".
            chunk_size: Size of document chunks.
        """
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.index_backend = index_backend
        self.chunk_size = chunk_size

        # Initialize document store
        self.doc_store = DocumentStore(chunk_size=chunk_size)

        # Initialize index
        if index_backend == "faiss":
            try:
                self.index = FAISSIndex(embedding_dim)
            except ImportError:
                logger.warning("FAISS not available, falling back to numpy")
                self.index = NumpyIndex(embedding_dim)
        else:
            self.index = NumpyIndex(embedding_dim)

        # Embedding model (can be set externally)
        self.embedding_model: tf.keras.Model | None = None
        self._embeddings_cache: dict[str, np.ndarray] = {}

    def set_embedding_model(self, model: tf.keras.Model) -> None:
        """Set the embedding model for document encoding.

        Args:
            model: Keras model that produces embeddings from text.
        """
        self.embedding_model = model

    def add_documents(
        self,
        documents: list[str],
        embeddings: np.ndarray | None = None,
        doc_ids: list[str] | None = None,
    ) -> None:
        """Add documents to the index.

        Args:
            documents: List of document texts.
            embeddings: Pre-computed embeddings (optional).
            doc_ids: Optional document IDs.
        """
        all_chunk_ids = []
        all_chunk_embeddings = []

        for i, doc in enumerate(documents):
            doc_id = doc_ids[i] if doc_ids else None
            chunk_ids = self.doc_store.add_document(doc, doc_id=doc_id)
            all_chunk_ids.extend(chunk_ids)

            # Get or compute embeddings for each chunk
            for _j, chunk_id in enumerate(chunk_ids):
                chunk_content = self.doc_store.get_chunk(chunk_id)

                if embeddings is not None and i < len(embeddings):
                    # Use provided embedding (for the whole doc)
                    chunk_emb = embeddings[i]
                elif self.embedding_model is not None:
                    # Compute embedding
                    chunk_emb = self._compute_embedding(chunk_content)
                else:
                    # Random embedding fallback (for testing)
                    logger.warning("No embedding model set, using random embeddings")
                    chunk_emb = np.random.randn(self.embedding_dim)

                all_chunk_embeddings.append(chunk_emb)
                self._embeddings_cache[chunk_id] = chunk_emb

        # Add to index
        if all_chunk_embeddings:
            embeddings_array = np.array(all_chunk_embeddings)
            self.index.add(embeddings_array, all_chunk_ids)

        logger.info(
            f"Added {len(documents)} documents " f"({len(all_chunk_ids)} chunks) to RAG index"
        )

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text using the embedding model."""
        if self.embedding_model is None:
            raise ValueError("Embedding model not set")

        # Tokenize and embed (implementation depends on tokenizer)
        # This is a simplified version
        embedding = self.embedding_model(tf.constant([[ord(c) % 256 for c in text[:128]]]))
        return embedding.numpy().squeeze()

    def retrieve(
        self,
        query: str | np.ndarray | tf.Tensor,
        k: int | None = None,
    ) -> RetrievedContext:
        """Retrieve relevant documents for a query.

        Args:
            query: Query text, embedding vector, or tensor.
            k: Number of documents to retrieve (defaults to self.top_k).

        Returns:
            RetrievedContext with documents and scores.
        """
        k = k or self.top_k

        # Get query embedding
        if isinstance(query, str):
            if self.embedding_model is not None:
                query_emb = self._compute_embedding(query)
            else:
                # Fallback: random (for testing)
                query_emb = np.random.randn(self.embedding_dim)
        elif isinstance(query, tf.Tensor):
            query_emb = query.numpy()
        else:
            query_emb = query

        # Search index
        scores, indices = self.index.search(query_emb, k)

        # Gather documents
        documents = []
        chunk_ids = []

        if hasattr(self.index, "ids"):
            for idx in indices:
                if 0 <= idx < len(self.index.ids):
                    chunk_id = self.index.ids[idx]
                    chunk_content = self.doc_store.get_chunk(chunk_id)
                    if chunk_content:
                        documents.append(chunk_content)
                        chunk_ids.append(chunk_id)

        return RetrievedContext(
            documents=documents,
            scores=scores.tolist() if isinstance(scores, np.ndarray) else list(scores),
            ids=chunk_ids,
        )

    def save_index(self, path: str | Path) -> None:
        """Save the index to disk."""
        path = Path(path)
        self.index.save(str(path / "index"))

        # Save document store
        with open(path / "doc_store.json", "w") as f:
            json.dump(
                {
                    "documents": self.doc_store.documents,
                    "chunks": self.doc_store.chunks,
                    "chunk_to_doc": self.doc_store.chunk_to_doc,
                },
                f,
            )

    def load_index(self, path: str | Path) -> None:
        """Load the index from disk."""
        path = Path(path)
        self.index.load(str(path / "index"))

        # Load document store
        with open(path / "doc_store.json") as f:
            data = json.load(f)
            self.doc_store.documents = data["documents"]
            self.doc_store.chunks = data["chunks"]
            self.doc_store.chunk_to_doc = data["chunk_to_doc"]

    def call(
        self,
        query_embeddings: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass for RAG layer.

        Args:
            query_embeddings: Query embeddings [batch, embedding_dim].
            training: Whether in training mode.

        Returns:
            Retrieved context embeddings [batch, top_k, embedding_dim].
        """
        batch_size = tf.shape(query_embeddings)[0]

        # Retrieve for each query in the batch
        # Note: This is a simple implementation; production would use
        # batched retrieval
        contexts = []
        for i in range(batch_size):
            query = query_embeddings[i].numpy()
            result = self.retrieve(query, k=self.top_k)

            # Get embeddings for retrieved chunks
            chunk_embeddings = []
            for chunk_id in result.ids:
                if chunk_id in self._embeddings_cache:
                    chunk_embeddings.append(self._embeddings_cache[chunk_id])
                else:
                    chunk_embeddings.append(np.zeros(self.embedding_dim))

            # Pad to top_k
            while len(chunk_embeddings) < self.top_k:
                chunk_embeddings.append(np.zeros(self.embedding_dim))

            contexts.append(np.stack(chunk_embeddings[: self.top_k]))

        return tf.constant(np.stack(contexts), dtype=tf.float32)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "top_k": self.top_k,
                "index_backend": self.index_backend,
                "chunk_size": self.chunk_size,
            }
        )
        return config
