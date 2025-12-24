# highnoon/retrieval/__init__.py
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

"""HighNoon Retrieval Module - RAG Integration.

This module provides Retrieval-Augmented Generation infrastructure:
- RAGModule: Retrieval layer for in-context document injection
- EmbeddingIndex: FAISS/Annoy-compatible vector index wrapper
- DocumentStore: Simple document storage and retrieval
"""

from highnoon.retrieval.rag_module import DocumentStore, EmbeddingIndex, RAGModule, RetrievedContext

__all__ = [
    "RAGModule",
    "EmbeddingIndex",
    "DocumentStore",
    "RetrievedContext",
]
