## File: src/dspy_refrag/retriever.py

"""
Retriever base classes and simple implementation for DSPy REFRAG integration.

This module provides:
- An abstract base class `Retriever` for building custom retrievers.
- `SimpleRetriever` as a basic example with random embeddings (for testing).
- Shared utilities imported from common.py.

For production retrievers, see separate files:
- weaviate_retriever.py for Weaviate
- faiss_retriever.py for FAISS
- pinecone_retriever.py for Pinecone
- psql_retriever.py for PostgreSQL
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np

from .common import Passage


class Retriever(ABC):
    """
    Abstract base class for retrievers in DSPy REFRAG.

    Subclass this to implement custom retrievers (e.g., FAISS, Pinecone, Weaviate).
    """

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed the query into a vector.

        Args:
            query: The query string.

        Returns:
            Query vector as numpy array.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Passage]:
        """
        Retrieve top-k passages for the query.

        Args:
            query: The query string.
            k: Number of passages to retrieve.

        Returns:
            List of Passage objects.
        """
        pass

    def add_passages(self, passages: List[Passage]):
        """
        Optional: Add passages to the retriever's index.

        Args:
            passages: List of Passage objects to add.
        """
        raise NotImplementedError("Add passages not implemented for this retriever.")


from .common import make_ollama_embedder


class SimpleRetriever(Retriever):
    """
    Simple retriever example with random embeddings and a small corpus.

    For production, use a real vector DB like WeaviateRetriever.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        corpus: Optional[List[Passage]] = None,
        embedder: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.embed_dim = embed_dim
        self.embedder = embedder or (lambda text: self._rand_vec())
        if corpus is None:
            self._corpus = [
                Passage(
                    text="Doc A about cats.",
                    vector=self._rand_vec(),
                    metadata={"id": "a"},
                ),
                Passage(
                    text="Doc B about dogs.",
                    vector=self._rand_vec(),
                    metadata={"id": "b"},
                ),
                Passage(
                    text="Doc C about birds.",
                    vector=self._rand_vec(),
                    metadata={"id": "c"},
                ),
            ]
        else:
            self._corpus = corpus

    def _rand_vec(self) -> np.ndarray:
        """Generate a random normalized vector."""
        v = np.random.randn(self.embed_dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed query using provided embedder or random fallback."""
        return self.embedder(query)

    def retrieve(self, query: str, k: int = 3) -> List[Passage]:
        """Retrieve top-k passages based on dot product similarity."""
        qv = self.embed_query(query)
        scores = [float(np.dot(p.vector, qv)) for p in self._corpus]
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._corpus[i] for i in idxs]
