## File: tests/test_weaviate_retriever.py

"""Tests for WeaviateRetriever.

Note: Run Weaviate using docker-compose -f tests/docker-compose.yml up before running these tests.
"""

import numpy as np
import pytest
import requests

from dspy_refrag.weaviate_retriever import (Passage, WeaviateRetriever,
                                            make_ollama_embedder)


def _service_up(url: str) -> bool:
    try:
        requests.get(url, timeout=2)
        return True
    except Exception:
        return False


def test_weaviate_retriever_basic():
    """Test basic retrieval functionality using a simple embedder."""
    if not _service_up("http://localhost:8080/v1/.well-known/ready"):
        pytest.skip("Weaviate not running on :8080")

    # Use a simple random embedder for testing
    def simple_embedder(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)  # Deterministic for testing
        return np.random.randn(384).astype(np.float32)  # 384-dim vector

    retriever = WeaviateRetriever(collection_name="TestBasic", embedder=simple_embedder)

    # Add sample passages, embedding with simple embedder
    texts = [
        "Doc A about cats.",
        "Doc B about dogs.",
        "Doc C about birds.",
    ]
    passages = [
        Passage(text=t, vector=simple_embedder(t), metadata={"id": i})
        for i, t in enumerate(texts)
    ]
    retriever.add_passages(passages)

    # Retrieve for a query
    results = retriever.retrieve("animals", k=2)

    # Assertions
    assert len(results) == 2
    assert all(isinstance(res, Passage) for res in results)
    assert all(
        hasattr(res, "text") and hasattr(res, "vector") and hasattr(res, "metadata")
        for res in results
    )
    assert all(isinstance(res.vector, np.ndarray) for res in results)

    retriever.close()


def test_weaviate_retriever_empty_query():
    """Test retrieval with empty query using simple embedder."""
    if not _service_up("http://localhost:8080/v1/.well-known/ready"):
        pytest.skip("Weaviate not running on :8080")

    def simple_embedder(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(384).astype(np.float32)

    retriever = WeaviateRetriever(
        collection_name="TestEmptyQuery", embedder=simple_embedder
    )
    results = retriever.retrieve("", k=1)
    assert isinstance(results, list)
    retriever.close()


def test_weaviate_retriever_k_parameter():
    """Test retrieval with different k values using simple embedder."""
    if not _service_up("http://localhost:8080/v1/.well-known/ready"):
        pytest.skip("Weaviate not running on :8080")

    def simple_embedder(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(384).astype(np.float32)

    retriever = WeaviateRetriever(
        collection_name="TestKParameter", embedder=simple_embedder
    )
    passages = [
        Passage(text=f"Doc {i}", vector=simple_embedder(f"Doc {i}"), metadata={"id": i})
        for i in range(5)
    ]
    retriever.add_passages(passages)

    results = retriever.retrieve("test", k=3)
    assert len(results) == 3
    retriever.close()


def test_weaviate_retriever_initialization():
    """Test different initialization scenarios."""
    if not _service_up("http://localhost:8080/v1/.well-known/ready"):
        pytest.skip("Weaviate not running on :8080")

    def simple_embedder(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(384).astype(np.float32)

    # Test with custom collection name
    retriever = WeaviateRetriever(
        collection_name="TestInitCollection", embedder=simple_embedder
    )
    assert retriever.collection_name == "TestInitCollection"
    retriever.close()

    # Test with embed_dim
    retriever = WeaviateRetriever(
        collection_name="TestInitEmbedDim", embed_dim=256, embedder=simple_embedder
    )
    assert retriever.embed_dim == 256
    retriever.close()


def test_weaviate_retriever_no_passages():
    """Test retrieval when no passages have been added."""
    if not _service_up("http://localhost:8080/v1/.well-known/ready"):
        pytest.skip("Weaviate not running on :8080")

    def simple_embedder(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(384).astype(np.float32)

    retriever = WeaviateRetriever(
        collection_name="TestNoPassages", embedder=simple_embedder
    )
    results = retriever.retrieve("test", k=1)
    assert isinstance(results, list)
    # Should return empty list when no passages
    assert len(results) == 0
    retriever.close()


def test_weaviate_retriever_close():
    """Test that close method works correctly."""
    if not _service_up("http://localhost:8080/v1/.well-known/ready"):
        pytest.skip("Weaviate not running on :8080")

    def simple_embedder(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(384).astype(np.float32)

    retriever = WeaviateRetriever(embedder=simple_embedder)
    retriever.close()
    # Should not raise an error when closing again
    retriever.close()
