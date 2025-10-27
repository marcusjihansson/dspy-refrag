"""
Common utilities and data structures shared across dspy_refrag modules.

This module contains shared items like Passage dataclass and embedder factories
to avoid duplication across retriever implementations.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Passage:
    """Represents a document passage with text, vector embedding, and metadata."""

    text: str
    vector: np.ndarray
    metadata: dict


def make_ollama_embedder(
    api_endpoint: str = "http://localhost:11434",
    model: str = "nomic-embed-text:latest",
    normalize: bool = True,
) -> Callable[[str], np.ndarray]:
    """
    Factory for Ollama-based embedder.

    Args:
        api_endpoint: Ollama API endpoint.
        model: Embedding model name.
        normalize: Whether to normalize vectors.

    Returns:
        Function that embeds text to vector.
    """
    import requests

    def _embed(text: str) -> np.ndarray:
        resp = requests.post(
            f"{api_endpoint.rstrip('/')}/api/embeddings",
            json={"model": model, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding")
        if embedding is None:
            raise ValueError(f"Unexpected response from Ollama: {data}")
        vec = np.array(embedding, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError("Unexpected embedding shape from Ollama")
        if normalize:
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
        return vec

    return _embed
