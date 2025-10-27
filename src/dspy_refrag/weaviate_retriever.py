## File: src/dspy_refrag/weaviate_retriever.py

"""Weaviate-based retriever for REFRAG integration.

This retriever connects to a local Weaviate instance (run via Docker) and supports
vector retrieval for REFRAG workflows.
"""

import json
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.query import MetadataQuery


from .common import Passage, make_ollama_embedder


class WeaviateRetriever:
    def __init__(
        self,
        collection_name: str = "DSPyRefrag",
        embed_dim: Optional[int] = None,
        weaviate_url: str = "http://localhost:8080",
        embedder: Optional[Callable[[str], np.ndarray]] = None,
        ollama_endpoint: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.embed_dim: Optional[int] = embed_dim
        # Prefer provided embedder; otherwise, auto-configure Ollama if params given
        if embedder is not None:
            self._embedder = embedder
        elif ollama_endpoint or ollama_model:
            self._embedder = make_ollama_embedder(
                api_endpoint=ollama_endpoint or "http://localhost:11434",
                model=ollama_model or "nomic-embed-text:latest",
            )
        else:
            self._embedder = None
        # Accept either a host or a full http URL; normalize to host/port for connect_to_local
        host = weaviate_url
        port = 8080
        try:
            if isinstance(weaviate_url, str) and weaviate_url.startswith("http"):
                # Parse http://host:port
                import urllib.parse as _urlparse

                parsed = _urlparse.urlparse(weaviate_url)
                host = parsed.hostname or "localhost"
                port = parsed.port or 8080
            else:
                # If a hostname is provided, keep defaults
                host = weaviate_url or "localhost"
        except Exception:
            host = "localhost"
            port = 8080
        self.client = weaviate.connect_to_local(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the collection exists in Weaviate."""
        if not self.client.collections.exists(self.collection_name):
            self.client.collections.create(
                name=self.collection_name,
                properties=[
                    weaviate.classes.config.Property(
                        name="text", data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="metadata", data_type=weaviate.classes.config.DataType.TEXT
                    ),
                ],
                vectorizer_config=Configure.Vectorizer.none(),
            )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed the query using a provided embedder or a fallback."""
        if self._embedder is not None:
            vec = self._embedder(query)
            # set embed_dim if not set
            if self.embed_dim is None and isinstance(vec, np.ndarray):
                self.embed_dim = int(vec.shape[0])
            return vec.astype(np.float32)
        # Fallback: use random vector but require a known dimension
        if self.embed_dim is None:
            raise ValueError(
                "embed_dim is not set and no embedder provided; cannot embed query. "
                "Provide embedder=..., or set embed_dim, or add passages first to infer dimension."
            )
        v = np.random.randn(self.embed_dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def add_passages(self, passages: List[Passage]):
        """Add passages to the Weaviate collection and infer embed_dim if needed."""
        if not passages:
            return
        if self.embed_dim is None:
            first_vec = passages[0].vector
            if isinstance(first_vec, np.ndarray):
                self.embed_dim = int(first_vec.shape[0])
            else:
                self.embed_dim = int(len(first_vec))
        collection = self.client.collections.get(self.collection_name)
        with collection.batch.dynamic() as batch:
            for passage in passages:
                batch.add_object(
                    properties={
                        "text": passage.text,
                        "metadata": json.dumps(passage.metadata),
                    },
                    vector=passage.vector.flatten() if passage.vector.ndim > 1 else passage.vector.tolist(),
                )

    def retrieve(self, query: str, k: int = 3) -> List[Passage]:
        """Retrieve top-k passages with vectors for the query."""
        collection = self.client.collections.get(self.collection_name)
        qv = self.embed_query(query)

        # Ensure vector is 1D
        if qv.ndim > 1:
            qv = qv.flatten()
        response = collection.query.near_vector(
            near_vector=qv.tolist(),
            limit=k,
            return_metadata=MetadataQuery(distance=True),
            include_vector=True,
        )

        results = []
        for obj in response.objects:
            # Parse metadata back from JSON string
            metadata_str = (
                str(obj.properties["metadata"]) if obj.properties["metadata"] else "{}"
            )
            metadata = json.loads(metadata_str)
            results.append(
                Passage(
                    text=str(obj.properties["text"]),
                    vector=np.array(obj.vector["default"]),
                    metadata=metadata,
                )
            )

        return results

    def close(self):
        """Close the Weaviate client."""
        self.client.close()
