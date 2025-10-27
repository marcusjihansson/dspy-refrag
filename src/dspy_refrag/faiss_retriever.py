"""
FAISS-based retriever for REFRAG integration.

This retriever uses FAISS for efficient similarity search on vector embeddings.
Currently scaffolding - implement with real FAISS code for production use.
"""

from typing import Callable, List

import numpy as np

from .common import Passage
from .retriever import Retriever


class FAISSRetriever(Retriever):
    """
    Scaffolding for FAISS-based retriever.

    IMPLEMENTATION TEMPLATE:
    This class provides the interface structure for implementing a FAISS-based
    retriever. Users should replace the NotImplementedError sections with actual
    FAISS implementation code.

    Example Implementation Steps:
    1. Install FAISS: `pip install faiss-cpu` or `pip install faiss-gpu`
    2. Load your index: `self.index = faiss.read_index(index_path)`
    3. Load passage metadata mapping: `self.id_to_passage = {...}`
    4. Implement search logic in retrieve() method

    Key FAISS Operations:
    - faiss.IndexFlatIP: For inner product similarity
    - faiss.IndexFlatL2: For L2 distance
    - index.search(query_vec, k): Returns distances and indices
    """

    def __init__(self, index_path: str, embedder: Callable[[str], np.ndarray]):
        """
        Initialize FAISS retriever.

        Implementation template:
        ```python
        import faiss
        self.index = faiss.read_index(index_path)
        self.passages = self._load_passage_metadata(index_path)  # Your implementation
        self.embedder = embedder
        ```
        """
        # TODO: Load FAISS index from index_path
        # self.index = faiss.read_index(index_path)
        # self.passages = ...  # Load passages metadata
        self.embedder = embedder
        raise NotImplementedError(
            "FAISSRetriever is scaffolding for user implementation. "
            "See class docstring for implementation guidance."
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder(query)

    def retrieve(self, query: str, k: int = 3) -> List[Passage]:
        """
        Retrieve similar passages using FAISS index.

        Implementation template:
        ```python
        query_vec = self.embed_query(query).reshape(1, -1)
        distances, indices = self.index.search(query_vec, k)

        passages = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 indicates no match found
                passage_data = self.passages[idx]
                passages.append(Passage(
                    text=passage_data['text'],
                    vector=passage_data['vector'],
                    metadata={'score': float(distances[0][i]), **passage_data.get('metadata', {})}
                ))
        return passages
        ```
        """
        # TODO: Query FAISS index and return passages
        raise NotImplementedError(
            "Implement FAISS query logic. See method docstring for template."
        )
