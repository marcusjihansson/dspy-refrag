import os
from typing import Any, Dict, List, Optional

import dspy
import numpy as np

from .retriever import SimpleRetriever
from .sensor import Sensor
from .serializer_vector import VectorAwareSerializer

# DSPy base classes
ModuleBase = dspy.Module
SignatureBase = dspy.Signature
LMBase = dspy.LM


def maybe_configure_openrouter_env() -> str | None:
    """If OPENROUTER_API_KEY is present, set OpenAI-compatible env vars and return model name."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        return None
    os.environ.setdefault("OPENAI_API_KEY", openrouter_key)
    os.environ.setdefault("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    # Let user override the model; default to a reasonable OpenRouter alias
    model = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
    return model


class REFRAGContext:
    # Context class for REFRAG results
    def __init__(
        self,
        query: str,
        chunk_vectors: Optional[List[List[float]]] = None,
        chunk_metadata: Optional[List[dict]] = None,
        answer: Optional[str] = None,
    ):
        self.query = query
        self.chunk_vectors = chunk_vectors
        self.chunk_metadata = chunk_metadata
        self.answer = answer


class REFRAGModule(ModuleBase):
    def __init__(
        self,
        retriever: Optional[SimpleRetriever] = None,
        lm: Optional[Any] = None,
        sensor: Optional[Sensor] = None,
        k: int = 5,
        budget: int = 2,
        lm_model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.retriever = retriever or SimpleRetriever()
        self.sensor = sensor or Sensor()
        self.k = k
        self.budget = budget
        self.serializer = VectorAwareSerializer()

        # Configure OpenRouter if available
        openrouter_model = maybe_configure_openrouter_env()
        if openrouter_model:
            lm_model = openrouter_model

        # Initialize LM with dspy.LM if no custom lm provided
        if lm is None:
            # Securely get API key from environment if not provided
            key = api_key or os.getenv("OPENAI_API_KEY")
            if key:
                self.lm = LMBase(model=lm_model, api_key=key)
            else:
                self.lm = None  # No LM if no key
        else:
            self.lm = lm

    def forward(self, query: str) -> REFRAGContext:
        # 1) retrieve
        passages = self.retriever.retrieve(query, k=self.k)
        vectors = [p.vector.tolist() for p in passages]
        metadata = [p.metadata for p in passages]

        # 2) decide expansion
        qv = self.retriever.embed_query(query)
        selected_idxs = self.sensor.select(
            qv, [np.asarray(v) for v in vectors], budget=self.budget
        )

        # 3) attach selection info
        metadata_with_selection = []
        for i, m in enumerate(metadata):
            meta = dict(m)
            meta.setdefault("selected", i in selected_idxs)
            metadata_with_selection.append(meta)

        # 4) Call LM if available, else return context without answer
        if self.lm is None:
            return REFRAGContext(
                query=query,
                chunk_vectors=vectors,
                chunk_metadata=metadata_with_selection,
                answer=None,
            )

        # Build prompt including context from vectors and metadata
        context_str = "\n".join(
            [
                f"Passage {i}: {p['text']} (selected: {p.get('selected', False)})"
                for i, p in enumerate(metadata_with_selection)
            ]
        )
        prompt = f"Use the following context to answer the query.\n\nContext:\n{context_str}\n\nQuery: {query}\n\nAnswer:"

        try:
            # Call LM with prompt (dspy.LM style)
            resp = self.lm(prompt)
            # Assume resp is a string or has .text attribute
            answer = resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
        except Exception as e:
            # Handle LM failures gracefully
            answer = f"Error calling LM: {str(e)}"

        return REFRAGContext(
            query=query,
            chunk_vectors=vectors,
            chunk_metadata=metadata_with_selection,
            answer=answer,
        )
