"""
Benchmarking-specific REFRAG module that uses RequestLM for consistent metrics tracking.
This wraps the core REFRAG implementation while providing proper token usage and metrics collection.
"""

import os
from typing import Any, Dict, List, Optional

import dspy
import numpy as np
from request_lm import RequestLM as OpenRouterRequestLM, RequestLMResponse
from request_lm_github import RequestLM as GitHubRequestLM

# Import core REFRAG components
from dspy_refrag import SimpleRetriever
from dspy_refrag.sensor import Sensor
from dspy_refrag.serializer_vector import VectorAwareSerializer

# DSPy base classes
ModuleBase = dspy.Module


class REFRAGContext:
    """Context class for REFRAG results with benchmarking metadata."""

    def __init__(
        self,
        query: str,
        chunk_vectors: Optional[List[List[float]]] = None,
        chunk_metadata: Optional[List[dict]] = None,
        answer: Optional[str] = None,
        token_usage: int = 0,
        prompt_chars: int = 0,
        context_chars: int = 0,
    ):
        self.query = query
        self.chunk_vectors = chunk_vectors
        self.chunk_metadata = chunk_metadata
        self.answer = answer
        self.token_usage = token_usage
        self.prompt_chars = prompt_chars
        self.context_chars = context_chars


class REFRAGBenchmarkModule(ModuleBase):
    """REFRAG module optimized for benchmarking with RequestLM integration."""

    def __init__(
        self,
        retriever: Optional[SimpleRetriever] = None,
        lm: Optional[Any] = None,
        sensor: Optional[Sensor] = None,
        k: int = 5,
        budget: int = 2,
        lm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.retriever = retriever or SimpleRetriever()
        self.sensor = sensor or Sensor()
        self.k = k
        self.budget = budget
        self.serializer = VectorAwareSerializer()

        # Initialize LM with RequestLM if no custom lm provided
        if lm is None:
            # Choose LM based on available environment variables
            if os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_BASE_URL"):
                self.lm = GitHubRequestLM(
                    model=lm_model,
                    api_key=api_key,
                    base_url=os.getenv("GITHUB_BASE_URL"),
                )
            elif os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_BASE_URL"):
                self.lm = OpenRouterRequestLM(
                    model=lm_model,
                    api_key=api_key,
                    base_url=os.getenv("OPENROUTER_BASE_URL"),
                )
            else:
                self.lm = None
        else:
            self.lm = lm

    def forward(self, query: str) -> REFRAGContext:
        """Forward pass with detailed metrics tracking."""
        # 1) Retrieve initial passages
        passages = self.retriever.retrieve(query, k=self.k)
        vectors = [p.vector.tolist() for p in passages]
        metadata = [p.metadata for p in passages]

        # 2) Use sensor to decide which passages to expand/select
        qv = self.retriever.embed_query(query)
        selected_idxs = self.sensor.select(
            qv, [np.asarray(v) for v in vectors], budget=self.budget
        )

        # 3) Attach selection info to metadata
        metadata_with_selection = []
        selected_count = 0
        for i, m in enumerate(metadata):
            meta = dict(m)
            is_selected = i in selected_idxs
            meta["selected"] = is_selected
            if is_selected:
                selected_count += 1
            metadata_with_selection.append(meta)

        # 4) Build context string from selected passages
        context_parts = []
        for i, meta in enumerate(metadata_with_selection):
            text = meta.get("text", "")
            selected = meta.get("selected", False)
            if selected:  # Only include selected passages in context
                context_parts.append(f"Passage {i}: {text}")

        context_str = "\n".join(context_parts)
        context_chars = len(context_str)

        # 5) Build final prompt
        prompt = f"Use the following context to answer the query.\n\nContext:\n{context_str}\n\nQuery: {query}\n\nAnswer:"
        prompt_chars = len(prompt)

        # 6) Call LM if available
        answer = None
        token_usage = 0

        if self.lm is not None:
            try:
                resp = self.lm(prompt)
                if isinstance(resp, RequestLMResponse):
                    answer = resp.text
                    token_usage = resp.usage.get("total_tokens", 0) or 0
                else:
                    # Handle other response types
                    answer = (
                        resp
                        if isinstance(resp, str)
                        else getattr(resp, "text", str(resp))
                    )
                    if hasattr(resp, "usage"):
                        if isinstance(resp.usage, dict):
                            token_usage = resp.usage.get("total_tokens", 0)
                        elif hasattr(resp.usage, "total_tokens"):
                            token_usage = resp.usage.total_tokens
            except Exception as e:
                answer = f"Error calling LM: {str(e)}"
                token_usage = 0

        return REFRAGContext(
            query=query,
            chunk_vectors=vectors,
            chunk_metadata=metadata_with_selection,
            answer=answer,
            token_usage=token_usage,
            prompt_chars=prompt_chars,
            context_chars=context_chars,
        )

    def __call__(self, query: str):
        """Make the module callable and return dict format for benchmark compatibility."""
        ctx = self.forward(query)

        # Calculate metrics
        retrieved = len(ctx.chunk_metadata) if ctx.chunk_metadata else 0
        selected = sum(
            1 for m in (ctx.chunk_metadata or []) if m.get("selected", False)
        )

        return {
            "answer": ctx.answer,
            "token_usage": ctx.token_usage,
            "meta": {
                "retrieved": retrieved,
                "selected": selected,
                "context_chars": ctx.context_chars,
                "prompt_chars": ctx.prompt_chars,
            },
        }
