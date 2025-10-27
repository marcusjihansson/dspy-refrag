import os
from typing import Any, Optional

import dspy
import numpy as np

from request_lm import RequestLM as OpenRouterRequestLM
from request_lm_github import RequestLM as GitHubRequestLM, RequestLMResponse

from dspy_refrag import SimpleRetriever

# DSPy base classes
ModuleBase = dspy.Module
LMBase = dspy.LM


class SimpleRAGModule(ModuleBase):
    def __init__(
        self,
        retriever: Optional[SimpleRetriever] = None,
        lm: Optional[Any] = None,
        k: int = 5,
        lm_model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.retriever = retriever or SimpleRetriever()
        self.k = k

        # Use provided model or default

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
                self.lm = None  # No LM if no key
        else:
            self.lm = lm

    def forward(self, query: str):
        # Retrieve passages
        passages = self.retriever.retrieve(query, k=self.k)
        context_str = "\n".join([p.text for p in passages])

        # Build prompt
        prompt = f"Use the following context to answer the query.\n\nContext:\n{context_str}\n\nQuery: {query}\n\nAnswer:"

        # Precompute meta info about the constructed prompt/context
        meta = {
            "retrieved": len(passages),
            "context_chars": len(context_str),
            "prompt_chars": len(prompt),
        }

        # Call LM if available
        if self.lm is None:
            return {"answer": None, "token_usage": 0, "meta": meta}

        try:
            # Call LM
            resp = self.lm(prompt)
            # If using RequestLM, it returns RequestLMResponse
            if isinstance(resp, RequestLMResponse):
                answer = resp.text
            else:
                # Assume resp is a string or has .text attribute
                answer = (
                    resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
                )

            # Debug: Print response structure to understand format
            if os.getenv("DEBUG_TOKEN_USAGE"):
                print(f"Response type: {type(resp)}")
                print(f"Response attributes: {dir(resp)}")
                if hasattr(resp, "usage"):
                    print(f"Usage type: {type(resp.usage)}")
                    print(f"Usage: {resp.usage}")

            # Extract token usage from OpenRouter/OpenAI response format
            token_usage = 0
            if isinstance(resp, RequestLMResponse):
                token_usage = resp.usage.get("total_tokens", 0) or 0
            elif hasattr(resp, "usage") and hasattr(resp.usage, "total_tokens"):
                # Direct access to usage.total_tokens (OpenRouter format)
                token_usage = resp.usage.total_tokens
            elif hasattr(resp, "usage") and isinstance(resp.usage, dict):
                # Usage as dict format
                token_usage = resp.usage.get("total_tokens", 0)
            elif hasattr(resp, "token_usage"):
                # Legacy format
                usage = getattr(resp, "token_usage")
                if isinstance(usage, dict):
                    token_usage = usage.get("total_tokens", 0)
                else:
                    token_usage = usage if isinstance(usage, (int, float)) else 0
            elif hasattr(resp, "_response") and hasattr(resp._response, "usage"):
                # Some DSPy responses might wrap the original response
                usage = resp._response.usage
                if hasattr(usage, "total_tokens"):
                    token_usage = usage.total_tokens
                elif isinstance(usage, dict):
                    token_usage = usage.get("total_tokens", 0)
        except Exception as e:
            answer = f"Error calling LM: {str(e)}"
            token_usage = 0

        # Add answer length into meta
        meta["answer_chars"] = len(answer) if isinstance(answer, str) else 0

        return {"answer": answer, "token_usage": token_usage, "meta": meta}


def _build_retriever_from_data():
    """Build a SimpleRetriever using Ollama embeddings from the data directory."""
    from pathlib import Path

    from dspy_refrag.common import make_ollama_embedder
    from dspy_refrag.data_ingest import build_corpus_from_data

    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
    embedder = make_ollama_embedder(api_endpoint=ollama_base, model=ollama_model)

    data_dir = Path(__file__).resolve().parents[1] / "data"
    print(f"Building corpus from: {data_dir}")
    corpus = build_corpus_from_data(embedder, data_dir)
    print(
        f"Built {len(corpus)} chunks from {len({p.metadata['source'] for p in corpus})} PDFs"
    )
    return SimpleRetriever(embedder=embedder, corpus=corpus)


def benchmark_once(query: str):
    """Run a single benchmark with the given query."""
    retriever = _build_retriever_from_data()
    rag_module = SimpleRAGModule(retriever=retriever, k=5)
    module = rag_module
    import time

    start = time.time()
    out = module.forward(query)
    elapsed = time.time() - start
    import json

    print(json.dumps({"query": query, "elapsed": elapsed, **out}, indent=2))


if __name__ == "__main__":
    # Example usage over DSPy-related queries
    queries = [
        "What is DSPy?",
        "Explain the core idea behind programming, not prompting.",
        "Name some typical components in a DSPy workflow.",
    ]
    for q in queries:
        benchmark_once(q)
