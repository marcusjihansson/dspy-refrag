## File: examples/quickstart.py

"""Quickstart: REFRAG with Ollama embeddings and optional OpenRouter LM

This example shows how to:
- Load PDFs from the local data/ directory
- Chunk and embed them with Ollama (nomic-embed-text:latest)
- Build an in-memory retriever (SimpleRetriever)
- Run REFRAG to retrieve and (optionally) answer using OpenRouter via an OpenAI-compatible API

Requirements:
- Ollama running locally with the model pulled:
  ollama pull nomic-embed-text:latest
  ollama serve
- Python packages: requests, numpy, pypdf (optional for PDF extraction)

Run:
  python examples/quickstart.py "your query here"

Environment variables for OpenRouter (optional, only if you want LM answering):
- OPENROUTER_API_KEY (required if using OpenRouter LM)
- OPENROUTER_MODEL (optional, defaults to "openrouter/auto")

This script will map OpenRouter to OpenAI-compatible environment variables that dspy.LM can use:
- OPENAI_API_KEY     <- OPENROUTER_API_KEY
- OPENAI_API_BASE    <- https://openrouter.ai/api/v1
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np

from dspy_refrag import REFRAGModule, SimpleRetriever
from dspy_refrag.common import Passage, make_ollama_embedder

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using pypdf if available, otherwise raise a helpful error."""
    try:
        from pypdf import PdfReader  # lightweight and commonly available
    except Exception:  # pragma: no cover
        raise RuntimeError(
            "pypdf is required to extract text from PDFs. Install with: pip install pypdf"
        )

    text_parts: List[str] = []
    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    """Naive char-based chunking with overlap. Works well enough for a quickstart."""
    if max_chars <= 0:
        return [text]
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        chunk = text[i:j]
        if chunk.strip():
            chunks.append(chunk.strip())
        i = j - overlap if j < n else j
        if i < 0:
            i = 0
    return chunks


def build_corpus_from_data(embedder) -> List[Passage]:
    """Read PDFs from data/, chunk, embed with Ollama, and build a corpus of Passages."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    pdfs = sorted([p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {DATA_DIR}. Add PDFs to run this example."
        )

    corpus: List[Passage] = []
    for pdf_path in pdfs:
        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"Warning: failed to extract {pdf_path.name}: {e}")
            continue

        # Simple filtering to avoid embedding empty docs
        if not text or len(text.strip()) < 20:
            print(f"Warning: no usable text in {pdf_path.name}")
            continue

        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            vec = embedder(chunk)
            if vec.ndim != 1:
                raise ValueError("Embedder must return a 1-D vector")
            # Normalize for dot-product retrieval
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            corpus.append(
                Passage(
                    text=chunk,
                    vector=vec.astype(np.float32),
                    metadata={
                        "source": str(pdf_path.name),
                        "chunk_id": idx,
                        "char_len": len(chunk),
                        "text": chunk,  # include text so REFRAG prompt building works if LM is used
                    },
                )
            )
    if not corpus:
        raise RuntimeError(
            "Corpus ended up empty. Check PDF extraction and Ollama embedding."
        )
    return corpus


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


def main():
    query = "What is DSPy and how does it relate to programmatic prompting?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    # 1) Create Ollama embedder (requires Ollama running locally)
    embedder = make_ollama_embedder(model="nomic-embed-text:latest")

    # 2) Build an in-memory corpus from PDFs in data/
    print(f"Building corpus from: {DATA_DIR}")
    corpus = build_corpus_from_data(embedder)
    print(
        f"Built {len(corpus)} chunks from {len({p.metadata['source'] for p in corpus})} PDFs"
    )

    # 3) Create a SimpleRetriever over our corpus wired to the same embedder used for the query
    retriever = SimpleRetriever(embedder=embedder, corpus=corpus)

    # 4) Optionally configure OpenRouter for answering; otherwise run retrieval-only
    lm_model = maybe_configure_openrouter_env()

    if lm_model:
        print(f"Using OpenRouter LM via OpenAI-compatible API: model={lm_model}")
        module = REFRAGModule(
            retriever=retriever, lm=None, k=5, budget=3, lm_model=lm_model
        )
    else:
        print(
            "No OPENROUTER_API_KEY found. Running in retrieval-only mode (no LM call)."
        )
        module = REFRAGModule(retriever=retriever, lm=None, k=5, budget=3)

    # 5) Run REFRAG
    ctx = module.forward(query)

    # 6) Display results
    print("\n=== REFRAG Results ===")
    print("Query:", ctx.query)
    print("Retrieved chunks:", len(ctx.chunk_vectors))
    # Show a preview of top-3 chunks' metadata and text starts
    for i, meta in enumerate(ctx.chunk_metadata[:3]):
        text_preview = (meta.get("text") or "").strip()
        if not text_preview and i < len(corpus):
            text_preview = corpus[i].text[:200].replace("\n", " ") + (
                "..." if len(corpus[i].text) > 200 else ""
            )
        print(
            f"- Passage {i}: source={meta.get('source')} chunk_id={meta.get('chunk_id')} selected={meta.get('selected')}\n  {text_preview}"
        )

    print("\nAnswer:", ctx.answer)


if __name__ == "__main__":
    main()
