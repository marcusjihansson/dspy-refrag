from __future__ import annotations

from pathlib import Path
from typing import List, Callable

import numpy as np

from .common import Passage


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using pypdf.

    Raises a clear error if pypdf is not installed.
    """
    try:
        from pypdf import PdfReader
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pypdf is required to extract text from PDFs. Install with: pip install pypdf"
        ) from e

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
    """Naive char-based chunking with overlap."""
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


def build_corpus_from_data(
    embedder: Callable[[str], np.ndarray],
    data_dir: Path,
    max_chars: int = 1500,
    overlap: int = 200,
) -> List[Passage]:
    """Read PDFs from data_dir, chunk, embed with `embedder`, and build a corpus of Passages."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pdfs = sorted([p for p in data_dir.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {data_dir}. Add PDFs to run the benchmarks."
        )

    corpus: List[Passage] = []
    for pdf_path in pdfs:
        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception as e:
            # Skip files we can't parse but continue with others
            print(f"Warning: failed to extract {pdf_path.name}: {e}")
            continue

        if not text or len(text.strip()) < 20:
            print(f"Warning: no usable text in {pdf_path.name}")
            continue

        chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
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
                        "text": chunk,
                    },
                )
            )
    if not corpus:
        raise RuntimeError(
            "Corpus ended up empty. Check PDF extraction and embedding configuration."
        )
    return corpus
