## File: tests/test_refrag_basic.py
from dspy_refrag import REFRAGModule, SimpleRetriever


def test_refrag_forward_no_lm():
    r = SimpleRetriever()
    module = REFRAGModule(retriever=r, lm=None, k=2, budget=1)
    ctx = module.forward("Tell me about dogs")
    assert ctx.query == "Tell me about dogs"
    assert ctx.chunk_vectors is not None
    assert len(ctx.chunk_vectors) == 2
    assert isinstance(ctx.chunk_metadata, list)
