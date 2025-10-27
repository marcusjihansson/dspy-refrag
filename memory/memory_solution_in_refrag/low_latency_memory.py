import json
from datetime import datetime
from typing import Dict, List, Optional

import dspy

from dspy_refrag.retriever import SimpleRetriever
from .memory_store import MemoryStore, InMemoryStore, MemoryRecord


class QuickFactsRetrieval(dspy.Module):
    """C) Low latency QnA for quick retrieval"""

    def __init__(self, memory: Optional[MemoryStore] = None):
        super().__init__()
        self.memory = memory or InMemoryStore()
        self.retrieve = SimpleRetriever()  # Core dspy_refrag retriever
        self.generate = dspy.Predict("context, question -> answer")

    def forward(self, question: str, use_cache: bool = True):
        # First, check if we have a cached answer
        if use_cache:
            cached = self.memory.search(session_id="default", query=question, k=1, type_filter="qna")
            if cached and self._is_similar_enough(question, cached[0].question or ""):
                return dspy.Prediction(answer=cached[0].answer or "", from_cache=True)

        # If not cached, retrieve from documents
        passages = self.retrieve.retrieve(question, k=3)
        context = [p.text for p in passages]
        try:
            result = self.generate(context=context, question=question)
        except Exception:
            # Fallback when no LM is configured
            heuristic = self._heuristic_answer(context, question)
            result = dspy.Prediction(answer=heuristic)

        # Cache the result
        self.memory.save(
            MemoryRecord(
                type="qna",
                session_id="default",
                question=question,
                answer=result.answer,
                score=1.0,
                meta={"context": context},
            )
        )

        return dspy.Prediction(answer=result.answer, from_cache=False)

    def _is_similar_enough(self, q1: str, q2: str, threshold: float = 0.95) -> bool:
        """Simple similarity check - you can improve this"""
        return q1.lower().strip() == q2.lower().strip()

    def _heuristic_answer(self, context: List[str], question: str) -> str:
        if context:
            return f"[heuristic] Based on context: {context[0]}"
        return "[heuristic] No LM configured; no context available."
