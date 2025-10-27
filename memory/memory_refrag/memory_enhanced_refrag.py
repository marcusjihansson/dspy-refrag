import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import dspy

from dspy_refrag.retriever import SimpleRetriever
from memory.memory_solution_in_refrag.low_latency_memory import \
    QuickFactsRetrieval
from memory.memory_solution_in_refrag.memory_store import (InMemoryStore,
                                                           MemoryRecord,
                                                           MemoryStore)


class MemoryEnhancedREFRAG(dspy.Module):
    """REFRAG implementation with a pluggable MemoryStore (DB-agnostic)."""

    def __init__(
        self,
        memory: Optional[MemoryStore] = None,
        retriever: Optional[Any] = None,
        session_id: str = "default",
    ):
        super().__init__()
        self.memory = memory or InMemoryStore()
        self.session_id = session_id

        # REFRAG components: prefer provided retriever, else SimpleRetriever
        self.retrieve = retriever or SimpleRetriever()
        self.generate_query = dspy.ChainOfThought("question, context -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

        # Quick facts for common queries (low-latency memory)
        self.quick_facts = QuickFactsRetrieval(self.memory)

    def forward(self, question: str, max_iterations: int = 3):
        # Try quick retrieval first
        quick_result = self.quick_facts(question, use_cache=True)
        if quick_result.from_cache:
            return quick_result

        # Get conversation context from memory
        recent_context = self._get_conversation_context()

        # Iterative retrieval and refinement
        all_context = []
        for i in range(max_iterations):
            # Generate refined search query
            try:
                query_result = self.generate_query(
                    question=question, context="\n".join(all_context + [recent_context])
                )
                search_query = getattr(query_result, "search_query", question)
            except Exception:
                # Fallback if no LM configured
                search_query = question

            # Retrieve documents
            passages = self.retrieve.retrieve(search_query, k=5)
            all_context.extend([p.text for p in passages])

            # Check if we have enough information
            if self._has_sufficient_info(passages, question):
                break

        # Generate final answer
        final_context = "\n\n".join(all_context + [recent_context])
        try:
            result = self.generate_answer(context=final_context, question=question)
        except Exception:
            # Fallback if no LM configured
            result = dspy.Prediction(
                answer=f"[heuristic] Context suggests: {final_context[:120]}"
            )

        # Store in memory for future use
        # Store in memory for future use (A: reasoning QnA)
        rec = MemoryRecord(
            type="qna",
            session_id=self.session_id,
            question=question,
            answer=result.answer,
            meta={
                "reasoning": getattr(result, "reasoning", ""),
                "context": final_context,
            },
        )
        self.memory.save(rec)

        return result

    def _get_conversation_context(self) -> str:
        """Get recent conversation history"""
        recent = self.memory.get_recent_qna(session_id=self.session_id, limit=5)
        if recent:
            return "Recent conversation:\n" + "\n".join(
                [f"Q: {item.question}\nA: {item.answer}" for item in reversed(recent)]
            )
        return ""

    def _has_sufficient_info(self, passages: List[Any], question: str) -> bool:
        """Heuristic to determine if we have enough information"""
        # You can implement more sophisticated logic here
        total_length = sum(len(getattr(p, "text", str(p))) for p in passages)
        return total_length > 500  # Simple threshold


if __name__ == "__main__":
    # Minimal example to show memory usage with REFRAG
    try:
        import os

        import dspy

        # Configure DSPy with OpenRouter free model if available
        if os.getenv("OPENROUTER_API_KEY"):
            dspy.configure(lm=dspy.LM("openrouter/openrouter/andromeda:free"))
        else:
            # Heuristic mode: DSPy modules will fall back to simple predictions
            pass
    except Exception:
        pass

    from dspy_refrag.common import Passage

    # Simple in-memory setup
    mem = InMemoryStore()
    retriever = SimpleRetriever()

    # Seed the retriever with a couple of passages (as an example)
    retriever.passages = [
        Passage(
            text="Doc A about cats. Cats are friendly.", vector=None, metadata=None
        ),
        Passage(text="Doc B about dogs. Dogs are loyal.", vector=None, metadata=None),
    ]

    system = MemoryEnhancedREFRAG(memory=mem, retriever=retriever, session_id="demo")
    q1 = "What animal is loyal?"
    out1 = system(question=q1)
    print("Q1:", q1)
    print("A1:", out1.answer)

    # Ask again to exercise low-latency memory usage
    out2 = system(question=q1)
    print("A1 (cached or improved):", out2.answer)
