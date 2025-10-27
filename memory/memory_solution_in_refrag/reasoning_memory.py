import json
from datetime import datetime
from typing import Dict, List, Optional

import dspy

from .memory_store import MemoryStore, InMemoryStore, MemoryRecord


class ReasoningWithMemory(dspy.Module):
    """A) Store previous QnA for further reasoning paths"""

    def __init__(self, memory: Optional[MemoryStore] = None, session_id: str = "default"): 
        super().__init__()
        self.memory = memory or InMemoryStore()
        self.session_id = session_id
        self.cot = dspy.ChainOfThought("context, question -> reasoning, answer")

    def forward(self, question: str, use_memory: bool = True):
        context = ""

        if use_memory:
            # Retrieve similar past interactions
            similar = self.memory.search(
                session_id=self.session_id,
                query=question,
                k=3,
                type_filter="qna",
            )

            if similar:
                context = "Relevant past interactions:\n" + "\n".join(
                    [
                        f"Q: {item.question}\nA: {item.answer}"
                        for item in similar
                    ]
                )

        # Generate response
        try:
            result = self.cot(context=context, question=question)
        except Exception:
            result = dspy.Prediction(reasoning="[heuristic]", answer="[heuristic] No LM configured.")

        # Store in memory
        self.memory.save(
            MemoryRecord(
                type="qna",
                session_id=self.session_id,
                question=question,
                answer=result.answer,
                meta={"reasoning": result.reasoning, "context": context},
            )
        )

        return result
