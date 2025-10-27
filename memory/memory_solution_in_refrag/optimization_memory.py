import json
from datetime import datetime
from typing import Dict, List, Optional

import dspy
import numpy as np

from dspy_refrag.common import Passage
from .memory_store import MemoryStore, InMemoryStore, MemoryRecord


class OptimizationMemory(dspy.Module):
    """B) Store QnA for optimization (MIPRO, etc.)"""

    def __init__(self, memory: Optional[MemoryStore] = None):
        super().__init__()
        self.memory = memory or InMemoryStore()
        self.predictor = dspy.Predict("question -> answer")

    def forward(self, question: str):
        result = self.predictor(question=question)
        return result

    def store_for_optimization(
        self,
        question: str,
        answer: str,
        ground_truth: Optional[str] = None,
        score: Optional[float] = None,
    ):
        """Store high-quality QnA pairs for training optimizers"""
        metadata = {}
        if ground_truth:
            metadata["ground_truth"] = ground_truth

        self.memory.save(
            MemoryRecord(
                type="optimization",
                session_id="optimization_dataset",
                question=question,
                answer=answer,
                score=score,
                meta=metadata,
            )
        )

    def get_training_examples(self, min_score: float = 0.7, limit: int = 100):
        """Retrieve high-quality examples for MIPRO optimization"""
        # Get all optimization examples with good scores
        examples = (
            self.memory.client.query.get(
                self.memory.collection_name,
                ["question", "answer", "reasoning", "metadata", "score", "vector"],
            )
            .with_where(
                {
                    "operator": "And",
                    "operands": [
                        {
                            "path": ["memory_type"],
                            "operator": "Equal",
                            "valueString": "optimization",
                        },
                        {
                            "path": ["score"],
                            "operator": "GreaterThan",
                            "valueNumber": min_score,
                        },
                    ],
                }
            )
            .with_limit(limit)
            .do()
        )

        if "data" in examples and "Get" in examples["data"]:
            return [
                Passage(
                    text=item["question"],
                    vector=np.array(item["vector"]),
                    metadata={k: v for k, v in item.items() if k not in ["question", "vector"]},
                )
                for item in examples["data"]["Get"][self.memory.collection_name]
            ]
        return []
