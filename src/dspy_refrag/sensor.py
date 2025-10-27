## File: src/dspy_refrag/sensor.py

"""
Sensor policy for selecting relevant chunks in DSPy REFRAG.

Provides heuristic and learned selection modes, with scaffolding for advanced algorithms.
"""

from typing import List, Optional

import numpy as np


class Sensor:
    """
    Sensor for selecting top-k chunks based on query similarity.

    Modes:
    - heuristic: Dot product similarity.
    - learned: Simple ML-based selection (scaffolding).
    """

    def __init__(
        self,
        mode: str = "heuristic",
        threshold: Optional[float] = None,
        learned_weights: Optional[np.ndarray] = None,
    ):
        assert mode in ("heuristic", "learned")
        self.mode = mode
        self.threshold = threshold  # Optional similarity threshold
        self.learned_weights = learned_weights  # For learned mode

    def select(
        self, query_vec: np.ndarray, chunk_vecs: List[np.ndarray], budget: int = 2
    ) -> List[int]:
        """
        Select top-k chunk indices based on mode.

        Args:
            query_vec: Query embedding vector.
            chunk_vecs: List of chunk embedding vectors.
            budget: Number of chunks to select.

        Returns:
            List of selected indices.
        """
        if self.mode == "heuristic":
            scores = [float(np.dot(query_vec, cv)) for cv in chunk_vecs]
            if self.threshold is not None:
                # Filter by threshold
                filtered = [(i, s) for i, s in enumerate(scores) if s >= self.threshold]
                ranked = sorted(filtered, key=lambda x: x[1], reverse=True)
                return [i for i, _ in ranked[:budget]]
            else:
                ranked = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )
                return ranked[:budget]
        elif self.mode == "learned":
            # Simple learned policy scaffolding (e.g., weighted sum)
            if self.learned_weights is None:
                # Fallback to heuristic if no weights
                return self.select(query_vec, chunk_vecs, budget)  # Recursive fallback
            scores = []
            for cv in chunk_vecs:
                # Example: weighted combination (scaffolding - replace with real model)
                combined = np.dot(self.learned_weights, np.concatenate([query_vec, cv]))
                scores.append(float(combined))
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return ranked[:budget]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# Example usage
def example_sensor_usage():
    """Example of using Sensor."""
    query_vec = np.array([0.1, 0.2, 0.3])
    chunk_vecs = [np.array([0.1, 0.1, 0.1]), np.array([0.2, 0.2, 0.2])]
    sensor = Sensor(mode="heuristic", threshold=0.5)
    selected = sensor.select(query_vec, chunk_vecs, budget=1)
    return selected
