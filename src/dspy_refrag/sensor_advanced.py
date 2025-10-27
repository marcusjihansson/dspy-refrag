"""
Advanced Sensor policy for selecting relevant chunks in DSPy REFRAG.
Provides sophisticated selection strategies including diversity-aware selection,
uncertainty-based sampling, and ensemble methods.

File: src/dspy_refrag/advanced_sensor.py
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np


class SelectionStrategy(Enum):
    """Available selection strategies."""

    SIMILARITY = "similarity"  # Pure similarity-based
    MMR = "mmr"  # Maximal Marginal Relevance
    UNCERTAINTY = "uncertainty"  # Uncertainty-based sampling
    ENSEMBLE = "ensemble"  # Ensemble of strategies
    ADAPTIVE = "adaptive"  # Adaptive threshold-based


@dataclass
class SelectionConfig:
    """Configuration for advanced selection."""

    strategy: SelectionStrategy = SelectionStrategy.SIMILARITY
    diversity_lambda: float = 0.5  # For MMR: balance relevance vs diversity
    temperature: float = 1.0  # For uncertainty sampling
    ensemble_weights: Optional[List[float]] = None  # For ensemble
    adaptive_percentile: float = 0.75  # For adaptive thresholding
    min_score: Optional[float] = None  # Minimum similarity score


class AdvancedSensor:
    """
    Advanced sensor with multiple selection strategies.

    Strategies:
    - SIMILARITY: Standard cosine/dot product similarity
    - MMR: Maximal Marginal Relevance for diversity
    - UNCERTAINTY: Probability-based sampling from softmax scores
    - ENSEMBLE: Weighted combination of multiple strategies
    - ADAPTIVE: Dynamic threshold based on score distribution
    """

    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or SelectionConfig()
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.diversity_lambda < 0 or self.config.diversity_lambda > 1:
            raise ValueError("diversity_lambda must be in [0, 1]")
        if self.config.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.config.adaptive_percentile < 0 or self.config.adaptive_percentile > 1:
            raise ValueError("adaptive_percentile must be in [0, 1]")

    def select(
        self,
        query_vec: np.ndarray,
        chunk_vecs: List[np.ndarray],
        budget: int = 2,
        metadata: Optional[List[dict]] = None,
    ) -> List[int]:
        """
        Select top-k chunk indices using configured strategy.

        Args:
            query_vec: Query embedding vector (normalized recommended).
            chunk_vecs: List of chunk embedding vectors.
            budget: Number of chunks to select.
            metadata: Optional metadata for each chunk (for advanced strategies).

        Returns:
            List of selected chunk indices.
        """
        if not chunk_vecs:
            return []

        budget = min(budget, len(chunk_vecs))

        strategy_map = {
            SelectionStrategy.SIMILARITY: self._select_similarity,
            SelectionStrategy.MMR: self._select_mmr,
            SelectionStrategy.UNCERTAINTY: self._select_uncertainty,
            SelectionStrategy.ENSEMBLE: self._select_ensemble,
            SelectionStrategy.ADAPTIVE: self._select_adaptive,
        }

        return strategy_map[self.config.strategy](
            query_vec, chunk_vecs, budget, metadata
        )

    def _compute_similarities(
        self, query_vec: np.ndarray, chunk_vecs: List[np.ndarray]
    ) -> np.ndarray:
        """Compute similarity scores between query and chunks."""
        scores = np.array([np.dot(query_vec, cv) for cv in chunk_vecs])
        return scores

    def _select_similarity(
        self,
        query_vec: np.ndarray,
        chunk_vecs: List[np.ndarray],
        budget: int,
        metadata: Optional[List[dict]] = None,
    ) -> List[int]:
        """Standard similarity-based selection."""
        scores = self._compute_similarities(query_vec, chunk_vecs)

        # Apply minimum score threshold if configured
        if self.config.min_score is not None:
            valid_indices = np.where(scores >= self.config.min_score)[0]
            if len(valid_indices) == 0:
                return []
            scores = scores[valid_indices]
            ranked = np.argsort(scores)[::-1][:budget]
            return valid_indices[ranked].tolist()

        # Standard top-k selection
        ranked = np.argsort(scores)[::-1][:budget]
        return ranked.tolist()

    def _select_mmr(
        self,
        query_vec: np.ndarray,
        chunk_vecs: List[np.ndarray],
        budget: int,
        metadata: Optional[List[dict]] = None,
    ) -> List[int]:
        """
        Maximal Marginal Relevance selection for diversity.
        Balances relevance to query with diversity from already-selected chunks.
        """
        similarities = self._compute_similarities(query_vec, chunk_vecs)
        selected_indices = []
        remaining_indices = list(range(len(chunk_vecs)))

        # Precompute pairwise similarities between chunks
        chunk_matrix = np.array(chunk_vecs)
        pairwise = np.dot(chunk_matrix, chunk_matrix.T)

        lambda_param = self.config.diversity_lambda

        for _ in range(budget):
            if not remaining_indices:
                break

            if not selected_indices:
                # First selection: pure similarity
                best_idx = remaining_indices[np.argmax(similarities[remaining_indices])]
            else:
                # MMR: balance similarity and diversity
                mmr_scores = []
                for idx in remaining_indices:
                    relevance = similarities[idx]
                    # Max similarity to already selected chunks
                    redundancy = max(
                        pairwise[idx, sel_idx] for sel_idx in selected_indices
                    )
                    mmr = lambda_param * relevance - (1 - lambda_param) * redundancy
                    mmr_scores.append(mmr)
                best_idx = remaining_indices[np.argmax(mmr_scores)]

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return selected_indices

    def _select_uncertainty(
        self,
        query_vec: np.ndarray,
        chunk_vecs: List[np.ndarray],
        budget: int,
        metadata: Optional[List[dict]] = None,
    ) -> List[int]:
        """
        Uncertainty-based sampling using temperature-scaled softmax.
        Adds stochasticity to selection for exploration.
        """
        similarities = self._compute_similarities(query_vec, chunk_vecs)

        # Temperature-scaled softmax
        scaled_scores = similarities / self.config.temperature
        exp_scores = np.exp(
            scaled_scores - np.max(scaled_scores)
        )  # Numerical stability
        probs = exp_scores / np.sum(exp_scores)

        # Sample without replacement
        selected = np.random.choice(
            len(chunk_vecs), size=budget, replace=False, p=probs
        )

        return selected.tolist()

    def _select_ensemble(
        self,
        query_vec: np.ndarray,
        chunk_vecs: List[np.ndarray],
        budget: int,
        metadata: Optional[List[dict]] = None,
    ) -> List[int]:
        """
        Ensemble selection: combines multiple strategies with voting.
        """
        if self.config.ensemble_weights is None:
            # Default: equal weights for similarity and MMR
            strategies = [SelectionStrategy.SIMILARITY, SelectionStrategy.MMR]
            weights = [0.5, 0.5]
        else:
            strategies = [
                SelectionStrategy.SIMILARITY,
                SelectionStrategy.MMR,
                SelectionStrategy.UNCERTAINTY,
            ]
            weights = self.config.ensemble_weights

        # Get selections from each strategy
        all_selections = []
        for strategy in strategies:
            temp_config = SelectionConfig(strategy=strategy)
            temp_sensor = AdvancedSensor(temp_config)
            selections = temp_sensor.select(query_vec, chunk_vecs, budget * 2, metadata)
            all_selections.append(selections)

        # Weighted voting
        vote_scores = np.zeros(len(chunk_vecs))
        for selections, weight in zip(all_selections, weights):
            for rank, idx in enumerate(selections):
                # Higher rank = more votes (inverse rank weighting)
                vote_scores[idx] += weight * (len(selections) - rank)

        # Select top budget items by votes
        top_indices = np.argsort(vote_scores)[::-1][:budget]
        return top_indices.tolist()

    def _select_adaptive(
        self,
        query_vec: np.ndarray,
        chunk_vecs: List[np.ndarray],
        budget: int,
        metadata: Optional[List[dict]] = None,
    ) -> List[int]:
        """
        Adaptive threshold selection based on score distribution.
        Dynamically determines threshold from percentile of scores.
        """
        similarities = self._compute_similarities(query_vec, chunk_vecs)

        # Compute adaptive threshold
        threshold = np.percentile(similarities, self.config.adaptive_percentile * 100)

        # Select chunks above threshold
        valid_indices = np.where(similarities >= threshold)[0]

        if len(valid_indices) == 0:
            # Fallback: select top-1
            return [int(np.argmax(similarities))]

        # Rank valid chunks and select top budget
        valid_scores = similarities[valid_indices]
        ranked = np.argsort(valid_scores)[::-1][:budget]

        return valid_indices[ranked].tolist()

    def get_scores(
        self, query_vec: np.ndarray, chunk_vecs: List[np.ndarray]
    ) -> np.ndarray:
        """
        Get similarity scores for all chunks (useful for debugging/analysis).

        Returns:
            Array of similarity scores.
        """
        return self._compute_similarities(query_vec, chunk_vecs)


# Example usage and comparison
def example_usage():
    """Demonstrate advanced sensor capabilities."""
    np.random.seed(42)

    # Create sample data
    query_vec = np.random.randn(128)
    query_vec = query_vec / np.linalg.norm(query_vec)

    chunk_vecs = []
    for i in range(10):
        vec = np.random.randn(128)
        vec = vec / np.linalg.norm(vec)
        chunk_vecs.append(vec)

    print("Advanced Sensor Examples")
    print("=" * 50)

    # Example 1: MMR for diversity
    config_mmr = SelectionConfig(
        strategy=SelectionStrategy.MMR,
        diversity_lambda=0.7,  # High diversity
    )
    sensor_mmr = AdvancedSensor(config_mmr)
    selected_mmr = sensor_mmr.select(query_vec, chunk_vecs, budget=3)
    print(f"\nMMR Selection (diverse): {selected_mmr}")

    # Example 2: Uncertainty sampling
    config_uncertain = SelectionConfig(
        strategy=SelectionStrategy.UNCERTAINTY,
        temperature=0.5,  # Lower temperature = less random
    )
    sensor_uncertain = AdvancedSensor(config_uncertain)
    selected_uncertain = sensor_uncertain.select(query_vec, chunk_vecs, budget=3)
    print(f"Uncertainty Selection: {selected_uncertain}")

    # Example 3: Adaptive threshold
    config_adaptive = SelectionConfig(
        strategy=SelectionStrategy.ADAPTIVE,
        adaptive_percentile=0.8,  # Top 20%
    )
    sensor_adaptive = AdvancedSensor(config_adaptive)
    selected_adaptive = sensor_adaptive.select(query_vec, chunk_vecs, budget=3)
    print(f"Adaptive Selection: {selected_adaptive}")

    # Example 4: Ensemble
    config_ensemble = SelectionConfig(
        strategy=SelectionStrategy.ENSEMBLE,
        ensemble_weights=[0.4, 0.4, 0.2],  # Favor similarity and MMR
    )
    sensor_ensemble = AdvancedSensor(config_ensemble)
    selected_ensemble = sensor_ensemble.select(query_vec, chunk_vecs, budget=3)
    print(f"Ensemble Selection: {selected_ensemble}")

    # Show scores
    sensor_simple = AdvancedSensor(
        SelectionConfig(strategy=SelectionStrategy.SIMILARITY)
    )
    scores = sensor_simple.get_scores(query_vec, chunk_vecs)
    print(f"\nAll similarity scores: {scores[:5]}...")  # Show first 5


if __name__ == "__main__":
    example_usage()
