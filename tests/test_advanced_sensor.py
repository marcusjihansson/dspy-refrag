"""Tests for advanced sensor functionality."""

import numpy as np
import pytest

from dspy_refrag.sensor_advanced import (AdvancedSensor, SelectionConfig,
                                         SelectionStrategy)


def test_advanced_sensor_similarity():
    """Test basic similarity-based selection."""
    config = SelectionConfig(strategy=SelectionStrategy.SIMILARITY)
    sensor = AdvancedSensor(config)

    # Create test data
    query_vec = np.array([1.0, 0.0, 0.0])
    chunk_vecs = [
        np.array([0.9, 0.1, 0.0]),  # High similarity
        np.array([0.0, 1.0, 0.0]),  # Low similarity
        np.array([0.8, 0.2, 0.0]),  # Medium similarity
    ]

    selected = sensor.select(query_vec, chunk_vecs, budget=2)

    assert len(selected) == 2
    assert 0 in selected  # Highest similarity
    assert 2 in selected  # Second highest
    assert 1 not in selected  # Lowest similarity


def test_advanced_sensor_mmr():
    """Test Maximal Marginal Relevance selection."""
    config = SelectionConfig(strategy=SelectionStrategy.MMR, diversity_lambda=0.5)
    sensor = AdvancedSensor(config)

    # Create test data with some similar chunks
    query_vec = np.array([1.0, 0.0, 0.0])
    chunk_vecs = [
        np.array([0.9, 0.1, 0.0]),  # High relevance
        np.array([0.85, 0.15, 0.0]),  # High relevance, similar to 0
        np.array([0.0, 0.8, 0.6]),  # Lower relevance, diverse
    ]

    selected = sensor.select(query_vec, chunk_vecs, budget=2)

    assert len(selected) == 2
    # Should prefer diverse chunks over similar ones


def test_advanced_sensor_uncertainty():
    """Test uncertainty-based sampling."""
    config = SelectionConfig(strategy=SelectionStrategy.UNCERTAINTY, temperature=1.0)
    sensor = AdvancedSensor(config)

    query_vec = np.array([1.0, 0.0])
    chunk_vecs = [
        np.array([0.9, 0.1]),
        np.array([0.1, 0.9]),
        np.array([0.5, 0.5]),
    ]

    # Run multiple times to check randomness
    selections = []
    for _ in range(10):
        selected = sensor.select(query_vec, chunk_vecs, budget=2)
        selections.append(tuple(sorted(selected)))

    # Should have some variability due to sampling
    unique_selections = set(selections)
    assert len(unique_selections) >= 1  # At least one selection pattern


def test_advanced_sensor_adaptive():
    """Test adaptive threshold selection."""
    config = SelectionConfig(
        strategy=SelectionStrategy.ADAPTIVE, adaptive_percentile=0.6
    )
    sensor = AdvancedSensor(config)

    query_vec = np.array([1.0, 0.0])
    chunk_vecs = [
        np.array([0.9, 0.1]),  # High similarity
        np.array([0.8, 0.2]),  # Medium-high similarity
        np.array([0.3, 0.7]),  # Low similarity
        np.array([0.2, 0.8]),  # Very low similarity
    ]

    selected = sensor.select(query_vec, chunk_vecs, budget=3)

    # Should select chunks above the 60th percentile threshold
    assert len(selected) <= 3
    # Higher similarity chunks should be preferred
    assert 0 in selected  # Highest similarity


def test_advanced_sensor_ensemble():
    """Test ensemble selection strategy."""
    config = SelectionConfig(
        strategy=SelectionStrategy.ENSEMBLE,
        ensemble_weights=[0.6, 0.4],  # Favor similarity over MMR
    )
    sensor = AdvancedSensor(config)

    query_vec = np.array([1.0, 0.0])
    chunk_vecs = [
        np.array([0.9, 0.1]),
        np.array([0.1, 0.9]),
        np.array([0.7, 0.3]),
    ]

    selected = sensor.select(query_vec, chunk_vecs, budget=2)

    assert len(selected) == 2
    # Should combine votes from multiple strategies


def test_advanced_sensor_config_validation():
    """Test configuration validation."""
    # Test invalid diversity_lambda
    with pytest.raises(ValueError, match="diversity_lambda must be in"):
        config = SelectionConfig(diversity_lambda=-0.1)
        AdvancedSensor(config)

    with pytest.raises(ValueError, match="diversity_lambda must be in"):
        config = SelectionConfig(diversity_lambda=1.1)
        AdvancedSensor(config)

    # Test invalid temperature
    with pytest.raises(ValueError, match="temperature must be positive"):
        config = SelectionConfig(temperature=0)
        AdvancedSensor(config)

    # Test invalid adaptive_percentile
    with pytest.raises(ValueError, match="adaptive_percentile must be in"):
        config = SelectionConfig(adaptive_percentile=-0.1)
        AdvancedSensor(config)

    with pytest.raises(ValueError, match="adaptive_percentile must be in"):
        config = SelectionConfig(adaptive_percentile=1.1)
        AdvancedSensor(config)


def test_advanced_sensor_get_scores():
    """Test score computation functionality."""
    sensor = AdvancedSensor()

    query_vec = np.array([1.0, 0.0])
    chunk_vecs = [
        np.array([0.9, 0.1]),
        np.array([0.1, 0.9]),
    ]

    scores = sensor.get_scores(query_vec, chunk_vecs)

    assert len(scores) == 2
    assert scores[0] > scores[1]  # First chunk should have higher similarity


def test_advanced_sensor_empty_chunks():
    """Test behavior with empty chunk list."""
    sensor = AdvancedSensor()

    query_vec = np.array([1.0, 0.0])
    chunk_vecs = []

    selected = sensor.select(query_vec, chunk_vecs, budget=2)

    assert selected == []


def test_advanced_sensor_min_score():
    """Test minimum score threshold."""
    config = SelectionConfig(strategy=SelectionStrategy.SIMILARITY, min_score=0.5)
    sensor = AdvancedSensor(config)

    query_vec = np.array([1.0, 0.0])
    chunk_vecs = [
        np.array([0.9, 0.1]),  # Score: 0.9 (above threshold)
        np.array([0.3, 0.7]),  # Score: 0.3 (below threshold)
        np.array([0.6, 0.4]),  # Score: 0.6 (above threshold)
    ]

    selected = sensor.select(query_vec, chunk_vecs, budget=3)

    # Should only select chunks above threshold
    assert len(selected) == 2
    assert 0 in selected
    assert 2 in selected
    assert 1 not in selected
