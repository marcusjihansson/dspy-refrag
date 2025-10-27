"""Integration tests for dspy-refrag components."""

import numpy as np
import pytest

from dspy_refrag import REFRAGModule, SimpleRetriever
from dspy_refrag.sensor_advanced import AdvancedSensor, SelectionConfig, SelectionStrategy


def test_refrag_with_advanced_sensor():
    """Test REFRAGModule with advanced sensor."""
    # Create advanced sensor with MMR strategy
    sensor_config = SelectionConfig(
        strategy=SelectionStrategy.MMR,
        diversity_lambda=0.6
    )
    advanced_sensor = AdvancedSensor(sensor_config)
    
    # Create REFRAG module with advanced sensor
    retriever = SimpleRetriever()
    module = REFRAGModule(
        retriever=retriever,
        sensor=advanced_sensor,
        lm=None,
        k=3,
        budget=2
    )
    
    # Test forward pass
    ctx = module.forward("What is machine learning?")
    
    assert ctx.query == "What is machine learning?"
    assert ctx.chunk_vectors is not None
    assert len(ctx.chunk_vectors) == 3  # k=3
    assert ctx.chunk_metadata is not None
    assert len(ctx.chunk_metadata) == 3
    
    # Check that exactly 2 chunks were selected (budget=2)
    selected_count = sum(1 for meta in ctx.chunk_metadata if meta.get('selected', False))
    assert selected_count == 2


def test_refrag_uncertainty_sampling():
    """Test REFRAGModule with uncertainty sampling sensor."""
    sensor_config = SelectionConfig(
        strategy=SelectionStrategy.UNCERTAINTY,
        temperature=0.8
    )
    uncertainty_sensor = AdvancedSensor(sensor_config)
    
    retriever = SimpleRetriever()
    module = REFRAGModule(
        retriever=retriever,
        sensor=uncertainty_sensor,
        lm=None,
        k=4,
        budget=3
    )
    
    ctx = module.forward("Tell me about neural networks")
    
    # SimpleRetriever returns only 3 chunks by default
    assert len(ctx.chunk_vectors) == 3
    assert len(ctx.chunk_metadata) == 3
    
    # Check selection count
    selected_count = sum(1 for meta in ctx.chunk_metadata if meta.get('selected', False))
    assert selected_count <= 3  # Can't select more than available


def test_refrag_adaptive_selection():
    """Test REFRAGModule with adaptive selection sensor."""
    sensor_config = SelectionConfig(
        strategy=SelectionStrategy.ADAPTIVE,
        adaptive_percentile=0.7
    )
    adaptive_sensor = AdvancedSensor(sensor_config)
    
    retriever = SimpleRetriever()
    module = REFRAGModule(
        retriever=retriever,
        sensor=adaptive_sensor,
        lm=None,
        k=5,
        budget=3
    )
    
    ctx = module.forward("Explain deep learning")
    
    # SimpleRetriever returns only 3 chunks by default
    assert len(ctx.chunk_vectors) == 3
    assert len(ctx.chunk_metadata) == 3
    
    # Adaptive selection might select fewer than budget if threshold not met
    selected_count = sum(1 for meta in ctx.chunk_metadata if meta.get('selected', False))
    assert selected_count <= 3
    assert selected_count >= 1  # Should select at least one


def test_serializer_integration():
    """Test integration with vector serializer."""
    from dspy_refrag import VectorAwareSerializer, Fragment
    
    serializer = VectorAwareSerializer()
    
    # Create test fragments using correct API
    fragments = [
        Fragment(
            text=f"Test fragment {i}",
            embedding=[0.1 * i, 0.2 * i, 0.3 * i],
            metadata={"id": i},
            fragment_id=f"frag-{i}"
        )
        for i in range(3)
    ]
    
    # Test serialization (returns string for JSON format)
    serialized = serializer.serialize_fragments(fragments)
    assert isinstance(serialized, str)  # JSON format returns string
    
    # Test deserialization
    deserialized_fragments = serializer.deserialize_fragments(serialized)
    
    assert len(deserialized_fragments) == 3
    
    # Check preservation
    for orig, deser in zip(fragments, deserialized_fragments):
        assert orig.text == deser.text
        assert orig.fragment_id == deser.fragment_id


def test_fragment_validation():
    """Test Fragment data structure validation."""
    from dspy_refrag import Fragment
    
    # Valid fragment
    fragment = Fragment(
        text="This is a test fragment",
        embedding=[1.0, 2.0, 3.0],
        metadata={"source": "test"},
        fragment_id="test-1"
    )
    
    assert fragment.fragment_id == "test-1"
    assert fragment.text == "This is a test fragment"
    assert len(fragment.embedding) == 3
    assert fragment.metadata["source"] == "test"
    
    # Test that validation happens in __post_init__
    # Invalid fragments should raise errors during creation
    with pytest.raises(ValueError, match="Fragment text cannot be empty"):
        Fragment(
            text="",
            embedding=[1.0, 2.0],
            metadata={},
            fragment_id="test"
        )


def test_multiple_serializers():
    """Test different serialization options."""
    from dspy_refrag import (
        JSONFragmentSerializer,
        PickleSerializer,
        UnifiedSerializer,
        Fragment
    )
    
    # Create test fragment
    fragment = Fragment(
        text="Test text",
        embedding=[1.0, 2.0, 3.0],
        metadata={"key": "value"},
        fragment_id="test"
    )
    
    # Test JSON serializer
    json_serializer = JSONFragmentSerializer()
    json_data = json_serializer.serialize_fragments([fragment])
    json_fragments = json_serializer.deserialize_fragments(json_data)
    assert len(json_fragments) == 1
    assert json_fragments[0].fragment_id == "test"
    
    # Test Pickle serializer
    pickle_serializer = PickleSerializer()
    pickle_data = pickle_serializer.serialize_fragments([fragment])
    pickle_fragments = pickle_serializer.deserialize_fragments(pickle_data)
    assert len(pickle_fragments) == 1
    assert pickle_fragments[0].fragment_id == "test"
    
    # Test Unified serializer
    unified_serializer = UnifiedSerializer()
    unified_data = unified_serializer.serialize([fragment])
    unified_fragments = unified_serializer.deserialize(unified_data)
    assert len(unified_fragments) == 1
    assert unified_fragments[0].fragment_id == "test"