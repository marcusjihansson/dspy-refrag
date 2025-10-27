from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Fragment:
    """
    Represents a document fragment with text, embedding, and metadata.

    Used in DSPy REFRAG for vector-based retrieval and processing.
    """

    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    fragment_id: str
    parent_doc_id: Optional[str] = None

    def __post_init__(self):
        """Validate fields after initialization."""
        if not self.text.strip():
            raise ValueError("Fragment text cannot be empty.")
        if not self.embedding:
            raise ValueError("Fragment embedding cannot be empty.")
        if len(self.embedding) == 0:
            raise ValueError("Fragment embedding must have at least one dimension.")
        if not self.fragment_id:
            raise ValueError("Fragment ID cannot be empty.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fragment":
        """Deserialize from dictionary."""
        return cls(**data)

    def get_embedding_array(self) -> np.ndarray:
        """Get embedding as numpy array."""
        return np.array(self.embedding, dtype=np.float32)

    def validate_embedding_dim(self, expected_dim: Optional[int] = None) -> bool:
        """Validate embedding dimensions."""
        if expected_dim and len(self.embedding) != expected_dim:
            raise ValueError(
                f"Embedding dim {len(self.embedding)} != expected {expected_dim}"
            )
        return True


# Scaffolding for custom fragment types
@dataclass
class CustomFragment(Fragment):
    """
    Example scaffolding for custom fragment types.

    Extend Fragment for specialized use cases, e.g., with additional fields.
    """

    custom_field: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Add custom validation if needed
        if self.custom_field and len(self.custom_field) > 100:
            raise ValueError("Custom field too long.")


# Example usage
def example_fragment_creation():
    """Example of creating and validating fragments."""
    frag = Fragment(
        text="Sample text",
        embedding=[0.1, 0.2, 0.3],
        metadata={"source": "test"},
        fragment_id="frag_001",
    )
    frag.validate_embedding_dim(3)
    return frag
