from typing import List

from .fragment import Fragment
from .serializer_json import JSONFragmentSerializer
from .serializer_unified import UnifiedSerializer


class VectorQuantizer:
    """Placeholder for a vector quantizer. No-op by default."""

    def quantize(self, vec: List[float]) -> List[float]:  # pragma: no cover
        return vec


class VectorAwareSerializer:
    """High-level serializer that ensures vectors are normalized and serializes to JSON by default.

    This keeps backward compatibility for code that expects a VectorAwareSerializer type.
    """

    def __init__(self, default_format: str = "json"):
        self._unified = UnifiedSerializer(default_format=default_format)
        self._json = JSONFragmentSerializer(normalize_vectors=True)

    def serialize_fragments(self, frags: List[Fragment], format: str = "json"):
        if format == "json":
            return self._json.serialize_fragments(frags)
        return self._unified.serialize(frags, format=format)

    def deserialize_fragments(self, data, format: str = "json"):
        if format == "json":
            return self._json.deserialize_fragments(data)
        return self._unified.deserialize(data, format=format)
