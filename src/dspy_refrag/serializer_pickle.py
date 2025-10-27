import pickle
from typing import Any, List

from .fragment import Fragment


class PickleSerializer:
    """Python pickle serialization (fast but Python-only)"""

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol

    def serialize_fragment(self, fragment: Fragment) -> bytes:
        return pickle.dumps(fragment, protocol=self.protocol)

    def serialize_fragments(self, fragments: List[Fragment]) -> bytes:
        return pickle.dumps(fragments, protocol=self.protocol)

    def deserialize_fragment(self, data: bytes) -> Fragment:
        return pickle.loads(data)

    def deserialize_fragments(self, data: bytes):
        return pickle.loads(data)

    def serialize_object(self, obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=self.protocol)

    def deserialize_object(self, data: bytes) -> Any:
        return pickle.loads(data)
