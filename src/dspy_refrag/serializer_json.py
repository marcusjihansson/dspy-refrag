import json
from typing import List

import numpy as np

from .fragment import Fragment


class JSONFragmentSerializer:
    """Serializes fragments to JSON for storage and retrieval"""

    def __init__(self, normalize_vectors: bool = True):
        self.normalize_vectors = normalize_vectors

    def serialize_fragment(self, fragment: Fragment) -> str:
        data = fragment.to_dict()
        if self.normalize_vectors and data["embedding"]:
            data["embedding"] = self._normalize(data["embedding"])
        return json.dumps(data)

    def serialize_fragments(self, fragments: List[Fragment]) -> str:
        data = [f.to_dict() for f in fragments]
        if self.normalize_vectors:
            for item in data:
                if item["embedding"]:
                    item["embedding"] = self._normalize(item["embedding"])
        return json.dumps(data)

    def deserialize_fragment(self, json_str: str) -> Fragment:
        data = json.loads(json_str)
        return Fragment.from_dict(data)

    def deserialize_fragments(self, json_str: str):
        data = json.loads(json_str)
        return [Fragment.from_dict(item) for item in data]

    def _normalize(self, vector):
        arr = np.array(vector)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()
