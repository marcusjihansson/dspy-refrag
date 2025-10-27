from typing import Any, Dict, List, Optional

from .fragment import Fragment


class ProtobufStyleSerializer:
    """Protobuf-style serializer with structured message format"""

    @staticmethod
    def serialize_message(msg_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"__type__": msg_type, "__version__": "1.0", "data": data}

    def serialize_fragment_batch(self, fragments: List[Fragment], batch_id: str) -> Dict[str, Any]:
        return self.serialize_message(
            "FragmentBatch",
            {
                "batch_id": batch_id,
                "count": len(fragments),
                "fragments": [f.to_dict() for f in fragments],
            },
        )

    def serialize_retrieval_result(
        self, query: str, fragments: List[Fragment], scores: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        results = []
        for i, fragment in enumerate(fragments):
            result = {"fragment": fragment.to_dict(), "rank": i + 1}
            if scores:
                result["score"] = scores[i]
            results.append(result)

        return self.serialize_message(
            "RetrievalResult",
            {"query": query, "results": results, "total_count": len(fragments)},
        )
