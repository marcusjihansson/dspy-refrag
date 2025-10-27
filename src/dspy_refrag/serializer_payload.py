from typing import Any, Dict, List, Optional

from .fragment import Fragment


class DSPyPayloadSerializer:
    """Creates DSPy-compatible payloads with fragment embeddings"""

    def create_payload(
        self, query: str, fragments: List[Fragment], max_fragments: Optional[int] = None
    ) -> Dict[str, Any]:
        if max_fragments:
            fragments = fragments[:max_fragments]

        payload = {
            "query": query,
            "fragments": [
                {
                    "text": f.text,
                    "metadata": f.metadata,
                    "fragment_id": f.fragment_id,
                    "parent_doc_id": f.parent_doc_id,
                }
                for f in fragments
            ],
            "embeddings": [f.embedding for f in fragments],
            "num_fragments": len(fragments),
        }
        return payload

    def create_training_example(
        self, question: str, answer: str, supporting_fragments: List[Fragment]
    ) -> Dict[str, Any]:
        context = "\n\n".join(
            [f"[Fragment {i + 1}]: {f.text}" for i, f in enumerate(supporting_fragments)]
        )
        return {
            "question": question,
            "context": context,
            "answer": answer,
            "fragments": [f.to_dict() for f in supporting_fragments],
        }
