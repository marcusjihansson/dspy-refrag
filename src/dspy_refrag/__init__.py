"""dspy_refrag public API
Expose the primary REFRAG module and helpers for users to import."""

from .refrag import REFRAGContext, REFRAGModule
from .retriever import SimpleRetriever
from .sensor import Sensor
from .serializer_vector import VectorAwareSerializer, VectorQuantizer
from .fragment import Fragment
from .serializer_json import JSONFragmentSerializer
from .serializer_pickle import PickleSerializer
from .serializer_unified import UnifiedSerializer
from .serializer_protobuf import ProtobufStyleSerializer
from .faiss_retriever import FAISSRetriever
from .pinecone_retriever import PineconeRetriever
from .psql_retriever import PSQLRetriever

__all__ = [
    "REFRAGModule",
    "REFRAGContext",
    "SimpleRetriever",
    "Sensor",
    "VectorAwareSerializer",
    "VectorQuantizer",
    "Fragment",
    "JSONFragmentSerializer",
    "PickleSerializer",
    "UnifiedSerializer",
    "ProtobufStyleSerializer",
    "FAISSRetriever",
    "PineconeRetriever",
    "PSQLRetriever",
]
