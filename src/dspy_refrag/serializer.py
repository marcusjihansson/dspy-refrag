"""Compatibility facade for serializer classes.

This module remains to avoid breaking imports. New implementations live in
separate modules: fragment, serializer_json, serializer_pickle, serializer_msgpack,
serializer_unified, serializer_vector, serializer_payload, serializer_protobuf.
"""

from .fragment import Fragment
from .serializer_json import JSONFragmentSerializer
from .serializer_pickle import PickleSerializer
from .serializer_unified import UnifiedSerializer

try:
    from .serializer_msgpack import MsgPackSerializer  # optional
except Exception:  # pragma: no cover
    MsgPackSerializer = None  # type: ignore
from .serializer_payload import DSPyPayloadSerializer
from .serializer_protobuf import ProtobufStyleSerializer
from .serializer_vector import VectorAwareSerializer, VectorQuantizer

__all__ = [
    "Fragment",
    "JSONFragmentSerializer",
    "PickleSerializer",
    "UnifiedSerializer",
    "MsgPackSerializer",
    "DSPyPayloadSerializer",
    "ProtobufStyleSerializer",
    "VectorAwareSerializer",
    "VectorQuantizer",
]
