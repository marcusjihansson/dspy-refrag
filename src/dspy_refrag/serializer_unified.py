from typing import List, Optional, Union

from .fragment import Fragment
from .serializer_json import JSONFragmentSerializer
from .serializer_pickle import PickleSerializer

try:
    from .serializer_msgpack import MsgPackSerializer  # optional
    _HAS_MSGPACK = True
except Exception:  # pragma: no cover
    MsgPackSerializer = None  # type: ignore
    _HAS_MSGPACK = False


class UnifiedSerializer:
    """Unified serializer that supports multiple formats"""

    SUPPORTED_FORMATS = ["json", "msgpack", "pickle"]

    def __init__(self, default_format: str = "json"):
        if default_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {default_format}. Supported: {self.SUPPORTED_FORMATS}"
            )
        self.default_format = default_format
        self.json_ser = JSONFragmentSerializer()
        self.pickle_ser = PickleSerializer()
        self.msgpack_ser = MsgPackSerializer() if _HAS_MSGPACK else None

    def serialize(self, fragments: List[Fragment], format: Optional[str] = None) -> Union[str, bytes]:
        fmt = format or self.default_format
        if fmt == "json":
            return self.json_ser.serialize_fragments(fragments)
        if fmt == "msgpack":
            if not self.msgpack_ser:
                raise RuntimeError("msgpack not available")
            return self.msgpack_ser.serialize_fragments(fragments)
        if fmt == "pickle":
            return self.pickle_ser.serialize_fragments(fragments)
        raise ValueError(f"Unsupported format: {fmt}")

    def deserialize(self, data, format: Optional[str] = None):
        fmt = format or self.default_format
        if fmt == "json":
            return self.json_ser.deserialize_fragments(data)
        if fmt == "msgpack":
            if not self.msgpack_ser:
                raise RuntimeError("msgpack not available")
            return self.msgpack_ser.deserialize_fragments(data)
        if fmt == "pickle":
            return self.pickle_ser.deserialize_fragments(data)
        raise ValueError(f"Unsupported format: {fmt}")
