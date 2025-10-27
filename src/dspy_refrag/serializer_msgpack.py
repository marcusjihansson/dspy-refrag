from typing import Any, Dict, List

try:
    import msgpack  # type: ignore
    MSGPACK_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    msgpack = None  # type: ignore
    MSGPACK_AVAILABLE = False

from .fragment import Fragment


class MsgPackSerializer:
    """Binary serialization using MessagePack (more efficient than JSON)"""

    def __init__(self):
        if not MSGPACK_AVAILABLE:
            raise RuntimeError("msgpack not available. Install with: uv add msgpack")

    def serialize_fragment(self, fragment: Fragment) -> bytes:
        data = fragment.to_dict()
        return msgpack.packb(data, use_bin_type=True)

    def serialize_fragments(self, fragments: List[Fragment]) -> bytes:
        data = [f.to_dict() for f in fragments]
        return msgpack.packb(data, use_bin_type=True)

    def deserialize_fragment(self, data: bytes) -> Fragment:
        unpacked = msgpack.unpackb(data, raw=False)
        return Fragment.from_dict(unpacked)

    def deserialize_fragments(self, data: bytes):
        unpacked = msgpack.unpackb(data, raw=False)
        return [Fragment.from_dict(item) for item in unpacked]
