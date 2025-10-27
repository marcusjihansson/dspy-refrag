from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
import json
import os
import threading


@dataclass
class MemoryRecord:
    type: str  # "qna", "reasoning", "optimization", "signal"
    session_id: str
    question: Optional[str] = None
    answer: Optional[str] = None
    context_refs: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    score: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    derived_from: List[str] = field(default_factory=list)
    prompt_deltas: Optional[str] = None
    rationale: Optional[str] = None
    improved_metrics: Optional[Dict[str, Any]] = None
    tool_calls: Optional[Dict[str, Any]] = None
    model_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    def save(self, record: MemoryRecord) -> str:
        raise NotImplementedError

    def batch_save(self, records: Sequence[MemoryRecord]) -> List[str]:
        return [self.save(r) for r in records]

    def get_recent_by_type(self, session_id: str, type_: str, limit: int = 5) -> List[MemoryRecord]:
        raise NotImplementedError

    def get_recent_qna(self, session_id: str, limit: int = 5) -> List[MemoryRecord]:
        return self.get_recent_by_type(session_id, "qna", limit)

    def search(self, session_id: str, query: str, k: int = 5, type_filter: Optional[str] = None) -> List[MemoryRecord]:
        """Simple lexical search fallback; can be overridden with vectors."""
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._records: List[MemoryRecord] = []
        self._lock = threading.Lock()

    def save(self, record: MemoryRecord) -> str:
        with self._lock:
            self._records.append(record)
            return str(len(self._records) - 1)

    def get_recent_by_type(self, session_id: str, type_: str, limit: int = 5) -> List[MemoryRecord]:
        with self._lock:
            items = [r for r in reversed(self._records) if r.session_id == session_id and r.type == type_]
            return items[:limit]

    def search(self, session_id: str, query: str, k: int = 5, type_filter: Optional[str] = None) -> List[MemoryRecord]:
        with self._lock:
            hits: List[tuple[float, MemoryRecord]] = []
            q = query.lower()
            for r in self._records:
                if r.session_id != session_id:
                    continue
                if type_filter and r.type != type_filter:
                    continue
                text = (r.question or "") + "\n" + (r.answer or "")
                score = text.lower().count(q)
                if score > 0:
                    hits.append((float(score), r))
            hits.sort(key=lambda x: x[0], reverse=True)
            return [r for _, r in hits[:k]]


class LocalJSONStore(MemoryStore):
    def __init__(self, file_path: str = "memory_store.json") -> None:
        self.file_path = file_path
        self._lock = threading.Lock()
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load(self) -> List[Dict[str, Any]]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_all(self, records: List[Dict[str, Any]]) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def save(self, record: MemoryRecord) -> str:
        with self._lock:
            records = self._load()
            records.append(record.__dict__)
            self._save_all(records)
            return str(len(records) - 1)

    def get_recent_by_type(self, session_id: str, type_: str, limit: int = 5) -> List[MemoryRecord]:
        with self._lock:
            data = self._load()
            items = [MemoryRecord(**r) for r in data if r.get("session_id") == session_id and r.get("type") == type_]
            items.sort(key=lambda r: r.created_at, reverse=True)
            return items[:limit]

    def search(self, session_id: str, query: str, k: int = 5, type_filter: Optional[str] = None) -> List[MemoryRecord]:
        with self._lock:
            data = self._load()
            q = query.lower()
            hits: List[tuple[float, MemoryRecord]] = []
            for rec in data:
                if rec.get("session_id") != session_id:
                    continue
                if type_filter and rec.get("type") != type_filter:
                    continue
                text = (rec.get("question") or "") + "\n" + (rec.get("answer") or "")
                score = text.lower().count(q)
                if score > 0:
                    hits.append((float(score), MemoryRecord(**rec)))
            hits.sort(key=lambda x: x[0], reverse=True)
            return [r for _, r in hits[:k]]
