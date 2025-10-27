"""
PostgreSQL-based retriever for REFRAG integration.

This retriever uses PostgreSQL with pgvector extension for vector storage and similarity search.
"""

from typing import Callable, List

import numpy as np
import psycopg2

from .common import Passage
from .retriever import Retriever


class PSQLRetriever(Retriever):
    """
    PostgreSQL retriever using pgvector for vector similarity search.
    """

    def __init__(
        self,
        db_url: str,
        embedder: Callable[[str], np.ndarray],
        table_name: str = "passages",
    ):
        self.db_url = db_url
        self.embedder = embedder
        self.table_name = table_name
        self._conn = None
        self._init_db()

    def _init_db(self):
        """Initialize database connection and table if needed."""
        self._conn = psycopg2.connect(self.db_url)
        if self._conn:
            with self._conn.cursor() as cur:
                # Create table with pgvector support
                cur.execute(
                    f"""
                    CREATE EXTENSION IF NOT EXISTS vector;
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        text TEXT,
                        vector vector(768),  -- Adjust dimension as needed
                        metadata JSONB
                    );
                """
                )
                self._conn.commit()

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder(query)

    def retrieve(self, query: str, k: int = 3) -> List[Passage]:
        qv = self.embed_query(query)
        if not self._conn:
            raise RuntimeError("Database connection not initialized.")
        with self._conn.cursor() as cur:
            # Use vector similarity search
            cur.execute(
                f"""
                SELECT text, vector, metadata
                FROM {self.table_name}
                ORDER BY vector <-> %s::vector
                LIMIT %s;
            """,
                (qv.tolist(), k),
            )
            rows = cur.fetchall()
            return [
                Passage(text=row[0], vector=np.array(row[1]), metadata=row[2] or {})
                for row in rows
            ]

    def add_passages(self, passages: List[Passage]):
        if not self._conn:
            raise RuntimeError("Database connection not initialized.")
        with self._conn.cursor() as cur:
            for p in passages:
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} (text, vector, metadata)
                    VALUES (%s, %s::vector, %s);
                """,
                    (p.text, p.vector.tolist(), p.metadata),
                )
            self._conn.commit()

    def __del__(self):
        if self._conn:
            self._conn.close()
