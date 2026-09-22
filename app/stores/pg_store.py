"""Postgres + pgvector backend — the production path.

Differences from the SQLite backend that matter at scale:

- Dense search runs *in the database* via pgvector's cosine operator (`<=>`) with
  an **HNSW** index, instead of scanning every vector in Python.
- Sparse search uses a generated `tsvector` column with a **GIN** index and
  `ts_rank_cd` for scoring.
- A connection pool serves concurrent API requests.

Note on ranking: vanilla Postgres full-text gives tf-idf-style `ts_rank_cd`, not
true Okapi BM25. It is the right sparse signal for a stock Postgres deployment;
for exact BM25 add the ParadeDB `pg_search` extension and swap the one query in
`bm25_search`. RRF only consumes *ranks*, so the fusion stage is unaffected.

Enable with:  DB_BACKEND=postgres  PG_DSN=postgresql://user:pass@host:5432/rag
"""
from __future__ import annotations

import json
from typing import Iterable, Optional

import numpy as np

from ..models import Chunk, Document, SourceInfo
from .base import tokenize


def _vec_literal(vec: np.ndarray) -> str:
    """pgvector accepts a '[1,2,3]' text literal, cast to ::vector."""
    return "[" + ",".join(f"{float(x):.8f}" for x in np.asarray(vec).ravel()) + "]"


class PostgresStore:
    name = "postgres+pgvector"

    def __init__(self, dsn: str, embed_dim: int):
        if not embed_dim:
            raise ValueError("PostgresStore requires a known embedding dimension.")
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as e:  # pragma: no cover - dependency guard
            raise ImportError(
                "Postgres backend needs `psycopg[binary]` and `psycopg-pool`. "
                "Install with: pip install -r requirements-pg.txt"
            ) from e

        self.dsn = dsn
        self.embed_dim = embed_dim
        self.pool = ConnectionPool(dsn, min_size=1, max_size=8, open=True)
        self._migrate()

    # ---------------------------------------------------------------- schema
    def _migrate(self) -> None:
        d = int(self.embed_dim)
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id       TEXT PRIMARY KEY,
                        source       TEXT NOT NULL,
                        title        TEXT DEFAULT '',
                        content_hash TEXT UNIQUE,
                        n_chunks     INTEGER DEFAULT 0,
                        n_chars      INTEGER DEFAULT 0,
                        created_at   TEXT DEFAULT '',
                        metadata     JSONB DEFAULT '{}'::jsonb
                    );""")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id       TEXT PRIMARY KEY,
                        doc_id         TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                        source         TEXT NOT NULL,
                        ordinal        INTEGER NOT NULL,
                        text           TEXT NOT NULL,
                        char_start     INTEGER NOT NULL,
                        char_end       INTEGER NOT NULL,
                        token_estimate INTEGER DEFAULT 0,
                        metadata       JSONB DEFAULT '{{}}'::jsonb,
                        embedding      vector({d}),
                        tsv            tsvector GENERATED ALWAYS AS
                                         (to_tsvector('english', text)) STORED
                    );""")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN(tsv);")
                # HNSW for approximate-NN cosine search.
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_embedding "
                    "ON chunks USING hnsw (embedding vector_cosine_ops);"
                )
            conn.commit()

    # ---------------------------------------------------------------- writes
    def upsert_document(self, doc: Document) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO documents(doc_id, source, title, content_hash, n_chunks,
                                         n_chars, created_at, metadata)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                   ON CONFLICT(doc_id) DO UPDATE SET
                     source=EXCLUDED.source, title=EXCLUDED.title,
                     content_hash=EXCLUDED.content_hash, n_chunks=EXCLUDED.n_chunks,
                     n_chars=EXCLUDED.n_chars, metadata=EXCLUDED.metadata""",
                (doc.doc_id, doc.source, doc.title, doc.content_hash, doc.n_chunks,
                 doc.n_chars, doc.created_at, json.dumps(doc.metadata)),
            )
            conn.commit()

    def insert_chunks(self, chunks: Iterable[Chunk],
                      embeddings: Optional[np.ndarray] = None) -> None:
        chunks = list(chunks)
        if not chunks:
            return
        rows = []
        for i, ch in enumerate(chunks):
            emb = _vec_literal(embeddings[i]) if embeddings is not None else None
            rows.append((ch.chunk_id, ch.doc_id, ch.source, ch.ordinal, ch.text,
                         ch.char_start, ch.char_end, ch.token_estimate,
                         json.dumps(ch.metadata), emb))
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks(chunk_id, doc_id, source, ordinal, text,
                                          char_start, char_end, token_estimate,
                                          metadata, embedding)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::vector)
                       ON CONFLICT(chunk_id) DO UPDATE SET
                         text=EXCLUDED.text, embedding=EXCLUDED.embedding,
                         metadata=EXCLUDED.metadata""",
                    rows,
                )
            conn.commit()

    def delete_document(self, doc_id: str) -> int:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE doc_id=%s", (doc_id,))
                n = cur.fetchone()[0]
                cur.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))  # cascades
            conn.commit()
        return int(n)

    # ---------------------------------------------------------------- reads
    def has_hash(self, content_hash: str) -> Optional[str]:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT doc_id FROM documents WHERE content_hash=%s", (content_hash,)
            ).fetchone()
        return row[0] if row else None

    _COLS = ("chunk_id, doc_id, source, ordinal, text, char_start, char_end, "
             "token_estimate, metadata")

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self.pool.connection() as conn:
            row = conn.execute(
                f"SELECT {self._COLS} FROM chunks WHERE chunk_id=%s", (chunk_id,)
            ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {self._COLS} FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,)
            ).fetchall()
        chunks = [self._row_to_chunk(r) for r in rows]
        return {c.chunk_id: c for c in chunks}

    def all_chunks(self) -> list[Chunk]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {self._COLS} FROM chunks ORDER BY doc_id, ordinal"
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def doc_chunks(self, doc_id: str) -> list[Chunk]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {self._COLS} FROM chunks WHERE doc_id=%s ORDER BY ordinal",
                (doc_id,),
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    # ---------------------------------------------------------------- retrieval
    def bm25_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        ts_query = " | ".join(terms)  # OR, mirroring the SQLite backend
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT chunk_id, ts_rank_cd(tsv, q) AS score
                   FROM chunks, to_tsquery('english', %s) q
                   WHERE tsv @@ q
                   ORDER BY score DESC LIMIT %s""",
                (ts_query, limit),
            ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def dense_search(self, query_vec: np.ndarray, limit: int) -> list[tuple[str, float]]:
        if query_vec is None or np.asarray(query_vec).size == 0:
            return []
        lit = _vec_literal(query_vec)
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score
                   FROM chunks WHERE embedding IS NOT NULL
                   ORDER BY embedding <=> %s::vector LIMIT %s""",
                (lit, lit, limit),
            ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    # ---------------------------------------------------------------- introspection
    def list_sources(self) -> list[SourceInfo]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT doc_id, source, title, n_chunks, n_chars, created_at
                   FROM documents ORDER BY created_at DESC"""
            ).fetchall()
        return [
            SourceInfo(doc_id=r[0], source=r[1], title=r[2] or "", n_chunks=r[3],
                       n_chars=r[4], created_at=r[5] or "")
            for r in rows
        ]

    def stats(self) -> dict:
        with self.pool.connection() as conn:
            n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            n_emb = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        return {"documents": n_docs, "chunks": n_chunks, "embedded_chunks": n_emb,
                "backend": self.name}

    def close(self) -> None:
        self.pool.close()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _row_to_chunk(r) -> Chunk:
        meta = r[8]
        if isinstance(meta, str):
            meta = json.loads(meta or "{}")
        return Chunk(
            chunk_id=r[0], doc_id=r[1], source=r[2], ordinal=r[3], text=r[4],
            char_start=r[5], char_end=r[6], token_estimate=r[7], metadata=meta or {},
        )
