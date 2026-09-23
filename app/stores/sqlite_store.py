"""SQLite backend — the zero-infra default.

- `chunks`      : chunk rows + float32 embedding blobs.
- `chunks_fts`  : FTS5 virtual table giving native BM25.
- `documents`   : one row per ingested document.

Dense search is an exact brute-force cosine scan over an in-memory matrix, which
is ideal at portfolio corpus scale. The single connection is opened with
check_same_thread=False and guarded by a lock, so it is safe under FastAPI's
request threadpool.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..models import Chunk, Document, SourceInfo
from ..retrieval.dense import cosine_topk
from .base import tokenize

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    title        TEXT DEFAULT '',
    content_hash TEXT DEFAULT '',
    n_chunks     INTEGER DEFAULT 0,
    n_chars      INTEGER DEFAULT 0,
    created_at   TEXT DEFAULT '',
    metadata     TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
    rowid          INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id       TEXT UNIQUE NOT NULL,
    doc_id         TEXT NOT NULL,
    source         TEXT NOT NULL,
    ordinal        INTEGER NOT NULL,
    text           TEXT NOT NULL,
    char_start     INTEGER NOT NULL,
    char_end       INTEGER NOT NULL,
    token_estimate INTEGER DEFAULT 0,
    metadata       TEXT DEFAULT '{}',
    embedding      BLOB,
    embed_dim      INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    tokenize = 'porter unicode61'
);
"""


def build_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Each token is double-quoted (so FTS5 syntax chars in user input can't break
    the query) and joined with OR to maximise candidate recall — RRF and the
    reranker handle precision downstream.
    """
    terms = tokenize(query)
    if not terms:
        return '""'
    return " OR ".join(f'"{t}"' for t in terms)


class SQLiteStore:
    name = "sqlite"

    def __init__(self, path: str | Path, embed_dim: int = 0):
        self.path = str(path)
        self.embed_dim = embed_dim
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self._lock = threading.RLock()
        self._emb_cache: Optional[tuple[list[str], np.ndarray]] = None

    # ---------------------------------------------------------------- writes
    def upsert_document(self, doc: Document) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO documents(doc_id, source, title, content_hash, n_chunks, n_chars, created_at, metadata)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(doc_id) DO UPDATE SET
                     source=excluded.source, title=excluded.title, content_hash=excluded.content_hash,
                     n_chunks=excluded.n_chunks, n_chars=excluded.n_chars, metadata=excluded.metadata""",
                (
                    doc.doc_id, doc.source, doc.title, doc.content_hash,
                    doc.n_chunks, doc.n_chars, doc.created_at, json.dumps(doc.metadata),
                ),
            )
            self.conn.commit()

    def insert_chunks(self, chunks: Iterable[Chunk],
                      embeddings: Optional[np.ndarray] = None) -> None:
        chunks = list(chunks)
        with self._lock:
            for i, ch in enumerate(chunks):
                emb_blob, dim = None, 0
                if embeddings is not None:
                    vec = np.asarray(embeddings[i], dtype=np.float32)
                    emb_blob, dim = vec.tobytes(), int(vec.shape[0])
                self.conn.execute(
                    """INSERT OR REPLACE INTO chunks
                       (chunk_id, doc_id, source, ordinal, text, char_start, char_end,
                        token_estimate, metadata, embedding, embed_dim)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        ch.chunk_id, ch.doc_id, ch.source, ch.ordinal, ch.text,
                        ch.char_start, ch.char_end, ch.token_estimate,
                        json.dumps(ch.metadata), emb_blob, dim,
                    ),
                )
                self.conn.execute(
                    "INSERT INTO chunks_fts(chunk_id, text) VALUES(?, ?)",
                    (ch.chunk_id, ch.text),
                )
            self.conn.commit()
            self._emb_cache = None  # invalidate

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            rows = self.conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)
            ).fetchall()
            for r in rows:
                self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (r["chunk_id"],))
            self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            self.conn.commit()
            self._emb_cache = None
            return len(rows)

    # ---------------------------------------------------------------- reads
    def has_hash(self, content_hash: str) -> Optional[str]:
        with self._lock:
            row = self.conn.execute(
                "SELECT doc_id FROM documents WHERE content_hash=?", (content_hash,)
            ).fetchone()
        return row["doc_id"] if row else None

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)
            ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
            ).fetchall()
        return {r["chunk_id"]: self._row_to_chunk(r) for r in rows}

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM chunks ORDER BY doc_id, ordinal"
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def doc_chunks(self, doc_id: str) -> list[Chunk]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM chunks WHERE doc_id=? ORDER BY ordinal", (doc_id,)
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    # ---------------------------------------------------------------- retrieval
    def bm25_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        match = build_fts_query(query)
        try:
            with self._lock:
                rows = self.conn.execute(
                    """SELECT chunk_id, bm25(chunks_fts) AS score
                       FROM chunks_fts WHERE chunks_fts MATCH ?
                       ORDER BY score ASC LIMIT ?""",
                    (match, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        # SQLite bm25() is negative; more-negative == better. Flip so higher == better.
        return [(r["chunk_id"], -float(r["score"])) for r in rows]

    def dense_search(self, query_vec: np.ndarray, limit: int) -> list[tuple[str, float]]:
        ids, matrix = self._all_embeddings()
        return cosine_topk(query_vec, ids, matrix, limit)

    def _all_embeddings(self) -> tuple[list[str], np.ndarray]:
        with self._lock:
            if self._emb_cache is not None:
                return self._emb_cache
            rows = self.conn.execute(
                "SELECT chunk_id, embedding, embed_dim FROM chunks WHERE embedding IS NOT NULL"
            ).fetchall()
            ids: list[str] = []
            vecs: list[np.ndarray] = []
            for r in rows:
                dim = r["embed_dim"] or 0
                if not dim:
                    continue
                ids.append(r["chunk_id"])
                vecs.append(np.frombuffer(r["embedding"], dtype=np.float32, count=dim))
            mat = np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)
            self._emb_cache = (ids, mat)
            return self._emb_cache

    # ---------------------------------------------------------------- introspection
    def list_sources(self) -> list[SourceInfo]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [
            SourceInfo(
                doc_id=r["doc_id"], source=r["source"], title=r["title"] or "",
                n_chunks=r["n_chunks"], n_chars=r["n_chars"],
                created_at=r["created_at"] or "",
                injection_flagged=bool(
                    json.loads(r["metadata"] or "{}").get("injection_flagged", False)
                ),
            )
            for r in rows
        ]

    def stats(self) -> dict:
        with self._lock:
            n_docs = self.conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
            n_chunks = self.conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
            n_emb = self.conn.execute(
                "SELECT COUNT(*) c FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()["c"]
        return {"documents": n_docs, "chunks": n_chunks, "embedded_chunks": n_emb,
                "backend": self.name}

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"], doc_id=row["doc_id"], source=row["source"],
            ordinal=row["ordinal"], text=row["text"],
            char_start=row["char_start"], char_end=row["char_end"],
            token_estimate=row["token_estimate"],
            metadata=json.loads(row["metadata"] or "{}"),
        )
