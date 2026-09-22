"""The storage contract.

Everything above this layer (ingestion, retrieval, generation, API) talks only to
`Store`. Two backends implement it:

- `SQLiteStore`   : zero-infra default. FTS5 for BM25, NumPy cosine for dense.
- `PostgresStore` : production. tsvector/ts_rank for sparse, pgvector HNSW for dense.

Keeping both retrieval calls (`bm25_search`, `dense_search`) on the Store is what
lets the Postgres backend push vector search into the database instead of scanning
in Python.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Protocol, runtime_checkable

import numpy as np

from ..models import Chunk, Document, SourceInfo

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(query: str) -> list[str]:
    return _WORD_RE.findall(query.lower())


@runtime_checkable
class Store(Protocol):
    """Storage + retrieval primitives. Scores are always higher == better."""

    name: str

    # --- writes ---
    def upsert_document(self, doc: Document) -> None: ...
    def insert_chunks(self, chunks: Iterable[Chunk],
                      embeddings: Optional[np.ndarray] = None) -> None: ...
    def delete_document(self, doc_id: str) -> int: ...

    # --- reads ---
    def has_hash(self, content_hash: str) -> Optional[str]: ...
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]: ...
    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]: ...
    def all_chunks(self) -> list[Chunk]: ...
    def doc_chunks(self, doc_id: str) -> list[Chunk]: ...

    # --- retrieval ---
    def bm25_search(self, query: str, limit: int) -> list[tuple[str, float]]: ...
    def dense_search(self, query_vec: np.ndarray, limit: int) -> list[tuple[str, float]]: ...

    # --- introspection ---
    def list_sources(self) -> list[SourceInfo]: ...
    def stats(self) -> dict: ...
    def close(self) -> None: ...
