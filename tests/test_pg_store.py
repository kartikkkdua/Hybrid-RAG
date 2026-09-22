"""Postgres + pgvector backend tests.

Skipped automatically unless a database is reachable. Run one with:

    docker compose up -d db
    PG_TEST_DSN=postgresql://postgres:postgres@localhost:55432/rag pytest tests/test_pg_store.py
"""
from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

DSN = os.getenv("PG_TEST_DSN", "")


def _pg_available(dsn: str) -> bool:
    if not dsn:
        return False
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _pg_available(DSN), reason="no Postgres available (set PG_TEST_DSN)"
)


@pytest.fixture()
def pg_service():
    from app.config import Settings
    from app.service import RAGService

    settings = Settings(DB_BACKEND="postgres", PG_DSN=DSN,
                        EMBED_BACKEND="tfidf", RERANK_BACKEND="lexical")
    svc = RAGService(settings=settings)
    yield svc
    svc.close()


SAMPLE = ("Reciprocal Rank Fusion merges ranked lists by summing one over k plus rank. "
          "The constant k is commonly set to sixty. BM25 is a sparse lexical ranking "
          "function. Dense retrieval encodes text into vectors and matches on meaning.")


def test_roundtrip_and_hybrid_search(pg_service):
    src = f"pgtest-{uuid.uuid4().hex[:8]}.md"
    r = pg_service.ingest_text(SAMPLE, source=src, title="pg test")
    assert r.n_chunks >= 1

    resp = pg_service.search("what constant does RRF use?", top_k=3)
    assert resp.results
    assert "bm25" in resp.stages and "dense" in resp.stages
    top = resp.results[0]
    assert top.rrf_score is not None and top.rank == 1

    # cleanup
    assert pg_service.delete_document(r.doc_id) >= 1


def test_dense_search_uses_pgvector(pg_service):
    src = f"pgtest-{uuid.uuid4().hex[:8]}.md"
    r = pg_service.ingest_text(SAMPLE, source=src)
    qv = pg_service.embedder.embed_query("vectors and semantic meaning")
    hits = pg_service.db.dense_search(np.asarray(qv), 3)
    assert hits
    # pgvector cosine similarity is 1 - distance, so scores live in [-1, 1]
    assert all(-1.01 <= s <= 1.01 for _cid, s in hits)
    pg_service.delete_document(r.doc_id)


def test_delete_cascades(pg_service):
    src = f"pgtest-{uuid.uuid4().hex[:8]}.md"
    r = pg_service.ingest_text(SAMPLE, source=src)
    before = pg_service.stats()["chunks"]
    deleted = pg_service.delete_document(r.doc_id)
    after = pg_service.stats()["chunks"]
    assert deleted >= 1
    assert after == before - deleted
    assert pg_service.db.doc_chunks(r.doc_id) == []
