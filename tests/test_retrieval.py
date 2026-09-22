SAMPLE = """Reciprocal Rank Fusion merges ranked lists by summing one over k plus rank.
The constant k is commonly set to sixty. BM25 is a sparse lexical ranking function.
Dense retrieval encodes text into vectors and matches on semantic meaning.
A cross-encoder reranker reads the query and passage together for higher precision."""


def _ingest(service):
    service.ingest_text(SAMPLE, source="doc.md", title="doc")


def test_search_returns_scored_results(service):
    _ingest(service)
    resp = service.search("what constant does RRF use?", top_k=3)
    assert resp.results
    top = resp.results[0]
    # hybrid stages populated
    assert "bm25" in resp.stages and "dense" in resp.stages and "fusion" in resp.stages
    assert top.rrf_score is not None
    assert top.rank == 1


def test_bm25_only_and_dense_only(service):
    _ingest(service)
    bm25_only = service.search("cross-encoder reranker", top_k=3, dense=False, rerank=False)
    dense_only = service.search("cross-encoder reranker", top_k=3, bm25=False, rerank=False)
    assert bm25_only.results and dense_only.results
    # bm25-only path should not carry dense scores and vice versa
    assert all(r.dense_score is None for r in bm25_only.results)
    assert all(r.bm25_score is None for r in dense_only.results)


def test_dedup_on_reingest(service):
    _ingest(service)
    r = service.ingest_text(SAMPLE, source="doc.md")
    assert r.skipped is True
    assert service.stats()["documents"] == 1
