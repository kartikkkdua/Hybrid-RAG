"""Semantic answer cache.

The interesting tests are not "does it cache" but "does it ever serve something
it shouldn't": a stale answer after the corpus changed, a cross-variant collision,
or a cached refusal that hides a newly-ingested document.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.cache import SemanticCache, normalize_query

CORPUS = (
    "BM25 weights rare terms using inverse document frequency and saturates "
    "term frequency.\n\n"
    "A cross-encoder reads the query and passage together in a single forward "
    "pass, which is slower but more accurate than a bi-encoder."
)
VARIANT = "grounded|8|True|model-x"
PAYLOAD = {"answer": "cached answer", "usage": {"cost_usd": 0.002, "latency_ms": 800.0}}


def _v(*xs) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


# ------------------------------------------------------------------ unit
def test_normalize_query_folds_case_and_whitespace():
    assert normalize_query("  What   IS\nBM25? ") == "what is bm25?"


def test_exact_hit_ignores_case_and_spacing():
    c = SemanticCache()
    c.store("what is bm25?", _v(1, 0, 0), PAYLOAD, "fp1", VARIANT)
    payload, kind = c.lookup("What  IS   BM25?", _v(1, 0, 0), "fp1", VARIANT)
    assert kind == "exact" and payload["answer"] == "cached answer"


def test_semantic_hit_above_threshold():
    c = SemanticCache(threshold=0.9)
    c.store("what is bm25?", _v(1, 0, 0), PAYLOAD, "fp1", VARIANT)
    _, kind = c.lookup("explain bm25", _v(0.97, 0.24, 0), "fp1", VARIANT)
    assert kind == "semantic"


def test_dissimilar_query_misses():
    c = SemanticCache(threshold=0.9)
    c.store("what is bm25?", _v(1, 0, 0), PAYLOAD, "fp1", VARIANT)
    _, kind = c.lookup("unrelated topic", _v(0, 1, 0), "fp1", VARIANT)
    assert kind == "miss"


def test_corpus_change_invalidates():
    """The critical property: an answer is only valid for the corpus it used."""
    c = SemanticCache()
    c.store("q", _v(1, 0, 0), PAYLOAD, "fp1", VARIANT)
    _, kind = c.lookup("q", _v(1, 0, 0), "fp2", VARIANT)
    assert kind == "miss"
    assert c.stats.invalidations >= 1


def test_variants_do_not_collide():
    """Different top_k / mode / model produce different answers."""
    c = SemanticCache(threshold=0.5)
    c.store("q", _v(1, 0, 0), PAYLOAD, "fp1", "grounded|8|True|model-x")
    _, kind = c.lookup("q", _v(1, 0, 0), "fp1", "extractive|3|False|model-y")
    assert kind == "miss"


def test_lru_eviction_is_bounded():
    c = SemanticCache(max_entries=3)
    for i in range(5):
        c.store(f"q{i}", _v(1, 0, 0), PAYLOAD, "fp1", VARIANT)
    assert len(c._entries) == 3
    assert c.stats.evictions == 2


def test_ttl_expiry():
    c = SemanticCache(ttl_seconds=-1)  # everything is already expired
    c.store("q", _v(1, 0, 0), PAYLOAD, "fp1", VARIANT)
    _, kind = c.lookup("q", _v(1, 0, 0), "fp1", VARIANT)
    assert kind == "miss"


def test_disabled_cache_never_hits():
    c = SemanticCache(enabled=False)
    c.store("q", _v(1, 0, 0), PAYLOAD, "fp1", VARIANT)
    assert c.lookup("q", _v(1, 0, 0), "fp1", VARIANT)[1] == "miss"


def test_stats_track_savings():
    c = SemanticCache()
    c.store("q", _v(1, 0, 0), PAYLOAD, "fp1", VARIANT)
    c.lookup("q", _v(1, 0, 0), "fp1", VARIANT)
    d = c.as_dict()
    assert d["exact_hits"] == 1
    assert d["cost_saved_usd"] == pytest.approx(0.002)
    assert d["latency_saved_ms"] == pytest.approx(800.0)


# ------------------------------------------------------------- end-to-end
def test_second_identical_question_is_served_from_cache(service):
    service.ingest_text(CORPUS, source="c.md")
    q = "What does BM25 do with rare terms?"
    first = service.answer(q, top_k=3, mode="extractive")
    second = service.answer(q, top_k=3, mode="extractive")
    assert first.cached is False
    assert second.cached is True and second.cache_kind == "exact"
    assert second.answer == first.answer


def test_ingesting_a_document_invalidates_cached_answers(service):
    """A new document could change the answer, so the cache must not win."""
    service.ingest_text(CORPUS, source="c.md")
    q = "What does BM25 do with rare terms?"
    service.answer(q, top_k=3, mode="extractive")
    service.ingest_text("An unrelated note about vector indexes.", source="d.md")
    after = service.answer(q, top_k=3, mode="extractive")
    assert after.cached is False


def test_deleting_a_document_invalidates_cached_answers(service):
    r = service.ingest_text(CORPUS, source="c.md")
    q = "What does BM25 do with rare terms?"
    service.answer(q, top_k=3, mode="extractive")
    service.delete_document(r.doc_id)
    after = service.answer(q, top_k=3, mode="extractive")
    assert after.cached is False


def test_refusals_are_not_cached(service):
    """An empty corpus refusal must not persist after documents arrive."""
    q = "What does BM25 do with rare terms?"
    first = service.answer(q, top_k=3, mode="extractive")
    assert first.refused is True

    service.ingest_text(CORPUS, source="c.md")
    second = service.answer(q, top_k=3, mode="extractive")
    assert second.cached is False
    assert second.refused is False


def test_use_cache_false_bypasses(service):
    service.ingest_text(CORPUS, source="c.md")
    q = "What does BM25 do with rare terms?"
    service.answer(q, top_k=3, mode="extractive")
    again = service.answer(q, top_k=3, mode="extractive", use_cache=False)
    assert again.cached is False


def test_cache_stats_surface_in_service(service):
    service.ingest_text(CORPUS, source="c.md")
    q = "What does BM25 do with rare terms?"
    service.answer(q, top_k=3, mode="extractive")
    service.answer(q, top_k=3, mode="extractive")
    stats = service.stats()["cache"]
    assert stats["enabled"] is True
    assert stats["exact_hits"] == 1
    assert 0.0 < stats["hit_rate"] <= 1.0


# ------------------------------------------------- semantic safety (the hard part)
# Each section is long enough to become its own chunk, so "same evidence" is a
# meaningful check rather than an artefact of everything landing in one chunk.
ADVERSARIAL_CORPUS = (
    "BM25 term frequency saturation is controlled by the parameter k1, which is "
    "typically set between 1.2 and 2.0 for text retrieval. Naive term frequency "
    "scoring rewards repetition without limit, so a document repeating a word "
    "fifty times would score fifty times higher than one mentioning it once. "
    "Saturation means the contribution of each additional occurrence shrinks, on "
    "the intuition that the tenth occurrence of a word tells you far less than "
    "the second did. Tuning k1 upward makes the function behave more like raw "
    "counting, while tuning it toward zero makes every term effectively binary. "
    "The parameter interacts with length normalisation and is usually left at "
    "its default unless the collection is unusual.\n\n"
    "In HNSW the parameter m controls how many edges each node keeps in the "
    "proximity graph, while ef_search trades accuracy against speed at query "
    "time. A larger m produces a denser graph that is more navigable and gives "
    "higher recall, at the cost of memory, since every edge is stored alongside "
    "the vector itself. The ef_construction setting plays the same role during "
    "index building, where each insertion performs its own search to find "
    "neighbours. Raising ef_search widens the beam explored at query time, which "
    "increases recall and latency together. None of these parameters requires a "
    "separate training pass, which is the main operational advantage HNSW holds "
    "over a clustering-based index.\n\n"
    "Product quantization compresses vectors by splitting them into subvectors "
    "and replacing each with a centroid identifier from a learned codebook. "
    "Compression of sixteen to thirty-two times is routine, which is what makes "
    "billion-scale indexes fit in memory at all. The cost is that distances "
    "become approximate, because a vector is now represented by the centroids "
    "nearest its parts rather than by its own coordinates. For that reason "
    "quantization is usually paired with a reranking pass over the "
    "full-precision vectors of the top candidates, recovering most of the "
    "accuracy lost to compression while keeping the memory win."
)


def test_semantic_hit_requires_matching_evidence(service):
    """Embedding similarity alone is not sufficient.

    Measured on this corpus, "k1 in BM25" vs "k1 in HNSW" scores 0.863 while a
    genuine paraphrase can score 0.835 — the classes overlap, so a threshold
    alone would serve the wrong answer. A semantic candidate must also retrieve
    the same evidence.
    """
    service.ingest_text(ADVERSARIAL_CORPUS, source="params.md")

    first = service.answer("What does the k1 parameter control in BM25?",
                           top_k=3, mode="extractive")
    assert first.cached is False

    # A different question that is lexically and semantically very close.
    other = service.answer("What does the m parameter control in HNSW?",
                           top_k=3, mode="extractive")
    assert other.cached is False, "served a cached answer for a different question"
    # and it must be grounded in different evidence
    assert other.citations[0].chunk_id != first.citations[0].chunk_id


def test_exact_hits_skip_verification(service):
    """An identical query string needs no evidence check."""
    service.ingest_text(ADVERSARIAL_CORPUS, source="params.md")
    q = "What does the k1 parameter control in BM25?"
    service.answer(q, top_k=3, mode="extractive")
    again = service.answer(q, top_k=3, mode="extractive")
    assert again.cached is True and again.cache_kind == "exact"


def test_rejected_semantic_matches_are_counted(service):
    """Rejections are observable, not silent."""
    service.ingest_text(ADVERSARIAL_CORPUS, source="params.md")
    service.answer("What does the k1 parameter control in BM25?", top_k=3, mode="extractive")
    service.answer("What does the m parameter control in HNSW?", top_k=3, mode="extractive")
    service.answer("How does product quantization compress vectors?", top_k=3, mode="extractive")
    stats = service.stats()["cache"]
    # Either they missed outright or were verified and rejected; never served.
    assert stats["semantic_hits"] == 0
    assert stats["misses"] + stats["rejected_semantic"] >= 2
