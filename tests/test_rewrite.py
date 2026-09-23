"""Conversational query rewriting — history-aware retrieval.

The bug this fixes: retrieval has no memory, so in a multi-turn UI a follow-up
like "why that value?" gets searched literally and matches nothing useful.
"""
from __future__ import annotations

from app.rewrite import condense_query, needs_rewrite

HISTORY = [
    {"role": "user", "text": "What constant does Reciprocal Rank Fusion use?"},
    {"role": "assistant", "text": "The constant k is commonly set to 60."},
]

CORPUS = (
    "Reciprocal Rank Fusion combines ranked lists. The constant k is commonly "
    "set to 60 because it dampens the influence of the very top ranks.\n\n"
    "Cross-encoder rerankers read the query and passage together. They are far "
    "slower than bi-encoders and are applied only to a shortlist.\n\n"
    "Recall at k measures whether a known relevant chunk appears in the top k "
    "results returned by the retriever."
)


# --------------------------------------------------------------- classifier
def test_self_contained_question_is_not_rewritten():
    assert needs_rewrite("What does nDCG capture?", HISTORY)[0] is False
    r = condense_query("What does nDCG capture?", HISTORY)
    assert r.rewritten is False and r.query == "What does nDCG capture?"


def test_referring_expression_triggers_rewrite():
    assert needs_rewrite("why is it set that way?", HISTORY)[0] is True


def test_continuation_opener_triggers_rewrite():
    assert needs_rewrite("why that value?", HISTORY)[0] is True


def test_short_question_with_acronym_stands_alone():
    # "BM25" is distinctive enough to retrieve on despite being short.
    assert needs_rewrite("explain BM25", HISTORY)[0] is False
    # ...without a distinctive token it is not.
    assert needs_rewrite("explain more", HISTORY)[0] is True


def test_no_history_means_no_rewrite():
    r = condense_query("why that value?", [])
    assert r.rewritten is False and r.query == "why that value?"


def test_rewrite_is_additive_and_keeps_original():
    r = condense_query("why that value?", HISTORY)
    assert r.original == "why that value?"
    assert r.query.startswith("why that value?")   # never loses the user's words
    assert "fusion" in r.query.lower()             # gains context from history


# --------------------------------------------------------------- end-to-end
def test_followup_retrieves_the_right_passage(service):
    """The payoff: with history the follow-up finds the RRF passage."""
    service.ingest_text(CORPUS, source="rrf.md", title="rrf")

    with_history = service.search("why that value?", top_k=3, history=HISTORY)
    assert with_history.rewritten is True
    assert with_history.query == "why that value?"          # user's wording preserved
    assert with_history.search_query != "why that value?"   # but we searched wider
    assert "dampens" in with_history.results[0].text        # found the explanation


def test_answer_carries_rewrite_metadata(service):
    service.ingest_text(CORPUS, source="rrf.md")
    ans = service.answer("why that value?", top_k=3, mode="extractive", history=HISTORY)
    assert ans.rewritten is True
    assert ans.rewrite_method in ("heuristic", "llm")
    assert ans.search_query
