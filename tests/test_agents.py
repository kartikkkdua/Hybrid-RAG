"""Multi-agent graph tests.

These run with no API key: every node has a heuristic fallback, so routing,
decomposition, escalation and termination are all deterministically testable.
"""
from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from app.agents.nodes import MAX_ITERATIONS  # noqa: E402
from app.models import Answer  # noqa: E402

SAMPLE = (
    "Reciprocal Rank Fusion merges ranked lists by summing one over k plus rank. "
    "The constant k is commonly set to sixty.\n\n"
    "A cross-encoder reads the query and passage together in a single forward pass. "
    "A bi-encoder embeds them separately.\n\n"
    "nDCG discounts relevant results that appear further down the list, so it "
    "captures graded relevance and ordering."
)


@pytest.fixture()
def svc(service):
    service.ingest_text(SAMPLE, source="agents.md", title="agents")
    return service


def _nodes(answer) -> list[str]:
    return [t.node for t in answer.trace]


def test_simple_question_takes_lookup_route(svc):
    a = svc.agent_answer("What constant does RRF use?", top_k=4)
    assert a.route == "lookup"
    assert "lookup" in _nodes(a)
    assert "planner" not in _nodes(a)
    assert a.iterations == 1


def test_multipart_question_takes_research_route(svc):
    a = svc.agent_answer(
        "How does a cross-encoder differ from a bi-encoder, and what does nDCG capture?",
        top_k=6,
    )
    assert a.route == "research"
    assert len(a.subquestions) > 1
    nodes = _nodes(a)
    assert "planner" in nodes and "researcher" in nodes


def test_empty_corpus_is_rejected(service):
    a = service.agent_answer("anything at all?")
    assert a.route == "reject"
    assert a.refused is True
    assert "empty" in a.refusal_reason.lower()


def test_ungrounded_draft_escalates_then_terminates(svc, monkeypatch):
    """The critic rejects a citation-less draft; the run escalates into the
    research path and still terminates at MAX_ITERATIONS rather than looping."""

    def no_citations(query, retrieved, model=None, mode="grounded"):
        return Answer(answer="A confident but unsupported claim.", citations=[],
                      refused=False, retrieved=retrieved)

    monkeypatch.setattr(svc.generator, "answer", no_citations)

    a = svc.agent_answer("What constant does RRF use?", top_k=4)

    assert a.refused is True                       # never ships an ungrounded answer
    assert a.iterations == MAX_ITERATIONS          # bounded, terminated
    assert a.escalated is True                     # lookup handed off to the researcher
    assert "planner" in _nodes(a)
    assert _nodes(a).count("critic") == MAX_ITERATIONS


def test_trace_is_ordered_and_timed(svc):
    a = svc.agent_answer("What constant does RRF use?", top_k=4)
    assert _nodes(a)[0] == "router"
    assert _nodes(a)[-1] == "critic"
    assert all(t.ms >= 0 for t in a.trace)
