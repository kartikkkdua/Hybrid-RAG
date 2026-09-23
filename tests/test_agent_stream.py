"""Streaming the multi-agent graph.

The graph branches and loops, so there is no token stream to forward. What is
worth streaming is the control flow — the user sees it route, decompose,
research and self-critique rather than watching a spinner.
"""
from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

SAMPLE = (
    "Reciprocal Rank Fusion merges ranked lists using one over k plus rank, with "
    "k commonly set to sixty.\n\n"
    "A cross-encoder reads the query and passage together in a single forward "
    "pass, unlike a bi-encoder which embeds them separately.\n\n"
    "nDCG discounts relevant results appearing further down the list, so it "
    "captures graded relevance and ordering."
)


@pytest.fixture()
def svc(service):
    service.ingest_text(SAMPLE, source="agents.md")
    return service


def _collect(svc, question, top_k=4):
    nodes, final = [], None
    for kind, payload in svc.agent_answer_streaming(question, top_k=top_k):
        if kind == "node":
            nodes.append(payload)
        else:
            final = payload
    return nodes, final


def test_nodes_stream_before_the_answer(svc):
    nodes, final = _collect(svc, "What constant does RRF use?")
    assert nodes, "no node events were streamed"
    assert final is not None
    assert nodes[0]["node"] == "router"
    assert all({"node", "detail", "ms"} <= set(n) for n in nodes)


def test_streamed_trace_matches_the_final_answer(svc):
    """Streaming must not diverge from what the blocking path would produce."""
    nodes, final = _collect(svc, "What constant does RRF use?")
    assert [n["node"] for n in nodes] == [t.node for t in final.trace]


def test_streaming_matches_blocking_route(svc):
    q = "How does a cross-encoder differ from a bi-encoder, and what does nDCG capture?"
    _, streamed = _collect(svc, q, top_k=6)
    blocking = svc.agent_answer(q, top_k=6)
    assert streamed.route == blocking.route == "research"
    assert [t.node for t in streamed.trace] == [t.node for t in blocking.trace]


def test_router_step_carries_the_route_for_live_rendering(svc):
    """The UI reads the route from the router step before the final event."""
    nodes, _ = _collect(svc, "What constant does RRF use?")
    router = nodes[0]
    assert router["data"].get("route") in ("lookup", "research", "reject")


def test_empty_corpus_streams_a_rejection(service):
    nodes, final = _collect(service, "anything?")
    assert final.refused is True
    assert [n["node"] for n in nodes][:1] == ["router"]
