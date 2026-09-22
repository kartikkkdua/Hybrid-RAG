"""Shared state for the multi-agent graph.

LangGraph threads one dict through every node; each node returns a partial update
that gets merged. Keeping evidence as serialized chunks (not model objects) makes
the state trivially JSON-dumpable for the API and the UI trace.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict


def _extend(left: list, right: list) -> list:
    """Reducer: append new items instead of overwriting (used for trace/evidence)."""
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    # input
    question: str
    top_k: int

    # routing
    route: str  # "lookup" | "research" | "reject"
    route_reason: str

    # research
    subquestions: list[str]
    evidence: list[dict]  # serialized RetrievedChunk, deduped by chunk_id

    # drafting
    draft: str
    citations: list[dict]

    # reflection
    critique: str
    grounded: bool
    iterations: int
    escalated: bool

    # output
    refused: bool
    refusal_reason: str
    usage: dict[str, Any]
    trace: Annotated[list[dict], _extend]
