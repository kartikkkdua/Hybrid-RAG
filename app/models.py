"""Pydantic data contracts shared across ingestion, retrieval, generation and the API.

The `Answer` / `Citation` pair is the structured-extraction output contract:
the model must return an answer plus a list of citations, each tied to a chunk
and a verbatim quote whose character span we independently verify.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Corpus objects
# --------------------------------------------------------------------------- #
class Document(BaseModel):
    doc_id: str
    source: str
    title: str = ""
    content_hash: str = ""
    n_chunks: int = 0
    n_chars: int = 0
    created_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    ordinal: int
    text: str
    char_start: int  # absolute offset within the parent document
    char_end: int
    token_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    text: str
    char_start: int
    char_end: int
    ordinal: int
    # scores from each stage (None if the stage did not run / did not surface it)
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    score: float = 0.0  # final ordering score
    rank: int = 0


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"] = "user"
    text: str = ""


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    rerank: bool = True
    dense: bool = True
    bm25: bool = True
    history: list[HistoryTurn] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str                       # the user's original wording
    results: list[RetrievedChunk]
    latency_ms: float = 0.0
    stages: dict[str, Any] = Field(default_factory=dict)
    search_query: str = ""           # what retrieval actually used
    rewritten: bool = False          # true when history was folded in
    rewrite_method: str = "none"     # none | heuristic | llm


# --------------------------------------------------------------------------- #
# Generation / citations (the structured-output contract)
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    """A single grounded claim.

    The LLM supplies `claim`, `chunk_id` and a verbatim `quote`.
    We fill `char_start` / `char_end` (absolute, within the source document) and
    `verified` after locating the quote inside the cited chunk.
    """

    claim: str
    chunk_id: str
    quote: str
    source: str = ""
    char_start: int = -1
    char_end: int = -1
    verified: bool = False


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str = ""


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    verification: dict[str, Any] = Field(default_factory=dict)
    search_query: str = ""
    rewritten: bool = False
    rewrite_method: str = "none"


class AnswerRequest(BaseModel):
    query: str
    top_k: int = 8
    rerank: bool = True
    mode: Literal["grounded", "extractive"] = "grounded"
    history: list[HistoryTurn] = Field(default_factory=list)


class TraceStep(BaseModel):
    """One node execution in the agent graph — what the UI renders as the trace."""

    node: str
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    ms: float = 0.0


class AgentAnswer(Answer):
    """An Answer produced by the multi-agent graph, plus how it got there."""

    route: str = ""
    route_reason: str = ""
    subquestions: list[str] = Field(default_factory=list)
    iterations: int = 0
    escalated: bool = False
    critique: str = ""
    trace: list[TraceStep] = Field(default_factory=list)


class IngestRequest(BaseModel):
    text: Optional[str] = None
    source: Optional[str] = None
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    source: str
    n_chunks: int
    n_chars: int
    skipped: bool = False


class SourceInfo(BaseModel):
    doc_id: str
    source: str
    title: str
    n_chunks: int
    n_chars: int
    created_at: str
