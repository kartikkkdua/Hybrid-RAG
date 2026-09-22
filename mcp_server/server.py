"""MCP server exposing the hybrid retriever as four tools.

Run standalone (stdio):   python -m mcp_server.server
Register in an MCP host (e.g. Claude Desktop) with:
    { "command": "python", "args": ["-m", "mcp_server.server"], "cwd": "<repo>" }

Tools:
  search_corpus    - hybrid retrieval, returns ranked passages with scores
  answer_question  - grounded answer with verified citations
  ingest_document  - add a document to the corpus
  list_sources     - list ingested documents

Any MCP-aware agent can now use this project's retrieval + citation stack as a
backend without bespoke HTTP glue.
"""
from __future__ import annotations

from mcp.server import MCPServer

from app.service import get_service

mcp = MCPServer("hybrid-rag")


@mcp.tool()
def search_corpus(query: str, top_k: int = 8, rerank: bool = True) -> dict:
    """Hybrid search (BM25 + dense + RRF + cross-encoder rerank) over the corpus.

    Returns the top passages with per-stage scores and total latency.
    """
    resp = get_service().search(query, top_k=top_k, rerank=rerank)
    return {
        "query": query,
        "latency_ms": resp.latency_ms,
        "stages": resp.stages,
        "results": [
            {
                "rank": r.rank,
                "chunk_id": r.chunk_id,
                "source": r.source,
                "score": round(r.score, 4),
                "rrf_score": r.rrf_score,
                "rerank_score": r.rerank_score,
                "text": r.text,
                "char_start": r.char_start,
                "char_end": r.char_end,
            }
            for r in resp.results
        ],
    }


@mcp.tool()
def answer_question(query: str, top_k: int = 8) -> dict:
    """Answer a question grounded in the corpus, with verified inline citations.

    Refuses (refused=true) when the corpus does not support an answer.
    """
    ans = get_service().answer(query, top_k=top_k, mode="grounded")
    return {
        "answer": ans.answer,
        "refused": ans.refused,
        "refusal_reason": ans.refusal_reason,
        "citations": [
            {
                "claim": c.claim,
                "source": c.source,
                "chunk_id": c.chunk_id,
                "quote": c.quote,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "verified": c.verified,
            }
            for c in ans.citations
        ],
        "usage": ans.usage.model_dump(),
        "verification": ans.verification,
    }


@mcp.tool()
def research_question(query: str, top_k: int = 8) -> dict:
    """Answer a hard/multi-part question with the multi-agent graph.

    Routes the question (single-hop vs decomposed research), gathers evidence
    across sub-questions, self-critiques the draft, and returns the node trace.
    """
    svc = get_service()
    if not svc.agent_available():
        return {"error": "langgraph is not installed on the server"}
    a = svc.agent_answer(query, top_k=top_k)
    return {
        "answer": a.answer,
        "refused": a.refused,
        "refusal_reason": a.refusal_reason,
        "route": a.route,
        "subquestions": a.subquestions,
        "iterations": a.iterations,
        "escalated": a.escalated,
        "critique": a.critique,
        "citations": [
            {"source": c.source, "chunk_id": c.chunk_id, "quote": c.quote,
             "char_start": c.char_start, "char_end": c.char_end, "verified": c.verified}
            for c in a.citations
        ],
        "trace": [{"node": t.node, "detail": t.detail, "ms": t.ms} for t in a.trace],
        "usage": a.usage.model_dump(),
    }


@mcp.tool()
def ingest_document(text: str, source: str = "mcp-inline", title: str = "") -> dict:
    """Ingest a document (raw text) into the corpus: chunk, embed and index it."""
    resp = get_service().ingest_text(text, source=source, title=title)
    return resp.model_dump()


@mcp.tool()
def list_sources() -> dict:
    """List the documents currently in the corpus."""
    svc = get_service()
    return {
        "stats": svc.stats(),
        "sources": [s.model_dump() for s in svc.list_sources()],
    }


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
