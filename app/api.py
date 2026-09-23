"""FastAPI surface for the RAG service.

All JSON/SSE endpoints live under /api/* so the built React SPA (mounted at /)
and the dev Vite proxy can share one path scheme.

  POST /api/ingest        ingest raw text
  POST /api/ingest/path   ingest a file or directory on the server
  POST /api/search        hybrid retrieval (returns per-stage scores)
  POST /api/answer        grounded answer with verified citations
  GET  /api/answer/stream live token streaming (SSE)
  GET  /api/sources       list ingested documents
  GET  /api/stats         corpus + backend info
  GET  /api/healthz
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .generation import SYSTEM_PROMPT, _format_context
from .models import AnswerRequest, IngestRequest, SearchRequest
from .service import get_service

app = FastAPI(title="Hybrid RAG", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

api = APIRouter(prefix="/api")


@api.get("/healthz")
def healthz():
    return {"status": "ok"}


@api.get("/stats")
def stats():
    return get_service().stats()


@api.post("/ingest")
def ingest(req: IngestRequest):
    svc = get_service()
    if not req.text:
        return {"error": "text is required"}
    resp = svc.ingest_text(req.text, source=req.source or "inline", title=req.title,
                           metadata=req.metadata)
    return resp.model_dump()


@api.post("/ingest/path")
def ingest_path(payload: dict):
    svc = get_service()
    path = payload.get("path")
    if not path:
        return {"error": "path is required"}
    return {"ingested": [r.model_dump() for r in svc.ingest_path(path)]}


@api.post("/search")
def search(req: SearchRequest):
    svc = get_service()
    resp = svc.search(req.query, top_k=req.top_k, rerank=req.rerank,
                      dense=req.dense, bm25=req.bm25, history=req.history)
    return resp.model_dump()


@api.post("/answer")
def answer(req: AnswerRequest):
    svc = get_service()
    ans = svc.answer(req.query, top_k=req.top_k, rerank=req.rerank, mode=req.mode,
                     history=req.history)
    return ans.model_dump()


@api.post("/agent")
def agent(req: AnswerRequest):
    """Answer via the multi-agent graph, returning the full node trace."""
    svc = get_service()
    if not svc.agent_available():
        raise HTTPException(status_code=501, detail="langgraph is not installed")
    return svc.agent_answer(req.query, top_k=req.top_k,
                            history=req.history).model_dump()


@api.get("/sources")
def sources():
    return {"sources": [s.model_dump() for s in get_service().list_sources()]}


@api.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Ingest an uploaded .pdf/.txt/.md file."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    resp = get_service().ingest_upload(file.filename or "upload", raw)
    return resp.model_dump()


@api.get("/documents/{doc_id}")
def get_document(doc_id: str):
    doc = get_service().document(doc_id)
    if doc["document"] is None:
        raise HTTPException(status_code=404, detail="document not found")
    return doc


@api.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    n = get_service().delete_document(doc_id)
    return {"doc_id": doc_id, "deleted_chunks": n}


@api.get("/answer/stream")
async def answer_stream(query: str, top_k: int = 8, rerank: bool = True,
                        dense: bool = True, bm25: bool = True, history: str = ""):
    """Stream retrieval metadata, then live answer tokens, then verified citations.

    `history` is a JSON array of {role, text} turns. EventSource can only issue
    GET, so the client sends a truncated window of recent turns in the query
    string; it is only used to condense a follow-up into a standalone query.
    """
    svc = get_service()
    turns = []
    if history:
        try:
            parsed = json.loads(history)
            if isinstance(parsed, list):
                turns = [t for t in parsed if isinstance(t, dict)]
        except (json.JSONDecodeError, TypeError):
            turns = []
    sr = svc.search(query, top_k=top_k, rerank=rerank, dense=dense, bm25=bm25,
                    history=turns)

    async def event_gen():
        yield {"event": "retrieval", "data": json.dumps(
            {"results": [r.model_dump() for r in sr.results], "stages": sr.stages,
             "latency_ms": sr.latency_ms, "search_query": sr.search_query,
             "rewritten": sr.rewritten, "rewrite_method": sr.rewrite_method})}

        if not sr.results:
            empty = svc.stats().get("chunks", 0) == 0
            yield {"event": "done", "data": json.dumps({
                "refused": True,
                "refusal_reason": (
                    "The corpus is empty — add a document before asking."
                    if empty else
                    "No passage in the corpus was relevant to this question."
                ),
            })}
            return

        # No LLM configured: emit the extractive answer as one chunk.
        if not svc.llm.available:
            ans = svc.generator.answer(query, sr.results, mode="extractive")
            yield {"event": "token", "data": json.dumps({"text": ans.answer})}
            yield {"event": "done", "data": json.dumps(
                {"answer": ans.answer,
                 "citations": [c.model_dump() for c in ans.citations],
                 "usage": ans.usage.model_dump(), "refused": ans.refused,
                 "verification": ans.verification})}
            return

        # Live LLM streaming (raw JSON tokens); verify citations from the buffer at end.
        context = _format_context(sr.results)
        user = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nReturn the JSON object now."
        buffer = ""
        usage = None
        for kind, payload in svc.llm.stream(SYSTEM_PROMPT, user, max_tokens=1200):
            if kind == "delta":
                buffer += payload
                yield {"event": "token", "data": json.dumps({"text": payload})}
            elif kind == "usage":
                usage = payload

        from .generation import _extract_json, _RawAnswer, _verify_citations
        data = _extract_json(buffer)
        citations, verification, refused, answer_text = [], {}, False, buffer
        if data:
            try:
                raw = _RawAnswer.model_validate(data)
                cits, verification = _verify_citations(raw, sr.results)
                citations = [c.model_dump() for c in cits if c.verified]
                answer_text = raw.answer
                refused = (not raw.answerable) or (len(citations) == 0)
            except Exception:
                pass
        yield {"event": "done", "data": json.dumps({
            "answer": answer_text, "citations": citations,
            "usage": usage.model_dump() if usage else {},
            "verification": verification, "refused": refused})}

    return EventSourceResponse(event_gen())


app.include_router(api)

# Serve the built SPA at / when present (production / Docker). API routes above
# take precedence because they are registered first.
_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
