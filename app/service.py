"""RAGService: the single composition root.

CLI, FastAPI and the MCP server all build a RAGService and call the same methods,
so there is exactly one wiring of db + embedder + retriever + llm + generator.
"""
from __future__ import annotations

from typing import Any, Optional

from .config import Settings, get_settings
from .embeddings import build_embedder
from .generation import Generator
from .ingest import Ingestor
from .llm import LLMClient
from .models import (Answer, Citation, IngestResponse, SearchResponse,
                     SourceInfo, Usage)
from .cache import SemanticCache
from .retrieval import HybridRetriever
from .rewrite import condense_query
from .stores import build_store


def _evidence_overlap(cached_payload: dict, results) -> float:
    """Jaccard overlap between a cached answer's evidence and fresh retrieval."""
    old = {c.get("chunk_id") for c in (cached_payload.get("retrieved") or [])}
    new = {r.chunk_id for r in results}
    if not old or not new:
        return 0.0
    return len(old & new) / len(old | new)


def _same_top_hit(cached_payload: dict, results) -> bool:
    old = (cached_payload.get("retrieved") or [])
    if not old or not results:
        return False
    return old[0].get("chunk_id") == results[0].chunk_id


def _as_turns(history) -> list[dict]:
    """Accept dicts or pydantic HistoryTurn objects from any caller."""
    out = []
    for t in history or []:
        if hasattr(t, "model_dump"):
            t = t.model_dump()
        if isinstance(t, dict) and t.get("text"):
            out.append({"role": t.get("role", "user"), "text": str(t["text"])})
    return out


class RAGService:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        # Embedder first: the Postgres backend needs the vector dimension to
        # create its `vector(d)` column and HNSW index.
        self.embedder = build_embedder(self.settings)
        self.db = build_store(self.settings, embed_dim=self.embedder.dim)
        self.retriever = HybridRetriever(self.db, self.embedder, self.settings)
        self.llm = LLMClient(self.settings)
        self.generator = Generator(self.db, self.llm, self.settings)
        self.ingestor = Ingestor(self.db, self.embedder, self.settings)
        self.cache = SemanticCache(
            enabled=self.settings.cache_enabled,
            threshold=self.settings.cache_threshold,
            max_entries=self.settings.cache_max_entries,
            ttl_seconds=self.settings.cache_ttl_seconds,
        )
        # Bumped on every corpus mutation. Combined with the row counts below it
        # forms the fingerprint that invalidates cached answers.
        self._corpus_version = 0

    def corpus_fingerprint(self) -> str:
        """Identifies the corpus an answer was grounded in.

        Row counts come from the store, so a change made by another process is
        still seen; the local version counter additionally catches a same-process
        delete-then-add that happens to leave the counts unchanged.
        """
        st = self.db.stats()
        return f"{st.get('documents', 0)}:{st.get('chunks', 0)}:{self._corpus_version}"

    # -- ingestion --
    def ingest_text(self, text: str, source: str, title: str = "",
                    metadata: Optional[dict[str, Any]] = None) -> IngestResponse:
        r = self.ingestor.ingest_text(text, source, title, metadata)
        if not r.skipped:
            self._corpus_version += 1
        return r

    def ingest_path(self, path: str) -> list[IngestResponse]:
        rs = self.ingestor.ingest_path(path)
        if any(not r.skipped for r in rs):
            self._corpus_version += 1
        return rs

    def ingest_upload(self, filename: str, raw: bytes) -> IngestResponse:
        """Ingest an uploaded file's bytes (PDF sniffed by extension/magic)."""
        import tempfile
        from pathlib import Path as _P

        suffix = _P(filename).suffix.lower()
        if suffix == ".pdf" or raw[:5] == b"%PDF-":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                tmp.write(raw)
                tmp.flush()
                from .chunking import extract_pdf_text

                text = extract_pdf_text(tmp.name)
        else:
            text = raw.decode("utf-8", errors="ignore")
        return self.ingest_text(text, source=filename, title=_P(filename).stem)

    def delete_document(self, doc_id: str) -> int:
        n = self.db.delete_document(doc_id)
        self._corpus_version += 1
        return n

    def document(self, doc_id: str) -> dict:
        """Full document view: metadata + its chunks (for the source viewer)."""
        chunks = self.db.doc_chunks(doc_id)
        info = next((s for s in self.db.list_sources() if s.doc_id == doc_id), None)
        return {
            "document": info.model_dump() if info else None,
            "chunks": [c.model_dump() for c in chunks],
        }

    # -- retrieval --
    def search(self, query: str, top_k: Optional[int] = None, rerank: bool = True,
               dense: bool = True, bm25: bool = True,
               history: Optional[list] = None, query_vec=None) -> SearchResponse:
        """Hybrid search. When `history` is given, a follow-up question is first
        condensed into a standalone query — retrieval has no memory of its own."""
        rw = condense_query(query, _as_turns(history), self.llm)
        resp = self.retriever.search(
            rw.query, top_k=top_k, use_rerank=rerank, use_dense=dense, use_bm25=bm25,
            query_vec=query_vec,
        )
        # Report the user's wording as `query`, and what we actually searched.
        resp.query = query
        resp.search_query = rw.query
        resp.rewritten = rw.rewritten
        resp.rewrite_method = rw.method
        return resp

    # -- generation --
    def answer(self, query: str, top_k: Optional[int] = None, rerank: bool = True,
               mode: str = "grounded", model: Optional[str] = None,
               history: Optional[list] = None, use_cache: bool = True) -> Answer:
        # Condense first: the cache is keyed on what we would actually search,
        # so two different follow-ups that mean the same thing can share an entry.
        rw = condense_query(query, _as_turns(history), self.llm)
        variant = f"{mode}|{top_k or self.settings.top_k_rerank}|{rerank}|{model or self.settings.gen_model}"
        fingerprint = self.corpus_fingerprint()
        query_vec = self.embedder.embed_query(rw.query)

        sr = None
        if use_cache:
            payload, kind = self.cache.lookup(rw.query, query_vec, fingerprint, variant)
            if payload is not None:
                if kind == "exact":
                    cached = Answer.model_validate(payload)
                    cached.cached = True
                    cached.cache_kind = kind
                    return cached
                # Semantic candidate: embedding similarity alone cannot tell a
                # paraphrase from a near-miss (see app/cache.py), so confirm the
                # new query actually retrieves the same evidence. Retrieval is
                # ~10ms; a wrong cached answer is unbounded damage.
                sr = self.search(query, top_k=top_k, rerank=rerank, history=history,
                                 query_vec=query_vec)
                if (_same_top_hit(payload, sr.results)
                        and _evidence_overlap(payload, sr.results) >= self.settings.cache_min_overlap):
                    cached = Answer.model_validate(payload)
                    cached.cached = True
                    cached.cache_kind = kind
                    return cached
                self.cache.stats.rejected_semantic += 1  # verified and thrown out

        if sr is None:
            sr = self.search(query, top_k=top_k, rerank=rerank, history=history,
                             query_vec=query_vec)
        # Generation gets the ORIGINAL question so the answer addresses what the
        # user actually asked, not the expanded retrieval query.
        ans = self.generator.answer(query, sr.results, model=model, mode=mode)
        ans.retrieved = sr.results
        ans.search_query = sr.search_query
        ans.rewritten = sr.rewritten
        ans.rewrite_method = sr.rewrite_method

        # Only cache answers worth replaying: a refusal may simply mean the right
        # document has not been ingested yet.
        if use_cache and not ans.refused:
            self.cache.store(rw.query, query_vec, ans.model_dump(), fingerprint, variant)
        return ans

    def lookup_cached(self, query: str, top_k: Optional[int] = None,
                      rerank: bool = True, mode: str = "grounded",
                      model: Optional[str] = None,
                      history: Optional[list] = None) -> Optional[Answer]:
        """Cache probe without generating. Used by the streaming endpoint so a
        cache hit can be returned instantly instead of re-streaming tokens."""
        if not self.cache.enabled:
            return None
        rw = condense_query(query, _as_turns(history), self.llm)
        variant = f"{mode}|{top_k or self.settings.top_k_rerank}|{rerank}|{model or self.settings.gen_model}"
        qv = self.embedder.embed_query(rw.query)
        payload, kind = self.cache.lookup(rw.query, qv, self.corpus_fingerprint(), variant)
        if payload is None:
            return None
        if kind == "semantic":
            # Same verification as `answer()` — never serve an unverified
            # semantic match (see app/cache.py for why similarity is not enough).
            sr = self.search(query, top_k=top_k, rerank=rerank, history=history,
                             query_vec=qv)
            if not (_same_top_hit(payload, sr.results)
                    and _evidence_overlap(payload, sr.results) >= self.settings.cache_min_overlap):
                self.cache.stats.rejected_semantic += 1
                return None
        ans = Answer.model_validate(payload)
        ans.cached = True
        ans.cache_kind = kind
        return ans

    def cache_streamed_answer(self, query: str, *, top_k, rerank, history,
                              answer_text, citations, usage, verification,
                              retrieved, search_query, rewritten,
                              rewrite_method, mode: str = "grounded",
                              model: Optional[str] = None) -> None:
        """Store an answer assembled by the streaming endpoint.

        Streaming builds its answer incrementally rather than through
        `answer()`, so it has to hand the finished result back to the cache.
        """
        if not self.cache.enabled:
            return
        rw = condense_query(query, _as_turns(history), self.llm)
        variant = f"{mode}|{top_k or self.settings.top_k_rerank}|{rerank}|{model or self.settings.gen_model}"
        ans = Answer(
            answer=answer_text,
            citations=[Citation.model_validate(c) for c in citations],
            refused=False, retrieved=retrieved,
            usage=Usage.model_validate(usage) if usage else Usage(),
            verification=verification or {}, search_query=search_query,
            rewritten=rewritten, rewrite_method=rewrite_method,
        )
        qv = self.embedder.embed_query(rw.query)
        self.cache.store(rw.query, qv, ans.model_dump(),
                         self.corpus_fingerprint(), variant)

    # -- multi-agent --
    def agent_answer(self, query: str, top_k: int = 8, history: Optional[list] = None):
        """Run the LangGraph multi-agent graph (router → research → critic).

        Imported lazily: langgraph is an optional extra, so the core app keeps
        working without it.
        """
        from .agents.graph import get_runner

        rw = condense_query(query, _as_turns(history), self.llm)
        ans = get_runner(self).run(rw.query, top_k=top_k)
        ans.search_query = rw.query
        ans.rewritten = rw.rewritten
        ans.rewrite_method = rw.method
        return ans

    def agent_answer_streaming(self, query: str, top_k: int = 8,
                               history: Optional[list] = None):
        """Yield the agent's node events as they happen, then the final answer."""
        from .agents.graph import get_runner

        rw = condense_query(query, _as_turns(history), self.llm)
        for kind, payload in get_runner(self).run_streaming(rw.query, top_k=top_k):
            if kind == "done":
                payload.search_query = rw.query
                payload.rewritten = rw.rewritten
                payload.rewrite_method = rw.method
            yield kind, payload

    @staticmethod
    def agent_available() -> bool:
        try:
            import langgraph  # noqa: F401

            return True
        except ImportError:
            return False

    # -- introspection --
    def list_sources(self) -> list[SourceInfo]:
        return self.db.list_sources()

    def stats(self) -> dict:
        s = self.db.stats()
        s["embedder"] = self.embedder.name
        s["llm_available"] = self.llm.available
        s["agent_available"] = self.agent_available()
        s["cache"] = self.cache.as_dict()
        return s

    def close(self) -> None:
        self.db.close()


_service: Optional[RAGService] = None


def get_service() -> RAGService:
    global _service
    if _service is None:
        _service = RAGService()
    return _service
