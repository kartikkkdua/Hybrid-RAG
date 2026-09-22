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
from .models import Answer, IngestResponse, SearchResponse, SourceInfo
from .retrieval import HybridRetriever
from .stores import build_store


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

    # -- ingestion --
    def ingest_text(self, text: str, source: str, title: str = "",
                    metadata: Optional[dict[str, Any]] = None) -> IngestResponse:
        return self.ingestor.ingest_text(text, source, title, metadata)

    def ingest_path(self, path: str) -> list[IngestResponse]:
        return self.ingestor.ingest_path(path)

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
        return self.db.delete_document(doc_id)

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
               dense: bool = True, bm25: bool = True) -> SearchResponse:
        return self.retriever.search(
            query, top_k=top_k, use_rerank=rerank, use_dense=dense, use_bm25=bm25
        )

    # -- generation --
    def answer(self, query: str, top_k: Optional[int] = None, rerank: bool = True,
               mode: str = "grounded", model: Optional[str] = None) -> Answer:
        sr = self.search(query, top_k=top_k, rerank=rerank)
        ans = self.generator.answer(query, sr.results, model=model, mode=mode)
        ans.retrieved = sr.results
        return ans

    # -- multi-agent --
    def agent_answer(self, query: str, top_k: int = 8):
        """Run the LangGraph multi-agent graph (router → research → critic).

        Imported lazily: langgraph is an optional extra, so the core app keeps
        working without it.
        """
        from .agents.graph import get_runner

        return get_runner(self).run(query, top_k=top_k)

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
        return s

    def close(self) -> None:
        self.db.close()


_service: Optional[RAGService] = None


def get_service() -> RAGService:
    global _service
    if _service is None:
        _service = RAGService()
    return _service
