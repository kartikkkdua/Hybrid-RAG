"""Ingestion pipeline: raw text / PDF / directory -> chunks -> embeddings -> store.

Content-hash dedup means re-ingesting an unchanged document is a no-op. This is
the PDF-chat ingestion path too — PDFs are just a text-extraction step in front
of the same chunk/embed/store flow.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
from pathlib import Path
from typing import Any, Optional

from .chunking import chunk_text, extract_pdf_text
from .config import Settings
from .stores.base import Store
from .embeddings import Embedder
from .models import Chunk, Document, IngestResponse


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Ingestor:
    def __init__(self, db: Store, embedder: Embedder, settings: Settings):
        self.db = db
        self.embedder = embedder
        self.settings = settings

    def ingest_text(
        self,
        text: str,
        source: str,
        title: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> IngestResponse:
        metadata = metadata or {}
        content_hash = _hash(text)
        existing = self.db.has_hash(content_hash)
        if existing:
            return IngestResponse(
                doc_id=existing, source=source, n_chunks=0, n_chars=len(text), skipped=True
            )

        doc_id = f"doc_{content_hash[:12]}"
        text_chunks = chunk_text(text, self.settings.chunk_tokens, self.settings.chunk_overlap)
        chunks = [
            Chunk(
                chunk_id=f"{doc_id}::{tc.ordinal}",
                doc_id=doc_id,
                source=source,
                ordinal=tc.ordinal,
                text=tc.text,
                char_start=tc.char_start,
                char_end=tc.char_end,
                token_estimate=tc.token_estimate,
                metadata={"title": title, **metadata},
            )
            for tc in text_chunks
        ]

        embeddings = None
        if chunks:
            embeddings = self.embedder.embed_documents([c.text for c in chunks])

        doc = Document(
            doc_id=doc_id,
            source=source,
            title=title,
            content_hash=content_hash,
            n_chunks=len(chunks),
            n_chars=len(text),
            created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            metadata=metadata,
        )
        self.db.upsert_document(doc)
        if chunks:
            self.db.insert_chunks(chunks, embeddings)
        return IngestResponse(
            doc_id=doc_id, source=source, n_chunks=len(chunks), n_chars=len(text)
        )

    def ingest_file(self, path: str | Path, title: str = "") -> IngestResponse:
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            text = extract_pdf_text(str(path))
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(text, source=path.name, title=title or path.stem)

    def ingest_path(self, path: str | Path, pattern: str = "*") -> list[IngestResponse]:
        path = Path(path)
        if path.is_file():
            return [self.ingest_file(path)]
        results = []
        exts = {".txt", ".md", ".pdf"}
        for p in sorted(path.rglob(pattern)):
            if p.is_file() and p.suffix.lower() in exts:
                results.append(self.ingest_file(p))
        return results
