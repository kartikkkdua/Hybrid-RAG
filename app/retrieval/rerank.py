"""Reranking stage.

- `CrossEncoderReranker`: a query–passage cross-encoder (recommended; needs the
  `ml` extra). Jointly encodes (query, chunk) for a precise relevance score.
- `LexicalReranker`: token-overlap fallback so the stage always runs.

Both expose `rerank(query, candidates) -> [(chunk_id, score)]` sorted best-first.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..config import Settings

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # lazy import

        self._model = CrossEncoder(model_name)
        self.name = f"cross-encoder:{model_name}"

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = [[query, text] for _cid, text in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(
            ((cid, float(s)) for (cid, _t), s in zip(candidates, scores)),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked


class LexicalReranker:
    """IDF-free token-overlap scorer with a small length normalisation."""

    name = "lexical"

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        q_terms = Counter(_WORD_RE.findall(query.lower()))
        if not q_terms:
            return [(cid, 0.0) for cid, _ in candidates]
        scored = []
        for cid, text in candidates:
            d_terms = Counter(_WORD_RE.findall(text.lower()))
            overlap = sum(min(c, d_terms.get(t, 0)) for t, c in q_terms.items())
            norm = math.log1p(len(d_terms) + 1)
            scored.append((cid, overlap / norm))
        return sorted(scored, key=lambda x: x[1], reverse=True)


class NoOpReranker:
    name = "none"

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        # preserve incoming order, decreasing pseudo-scores
        return [(cid, float(len(candidates) - i)) for i, (cid, _t) in enumerate(candidates)]


def build_reranker(settings: Settings):
    backend = (settings.rerank_backend or "auto").lower()
    if backend == "none":
        return NoOpReranker()
    if backend in ("cross_encoder", "auto"):
        try:
            return CrossEncoderReranker(settings.rerank_model)
        except Exception:
            if backend == "cross_encoder":
                raise
    return LexicalReranker()
