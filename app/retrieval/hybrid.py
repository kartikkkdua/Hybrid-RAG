"""Orchestrates the full hybrid pipeline and records per-stage timing/scores.

    query
      ├─ BM25 (FTS5)        top_k_bm25
      ├─ dense (cosine)     top_k_dense
      ├─ RRF fusion         merge the two ranked lists
      └─ cross-encoder      rerank the fused shortlist  ->  top_k
"""
from __future__ import annotations

import time
from typing import Optional

from ..config import Settings
from ..embeddings import Embedder
from ..models import RetrievedChunk, SearchResponse
from ..stores.base import Store
from .fusion import reciprocal_rank_fusion
from .rerank import build_reranker


class HybridRetriever:
    def __init__(
        self,
        db: Store,
        embedder: Embedder,
        settings: Settings,
        reranker=None,
    ):
        self.db = db
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker  # lazily built on first rerank if None

    def _get_reranker(self):
        if self.reranker is None:
            self.reranker = build_reranker(self.settings)
        return self.reranker

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_rerank: bool = True,
        query_vec=None,
    ) -> SearchResponse:
        s = self.settings
        top_k = top_k or s.top_k_rerank
        stages: dict = {}
        t0 = time.perf_counter()

        ranked_lists: dict[str, list[tuple[str, float]]] = {}
        bm25_scores: dict[str, float] = {}
        dense_scores: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}
        dense_ranks: dict[str, int] = {}

        # --- sparse ---
        if use_bm25:
            tb = time.perf_counter()
            bm25 = self.db.bm25_search(query, s.top_k_bm25)
            ranked_lists["bm25"] = bm25
            bm25_scores = {cid: sc for cid, sc in bm25}
            bm25_ranks = {cid: i + 1 for i, (cid, _) in enumerate(bm25)}
            stages["bm25"] = {"hits": len(bm25), "ms": round((time.perf_counter() - tb) * 1000, 2)}

        # --- dense ---
        if use_dense:
            td = time.perf_counter()
            # Reuse a caller-supplied vector when available (the answer path
            # already embeds the query for the semantic cache).
            qv = query_vec if query_vec is not None else self.embedder.embed_query(query)
            # The store owns the vector search: NumPy cosine on SQLite,
            # pgvector HNSW on Postgres.
            dense = self.db.dense_search(qv, s.top_k_dense)
            ranked_lists["dense"] = dense
            dense_scores = {cid: sc for cid, sc in dense}
            dense_ranks = {cid: i + 1 for i, (cid, _) in enumerate(dense)}
            stages["dense"] = {
                "hits": len(dense),
                "embedder": self.embedder.name,
                "ms": round((time.perf_counter() - td) * 1000, 2),
            }

        # --- fuse ---
        fused = reciprocal_rank_fusion(ranked_lists, k=s.rrf_k)
        stages["fusion"] = {"method": "rrf", "k": s.rrf_k, "candidates": len(fused)}

        # shortlist to rerank: a few multiples of top_k, capped
        shortlist_n = max(top_k, s.top_k_rerank) * 3
        shortlist = fused[:shortlist_n]
        chunk_map = self.db.get_chunks([f.chunk_id for f in shortlist])

        # --- rerank ---
        rerank_scores: dict[str, float] = {}
        rerank_name = "disabled"
        if use_rerank and shortlist:
            tr = time.perf_counter()
            reranker = self._get_reranker()
            rerank_name = reranker.name
            cands = [(f.chunk_id, chunk_map[f.chunk_id].text) for f in shortlist if f.chunk_id in chunk_map]
            reranked = reranker.rerank(query, cands)
            rerank_scores = {cid: sc for cid, sc in reranked}
            order = [cid for cid, _ in reranked]
            stages["rerank"] = {
                "model": rerank_name,
                "ms": round((time.perf_counter() - tr) * 1000, 2),
            }
        else:
            order = [f.chunk_id for f in shortlist]
            stages["rerank"] = {"model": rerank_name}

        fused_by_id = {f.chunk_id: f for f in fused}
        results: list[RetrievedChunk] = []
        for rank, cid in enumerate(order[:top_k], start=1):
            ch = chunk_map.get(cid)
            if ch is None:
                continue
            f = fused_by_id.get(cid)
            final_score = rerank_scores.get(cid, f.rrf_score if f else 0.0)
            results.append(
                RetrievedChunk(
                    chunk_id=ch.chunk_id, doc_id=ch.doc_id, source=ch.source,
                    text=ch.text, char_start=ch.char_start, char_end=ch.char_end,
                    ordinal=ch.ordinal,
                    bm25_score=bm25_scores.get(cid),
                    dense_score=dense_scores.get(cid),
                    bm25_rank=bm25_ranks.get(cid),
                    dense_rank=dense_ranks.get(cid),
                    rrf_score=(f.rrf_score if f else None),
                    rerank_score=rerank_scores.get(cid),
                    score=final_score,
                    rank=rank,
                )
            )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return SearchResponse(query=query, results=results, latency_ms=latency_ms, stages=stages)
