"""Hybrid retrieval: BM25 (sparse) + dense vectors, fused with RRF, then
cross-encoder reranked."""

from .hybrid import HybridRetriever

__all__ = ["HybridRetriever"]
