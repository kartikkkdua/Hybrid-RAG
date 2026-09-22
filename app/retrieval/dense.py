"""Dense vector search: brute-force cosine over in-memory embeddings.

Exact (no ANN approximation), which is ideal at portfolio corpus scale and keeps
the code dependency-free. For large corpora swap this for pgvector / FAISS /
hnswlib behind the same (query_vec, top_k) -> [(chunk_id, score)] signature.
"""
from __future__ import annotations

import numpy as np


def cosine_topk(
    query_vec: np.ndarray, ids: list[str], matrix: np.ndarray, top_k: int
) -> list[tuple[str, float]]:
    if matrix.shape[0] == 0 or query_vec.size == 0:
        return []
    if matrix.shape[1] != query_vec.shape[0]:
        # dimension mismatch (e.g. corpus embedded with a different backend)
        return []
    q = np.asarray(query_vec, dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)
    scores = matrix @ q  # matrix rows are pre-normalised → dot == cosine
    k = min(top_k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(ids[i], float(scores[i])) for i in idx]
