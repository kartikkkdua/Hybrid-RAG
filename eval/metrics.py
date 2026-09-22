"""Retrieval metrics computed against a gold set.

A gold item pairs a question with the chunk_ids that truly answer it. Given the
ranked chunk_ids a configuration returned, we compute:

- recall@k : fraction of gold chunks present in the top k
- hit@k    : 1 if any gold chunk is in the top k
- MRR      : 1 / rank of the first gold hit (0 if none)
- nDCG@k   : ordering-aware, graded by presence in the gold set
"""
from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for r in topk if r in relevant)
    return hits / len(relevant)


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        rel = 1.0 if r in relevant else 0.0
        dcg += rel / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(per_query: list[dict]) -> dict:
    """Mean of each numeric metric across queries."""
    if not per_query:
        return {}
    keys = [k for k, v in per_query[0].items() if isinstance(v, (int, float))]
    return {k: round(sum(q[k] for q in per_query) / len(per_query), 4) for k in keys}
