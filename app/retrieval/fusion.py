"""Reciprocal Rank Fusion (RRF).

RRF combines ranked lists using only *ranks*, not raw scores, so incomparable
scales (BM25's log-odds vs cosine's [-1,1]) merge cleanly:

    rrf(d) = sum_over_lists  1 / (k + rank_in_list(d))

`k` (default 60, from Cormack et al. 2009) damps the influence of top ranks.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fused:
    chunk_id: str
    rrf_score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)  # list_name -> 1-based rank


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[str, float]]], k: int = 60
) -> list[Fused]:
    """`ranked_lists`: {list_name: [(chunk_id, score) best-first]}."""
    fused: dict[str, Fused] = {}
    for name, results in ranked_lists.items():
        for rank, (chunk_id, _score) in enumerate(results, start=1):
            f = fused.setdefault(chunk_id, Fused(chunk_id))
            f.rrf_score += 1.0 / (k + rank)
            f.ranks[name] = rank
    return sorted(fused.values(), key=lambda f: f.rrf_score, reverse=True)
