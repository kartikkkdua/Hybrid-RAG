"""Semantic answer cache.

Identical and near-identical questions are common in practice, and answering them
twice costs real money. This cache serves both:

- **exact**    — the same query string, normalised. Free to check.
- **semantic** — a different wording of the same question, matched by cosine
                 similarity against cached query embeddings above a threshold.

The hard part is not caching, it is **not serving the wrong answer**. Two
independent failure modes:

1. *Stale* — an answer is only valid for the corpus it was grounded in, so every
   entry records a corpus fingerprint and is discarded when the corpus changes. A
   cache that keeps citing a document you deleted is worse than no cache at all.

2. *Confused* — embedding similarity alone cannot separate a paraphrase from a
   near-miss. Measured on this corpus with bge-small:

       "What does nDCG capture?" / "What does nDCG measure?"        0.835  (same)
       "k1 parameter in BM25?"   / "k1 parameter in HNSW?"          0.863  (DIFFERENT)

   The classes *overlap*: a distinct question scored higher than a genuine
   paraphrase, so no threshold is simultaneously useful and safe. A semantic
   candidate is therefore **verified against retrieval before it is served**
   (see `RAGService.answer`): retrieval costs ~10ms, generation costs seconds, so
   confirming that the new query retrieves the same evidence is cheap insurance.
   Exact hits skip verification — the query string is identical.

Entries also carry the variant they were produced under (mode, top_k, rerank,
model), because those change the answer and must not collide.

Scope: in-process and bounded (LRU + TTL). A multi-process deployment would move
this to Redis; the interface is the same.
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

_WS = re.compile(r"\s+")


def normalize_query(q: str) -> str:
    return _WS.sub(" ", q or "").strip().lower()


@dataclass
class CacheEntry:
    query: str
    vector: np.ndarray
    payload: dict[str, Any]
    fingerprint: str
    variant: str
    created_at: float = field(default_factory=time.time)
    hits: int = 0


@dataclass
class CacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    rejected_semantic: int = 0   # similar enough, but retrieval disagreed
    cost_saved_usd: float = 0.0
    latency_saved_ms: float = 0.0

    @property
    def lookups(self) -> int:
        return self.exact_hits + self.semantic_hits + self.misses

    def as_dict(self, entries: int = 0) -> dict:
        total = self.lookups
        hits = self.exact_hits + self.semantic_hits
        return {
            "entries": entries,
            "lookups": total,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": round(hits / total, 3) if total else 0.0,
            "cost_saved_usd": round(self.cost_saved_usd, 6),
            "latency_saved_ms": round(self.latency_saved_ms, 1),
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "rejected_semantic": self.rejected_semantic,
        }


class SemanticCache:
    def __init__(self, enabled: bool = True, threshold: float = 0.95,
                 max_entries: int = 256, ttl_seconds: float = 3600.0):
        self.enabled = enabled
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats()

    # ------------------------------------------------------------------ read
    def lookup(self, query: str, vector: Optional[np.ndarray], fingerprint: str,
               variant: str) -> tuple[Optional[dict], str]:
        """Return (payload, kind) where kind is 'exact' | 'semantic' | 'miss'."""
        if not self.enabled:
            return None, "miss"
        with self._lock:
            self._evict_stale(fingerprint)
            key = self._key(query, variant)

            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                entry.hits += 1
                self._record_hit(entry, exact=True)
                return entry.payload, "exact"

            if vector is not None and self._entries:
                best, score = self._nearest(vector, variant)
                if best is not None and score >= self.threshold:
                    best.hits += 1
                    self._record_hit(best, exact=False)
                    return best.payload, "semantic"

            self.stats.misses += 1
            return None, "miss"

    def _nearest(self, vector: np.ndarray, variant: str) -> tuple[Optional[CacheEntry], float]:
        v = np.asarray(vector, dtype=np.float32)
        n = np.linalg.norm(v) or 1.0
        v = v / n
        best, best_score = None, -1.0
        for entry in self._entries.values():
            if entry.variant != variant or entry.vector.shape != v.shape:
                continue
            score = float(np.dot(entry.vector, v))
            if score > best_score:
                best, best_score = entry, score
        return best, best_score

    def _record_hit(self, entry: CacheEntry, exact: bool) -> None:
        if exact:
            self.stats.exact_hits += 1
        else:
            self.stats.semantic_hits += 1
        usage = (entry.payload or {}).get("usage") or {}
        self.stats.cost_saved_usd += float(usage.get("cost_usd") or 0.0)
        self.stats.latency_saved_ms += float(usage.get("latency_ms") or 0.0)

    # ----------------------------------------------------------------- write
    def store(self, query: str, vector: Optional[np.ndarray], payload: dict,
              fingerprint: str, variant: str) -> None:
        if not self.enabled:
            return
        v = np.zeros(0, dtype=np.float32)
        if vector is not None:
            v = np.asarray(vector, dtype=np.float32)
            v = v / (np.linalg.norm(v) or 1.0)
        with self._lock:
            key = self._key(query, variant)
            self._entries[key] = CacheEntry(
                query=query, vector=v, payload=payload,
                fingerprint=fingerprint, variant=variant,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self.stats.evictions += 1

    # ------------------------------------------------------------ invalidate
    def _evict_stale(self, fingerprint: str) -> None:
        """Drop entries grounded in a different corpus, or past their TTL."""
        now = time.time()
        dead = [
            k for k, e in self._entries.items()
            if e.fingerprint != fingerprint or (now - e.created_at) > self.ttl_seconds
        ]
        for k in dead:
            del self._entries[k]
        self.stats.invalidations += len(dead)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def as_dict(self) -> dict:
        with self._lock:
            d = self.stats.as_dict(entries=len(self._entries))
        d["enabled"] = self.enabled
        d["threshold"] = self.threshold
        return d

    @staticmethod
    def _key(query: str, variant: str) -> str:
        return f"{variant}::{normalize_query(query)}"
