"""The agents.

Each node is a plain function of (state) -> partial state update, closed over a
RAGService. Every LLM-backed decision has a deterministic heuristic fallback, so
the whole graph still runs (and is testable) with no API key.

    router      classify the question: lookup | research | reject
    lookup      single-hop retrieve + answer
    planner     decompose a hard question into sub-questions
    researcher  retrieve per sub-question, accumulate deduped evidence
    synthesizer draft a grounded answer over all evidence
    critic      self-reflection; can escalate lookup -> research, or re-research
"""
from __future__ import annotations

import re
import time
from typing import Any

from ..generation import _extract_json
from ..models import RetrievedChunk
from ..service import RAGService
from .state import AgentState

MAX_ITERATIONS = 2
MAX_SUBQUESTIONS = 4

_CONNECTORS = re.compile(r"\s+and\s+|\s*;\s*|\s*\?\s*", re.IGNORECASE)
_RESEARCHY = re.compile(
    r"\b(compare|difference|differences|versus|vs\.?|trade-?offs?|how does .* differ|"
    r"both|each|relationship between|pros and cons|why .* and)\b",
    re.IGNORECASE,
)
_STOP = {
    "what", "which", "does", "do", "is", "are", "the", "a", "an", "of", "in", "on",
    "for", "to", "and", "or", "how", "why", "when", "use", "used", "its", "it",
    "that", "this", "with", "from", "by", "about", "can", "should", "would",
}


def _step(node: str, detail: str, t0: float, **data) -> dict:
    return {"node": node, "detail": detail,
            "ms": round((time.perf_counter() - t0) * 1000, 2), "data": data}


def _add_usage(state: AgentState, usage) -> dict:
    cur = dict(state.get("usage") or {})
    cur["input_tokens"] = cur.get("input_tokens", 0) + getattr(usage, "input_tokens", 0)
    cur["output_tokens"] = cur.get("output_tokens", 0) + getattr(usage, "output_tokens", 0)
    cur["cost_usd"] = round(cur.get("cost_usd", 0.0) + getattr(usage, "cost_usd", 0.0), 6)
    cur["latency_ms"] = round(cur.get("latency_ms", 0.0) + getattr(usage, "latency_ms", 0.0), 2)
    if getattr(usage, "model", ""):
        cur["model"] = usage.model
    return cur


def _keywords(text: str, n: int = 6) -> list[str]:
    words = [w for w in re.findall(r"[A-Za-z0-9-]+", text.lower()) if w not in _STOP]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:n]


# --------------------------------------------------------------------------- #
# router
# --------------------------------------------------------------------------- #
_ROUTER_SYS = """You route questions for a retrieval system. Reply ONLY with JSON:
{"route": "lookup" | "research", "reason": "<short>"}

- "lookup"   : a single focused retrieval can answer it (one fact, one definition).
- "research" : needs decomposition — comparisons, multi-part questions, anything
               requiring evidence from several places."""


def make_router(svc: RAGService):
    def router(state: AgentState) -> dict:
        t0 = time.perf_counter()
        q = state["question"]

        if svc.db.stats().get("chunks", 0) == 0:
            return {
                "route": "reject",
                "route_reason": "The corpus is empty — add a document before asking.",
                "trace": [_step("router", "corpus empty -> reject", t0, route="reject")],
            }

        if svc.llm.available:
            try:
                res = svc.llm.complete(_ROUTER_SYS, f"QUESTION: {q}", max_tokens=150)
                data = _extract_json(res.text) or {}
                route = data.get("route")
                if route in ("lookup", "research"):
                    return {
                        "route": route,
                        "route_reason": str(data.get("reason", ""))[:200],
                        "usage": _add_usage(state, res.usage),
                        "trace": [_step("router", f"LLM -> {route}", t0, route=route)],
                    }
            except Exception:
                pass  # fall through to the heuristic

        # Heuristic: multi-clause or comparative questions need research.
        parts = [p for p in _CONNECTORS.split(q) if p and p.strip()]
        route = "research" if (_RESEARCHY.search(q) or len(parts) > 1
                               or len(q.split()) > 14) else "lookup"
        return {
            "route": route,
            "route_reason": "heuristic: " + (
                "comparative/multi-part" if route == "research" else "single focused question"
            ),
            "trace": [_step("router", f"heuristic -> {route}", t0, route=route)],
        }

    return router


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
_PLANNER_SYS = """Break the question into 2-4 independent sub-questions, each
answerable by a single corpus search. Reply ONLY with JSON:
{"subquestions": ["...", "..."]}"""


def make_planner(svc: RAGService):
    def planner(state: AgentState) -> dict:
        t0 = time.perf_counter()
        q = state["question"]
        iteration = state.get("iterations", 0)

        subs: list[str] = []
        usage_update = None
        if svc.llm.available:
            try:
                hint = ""
                if iteration:
                    hint = ("\nThe previous attempt was not well grounded; propose "
                            "DIFFERENT, broader sub-questions.")
                res = svc.llm.complete(_PLANNER_SYS, f"QUESTION: {q}{hint}", max_tokens=400)
                data = _extract_json(res.text) or {}
                subs = [str(s) for s in (data.get("subquestions") or []) if str(s).strip()]
                usage_update = res.usage
            except Exception:
                subs = []

        if not subs:
            # Heuristic decomposition: split on connectors, else widen by keywords.
            parts = [p.strip() for p in _CONNECTORS.split(q) if p and len(p.strip()) > 8]
            subs = parts if len(parts) > 1 else [q]
            kws = _keywords(q)
            if kws:
                subs.append(" ".join(kws))  # a keyword-only pass favours BM25

        subs = subs[:MAX_SUBQUESTIONS]
        out: dict[str, Any] = {
            "subquestions": subs,
            "trace": [_step("planner", f"{len(subs)} sub-questions", t0, subquestions=subs)],
        }
        if usage_update:
            out["usage"] = _add_usage(state, usage_update)
        return out

    return planner


# --------------------------------------------------------------------------- #
# researcher
# --------------------------------------------------------------------------- #
def make_researcher(svc: RAGService):
    def researcher(state: AgentState) -> dict:
        t0 = time.perf_counter()
        top_k = state.get("top_k", 8)
        subs = state.get("subquestions") or [state["question"]]

        # Dedup across sub-question searches; keep the best score per chunk.
        best: dict[str, dict] = {c["chunk_id"]: c for c in (state.get("evidence") or [])}
        per_sub = []
        for sq in subs:
            sr = svc.search(sq, top_k=max(3, top_k // 2))
            per_sub.append({"subquestion": sq, "hits": len(sr.results)})
            for r in sr.results:
                d = r.model_dump()
                prev = best.get(r.chunk_id)
                if prev is None or (d.get("score") or 0) > (prev.get("score") or 0):
                    best[r.chunk_id] = d

        evidence = sorted(best.values(), key=lambda d: d.get("score") or 0, reverse=True)
        evidence = evidence[: max(top_k, 8)]
        return {
            "evidence": evidence,
            "trace": [_step("researcher", f"{len(evidence)} unique passages from "
                            f"{len(subs)} searches", t0, searches=per_sub)],
        }

    return researcher


# --------------------------------------------------------------------------- #
# lookup (single hop)
# --------------------------------------------------------------------------- #
def make_lookup(svc: RAGService):
    def lookup(state: AgentState) -> dict:
        t0 = time.perf_counter()
        sr = svc.search(state["question"], top_k=state.get("top_k", 8))
        evidence = [r.model_dump() for r in sr.results]
        return {
            "evidence": evidence,
            "subquestions": [state["question"]],
            "trace": [_step("lookup", f"{len(evidence)} passages (single hop)", t0)],
        }

    return lookup


# --------------------------------------------------------------------------- #
# synthesizer
# --------------------------------------------------------------------------- #
def make_synthesizer(svc: RAGService):
    def synthesizer(state: AgentState) -> dict:
        t0 = time.perf_counter()
        evidence = state.get("evidence") or []
        if not evidence:
            return {
                "refused": True,
                "refusal_reason": "No passage in the corpus was relevant to this question.",
                "draft": "", "citations": [], "grounded": False,
                "trace": [_step("synthesizer", "no evidence -> refuse", t0)],
            }

        chunks = [RetrievedChunk(**c) for c in evidence]
        ans = svc.generator.answer(state["question"], chunks, mode="grounded")
        return {
            "draft": ans.answer,
            "citations": [c.model_dump() for c in ans.citations],
            "refused": ans.refused,
            "refusal_reason": ans.refusal_reason,
            "usage": _add_usage(state, ans.usage),
            "trace": [_step("synthesizer",
                            f"{len(ans.citations)} verified citations"
                            + (" (refused)" if ans.refused else ""),
                            t0, refused=ans.refused)],
        }

    return synthesizer


# --------------------------------------------------------------------------- #
# critic (self-reflection)
# --------------------------------------------------------------------------- #
_CRITIC_SYS = """You audit a drafted answer against the question. Reply ONLY with JSON:
{"grounded": true|false, "critique": "<one sentence>"}

Set grounded=false if the answer fails to address the question, is vague, or
leaves an obvious part of a multi-part question unanswered."""


def make_critic(svc: RAGService):
    def critic(state: AgentState) -> dict:
        t0 = time.perf_counter()
        draft = state.get("draft") or ""
        citations = state.get("citations") or []
        iterations = state.get("iterations", 0) + 1

        # Hard gate first: no verified citation == not grounded, no LLM needed.
        verified = [c for c in citations if c.get("verified")]
        if not draft.strip() or not verified:
            return {
                "grounded": False, "iterations": iterations,
                "critique": "No verified citation supports the draft.",
                "trace": [_step("critic", "ungrounded (no verified citations)", t0,
                                grounded=False, iteration=iterations)],
            }

        grounded, critique = True, "Answer is supported by verified citations."
        usage_update = None
        if svc.llm.available:
            try:
                res = svc.llm.complete(
                    _CRITIC_SYS,
                    f"QUESTION: {state['question']}\n\nDRAFT ANSWER: {draft}",
                    max_tokens=200,
                )
                data = _extract_json(res.text) or {}
                grounded = bool(data.get("grounded", True))
                critique = str(data.get("critique", ""))[:300] or critique
                usage_update = res.usage
            except Exception:
                pass

        out: dict[str, Any] = {
            "grounded": grounded, "critique": critique, "iterations": iterations,
            "trace": [_step("critic", ("accepted" if grounded else "rejected") +
                            f" (iteration {iterations})", t0,
                            grounded=grounded, critique=critique)],
        }
        if usage_update:
            out["usage"] = _add_usage(state, usage_update)
        return out

    return critic


# --------------------------------------------------------------------------- #
# reject
# --------------------------------------------------------------------------- #
def make_reject(svc: RAGService):
    def reject(state: AgentState) -> dict:
        t0 = time.perf_counter()
        return {
            "refused": True,
            "refusal_reason": state.get("route_reason") or "Out of scope for this corpus.",
            "draft": "", "citations": [], "grounded": False,
            "trace": [_step("reject", "refused", t0)],
        }

    return reject
