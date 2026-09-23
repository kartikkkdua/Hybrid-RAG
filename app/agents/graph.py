"""The multi-agent graph.

                        ┌──────────┐
      question ───────▶ │  router  │──── reject ───────────────┐
                        └────┬─────┘                           │
                   lookup    │    research                     │
              ┌──────────────┘        └────────┐               │
              ▼                                ▼               │
        ┌──────────┐                     ┌──────────┐          │
        │  lookup  │                     │ planner  │          │
        └────┬─────┘                     └────┬─────┘          │
             │                                ▼                │
             │                          ┌────────────┐         │
             │                          │ researcher │         │
             │                          └────┬───────┘         │
             └──────────────┬────────────────┘                 │
                            ▼                                  │
                    ┌───────────────┐                          │
                    │  synthesizer  │                          │
                    └───────┬───────┘                          │
                            ▼                                  │
                     ┌────────────┐   grounded / max iters     │
                     │   critic   │────────────────────────────┴──▶ END
                     └──────┬─────┘
                            │ not grounded
                            ▼
                 escalate: lookup ⇒ planner   (hand-off to the researcher)
                           research ⇒ planner (re-plan, broader sub-questions)

Two patterns a single-agent RAG chain can't express live here:
1. **Routing** — cheap single-hop for simple questions, decomposition for hard ones.
2. **Supervised escalation** — the critic can reject a draft and hand a lookup off
   to the research path, bounded by MAX_ITERATIONS so it always terminates.
"""
from __future__ import annotations

import time
from typing import Optional

from langgraph.graph import END, START, StateGraph

from ..models import AgentAnswer, Citation, RetrievedChunk, TraceStep, Usage
from ..service import RAGService
from .nodes import (
    MAX_ITERATIONS,
    make_critic,
    make_lookup,
    make_planner,
    make_reject,
    make_researcher,
    make_synthesizer,
)
from .state import AgentState


def _route_edge(state: AgentState) -> str:
    route = state.get("route", "lookup")
    if route == "reject":
        return "reject"
    return "planner" if route == "research" else "lookup"


def _critic_edge(state: AgentState) -> str:
    """Finish, or escalate for another bounded attempt."""
    if state.get("grounded") or state.get("iterations", 0) >= MAX_ITERATIONS:
        return END
    return "planner"  # both lookup and research escalate into (re-)planning


def build_graph(svc: RAGService):
    g = StateGraph(AgentState)

    g.add_node("router", make_router_node(svc))
    g.add_node("lookup", make_lookup(svc))
    g.add_node("planner", make_planner(svc))
    g.add_node("researcher", make_researcher(svc))
    g.add_node("synthesizer", make_synthesizer(svc))
    g.add_node("critic", make_critic(svc))
    g.add_node("reject", make_reject(svc))

    g.add_edge(START, "router")
    g.add_conditional_edges("router", _route_edge,
                            {"lookup": "lookup", "planner": "planner", "reject": "reject"})
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "synthesizer")
    g.add_edge("lookup", "synthesizer")
    g.add_edge("synthesizer", "critic")
    g.add_conditional_edges("critic", _critic_edge, {END: END, "planner": "planner"})
    g.add_edge("reject", END)

    return g.compile()


def make_router_node(svc: RAGService):
    from .nodes import make_router

    return make_router(svc)


class AgentRunner:
    """Wraps the compiled graph and converts final state into an `AgentAnswer`."""

    def __init__(self, svc: RAGService):
        self.svc = svc
        self.graph = build_graph(svc)

    @staticmethod
    def _init(question: str, top_k: int) -> AgentState:
        return {"question": question, "top_k": top_k, "iterations": 0,
                "trace": [], "usage": {}}

    def run(self, question: str, top_k: int = 8) -> AgentAnswer:
        t0 = time.perf_counter()
        final = self.graph.invoke(self._init(question, top_k))
        return self._to_answer(final, round((time.perf_counter() - t0) * 1000, 2))

    def run_streaming(self, question: str, top_k: int = 8):
        """Yield ('node', step) as each agent runs, then ('done', AgentAnswer).

        The graph branches and loops, so there is no token stream to forward —
        what is worth streaming is the *control flow*: the user sees it route,
        decompose, research and self-critique instead of staring at a spinner.
        """
        t0 = time.perf_counter()
        merged: dict = dict(self._init(question, top_k))
        for chunk in self.graph.stream(self._init(question, top_k)):
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                # `trace` uses an append reducer in the graph; merge it by hand
                # here because we are accumulating updates ourselves.
                steps = update.get("trace") or []
                merged["trace"] = (merged.get("trace") or []) + steps
                for k, v in update.items():
                    if k != "trace":
                        merged[k] = v
                for step in steps:
                    yield "node", step
        yield "done", self._to_answer(merged, round((time.perf_counter() - t0) * 1000, 2))

    def _to_answer(self, s: AgentState, wall_ms: float) -> AgentAnswer:
        u = s.get("usage") or {}
        usage = Usage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cost_usd=u.get("cost_usd", 0.0),
            latency_ms=wall_ms,
            model=u.get("model", "agent"),
        )
        citations = [Citation(**c) for c in (s.get("citations") or [])]
        evidence = [RetrievedChunk(**c) for c in (s.get("evidence") or [])]
        trace = [TraceStep(**t) for t in (s.get("trace") or [])]
        refused = bool(s.get("refused")) or not s.get("grounded", False)
        reason = s.get("refusal_reason") or (
            "" if not refused else (s.get("critique") or "Could not ground an answer.")
        )
        # A run that escalated visited the planner after a lookup route.
        escalated = s.get("route") == "lookup" and any(
            t.node == "planner" for t in trace
        )
        return AgentAnswer(
            answer="" if refused else (s.get("draft") or ""),
            citations=citations,
            refused=refused,
            refusal_reason=reason,
            retrieved=evidence,
            usage=usage,
            verification={"citations_total": len(citations),
                          "citations_verified": sum(1 for c in citations if c.verified),
                          "grounded": bool(s.get("grounded"))},
            route=s.get("route", ""),
            route_reason=s.get("route_reason", ""),
            subquestions=s.get("subquestions") or [],
            iterations=s.get("iterations", 0),
            escalated=escalated,
            critique=s.get("critique", ""),
            trace=trace,
        )


_runner: Optional[AgentRunner] = None


def get_runner(svc: RAGService) -> AgentRunner:
    """Cached per-process runner (compiling the graph is not free)."""
    global _runner
    if _runner is None or _runner.svc is not svc:
        _runner = AgentRunner(svc)
    return _runner
