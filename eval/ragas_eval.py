"""RAGAS-style answer metrics using an LLM as judge.

These are opt-in (they cost tokens) and only run when an Anthropic key is set.

- faithfulness      : of the claims in the answer, how many are supported by the
                      retrieved context. Penalises hallucination.
- answer_relevance  : does the answer actually address the question.

Each is a single judged call returning small JSON, so cost stays predictable.
"""
from __future__ import annotations

import json
from typing import Optional

from app.generation import _extract_json
from app.llm import LLMClient

_FAITHFULNESS_SYS = """You are a strict evaluator. Given an ANSWER and the CONTEXT it \
was supposed to be grounded in, break the answer into atomic factual claims and decide \
how many are directly supported by the context. Reply ONLY with JSON: \
{"total_claims": int, "supported_claims": int, "unsupported": ["..."]}"""

_RELEVANCE_SYS = """You are a strict evaluator. Rate from 0.0 to 1.0 how well the ANSWER \
addresses the QUESTION (ignore truthfulness, only relevance). Reply ONLY with JSON: \
{"answer_relevance": float, "reason": "..."}"""


def faithfulness(answer: str, contexts: list[str], llm: LLMClient,
                 model: Optional[str] = None) -> Optional[dict]:
    if not llm.available or not answer.strip():
        return None
    ctx = "\n\n".join(contexts)
    user = f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}\n\nReturn the JSON."
    res = llm.complete(_FAITHFULNESS_SYS, user, model=model or llm.settings.judge_model,
                       max_tokens=500)
    data = _extract_json(res.text) or {}
    total = max(int(data.get("total_claims", 0)), 0)
    supported = max(int(data.get("supported_claims", 0)), 0)
    score = round(supported / total, 3) if total else 0.0
    return {"faithfulness": score, "total_claims": total,
            "supported_claims": supported, "usage": res.usage.model_dump()}


def answer_relevance(question: str, answer: str, llm: LLMClient,
                     model: Optional[str] = None) -> Optional[dict]:
    if not llm.available or not answer.strip():
        return None
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nReturn the JSON."
    res = llm.complete(_RELEVANCE_SYS, user, model=model or llm.settings.judge_model,
                       max_tokens=300)
    data = _extract_json(res.text) or {}
    try:
        score = float(data.get("answer_relevance", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return {"answer_relevance": round(score, 3), "usage": res.usage.model_dump()}
