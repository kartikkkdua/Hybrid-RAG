"""Conversational query rewriting (history-aware retrieval).

A multi-turn UI creates a retrieval bug that is easy to miss: the second turn of

    user: What constant does RRF use?
    user: why that value?

retrieves on "why that value" alone, which matches nothing useful. Retrieval has
no memory — only generation sees the conversation.

So before retrieving we *condense* history + follow-up into a standalone query.
The rewritten query is used for RETRIEVAL ONLY; generation still receives the
user's original wording, so the answer addresses what they actually asked.

Rewriting is skipped when the question already stands on its own, which keeps the
common case free (no extra LLM call).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .generation import _extract_json
from .llm import LLMClient

MAX_HISTORY_TURNS = 6
MAX_TURN_CHARS = 400

# Referring expressions that can only resolve against earlier turns.
_ANAPHORA = re.compile(
    r"\b(it|its|it's|that|this|these|those|they|them|their|there|"
    r"the same|the former|the latter|both|instead)\b",
    re.IGNORECASE,
)
# Openings that signal a continuation rather than a fresh question.
_FOLLOWUP_START = re.compile(
    r"^\s*(why|why not|how come|what about|how about|and\b|but\b|so\b|then\b|"
    r"ok\b|okay\b|also\b|compared to)",
    re.IGNORECASE,
)
# A token that carries its own specificity: an acronym (nDCG, BM25, RRF), a
# proper noun, or a number. Their presence means a short question can still
# stand on its own.
_DISTINCTIVE = re.compile(r"[A-Z]{2,}|[a-z]+[A-Z]|\b[A-Z][a-z]{2,}|\d")

_STOP = {
    "what", "which", "does", "do", "did", "is", "are", "was", "were", "the", "a",
    "an", "of", "in", "on", "for", "to", "and", "or", "how", "why", "when", "that",
    "this", "with", "from", "by", "about", "can", "could", "should", "would", "use",
    "used", "its", "it", "be", "have", "has", "you", "your", "me", "my",
}


@dataclass
class Rewrite:
    query: str              # what retrieval should use
    original: str           # what the user typed
    rewritten: bool = False
    method: str = "none"    # none | heuristic | llm
    reason: str = ""


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z0-9-]+", text.lower()) if w not in _STOP]


def _recent(history: list[dict], role: Optional[str] = None) -> list[dict]:
    turns = [t for t in (history or []) if not role or t.get("role") == role]
    return turns[-MAX_HISTORY_TURNS:]


def needs_rewrite(question: str, history: list[dict]) -> tuple[bool, str]:
    """Cheap test for whether the question can stand alone."""
    if not history:
        return False, "no history"
    q = question.strip()
    if _FOLLOWUP_START.match(q):
        return True, "continuation opener"
    if _ANAPHORA.search(q):
        return True, "contains a referring expression"
    # Short questions only need history when nothing in them is distinctive:
    # "why that value?" does, "What does nDCG capture?" does not.
    if len(_content_words(q)) <= 3 and not _DISTINCTIVE.search(q[1:]):
        return True, "too few content words to retrieve on"
    return False, "question is self-contained"


def heuristic_rewrite(question: str, history: list[dict]) -> str:
    """Append salient terms from the most recent user turn.

    Deliberately additive rather than generative: it cannot invent a wrong
    question, and extra key terms help both BM25 and dense retrieval.
    """
    prev_users = _recent(history, "user")
    terms: list[str] = []
    for turn in reversed(prev_users):
        for w in _content_words(turn.get("text", "")):
            if w not in terms:
                terms.append(w)
        if len(terms) >= 6:
            break
    if not terms:
        return question
    return f"{question.strip()} {' '.join(terms[:6])}"


_LLM_SYS = """Rewrite the user's follow-up into a single standalone search query \
that makes sense without the conversation. Keep it short and keep the user's \
terminology. Do not answer it. Reply ONLY with JSON: {"query": "..."}"""


def condense_query(
    question: str,
    history: Optional[list[dict]] = None,
    llm: Optional[LLMClient] = None,
) -> Rewrite:
    history = history or []
    needed, reason = needs_rewrite(question, history)
    if not needed:
        return Rewrite(query=question, original=question, reason=reason)

    if llm is not None and llm.available:
        try:
            convo = "\n".join(
                f"{t.get('role', 'user')}: {str(t.get('text', ''))[:MAX_TURN_CHARS]}"
                for t in _recent(history)
            )
            res = llm.complete(
                _LLM_SYS, f"CONVERSATION:\n{convo}\n\nFOLLOW-UP: {question}",
                max_tokens=150,
            )
            data = _extract_json(res.text) or {}
            q = str(data.get("query", "")).strip()
            if q:
                return Rewrite(query=q, original=question, rewritten=True,
                               method="llm", reason=reason)
        except Exception:
            pass  # fall through to the heuristic

    q = heuristic_rewrite(question, history)
    return Rewrite(query=q, original=question, rewritten=q != question,
                   method="heuristic", reason=reason)
