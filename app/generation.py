"""Grounded generation with verified inline citations.

The LLM is constrained to a JSON contract:

    {"answerable": bool,
     "answer": str,
     "citations": [{"claim": str, "chunk_id": str, "quote": str}]}

We then *independently verify* every citation: the verbatim `quote` must be
locatable inside the cited chunk. When it is, we map it to an absolute
`char_start`/`char_end` in the source document (so the UI can highlight it) and
mark the citation `verified`. Unverifiable quotes are dropped. If nothing can be
grounded, we refuse rather than hallucinate.

This module also provides an extractive fallback used when no API key is present,
so the pipeline demonstrably works end-to-end without an LLM.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, ValidationError

from .config import Settings
from .stores.base import Store
from .llm import LLMClient
from .models import Answer, Citation, RetrievedChunk, Usage

MAX_QUOTE_CHARS = 400

SYSTEM_PROMPT = """You are a precise question-answering assistant. Answer ONLY using the \
numbered CONTEXT passages provided. Never use outside knowledge.

Rules:
- If the context does not contain enough information to answer, set "answerable" \
to false and leave "answer" as a short explanation of what is missing.
- Every factual sentence in "answer" must be supported by a citation.
- Each citation's "quote" MUST be copied verbatim (exact substring) from the cited \
passage's text — do not paraphrase the quote.
- "chunk_id" must be one of the provided passage ids.

Return ONLY a JSON object, no prose, with this exact shape:
{"answerable": true, "answer": "...", "citations": [{"claim": "...", "chunk_id": "...", "quote": "..."}]}
"""


class _RawCitation(BaseModel):
    claim: str = ""
    chunk_id: str
    quote: str


class _RawAnswer(BaseModel):
    answerable: bool = True
    answer: str = ""
    citations: list[_RawCitation] = []


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, ch in enumerate(chunks, start=1):
        blocks.append(f"[Passage {i}] (chunk_id={ch.chunk_id}, source={ch.source})\n{ch.text}")
    return "\n\n".join(blocks)


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort: parse the first balanced JSON object in the text."""
    text = text.strip()
    # strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth, start = 0, None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def locate_span(haystack: str, needle: str) -> Optional[tuple[int, int]]:
    """Find `needle` in `haystack`, tolerant of whitespace and case differences.

    Returns (start, end) offsets in the ORIGINAL haystack, or None.
    """
    needle = needle.strip()
    if not needle:
        return None
    idx = haystack.find(needle)
    if idx >= 0:
        return idx, idx + len(needle)
    # whitespace/case-tolerant: build a regex where whitespace runs match \s+
    parts = re.split(r"\s+", needle)
    pattern = r"\s+".join(re.escape(p) for p in parts if p)
    if not pattern:
        return None
    m = re.search(pattern, haystack, flags=re.IGNORECASE)
    if m:
        return m.start(), m.end()
    return None


def _verify_citations(
    raw: _RawAnswer, chunks: list[RetrievedChunk]
) -> tuple[list[Citation], dict]:
    by_id = {c.chunk_id: c for c in chunks}
    verified: list[Citation] = []
    n_total = len(raw.citations)
    n_ok = 0
    for rc in raw.citations:
        ch = by_id.get(rc.chunk_id)
        cit = Citation(claim=rc.claim, chunk_id=rc.chunk_id, quote=rc.quote[:MAX_QUOTE_CHARS])
        if ch is not None:
            span = locate_span(ch.text, rc.quote[:MAX_QUOTE_CHARS])
            if span is not None:
                local_start, local_end = span
                cit.source = ch.source
                cit.char_start = ch.char_start + local_start
                cit.char_end = ch.char_start + local_end
                cit.quote = ch.text[local_start:local_end]
                cit.verified = True
                n_ok += 1
        verified.append(cit)
    stats = {
        "citations_total": n_total,
        "citations_verified": n_ok,
        "verified_ratio": round(n_ok / n_total, 3) if n_total else 0.0,
    }
    return verified, stats


class Generator:
    def __init__(self, db: Store, llm: LLMClient, settings: Settings):
        self.db = db
        self.llm = llm
        self.settings = settings

    def answer(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        model: Optional[str] = None,
        mode: str = "grounded",
    ) -> Answer:
        if not retrieved:
            # Distinguish "nothing indexed yet" from "nothing relevant" — they need
            # very different things from the user.
            empty_corpus = self.db.stats().get("chunks", 0) == 0
            reason = (
                "The corpus is empty — add a document before asking."
                if empty_corpus
                else "No passage in the corpus was relevant to this question."
            )
            return Answer(answer="", refused=True, refusal_reason=reason,
                          retrieved=retrieved)

        if mode == "extractive" or not self.llm.available:
            return self._extractive(query, retrieved)

        context = _format_context(retrieved)
        user = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nReturn the JSON object now."

        raw, usage, parse_note = self._call_with_retry(user, model)
        if raw is None:
            # Could not get valid structured output; degrade to extractive.
            ans = self._extractive(query, retrieved)
            ans.usage = usage
            ans.verification["structured_output"] = parse_note
            return ans

        citations, stats = _verify_citations(raw, retrieved)
        verified = [c for c in citations if c.verified]

        refused = (not raw.answerable) or (len(verified) == 0)
        refusal_reason = ""
        if not raw.answerable:
            refusal_reason = raw.answer or "The context does not contain the answer."
        elif len(verified) == 0:
            refusal_reason = "No answer citation could be verified against the sources."

        stats["structured_output"] = parse_note
        return Answer(
            answer=("" if refused else raw.answer),
            citations=verified,
            refused=refused,
            refusal_reason=refusal_reason,
            retrieved=retrieved,
            usage=usage,
            verification=stats,
        )

    def _call_with_retry(
        self, user: str, model: Optional[str]
    ) -> tuple[Optional[_RawAnswer], Usage, str]:
        """Call the LLM, parse+validate; retry once with a corrective message."""
        total = Usage(model=model or self.settings.gen_model)
        attempt_user = user
        for attempt in range(2):
            res = self.llm.complete(
                SYSTEM_PROMPT, attempt_user, model=model, max_tokens=1200, temperature=0.0
            )
            total.input_tokens += res.usage.input_tokens
            total.output_tokens += res.usage.output_tokens
            total.cost_usd = round(total.cost_usd + res.usage.cost_usd, 6)
            total.latency_ms = round(total.latency_ms + res.usage.latency_ms, 2)
            total.model = res.usage.model
            data = _extract_json(res.text)
            if data is not None:
                try:
                    return _RawAnswer.model_validate(data), total, f"ok (attempt {attempt + 1})"
                except ValidationError as e:
                    err = str(e)[:300]
            else:
                err = "response was not valid JSON"
            attempt_user = (
                user
                + f"\n\nYour previous reply could not be parsed ({err}). "
                "Reply with ONLY the JSON object in the required shape."
            )
        return None, total, f"failed: {err}"

    def _extractive(self, query: str, retrieved: list[RetrievedChunk]) -> Answer:
        """LLM-free fallback: quote the top passage as the answer."""
        top = retrieved[0]
        snippet = top.text.strip()
        snippet = snippet[:500] + ("…" if len(snippet) > 500 else "")
        cit = Citation(
            claim=snippet, chunk_id=top.chunk_id, quote=snippet.rstrip("…").strip(),
            source=top.source, char_start=top.char_start,
            char_end=top.char_start + len(snippet.rstrip("…").strip()), verified=True,
        )
        return Answer(
            answer=snippet,
            citations=[cit],
            refused=False,
            retrieved=retrieved,
            usage=Usage(model="extractive-fallback"),
            verification={"mode": "extractive", "citations_verified": 1, "citations_total": 1},
        )
