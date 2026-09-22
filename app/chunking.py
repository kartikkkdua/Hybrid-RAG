"""Chunking with exact character offsets.

Every chunk records the absolute `char_start`/`char_end` of its text within the
parent document. That is the anchor the citation layer uses: a model-supplied
quote is located inside the chunk, and the offset is mapped back to the document
so a citation can be verified and highlighted in the original source.

Token counts are estimated (≈4 chars/token) to avoid a hard tokenizer dependency;
this only affects chunk sizing, not correctness.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_CHARS_PER_TOKEN = 4
# Split on paragraph and sentence boundaries, keeping the delimiter with the text.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass
class TextChunk:
    text: str
    char_start: int
    char_end: int
    ordinal: int
    token_estimate: int


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _segments(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of sentence-ish segments over the raw text."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _SENT_SPLIT.finditer(text):
        end = m.start()
        if end > pos:
            spans.append((pos, end))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans or [(0, len(text))]


def chunk_text(text: str, chunk_tokens: int = 350, overlap_tokens: int = 60) -> list[TextChunk]:
    """Greedy sentence-packing into ~`chunk_tokens` windows with token overlap.

    Offsets are preserved exactly: `text[chunk.char_start:chunk.char_end]` equals
    the chunk's text (modulo surrounding whitespace trimmed only at the edges).
    """
    if not text.strip():
        return []

    max_chars = chunk_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN
    spans = _segments(text)

    chunks: list[TextChunk] = []
    cur_start = spans[0][0]
    cur_end = cur_start
    ordinal = 0

    def flush(start: int, end: int, ordinal: int) -> TextChunk | None:
        raw = text[start:end]
        stripped = raw.strip()
        if not stripped:
            return None
        # tighten offsets to the stripped content
        lead = len(raw) - len(raw.lstrip())
        s = start + lead
        e = s + len(stripped)
        return TextChunk(stripped, s, e, ordinal, estimate_tokens(stripped))

    for seg_start, seg_end in spans:
        # would adding this segment overflow the window?
        if seg_end - cur_start > max_chars and cur_end > cur_start:
            ch = flush(cur_start, cur_end, ordinal)
            if ch:
                chunks.append(ch)
                ordinal += 1
            # start next window with overlap looking back from cur_end
            back = max(cur_start, cur_end - overlap_chars)
            cur_start = back
        cur_end = seg_end

    ch = flush(cur_start, cur_end, ordinal)
    if ch:
        chunks.append(ch)

    # Very long single segments (no sentence breaks) get hard-split.
    return _hard_split_oversize(chunks, text, max_chars)


def _hard_split_oversize(
    chunks: list[TextChunk], text: str, max_chars: int
) -> list[TextChunk]:
    out: list[TextChunk] = []
    ordinal = 0
    for ch in chunks:
        if len(ch.text) <= max_chars * 1.5:
            out.append(TextChunk(ch.text, ch.char_start, ch.char_end, ordinal, ch.token_estimate))
            ordinal += 1
            continue
        pos = ch.char_start
        end = ch.char_end
        while pos < end:
            sub_end = min(pos + max_chars, end)
            raw = text[pos:sub_end]
            stripped = raw.strip()
            if stripped:
                lead = len(raw) - len(raw.lstrip())
                s = pos + lead
                out.append(
                    TextChunk(stripped, s, s + len(stripped), ordinal, estimate_tokens(stripped))
                )
                ordinal += 1
            pos = sub_end
    return out


def normalize_extracted_text(text: str) -> str:
    """Repair the whitespace of PDF-extracted text.

    Many PDFs extract with a newline between *every* token, e.g.
    ``"Experiment\\n \\n-\\n \\n7\\n \\nAim:"``. Left alone that wrecks chunking
    (no real paragraph boundaries) and dilutes BM25 tokenisation. We detect that
    pathological case by newline density and collapse aggressively; otherwise we
    apply the normal rule: blank lines are paragraph breaks, single newlines are
    soft wraps and become spaces.
    """
    if not text:
        return text
    newline_ratio = text.count("\n") / max(len(text), 1)
    if newline_ratio > 0.08:
        # One-token-per-line extraction: paragraph structure is already lost.
        return re.sub(r"\s+", " ", text).strip()
    # Normal layout: protect blank-line paragraph breaks, then unwrap soft breaks.
    text = re.sub(r"[ \t]*\n[ \t]*\n[ \t\n]*", "\n\n", text)
    text = re.sub(r"(?<!\n)[ \t]*\n[ \t]*(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_pdf_text(path: str) -> str:
    """Extract text from a PDF, one normalized block per page.

    Pages are joined with blank lines so the paragraph splitter treats page
    breaks as boundaries.
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [normalize_extracted_text(p.extract_text() or "") for p in reader.pages]
    return "\n\n".join(p for p in pages if p)
