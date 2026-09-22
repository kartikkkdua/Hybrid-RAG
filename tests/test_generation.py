from app.generation import _extract_json, _RawAnswer, _verify_citations, locate_span
from app.models import RetrievedChunk


def test_locate_span_exact_and_whitespace_tolerant():
    hay = "The constant k is commonly set to sixty in RRF."
    start = hay.index("commonly set to sixty")
    assert locate_span(hay, "commonly set to sixty") == (start, start + len("commonly set to sixty"))
    # newline / multiple spaces in the needle still match
    span = locate_span(hay, "commonly   set\nto sixty")
    assert span is not None and hay[span[0]:span[1]] == "commonly set to sixty"
    assert locate_span(hay, "not present here") is None


def test_extract_json_from_fenced_and_noisy_output():
    assert _extract_json('```json\n{"answerable": true, "answer": "x"}\n```')["answer"] == "x"
    assert _extract_json('sure! {"answer": "y", "answerable": false} done')["answer"] == "y"
    assert _extract_json("no json at all") is None


def test_verify_citations_maps_absolute_offsets():
    chunk = RetrievedChunk(
        chunk_id="c1", doc_id="d1", source="doc.md",
        text="Dense retrieval encodes text into vectors.",
        char_start=100, char_end=142, ordinal=0,
    )
    raw = _RawAnswer(
        answerable=True,
        answer="It uses vectors.",
        citations=[
            {"claim": "uses vectors", "chunk_id": "c1", "quote": "encodes text into vectors"},
            {"claim": "hallucinated", "chunk_id": "c1", "quote": "this text is not present"},
        ],
    )
    cits, stats = _verify_citations(raw, [chunk])
    assert stats["citations_total"] == 2
    assert stats["citations_verified"] == 1
    ok = [c for c in cits if c.verified][0]
    # absolute offset = chunk.char_start + local index of the quote
    assert ok.char_start == 100 + chunk.text.index("encodes text into vectors")
    assert ok.quote == "encodes text into vectors"


def test_extractive_answer_without_llm(service):
    service.ingest_text(
        "BM25 is a sparse lexical ranking function used for keyword retrieval.",
        source="doc.md",
    )
    ans = service.answer("what is BM25?", top_k=2, mode="extractive")
    assert ans.refused is False
    assert ans.citations and ans.citations[0].verified
    assert ans.usage.model == "extractive-fallback"
