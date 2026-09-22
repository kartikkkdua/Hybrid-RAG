from app.chunking import chunk_text


def test_offsets_are_exact():
    text = ("Alpha sentence one. Beta sentence two. Gamma sentence three. "
            "Delta four. Epsilon five. Zeta six. Eta seven. Theta eight.")
    chunks = chunk_text(text, chunk_tokens=8, overlap_tokens=2)
    assert len(chunks) >= 2
    for ch in chunks:
        # the recorded span must reproduce the chunk text verbatim
        assert text[ch.char_start:ch.char_end] == ch.text


def test_chunks_cover_content_and_overlap():
    text = " ".join(f"Sentence number {i} here." for i in range(40))
    chunks = chunk_text(text, chunk_tokens=20, overlap_tokens=5)
    assert len(chunks) > 1
    # ordinals are contiguous
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    # every chunk's text is a real substring at its offset
    for ch in chunks:
        assert text[ch.char_start:ch.char_end] == ch.text


def test_empty_text():
    assert chunk_text("   ", 100, 10) == []


def test_normalize_pathological_pdf_text():
    from app.chunking import normalize_extracted_text

    # pypdf often emits a newline between every token.
    raw = "Experiment\n \n-\n \n7\n \n \nAim:\n \nTo\n \ndeploy\n \nAWS"
    assert normalize_extracted_text(raw) == "Experiment - 7 Aim: To deploy AWS"


def test_normalize_preserves_paragraphs_in_normal_text():
    from app.chunking import normalize_extracted_text

    raw = ("This is a sentence that was soft-wrapped\nacross two lines.\n\n"
           "This is a separate paragraph.")
    out = normalize_extracted_text(raw)
    assert out == ("This is a sentence that was soft-wrapped across two lines.\n\n"
                   "This is a separate paragraph.")
