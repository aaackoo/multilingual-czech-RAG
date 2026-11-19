"""Unit tests for rag.chunking (pure logic, no I/O)."""

from itertools import pairwise

import pytest

from rag.chunking import chunk_text


def _paragraphs(*lengths: int) -> str:
    """Paragraphs of distinct letters ('a' * n, 'b' * n, ...) separated by blank lines."""
    return "\n\n".join(chr(ord("a") + i) * n for i, n in enumerate(lengths))


def test_empty_and_whitespace_input_returns_no_chunks():
    assert chunk_text("", "doc.txt", chunk_size=100, overlap=10) == []
    assert chunk_text("   \n\n  ", "doc.txt", chunk_size=100, overlap=10) == []


def test_short_input_is_a_single_chunk():
    chunks = chunk_text("  Praha leží na Vltavě.  ", "praha.txt", chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text == "Praha leží na Vltavě."


def test_source_and_chunk_index_are_set():
    chunks = chunk_text(_paragraphs(60, 60, 60, 60), "brno.txt", chunk_size=130, overlap=0)

    assert len(chunks) > 1
    assert all(c.source == "brno.txt" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_respect_chunk_size_when_paragraphs_are_small():
    text = _paragraphs(*[40] * 20)
    chunks = chunk_text(text, "doc.txt", chunk_size=200, overlap=50)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_splits_on_paragraphs_first():
    # Each paragraph fits on its own, so no paragraph should be cut mid-way.
    text = _paragraphs(70, 70, 70)
    chunks = chunk_text(text, "doc.txt", chunk_size=100, overlap=0)

    for chunk in chunks:
        for word in chunk.text.split(" "):
            assert len(word) == 70 and len(set(word)) == 1


def test_overlap_carries_trailing_text_into_next_chunk():
    text = _paragraphs(*[30] * 8)
    chunks = chunk_text(text, "doc.txt", chunk_size=100, overlap=30)

    assert len(chunks) > 1
    for prev, nxt in pairwise(chunks):
        last_piece = prev.text.split(" ")[-1]
        assert nxt.text.startswith(last_piece)


def test_long_paragraph_falls_back_to_sentences():
    sentence = "Toto je jedna česká věta o historii"  # 35 chars
    paragraph = ". ".join([sentence] * 6) + "."
    chunks = chunk_text(paragraph, "doc.txt", chunk_size=80, overlap=0)

    assert len(chunks) > 1
    assert all(len(c.text) <= 80 for c in chunks)
    # Cut on sentence boundaries, so no sentence is split mid-word.
    for chunk in chunks:
        assert sentence in chunk.text


def test_unbreakable_text_is_hard_cut():
    chunks = chunk_text("x" * 250, "doc.txt", chunk_size=100, overlap=0)
    pieces = [p for c in chunks for p in c.text.split(" ")]

    # No separator applies, so the text is cut into chunk_size slices.
    assert {len(p) for p in pieces} == {100, 50}
    assert chunks[-1].text.endswith("x" * 50)


@pytest.mark.xfail(
    strict=True,
    reason="Known limitation: the overlap carried into a new window is not counted "
    "against chunk_size, so pieces close to chunk_size produce oversized chunks.",
)
def test_chunk_size_is_a_hard_limit_for_large_paragraphs():
    chunks = chunk_text(_paragraphs(500, 500, 500), "doc.txt", chunk_size=800, overlap=150)

    assert all(len(c.text) <= 800 for c in chunks)
