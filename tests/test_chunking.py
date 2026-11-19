"""Unit tests for rag.chunking (pure logic, no I/O)."""

from itertools import pairwise

import pytest

from rag.chunking import chunk_text


def _paragraphs(*lengths: int) -> str:
    """Paragraphs of distinct letters ('a' * n, 'b' * n, ...) separated by blank lines."""
    return "\n\n".join(chr(ord("a") + i) * n for i, n in enumerate(lengths))


def _words(n: int, word: str = "slovo") -> str:
    return " ".join([word] * n)


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


def test_splits_on_paragraphs_first():
    # Each paragraph fits on its own, so no paragraph should be cut mid-way.
    text = _paragraphs(70, 70, 70)
    chunks = chunk_text(text, "doc.txt", chunk_size=100, overlap=0)

    for chunk in chunks:
        for word in chunk.text.split(" "):
            assert len(word) == 70 and len(set(word)) == 1


def test_long_paragraph_falls_back_to_sentences():
    sentence = "Toto je jedna česká věta o historii."  # 36 chars
    paragraph = " ".join([sentence] * 6)
    chunks = chunk_text(paragraph, "doc.txt", chunk_size=80, overlap=0)

    assert len(chunks) > 1
    # Cut on sentence boundaries, so every chunk is made of whole sentences.
    for chunk in chunks:
        assert chunk.text.replace(sentence, "").strip() == ""


def test_sentence_punctuation_is_kept():
    text = "První věta. Druhá věta. Třetí věta."
    chunks = chunk_text(text, "doc.txt", chunk_size=25, overlap=0)

    assert [c.text for c in chunks] == ["První věta. Druhá věta.", "Třetí věta."]


def test_overlap_carries_whole_trailing_pieces_into_next_chunk():
    text = _paragraphs(*[30] * 8)
    chunks = chunk_text(text, "doc.txt", chunk_size=100, overlap=30)

    assert len(chunks) > 1
    for prev, nxt in pairwise(chunks):
        last_piece = prev.text.split(" ")[-1]
        assert nxt.text.startswith(last_piece + " ")


def test_overlap_from_large_piece_is_a_word_boundary_tail():
    # Paragraphs too large to carry whole: overlap falls back to trailing words.
    text = "\n\n".join(_words(80, w) for w in ("alfa", "beta", "gama"))  # ~400 chars each
    chunks = chunk_text(text, "doc.txt", chunk_size=800, overlap=150)

    assert len(chunks) > 1
    for prev, nxt in pairwise(chunks):
        prev_words, nxt_words = prev.text.split(" "), nxt.text.split(" ")
        carried = next(i for i, w in enumerate(nxt_words) if w != nxt_words[0])
        assert 0 < len(" ".join(nxt_words[:carried])) <= 150
        assert nxt_words[:carried] == prev_words[-carried:]


def test_zero_overlap_carries_nothing():
    chunks = chunk_text(_paragraphs(60, 60, 60), "doc.txt", chunk_size=100, overlap=0)

    assert [c.text for c in chunks] == ["a" * 60, "b" * 60, "c" * 60]


def test_unbreakable_text_is_hard_cut():
    chunks = chunk_text("x" * 250, "doc.txt", chunk_size=100, overlap=0)

    assert [c.text for c in chunks] == ["x" * 100, "x" * 100, "x" * 50]


def test_chunk_size_is_a_hard_limit_for_large_paragraphs():
    chunks = chunk_text(_paragraphs(500, 500, 500), "doc.txt", chunk_size=800, overlap=150)

    assert all(len(c.text) <= 800 for c in chunks)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            " ".join(f"Věta číslo {i} je krátká." for i in range(200)), id="short-sentences"
        ),
        pytest.param(_paragraphs(700, 650, 790, 500), id="large-paragraphs"),
        pytest.param("x" * 3000, id="one-long-word"),
        pytest.param(
            "Krátký odstavec.\n\n"
            + _words(150)
            + "\n\n"
            + "y" * 900
            + "\nŘádek. Další věta. "
            + _words(40, "konec"),
            id="mixed",
        ),
    ],
)
@pytest.mark.parametrize("chunk_size,overlap", [(800, 150), (200, 50), (100, 99)])
def test_no_chunk_exceeds_chunk_size(text, chunk_size, overlap):
    chunks = chunk_text(text, "doc.txt", chunk_size=chunk_size, overlap=overlap)

    assert chunks
    assert all(len(c.text) <= chunk_size for c in chunks)
