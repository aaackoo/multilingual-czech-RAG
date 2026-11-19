"""Unit tests for the pure parts of rag.retrieval (no database, no embeddings)."""

import pytest

from rag.retrieval import RRF_K, RetrievedChunk, _tokenize, reciprocal_rank_fusion


def _chunk(chunk_id: int, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id, source="doc.txt", chunk_index=chunk_id, text=f"t{chunk_id}", score=score
    )


def test_rrf_items_in_both_lists_rank_above_items_in_one():
    dense = [_chunk(1), _chunk(2), _chunk(3)]
    sparse = [_chunk(4), _chunk(5), _chunk(3)]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert fused[0].id == 3
    assert {c.id for c in fused} == {1, 2, 3, 4, 5}


def test_rrf_uses_rank_not_raw_score():
    # Raw scores are on different scales and deliberately misleading.
    dense = [_chunk(1, score=0.2), _chunk(2, score=0.1)]
    sparse = [_chunk(3, score=50.0), _chunk(4, score=99.0)]

    fused = reciprocal_rank_fusion([dense, sparse])

    # Rank 1 in either list beats rank 2 in either list, whatever the raw score.
    assert {c.id for c in fused[:2]} == {1, 3}
    assert {c.id for c in fused[2:]} == {2, 4}


def test_rrf_score_formula_and_top_k():
    fused = reciprocal_rank_fusion([[_chunk(1), _chunk(2)], [_chunk(1)]], top_k=1)

    assert [c.id for c in fused] == [1]
    assert fused[0].score == pytest.approx(2 / (RRF_K + 1))


def test_rrf_ties_keep_first_seen_order():
    # Same rank in different lists gives an equal RRF score; sort is stable.
    fused = reciprocal_rank_fusion([[_chunk(1)], [_chunk(2)]])

    assert [c.id for c in fused] == [1, 2]
    assert fused[0].score == fused[1].score


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert [c.id for c in reciprocal_rank_fusion([[], [_chunk(7)]])] == [7]


def test_tokenize_lowercases_and_keeps_czech_diacritics():
    assert _tokenize("Kde leží Praha? Na Vltavě, 1348!") == [
        "kde",
        "leží",
        "praha",
        "na",
        "vltavě",
        "1348",
    ]
