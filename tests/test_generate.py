"""Unit tests for prompt assembly in rag.generate (no LLM calls)."""

from rag.generate import SYSTEM_PROMPT, build_context_block, build_messages
from rag.retrieval import RetrievedChunk


def _chunks(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(id=i, source="doc.txt", chunk_index=i, text=t, score=0.0)
        for i, t in enumerate(texts)
    ]


def test_context_block_numbers_chunks_from_one_in_order():
    block = build_context_block(_chunks("Praha leží na Vltavě.", "Brno je na Moravě."))

    assert block == "[1] Praha leží na Vltavě.\n\n[2] Brno je na Moravě."


def test_context_block_empty():
    assert build_context_block([]) == ""


def test_build_messages_has_system_prompt_and_numbered_context():
    chunks = _chunks("první", "druhý", "třetí")
    messages = build_messages("Kdo napsal Babičku?", chunks)

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    user = messages[1]["content"]
    assert build_context_block(chunks) in user
    assert user.endswith("Otázka: Kdo napsal Babičku?")
    # Citation [n] maps back to chunks[n - 1].
    for n, chunk in enumerate(chunks, start=1):
        assert f"[{n}] {chunk.text}" in user
