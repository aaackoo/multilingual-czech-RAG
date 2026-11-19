"""
Orchestrates retrieval -> (optional) rerank -> generation into one call.

This is the one place that decides how the earlier stages compose, so
api/main.py and cli.py both call `answer_query()` instead of each
duplicating this wiring themselves.
"""

from dataclasses import dataclass

from rag.config import get_settings
from rag.generate import generate_answer
from rag.rerank import rerank
from rag.retrieval import RetrievedChunk, hybrid_search


@dataclass
class Citation:
    source: str
    chunk_index: int
    snippet: str


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]  # exposed for debugging / API responses


def answer_query(
    question: str,
    top_k: int | None = None,
    sources: list[str] | None = None,
    rerank_enabled: bool | None = None,
) -> AnswerResult:
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    use_rerank = settings.rerank_enabled if rerank_enabled is None else rerank_enabled

    # Always retrieve a wider candidate set than we'll actually use -- the
    # reranker (when enabled) needs a shortlist to sort, not just top_k.
    candidates = hybrid_search(question, top_k=settings.retrieval_candidates, sources=sources)

    if use_rerank:
        final_chunks = rerank(question, candidates, top_k=top_k)
    else:
        final_chunks = candidates[:top_k]

    answer = generate_answer(question, final_chunks)

    # Citations map directly to the chunks actually sent to the LLM, not a
    # post-hoc guess -- what you see is what the model was grounded on.
    citations = [
        Citation(source=c.source, chunk_index=c.chunk_index, snippet=c.text[:160])
        for c in final_chunks
    ]
    return AnswerResult(answer=answer, citations=citations, retrieved_chunks=final_chunks)
