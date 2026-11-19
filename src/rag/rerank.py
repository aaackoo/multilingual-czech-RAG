"""
Stage 4: optional reranking with a local cross-encoder.

Two-stage retrieval rationale: stage 3 (bi-encoder + BM25) is cheap enough
to run over the whole corpus because it scores the query and each document
independently (a bi-encoder embeds them separately, then compares vectors).
That independence is also its weakness -- it can't model fine-grained
interactions between query and document terms. A cross-encoder scores a
(query, document) PAIR jointly in a single forward pass, which is far more
precise, but far too slow to run over an entire corpus. So the standard
pattern -- and what this module implements -- is: retrieve a wide net
cheaply (stage 3, top ~20), then rerank only that shortlist with the
expensive-but-precise model, keeping only the top few for generation.

Runs locally (BAAI/bge-reranker-v2-m3, multilingual incl. Czech) rather than
via an API specifically so this optional stage needs zero extra API keys --
consistent with this demo defaulting to a single OpenAI key. Toggle with
RERANK_ENABLED in .env.
"""

from dataclasses import replace
from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag.config import get_settings
from rag.retrieval import RetrievedChunk


@lru_cache
def _get_model() -> CrossEncoder:
    # Cached at module/process scope: instantiating a CrossEncoder loads
    # ~1-2GB of weights from disk (downloaded from HuggingFace on first use),
    # which should happen once per process, not once per query.
    settings = get_settings()
    return CrossEncoder(settings.rerank_model)


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Re-scores `candidates` with the cross-encoder and returns the top_k,
    re-ranked. `candidates.score` (RRF score) is replaced with the
    cross-encoder score, since the two are on unrelated scales."""
    if not candidates:
        return []
    model = _get_model()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [replace(c, score=float(s)) for c, s in ranked[:top_k]]
