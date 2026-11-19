"""
Stage 3: hybrid retrieval -- dense (pgvector) + sparse (BM25), combined via
Reciprocal Rank Fusion (RRF).

Why hybrid at all: dense embedding search is great at semantic/paraphrase
matches ("kde leží Praha" ~ "Praha se nachází na Vltavě") but can miss exact
keyword/name matches that don't cluster well in embedding space (e.g. rare
proper nouns, numbers, dates). BM25 is the opposite -- great at exact term
overlap, blind to paraphrase. Fusing both ranked lists is cheap and tends to
beat either one alone, which is why it's the standard first move in
production RAG systems before reaching for anything fancier.

Why RRF specifically: it combines ranked lists using only each item's *rank*
in each list (score = sum of 1/(k + rank)), not the raw scores. Dense cosine
similarity and BM25 scores live on totally different, unnormalized scales,
so averaging raw scores would require calibration; RRF sidesteps that
entirely and is a couple of lines of code.
"""

import re
from dataclasses import dataclass, replace

from psycopg import Connection
from rank_bm25 import BM25Okapi

from rag.config import get_settings
from rag.db import get_connection
from rag.llm import embed

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

RRF_K = 60  # standard default from the original RRF paper; low sensitivity to exact value


@dataclass
class RetrievedChunk:
    id: int
    source: str
    chunk_index: int
    text: str
    score: float  # meaning depends on which stage produced this object


def _tokenize(text: str) -> list[str]:
    # Lowercase + regex word split, no stemming. Postgres ships no Czech
    # text-search config out of the box (only e.g. english/german/french/...),
    # so rather than silently mis-stemming Czech words with the wrong
    # language rules, we use plain tokenization -- an honest, if less
    # recall-heavy, baseline. A production system would add a proper Czech
    # stemmer/lemmatizer here.
    return _TOKEN_RE.findall(text.lower())


def _dense_search(
    conn: Connection, query_embedding: list[float], top_k: int, sources: list[str] | None
) -> list[RetrievedChunk]:
    # ::vector cast is required: psycopg's default adapter for a Python list
    # of floats binds it as `double precision[]`, which pgvector's <=>
    # operator doesn't accept without an explicit cast to its own type.
    sql = (
        "SELECT id, source, chunk_index, text, 1 - (embedding <=> %s::vector) AS score FROM chunks"
    )
    params: list = [query_embedding]
    if sources:
        sql += " WHERE source = ANY(%s)"
        params.append(sources)
    sql += " ORDER BY score DESC LIMIT %s"
    params.append(top_k)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        RetrievedChunk(id=r[0], source=r[1], chunk_index=r[2], text=r[3], score=r[4]) for r in rows
    ]


def _sparse_search(
    conn: Connection, query: str, top_k: int, sources: list[str] | None
) -> list[RetrievedChunk]:
    sql = "SELECT id, source, chunk_index, text FROM chunks"
    params: list = []
    if sources:
        sql += " WHERE source = ANY(%s)"
        params.append(sources)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        return []

    # Rebuilt on every call rather than persisted/cached: the demo corpus is
    # tiny (tens of chunks), so this costs microseconds. At real scale you'd
    # maintain this index incrementally (or use Postgres/Elasticsearch full-
    # text search) instead of rescanning the whole table per query.
    corpus_tokens = [_tokenize(r[3]) for r in rows]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(zip(rows, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [
        RetrievedChunk(id=r[0], source=r[1], chunk_index=r[2], text=r[3], score=float(s))
        for r, s in ranked
    ]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = RRF_K, top_k: int | None = None
) -> list[RetrievedChunk]:
    rrf_scores: dict[int, float] = {}
    chunks_by_id: dict[int, RetrievedChunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunks_by_id[chunk.id] = chunk

    fused = sorted(chunks_by_id.values(), key=lambda c: rrf_scores[c.id], reverse=True)
    fused = [replace(c, score=rrf_scores[c.id]) for c in fused]
    return fused[:top_k] if top_k is not None else fused


def hybrid_search(
    query: str, top_k: int | None = None, sources: list[str] | None = None
) -> list[RetrievedChunk]:
    """Dense + sparse search over `query`, fused via RRF.

    `sources`, if given, restricts retrieval to chunks from those filenames
    (metadata filtering) -- applied identically to both the dense and
    sparse legs before fusion, so it's a hard filter, not a soft rerank.
    """
    settings = get_settings()
    top_k = top_k or settings.retrieval_candidates
    query_embedding = embed([query])[0]

    with get_connection() as conn:
        dense = _dense_search(conn, query_embedding, top_k, sources)
        sparse = _sparse_search(conn, query, top_k, sources)

    return reciprocal_rank_fusion([dense, sparse], top_k=top_k)
