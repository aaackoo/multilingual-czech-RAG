"""
FastAPI wrapper around the RAG pipeline: POST /query.

Kept intentionally thin -- all real logic lives in rag.pipeline, so the
request handler is the same "one orchestration call" whether it's driven by
this HTTP endpoint or cli.py's `query` subcommand.

Run with:  uv run uvicorn api.main:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

from rag.pipeline import answer_query

app = FastAPI(title="Multilingual RAG demo", version="0.1.0")


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    sources: list[str] | None = None  # optional metadata filter, e.g. ["praha.txt"]
    rerank: bool | None = None  # overrides RERANK_ENABLED for this request


class CitationResponse(BaseModel):
    source: str
    chunk_index: int
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = answer_query(
        request.question,
        top_k=request.top_k,
        sources=request.sources,
        rerank_enabled=request.rerank,
    )
    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationResponse(source=c.source, chunk_index=c.chunk_index, snippet=c.snippet)
            for c in result.citations
        ],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
