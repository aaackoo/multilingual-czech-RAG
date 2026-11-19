# Multilingual (Czech) RAG

A small, model-agnostic RAG pipeline over Czech documents: hybrid retrieval
(dense + BM25), optional cross-encoder reranking, grounded generation with
citations, and an LLM-as-judge evaluation. It is a portfolio piece that
favours a clear, well-explained pipeline over scale; every module is
commented with *why*, not just *what*.

## Architecture at a glance

```
./data/raw/*.txt,*.pdf
        │  stage 1: chunking.py + ingest.py
        ▼
   chunks (text, source, position)
        │  stage 2: embed_store.py  (LiteLLM embeddings)
        ▼
   Postgres + pgvector  ◄──────────────┐
        │  stage 3: retrieval.py       │
        │  dense (pgvector) + BM25,    │  cli.py / api/main.py
        │  fused with RRF              │  (entry points)
        ▼                              │
   candidate chunks                    │
        │  stage 4: rerank.py          │
        │  local cross-encoder         │
        │  (optional, RERANK_ENABLED)  │
        ▼                              │
   top-k chunks                        │
        │  stage 5: generate.py        │
        │  grounded prompt + LiteLLM   │
        ▼                              │
   answer + citations  ─────────────────┘
        │
        │  stage 6: evaluate.py (offline, against data/golden_eval.json)
        ▼
   faithfulness / context precision / answer relevancy report
```

All LLM and embedding calls go through `src/rag/llm.py` (a LiteLLM
`Router`), the only file you touch to swap providers.

## Setup

Requirements: Python 3.11 (pinned in `.python-version`),
[uv](https://docs.astral.sh/uv/), Docker.

```bash
uv sync                      # create .venv and install locked dependencies
cp .env.example .env         # then set OPENAI_API_KEY, the only required key
docker compose up -d         # start Postgres + pgvector
```

$5 of OpenAI credit covers this demo many times over. The first reranked
query downloads the open-weight `BAAI/bge-reranker-v2-m3` model (1 to 2 GB)
from HuggingFace; no key is needed.

## Running the pipeline

```bash
uv run python cli.py ingest                          # load ./data/raw into pgvector
uv run python cli.py query "Na které řece leží Praha?"

uv run uvicorn api.main:app --reload                 # or serve the API
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "Kdo napsal Babičku?"}'
```

`POST /query` takes `{question, top_k?, sources?, rerank?}` and returns
`{answer, citations}`; each citation is the exact chunk (`source`,
`chunk_index`, snippet) the model was given.

## Running the evaluation

```bash
uv run python evaluate.py
```

Runs every question in `data/golden_eval.json` through the pipeline,
prints per-question and average scores for the three metrics below, and
writes details to `eval_report.json`. Extend the golden set by adding
`{question, ground_truth_answer}` pairs.

## Tests

```bash
uv run pytest                 # unit tests: offline, a few seconds
uv run pytest -m quality -v   # quality gate: live pipeline, real API cost
uv run ruff check . && uv run ruff format --check .
```

Unit tests cover chunking, Reciprocal Rank Fusion, tokenisation and prompt
assembly, with no network, database, API key or model download. The
quality tests (`tests/test_quality.py`) assert that the evaluation averages
stay above baseline thresholds. They take about 3 minutes, need Postgres
with the corpus ingested, and are skipped by a plain `pytest`.

## Continuous integration

`.github/workflows/ci.yml` has two jobs:

- **`build`** (every push and PR): compile check, ruff, unit tests. About
  30 seconds, no external services.
- **`quality-eval`** (manual `workflow_dispatch`): starts Postgres +
  pgvector, ingests the corpus and runs `pytest -m quality` against live
  OpenAI. It is manual because every run costs API credit. It needs an
  `OPENAI_API_KEY` repository secret.

## Key design decisions

**Chunking: character-based, paragraph-first, with overlap.** Text is split
on paragraphs, then sentences, then raw characters only if still too long,
targeting about 800 characters with about 150 of overlap. Characters rather
than tokens, because token boundaries differ per vendor's tokenizer and
would tie this stage to one provider. Overlap carries whole trailing pieces
into the next chunk, so it starts on a natural boundary.

**Embeddings: OpenAI `text-embedding-3-small`, via LiteLLM.** It handles Czech
well. Dedicated multilingual models (`BAAI/bge-m3`, Cohere
`embed-multilingual-v3.0`) are probably a little better on non-English
text, but the demo runs on a single API key. Switching to Cohere means
setting `EMBEDDING_MODEL` and changing `EMBEDDING_DIM` and the `vector(...)`
column in `sql/init.sql` to 1024; nothing else references a provider.

**Hybrid retrieval: dense (pgvector) + BM25 (`rank_bm25`), fused with
Reciprocal Rank Fusion.** Dense search handles paraphrase but can miss exact
names and numbers; BM25 is the opposite, and fusing both usually beats
either. RRF uses only ranks (`score = Σ 1/(60 + rank)`), so the
incomparable cosine and BM25 scales never need calibrating. BM25 runs in
memory because **stock PostgreSQL has no `czech` text-search
configuration**, so `ts_rank` would fall back to `simple` with no stemming
anyway.

**Reranking: optional local cross-encoder (`BAAI/bge-reranker-v2-m3`).**
Two-stage retrieval: stage 3 scores query and document independently,
which is cheap but misses fine-grained interactions. A cross-encoder scores
each pair jointly, which is more precise but too slow for the whole corpus,
so it only reranks stage 3's shortlist. It runs locally, so it needs no
extra API key. Toggle with `RERANK_ENABLED`.

**Generation and citations.** The prompt numbers each chunk, asks the model to
cite those numbers, and to say it doesn't know when the context has no
answer. Groundedness comes from the prompt, not from retrieval alone.
Returned citations are the chunks actually sent to the LLM.

**Model-agnostic routing via LiteLLM.** `src/rag/llm.py` builds one `Router` with
aliases (`generation`, `embedding`, `judge`) that the code uses instead of
provider strings. A commented example adds a second model under the same
alias; the Router pools same-named entries and fails over automatically, so
fallback needs a second API key, not a code change. The Router is built on
first use, so importing the package makes no network calls.

**Evaluation: custom LLM-as-judge, not RAGAS.** Each metric is a short prompt
plus JSON parsing, readable end to end. Three signals:

- *Faithfulness / groundedness.* The judge checks each claim in the answer
  against the retrieved context. This isolates **generation**.
- *Context precision.* The fraction of retrieved chunks the judge
  marks relevant. This isolates **retrieval (+ rerank)**.

  A live run on the sample corpus (18 chunks, 10 documents) scored
  faithfulness 1.00, answer relevancy 1.00 and context precision **0.31**.
  That is not a retrieval bug: with `RETRIEVAL_TOP_K=5` and most questions
  answerable from 1 or 2 chunks, at least 3 of 5 chunks must be irrelevant.
  Growing the corpus from 9 to 18 chunks moved it from 0.26 to 0.31.
  `top_k` is kept as is on purpose, since this illustrates exactly what the
  metric measures, and faithfulness at 1.00 shows generation ignored the
  noise.
- *Answer relevancy.* An LLM-judge score for whether the answer
  addresses the question (truth is faithfulness's job), next to the cosine
  similarity of generated and golden answer embeddings. Sharp disagreement
  is worth a manual look.

An LLM judging an LLM pipeline risks self-preference bias.
`EVAL_JUDGE_MODEL` is separate from `GENERATION_MODEL` so a different model
can reduce, though not remove, that risk.

## Known limitations (by design, for a demo)

- BM25 is rebuilt from a full table scan per query; at scale you would use
  a persisted sparse index (Postgres FTS, Elasticsearch, OpenSearch).
- No Czech stemming or lemmatisation in the sparse path.
- Overlap is not counted against the chunk size, so very long paragraphs
  can yield oversized chunks. This does not occur on the sample corpus and
  is documented by an `xfail` unit test.

## Environment variables

See `.env.example` for all settings and defaults. Only `OPENAI_API_KEY`
is required.
