# Multilingual (Czech) RAG

A small, model-agnostic Retrieval-Augmented Generation pipeline over a
corpus of Czech-language documents: hybrid retrieval (dense + BM25), an
optional cross-encoder reranking stage, grounded generation with citations,
and a quality-evaluation script with an LLM-as-judge report.

Built as a portfolio piece — prioritizes a clear, correct, well-explained
pipeline over scale or features. Every module is commented with *why*, not
just *what*.

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

Every LLM and embedding call in every stage goes through `src/rag/llm.py`,
which wraps a LiteLLM `Router`. That's the one file you'd touch to swap
providers.

## Setup

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker.

```bash
# 1. Install dependencies (creates .venv, resolves + locks versions)
uv sync

# 2. Configure environment
cp .env.example .env
# edit .env and set OPENAI_API_KEY — this is the only required key.
# Get one at https://platform.openai.com/api-keys ($5 of credit easily
# covers this whole demo many times over).

# 3. Start Postgres + pgvector
docker compose up -d
```

The first reranker call downloads `BAAI/bge-reranker-v2-m3` (~1-2GB) from
HuggingFace on demand — no key required, it's an open-weight model. This
happens automatically the first time you query with reranking enabled
(the default).

## Running the pipeline

```bash
# Ingest the sample Czech documents in ./data/raw into pgvector
uv run python cli.py ingest

# Ask a question from the CLI
uv run python cli.py query "Na které řece leží Praha?"

# Or serve the API
uv run uvicorn api.main:app --reload
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "Kdo napsal Babičku?"}'
```

`POST /query` accepts `{question, top_k?, sources?, rerank?}` and returns
`{answer, citations}`, where each citation maps to the exact chunk
(`source`, `chunk_index`, text snippet) the model was grounded on.

## Running the evaluation

```bash
uv run python evaluate.py
```

Runs every question in `data/golden_eval.json` through the full pipeline
and prints a per-question + averaged report for three metrics (see below),
plus writes `eval_report.json` with the full detail. Extend the golden set
by adding more `{question, ground_truth_answer}` pairs to that file.

**CI gate.** `tests/test_quality.py` is a thin pytest wrapper around the
same metrics, asserting the aggregate averages stay above baseline
thresholds — for wiring this eval into CI rather than only reading it as a
report. It's marked `quality` and excluded from a bare `pytest` run (real
API calls, ~3 minutes) — run it explicitly:

```bash
uv run pytest -m quality -v
```

## Continuous integration

`.github/workflows/ci.yml` has two jobs:

- **`build`** — runs on every push/PR. `uv sync`, a compile-check of every
  module, and a bare `pytest` run. Free and fast (~30s); touches no
  external services.
- **`quality-eval`** — runs the real pipeline: spins up a Postgres +
  pgvector service container, ingests the sample corpus, and runs
  `pytest -m quality` against live OpenAI calls. This costs real API money
  and a few minutes per run, so it's **manually triggered** (`workflow_dispatch`
  from the Actions tab) rather than on every push — automatically spending
  API credit on every commit is a decision worth making deliberately, not a
  CI default. To enable it: add an `OPENAI_API_KEY` secret under the repo's
  Settings → Secrets and variables → Actions.

## Key design decisions

**Chunking — character-based, paragraph-first, with overlap.** Chunks are
split on paragraph breaks first, falling back to sentences and then raw
characters only when a unit is still too long, targeting ~800 characters
with ~150 characters of overlap. Chunking is done by *character* count, not
tokens, deliberately: token boundaries are tokenizer-specific (OpenAI's
tokenizer differs from Cohere's or Anthropic's), so token-based chunking
would silently couple this stage to one vendor — a poor fit for a
model-agnostic design. Overlap is implemented by carrying the trailing
pieces of one chunk into the start of the next, so it lands on a natural
boundary rather than a fixed character offset.

**Embeddings — OpenAI `text-embedding-3-small`, via LiteLLM.** The corpus
is Czech, so the embedding model needs real multilingual capability.
`text-embedding-3-small` was trained on a broad multilingual corpus and
works well here; it was chosen over dedicated multilingual models like
`BAAI/bge-m3` or Cohere's `embed-multilingual-v3.0` (both of which likely
edge it out specifically on non-English retrieval quality) because this
demo runs on a single API key with no signup friction. If you already have
a Cohere key, swapping `EMBEDDING_MODEL=cohere/embed-multilingual-v3.0` in
`.env` (and updating `EMBEDDING_DIM` + the `vector(...)` column in
`sql/init.sql` to match its 1024 dimensions) is the entire migration — nothing
else in the codebase references a specific provider.

**Hybrid retrieval — dense (pgvector) + BM25 (`rank_bm25`), fused with
Reciprocal Rank Fusion.** Dense embedding search is strong at
semantic/paraphrase matches but can miss exact keyword or name matches that
don't cluster well in embedding space; BM25 is the opposite. Fusing both is
cheap and reliably beats either alone. RRF combines the two ranked lists
using only each item's *rank* (`score = Σ 1/(60 + rank)`), not raw scores —
dense cosine similarity and BM25 scores live on incomparable, unnormalized
scales, so RRF sidesteps a score-calibration problem entirely. BM25 runs
in-memory via `rank_bm25` rather than Postgres full-text search because
**stock PostgreSQL ships no `czech` text-search configuration** (only e.g.
english/german/french/...), so `ts_rank` would silently fall back to
`simple` (no stemming) anyway; `rank_bm25` gives an honest, literal BM25
score instead of a mismatched substitute. At real scale you'd want a
persisted sparse index (Postgres FTS, Elasticsearch, OpenSearch) instead of
rebuilding BM25 from a full table scan per query — noted in `retrieval.py`.

**Reranking — optional, local cross-encoder (`BAAI/bge-reranker-v2-m3`).**
This is the two-stage retrieval pattern: stage 3's bi-encoder + BM25 are
cheap enough to run over the whole corpus because they score the query and
each document independently, but that independence is also their weakness —
they can't model fine-grained query–document interactions. A cross-encoder
scores a (query, document) *pair* jointly in one forward pass, which is far
more precise but too slow to run over an entire corpus — so it only ever
reranks the small shortlist stage 3 already narrowed down. It runs locally
rather than via a rerank API so this optional stage needs zero extra API
keys; toggle with `RERANK_ENABLED` in `.env`.

**Generation & citations.** The prompt numbers each retrieved chunk and
instructs the model to cite the numbers it used, and to say it doesn't know
rather than invent an answer when the context doesn't contain one. This is
the actual mechanism that makes an answer "grounded" — groundedness is a
property of how the prompt asks the model to use retrieved context, not
something retrieval guarantees on its own. Citations returned to the caller
map directly to the chunks that were actually sent to the LLM, not a
post-hoc guess.

**Model-agnostic routing via LiteLLM.** `src/rag/llm.py` builds a single
LiteLLM `Router` with named aliases (`generation`, `embedding`, `judge`)
that the rest of the codebase calls instead of a concrete provider string.
This demo runs on one OpenAI key, but the file includes a commented example
of adding a second model under the same alias (e.g. an Anthropic model
under `generation`) — LiteLLM's Router treats same-named entries as a pool
and retries the next one on failure automatically. That's the actual
fallback/routing mechanism; it only needs a second API key in `.env` to
switch on, not a code change.

**Evaluation — custom lightweight LLM-as-judge, not RAGAS.** Every metric
here is a short, readable prompt plus JSON parsing you can read end to end,
rather than a framework whose default integration expects LangChain-wrapped
models. It measures three complementary things:

- *Faithfulness / groundedness* — the judge breaks the generated answer
  into individual claims and checks each against the retrieved context.
  This isolates the **generation** step: a pipeline can retrieve perfectly
  and still hallucinate on top of good context.
- *Context precision* — the judge marks each retrieved chunk
  relevant/irrelevant to the question; the score is the fraction relevant.
  This isolates the **retrieval (+ rerank)** step, independent of what
  generation did with the chunks it was given.

  On this repo's sample corpus (18 chunks across 10 documents), a live run
  measured faithfulness = 1.00 and answer relevancy = 1.00, but context
  precision = **0.31**. That's not a retrieval bug: with `RETRIEVAL_TOP_K=5`
  and any single question realistically answerable from 1-2 chunks, at
  least 3-4 of the 5 retrieved chunks are structurally guaranteed to be
  "irrelevant" regardless of retrieval quality — the metric is correctly
  detecting that `top_k` is oversized relative to this demo corpus. Doubling
  the corpus from 4 to 10 documents (9→18 chunks) only moved this from 0.26
  to 0.31, which itself is a useful data point: context precision converges
  toward 1.0 as corpus size grows relative to `top_k`, but doing so
  meaningfully would take a corpus an order of magnitude larger than a demo
  needs. Faithfulness staying at 1.00 despite that noise confirms generation
  successfully ignored the irrelevant chunks it was handed. This is kept
  as-is deliberately (rather than lowering `top_k` to make the number look
  better) because it's a genuine, explainable illustration of what context
  precision is for: it would tighten toward 1.0 on its own as the corpus
  grows relative to `top_k`.
- *Answer relevancy* — reported as two independent signals side by side: an
  LLM-judge score for "does the answer address what was asked" (deliberately
  separate from whether it's *true* — that's faithfulness's job), and a
  cheap non-LLM sanity check, cosine similarity between the generated
  answer's embedding and the golden answer's embedding. When the two
  disagree sharply, that's usually worth a manual look.

Worth saying out loud: using an LLM to judge a pipeline that itself calls an
LLM risks the judge favoring the generator's own style ("self-preference
bias"). `EVAL_JUDGE_MODEL` is a separate config knob from `GENERATION_MODEL`
for exactly this reason — pointing it at a different model reduces (not
eliminates) that risk.

## Known limitations (by design, for a demo)

- BM25 is rebuilt from a full table scan on every query — fine at this
  corpus size, not how you'd do it at scale.
- No Czech-specific stemming/lemmatization in the sparse retrieval path
  (see the Postgres FTS note above).
- No automated test suite — correctness is demonstrated by `evaluate.py`
  and the manual verification steps used while building each stage, not
  unit tests, to keep this a lean demo rather than a production system.

## Environment variables

See `.env.example` for the full list with defaults. Only `OPENAI_API_KEY`
is required to run everything end to end.
