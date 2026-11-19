-- Enables the pgvector type/operators used for dense (embedding) similarity search.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per chunk. We keep text + metadata alongside the embedding so both
-- the dense (pgvector) and sparse (BM25, built from `text`) retrieval paths
-- can be served from this single table -- no separate search index to keep
-- in sync.
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,          -- filename the chunk came from, e.g. "praha.txt"
    chunk_index INT NOT NULL,      -- position of this chunk within its source document
    text TEXT NOT NULL,
    -- 1536 = OpenAI text-embedding-3-small dimensionality. If EMBEDDING_MODEL
    -- changes to a model with a different dimension, this column (and the
    -- index below) must be recreated to match.
    embedding VECTOR(1536) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, chunk_index)
);

-- HNSW gives fast approximate nearest-neighbor search on cosine distance and
-- is the current recommended pgvector index type (better recall/speed
-- tradeoff than IVFFlat, and unlike IVFFlat needs no "train on existing data"
-- step -- important since we build the index before any rows exist).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source);
