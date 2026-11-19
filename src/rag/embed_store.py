"""
Stage 2: embed chunks via LiteLLM and upsert them into pgvector.

Batching matters here: embedding one chunk per API call would be slow and
wasteful (network round-trip dominates), so we send chunks to the provider
in batches. `ON CONFLICT (source, chunk_index) DO UPDATE` makes re-running
ingestion idempotent -- rerunning `cli.py ingest` after editing a document
updates its rows instead of duplicating them.
"""

from rag.chunking import Chunk
from rag.db import get_connection
from rag.llm import embed

BATCH_SIZE = 32


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_and_store(chunks: list[Chunk]) -> int:
    """Embeds `chunks` and upserts them into the `chunks` table. Returns the
    number of rows written."""
    written = 0
    with get_connection() as conn:
        for batch in _batched(chunks, BATCH_SIZE):
            vectors = embed([c.text for c in batch])
            with conn.cursor() as cur:
                for chunk, vector in zip(batch, vectors):
                    cur.execute(
                        """
                        INSERT INTO chunks (source, chunk_index, text, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (source, chunk_index)
                        DO UPDATE SET text = EXCLUDED.text,
                                      embedding = EXCLUDED.embedding
                        """,
                        (chunk.source, chunk.chunk_index, chunk.text, vector),
                    )
                    written += 1
    return written
