"""
Small CLI for the RAG pipeline.

    uv run python cli.py ingest
    uv run python cli.py query "Na které řece leží Praha?"

Uses argparse (stdlib) rather than a CLI framework -- two subcommands don't
need one, and it keeps the dependency list honest about what this demo
actually needs.
"""

import argparse

from rag.embed_store import embed_and_store
from rag.ingest import load_and_chunk_directory


def cmd_ingest(args: argparse.Namespace) -> None:
    chunks = load_and_chunk_directory(args.data_dir)
    print(f"Chunked {len(chunks)} pieces from {args.data_dir}, embedding + storing...")
    written = embed_and_store(chunks)
    print(f"Stored/updated {written} chunks in pgvector.")


def cmd_query(args: argparse.Namespace) -> None:
    # Imported lazily so `cli.py ingest` works even before pipeline.py exists.
    from rag.pipeline import answer_query

    result = answer_query(args.question, top_k=args.top_k)

    print(result.answer)
    print("\nZdroje (citations):")
    for c in result.citations:
        print(f"  [{c.source} #{c.chunk_index}] {c.snippet}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multilingual RAG demo CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Load ./data/raw, chunk, embed, and store in pgvector")
    p_ingest.add_argument("--data-dir", default="data/raw")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Ask a question against the ingested corpus")
    p_query.add_argument("question")
    p_query.add_argument("--top-k", type=int, default=None)
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
