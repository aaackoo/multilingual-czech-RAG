"""
Stage 1: load raw documents and split them into chunks.

Supports .txt (read directly) and .pdf (text extracted via pypdf). Both are
normalized to plain text before chunking, so the rest of the pipeline never
needs to know which format a source document came from.

Run directly for a quick, DB-free sanity check:
    uv run python -m rag.ingest
"""

from pathlib import Path

from pypdf import PdfReader

from rag.chunking import Chunk, chunk_text
from rag.config import get_settings


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


_LOADERS = {".txt": _read_txt, ".pdf": _read_pdf}


def load_and_chunk_directory(data_dir: Path | str = "data/raw") -> list[Chunk]:
    settings = get_settings()
    data_dir = Path(data_dir)
    all_chunks: list[Chunk] = []
    for path in sorted(data_dir.iterdir()):
        loader = _LOADERS.get(path.suffix.lower())
        if loader is None:
            continue
        text = loader(path)
        all_chunks.extend(
            chunk_text(
                text,
                source=path.name,
                chunk_size=settings.chunk_size_chars,
                overlap=settings.chunk_overlap_chars,
            )
        )
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_directory()
    print(f"Loaded {len(chunks)} chunks from ./data/raw\n")

    by_source: dict[str, int] = {}
    for c in chunks:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count} chunks")

    print("\nPreview of first chunk:")
    print(f"  source={chunks[0].source} index={chunks[0].chunk_index}")
    print(f"  text={chunks[0].text[:200]!r}")
