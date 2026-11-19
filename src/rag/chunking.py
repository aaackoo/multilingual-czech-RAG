"""
Chunking: split raw document text into overlapping windows.

Design choices, both deliberate:

- Character count, not tokens. Token boundaries are tokenizer-specific
  (OpenAI's cl100k_base differs from Cohere's or Anthropic's tokenizer), so
  chunking by token count would silently couple this stage to one vendor's
  tokenizer -- a bad fit for a system whose whole point is provider
  agnosticism. Characters are a good-enough, provider-neutral proxy for
  "how much text fits in a chunk."
- Paragraph-first, recursive splitting. We prefer to cut on natural
  boundaries (blank lines, then sentences) rather than truncating
  mid-sentence: retrieval quality depends on each chunk being a coherent,
  self-contained unit of meaning, which both BM25 and embedding models
  represent better than an arbitrary character slice.
- Overlap. Consecutive chunks repeat a trailing slice of text so a fact
  stated right at a chunk boundary isn't only ever visible to one chunk.
"""

from dataclasses import dataclass

# Tried in order, most to least "natural" -- paragraph breaks first, falling
# back to sentence ends, then plain whitespace, only for pieces still too
# long after the previous separator.
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def _split_recursive(text: str, max_len: int, separators: list[str]) -> list[str]:
    """Break `text` into pieces no longer than max_len, preferring earlier
    (more natural) separators over later ones."""
    if len(text) <= max_len:
        return [text]
    if not separators:
        # No separator left to try -- hard cut. Only triggers on pathological
        # input (e.g. one very long word); never happens on our sample corpus.
        return [text[i : i + max_len] for i in range(0, len(text), max_len)]

    sep, *rest = separators
    pieces: list[str] = []
    for part in text.split(sep):
        if len(part) <= max_len:
            pieces.append(part)
        else:
            pieces.extend(_split_recursive(part, max_len, rest))
    return pieces


def _merge_with_overlap(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack small pieces into ~chunk_size windows. When a window is
    full, start the next one by repeating enough trailing pieces from the
    current window to cover `overlap` characters."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        added_len = len(piece) + (1 if current else 0)  # +1 for the joining space
        if current and current_len + added_len > chunk_size:
            chunks.append(" ".join(current))

            overlap_pieces: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                overlap_pieces.insert(0, p)
                overlap_len += len(p)
                if overlap_len >= overlap:
                    break
            current = overlap_pieces
            current_len = sum(len(p) for p in current) + max(0, len(current) - 1)

        current.append(piece)
        current_len += added_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> list[Chunk]:
    text = text.strip()
    if not text:
        return []
    pieces = _split_recursive(text, chunk_size, SEPARATORS)
    merged = _merge_with_overlap(pieces, chunk_size=chunk_size, overlap=overlap)
    return [Chunk(text=t, source=source, chunk_index=i) for i, t in enumerate(merged)]
