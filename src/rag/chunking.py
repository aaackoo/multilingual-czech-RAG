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
  The overlap is counted against chunk_size, which is a hard limit.
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
    # str.split drops the separator. Re-attach its visible part so sentences
    # keep their full stop ("Věta." not "Věta"); whitespace-only separators
    # have nothing visible to keep, and pieces are later joined with spaces.
    kept = sep.rstrip()
    parts = text.split(sep)
    parts = [part + kept for part in parts[:-1]] + parts[-1:]

    pieces: list[str] = []
    for part in parts:
        if len(part) <= max_len:
            pieces.append(part)
        else:
            pieces.extend(_split_recursive(part, max_len, rest))
    return pieces


def _take_from_end(items: list[str], budget: int) -> list[str]:
    """The longest run of trailing `items` that, joined with spaces, fits in
    `budget` characters. Returns [] if not even the last item fits."""
    taken: list[str] = []
    for item in reversed(items):
        if len(" ".join([item, *taken])) > budget:
            break
        taken.insert(0, item)
    return taken


def _overlap_tail(pieces: list[str], budget: int) -> list[str]:
    """Trailing text of a finished chunk to repeat at the start of the next
    one, at most `budget` characters. Whole pieces are preferred; if even the
    last piece is too long, its trailing words are used instead."""
    tail = _take_from_end(pieces, budget)
    if tail:
        return tail
    # Split on single spaces (not any whitespace) so the tail is an exact copy
    # of the original text, line breaks included.
    words = _take_from_end(pieces[-1].split(" "), budget)
    tail_text = " ".join(words).strip()
    return [tail_text] if tail_text else []


def _merge_with_overlap(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack pieces into windows of at most chunk_size characters.
    When a window is full, the next one starts with up to `overlap`
    characters of its trailing text, but only as much as still leaves room
    for the next piece, so chunk_size is a hard limit."""
    chunks: list[str] = []
    current: list[str] = []

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if current and len(" ".join([*current, piece])) > chunk_size:
            chunks.append(" ".join(current))
            # Room left for overlap once the new piece and its joining space
            # are added. Can be zero or negative, meaning no overlap.
            budget = min(overlap, chunk_size - len(piece) - 1)
            current = _overlap_tail(current, budget)
        current.append(piece)

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
