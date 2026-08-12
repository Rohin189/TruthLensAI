"""Pure chunking logic — no I/O here, just text in, chunks out."""

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def chunk_text_with_offsets(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[tuple[str, int]]:
    """Same as chunk_text, but also returns each chunk's starting character offset
    in the original text — needed to map chunks back to page/section."""
    if len(text) <= chunk_size:
        return [(text, 0)] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunk_start = start

        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.5:
                chunk = chunk[: last_period + 1]
                end = start + last_period + 1

        if chunk.strip():
            chunks.append((chunk.strip(), chunk_start))
        start = end - overlap

    return chunks