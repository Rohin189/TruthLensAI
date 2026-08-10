"""Pure chunking logic — no I/O here, just text in, chunks out."""

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.5:
                chunk = chunk[: last_period + 1]
                end = start + last_period + 1

        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap

    return chunks