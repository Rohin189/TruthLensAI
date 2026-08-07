"""
Chunking for TruthLens AI ingestion pipeline.

Splits cleaned text into overlapping chunks sized for the embedding model's
context window. Uses a simple whitespace-token approximation (good enough
for chunk sizing; the embedding model does its own tokenization later).
Also does lightweight section-heading detection common in research papers
(Abstract, Introduction, Related Work, Methodology, Results, Conclusion, etc.)
so chunk metadata can carry a `section` label for better retrieval.
"""
import re
from dataclasses import dataclass, field
from uuid import uuid4

SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s*)?"
    r"(Abstract|Introduction|Related Work|Background|Methodology|Method|"
    r"Approach|Experiments?|Results?|Discussion|Conclusion|References|"
    r"Appendix|Evaluation|Limitations|Future Work)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page_number: int | None
    section: str | None
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def detect_section(text: str) -> str | None:
    match = SECTION_HEADING_PATTERN.search(text)
    return match.group(1).title() if match else None


def _tokenize_words(text: str) -> list[str]:
    return text.split()


def chunk_text(
    text: str,
    source_file: str,
    page_number: int | None = None,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    start_index: int = 0,
) -> list[Chunk]:
    """Split a block of text into overlapping word-based chunks.

    chunk_size / chunk_overlap are measured in whitespace-split words,
    which approximates ~1.3 tokens/word for most embedding tokenizers.
    """
    words = _tokenize_words(text)
    if not words:
        return []

    chunks: list[Chunk] = []
    step = max(chunk_size - chunk_overlap, 1)
    idx = start_index
    section = detect_section(text)

    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunk_text_str = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=str(uuid4()),
                text=chunk_text_str,
                source_file=source_file,
                page_number=page_number,
                section=section,
                chunk_index=idx,
            )
        )
        idx += 1
        if start + chunk_size >= len(words):
            break

    return chunks
