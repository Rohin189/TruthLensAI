"""
Document-level metadata extraction for TruthLens AI.

Best-effort extraction of paper metadata (title, arXiv ID, year) from the
first page of a research-paper PDF. This is heuristic, not a full parser —
good enough to populate citation/source display in the frontend without
needing a dedicated metadata-extraction model.
"""
import re
from dataclasses import dataclass

ARXIV_ID_PATTERN = re.compile(r"arXiv:(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class DocumentMetadata:
    source_file: str
    title: str | None = None
    arxiv_id: str | None = None
    year: str | None = None


def extract_title(first_page_text: str) -> str | None:
    """Heuristic: the title is usually the first non-empty line that isn't
    an arXiv footer, and is reasonably short (not a full sentence/paragraph).
    """
    lines = [l.strip() for l in first_page_text.splitlines() if l.strip()]
    for line in lines:
        if ARXIV_ID_PATTERN.search(line):
            continue
        word_count = len(line.split())
        if 2 <= word_count <= 25:
            return line
    return None


def extract_metadata(first_page_text: str, source_file: str) -> DocumentMetadata:
    arxiv_match = ARXIV_ID_PATTERN.search(first_page_text)
    year_match = YEAR_PATTERN.search(first_page_text)

    return DocumentMetadata(
        source_file=source_file,
        title=extract_title(first_page_text),
        arxiv_id=arxiv_match.group(1) if arxiv_match else None,
        year=year_match.group(0) if year_match else None,
    )
