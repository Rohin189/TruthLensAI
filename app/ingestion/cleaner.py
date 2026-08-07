"""
Text cleaning for TruthLens AI ingestion pipeline.

Research-paper PDFs are messy: hyphenated line breaks, repeated headers/
footers, page numbers, ligature artifacts, and inconsistent whitespace.
These functions normalize that before chunking.
"""
import re


def dehyphenate(text: str) -> str:
    """Rejoin words split across a line break with a trailing hyphen.
    e.g. "hallucina-\ntion" -> "hallucination"
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace, but preserve paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_common_artifacts(text: str) -> str:
    """Remove standalone page numbers and common arXiv footer patterns."""
    # Standalone numeric lines (likely page numbers)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # arXiv preprint footer, e.g. "arXiv:2401.12345v2 [cs.CL] 3 Jan 2024"
    text = re.sub(r"arXiv:\d{4}\.\d{4,5}(v\d+)?\s*\[[\w.]+\]\s*\d{1,2}\s+\w+\s+\d{4}", "", text)
    return text


def clean_text(text: str) -> str:
    """Apply the full cleaning pipeline in order."""
    text = dehyphenate(text)
    text = strip_common_artifacts(text)
    text = collapse_whitespace(text)
    return text
