"""
Document loaders for TruthLens AI ingestion pipeline.

Supports PDF (via pypdf) and DOCX (via python-docx). Each loader returns a
list of page/section-level text blocks with minimal structural metadata,
so downstream chunking can respect natural document boundaries instead of
blindly splitting by character count.
"""
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument


@dataclass
class RawBlock:
    """A single unit of extracted text before chunking."""
    text: str
    source_file: str
    page_number: int | None = None  # 1-indexed for PDFs, None for DOCX paragraphs
    extra: dict = field(default_factory=dict)


def load_pdf(path: str | Path) -> list[RawBlock]:
    """Load a PDF and return one RawBlock per page."""
    path = Path(path)
    reader = PdfReader(str(path))
    blocks: list[RawBlock] = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            blocks.append(RawBlock(text=text, source_file=path.name, page_number=i))

    return blocks


def load_docx(path: str | Path) -> list[RawBlock]:
    """Load a DOCX and return one RawBlock per non-empty paragraph group.

    Paragraphs are merged in groups of ~5 to avoid overly small chunks
    (a single DOCX paragraph is often just a sentence).
    """
    path = Path(path)
    doc = DocxDocument(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    blocks: list[RawBlock] = []
    group_size = 5
    for i in range(0, len(paragraphs), group_size):
        group = paragraphs[i : i + group_size]
        text = "\n".join(group)
        if text.strip():
            blocks.append(RawBlock(text=text, source_file=path.name, page_number=None))

    return blocks


def load_document(path: str | Path) -> list[RawBlock]:
    """Dispatch to the correct loader based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix == ".docx":
        return load_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix} ({path})")
