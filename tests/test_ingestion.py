"""Basic sanity tests for the ingestion pipeline (no network calls)."""
from app.ingestion.cleaner import clean_text, truncate_references_section
from app.ingestion.chunker import chunk_text


def test_clean_text_removes_page_numbers():
    raw = "Some text\n\n42\n\nMore text"
    cleaned = clean_text(raw)
    assert "42" not in cleaned.split("\n")


def test_truncate_references_section():
    text = "Body content here.\n\nReferences\n[1] Some citation."
    result = truncate_references_section(text)
    assert "References" not in result
    assert "Body content" in result


def test_chunk_text_respects_overlap():
    text = "a" * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert len(chunks) >= 2
    assert all(len(c) <= 800 for c in chunks)