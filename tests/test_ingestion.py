from app.ingestion.chunker import chunk_text, detect_section
from app.ingestion.cleaner import clean_text, dehyphenate, strip_common_artifacts
from app.ingestion.metadata import extract_metadata


def test_dehyphenate_rejoins_split_word():
    assert dehyphenate("hallucina-\ntion") == "hallucination"


def test_strip_common_artifacts_removes_arxiv_footer():
    text = "Some content\narXiv:2401.12345v2 [cs.CL] 3 Jan 2024\nMore content"
    cleaned = strip_common_artifacts(text)
    assert "arXiv:2401.12345" not in cleaned


def test_strip_common_artifacts_removes_page_number_line():
    text = "Some content\n42\nMore content"
    cleaned = strip_common_artifacts(text)
    assert "\n42\n" not in cleaned


def test_clean_text_full_pipeline():
    raw = "This is a hallucina-\ntion example.\n\n\n\nExtra   spaces."
    cleaned = clean_text(raw)
    assert "hallucination example" in cleaned
    assert "   " not in cleaned


def test_detect_section_finds_heading():
    assert detect_section("Introduction\nSome text here.") == "Introduction"
    assert detect_section("Just a random sentence.") is None


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, source_file="test.pdf", chunk_size=30, chunk_overlap=10)
    assert len(chunks) > 1
    # consecutive chunks should overlap by ~10 words
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-10:] == second_words[:10]


def test_chunk_text_empty_input_returns_empty_list():
    assert chunk_text("", source_file="test.pdf") == []


def test_extract_metadata_finds_arxiv_id_and_title():
    first_page = (
        "TruthLens AI: A Hallucination-Aware Framework\n"
        "Jane Doe, John Smith\n"
        "arXiv:2401.12345v2 [cs.CL] 3 Jan 2024\n"
        "Abstract\nWe propose..."
    )
    meta = extract_metadata(first_page, source_file="paper.pdf")
    assert meta.arxiv_id == "2401.12345"
    assert meta.year == "2024"
    assert "TruthLens AI" in meta.title
