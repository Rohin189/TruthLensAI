"""Extract raw text from downloaded PDF files."""
from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(filepath: Path) -> list[dict]:
    """Returns [{page_number, text}, ...] for a single PDF."""
    reader = PdfReader(str(filepath))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page_number": i + 1, "text": text})
    return pages


def load_all_pdfs(raw_dir: Path = Path("data/raw")) -> dict[str, list[dict]]:
    """Returns {filename: pages} for every PDF in raw_dir."""
    results = {}
    pdf_files = list(raw_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process")

    for pdf_path in pdf_files:
        try:
            pages = extract_text_from_pdf(pdf_path)
            results[pdf_path.name] = pages
            print(f"  ok {pdf_path.name}: {len(pages)} pages")
        except Exception as e:
            print(f"  x {pdf_path.name}: failed ({e})")

    return results