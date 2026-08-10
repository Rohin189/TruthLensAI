"""Orchestrates the full ingestion pipeline: download -> extract -> clean -> chunk."""
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from app.ingestion.metadata import run_all_queries, load_manifest, MANIFEST_PATH
from app.ingestion.loaders import load_all_pdfs
from app.ingestion.cleaner import clean_text, truncate_references_section, sanitize_text
from app.ingestion.chunker import chunk_text

PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    arxiv_id: str
    title: str
    chunk_index: int
    text: str


def run_download_stage():
    print("=== Stage 1: Downloading papers from arXiv ===")
    run_all_queries()


def run_processing_stage(manifest_path: Path = MANIFEST_PATH) -> list[Chunk]:
    print("\n=== Stage 2: Extracting, cleaning, chunking ===")
    manifest = load_manifest()
    filename_to_arxiv = {v["filename"]: (k, v["title"]) for k, v in manifest.items()}

    all_pages = load_all_pdfs()
    all_chunks: list[Chunk] = []

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for filename, pages in all_pages.items():
        arxiv_id, title = filename_to_arxiv.get(filename, ("unknown", filename))
        full_text = "\n".join(p["text"] for p in pages)
        full_text = truncate_references_section(full_text)
        full_text = clean_text(full_text)
        full_text = sanitize_text(full_text)

        # save intermediate cleaned text for inspection/debugging
        (PROCESSED_DIR / f"{arxiv_id.replace('/', '_')}.txt").write_text(
            full_text, encoding="utf-8"
        )

        for idx, chunk_str in enumerate(chunk_text(full_text)):
            all_chunks.append(Chunk(
                chunk_id=f"{arxiv_id}_{idx}",
                source_file=filename,
                arxiv_id=arxiv_id,
                title=title,
                chunk_index=idx,
                text=chunk_str,
            ))

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNKS_DIR / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c)) + "\n")

    print(f"\nWrote {len(all_chunks)} chunks from {len(all_pages)} documents to {out_path}")
    return all_chunks


def run_full_pipeline():
    run_download_stage()
    return run_processing_stage()