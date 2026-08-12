"""Orchestrates the full ingestion pipeline: download -> extract -> clean -> chunk,
enriched with page/section/author metadata for the Evidence Viewer."""
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from app.ingestion.metadata import run_all_queries, load_manifest, MANIFEST_PATH
from app.ingestion.loaders import load_all_pdfs
from app.ingestion.cleaner import clean_text, truncate_references_section, sanitize_text
from app.ingestion.chunker import chunk_text_with_offsets
from app.ingestion.structure import (
    build_marked_text,
    get_page_offsets,
    get_section_offsets,
    page_for_offset,
    section_for_offset,
    strip_page_markers,
)

PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    arxiv_id: str
    title: str
    authors: list[str]
    page_number: int
    section: str
    chunk_index: int
    text: str


def run_download_stage():
    print("=== Stage 1: Downloading papers from arXiv ===")
    run_all_queries()


def run_processing_stage(manifest_path: Path = MANIFEST_PATH) -> list[Chunk]:
    print("\n=== Stage 2: Extracting, cleaning, chunking ===")
    manifest = load_manifest()
    filename_to_meta = {
        v["filename"]: (k, v["title"], v.get("authors", []))
        for k, v in manifest.items()
    }

    all_pages = load_all_pdfs()
    all_chunks: list[Chunk] = []

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for filename, pages in all_pages.items():
        arxiv_id, title, authors = filename_to_meta.get(filename, ("unknown", filename, []))

        # build text WITH page markers so we can track page numbers through cleaning
        marked_text = build_marked_text(pages)
        marked_text = truncate_references_section(marked_text)
        marked_text = clean_text(marked_text)
        marked_text = sanitize_text(marked_text)

        page_offsets = get_page_offsets(marked_text)
        section_offsets = get_section_offsets(marked_text)

        # save intermediate cleaned text (markers stripped) for inspection/debugging
        (PROCESSED_DIR / f"{arxiv_id.replace('/', '_')}.txt").write_text(
            strip_page_markers(marked_text), encoding="utf-8"
        )

        for idx, (chunk_str, offset) in enumerate(chunk_text_with_offsets(marked_text)):
            page_num = page_for_offset(offset, page_offsets)
            section = section_for_offset(offset, section_offsets)
            clean_chunk_text = strip_page_markers(chunk_str)

            if not clean_chunk_text:
                continue

            all_chunks.append(Chunk(
                chunk_id=f"{arxiv_id}_{idx}",
                source_file=filename,
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                page_number=page_num,
                section=section,
                chunk_index=idx,
                text=clean_chunk_text,
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