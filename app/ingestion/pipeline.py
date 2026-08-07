"""
Ingestion pipeline orchestrator for TruthLens AI.

load_document() -> clean_text() -> chunk_text() -> attach DocumentMetadata
-> persist to data/chunks/<file_stem>.jsonl

Each line in the output JSONL is one chunk ready for the embedding stage
(Phase 4), with all metadata needed for citation display and trust scoring
already attached.
"""
import json
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from app.ingestion.chunker import chunk_text
from app.ingestion.cleaner import clean_text
from app.ingestion.loaders import load_document
from app.ingestion.metadata import extract_metadata


def ingest_file(
    path: str | Path,
    output_dir: str | Path = "data/chunks",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> Path:
    """Run the full ingestion pipeline for a single file and write chunks
    to a JSONL file. Returns the output path.
    """
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {path.name}")
    raw_blocks = load_document(path)
    if not raw_blocks:
        logger.warning(f"No extractable text in {path.name}")
        return output_dir / f"{path.stem}.jsonl"

    # Document-level metadata comes from the first block (first page for PDFs)
    doc_meta = extract_metadata(raw_blocks[0].text, source_file=path.name)
    logger.info(
        f"Metadata for {path.name}: title={doc_meta.title!r}, "
        f"arxiv_id={doc_meta.arxiv_id}, year={doc_meta.year}"
    )

    all_chunks = []
    running_index = 0
    for block in raw_blocks:
        cleaned = clean_text(block.text)
        if not cleaned:
            continue
        block_chunks = chunk_text(
            cleaned,
            source_file=path.name,
            page_number=block.page_number,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_index=running_index,
        )
        for c in block_chunks:
            c.metadata.update(
                {
                    "doc_title": doc_meta.title,
                    "arxiv_id": doc_meta.arxiv_id,
                    "year": doc_meta.year,
                }
            )
        all_chunks.extend(block_chunks)
        running_index += len(block_chunks)

    output_path = output_dir / f"{path.stem}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(all_chunks)} chunks to {output_path}")
    return output_path


def ingest_directory(
    input_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/chunks",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Path]:
    """Run ingestion over every supported file in input_dir."""
    input_dir = Path(input_dir)
    supported = {".pdf", ".docx"}
    files = [f for f in input_dir.glob("*") if f.suffix.lower() in supported]

    if not files:
        logger.warning(f"No supported files found in {input_dir}")
        return []

    outputs = []
    for f in files:
        try:
            outputs.append(
                ingest_file(f, output_dir=output_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            )
        except Exception as e:
            logger.error(f"Failed to ingest {f.name}: {e}")

    return outputs
