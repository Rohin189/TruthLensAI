#!/usr/bin/env python3
"""
CLI entrypoint for Phase 3: Document Ingestion.

Usage:
    python scripts/ingest_all.py
    python scripts/ingest_all.py --input data/raw --output data/chunks
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.pipeline import ingest_directory  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs/DOCX into chunked JSONL")
    parser.add_argument("--input", default="data/raw", help="Directory of raw documents")
    parser.add_argument("--output", default="data/chunks", help="Directory for output JSONL files")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    outputs = ingest_directory(
        input_dir=args.input,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"\nIngested {len(outputs)} file(s):")
    for o in outputs:
        print(f"  - {o}")


if __name__ == "__main__":
    main()
