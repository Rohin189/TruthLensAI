"""CLI entrypoint: run the full ingestion pipeline."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline()