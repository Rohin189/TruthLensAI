"""CLI entrypoint: generate embeddings for all chunks."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.embeddings.pipeline import run_embedding_pipeline

if __name__ == "__main__":
    run_embedding_pipeline()  