"""CLI entrypoint: build the persistent ChromaDB collection."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.vectorstore.chroma_store import build_collection

if __name__ == "__main__":
    build_collection(reset=True)