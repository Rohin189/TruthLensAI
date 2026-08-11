"""Orchestrates embedding generation for all chunks."""
import json
import numpy as np
from pathlib import Path

from app.embeddings.embedder import embed_texts

CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
EMBEDDINGS_DIR = Path("data/embeddings")


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def run_embedding_pipeline(chunks_path: Path = CHUNKS_PATH):
    print(f"Loading chunks from {chunks_path}...")
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]

    print("Generating embeddings...")
    embeddings = embed_texts(texts)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # save vectors as a numpy array — fast to load, compact
    vectors = np.array(embeddings, dtype=np.float32)
    np.save(EMBEDDINGS_DIR / "vectors.npy", vectors)

    # save metadata aligned by index (row i in vectors.npy <-> line i here)
    with (EMBEDDINGS_DIR / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            meta = {k: v for k, v in chunk.items() if k != "text"}
            meta["text"] = chunk["text"]  # keep text too, Phase 5 needs it for storage
            f.write(json.dumps(meta) + "\n")

    print(f"\nSaved {vectors.shape[0]} embeddings (dim={vectors.shape[1]}) to {EMBEDDINGS_DIR}")
    return vectors, chunks