"""ChromaDB persistent vector store — load embeddings, query by similarity."""
import json
import numpy as np
import chromadb
from pathlib import Path

CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "truthlens_papers"

EMBEDDINGS_PATH = Path("data/embeddings/vectors.npy")
METADATA_PATH = Path("data/embeddings/metadata.jsonl")

# Chroma caps how many items you can add in a single call
BATCH_SIZE = 500


def get_client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_or_create_collection(client: chromadb.ClientAPI):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # matches our normalized embeddings
    )


def load_embeddings_and_metadata():
    vectors = np.load(EMBEDDINGS_PATH)
    metadata = []
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    assert len(vectors) == len(metadata), (
        f"Mismatch: {len(vectors)} vectors vs {len(metadata)} metadata rows"
    )
    return vectors, metadata


def build_collection(reset: bool = False):
    """Load all embeddings + metadata into ChromaDB, batched to avoid overload."""
    client = get_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass  # didn't exist yet, fine

    collection = get_or_create_collection(client)

    existing_count = collection.count()
    if existing_count > 0 and not reset:
        print(f"Collection already has {existing_count} items. Use reset=True to rebuild.")
        return collection

    vectors, metadata = load_embeddings_and_metadata()
    print(f"Loading {len(vectors)} vectors into ChromaDB...")

    for start in range(0, len(vectors), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(vectors))

        ids = [metadata[i]["chunk_id"] for i in range(start, end)]
        embeddings = vectors[start:end].tolist()
        documents = [metadata[i]["text"] for i in range(start, end)]

        # Chroma metadata values must be str/int/float/bool — authors list needs joining
        metadatas = []
        for i in range(start, end):
            m = metadata[i]
            metadatas.append({
                "arxiv_id": m["arxiv_id"],
                "title": m["title"],
                "authors": ", ".join(m.get("authors", [])),
                "page_number": m.get("page_number", -1),
                "section": m.get("section", "Unknown"),
                "chunk_index": m.get("chunk_index", -1),
                "source_file": m.get("source_file", ""),
            })

        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        print(f"  Added {end}/{len(vectors)}")

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} items.")
    return collection


def query_collection(query_embedding: list[float], n_results: int = 5) -> dict:
    """Query the collection with a pre-computed embedding vector."""
    client = get_client()
    collection = get_or_create_collection(client)
    return collection.query(query_embeddings=[query_embedding], n_results=n_results)