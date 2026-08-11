"""Generate embeddings for text chunks using BAAI/bge-base-en-v1.5."""
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32  # safe for 8GB VRAM; lower to 16 if you hit OOM

_model = None  # lazy-loaded singleton, avoid reloading on every call


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {MODEL_NAME} on {device}...")
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def embed_texts(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    """Embed a list of texts, returns list of embedding vectors (as plain lists)."""
    model = get_model()
    # bge models recommend a query prefix for queries, but NOT for documents/passages
    # we're embedding document chunks here, so no prefix needed
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # important: enables cosine similarity via dot product
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single search query. bge models want an instruction prefix for queries."""
    model = get_model()
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    embedding = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
    return embedding.tolist()