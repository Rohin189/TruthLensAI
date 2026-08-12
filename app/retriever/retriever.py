"""Retriever: clean interface over ChromaDB for semantic search."""
from dataclasses import dataclass

from app.embeddings.embedder import embed_query
from app.vectorstore.chroma_store import get_client, get_or_create_collection


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    title: str
    arxiv_id: str
    authors: str
    page_number: int
    section: str
    similarity_score: float  # 1.0 = identical, 0.0 = unrelated (cosine similarity)


class Retriever:
    def __init__(self, similarity_threshold: float = 0.3):
        """
        similarity_threshold: chunks scoring below this get filtered out.
        0.3 is a permissive default for bge-base cosine similarity — tune this
        once you see real query results in Phase 7.
        """
        self.similarity_threshold = similarity_threshold
        self._client = get_client()
        self._collection = get_or_create_collection(self._client)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        section_filter: str | None = None,
        arxiv_id_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top_k most relevant chunks for a natural language query."""
        query_embedding = embed_query(query)

        where_clause = None
        if section_filter:
            where_clause = {"section": section_filter}
        elif arxiv_id_filter:
            where_clause = {"arxiv_id": arxiv_id_filter}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
        )

        chunks = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]  # cosine distance, lower = more similar

        for i in range(len(ids)):
            similarity = 1 - distances[i]  # convert distance to similarity score

            if similarity < self.similarity_threshold:
                continue

            meta = metadatas[i]
            chunks.append(RetrievedChunk(
                chunk_id=ids[i],
                text=documents[i],
                title=meta.get("title", "Unknown"),
                arxiv_id=meta.get("arxiv_id", "Unknown"),
                authors=meta.get("authors", ""),
                page_number=meta.get("page_number", -1),
                section=meta.get("section", "Unknown"),
                similarity_score=round(similarity, 4),
            ))

        return chunks

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a single context string for the LLM prompt."""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {c.title} (arXiv:{c.arxiv_id}), "
                f"Page {c.page_number}, Section: {c.section}]\n{c.text}"
            )
        return "\n\n".join(parts)