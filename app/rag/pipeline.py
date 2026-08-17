"""Base RAG pipeline: retrieve context, generate grounded answer with citations."""
from app.retriever.retriever import Retriever
from app.rag.llm import generate

SYSTEM_INSTRUCTION = """You are a research assistant answering questions using ONLY the provided sources.

Rules:
- Answer using only information present in the sources below.
- If the sources don't contain enough information to answer, say so clearly.
- When you state a fact, reference which source it came from, e.g. (Source 1).
- Do not use outside knowledge not present in the sources.
"""


def build_prompt(query: str, context: str) -> str:
    return f"""{SYSTEM_INSTRUCTION}

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""


def answer_question(query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> dict:
    retriever = Retriever(similarity_threshold=similarity_threshold)
    chunks = retriever.retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "No relevant information found in the corpus for this question.",
            "sources": [],
        }

    context = retriever.format_context(chunks)
    prompt = build_prompt(query, context)
    answer = generate(prompt)

    return {
        "answer": answer,
        "sources": [
            {
                "title": c.title,
                "arxiv_id": c.arxiv_id,
                "page": c.page_number,
                "section": c.section,
                "similarity": c.similarity_score,
            }
            for c in chunks
        ],
    }