"""
TruthLens AI — FastAPI entrypoint.

Run locally:
    uvicorn app.api.main:app --reload --port 8000

Endpoints are stubbed now and will be wired to real modules as each
phase (ingestion, retrieval, RAG, verification, trust score) is built.
"""
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from app.utils.config import settings

app = FastAPI(
    title="TruthLens AI",
    description="Hallucination-aware RAG with evidence-based claim verification",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    trust_score: float | None = None
    supported_claims: int | None = None
    total_claims: int | None = None
    sources: list[str] = []


@app.get("/health")
def health():
    return {"status": "ok", "vector_db": settings.vector_db, "model": settings.base_model_name}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # TODO(Phase 3): save to data/raw, trigger ingestion pipeline
    return {"filename": file.filename, "status": "received (ingestion pipeline not yet wired)"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # TODO(Phase 7): retrieve -> generate
    # TODO(Phase 9-11): verify claims -> trust score -> self-correct
    return AskResponse(answer=f"[stub] You asked: {req.question}", sources=[])


@app.get("/retrieve")
def retrieve(query: str, k: int = settings.top_k):
    # TODO(Phase 6): call retriever module
    return {"query": query, "k": k, "results": []}


@app.post("/verify")
def verify(answer: str):
    # TODO(Phase 9): claim extraction + evidence matching
    return {"answer": answer, "claims": [], "unsupported_claims": []}


@app.get("/trust-score")
def trust_score(question: str, answer: str):
    # TODO(Phase 10): compute composite trust score
    return {"question": question, "answer": answer, "trust_score": None}
