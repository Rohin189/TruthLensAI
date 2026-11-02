"""Full TruthLens pipeline (Stages 3-5): LoRA RAG -> claim verification -> trust score -> self-correction.

    retrieve -> build_prompt -> generate (LoRA adapter if present, greedy)
             -> extract + verify claims (Qwen judge on base weights)
             -> claim-based trust score
             -> if any claim is flagged: self-correct and keep the best candidate
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.rag.llm import attach_lora_adapter, generate, get_model
from app.rag.pipeline import build_prompt
from app.retriever.retriever import Retriever
from app.self_correction.corrector import Candidate, correct, score
from app.verification.verifier import AnswerVerification, verify_answer

DEFAULT_ADAPTER = "models/lora_adapters/qwen25-7b-truthlens-lora"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
MAX_NEW_TOKENS = 512
NO_SOURCES_ANSWER = "No relevant information found in the corpus for this question."


@dataclass
class TruthLensResult:
    question: str
    answer: str
    sources: list[dict]               # metadata + text + similarity, in prompt order
    best: Candidate
    candidates: list[Candidate] = field(default_factory=list)  # original first
    adapter_used: bool = False

    @property
    def corrected(self) -> bool:
        return self.best.strategy != "original"


def load_generator(adapter_path: str | None = DEFAULT_ADAPTER) -> bool:
    """Attach the LoRA adapter if it exists locally. Returns whether it is in use."""
    if adapter_path and (Path(adapter_path) / "adapter_config.json").exists():
        attach_lora_adapter(adapter_path)
        return True
    return False


def run_truthlens(question: str, adapter_path: str | None = DEFAULT_ADAPTER, self_correct: bool = True) -> TruthLensResult:
    adapter_used = load_generator(adapter_path)
    retriever = Retriever(similarity_threshold=SIMILARITY_THRESHOLD)
    chunks = retriever.retrieve(question, top_k=TOP_K)
    sources = [{"title": c.title, "arxiv_id": c.arxiv_id, "page": c.page_number, "section": c.section,
                "similarity": c.similarity_score, "text": c.text} for c in chunks]
    similarities = [s["similarity"] for s in sources]

    if not chunks:
        v = AnswerVerification(is_refusal=True)
        best = Candidate("original", NO_SOURCES_ANSWER, v, score(v, similarities))
        return TruthLensResult(question, NO_SOURCES_ANSWER, sources, best, [best], adapter_used)

    texts = [c.text for c in chunks]
    prompt = build_prompt(question, retriever.format_context(chunks))

    def generate_fn(p: str) -> str:
        # an adapter attached earlier in this process stays on the model; switch it off for base generation
        model = get_model()
        ctx = nullcontext() if adapter_used or not hasattr(model, "disable_adapter") else model.disable_adapter()
        with ctx:
            return generate(p, max_new_tokens=MAX_NEW_TOKENS, temperature=0)

    answer = generate_fn(prompt)
    verification = verify_answer(answer, question, texts)
    if self_correct:
        result = correct(question, prompt, texts, similarities, answer, verification, generate_fn)
        best, candidates = result.best, result.candidates
    else:
        best = Candidate("original", answer, verification, score(verification, similarities))
        candidates = [best]
    return TruthLensResult(question, best.answer, sources, best, candidates, adapter_used)


VERDICT_SEVERITY = ["supported", "uncited_supported", "misattributed", "unsupported", "contradicted"]


def candidate_to_dict(c: Candidate) -> dict:
    """Plain-JSON view of a candidate. Sentences carry their worst verdict, so a viewer can highlight the
    answer without importing the extractor (which loads torch)."""
    from app.verification.claim_extractor import split_sentences
    worst: dict[int, str] = {}
    for cv in c.verification.claims:
        if cv.sentence_index not in worst or (VERDICT_SEVERITY.index(cv.verdict)
                                              > VERDICT_SEVERITY.index(worst[cv.sentence_index])):
            worst[cv.sentence_index] = cv.verdict
    return {
        "strategy": c.strategy,
        "answer": c.answer,
        "trust": asdict(c.trust),
        "claims": [{"claim": cv.claim, "cited_sources": cv.cited_sources, "verdict": cv.verdict,
                    "judge_reason": cv.judge_reason, "sentence_index": cv.sentence_index}
                   for cv in c.verification.claims],
        "sentences": [{"text": s, "verdict": worst.get(i)} for i, s in enumerate(split_sentences(c.answer))],
    }


def result_to_dict(r: TruthLensResult) -> dict:
    return {"question": r.question, "answer": r.answer, "sources": r.sources, "adapter_used": r.adapter_used,
            "best": candidate_to_dict(r.best), "candidates": [candidate_to_dict(c) for c in r.candidates]}
