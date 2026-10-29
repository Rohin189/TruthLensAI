"""Composite trust score for RAG answers — a heuristic, not a calibrated probability."""
import re
from dataclasses import dataclass

REFUSAL_PATTERNS = [
    r"do(?:es)? not contain",
    r"cannot answer",
    r"can't answer",
    r"no relevant information",
    r"not covered in",
    r"insufficient information",
    r"not enough information",
    r"i don'?t have (?:enough )?information",
]

REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)
SOURCE_MENTION_RE = re.compile(r"Source\s+(\d+)", re.IGNORECASE)


@dataclass
class TrustScoreResult:
    score: int  # 0-100
    label: str  # "Low" / "Moderate" / "High"
    retrieval_confidence: float  # 0-1, avg similarity of retrieved chunks
    citation_coverage: float  # 0-1, fraction of retrieved sources actually cited
    is_refusal: bool  # whether the answer appears to be a grounded refusal
    num_sources_retrieved: int
    num_sources_cited: int


def label_for_score(score: int) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


def compute_trust_score(answer: str, sources: list[dict]) -> TrustScoreResult:
    """
    sources: list of dicts as returned by Retriever, each with a 'similarity' key.
    """
    num_sources = len(sources)
    is_refusal = bool(REFUSAL_RE.search(answer))

    if num_sources == 0:
        retrieval_confidence = 0.0
    else:
        retrieval_confidence = sum(s["similarity"] for s in sources) / num_sources

    cited_indices = {int(m) for m in SOURCE_MENTION_RE.findall(answer)}
    valid_cited = {i for i in cited_indices if 1 <= i <= num_sources}
    num_cited = len(valid_cited)
    citation_coverage = (num_cited / num_sources) if num_sources > 0 else 0.0

    if is_refusal:
        # A refusal is only "trustworthy" if it's ALSO backed by weak retrieval —
        # i.e. the system is refusing because there's genuinely nothing relevant.
        # Score reflects "how justified is this refusal", capped low regardless.
        score = round(15 + 10 * (1 - retrieval_confidence))
        score = min(score, 30)
    else:
        # Weighted blend: retrieval quality matters more than citation formatting,
        # but an answer that cites nothing despite having sources is a red flag.
        score = round(100 * (0.65 * retrieval_confidence + 0.35 * citation_coverage))

    score = max(0, min(100, score))

    return TrustScoreResult(
        score=score,
        label=label_for_score(score),
        retrieval_confidence=round(retrieval_confidence, 4),
        citation_coverage=round(citation_coverage, 4),
        is_refusal=is_refusal,
        num_sources_retrieved=num_sources,
        num_sources_cited=num_cited,
    )