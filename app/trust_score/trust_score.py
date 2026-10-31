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

# ---------- Phase 10: claim-based trust score ----------
# Built on Phase 9 claim verdicts (app/verification/verifier.py). Still a heuristic: the
# weights are design choices, not fitted, and calibration is untested (Phase 12).

# bge similarities are compressed: off-topic questions top out at ~0.45-0.52 and answerable
# eval questions reach 0.67-0.87 (measured on the 50 Phase 8 items), so rescale the best source.
RELEVANCE_FLOOR = 0.50
RELEVANCE_CEIL = 0.80

# credit per claim verdict: true-but-badly-attributed content earns partial credit
CLAIM_CREDIT = {
    "supported": 1.0,
    "misattributed": 0.75,
    "uncited_supported": 0.75,
    "unsupported": 0.0,
    "contradicted": 0.0,
}
W_SUPPORT, W_CITATION, W_RELEVANCE = 0.60, 0.15, 0.25
CAP_UNSUPPORTED = 69   # an unverifiable claim rules out "High"
CAP_CONTRADICTED = 39  # a contradicted claim means "Low"


@dataclass
class ClaimTrustResult:
    score: int
    label: str
    is_refusal: bool
    retrieval_relevance: float
    support: float | None             # mean claim credit
    citation_precision: float | None  # supported / claims that cite a source
    num_claims: int
    num_supported: int
    num_flagged: int                  # unsupported + contradicted
    cap_applied: str | None

    def summary_line(self) -> str:
        if self.is_refusal:
            return f"Trust Score: {self.score}/100 ({self.label}) - refusal; retrieval relevance {self.retrieval_relevance:.2f}"
        return (f"Trust Score: {self.score}/100 ({self.label}) - "
                f"{self.num_supported}/{self.num_claims} claims supported by their cited source")


def retrieval_relevance(similarities: list[float]) -> float:
    if not similarities:
        return 0.0
    return min(1.0, max(0.0, (max(similarities) - RELEVANCE_FLOOR) / (RELEVANCE_CEIL - RELEVANCE_FLOOR)))


def compute_claim_trust_score(
    verdicts: list[str], cited: list[bool], similarities: list[float], is_refusal: bool
) -> ClaimTrustResult:
    """verdicts / cited: one entry per claim (primary verdict; whether the claim cites a source).
    similarities: query-source cosine similarity of each retrieved source."""
    relevance = retrieval_relevance(similarities)

    if is_refusal or not verdicts:
        # how justified is declining: strong sources make a refusal suspect, weak ones justify it
        score = round(100 * (1 - relevance))
        return ClaimTrustResult(score, label_for_score(score), True, round(relevance, 3),
                                None, None, 0, 0, 0, None)

    n = len(verdicts)
    support = sum(CLAIM_CREDIT[v] for v in verdicts) / n
    num_cited = sum(cited)
    num_supported = sum(v == "supported" for v in verdicts)
    citation_precision = num_supported / num_cited if num_cited else 0.0
    score = round(100 * (W_SUPPORT * support + W_CITATION * citation_precision + W_RELEVANCE * relevance))

    cap = None
    if "contradicted" in verdicts and score > CAP_CONTRADICTED:
        score, cap = CAP_CONTRADICTED, "contradicted claim"
    elif "unsupported" in verdicts and score > CAP_UNSUPPORTED:
        score, cap = CAP_UNSUPPORTED, "unsupported claim"

    flagged = sum(v in ("unsupported", "contradicted") for v in verdicts)
    return ClaimTrustResult(score, label_for_score(score), False, round(relevance, 3), round(support, 3),
                            round(citation_precision, 3), n, num_supported, flagged, cap)
