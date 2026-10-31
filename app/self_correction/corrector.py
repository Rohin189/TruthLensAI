"""Self-correction (Phase 11): repair an answer whose claims failed verification.

Triggered only when the verifier flags at least one claim as unsupported or contradicted.
Two candidate repairs are built and every candidate (original included) is scored with the
claim-based trust score. An answer is eligible only if at least half of its claim credit is
verified (MIN_SUPPORT_TO_ANSWER) and none is contradicted; the highest-scoring eligible answer wins, ties going to the
least-edited candidate. If no answer is eligible, a refusal is returned instead: declining is
better than a mostly unverified answer, even though a refusal's own score is low when the
sources looked relevant.

  regenerate: same sources, prompt extended with the flagged statements and an instruction
              to drop or correct them; the new answer is re-verified from scratch.
  prune:      delete the sentences that produced flagged claims; the remaining claims keep
              their verdicts, so no new judge calls are needed. If nothing survives, the
              answer becomes a refusal.

Caveat: candidates are chosen by the same judge that flagged them, so the trust score of a
corrected answer is optimistic by construction; corrected answers must be checked by hand.
"""
from dataclasses import dataclass, field, replace

from app.trust_score.trust_score import ClaimTrustResult, compute_claim_trust_score
from app.verification.claim_extractor import split_sentences
from app.verification.verifier import CONTRADICTED, UNSUPPORTED, AnswerVerification, verify_answer

FLAGGED = (UNSUPPORTED, CONTRADICTED)
MIN_SUPPORT_TO_ANSWER = 0.5  # chosen after reviewing the 7 flagged Phase 8 answers; not validated on held-out data
PRUNED_REFUSAL = "The retrieved sources do not contain enough verifiable information to answer this question."

FEEDBACK_TEMPLATE = """A fact-checker reviewed a previous answer to this question and found that these statements are NOT supported by the sources they cite:
{flagged}

Write a new answer that leaves out or corrects these statements. Use only facts stated in the sources and cite each one, e.g. (Source 1). If the sources do not answer the question, say so clearly.

ANSWER:"""


@dataclass
class Candidate:
    strategy: str                 # original / regenerate / prune
    answer: str
    verification: AnswerVerification
    trust: ClaimTrustResult


@dataclass
class CorrectionResult:
    best: Candidate
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def corrected(self) -> bool:
        return self.best.strategy != "original"


def score(verification: AnswerVerification, similarities: list[float]) -> ClaimTrustResult:
    return compute_claim_trust_score([c.verdict for c in verification.claims],
                                     [bool(c.cited_sources) for c in verification.claims],
                                     similarities, verification.is_refusal)


def needs_correction(verification: AnswerVerification) -> bool:
    return not verification.is_refusal and verification.count(*FLAGGED) > 0


def feedback_prompt(prompt: str, verification: AnswerVerification) -> str:
    flagged = "\n".join(f"- {c.claim}" for c in verification.claims if c.verdict in FLAGGED)
    head = prompt.rsplit("ANSWER:", 1)[0]  # build_prompt ends with "ANSWER:"
    return head + FEEDBACK_TEMPLATE.format(flagged=flagged)


def prune(answer: str, verification: AnswerVerification) -> tuple[str, AnswerVerification]:
    bad = {c.sentence_index for c in verification.claims if c.verdict in FLAGGED}
    kept_idx = [i for i in range(len(split_sentences(answer))) if i not in bad]
    kept = [split_sentences(answer)[i] for i in kept_idx]
    new_index = {old: new for new, old in enumerate(kept_idx)}  # kept sentences go one per line
    claims = [replace(c, sentence_index=new_index[c.sentence_index])
              for c in verification.claims if c.sentence_index not in bad]
    if not claims:
        return PRUNED_REFUSAL, AnswerVerification(is_refusal=True)
    return "\n".join(kept), AnswerVerification(is_refusal=False, claims=claims)


def correct(question: str, prompt: str, sources: list[str], similarities: list[float],
            answer: str, verification: AnswerVerification, generate_fn) -> CorrectionResult:
    """prompt: the build_prompt() text that produced `answer`; generate_fn(prompt) -> str."""
    original = Candidate("original", answer, verification, score(verification, similarities))
    if not needs_correction(verification):
        return CorrectionResult(original, [original])

    candidates = [original]
    new_answer = generate_fn(feedback_prompt(prompt, verification))
    new_v = verify_answer(new_answer, question, sources)
    candidates.append(Candidate("regenerate", new_answer, new_v, score(new_v, similarities)))

    pruned_answer, pruned_v = prune(answer, verification)
    candidates.append(Candidate("prune", pruned_answer, pruned_v, score(pruned_v, similarities)))

    return CorrectionResult(select(candidates), candidates)


def select(candidates: list[Candidate]) -> Candidate:
    edit_rank = {"original": 0, "prune": 1, "regenerate": 2}
    # a contradicted claim means the source says the opposite: never show it if a refusal is available
    eligible = [c for c in candidates if not c.trust.is_refusal and c.trust.support >= MIN_SUPPORT_TO_ANSWER
                and c.verification.count(CONTRADICTED) == 0]
    refusals = [c for c in candidates if c.trust.is_refusal]
    pool = eligible or refusals or candidates
    return max(pool, key=lambda c: (c.trust.score, -edit_rank[c.strategy]))
