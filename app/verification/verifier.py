"""Evidence-based claim verification: Qwen judge (primary) with an NLI veto (sensitivity check).

Each claim is checked against the source(s) it cites:
  1. Qwen (base weights, adapter disabled) reads the claim and the full cited passage(s)
     and labels it SUPPORTED / CONTRADICTED / NOT_SUPPORTED with a one-line reason.
     This label decides the primary verdict.
  2. An independent NLI cross-encoder scores the same claim. A second verdict is computed
     in which a SUPPORTED label is vetoed when NLI entailment < NLI_VETO_THRESHOLD; it is
     reported as a sensitivity check, not used as the primary result.

History (see docs/phase9_verification_findings.md): NLI alone rejected about half of truly
supported claims (short windows lose context; strict about paraphrase). Qwen + veto looked
best on the 38-claim dev set, but a manual audit of 13 vetoes on real answers found 10 of
them wrong, concentrated on the base model's longer, paraphrased claims, which biased the
Stage 2 vs Stage 3 comparison. The veto was therefore demoted to a sensitivity check.
"""
import re
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from sentence_transformers import CrossEncoder

from app.rag.llm import generate, get_model
from app.verification.claim_extractor import Claim, extract_claims, split_sentences

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
# sensitivity check only; chosen on the 38-claim dev set
NLI_VETO_THRESHOLD = 0.01
WINDOW_SIZES = (1, 2, 3)     # consecutive source sentences scored alongside the whole chunk
BATCH_SIZE = 16

# claim-level verdicts
SUPPORTED = "supported"              # cited source supports the claim
MISATTRIBUTED = "misattributed"      # another retrieved source supports it, the cited one does not
CONTRADICTED = "contradicted"        # judge finds the cited source (or, if uncited, any source) incompatible
UNSUPPORTED = "unsupported"          # no retrieved source supports it
UNCITED_SUPPORTED = "uncited_supported"  # no citation given, but a retrieved source supports it

JUDGE_SUPPORTED, JUDGE_CONTRADICTED, JUDGE_NOT_SUPPORTED = "SUPPORTED", "CONTRADICTED", "NOT_SUPPORTED"
JUDGE_PROMPT = """You are a strict fact-checker. Decide whether the CLAIM is supported by the SOURCE passage.

Use ONLY the source passage, not your own knowledge. The question is given only to clarify what words like "it" or "this" in the claim refer to.

Answer with exactly one label:
SUPPORTED - the source states or clearly implies everything in the claim.
CONTRADICTED - the source states something incompatible with the claim.
NOT_SUPPORTED - the source does not establish the claim (missing, different, or overstated details).

QUESTION: {question}

SOURCE:
{source}

CLAIM: {claim}

Reply in the form:
LABEL: <SUPPORTED|CONTRADICTED|NOT_SUPPORTED>
REASON: <one short sentence>"""
JUDGE_LABEL_RE = re.compile(r"LABEL:\s*(NOT_SUPPORTED|SUPPORTED|CONTRADICTED)")
JUDGE_REASON_RE = re.compile(r"REASON:\s*(.*)")

_nli_model = None


# ---------- NLI ----------

def get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {NLI_MODEL_NAME} on {device}...")
        _nli_model = CrossEncoder(NLI_MODEL_NAME, device=device)
    return _nli_model


def _label_index(name: str) -> int:
    # read from the model config instead of assuming an order
    id2label = get_nli_model().model.config.id2label
    return next(i for i, label in id2label.items() if label.lower() == name)


def normalize_source(text: str) -> str:
    """Undo PDF line wrapping: rejoin hyphenated breaks, collapse whitespace."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def premise_windows(source_text: str) -> list[str]:
    """The whole chunk plus sliding windows of 1-3 sentences."""
    text = normalize_source(source_text)
    sentences = split_sentences(text)
    windows = [text]
    for size in WINDOW_SIZES:
        if size >= len(sentences):
            break
        windows.extend(" ".join(sentences[i:i + size]) for i in range(len(sentences) - size + 1))
    return windows


def nli_entailment(claim: str, sources: list[str]) -> list[float]:
    """Max entailment probability over each source's windows."""
    model = get_nli_model()
    entail_i = _label_index("entailment")
    pairs, owners = [], []
    for s_idx, source in enumerate(sources):
        for window in premise_windows(source):
            pairs.append((window, claim))
            owners.append(s_idx)
    probs = model.predict(pairs, apply_softmax=True, batch_size=BATCH_SIZE, show_progress_bar=False)
    best = [0.0] * len(sources)
    for s_idx, p in zip(owners, probs):
        best[s_idx] = max(best[s_idx], float(p[entail_i]))
    return best


# ---------- Qwen judge ----------

def _generate_with_base_weights(prompt: str) -> str:
    """The judge always uses base Qwen, even when a LoRA adapter is attached for generation."""
    model = get_model()
    try:
        from peft import PeftModel
        ctx = model.disable_adapter() if isinstance(model, PeftModel) else nullcontext()
    except ImportError:
        ctx = nullcontext()
    with ctx:
        return generate(prompt, max_new_tokens=80, temperature=0)


def judge(claim: str, question: str, sources: dict[int, str]) -> tuple[str, str]:
    """sources: {source number: text}. Returns (label, reason)."""
    source_block = "\n\n".join(f"[Source {n}] {normalize_source(t)}" for n, t in sources.items())
    raw = _generate_with_base_weights(JUDGE_PROMPT.format(question=question, source=source_block, claim=claim))
    label = JUDGE_LABEL_RE.search(raw)
    reason = JUDGE_REASON_RE.search(raw)
    # an unparseable reply is treated as not supported rather than guessed
    return (label.group(1) if label else JUDGE_NOT_SUPPORTED), (reason.group(1).strip() if reason else raw.strip()[:200])


@dataclass
class Judgement:
    judge_supported: bool      # primary: the judge's own SUPPORTED label
    supported: bool            # sensitivity: SUPPORTED and not vetoed by NLI
    judge_label: str
    judge_reason: str
    nli_entailment: float      # best NLI entailment among the judged sources
    nli_by_source: dict[int, float]
    vetoed: bool               # judge said SUPPORTED but NLI found the claim absent


def judge_with_veto(claim: str, question: str, sources: dict[int, str]) -> Judgement:
    label, reason = judge(claim, question, sources)
    by_source = dict(zip(sources, nli_entailment(claim, list(sources.values()))))
    entail = max(by_source.values(), default=0.0)
    vetoed = label == JUDGE_SUPPORTED and entail < NLI_VETO_THRESHOLD
    return Judgement(label == JUDGE_SUPPORTED, label == JUDGE_SUPPORTED and not vetoed, label, reason,
                     round(entail, 4), by_source, vetoed)


# ---------- claim verdicts ----------

@dataclass
class ClaimVerdict:
    claim: str
    cited_sources: list[int]
    verdict: str                 # primary (Qwen judge)
    verdict_with_veto: str       # sensitivity check (NLI veto applied)
    judge_label: str             # judge's label on the cited sources (or all sources if uncited)
    judge_reason: str
    nli_entailment: float
    vetoed: bool
    supporting_sources: list[int]  # primary: cited sources if supported, likeliest other source if misattributed
    sentence_index: int


def verify_claim(claim: Claim, question: str, sources: list[str]) -> ClaimVerdict:
    """sources: retrieved chunk texts in prompt order (Source 1 = sources[0])."""
    n = len(sources)
    cited = [c for c in claim.cited_sources if 1 <= c <= n]  # out-of-range citations count as uncited
    others = [i for i in range(1, n + 1) if i not in cited]

    def build(primary: str, with_veto: str, j: Judgement, supporting: list[int]) -> ClaimVerdict:
        return ClaimVerdict(claim.text, claim.cited_sources, primary, with_veto, j.judge_label, j.judge_reason,
                            j.nli_entailment, j.vetoed, supporting, claim.sentence_index)

    if not cited:
        j = judge_with_veto(claim.text, question, {i: sources[i - 1] for i in range(1, n + 1)})

        def uncited(ok: bool) -> str:
            if ok:
                return UNCITED_SUPPORTED
            return CONTRADICTED if j.judge_label == JUDGE_CONTRADICTED else UNSUPPORTED

        return build(uncited(j.judge_supported), uncited(j.supported), j, [])

    j = judge_with_veto(claim.text, question, {c: sources[c - 1] for c in cited})
    alt: Judgement | None = None

    def cited_verdict(ok: bool, use_veto: bool) -> str:
        nonlocal alt
        if ok:
            return SUPPORTED
        if j.judge_label == JUDGE_CONTRADICTED:
            return CONTRADICTED
        if others:
            # the claim may be true but cited to the wrong source; judged lazily, once, for either policy
            if alt is None:
                alt = judge_with_veto(claim.text, question, {i: sources[i - 1] for i in others})
            if alt.supported if use_veto else alt.judge_supported:
                return MISATTRIBUTED
        return UNSUPPORTED

    primary = cited_verdict(j.judge_supported, use_veto=False)
    with_veto = cited_verdict(j.supported, use_veto=True)
    if primary == SUPPORTED:
        supporting = cited
    elif primary == MISATTRIBUTED:
        supporting = [max(alt.nli_by_source, key=alt.nli_by_source.get)]
    else:
        supporting = []
    return build(primary, with_veto, j, supporting)


@dataclass
class AnswerVerification:
    is_refusal: bool
    claims: list[ClaimVerdict] = field(default_factory=list)

    def count(self, *verdicts: str) -> int:
        return sum(c.verdict in verdicts for c in self.claims)

    @property
    def num_claims(self) -> int:
        return len(self.claims)

    @property
    def claim_support_ratio(self) -> float | None:
        """Claims supported by the source they cite. None for refusals (no claims to score)."""
        return self.count(SUPPORTED) / self.num_claims if self.num_claims else None

    @property
    def grounded_ratio(self) -> float | None:
        """Claims supported by any retrieved source, regardless of attribution."""
        if not self.num_claims:
            return None
        return self.count(SUPPORTED, MISATTRIBUTED, UNCITED_SUPPORTED) / self.num_claims

    def summary(self) -> dict:
        verdicts = (SUPPORTED, MISATTRIBUTED, CONTRADICTED, UNSUPPORTED, UNCITED_SUPPORTED)
        return {
            "is_refusal": self.is_refusal,
            "num_claims": self.num_claims,
            **{v: self.count(v) for v in verdicts},
            "claim_support_ratio": None if self.claim_support_ratio is None else round(self.claim_support_ratio, 3),
            "grounded_ratio": None if self.grounded_ratio is None else round(self.grounded_ratio, 3),
            "with_veto": {v: sum(c.verdict_with_veto == v for c in self.claims) for v in verdicts},
        }


def verify_answer(answer: str, question: str, sources: list[str]) -> AnswerVerification:
    """sources: retrieved chunk texts in prompt order (Source 1 = sources[0])."""
    extraction = extract_claims(answer, question)
    if extraction.is_refusal:
        return AnswerVerification(is_refusal=True)
    return AnswerVerification(is_refusal=False, claims=[verify_claim(c, question, sources) for c in extraction.claims])
