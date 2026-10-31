"""Split a RAG answer into atomic, self-contained claims, each carrying its cited sources.

Claims are extracted per sentence so citations map cleanly: every claim inherits the
(Source N) citations of the sentence it came from. Long or compound sentences are
decomposed by Qwen; short ones are used as-is. A decomposition that introduces content
not present in the sentence (or question) is rejected in favour of the sentence itself.
"""
import re
from dataclasses import dataclass, field

from app.finetuning.rag_formatter import content_terms
from app.rag.llm import generate
from app.trust_score.trust_score import REFUSAL_RE

DECOMPOSE_MIN_WORDS = 25     # shorter sentences are kept whole: splitting them distorted definitions
MIN_CLAIM_WORDS = 3
MAX_CLAIMS_PER_SENTENCE = 6
MIN_TERM_OVERLAP = 0.7       # share of a claim's content words that must come from sentence/question

# "(Source 1)", "(Source 1, Source 3)", "Sources 2 and 4", "According to Source 5,"
CITATION_GROUP_RE = re.compile(r"Sources?\s+\d+(?:\s*(?:,|and|&)\s*(?:Source\s+)?\d+)*", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+")
CITATION_PAREN_RE = re.compile(r"\s*\((?:[^()]*?Sources?\s+\d+[^()]*?)\)", re.IGNORECASE)
LEAD_IN_RE = re.compile(r"^(?:According to|As (?:stated|noted|described) in|Based on)\s+Sources?\s+[\d,\sand&]+,?\s*", re.IGNORECASE)
# mid-sentence attributions: ", as defined in Source 1,", " (as noted in Sources 2 and 3)", " in Source 4"
INLINE_ATTRIBUTION_RE = re.compile(
    r",?\s*(?:as\s+(?:\w+\s+)?(?:in|by)|according\s+to|in|from|per|by)\s+Sources?\s+\d+(?:\s*(?:,|and|&)\s*(?:Source\s+)?\d+)*\s*,?",
    re.IGNORECASE,
)

DECOMPOSE_PROMPT = """Break the sentence below into short, independent factual claims.

Rules:
- Each claim states one fact and must make sense on its own.
- Keep definitions, conditions, and qualifiers together with what they describe. Never split a phrase so that its meaning changes.
- Do not create claims that only restate the subject or add no information.
- Replace pronouns and vague references ("it", "this method", "they") with what they refer to, using the question for context.
- Use only information in the sentence. Do not add, infer, or explain anything.
- Output one claim per line, each starting with "- ". No other text.

Example:
SENTENCE: LoRA freezes the pretrained weights and trains low-rank matrices that cannot exceed rank r, which cuts trainable parameters by up to 10,000 times on GPT-3.
CLAIMS:
- LoRA freezes the pretrained weights.
- LoRA trains low-rank matrices that cannot exceed rank r.
- LoRA cuts trainable parameters by up to 10,000 times on GPT-3.

QUESTION: {question}
SENTENCE: {sentence}

CLAIMS:"""


@dataclass
class Claim:
    text: str
    cited_sources: list[int]          # 1-based source numbers from the parent sentence
    sentence: str                     # parent sentence (citations stripped)
    sentence_index: int
    decomposed: bool = False          # True if Qwen split the sentence


@dataclass
class ExtractionResult:
    is_refusal: bool
    claims: list[Claim] = field(default_factory=list)


def is_refusal_answer(answer: str) -> bool:
    # a refusal phrase alone is not enough: correct answers can *describe* a limitation
    # ("LLMs cannot answer questions about...") while still citing sources
    return bool(REFUSAL_RE.search(answer)) and not CITATION_GROUP_RE.search(answer)


def parse_citations(text: str) -> list[int]:
    nums = set()
    for group in CITATION_GROUP_RE.findall(text):
        nums.update(int(n) for n in NUMBER_RE.findall(group))
    return sorted(nums)


def strip_citations(text: str) -> str:
    text = LEAD_IN_RE.sub("", text.strip())
    text = CITATION_PAREN_RE.sub("", text)
    text = INLINE_ATTRIBUTION_RE.sub(" ", text)
    text = re.sub(r"\*\*|__|`", "", text)          # markdown emphasis
    text = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s+", "", text)  # list markers
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "etc.", "vs.", "Fig.", "Eq.", "approx.", "resp.")
_DOT = "․"  # placeholder so abbreviation dots are not read as sentence ends


def split_sentences(answer: str) -> list[str]:
    """Split on sentence ends and on line breaks (list items, paragraphs)."""
    for abbr in ABBREVIATIONS:
        answer = answer.replace(abbr, abbr.replace(".", _DOT))
    sentences = []
    for block in re.split(r"\n+", answer):
        sentences.extend(s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", block.strip()) if s.strip())
    return [s.replace(_DOT, ".") for s in sentences]


def _valid_decomposition(claims: list[str], sentence: str, question: str) -> bool:
    if not claims or len(claims) > MAX_CLAIMS_PER_SENTENCE:
        return False
    allowed = content_terms(sentence) | content_terms(question)
    for c in claims:
        if len(c.split()) < MIN_CLAIM_WORDS or CITATION_GROUP_RE.search(c):
            return False  # fragments, or claims about the citation rather than the content
        terms = content_terms(c)
        if terms and len(terms & allowed) / len(terms) < MIN_TERM_OVERLAP:
            return False  # decomposition invented content
    return True


def decompose_sentence(sentence: str, question: str) -> tuple[list[str], bool]:
    if len(sentence.split()) < DECOMPOSE_MIN_WORDS:
        return [sentence], False
    raw = generate(DECOMPOSE_PROMPT.format(question=question, sentence=sentence), max_new_tokens=200, temperature=0)
    claims = [re.sub(r"^\s*[-*•]\s*", "", line).strip() for line in raw.splitlines() if line.strip().startswith(("-", "*", "•"))]
    if _valid_decomposition(claims, sentence, question):
        return claims, True
    return [sentence], False


def extract_claims(answer: str, question: str) -> ExtractionResult:
    if is_refusal_answer(answer):
        return ExtractionResult(is_refusal=True)

    claims = []
    for i, raw_sentence in enumerate(split_sentences(answer)):
        cited = parse_citations(raw_sentence)
        sentence = strip_citations(raw_sentence)
        if len(sentence.split()) < MIN_CLAIM_WORDS or sentence.endswith(":"):
            continue  # fragments and list headers ("The categories are:")
        texts, decomposed = decompose_sentence(sentence, question)
        claims.extend(Claim(t, cited, sentence, i, decomposed) for t in texts)
    return ExtractionResult(is_refusal=False, claims=claims)
