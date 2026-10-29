"""Convert the Phase 8.1 Q&A pairs into RAG-format training examples.

Each example uses the exact inference prompt (build_prompt): the question plus
retrieved sources, with the answer citing (Source N). This trains the model to
stay grounded inside the RAG pipeline instead of answering from memory.
"""
import json
import random
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from app.embeddings.embedder import embed_texts
from app.rag.llm import MODEL_NAME, generate
from app.rag.pipeline import build_prompt
from app.retriever.retriever import Retriever
from app.trust_score.trust_score import REFUSAL_RE

INPUT_PATH = Path("data/finetuning/instructions.jsonl")
REWRITE_CACHE_PATH = Path("data/finetuning/question_rewrites.jsonl")
EVAL_QUESTIONS_PATH = Path("data/eval/questions.json")
TRAIN_PATH = Path("data/finetuning/train.jsonl")
VAL_PATH = Path("data/finetuning/val.jsonl")
REPORT_PATH = Path("data/finetuning/build_report.json")

SEED = 42
NUM_SOURCES = 5            # matches answer_question's top_k
GOLD_MIN_RECALL = 0.5      # answer content words that must appear in the gold chunk
REFUSAL_MAX_RECALL = 0.2   # refusal context must NOT contain the answer
REFUSAL_COUNT = 50
EVAL_OVERLAP_MAX_SIM = 0.85
VAL_FRACTION = 0.1
REWRITE_ATTEMPTS = 2
MAX_TOKENS = 2048          # Colab max_seq_length; longer examples are dropped, not truncated

DEICTIC_RE = re.compile(
    r"\b(excerpt|passage|this paper|the paper|this study|the study|the authors|"
    r"this research|the research)\b",
    re.IGNORECASE,
)
BAD_OUTPUT_RE = re.compile(r"\b(excerpt|the text)\b", re.IGNORECASE)

REWRITE_PROMPT = """Rewrite the question below so it makes sense on its own, without seeing any document.

Rules:
- Name the specific method, system, dataset, or topic instead of saying "the paper", "the excerpt", "the study", "the authors", or "the research".
- The rewritten question must still be answered by the given answer.
- Output ONLY the rewritten question on a single line, nothing else.

PAPER TITLE: {title}
ORIGINAL QUESTION: {question}
ANSWER: {answer}

REWRITTEN QUESTION:"""

REWRITE_RETRY_PROMPT = """This question still contains the phrase "{phrase}", which a reader cannot resolve without the document.

Replace "{phrase}" with the specific name of the method, system, model, or topic it refers to (use the title and answer below).
Example: "What method does the paper propose for filtering noisy labels?" -> "What method does confidence-based filtering use to remove noisy relevance labels?"

Output ONLY the rewritten question on a single line. Do not use the words "paper", "study", "authors", "research", "excerpt", or "passage".

PAPER TITLE: {title}
QUESTION: {question}
ANSWER: {answer}

REWRITTEN QUESTION:"""

REFUSAL_ANSWERS = [
    "The provided sources do not contain information to answer this question.",
    "The sources do not contain enough information to answer this question.",
    "I cannot answer this question based on the provided sources, as they do not contain the relevant information.",
]

STOPWORDS = {
    "this", "that", "with", "from", "which", "their", "they", "these", "those", "have",
    "been", "were", "also", "such", "into", "than", "more", "most", "other", "using",
    "used", "uses", "based", "while", "where", "when", "what", "will", "would", "could",
    "should", "about", "between", "through", "both", "each", "only", "some", "them",
    "there", "being", "does", "over", "well", "like", "many", "much", "very", "within",
}


# ---------- helpers ----------

def content_terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower())
    return {w for w in words if len(w) >= 4 and w not in STOPWORDS}


def term_recall(answer: str, text: str) -> float:
    """Fraction of the answer's content words present in text."""
    answer_terms = content_terms(answer)
    if not answer_terms:
        return 0.0
    return len(answer_terms & content_terms(text)) / len(answer_terms)


def add_citations(answer: str, gold_texts: dict[int, str]) -> str:
    """Append (Source N) to every sentence, citing the gold chunk that best supports it."""
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    cited = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        source_num = max(gold_texts, key=lambda n: term_recall(s, gold_texts[n]))
        if s[-1] in ".!?":
            cited.append(f"{s[:-1]} (Source {source_num}){s[-1]}")
        else:
            cited.append(f"{s} (Source {source_num}).")
    return " ".join(cited)


def find_gold_chunks(answer: str, candidates) -> tuple[float, list]:
    """Best single chunk; if it falls short, the best pair (answers often span adjacent chunks)."""
    if not candidates:
        return 0.0, []
    best_recall, best = max(((term_recall(answer, c.text), [c]) for c in candidates), key=lambda x: x[0])
    if best_recall >= GOLD_MIN_RECALL:
        return best_recall, best

    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            recall = term_recall(answer, a.text + " " + b.text)
            # second chunk must add real support, not just pad the score
            if recall > best_recall and recall - max(term_recall(answer, a.text), term_recall(answer, b.text)) >= 0.1:
                best_recall, best = recall, [a, b]
    return best_recall, best


def to_messages(prompt: str, answer: str) -> list[dict]:
    # generate() sends a single user turn, so training mirrors that exactly
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]


# ---------- steps ----------

def load_pairs() -> tuple[list[dict], dict]:
    pairs, seen = [], set()
    stats = {"input_pairs": 0, "dropped_duplicate": 0, "dropped_bad_output": 0}
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            stats["input_pairs"] += 1
            key = p["instruction"].strip().lower()
            if key in seen:
                stats["dropped_duplicate"] += 1
                continue
            if BAD_OUTPUT_RE.search(p["output"]):
                stats["dropped_bad_output"] += 1
                continue
            seen.add(key)
            pairs.append(p)
    return pairs, stats


def is_valid_rewrite(question: str) -> bool:
    return bool(question) and question.endswith("?") and not DEICTIC_RE.search(question)


def rewrite_deictic_questions(pairs: list[dict]) -> dict:
    """Rewrite questions that point at an unseen document. Cached so reruns skip Qwen."""
    cache = {}  # (original, attempt) -> rewritten
    if REWRITE_CACHE_PATH.exists():
        with REWRITE_CACHE_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                cache[(row["original"], row.get("attempt", 1))] = row["rewritten"]

    stats = {"deictic": 0, "rewritten": 0, "rewritten_on_retry": 0, "dropped_rewrite_failed": 0}
    targets = [p for p in pairs if DEICTIC_RE.search(p["instruction"])]
    stats["deictic"] = len(targets)

    with REWRITE_CACHE_PATH.open("a", encoding="utf-8") as cache_f:
        for i, p in enumerate(targets, 1):
            original = p["instruction"]
            rewritten = ""
            for attempt in range(1, REWRITE_ATTEMPTS + 1):
                key = (original, attempt)
                if key not in cache:
                    print(f"  [rewrite {i}/{len(targets)}, attempt {attempt}] {original[:70]}")
                    if attempt == 1:
                        prompt = REWRITE_PROMPT.format(title=p["source_title"], question=original, answer=p["output"])
                    else:
                        bad = DEICTIC_RE.search(rewritten)
                        prompt = REWRITE_RETRY_PROMPT.format(
                            title=p["source_title"], question=rewritten or original,
                            answer=p["output"], phrase=bad.group(0) if bad else "the paper",
                        )
                    raw = generate(prompt, max_new_tokens=80, temperature=0.2)
                    cache[key] = raw.strip().splitlines()[0].strip().strip('"') if raw.strip() else ""
                    cache_f.write(json.dumps({"original": original, "attempt": attempt, "rewritten": cache[key]}) + "\n")
                    cache_f.flush()
                rewritten = cache[key]
                if is_valid_rewrite(rewritten):
                    stats["rewritten_on_retry"] += attempt > 1
                    break

            if is_valid_rewrite(rewritten):
                p["original_instruction"] = original
                p["instruction"] = rewritten
                stats["rewritten"] += 1
            else:
                p["drop"] = True
                stats["dropped_rewrite_failed"] += 1

    return stats


def remove_eval_overlap(pairs: list[dict]) -> tuple[list[dict], dict]:
    eval_questions = [q["question"] for q in json.loads(EVAL_QUESTIONS_PATH.read_text(encoding="utf-8"))]
    eval_vecs = np.array(embed_texts(eval_questions))
    train_vecs = np.array(embed_texts([p["instruction"] for p in pairs]))
    sims = train_vecs @ eval_vecs.T  # normalized embeddings -> cosine similarity

    kept, removed = [], []
    for p, row in zip(pairs, sims):
        j = int(row.argmax())
        if row[j] >= EVAL_OVERLAP_MAX_SIM:
            removed.append({"train": p["instruction"], "eval": eval_questions[j], "sim": round(float(row[j]), 3)})
        else:
            kept.append(p)

    top_sims = sorted(
        ({"train": p["instruction"], "eval": eval_questions[int(r.argmax())], "sim": round(float(r.max()), 3)}
         for p, r in zip(pairs, sims)),
        key=lambda x: -x["sim"],
    )[:10]
    return kept, {"dropped_eval_overlap": len(removed), "eval_overlap_removed": removed, "eval_top_similarities": top_sims}


def build_answerable(pairs: list[dict], retriever: Retriever, rng: random.Random) -> tuple[list[dict], dict]:
    examples, recalls = [], []
    stats = {"dropped_no_gold": 0}

    for p in pairs:
        question = p["instruction"]
        candidates = retriever.retrieve(question, top_k=5, arxiv_id_filter=p["source_arxiv_id"])
        best_recall, golds = find_gold_chunks(p["output"], candidates)
        recalls.append(best_recall)
        if best_recall < GOLD_MIN_RECALL:
            stats["dropped_no_gold"] += 1
            continue

        gold_ids = {g.chunk_id for g in golds}
        pool = retriever.retrieve(question, top_k=NUM_SOURCES + 5)
        chunks = [c for c in pool if c.chunk_id not in gold_ids][: NUM_SOURCES - len(golds)]
        for g in golds:
            chunks.insert(rng.randint(0, len(chunks)), g)
        gold_positions = {i + 1: c.text for i, c in enumerate(chunks) if c.chunk_id in gold_ids}

        prompt = build_prompt(question, retriever.format_context(chunks))
        answer = add_citations(p["output"], gold_positions)
        examples.append({
            "messages": to_messages(prompt, answer),
            "meta": {
                "type": "answerable",
                "source_arxiv_id": p["source_arxiv_id"],
                "gold_chunk_ids": [g.chunk_id for g in golds],
                "gold_positions": sorted(gold_positions),
                "gold_recall": round(best_recall, 3),
                "rewritten": "original_instruction" in p,
            },
        })

    stats["gold_recall_mean"] = round(float(np.mean(recalls)), 3) if recalls else 0.0
    stats["gold_recall_histogram"] = {
        f"{lo:.1f}-{lo + 0.1:.1f}": int(sum(lo <= r < lo + 0.1 or (lo >= 0.9 and r == 1.0) for r in recalls))
        for lo in np.arange(0, 1.0, 0.1)
    }
    return examples, stats


def build_refusals(pairs: list[dict], retriever: Retriever, rng: random.Random) -> tuple[list[dict], dict]:
    """Same question, but only chunks from OTHER papers that don't contain the answer."""
    examples, attempts = [], 0
    candidates = pairs[:]
    rng.shuffle(candidates)

    for p in candidates:
        if len(examples) >= REFUSAL_COUNT:
            break
        attempts += 1
        question = p["instruction"]
        pool = retriever.retrieve(question, top_k=NUM_SOURCES * 4)
        chunks = [c for c in pool if c.arxiv_id != p["source_arxiv_id"]][:NUM_SOURCES]
        if len(chunks) < NUM_SOURCES:
            continue
        if term_recall(p["output"], " ".join(c.text for c in chunks)) > REFUSAL_MAX_RECALL:
            continue  # other papers may genuinely answer it; refusing would be wrong

        answer = rng.choice(REFUSAL_ANSWERS)
        assert REFUSAL_RE.search(answer), "refusal template must be detected by the trust score"
        examples.append({
            "messages": to_messages(build_prompt(question, retriever.format_context(chunks)), answer),
            "meta": {"type": "refusal", "source_arxiv_id": p["source_arxiv_id"]},
        })

    return examples, {"refusal_attempts": attempts, "refusals_built": len(examples)}


def split_by_paper(examples: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    papers = sorted({e["meta"]["source_arxiv_id"] for e in examples})
    rng.shuffle(papers)
    val_papers = set(papers[: max(1, round(len(papers) * VAL_FRACTION))])
    train = [e for e in examples if e["meta"]["source_arxiv_id"] not in val_papers]
    val = [e for e in examples if e["meta"]["source_arxiv_id"] in val_papers]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def drop_long_examples(examples: list[dict]) -> tuple[list[dict], dict]:
    # tokenizer only: avoids loading the 7B weights when every rewrite is cached
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    lengths = [len(tokenizer.apply_chat_template(e["messages"], tokenize=True)) for e in examples]
    kept = [e for e, n in zip(examples, lengths) if n <= MAX_TOKENS]
    kept_lengths = [n for n in lengths if n <= MAX_TOKENS]
    return kept, {
        "dropped_too_long": len(examples) - len(kept),
        "tokens_mean": int(np.mean(kept_lengths)),
        "tokens_max": int(max(kept_lengths)),
    }


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- orchestrator ----------

def build_rag_dataset():
    rng = random.Random(SEED)
    report = {}

    pairs, stats = load_pairs()
    report.update(stats)
    print(f"Loaded {len(pairs)} pairs ({stats['dropped_duplicate']} duplicates, {stats['dropped_bad_output']} bad outputs dropped)")

    print("Rewriting document-dependent questions...")
    report.update(rewrite_deictic_questions(pairs))
    pairs = [p for p in pairs if not p.get("drop")]

    print("Checking overlap with eval questions...")
    pairs, stats = remove_eval_overlap(pairs)
    report.update(stats)

    retriever = Retriever(similarity_threshold=0.0)  # build-time: never filter candidates

    print("Building answerable examples...")
    answerable, stats = build_answerable(pairs, retriever, rng)
    report.update(stats)

    print("Building refusal examples...")
    refusals, stats = build_refusals(pairs, retriever, rng)
    report.update(stats)

    examples, stats = drop_long_examples(answerable + refusals)
    report.update(stats)

    train, val = split_by_paper(examples, rng)
    report.update({
        "answerable_examples": sum(e["meta"]["type"] == "answerable" for e in examples),
        "refusal_examples": sum(e["meta"]["type"] == "refusal" for e in examples),
        "train_examples": len(train),
        "val_examples": len(val),
        "val_refusals": sum(e["meta"]["type"] == "refusal" for e in val),
    })

    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VAL_PATH, val)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nTrain: {len(train)} -> {TRAIN_PATH}")
    print(f"Val:   {len(val)} -> {VAL_PATH}")
    print(f"Report -> {REPORT_PATH}")
    return report
