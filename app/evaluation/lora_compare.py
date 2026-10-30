"""Compare Base RAG vs LoRA-fine-tuned RAG (Stage 2 vs Stage 3) under identical conditions.

Both systems see the same retrieved sources and decode greedily from the same loaded
weights; the only difference is whether the LoRA adapter is active.
"""
import json
import re
from datetime import datetime
from pathlib import Path

from app.finetuning.rag_formatter import term_recall
from app.rag.llm import attach_lora_adapter, generate, get_model
from app.rag.pipeline import build_prompt
from app.retriever.retriever import Retriever
from app.trust_score.trust_score import REFUSAL_RE, compute_trust_score

QUESTIONS_PATH = Path("data/eval/questions.json")
VAL_PATH = Path("data/finetuning/val.jsonl")
RESULTS_PATH = Path("data/eval/results_lora.jsonl")
VAL_RESULTS_PATH = Path("data/eval/results_lora_val.jsonl")
SUMMARY_PATH = Path("docs/evaluation_lora.md")
DEFAULT_ADAPTER = "models/lora_adapters/qwen25-7b-truthlens-lora"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
MAX_NEW_TOKENS = 512
# out-of-corpus (q11) and unanswerable (q12): the correct behaviour is to refuse
EXPECT_REFUSAL = {"q11", "q12"}

SYSTEMS = ("base", "lora")
SOURCE_RE = re.compile(r"Source\s+(\d+)", re.IGNORECASE)
SOURCE_HEADER_RE = re.compile(r"^\[Source (\d+): .*\(arXiv:([^)]+)\)", re.MULTILINE)


# ---------- metrics ----------

def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.split()) >= 3]


def answer_metrics(answer: str, num_sources: int) -> dict:
    cited = {int(n) for n in SOURCE_RE.findall(answer)}
    sentences = split_sentences(answer)
    uncited = [s for s in sentences if not SOURCE_RE.search(s)]
    return {
        "is_refusal": bool(REFUSAL_RE.search(answer)),
        "cited_sources": sorted(cited),
        "invalid_citations": sorted(n for n in cited if not 1 <= n <= num_sources),
        "num_sentences": len(sentences),
        "uncited_sentences": len(uncited),
        "words": len(answer.split()),
    }


def run_both(prompt: str) -> dict[str, str]:
    with get_model().disable_adapter():
        base = generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0)
    lora = generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0)
    return {"base": base, "lora": lora}


# ---------- eval set 1: the fixed 12 questions ----------

def eval_questions(limit: int | None = None) -> list[dict]:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))[:limit]
    retriever = Retriever(similarity_threshold=SIMILARITY_THRESHOLD)
    results = []

    for q in questions:
        print(f"\n{'=' * 80}\n{q['id']}: {q['question']}\n{'=' * 80}")
        chunks = retriever.retrieve(q["question"], top_k=TOP_K)
        sources = [
            {"title": c.title, "arxiv_id": c.arxiv_id, "page": c.page_number,
             "section": c.section, "similarity": c.similarity_score}
            for c in chunks
        ]
        if chunks:
            answers = run_both(build_prompt(q["question"], retriever.format_context(chunks)))
        else:  # same short-circuit as answer_question
            answers = {s: "No relevant information found in the corpus for this question." for s in SYSTEMS}

        entry = {"id": q["id"], "question": q["question"], "expect_refusal": q["id"] in EXPECT_REFUSAL,
                 "sources": sources, "timestamp": datetime.utcnow().isoformat()}
        for system in SYSTEMS:
            trust = compute_trust_score(answers[system], sources)
            metrics = answer_metrics(answers[system], len(sources))
            metrics["behaviour_correct"] = metrics["is_refusal"] == entry["expect_refusal"]
            entry[system] = {"answer": answers[system], "trust_score": trust.score,
                             "trust_label": trust.label, **metrics}
            print(f"\n  [{system}] trust {trust.score} | refusal {metrics['is_refusal']} | "
                  f"uncited {metrics['uncited_sentences']}/{metrics['num_sentences']}\n  {answers[system][:200]}")
        results.append(entry)

    return results


# ---------- eval set 2: held-out val examples with known gold sources ----------

def eval_val(limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in VAL_PATH.open(encoding="utf-8")][:limit]
    results = []

    for i, row in enumerate(rows, 1):
        prompt, target = row["messages"][0]["content"], row["messages"][1]["content"]
        meta = row["meta"]
        print(f"  [val {i}/{len(rows)}] {meta['type']}")
        answers = run_both(prompt)

        # position -> arXiv id, so a citation to a sibling chunk of the gold paper still counts as on-paper
        position_paper = {int(n): arxiv for n, arxiv in SOURCE_HEADER_RE.findall(prompt)}
        entry = {"type": meta["type"], "source_arxiv_id": meta["source_arxiv_id"], "target": target}
        for system in SYSTEMS:
            m = answer_metrics(answers[system], TOP_K)
            cited = set(m["cited_sources"])
            if meta["type"] == "answerable":
                gold = set(meta["gold_positions"])
                m.update({
                    "cites_gold": bool(cited & gold),
                    "cites_gold_paper": any(position_paper.get(n) == meta["source_arxiv_id"] for n in cited),
                    "citations_all_gold": bool(cited) and cited <= gold,
                    "content_recall": round(term_recall(target, answers[system]), 3),
                    "behaviour_correct": not m["is_refusal"],
                })
            else:
                m["behaviour_correct"] = m["is_refusal"] and not cited
            entry[system] = {"answer": answers[system], **m}
        results.append(entry)

    return results


# ---------- aggregation + report ----------

def mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 3) if values else 0.0


def summarize_questions(results: list[dict]) -> dict:
    summary = {}
    for system in SYSTEMS:
        answered = [r[system] for r in results if not r["expect_refusal"]]
        summary[system] = {
            "behaviour_correct": f"{sum(r[system]['behaviour_correct'] for r in results)}/{len(results)}",
            "trust_answerable_mean": mean(a["trust_score"] for a in answered),
            "trust_refusal_mean": mean(r[system]["trust_score"] for r in results if r["expect_refusal"]),
            "uncited_sentence_rate": mean(
                a["uncited_sentences"] / a["num_sentences"] for a in answered if a["num_sentences"]
            ),
            "invalid_citations": sum(len(r[system]["invalid_citations"]) for r in results),
            "mean_words": mean(r[system]["words"] for r in results),
        }
    return summary


def summarize_val(results: list[dict]) -> dict:
    summary = {}
    answerable = [r for r in results if r["type"] == "answerable"]
    refusals = [r for r in results if r["type"] == "refusal"]
    for system in SYSTEMS:
        summary[system] = {
            "answered_when_answerable": f"{sum(r[system]['behaviour_correct'] for r in answerable)}/{len(answerable)}",
            "cites_gold_source": f"{sum(r[system]['cites_gold'] for r in answerable)}/{len(answerable)}",
            "cites_gold_paper": f"{sum(r[system]['cites_gold_paper'] for r in answerable)}/{len(answerable)}",
            "all_citations_gold": f"{sum(r[system]['citations_all_gold'] for r in answerable)}/{len(answerable)}",
            "content_recall_mean": mean(r[system]["content_recall"] for r in answerable),
            "uncited_sentence_rate": mean(
                r[system]["uncited_sentences"] / r[system]["num_sentences"]
                for r in answerable if r[system]["num_sentences"]
            ),
            "refused_when_unanswerable": f"{sum(r[system]['behaviour_correct'] for r in refusals)}/{len(refusals)}",
            "mean_words": mean(r[system]["words"] for r in results),
        }
    return summary


def table(summary: dict) -> list[str]:
    keys = list(summary["base"])
    lines = ["| Metric | Base RAG | LoRA RAG |", "|---|---|---|"]
    lines += [f"| {k} | {summary['base'][k]} | {summary['lora'][k]} |" for k in keys]
    return lines


def write_markdown(q_results: list[dict], q_summary: dict, val_summary: dict | None, adapter_path: str):
    lines = [
        "# Stage 2 vs Stage 3: Base RAG vs LoRA-fine-tuned RAG",
        f"\nGenerated: {datetime.utcnow().isoformat()}",
        f"\nAdapter: `{adapter_path}`",
        "\nBoth systems receive identical retrieved sources (top_k=5, threshold 0.3) and decode greedily "
        "from the same 4-bit weights; only the LoRA adapter differs. Base RAG is re-run here rather than "
        "reusing `results_baseline.jsonl`, which used temperature 0.3 sampling.",
        "\nMetrics are rule-based proxies (RAGAS/DeepEval come in Phase 12). `is_refusal` uses the interim "
        "`REFUSAL_RE`, which has known false positives (see `docs/trust_score_interim.md`).",
        "\n## Fixed question set (12)\n",
        *table(q_summary),
    ]
    if val_summary:
        lines += ["\n## Held-out validation set (papers unseen in training)\n", *table(val_summary)]

    lines.append("\n---\n\n## Per-question answers\n")
    for r in q_results:
        lines.append(f"### {r['id']}: {r['question']}\n")
        lines.append(f"Expected: {'refuse' if r['expect_refusal'] else 'answer'}\n")
        for system, label in (("base", "Base RAG"), ("lora", "LoRA RAG")):
            a = r[system]
            lines.append(
                f"**{label}** (Trust Score: {a['trust_score']}/100 {a['trust_label']}; "
                f"uncited sentences {a['uncited_sentences']}/{a['num_sentences']}; "
                f"{'correct' if a['behaviour_correct'] else 'INCORRECT'} behaviour)\n"
            )
            lines.append(f"> {a['answer']}\n".replace("\n\n", "\n>\n> "))
        lines.append("---\n")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_lora_comparison(adapter_path: str = DEFAULT_ADAPTER, skip_val: bool = False, limit: int | None = None):
    if not Path(adapter_path, "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter_config.json in {adapter_path}; unzip the Colab adapter there first.")
    attach_lora_adapter(adapter_path)

    q_results = eval_questions(limit)
    q_summary = summarize_questions(q_results)
    write_jsonl(RESULTS_PATH, q_results)

    val_summary = None
    if not skip_val:
        print("\nRunning held-out validation set...")
        val_results = eval_val(limit)
        val_summary = summarize_val(val_results)
        write_jsonl(VAL_RESULTS_PATH, val_results)

    write_markdown(q_results, q_summary, val_summary, adapter_path)

    print("\n" + "\n".join(table(q_summary)))
    if val_summary:
        print("\n" + "\n".join(table(val_summary)))
    print(f"\nDone. Results -> {RESULTS_PATH}, {VAL_RESULTS_PATH if val_summary else ''} Summary -> {SUMMARY_PATH}")
    return q_summary, val_summary
