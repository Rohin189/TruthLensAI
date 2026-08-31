"""Compare Baseline ① (No-RAG) vs Baseline ② (Base RAG) on a fixed question set."""
import json
from pathlib import Path
from datetime import datetime

from app.rag.pipeline import answer_question, answer_question_no_rag

QUESTIONS_PATH = Path("data/eval/questions.json")
RESULTS_PATH = Path("data/eval/results_baseline.jsonl")
SUMMARY_PATH = Path("docs/evaluation_baseline.md")


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def run_comparison():
    questions = load_questions()
    results = []

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for q in questions:
        print(f"\n{'='*80}\n{q['id']}: {q['question']}\n{'='*80}")

        print("  Running No-RAG...")
        no_rag_result = answer_question_no_rag(q["question"])

        print("  Running Base RAG...")
        rag_result = answer_question(q["question"])

        entry = {
            "id": q["id"],
            "question": q["question"],
            "no_rag_answer": no_rag_result["answer"],
            "rag_answer": rag_result["answer"],
            "rag_sources": rag_result["sources"],
            "timestamp": datetime.utcnow().isoformat(),
        }
        results.append(entry)

        print(f"\n  [No-RAG]: {no_rag_result['answer'][:150]}...")
        print(f"\n  [Base RAG]: {rag_result['answer'][:150]}...")

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    write_markdown_summary(results)
    print(f"\n\nDone. Results saved to {RESULTS_PATH} and {SUMMARY_PATH}")


def write_markdown_summary(results: list[dict]):
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Baseline Comparison: No-RAG vs Base RAG",
        f"\nGenerated: {datetime.utcnow().isoformat()}",
        f"\nQuestions evaluated: {len(results)}",
        "\n---\n",
    ]

    for r in results:
        lines.append(f"## {r['id']}: {r['question']}\n")
        lines.append(f"**No-RAG Answer:**\n> {r['no_rag_answer']}\n")
        lines.append(f"**Base RAG Answer:**\n> {r['rag_answer']}\n")

        if r["rag_sources"]:
            lines.append("**RAG Sources:**")
            for s in r["rag_sources"]:
                lines.append(f"- {s['title']} (arXiv:{s['arxiv_id']}), Page {s['page']}, "
                              f"Section: {s['section']} [similarity: {s['similarity']}]")
        else:
            lines.append("**RAG Sources:** none retrieved")

        lines.append("\n---\n")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")