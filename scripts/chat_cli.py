"""Interactive CLI for the full TruthLens pipeline (LoRA RAG + claim verification + trust score + self-correction).

Usage:
    python scripts\\chat_cli.py                 # full pipeline
    python scripts\\chat_cli.py --no-correct    # Stage 4: verify and score, no self-correction
    python scripts\\chat_cli.py --no-adapter    # base Qwen as the generator
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.rag.truthlens import DEFAULT_ADAPTER, run_truthlens

VERDICT_TAG = {
    "supported": "OK  ",
    "misattributed": "MIS ",
    "uncited_supported": "UNC ",
    "unsupported": "FAIL",
    "contradicted": "CONT",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-correct", action="store_true", help="skip self-correction")
    parser.add_argument("--no-adapter", action="store_true", help="generate with base Qwen")
    args = parser.parse_args()
    adapter = None if args.no_adapter else DEFAULT_ADAPTER

    print("=== TruthLens AI (CLI) ===")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        print("\nThinking (generate -> verify claims -> score)...\n")
        result = run_truthlens(query, adapter_path=adapter, self_correct=not args.no_correct)
        best = result.best

        print(f"ANSWER:\n{result.answer}\n")
        print(best.trust.summary_line())
        if best.trust.cap_applied:
            print(f"  capped: {best.trust.cap_applied}")
        if len(result.candidates) > 1:
            tried = ", ".join(f"{c.strategy} {c.trust.score}" for c in result.candidates)
            print(f"  self-correction: kept '{best.strategy}' (candidates: {tried})")

        if best.verification.claims:
            print("\nCLAIMS:")
            for c in best.verification.claims:
                cited = ",".join(map(str, c.cited_sources)) or "-"
                print(f"  [{VERDICT_TAG[c.verdict]}] (S{cited}) {c.claim}")

        print("\nSOURCES:")
        for i, s in enumerate(result.sources, 1):
            print(f"  [{i}] {s['title']} (arXiv:{s['arxiv_id']}), Page {s['page']}, "
                  f"Section: {s['section']} [similarity: {s['similarity']}]")
        print("\n" + "=" * 80 + "\n")
