"""Phase 11: apply self-correction to the Stage 4 (LoRA RAG + verification) answers from Phase 8/9.

Only answers with at least one flagged claim are corrected (the rest pass through unchanged, so
Stage 5 equals Stage 4 on them). Each uses the same prompt and sources as the Phase 8 run and the
stored Phase 9 verdicts as its starting point, so only the correction step is new.

Output: data/eval/self_correction_phase8.jsonl (every candidate per item, for manual review).
Usage: python scripts\\evaluate_self_correction.py [--system lora|base]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.embeddings.embedder import embed_query, embed_texts
from app.evaluation.lora_compare import SIMILARITY_THRESHOLD, TOP_K, VAL_PATH
from app.rag.llm import generate, get_model
from app.rag.pipeline import build_prompt
from app.rag.truthlens import DEFAULT_ADAPTER, MAX_NEW_TOKENS, load_generator
from app.retriever.retriever import Retriever
from app.self_correction.corrector import correct, needs_correction
from app.verification.phase8_audit import VERIFICATION_PATH, load_items
from app.verification.verifier import AnswerVerification, ClaimVerdict

OUT_PATH = Path("data/eval/self_correction_phase8.jsonl")


def prompts_by_item(items: dict) -> dict[str, str]:
    retriever = Retriever(similarity_threshold=SIMILARITY_THRESHOLD)
    prompts = {}
    for item_id, item in items.items():
        if item["set"] == "questions":
            chunks = retriever.retrieve(item["question"], top_k=TOP_K)
            prompts[item_id] = build_prompt(item["question"], retriever.format_context(chunks))
    for i, line in enumerate(VAL_PATH.open(encoding="utf-8")):
        prompts[f"val{i + 1:02d}"] = json.loads(line)["messages"][0]["content"]
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="lora", choices=("lora", "base"))
    args = parser.parse_args()

    items = {i["item_id"]: i for i in load_items()}
    prompts = prompts_by_item(items)
    similarities = {}
    for item_id, item in items.items():
        q = np.array(embed_query(item["question"]))
        similarities[item_id] = (np.array(embed_texts(item["sources"])) @ q).tolist()

    load_generator(DEFAULT_ADAPTER)

    def generate_fn(p: str) -> str:
        if args.system == "base":
            with get_model().disable_adapter():
                return generate(p, max_new_tokens=MAX_NEW_TOKENS, temperature=0)
        return generate(p, max_new_tokens=MAX_NEW_TOKENS, temperature=0)

    rows = [r for r in map(json.loads, VERIFICATION_PATH.open(encoding="utf-8")) if r["system"] == args.system]
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            v = AnswerVerification(r["is_refusal"], [ClaimVerdict(**c) for c in r["claims"]])
            if not needs_correction(v):
                continue
            item = items[r["item_id"]]
            print(f"  [correct] {r['item_id']} {args.system}")
            result = correct(item["question"], prompts[r["item_id"]], item["sources"], similarities[r["item_id"]],
                             item["answers"][args.system], v, generate_fn)
            out = {"item_id": r["item_id"], "system": args.system, "question": item["question"],
                   "best": result.best.strategy, "candidates": []}
            for c in result.candidates:
                out["candidates"].append({
                    "strategy": c.strategy, "answer": c.answer, "score": c.trust.score, "label": c.trust.label,
                    "is_refusal": c.verification.is_refusal, "summary": c.verification.summary(),
                    "flagged_claims": [x.claim for x in c.verification.claims if x.verdict in ("unsupported", "contradicted")],
                })
                print(f"     {c.strategy:<10} Trust Score: {c.trust.score}/100 ({c.trust.label})")
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
