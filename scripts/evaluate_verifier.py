"""Measure the claim verifier against hand-labelled gold claims (Phase 9).

Each gold claim is labelled against its cited source(s) only, so this evaluates the
cited-source judgement every claim verdict starts from, under two policies:
  primary:   the Qwen judge's own label
  with_veto: Qwen SUPPORTED overridden when NLI entailment < NLI_VETO_THRESHOLD (sensitivity check)
Sets:
  dev:     data/eval/verifier_gold.jsonl    (38 claims, oversampled failure cases; used for design choices)
  heldout: data/eval/verifier_heldout.jsonl (40 random claims, labelled before the verifier saw them)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

SETS = {"dev": Path("data/eval/verifier_gold.jsonl"), "heldout": Path("data/eval/verifier_heldout.jsonl")}
PREDICTIONS_PATH = Path("data/eval/verifier_predictions.jsonl")
POLICIES = {"primary": "judge_supported", "with_veto": "veto_supported"}


def predict(name: str, rows: list[dict]) -> list[dict]:
    from app.verification.verifier import judge_with_veto

    preds = []
    for r in rows:
        j = judge_with_veto(r["claim"], r["question"], {int(n): t for n, t in r["cited_texts"].items()})
        preds.append({"set": name, "gold_id": r["gold_id"], "judge_supported": j.judge_supported,
                      "veto_supported": j.supported, "judge_label": j.judge_label, "judge_reason": j.judge_reason,
                      "nli_entailment": j.nli_entailment, "vetoed": j.vetoed})
    return preds


def score(preds: list[dict], gold: dict[str, str], key: str) -> dict:
    pos = [p for p in preds if gold[p["gold_id"]] == "supported"]
    neg = [p for p in preds if gold[p["gold_id"]] != "supported"]
    tp = sum(p[key] for p in pos)
    tn = sum(not p[key] for p in neg)
    flagged = sum(p[key] for p in preds)
    return {
        "agreement": f"{tp + tn}/{len(preds)}",
        "accuracy": round((tp + tn) / len(preds), 3),
        "supported_recall": f"{tp}/{len(pos)}",
        "unsupported_caught": f"{tn}/{len(neg)}",
        "precision_of_supported": round(tp / flagged, 3) if flagged else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=[*SETS, "both"], default="both")
    parser.add_argument("--rescore", action="store_true",
                        help="re-score saved predictions against the current labels instead of re-running the judge")
    args = parser.parse_args()

    saved = {}
    if args.rescore:
        saved = {(p["set"], p["gold_id"]): p for p in map(json.loads, PREDICTIONS_PATH.open(encoding="utf-8"))}

    all_preds = []
    for name in (SETS if args.set == "both" else [args.set]):
        rows = [json.loads(line) for line in SETS[name].open(encoding="utf-8")]
        gold = {r["gold_id"]: r["label"] for r in rows}
        preds = [saved[(name, r["gold_id"])] for r in rows] if args.rescore else predict(name, rows)
        all_preds += preds

        print(f"\n== {name}: {sum(l == 'supported' for l in gold.values())} supported / "
              f"{sum(l != 'supported' for l in gold.values())} not supported ==")
        for policy, key in POLICIES.items():
            print(f"  {policy:<9} {json.dumps(score(preds, gold, key))}")
        for p in preds:
            if p["judge_supported"] != (gold[p["gold_id"]] == "supported"):
                print(f"    primary miss {p['gold_id']}: gold={gold[p['gold_id']]} judge={p['judge_label']} | {p['judge_reason'][:110]}")

    if not args.rescore:
        with PREDICTIONS_PATH.open("w", encoding="utf-8") as f:
            for p in all_preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\nPredictions -> {PREDICTIONS_PATH}")
