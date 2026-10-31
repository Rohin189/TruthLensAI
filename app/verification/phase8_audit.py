"""Run claim extraction + verification over the Phase 8 answers (Stage 2 base vs Stage 3 LoRA).

Two cached stages, because they need different models:
  extract: Qwen splits each answer into claims  -> data/eval/claims_phase8.jsonl
  verify:  NLI checks each claim vs its sources -> data/eval/verification_phase8.jsonl
Re-running `verify` (e.g. after threshold tuning) reuses the extracted claims.
"""
import json
import re
from dataclasses import asdict
from pathlib import Path

from app.evaluation.lora_compare import RESULTS_PATH, SIMILARITY_THRESHOLD, TOP_K, VAL_PATH, VAL_RESULTS_PATH
from app.retriever.retriever import Retriever

CLAIMS_PATH = Path("data/eval/claims_phase8.jsonl")
VERIFICATION_PATH = Path("data/eval/verification_phase8.jsonl")
SYSTEMS = ("base", "lora")

SOURCE_SPLIT_RE = re.compile(r"\n\n(?=\[Source \d+:)")
SOURCE_HEADER_RE = re.compile(r"^\[Source \d+:[^\]]*\]\n", re.MULTILINE)


def sources_from_prompt(prompt: str) -> list[str]:
    block = prompt.split("SOURCES:\n", 1)[1].split("\n\nQUESTION:", 1)[0]
    return [SOURCE_HEADER_RE.sub("", part, count=1) for part in SOURCE_SPLIT_RE.split(block)]


def load_items() -> list[dict]:
    """Phase 8 eval items with source texts in prompt order and both systems' answers."""
    items = []

    retriever = Retriever(similarity_threshold=SIMILARITY_THRESHOLD)
    for line in RESULTS_PATH.open(encoding="utf-8"):
        r = json.loads(line)
        chunks = retriever.retrieve(r["question"], top_k=TOP_K)
        # results_lora.jsonl stores source metadata only; retrieval is deterministic, but check
        got = [(c.arxiv_id, c.page_number, c.section) for c in chunks]
        want = [(s["arxiv_id"], s["page"], s["section"]) for s in r["sources"]]
        assert got == want, f"{r['id']}: re-retrieved sources differ from the Phase 8 run"
        items.append({
            "item_id": r["id"], "set": "questions", "question": r["question"],
            "expect_refusal": r["expect_refusal"], "sources": [c.text for c in chunks],
            "answers": {s: r[s]["answer"] for s in SYSTEMS},
        })

    val_rows = [json.loads(l) for l in VAL_PATH.open(encoding="utf-8")]
    val_results = [json.loads(l) for l in VAL_RESULTS_PATH.open(encoding="utf-8")]
    for i, (row, r) in enumerate(zip(val_rows, val_results)):
        prompt = row["messages"][0]["content"]
        items.append({
            "item_id": f"val{i + 1:02d}", "set": "val",
            "question": re.search(r"QUESTION: (.*)", prompt).group(1),
            "expect_refusal": r["type"] == "refusal", "sources": sources_from_prompt(prompt),
            "gold_positions": row["meta"].get("gold_positions", []),
            "answers": {s: r[s]["answer"] for s in SYSTEMS},
        })
    return items


def run_extract(items: list[dict]):
    from app.verification.claim_extractor import extract_claims

    done = set()
    if CLAIMS_PATH.exists():
        done = {(r["item_id"], r["system"]) for r in map(json.loads, CLAIMS_PATH.open(encoding="utf-8"))}

    with CLAIMS_PATH.open("a", encoding="utf-8") as f:
        for item in items:
            for system in SYSTEMS:
                if (item["item_id"], system) in done:
                    continue
                print(f"  [extract] {item['item_id']} {system}")
                result = extract_claims(item["answers"][system], item["question"])
                f.write(json.dumps({
                    "item_id": item["item_id"], "system": system, "is_refusal": result.is_refusal,
                    "claims": [asdict(c) for c in result.claims],
                }, ensure_ascii=False) + "\n")
                f.flush()


def run_verify(items: list[dict]) -> list[dict]:
    from app.verification.claim_extractor import Claim
    from app.verification.verifier import AnswerVerification, verify_claim

    by_id = {i["item_id"]: i for i in items}
    records = [json.loads(line) for line in CLAIMS_PATH.open(encoding="utf-8")]
    rows = []
    for k, rec in enumerate(records, 1):
        item = by_id[rec["item_id"]]
        print(f"  [verify {k}/{len(records)}] {rec['item_id']} {rec['system']} ({len(rec['claims'])} claims)")
        verification = AnswerVerification(is_refusal=rec["is_refusal"])
        for c in rec["claims"]:
            verification.claims.append(verify_claim(Claim(**c), item["question"], item["sources"]))
        rows.append({
            "item_id": rec["item_id"], "set": item["set"], "system": rec["system"],
            "expect_refusal": item["expect_refusal"], **verification.summary(),
            "claims": [asdict(v) for v in verification.claims],
        })

    with VERIFICATION_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows
