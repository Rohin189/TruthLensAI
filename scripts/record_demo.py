"""Record full-pipeline runs for the Streamlit replay mode (hosted demo without a GPU).

Runs every demo question under all four sidebar settings (LoRA on/off x self-correction on/off)
and writes data/demo/recorded_runs.json. Resumable: runs already in the file are skipped.

Usage (needs the GPU free):
    python scripts\\record_demo.py
    python scripts\\record_demo.py --fresh     # discard existing recordings
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app.rag.truthlens import DEFAULT_ADAPTER, result_to_dict, run_truthlens

OUT = ROOT / "data/demo/recorded_runs.json"
QUESTIONS = [  # docs/demo_guide.md, in demo order
    "What is the difference between intrinsic and extrinsic hallucinations?",
    "What causes hallucinations in large language models?",
    "Why does LoRA increase the number of trainable parameters compared to full fine-tuning?",
    "What are the main challenges of deploying RAG in industry?",
    "What is quantum entanglement?",
    "How many trainable parameters does GPT-5 have?",
    "What is football?",
]
SETTINGS = [(True, True), (True, False), (False, True), (False, False)]  # (use_adapter, self_correct)


def key(q: str, use_adapter: bool, self_correct: bool) -> str:
    return f"{q}|adapter={int(use_adapter)}|correct={int(self_correct)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    data = {"runs": {}} if args.fresh or not OUT.exists() else json.loads(OUT.read_text(encoding="utf-8"))
    data["questions"] = QUESTIONS
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # group by adapter setting so the adapter is toggled as rarely as possible
    for use_adapter, self_correct in SETTINGS:
        for q in QUESTIONS:
            k = key(q, use_adapter, self_correct)
            if k in data["runs"]:
                continue
            t0 = time.time()
            result = run_truthlens(q, adapter_path=DEFAULT_ADAPTER if use_adapter else None,
                                   self_correct=self_correct)
            run = result_to_dict(result)
            run.update(use_adapter=use_adapter, self_correct=self_correct, seconds=round(time.time() - t0, 1))
            data["runs"][k] = run
            data["recorded_on"] = date.today().isoformat()
            OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            t = run["best"]["trust"]
            print(f"[{len(data['runs'])}/{len(QUESTIONS) * len(SETTINGS)}] adapter={use_adapter} "
                  f"correct={self_correct} {t['score']}/100 {t['label']} ({run['seconds']}s) {q[:60]}",
                  flush=True)
    print(f"Wrote {OUT}")
