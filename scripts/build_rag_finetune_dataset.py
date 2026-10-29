"""CLI entrypoint: convert instruction pairs into RAG-format train/val sets."""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.finetuning.rag_formatter import build_rag_dataset

if __name__ == "__main__":
    report = build_rag_dataset()
    summary = {k: v for k, v in report.items() if k not in ("eval_overlap_removed", "eval_top_similarities")}
    print(json.dumps(summary, indent=2))
