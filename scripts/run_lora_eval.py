"""CLI entrypoint: Base RAG vs LoRA-fine-tuned RAG comparison (Phase 8.4)."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.evaluation.lora_compare import DEFAULT_ADAPTER, run_lora_comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="folder containing adapter_config.json")
    parser.add_argument("--skip-val", action="store_true", help="only run the 12 fixed questions")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N items (smoke test)")
    args = parser.parse_args()
    run_lora_comparison(args.adapter, skip_val=args.skip_val, limit=args.limit)
