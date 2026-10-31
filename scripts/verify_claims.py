"""CLI entrypoint: claim extraction + NLI verification over the Phase 8 answers (Phase 9)."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.verification.phase8_audit import CLAIMS_PATH, VERIFICATION_PATH, load_items, run_extract, run_verify

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["extract", "verify", "all"], default="all")
    args = parser.parse_args()

    items = load_items()
    print(f"Loaded {len(items)} items")
    if args.stage in ("extract", "all"):
        run_extract(items)
        print(f"Claims -> {CLAIMS_PATH}")
    if args.stage in ("verify", "all"):
        rows = run_verify(items)
        print(f"Verified {len(rows)} answers -> {VERIFICATION_PATH}")
