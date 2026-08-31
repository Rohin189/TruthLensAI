"""CLI entrypoint: run the No-RAG vs Base RAG baseline comparison."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.evaluation.baseline_compare import run_comparison

if __name__ == "__main__":
    run_comparison()