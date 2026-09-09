"""CLI entrypoint: generate the LoRA instruction dataset."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.finetuning.dataset_builder import build_dataset

if __name__ == "__main__":
    build_dataset()