"""Generate instruction-tuning Q&A pairs from the corpus using local Qwen."""
import json
import re
from pathlib import Path

from app.rag.llm import generate

MANIFEST_PATH = Path("data/raw/manifest.json")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = Path("data/finetuning/instructions.jsonl")

QA_PER_PAPER = 3

GENERATION_PROMPT = """You are creating training data for an AI research assistant.
Based on the following paper excerpt, write exactly {n} question-answer pairs that
test understanding of specific claims, methods, or findings in this text.

Rules:
- Questions must be answerable ONLY using the excerpt below (no outside knowledge).
- Answers must be factual, concise (2-4 sentences), and directly grounded in the text.
- Do not reference "the paper" or "the excerpt" in the question itself — phrase questions naturally.

Format your response EXACTLY like this, with no extra text:
Q1: <question>
A1: <answer>
Q2: <question>
A2: <answer>
Q3: <question>
A3: <answer>

PAPER TITLE: {title}

EXCERPT:
{excerpt}
"""


def parse_qa_pairs(raw_output: str) -> list[dict]:
    pairs = []
    pattern = re.compile(r"Q\d+:\s*(.+?)\s*A\d+:\s*(.+?)(?=Q\d+:|$)", re.DOTALL)
    for match in pattern.finditer(raw_output):
        question = match.group(1).strip()
        answer = match.group(2).strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})
    return pairs


def build_dataset(qa_per_paper: int = QA_PER_PAPER):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_examples = []
    papers = list(manifest.items())
    print(f"Generating Q&A pairs for {len(papers)} papers...")

    for i, (arxiv_id, meta) in enumerate(papers, 1):
        processed_file = PROCESSED_DIR / f"{arxiv_id.replace('/', '_')}.txt"
        if not processed_file.exists():
            continue

        full_text = processed_file.read_text(encoding="utf-8")
        excerpt = full_text[:2500]  # abstract + intro region, roughly

        prompt = GENERATION_PROMPT.format(n=qa_per_paper, title=meta["title"], excerpt=excerpt)

        print(f"  [{i}/{len(papers)}] {meta['title'][:60]}")
        try:
            raw_output = generate(prompt, max_new_tokens=600, temperature=0.4)
            pairs = parse_qa_pairs(raw_output)
        except Exception as e:
            print(f"    x Failed: {e}")
            continue

        for pair in pairs:
            all_examples.append({
                "instruction": pair["question"],
                "input": "",
                "output": pair["answer"],
                "source_arxiv_id": arxiv_id,
                "source_title": meta["title"],
            })

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nGenerated {len(all_examples)} training examples -> {OUTPUT_PATH}")
    return all_examples