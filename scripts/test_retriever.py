"""Manual sanity test for the retriever."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.retriever.retriever import Retriever

if __name__ == "__main__":
    retriever = Retriever(similarity_threshold=0.3)

    test_queries = [
        "What causes hallucinations in large language models?",
        "How does LoRA reduce the number of trainable parameters?",
        "What is retrieval augmented generation?",
    ]

    for q in test_queries:
        print(f"\n{'='*80}\nQuery: {q}\n{'='*80}")
        results = retriever.retrieve(q, top_k=3)

        if not results:
            print("  No results above similarity threshold.")
            continue

        for r in results:
            print(f"\n  [{r.similarity_score}] {r.title}")
            print(f"  Page {r.page_number}, Section: {r.section}")
            print(f"  {r.text[:150]}...")