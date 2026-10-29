"""Interactive CLI loop for testing the Base RAG pipeline."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.rag.pipeline import answer_question

if __name__ == "__main__":
    print("=== TruthLens AI — Base RAG (CLI) ===")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        print("\nThinking...\n")
        result = answer_question(query)

        print(f"ANSWER:\n{result['answer']}\n")

        print(f"TRUST SCORE: {result['trust_score']}/100 ({result['trust_label']})")
        print(f"  Retrieval confidence: {result['trust_details']['retrieval_confidence']}")
        print(f"  Citation coverage: {result['trust_details']['citation_coverage']}")
        print(f"  Sources: {result['trust_details']['sources_cited']}/{result['trust_details']['sources_retrieved']} cited")

        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"  - {s['title']} (arXiv:{s['arxiv_id']}), Page {s['page']}, "
                  f"Section: {s['section']} [similarity: {s['similarity']}]")
        print("\n" + "=" * 80 + "\n")