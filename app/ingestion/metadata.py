"""arXiv paper search, download, and manifest tracking."""
import arxiv
import json
import time
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "manifest.json"

DEFAULT_CATEGORIES = ["cs.CL", "cs.LG", "cs.AI"]

DEFAULT_QUERIES = [
    "retrieval augmented generation",
    "large language model hallucination",
    "parameter efficient fine-tuning LoRA",
    "hallucination detection NLP",
    "self-correction language models",
]


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def download_papers(
    query: str,
    max_results: int = 30,
    categories: list[str] | None = None,
    sort_by=arxiv.SortCriterion.Relevance,
) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    categories = categories or DEFAULT_CATEGORIES
    cat_filter = " OR ".join(f"cat:{c}" for c in categories)
    full_query = f"({query}) AND ({cat_filter})"

    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    search = arxiv.Search(query=full_query, max_results=max_results, sort_by=sort_by)

    downloaded = 0
    for result in client.results(search):
        arxiv_id = result.get_short_id()
        if arxiv_id in manifest:
            continue

        filename = f"{arxiv_id.replace('/', '_')}.pdf"
        try:
            result.download_pdf(dirpath=str(RAW_DIR), filename=filename)
        except Exception as e:
            print(f"  x Failed to download {arxiv_id}: {e}")
            continue

        manifest[arxiv_id] = {
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "published": result.published.isoformat(),
            "categories": result.categories,
            "summary": result.summary,
            "pdf_url": result.pdf_url,
            "filename": filename,
            "downloaded_at": datetime.utcnow().isoformat(),
        }
        downloaded += 1
        print(f"  [{downloaded}] {result.title[:70]}")
        time.sleep(1)

    save_manifest(manifest)
    print(f"\nDownloaded {downloaded} new papers. Total in manifest: {len(manifest)}")
    return manifest


def run_all_queries(queries: list[str] | None = None, max_results_per_query: int = 30) -> dict:
    queries = queries or DEFAULT_QUERIES
    for q in queries:
        print(f"\n--- Query: {q} ---")
        try:
            download_papers(query=q, max_results=max_results_per_query)
        except Exception as e:
            print(f"  !! Query failed, skipping: {q} ({e})")
            continue
    return load_manifest()