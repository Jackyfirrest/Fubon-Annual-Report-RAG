from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.retriever import HybridRetriever
from src.utils import read_jsonl


def main() -> None:
    chunks = read_jsonl(settings.processed_chunks_path)
    retriever = HybridRetriever(
        chunks,
        bm25_weight=settings.bm25_weight,
        tfidf_weight=settings.tfidf_weight,
        overlap_weight=settings.overlap_weight,
    )
    retriever.save(settings.hybrid_index_path)
    print(f"[OK] hybrid index saved to {settings.hybrid_index_path}")


if __name__ == "__main__":
    main()