from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.retriever import BM25Retriever
from src.utils import read_jsonl


def main() -> None:
    chunks = read_jsonl(settings.processed_chunks_path)
    retriever = BM25Retriever(chunks)
    retriever.save(settings.bm25_index_path)
    print(f"[OK] BM25 index saved to {settings.bm25_index_path}")


if __name__ == "__main__":
    main()
