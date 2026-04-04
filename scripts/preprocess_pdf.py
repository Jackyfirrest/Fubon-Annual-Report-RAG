from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.pdf_parser import extract_pdf_pages
from src.chunking import build_chunks_from_pages
from src.utils import write_jsonl


def main() -> None:
    pages = extract_pdf_pages(settings.raw_pdf_path)
    chunks = build_chunks_from_pages(pages)

    write_jsonl(pages, settings.processed_pages_path)
    write_jsonl(chunks, settings.processed_chunks_path)

    print(f"[OK] pages saved to {settings.processed_pages_path}: {len(pages)} pages")
    print(f"[OK] chunks saved to {settings.processed_chunks_path}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
