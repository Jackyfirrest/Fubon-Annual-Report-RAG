from __future__ import annotations

from typing import List, Dict
import fitz

from .utils import normalize_whitespace


def extract_pdf_pages(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages: List[Dict] = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = normalize_whitespace(text)
        pages.append(
            {
                "page_num": i + 1,
                "text": text,
            }
        )

    return pages
