from __future__ import annotations

import re
from typing import List, Dict

from .utils import normalize_whitespace


def split_into_paragraphs(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def build_chunks_from_pages(pages: List[Dict], min_chars: int = 120, max_chars: int = 1200) -> List[Dict]:
    """
    先以頁面為基礎，再依段落做 chunk。
    若段落太短會合併，太長則切小段。
    """
    chunks: List[Dict] = []
    chunk_id = 0

    for page in pages:
        page_num = page["page_num"]
        paragraphs = split_into_paragraphs(page["text"])

        buffer = ""
        for para in paragraphs:
            if len(para) > max_chars:
                if buffer:
                    chunks.append(
                        {
                            "chunk_id": f"chunk_{chunk_id}",
                            "page_num": page_num,
                            "text": buffer.strip(),
                        }
                    )
                    chunk_id += 1
                    buffer = ""

                for start in range(0, len(para), max_chars):
                    piece = para[start : start + max_chars]
                    chunks.append(
                        {
                            "chunk_id": f"chunk_{chunk_id}",
                            "page_num": page_num,
                            "text": piece.strip(),
                        }
                    )
                    chunk_id += 1
                continue

            candidate = f"{buffer}\n\n{para}".strip() if buffer else para
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(
                        {
                            "chunk_id": f"chunk_{chunk_id}",
                            "page_num": page_num,
                            "text": buffer.strip(),
                        }
                    )
                    chunk_id += 1
                buffer = para

        if buffer:
            if len(buffer) < min_chars and chunks and chunks[-1]["page_num"] == page_num:
                chunks[-1]["text"] = f"{chunks[-1]['text']}\n\n{buffer}".strip()
            else:
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_id}",
                        "page_num": page_num,
                        "text": buffer.strip(),
                    }
                )
                chunk_id += 1

    return chunks
