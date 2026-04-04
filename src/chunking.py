from __future__ import annotations

import re
from typing import List, Dict

from .utils import normalize_whitespace


def split_into_paragraphs(text: str) -> List[str]:
    text = text.replace("。", "。\n")
    text = normalize_whitespace(text)
    if not text:
        return []
    paragraphs = re.split(r"\n\n+|\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def guess_section_title(paragraphs: List[str]) -> str:
    for p in paragraphs[:5]:
        if len(p) <= 35 and not re.search(r"[，。；：:]", p):
            return p
    return ""


def build_chunks_from_pages(pages: List[Dict], min_chars: int = 80, max_chars: int = 900) -> List[Dict]:
    chunks: List[Dict] = []
    chunk_id = 0

    for page in pages:
        page_num = page["page_num"]
        paragraphs = split_into_paragraphs(page["text"])
        section_title = guess_section_title(paragraphs)

        buffer = ""
        local_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) > max_chars:
                if buffer:
                    chunks.append(
                        {
                            "chunk_id": f"chunk_{chunk_id}",
                            "page_num": page_num,
                            "chunk_index_on_page": local_idx,
                            "section_title": section_title,
                            "text": buffer.strip(),
                        }
                    )
                    chunk_id += 1
                    local_idx += 1
                    buffer = ""

                for start in range(0, len(para), max_chars):
                    piece = para[start : start + max_chars]
                    chunks.append(
                        {
                            "chunk_id": f"chunk_{chunk_id}",
                            "page_num": page_num,
                            "chunk_index_on_page": local_idx,
                            "section_title": section_title,
                            "text": piece.strip(),
                        }
                    )
                    chunk_id += 1
                    local_idx += 1
                continue

            candidate = f"{buffer}\n{para}".strip() if buffer else para
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(
                        {
                            "chunk_id": f"chunk_{chunk_id}",
                            "page_num": page_num,
                            "chunk_index_on_page": local_idx,
                            "section_title": section_title,
                            "text": buffer.strip(),
                        }
                    )
                    chunk_id += 1
                    local_idx += 1
                buffer = para

        if buffer:
            if len(buffer) < min_chars and chunks and chunks[-1]["page_num"] == page_num:
                chunks[-1]["text"] = f"{chunks[-1]['text']}\n{buffer}".strip()
            else:
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_id}",
                        "page_num": page_num,
                        "chunk_index_on_page": local_idx,
                        "section_title": section_title,
                        "text": buffer.strip(),
                    }
                )
                chunk_id += 1

    for global_idx, chunk in enumerate(chunks):
        chunk["global_chunk_index"] = global_idx

    return chunks