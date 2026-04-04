from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, List


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(records: Iterable[dict], path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(data: Any, path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_float(text: str):
    try:
        return float(text)
    except Exception:
        return None


def tokenize_for_bm25(text: str) -> list[str]:
    """
    簡單 tokenizer：
    - 英數以字串切分
    - 中文以單字切分
    - 保留百分比、數字、英文縮寫附近資訊
    """
    text = normalize_whitespace(text).lower()
    parts = re.findall(r"[a-zA-Z0-9\.\-%]+|[\u4e00-\u9fff]", text)
    return parts


def truncate_text(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]
