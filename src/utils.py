from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, List
from difflib import SequenceMatcher


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
    text = normalize_whitespace(text).lower()
    parts = re.findall(r"[a-zA-Z0-9\.\-%]+|[\u4e00-\u9fff]", text)
    return parts


def tokenize_keywords(text: str) -> list[str]:
    text = normalize_whitespace(text).lower()
    tokens = re.findall(r"[a-zA-Z0-9\.\-%]+|[\u4e00-\u9fff]{1,4}", text)
    stop_chars = {"請", "問", "是", "多少", "為何", "哪些", "有", "的", "與", "及", "和", "年", "度", "在"}
    return [t for t in tokens if t not in stop_chars]


def truncate_text(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def min_max_scale(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-12:
        return [1.0 if vmax > 0 else 0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def partial_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if short in long_:
        return 1.0
    best = 0.0
    step = max(1, len(short) // 4)
    for i in range(0, max(1, len(long_) - len(short) + 1), step):
        cand = long_[i : i + len(short)]
        best = max(best, SequenceMatcher(None, short, cand).ratio())
    return best