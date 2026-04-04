from __future__ import annotations

import re
from typing import List, Set


REFUSAL_PATTERNS = [
    r"年報未提供此資訊",
    r"無法根據文件推論",
    r"資料不足",
    r"未揭露",
]


def contains_refusal(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in REFUSAL_PATTERNS)


def normalize_number(num: str) -> str:
    return num.replace(",", "").strip()


def extract_numbers(text: str) -> List[str]:
    nums = re.findall(r"\d+[\d,\.]*%?|\d+[\d,\.]*", text)
    return [normalize_number(n) for n in nums]


def find_unsupported_numbers(answer: str, evidence_text: str) -> Set[str]:
    answer_numbers = set(extract_numbers(answer))
    evidence_numbers = set(extract_numbers(evidence_text))
    if not answer_numbers:
        return set()
    unsupported = {num for num in answer_numbers if num not in evidence_numbers}
    return unsupported


def detect_possible_numeric_hallucination(answer: str, evidence_text: str) -> bool:
    unsupported = find_unsupported_numbers(answer, evidence_text)
    return len(unsupported) > 0


def label_hallucination(pred_answer: str, evidence_text: str, is_refusal_expected: bool = False) -> str:
    if is_refusal_expected:
        return "correct_refusal" if contains_refusal(pred_answer) else "hallucinated_non_refusal"

    if contains_refusal(pred_answer):
        return "over_refusal"

    if detect_possible_numeric_hallucination(pred_answer, evidence_text):
        return "possible_numeric_hallucination"

    return "no_obvious_hallucination"