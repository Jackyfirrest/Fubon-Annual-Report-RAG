from __future__ import annotations

import re
from typing import List, Dict, Any


REFUSAL_PATTERNS = [
    r"年報未提供此資訊",
    r"無法根據文件推論",
    r"資料不足",
    r"未揭露",
]


def contains_refusal(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in REFUSAL_PATTERNS)


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d+[\d,\.]*%?|\d+[\d,\.]*", text)


def detect_possible_numeric_hallucination(answer: str, evidence_text: str) -> bool:
    """
    很簡單的數值 hallucination 檢查：
    若 answer 裡的數字完全沒出現在 evidence 中，視為可疑。
    計算題可能誤判，因此結果只作為 heuristic flag。
    """
    answer_numbers = extract_numbers(answer)
    if not answer_numbers:
        return False

    evidence_numbers = set(extract_numbers(evidence_text))
    unsupported = [num for num in answer_numbers if num not in evidence_numbers]
    return len(unsupported) > 0


def label_hallucination(pred_answer: str, evidence_text: str, is_refusal_expected: bool = False) -> str:
    if is_refusal_expected:
        return "correct_refusal" if contains_refusal(pred_answer) else "hallucinated_non_refusal"

    if contains_refusal(pred_answer):
        return "over_refusal"

    if detect_possible_numeric_hallucination(pred_answer, evidence_text):
        return "possible_numeric_hallucination"

    return "no_obvious_hallucination"
