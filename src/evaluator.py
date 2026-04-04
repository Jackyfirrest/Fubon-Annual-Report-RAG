from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .hallucination import label_hallucination, find_unsupported_numbers
from .config import settings
from .utils import fuzzy_ratio, partial_ratio


@dataclass
class EvaluationResult:
    is_correct: bool
    hallucination_label: str
    note: str
    match_method: str
    needs_manual_review: bool


REFUSAL_QUESTIONS = {"28", "29"}


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("％", "%")
    text = text.replace("臺", "台")
    text = text.replace("　", " ")
    text = text.replace("～", "~")
    text = re.sub(r"[：:;；、，,。.\(\)\[\]「」『』／/]", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def split_answer_items(text: str) -> List[str]:
    text = text.replace("；", "|").replace(";", "|").replace("，", "|").replace(",", "|")
    text = text.replace(" / ", "|").replace("、", "|")
    return [t.strip() for t in text.split("|") if t.strip()]


def extract_numbers(text: str) -> List[str]:
    nums = re.findall(r"\d+[\d,\.]*%?|\d+[\d,\.]*", text)
    return [n.replace(",", "") for n in nums]


def numeric_consistency_score(pred_answer: str, gold_answer: str) -> float:
    pred_nums = set(extract_numbers(pred_answer))
    gold_nums = set(extract_numbers(gold_answer))
    if not gold_nums:
        return 1.0
    hit = len(pred_nums & gold_nums)
    return hit / max(1, len(gold_nums))


def item_coverage_score(pred_answer: str, gold_answer: str) -> float:
    gold_items = split_answer_items(gold_answer)
    if len(gold_items) <= 1:
        return 1.0
    pred_norm = normalize_text(pred_answer)
    hits = 0
    for item in gold_items:
        item_norm = normalize_text(item)
        if item_norm and item_norm in pred_norm:
            hits += 1
    return hits / max(1, len(gold_items))


def evaluate_prediction(
    question_id: str,
    pred_answer: str,
    gold_answer: str,
    evidence_text: str,
) -> EvaluationResult:
    is_refusal_expected = question_id in REFUSAL_QUESTIONS

    pred_norm = normalize_text(pred_answer)
    gold_norm = normalize_text(gold_answer)

    if is_refusal_expected:
        is_correct = any(x in pred_answer for x in ["無法", "未提供", "未揭露", "資料不足"])
        hallucination_label = label_hallucination(pred_answer, evidence_text, is_refusal_expected=True)
        return EvaluationResult(
            is_correct=is_correct,
            hallucination_label=hallucination_label,
            note="refusal question",
            match_method="refusal_rule",
            needs_manual_review=False,
        )

    if not pred_norm:
        return EvaluationResult(
            is_correct=False,
            hallucination_label=label_hallucination(pred_answer, evidence_text, is_refusal_expected=False),
            note="empty answer",
            match_method="empty",
            needs_manual_review=False,
        )

    if gold_norm in pred_norm or pred_norm in gold_norm:
        return EvaluationResult(
            is_correct=True,
            hallucination_label=label_hallucination(pred_answer, evidence_text, is_refusal_expected=False),
            note="substring match",
            match_method="substring",
            needs_manual_review=False,
        )

    fuzzy = fuzzy_ratio(pred_norm, gold_norm)
    partial = partial_ratio(pred_norm, gold_norm)
    num_score = numeric_consistency_score(pred_answer, gold_answer)
    item_score = item_coverage_score(pred_answer, gold_answer)
    unsupported_nums = find_unsupported_numbers(pred_answer, evidence_text)

    if num_score == 1.0 and item_score >= 0.8 and partial >= 0.75:
        return EvaluationResult(
            is_correct=True,
            hallucination_label=label_hallucination(pred_answer, evidence_text, is_refusal_expected=False),
            note=f"semantic-ish match: num_score={num_score:.2f}, item_score={item_score:.2f}, partial={partial:.2f}",
            match_method="numeric_item_match",
            needs_manual_review=False,
        )

    if fuzzy >= settings.fuzzy_threshold or partial >= settings.partial_fuzzy_threshold:
        return EvaluationResult(
            is_correct=True,
            hallucination_label=label_hallucination(pred_answer, evidence_text, is_refusal_expected=False),
            note=f"fuzzy match: fuzzy={fuzzy:.2f}, partial={partial:.2f}",
            match_method="fuzzy",
            needs_manual_review=False,
        )

    needs_manual_review = (
        (num_score >= 0.5 and partial >= 0.65)
        or (item_score >= 0.5 and partial >= 0.60)
        or (len(unsupported_nums) == 0 and partial >= 0.55)
    )

    hallucination_label = label_hallucination(pred_answer, evidence_text, is_refusal_expected=False)
    return EvaluationResult(
        is_correct=False,
        hallucination_label=hallucination_label,
        note=f"no auto match: fuzzy={fuzzy:.2f}, partial={partial:.2f}, num_score={num_score:.2f}, item_score={item_score:.2f}, unsupported_nums={sorted(list(unsupported_nums))}",
        match_method="no_match",
        needs_manual_review=needs_manual_review,
    )