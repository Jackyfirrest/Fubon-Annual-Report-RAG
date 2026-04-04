from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any

from .hallucination import label_hallucination


@dataclass
class EvaluationResult:
    is_correct: bool
    hallucination_label: str
    note: str


REFUSAL_QUESTIONS = {"28", "29"}


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("％", "%")
    text = text.replace("臺", "台")
    text = re.sub(r"\s+", "", text)
    return text


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
        is_correct = "無法" in pred_answer or "未提供" in pred_answer or "未揭露" in pred_answer
        hallucination_label = label_hallucination(pred_answer, evidence_text, is_refusal_expected=True)
        return EvaluationResult(
            is_correct=is_correct,
            hallucination_label=hallucination_label,
            note="refusal question",
        )

    is_correct = gold_norm in pred_norm or pred_norm in gold_norm

    hallucination_label = label_hallucination(pred_answer, evidence_text, is_refusal_expected=False)
    note = "substring match"

    return EvaluationResult(
        is_correct=is_correct,
        hallucination_label=hallucination_label,
        note=note,
    )
