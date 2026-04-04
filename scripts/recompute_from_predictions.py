from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.utils import write_json


TRUE_SET = {"1", "true", "t", "yes", "y"}
FALSE_SET = {"0", "false", "f", "no", "n"}


def parse_bool_like(x):
    if pd.isna(x):
        return None

    if isinstance(x, bool):
        return x

    if isinstance(x, (int, float)):
        if x == 1:
            return True
        if x == 0:
            return False

    s = str(x).strip().lower()
    if s in TRUE_SET:
        return True
    if s in FALSE_SET:
        return False
    return None


def main() -> None:
    pred_df = pd.read_csv(settings.predictions_path)

    if "final_is_correct" not in pred_df.columns:
        raise ValueError("predictions.csv 缺少 final_is_correct 欄位。")

    parsed_final = pred_df["final_is_correct"].apply(parse_bool_like)

    if parsed_final.isna().any():
        bad_rows = pred_df.loc[parsed_final.isna(), ["question_id", "final_is_correct"]]
        raise ValueError(
            "final_is_correct 有無法解析的值，請修正後再重跑。\n"
            + bad_rows.to_string(index=False)
        )

    pred_df["final_is_correct"] = parsed_final.astype(bool)

    overall_accuracy = float(pred_df["final_is_correct"].mean()) if len(pred_df) > 0 else 0.0
    by_category = pred_df.groupby("category")["final_is_correct"].mean().to_dict()
    by_question_type = pred_df.groupby("question_type")["final_is_correct"].mean().to_dict()
    hallucination_counts = pred_df["hallucination_label"].value_counts().to_dict()

    if "match_method" in pred_df.columns:
        match_method_counts = pred_df["match_method"].value_counts().to_dict()
    else:
        match_method_counts = {}

    if "needs_manual_review" in pred_df.columns:
        n_needs_manual_review = int(pred_df["needs_manual_review"].astype(bool).sum())
    else:
        n_needs_manual_review = 0

    n_manual_override = 0
    if "is_correct" in pred_df.columns:
        parsed_auto = pred_df["is_correct"].apply(parse_bool_like)
        if parsed_auto.notna().all():
            pred_df["is_correct"] = parsed_auto.astype(bool)
            n_manual_override = int((pred_df["is_correct"] != pred_df["final_is_correct"]).sum())

    summary = {
        "n_questions": int(len(pred_df)),
        "overall_accuracy": overall_accuracy,
        "accuracy_by_category": by_category,
        "accuracy_by_question_type": by_question_type,
        "hallucination_label_counts": hallucination_counts,
        "match_method_counts": match_method_counts,
        "n_needs_manual_review": n_needs_manual_review,
        "n_manual_override": n_manual_override,
    }

    pred_df.to_csv(settings.predictions_path, index=False, encoding="utf-8-sig")

    error_df = pred_df[~pred_df["final_is_correct"]].copy()
    error_df.to_csv(settings.error_analysis_path, index=False, encoding="utf-8-sig")

    write_json(summary, settings.evaluation_summary_path)

    print("[OK] Recomputed summary from reviewed predictions.csv")
    print(summary)
    print(f"[OK] summary saved to {settings.evaluation_summary_path}")
    print(f"[OK] error analysis saved to {settings.error_analysis_path}")


if __name__ == "__main__":
    main()