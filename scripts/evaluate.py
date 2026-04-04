from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.data_loader import load_qa_excel
from src.retriever import HybridRetriever
from src.prompt_builder import build_user_prompt
from src.generator import GeminiGenerator
from src.evaluator import evaluate_prediction
from src.utils import write_json

import time

def main() -> None:
    qa_df = load_qa_excel(settings.raw_qa_path)
    retriever = HybridRetriever.load(settings.hybrid_index_path)
    generator = GeminiGenerator()

    rows = []

    for _, row in qa_df.iterrows():
        qid = str(row["question_id"])
        question = row["question"]
        gold_answer = row["gold_answer"]
        gold_pages = row["gold_pages"]
        category = row["category"]
        question_type = row["question_type"]

        retrieval_results = retriever.retrieve(
            question,
            top_k=settings.top_k,
            expanded_top_k=settings.expanded_top_k,
            neighbor_window=settings.neighbor_window,
        )
        evidence_text = "\n\n".join([f"[Page {r.page_num}] {r.text}" for r in retrieval_results])
        prompt = build_user_prompt(question, retrieval_results, settings.max_context_chars)
        response = generator.generate_json(prompt)

        time.sleep(5)  # add delay to avoid hitting rate limits

        pred_answer = str(response.get("answer", "")).strip()
        pred_citations = response.get("citations", [])
        is_refusal = bool(response.get("is_refusal", False))
        reasoning_note = str(response.get("reasoning_note", "")).strip()

        eval_result = evaluate_prediction(
            question_id=qid,
            pred_answer=pred_answer,
            gold_answer=gold_answer,
            evidence_text=evidence_text,
        )

        retrieved_pages = [r.page_num for r in retrieval_results]

        rows.append(
            {
                "question_id": qid,
                "category": category,
                "question_type": question_type,
                "question": question,
                "gold_answer": gold_answer,
                "gold_pages": gold_pages,
                "pred_answer": pred_answer,
                "pred_citations": pred_citations,
                "retrieved_pages": retrieved_pages,
                "model_is_refusal": is_refusal,
                "reasoning_note": reasoning_note,
                "is_correct": eval_result.is_correct,
                "final_is_correct": eval_result.is_correct,  # 之後可人工覆寫
                "hallucination_label": eval_result.hallucination_label,
                "evaluation_note": eval_result.note,
                "match_method": eval_result.match_method,
                "needs_manual_review": eval_result.needs_manual_review,
            }
        )
        print(
            f"[DONE] Q{qid} correct={eval_result.is_correct} "
            f"manual_review={eval_result.needs_manual_review} "
            f"hallucination={eval_result.hallucination_label}"
        )

    pred_df = pd.DataFrame(rows)
    Path("results").mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(settings.predictions_path, index=False, encoding="utf-8-sig")

    overall_accuracy = float(pred_df["final_is_correct"].mean()) if len(pred_df) > 0 else 0.0
    by_category = pred_df.groupby("category")["final_is_correct"].mean().to_dict()
    by_question_type = pred_df.groupby("question_type")["final_is_correct"].mean().to_dict()
    hallucination_counts = pred_df["hallucination_label"].value_counts().to_dict()
    match_method_counts = pred_df["match_method"].value_counts().to_dict()

    summary = {
        "n_questions": int(len(pred_df)),
        "overall_accuracy": overall_accuracy,
        "accuracy_by_category": by_category,
        "accuracy_by_question_type": by_question_type,
        "hallucination_label_counts": hallucination_counts,
        "match_method_counts": match_method_counts,
        "n_needs_manual_review": int(pred_df["needs_manual_review"].sum()),
    }

    write_json(summary, settings.evaluation_summary_path)

    error_df = pred_df[~pred_df["final_is_correct"].astype(bool)].copy()
    error_df.to_csv(settings.error_analysis_path, index=False, encoding="utf-8-sig")

    print("\n=== Summary ===")
    print(summary)
    print(f"[OK] predictions saved to {settings.predictions_path}")
    print(f"[OK] summary saved to {settings.evaluation_summary_path}")
    print(f"[OK] error analysis saved to {settings.error_analysis_path}")


if __name__ == "__main__":
    main()