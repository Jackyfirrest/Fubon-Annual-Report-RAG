#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "[Fubon Annual Report RAG] Full Reproducible Pipeline"
echo "=================================================="

if [ ! -f ".env" ]; then
  echo "[ERROR] .env not found. Please create .env first."
  exit 1
fi

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "[OK] Activated virtual environment."
fi

mkdir -p data/raw data/processed results

PDF_PATH="${RAW_PDF_PATH:-data/raw/20250516180014-7.pdf}"
QA_PATH="${RAW_QA_PATH:-data/raw/題目一_附件_問答集.xlsx}"

if [ ! -f "$PDF_PATH" ]; then
  echo "[ERROR] PDF not found: $PDF_PATH"
  exit 1
fi

if [ ! -f "$QA_PATH" ]; then
  echo "[ERROR] QA Excel not found: $QA_PATH"
  exit 1
fi

echo
echo "==================== Step 1/4: preprocess_pdf ===================="
python scripts/preprocess_pdf.py

echo
echo "==================== Step 2/4: build_sparse_index ===================="
python scripts/build_index.py

echo
echo "==================== Step 3/4: build_dense_index ===================="
python scripts/build_dense_index.py

echo
echo "==================== Step 4/4: evaluate ===================="
python scripts/evaluate.py

echo
echo "==================== Final Summary ===================="
python - <<'PY'
import json
from pathlib import Path

summary_path = Path("results/evaluation_summary.json")
with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

print(f"n_questions          : {summary.get('n_questions')}")
print(f"overall_accuracy     : {summary.get('overall_accuracy')}")
print(f"n_needs_manual_review: {summary.get('n_needs_manual_review', 0)}")

print("\naccuracy_by_category:")
for k, v in summary.get("accuracy_by_category", {}).items():
    print(f"  - {k}: {v}")

print("\naccuracy_by_question_type:")
for k, v in summary.get("accuracy_by_question_type", {}).items():
    print(f"  - {k}: {v}")

print("\n[OK] Files generated:")
print("  - results/predictions.csv")
print("  - results/evaluation_summary.json")
print("  - results/error_analysis.csv")
PY

echo
echo "[DONE] Pipeline completed successfully."