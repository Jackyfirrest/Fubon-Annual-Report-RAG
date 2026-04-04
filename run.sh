#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "[Fubon Annual Report RAG] Full Reproducible Pipeline"
echo "Project root: $PROJECT_ROOT"
echo "=================================================="

# -----------------------------
# 0. Basic checks
# -----------------------------
if [ ! -f ".env" ]; then
  echo "[ERROR] .env not found. Please create .env first."
  echo "You can copy from .env.example:"
  echo "  cp .env.example .env"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[WARNING] .venv not found."
  echo "Recommended setup:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
fi

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "[OK] Activated virtual environment."
fi

mkdir -p data/raw data/processed data/outputs results slides/figures

# -----------------------------
# 1. Check raw data
# -----------------------------
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

echo "[OK] Raw files found:"
echo "  PDF: $PDF_PATH"
echo "  QA : $QA_PATH"

# -----------------------------
# 2. Run preprocess
# -----------------------------
echo
echo "==================== Step 1/3: preprocess_pdf ===================="
python scripts/preprocess_pdf.py

# -----------------------------
# 3. Build hybrid index
# -----------------------------
echo
echo "==================== Step 2/3: build_index ===================="
python scripts/build_index.py

# -----------------------------
# 4. Evaluate all questions
# -----------------------------
echo
echo "==================== Step 3/3: evaluate ===================="
python scripts/evaluate.py

# -----------------------------
# 5. Print summary
# -----------------------------
echo
echo "==================== Final Summary ===================="
python - <<'PY'
import json
from pathlib import Path

summary_path = Path("results/evaluation_summary.json")
if not summary_path.exists():
    print("[ERROR] results/evaluation_summary.json not found.")
    raise SystemExit(1)

with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

print(f"n_questions          : {summary.get('n_questions')}")
print(f"overall_accuracy     : {summary.get('overall_accuracy')}")
print(f"n_needs_manual_review: {summary.get('n_needs_manual_review', 0)}")
print(f"n_manual_override    : {summary.get('n_manual_override', 0)}")

print("\naccuracy_by_category:")
for k, v in summary.get("accuracy_by_category", {}).items():
    print(f"  - {k}: {v}")

print("\naccuracy_by_question_type:")
for k, v in summary.get("accuracy_by_question_type", {}).items():
    print(f"  - {k}: {v}")

print("\nhallucination_label_counts:")
for k, v in summary.get("hallucination_label_counts", {}).items():
    print(f"  - {k}: {v}")

print("\n[OK] Files generated:")
print("  - results/predictions.csv")
print("  - results/evaluation_summary.json")
print("  - results/error_analysis.csv")
PY

echo
echo "=================================================="
echo "[DONE] Pipeline completed successfully."
echo "=================================================="