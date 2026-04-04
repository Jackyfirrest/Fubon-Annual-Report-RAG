#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "[Fubon Annual Report RAG] Recompute From Reviewed Predictions"
echo "=================================================="

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "[OK] Activated virtual environment."
fi

if [ ! -f "results/predictions.csv" ]; then
  echo "[ERROR] results/predictions.csv not found."
  echo "Please run ./run.sh first."
  exit 1
fi

python scripts/recompute_from_predictions.py

echo
echo "==================== Updated Summary ===================="
python - <<'PY'
import json
from pathlib import Path

summary_path = Path("results/evaluation_summary.json")
with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

print(f"n_questions          : {summary.get('n_questions')}")
print(f"overall_accuracy     : {summary.get('overall_accuracy')}")
print(f"n_manual_override    : {summary.get('n_manual_override', 0)}")

print("\naccuracy_by_category:")
for k, v in summary.get("accuracy_by_category", {}).items():
    print(f"  - {k}: {v}")

print("\n[OK] Updated files:")
print("  - results/evaluation_summary.json")
print("  - results/error_analysis.csv")
PY

echo
echo "[DONE] Recompute finished."