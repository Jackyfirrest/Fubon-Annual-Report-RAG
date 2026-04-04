# Fubon Annual Report RAG

A RAG-based QA system for Fubon Financial 2024 annual report, designed to answer annual report questions with citations and hallucination detection.

## Project Goal

Build a retrieval-augmented generation system that:

- answers questions from the annual report
- cites source pages
- detects or refuses unsupported answers
- evaluates performance using the provided QA set

## Project Structure

- `data/raw/`: raw PDF and QA dataset
- `src/`: core modules
- `scripts/`: runnable scripts
- `results/`: predictions and evaluation outputs

## Reproducible Pipeline

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set GEMINI_API_KEY in .env.

### 2. Put raw files into data/raw/

20250516180014-7.pdf
題目一_附件_問答集.xlsx

### 3. Run full pipeline

```bash
./run.sh
```

Outputs:

results/predictions.csv
results/evaluation_summary.json
results/error_analysis.csv

### 4. Manual review

Edit results/predictions.csv, update final_is_correct, then run:

```bash
./rerun_from_review.sh
```