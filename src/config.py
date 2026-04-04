from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    raw_pdf_path: str = os.getenv("RAW_PDF_PATH", "data/raw/20250516180014-7.pdf")
    raw_qa_path: str = os.getenv("RAW_QA_PATH", "data/raw/題目一_附件_問答集.xlsx")
    processed_pages_path: str = "data/processed/pages.jsonl"
    processed_chunks_path: str = "data/processed/chunks.jsonl"
    bm25_index_path: str = "data/processed/bm25.pkl"
    predictions_path: str = "results/predictions.csv"
    evaluation_summary_path: str = "results/evaluation_summary.json"
    error_analysis_path: str = "results/error_analysis.csv"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))


settings = Settings()
