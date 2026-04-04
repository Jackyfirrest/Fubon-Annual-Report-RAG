from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    raw_pdf_path: str = os.getenv("RAW_PDF_PATH", "data/raw/20250516180014-7.pdf")
    raw_qa_path: str = os.getenv("RAW_QA_PATH", "data/raw/題目一_附件_問答集.xlsx")

    processed_pages_path: str = os.getenv("PROCESSED_PAGES_PATH", "data/processed/pages.jsonl")
    processed_chunks_path: str = os.getenv("PROCESSED_CHUNKS_PATH", "data/processed/chunks.jsonl")

    hybrid_index_path: str = os.getenv("HYBRID_INDEX_PATH", "data/processed/hybrid_index.pkl")

    predictions_path: str = os.getenv("PREDICTIONS_PATH", "results/predictions.csv")
    evaluation_summary_path: str = os.getenv("EVALUATION_SUMMARY_PATH", "results/evaluation_summary.json")
    error_analysis_path: str = os.getenv("ERROR_ANALYSIS_PATH", "results/error_analysis.csv")
    manual_review_path: str = os.getenv("MANUAL_REVIEW_PATH", "results/manual_review.csv")

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    top_k: int = int(os.getenv("TOP_K", "8"))
    expanded_top_k: int = int(os.getenv("EXPANDED_TOP_K", "10"))
    neighbor_window: int = int(os.getenv("NEIGHBOR_WINDOW", "1"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "16000"))
    max_subquestions: int = int(os.getenv("MAX_SUBQUESTIONS", "3"))

    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.55"))
    tfidf_weight: float = float(os.getenv("TFIDF_WEIGHT", "0.25"))
    overlap_weight: float = float(os.getenv("OVERLAP_WEIGHT", "0.20"))

    fuzzy_threshold: float = float(os.getenv("FUZZY_THRESHOLD", "0.82"))
    partial_fuzzy_threshold: float = float(os.getenv("PARTIAL_FUZZY_THRESHOLD", "0.90"))


settings = Settings()