from __future__ import annotations

import pandas as pd


COLUMN_MAP = {
    "類別": "category",
    "類型": "question_type",
    "題號": "question_id",
    "題目": "question",
    "答案": "gold_answer",
    "來源頁數（PDF）": "gold_pages",
    "來源頁數(PDF)": "gold_pages",
    "來源頁數": "gold_pages",
}


REQUIRED_COLUMNS = [
    "question_id",
    "category",
    "question_type",
    "question",
    "gold_answer",
    "gold_pages",
]


def load_qa_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns={c: COLUMN_MAP.get(c, c) for c in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"問答集缺少必要欄位: {missing}; 實際欄位={list(df.columns)}")

    df = df[REQUIRED_COLUMNS].copy()
    df["question_id"] = df["question_id"].astype(str)
    for col in ["category", "question_type", "question", "gold_answer", "gold_pages"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df
