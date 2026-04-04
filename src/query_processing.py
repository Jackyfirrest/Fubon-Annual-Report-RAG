from __future__ import annotations

import re
from typing import List


QUESTION_TYPE_KEYWORDS = {
    "calculation": ["計算", "比例", "成長率", "減碳", "相較", "合計", "多少%", "百分比"],
    "refusal": ["預測", "推估", "國泰金控", "未提供", "未揭露"],
    "summary": ["簡述", "總結", "彙整", "共同", "策略", "措施", "行動", "比較", "列出", "有哪些"],
    "multi_hop": ["與", "及", "和", "各是什麼", "各是多少", "分別", "橫跨", "前後關聯"],
}

SYNONYM_MAP = {
    "稅後淨利": ["合併稅後淨利", "全年稅後淨利", "淨利"],
    "每股盈餘": ["eps"],
    "總資產": ["合併總資產", "資產"],
    "資本適足率": ["car"],
    "逾放比": ["npl", "不良放款比率"],
    "股東權益報酬率": ["roe"],
    "資產報酬率": ["roa"],
    "溫室氣體": ["範疇一", "範疇二", "排放"],
    "富邦人壽": ["人壽"],
    "台北富邦銀行": ["北富銀", "銀行"],
    "富邦證券": ["證券"],
    "富邦產險": ["產險"],
}


def detect_question_mode(question: str) -> str:
    for mode, keywords in QUESTION_TYPE_KEYWORDS.items():
        if any(k in question for k in keywords):
            return mode
    return "lookup"


def split_subquestions(question: str, max_subquestions: int = 3) -> List[str]:
    q = question.strip().rstrip("？?")
    separators = ["；", ";", "。", "，", ",", "以及", "並且"]
    tmp = q
    for sep in separators:
        tmp = tmp.replace(sep, "|")

    if "？" in question or "?" in question:
        parts = [p.strip() for p in re.split(r"[？?]", question) if p.strip()]
        if len(parts) > 1:
            return parts[:max_subquestions]

    if "分別" in q or "各是" in q or "各為" in q:
        parts = [p.strip() for p in tmp.split("|") if p.strip()]
        return parts[:max_subquestions] if len(parts) > 1 else [q]

    if ("和" in q or "與" in q or "及" in q) and ("多少" in q or "為何" in q or "是什麼" in q):
        return [q]

    parts = [p.strip() for p in tmp.split("|") if p.strip()]
    return parts[:max_subquestions] if len(parts) > 1 else [q]


def expand_query(question: str) -> List[str]:
    queries = [question.strip()]
    for key, synonyms in SYNONYM_MAP.items():
        if key in question:
            queries.append(question + " " + " ".join(synonyms))
    if "2024" in question and "113" not in question:
        queries.append(question.replace("2024", "113"))
    if "113" in question and "2024" not in question:
        queries.append(question.replace("113", "2024"))
    if "年報" not in question:
        queries.append("富邦金控 年報 " + question)
    seen = []
    for q in queries:
        if q not in seen:
            seen.append(q)
    return seen[:4]