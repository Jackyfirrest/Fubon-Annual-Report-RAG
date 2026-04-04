from __future__ import annotations

from typing import List

from .retriever import RetrievalResult
from .query_processing import detect_question_mode


SYSTEM_PROMPT = """你是一個金融文件問答助手。你只能根據提供的富邦金控年報內容回答問題。

強制規則：
1. 只能使用 evidence，不可自行猜測，不可補充年報沒有寫的資訊。
2. 若 evidence 不足，必須拒答，固定使用：年報未提供此資訊，無法根據文件推論。
3. 若問題包含多個子問題，請分點完整回答，不可漏答。
4. 若涉及數字，答案中的每個數字都必須可被 evidence 直接支持；若無法支持就拒答。
5. 若是計算題，請先列出使用到的原始數值，再給最終結果。
6. 你必須輸出 JSON，不能輸出 markdown code fence。
"""

JSON_SCHEMA_INSTRUCTION = """
請輸出以下 JSON 格式：
{
  "answer": "最終回答",
  "citations": [頁碼整數, 頁碼整數],
  "is_refusal": true 或 false,
  "reasoning_note": "簡短說明依據哪些頁碼、哪些關鍵數字或為何拒答"
}
"""


def build_context(results: List[RetrievalResult], max_chars: int = 16000) -> str:
    sections = []
    total = 0
    for r in results:
        title = f" | section={r.section_title}" if r.section_title else ""
        block = f"[Page {r.page_num}{title}]\n{r.text}\n"
        total += len(block)
        if total > max_chars:
            break
        sections.append(block)
    return "\n".join(sections)


def build_user_prompt(question: str, results: List[RetrievalResult], max_context_chars: int = 16000) -> str:
    context = build_context(results, max_context_chars)
    mode = detect_question_mode(question)

    extra_instruction = {
        "lookup": "請直接回答，並附上最相關頁碼。",
        "summary": "請先彙整重點，再輸出精簡答案，避免遺漏子公司或重點項目。",
        "multi_hop": "這題可能涉及跨頁或多子問題，請先逐一比對 evidence 再回答。",
        "calculation": "這題需要計算，請務必把使用到的原始數值與計算結果寫清楚。",
        "refusal": "若 evidence 沒有直接答案，請拒答，不可預測或推論。",
    }[mode]

    return f"""以下是從富邦金控 2024 年報檢索出的 evidence：

{context}

問題：{question}

補充要求：{extra_instruction}

請嚴格根據以上 evidence 回答。
{JSON_SCHEMA_INSTRUCTION}
"""