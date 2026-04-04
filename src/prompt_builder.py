from __future__ import annotations

from typing import List

from .retriever import RetrievalResult
from .utils import truncate_text


SYSTEM_PROMPT = """你是一個金融文件問答助手。你只能根據提供的年報內容回答問題。
規則：
1. 只能使用提供的 evidence，不可自行猜測。
2. 若 evidence 無法支持答案，必須明確拒答，使用：年報未提供此資訊，無法根據文件推論。
3. 若問題包含多個子問題，請逐一回答。
4. 若涉及數字，優先逐字引用 evidence 中能支持的數值。
5. 請輸出 JSON，不要加 markdown code fence。
"""


JSON_SCHEMA_INSTRUCTION = """
請輸出以下 JSON 格式：
{
  "answer": "最終回答",
  "citations": [頁碼整數, 頁碼整數],
  "is_refusal": true 或 false,
  "reasoning_note": "簡短說明是依據哪些 evidence 或為何拒答"
}
"""


def build_context(results: List[RetrievalResult], max_chars: int = 12000) -> str:
    sections = []
    total = 0
    for r in results:
        block = f"[Page {r.page_num}]\n{r.text}\n"
        total += len(block)
        if total > max_chars:
            break
        sections.append(block)
    return "\n".join(sections)


def build_user_prompt(question: str, results: List[RetrievalResult], max_context_chars: int = 12000) -> str:
    context = build_context(results, max_context_chars)
    return f"""以下是從富邦金控 2024 年報檢索出的 evidence：

{context}

問題：{question}

請嚴格根據以上 evidence 回答。
{JSON_SCHEMA_INSTRUCTION}
"""
