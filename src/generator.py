from __future__ import annotations

import json
from typing import Dict, Any

from google import genai

from .config import settings
from .prompt_builder import SYSTEM_PROMPT


class GeminiGenerator:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY 未設定，請先在 .env 設定。")
        self.client = genai.Client(api_key=self.api_key)

    def generate_json(self, user_prompt: str) -> Dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"{SYSTEM_PROMPT}\n\n{user_prompt}",
        )
        text = (response.text or "").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "answer": text,
                "citations": [],
                "is_refusal": False,
                "reasoning_note": "模型未輸出合法 JSON，已回傳原始文字。",
            }
