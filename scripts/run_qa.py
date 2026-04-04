from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.retriever import BM25Retriever
from src.prompt_builder import build_user_prompt
from src.generator import GeminiGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True, help="單題測試問題")
    parser.add_argument("--top_k", type=int, default=settings.top_k)
    args = parser.parse_args()

    retriever = BM25Retriever.load(settings.bm25_index_path)
    results = retriever.retrieve(args.question, top_k=args.top_k)

    print("\n=== Retrieved Evidence ===")
    for i, r in enumerate(results, start=1):
        print(f"[{i}] page={r.page_num}, score={r.score:.4f}")
        print(r.text[:500])
        print("-" * 80)

    prompt = build_user_prompt(args.question, results, settings.max_context_chars)
    generator = GeminiGenerator()
    response = generator.generate_json(prompt)

    print("\n=== Model Response ===")
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
