from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List, Dict

from rank_bm25 import BM25Okapi

from .utils import tokenize_for_bm25


@dataclass
class RetrievalResult:
    chunk_id: str
    page_num: int
    text: str
    score: float


class BM25Retriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize_for_bm25(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        tokenized_query = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[RetrievalResult] = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk["chunk_id"],
                    page_num=int(chunk["page_num"]),
                    text=chunk["text"],
                    score=float(scores[idx]),
                )
            )
        return results

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "tokenized_corpus": self.tokenized_corpus,
                    "bm25": self.bm25,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls.__new__(cls)
        obj.chunks = payload["chunks"]
        obj.tokenized_corpus = payload["tokenized_corpus"]
        obj.bm25 = payload["bm25"]
        return obj
