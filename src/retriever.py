from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import tokenize_for_bm25, tokenize_keywords, min_max_scale
from .query_processing import expand_query, split_subquestions, detect_question_mode


@dataclass
class RetrievalResult:
    chunk_id: str
    page_num: int
    text: str
    score: float
    section_title: str = ""


class HybridRetriever:
    def __init__(self, chunks: List[Dict], bm25_weight: float = 0.55, tfidf_weight: float = 0.25, overlap_weight: float = 0.20):
        self.chunks = chunks
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight
        self.overlap_weight = overlap_weight

        self.tokenized_corpus = [tokenize_for_bm25(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.tfidf_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform([chunk["text"] for chunk in chunks])

        self.chunk_id_to_idx = {chunk["chunk_id"]: i for i, chunk in enumerate(chunks)}

    def _keyword_overlap_score(self, query: str, chunk_text: str) -> float:
        q_tokens = set(tokenize_keywords(query))
        if not q_tokens:
            return 0.0
        hit = sum(1 for t in q_tokens if t in chunk_text.lower())
        return hit / max(1, len(q_tokens))

    def _single_query_scores(self, query: str) -> np.ndarray:
        tokenized_query = tokenize_for_bm25(query)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query), dtype=float)
        bm25_scores = np.array(min_max_scale(bm25_scores.tolist()), dtype=float)

        query_vec = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix @ query_vec.T).toarray().reshape(-1)
        tfidf_scores = np.array(min_max_scale(tfidf_scores.tolist()), dtype=float)

        overlap_scores = np.array([self._keyword_overlap_score(query, c["text"]) for c in self.chunks], dtype=float)

        return self.bm25_weight * bm25_scores + self.tfidf_weight * tfidf_scores + self.overlap_weight * overlap_scores

    def retrieve(self, query: str, top_k: int = 8, expanded_top_k: int = 10, neighbor_window: int = 1) -> List[RetrievalResult]:
        question_mode = detect_question_mode(query)
        subquestions = split_subquestions(query)
        queries = []
        for q in [query] + subquestions:
            queries.extend(expand_query(q))
        dedup_queries = []
        for q in queries:
            if q not in dedup_queries:
                dedup_queries.append(q)

        total_scores = np.zeros(len(self.chunks), dtype=float)
        for q in dedup_queries:
            total_scores += self._single_query_scores(q)

        if question_mode in {"summary", "multi_hop", "calculation"}:
            page_bonus = np.array([0.08 if len(c["text"]) > 120 else 0.0 for c in self.chunks], dtype=float)
            total_scores += page_bonus

        ranked_indices = np.argsort(-total_scores)[:top_k]

        expanded_indices = set(ranked_indices.tolist())
        for idx in ranked_indices:
            center = self.chunks[idx]["global_chunk_index"]
            for offset in range(-neighbor_window, neighbor_window + 1):
                ni = center + offset
                if 0 <= ni < len(self.chunks):
                    expanded_indices.add(ni)

        final_indices = sorted(
            list(expanded_indices),
            key=lambda i: total_scores[i],
            reverse=True,
        )[:expanded_top_k]

        results: List[RetrievalResult] = []
        used_pages = set()
        for idx in final_indices:
            chunk = self.chunks[idx]
            page_num = int(chunk["page_num"])
            section_title = chunk.get("section_title", "")

            diversity_bonus = 0.02 if page_num not in used_pages else 0.0
            score = float(total_scores[idx] + diversity_bonus)
            used_pages.add(page_num)

            results.append(
                RetrievalResult(
                    chunk_id=chunk["chunk_id"],
                    page_num=page_num,
                    text=chunk["text"],
                    score=score,
                    section_title=section_title,
                )
            )

        results = sorted(results, key=lambda r: r.score, reverse=True)
        return results

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "tokenized_corpus": self.tokenized_corpus,
                    "bm25": self.bm25,
                    "tfidf_vectorizer": self.tfidf_vectorizer,
                    "tfidf_matrix": self.tfidf_matrix,
                    "bm25_weight": self.bm25_weight,
                    "tfidf_weight": self.tfidf_weight,
                    "overlap_weight": self.overlap_weight,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "HybridRetriever":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls.__new__(cls)
        obj.chunks = payload["chunks"]
        obj.tokenized_corpus = payload["tokenized_corpus"]
        obj.bm25 = payload["bm25"]
        obj.tfidf_vectorizer = payload["tfidf_vectorizer"]
        obj.tfidf_matrix = payload["tfidf_matrix"]
        obj.bm25_weight = payload["bm25_weight"]
        obj.tfidf_weight = payload["tfidf_weight"]
        obj.overlap_weight = payload["overlap_weight"]
        obj.chunk_id_to_idx = {chunk["chunk_id"]: i for i, chunk in enumerate(obj.chunks)}
        return obj