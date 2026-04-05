from __future__ import annotations

import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []

    def add(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        if len(vectors) != len(metadata):
            raise ValueError("vectors 與 metadata 長度不一致。")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[float, Dict]]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((float(score), self.metadata[idx]))
        return results

    def save(self, index_path: str, meta_path: str) -> None:
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "dim": self.dim,
                    "metadata": self.metadata,
                },
                f,
            )

    @classmethod
    def load(cls, index_path: str, meta_path: str) -> "FaissVectorStore":
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            payload = pickle.load(f)

        obj = cls.__new__(cls)
        obj.dim = payload["dim"]
        obj.index = index
        obj.metadata = payload["metadata"]
        return obj