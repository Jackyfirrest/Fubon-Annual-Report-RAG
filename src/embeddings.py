from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings


class LocalSentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize_embeddings: bool | None = None,
    ):
        self.model_name = model_name or settings.local_embedding_model
        self.device = device or settings.local_embedding_device
        self.normalize_embeddings = (
            settings.local_embedding_normalize
            if normalize_embeddings is None
            else normalize_embeddings
        )

        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_texts(self, texts: List[str], batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, settings.local_embedding_dim), dtype=np.float32)

        batch_size = batch_size or settings.local_embedding_batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )[0]
        return vec.astype(np.float32)