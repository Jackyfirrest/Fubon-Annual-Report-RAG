from __future__ import annotations

import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.utils import read_jsonl
from src.embeddings import LocalSentenceTransformerEmbedder
from src.vector_store import FaissVectorStore


def batched(items: List[dict], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main() -> None:
    chunks = read_jsonl(settings.processed_chunks_path)

    embedder = LocalSentenceTransformerEmbedder(
        model_name=settings.local_embedding_model,
        device=settings.local_embedding_device,
        normalize_embeddings=settings.local_embedding_normalize,
    )

    store = FaissVectorStore(dim=settings.local_embedding_dim)

    total = len(chunks)
    print(f"[INFO] Building dense index for {total} chunks...")
    print(f"[INFO] Local embedding model: {settings.local_embedding_model}")
    print(f"[INFO] Device: {settings.local_embedding_device}")

    for batch_idx, batch in enumerate(batched(chunks, settings.local_embedding_batch_size), start=1):
        texts = [c["text"] for c in batch]
        vectors = embedder.embed_texts(texts, batch_size=settings.local_embedding_batch_size)

        metadata = []
        for c in batch:
            metadata.append(
                {
                    "chunk_id": c["chunk_id"],
                    "page_num": c["page_num"],
                    "text": c["text"],
                    "section_title": c.get("section_title", ""),
                    "global_chunk_index": c.get("global_chunk_index", -1),
                }
            )

        store.add(vectors, metadata)
        print(
            f"[OK] Embedded batch {batch_idx}, "
            f"processed {min(batch_idx * settings.local_embedding_batch_size, total)}/{total}"
        )

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    store.save(settings.dense_index_path, settings.dense_meta_path)
    print(f"[OK] Saved FAISS index to {settings.dense_index_path}")
    print(f"[OK] Saved FAISS metadata to {settings.dense_meta_path}")


if __name__ == "__main__":
    main()