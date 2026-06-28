"""向量数据库封装（ChromaDB）"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from src.services.embedding_service import EmbeddingService


class VectorStore:
    """ChromaDB 向量存储封装"""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self._embedding = embedding_service
        persist_dir = Path(settings.VECTOR_DB_PATH_RESOLVED)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.Client(
            ChromaSettings(
                persist_directory=str(persist_dir),
                anonymized_telemetry=False,
            )
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.VECTOR_DB_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add_texts(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """添加文本到向量库"""
        if self._embedding:
            embeddings = self._embedding.embed(texts)
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas or [{}] * len(ids),
            )
        else:
            self._collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas or [{}] * len(ids),
            )

    def similarity_search(
        self, query: str, top_k: int = 5
    ) -> list[dict]:
        """相似度搜索"""
        if self._embedding:
            query_emb = self._embedding.embed([query])[0]
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
            )
        else:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
            )

        items = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                items.append({
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })
        return items

    def count(self) -> int:
        return self._collection.count()

    def delete_collection(self) -> None:
        self._client.delete_collection(settings.VECTOR_DB_COLLECTION)
