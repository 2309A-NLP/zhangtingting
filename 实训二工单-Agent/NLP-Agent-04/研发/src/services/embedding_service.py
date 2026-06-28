"""Embedding 服务封装"""

from __future__ import annotations

from typing import Optional

from config import settings


class EmbeddingService:
    """文本向量化服务"""

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model = None  # 延迟加载

    def _lazy_load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转为向量"""
        self._lazy_load()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]
