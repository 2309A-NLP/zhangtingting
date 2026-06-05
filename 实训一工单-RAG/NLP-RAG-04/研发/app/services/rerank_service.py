# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from app.config import settings


class RerankService:
    def __init__(self, model_path: str = "") -> None:
        self.model_path = Path(model_path or settings.model_dir / "reranker" / "bge-reranker-v2-m3")
        self.backend = "disabled"
        self._reranker = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            return
        try:
            from FlagEmbedding import FlagReranker

            self._reranker = FlagReranker(str(self.model_path), use_fp16=False)
            self.backend = f"FlagReranker:{self.model_path.name}"
        except Exception:
            self._reranker = None
            self.backend = "disabled"

    def is_enabled(self) -> bool:
        return self._reranker is not None

    def rerank(self, query: str, candidates: List[Dict[str, object]], top_n: int) -> List[Dict[str, object]]:
        if not self.is_enabled() or not candidates:
            return candidates[:top_n]
        pairs = [[query, str(item.get("text") or "")] for item in candidates]
        scores = self._reranker.compute_score(pairs, normalize=True)
        reranked: List[Dict[str, object]] = []
        for item, score in zip(candidates, scores):
            enriched = dict(item)
            enriched["rerank_score"] = float(score)
            reranked.append(enriched)
        reranked.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
        return reranked[:top_n]
