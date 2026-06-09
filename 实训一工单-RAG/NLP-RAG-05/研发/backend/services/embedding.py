# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np


class EmbeddingService:
    def __init__(self, model_dir: Path, configured_path: str = "") -> None:
        self.model_dir = model_dir
        self.configured_path = configured_path
        self.backend = "hashing"
        self._encoder = None
        self.dimension = 384
        self._load_local_model()

    def _load_local_model(self) -> None:
        model_path = Path(self.configured_path) if self.configured_path else None
        candidates = []
        if model_path and model_path.exists():
            candidates.append(model_path)
        if self.model_dir.exists():
            candidates.extend([item for item in self.model_dir.iterdir() if item.is_dir()])
            candidates.extend([item for item in self.model_dir.rglob("*") if item.is_dir()])
        visited = set()
        for candidate in candidates:
            candidate_key = str(candidate.resolve())
            if candidate_key in visited:
                continue
            visited.add(candidate_key)
            if (candidate / "config.json").exists() and (candidate / "modules.json").exists():
                try:
                    from sentence_transformers import SentenceTransformer

                    self._encoder = SentenceTransformer(str(candidate), device="cpu")
                    self.backend = f"sentence_transformers:{candidate.name}"
                    sample = self._encoder.encode(["warmup"], normalize_embeddings=True)
                    self.dimension = int(sample.shape[1])
                    return
                except Exception:
                    pass
            if (candidate / "tokenizer.json").exists():
                try:
                    from transformers import AutoModel, AutoTokenizer
                    import torch

                    tokenizer = AutoTokenizer.from_pretrained(str(candidate), local_files_only=True)
                    model = AutoModel.from_pretrained(str(candidate), local_files_only=True)
                    model.eval()
                    hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 768) or 768)

                    def encoder(texts: List[str]) -> np.ndarray:
                        with torch.no_grad():
                            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
                            outputs = model(**inputs)
                            hidden = outputs.last_hidden_state.mean(dim=1)
                            norms = torch.linalg.norm(hidden, dim=1, keepdim=True).clamp_min(1e-12)
                            return (hidden / norms).cpu().numpy()

                    self._encoder = encoder
                    self.backend = f"transformers:{candidate.name}"
                    self.dimension = hidden_size
                    return
                except Exception:
                    continue

    def _hash_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in text:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dimension
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        text_list = list(texts)
        if self._encoder is None:
            return [self._hash_embed(text).tolist() for text in text_list]
        if callable(self._encoder) and self.backend.startswith("transformers:"):
            return self._encoder(text_list).tolist()
        vectors = self._encoder.encode(text_list, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def describe(self) -> str:
        return json.dumps({"backend": self.backend, "dimension": self.dimension}, ensure_ascii=False)
