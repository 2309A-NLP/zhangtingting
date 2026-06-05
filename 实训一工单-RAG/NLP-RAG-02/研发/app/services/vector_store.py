# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from app.config import settings
from app.services.text_utils import Chunk, keyword_overlap_score, normalize_whitespace


class SimpleBM25Index:
    def __init__(self, corpus_tokens: List[List[str]], k1: float, b: float) -> None:
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(tokens) for tokens in corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.term_frequencies = [Counter(tokens) for tokens in corpus_tokens]
        self.idf = self._build_idf(corpus_tokens)

    def _build_idf(self, corpus_tokens: List[List[str]]) -> Dict[str, float]:
        doc_freq: Counter[str] = Counter()
        for tokens in corpus_tokens:
            doc_freq.update(set(tokens))

        total_docs = len(corpus_tokens)
        idf: Dict[str, float] = {}
        for token, freq in doc_freq.items():
            idf[token] = math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
        return idf

    def score(self, query_tokens: List[str]) -> List[float]:
        if not query_tokens or not self.corpus_tokens:
            return [0.0] * len(self.corpus_tokens)

        query_tf = Counter(query_tokens)
        scores = [0.0] * len(self.corpus_tokens)
        for index, doc_tf in enumerate(self.term_frequencies):
            doc_len = self.doc_lengths[index] or 1
            denom_norm = self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-6))
            total = 0.0
            for token, qtf in query_tf.items():
                tf = doc_tf.get(token, 0)
                if tf <= 0:
                    continue
                idf = self.idf.get(token, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + denom_norm
                total += idf * (numerator / max(denominator, 1e-6)) * qtf
            scores[index] = total
        return scores


class MilvusVectorStore:
    def __init__(self, uri: str, collection_name: str, dimension: int) -> None:
        self.uri = uri
        self.collection_name = collection_name
        self.dimension = dimension
        self.connected = False
        self.collection = None
        self.fallback_records: List[Dict[str, object]] = []
        self.record_map: Dict[str, Dict[str, object]] = {}
        self.bm25_index: SimpleBM25Index | None = None
        try:
            self._connect()
            self.collection = self._ensure_collection()
        except Exception:
            self.connected = False

    def _connect(self) -> None:
        if self.connected:
            return
        connections.connect(alias="default", uri=self.uri)
        self.connected = True

    def _ensure_collection(self) -> Collection:
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            collection.load()
            return collection
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="logical_page", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            ],
            description="Prospectus RAG chunks",
        )
        collection = Collection(name=self.collection_name, schema=schema)
        collection.create_index(
            field_name="embedding",
            index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
        )
        collection.load()
        return collection

    def recreate_for_dimension(self, dimension: int) -> None:
        if not self.connected:
            self.dimension = dimension
            self._clear_local_indexes()
            return
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()
        self.dimension = dimension
        self.collection = self._ensure_collection()

    def clear(self) -> None:
        self._clear_local_indexes()
        if not self.connected:
            return
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()
        self.collection = self._ensure_collection()

    def _clear_local_indexes(self) -> None:
        self.fallback_records = []
        self.record_map = {}
        self.bm25_index = None

    def load_runtime_records(self, chunks: Iterable[Chunk], embeddings: List[List[float]]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            self._clear_local_indexes()
            return 0

        self.dimension = len(embeddings[0])
        self.fallback_records = [
            {
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "logical_page": chunk.logical_page,
                "text": chunk.text,
                "embedding": np.array(embedding, dtype=np.float32),
                "metadata": dict(chunk.metadata),
            }
            for chunk, embedding in zip(chunk_list, embeddings)
        ]
        self.record_map = {str(item["chunk_id"]): item for item in self.fallback_records}
        corpus_tokens = [self._tokenize_for_bm25(str(item["text"])) for item in self.fallback_records]
        self.bm25_index = SimpleBM25Index(corpus_tokens, settings.bm25_k1, settings.bm25_b)
        return len(self.fallback_records)

    def upsert_chunks(self, chunks: Iterable[Chunk], embeddings: List[List[float]]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        self.load_runtime_records(chunk_list, embeddings)

        if not self.connected:
            return len(self.fallback_records)

        if self.collection is not None and self.collection.schema.fields[-1].params["dim"] != len(embeddings[0]):
            self.recreate_for_dimension(len(embeddings[0]))
        data = [
            [chunk.chunk_id for chunk in chunk_list],
            [chunk.page_number for chunk in chunk_list],
            [chunk.logical_page or "" for chunk in chunk_list],
            [chunk.text[:8190] for chunk in chunk_list],
            embeddings,
        ]
        self.collection.insert(data)
        self.collection.flush()
        self.collection.load()
        return len(chunk_list)

    def search(self, query_embedding: List[float], top_k: int, query_text: str = "") -> List[Dict[str, object]]:
        candidate_limit = max(top_k * 4, settings.retrieval_candidate_limit)

        if not self.connected:
            return self._hybrid_fallback_search(query_embedding, top_k, query_text, candidate_limit)

        dense_candidates = self._dense_search_milvus(query_embedding, candidate_limit)
        bm25_candidates = self._bm25_search(query_text, candidate_limit)
        return self._merge_candidates(dense_candidates, bm25_candidates, query_text, top_k)

    def _dense_search_milvus(self, query_embedding: List[float], candidate_limit: int) -> List[Dict[str, object]]:
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=candidate_limit,
            output_fields=["page_number", "logical_page", "text"],
        )
        dense_candidates: List[Dict[str, object]] = []
        for hit in results[0]:
            chunk_id = str(hit.id)
            local_record = self.record_map.get(chunk_id, {})
            dense_candidates.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": int(hit.entity.get("page_number")),
                    "logical_page": hit.entity.get("logical_page") or None,
                    "text": local_record.get("text", hit.entity.get("text")),
                    "dense_score": float(hit.score),
                    "metadata": dict(local_record.get("metadata") or {}),
                }
            )
        return dense_candidates

    def _hybrid_fallback_search(
        self,
        query_embedding: List[float],
        top_k: int,
        query_text: str,
        candidate_limit: int,
    ) -> List[Dict[str, object]]:
        if not self.fallback_records:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query) or 1.0
        dense_candidates: List[Dict[str, object]] = []
        for item in self.fallback_records:
            vector = item["embedding"]
            vector_norm = np.linalg.norm(vector) or 1.0
            dense_score = float(np.dot(query, vector) / (query_norm * vector_norm))
            dense_candidates.append(
                {
                    "chunk_id": item["chunk_id"],
                    "page_number": item["page_number"],
                    "logical_page": item["logical_page"],
                    "text": item["text"],
                    "dense_score": dense_score,
                    "metadata": dict(item.get("metadata") or {}),
                }
            )
        dense_candidates.sort(key=lambda item: item["dense_score"], reverse=True)
        bm25_candidates = self._bm25_search(query_text, candidate_limit)
        return self._merge_candidates(dense_candidates[:candidate_limit], bm25_candidates, query_text, top_k)

    def _bm25_search(self, query_text: str, candidate_limit: int) -> List[Dict[str, object]]:
        if not query_text or self.bm25_index is None or not self.fallback_records:
            return []

        query_tokens = self._tokenize_for_bm25(query_text)
        if not query_tokens:
            return []

        scores = self.bm25_index.score(query_tokens)
        candidates: List[Dict[str, object]] = []
        for index, score in enumerate(scores):
            if score <= 0:
                continue
            item = self.fallback_records[index]
            candidates.append(
                {
                    "chunk_id": item["chunk_id"],
                    "page_number": item["page_number"],
                    "logical_page": item["logical_page"],
                    "text": item["text"],
                    "bm25_score": float(score),
                    "metadata": dict(item.get("metadata") or {}),
                }
            )
        candidates.sort(key=lambda item: item["bm25_score"], reverse=True)
        return candidates[:candidate_limit]

    def _merge_candidates(
        self,
        dense_candidates: List[Dict[str, object]],
        bm25_candidates: List[Dict[str, object]],
        query_text: str,
        top_k: int,
    ) -> List[Dict[str, object]]:
        dense_score_map = {str(item["chunk_id"]): float(item.get("dense_score", 0.0)) for item in dense_candidates}
        bm25_score_map = {str(item["chunk_id"]): float(item.get("bm25_score", 0.0)) for item in bm25_candidates}
        dense_norm = self._normalize_score_map(dense_score_map)
        bm25_norm = self._normalize_score_map(bm25_score_map)

        merged_candidates: Dict[str, Dict[str, object]] = {}
        for item in dense_candidates + bm25_candidates:
            chunk_id = str(item["chunk_id"])
            if chunk_id not in merged_candidates:
                record = self.record_map.get(chunk_id, {})
                merged_candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "page_number": int(item.get("page_number") or record.get("page_number") or 0),
                    "logical_page": item.get("logical_page") or record.get("logical_page"),
                    "text": str(item.get("text") or record.get("text") or ""),
                    "metadata": dict(item.get("metadata") or record.get("metadata") or {}),
                }

        results: List[Dict[str, object]] = []
        for chunk_id, item in merged_candidates.items():
            text = str(item["text"])
            metadata = dict(item.get("metadata") or {})
            overlap_score = keyword_overlap_score(query_text, text)
            bonus = self._metadata_bonus(metadata, query_text)
            raw_score = (
                dense_norm.get(chunk_id, 0.0) * settings.hybrid_dense_weight
                + bm25_norm.get(chunk_id, 0.0) * settings.hybrid_lexical_weight
                + overlap_score * settings.hybrid_overlap_weight
                + bonus
            )
            display_score = max(0.0, min(1.0, raw_score))
            results.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": item["page_number"],
                    "logical_page": item.get("logical_page"),
                    "text": text,
                    "score": display_score,
                    "raw_score": raw_score,
                    "dense_score": dense_score_map.get(chunk_id, 0.0),
                    "bm25_score": bm25_score_map.get(chunk_id, 0.0),
                    "metadata": metadata,
                }
            )
        results.sort(key=lambda item: item["raw_score"], reverse=True)
        return results[:top_k]

    def _normalize_score_map(self, score_map: Dict[str, float]) -> Dict[str, float]:
        if not score_map:
            return {}
        values = list(score_map.values())
        min_value = min(values)
        max_value = max(values)
        if math.isclose(min_value, max_value):
            if max_value <= 0:
                return {key: 0.0 for key in score_map}
            return {key: 1.0 for key in score_map}
        return {
            key: (value - min_value) / (max_value - min_value)
            for key, value in score_map.items()
        }

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        normalized = normalize_whitespace(text).lower()
        if not normalized:
            return []

        tokens: List[str] = []
        segments = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", normalized)
        for segment in segments:
            if re.fullmatch(r"[a-z0-9]+", segment):
                tokens.append(segment)
                continue

            if len(segment) == 1:
                tokens.append(segment)
                continue

            for n in (2, 3):
                if len(segment) >= n:
                    tokens.extend(segment[index : index + n] for index in range(len(segment) - n + 1))
            if len(segment) <= 8:
                tokens.append(segment)
        return tokens

    def _metadata_bonus(self, metadata: Dict[str, object], query_text: str) -> float:
        page_type = str(metadata.get("page_type") or "")
        field_title = str(metadata.get("field_title") or "")
        section_title = str(metadata.get("section_title") or "")
        query = query_text or ""
        bonus = 0.0
        if page_type == "structured":
            bonus += 0.04
        if page_type == "vlm_structured":
            bonus += 0.06
        if page_type == "table_analysis":
            bonus += 0.03
        if page_type == "table_markdown":
            bonus += 0.06
        if field_title and field_title in query:
            bonus += 0.08
        if section_title and any(token in section_title for token in ["募集资金", "发行人基本情况", "主要客户", "供应商"]):
            bonus += 0.04
        if "法定代表人" in query and field_title == "法定代表人":
            bonus += 0.08
        if "注册资本" in query and field_title == "注册资本":
            bonus += 0.08
        if "补充流动资金" in query and field_title == "补充流动资金":
            bonus += 0.08
        if "技术标准" in query and field_title == "技术标准":
            bonus += 0.06
        if "供应商" in query and field_title in {"重要供应商", "供应商领域"}:
            bonus += 0.06
        if "上游" in query and field_title in {"上游企业", "上游行业"}:
            bonus += 0.05
        if "下游" in query and field_title in {"下游行业", "下游应用行业"}:
            bonus += 0.05
        return bonus

    def count(self) -> int:
        if not self.connected:
            return len(self.fallback_records)
        return self.collection.num_entities
