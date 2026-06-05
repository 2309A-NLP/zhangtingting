# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.text_utils import Chunk


class MilvusVectorStore:
    def __init__(self, uri: str, collection_name: str, dimension: int) -> None:
        self.uri = uri
        self.collection_name = collection_name
        self.dimension = dimension
        self.connected = False
        self.collection = None
        self.fallback_records: List[Dict[str, object]] = []
        self.fallback_vectorizer: TfidfVectorizer | None = None
        self.fallback_matrix = None
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
            return Collection(self.collection_name)
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
            self.fallback_records = []
            self.fallback_vectorizer = None
            self.fallback_matrix = None
            return
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()
        self.dimension = dimension
        self.collection = self._ensure_collection()

    def clear(self) -> None:
        if not self.connected:
            self.fallback_records = []
            self.fallback_vectorizer = None
            self.fallback_matrix = None
            return
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()
        self.collection = self._ensure_collection()

    def upsert_chunks(self, chunks: Iterable[Chunk], embeddings: List[List[float]]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0
        if not self.connected:
            self.dimension = len(embeddings[0])
            self.fallback_records = [
                {
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "logical_page": chunk.logical_page,
                    "text": chunk.text,
                    "embedding": np.array(embedding, dtype=np.float32),
                }
                for chunk, embedding in zip(chunk_list, embeddings)
            ]
            corpus = [str(item["text"]) for item in self.fallback_records]
            self.fallback_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
            self.fallback_matrix = self.fallback_vectorizer.fit_transform(corpus)
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
        if not self.connected:
            if not self.fallback_records or self.fallback_vectorizer is None or self.fallback_matrix is None:
                return []
            query = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query) or 1.0
            tfidf_query = self.fallback_vectorizer.transform([query_text])
            tfidf_scores = cosine_similarity(tfidf_query, self.fallback_matrix).flatten()
            keywords = [token for token in query_text if token.strip()] if query_text else []
            matches: List[Dict[str, object]] = []
            for index, item in enumerate(self.fallback_records):
                vector = item["embedding"]
                vector_norm = np.linalg.norm(vector) or 1.0
                semantic_score = float(np.dot(query, vector) / (query_norm * vector_norm))
                text = str(item["text"])
                keyword_hits = sum(1 for token in keywords if token in text)
                lexical_score = keyword_hits / max(1, len(set(keywords)))
                exact_bonus = 0.5 if query_text and query_text.replace("？", "").replace("?", "") in text else 0.0
                tfidf_score = float(tfidf_scores[index])
                score = semantic_score * 0.15 + tfidf_score * 0.55 + lexical_score * 0.30 + exact_bonus
                matches.append(
                    {
                        "chunk_id": item["chunk_id"],
                        "page_number": item["page_number"],
                        "logical_page": item["logical_page"],
                        "text": item["text"],
                        "score": score,
                    }
                )
            matches.sort(key=lambda item: item["score"], reverse=True)
            return matches[:top_k]
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=max(top_k * 3, top_k),
            output_fields=["page_number", "logical_page", "text"],
        )
        matches: List[Dict[str, object]] = []
        keywords = [token for token in query_text if token.strip()] if query_text else []
        for hit in results[0]:
            text = hit.entity.get("text")
            keyword_hits = sum(1 for token in keywords if token in text)
            lexical_score = keyword_hits / max(1, len(set(keywords)))
            reranked_score = float(hit.score) * 0.7 + lexical_score * 0.3
            matches.append(
                {
                    "chunk_id": hit.id,
                    "page_number": int(hit.entity.get("page_number")),
                    "logical_page": hit.entity.get("logical_page") or None,
                    "text": text,
                    "score": reranked_score,
                }
            )
        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:top_k]

    def count(self) -> int:
        if not self.connected:
            return len(self.fallback_records)
        return self.collection.num_entities
