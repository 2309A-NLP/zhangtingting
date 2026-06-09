# å·¥åç¼å·ï¼äººå·¥æºè½ NLP-RAG-å¾ååå®¹è§£æåæ£ç´¢ä¼å
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Dict, Iterable, List

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from backend.config import settings
from backend.services.text_utils import Chunk, normalize_whitespace
from backend.utils.retrieval import SimpleBM25Index, keyword_overlap_score

FIELD_TEXT_HINTS: Dict[str, List[str]] = {
    "åç¨é¢åæ¶å¥": ["åç¨é¢åæ¶å¥", "åç¨æ¶å¥", "ååæ¶å¥", "å½é²å®¢æ·éå®é¢", "åæ¹å¸åºæ¶å¥", "åéç¨æ·", "ååéå®é¢"],
    "åç¨æ¶å¥å æ¯": ["å ä¸»è¥ä¸å¡æ¶å¥çæ¯é", "å ä¸»è¥ä¸å¡æ¶å¥æ¯é", "æ¶å¥å æ¯", "å½é²å®¢æ·éå®é¢", "ååéå®é¢", "æ¯éåå«ä¸º"],
    "ææ¯æ å": ["ææ¯æ å", "è§é¢ææ¥ç³»ç»ææ¯æ å", "åä¸å¶å®", "è§é¢ææ¯è§è"],
    "ç»ç»ç»æå¾": ["ç»ç»ç»æå¾", "ç»ç»æ¶æå¾", "ç»ç»æºæå¾"],
    "éå®é¨ä¸å±é¨é¨": ["éå®é¨ä¸å±é¨é¨", "éå®é¨", "é¨é¨ææ", "ä¸è®¾é¨é¨"],
    "å¤§å®¢æ·éå®é¨ä¸å±éå®å¤": ["å¤§å®¢æ·éå®é¨ä¸å±éå®å¤", "å¤§å®¢æ·éå®é¨", "éå®å¤ææ", "ä¸å±éå®å¤"],
    "å¢é¿çæå¿«è¡ä¸": ["å¢é¿çæå¿«è¡ä¸", "å¢é¿æå¿«è¡ä¸", "å¢éæå¿«è¡ä¸", "å¢é¿çæé«è¡ä¸"],
    "è´å¢é¿è¡ä¸": ["è´å¢é¿è¡ä¸", "å¢é¿çä¸ºè´è¡ä¸", "è´å¢é¿"],
    "å¸åºåºç¨ç»æå¢é¿å¾": ["å¸åºåºç¨ç»æä¸å¢é¿å¾", "åºç¨ç»æä¸å¢é¿å¾", "å¢é¿å¾"],
}



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

    def _load_runtime_chunk_records(
        self,
        chunk_list: List[Chunk],
        embeddings: List[List[float]] | None = None,
    ) -> int:
        self.fallback_records = []
        for index, chunk in enumerate(chunk_list):
            chunk_text = chunk.search_text or chunk.normalized_text or chunk.text
            record = {
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "logical_page": chunk.logical_page,
                "text": chunk_text,
                "metadata": dict(chunk.metadata),
            }
            if embeddings is not None:
                record["embedding"] = np.array(embeddings[index], dtype=np.float32)
            self.fallback_records.append(record)
        self.record_map = {str(item["chunk_id"]): item for item in self.fallback_records}
        corpus_tokens = [self._tokenize_for_bm25(str(item["text"])) for item in self.fallback_records]
        self.bm25_index = SimpleBM25Index(corpus_tokens, settings.bm25_k1, settings.bm25_b)
        return len(self.fallback_records)

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
                FieldSchema(name="source_pdf", dtype=DataType.VARCHAR, max_length=256),
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
        return self._load_runtime_chunk_records(chunk_list, embeddings)

    def load_runtime_chunks(self, chunks: Iterable[Chunk]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            self._clear_local_indexes()
            return 0
        return self._load_runtime_chunk_records(chunk_list)

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
            [(chunk.search_text or chunk.normalized_text or chunk.text)[:8190] for chunk in chunk_list],
            [str(chunk.metadata.get("source_pdf") or "")[:250] for chunk in chunk_list],
            embeddings,
        ]
        self.collection.insert(data)
        self.collection.flush()
        self.collection.load()
        return len(chunk_list)

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        query_text: str = "",
        question_type: str = "",
        preferred_block_types: List[str] | None = None,
    ) -> List[Dict[str, object]]:
        candidate_limit = max(top_k * 4, settings.retrieval_candidate_limit)
        preferred_block_types = preferred_block_types or []

        if not self.connected:
            return self._hybrid_fallback_search(
                query_embedding,
                top_k,
                query_text,
                candidate_limit,
                question_type=question_type,
                preferred_block_types=preferred_block_types,
            )

        dense_candidates = self._dense_search_milvus(query_embedding, candidate_limit)
        bm25_candidates = self._bm25_search(query_text, candidate_limit)
        return self._merge_candidates(
            dense_candidates,
            bm25_candidates,
            query_text,
            top_k,
            question_type=question_type,
            preferred_block_types=preferred_block_types,
        )

    def _dense_search_milvus(self, query_embedding: List[float], candidate_limit: int) -> List[Dict[str, object]]:
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=candidate_limit,
            output_fields=["page_number", "logical_page", "text", "source_pdf"],
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
                    "metadata": {**dict(local_record.get("metadata") or {}), "source_pdf": str(hit.entity.get("source_pdf") or (local_record.get("metadata") or {}).get("source_pdf") or "")},
                }
            )
        return dense_candidates

    def _hybrid_fallback_search(
        self,
        query_embedding: List[float],
        top_k: int,
        query_text: str,
        candidate_limit: int,
        question_type: str = "",
        preferred_block_types: List[str] | None = None,
    ) -> List[Dict[str, object]]:
        if not self.fallback_records:
            return []
        preferred_block_types = preferred_block_types or []

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
        return self._merge_candidates(
            dense_candidates[:candidate_limit],
            bm25_candidates,
            query_text,
            top_k,
            question_type=question_type,
            preferred_block_types=preferred_block_types,
        )

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

    def search_by_metadata(
        self,
        query_text: str,
        top_k: int,
        source_pdfs: List[str] | None = None,
        field_keys: List[str] | None = None,
        preferred_sections: List[str] | None = None,
        query_tags: List[str] | None = None,
        preferred_block_types: List[str] | None = None,
        question_type: str = "",
    ) -> List[Dict[str, object]]:
        if not query_text or not self.fallback_records:
            return []

        source_pdfs = source_pdfs or []
        field_keys = field_keys or []
        preferred_sections = preferred_sections or []
        query_tags = query_tags or []
        preferred_block_types = preferred_block_types or []
        prefer_tables = "table" in query_tags
        prefer_lists = "list" in query_tags
        prefer_org_chart = "org_chart" in query_tags
        prefer_chart = "chart_analysis" in query_tags
        query_tokens = set(self._tokenize_for_bm25(query_text))
        if not query_tokens:
            return []
        query_payload = self._build_query_payload(query_text, query_tags, field_keys)

        results: List[Dict[str, object]] = []
        for item in self.fallback_records:
            metadata = dict(item.get("metadata") or {})
            source_pdf = str(metadata.get("source_pdf") or "")
            if source_pdfs and source_pdf and source_pdf not in source_pdfs:
                continue

            text = str(item.get("text") or "")
            tokens = set(self._tokenize_for_bm25(text))
            if not tokens:
                continue

            overlap = len(query_tokens & tokens) / max(1, len(query_tokens))
            if overlap <= 0:
                continue

            page_type = str(metadata.get("page_type") or "")
            primary_type = str(metadata.get("primary_type") or "")
            sub_type = str(metadata.get("sub_type") or "")
            section_title = str(metadata.get("section_title") or "")
            field_title = str(metadata.get("field_title") or "")
            content_tags = str(metadata.get("content_tags") or "")
            structured_facts = self._load_structured_facts(metadata)
            raw_score = overlap

            if preferred_block_types and primary_type in preferred_block_types:
                preferred_index = preferred_block_types.index(primary_type)
                raw_score += max(0.05, 0.18 - preferred_index * 0.04)
            elif preferred_block_types:
                raw_score -= 0.06

            hint_hits = 0
            for field_key in field_keys:
                hints = FIELD_TEXT_HINTS.get(field_key, [field_key])
                if any(hint and (hint in text or hint in field_title or hint in section_title) for hint in hints):
                    hint_hits += 1
            if hint_hits:
                raw_score += min(0.24, hint_hits * 0.12)

            if prefer_tables and page_type in {"table_markdown", "structured", "vlm_structured", "table_analysis"}:
                raw_score += 0.22
            if prefer_tables and primary_type == "table":
                raw_score += 0.12
            if prefer_org_chart and page_type in {"org_chart_summary", "vlm_structured"}:
                raw_score += 0.28
            if prefer_org_chart and sub_type == "org_chart":
                raw_score += 0.12
            if prefer_chart and page_type in {"chart_summary", "vlm_structured", "table_analysis"}:
                raw_score += 0.26
            if prefer_chart and sub_type == "chart_summary":
                raw_score += 0.12
            if prefer_lists and any(marker in text for marker in ["|", "ã", "åå«", "åæ¬", "å¦ä¸"]):
                raw_score += 0.10
            if prefer_org_chart and any(marker in text for marker in ["ç»ç»ç»æå¾", "ç»ç»æ¶æå¾", "ä¸å±é¨é¨", "éå®å¤", "å±", "ææ"]):
                raw_score += 0.18
            if prefer_chart and any(marker in text for marker in ["å¢é¿ç", "æå¿«", "è´å¢é¿", "æå¤§", "æå°", "è¡ä¸"]):
                raw_score += 0.18
            if query_tags and any(tag and tag in content_tags for tag in query_tags):
                raw_score += 0.08
            if preferred_sections and any(section and section in section_title for section in preferred_sections):
                raw_score += 0.18
            if field_keys and any(field_key and field_key in text for field_key in field_keys):
                raw_score += 0.20
            raw_score += self._field_key_alignment_bonus(
                field_keys=field_keys,
                field_title=field_title,
                text=text,
                question_type=question_type,
            )
            if "related_party" in query_tags and any(token in text for token in ["å³èæ¹", "å³èå³ç³»", "å³èäº¤æ"]):
                raw_score += 0.20
            if "non_control_related_party" in query_tags and any(
                token in text for token in ["ä¸å­å¨æ§å¶å³ç³»", "éæ§å¶å³ç³»", "ä¸ååä¸æ§å¶"]
            ):
                raw_score += 0.22
            if "fundraising" in query_tags and any(token in text for token in ["åéèµé", "åæé¡¹ç®", "è¡¥åæµå¨èµé"]):
                raw_score += 0.18
            if "issuance" in query_tags and any(token in text for token in ["åè¡è¡æ°", "åè¡æ°é", "åè¡åæ»è¡æ¬", "åè¡æ¯ä¾"]):
                raw_score += 0.18
            if "military_revenue" in query_tags and any(
                token in text for token in ["å½é²å®¢æ·", "åæ¹å¸åº", "åéç¨æ·", "ååéå®", "ä¸»è¥ä¸å¡æ¶å¥çæ¯é"]
            ):
                raw_score += 0.24
            if "technical_standard" in query_tags and any(
                token in text for token in ["ææ¯æ å", "åä¸å¶å®", "è§é¢ææ¯è§è", "æ ¸å¿ææ¯ä¼å¿", "ç«äºå°ä½"]
            ):
                raw_score += 0.22
            if "org_chart" in query_tags and any(
                token in text for token in ["ç»ç»ç»æå¾", "ç»ç»æ¶æå¾", "ç»ç»æºæå¾", "ä¸å±é¨é¨", "éå®é¨", "éå®å¤"]
            ):
                raw_score += 0.26
            if "chart_analysis" in query_tags and any(
                token in text for token in ["å¢é¿çæå¿«", "è´å¢é¿", "å¢é¿å¾", "åºç¨ç»æ", "è¡ä¸å¢é¿ç", "å¢é¿æå¿«è¡ä¸"]
            ):
                raw_score += 0.26
            if source_pdfs and source_pdf in source_pdfs:
                raw_score += 0.12

            raw_score += self._question_type_bonus(
                question_type=question_type,
                text=text,
                metadata=metadata,
                field_keys=field_keys,
                query_text=query_text,
                structured_facts=structured_facts,
                query_payload=query_payload,
            )
            raw_score += self._structured_facts_bonus(
                query_payload=query_payload,
                structured_facts=structured_facts,
                primary_type=primary_type,
                sub_type=sub_type,
            )
            if structured_facts and any(
                self._is_placeholder_value(str(fact.get("value") or "")) for fact in structured_facts
            ):
                raw_score -= 0.18

            results.append(
                {
                    "chunk_id": item["chunk_id"],
                    "page_number": item["page_number"],
                    "logical_page": item["logical_page"],
                    "text": text,
                    "score": max(0.0, min(1.0, raw_score)),
                    "raw_score": raw_score,
                    "metadata": metadata,
                    "specialized_score": raw_score,
                }
            )

        results.sort(key=lambda item: float(item.get("raw_score", 0.0)), reverse=True)
        return results[:top_k]

    def _question_type_bonus(
        self,
        *,
        question_type: str,
        text: str,
        metadata: Dict[str, object],
        field_keys: List[str],
        query_text: str,
        structured_facts: List[Dict[str, str]],
        query_payload: Dict[str, object],
    ) -> float:
        if not question_type:
            return 0.0

        primary_type = str(metadata.get("primary_type") or "")
        sub_type = str(metadata.get("sub_type") or "")
        field_title = str(metadata.get("field_title") or "")
        section_title = str(metadata.get("section_title") or "")
        content_tags = str(metadata.get("content_tags") or "")
        payload = f"{section_title}\n{field_title}\n{text}"
        bonus = 0.0
        facts_count = len(structured_facts)

        has_number = bool(re.search(r"\d[\d,]*(?:\.\d+)?", text))
        has_list_markers = any(marker in text for marker in ["|", "åæ¬", "å¦ä¸", "åå«", "åå"])
        has_org_markers = any(marker in payload for marker in ["ä¸è®¾", "ææ", "å±", "é¨é¨", "éå®å¤", "ç»ç»ç»æ"])
        has_chart_markers = any(marker in payload for marker in ["å¢é¿ç", "æå¿«", "è´å¢é¿", "æé«", "æä½", "è¡ä¸"])

        if question_type == "field_lookup":
            if primary_type == "form":
                bonus += 0.16
            if field_title and any(field_key in field_title or field_title in field_key for field_key in field_keys):
                bonus += 0.22
            elif field_title:
                bonus -= 0.12
            if any(token in payload for token in ["åè¡äººåºæ¬æåµ", "å¬å¸æ¦åµ", "åè¡äººç®ä»", "åºæ¬æåµ"]):
                bonus += 0.18
            if "æ³¨åèµæ¬" in query_text and any(token in payload for token in ["å­å¬å¸", "æ§è¡å­å¬å¸", "åè¡å¬å¸"]):
                bonus -= 0.28
            if "æ³å®ä»£è¡¨äºº" in query_text and any(token in payload for token in ["å­å¬å¸", "æ§è¡å­å¬å¸", "åè¡å¬å¸"]):
                bonus -= 0.28
            if has_number:
                bonus += 0.04
            if facts_count:
                bonus += min(0.12, facts_count * 0.03)
            if primary_type == "text" and not field_title:
                bonus -= 0.06

        elif question_type == "table_numeric":
            if primary_type == "table":
                bonus += 0.18
            if primary_type == "form":
                bonus += 0.10
            if has_number:
                bonus += 0.12
            if any(token in payload for token in ["éé¢", "å æ¯", "æ¯ä¾", "æ¶å¥", "è¡æ°", "æ°é"]):
                bonus += 0.12
            if any(token in payload for token in ["æ¬æ¬¡åè¡æ¦åµ", "åè¡åºæ¬æåµ", "åè¡æ¹æ¡", "åéèµéè¿ç¨", "åéèµéæèµé¡¹ç®", "åæé¡¹ç®"]):
                bonus += 0.18
            if primary_type == "text" and not has_number:
                bonus -= 0.08

        elif question_type == "table_list":
            if primary_type == "table":
                bonus += 0.18
            if primary_type == "form":
                bonus += 0.08
            if has_list_markers:
                bonus += 0.14
            if any(token in payload for token in ["é¡¹ç®", "ä¼ä¸", "å³èæ¹", "åå", "åæ¬"]):
                bonus += 0.10
            if any(token in payload for token in ["åéèµéè¿ç¨", "åéèµéæèµé¡¹ç®", "åæé¡¹ç®", "å³èæ¹", "å³èå³ç³»"]):
                bonus += 0.18
            if "ä¸å­å¨æ§å¶å³ç³»" in query_text and any(token in payload for token in ["æ¾ä¸ºå³èæ¹", "ç®åå·²ä¸å­å¨å³èå³ç³»"]):
                bonus -= 0.18
            if primary_type == "text" and not has_list_markers:
                bonus -= 0.06

        elif question_type == "org_structure":
            if sub_type == "org_chart":
                bonus += 0.26
            if primary_type == "figure":
                bonus += 0.14
            if has_org_markers:
                bonus += 0.18
            if has_number:
                bonus += 0.06
            if "organization_structure" in content_tags:
                bonus += 0.10
            if primary_type == "text" and not has_org_markers:
                bonus -= 0.10

        elif question_type == "chart_trend":
            if sub_type in {"chart", "chart_summary"}:
                bonus += 0.24
            if primary_type == "figure":
                bonus += 0.12
            if has_chart_markers:
                bonus += 0.18
            if any(token in payload for token in ["åºç¨ç»æ", "å¢é¿å¾", "è´å¢é¿", "æå¿«è¡ä¸"]):
                bonus += 0.10
            if "chart_analysis" in content_tags:
                bonus += 0.10
            if primary_type == "text" and not has_chart_markers:
                bonus -= 0.10

        elif question_type == "fact_text":
            if primary_type == "text":
                bonus += 0.08
            if field_title:
                bonus += 0.03
            if "ä¸æ¸¸" in query_text and any(token in payload for token in ["ä¸æ¸¸", "çµå­åå¨ä»¶å¶é ä¼ä¸", "éå±å£³ä½å¶é ä¼ä¸"]):
                bonus += 0.18
            if "ä¸æ¸¸" in query_text and any(token in payload for token in ["ä¸æ¸¸", "åé", "æ¿åºæºå³", "è½æº"]):
                bonus += 0.18
            if "ææ¯æ å" in query_text and any(token in payload for token in ["åä¸å¶å®", "è§é¢ææ¯è§è", "ææ¯æ å"]):
                bonus += 0.18
            if any(token in query_text for token in ["éè¦ä¾åºå", "ä¾åºå"]) and any(token in payload for token in ["å½é²åéè§é¢ææ¥é¢å", "åéè§é¢ææ¥é¢å", "éè¦ä¾åºå"]):
                bonus += 0.16

        if facts_count:
            if question_type == "table_numeric":
                bonus += min(0.14, facts_count * 0.035)
            elif question_type == "table_list":
                bonus += min(0.12, facts_count * 0.03)
            elif question_type in {"org_structure", "chart_trend"}:
                bonus += min(0.16, facts_count * 0.04)
            elif question_type == "fact_text":
                bonus += min(0.08, facts_count * 0.02)

        bonus += self._answer_pattern_bonus(
            question_type=question_type,
            text=text,
            query_payload=query_payload,
            primary_type=primary_type,
            sub_type=sub_type,
        )

        return bonus

    def _merge_candidates(
        self,
        dense_candidates: List[Dict[str, object]],
        bm25_candidates: List[Dict[str, object]],
        query_text: str,
        top_k: int,
        question_type: str = "",
        preferred_block_types: List[str] | None = None,
    ) -> List[Dict[str, object]]:
        preferred_block_types = preferred_block_types or []
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
            primary_type = str(metadata.get("primary_type") or "")
            sub_type = str(metadata.get("sub_type") or "")
            structured_facts = self._load_structured_facts(metadata)
            query_payload = self._build_query_payload(query_text, [], [])
            if preferred_block_types and primary_type in preferred_block_types:
                bonus += max(0.04, 0.12 - preferred_block_types.index(primary_type) * 0.03)
            elif preferred_block_types:
                bonus -= 0.04
            bonus += self._structured_facts_bonus(
                query_payload=query_payload,
                structured_facts=structured_facts,
                primary_type=primary_type,
                sub_type=sub_type,
            )
            bonus += self._answer_pattern_bonus(
                question_type=question_type,
                text=text,
                query_payload=query_payload,
                primary_type=primary_type,
                sub_type=sub_type,
            )
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

    def _build_query_payload(
        self,
        query_text: str,
        query_tags: List[str],
        field_keys: List[str],
    ) -> Dict[str, object]:
        normalized = normalize_whitespace(query_text)
        query_tokens = set(self._tokenize_for_bm25(normalized))
        return {
            "text": normalized,
            "tokens": query_tokens,
            "asks_count": any(token in normalized for token in ["å ä¸ª", "å¤å°", "å å®¶", "å é¡¹", "æ°é", "ä¸ªéå®å¤", "ä¸ªé¨é¨"]),
            "asks_fastest": any(token in normalized for token in ["æå¿«", "æé«å¢é", "å¢é¿çæé«", "å¢éæå¿«"]),
            "asks_negative": any(token in normalized for token in ["è´å¢é¿", "ä¸ºè´", "ä¸é"]),
            "asks_ratio": any(token in normalized for token in ["æ¯ä¾", "å æ¯", "æ¯é"]),
            "asks_amount": any(token in normalized for token in ["éé¢", "æ¶å¥", "åéèµé", "è¡æ°", "æ°é"]),
            "tags": set(query_tags or []),
            "field_keys": set(field_keys or []),
        }

    def _load_structured_facts(self, metadata: Dict[str, object]) -> List[Dict[str, str]]:
        payload = metadata.get("structured_facts") or ""
        if not payload:
            return []
        try:
            value = json.loads(str(payload))
        except Exception:
            return []
        if not isinstance(value, list):
            return []
        facts: List[Dict[str, str]] = []
        for item in value:
            if isinstance(item, dict):
                facts.append({str(key): str(val) for key, val in item.items() if val is not None})
        return facts

    def _is_placeholder_value(self, value: str) -> bool:
        normalized = normalize_whitespace(value)
        if not normalized:
            return True
        placeholders = [
            "æ ",
            "ææ ",
            "æªæ«é²",
            "æªæä¾",
            "æªæ£ç´¢å°",
            "æ²¡æ",
            "æªç¥",
            "ä¸éç¨",
            "æ å·ä½",
            "æ ç¸å³",
        ]
        return any(normalized == item or normalized.startswith(item) for item in placeholders)

    def _field_key_alignment_bonus(
        self,
        *,
        field_keys: List[str],
        field_title: str,
        text: str,
        question_type: str,
    ) -> float:
        if not field_keys:
            return 0.0

        payload = f"{field_title}\n{text}"
        exact_hit = any(field_key and field_key == field_title for field_key in field_keys)
        partial_hit = any(
            field_key
            and (
                field_key in payload
                or field_title in field_key
                or any(token and token in payload for token in FIELD_TEXT_HINTS.get(field_key, []))
            )
            for field_key in field_keys
        )

        if exact_hit:
            return 0.28
        if partial_hit:
            return 0.14
        if question_type == "field_lookup" and field_title:
            return -0.18
        return 0.0

    def _structured_facts_bonus(
        self,
        *,
        query_payload: Dict[str, object],
        structured_facts: List[Dict[str, str]],
        primary_type: str,
        sub_type: str,
    ) -> float:
        if not structured_facts:
            return 0.0

        query_tokens = set(query_payload.get("tokens") or set())
        asks_count = bool(query_payload.get("asks_count"))
        asks_fastest = bool(query_payload.get("asks_fastest"))
        asks_negative = bool(query_payload.get("asks_negative"))
        asks_ratio = bool(query_payload.get("asks_ratio"))
        asks_amount = bool(query_payload.get("asks_amount"))

        best = 0.0
        for fact in structured_facts:
            title = str(fact.get("title") or fact.get("field") or "")
            value = str(fact.get("value") or "")
            evidence = str(fact.get("evidence") or "")
            payload = normalize_whitespace(f"{title} {value} {evidence}")
            fact_tokens = set(self._tokenize_for_bm25(payload))
            overlap = len(query_tokens & fact_tokens) / max(1, len(query_tokens)) if query_tokens else 0.0
            score = overlap * 0.24
            if asks_count and any(token in payload for token in ["ä¸ª", "å®¶", "é¡¹", "å¤", "é¨é¨", "æ°é"]):
                score += 0.10
            if asks_fastest and any(token in payload for token in ["æå¿«", "æé«", "æå¤§", "ç¬¬ä¸"]):
                score += 0.12
            if asks_negative and any(token in payload for token in ["è´å¢é¿", "ä¸é", "ä¸ºè´"]):
                score += 0.12
            if asks_ratio and any(token in payload for token in ["å æ¯", "æ¯ä¾", "æ¯é"]):
                score += 0.10
            if asks_amount and bool(re.search(r"\d[\d,]*(?:\.\d+)?", payload)):
                score += 0.08
            if title and any(
                field_key and (field_key == title or field_key in title or title in field_key)
                for field_key in query_payload.get("field_keys") or set()
            ):
                score += 0.12
            if self._is_placeholder_value(value):
                score -= 0.20
            best = max(best, score)

        if primary_type == "table":
            best += 0.02
        if sub_type in {"org_chart", "chart_summary"}:
            best += 0.03
        return min(0.34, best)

    def _answer_pattern_bonus(
        self,
        *,
        question_type: str,
        text: str,
        query_payload: Dict[str, object],
        primary_type: str,
        sub_type: str,
    ) -> float:
        payload = normalize_whitespace(text)
        bonus = 0.0

        if question_type == "org_structure":
            if query_payload.get("asks_count") and re.search(r"(ç±|å±|è®¾æ).{0,10}\d+.{0,6}(ä¸ª|å®¶|é¡¹|å¤|é¨é¨)", payload):
                bonus += 0.18
            if primary_type == "figure" or sub_type == "org_chart":
                bonus += 0.06

        if question_type == "chart_trend":
            if query_payload.get("asks_fastest") and any(token in payload for token in ["æå¿«", "æé«", "æå¤§"]):
                bonus += 0.18
            if query_payload.get("asks_negative") and any(token in payload for token in ["è´å¢é¿", "ä¸é", "ä¸ºè´"]):
                bonus += 0.18
            if primary_type == "figure" or sub_type in {"chart", "chart_summary"}:
                bonus += 0.06

        if question_type == "table_numeric":
            if query_payload.get("asks_amount") and bool(re.search(r"\d[\d,]*(?:\.\d+)?", payload)):
                bonus += 0.08
            if query_payload.get("asks_ratio") and any(token in payload for token in ["æ¯ä¾", "å æ¯", "æ¯é", "%"]):
                bonus += 0.08

        if question_type == "table_list":
            if any(token in payload for token in ["åæ¬", "å¦ä¸", "åå«", "åå"]):
                bonus += 0.08

        return min(0.26, bonus)

    def _metadata_bonus(self, metadata: Dict[str, object], query_text: str) -> float:
        page_type = str(metadata.get("page_type") or "")
        primary_type = str(metadata.get("primary_type") or "")
        sub_type = str(metadata.get("sub_type") or "")
        field_title = str(metadata.get("field_title") or "")
        section_title = str(metadata.get("section_title") or "")
        content_tags = str(metadata.get("content_tags") or "")
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
        if page_type == "org_chart_summary":
            bonus += 0.12
        if page_type == "chart_summary":
            bonus += 0.12
        if primary_type == "table":
            bonus += 0.02
        if sub_type == "org_chart":
            bonus += 0.04
        if sub_type == "chart_summary":
            bonus += 0.04
        if field_title and field_title in query:
            bonus += 0.08
        if section_title and any(token in section_title for token in ["åéèµé", "åè¡äººåºæ¬æåµ", "ä¸»è¦å®¢æ·", "ä¾åºå"]):
            bonus += 0.04
        if any(token in content_tags for token in ["fundraising", "organization_structure", "chart_analysis"]):
            bonus += 0.02
        if "æ³å®ä»£è¡¨äºº" in query and field_title == "æ³å®ä»£è¡¨äºº":
            bonus += 0.08
        if "æ³¨åèµæ¬" in query and field_title == "æ³¨åèµæ¬":
            bonus += 0.08
        if "è¡¥åæµå¨èµé" in query and field_title == "è¡¥åæµå¨èµé":
            bonus += 0.08
        if "ææ¯æ å" in query and field_title == "ææ¯æ å":
            bonus += 0.06
        if any(token in query for token in ["åç¨é¢åæ¶å¥", "æ¥èªåç¨é¢åçæ¶å¥", "å½é²å®¢æ·éå®é¢", "åæ¹å¸åºæ¶å¥"]) and any(
            token in field_title for token in ["åç¨é¢åæ¶å¥", "åç¨æ¶å¥", "å½é²å®¢æ·éå®é¢", "ååæ¶å¥"]
        ):
            bonus += 0.10
        if any(token in query for token in ["ä¸»è¥ä¸å¡æ¶å¥çæ¯é", "æ¶å¥å æ¯", "æ¯éåå«ä¸º"]) and any(
            token in field_title for token in ["åç¨æ¶å¥å æ¯", "å½é²å®¢æ·éå®é¢å æ¯", "ä¸»è¥ä¸å¡æ¶å¥æ¯é"]
        ):
            bonus += 0.10
        if "ä¾åºå" in query and field_title in {"éè¦ä¾åºå", "ä¾åºåé¢å"}:
            bonus += 0.06
        if "ä¸æ¸¸" in query and field_title in {"ä¸æ¸¸ä¼ä¸", "ä¸æ¸¸è¡ä¸"}:
            bonus += 0.05
        if "ä¸æ¸¸" in query and field_title in {"ä¸æ¸¸è¡ä¸", "ä¸æ¸¸åºç¨è¡ä¸"}:
            bonus += 0.05
        if any(token in query for token in ["ç»ç»ç»æå¾", "ç»ç»æ¶æå¾", "éå®é¨", "å¤§å®¢æ·éå®é¨", "éå®å¤"]) and any(
            token in field_title for token in ["éå®é¨ä¸å±é¨é¨", "å¤§å®¢æ·éå®é¨ä¸å±éå®å¤", "ç»ç»ç»æå¾"]
        ):
            bonus += 0.12
        if any(token in query for token in ["å¢é¿çæå¿«", "å¢é¿æå¿«", "è´å¢é¿", "åºç¨ç»æä¸å¢é¿å¾", "å¢é¿å¾"]) and any(
            token in field_title for token in ["å¢é¿çæå¿«è¡ä¸", "è´å¢é¿è¡ä¸", "å¸åºåºç¨ç»æå¢é¿å¾", "è¡ä¸å¢é¿ç"]
        ):
            bonus += 0.12
        return bonus

    def count(self) -> int:
        if not self.connected:
            return len(self.fallback_records)
        return self.collection.num_entities
