# ������ţ��˹����� NLP-RAG-ͼ�����ݽ����������Ż�
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

import fitz

from backend.config import settings
from backend.schemas import QueryResponse, SourceChunk
from backend.services.embedding import EmbeddingService
from backend.services.legacy_pdf_pipeline import load_resolved_pages_as_parser_output, run_legacy_pdf_pipeline
from backend.services.llm_client import LLMClient
from backend.services.pdf_parser import PDFParser
from backend.services.pdf_vlm_client import PDFVLMClient
from backend.services.query_understanding import analyze_query
from backend.services.rerank_service import RerankService
from backend.services.text_utils import dedupe_preserve_order
from backend.services.vector_store import MilvusVectorStore
from backend.services.rag_pipeline._scoring import (
    rerank_for_answerability,
    apply_company_routing,
    select_answer_contexts,
)
from backend.services.rag_pipeline._chunks import (
    build_main_chunks,
    build_pdf_intelligence_chunks,
    build_enhanced_chunks,
    build_candidate_from_page,
)
from backend.services.rag_pipeline._vlm import (
    build_vlm_chunks,
    load_pdf_vlm_items,
)
from backend.services.rag_pipeline.rag_utils import (
    is_valid_enhanced_item,
    write_redacted_export,
    main_manifest_path,
    enhance_manifest_path,
    parsed_cache_path,
    redacted_cache_path,
    default_pdf_paths,
    resolve_target_pdfs,
)
from backend.services.rag_pipeline.upload_registry import (
    build_upload_id,
    build_uploaded_collection_names,
    discover_uploaded_documents_from_milvus,
    ensure_uploads_root,
    file_sha1,
    merge_uploaded_documents,
    read_upload_registry,
    upsert_upload_registry_entry,
    upload_artifact_dir,
    upload_chunk_manifest_path,
    upload_enhanced_pages_path,
    upload_manifest_path,
    upload_parsed_pages_path,
    upload_pdf_intelligence_dir,
    upload_redacted_pages_path,
    upload_root,
    upload_source_dir,
    upload_vlm_failure_path,
)
from backend.utils.logging import get_logger


CorpusName = Literal["default", "uploaded"]
logger = get_logger(__name__)


def _read_json_file(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class RAGPipeline:
    def __init__(self) -> None:
        self.parser = PDFParser(ocr_lang=settings.ocr_lang)
        self.embedder = EmbeddingService(settings.model_dir / "embedding", configured_path=settings.embedding_model_path)
        self.vector_store = MilvusVectorStore(settings.milvus_uri, settings.collection_name, self.embedder.dimension)
        self.uploaded_vector_store: MilvusVectorStore | None = None
        self.active_upload_id = ""
        self.uploaded_pdf_name = ""
        self.uploaded_pages_cache: List[Dict[str, object]] | None = None
        self.uploaded_registry: List[Dict[str, object]] = read_upload_registry()
        self.llm = LLMClient(
            provider=settings.llm_provider,
            api_url=settings.llm_api_url,
            api_key=settings.llm_api_key,
            model_name=settings.llm_model_name,
            fallback_api_url=settings.llm_fallback_api_url,
            fallback_api_key=settings.llm_fallback_api_key,
            fallback_model_name=settings.llm_fallback_model_name,
        )
        self.pdf_vlm = PDFVLMClient()
        self.reranker = RerankService(settings.reranker_model_path) if settings.reranker_enabled else None
        self.pages_cache: List[Dict[str, object]] | None = None
        self.pdf_company_map: Dict[str, List[str]] = {}
        self.upload_dir = settings.artifact_dir / "uploads"
        settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._restore_latest_uploaded_state()

    # -------------------------------------------------------------------------
    # Uploaded registry
    # -------------------------------------------------------------------------

    def _restore_latest_uploaded_state(self) -> None:
        if not self.uploaded_registry:
            return
        latest = self.uploaded_registry[0]
        upload_id = str(latest.get("upload_id") or "")
        collection_name = str(latest.get("text_collection_name") or "")
        filename = str(latest.get("filename") or "")
        if not upload_id or not collection_name:
            return
        self.active_upload_id = upload_id
        self.uploaded_pdf_name = filename
        self.uploaded_vector_store = MilvusVectorStore(settings.milvus_uri, collection_name, self.embedder.dimension)

    def _uploaded_registry_entry(self, upload_id: str) -> Dict[str, object] | None:
        for item in self.uploaded_registry:
            if str(item.get("upload_id") or "") == upload_id:
                return item
        return None

    def _set_active_upload(self, upload_id: str) -> Dict[str, object]:
        entry = self._uploaded_registry_entry(upload_id)
        if entry is None:
            raise ValueError(f"Unknown upload_id: {upload_id}")
        collection_name = str(entry.get("text_collection_name") or "")
        if not collection_name:
            raise ValueError(f"Upload missing collection name: {upload_id}")
        self.active_upload_id = upload_id
        self.uploaded_pdf_name = str(entry.get("filename") or "")
        self.uploaded_vector_store = MilvusVectorStore(settings.milvus_uri, collection_name, self.embedder.dimension)
        return entry

    def list_uploaded_documents(self) -> List[Dict[str, object]]:
        registry_items = read_upload_registry()
        discovered_items = discover_uploaded_documents_from_milvus()
        self.uploaded_registry = merge_uploaded_documents(registry_items, discovered_items)
        return self.uploaded_registry

    def _resolve_uploaded_store(self, upload_id: str | None = None) -> tuple[MilvusVectorStore, Dict[str, object] | None]:
        self.uploaded_registry = self.list_uploaded_documents()
        target_id = (upload_id or self.active_upload_id or "").strip()
        if not target_id and self.uploaded_registry:
            target_id = str(self.uploaded_registry[0].get("upload_id") or "")
        if not target_id:
            raise ValueError("No uploaded PDF available.")
        entry = self._set_active_upload(target_id)
        if self.uploaded_vector_store is None:
            raise ValueError("Uploaded vector store not initialized.")
        return self.uploaded_vector_store, entry

    def _resolve_uploaded_artifact_dir(self, upload_entry: Dict[str, object] | None) -> Path | None:
        if upload_entry is None:
            return None
        artifact_dir = str(upload_entry.get("artifact_dir") or "").strip()
        if artifact_dir:
            path = Path(artifact_dir)
            if path.exists():
                return path
        upload_id = str(upload_entry.get("upload_id") or "").strip()
        if upload_id:
            path = upload_artifact_dir(upload_id)
            if path.exists():
                return path
        return None

    def _build_uploaded_text_rows(self, items: List[Dict[str, object]], upload_entry: Dict[str, object] | None) -> List[Dict[str, object]]:
        doc_name = str((upload_entry or {}).get("filename") or "")
        rows: List[Dict[str, object]] = []
        for item in items:
            metadata = dict(item.get("metadata") or {})
            page_number = int(item.get("page_number") or 0)
            text = str(item.get("text") or "")
            rows.append(
                {
                    "page_start": page_number,
                    "page_end": page_number,
                    "page_number": page_number,
                    "logical_page": str(item.get("logical_page") or ""),
                    "chunk_id": str(item.get("chunk_id") or metadata.get("chunk_id") or ""),
                    "doc_name": str(metadata.get("doc_name") or metadata.get("source_pdf") or doc_name),
                    "text": text,
                    "preview_text": text,
                    "window_text": text,
                    "source_pages": [page_number] if page_number > 0 else [],
                    "score": float(item.get("rerank_score", item.get("raw_score", item.get("score", 0.0))) or 0.0),
                }
            )
        return rows

    def _normalize_uploaded_contexts(self, contexts: List[Dict[str, object]]) -> List[Dict[str, object]]:
        normalized: List[Dict[str, object]] = []
        for index, item in enumerate(contexts, start=1):
            payload = dict(item)
            metadata = dict(payload.get("metadata") or {})
            payload["metadata"] = metadata
            chunk_id = str(
                payload.get("chunk_id")
                or metadata.get("chunk_id")
                or metadata.get("table_id")
                or metadata.get("visual_id")
                or f"uploaded_ctx_{index}"
            )
            payload["chunk_id"] = chunk_id
            payload["score"] = float(payload.get("score", payload.get("rerank_score", 0.0)) or 0.0)
            payload["logical_page"] = payload.get("logical_page") or metadata.get("logical_page") or ""
            normalized.append(payload)
        return normalized

    def _expand_uploaded_answer_contexts(
        self,
        query: str,
        *,
        retrieval_rows: List[Dict[str, object]],
        answer_contexts: List[Dict[str, object]],
        upload_entry: Dict[str, object] | None,
        top_k: int,
    ) -> List[Dict[str, object]]:
        artifact_dir = self._resolve_uploaded_artifact_dir(upload_entry)
        if artifact_dir is None:
            return answer_contexts
        try:
            from backend.retrieval import quality_check as qc

            table_lookup = qc.build_table_lookup(artifact_dir)
            visual_lookup = qc.build_visual_lookup(artifact_dir)
            if not table_lookup and not visual_lookup:
                return answer_contexts

            text_rows = self._build_uploaded_text_rows(
                retrieval_rows or answer_contexts,
                upload_entry,
            )
            if not text_rows:
                return answer_contexts

            table_rows = qc.search_tables_keyword(query, artifact_dir, table_top_k=max(5, top_k)) if table_lookup else []
            expanded = qc.build_llm_contexts(
                query=query,
                text_rows=text_rows,
                table_rows=table_rows,
                visual_rows=[],
                table_lookup=table_lookup,
                visual_lookup=visual_lookup,
                limit=max(top_k, settings.generation_top_n),
            )
            if not expanded:
                return answer_contexts
            return self._normalize_uploaded_contexts(expanded)
        except Exception:
            logger.exception("[query-uploaded] expand_contexts_failed")
            return answer_contexts

    # -------------------------------------------------------------------------
    # Company map
    # -------------------------------------------------------------------------

    def _build_pdf_company_map(self, pages: List[Dict[str, object]]) -> Dict[str, List[str]]:
        company_pattern = re.compile(
            r"[\u4e00-\u9fffA-Za-z0-9����()]{4,}?(?:�ɷ����޹�˾|�������ι�˾|���Źɷ����޹�˾|�������޹�˾)"
        )
        grouped_pages: Dict[str, List[Dict[str, object]]] = {}
        for page in pages:
            grouped_pages.setdefault(str(page.get("source_pdf") or ""), []).append(page)

        mapping: Dict[str, List[str]] = {}
        for source_pdf, pdf_pages in grouped_pages.items():
            sample_text = "\n".join(str(page.get("text") or "")[:1200] for page in pdf_pages[:12])
            matches = company_pattern.findall(sample_text)
            matches.sort(key=len, reverse=True)
            aliases: List[str] = []
            from backend.services.rag_pipeline.rag_utils import get_company_aliases
            for company in matches[:5]:
                aliases.extend(get_company_aliases(company))
            if aliases:
                mapping[source_pdf] = dedupe_preserve_order(aliases)
        return mapping

    def _build_pdf_company_map_from_files(self) -> Dict[str, List[str]]:
        company_pattern = re.compile(
            r"[\u4e00-\u9fffA-Za-z0-9����()]{4,}?(?:�ɷ����޹�˾|�������ι�˾|���Źɷ����޹�˾|�������޹�˾)"
        )
        mapping: Dict[str, List[str]] = {}
        from backend.services.rag_pipeline.rag_utils import get_company_aliases
        for pdf_path in default_pdf_paths():
            sample_parts: List[str] = []
            try:
                with fitz.open(pdf_path) as document:
                    for index in range(min(6, len(document))):
                        sample_parts.append(document[index].get_text("text"))
            except Exception:
                continue
            sample_text = "\n".join(sample_parts)
            matches = company_pattern.findall(sample_text)
            matches.sort(key=len, reverse=True)
            aliases: List[str] = []
            for company in matches[:5]:
                aliases.extend(get_company_aliases(company))
            if aliases:
                mapping[pdf_path.name] = dedupe_preserve_order(aliases)
        return mapping

    def _ensure_pdf_company_map(self) -> Dict[str, List[str]]:
        if self.pdf_company_map:
            return self.pdf_company_map

        pages = self.pages_cache
        if pages is None and parsed_cache_path().exists():
            payload = json.loads(parsed_cache_path().read_text(encoding="utf-8"))
            pages = list(payload.get("pages") or []) if isinstance(payload, dict) else list(payload)
            self.pages_cache = pages
        if not pages:
            self.pdf_company_map = self._build_pdf_company_map_from_files()
            return self.pdf_company_map

        self.pdf_company_map = self._build_pdf_company_map(pages)
        if not self.pdf_company_map:
            self.pdf_company_map = self._build_pdf_company_map_from_files()
        return self.pdf_company_map

    def _resolve_target_pdfs(self, target_company: str) -> List[str]:
        if not target_company:
            return []
        normalized_target = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9����()]", "", target_company).strip()
        if not normalized_target:
            return []
        from backend.services.rag_pipeline.rag_utils import get_company_aliases
        pdf_company_map = self._ensure_pdf_company_map()
        target_aliases = get_company_aliases(normalized_target)
        matched_pdfs: List[str] = []
        for source_pdf, aliases in pdf_company_map.items():
            if any(
                target_alias in alias or alias in target_alias
                for target_alias in target_aliases
                for alias in aliases
            ):
                matched_pdfs.append(source_pdf)
        return dedupe_preserve_order(matched_pdfs)

    # -------------------------------------------------------------------------
    # Page loading
    # -------------------------------------------------------------------------

    def _load_or_parse_pages(
        self,
        force_parse: bool = False,
        pdf_path: Path | None = None,
        cache_path: Path | None = None,
    ) -> List[Dict[str, object]]:
        if pdf_path is not None:
            target_cache_path = cache_path or (settings.artifact_dir / "uploaded_parsed_pages.json")
            target_cache_path.parent.mkdir(parents=True, exist_ok=True)
            if target_cache_path.exists() and not force_parse:
                payload = json.loads(target_cache_path.read_text(encoding="utf-8"))
                return list(payload.get("pages") or []) if isinstance(payload, dict) else list(payload)
            pages = self.parser.parse(pdf_path)
            target_cache_path.write_text(json.dumps({"pages": pages}, ensure_ascii=False), encoding="utf-8")
            return pages

        cache_path = parsed_cache_path()
        if cache_path.exists() and not force_parse:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return list(payload.get("pages") or []) if isinstance(payload, dict) else list(payload)

        pdf_paths = default_pdf_paths()
        all_pages: List[Dict[str, object]] = []
        for target_pdf in pdf_paths:
            all_pages.extend(self.parser.parse(target_pdf))

        cache_path.write_text(
            json.dumps({"pdfs": [str(path) for path in pdf_paths], "pages": all_pages}, ensure_ascii=False),
            encoding="utf-8",
        )
        return all_pages

    def _ensure_pages_cache_loaded(self) -> List[Dict[str, object]]:
        if self.pages_cache is not None:
            return self.pages_cache
        cache_path = parsed_cache_path()
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.pages_cache = list(payload.get("pages") or []) if isinstance(payload, dict) else list(payload)
            return self.pages_cache
        self.pages_cache = self._load_or_parse_pages(force_parse=False)
        return self.pages_cache

    # -------------------------------------------------------------------------
    # Retrieval helpers
    # -------------------------------------------------------------------------

    def _build_runtime_default_index_if_needed(self) -> None:
        if self.vector_store.fallback_records:
            return
        pages = self.pages_cache
        if pages is None and parsed_cache_path().exists():
            payload = json.loads(parsed_cache_path().read_text(encoding="utf-8"))
            pages = list(payload.get("pages") or []) if isinstance(payload, dict) else list(payload)
            self.pages_cache = pages
        if pages is None:
            try:
                pages = self._load_or_parse_pages(force_parse=False)
                self.pages_cache = pages
            except Exception:
                pages = None
        if not pages:
            return
        chunks = build_main_chunks(pages) + build_pdf_intelligence_chunks(pages)
        self.vector_store.load_runtime_chunks(chunks)
        self.pdf_company_map = self._build_pdf_company_map(pages)
        if not self.pdf_company_map:
            self.pdf_company_map = self._build_pdf_company_map_from_files()

    def _merge_specialized_matches(
        self,
        base_matches: List[Dict[str, object]],
        extra_matches: List[Dict[str, object]],
        top_k: int,
    ) -> List[Dict[str, object]]:
        if not extra_matches:
            return base_matches
        merged: Dict[str, Dict[str, object]] = {str(item["chunk_id"]): dict(item) for item in base_matches}
        for item in extra_matches:
            chunk_id = str(item["chunk_id"])
            if chunk_id not in merged:
                enriched = dict(item)
                enriched["score"] = max(
                    float(enriched.get("score", 0.0)),
                    min(1.0, float(enriched.get("raw_score", 0.0))),
                )
                merged[chunk_id] = enriched
                continue
            merged_item = merged[chunk_id]
            merged_item["score"] = max(float(merged_item.get("score", 0.0)), float(item.get("score", 0.0)))
            merged_item["raw_score"] = max(
                float(merged_item.get("raw_score", 0.0)),
                float(item.get("raw_score", 0.0)),
            )
            merged_item["specialized_score"] = max(
                float(merged_item.get("specialized_score", 0.0)),
                float(item.get("specialized_score", 0.0)),
            )
            merged_item["raw_score"] = max(
                float(merged_item.get("raw_score", 0.0)),
                float(item.get("raw_score", 0.0)),
            ) + min(0.06, float(item.get("specialized_score", 0.0)) * 0.05)
            merged_item["metadata"] = {**dict(item.get("metadata") or {}), **dict(merged_item.get("metadata") or {})}

        ranked = list(merged.values())
        ranked.sort(
            key=lambda item: (
                float(item.get("specialized_score", 0.0)),
                float(item.get("multi_query_score", 0.0)),
                float(item.get("raw_score", item.get("score", 0.0))),
            ),
            reverse=True,
        )
        candidate_limit = max(top_k, settings.reranker_candidate_limit, settings.multi_query_top_k)
        return ranked[:candidate_limit]

    def _retrieve_specialized(
        self,
        vector_store: MilvusVectorStore,
        intent,
        top_k: int,
        target_pdfs: List[str],
    ) -> List[Dict[str, object]]:
        preferred_sections = list(intent.preferred_sections or [])
        query_tags = list(intent.query_tags or [])
        field_keys = list(intent.field_keys or [])
        if not preferred_sections and not query_tags and not field_keys:
            return []
        candidate_limit = max(top_k, settings.multi_query_top_k)
        return vector_store.search_by_metadata(
            query_text=intent.rewritten_query,
            top_k=candidate_limit,
            source_pdfs=target_pdfs,
            field_keys=field_keys,
            preferred_sections=preferred_sections,
            query_tags=query_tags,
            preferred_block_types=list(intent.preferred_block_types or []),
            question_type=str(intent.question_type or ""),
        )

    def _normalize_lookup_text(self, text: str) -> str:
        return re.sub(r"\s+", "", text or "")

    def _find_page_record(self, source_pdf: str, page_number: int) -> Dict[str, object] | None:
        for page in self._ensure_pages_cache_loaded():
            if str(page.get("source_pdf") or "") == source_pdf and int(page.get("page_number") or 0) == page_number:
                return page
        return None

    def _build_candidate_from_vlm_item(
        self,
        *,
        source_pdf: str,
        page_number: int,
        title: str,
        value: str,
        evidence: str,
        item_type: str,
        score: float = 1.38,
    ) -> Dict[str, object]:
        page = self._find_page_record(source_pdf, page_number) or {
            "source_pdf": source_pdf,
            "page_number": page_number,
            "logical_page": None,
            "page_type": "vlm_structured",
            "primary_type": "mixed",
            "sub_type": "visual_summary",
            "section_title": "",
        }
        vlm_page_type = "vlm_structured"
        vlm_sub_type = "visual_summary"
        if item_type.startswith("org_chart"):
            vlm_page_type = "org_chart_summary"
            vlm_sub_type = "org_chart"
        elif item_type.startswith("chart_"):
            vlm_page_type = "chart_summary"
            vlm_sub_type = "chart_summary"
        text = (
            f"�ֶΣ�{title}\n"
            f"ֵ��{value}\n"
            f"֤�ݣ�{evidence}\n"
            f"ҳ�룺{page_number}"
        ).strip()
        return build_candidate_from_page(
            page,
            text=text,
            score=score,
            field_title=title,
            page_type=vlm_page_type,
            primary_type="figure" if vlm_page_type in {"org_chart_summary", "chart_summary"} else "form",
            sub_type=vlm_sub_type,
            source="pdf_vlm_cache",
            structured_facts=[{
                "title": title,
                "value": value,
                "evidence": evidence,
                "type": item_type,
            }],
            content_tags=(
                ["organization_structure"]
                if item_type.startswith("org_chart")
                else ["chart_analysis"]
                if item_type.startswith("chart_")
                else []
            ),
        )

    def _match_company_pages(
        self,
        source_pdfs: List[str],
        *,
        required_terms: List[str] | None = None,
        optional_terms: List[str] | None = None,
        score: float = 1.34,
        field_title: str = "",
    ) -> List[Dict[str, object]]:
        required_terms = required_terms or []
        optional_terms = optional_terms or []
        candidates: List[Dict[str, object]] = []
        for page in self._ensure_pages_cache_loaded():
            source_pdf = str(page.get("source_pdf") or "")
            if source_pdfs and source_pdf not in source_pdfs:
                continue
            blob = f"{page.get('text') or ''}\n{page.get('tables_markdown') or ''}"
            normalized_blob = self._normalize_lookup_text(blob)
            if required_terms and not all(self._normalize_lookup_text(term) in normalized_blob for term in required_terms):
                continue
            if optional_terms and not any(self._normalize_lookup_text(term) in normalized_blob for term in optional_terms):
                continue
            candidates.append(
                build_candidate_from_page(
                    page,
                    text=blob.strip(),
                    score=score,
                    field_title=field_title,
                    source="parsed_page_fallback",
                )
            )
        return candidates

    def _score_structured_candidate(self, intent, item: Dict[str, object], target_pdfs: List[str]) -> float:
        from backend.services.rag_pipeline._scoring import answerability_bonus, answer_context_bonus
        metadata = dict(item.get("metadata") or {})
        source_pdf = str(metadata.get("source_pdf") or "")
        raw_score = 0.18
        if target_pdfs and source_pdf in target_pdfs:
            raw_score += 0.18
        raw_score += answerability_bonus(intent, item)
        raw_score += answer_context_bonus(intent, item)
        if str(metadata.get("page_type") or "") in {"structured", "vlm_structured", "table_analysis", "org_chart_summary", "chart_summary"}:
            raw_score += 0.12
        if str(metadata.get("primary_type") or "") == "table":
            raw_score += 0.06
        if str(metadata.get("sub_type") or "") in {"org_chart", "chart_summary"}:
            raw_score += 0.08
        text = str(item.get("text") or "")
        field_title = str(metadata.get("field_title") or "")
        payload = self._normalize_lookup_text(f"{field_title}\n{text}")
        for field_key in list(intent.field_keys or []):
            normalized_key = self._normalize_lookup_text(field_key)
            if normalized_key and normalized_key in payload:
                raw_score += 0.10
        for tag in list(intent.query_tags or []):
            for token_map in [
                ("fundraising", ["ļ���ʽ�", "ļͶ��Ŀ", "���������ʽ�"]),
                ("related_party", ["������", "�عɹɶ�", "ʵ�ʿ�����", "�ֹɱ���"]),
                ("military_revenue", ["��������", "�����ͻ�", "��Ӫҵ���������"]),
                ("technical_standard", ["�����ƶ�", "������׼", "�淶"]),
                ("org_chart", ["��֯�ṹ", "���۲�", "���۴�", "����"]),
                ("chart_analysis", ["������", "������", "���", "ͼ"]),
            ]:
                tag_name, tokens = token_map
                if tag == tag_name and any(self._normalize_lookup_text(t) in payload for t in tokens):
                    raw_score += 0.08
        return raw_score

    def _retrieve_structured_candidates(self, intent, target_pdfs: List[str], top_k: int) -> List[Dict[str, object]]:
        target_company = str(intent.target_company or "")
        if not target_pdfs:
            target_pdfs = self._resolve_target_pdfs(target_company)

        candidates: List[Dict[str, object]] = []
        pages = self._ensure_pages_cache_loaded()
        for page in pages:
            source_pdf = str(page.get("source_pdf") or "")
            if target_pdfs and source_pdf not in target_pdfs:
                continue
            text = str(page.get("text") or "").strip()
            if text:
                page_candidate = build_candidate_from_page(
                    page,
                    text=text,
                    score=0.32,
                    source="parsed_page_structured",
                )
                page_candidate["raw_score"] = self._score_structured_candidate(intent, page_candidate, target_pdfs)
                page_candidate["score"] = min(1.0, float(page_candidate["raw_score"]))
                page_candidate["specialized_score"] = float(page_candidate["raw_score"])
                candidates.append(page_candidate)

            table_text = str(page.get("tables_markdown") or "").strip()
            if table_text:
                table_candidate = build_candidate_from_page(
                    page,
                    text=table_text,
                    score=0.36,
                    page_type="table_markdown",
                    primary_type="table",
                    sub_type=str(page.get("sub_type") or "simple_table"),
                    source="parsed_table_structured",
                )
                table_candidate["raw_score"] = self._score_structured_candidate(intent, table_candidate, target_pdfs) + 0.06
                table_candidate["score"] = min(1.0, float(table_candidate["raw_score"]))
                table_candidate["specialized_score"] = float(table_candidate["raw_score"])
                candidates.append(table_candidate)

        for source_pdf in target_pdfs:
            for raw_item in load_pdf_vlm_items(source_pdf):
                item = self._build_candidate_from_vlm_item(
                    source_pdf=source_pdf,
                    page_number=int(raw_item.get("page_number") or 0),
                    title=str(raw_item.get("title") or ""),
                    value=str(raw_item.get("value") or ""),
                    evidence=str(raw_item.get("evidence") or ""),
                    item_type=str(raw_item.get("item_type") or "field"),
                )
                raw_score = self._score_structured_candidate(intent, item, target_pdfs) + 0.10
                item["raw_score"] = raw_score
                item["score"] = min(1.0, raw_score)
                item["specialized_score"] = raw_score
                candidates.append(item)

        deduped: Dict[str, Dict[str, object]] = {}
        for item in candidates:
            chunk_id = str(item.get("chunk_id"))
            existing = deduped.get(chunk_id)
            if existing is None or float(item.get("raw_score", 0.0)) > float(existing.get("raw_score", 0.0)):
                deduped[chunk_id] = item

        ranked = list(deduped.values())
        ranked.sort(
            key=lambda item: (
                float(item.get("specialized_score", item.get("raw_score", 0.0))),
                float(item.get("raw_score", 0.0)),
            ),
            reverse=True,
        )
        return ranked[: max(top_k, settings.multi_query_top_k)]

    # -------------------------------------------------------------------------
    # Ingest
    # -------------------------------------------------------------------------

    def _build_pages_by_pdf(self, pages: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
        pages_by_pdf: Dict[str, List[Dict[str, object]]] = {}
        for page in pages:
            source_pdf_path = str(page.get("source_pdf_path") or "").strip()
            source_pdf_name = str(page.get("source_pdf") or "").strip()
            group_key = source_pdf_path or source_pdf_name
            if not group_key:
                continue
            pages_by_pdf.setdefault(group_key, []).append(page)
        return pages_by_pdf

    def _run_vlm_for_pdfs(
        self,
        *,
        pdf_paths: List[Path],
        pages_by_pdf: Dict[str, List[Dict[str, object]]],
        cache_root: Path | None = None,
        failure_path: Path | None = None,
    ) -> tuple[List, List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
        vlm_chunks_list: List = []
        vlm_selected_pages: List[Dict[str, object]] = []
        vlm_failed_pages: List[Dict[str, object]] = []
        vlm_cache_hit_pages: List[Dict[str, object]] = []
        vlm_api_success_pages: List[Dict[str, object]] = []

        for pdf_path in pdf_paths:
            current_pages = pages_by_pdf.get(str(pdf_path), [])
            if not current_pages:
                current_pages = pages_by_pdf.get(pdf_path.name, [])
            if not current_pages:
                continue
            (
                current_vlm_chunks,
                current_selected_pages,
                current_failed_pages,
                current_cache_hit_pages,
                current_api_success_pages,
            ) = build_vlm_chunks(
                current_pages,
                pdf_path,
                self.pdf_vlm,
                cache_root=cache_root,
                failure_path=failure_path,
            )
            vlm_chunks_list.extend(current_vlm_chunks)
            vlm_selected_pages.extend({"source_pdf": pdf_path.name, "page_number": page_number} for page_number in current_selected_pages)
            vlm_failed_pages.extend({"source_pdf": pdf_path.name, "page_number": page_number} for page_number in current_failed_pages)
            vlm_cache_hit_pages.extend({"source_pdf": pdf_path.name, "page_number": page_number} for page_number in current_cache_hit_pages)
            vlm_api_success_pages.extend({"source_pdf": pdf_path.name, "page_number": page_number} for page_number in current_api_success_pages)
        return (
            vlm_chunks_list,
            vlm_selected_pages,
            vlm_failed_pages,
            vlm_cache_hit_pages,
            vlm_api_success_pages,
        )

    def _build_all_chunks_for_pages(
        self,
        *,
        pages: List[Dict[str, object]],
        pdf_paths: List[Path],
        cache_root: Path | None = None,
        failure_path: Path | None = None,
        log_prefix: str = "[ingest]",
    ) -> tuple[List, Dict[str, object]]:
        pages_by_pdf = self._build_pages_by_pdf(pages)
        logger.info("%s stage=chunk_main start pages=%s", log_prefix, len(pages))
        base_chunks = build_main_chunks(pages)
        logger.info("%s stage=chunk_main done chunks=%s", log_prefix, len(base_chunks))
        logger.info("%s stage=chunk_intelligence start", log_prefix)
        intelligence_chunks = build_pdf_intelligence_chunks(pages)
        logger.info("%s stage=chunk_intelligence done chunks=%s", log_prefix, len(intelligence_chunks))
        logger.info("%s stage=chunk_enhanced start", log_prefix)
        enhanced_chunks = build_enhanced_chunks(pages, self.llm)
        logger.info("%s stage=chunk_enhanced done chunks=%s", log_prefix, len(enhanced_chunks))
        logger.info("%s stage=chunk_vlm start pdf_count=%s", log_prefix, len(pdf_paths))
        (
            vlm_chunks_list,
            vlm_selected_pages,
            vlm_failed_pages,
            vlm_cache_hit_pages,
            vlm_api_success_pages,
        ) = self._run_vlm_for_pdfs(
            pdf_paths=pdf_paths,
            pages_by_pdf=pages_by_pdf,
            cache_root=cache_root,
            failure_path=failure_path,
        )
        logger.info(
            "%s stage=chunk_vlm done chunks=%s selected_pages=%s cache_hits=%s api_success=%s failed=%s",
            log_prefix,
            len(vlm_chunks_list),
            len(vlm_selected_pages),
            len(vlm_cache_hit_pages),
            len(vlm_api_success_pages),
            len(vlm_failed_pages),
        )
        all_chunks = base_chunks + intelligence_chunks + enhanced_chunks + vlm_chunks_list
        stats = {
            "base_chunks": len(base_chunks),
            "pdf_intelligence_chunks": len(intelligence_chunks),
            "enhanced_chunks": len(enhanced_chunks),
            "vlm_enhanced_chunks": len(vlm_chunks_list),
            "vlm_selected_pages": vlm_selected_pages,
            "vlm_failed_pages": vlm_failed_pages,
            "vlm_cache_hit_pages": vlm_cache_hit_pages,
            "vlm_api_success_pages": vlm_api_success_pages,
        }
        return all_chunks, stats

    def _build_manifest_payload(
        self,
        *,
        pdf_paths: List[Path],
        inserted: int,
        collection_name: str,
        stats: Dict[str, object],
    ) -> Dict[str, object]:
        return {
            "pdfs": [str(path) for path in pdf_paths],
            "pdf_count": len(pdf_paths),
            "mode": "heavy_unified",
            "chunks": inserted,
            **stats,
            "collection_name": collection_name,
            "embedding_backend": self.embedder.backend,
            "dimension": self.embedder.dimension,
            "vector_store": "milvus" if self.vector_store.connected else "in_memory",
            "pdf_parser_backend": settings.pdf_parser_backend,
        }

    def _write_upload_stage_outputs(
        self,
        *,
        upload_id: str,
        pages: List[Dict[str, object]],
        manifest_payload: Dict[str, object],
    ) -> None:
        parsed_path = upload_parsed_pages_path(upload_id)
        redacted_path = upload_redacted_pages_path(upload_id)
        chunk_manifest_path = upload_chunk_manifest_path(upload_id)
        enhanced_pages_path = upload_enhanced_pages_path(upload_id)

        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        redacted_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced_pages_path.parent.mkdir(parents=True, exist_ok=True)

        parsed_path.write_text(json.dumps({"pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8")
        write_redacted_export(pages, redacted_path)
        chunk_manifest_path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        enhanced_pages_path.write_text(
            json.dumps({"pages": pages}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _find_resumable_upload(self, filename: str, file_hash: str) -> Dict[str, object] | None:
        for root in sorted(ensure_uploads_root().iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            if not root.is_dir():
                continue
            manifest = _read_json_file(root / "manifest.json")
            if not manifest or str(manifest.get("filename") or "") != filename:
                continue
            source_path_value = str(manifest.get("pdf_path") or "").strip()
            if not source_path_value:
                continue
            source_path = Path(source_path_value)
            if not source_path.exists():
                continue
            try:
                source_hash = file_sha1(source_path)
            except Exception:
                continue
            if source_hash != file_hash:
                continue
            stages = manifest.get("stages") or {}
            stage5_value = str(stages.get("stage5") or "").strip() if isinstance(stages, dict) else ""
            stage5_path = Path(stage5_value) if stage5_value else (root / "artifacts" / "stage5_persist_manifest.json")
            if stage5_path.exists():
                continue
            return {
                "upload_id": root.name,
                "root": root,
                "manifest": manifest,
            }
        return None

    def ingest_main(self, force: bool = False) -> int:
        pdf_paths = default_pdf_paths()
        started = time.perf_counter()
        logger.info("[ingest-main] start pdf_count=%s force=%s", len(pdf_paths), force)
        logger.info("[ingest-main] stage=parse start")
        pages = self._load_or_parse_pages(force_parse=force)
        logger.info("[ingest-main] parse done pages=%s elapsed_ms=%s", len(pages), int((time.perf_counter() - started) * 1000))
        logger.info("[ingest-main] stage=redact_export start")
        write_redacted_export(pages, redacted_cache_path())
        logger.info("[ingest-main] stage=redact_export done")
        all_chunks, stats = self._build_all_chunks_for_pages(pages=pages, pdf_paths=pdf_paths, log_prefix="[ingest-main]")
        logger.info("[ingest-main] stage=embedding start chunks=%s", len(all_chunks))
        embeddings = self.embedder.embed_texts(chunk.text for chunk in all_chunks)
        logger.info("[ingest-main] stage=embedding done vectors=%s", len(embeddings))
        logger.info("[ingest-main] stage=milvus_reset start collection=%s", self.vector_store.collection_name)
        self.vector_store.clear()
        logger.info("[ingest-main] stage=milvus_upsert start")
        inserted = self.vector_store.upsert_chunks(all_chunks, embeddings)
        logger.info("[ingest-main] stage=milvus_upsert done inserted=%s", inserted)
        manifest_payload = self._build_manifest_payload(
            pdf_paths=pdf_paths,
            inserted=inserted,
            collection_name=self.vector_store.collection_name,
            stats=stats,
        )
        main_manifest_path().write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        enhance_manifest_path().write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.pages_cache = pages
        self.pdf_company_map = self._build_pdf_company_map(pages)
        logger.info(
            "[ingest-main] done chunks=%s elapsed_ms=%s",
            inserted,
            int((time.perf_counter() - started) * 1000),
        )
        return inserted

    def ingest_enhancement(self, force: bool = False) -> int:
        return self.ingest_main(force=force)

    def reset_default_collection(self) -> None:
        self.vector_store.clear()
        self.pages_cache = None
        self.pdf_company_map = {}
        self.vector_store._clear_local_indexes()
        self.uploaded_pages_cache = None
        self.uploaded_pdf_name = ""
        self.active_upload_id = ""
        self.uploaded_vector_store = None
        for path in [
            main_manifest_path(),
            enhance_manifest_path(),
            parsed_cache_path(),
            redacted_cache_path(),
            settings.artifact_dir / "pdf_vlm_last_failure.json",
        ]:
            path.unlink(missing_ok=True)
        pdf_vlm_cache_dir = settings.artifact_dir / "pdf_vlm_cache"
        if pdf_vlm_cache_dir.exists():
            shutil.rmtree(pdf_vlm_cache_dir, ignore_errors=True)

    def ingest_uploaded_pdf(self, pdf_path: Path, original_filename: str) -> Dict[str, object]:
        source_hash = file_sha1(pdf_path)
        resumable = self._find_resumable_upload(original_filename, source_hash)
        if resumable is not None:
            upload_id = str(resumable["upload_id"])
            source_dir = upload_source_dir(upload_id)
            source_dir.mkdir(parents=True, exist_ok=True)
            artifact_dir = upload_artifact_dir(upload_id)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            source_target = source_dir / Path(original_filename).name
            if pdf_path.resolve() != source_target.resolve():
                shutil.copy2(pdf_path, source_target)
            logger.info("[upload-pdf] resume upload_id=%s filename=%s artifact_dir=%s", upload_id, original_filename, artifact_dir)
        else:
            upload_id = build_upload_id(original_filename)
            source_dir = upload_source_dir(upload_id)
            source_dir.mkdir(parents=True, exist_ok=True)
            artifact_dir = upload_artifact_dir(upload_id)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            source_target = source_dir / Path(original_filename).name
            if pdf_path.resolve() != source_target.resolve():
                shutil.copy2(pdf_path, source_target)
        started = time.perf_counter()
        pending_registry_entry = {
            "upload_id": upload_id,
            "filename": original_filename,
            "stored_pdf_path": str(source_target),
            "artifact_dir": str(artifact_dir),
            "text_collection_name": "",
            "visual_collection_name": "",
            "mongo_collection_name": "",
            "chunks": 0,
            "file_sha1": file_sha1(source_target),
            "uploaded_at": int(time.time()),
            "status": "processing",
        }
        self.uploaded_registry = upsert_upload_registry_entry(pending_registry_entry)
        self.active_upload_id = upload_id
        self.uploaded_pdf_name = original_filename
        logger.info("[upload-pdf] ingest start upload_id=%s filename=%s", upload_id, original_filename)
        legacy_result = run_legacy_pdf_pipeline(
            pdf_path=source_target,
            artifact_dir=artifact_dir,
            original_filename=original_filename,
            upload_id=upload_id,
        )
        pages = load_resolved_pages_as_parser_output(legacy_result, source_target)
        upload_parsed_pages_path(upload_id).parent.mkdir(parents=True, exist_ok=True)
        upload_parsed_pages_path(upload_id).write_text(
            json.dumps({"pages": pages}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_payload = {
            "upload_id": upload_id,
            "filename": original_filename,
            "pdf_path": str(source_target),
            "artifact_dir": str(artifact_dir),
            "text_collection_name": legacy_result.text_collection_name,
            "visual_collection_name": legacy_result.visual_collection_name,
            "mongo_collection_name": legacy_result.mongo_collection_name,
            "stages": {
                "stage0": str(legacy_result.stage0_path),
                "stage1": str(legacy_result.stage1_path),
                "stage2": str(legacy_result.stage2_dir),
                "stage3": str(legacy_result.stage3_dir),
                "stage4_text": str(legacy_result.stage4_text_dir),
                "stage4_visual": str(legacy_result.stage4_visual_dir),
                "stage5": str(legacy_result.stage5_manifest_path),
            },
        }
        upload_manifest_path(upload_id).write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("[upload-pdf] stage=manifest done upload_id=%s", upload_id)
        self._write_upload_stage_outputs(upload_id=upload_id, pages=pages, manifest_payload=manifest_payload)
        stage5_manifest = json.loads(legacy_result.stage5_manifest_path.read_text(encoding="utf-8"))
        total_inserted = sum(
            step.get("count", 0)
            for step in stage5_manifest.get("steps", {}).values()
            if isinstance(step, dict) and "count" in step
        )
        registry_entry = {
            "upload_id": upload_id,
            "filename": original_filename,
            "stored_pdf_path": str(source_target),
            "artifact_dir": str(artifact_dir),
            "text_collection_name": legacy_result.text_collection_name,
            "visual_collection_name": legacy_result.visual_collection_name,
            "mongo_collection_name": legacy_result.mongo_collection_name,
            "chunks": total_inserted,
            "file_sha1": file_sha1(source_target),
            "uploaded_at": int(time.time()),
            "status": "ready",
        }
        self.uploaded_registry = upsert_upload_registry_entry(registry_entry)
        self.active_upload_id = upload_id
        self.uploaded_pdf_name = original_filename
        self.uploaded_pages_cache = pages
        self.uploaded_vector_store = MilvusVectorStore(
            settings.milvus_uri, legacy_result.text_collection_name, self.embedder.dimension,
        )
        logger.info(
            "[upload-pdf] ingest done upload_id=%s chunks=%s collection=%s elapsed_ms=%s",
            upload_id,
            total_inserted,
            legacy_result.text_collection_name,
            int((time.perf_counter() - started) * 1000),
        )
        return {
            "upload_id": upload_id,
            "filename": original_filename,
            "chunks": total_inserted,
            "collection_name": legacy_result.text_collection_name,
            "visual_collection_name": legacy_result.visual_collection_name,
            "mongo_collection_name": legacy_result.mongo_collection_name,
        }

    def save_uploaded_pdf(self, filename: str, content: bytes) -> Path:
        safe_name = Path(filename).name or "uploaded.pdf"
        target = self.upload_dir / safe_name
        target.write_bytes(content)
        return target

    def save_uploaded_pdf_stream(self, filename: str, upload_file) -> Path:
        safe_name = Path(filename).name or "uploaded.pdf"
        target = self.upload_dir / safe_name
        with target.open("wb") as handle:
            while True:
                chunk = upload_file.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        return target

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    def _build_retrieval_queries(
        self,
        original_query: str,
        rewritten_query: str,
        search_queries: List[str],
        sub_questions: List[str],
        rewrite_strategy: str,
    ) -> List[str]:
        simple_queries = [original_query.strip(), rewritten_query.strip()]
        expanded_queries = [*simple_queries, *search_queries]
        decomposed_queries = [*simple_queries, *sub_questions, *search_queries]

        if rewrite_strategy == "decomposed":
            queries = decomposed_queries
            query_limit = max(settings.multi_query_max_queries, 6)
        elif rewrite_strategy == "expanded":
            queries = expanded_queries
            query_limit = max(settings.multi_query_max_queries, 5)
        else:
            queries = [*simple_queries, *search_queries[:2]]
            query_limit = min(settings.multi_query_max_queries, 3)

        deduped = dedupe_preserve_order([q for q in queries if q])
        if not settings.multi_query_enabled:
            return deduped[:1] if deduped else []
        return deduped[:query_limit]

    def _retrieve_multi_query(
        self,
        vector_store: MilvusVectorStore,
        retrieval_queries: List[str],
        top_k: int,
        question_type: str = "",
        preferred_block_types: List[str] | None = None,
    ) -> List[Dict[str, object]]:
        if not retrieval_queries:
            return []
        merged: Dict[str, Dict[str, object]] = {}
        per_query_top_k = max(top_k, settings.multi_query_top_k)
        preferred_block_types = preferred_block_types or []

        for query_index, retrieval_query in enumerate(retrieval_queries):
            query_embedding = self.embedder.embed_query(retrieval_query)
            matches = vector_store.search(
                query_embedding,
                per_query_top_k,
                query_text=retrieval_query,
                question_type=question_type,
                preferred_block_types=preferred_block_types,
            )
            query_weight = 1.0 if query_index == 0 else 0.92

            for rank, item in enumerate(matches, start=1):
                chunk_id = str(item["chunk_id"])
                rrf_score = query_weight / (settings.rrf_k + rank)
                if chunk_id not in merged:
                    merged[chunk_id] = {
                        **item,
                        "retrieval_queries": [retrieval_query],
                        "multi_query_score": float(item.get("score", 0.0)) + rrf_score,
                    }
                    continue

                merged_item = merged[chunk_id]
                merged_item["score"] = max(float(merged_item.get("score", 0.0)), float(item.get("score", 0.0)))
                merged_item["dense_score"] = max(
                    float(merged_item.get("dense_score", 0.0)),
                    float(item.get("dense_score", 0.0)),
                )
                merged_item["bm25_score"] = max(
                    float(merged_item.get("bm25_score", 0.0)),
                    float(item.get("bm25_score", 0.0)),
                )
                merged_item["multi_query_score"] = float(merged_item.get("multi_query_score", 0.0)) + rrf_score
                merged_item["retrieval_queries"] = dedupe_preserve_order(
                    [*list(merged_item.get("retrieval_queries") or []), retrieval_query]
                )

        ranked = list(merged.values())
        ranked.sort(
            key=lambda item: (
                float(item.get("multi_query_score", 0.0)),
                float(item.get("score", 0.0)),
            ),
            reverse=True,
        )
        candidate_limit = max(top_k, settings.reranker_candidate_limit, settings.multi_query_top_k)
        return ranked[:candidate_limit]

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        use_llm: bool = True,
        corpus: CorpusName = "default",
        upload_id: str | None = None,
    ) -> QueryResponse:
        started = time.perf_counter()
        intent = analyze_query(query)
        top_k = top_k or settings.top_k
        if corpus == "uploaded":
            vector_store, upload_entry = self._resolve_uploaded_store(upload_id)
            logger.info(
                "[query-uploaded] start upload_id=%s filename=%s collection=%s query=%s",
                upload_entry.get("upload_id") if upload_entry else "",
                upload_entry.get("filename") if upload_entry else "",
                vector_store.collection_name,
                query[:80],
            )
        else:
            vector_store = self.vector_store
            upload_entry = None
        if corpus == "default":
            should_build_runtime_fallback = True
            if self.vector_store.connected:
                try:
                    if self.vector_store.count() > 0:
                        should_build_runtime_fallback = False
                        if not self.pdf_company_map:
                            self.pdf_company_map = self._build_pdf_company_map_from_files()
                except Exception:
                    should_build_runtime_fallback = True
            if should_build_runtime_fallback:
                self._build_runtime_default_index_if_needed()

        target_pdfs = self._resolve_target_pdfs(intent.target_company) if corpus == "default" else []

        retrieval_queries = self._build_retrieval_queries(
            query,
            intent.rewritten_query,
            intent.search_queries,
            intent.sub_questions,
            intent.rewrite_strategy,
        )
        stage_started = time.perf_counter()
        matches = self._retrieve_multi_query(
            vector_store=vector_store,
            retrieval_queries=retrieval_queries,
            top_k=top_k,
            question_type=str(intent.question_type or ""),
            preferred_block_types=list(intent.preferred_block_types or []),
        )
        logger.info(
            "[query-%s] multi_query_ms=%s collection=%s query_count=%s hits=%s",
            corpus,
            int((time.perf_counter() - stage_started) * 1000),
            vector_store.collection_name,
            len(retrieval_queries),
            len(matches),
        )
        stage_started = time.perf_counter()
        specialized_matches = self._retrieve_specialized(
            vector_store=vector_store,
            intent=intent,
            top_k=top_k,
            target_pdfs=target_pdfs,
        )
        logger.info(
            "[query-%s] specialized_ms=%s collection=%s hits=%s",
            corpus,
            int((time.perf_counter() - stage_started) * 1000),
            vector_store.collection_name,
            len(specialized_matches),
        )
        stage_started = time.perf_counter()
        structured_matches = self._retrieve_structured_candidates(intent, target_pdfs, top_k) if corpus == "default" else []
        logger.info(
            "[query-%s] structured_ms=%s collection=%s hits=%s",
            corpus,
            int((time.perf_counter() - stage_started) * 1000),
            vector_store.collection_name,
            len(structured_matches),
        )
        matches = self._merge_specialized_matches(matches, specialized_matches, top_k)
        matches = self._merge_specialized_matches(matches, structured_matches, top_k)
        matches = apply_company_routing(matches, target_pdfs, top_k)
        rerank_query = query if intent.rewrite_strategy == "decomposed" else intent.rewritten_query
        if self.reranker is not None and self.reranker.is_enabled() and matches:
            candidate_limit = min(len(matches), max(top_k, settings.reranker_candidate_limit, settings.multi_query_top_k))
            stage_started = time.perf_counter()
            matches = self.reranker.rerank(
                rerank_query,
                matches[:candidate_limit],
                max(top_k, settings.reranker_top_n),
            )
            logger.info(
                "[query-%s] rerank_ms=%s collection=%s candidate_limit=%s hits=%s",
                corpus,
                int((time.perf_counter() - stage_started) * 1000),
                vector_store.collection_name,
                candidate_limit,
                len(matches),
            )
            matches = apply_company_routing(matches, target_pdfs, top_k)
        else:
            matches = matches[:top_k]

        matches = rerank_for_answerability(matches, intent, top_k)
        answer_contexts = select_answer_contexts(matches, intent, top_k, target_pdfs)
        if corpus == "uploaded":
            answer_contexts = self._expand_uploaded_answer_contexts(
                answer_query := (query if intent.rewrite_strategy == "decomposed" else intent.rewritten_query),
                retrieval_rows=matches,
                answer_contexts=answer_contexts,
                upload_entry=upload_entry,
                top_k=top_k,
            )
        else:
            answer_query = query if intent.rewrite_strategy == "decomposed" else intent.rewritten_query
        stage_started = time.perf_counter()
        answer = self.llm.answer(answer_query, answer_contexts, intent=intent) if use_llm else "\n".join(
            [f"��{item['page_number']}ҳ��{item['text'][:180]}" for item in answer_contexts]
        )
        logger.info(
            "[query-%s] answer_ms=%s collection=%s contexts=%s use_llm=%s",
            corpus,
            int((time.perf_counter() - stage_started) * 1000),
            vector_store.collection_name,
            len(answer_contexts),
            use_llm,
        )
        citations = [
            SourceChunk(
                chunk_id=str(
                    item.get("chunk_id")
                    or item.get("metadata", {}).get("chunk_id")
                    or item.get("metadata", {}).get("table_id")
                    or item.get("metadata", {}).get("visual_id")
                    or f"ctx_{index + 1}"
                ),
                page_number=int(item["page_number"]),
                logical_page=item.get("logical_page"),
                score=float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
                text=str(item["text"]),
                metadata={
                    "logical_page": str(item.get("logical_page") or ""),
                    "corpus": corpus,
                    **dict(item.get("metadata") or {}),
                },
            )
            for index, item in enumerate(answer_contexts)
        ]
        latency_ms = int((time.perf_counter() - started) * 1000)
        grounded = bool(matches) and ("�޷������й�˵��������" not in answer) and ("δ���������֤��" not in answer)
        rerank_mode = (
            f"dense+bm25+multi_query[{intent.rewrite_strategy}]+specialized+company_route+rerank"
            if target_pdfs
            else f"dense+bm25+multi_query[{intent.rewrite_strategy}]+specialized+rerank"
        ) if (self.reranker is not None and self.reranker.is_enabled()) else (
            f"dense+bm25+multi_query[{intent.rewrite_strategy}]+specialized+company_route"
            if target_pdfs
            else f"dense+bm25+multi_query[{intent.rewrite_strategy}]+specialized"
        )
        return QueryResponse(
            answer=answer,
            citations=citations,
            intent=intent,
            latency_ms=latency_ms,
            retrieval_mode=rerank_mode,
            grounded=grounded,
            answer_source=str(self.llm.last_call_details.get("answer_source") or ""),
            llm_status=str(self.llm.last_call_details.get("status") or ""),
            llm_error=str(self.llm.last_call_details.get("last_error") or ""),
        )

    def health(self) -> Dict[str, object]:
        uploaded_documents = [
            {
                "upload_id": str(item.get("upload_id") or ""),
                "filename": str(item.get("filename") or ""),
                "chunks": int(item.get("chunks") or 0),
                "collection_name": str(item.get("text_collection_name") or ""),
                "uploaded_at": int(item.get("uploaded_at") or 0),
                "status": str(item.get("status") or "ready"),
            }
            for item in self.list_uploaded_documents()
        ]
        active_uploaded_count = 0
        if self.uploaded_vector_store is not None:
            try:
                active_uploaded_count = self.uploaded_vector_store.count()
            except Exception:
                active_uploaded_count = 0
        return {
            "status": "ok",
            "pdf_exists": bool(settings.pdf_paths) and all(path.exists() for path in settings.pdf_paths),
            "milvus_uri": settings.milvus_uri,
            "embedding_backend": self.embedder.backend,
            "llm_provider": self.llm.provider,
            "llm_api_url": self.llm.api_url,
            "llm_model_name": self.llm.model_name,
            "llm_fallback_api_url": self.llm.fallback_api_url,
            "llm_fallback_model_name": self.llm.fallback_model_name,
            "vector_store": "milvus" if self.vector_store.connected else "in_memory",
            "uploaded_pdf_active": bool(uploaded_documents),
            "uploaded_pdf_name": self.uploaded_pdf_name,
            "pdf_parser_backend": settings.pdf_parser_backend,
            "pdf_vlm_enabled": self.pdf_vlm.is_enabled(),
            "pdf_vlm_model_name": settings.pdf_vlm_model_name,
            "multi_query_enabled": settings.multi_query_enabled,
            "bm25_k1": settings.bm25_k1,
            "bm25_b": settings.bm25_b,
            "reranker_enabled": bool(self.reranker and self.reranker.is_enabled()),
            "reranker_backend": self.reranker.backend if self.reranker else "disabled",
            "reranker_model_path": settings.reranker_model_path,
            "default_collection_name": self.vector_store.collection_name,
            "default_collection_count": self.vector_store.count(),
            "uploaded_documents": uploaded_documents,
            "runtime_fallback_active": bool(self.vector_store.fallback_records),
        }
