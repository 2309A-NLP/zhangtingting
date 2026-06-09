# ������ţ��˹����� NLP-RAG-ͼ�����ݽ����������Ż�
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Literal

import fitz

from backend.config import settings
from backend.schemas import QueryResponse, SourceChunk
from backend.services.embedding import EmbeddingService
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


CorpusName = Literal["default", "uploaded"]


class RAGPipeline:
    def __init__(self) -> None:
        self.parser = PDFParser(ocr_lang=settings.ocr_lang)
        self.embedder = EmbeddingService(settings.model_dir / "embedding", configured_path=settings.embedding_model_path)
        self.vector_store = MilvusVectorStore(settings.milvus_uri, settings.collection_name, self.embedder.dimension)
        self.uploaded_vector_store = MilvusVectorStore(
            settings.milvus_uri,
            f"{settings.collection_name}_uploaded",
            self.embedder.dimension,
        )
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
        self.uploaded_pages_cache: List[Dict[str, object]] | None = None
        self.uploaded_pdf_name = ""
        self.upload_dir = settings.artifact_dir / "uploads"
        settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

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

    def _load_or_parse_pages(self, force_parse: bool = False, pdf_path: Path | None = None) -> List[Dict[str, object]]:
        if pdf_path is not None:
            cache_path = settings.artifact_dir / "uploaded_parsed_pages.json"
            if cache_path.exists() and not force_parse:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                return list(payload.get("pages") or []) if isinstance(payload, dict) else list(payload)
            pages = self.parser.parse(pdf_path)
            cache_path.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
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

    def ingest_main(self, force: bool = False) -> int:
        pdf_paths = default_pdf_paths()
        pages = self._load_or_parse_pages(force_parse=force)
        write_redacted_export(pages, redacted_cache_path())
        pages_by_pdf: Dict[str, List[Dict[str, object]]] = {}
        for page in pages:
            source_pdf_path = str(page.get("source_pdf_path") or "").strip()
            source_pdf_name = str(page.get("source_pdf") or "").strip()
            group_key = source_pdf_path or source_pdf_name
            if not group_key:
                continue
            pages_by_pdf.setdefault(group_key, []).append(page)

        base_chunks = build_main_chunks(pages)
        intelligence_chunks = build_pdf_intelligence_chunks(pages)
        enhanced_chunks = build_enhanced_chunks(pages, self.llm)

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
            ) = build_vlm_chunks(current_pages, pdf_path, self.pdf_vlm)
            vlm_chunks_list.extend(current_vlm_chunks)
            vlm_selected_pages.extend({"source_pdf": pdf_path.name, "page_number": page_number} for page_number in current_selected_pages)
            vlm_failed_pages.extend({"source_pdf": pdf_path.name, "page_number": page_number} for page_number in current_failed_pages)
            vlm_cache_hit_pages.extend({"source_pdf": pdf_path.name, "page_number": page_number} for page_number in current_cache_hit_pages)
            vlm_api_success_pages.extend({"source_pdf": pdf_path.name, "page_number": page_number} for page_number in current_api_success_pages)

        all_chunks = base_chunks + intelligence_chunks + enhanced_chunks + vlm_chunks_list
        embeddings = self.embedder.embed_texts(chunk.text for chunk in all_chunks)
        self.vector_store.clear()
        inserted = self.vector_store.upsert_chunks(all_chunks, embeddings)
        manifest_payload = {
            "pdfs": [str(path) for path in pdf_paths],
            "pdf_count": len(pdf_paths),
            "mode": "heavy_unified",
            "chunks": inserted,
            "base_chunks": len(base_chunks),
            "pdf_intelligence_chunks": len(intelligence_chunks),
            "enhanced_chunks": len(enhanced_chunks),
            "vlm_enhanced_chunks": len(vlm_chunks_list),
            "vlm_selected_pages": vlm_selected_pages,
            "vlm_failed_pages": vlm_failed_pages,
            "vlm_cache_hit_pages": vlm_cache_hit_pages,
            "vlm_api_success_pages": vlm_api_success_pages,
            "collection_name": self.vector_store.collection_name,
            "embedding_backend": self.embedder.backend,
            "dimension": self.embedder.dimension,
            "vector_store": "milvus" if self.vector_store.connected else "in_memory",
            "pdf_parser_backend": settings.pdf_parser_backend,
        }
        main_manifest_path().write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        enhance_manifest_path().write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.pages_cache = pages
        self.pdf_company_map = self._build_pdf_company_map(pages)
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
        self.uploaded_vector_store._clear_local_indexes()
        for path in [
            main_manifest_path(),
            enhance_manifest_path(),
            parsed_cache_path(),
            redacted_cache_path(),
            settings.artifact_dir / "pdf_vlm_last_failure.json",
            settings.artifact_dir / "uploaded_parsed_pages.json",
            settings.artifact_dir / "uploaded_parsed_pages_redacted.json",
            settings.artifact_dir / "uploaded_ingest_manifest.json",
        ]:
            path.unlink(missing_ok=True)
        pdf_vlm_cache_dir = settings.artifact_dir / "pdf_vlm_cache"
        if pdf_vlm_cache_dir.exists():
            shutil.rmtree(pdf_vlm_cache_dir, ignore_errors=True)

    def ingest_uploaded_pdf(self, pdf_path: Path, original_filename: str) -> int:
        pages = self.parser.parse(pdf_path)
        self.uploaded_vector_store.clear()
        chunks = build_main_chunks(pages) + build_pdf_intelligence_chunks(pages)
        embeddings = self.embedder.embed_texts(chunk.text for chunk in chunks)
        inserted = self.uploaded_vector_store.upsert_chunks(chunks, embeddings)
        (settings.artifact_dir / "uploaded_ingest_manifest.json").write_text(
            json.dumps({"pdf": original_filename, "mode": "main", "chunks": inserted}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (settings.artifact_dir / "uploaded_parsed_pages.json").write_text(
            json.dumps(pages, ensure_ascii=False), encoding="utf-8",
        )
        write_redacted_export(pages, settings.artifact_dir / "uploaded_parsed_pages_redacted.json")
        self.uploaded_pages_cache = pages
        self.uploaded_pdf_name = original_filename
        return inserted

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
    ) -> QueryResponse:
        started = time.perf_counter()
        intent = analyze_query(query)
        top_k = top_k or settings.top_k
        vector_store = self.uploaded_vector_store if corpus == "uploaded" else self.vector_store
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
        matches = self._retrieve_multi_query(
            vector_store=vector_store,
            retrieval_queries=retrieval_queries,
            top_k=top_k,
            question_type=str(intent.question_type or ""),
            preferred_block_types=list(intent.preferred_block_types or []),
        )
        specialized_matches = self._retrieve_specialized(
            vector_store=vector_store,
            intent=intent,
            top_k=top_k,
            target_pdfs=target_pdfs,
        )
        structured_matches = self._retrieve_structured_candidates(intent, target_pdfs, top_k)
        matches = self._merge_specialized_matches(matches, specialized_matches, top_k)
        matches = self._merge_specialized_matches(matches, structured_matches, top_k)
        matches = apply_company_routing(matches, target_pdfs, top_k)
        rerank_query = query if intent.rewrite_strategy == "decomposed" else intent.rewritten_query
        if self.reranker is not None and self.reranker.is_enabled() and matches:
            candidate_limit = min(len(matches), max(top_k, settings.reranker_candidate_limit, settings.multi_query_top_k))
            matches = self.reranker.rerank(
                rerank_query,
                matches[:candidate_limit],
                max(top_k, settings.reranker_top_n),
            )
            matches = apply_company_routing(matches, target_pdfs, top_k)
        else:
            matches = matches[:top_k]

        matches = rerank_for_answerability(matches, intent, top_k)
        answer_contexts = select_answer_contexts(matches, intent, top_k, target_pdfs)
        answer_query = query if intent.rewrite_strategy == "decomposed" else intent.rewritten_query
        answer = self.llm.answer(answer_query, answer_contexts, intent=intent) if use_llm else "\n".join(
            [f"��{item['page_number']}ҳ��{item['text'][:180]}" for item in answer_contexts]
        )
        citations = [
            SourceChunk(
                chunk_id=str(item["chunk_id"]),
                page_number=int(item["page_number"]),
                logical_page=item.get("logical_page"),
                score=float(item.get("rerank_score", item["score"])),
                text=str(item["text"]),
                metadata={
                    "logical_page": str(item.get("logical_page") or ""),
                    "corpus": corpus,
                    **dict(item.get("metadata") or {}),
                },
            )
            for item in answer_contexts
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
        )

    def health(self) -> Dict[str, object]:
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
            "uploaded_pdf_active": self.uploaded_vector_store.count() > 0,
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
            "uploaded_collection_name": self.uploaded_vector_store.collection_name,
            "uploaded_collection_count": self.uploaded_vector_store.count(),
            "runtime_fallback_active": bool(self.vector_store.fallback_records),
        }
