# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Literal

import fitz

from app.config import settings
from app.schemas import QueryResponse, SourceChunk
from app.services.embedding import EmbeddingService
from app.services.llm_client import LLMClient
from app.services.pdf_parser import PDFParser
from app.services.pdf_vlm_client import PDFVLMClient
from app.services.query_understanding import analyze_query
from app.services.rerank_service import RerankService
from app.services.text_utils import Chunk, build_chunks, dedupe_preserve_order, stable_chunk_id
from app.services.vector_store import MilvusVectorStore


CorpusName = Literal["default", "uploaded"]


class RAGPipeline:
    def __init__(self) -> None:
        self.parser = PDFParser(ocr_lang=settings.ocr_lang)
        self.embedder = EmbeddingService(settings.model_dir, configured_path=settings.embedding_model_path)
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
        self.uploaded_pages_cache: List[Dict[str, object]] | None = None
        self.uploaded_pdf_name = ""
        self.upload_dir = settings.artifact_dir / "uploads"
        settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _looks_like_mojibake(self, text: str) -> bool:
        sample = (text or "").strip()
        if not sample:
            return False
        markers = [
            "銆",
            "锛",
            "鈥",
            "鏈€",
            "鍏",
            "鍙戣",
            "璇",
            "鐨勬",
            "涓€",
            "鍒嗗埆",
            "闈",
            "閲",
        ]
        hits = sum(sample.count(marker) for marker in markers)
        return hits >= 3

    def _sanitize_vlm_context(self, page: Dict[str, object]) -> tuple[str, str]:
        local_text = str(page.get("text") or "")
        table_markdown = str(page.get("tables_markdown") or "")
        if self._looks_like_mojibake(local_text):
            local_text = ""
        if self._looks_like_mojibake(table_markdown):
            table_markdown = ""
        return local_text, table_markdown

    def _is_valid_enhanced_item(self, item: Dict[str, object]) -> bool:
        title = str(item.get("title") or "").strip()
        value = str(item.get("value") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        item_type = str(item.get("type") or "").strip()

        if not title or not value:
            return False

        low_confidence_markers = [
            "无具体",
            "未披露",
            "未说明",
            "无法判断",
            "无法确定",
            "未检索到",
            "没有提及",
            "不详",
        ]
        if any(marker in value for marker in low_confidence_markers):
            return False

        # 字段抽取类增强必须附带一定长度的原文证据，避免短幻觉块污染主检索。
        if item_type in {"field", "fact", "table_fact", "table_summary"} and len(evidence) < 6:
            return False

        return True

    def _write_redacted_export(self, pages: List[Dict[str, object]], output_path: Path) -> None:
        redacted_export = [
            {
                "page_number": page["page_number"],
                "logical_page": page["logical_page"],
                "page_type": page.get("page_type", "text"),
                "section_title": page.get("section_title", ""),
                "source": page.get("source", "builtin"),
                "text": page.get("redacted_text", page["text"]),
                "redaction_stats": page["redaction_stats"],
            }
            for page in pages
        ]
        output_path.write_text(json.dumps(redacted_export, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_pdf_vlm_failure(self, pdf_path: Path, page_number: int, error: str, failed_pages: List[int]) -> None:
        failure_path = settings.artifact_dir / "pdf_vlm_last_failure.json"
        failure_path.write_text(
            json.dumps(
                {
                    "pdf": str(pdf_path),
                    "page_number": page_number,
                    "failed_pages": failed_pages,
                    "error": error,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _main_manifest_path(self) -> Path:
        return settings.artifact_dir / "ingest_manifest.json"

    def _enhance_manifest_path(self) -> Path:
        return settings.artifact_dir / "enhance_manifest.json"

    def _parsed_cache_path(self) -> Path:
        return settings.artifact_dir / "parsed_pages.json"

    def _redacted_cache_path(self) -> Path:
        return settings.artifact_dir / "parsed_pages_redacted.json"

    def _load_or_parse_pages(self, force_parse: bool = False, pdf_path: Path | None = None) -> List[Dict[str, object]]:
        target_pdf = pdf_path or settings.pdf_path
        parsed_cache_path = self._parsed_cache_path() if pdf_path is None else settings.artifact_dir / "uploaded_parsed_pages.json"
        if parsed_cache_path.exists() and not force_parse:
            pages = json.loads(parsed_cache_path.read_text(encoding="utf-8"))
        else:
            pages = self.parser.parse(target_pdf)
            parsed_cache_path.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
        return pages

    def _build_main_chunks(self, pages: List[Dict[str, object]]) -> List[Chunk]:
        return build_chunks(pages, settings.chunk_size, settings.chunk_overlap)

    def _build_enhanced_chunks(self, pages: List[Dict[str, object]]) -> List[Chunk]:
        if not settings.llm_enhancement_enabled:
            return []

        enhanced_chunks: List[Chunk] = []
        page_budget = min(len(pages), settings.llm_enhancement_max_pages)
        keywords = ["注册资本", "法定代表人", "补充流动资金", "技术标准", "一等奖", "供应商", "上游", "下游"]
        candidate_pages = [
            page
            for page in pages[:page_budget]
            if page.get("tables_markdown")
            or str(page.get("page_type") or "") in {"table", "ocr"}
            or any(keyword in str(page.get("text") or "") for keyword in keywords)
        ]

        for page in candidate_pages:
            structured_items = [
                item for item in self.llm.structure_page(page) if self._is_valid_enhanced_item(item)
            ]
            for index, item in enumerate(structured_items):
                text = (
                    f"字段：{item['title']}\n"
                    f"值：{item['value']}\n"
                    f"证据：{item.get('evidence') or ''}\n"
                    f"页码：{page['page_number']}"
                ).strip()
                chunk_id = stable_chunk_id(int(page["page_number"]), 100000 + index, text)
                enhanced_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=text,
                        page_number=int(page["page_number"]),
                        logical_page=page.get("logical_page"),
                        metadata={
                            "page_number": str(page["page_number"]),
                            "logical_page": str(page.get("logical_page") or ""),
                            "page_type": "structured",
                            "section_title": str(page.get("section_title") or ""),
                            "has_table": "1" if page.get("tables_markdown") else "0",
                            "has_ocr": "1" if page.get("handwriting") else "0",
                            "source": "llm_enhanced",
                            "field_title": str(item["title"]),
                            "field_type": str(item.get("type") or "field"),
                        },
                    )
                )

            if settings.llm_table_analysis_enabled and page.get("tables_markdown"):
                table_items = [
                    item for item in self.llm.analyze_table(page) if self._is_valid_enhanced_item(item)
                ]
                for table_index, item in enumerate(table_items, start=len(structured_items)):
                    text = (
                        f"表格分析：{item['title']}\n"
                        f"结论：{item['value']}\n"
                        f"证据：{item.get('evidence') or ''}\n"
                        f"页码：{page['page_number']}"
                    ).strip()
                    chunk_id = stable_chunk_id(int(page["page_number"]), 200000 + table_index, text)
                    enhanced_chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=text,
                            page_number=int(page["page_number"]),
                            logical_page=page.get("logical_page"),
                            metadata={
                                "page_number": str(page["page_number"]),
                                "logical_page": str(page.get("logical_page") or ""),
                                "page_type": "table_analysis",
                                "section_title": str(page.get("section_title") or ""),
                                "has_table": "1",
                                "has_ocr": "1" if page.get("handwriting") else "0",
                                "source": "llm_table_analysis",
                                "field_title": str(item["title"]),
                                "field_type": str(item.get("type") or "table_trend"),
                            },
                        )
                    )
        return enhanced_chunks

    def _select_pages_for_pdf_vlm(self, pages: List[Dict[str, object]]) -> List[int]:
        selected: List[int] = []
        high_priority: List[int] = []
        normal_priority: List[int] = []

        for index, page in enumerate(pages):
            text = str(page.get("text") or "")
            table_markdown = str(page.get("tables_markdown") or "")
            page_type = str(page.get("page_type") or "")
            metadata = page.get("parse_metadata") or {}
            image_count = int(metadata.get("image_count") or 0)

            weak_text = len(text.strip()) < settings.pdf_vlm_min_text_chars
            complex_table = bool(table_markdown) and len(table_markdown) >= settings.pdf_vlm_table_trigger_chars
            ocr_heavy = page_type == "ocr"
            image_heavy = image_count >= settings.pdf_vlm_image_trigger_count
            key_field_page = any(
                keyword in text
                for keyword in ["法定代表人", "注册资本", "补充流动资金", "技术标准", "供应商", "上游", "下游"]
            )

            if key_field_page and (complex_table or ocr_heavy or page_type == "table"):
                high_priority.append(index)
            elif weak_text or complex_table or ocr_heavy or image_heavy:
                normal_priority.append(index)

        for index in high_priority + normal_priority:
            if index not in selected:
                selected.append(index)
            if len(selected) >= settings.pdf_vlm_max_pages:
                break
        return selected

    def _run_pdf_vlm_enhancement(self, pages: List[Dict[str, object]], pdf_path: Path):
        if not self.pdf_vlm.is_enabled():
            return [], [], [], [], []

        selected_indexes = self._select_pages_for_pdf_vlm(pages)
        if not selected_indexes:
            return [], [], [], [], []

        doc = fitz.open(str(pdf_path))
        enhanced_chunks: List[Chunk] = []
        failed_pages: List[int] = []
        cache_hit_pages: List[int] = []
        api_success_pages: List[int] = []
        cache_dir = settings.artifact_dir / "pdf_vlm_cache" / pdf_path.stem
        try:
            for page_index in selected_indexes:
                page = pages[page_index]
                page_number = int(page["page_number"])
                pdf_page = doc.load_page(page_index)
                pix = pdf_page.get_pixmap(
                    matrix=fitz.Matrix(settings.pdf_vlm_render_scale, settings.pdf_vlm_render_scale),
                    alpha=False,
                )
                try:
                    image_first_result = self.pdf_vlm.enhance_page(
                        page_number=page_number,
                        logical_page=page.get("logical_page"),
                        local_text="",
                        table_markdown="",
                        image_bytes=pix.tobytes("png"),
                        cache_dir=cache_dir,
                        mode="image_only",
                        force_items=True,
                        cache_variant="image_first",
                    )
                    result = image_first_result

                    if not list(result.get("items") or []):
                        safe_local_text, safe_table_markdown = self._sanitize_vlm_context(page)
                        if safe_local_text or safe_table_markdown:
                            result = self.pdf_vlm.enhance_page(
                                page_number=page_number,
                                logical_page=page.get("logical_page"),
                                local_text=safe_local_text,
                                table_markdown=safe_table_markdown,
                                image_bytes=pix.tobytes("png"),
                                cache_dir=cache_dir,
                                mode="full",
                                force_items=True,
                                cache_variant="assist",
                            )
                except Exception as exc:
                    failed_pages.append(page_number)
                    self._write_pdf_vlm_failure(pdf_path, page_number, str(exc), failed_pages)
                    raise RuntimeError(
                        f"PDF VLM enhancement failed on page {page_number}. "
                        f"failed_pages={failed_pages}. detail={exc}"
                    ) from exc

                if result.get("status") == "cache_hit":
                    cache_hit_pages.append(page_number)
                elif result.get("status") == "api_success":
                    api_success_pages.append(page_number)
                elif result.get("status") == "failed":
                    failed_pages.append(page_number)

                items = [item for item in list(result.get("items") or []) if self._is_valid_enhanced_item(item)]
                for item_index, item in enumerate(items):
                    text = (
                        f"字段：{item['title']}\n"
                        f"值：{item['value']}\n"
                        f"证据：{item.get('evidence') or ''}\n"
                        f"页码：{page_number}"
                    ).strip()
                    chunk_id = stable_chunk_id(page_number, 300000 + item_index, text)
                    enhanced_chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=text,
                            page_number=page_number,
                            logical_page=page.get("logical_page"),
                            metadata={
                                "page_number": str(page_number),
                                "logical_page": str(page.get("logical_page") or ""),
                                "page_type": "vlm_structured",
                                "section_title": str(page.get("section_title") or ""),
                                "has_table": "1" if page.get("tables_markdown") else "0",
                                "has_ocr": "1" if page.get("handwriting") else "0",
                                "source": "pdf_vlm_enhanced",
                                "field_title": str(item["title"]),
                                "field_type": str(item.get("type") or "field"),
                            },
                        )
                    )
        finally:
            doc.close()

        selected_pages = [int(pages[index]["page_number"]) for index in selected_indexes]
        return enhanced_chunks, selected_pages, failed_pages, cache_hit_pages, api_success_pages

    def _persist_runtime_index(self, vector_store: MilvusVectorStore, chunks: List[Chunk]) -> None:
        if not chunks:
            vector_store.load_runtime_records([], [])
            return
        embeddings = self.embedder.embed_texts(chunk.text for chunk in chunks)
        vector_store.load_runtime_records(chunks, embeddings)

    def ingest_main(self, force: bool = False) -> int:
        pages = self._load_or_parse_pages(force_parse=force)
        self._write_redacted_export(pages, self._redacted_cache_path())
        main_chunks = self._build_main_chunks(pages)
        embeddings = self.embedder.embed_texts(chunk.text for chunk in main_chunks)
        inserted = self.vector_store.upsert_chunks(main_chunks, embeddings)
        self._main_manifest_path().write_text(
            json.dumps(
                {
                    "pdf": str(settings.pdf_path),
                    "mode": "main",
                    "chunks": inserted,
                    "base_chunks": len(main_chunks),
                    "embedding_backend": self.embedder.backend,
                    "dimension": self.embedder.dimension,
                    "vector_store": "milvus" if self.vector_store.connected else "in_memory",
                    "pdf_parser_backend": settings.pdf_parser_backend,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.pages_cache = pages
        return inserted

    def ingest_enhancement(self, force: bool = False) -> int:
        pages = self._load_or_parse_pages(force_parse=False)
        base_chunks = self._build_main_chunks(pages)
        enhanced_chunks = self._build_enhanced_chunks(pages)
        (
            vlm_chunks,
            vlm_selected_pages,
            vlm_failed_pages,
            vlm_cache_hit_pages,
            vlm_api_success_pages,
        ) = self._run_pdf_vlm_enhancement(pages, settings.pdf_path)
        all_chunks = base_chunks + enhanced_chunks + vlm_chunks
        embeddings = self.embedder.embed_texts(chunk.text for chunk in all_chunks)
        self.vector_store.clear()
        inserted = self.vector_store.upsert_chunks(all_chunks, embeddings)
        self._enhance_manifest_path().write_text(
            json.dumps(
                {
                    "pdf": str(settings.pdf_path),
                    "mode": "enhancement",
                    "chunks": inserted,
                    "base_chunks": len(base_chunks),
                    "enhanced_chunks": len(enhanced_chunks),
                    "vlm_enhanced_chunks": len(vlm_chunks),
                    "vlm_selected_pages": vlm_selected_pages,
                    "vlm_failed_pages": vlm_failed_pages,
                    "vlm_cache_hit_pages": vlm_cache_hit_pages,
                    "vlm_api_success_pages": vlm_api_success_pages,
                    "embedding_backend": self.embedder.backend,
                    "dimension": self.embedder.dimension,
                    "vector_store": "milvus" if self.vector_store.connected else "in_memory",
                    "pdf_parser_backend": settings.pdf_parser_backend,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.pages_cache = pages
        return inserted

    def reset_default_collection(self) -> None:
        self.vector_store.clear()
        for path in [
            self._main_manifest_path(),
            self._enhance_manifest_path(),
            settings.artifact_dir / "pdf_vlm_last_failure.json",
        ]:
            path.unlink(missing_ok=True)

    def _build_runtime_default_index_if_needed(self) -> None:
        # 只有 Milvus 不可用时才需要构建本地运行时兜底索引。
        # 否则在查询路径里重建 chunks，会把首个请求拖到超时。
        if self.vector_store.connected:
            return

        if self.vector_store.fallback_records:
            return

        pages = self.pages_cache
        if pages is None and self._parsed_cache_path().exists():
            pages = json.loads(self._parsed_cache_path().read_text(encoding="utf-8"))
            self.pages_cache = pages
        if not pages:
            return

        chunks = self._build_main_chunks(pages)
        self._persist_runtime_index(self.vector_store, chunks)

    def ingest_uploaded_pdf(self, pdf_path: Path, original_filename: str) -> int:
        pages = self.parser.parse(pdf_path)
        self.uploaded_vector_store.clear()
        chunks = self._build_main_chunks(pages)
        embeddings = self.embedder.embed_texts(chunk.text for chunk in chunks)
        inserted = self.uploaded_vector_store.upsert_chunks(chunks, embeddings)
        (settings.artifact_dir / "uploaded_ingest_manifest.json").write_text(
            json.dumps(
                {
                    "pdf": original_filename,
                    "mode": "main",
                    "chunks": inserted,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (settings.artifact_dir / "uploaded_parsed_pages.json").write_text(
            json.dumps(pages, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_redacted_export(pages, settings.artifact_dir / "uploaded_parsed_pages_redacted.json")
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

    def _build_retrieval_queries(self, original_query: str, rewritten_query: str, search_queries: List[str]) -> List[str]:
        queries = [original_query.strip(), rewritten_query.strip(), *search_queries]
        deduped = dedupe_preserve_order([query for query in queries if query])
        if not settings.multi_query_enabled:
            return deduped[:1] if deduped else []
        return deduped[: settings.multi_query_max_queries]

    def _retrieve_multi_query(
        self,
        vector_store: MilvusVectorStore,
        retrieval_queries: List[str],
        top_k: int,
    ) -> List[Dict[str, object]]:
        if not retrieval_queries:
            return []

        merged: Dict[str, Dict[str, object]] = {}
        per_query_top_k = max(top_k, settings.multi_query_top_k)

        for query_index, retrieval_query in enumerate(retrieval_queries):
            query_embedding = self.embedder.embed_query(retrieval_query)
            matches = vector_store.search(query_embedding, per_query_top_k, query_text=retrieval_query)
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
            self._build_runtime_default_index_if_needed()

        retrieval_queries = self._build_retrieval_queries(query, intent.rewritten_query, intent.search_queries)
        matches = self._retrieve_multi_query(
            vector_store=vector_store,
            retrieval_queries=retrieval_queries,
            top_k=top_k,
        )
        if self.reranker is not None and self.reranker.is_enabled() and matches:
            candidate_limit = min(len(matches), max(top_k, settings.reranker_candidate_limit, settings.multi_query_top_k))
            matches = self.reranker.rerank(
                intent.rewritten_query,
                matches[:candidate_limit],
                max(top_k, settings.reranker_top_n),
            )
        else:
            matches = matches[:top_k]

        answer = self.llm.answer(intent.rewritten_query, matches) if use_llm else "\n".join(
            [f"第{item['page_number']}页：{item['text'][:180]}" for item in matches]
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
            for item in matches
        ]
        latency_ms = int((time.perf_counter() - started) * 1000)
        grounded = bool(matches) and ("无法基于招股说明书作答" not in answer) and ("未检索到相关证据" not in answer)
        return QueryResponse(
            answer=answer,
            citations=citations,
            intent=intent,
            latency_ms=latency_ms,
            retrieval_mode="dense+bm25+multi_query+rerank"
            if self.reranker is not None and self.reranker.is_enabled()
            else "dense+bm25+multi_query",
            grounded=grounded,
        )

    def health(self) -> Dict[str, object]:
        return {
            "status": "ok",
            "pdf_exists": settings.pdf_path.exists(),
            "milvus_uri": settings.milvus_uri,
            "embedding_backend": self.embedder.backend,
            "llm_provider": settings.llm_provider,
            "llm_api_url": settings.llm_api_url,
            "llm_model_name": settings.llm_model_name,
            "llm_fallback_api_url": settings.llm_fallback_api_url,
            "llm_fallback_model_name": settings.llm_fallback_model_name,
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
        }
