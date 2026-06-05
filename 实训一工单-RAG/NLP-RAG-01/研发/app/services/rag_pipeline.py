from __future__ import annotations

# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统

import json
import time
from pathlib import Path
from typing import Dict, List, Literal

from app.config import settings
from app.schemas import QueryResponse, SourceChunk
from app.services.embedding import EmbeddingService
from app.services.llm_client import LLMClient
from app.services.pdf_parser import PDFParser
from app.services.query_understanding import analyze_query
from app.services.text_utils import build_chunks
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
        )
        self.pages_cache: List[Dict[str, object]] | None = None
        self.uploaded_pages_cache: List[Dict[str, object]] | None = None
        self.uploaded_pdf_name = ""
        self.upload_dir = settings.artifact_dir / "uploads"
        settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _write_redacted_export(self, pages: List[Dict[str, object]], output_path: Path) -> None:
        redacted_export = [
            {
                "page_number": page["page_number"],
                "logical_page": page["logical_page"],
                "text": page.get("redacted_text", page["text"]),
                "redaction_stats": page["redaction_stats"],
            }
            for page in pages
        ]
        output_path.write_text(
            json.dumps(redacted_export, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _ingest_pages(
        self,
        pages: List[Dict[str, object]],
        vector_store: MilvusVectorStore,
        manifest_path: Path,
        pdf_label: str,
        redacted_output_path: Path,
    ) -> int:
        self._write_redacted_export(pages, redacted_output_path)
        chunks = build_chunks(pages, settings.chunk_size, settings.chunk_overlap)
        embeddings = self.embedder.embed_texts(chunk.text for chunk in chunks)
        inserted = vector_store.upsert_chunks(chunks, embeddings)
        manifest_path.write_text(
            json.dumps(
                {
                    "pdf": pdf_label,
                    "chunks": inserted,
                    "embedding_backend": self.embedder.backend,
                    "dimension": self.embedder.dimension,
                    "vector_store": "milvus" if vector_store.connected else "in_memory",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return inserted

    def ingest(self, force: bool = False) -> int:
        manifest_path = settings.artifact_dir / "ingest_manifest.json"
        parsed_cache_path = settings.artifact_dir / "parsed_pages.json"
        if manifest_path.exists() and not force and self.vector_store.count() > 0:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            return int(data["chunks"])
        if parsed_cache_path.exists() and not force:
            pages = json.loads(parsed_cache_path.read_text(encoding="utf-8"))
        else:
            pages = self.parser.parse(settings.pdf_path)
            parsed_cache_path.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
        inserted = self._ingest_pages(
            pages=pages,
            vector_store=self.vector_store,
            manifest_path=manifest_path,
            pdf_label=str(settings.pdf_path),
            redacted_output_path=settings.artifact_dir / "parsed_pages_redacted.json",
        )
        self.pages_cache = pages
        return inserted

    def ingest_uploaded_pdf(self, pdf_path: Path, original_filename: str) -> int:
        pages = self.parser.parse(pdf_path)
        self.uploaded_vector_store.clear()
        inserted = self._ingest_pages(
            pages=pages,
            vector_store=self.uploaded_vector_store,
            manifest_path=settings.artifact_dir / "uploaded_ingest_manifest.json",
            pdf_label=original_filename,
            redacted_output_path=settings.artifact_dir / "uploaded_parsed_pages_redacted.json",
        )
        (settings.artifact_dir / "uploaded_parsed_pages.json").write_text(
            json.dumps(pages, ensure_ascii=False),
            encoding="utf-8",
        )
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
        query_embedding = self.embedder.embed_query(intent.rewritten_query)
        vector_store = self.uploaded_vector_store if corpus == "uploaded" else self.vector_store
        matches = vector_store.search(query_embedding, top_k, query_text=intent.rewritten_query)
        answer = self.llm.answer(intent.rewritten_query, matches) if use_llm else "\n".join(
            [f"第{item['page_number']}页：{item['text'][:180]}" for item in matches]
        )
        citations = [
            SourceChunk(
                chunk_id=str(item["chunk_id"]),
                page_number=int(item["page_number"]),
                logical_page=item.get("logical_page"),
                score=float(item["score"]),
                text=str(item["text"]),
                metadata={"logical_page": str(item.get("logical_page") or ""), "corpus": corpus},
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
            retrieval_mode=self.embedder.backend,
            grounded=grounded,
        )

    def health(self) -> Dict[str, object]:
        return {
            "status": "ok",
            "pdf_exists": settings.pdf_path.exists(),
            "milvus_uri": settings.milvus_uri,
            "embedding_backend": self.embedder.backend,
            "llm_provider": settings.llm_provider,
            "vector_store": "milvus" if self.vector_store.connected else "in_memory",
            "uploaded_pdf_active": self.uploaded_vector_store.count() > 0,
            "uploaded_pdf_name": self.uploaded_pdf_name,
        }
