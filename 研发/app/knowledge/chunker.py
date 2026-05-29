from __future__ import annotations

import uuid

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import emit_runtime_trace, log_timed, preview_text
from app.knowledge.models import ChunkedDocument, CleanDocument

logger = get_logger(__name__)


class SemanticChunker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logger
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            length_function=self._token_length,
            keep_separator=True,
            is_separator_regex=False,
        )

    @log_timed("chunk_split")
    def split(self, document: CleanDocument, role_category: str = "general") -> list[ChunkedDocument]:
        emit_runtime_trace(
            self.logger,
            "chunk_split_entered",
            doc_id=document.doc_id,
            role_category=role_category,
            clean_text_preview=preview_text(document.clean_text, 160),
        )
        raw_chunks = self.splitter.split_text(document.clean_text)
        chunks: list[ChunkedDocument] = []
        for index, chunk_text in enumerate(raw_chunks):
            normalized = chunk_text.strip()
            if not normalized:
                continue

            chunk_uuid = uuid.uuid4().hex
            heading_path = self._infer_heading_path(document, normalized)
            chunks.append(
                ChunkedDocument(
                    id=chunk_uuid,
                    doc_id=document.doc_id,
                    chunk_id=f"{document.doc_id}_{index}",
                    user_id=document.user_id,
                    role_id=document.role_id,
                    role_category=role_category,
                    text=normalized,
                    token_count=self._token_length(normalized),
                    chunk_index=index,
                    source=document.source_uri,
                    heading_path=heading_path,
                    metadata={
                        **document.metadata,
                        "title": document.title,
                        "file_name": document.file_name,
                        "parser_name": document.parser_name,
                        "removed_items": document.removed_items,
                    },
                )
            )

        logger.info("document_chunked", doc_id=document.doc_id, chunk_count=len(chunks))
        emit_runtime_trace(
            self.logger,
            "chunk_split_completed",
            doc_id=document.doc_id,
            chunk_count=len(chunks),
            first_chunk=preview_text(chunks[0].text, 120) if chunks else "",
        )
        return chunks

    def _infer_heading_path(self, document: CleanDocument, chunk_text: str) -> str:
        for section in document.sections:
            if section.content and section.content[:50] in chunk_text:
                return section.heading
        return document.title

    def _token_length(self, value: str) -> int:
        return len(self.encoding.encode(value))
