from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path

import fitz
import numpy as np
from bs4 import BeautifulSoup
from paddleocr import PaddleOCR
from pypdf import PdfReader

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import emit_runtime_trace, log_timed, preview_text
from app.knowledge.models import DocumentSection, ParsedDocument, RawDocument
from app.knowledge.pdf_parser import ComplexPdfParser

logger = get_logger(__name__)

HEADING_PATTERN = re.compile(
    r"^\s*("
    r"(chapter|section|appendix)\s+\S+"
    r"|"
    r"[0-9]+(\.[0-9]+)*"
    r"|"
    r"[一二三四五六七八九十百千万亿]+[、.)）]"
    r"|"
    r"第[一二三四五六七八九十百千万亿\d]+[章节条款]"
    r")",
    re.IGNORECASE | re.UNICODE,
)


class DocumentLoader:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logger
        self._ocr: PaddleOCR | None = None
        self._pdf_parser = ComplexPdfParser()

    @log_timed("document_load")
    async def load(self, raw_document: RawDocument) -> ParsedDocument:
        suffix = Path(raw_document.file_name).suffix.lower()
        emit_runtime_trace(
            self.logger,
            "document_load_entered",
            task_id=raw_document.task_id,
            doc_id=raw_document.file_id,
            file_name=raw_document.file_name,
            suffix=suffix,
            local_path=raw_document.local_path,
            source_uri=raw_document.source_uri,
        )

        if suffix == ".pdf":
            emit_runtime_trace(self.logger, "document_load_route_selected", parser="pdf")
            parsed = await self._load_pdf(raw_document)
        elif suffix == ".txt":
            emit_runtime_trace(self.logger, "document_load_route_selected", parser="txt")
            parsed = await self._load_txt(raw_document)
        elif suffix == ".json":
            emit_runtime_trace(self.logger, "document_load_route_selected", parser="json")
            parsed = await self._load_json(raw_document)
        elif suffix in {".html", ".htm"}:
            emit_runtime_trace(self.logger, "document_load_route_selected", parser="html")
            parsed = await self._load_html(raw_document)
        else:
            emit_runtime_trace(
                self.logger,
                "document_load_unsupported_type",
                file_name=raw_document.file_name,
                suffix=suffix,
            )
            raise ValueError(f"Unsupported file type: {suffix}")

        emit_runtime_trace(
            self.logger,
            "document_load_completed",
            doc_id=parsed.doc_id,
            parser_name=parsed.parser_name,
            title=parsed.title,
            section_count=len(parsed.sections),
            table_count=len(parsed.tables),
            text_chars=len(parsed.plain_text),
            text_preview=preview_text(parsed.plain_text, 160),
        )
        logger.info(
            "document_loaded",
            doc_id=parsed.doc_id,
            parser=parsed.parser_name,
            file_name=raw_document.file_name,
        )
        return parsed

    async def _load_pdf(self, raw_document: RawDocument) -> ParsedDocument:
        emit_runtime_trace(
            self.logger,
            "document_load_pdf_started",
            doc_id=raw_document.file_id,
            file_name=raw_document.file_name,
        )
        parsed = await self._pdf_parser.parse(raw_document)
        emit_runtime_trace(
            self.logger,
            "document_load_pdf_finished",
            doc_id=parsed.doc_id,
            parser_name=parsed.parser_name,
            text_preview=preview_text(parsed.plain_text, 160),
        )
        return parsed

    def _parse_pdf_sync(self, raw_document: RawDocument) -> ParsedDocument:
        reader = PdfReader(raw_document.local_path)
        emit_runtime_trace(
            self.logger,
            "document_load_pdf_reader_ready",
            doc_id=raw_document.file_id,
            page_count=len(reader.pages),
        )

        pages: list[str] = []
        for page_index, page in enumerate(reader.pages):
            page_text = (page.extract_text() or "").strip()
            pages.append(page_text)
            if page_index < 3:
                emit_runtime_trace(
                    self.logger,
                    "document_load_pdf_page_extracted",
                    page_index=page_index,
                    chars=len(page_text),
                    preview=preview_text(page_text, 120),
                )

        text = "\n\n".join(page for page in pages if page).strip()
        parser_name = "pypdf"
        if not text and self.settings.ocr_enabled:
            emit_runtime_trace(
                self.logger,
                "document_load_pdf_ocr_fallback_started",
                doc_id=raw_document.file_id,
                local_path=raw_document.local_path,
            )
            text = self._ocr_pdf(raw_document.local_path)
            parser_name = "paddleocr"
            emit_runtime_trace(
                self.logger,
                "document_load_pdf_ocr_fallback_finished",
                doc_id=raw_document.file_id,
                chars=len(text),
                preview=preview_text(text, 160),
            )

        sections = self._extract_sections(text)
        title = sections[0].heading if sections and sections[0].heading else Path(raw_document.file_name).stem

        return ParsedDocument(
            doc_id=raw_document.file_id,
            user_id=raw_document.user_id,
            role_id=raw_document.role_id,
            title=title,
            plain_text=text,
            source_uri=raw_document.source_uri,
            file_name=raw_document.file_name,
            content_type=raw_document.content_type,
            parser_name=parser_name,
            sections=sections,
            tables=[],
            metadata={"source_type": raw_document.source_type, **raw_document.metadata},
        )

    async def _load_txt(self, raw_document: RawDocument) -> ParsedDocument:
        text = await asyncio.to_thread(Path(raw_document.local_path).read_text, "utf-8")
        sections = self._extract_sections(text)
        emit_runtime_trace(
            self.logger,
            "document_load_txt_ready",
            doc_id=raw_document.file_id,
            section_count=len(sections),
            chars=len(text),
            preview=preview_text(text, 160),
        )
        return ParsedDocument(
            doc_id=raw_document.file_id,
            user_id=raw_document.user_id,
            role_id=raw_document.role_id,
            title=Path(raw_document.file_name).stem,
            plain_text=text,
            source_uri=raw_document.source_uri,
            file_name=raw_document.file_name,
            content_type=raw_document.content_type,
            parser_name="txt",
            sections=sections,
            tables=[],
            metadata={"source_type": raw_document.source_type, **raw_document.metadata},
        )

    async def _load_json(self, raw_document: RawDocument) -> ParsedDocument:
        text = await asyncio.to_thread(Path(raw_document.local_path).read_text, "utf-8")
        data = json.loads(text)
        normalized = json.dumps(data, ensure_ascii=False, indent=2)
        sections = [DocumentSection(heading="JSON", level=1, content=normalized)]
        emit_runtime_trace(
            self.logger,
            "document_load_json_ready",
            doc_id=raw_document.file_id,
            chars=len(normalized),
            preview=preview_text(normalized, 160),
        )
        return ParsedDocument(
            doc_id=raw_document.file_id,
            user_id=raw_document.user_id,
            role_id=raw_document.role_id,
            title=Path(raw_document.file_name).stem,
            plain_text=normalized,
            source_uri=raw_document.source_uri,
            file_name=raw_document.file_name,
            content_type=raw_document.content_type,
            parser_name="json",
            sections=sections,
            tables=[],
            metadata={"source_type": raw_document.source_type, **raw_document.metadata},
        )

    async def _load_html(self, raw_document: RawDocument) -> ParsedDocument:
        html = await asyncio.to_thread(Path(raw_document.local_path).read_text, "utf-8")
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.string.strip() if soup.title and soup.title.string else Path(raw_document.file_name).stem
        sections = self._extract_html_sections(soup)
        plain_text = "\n\n".join(section.content for section in sections if section.content).strip()
        tables = self._extract_html_tables(soup)
        emit_runtime_trace(
            self.logger,
            "document_load_html_ready",
            doc_id=raw_document.file_id,
            section_count=len(sections),
            table_count=len(tables),
            preview=preview_text(plain_text, 160),
        )
        return ParsedDocument(
            doc_id=raw_document.file_id,
            user_id=raw_document.user_id,
            role_id=raw_document.role_id,
            title=title,
            plain_text=plain_text,
            source_uri=raw_document.source_uri,
            file_name=raw_document.file_name,
            content_type=raw_document.content_type,
            parser_name="html",
            sections=sections,
            tables=tables,
            metadata={"source_type": raw_document.source_type, **raw_document.metadata},
        )

    def _get_ocr(self) -> PaddleOCR:
        if self._ocr is None:
            emit_runtime_trace(
                self.logger,
                "document_load_ocr_model_loading",
                language=self.settings.ocr_language,
            )
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.settings.ocr_language,
                show_log=False,
            )
            emit_runtime_trace(
                self.logger,
                "document_load_ocr_model_loaded",
                language=self.settings.ocr_language,
            )
        return self._ocr

    def _ocr_pdf(self, pdf_path: str) -> str:
        pdf = fitz.open(pdf_path)
        texts: list[str] = []
        ocr = self._get_ocr()

        emit_runtime_trace(
            self.logger,
            "document_load_ocr_pdf_opened",
            pdf_path=pdf_path,
            page_count=pdf.page_count,
        )

        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            result = ocr.ocr(image, cls=True)
            page_lines: list[str] = []
            for line_group in result or []:
                for line in line_group:
                    if len(line) > 1 and line[1]:
                        page_lines.append(str(line[1][0]).strip())
            page_text = "\n".join(page_lines)
            texts.append(page_text)
            if page_index < 3:
                emit_runtime_trace(
                    self.logger,
                    "document_load_ocr_page_finished",
                    page_index=page_index,
                    line_count=len(page_lines),
                    preview=preview_text(page_text, 120),
                )

        pdf.close()
        return "\n\n".join(part for part in texts if part).strip()

    def _extract_html_sections(self, soup: BeautifulSoup) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        current_heading = soup.title.string.strip() if soup.title and soup.title.string else "HTML"
        current_level = 1
        buffer: list[str] = []

        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            name = element.name.lower()
            text = element.get_text(" ", strip=True)
            if not text:
                continue
            if name.startswith("h"):
                if buffer:
                    sections.append(
                        DocumentSection(
                            heading=current_heading,
                            level=current_level,
                            content="\n".join(buffer).strip(),
                        )
                    )
                    buffer.clear()
                current_heading = text
                current_level = int(name[1])
            else:
                buffer.append(text)

        if buffer:
            sections.append(
                DocumentSection(
                    heading=current_heading,
                    level=current_level,
                    content="\n".join(buffer).strip(),
                )
            )

        if not sections:
            body_text = soup.get_text("\n", strip=True)
            sections.append(DocumentSection(heading=current_heading, level=1, content=body_text))

        return sections

    def _extract_html_tables(self, soup: BeautifulSoup) -> list[dict[str, str]]:
        tables: list[dict[str, str]] = []
        for index, table in enumerate(soup.find_all("table"), start=1):
            tables.append(
                {
                    "caption": f"table_{index}",
                    "html": str(table),
                    "text": table.get_text(" | ", strip=True),
                }
            )
        return tables

    def _extract_sections(self, text: str) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        current_heading = ""
        buffer: list[str] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if HEADING_PATTERN.match(line) and len(line) <= 120:
                if buffer:
                    sections.append(
                        DocumentSection(
                            heading=current_heading or "Body",
                            level=1,
                            content="\n".join(buffer).strip(),
                        )
                    )
                    buffer.clear()
                current_heading = line
            else:
                buffer.append(line)

        if buffer:
            sections.append(
                DocumentSection(
                    heading=current_heading or "Body",
                    level=1,
                    content="\n".join(buffer).strip(),
                )
            )

        if not sections and text.strip():
            sections.append(DocumentSection(heading="Body", level=1, content=text.strip()))
        return sections


def build_raw_document(
    user_id: str,
    role_id: str,
    file_name: str,
    content_type: str,
    local_path: str,
    file_id: str | None = None,
    task_id: str | None = None,
    source_uri: str | None = None,
    source_type: str = "upload",
    metadata: dict[str, str] | None = None,
) -> RawDocument:
    resolved_file_id = file_id or uuid.uuid4().hex
    resolved_task_id = task_id or resolved_file_id
    return RawDocument(
        file_id=resolved_file_id,
        task_id=resolved_task_id,
        user_id=user_id,
        role_id=role_id,
        file_name=file_name,
        content_type=content_type,
        source_uri=source_uri or local_path,
        local_path=local_path,
        source_type=source_type,
        metadata=metadata or {},
    )
