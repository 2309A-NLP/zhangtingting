# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统
from __future__ import annotations

import logging
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import fitz
import pdfplumber

from app.config import settings
from app.services.external_pdf_parser import ExternalPDFParser
from app.services.redaction import redact_sensitive_text
from app.services.text_utils import normalize_whitespace

try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover
    PaddleOCR = None


LOGICAL_PAGE_PATTERN = re.compile(r"1-1-(\d+)")

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)


class PDFParser:
    def __init__(self, ocr_lang: str = "ch") -> None:
        self.ocr_lang = ocr_lang
        self._ocr = None
        self.external_parser = ExternalPDFParser()

    def _get_ocr(self):
        if self._ocr is None and PaddleOCR is not None:
            try:
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self.ocr_lang, show_log=False)
            except Exception as exc:
                if "show_log" not in str(exc):
                    raise
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self.ocr_lang)
        return self._ocr

    def _header_footer_candidates(self, pages: List[fitz.Page]) -> Tuple[set[str], set[str]]:
        headers: List[str] = []
        footers: List[str] = []
        for page in pages[: min(20, len(pages))]:
            text_lines = [line.strip() for line in page.get_text("text").splitlines() if line.strip()]
            if text_lines:
                headers.append(text_lines[0])
                footers.append(text_lines[-1])
        header_counts = Counter(headers)
        footer_counts = Counter(footers)
        header_repeats = {item for item, count in header_counts.items() if count >= 3}
        footer_repeats = {item for item, count in footer_counts.items() if count >= 3}
        return header_repeats, footer_repeats

    def _strip_watermark_lines(self, text: str, headers: set[str], footers: set[str]) -> str:
        lines = [line.strip() for line in text.splitlines()]
        kept: List[str] = []
        for line in lines:
            if not line:
                kept.append("")
                continue
            if line in headers or line in footers:
                continue
            if LOGICAL_PAGE_PATTERN.fullmatch(line):
                continue
            kept.append(line)
        return "\n".join(kept)

    def _looks_like_table_page(self, text: str) -> bool:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        structured_lines = sum(
            1
            for line in lines
            if len(re.findall(r"\s{2,}|\t|[:：]", line)) >= 1 or len(re.findall(r"\d", line)) >= 3
        )
        return structured_lines >= max(3, len(lines) // 4)

    def _extract_tables(self, plumber_page) -> str:
        markdown_tables: List[str] = []
        tables = plumber_page.extract_tables()
        for table in tables:
            if not table or len(table) < 2:
                continue
            rows = [["" if cell is None else str(cell).strip() for cell in row] for row in table]
            header = rows[0]
            markdown_tables.append(
                "\n".join(
                    [
                        "| " + " | ".join(header) + " |",
                        "| " + " | ".join(["---"] * len(header)) + " |",
                        *["| " + " | ".join(row) + " |" for row in rows[1:]],
                    ]
                )
            )
        return "\n\n".join(markdown_tables[:5])

    def _extract_handwriting(self, page: fitz.Page) -> str:
        ocr = self._get_ocr()
        if ocr is None:
            return ""
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        try:
            result = ocr.ocr(pix.tobytes("png"), cls=True)
        except TypeError:
            result = ocr.ocr(pix.tobytes("png"))
        lines: List[str] = []
        for line_group in result or []:
            for line in line_group:
                if not line or len(line) < 2:
                    continue
                text = str(line[1][0]).strip()
                if text:
                    lines.append(text)
        return "\n".join(lines)

    def _guess_section_title(self, text: str) -> str:
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            if len(candidate) <= 40 and re.search(r"(第.+[章节])|^[一二三四五六七八九十]+、", candidate):
                return candidate
        return ""

    def _post_process_pages(self, pages: List[Dict[str, object]], source: str) -> List[Dict[str, object]]:
        processed: List[Dict[str, object]] = []
        for page in pages:
            text = normalize_whitespace(str(page.get("text") or ""))
            tables_markdown = normalize_whitespace(str(page.get("tables_markdown") or ""))
            handwriting = normalize_whitespace(str(page.get("handwriting") or ""))

            if settings.enable_redaction:
                redacted_text, redaction_stats = redact_sensitive_text(text)
            else:
                redacted_text, redaction_stats = text, {}

            section_title = str(page.get("section_title") or self._guess_section_title(text))
            page_type = str(page.get("page_type") or ("table" if tables_markdown else "text"))
            processed.append(
                {
                    "page_number": int(page["page_number"]),
                    "logical_page": page.get("logical_page"),
                    "text": text,
                    "redacted_text": redacted_text,
                    "raw_text": str(page.get("raw_text") or text),
                    "tables_markdown": tables_markdown,
                    "handwriting": handwriting,
                    "redaction_stats": redaction_stats if redaction_stats else dict(page.get("redaction_stats") or {}),
                    "page_type": page_type,
                    "section_title": section_title,
                    "source": source,
                    "parse_metadata": dict(page.get("parse_metadata") or {}),
                }
            )
        return processed

    def _parse_builtin(self, pdf_path: Path) -> List[Dict[str, object]]:
        doc = fitz.open(str(pdf_path))
        header_pages = [doc.load_page(i) for i in range(min(20, doc.page_count))]
        header_candidates, footer_candidates = self._header_footer_candidates(header_pages)
        pages: List[Dict[str, object]] = []
        ocr_pages_used = 0
        allow_aggressive_ocr = doc.page_count <= settings.large_pdf_page_threshold
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pdfplumber.open(str(pdf_path)) as pdf:
                for i in range(doc.page_count):
                    page = doc.load_page(i)
                    raw_text = page.get_text("text")
                    cleaned_text = self._strip_watermark_lines(raw_text, header_candidates, footer_candidates)
                    tables = ""
                    if self._looks_like_table_page(cleaned_text):
                        try:
                            tables = self._extract_tables(pdf.pages[i])
                        except Exception:
                            tables = ""
                    handwriting = ""
                    should_ocr = len(cleaned_text.strip()) < 40 and (
                        allow_aggressive_ocr or ocr_pages_used < settings.max_ocr_pages_per_pdf
                    )
                    if should_ocr:
                        handwriting = self._extract_handwriting(page)
                        ocr_pages_used += 1
                    logical_page_match = LOGICAL_PAGE_PATTERN.search(raw_text)
                    logical_page = logical_page_match.group(1) if logical_page_match else None
                    merged_text = normalize_whitespace("\n\n".join(part for part in [cleaned_text, handwriting] if part))
                    pages.append(
                        {
                            "page_number": i + 1,
                            "logical_page": logical_page,
                            "text": merged_text,
                            "raw_text": raw_text,
                            "tables_markdown": tables,
                            "handwriting": handwriting,
                            "page_type": "table" if tables else ("ocr" if handwriting else "text"),
                            "section_title": self._guess_section_title(merged_text),
                        }
                    )
        return self._post_process_pages(pages, source="builtin")

    def parse(self, pdf_path: Path) -> List[Dict[str, object]]:
        if self.external_parser.is_enabled():
            try:
                pages = self.external_parser.parse(pdf_path)
                return self._post_process_pages(pages, source="parse2")
            except Exception:
                if settings.pdf_parser_backend.lower() == "parse2":
                    raise
        return self._parse_builtin(pdf_path)
