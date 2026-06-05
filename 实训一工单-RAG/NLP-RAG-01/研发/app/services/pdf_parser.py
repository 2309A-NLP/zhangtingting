# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统

from __future__ import annotations

import re
import warnings
from collections import Counter
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import fitz
import pdfplumber

from app.config import settings
from app.services.redaction import redact_sensitive_text
from app.services.text_utils import normalize_whitespace

try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - optional runtime path
    PaddleOCR = None


LOGICAL_PAGE_PATTERN = re.compile(r"1-1-(\d+)")

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)


class PDFParser:
    def __init__(self, ocr_lang: str = "ch") -> None:
        self.ocr_lang = ocr_lang
        self._ocr = None

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
        return "\n\n".join(markdown_tables[:3])

    def _extract_handwriting(self, page: fitz.Page) -> str:
        ocr = self._get_ocr()
        if ocr is None:
            return ""
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image_path = page.parent.name  # pragma: no cover - metadata only
        _ = image_path
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

    def parse(self, pdf_path: Path) -> List[Dict[str, object]]:
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
                    merged = normalize_whitespace("\n\n".join(part for part in [cleaned_text, tables, handwriting] if part))
                    redacted, redaction_stats = redact_sensitive_text(merged)
                    pages.append(
                        {
                            "page_number": i + 1,
                            "logical_page": logical_page,
                            "text": merged,
                            "redacted_text": redacted,
                            "raw_text": raw_text,
                            "tables_markdown": tables,
                            "handwriting": handwriting,
                            "redaction_stats": redaction_stats,
                        }
                    )
        return pages
