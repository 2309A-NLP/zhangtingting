from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import logging
import re
import warnings
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import fitz
import pdfplumber

from backend.config import settings
from backend.services.external_pdf_parser import ExternalPDFParser
from backend.services.legacy_pdf_pipeline import load_resolved_pages_as_parser_output, run_legacy_pdf_pipeline
from backend.services.pdf_intelligence import PDFIntelligencePipeline
from backend.services.redaction import redact_sensitive_text
from backend.services.text_utils import dedupe_preserve_order, derive_content_tags, normalize_whitespace

try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover
    PaddleOCR = None


LOGICAL_PAGE_PATTERN = re.compile(r"1-1-(\d+)")
FORM_FIELD_HINTS = [
    "法定代表人",
    "注册资本",
    "实收资本",
    "成立日期",
    "统一社会信用代码",
    "公司名称",
    "英文名称",
    "注册地址",
    "经营范围",
]
ORG_CHART_HINTS = ["组织结构图", "组织架构图", "组织机构图", "部门构成", "下设部门", "销售处"]
CHART_HINTS = ["增长图", "增长率", "应用结构", "市场规模", "图中可以看出", "市场应用结构"]
FINANCIAL_TABLE_HINTS = ["募集资金", "金额", "万元", "占比", "比例", "营业收入", "净利润", "股本", "发行股数"]

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class PDFParser:
    def __init__(self, ocr_lang: str = "ch", intelligence_output_root: Path | None = None) -> None:
        self.ocr_lang = ocr_lang
        self._ocr = None
        self.external_parser = ExternalPDFParser()
        self.intelligence_pipeline = (
            PDFIntelligencePipeline(output_root=intelligence_output_root) if settings.pdf_intelligence_enabled else None
        )

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
            if len(candidate) <= 40 and re.search(r"(第.+[章节])|^[一二三四五六七八九十]+\s*[、.]", candidate):
                return candidate
        return ""

    def _looks_like_multi_column_page(self, page: fitz.Page) -> bool:
        width = float(page.rect.width or 0)
        if width <= 0:
            return False
        left_hits = 0
        right_hits = 0
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, _, _, _, text = block[:5]
            if not str(text).strip():
                continue
            if x0 < width * 0.45:
                left_hits += 1
            elif x0 > width * 0.55:
                right_hits += 1
        return left_hits >= 2 and right_hits >= 2

    def _looks_like_key_value_form(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 3:
            return False
        hint_hits = sum(1 for line in lines if any(hint in line for hint in FORM_FIELD_HINTS))
        colon_hits = sum(1 for line in lines[:12] if re.search(r"[:：]", line))
        return hint_hits >= 2 or (hint_hits >= 1 and colon_hits >= 2) or colon_hits >= 4

    def _looks_like_org_chart_page(self, text: str, image_count: int) -> bool:
        return image_count >= 1 and any(keyword in text for keyword in ORG_CHART_HINTS)

    def _looks_like_chart_page(self, text: str, image_count: int) -> bool:
        return image_count >= 1 and any(keyword in text for keyword in CHART_HINTS)

    def _looks_like_financial_table(self, text: str, tables_markdown: str) -> bool:
        payload = f"{text}\n{tables_markdown}"
        return sum(1 for hint in FINANCIAL_TABLE_HINTS if hint in payload) >= 2

    def _classify_page(
        self,
        *,
        page: fitz.Page,
        text: str,
        tables_markdown: str,
        handwriting: str,
        section_title: str,
        parse_metadata: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        parse_metadata = dict(parse_metadata or {})
        image_count = int(parse_metadata.get("image_count") or len(page.get_images(full=True)))
        multi_column = self._looks_like_multi_column_page(page)
        key_value_like = self._looks_like_key_value_form(text)
        org_chart_like = self._looks_like_org_chart_page(text, image_count)
        chart_like = self._looks_like_chart_page(text, image_count)
        financial_table_like = self._looks_like_financial_table(text, tables_markdown)

        page_type = "table" if tables_markdown else ("ocr" if handwriting else "text")
        primary_type = "text"
        sub_type = "paragraph"
        type_confidence = 0.72
        candidate_types = ["text", "mixed"]
        layout_tags: List[str] = []

        if tables_markdown:
            page_type = "table"
            primary_type = "table"
            sub_type = "financial_table" if financial_table_like else ("key_value_table" if key_value_like else "simple_table")
            type_confidence = 0.88 if financial_table_like else 0.84
            candidate_types = ["table", "form"] if key_value_like else ["table", "mixed"]
            layout_tags.append("table_present")
        elif org_chart_like:
            page_type = "figure"
            primary_type = "figure"
            sub_type = "org_chart"
            type_confidence = 0.83
            candidate_types = ["figure", "mixed", "text"]
            layout_tags.append("visual_page")
        elif chart_like:
            page_type = "figure"
            primary_type = "figure"
            sub_type = "chart"
            type_confidence = 0.82
            candidate_types = ["figure", "mixed", "text"]
            layout_tags.append("visual_page")
        elif key_value_like:
            primary_type = "form"
            sub_type = "basic_info_form"
            type_confidence = 0.79
            candidate_types = ["form", "text", "mixed"]
        elif handwriting and len(text.strip()) < 120:
            page_type = "ocr"
            primary_type = "text"
            sub_type = "ocr_text"
            type_confidence = 0.68
            candidate_types = ["text", "mixed"]
        elif multi_column:
            primary_type = "text"
            sub_type = "multi_column_text"
            type_confidence = 0.77
            candidate_types = ["text", "mixed"]

        if handwriting:
            layout_tags.append("ocr_used")
        if multi_column:
            layout_tags.append("multi_column")
        if image_count >= settings.pdf_vlm_image_trigger_count:
            layout_tags.append("image_heavy")
        if section_title:
            layout_tags.append("sectioned_page")

        content_tags = derive_content_tags(section_title, "", text)
        if key_value_like:
            content_tags.append("field_lookup")
        if financial_table_like:
            content_tags.append("table_numeric")
        if org_chart_like:
            content_tags.append("organization_structure")
        if chart_like:
            content_tags.append("chart_analysis")

        return {
            "page_type": page_type,
            "primary_type": primary_type,
            "sub_type": sub_type,
            "type_confidence": type_confidence,
            "candidate_types": dedupe_preserve_order(candidate_types),
            "layout_tags": dedupe_preserve_order(layout_tags),
            "content_tags": dedupe_preserve_order(content_tags),
            "parse_metadata": {
                **parse_metadata,
                "image_count": image_count,
                "multi_column": multi_column,
                "key_value_like": key_value_like,
                "org_chart_like": org_chart_like,
                "chart_like": chart_like,
                "financial_table_like": financial_table_like,
                "has_table": bool(tables_markdown),
                "used_ocr": bool(handwriting),
            },
        }

    def _post_process_pages(self, pages: List[Dict[str, object]], source: str, pdf_path: Path) -> List[Dict[str, object]]:
        processed: List[Dict[str, object]] = []
        for page in pages:
            text = normalize_whitespace(str(page.get("text") or ""))
            tables_markdown = normalize_whitespace(str(page.get("tables_markdown") or ""))
            handwriting = normalize_whitespace(str(page.get("handwriting") or ""))
            unified_text = normalize_whitespace("\n\n".join(part for part in [text, tables_markdown, handwriting] if part))

            if settings.enable_redaction:
                redacted_text, redaction_stats = redact_sensitive_text(unified_text)
            else:
                redacted_text, redaction_stats = unified_text, {}

            section_title = str(page.get("section_title") or self._guess_section_title(unified_text))
            page_type = str(page.get("page_type") or ("table" if tables_markdown else "text"))
            content_tags = list(page.get("content_tags") or derive_content_tags(section_title, "", unified_text))
            processed.append(
                {
                    "page_number": int(page["page_number"]),
                    "logical_page": page.get("logical_page"),
                    "text": unified_text,
                    "redacted_text": redacted_text,
                    "raw_text": str(page.get("raw_text") or unified_text),
                    "tables_markdown": tables_markdown,
                    "handwriting": handwriting,
                    "redaction_stats": redaction_stats if redaction_stats else dict(page.get("redaction_stats") or {}),
                    "page_type": page_type,
                    "primary_type": str(page.get("primary_type") or "text"),
                    "sub_type": str(page.get("sub_type") or "paragraph"),
                    "type_confidence": float(page.get("type_confidence") or 0.72),
                    "candidate_types": list(page.get("candidate_types") or ["text"]),
                    "layout_tags": list(page.get("layout_tags") or []),
                    "content_tags": dedupe_preserve_order(content_tags),
                    "section_title": section_title,
                    "source": source,
                    "source_pdf": pdf_path.name,
                    "source_pdf_path": str(pdf_path),
                    "parse_metadata": dict(page.get("parse_metadata") or {}),
                }
            )
        return processed

    def _infer_page_from_element_id(self, element_id: str) -> int | None:
        match = re.search(r"_(\d{3})", element_id or "")
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _apply_pdf_intelligence(
        self,
        *,
        pdf_path: Path,
        pages: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        if not self.intelligence_pipeline or not pages:
            return pages

        started = time.perf_counter()
        logger.info("[pdf-parser] intelligence start pdf=%s pages=%s", pdf_path.name, len(pages))
        intelligence_payload = self.intelligence_pipeline.run(
            pdf_path=pdf_path,
            base_pages=pages,
            force=False,
        )
        logger.info(
            "[pdf-parser] intelligence done pdf=%s pages=%s elapsed_ms=%s",
            pdf_path.name,
            len(pages),
            int((time.perf_counter() - started) * 1000),
        )
        stage0_map = {
            int(item.get("page_index") or 0): item
            for item in list(intelligence_payload.get("stage0") or [])
            if int(item.get("page_index") or 0) > 0
        }
        stage1_map = {
            int(item.get("page_index") or 0): item
            for item in list(intelligence_payload.get("stage1") or [])
            if int(item.get("page_index") or 0) > 0
        }

        context_by_page: Dict[int, List[Dict[str, object]]] = {}
        for link in list(intelligence_payload.get("stage3") or []):
            page_index = self._infer_page_from_element_id(str(link.get("element_id") or ""))
            if not page_index:
                continue
            context_by_page.setdefault(page_index, []).append(link)

        facts_by_page: Dict[int, List[Dict[str, object]]] = {}
        for fact in list(intelligence_payload.get("stage4") or []):
            page_index = int(fact.get("page_number") or 0)
            if page_index <= 0:
                continue
            facts_by_page.setdefault(page_index, []).append(fact)

        enriched_pages: List[Dict[str, object]] = []
        for page in pages:
            page_number = int(page["page_number"])
            stage0 = stage0_map.get(page_number, {})
            stage1 = stage1_map.get(page_number, {})
            structured_facts = list(facts_by_page.get(page_number, []))
            context_links = list(context_by_page.get(page_number, []))
            parse_metadata = dict(page.get("parse_metadata") or {})
            parse_metadata.update(
                {
                    "pdf_intelligence_enabled": True,
                    "filter_region_count": len(list(stage0.get("filter_regions") or [])),
                    "layout_element_count": len(list(stage1.get("elements") or [])),
                    "context_link_count": len(context_links),
                    "structured_fact_count": len(structured_facts),
                    "stage0_page_type": str(stage0.get("page_type") or ""),
                    "ocr_required": bool(stage0.get("ocr_required") or False),
                    "ocr_decision_reason": str(stage0.get("ocr_decision_reason") or ""),
                    "text_span_count": int(stage0.get("text_span_count") or 0),
                    "text_line_count": int(stage0.get("text_line_count") or 0),
                    "text_block_count": int(stage0.get("text_block_count") or 0),
                    "text_char_count": int(stage0.get("text_char_count") or 0),
                    "image_coverage_ratio": float(stage0.get("image_coverage_ratio") or 0.0),
                    "watermark_score": float(stage0.get("watermark_score") or 0.0),
                }
            )
            layout_tags = dedupe_preserve_order(
                [
                    *list(page.get("layout_tags") or []),
                    "pdf_intelligence",
                    "filtered_regions" if stage0.get("filter_regions") else "",
                    "layout_analyzed" if stage1.get("elements") else "",
                    "structured_facts" if structured_facts else "",
                ]
            )
            content_tags = dedupe_preserve_order(
                [
                    *list(page.get("content_tags") or []),
                    "table_structured" if any(str(item.get("fact_type") or "").startswith("table") for item in structured_facts) else "",
                    "figure_structured" if any(item.get("primary_type") == "figure" for item in structured_facts) else "",
                ]
            )
            enriched_pages.append(
                {
                    **page,
                    "parse_metadata": parse_metadata,
                    "layout_tags": layout_tags,
                    "content_tags": content_tags,
                    "filter_regions": list(stage0.get("filter_regions") or []),
                    "layout_elements": list(stage1.get("elements") or []),
                    "context_links": context_links,
                    "structured_facts": structured_facts,
                    "pdf_intelligence": {
                        "stage0": stage0,
                        "stage1": stage1,
                        "context_links": context_links,
                        "structured_facts": structured_facts,
                    },
                }
            )
        return enriched_pages

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
                    section_title = self._guess_section_title(merged_text)
                    classification = self._classify_page(
                        page=page,
                        text=merged_text,
                        tables_markdown=tables,
                        handwriting=handwriting,
                        section_title=section_title,
                        parse_metadata={"image_count": len(page.get_images(full=True))},
                    )
                    pages.append(
                        {
                            "page_number": i + 1,
                            "logical_page": logical_page,
                            "text": merged_text,
                            "raw_text": raw_text,
                            "tables_markdown": tables,
                            "handwriting": handwriting,
                            "page_type": classification["page_type"],
                            "primary_type": classification["primary_type"],
                            "sub_type": classification["sub_type"],
                            "type_confidence": classification["type_confidence"],
                            "candidate_types": classification["candidate_types"],
                            "layout_tags": classification["layout_tags"],
                            "content_tags": classification["content_tags"],
                            "section_title": section_title,
                            "parse_metadata": classification["parse_metadata"],
                        }
                    )
        return self._post_process_pages(pages, source="builtin", pdf_path=pdf_path)

    def parse(self, pdf_path: Path) -> List[Dict[str, object]]:
        started = time.perf_counter()
        logger.info(
            "[pdf-parser] parse start pdf=%s backend=%s intelligence=%s",
            pdf_path.name,
            settings.pdf_parser_backend,
            bool(self.intelligence_pipeline),
        )
        if self.external_parser.is_enabled():
            try:
                external_started = time.perf_counter()
                logger.info("[pdf-parser] external parse start pdf=%s", pdf_path.name)
                pages = self.external_parser.parse(pdf_path)
                logger.info(
                    "[pdf-parser] external parse done pdf=%s pages=%s elapsed_ms=%s",
                    pdf_path.name,
                    len(pages),
                    int((time.perf_counter() - external_started) * 1000),
                )
                processed = self._post_process_pages(pages, source="parse2", pdf_path=pdf_path)
                result = self._apply_pdf_intelligence(pdf_path=pdf_path, pages=processed)
                logger.info(
                    "[pdf-parser] parse done pdf=%s pages=%s source=parse2 elapsed_ms=%s",
                    pdf_path.name,
                    len(result),
                    int((time.perf_counter() - started) * 1000),
                )
                return result
            except Exception:
                logger.exception("[pdf-parser] external parse failed pdf=%s", pdf_path.name)
                if settings.pdf_parser_backend.lower() == "parse2":
                    raise
        logger.info("[pdf-parser] builtin parse start pdf=%s", pdf_path.name)
        builtin_started = time.perf_counter()
        processed = self._parse_builtin(pdf_path)
        logger.info(
            "[pdf-parser] builtin parse done pdf=%s pages=%s elapsed_ms=%s",
            pdf_path.name,
            len(processed),
            int((time.perf_counter() - builtin_started) * 1000),
        )
        result = self._apply_pdf_intelligence(pdf_path=pdf_path, pages=processed)
        logger.info(
            "[pdf-parser] parse done pdf=%s pages=%s source=builtin elapsed_ms=%s",
            pdf_path.name,
            len(result),
            int((time.perf_counter() - started) * 1000),
        )
        return result
