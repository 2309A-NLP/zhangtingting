from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

from app.config import settings


PARSE2_BRIDGE = r"""
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import fitz
import pdfplumber

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
except Exception:
    pass

try:
    import camelot
except Exception:
    camelot = None

try:
    from rapidocr import RapidOCR
except Exception:
    RapidOCR = None


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


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def header_footer_candidates(pages):
    headers = []
    footers = []
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


def strip_watermark_lines(text, headers, footers):
    lines = [line.strip() for line in text.splitlines()]
    kept = []
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


def looks_like_table_page(text):
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    structured_lines = sum(
        1
        for line in lines
        if len(re.findall(r"\s{2,}|\t|[:：]", line)) >= 1 or len(re.findall(r"\d", line)) >= 3
    )
    return structured_lines >= max(3, len(lines) // 4)


def extract_tables_pdfplumber(plumber_page):
    markdown_tables = []
    for table in plumber_page.extract_tables() or []:
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


def extract_tables_camelot(pdf_path, page_number):
    if camelot is None:
        return ""
    markdown_tables = []
    for flavor in ["stream", "lattice"]:
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=str(page_number), flavor=flavor)
        except Exception:
            continue
        for table in tables:
            df = table.df.fillna("")
            if df.empty or df.shape[0] < 2:
                continue
            rows = [[str(cell).strip() for cell in row] for row in df.values.tolist()]
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
        if markdown_tables:
            break
    return "\n\n".join(markdown_tables[:5])


def guess_section_title(text):
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= 40 and re.search(r"(第.+[章节])|^[一二三四五六七八九十]+\s*[、.]", line):
            return line
    return ""


def dedupe(items):
    result = []
    seen = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def derive_content_tags(section_title, text):
    payload = f"{section_title}\n{text}"
    rules = {
        "fundraising": ["募集资金", "募投项目", "补充流动资金"],
        "revenue": ["营业收入", "主营业务收入", "收入构成"],
        "military_revenue": ["军用领域收入", "军用收入", "国防客户"],
        "shareholding": ["股本", "持股", "发行股数", "总股本"],
        "legal_representative": ["法定代表人"],
        "organization_structure": ["组织结构", "销售部", "销售处", "下设部门"],
        "chart_analysis": ["增长率", "增长图", "应用结构", "负增长"],
        "supplier": ["供应商", "上游", "下游"],
    }
    return dedupe([tag for tag, keywords in rules.items() if any(keyword in payload for keyword in keywords)])


def looks_like_multi_column_page(page):
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


def looks_like_key_value_form(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    hint_hits = sum(1 for line in lines if any(hint in line for hint in FORM_FIELD_HINTS))
    colon_hits = sum(1 for line in lines[:12] if re.search(r"[:：]", line))
    return hint_hits >= 2 or (hint_hits >= 1 and colon_hits >= 2) or colon_hits >= 4


def looks_like_org_chart_page(text, image_count):
    return image_count >= 1 and any(keyword in text for keyword in ORG_CHART_HINTS)


def looks_like_chart_page(text, image_count):
    return image_count >= 1 and any(keyword in text for keyword in CHART_HINTS)


def looks_like_financial_table(text, table_markdown):
    payload = f"{text}\n{table_markdown}"
    return sum(1 for hint in FINANCIAL_TABLE_HINTS if hint in payload) >= 2


def classify_page(page, text, table_markdown, ocr_text, section_title, image_count):
    multi_column = looks_like_multi_column_page(page)
    key_value_like = looks_like_key_value_form(text)
    org_chart_like = looks_like_org_chart_page(text, image_count)
    chart_like = looks_like_chart_page(text, image_count)
    financial_table_like = looks_like_financial_table(text, table_markdown)

    page_type = "table" if table_markdown else ("ocr" if ocr_text else "text")
    primary_type = "text"
    sub_type = "paragraph"
    type_confidence = 0.72
    candidate_types = ["text", "mixed"]
    layout_tags = []

    if table_markdown:
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
    elif ocr_text and len(text.strip()) < 120:
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

    if ocr_text:
        layout_tags.append("ocr_used")
    if multi_column:
        layout_tags.append("multi_column")
    if image_count >= 6:
        layout_tags.append("image_heavy")
    if section_title:
        layout_tags.append("sectioned_page")

    content_tags = derive_content_tags(section_title, text)
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
        "candidate_types": dedupe(candidate_types),
        "layout_tags": dedupe(layout_tags),
        "content_tags": dedupe(content_tags),
        "parse_metadata": {
            "image_count": image_count,
            "used_rapidocr": bool(ocr_text),
            "has_table": bool(table_markdown),
            "multi_column": multi_column,
            "key_value_like": key_value_like,
            "org_chart_like": org_chart_like,
            "chart_like": chart_like,
            "financial_table_like": financial_table_like,
        },
    }


def should_use_ocr(text, image_count):
    stripped = text.strip()
    if len(stripped) < 30:
        return True
    return len(stripped) < 120 and image_count >= 8


def build_rapidocr():
    if RapidOCR is None:
        return None
    try:
        return RapidOCR()
    except Exception:
        return None


def run_rapidocr(reader, page):
    if reader is None:
        return ""
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    try:
        result = reader(pix.tobytes("png"))
    except Exception:
        return ""
    lines = []
    payload = result[0] if isinstance(result, tuple) else result
    if payload is None:
        return ""
    txts = getattr(payload, "txts", None)
    if txts:
        for text in txts:
            value = str(text).strip()
            if value:
                lines.append(value)
    elif getattr(payload, "word_results", None):
        for item in payload.word_results or []:
            if not item:
                continue
            value = str(item[0]).strip()
            if value:
                lines.append(value)
    elif isinstance(payload, list):
        for item in payload:
            if not item:
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                maybe_text = item[1]
                if isinstance(maybe_text, (list, tuple)) and maybe_text:
                    value = str(maybe_text[0]).strip()
                    if value:
                        lines.append(value)
    return "\n".join(lines)


def parse(pdf_path):
    doc = fitz.open(str(pdf_path))
    preview_pages = [doc.load_page(i) for i in range(min(20, doc.page_count))]
    headers, footers = header_footer_candidates(preview_pages)
    reader = build_rapidocr()
    pages = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i in range(doc.page_count):
                page = doc.load_page(i)
                raw_text = page.get_text("text")
                cleaned_text = strip_watermark_lines(raw_text, headers, footers)

                table_markdown = ""
                if looks_like_table_page(cleaned_text):
                    table_markdown = extract_tables_camelot(pdf_path, i + 1)
                    if not table_markdown:
                        try:
                            table_markdown = extract_tables_pdfplumber(pdf.pages[i])
                        except Exception:
                            table_markdown = ""

                image_count = len(page.get_images(full=True))
                ocr_text = ""
                if should_use_ocr(cleaned_text, image_count):
                    ocr_text = run_rapidocr(reader, page)

                logical_page_match = LOGICAL_PAGE_PATTERN.search(raw_text)
                logical_page = logical_page_match.group(1) if logical_page_match else None
                page_text = normalize_whitespace("\n\n".join(part for part in [cleaned_text, ocr_text] if part))
                unified_text = normalize_whitespace("\n\n".join(part for part in [cleaned_text, table_markdown, ocr_text] if part))
                section_title = guess_section_title(unified_text)
                classification = classify_page(
                    page,
                    unified_text,
                    table_markdown,
                    ocr_text,
                    section_title,
                    image_count,
                )
                pages.append(
                    {
                        "page_number": i + 1,
                        "logical_page": logical_page,
                        "text": page_text,
                        "raw_text": raw_text,
                        "tables_markdown": table_markdown,
                        "handwriting": ocr_text,
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
    return pages


if __name__ == "__main__":
    pdf_path = Path(sys.argv[1])
    pages = parse(pdf_path)
    print(json.dumps(pages, ensure_ascii=False))
"""


class ExternalPDFParser:
    def __init__(self) -> None:
        self.backend = settings.pdf_parser_backend.lower()

    def is_enabled(self) -> bool:
        return self.backend in {"parse2", "auto"}

    def parse(self, pdf_path: Path) -> List[Dict[str, object]]:
        python_cmd = self._resolve_python()
        if not python_cmd:
            raise RuntimeError("parse2 parser backend is enabled but no external Python interpreter was configured.")

        with tempfile.NamedTemporaryFile("w", suffix="_parse2_bridge.py", delete=False, encoding="utf-8") as handle:
            handle.write(PARSE2_BRIDGE)
            script_path = Path(handle.name)

        try:
            completed = subprocess.run(
                [python_cmd, str(script_path), str(pdf_path)],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=settings.pdf_parser_timeout,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            payload = completed.stdout.strip()
            if not payload:
                raise RuntimeError("parse2 parser returned empty output.")
            return json.loads(payload)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"parse2 parser timed out after {settings.pdf_parser_timeout} seconds.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            raise RuntimeError(f"parse2 parser failed: {stderr or exc}") from exc
        finally:
            script_path.unlink(missing_ok=True)

    def _resolve_python(self) -> str:
        if settings.pdf_parser_python:
            return settings.pdf_parser_python

        if settings.pdf_parser_conda_env:
            conda_prefix = os.environ.get("CONDA_PREFIX", "")
            if conda_prefix and Path(conda_prefix).name == settings.pdf_parser_conda_env:
                return sys.executable

        return ""
