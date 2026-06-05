# 工单编号：人工智能 NLP-RAG-PDF 文档的表格解析及检索优化
from __future__ import annotations

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
        if len(re.findall(r"\s{2,}|\t|[:：|]", line)) >= 1 or len(re.findall(r"\d", line)) >= 3
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
        if len(line) <= 40 and re.search(r"[一二三四五六七八九十]|第.+节|第.+章", line):
            return line
    return ""


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
                merged = normalize_whitespace("\n\n".join(part for part in [cleaned_text, table_markdown, ocr_text] if part))
                pages.append(
                    {
                        "page_number": i + 1,
                        "logical_page": logical_page,
                        "text": normalize_whitespace("\n\n".join(part for part in [cleaned_text, ocr_text] if part)),
                        "redacted_text": merged,
                        "raw_text": raw_text,
                        "tables_markdown": table_markdown,
                        "handwriting": ocr_text,
                        "redaction_stats": {},
                        "page_type": "table" if table_markdown else ("ocr" if ocr_text else "text"),
                        "section_title": guess_section_title(merged),
                        "source": "parse2",
                        "parse_metadata": {
                            "image_count": image_count,
                            "used_rapidocr": bool(ocr_text),
                            "has_table": bool(table_markdown),
                        },
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
