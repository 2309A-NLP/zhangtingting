# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "pdf_intelligence_config.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "stage1_smoke_test"
DEFAULT_PAGE_RANGE: list[int] = []

TABLE_ROW_HEIGHT_MIN = 15
TABLE_ROW_HEIGHT_MAX = 50
TABLE_HLINE_MIN_RATIO = 0.3
TABLE_MIN_ROWS = 3
CROSS_PAGE_GAP_BOTTOM = 48.0
CROSS_PAGE_GAP_TOP = 120.0
CROSS_PAGE_MERGE_HIGH = 0.8
CROSS_PAGE_MERGE_LOW = 0.6
FIGURE_CLUSTER_GAP = 72.0
FIGURE_CLUSTER_OVERLAP = 0.12

FLOWCHART_KEYWORDS = ["\u6d41\u7a0b\u56fe", "\u7ec4\u7ec7\u67b6\u6784", "\u601d\u7ef4\u5bfc\u56fe", "\u7ed3\u6784\u56fe", "\u997c\u56fe", "\u67f1\u56fe", "\u6298\u56fe"]
DATAVIZ_KEYWORDS = ["\u8d8b\u52bf\u56fe", "\u67f1\u56fe", "\u6298\u56fe", "\u997c\u56fe", "\u6563\u70b9\u56fe", "\u70ed\u529b\u56fe", "\u9762\u79ef\u56fe", "\u96f7\u8fbe\u56fe", "\u6f0f\u6597\u56fe", "\u4eea\u8868\u76d8"]
TABLE_END_WORDS = ["\u6ce8", "\u91ca", "\u6765\u6e90", "\u9644\u6ce8", "\u5907\u6ce8"]
TABLE_HEADER_WORDS = ["\u9879\u76ee", "\u540d\u79f0", "\u91d1\u989d", "\u79d1\u76ee", "\u79d1\u76ee", "\u671f\u95f4"]
CONTINUATION_WORDS = ["\u7ee7", "\u63a5\u4e0a\u9875", "\u4e0b\u9875\u7ee7"]
UNIT_PREFIXES = ["\u5355\u4f4d\uff1a", "\u5355\u4f4d:", "\u91d1\u989d\uff1a"]

DATE_PATTERN = re.compile(r"\d{4}[-/\u5e74]\d{1,2}[-/\u6708]?\d{0,2}")
NUMBER_PATTERN = re.compile(r"^-?\d+(?:,\d{3})*(?:\.\d+)?$")
PERCENTAGE_PATTERN = re.compile(r"\d+\.\d+%|\d+%")
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
KEY_VALUE_PATTERN = re.compile(r"[:\uff1a]")


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[stage1 {timestamp}] {message}", flush=True)


def safe_pattern(value: str) -> str:
    try:
        return value.encode("latin1").decode("utf-8")
    except Exception:
        return value


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_pdf_path(config: dict[str, Any]) -> Path:
    configured_pdf = str(config.get("stage1_smoke_test_pdf") or "").strip()
    if configured_pdf:
        return Path(configured_pdf).expanduser()
    pdfs = sorted(PROJECT_ROOT.joinpath("data").glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError("No PDF found under data directory.")
    return pdfs[0]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_watermark_patterns() -> list[str]:
    config_path = PROJECT_ROOT / "config" / "watermark_patterns.json"
    if not config_path.exists():
        return []
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [safe_pattern(str(item).strip()) for item in payload if str(item).strip()]


def extract_text_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    blocks = []
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        if not str(text).strip():
            continue
        blocks.append({
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
            "text": normalize_whitespace(str(text)),
        })
    return blocks


def extract_text_lines(page: fitz.Page, filter_regions: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    from backend.pipeline.stages.stage1_layout_analysis._bbox import line_text_and_bbox, is_filtered_line
    lines: list[dict[str, Any]] = []
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks") or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or []:
            text, bbox = line_text_and_bbox(block, line)
            if not text:
                continue
            if filter_regions and is_filtered_line(bbox, filter_regions):
                continue
            lines.append({
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "text": text,
                "x": float(bbox[0]),
                "y": float(bbox[1]),
            })
    return lines
