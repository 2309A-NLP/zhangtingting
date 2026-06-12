from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化

import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


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

FLOWCHART_KEYWORDS = ["组织结构", "组织架构", "组织机构", "部门构成", "股权结构", "子公司", "分公司"]
DATAVIZ_KEYWORDS = ["增长率", "趋势", "同比", "环比", "柱状图", "折线图", "饼图", "图表", "市场规模", "市场份额"]
TABLE_END_WORDS = ["合计", "总计", "小计", "合计数", "总计数"]
TABLE_HEADER_WORDS = ["序号", "编号", "项目", "名称", "金额", "比例", "占比"]
CONTINUATION_WORDS = ["续表", "续上表", "下表续"]
UNIT_PREFIXES = ["单位：", "单位:", "金额单位"]

DATE_PATTERN = re.compile(r"\d{4}[-/年]\d{1,2}[-/月]?\d{0,2}")
NUMBER_PATTERN = re.compile(r"^-?\d+(?:,\d{3})*(?:\.\d+)?$")
PERCENTAGE_PATTERN = re.compile(r"\d+\.\d+%|\d+%")
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
KEY_VALUE_PATTERN = re.compile(r"[:：]")


@dataclass
class Region:
    region_id: str
    page_index: int
    region_type: str
    sub_type: str
    bbox: list[float]
    text: str = ""
    confidence: float = 0.0
    source: str = "pymupdf"
    col_count: int = 0
    row_count: int = 0
    header_names: list[str] = field(default_factory=list)
    xref: int = 0
    is_cross_page: bool = False
    merged_from: list[str] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "page_index": self.page_index,
            "region_type": self.region_type,
            "sub_type": self.sub_type,
            "bbox": self.bbox,
            "text": self.text[:200],
            "confidence": self.confidence,
            "source": self.source,
            "col_count": self.col_count,
            "row_count": self.row_count,
            "header_names": self.header_names,
            "xref": self.xref,
            "is_cross_page": self.is_cross_page,
            "merged_from": self.merged_from,
            "page_start": self.page_start or self.page_index,
            "page_end": self.page_end or self.page_index,
        }


@dataclass
class LayoutPage:
    page_index: int
    page_bbox: list[float]
    page_type: str = "text"
    sub_type: str = "paragraph"
    regions: list[Region] = field(default_factory=list)
    text_flow: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page_bbox": self.page_bbox,
            "page_type": self.page_type,
            "sub_type": self.sub_type,
            "regions": [r.to_dict() for r in self.regions],
            "text_flow": self.text_flow[:500],
        }


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


def bbox_area(bbox: list[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_distance(left: list[float], right: list[float]) -> float:
    dx = max(0.0, max(left[0], right[0]) - min(left[2], right[2]))
    dy = max(0.0, max(left[1], right[1]) - min(left[3], right[3]))
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return (dx * dx + dy * dy) ** 0.5


def is_inside(inner: list[float], outer: list[float]) -> bool:
    return outer[0] <= inner[0] and inner[2] <= outer[2] and outer[1] <= inner[1] and inner[3] <= outer[3]


def overlap_ratio(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    base = max(1.0, bbox_area(left))
    return inter / base


def bbox_overlap_ratio(left: list[float], right: list[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    base = max(1.0, (float(left[2]) - float(left[0])) * (float(left[3]) - float(left[1])))
    return inter / base


def merge_boxes(boxes: list[list[float]]) -> list[list[float]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged = [boxes[0][:]]
    for box in boxes[1:]:
        last = merged[-1]
        overlap_y = min(box[3], last[3]) - max(box[1], last[1])
        overlap_x = min(box[2], last[2]) - max(box[0], last[0])
        if overlap_x > 0 and overlap_y > 0:
            last[0] = min(last[0], box[0])
            last[1] = min(last[1], box[1])
            last[2] = max(last[2], box[2])
            last[3] = max(last[3], box[3])
        else:
            merged.append(box[:])
    return merged


def union_bbox(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _point_to_xy(value: Any) -> tuple[float, float] | None:
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _item_to_bbox(item: Any) -> list[float] | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    geom = item[1]
    if hasattr(geom, "x0") and hasattr(geom, "y0") and hasattr(geom, "x1") and hasattr(geom, "y1"):
        return [float(geom.x0), float(geom.y0), float(geom.x1), float(geom.y1)]
    if isinstance(geom, (list, tuple)):
        if len(geom) == 4 and all(isinstance(v, (int, float)) for v in geom):
            return [float(geom[0]), float(geom[1]), float(geom[2]), float(geom[3])]
        if len(geom) == 2:
            p0 = _point_to_xy(geom[0])
            p1 = _point_to_xy(geom[1])
            if p0 and p1:
                return [min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])]
    return None


def line_text_and_bbox(block: dict[str, Any], line: dict[str, Any]) -> tuple[str, list[float]]:
    text = normalize_whitespace("".join(str(span.get("text") or "") for span in line.get("spans") or []))
    bbox = list(line.get("bbox") or block.get("bbox") or [0.0, 0.0, 0.0, 0.0])
    return text, bbox


def is_filtered_line(bbox: list[float], filter_regions: list[dict[str, Any]]) -> bool:
    for region in filter_regions:
        if region["type"] not in {"header", "footer", "watermark_text"}:
            continue
        if bbox_overlap_ratio(bbox, list(region["bbox"])) >= 0.55:
            return True
    return False


def collect_repeated_lines(pages: list[fitz.Page]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for page in pages:
        text_dict = page.get_text("dict")
        page_height = float(page.rect.height or 0.0)
        for block in text_dict.get("blocks") or []:
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            if len(bbox) != 4:
                continue
            y0 = float(bbox[1])
            y1 = float(bbox[3])
            if y0 < 70.0 or y1 > page_height - 70.0:
                text = normalize_whitespace(
                    " ".join(
                        line_text_and_bbox(block, line)[0]
                        for line in block.get("lines") or []
                        if line_text_and_bbox(block, line)[0]
                    )
                )
                if text:
                    counter[text] += 1
    return counter


def collect_position_dense_candidates(pages: list[fitz.Page]) -> dict[tuple[str, int, int], int]:
    counter: dict[tuple[str, int, int], int] = defaultdict(int)
    for page in pages:
        page_width = float(page.rect.width or 1.0)
        page_height = float(page.rect.height or 1.0)
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks") or []:
            if block.get("type") != 0:
                continue
            for line in block.get("lines") or []:
                text, bbox = line_text_and_bbox(block, line)
                if not text or len(text) < 4:
                    continue
                cx = ((float(bbox[0]) + float(bbox[2])) / 2.0) / page_width
                cy = ((float(bbox[1]) + float(bbox[3])) / 2.0) / page_height
                if cy < 0.12 or cy > 0.88:
                    continue
                counter[(text, int(cx * 10), int(cy * 10))] += 1
    return counter


def detect_header_footer_regions(page: fitz.Page, text_dict: dict[str, Any], repeated_lines: Counter[str]) -> list[dict[str, Any]]:
    width = float(page.rect.width or 0.0)
    height = float(page.rect.height or 0.0)
    matches: list[dict[str, Any]] = []
    for block in text_dict.get("blocks") or []:
        if block.get("type") != 0:
            continue
        bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        if len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [float(value) for value in bbox]
        text = normalize_whitespace(
            " ".join(
                line_text_and_bbox(block, line)[0]
                for line in block.get("lines") or []
                if line_text_and_bbox(block, line)[0]
            )
        )
        if not text:
            continue
        repeated = repeated_lines.get(text, 0) >= 3
        page_no_like = bool(re.fullmatch(r"(?:-?\d+-?|1-1-\d+|第\s*\d+\s*页|\d+\s*/\s*\d+)", text))
        if repeated and y0 < min(60.0, height * 0.08):
            matches.append({"type": "header", "bbox": [x0, y0, x1, y1], "content": text})
        elif repeated and y1 > max(height - 60.0, height * 0.92):
            matches.append({"type": "footer", "bbox": [x0, y0, x1, y1], "content": text})
        elif page_no_like and (y0 < 70.0 or y1 > height - 70.0 or x1 > width * 0.8):
            matches.append({"type": "footer", "bbox": [x0, y0, x1, y1], "content": text})
    return matches


def detect_text_watermarks(
    page: fitz.Page,
    text_dict: dict[str, Any],
    watermark_patterns: list[str],
    position_dense_counter: dict[tuple[str, int, int], int],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    page_width = float(page.rect.width or 1.0)
    page_height = float(page.rect.height or 1.0)
    for block in text_dict.get("blocks") or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or []:
            text, bbox = line_text_and_bbox(block, line)
            if not text:
                continue
            spans = line.get("spans") or []
            cx = ((float(bbox[0]) + float(bbox[2])) / 2.0) / page_width
            cy = ((float(bbox[1]) + float(bbox[3])) / 2.0) / page_height
            is_position_dense = position_dense_counter.get((text, int(cx * 10), int(cy * 10)), 0) >= 3
            for span in spans:
                span_text = normalize_whitespace(str(span.get("text") or ""))
                if not span_text:
                    continue
                span_bbox = list(span.get("bbox") or bbox)
                if len(span_bbox) != 4:
                    continue
                span_width = max(0.0, float(span_bbox[2]) - float(span_bbox[0]))
                span_height = max(0.0, float(span_bbox[3]) - float(span_bbox[1]))
                font_size = float(span.get("size") or 0.0)
                color = int(span.get("color") or 0)
                is_pattern_match = any(pattern and pattern in span_text for pattern in watermark_patterns)
                is_large_overlay = len(span_text) >= 4 and (font_size >= 18.0 or span_width >= 180.0) and span_height >= 12.0
                is_light_color = color >= 0x888888
                is_center_overlay = 0.12 < cy < 0.88
                if is_pattern_match or is_position_dense or (is_large_overlay and is_light_color and is_center_overlay):
                    matches.append({"type": "watermark_text", "bbox": span_bbox, "content": span_text})
    return matches


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
            lines.append(
                {
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "text": text,
                    "x": float(bbox[0]),
                    "y": float(bbox[1]),
                }
            )
    return lines


def flatten_table_extract_rows(rows: list[list[Any]]) -> list[str]:
    flattened: list[str] = []
    for row in rows:
        parts = [normalize_whitespace(str(cell or "")) for cell in row if normalize_whitespace(str(cell or ""))]
        if parts:
            flattened.append(" | ".join(parts))
    return flattened


def collect_texts_in_bbox(text_blocks: list[dict[str, Any]], bbox: list[float]) -> list[str]:
    selected: list[tuple[float, float, str]] = []
    seen: set[tuple[float, float, str]] = set()
    for tb in text_blocks:
        if (
            is_inside(tb["bbox"], bbox)
            or overlap_ratio(tb["bbox"], bbox) >= 0.12
            or overlap_ratio(bbox, tb["bbox"]) >= 0.35
        ):
            key = (tb["bbox"][1], tb["bbox"][0], tb["text"])
            if key in seen:
                continue
            seen.add(key)
            selected.append((tb["bbox"][1], tb["bbox"][0], tb["text"]))
    selected.sort(key=lambda item: (item[0], item[1]))
    return [text for _, _, text in selected]


def is_continuation_label(text: str) -> bool:
    normalized = normalize_whitespace(text)
    return any(word in normalized for word in CONTINUATION_WORDS)


def is_unit_text(text: str) -> bool:
    normalized = normalize_whitespace(text)
    return any(normalized.startswith(prefix) for prefix in UNIT_PREFIXES)


def is_year_or_period_text(text: str) -> bool:
    normalized = normalize_whitespace(text)
    return bool(re.search(r"(?:19|20)\d{2}", normalized))


def row_looks_like_header(text: str) -> bool:
    normalized = normalize_whitespace(text)
    header_hits = sum(1 for word in TABLE_HEADER_WORDS if word in normalized)
    return (
        header_hits >= 1
        or ("项目" in normalized and is_year_or_period_text(normalized))
        or ("金额" in normalized and "占比" in normalized)
    )


def is_transition_helper_text(text: str) -> bool:
    normalized = normalize_whitespace(text)
    if not normalized:
        return True
    if is_continuation_label(normalized) or is_unit_text(normalized):
        return True
    if len(normalized) <= 24 and (row_looks_like_header(normalized) or is_year_or_period_text(normalized)):
        return True
    return False


def is_blocking_transition_text(text: str) -> bool:
    normalized = normalize_whitespace(text)
    if not normalized:
        return False
    if is_transition_helper_text(normalized):
        return False
    return len(normalized) >= 5


def get_texts_between_tables(text_blocks: list[dict[str, Any]], upper_bbox: list[float], lower_bbox: list[float]) -> list[str]:
    texts: list[tuple[float, float, str]] = []
    left = min(upper_bbox[0], lower_bbox[0]) - 40
    right = max(upper_bbox[2], lower_bbox[2]) + 40
    top = upper_bbox[3]
    bottom = lower_bbox[1]
    for block in text_blocks:
        bx0, by0, bx1, by1 = block["bbox"]
        if by0 >= top and by1 <= bottom and bx1 >= left and bx0 <= right:
            text = normalize_whitespace(block["text"])
            if text:
                texts.append((by0, bx0, text))
    texts.sort(key=lambda item: (item[0], item[1]))
    return [text for _, _, text in texts]


def merge_same_page_continuation_tables(table_areas: list[dict[str, Any]], text_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(table_areas) < 2:
        return table_areas
    ordered = sorted(table_areas, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    merged: list[dict[str, Any]] = []
    current = ordered[0]
    for nxt in ordered[1:]:
        between_texts = get_texts_between_tables(text_blocks, current["bbox"], nxt["bbox"])
        has_continuation_marker = any(is_continuation_label(text) for text in between_texts)
        only_helper_texts = bool(between_texts) and all(is_transition_helper_text(text) for text in between_texts)
        if has_continuation_marker and only_helper_texts:
            current = {
                "bbox": [
                    min(current["bbox"][0], nxt["bbox"][0]),
                    min(current["bbox"][1], nxt["bbox"][1]),
                    max(current["bbox"][2], nxt["bbox"][2]),
                    max(current["bbox"][3], nxt["bbox"][3]),
                ],
                "row_count": int(current.get("row_count") or 0) + int(nxt.get("row_count") or 0),
                "col_count": max(int(current.get("col_count") or 0), int(nxt.get("col_count") or 0)),
                "has_thick_border": bool(current.get("has_thick_border") or nxt.get("has_thick_border")),
                "is_pymupdf_table": bool(current.get("is_pymupdf_table") or nxt.get("is_pymupdf_table")),
                "is_pipe_table": bool(current.get("is_pipe_table") or nxt.get("is_pipe_table")),
                "is_wireless": bool(current.get("is_wireless") or nxt.get("is_wireless")),
                "is_same_page_continuation": True,
                "detector": "same_page_continuation",
                "confidence": max(float(current.get("confidence") or 0.0), float(nxt.get("confidence") or 0.0)),
                "text_lines": (current.get("text_lines") or []) + (nxt.get("text_lines") or []),
            }
            continue
        merged.append(current)
        current = nxt
    merged.append(current)
    return merged


def table_candidate_priority(candidate: dict[str, Any]) -> float:
    if candidate.get("is_pymupdf_table"):
        return 5.5
    if candidate.get("is_pipe_table"):
        return 5.0
    if candidate.get("is_top_continuation_table"):
        return 4.7
    if candidate.get("is_multiline_row_table"):
        return 4.2
    if candidate.get("has_thick_border"):
        return 3.8
    if candidate.get("is_wireless"):
        return 3.2
    return 3.0


def table_candidates_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if is_inside(left["bbox"], right["bbox"]) or is_inside(right["bbox"], left["bbox"]):
        return True
    return overlap_ratio(left["bbox"], right["bbox"]) >= 0.35 or overlap_ratio(right["bbox"], left["bbox"]) >= 0.35


def dedupe_table_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    groups: list[list[dict[str, Any]]] = []
    for candidate in sorted(candidates, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        placed = False
        for group in groups:
            if any(table_candidates_overlap(candidate, existing) for existing in group):
                group.append(candidate)
                placed = True
                break
        if not placed:
            groups.append([candidate])

    deduped: list[dict[str, Any]] = []
    for group in groups:
        best = max(
            group,
            key=lambda item: (
                table_candidate_priority(item),
                float(item.get("confidence") or 0.0),
                int(item.get("row_count") or 0),
                int(item.get("col_count") or 0),
            ),
        )
        deduped.append(best)
    return deduped


def detect_pipe_tables(page: fitz.Page) -> list[dict[str, Any]]:
    pipe_lines: list[dict[str, Any]] = []
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        text = normalize_whitespace(str(text))
        if not text:
            continue
        if text.count("|") >= 2:
            pipe_lines.append(
                {
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "text": text,
                    "y0": float(y0),
                    "y1": float(y1),
                }
            )

    if not pipe_lines:
        return []

    pipe_lines.sort(key=lambda item: item["y0"])
    groups: list[list[dict[str, Any]]] = []
    current = [pipe_lines[0]]
    for line in pipe_lines[1:]:
        gap = line["y0"] - current[-1]["y1"]
        if gap <= 20:
            current.append(line)
        else:
            if len(current) >= 2:
                groups.append(current)
            current = [line]
    if len(current) >= 2:
        groups.append(current)

    tables: list[dict[str, Any]] = []
    for group in groups:
        lines_text = [item["text"] for item in group]
        x0 = min(item["bbox"][0] for item in group)
        y0 = min(item["bbox"][1] for item in group)
        x1 = max(item["bbox"][2] for item in group)
        y1 = max(item["bbox"][3] for item in group)
        col_count = max(text.count("|") + 1 for text in lines_text)
        tables.append(
            {
                "bbox": [x0, y0, x1, y1],
                "row_count": len(group),
                "col_count": col_count,
                "has_thick_border": False,
                "is_pipe_table": True,
                "detector": "pipe_table",
                "confidence": 0.96,
                "text_lines": lines_text,
            }
        )
    return tables


def detect_pymupdf_tables(page: fitz.Page) -> list[dict[str, Any]]:
    try:
        finder = page.find_tables()
    except Exception:
        return []
    tables: list[dict[str, Any]] = []
    for table in getattr(finder, "tables", []) or []:
        bbox = [float(v) for v in table.bbox]
        if bbox_area(bbox) <= 800:
            continue
        text_lines = flatten_table_extract_rows(table.extract() or [])
        if not text_lines:
            continue
        tables.append(
            {
                "bbox": bbox,
                "row_count": int(getattr(table, "row_count", 0) or 0),
                "col_count": int(getattr(table, "col_count", 0) or 0),
                "has_thick_border": True,
                "is_pymupdf_table": True,
                "detector": "pymupdf_table",
                "confidence": 0.99,
                "text_lines": text_lines,
            }
        )
    return tables


def detect_multiline_row_tables(page: fitz.Page) -> list[dict[str, Any]]:
    row_blocks: list[dict[str, Any]] = []
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        x0, y0, x1, y1, raw_text = block[:5]
        raw_text = str(raw_text).strip()
        if not raw_text:
            continue
        parts = [normalize_whitespace(part) for part in raw_text.split("\n") if normalize_whitespace(part)]
        if len(parts) < 2:
            continue
        width = float(x1) - float(x0)
        if width < page.rect.width * 0.38:
            continue
        has_numeric_tail = any(NUMBER_PATTERN.fullmatch(part.replace(",", "")) or PERCENTAGE_PATTERN.search(part) for part in parts[1:])
        has_header_words = any(any(word in part for word in TABLE_HEADER_WORDS) for part in parts)
        if not has_numeric_tail and not has_header_words:
            continue
        if looks_like_flowchart_text("\n".join(parts)):
            continue
        row_blocks.append(
            {
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "parts": parts,
                "text": "\n".join(parts),
                "x0": float(x0),
                "x1": float(x1),
                "y0": float(y0),
                "y1": float(y1),
            }
        )

    if not row_blocks:
        return []

    row_blocks.sort(key=lambda item: item["y0"])
    groups: list[list[dict[str, Any]]] = []
    current = [row_blocks[0]]
    for block in row_blocks[1:]:
        prev = current[-1]
        gap = block["y0"] - prev["y1"]
        prev_width = max(1.0, prev["x1"] - prev["x0"])
        curr_width = max(1.0, block["x1"] - block["x0"])
        same_span = (
            (abs(block["x0"] - prev["x0"]) <= 25 and abs(block["x1"] - prev["x1"]) <= 25)
            or (abs(block["x1"] - prev["x1"]) <= 25 and min(prev_width, curr_width) / max(prev_width, curr_width) >= 0.72)
        )
        if gap <= 28 and same_span:
            current.append(block)
        else:
            if len(current) >= 2:
                groups.append(current)
            current = [block]
    if len(current) >= 2:
        groups.append(current)

    tables: list[dict[str, Any]] = []
    for group in groups:
        lines_text = [item["text"] for item in group]
        x0 = min(item["bbox"][0] for item in group)
        y0 = min(item["bbox"][1] for item in group)
        x1 = max(item["bbox"][2] for item in group)
        y1 = max(item["bbox"][3] for item in group)
        col_count = max(len(item["parts"]) for item in group)
        header_like = any(any(word in line for word in TABLE_HEADER_WORDS) for line in lines_text)
        numeric_rows = sum(
            1
            for line in lines_text
            if any(NUMBER_PATTERN.fullmatch(part.replace(",", "")) or PERCENTAGE_PATTERN.search(part) for part in line.split("\n")[1:])
        )
        if len(group) < 2 and not (header_like or numeric_rows >= 1):
            continue
        tables.append(
            {
                "bbox": [x0, y0, x1, y1],
                "row_count": len(group),
                "col_count": col_count,
                "has_thick_border": False,
                "is_multiline_row_table": True,
                "detector": "multiline_row_table",
                "confidence": 0.94,
                "text_lines": lines_text,
            }
        )
    return tables


def detect_top_continuation_table(page: fitz.Page) -> list[dict[str, Any]]:
    top_blocks: list[dict[str, Any]] = []
    continuation_label_found = False
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        x0, y0, x1, y1, raw_text = block[:5]
        if float(y0) > 200:
            continue
        raw_text = str(raw_text).strip()
        if not raw_text:
            continue
        normalized_block = normalize_whitespace(raw_text)
        if is_continuation_label(normalized_block):
            continuation_label_found = True
            continue
        parts = [normalize_whitespace(part) for part in raw_text.split("\n") if normalize_whitespace(part)]
        if len(parts) < 2:
            continue
        has_numeric_tail = any(NUMBER_PATTERN.fullmatch(part.replace(",", "")) or PERCENTAGE_PATTERN.search(part) for part in parts[1:])
        if not has_numeric_tail:
            continue
        top_blocks.append(
            {
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "parts": parts,
                "text": "\n".join(parts),
                "x0": float(x0),
                "x1": float(x1),
                "y0": float(y0),
                "y1": float(y1),
            }
        )

    min_required_blocks = 1 if continuation_label_found else 2
    if len(top_blocks) < min_required_blocks:
        return []

    top_blocks.sort(key=lambda item: item["y0"])
    group = [top_blocks[0]]
    for block in top_blocks[1:]:
        prev = group[-1]
        gap = block["y0"] - prev["y1"]
        if gap <= 18 and abs(block["x1"] - prev["x1"]) <= 35 and abs(block["x0"] - prev["x0"]) <= 35:
            group.append(block)
        else:
            break

    if len(group) < min_required_blocks:
        return []

    x0 = min(item["bbox"][0] for item in group)
    y0 = min(item["bbox"][1] for item in group)
    x1 = max(item["bbox"][2] for item in group)
    y1 = max(item["bbox"][3] for item in group)
    lines_text = [item["text"] for item in group]
    col_count = max(len(item["parts"]) for item in group)
    return [
        {
            "bbox": [x0, y0, x1, y1],
            "row_count": len(group),
            "col_count": col_count,
            "has_thick_border": False,
            "is_top_continuation_table": True,
            "detector": "top_continuation",
            "continuation_label_found": continuation_label_found,
            "confidence": 0.98,
            "text_lines": lines_text,
        }
    ]


def looks_like_flowchart_text(text: str) -> bool:
    normalized = normalize_whitespace(text)
    if any(keyword in normalized for keyword in ["股权结构图", "股权结构", "组织结构", "组织架构"]):
        return True
    vertical_tokens = re.findall(r"(?:[\u4e00-\u9fff]\s*\|\s*){2,}[\u4e00-\u9fff]?", normalized)
    if vertical_tokens:
        return True
    percentages = PERCENTAGE_PATTERN.findall(normalized)
    if len(percentages) >= 4 and any(keyword in normalized for keyword in ["兴图新科", "公司", "持股"]):
        return True
    return False


def extract_image_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    image_blocks = []
    seen: set[tuple[int, float, float, float, float]] = set()
    page_area = max(1.0, float(page.rect.width) * float(page.rect.height))
    for img in page.get_images(full=True):
        xref = int(img[0])
        for rect in page.get_image_rects(xref):
            bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
            area_ratio = bbox_area(bbox) / page_area
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = max(width, height) / max(1.0, min(width, height))
            # Filter tiled watermark slices and tiny repeated stamps.
            if area_ratio < 0.04 and width < page.rect.width * 0.35 and height < page.rect.height * 0.25:
                continue
            # Filter decorative line images and ultra-thin separator strips.
            if min(width, height) <= 4.0:
                continue
            if aspect_ratio >= 40 and area_ratio < 0.01:
                continue
            key = (xref, *bbox)
            if key in seen:
                continue
            seen.add(key)
            image_blocks.append({"bbox": bbox, "xref": xref})
    return image_blocks


def extract_drawing_segments(page: fitz.Page) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    h_segments = []
    complex_boxes = []
    for drawing in page.get_drawings():
        width = float(drawing.get("width") or drawing.get("linewidth") or 1.0)
        fill = drawing.get("fill")
        has_fill = fill is not None and fill not in {(1.0, 1.0, 1.0), (0, 0, 0)}
        for item in drawing.get("items") or []:
            op = item[0]
            bbox = _item_to_bbox(item)
            if not bbox:
                continue
            x0, y0, x1, y1 = bbox
            if op == "l":
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                if dy < 2 and dx > page.rect.width * TABLE_HLINE_MIN_RATIO:
                    h_segments.append({"bbox": bbox, "y": (y0 + y1) / 2.0, "linewidth": width})
                elif dx > 5 and dy > 5:
                    complex_boxes.append(bbox)
            elif op == "re":
                if bbox[3] - bbox[1] <= 4 or bbox[2] - bbox[0] > page.rect.width * TABLE_HLINE_MIN_RATIO:
                    h_segments.append({"bbox": bbox, "y": bbox[1], "linewidth": width})
                    h_segments.append({"bbox": bbox, "y": bbox[3], "linewidth": width})
                if has_fill or bbox_area(bbox) > 500:
                    complex_boxes.append(bbox)
            elif op == "c":
                complex_boxes.append(bbox)
    return h_segments, complex_boxes


def detect_table_by_lines(page: fitz.Page) -> list[dict[str, Any]]:
    h_lines, _ = extract_drawing_segments(page)
    if len(h_lines) < TABLE_MIN_ROWS + 1:
        return []
    h_lines.sort(key=lambda l: l["y"])
    groups = []
    current = [h_lines[0]]
    for line in h_lines[1:]:
        gap = line["y"] - current[-1]["y"]
        if TABLE_ROW_HEIGHT_MIN <= gap <= TABLE_ROW_HEIGHT_MAX:
            current.append(line)
        else:
            if len(current) >= TABLE_MIN_ROWS + 1:
                groups.append(current)
            current = [line]
    if len(current) >= TABLE_MIN_ROWS + 1:
        groups.append(current)

    tables = []
    text_blocks = page.get_text("blocks")
    for group in groups:
        y_min = group[0]["y"] - 5
        y_max = group[-1]["y"] + 5
        x_min = min(l["bbox"][0] for l in group)
        x_max = max(l["bbox"][2] for l in group)
        region_texts = []
        for block in text_blocks:
            if len(block) < 5:
                continue
            bx0, by0, bx1, by1, text = block[:5]
            if x_min - 10 <= bx0 and bx1 <= x_max + 10 and y_min <= by1 and by0 <= y_max and str(text).strip():
                region_texts.append((bx0, normalize_whitespace(str(text))))
        if len(region_texts) < 2:
            continue
        x_buckets = defaultdict(list)
        for x, text in region_texts:
            bucket = round(float(x) / 20) * 20
            x_buckets[bucket].append(text)
        if len([k for k, v in x_buckets.items() if len(v) >= 1]) >= 2:
            if (y_max - y_min) > page.rect.height * 0.92:
                continue
            tables.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "row_count": len(group) - 1,
                "col_count": len(x_buckets),
                "has_thick_border": any(seg["linewidth"] > 1.5 for seg in group),
                "detector": "line_table",
                "confidence": min(1.0, len(group) / 10),
            })
    return tables


def detect_table_by_text_alignment(page: fitz.Page, filter_regions: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    text_lines = extract_text_lines(page, filter_regions=filter_regions)

    if len(text_lines) < TABLE_MIN_ROWS * 2:
        return []

    text_lines = [line for line in text_lines if 70.0 <= line["y"] <= page.rect.height - 70.0]
    text_lines.sort(key=lambda l: l["y"])
    row_buckets = defaultdict(list)
    for line in text_lines:
        y_bucket = round(line["y"] / 18.0) * 18.0
        row_buckets[y_bucket].append(line)

    sorted_y = sorted(row_buckets.keys())
    table_candidates = []
    for i in range(len(sorted_y)):
        consecutive_rows = []
        for j in range(i, len(sorted_y)):
            if j > i and sorted_y[j] - sorted_y[j - 1] > 34:
                break
            if len(row_buckets[sorted_y[j]]) >= 2:
                consecutive_rows.append(sorted_y[j])
        if len(consecutive_rows) < TABLE_MIN_ROWS:
            continue
        all_lines = []
        for y in consecutive_rows:
            all_lines.extend(row_buckets[y])
        x_positions = [round(l["x"] / 24) * 24 for l in all_lines]
        x_counts = defaultdict(int)
        for x in x_positions:
            x_counts[x] += 1
        stable_cols = [x for x, c in x_counts.items() if c >= max(2, len(consecutive_rows) * 0.45)]
        if len(stable_cols) >= 2:
            lines_text = [l["text"] for l in all_lines]
            joined_text = "\n".join(lines_text)
            kv_count = sum(1 for t in lines_text if KEY_VALUE_PATTERN.search(t))
            numeric_count = sum(1 for t in lines_text if NUMBER_PATTERN.fullmatch(t.replace(",", "")) or PERCENTAGE_PATTERN.search(t))
            y_min = min(l["y"] for l in all_lines) - 5
            y_max = max(l["y"] for l in all_lines) + 10
            x_min = min(l["x"] for l in all_lines) - 10
            x_max = max(l["bbox"][2] for l in all_lines)
            if (y_max - y_min) > page.rect.height * 0.92:
                continue
            if len(stable_cols) < 3 and numeric_count < 1 and kv_count < 2:
                continue
            if looks_like_flowchart_text(joined_text):
                continue
            table_candidates.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "row_count": len(consecutive_rows),
                "col_count": len(stable_cols),
                "is_wireless": True,
                "kv_ratio": kv_count / max(len(lines_text), 1),
                "numeric_count": numeric_count,
                "detector": "text_alignment",
            })

    if not table_candidates:
        return []
    merged = merge_boxes([t["bbox"] for t in table_candidates])
    result = []
    for box in merged:
        contained = [t for t in table_candidates if is_inside(t["bbox"], box)]
        if contained:
            best = max(contained, key=lambda t: t["row_count"])
            result.append({
                "bbox": box,
                "row_count": best["row_count"],
                "col_count": best["col_count"],
                "has_thick_border": False,
                "is_wireless": True,
                "confidence": 0.7,
            })
    return result


def is_table_like_visual_region(
    page: fitz.Page,
    bbox: list[float],
    text_blocks: list[dict[str, Any]],
    filter_regions: list[dict[str, Any]] | None = None,
) -> bool:
    if filter_regions is None:
        filter_regions = []
    texts = collect_texts_in_bbox(text_blocks, bbox)
    if not texts:
        return False
    joined = "\n".join(texts)
    header_hits = sum(1 for word in TABLE_HEADER_WORDS if word in joined)
    numeric_hits = sum(1 for text in texts if NUMBER_PATTERN.search(text.replace(",", "")) or PERCENTAGE_PATTERN.search(text))
    if header_hits >= 2 and numeric_hits >= 2:
        return True
    lines = [line for line in extract_text_lines(page, filter_regions=filter_regions) if overlap_ratio(line["bbox"], bbox) >= 0.2]
    if len(lines) < 4:
        return False
    x_buckets: dict[int, int] = defaultdict(int)
    for line in lines:
        bucket = round(float(line["x"]) / 24.0) * 24
        x_buckets[bucket] += 1
    stable_cols = sum(1 for count in x_buckets.values() if count >= 2)
    return stable_cols >= 3 and numeric_hits >= 1


def detect_figure_regions(page: fitz.Page, exclude_bboxes: list[list[float]]) -> list[dict[str, Any]]:
    _, complex_boxes = extract_drawing_segments(page)
    if not complex_boxes:
        return []
    text_blocks = extract_text_blocks(page)
    merged = merge_boxes(complex_boxes)
    figures = []
    for box in merged:
        if bbox_area(box) <= 1000:
            continue
        overlap_with_table = any(overlap_ratio(box, table) > 0.3 for table in exclude_bboxes)
        if not overlap_with_table and not is_table_like_visual_region(page, box, text_blocks):
            figures.append({"bbox": box})
    return figures


def extract_nearby_text(page: fitz.Page, bbox: list[float], margin: float = 50) -> str:
    nearby = []
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        bx0, by0, bx1, by1, text = block[:5]
        if (abs(by0 - bbox[3]) < margin * 2 or abs(by1 - bbox[1]) < margin * 2 or (bbox[0] <= bx1 and bx0 <= bbox[2])):
            text = normalize_whitespace(str(text))
            if text:
                nearby.append(text)
    return " ".join(nearby)[:200]


def classify_figure(page: fitz.Page, bbox: list[float], nearby_text: str = "") -> str:
    area_ratio = bbox_area(bbox) / bbox_area([0, 0, page.rect.width, page.rect.height])
    text_lower = nearby_text.lower()
    if area_ratio <= 0.05:
        return "inline_image"
    h_lines, complex_boxes = extract_drawing_segments(page)
    features = {
        "has_arrow": False,
        "has_filled_box": False,
        "h_line_count": 0,
        "v_line_count": 0,
    }
    for seg in h_lines:
        if overlap_ratio(seg["bbox"], bbox) > 0.2:
            features["h_line_count"] += 1
    for box in complex_boxes:
        if overlap_ratio(box, bbox) > 0.2:
            features["has_filled_box"] = True
    if features["h_line_count"] >= 3 and any(kw in text_lower for kw in ["增长率", "趋势", "同比", "市场份额", "规模"]):
        return "data_viz"
    if features["has_filled_box"] and PERCENTAGE_PATTERN.search(nearby_text) and any(kw in nearby_text for kw in ["投资", "公司", "持股", "股份"]):
        return "flowchart"
    if any(kw in text_lower for kw in DATAVIZ_KEYWORDS):
        return "data_viz"
    if any(kw in text_lower for kw in FLOWCHART_KEYWORDS):
        return "flowchart"
    return "mixed_image"


def table_looks_like_flowchart(cell_texts: list[str]) -> bool:
    joined = normalize_whitespace("\n".join(cell_texts))
    if not joined:
        return False
    if looks_like_flowchart_text(joined):
        return True
    lines = [line.strip() for line in joined.split("\n") if line.strip()]
    if len(lines) < 3:
        return False
    numeric_lines = sum(1 for line in lines if NUMBER_PATTERN.search(line.replace(",", "")) or PERCENTAGE_PATTERN.search(line))
    short_lines = sum(1 for line in lines if len(line) <= 18)
    flow_words = [
        "流程",
        "需求",
        "设计",
        "开发",
        "调试",
        "测试",
        "验收",
        "服务",
        "模块",
        "系统",
        "生产",
        "交付",
    ]
    flow_hits = sum(1 for word in flow_words if word in joined)
    return numeric_lines <= max(1, len(lines) // 8) and short_lines >= max(3, len(lines) // 3) and flow_hits >= 2


def infer_table_sub_type(
    page: fitz.Page,
    bbox: list[float],
    cell_texts: list[str],
    embedded_visual_count: int,
) -> str:
    joined = normalize_whitespace("\n".join(cell_texts))
    numeric_hits = len(PERCENTAGE_PATTERN.findall(joined)) + len(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", joined))
    header_hits = sum(1 for word in TABLE_HEADER_WORDS if word in joined)
    timeline_hits = len(re.findall(r"\bY\d\b|\bQ[1-4]\b", joined))
    if embedded_visual_count > 0:
        return "table_with_embedded_image"
    if timeline_hits >= 4 and numeric_hits <= max(2, timeline_hits // 2):
        return "timeline_table"
    if len(joined) <= 40 and numeric_hits == 0 and header_hits <= 1 and bbox_area(bbox) > bbox_area([0.0, 0.0, page.rect.width, page.rect.height]) * 0.02:
        return "decorative_table"
    if header_hits >= 2 and numeric_hits >= 2:
        return "data_table"
    if KEY_VALUE_PATTERN.search(joined) and joined.count("\n") <= 8:
        return "key_value"
    return "data_table"


def cluster_visual_candidates(
    page: fitz.Page,
    image_blocks: list[dict[str, Any]],
    complex_figures: list[dict[str, Any]],
    table_boxes: list[list[float]],
    text_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for img in image_blocks:
        bbox = img["bbox"]
        if any(is_inside(bbox, tbox) or overlap_ratio(bbox, tbox) >= 0.45 for tbox in table_boxes):
            continue
        candidates.append({"bbox": bbox, "kind": "image", "xref": int(img["xref"])})
    for fig in complex_figures:
        bbox = fig["bbox"]
        if any(is_inside(bbox, tbox) or overlap_ratio(bbox, tbox) >= 0.3 or overlap_ratio(tbox, bbox) >= 0.45 for tbox in table_boxes):
            continue
        candidates.append({"bbox": bbox, "kind": "drawing", "xref": 0})
    if not candidates:
        return []

    groups: list[list[dict[str, Any]]] = []
    for candidate in sorted(candidates, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        placed = False
        for group in groups:
            if any(
                overlap_ratio(candidate["bbox"], existing["bbox"]) >= FIGURE_CLUSTER_OVERLAP
                or overlap_ratio(existing["bbox"], candidate["bbox"]) >= FIGURE_CLUSTER_OVERLAP
                or bbox_distance(candidate["bbox"], existing["bbox"]) <= FIGURE_CLUSTER_GAP
                for existing in group
            ):
                group.append(candidate)
                placed = True
                break
        if not placed:
            groups.append([candidate])

    merged_candidates: list[dict[str, Any]] = []
    for group in groups:
        bbox = union_bbox([item["bbox"] for item in group])
        if is_table_like_visual_region(page, bbox, text_blocks):
            continue
        xrefs = sorted({int(item["xref"]) for item in group if int(item["xref"]) > 0})
        nearby_text = extract_nearby_text(page, bbox)
        sub_type = classify_figure(page, bbox, nearby_text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if sub_type == "inline_image":
            if (
                len(group) >= 2
                or width >= page.rect.width * 0.18
                or height >= page.rect.height * 0.1
                or looks_like_flowchart_text(nearby_text)
            ):
                sub_type = "flowchart" if looks_like_flowchart_text(nearby_text) else "mixed_image"
        merged_candidates.append(
            {
                "bbox": bbox,
                "xref": xrefs[0] if len(xrefs) == 1 else 0,
                "xref_list": xrefs,
                "nearby_text": nearby_text,
                "sub_type": sub_type,
                "region_type": "image" if sub_type == "inline_image" else "figure",
                "confidence": 0.76 if sub_type == "inline_image" else 0.82,
                "source": "visual_cluster",
                "part_count": len(group),
            }
        )
    return merged_candidates


def assign_reading_order(regions: list[Region], page_bbox: list[float]) -> list[Region]:
    page_mid_x = (page_bbox[0] + page_bbox[2]) / 2.0

    def sort_key(region: Region) -> tuple[float, float, float]:
        cy = (region.bbox[1] + region.bbox[3]) / 2.0
        cx = (region.bbox[0] + region.bbox[2]) / 2.0
        y_bucket = round(cy / 12.0) * 12.0
        column_bias = 0.0 if cx <= page_mid_x else 0.2
        return (y_bucket, column_bias, cx)

    regions.sort(key=sort_key)
    for i, region in enumerate(regions, 1):
        region.reading_order = i  # type: ignore[attr-defined]
    return regions


def infer_page_type(regions: list[Region], page_bbox: list[float]) -> tuple[str, str]:
    page_area = bbox_area(page_bbox)
    table_area = sum(bbox_area(r.bbox) for r in regions if r.region_type == "table")
    figure_area = sum(bbox_area(r.bbox) for r in regions if r.region_type == "figure")
    image_area = sum(bbox_area(r.bbox) for r in regions if r.region_type == "image")
    if table_area / page_area >= 0.1:
        return "table", "data_table"
    if figure_area / page_area >= 0.15:
        if any(r.sub_type == "flowchart" for r in regions):
            return "figure", "flowchart"
        if any(r.sub_type == "data_viz" for r in regions):
            return "figure", "data_viz"
        return "figure", "mixed_image"
    if image_area / page_area >= 0.2:
        return "figure", "mixed_image"
    return "text", "paragraph"


def detect_cross_page_tables(pages: list[LayoutPage], config: dict[str, Any]) -> tuple[list[Region], list[dict[str, Any]]]:
    merge_high = float(config.get("cross_page_merge_high", CROSS_PAGE_MERGE_HIGH))
    merge_low = float(config.get("cross_page_merge_low", CROSS_PAGE_MERGE_LOW))
    merges: list[tuple[Region, Region, float, str]] = []
    debug_log: list[dict[str, Any]] = []
    log("开始跨页表格候选评分")

    for i in range(len(pages) - 1):
        page_a = pages[i]
        page_b = pages[i + 1]
        tables_a = [r for r in page_a.regions if r.region_type == "table"]
        tables_b = [r for r in page_b.regions if r.region_type == "table"]
        if not tables_a or not tables_b:
            continue
        for ta in tables_a:
            page_height = page_a.page_bbox[3]
            gap_to_bottom = page_height - ta.bbox[3]
            is_bottom = gap_to_bottom <= max(CROSS_PAGE_GAP_BOTTOM, page_height * 0.1)
            a_is_last_region = bool(page_a.regions) and ta.reading_order == max(r.reading_order for r in page_a.regions)
            if not is_bottom and gap_to_bottom <= page_height * 0.18:
                is_bottom = True
            if not is_bottom:
                continue
            for tb in tables_b:
                gap_to_top = tb.bbox[1] - page_b.page_bbox[1]
                is_top = gap_to_top <= max(CROSS_PAGE_GAP_TOP, page_height * 0.2)
                b_is_first_region = bool(page_b.regions) and tb.reading_order == min(r.reading_order for r in page_b.regions)
                if not is_top and gap_to_top <= page_height * 0.24:
                    is_top = True
                if not is_top:
                    continue
                align_score = column_alignment_score(ta, tb)
                type_score = column_type_match_score(ta, tb)
                cont_score = reading_continuity_score(page_a, page_b, ta, tb)
                confidence = round(align_score * 0.4 + type_score * 0.3 + cont_score * 0.3, 4)
                decision = "separate"
                if confidence >= merge_high:
                    decision = "auto_merge"
                elif confidence >= merge_low:
                    decision = "manual_review"
                candidate = {
                    "from_page": page_a.page_index,
                    "to_page": page_b.page_index,
                    "from_region": ta.region_id,
                    "to_region": tb.region_id,
                    "from_preview": ta.text[:100],
                    "to_preview": tb.text[:100],
                    "gap_to_bottom": round(gap_to_bottom, 4),
                    "gap_to_top": round(gap_to_top, 4),
                    "from_is_last_region": a_is_last_region,
                    "to_is_first_region": b_is_first_region,
                    "column_alignment": round(align_score, 4),
                    "column_type_match": round(type_score, 4),
                    "reading_continuity": round(cont_score, 4),
                    "confidence": confidence,
                    "decision": decision,
                }
                debug_log.append(candidate)
                log(
                    f"跨页候选 p{page_a.page_index}:{ta.region_id} -> p{page_b.page_index}:{tb.region_id} "
                    f"confidence={confidence} decision={decision}"
                )
                if decision == "auto_merge":
                    merges.append((ta, tb, confidence, decision))

    merged_tables: list[Region] = []
    if not merges:
        return merged_tables, debug_log

    best_outgoing: dict[str, tuple[Region, Region, float, str]] = {}
    best_incoming: dict[str, tuple[Region, Region, float, str]] = {}
    for ta, tb, conf, decision in merges:
        current_out = best_outgoing.get(ta.region_id)
        if current_out is None or conf > current_out[2]:
            best_outgoing[ta.region_id] = (ta, tb, conf, decision)
        current_in = best_incoming.get(tb.region_id)
        if current_in is None or conf > current_in[2]:
            best_incoming[tb.region_id] = (ta, tb, conf, decision)

    selected_edges: list[tuple[Region, Region, float, str]] = []
    for ta_id, edge in best_outgoing.items():
        _, tb, _, _ = edge
        incoming = best_incoming.get(tb.region_id)
        if incoming and incoming[0].region_id == ta_id:
            selected_edges.append(edge)

    successor: dict[str, tuple[Region, Region, float, str]] = {edge[0].region_id: edge for edge in selected_edges}
    predecessor: dict[str, tuple[Region, Region, float, str]] = {edge[1].region_id: edge for edge in selected_edges}
    visited: set[str] = set()

    start_region_ids = [region_id for region_id in successor if region_id not in predecessor]
    if not start_region_ids:
        start_region_ids = list(successor.keys())

    for start_region_id in start_region_ids:
        if start_region_id in visited:
            continue
        chain_regions: list[Region] = []
        chain_scores: list[float] = []
        current_region_id = start_region_id
        local_seen: set[str] = set()
        while current_region_id in successor and current_region_id not in local_seen:
            local_seen.add(current_region_id)
            ta, tb, conf, _ = successor[current_region_id]
            if not chain_regions:
                chain_regions.append(ta)
            chain_regions.append(tb)
            chain_scores.append(conf)
            current_region_id = tb.region_id

        deduped_chain: list[Region] = []
        chain_ids: set[str] = set()
        for region in chain_regions:
            if region.region_id in chain_ids:
                continue
            deduped_chain.append(region)
            chain_ids.add(region.region_id)

        if len(deduped_chain) < 2:
            continue

        merged_region_id = f"TABLE_{len(merged_tables)+1:04d}"
        merged = Region(
            region_id=merged_region_id,
            page_index=deduped_chain[0].page_index,
            region_type="table",
            sub_type="data_table",
            bbox=[
                min(region.bbox[0] for region in deduped_chain),
                min(region.bbox[1] for region in deduped_chain),
                max(region.bbox[2] for region in deduped_chain),
                max(region.bbox[3] for region in deduped_chain),
            ],
            text="\n".join(region.text for region in deduped_chain if region.text),
            confidence=round(sum(chain_scores) / max(len(chain_scores), 1), 4),
            is_cross_page=True,
            merged_from=[region.region_id for region in deduped_chain],
            page_start=deduped_chain[0].page_index,
            page_end=deduped_chain[-1].page_index,
            col_count=max(region.col_count for region in deduped_chain),
            row_count=sum(region.row_count for region in deduped_chain),
            source="cross_page_chain",
        )
        merged_tables.append(merged)
        for region in deduped_chain:
            visited.add(region.region_id)
            region.is_cross_page = True
            region.merged_from = [merged_region_id]

    return merged_tables, debug_log


def column_alignment_score(a: Region, b: Region) -> float:
    width = max(1.0, max(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0]))
    left_delta = abs(a.bbox[0] - b.bbox[0])
    right_delta = abs(a.bbox[2] - b.bbox[2])
    edge_score = max(0.0, 1.0 - ((left_delta + right_delta) / (2.0 * width)))
    cols_a = max(1, a.col_count)
    cols_b = max(1, b.col_count)
    count_score = min(cols_a, cols_b) / max(cols_a, cols_b)
    return max(0.0, min(1.0, 0.55 * count_score + 0.45 * edge_score))


def infer_column_types(text: str) -> list[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return []
    samples: list[list[str]] = [[], []]
    for line in lines[:10]:
        if KEY_VALUE_PATTERN.search(line):
            parts = re.split(r"[:：]", line, maxsplit=1)
            if len(parts) == 2:
                samples[0].append(parts[0].strip())
                samples[1].append(parts[1].strip())
    if not samples[0]:
        for line in lines[:10]:
            parts = re.split(r"\s{2,}", line.strip())
            for idx, part in enumerate(parts):
                while idx >= len(samples):
                    samples.append([])
                samples[idx].append(part.strip())
    result = []
    for col_samples in samples:
        if not col_samples:
            result.append("text")
            continue
        blob = " ".join(col_samples)
        if "%" in blob:
            result.append("percentage")
        elif DATE_PATTERN.search(blob):
            result.append("date")
        elif any(u in blob for u in ["元", "万元", "亿元"]):
            result.append("amount")
        elif all(NUMBER_PATTERN.fullmatch(s.replace(",", "")) for s in col_samples[:5] if s):
            result.append("number")
        elif CHINESE_PATTERN.search(blob):
            result.append("text")
        else:
            result.append("text")
    return result


def column_type_match_score(a: Region, b: Region) -> float:
    types_a = infer_column_types(a.text)
    types_b = infer_column_types(b.text)
    if not types_a or not types_b:
        return 0.0
    size = min(len(types_a), len(types_b))
    if size == 0:
        return 0.0
    matches = sum(1 for i in range(size) if types_a[i] == types_b[i])
    return matches / max(len(types_a), len(types_b))


def reading_continuity_score(page_a: LayoutPage, page_b: LayoutPage, table_a: Region, table_b: Region) -> float:
    orders_a = sorted(r.reading_order for r in page_a.regions)
    orders_b = sorted(r.reading_order for r in page_b.regions)
    a_is_last = table_a.reading_order == orders_a[-1]
    b_is_first = table_b.reading_order == orders_b[0]
    text_a_last = table_a.text.strip().split("\n")[-1] if table_a.text else ""
    text_b_first = table_b.text.strip().split("\n")[0] if table_b.text else ""
    not_ending = not any(text_a_last.endswith(w) for w in TABLE_END_WORDS)
    not_header = not any(w in text_b_first for w in TABLE_HEADER_WORDS)
    header_like_b = row_looks_like_header(text_b_first)
    pre_text_regions = [
        r for r in page_b.regions
        if r.region_type == "text" and r.reading_order < table_b.reading_order
    ]
    post_text_regions = [
        r for r in page_a.regions
        if r.region_type == "text" and r.reading_order > table_a.reading_order
    ]
    pre_has_continuation = any(is_continuation_label(r.text) for r in pre_text_regions)
    post_has_continuation = any(is_continuation_label(r.text) for r in post_text_regions)
    pre_blockers = [r for r in pre_text_regions if is_blocking_transition_text(r.text)]
    post_blockers = [r for r in post_text_regions if is_blocking_transition_text(r.text)]
    if pre_blockers or post_blockers:
        return 0.05

    explicit_continuation = pre_has_continuation or post_has_continuation
    top_continuation_hint = getattr(table_b, "source", "") == "top_continuation"
    if explicit_continuation or top_continuation_hint:
        if not_ending:
            return 1.0
        return 0.82

    if a_is_last and b_is_first and not_ending and not_header and not header_like_b:
        return 0.88
    if a_is_last and b_is_first and not_ending and header_like_b and table_b.row_count <= 3:
        return 0.82
    if a_is_last and b_is_first:
        return 0.2
    if a_is_last or b_is_first:
        return 0.12
    return 0.2


def build_text_flow(page: LayoutPage) -> str:
    parts = []
    for region in page.regions:
        if region.region_type == "text":
            parts.append(region.text)
        elif region.region_type == "table":
            tag = region.merged_from[0] if region.is_cross_page and region.merged_from else region.region_id
            parts.append(f"[TABLE:{tag}]")
        elif region.region_type == "figure":
            parts.append(f"[FIGURE:{region.region_id}]")
        elif region.region_type == "image":
            parts.append(f"[IMAGE:{region.region_id}]")
    return "\n\n".join(parts)


def analyze_page_layout(
    doc: fitz.Document,
    page_num: int,
    repeated_lines: Counter[str],
    watermark_patterns: list[str],
    position_dense_counter: dict[tuple[str, int, int], int],
) -> LayoutPage:
    page = doc[page_num]
    page_bbox = [0.0, 0.0, float(page.rect.width), float(page.rect.height)]
    regions: list[Region] = []
    region_counter = 0
    text_dict = page.get_text("dict")
    filter_regions = [
        *detect_header_footer_regions(page, text_dict, repeated_lines),
        *detect_text_watermarks(page, text_dict, watermark_patterns, position_dense_counter),
    ]

    text_blocks = extract_text_blocks(page)
    image_blocks = extract_image_blocks(page)
    table_regions_pymupdf = detect_pymupdf_tables(page)
    table_regions_pipe = detect_pipe_tables(page)
    table_regions_multiline = detect_multiline_row_tables(page)
    table_regions_top = detect_top_continuation_table(page)
    table_regions_line = detect_table_by_lines(page)
    table_regions_wireless = detect_table_by_text_alignment(page, filter_regions=filter_regions)

    table_candidates = (
        table_regions_pymupdf
        + table_regions_top
        + table_regions_pipe
        + table_regions_multiline
        + table_regions_line
        + table_regions_wireless
    )
    table_areas = dedupe_table_candidates(table_candidates)
    table_areas = merge_same_page_continuation_tables(table_areas, text_blocks)

    flowchart_like_table_indices: set[int] = set()
    for idx, ta in enumerate(table_areas):
        candidate_texts = ta.get("text_lines") or collect_texts_in_bbox(text_blocks, ta["bbox"])
        if table_looks_like_flowchart(candidate_texts):
            flowchart_like_table_indices.add(idx)

    table_boxes = [t["bbox"] for idx, t in enumerate(table_areas) if idx not in flowchart_like_table_indices]
    embedded_visual_counts: dict[int, int] = {}
    for idx, ta in enumerate(table_areas):
        if idx in flowchart_like_table_indices:
            continue
        table_bbox = ta["bbox"]
        embedded_visual_counts[idx] = sum(
            1
            for img in image_blocks
            if is_inside(img["bbox"], table_bbox)
            or overlap_ratio(img["bbox"], table_bbox) >= 0.45
            or overlap_ratio(table_bbox, img["bbox"]) >= 0.7
        )

    used_text_indices: set[int] = set()
    for ti, ta in enumerate(table_areas):
        region_counter += 1
        if ta.get("is_pipe_table"):
            cell_texts = ta.get("text_lines", [])
        else:
            cell_texts = ta.get("text_lines") or collect_texts_in_bbox(text_blocks, ta["bbox"])
        if ti in flowchart_like_table_indices:
            nearby_text = extract_nearby_text(page, ta["bbox"])
            figure_sub_type = "flowchart" if looks_like_flowchart_text("\n".join(cell_texts) + "\n" + nearby_text) else "mixed_image"
            regions.append(Region(
                region_id=f"p{page_num+1}_ft{ti+1}",
                page_index=page_num + 1,
                region_type="figure",
                sub_type=figure_sub_type,
                bbox=ta["bbox"],
                text="\n".join(cell_texts),
                confidence=0.9,
                source=str(ta.get("detector") or "flowchart_table"),
                col_count=int(ta.get("col_count") or 0),
                row_count=int(ta.get("row_count") or 0),
            ))
            continue
        table_sub_type = infer_table_sub_type(
            page,
            ta["bbox"],
            cell_texts,
            embedded_visual_counts.get(ti, 0),
        )
        regions.append(Region(
            region_id=f"p{page_num+1}_t{ti+1}",
            page_index=page_num + 1,
            region_type="table",
            sub_type=table_sub_type,
            bbox=ta["bbox"],
            text="\n".join(cell_texts),
            confidence=0.96 if ta.get("is_pipe_table") else (0.9 if not ta.get("is_wireless") else 0.8),
            source=str(ta.get("detector") or "pymupdf"),
            col_count=int(ta.get("col_count") or 0),
            row_count=int(ta.get("row_count") or 0),
        ))
        for idx, tb in enumerate(text_blocks):
            if (
                is_inside(tb["bbox"], ta["bbox"])
                or overlap_ratio(tb["bbox"], ta["bbox"]) >= 0.12
                or overlap_ratio(ta["bbox"], tb["bbox"]) >= 0.35
            ):
                used_text_indices.add(idx)

    for idx, tb in enumerate(text_blocks):
        if idx in used_text_indices:
            continue
        if is_filtered_line(tb["bbox"], filter_regions):
            continue
        text = tb["text"]
        region_counter += 1
        if KEY_VALUE_PATTERN.search(text) and len(text.splitlines()) <= 6:
            sub_type = "key_value"
        elif len(text) < 60 and text.endswith("："):
            sub_type = "title"
        else:
            sub_type = "paragraph"
        regions.append(Region(
            region_id=f"p{page_num+1}_r{region_counter}",
            page_index=page_num + 1,
            region_type="text",
            sub_type=sub_type,
            bbox=tb["bbox"],
            text=text,
            confidence=0.85 if sub_type == "title" else 0.8,
        ))

    complex_figures = detect_figure_regions(page, table_boxes)
    visual_regions = cluster_visual_candidates(page, image_blocks, complex_figures, table_boxes, text_blocks)
    image_counter = 0
    figure_counter = 0
    for visual in visual_regions:
        if any(is_inside(visual["bbox"], r.bbox) for r in regions):
            continue
        if any(overlap_ratio(visual["bbox"], tb["bbox"]) > 0.3 for tb in table_regions_line + table_regions_wireless):
            continue
        region_counter += 1
        if visual["region_type"] == "image":
            image_counter += 1
            region_id = f"p{page_num+1}_i{image_counter}"
        else:
            figure_counter += 1
            region_id = f"p{page_num+1}_f{figure_counter}"
        regions.append(Region(
            region_id=region_id,
            page_index=page_num + 1,
            region_type=visual["region_type"],
            sub_type=str(visual["sub_type"]),
            bbox=list(visual["bbox"]),
            text=str(visual["nearby_text"]),
            confidence=float(visual["confidence"]),
            source=str(visual["source"]),
            xref=int(visual.get("xref") or 0),
        ))

    regions = assign_reading_order(regions, page_bbox)
    page_type, sub_type = infer_page_type(regions, page_bbox)
    page = LayoutPage(
        page_index=page_num + 1,
        page_bbox=page_bbox,
        page_type=page_type,
        sub_type=sub_type,
        regions=regions,
    )
    page.text_flow = build_text_flow(page)
    return page


def main() -> None:
    started_at = time.time()
    config = load_config()
    pdf_path = resolve_pdf_path(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"PDF={pdf_path}")
    log(f"页码范围={DEFAULT_PAGE_RANGE}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    if DEFAULT_PAGE_RANGE:
        target_pages = [page_num for page_num in DEFAULT_PAGE_RANGE if page_num < total_pages]
    else:
        target_pages = list(range(total_pages))
    preview_pages = [doc.load_page(index) for index in target_pages]
    repeated_lines = collect_repeated_lines(preview_pages)
    watermark_patterns = load_watermark_patterns()
    position_dense_counter = collect_position_dense_candidates(preview_pages)

    pages: list[LayoutPage] = []
    for page_num in target_pages:
        log(f"分析第 {page_num + 1} 页...")
        page_start = time.time()
        layout_page = analyze_page_layout(
            doc,
            page_num,
            repeated_lines,
            watermark_patterns,
            position_dense_counter,
        )
        pages.append(layout_page)
        log(
            f"第 {page_num + 1} 页完成，区域数={len(layout_page.regions)}，"
            f"类型={layout_page.page_type}/{layout_page.sub_type}，耗时={time.time() - page_start:.2f}s"
        )

    log("开始跨页表格检测...")
    cross_page_tables, cross_page_debug = detect_cross_page_tables(pages, config)
    log(f"跨页表格检测完成，候选={len(cross_page_debug)}，合并={len(cross_page_tables)}")

    for page in pages:
        page.text_flow = build_text_flow(page)

    text_flow_output = {
        "pdf": str(pdf_path),
        "pages": [
            {
                "page_index": page.page_index,
                "page_type": page.page_type,
                "sub_type": page.sub_type,
                "text_flow": page.text_flow,
            }
            for page in pages
        ],
    }

    structure_objects = {
        "tables": [],
        "figures": [],
        "images": [],
    }

    merged_source_region_ids = {
        source_region_id
        for merged_table in cross_page_tables
        for source_region_id in merged_table.merged_from
    }
    for ct in cross_page_tables:
        structure_objects["tables"].append(ct.to_dict())

    for page in pages:
        for region in page.regions:
            if region.region_type == "table" and region.region_id not in merged_source_region_ids:
                structure_objects["tables"].append(region.to_dict())
            elif region.region_type == "figure":
                structure_objects["figures"].append(region.to_dict())
            elif region.region_type == "image":
                structure_objects["images"].append(region.to_dict())

    final_output = {
        "pdf": str(pdf_path),
        "total_pages_analyzed": len(pages),
        "table_summary": {
            "cross_page_merged_tables": len(cross_page_tables),
            "final_table_count": len(structure_objects["tables"]),
        },
        "layout_engine": {
            "primary": "PyMuPDF",
            "notes": [
                "阶段1主链路使用 PyMuPDF 做版面分析。",
                "执行顺序：文本/图片/绘图提取 -> 表格检测 -> 图表分类 -> 阅读顺序恢复 -> 跨页表格检测与合并。",
                "表格分为有线框与无线文本对齐两路检测。",
                "图表分类基于线条特征、关键词、坐标轴和填充框。",
            ],
        },
        "technology_notes": [
            {
                "step": "layout_segmentation",
                "technology": "PyMuPDF",
                "description": "提取文本块、图片块、绘制对象并生成 bbox、类型和阅读顺序。",
            },
            {
                "step": "table_detection",
                "technology": "线条聚类 + 文本对齐",
                "description": "同时使用有线框表格与无线文本对齐表格两路规则。",
            },
            {
                "step": "figure_classification",
                "technology": "规则分类",
                "description": "基于线条、箭头、坐标轴、填充框和关键词判断 data_viz / flowchart / mixed_image / inline_image。",
            },
            {
                "step": "cross_page_table_detection",
                "technology": "列对齐 + 列类型匹配 + 阅读连续性",
                "description": "扫描相邻页表格候选，满足阈值后合并为新表对象。",
            },
            {
                "step": "reading_order_recovery",
                "technology": "y-x 排序",
                "description": "按从上到下、从左到右恢复顺序，并在文本流中插入对象标志。",
            },
        ],
        "text_flow": text_flow_output,
        "structure_objects": structure_objects,
        "cross_page_table_debug": cross_page_debug,
        "repeated_header_footer_lines": repeated_lines,
        "page_details": [page.to_dict() for page in pages],
    }

    output_path = OUTPUT_DIR / "stage1_layout_analysis.json"
    output_path.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"结果写出完成: {output_path}")
    log(f"总耗时: {time.time() - started_at:.2f}s")
    log(f"分析页数: {len(pages)}")
    log(f"表格数: {len(structure_objects['tables'])}")
    log(f"图表数: {len(structure_objects['figures'])}")
    log(f"图片数: {len(structure_objects['images'])}")


if __name__ == "__main__":
    main()
