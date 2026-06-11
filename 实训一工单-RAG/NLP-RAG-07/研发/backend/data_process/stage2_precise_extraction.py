from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber

try:
    import camelot  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    camelot = None
    CAMELLOT_IMPORT_ERROR = exc
else:
    CAMELLOT_IMPORT_ERROR = None

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)


PROJECT_ROOT = Path(
    os.environ.get("NLP_RAG_PROJECT_ROOT") or Path(__file__).resolve().parents[3]
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "pdf_intelligence_config.json"
STAGE1_PATH = PROJECT_ROOT / "artifacts" / "stage1_smoke_test" / "stage1_layout_analysis.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "stage2_precise_extraction"

LINES_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "intersection_tolerance": 5,
    "snap_tolerance": 4,
    "join_tolerance": 4,
    "edge_min_length": 12,
}
TEXT_TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "text_x_tolerance": 3,
    "text_y_tolerance": 3,
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "min_words_vertical": 2,
    "min_words_horizontal": 1,
}
EMPTY_CELL_RE = re.compile(r"^\s*$")
MULTILINE_RE = re.compile(r"\n")
NUMBERISH_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?(?:%|万元|亿元|元)?$")
SUMMARY_WORDS = ("合计", "小计", "总计", "续表")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
STEP_RE = re.compile(r"\d+[、.]")
TABLE_BBOX_PADDING = {
    "left": 6.0,
    "right": 6.0,
    "top": 28.0,
    "bottom": 20.0,
}


@dataclass
class ExtractedTable:
    table_id: str
    source_region_id: str
    page_index: int
    bbox: list[float]
    sub_type: str
    strategy: str
    quality_score: float
    headers: list[str]
    rows: list[list[str]]
    col_count: int
    row_count: int
    extraction_backend: str
    is_cross_page_member: bool
    merged_from: list[str]
    raw_text_fallback_used: bool
    needs_vlm: bool
    vlm_reasons: list[str]
    extraction_notes: list[str]
    complexity_score: float = 0.0
    complexity_reasons: list[str] = None
    final_object_id: str = ""
    crop_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "source_region_id": self.source_region_id,
            "page_index": self.page_index,
            "bbox": self.bbox,
            "sub_type": self.sub_type,
            "strategy": self.strategy,
            "quality_score": round(self.quality_score, 4),
            "headers": self.headers,
            "rows": self.rows,
            "col_count": self.col_count,
            "row_count": self.row_count,
            "extraction_backend": self.extraction_backend,
            "is_cross_page_member": self.is_cross_page_member,
            "merged_from": self.merged_from,
            "raw_text_fallback_used": self.raw_text_fallback_used,
            "needs_vlm": self.needs_vlm,
            "vlm_reasons": self.vlm_reasons,
            "extraction_notes": self.extraction_notes,
            "complexity_score": round(self.complexity_score, 4),
            "complexity_reasons": self.complexity_reasons or [],
            "final_object_id": self.final_object_id,
            "crop_path": self.crop_path,
        }


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[stage2 {timestamp}] {message}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    return load_json(CONFIG_PATH)


def normalize_text(text: str) -> str:
    text = str(text or "").replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clamp_bbox(bbox: list[float], page_rect: fitz.Rect) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0 = max(page_rect.x0, min(page_rect.x1, x0))
    x1 = max(page_rect.x0, min(page_rect.x1, x1))
    y0 = max(page_rect.y0, min(page_rect.y1, y0))
    y1 = max(page_rect.y0, min(page_rect.y1, y1))
    if x1 <= x0:
        x1 = min(page_rect.x1, x0 + 1)
    if y1 <= y0:
        y1 = min(page_rect.y1, y0 + 1)
    return [x0, y0, x1, y1]


def expand_bbox(
    bbox: list[float],
    page_rect: fitz.Rect,
    *,
    left: float = 0.0,
    right: float = 0.0,
    top: float = 0.0,
    bottom: float = 0.0,
) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    expanded = [x0 - left, y0 - top, x1 + right, y1 + bottom]
    return clamp_bbox(expanded, page_rect)


def bbox_to_pdfplumber(bbox: list[float], page_height: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return (x0, page_height - y1, x1, page_height - y0)


def build_region_pdf_for_camelot(
    src_doc: fitz.Document,
    page_index: int,
    bbox: list[float],
    temp_dir: Path,
) -> Path:
    src_page = src_doc.load_page(page_index - 1)
    padded_bbox = expand_bbox(bbox, src_page.rect, **TABLE_BBOX_PADDING)
    rect = fitz.Rect(*padded_bbox)
    out_path = temp_dir / f"page_{page_index:04d}_{int(rect.x0)}_{int(rect.y0)}_{int(rect.x1)}_{int(rect.y1)}.pdf"
    region_doc = fitz.open()
    region_page = region_doc.new_page(width=rect.width, height=rect.height)
    region_page.show_pdf_page(region_page.rect, src_doc, page_index - 1, clip=rect)
    region_doc.save(out_path)
    region_doc.close()
    return out_path


def iter_target_pdfs(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.pdf"))


def build_layout_lookup(stage1: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(page["page_index"]): page for page in stage1.get("page_details", [])}


def extract_region_text_precisely(page: fitz.Page, bbox: list[float]) -> str:
    rect = fitz.Rect(*clamp_bbox(bbox, page.rect))
    words = page.get_text("words", clip=rect)
    if words:
        words_sorted = sorted(words, key=lambda item: (round(item[3], 1), item[0]))
        lines: list[list[str]] = []
        current_y: float | None = None
        current_line: list[str] = []
        for x0, y0, x1, y1, word, *_ in words_sorted:
            if current_y is None or abs(y1 - current_y) <= 3.0:
                current_line.append(str(word))
                current_y = y1 if current_y is None else current_y
            else:
                lines.append(current_line)
                current_line = [str(word)]
                current_y = y1
        if current_line:
            lines.append(current_line)
        return normalize_text("\n".join(" ".join(line) for line in lines))
    return normalize_text(page.get_textbox(rect))


def is_numberish_cell(cell: str) -> bool:
    value = normalize_text(cell).replace(" ", "")
    if not value:
        return False
    return bool(NUMBERISH_RE.match(value))


def first_nonempty_cell(row: list[str]) -> str:
    for cell in row:
        value = normalize_text(cell)
        if value:
            return value
    return ""


def cell_count_nonempty(row: list[str]) -> int:
    return sum(1 for cell in row if normalize_text(cell))


def is_labelish_cell(cell: str) -> bool:
    value = normalize_text(cell)
    if not value:
        return False
    if len(value) > 24:
        return False
    return bool(CJK_RE.search(value)) or any(keyword in value for keyword in ("日期", "名称", "类别", "简介", "负责人", "关系", "业务", "展示", "图例", "比例"))


def is_summary_row(row: list[str]) -> bool:
    lead = first_nonempty_cell(row)
    return bool(lead) and any(word in lead for word in SUMMARY_WORDS)


def is_kv_like_single_row(row: list[str]) -> bool:
    nonempty = [normalize_text(cell) for cell in row if normalize_text(cell)]
    if len(nonempty) < 2 or len(nonempty) > 4:
        return False
    if len(nonempty) == 2 and is_labelish_cell(nonempty[0]) and len(nonempty[1]) >= 1:
        return True
    if len(nonempty) >= 3 and sum(1 for cell in nonempty if is_labelish_cell(cell)) >= 2:
        return True
    return False


def is_single_row_table_candidate(grid: list[list[str]]) -> bool:
    if not grid:
        return False
    nonempty_rows = [row for row in grid if any(normalize_text(cell) for cell in row)]
    if len(nonempty_rows) != 1:
        return False
    row = nonempty_rows[0]
    col_count = len(row)
    numberish_count = sum(1 for cell in row if is_numberish_cell(cell))
    if is_summary_row(row) and col_count >= 2:
        return True
    if is_kv_like_single_row(row):
        return True
    return col_count >= 3 and numberish_count >= 2


def is_good_table_grid(grid: list[list[str]]) -> bool:
    if not grid:
        return False
    max_cols = max((len(row) for row in grid), default=0)
    if max_cols < 2:
        return False
    nonempty_rows = sum(1 for row in grid if any(normalize_text(cell) for cell in row))
    return nonempty_rows >= 2 or is_single_row_table_candidate(grid)


def normalize_grid(grid: list[list[Any]]) -> list[list[str]]:
    normalized: list[list[str]] = []
    width = max((len(row or []) for row in grid), default=0)
    for row in grid:
        cells = [normalize_text(cell) for cell in (row or [])]
        if len(cells) < width:
            cells.extend([""] * (width - len(cells)))
        normalized.append(cells)
    return normalized


def score_table_grid(grid: list[list[str]]) -> float:
    if not grid:
        return 0.0
    row_count = len(grid)
    col_count = max((len(row) for row in grid), default=0)
    if row_count == 0 or col_count == 0:
        return 0.0
    total_cells = max(1, row_count * col_count)
    nonempty_cells = sum(1 for row in grid for cell in row if normalize_text(cell))
    fill_ratio = nonempty_cells / total_cells
    consistent_cols = sum(1 for row in grid if len(row) == col_count) / row_count
    avg_chars = sum(len(normalize_text(cell)) for row in grid for cell in row) / total_cells
    header_signal = 0.0
    if grid:
        first_row = " ".join(grid[0])
        if any(keyword in first_row for keyword in ["项目", "名称", "金额", "比例", "日期", "收入", "成本", "股东"]):
            header_signal = 1.0
    score = (fill_ratio * 0.45) + (consistent_cols * 0.25) + min(avg_chars / 12.0, 1.0) * 0.15 + (header_signal * 0.15)
    return round(score, 4)


def parse_text_fallback(raw_text: str) -> list[list[str]]:
    lines = [normalize_text(line) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []
    grid: list[list[str]] = []
    for line in lines:
        if "|" in line:
            cells = [normalize_text(cell) for cell in line.split("|")]
            if len(cells) >= 2:
                grid.append(cells)
                continue
        parts = re.split(r"\s{2,}", line)
        parts = [normalize_text(part) for part in parts if normalize_text(part)]
        if len(parts) >= 2:
            grid.append(parts)
    return normalize_grid(grid)


def parse_kv_fallback(raw_text: str) -> list[list[str]]:
    lines = [normalize_text(line) for line in raw_text.splitlines() if normalize_text(line)]
    if not lines:
        return []
    grid: list[list[str]] = []
    for line in lines:
        if "|" in line:
            cells = [normalize_text(cell) for cell in line.split("|") if normalize_text(cell)]
            if 2 <= len(cells) <= 4:
                grid.append(cells)
        else:
            parts = [normalize_text(part) for part in re.split(r"\s{2,}", line) if normalize_text(part)]
            if 2 <= len(parts) <= 4:
                grid.append(parts)
    return normalize_grid(grid)


def looks_visual_non_table(region: dict[str, Any], raw_text: str) -> tuple[bool, str]:
    sub_type = str(region.get("sub_type") or "")
    if sub_type in {"decorative_table", "table_with_embedded_image"} and not bool(region.get("is_cross_page")):
        return True, f"sub_type_{sub_type}"
    text = normalize_text(raw_text)
    if not text:
        return False, ""
    token_source = text.replace("\n", " | ")
    tokens = [normalize_text(token) for token in token_source.split("|") if normalize_text(token)]
    short_ratio = (sum(1 for token in tokens if len(token) <= 2) / len(tokens)) if tokens else 0.0
    if any(keyword in text for keyword in ("图例", "产品展示", "原型机", "打样", "采购任务")):
        return True, "legend_or_visual_board"
    if short_ratio >= 0.6 and len(tokens) >= 8 and any("%" in token for token in tokens):
        return True, "equity_or_org_chart_like"
    if len(STEP_RE.findall(text)) >= 3 and ("系统" in text or "采购" in text or "安装" in text):
        return True, "flowchart_like"
    return False, ""


def score_camelot_candidate(table: Any, grid: list[list[str]]) -> float:
    base_score = score_table_grid(grid)
    parsing_report = getattr(table, "parsing_report", {}) or {}
    accuracy = float(parsing_report.get("accuracy", 0.0) or 0.0) / 100.0
    whitespace = float(parsing_report.get("whitespace", 0.0) or 0.0) / 100.0
    combined = (base_score * 0.65) + (accuracy * 0.3) + (max(0.0, 1.0 - whitespace) * 0.05)
    return round(min(1.0, combined), 4)


def extract_camelot_candidates(region_pdf_path: Path) -> list[tuple[str, str, list[list[str]], float, bool]]:
    if camelot is None:
        return []
    candidates: list[tuple[str, str, list[list[str]], float, bool]] = []
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(
                str(region_pdf_path),
                pages="1",
                flavor=flavor,
                suppress_stdout=True,
            )
        except Exception:
            continue
        for table in tables or []:
            try:
                grid = normalize_grid(table.df.values.tolist())
            except Exception:
                continue
            if not is_good_table_grid(grid):
                continue
            score = score_camelot_candidate(table, grid)
            candidates.append(("camelot", f"camelot_{flavor}", grid, score, False))
    return candidates


def extract_best_table_from_region(
    src_doc: fitz.Document,
    temp_dir: Path,
    pdf_path: Path,
    plumber_page: pdfplumber.page.Page,
    region: dict[str, Any],
    page_index: int,
) -> tuple[str, str, list[list[str]], float, bool]:
    bbox = bbox_to_pdfplumber(region["bbox"], plumber_page.height)
    cropped = plumber_page.crop(bbox)
    candidates: list[tuple[str, str, list[list[str]], float, bool]] = []
    if camelot is not None:
        try:
            region_pdf_path = build_region_pdf_for_camelot(src_doc, page_index, list(region["bbox"]), temp_dir)
            candidates.extend(extract_camelot_candidates(region_pdf_path))
        except Exception:
            pass
    for name, settings in [("lines", LINES_TABLE_SETTINGS), ("text", TEXT_TABLE_SETTINGS)]:
        try:
            tables = cropped.extract_tables(settings)
        except Exception:
            tables = []
        for table in tables or []:
            grid = normalize_grid(table)
            if not is_good_table_grid(grid):
                continue
            candidates.append(("pdfplumber", name, grid, score_table_grid(grid), False))
    fallback_grid = parse_text_fallback(str(region.get("text") or ""))
    if is_good_table_grid(fallback_grid):
        candidates.append(("text_fallback", "text_fallback", fallback_grid, score_table_grid(fallback_grid), True))
    if not candidates:
        kv_grid = parse_kv_fallback(str(region.get("text") or ""))
        if is_good_table_grid(kv_grid):
            return ("text_fallback", "kv_fallback", kv_grid, score_table_grid(kv_grid), True)
        route_vlm, reason = looks_visual_non_table(region, str(region.get("text") or ""))
        if route_vlm:
            return ("vlm_route", f"route_vlm:{reason}", [], 0.0, True)
        return ("none", "none", [], 0.0, True)
    candidates.sort(
        key=lambda item: (item[3], len(item[2]), max((len(r) for r in item[2]), default=0)),
        reverse=True,
    )
    return candidates[0]


def split_headers_and_rows(grid: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    if not grid:
        return [], []
    if len(grid) == 1:
        return ([], grid) if is_single_row_table_candidate(grid) else (grid[0], [])
    return grid[0], grid[1:]


def is_summary_fragment_table(table: ExtractedTable) -> bool:
    if table.row_count == 0:
        return False
    if table.headers and not table.rows:
        return is_summary_row(table.headers)
    if len(table.rows) == 1 and is_summary_row(table.rows[0]):
        return True
    return False


def stitch_summary_fragments(raw_tables: list[ExtractedTable]) -> tuple[list[ExtractedTable], dict[str, ExtractedTable]]:
    stitched: list[ExtractedTable] = []
    table_map: dict[str, ExtractedTable] = {}
    ordered = sorted(raw_tables, key=lambda item: (item.page_index, item.bbox[1], item.bbox[0]))
    for table in ordered:
        merged = False
        if stitched and is_summary_fragment_table(table):
            prev = stitched[-1]
            gap = float(table.bbox[1]) - float(prev.bbox[3])
            same_page = table.page_index == prev.page_index
            similar_cols = abs(table.col_count - prev.col_count) <= 2
            if same_page and gap <= 90.0 and similar_cols:
                fragment_rows = table.rows[:] if table.rows else ([table.headers[:]] if table.headers else [])
                if fragment_rows:
                    prev.rows.extend(fragment_rows)
                    prev.row_count = (1 if prev.headers else 0) + len(prev.rows)
                    prev.bbox = [
                        min(float(prev.bbox[0]), float(table.bbox[0])),
                        min(float(prev.bbox[1]), float(table.bbox[1])),
                        max(float(prev.bbox[2]), float(table.bbox[2])),
                        max(float(prev.bbox[3]), float(table.bbox[3])),
                    ]
                    if prev.extraction_backend == "none" and table.extraction_backend != "none":
                        prev.extraction_backend = table.extraction_backend
                        prev.strategy = table.strategy
                        prev.quality_score = max(prev.quality_score, table.quality_score)
                    prev.vlm_reasons = sorted(set(prev.vlm_reasons + ["summary_fragment_stitched"]))
                    prev.needs_vlm = prev.needs_vlm or table.needs_vlm
                    table_map[table.source_region_id] = prev
                    log(
                        "summary_fragment_stitched "
                        f"fragment={table.source_region_id} into={prev.source_region_id} "
                        f"page={table.page_index} gap={gap:.1f}"
                    )
                    merged = True
        if not merged:
            stitched.append(table)
            table_map[table.source_region_id] = table
    for table in stitched:
        table_map[table.source_region_id] = table
    return stitched, table_map


def table_vlm_reasons(
    grid: list[list[str]],
    region: dict[str, Any],
    quality_score: float,
    config: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    max_rows = int(config.get("table_partition_max_rows", 50))
    row_count = len(grid)
    col_count = max((len(row) for row in grid), default=0)
    total_cells = max(1, row_count * max(1, col_count))
    empty_cells = sum(1 for row in grid for cell in row if EMPTY_CELL_RE.match(cell))
    multiline_cells = sum(1 for row in grid for cell in row if MULTILINE_RE.search(cell))
    if row_count > max_rows:
        reasons.append(f"row_count_gt_{max_rows}")
    if bool(region.get("is_cross_page")):
        reasons.append("cross_page_table")
    if quality_score < 0.62:
        reasons.append("low_extraction_quality")
    if total_cells and (empty_cells / total_cells) > 0.35:
        reasons.append("many_empty_cells")
    if multiline_cells >= 4:
        reasons.append("many_multiline_cells")
    if region.get("sub_type") in {"table_with_embedded_image", "decorative_table"}:
        reasons.append(f"sub_type_{region.get('sub_type')}")
    return reasons


def crop_region_image(
    page: fitz.Page,
    bbox: list[float],
    output_path: Path,
    scale: float = 2.0,
) -> None:
    rect = fitz.Rect(*clamp_bbox(bbox, page.rect))
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(output_path)


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def cleanup_temp_dir(temp_dir: Path, retries: int = 6, delay_seconds: float = 0.75) -> bool:
    if not temp_dir.exists():
        return True
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(temp_dir)
            if attempt > 1:
                log(f"temp_cleanup_recovered dir={temp_dir} attempt={attempt}")
            return True
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay_seconds * attempt)
    log(f"temp_cleanup_failed dir={temp_dir} error={type(last_error).__name__}: {last_error}")
    return False


def cleanup_temp_dir_quietly(temp_dir: Path, retries: int = 3, delay_seconds: float = 0.25) -> None:
    if not temp_dir.exists():
        return
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(temp_dir)
            return
        except Exception:
            if attempt < retries:
                time.sleep(delay_seconds * attempt)


def patch_camelot_tempdir() -> None:
    if camelot is None:
        return
    try:
        import camelot.handlers as camelot_handlers  # type: ignore
        import camelot.utils as camelot_utils  # type: ignore
        import atexit
    except Exception as exc:
        log(f"camelot_temp_patch_skipped error={type(exc).__name__}: {exc}")
        return

    if getattr(camelot_handlers.TemporaryDirectory, "__name__", "") == "SafeCamelotTemporaryDirectory":
        return

    class SafeCamelotTemporaryDirectory:
        def __enter__(self) -> str:
            self.name = tempfile.mkdtemp(prefix="camelot_runtime_")
            atexit.register(cleanup_temp_dir_quietly, Path(self.name))
            return self.name

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

    camelot_handlers.TemporaryDirectory = SafeCamelotTemporaryDirectory
    camelot_utils.TemporaryDirectory = SafeCamelotTemporaryDirectory
    log("camelot_temp_patch_applied")


def compute_table_complexity(table: ExtractedTable, config: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    backend_penalty = {
        "camelot": 0.12 if table.strategy == "camelot_stream" else 0.05,
        "pdfplumber": 0.18 if table.strategy == "text" else 0.12,
        "text_fallback": 0.26,
        "vlm_route": 0.7,
        "none": 1.0,
    }.get(table.extraction_backend, 0.25)
    score += backend_penalty
    if table.quality_score < 0.82:
        penalty = min(0.35, max(0.0, 0.82 - table.quality_score))
        score += penalty
        reasons.append("quality_below_target")
    grid = ([table.headers] if table.headers else []) + table.rows
    total_cells = max(1, table.row_count * max(1, table.col_count))
    empty_cells = sum(1 for row in grid for cell in row if EMPTY_CELL_RE.match(str(cell)))
    multiline_cells = sum(1 for row in grid for cell in row if MULTILINE_RE.search(str(cell)))
    empty_ratio = empty_cells / total_cells
    multiline_ratio = multiline_cells / total_cells
    if empty_ratio > 0.18:
        score += min(0.24, empty_ratio * 0.35)
        reasons.append("empty_cell_ratio_high")
    if multiline_ratio > 0.08:
        score += min(0.2, multiline_ratio * 0.5)
        reasons.append("multiline_cell_ratio_high")
    max_rows = int(config.get("table_partition_max_rows", 50))
    if table.row_count > max_rows:
        score += 0.18
        reasons.append("row_count_high")
    if table.col_count >= 8:
        score += 0.08
        reasons.append("col_count_high")
    if table.is_cross_page_member:
        score += 0.22
        reasons.append("cross_page_member")
    if table.sub_type in {"table_with_embedded_image", "decorative_table"}:
        score += 0.2
        reasons.append(f"sub_type_{table.sub_type}")
    if table.strategy == "kv_fallback":
        score += 0.05
        reasons.append("kv_fallback")
    merged_cell_suspected = False
    for row in grid:
        if len(row) < 3:
            continue
        empties = sum(1 for cell in row if not normalize_text(str(cell)))
        if empties >= max(1, len(row) // 2):
            merged_cell_suspected = True
            break
    if merged_cell_suspected:
        score += 0.12
        reasons.append("merged_cell_suspected")
    for reason in table.vlm_reasons:
        if reason not in reasons:
            reasons.append(reason)
    return round(min(1.0, score), 4), reasons


def build_table_chains(
    stage1: dict[str, Any],
    raw_index: dict[str, ExtractedTable],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    chains: list[dict[str, Any]] = []
    fragment_to_chain: dict[str, str] = {}
    for table in stage1.get("structure_objects", {}).get("tables", []):
        if table.get("source") != "cross_page_chain":
            continue
        chain_id = str(table["region_id"])
        members: list[ExtractedTable] = []
        seen_final: set[str] = set()
        for source_region_id in table.get("merged_from") or []:
            member = raw_index.get(str(source_region_id))
            if member is None:
                continue
            if member.source_region_id in seen_final:
                continue
            seen_final.add(member.source_region_id)
            members.append(member)
            fragment_to_chain[member.source_region_id] = chain_id
        if not members:
            continue
        relation = {
            "chain_id": chain_id,
            "page_start": table.get("page_start"),
            "page_end": table.get("page_end"),
            "fragment_region_ids": [member.source_region_id for member in members],
            "fragment_object_ids": [],
            "fragment_pages": [member.page_index for member in members],
            "bbox": table.get("bbox"),
            "sub_type": table.get("sub_type"),
            "chain_type": "cross_page_table",
            "requires_vlm_merge": True,
            "relation_reason": "stage1_cross_page_chain",
        }
        chains.append(relation)
    return chains, fragment_to_chain


def load_vlm_results(output_dir: Path) -> dict[str, dict[str, Any]]:
    candidates = [
        output_dir / "vlm_results.jsonl",
        output_dir / "multimodal_results.jsonl",
    ]
    result_map: dict[str, dict[str, Any]] = {}
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                final_object_id = str(payload.get("final_object_id") or "")
                if final_object_id:
                    result_map[final_object_id] = payload
    return result_map


def table_task_prompt(table: ExtractedTable) -> str:
    return (
        "你将收到一个单页表格的截图与机器提取结果。"
        "请校对表头、合并单元格、单元格换行和数字列，输出标准 JSON。"
        "如果机器提取错位，请以图片为准纠正。"
        "输出字段必须包含 headers、rows、notes，不要输出解释性长文本。"
    )


def chain_task_prompt() -> str:
    return (
        "你将收到同一张跨页表格的多个页内片段截图、页码顺序和每页的结构化提取结果。"
        "请判断这些片段是否属于同一张表；若属于，请统一表头、去掉重复表头、处理续表与合计行，"
        "输出合并后的标准 JSON。输出字段必须包含 headers、rows、notes、merge_decision。"
        "如果某页存在合并单元格或复杂换行，请以图片为准恢复。"
    )


def figure_task_prompt(region_type: str) -> str:
    if region_type == "image":
        return "请识别该图片的关键信息，输出结构化 JSON，字段包含 summary、labels、numbers、notes。"
    return "请识别该图表/结构图的关键信息，输出结构化 JSON，字段包含 summary、labels、numbers、notes。"


def grouped_figure_task_prompt(task: dict[str, Any]) -> str:
    region_count = len(task.get("regions") or [])
    region_types = sorted({str(item.get("region_type") or "") for item in (task.get("regions") or [])})
    region_types_text = "/".join(item for item in region_types if item) or "figure_or_image"
    return (
        "请把同一页的图表/图片区域作为一个整体来判断。"
        "先判断这些裁剪区域是否属于同一个完整视觉对象的不同部分，还是多个彼此独立的对象；"
        "再输出页面级摘要、关键信息、数字和关系。"
        f"当前页面共 {region_count} 个区域，类型={region_types_text}。"
    )


def build_multimodal_objects(
    doc: fitz.Document,
    page_text_flow: list[dict[str, Any]],
    raw_tables: list[ExtractedTable],
    table_chains: list[dict[str, Any]],
    figure_tasks: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    vlm_results = load_vlm_results(output_dir)
    tasks: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    reference_map: dict[str, str] = {}
    crop_root = output_dir / "crops" / "tables"
    fragment_to_chain = {rid: chain["chain_id"] for chain in table_chains for rid in chain["fragment_region_ids"]}
    chain_registry_ids: dict[str, str] = {}
    for chain in table_chains:
        final_object_id = f"TABLE_FINAL_{chain['chain_id']}"
        chain_registry_ids[chain["chain_id"]] = final_object_id
        for rid in chain["fragment_region_ids"]:
            reference_map[rid] = final_object_id
        chain_result = vlm_results.get(final_object_id)
        chain_entry = {
            "final_object_id": final_object_id,
            "object_type": "cross_page_table",
            "source_ids": chain["fragment_region_ids"],
            "status": "resolved" if chain_result else "pending_vlm",
            "latest_content": chain_result.get("content") if chain_result else None,
            "latest_structured_content": chain_result.get("structured_content") if chain_result else None,
            "content_version": chain_result.get("version", 0) if chain_result else 0,
            "prompt_type": "cross_page_table_merge",
        }
        registry.append(chain_entry)
        task = {
            "task_id": f"vlm_task_{final_object_id}",
            "final_object_id": final_object_id,
            "task_type": "cross_page_table_merge",
            "prompt": chain_task_prompt(),
            "chain_id": chain["chain_id"],
            "fragment_region_ids": chain["fragment_region_ids"],
            "fragment_pages": chain["fragment_pages"],
            "relation": chain,
            "fragments": [],
        }
        tasks.append(task)

    chain_task_map = {task["final_object_id"]: task for task in tasks if task["task_type"] == "cross_page_table_merge"}
    chain_map = {chain["chain_id"]: chain for chain in table_chains}
    mm_threshold = float(config.get("multimodal_upgrade_threshold", 0.7))
    for table in raw_tables:
        table.complexity_score, table.complexity_reasons = compute_table_complexity(table, config)
        crop_path = crop_root / f"{table.source_region_id}.png"
        page = doc.load_page(table.page_index - 1)
        crop_region_image(page, table.bbox, crop_path)
        table.crop_path = str(crop_path)
        fragment_object_id = f"TABLE_FRAGMENT_{table.source_region_id}"
        chain_id = fragment_to_chain.get(table.source_region_id)
        if chain_id:
            final_object_id = chain_registry_ids[chain_id]
            table.final_object_id = final_object_id
            chain_map[chain_id]["fragment_object_ids"].append(fragment_object_id)
            chain_task_map[final_object_id]["fragments"].append(
                {
                    "fragment_object_id": fragment_object_id,
                    "source_region_id": table.source_region_id,
                    "page_index": table.page_index,
                    "crop_path": table.crop_path,
                    "backend": table.extraction_backend,
                    "strategy": table.strategy,
                    "quality_score": table.quality_score,
                    "complexity_score": table.complexity_score,
                    "headers": table.headers,
                    "rows": table.rows,
                    "col_count": table.col_count,
                    "row_count": table.row_count,
                    "complexity_reasons": table.complexity_reasons,
                }
            )
            continue
        final_object_id = f"TABLE_FINAL_{table.source_region_id}"
        table.final_object_id = final_object_id
        reference_map[table.source_region_id] = final_object_id
        needs_vlm = (
            table.extraction_backend in {"vlm_route", "none"}
            or table.needs_vlm
            or table.complexity_score >= mm_threshold
        )
        result = vlm_results.get(final_object_id)
        registry.append(
            {
                "final_object_id": final_object_id,
                "object_type": "table",
                "source_ids": [table.source_region_id],
                "status": "resolved" if result else ("pending_vlm" if needs_vlm else "structured"),
                "latest_content": result.get("content") if result else None,
                "latest_structured_content": result.get("structured_content") if result else {
                    "headers": table.headers,
                    "rows": table.rows,
                },
                "content_version": result.get("version", 0) if result else 0,
                "complexity_score": table.complexity_score,
                "complexity_reasons": table.complexity_reasons,
                "prompt_type": "single_table_understanding" if needs_vlm else "structured_only",
            }
        )
        if needs_vlm:
            tasks.append(
                {
                    "task_id": f"vlm_task_{final_object_id}",
                    "final_object_id": final_object_id,
                    "task_type": "single_table_understanding",
                    "prompt": table_task_prompt(table),
                    "source_region_id": table.source_region_id,
                    "page_index": table.page_index,
                    "crop_path": table.crop_path,
                    "backend": table.extraction_backend,
                    "strategy": table.strategy,
                    "quality_score": table.quality_score,
                    "complexity_score": table.complexity_score,
                    "complexity_reasons": table.complexity_reasons,
                    "headers": table.headers,
                    "rows": table.rows,
                }
            )

    for task in figure_tasks:
        source_region_ids = [str(item) for item in (task.get("source_region_ids") or []) if str(item)]
        if not source_region_ids and task.get("region_id"):
            source_region_ids = [str(task["region_id"])]
        final_object_id = str(task.get("final_object_id") or f"VISUAL_FINAL_PAGE_{int(task['page_index']):04d}")
        for region_id in source_region_ids:
            reference_map[region_id] = final_object_id
        result = vlm_results.get(final_object_id)
        registry.append(
            {
                "final_object_id": final_object_id,
                "object_type": str(task.get("object_type") or "visual_group"),
                "source_ids": source_region_ids,
                "status": "resolved" if result else "pending_vlm",
                "latest_content": result.get("content") if result else None,
                "latest_structured_content": result.get("structured_content") if result else None,
                "content_version": result.get("version", 0) if result else 0,
                "prompt_type": task["prompt_type"],
            }
        )
        task["final_object_id"] = final_object_id
        task["task_type"] = "figure_or_image_understanding"
        task["prompt"] = grouped_figure_task_prompt(task)
        task["task_id"] = f"vlm_task_{final_object_id}"
        tasks.append(task)

    resolved_text_flow: list[dict[str, Any]] = []
    registry_map = {item["final_object_id"]: item for item in registry}
    for page in page_text_flow:
        object_flow: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for item in page["object_flow"]:
            entry = dict(item)
            marker = entry.get("marker")
            if marker:
                region_id = entry["region_id"]
                final_object_id = reference_map.get(region_id, region_id)
                registry_entry = registry_map.get(final_object_id, {})
                resolved_marker = f"[{entry['region_type'].upper()}:{final_object_id}]"
                entry["final_object_id"] = final_object_id
                entry["resolved_status"] = registry_entry.get("status", "unknown")
                entry["resolved_preview"] = normalize_text(str(registry_entry.get("latest_content") or ""))[:120]
                entry["marker"] = resolved_marker
                text_parts.append(resolved_marker)
            else:
                text_parts.append(entry.get("text", ""))
            object_flow.append(entry)
        resolved_text_flow.append(
            {
                "page_index": page["page_index"],
                "page_type": page["page_type"],
                "sub_type": page["sub_type"],
                "object_flow": object_flow,
                "page_text_flow": "\n\n".join(part for part in text_parts if part).strip(),
            }
        )
    return tasks, registry, resolved_text_flow, reference_map


def build_text_regions(
    doc: fitz.Document,
    page_details: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text_regions: list[dict[str, Any]] = []
    page_text_flow: list[dict[str, Any]] = []
    for page_info in page_details:
        page_index = int(page_info["page_index"])
        page = doc.load_page(page_index - 1)
        ordered_regions = sorted(
            page_info.get("regions", []),
            key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])),
        )
        page_objects: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for order_index, region in enumerate(ordered_regions, start=1):
            region_type = str(region.get("region_type") or "")
            if region_type == "text":
                precise_text = extract_region_text_precisely(page, list(region["bbox"]))
                final_text = precise_text or normalize_text(str(region.get("text") or ""))
                if not final_text:
                    continue
                text_regions.append(
                    {
                        "region_id": region["region_id"],
                        "page_index": page_index,
                        "reading_order": order_index,
                        "region_type": region_type,
                        "sub_type": region.get("sub_type"),
                        "bbox": region["bbox"],
                        "text": final_text,
                        "source_backend": "pymupdf_precise_clip",
                        "fallback_used": not bool(precise_text),
                    }
                )
                text_parts.append(final_text)
                page_objects.append(
                    {
                        "region_id": region["region_id"],
                        "region_type": region_type,
                        "sub_type": region.get("sub_type"),
                        "bbox": region["bbox"],
                        "text": final_text,
                        "reading_order": order_index,
                    }
                )
            elif region_type in {"table", "figure", "image"}:
                marker = f"[{region_type.upper()}:{region['region_id']}]"
                text_parts.append(marker)
                page_objects.append(
                    {
                        "region_id": region["region_id"],
                        "region_type": region_type,
                        "sub_type": region.get("sub_type"),
                        "bbox": region["bbox"],
                        "marker": marker,
                        "reading_order": order_index,
                    }
                )
        page_text_flow.append(
            {
                "page_index": page_index,
                "page_type": page_info.get("page_type"),
                "sub_type": page_info.get("sub_type"),
                "object_flow": page_objects,
                "page_text_flow": "\n\n".join(text_parts).strip(),
            }
        )
    return text_regions, page_text_flow


def build_tables(
    pdf_path: Path,
    layout_by_page: dict[int, dict[str, Any]],
    stage1: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[ExtractedTable], dict[str, ExtractedTable]]:
    raw_tables: list[ExtractedTable] = []
    extracted_count = 0
    temp_dir = Path(tempfile.mkdtemp(prefix="stage2_camelot_"))
    try:
        with fitz.open(str(pdf_path)) as src_doc, pdfplumber.open(str(pdf_path)) as plumber_pdf:
            for page_index, page_info in layout_by_page.items():
                plumber_page = plumber_pdf.pages[page_index - 1]
                table_regions = sorted(
                    [region for region in page_info.get("regions", []) if region.get("region_type") == "table"],
                    key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])),
                )
                for region in table_regions:
                    backend, strategy, grid, quality_score, fallback_used = extract_best_table_from_region(
                        src_doc,
                        temp_dir,
                        pdf_path,
                        plumber_page,
                        region,
                        page_index,
                    )
                    headers, rows = split_headers_and_rows(grid)
                    reasons = table_vlm_reasons(grid, region, quality_score, config)
                    extracted = ExtractedTable(
                        table_id=region["region_id"],
                        source_region_id=region["region_id"],
                        page_index=page_index,
                        bbox=list(region["bbox"]),
                        sub_type=str(region.get("sub_type") or ""),
                        strategy=strategy,
                        quality_score=quality_score,
                        headers=headers,
                        rows=rows,
                        col_count=max((len(row) for row in grid), default=0),
                        row_count=len(grid),
                        extraction_backend=backend,
                        is_cross_page_member=bool(region.get("is_cross_page")),
                        merged_from=list(region.get("merged_from") or []),
                        raw_text_fallback_used=fallback_used,
                        needs_vlm=bool(reasons),
                        vlm_reasons=reasons,
                        extraction_notes=[],
                    )
                    if backend == "vlm_route":
                        extracted.needs_vlm = True
                        extracted.vlm_reasons = sorted(set(extracted.vlm_reasons + [strategy]))
                        extracted.extraction_notes = ["visual_non_table_route"]
                    elif strategy == "kv_fallback":
                        extracted.extraction_notes = ["kv_fallback"]
                    elif backend == "none":
                        extracted.extraction_notes = ["table_extract_failed"]
                    raw_tables.append(extracted)
                    extracted_count += 1
                    if strategy == "none" or quality_score < 0.55 or extracted_count % 25 == 0:
                        log(
                            "table_extracted "
                            f"page={page_index} region={region['region_id']} backend={backend} strategy={strategy} "
                            f"rows={extracted.row_count} cols={extracted.col_count} quality={quality_score:.3f} "
                            f"count={extracted_count}"
                        )
    finally:
        cleanup_temp_dir(temp_dir)

    raw_tables, raw_index = stitch_summary_fragments(raw_tables)
    return raw_tables, raw_index


def build_figure_tasks(
    doc: fitz.Document,
    layout_by_page: dict[int, dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    crop_root = output_dir / "crops"
    for page_index, page_info in layout_by_page.items():
        page = doc.load_page(page_index - 1)
        visual_regions: list[dict[str, Any]] = []
        for region in page_info.get("regions", []):
            if region.get("region_type") not in {"figure", "image"}:
                continue
            region_type = str(region["region_type"])
            crop_path = crop_root / region_type / f"{region['region_id']}.png"
            crop_region_image(page, list(region["bbox"]), crop_path)
            visual_regions.append(
                {
                    "region_id": region["region_id"],
                    "page_index": page_index,
                    "region_type": region_type,
                    "sub_type": region.get("sub_type"),
                    "bbox": region["bbox"],
                    "crop_path": str(crop_path),
                    "context_text": normalize_text(str(region.get("text") or "")),
                }
            )
        if not visual_regions:
            continue
        visual_regions.sort(key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])))
        tasks.append(
            {
                "group_id": f"visual_page_{page_index:04d}",
                "page_index": page_index,
                "region_type": "visual_group",
                "object_type": "visual_group",
                "sub_type": "grouped_same_page_visuals",
                "route": "vlm_direct",
                "prompt_type": "figure_or_image_understanding",
                "task_type": "figure_or_image_understanding",
                "source_region_ids": [str(item["region_id"]) for item in visual_regions],
                "crop_paths": [str(item["crop_path"]) for item in visual_regions],
                "regions": visual_regions,
                "region_count": len(visual_regions),
                "final_object_id": f"VISUAL_FINAL_PAGE_{page_index:04d}",
                "context_text": "\n".join(
                    text for text in (str(item.get("context_text") or "") for item in visual_regions) if text
                )[:1000],
            }
        )
    return tasks


def pick_pdf_path(stage1: dict[str, Any]) -> Path:
    pdf_value = stage1.get("pdf")
    if pdf_value:
        pdf_path = Path(str(pdf_value))
        if pdf_path.exists():
            return pdf_path
    data_dir = PROJECT_ROOT / "data"
    pdfs = iter_target_pdfs(data_dir)
    if not pdfs:
        raise FileNotFoundError(f"未在 {data_dir} 找到 PDF 文件")
    return pdfs[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precise extraction based on stage0/stage1 outputs")
    parser.add_argument("--project-root", type=str, default=None, help="Override project root")
    parser.add_argument("--stage1-path", type=str, default=None, help="Override stage1 layout json")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir")
    return parser.parse_args()


def main() -> None:
    global PROJECT_ROOT, STAGE1_PATH, OUTPUT_DIR, CONFIG_PATH
    args = parse_args()
    if args.project_root:
        PROJECT_ROOT = Path(args.project_root)
        CONFIG_PATH = PROJECT_ROOT / "config" / "pdf_intelligence_config.json"
        STAGE1_PATH = PROJECT_ROOT / "artifacts" / "stage1_smoke_test" / "stage1_layout_analysis.json"
        OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "stage2_precise_extraction"
    if args.stage1_path:
        STAGE1_PATH = Path(args.stage1_path)
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    start_time = time.time()
    log(f"project_root={PROJECT_ROOT}")
    log(f"stage1_path={STAGE1_PATH}")
    if not STAGE1_PATH.exists():
        raise FileNotFoundError(f"未找到 stage1 产物: {STAGE1_PATH}")

    config = load_config()
    stage1 = load_json(STAGE1_PATH)
    pdf_path = pick_pdf_path(stage1)
    layout_by_page = build_layout_lookup(stage1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"pdf_path={pdf_path}")
    log(f"pages={len(layout_by_page)}")
    if camelot is None:
        log(f"camelot_unavailable error={type(CAMELLOT_IMPORT_ERROR).__name__}: {CAMELLOT_IMPORT_ERROR}")
    else:
        log(f"camelot_available version={getattr(camelot, '__version__', 'unknown')}")
        patch_camelot_tempdir()
    with fitz.open(str(pdf_path)) as doc:
        text_regions, page_text_flow = build_text_regions(doc, stage1.get("page_details", []))
        figure_tasks = build_figure_tasks(doc, layout_by_page, OUTPUT_DIR)

    raw_tables, raw_index = build_tables(pdf_path, layout_by_page, stage1, config)
    table_chains, _ = build_table_chains(stage1, raw_index)
    with fitz.open(str(pdf_path)) as doc:
        multimodal_tasks, object_registry, resolved_page_text_flow, _ = build_multimodal_objects(
            doc,
            page_text_flow,
            raw_tables,
            table_chains,
            figure_tasks,
            OUTPUT_DIR,
            config,
        )
    backend_counts = Counter(item.extraction_backend for item in raw_tables)

    jsonl_write(OUTPUT_DIR / "text_regions.jsonl", text_regions)
    (OUTPUT_DIR / "page_text_flow.json").write_text(
        json.dumps(page_text_flow, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "page_text_flow_resolved.json").write_text(
        json.dumps(resolved_page_text_flow, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    jsonl_write(OUTPUT_DIR / "tables_raw.jsonl", [item.to_dict() for item in raw_tables])
    jsonl_write(OUTPUT_DIR / "table_chains.jsonl", table_chains)
    jsonl_write(OUTPUT_DIR / "figure_tasks.jsonl", figure_tasks)
    jsonl_write(OUTPUT_DIR / "multimodal_tasks.jsonl", multimodal_tasks)
    jsonl_write(OUTPUT_DIR / "object_registry.jsonl", object_registry)
    merged_path = OUTPUT_DIR / "tables_merged.jsonl"
    if merged_path.exists():
        merged_path.unlink()

    summary = {
        "project_root": str(PROJECT_ROOT),
        "pdf_path": str(pdf_path),
        "stage1_path": str(STAGE1_PATH),
        "output_dir": str(OUTPUT_DIR),
        "text_region_count": len(text_regions),
        "table_raw_count": len(raw_tables),
        "table_chain_count": len(table_chains),
        "figure_task_count": len(figure_tasks),
        "multimodal_task_count": len(multimodal_tasks),
        "table_vlm_count": sum(1 for item in raw_tables if item.needs_vlm or item.complexity_score >= float(config.get("multimodal_upgrade_threshold", 0.7))),
        "camelot_available": camelot is not None,
        "camelot_error": None if CAMELLOT_IMPORT_ERROR is None else f"{type(CAMELLOT_IMPORT_ERROR).__name__}: {CAMELLOT_IMPORT_ERROR}",
        "table_backend_counts": dict(backend_counts),
        "generated_files": [
            "text_regions.jsonl",
            "page_text_flow.json",
            "page_text_flow_resolved.json",
            "tables_raw.jsonl",
            "table_chains.jsonl",
            "figure_tasks.jsonl",
            "multimodal_tasks.jsonl",
            "object_registry.jsonl",
        ],
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log(
        "complete "
        f"text_regions={summary['text_region_count']} "
        f"tables_raw={summary['table_raw_count']} "
        f"table_chains={summary['table_chain_count']} "
        f"multimodal_tasks={summary['multimodal_task_count']} "
        f"figure_tasks={summary['figure_task_count']} "
        f"elapsed={summary['elapsed_seconds']}s"
    )


if __name__ == "__main__":
    main()
