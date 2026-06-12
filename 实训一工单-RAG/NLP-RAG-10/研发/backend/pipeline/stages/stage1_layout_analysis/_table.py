# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

import fitz

from backend.pipeline.stages.stage1_layout_analysis._bbox import (
    bbox_area,
    is_inside,
    is_filtered_line,
    line_text_and_bbox,
    merge_boxes,
    normalize_whitespace,
    overlap_ratio,
)
from backend.pipeline.stages.stage1_layout_analysis._config import (
    DATE_PATTERN,
    NUMBER_PATTERN,
    PERCENTAGE_PATTERN,
    CHINESE_PATTERN,
    KEY_VALUE_PATTERN,
    TABLE_HEADER_WORDS,
    TABLE_END_WORDS,
    TABLE_HLINE_MIN_RATIO,
    TABLE_MIN_ROWS,
    TABLE_ROW_HEIGHT_MAX,
    TABLE_ROW_HEIGHT_MIN,
    UNIT_PREFIXES,
    CONTINUATION_WORDS,
)

if TYPE_CHECKING:
    from backend.pipeline.stages.stage1_layout_analysis._layout import Region, LayoutPage


def flatten_table_extract_rows(rows: list[list]) -> list[str]:
    flattened: list[str] = []
    for row in rows:
        parts = [normalize_whitespace(str(cell or "")) for cell in row if normalize_whitespace(str(cell or ""))]
        if parts:
            flattened.append(" | ".join(parts))
    return flattened


def collect_texts_in_bbox(text_blocks: list[dict], bbox: list[float]) -> list[str]:
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
        or ("项目" in normalized and "名称" in normalized)
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


def get_texts_between_tables(text_blocks: list[dict], upper_bbox: list[float], lower_bbox: list[float]) -> list[str]:
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


def merge_same_page_continuation_tables(table_areas: list[dict], text_blocks: list[dict]) -> list[dict]:
    if len(table_areas) < 2:
        return table_areas
    ordered = sorted(table_areas, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    merged: list[dict] = []
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


def table_candidate_priority(candidate: dict) -> float:
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


def table_candidates_overlap(left: dict, right: dict) -> bool:
    if is_inside(left["bbox"], right["bbox"]) or is_inside(right["bbox"], left["bbox"]):
        return True
    return overlap_ratio(left["bbox"], right["bbox"]) >= 0.35 or overlap_ratio(right["bbox"], left["bbox"]) >= 0.35


def dedupe_table_candidates(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    groups: list[list[dict]] = []
    for candidate in sorted(candidates, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        placed = False
        for group in groups:
            if any(table_candidates_overlap(candidate, existing) for existing in group):
                group.append(candidate)
                placed = True
                break
        if not placed:
            groups.append([candidate])

    deduped: list[dict] = []
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


def detect_pipe_tables(page: fitz.Page) -> list[dict]:
    pipe_lines: list[dict] = []
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
    groups: list[list[dict]] = []
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

    tables: list[dict] = []
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


def detect_pymupdf_tables(page: fitz.Page) -> list[dict]:
    try:
        finder = page.find_tables()
    except Exception:
        return []
    tables: list[dict] = []
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


def detect_multiline_row_tables(page: fitz.Page) -> list[dict]:
    from backend.pipeline.stages.stage1_layout_analysis._visual import looks_like_flowchart_text
    row_blocks: list[dict] = []
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
    groups: list[list[dict]] = []
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

    tables: list[dict] = []
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


def detect_top_continuation_table(page: fitz.Page) -> list[dict]:
    top_blocks: list[dict] = []
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


def detect_table_by_lines(page: fitz.Page) -> list[dict]:
    from backend.pipeline.stages.stage1_layout_analysis._visual import extract_drawing_segments
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


def detect_table_by_text_alignment(page: fitz.Page, filter_regions: list[dict] | None = None) -> list[dict]:
    from backend.pipeline.stages.stage1_layout_analysis._visual import looks_like_flowchart_text
    from backend.pipeline.stages.stage1_layout_analysis._config import extract_text_lines
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


def is_table_like_visual_region(page: fitz.Page, bbox: list[float], text_blocks: list[dict], filter_regions: list[dict] | None = None) -> bool:
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
    from backend.pipeline.stages.stage1_layout_analysis._config import extract_text_lines
    lines = [line for line in extract_text_lines(page, filter_regions=filter_regions) if overlap_ratio(line["bbox"], bbox) >= 0.2]
    if len(lines) < 4:
        return False
    x_buckets: dict[int, int] = defaultdict(int)
    for line in lines:
        bucket = round(float(line["x"]) / 24.0) * 24
        x_buckets[bucket] += 1
    stable_cols = sum(1 for count in x_buckets.values() if count >= 2)
    return stable_cols >= 3 and numeric_hits >= 1


def table_looks_like_flowchart(cell_texts: list[str]) -> bool:
    from backend.pipeline.stages.stage1_layout_analysis._visual import looks_like_flowchart_text
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
    flow_words = ["开始", "结束", "流程", "输入", "输出", "判断", "处理", "申请", "审批", "通过", "驳回"]
    flow_hits = sum(1 for word in flow_words if word in joined)
    return numeric_lines <= max(1, len(lines) // 8) and short_lines >= max(3, len(lines) // 3) and flow_hits >= 2


def infer_table_sub_type(page: fitz.Page, bbox: list[float], cell_texts: list[str], embedded_visual_count: int) -> str:
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


def column_alignment_score(a: "Region", b: "Region") -> float:
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


def column_type_match_score(a: "Region", b: "Region") -> float:
    types_a = infer_column_types(a.text)
    types_b = infer_column_types(b.text)
    if not types_a or not types_b:
        return 0.0
    size = min(len(types_a), len(types_b))
    if size == 0:
        return 0.0
    matches = sum(1 for i in range(size) if types_a[i] == types_b[i])
    return matches / max(len(types_a), len(types_b))


def reading_continuity_score(page_a: "LayoutPage", page_b: "LayoutPage", table_a: "Region", table_b: "Region") -> float:
    orders_a = sorted(r.reading_order for r in page_a.regions)
    orders_b = sorted(r.reading_order for r in page_b.regions)
    a_is_last = table_a.reading_order == orders_a[-1]
    b_is_first = table_b.reading_order == orders_b[0]
    text_a_last = table_a.text.strip().split("\n")[-1] if table_a.text else ""
    text_b_first = table_b.text.strip().split("\n")[0] if table_b.text else ""
    not_ending = not any(text_a_last.endswith(w) for w in TABLE_END_WORDS)
    not_header = not any(w in text_b_first for w in TABLE_HEADER_WORDS)
    header_like_b = row_looks_like_header(text_b_first)
    pre_text_regions = [r for r in page_b.regions if r.region_type == "text" and r.reading_order < table_b.reading_order]
    post_text_regions = [r for r in page_a.regions if r.region_type == "text" and r.reading_order > table_a.reading_order]
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


def detect_cross_page_tables(pages: list["LayoutPage"], config: dict) -> tuple[list["Region"], list[dict]]:
    from backend.pipeline.stages.stage1_layout_analysis._layout import Region
    from backend.pipeline.stages.stage1_layout_analysis._config import log, CROSS_PAGE_MERGE_HIGH, CROSS_PAGE_MERGE_LOW, CROSS_PAGE_GAP_BOTTOM, CROSS_PAGE_GAP_TOP
    merge_high = float(config.get("cross_page_merge_high", CROSS_PAGE_MERGE_HIGH))
    merge_low = float(config.get("cross_page_merge_low", CROSS_PAGE_MERGE_LOW))
    merges: list[tuple[Region, Region, float, str]] = []
    debug_log: list[dict] = []
    log("正在分析跨页表格...")

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
                log(f"候选配对 p{page_a.page_index}:{ta.region_id} -> p{page_b.page_index}:{tb.region_id} confidence={confidence} decision={decision}")
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
