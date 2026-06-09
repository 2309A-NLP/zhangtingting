# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations

import re
from collections import defaultdict

import fitz

from backend.pipeline.stages.stage1_layout_analysis._bbox import (
    bbox_area,
    merge_boxes,
    normalize_whitespace,
)
from backend.pipeline.stages.stage1_layout_analysis._config import (
    NUMBER_PATTERN,
    PERCENTAGE_PATTERN,
    TABLE_HLINE_MIN_RATIO,
    TABLE_MIN_ROWS,
    TABLE_ROW_HEIGHT_MAX,
    TABLE_ROW_HEIGHT_MIN,
    TABLE_HEADER_WORDS,
)


def flatten_table_extract_rows(rows: list) -> list[str]:
    flattened: list[str] = []
    for row in rows:
        parts = [normalize_whitespace(str(cell or "")) for cell in row if normalize_whitespace(str(cell or ""))]
        if parts:
            flattened.append(" | ".join(parts))
    return flattened


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
            pipe_lines.append({
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "text": text,
                "y0": float(y0),
                "y1": float(y1),
            })

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
        tables.append({
            "bbox": [x0, y0, x1, y1],
            "row_count": len(group),
            "col_count": col_count,
            "has_thick_border": False,
            "is_pipe_table": True,
            "detector": "pipe_table",
            "confidence": 0.96,
            "text_lines": lines_text,
        })
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
        tables.append({
            "bbox": bbox,
            "row_count": int(getattr(table, "row_count", 0) or 0),
            "col_count": int(getattr(table, "col_count", 0) or 0),
            "has_thick_border": True,
            "is_pymupdf_table": True,
            "detector": "pymupdf_table",
            "confidence": 0.99,
            "text_lines": text_lines,
        })
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
        row_blocks.append({
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
            "parts": parts,
            "text": "\n".join(parts),
            "x0": float(x0),
            "x1": float(x1),
            "y0": float(y0),
            "y1": float(y1),
        })

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
        tables.append({
            "bbox": [x0, y0, x1, y1],
            "row_count": len(group),
            "col_count": col_count,
            "has_thick_border": False,
            "is_multiline_row_table": True,
            "detector": "multiline_row_table",
            "confidence": 0.94,
            "text_lines": lines_text,
        })
    return tables


def detect_top_continuation_table(page: fitz.Page) -> list[dict]:
    from backend.pipeline.stages.stage1_layout_analysis._table_utils import is_continuation_label
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
        top_blocks.append({
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
            "parts": parts,
            "text": "\n".join(parts),
            "x0": float(x0),
            "x1": float(x1),
            "y0": float(y0),
            "y1": float(y1),
        })

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
    return [{
        "bbox": [x0, y0, x1, y1],
        "row_count": len(group),
        "col_count": col_count,
        "has_thick_border": False,
        "is_top_continuation_table": True,
        "detector": "top_continuation",
        "continuation_label_found": continuation_label_found,
        "confidence": 0.98,
        "text_lines": lines_text,
    }]


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
    from backend.pipeline.stages.stage1_layout_analysis._table_utils import KEY_VALUE_PATTERN
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
        from backend.pipeline.stages.stage1_layout_analysis._bbox import is_inside
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
