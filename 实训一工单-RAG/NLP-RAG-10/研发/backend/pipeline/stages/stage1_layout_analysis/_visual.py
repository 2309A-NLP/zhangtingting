# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations

import re
from typing import Any

import fitz

from backend.pipeline.stages.stage1_layout_analysis._bbox import (
    bbox_area,
    bbox_overlap_ratio,
    is_inside,
    overlap_ratio,
    union_bbox,
    _item_to_bbox,
)
from backend.pipeline.stages.stage1_layout_analysis._config import (
    PERCENTAGE_PATTERN,
    DATAVIZ_KEYWORDS,
    FLOWCHART_KEYWORDS,
    normalize_whitespace,
    extract_text_blocks,
    extract_text_lines,
)


def looks_like_flowchart_text(text: str) -> bool:
    normalized = normalize_whitespace(text)
    if any(keyword in normalized for keyword in ["流程图", "结构图", "思维导图", "组织架构图"]):
        return True
    vertical_tokens = re.findall(r"(?:[\u4e00-\u9fff]\s*\|\s*){2,}[\u4e00-\u9fff]?", normalized)
    if vertical_tokens:
        return True
    percentages = PERCENTAGE_PATTERN.findall(normalized)
    if len(percentages) >= 4 and any(keyword in normalized for keyword in ["饼图", "柱图", "折图"]):
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
            if area_ratio < 0.04 and width < page.rect.width * 0.35 and height < page.rect.height * 0.25:
                continue
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


def extract_drawing_segments(page: fitz.Page) -> tuple[list[dict[str, Any]], list[dict[float]]]:
    from backend.pipeline.stages.stage1_layout_analysis._config import TABLE_HLINE_MIN_RATIO
    h_segments = []
    complex_boxes: list[list[float]] = []
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


def detect_figure_regions(page: fitz.Page, exclude_bboxes: list[list[float]]) -> list[dict[str, Any]]:
    from backend.pipeline.stages.stage1_layout_analysis._bbox import merge_boxes
    from backend.pipeline.stages.stage1_layout_analysis._table import is_table_like_visual_region
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
    if features["h_line_count"] >= 3 and any(kw in text_lower for kw in ["趋势图", "对比", "柱图", "折线图", "饼图"]):
        return "data_viz"
    if features["has_filled_box"] and PERCENTAGE_PATTERN.search(nearby_text) and any(kw in nearby_text for kw in ["公司", "业务", "部门", "产品"]):
        return "flowchart"
    if any(kw in text_lower for kw in DATAVIZ_KEYWORDS):
        return "data_viz"
    if any(kw in text_lower for kw in FLOWCHART_KEYWORDS):
        return "flowchart"
    return "mixed_image"


def bbox_distance(left: list[float], right: list[float]) -> float:
    dx = max(0.0, max(left[0], right[0]) - min(left[2], right[2]))
    dy = max(0.0, max(left[1], right[1]) - min(left[3], right[3]))
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return (dx * dx + dy * dy) ** 0.5


def cluster_visual_candidates(
    page: fitz.Page,
    image_blocks: list[dict[str, Any]],
    complex_figures: list[dict[str, Any]],
    table_boxes: list[list[float]],
    text_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from backend.pipeline.stages.stage1_layout_analysis._table import is_table_like_visual_region
    from backend.pipeline.stages.stage1_layout_analysis._config import FIGURE_CLUSTER_GAP, FIGURE_CLUSTER_OVERLAP
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
