# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations
from typing import Any

from backend.pipeline.stages.stage1_layout_analysis._config import (
    PERCENTAGE_PATTERN,
    normalize_whitespace,
)


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
