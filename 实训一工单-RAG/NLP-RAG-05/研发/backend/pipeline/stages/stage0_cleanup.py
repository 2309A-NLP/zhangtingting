from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_PAGE_INDICES: list[int] = []


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_pattern(value: str) -> str:
    try:
        return value.encode("latin1").decode("utf-8")
    except Exception:
        return value


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


def page_area(page: fitz.Page) -> float:
    return max(1.0, float(page.rect.width or 1.0) * float(page.rect.height or 1.0))


def line_text_and_bbox(block: dict[str, Any], line: dict[str, Any]) -> tuple[str, list[float]]:
    text = normalize_whitespace("".join(str(span.get("text") or "") for span in line.get("spans") or []))
    bbox = list(line.get("bbox") or block.get("bbox") or [0.0, 0.0, 0.0, 0.0])
    return text, bbox


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


def detect_header_footer_regions(
    page: fitz.Page,
    text_dict: dict[str, Any],
    repeated_lines: Counter[str],
) -> list[dict[str, Any]]:
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


def render_page_gray(page: fitz.Page, width: int = 220) -> tuple[int, int, bytes]:
    scale = width / max(float(page.rect.width or 1.0), 1.0)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False)
    return pix.width, pix.height, bytes(pix.samples)


def learn_visual_watermark_template(
    pages: list[fitz.Page],
    learn_pages: int = 5,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    training_pages = pages[: max(1, min(learn_pages, len(pages)))]
    rendered = [render_page_gray(page) for page in training_pages]
    if not rendered:
        return [], {"training_pages": 0, "active_cells": 0, "pixel_hits": 0}

    width = min(item[0] for item in rendered)
    height = min(item[1] for item in rendered)
    aligned_samples: list[bytes] = []
    for current_width, current_height, samples in rendered:
        if current_width == width and current_height == height:
            aligned_samples.append(samples)
            continue
        cropped = bytearray(width * height)
        for row in range(height):
            start = row * current_width
            cropped[row * width : (row + 1) * width] = samples[start : start + width]
        aligned_samples.append(bytes(cropped))

    hit_counter = [0] * (width * height)
    for index in range(width * height):
        values = [sample[index] for sample in aligned_samples]
        mean_value = sum(values) / len(values)
        row = index // width
        col = index % width
        cy = row / max(height, 1)
        cx = col / max(width, 1)

        if cy < 0.03 or cy > 0.97:
            continue
        if cx < 0.03 or cx > 0.97:
            continue
        if mean_value < 180 or mean_value > 252:
            continue
        hits = sum(1 for value in values if 185 <= value <= 252)
        if hits >= max(2, len(values) - 1):
            hit_counter[index] = hits

    block_w = 10
    block_h = 10
    grid_cols = max(1, width // block_w)
    grid_rows = max(1, height // block_h)
    active_cells: list[tuple[int, int]] = []

    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            hits = 0
            total = 0
            for y in range(grid_row * block_h, min((grid_row + 1) * block_h, height)):
                for x in range(grid_col * block_w, min((grid_col + 1) * block_w, width)):
                    total += 1
                    if hit_counter[y * width + x] > 0:
                        hits += 1
            if total and hits / total >= 0.12:
                active_cells.append((grid_row, grid_col))

    if not active_cells:
        return [], {
            "training_pages": len(training_pages),
            "render_size": {"width": width, "height": height},
            "pixel_hits": sum(1 for value in hit_counter if value > 0),
            "active_cells": 0,
            "boxes": 0,
        }

    active_set = set(active_cells)
    visited: set[tuple[int, int]] = set()
    boxes: list[dict[str, float]] = []

    for start in active_cells:
        if start in visited:
            continue
        stack = [start]
        cells: list[tuple[int, int]] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cells.append(current)
            row, col = current
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in active_set and neighbor not in visited:
                    stack.append(neighbor)

        if len(cells) < 2:
            continue

        min_row = min(cell[0] for cell in cells)
        max_row = max(cell[0] for cell in cells)
        min_col = min(cell[1] for cell in cells)
        max_col = max(cell[1] for cell in cells)
        x0 = min_col * block_w / width
        y0 = min_row * block_h / height
        x1 = min(1.0, ((max_col + 1) * block_w) / width)
        y1 = min(1.0, ((max_row + 1) * block_h) / height)
        area_ratio = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area_ratio < 0.002:
            continue
        boxes.append(
            {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "cell_count": float(len(cells)),
                "area_ratio": area_ratio,
            }
        )

    debug = {
        "training_pages": len(training_pages),
        "render_size": {"width": width, "height": height},
        "pixel_hits": sum(1 for value in hit_counter if value > 0),
        "active_cells": len(active_cells),
        "boxes": len(boxes),
    }
    return boxes, debug


def match_visual_watermark_template(page: fitz.Page, template_boxes: list[dict[str, float]]) -> list[dict[str, Any]]:
    if not template_boxes:
        return []

    width, height, samples = render_page_gray(page)
    matched: list[dict[str, Any]] = []
    for index, box in enumerate(template_boxes):
        x0 = max(0, min(width - 1, int(box["x0"] * width)))
        y0 = max(0, min(height - 1, int(box["y0"] * height)))
        x1 = max(x0 + 1, min(width, int(box["x1"] * width)))
        y1 = max(y0 + 1, min(height, int(box["y1"] * height)))

        values: list[int] = []
        for row in range(y0, y1):
            start = row * width
            values.extend(samples[start + x0 : start + x1])
        if not values:
            continue

        mean_value = sum(values) / len(values)
        hits = sum(1 for value in values if 185 <= value <= 252)
        hit_ratio = hits / max(1, len(values))
        if 175 <= mean_value <= 252 and hit_ratio >= 0.18:
            matched.append(
                {
                    "type": "watermark_visual",
                    "bbox": [
                        float(page.rect.width) * box["x0"],
                        float(page.rect.height) * box["y0"],
                        float(page.rect.width) * box["x1"],
                        float(page.rect.height) * box["y1"],
                    ],
                    "content": f"visual_template_{index}",
                }
            )
    return matched


def filtered_text_metrics(text_dict: dict[str, Any], filter_regions: list[dict[str, Any]]) -> dict[str, float]:
    text_span_count = 0
    text_line_count = 0
    text_block_count = 0
    text_char_count = 0
    text_area = 0.0

    for block in text_dict.get("blocks") or []:
        if block.get("type") != 0:
            continue
        kept_line_found = False
        for line in block.get("lines") or []:
            text, bbox = line_text_and_bbox(block, line)
            if not text or is_filtered_line(bbox, filter_regions):
                continue
            spans = line.get("spans") or []
            if spans:
                text_line_count += 1
                kept_line_found = True
            text_span_count += len(spans)
            text_char_count += sum(len(str(span.get("text") or "")) for span in spans)
            if len(bbox) == 4:
                text_area += max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
        if kept_line_found:
            text_block_count += 1

    return {
        "text_span_count": text_span_count,
        "text_line_count": text_line_count,
        "text_block_count": text_block_count,
        "text_char_count": text_char_count,
        "text_area": text_area,
    }


def filtered_text_preview(text_dict: dict[str, Any], filter_regions: list[dict[str, Any]]) -> str:
    kept_lines: list[str] = []
    for block in text_dict.get("blocks") or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or []:
            text, bbox = line_text_and_bbox(block, line)
            if not text or is_filtered_line(bbox, filter_regions):
                continue
            kept_lines.append(text)
    return "\n".join(kept_lines)[:1000]


def image_metrics(page: fitz.Page, text_dict: dict[str, Any]) -> dict[str, float]:
    blocks = list(text_dict.get("blocks") or [])
    image_blocks = [block for block in blocks if block.get("type") == 1]
    image_area = 0.0
    for block in image_blocks:
        bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        if len(bbox) == 4:
            image_area += max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    return {
        "image_count": len(image_blocks) + len(page.get_images(full=True)),
        "image_area": image_area,
    }


def region_area_ratio(region: dict[str, Any], page: fitz.Page) -> float:
    bbox = list(region.get("bbox") or [])
    if len(bbox) != 4:
        return 0.0
    area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    return area / page_area(page)


def looks_like_garbled_text(text: str) -> bool:
    sample = (text or "").strip()
    if not sample:
        return True
    junk_patterns = ["锟斤拷锟", "锟", "\ufffd"]
    if any(pattern in sample for pattern in junk_patterns):
        return True
    non_printable = sum(1 for char in sample if ord(char) < 32 and char not in "\n\r\t")
    return non_printable > 10


def watermark_score(filter_regions: list[dict[str, Any]], page: fitz.Page) -> float:
    if not filter_regions:
        return 0.0
    score = 0.0
    for region in filter_regions:
        score += min(0.5, region_area_ratio(region, page))
        if region["type"] in {"watermark_text", "watermark_visual"}:
            score += 0.18
    return min(1.0, score)


def watermark_presence_score(filter_regions: list[dict[str, Any]], page: fitz.Page) -> float:
    visual_regions = [region for region in filter_regions if region.get("type") == "watermark_visual"]
    text_regions = [region for region in filter_regions if region.get("type") == "watermark_text"]
    visual_area = sum(region_area_ratio(region, page) for region in visual_regions)
    text_area = sum(region_area_ratio(region, page) for region in text_regions)
    score = min(1.0, visual_area * 1.25 + text_area * 0.8)
    if len(visual_regions) >= 9:
        score = max(score, 0.95)
    elif len(visual_regions) >= 6:
        score = max(score, 0.8)
    return min(1.0, score)


def watermark_interference_score(
    filter_regions: list[dict[str, Any]],
    text_metrics: dict[str, float],
    page: fitz.Page,
) -> float:
    text_area_ratio = float(text_metrics.get("text_area") or 0.0) / page_area(page)
    visual_regions = [region for region in filter_regions if region.get("type") == "watermark_visual"]
    text_regions = [region for region in filter_regions if region.get("type") == "watermark_text"]
    visual_area = sum(region_area_ratio(region, page) for region in visual_regions)
    text_area = sum(region_area_ratio(region, page) for region in text_regions)

    score = min(1.0, text_area * 3.5 + min(0.3, visual_area * 0.35))
    if text_area_ratio < 0.015 and visual_area > 0.2:
        score = max(score, 0.45)
    return min(1.0, score)


def detect_page_type(
    page: fitz.Page,
    normalized_text: str,
    text_metrics: dict[str, float],
    image_metrics_value: dict[str, float],
    watermark_score_value: float,
) -> tuple[str, float, float]:
    text_span_count = int(text_metrics["text_span_count"])
    text_char_count = int(text_metrics["text_char_count"])
    text_coverage_ratio = float(text_metrics["text_area"]) / page_area(page)
    image_coverage_ratio = float(image_metrics_value["image_area"]) / page_area(page)

    page_type = "native"
    if text_char_count < 20 or text_span_count < 6 or looks_like_garbled_text(normalized_text):
        page_type = "scanned"
    elif text_char_count >= 20 and image_coverage_ratio >= 0.16 and watermark_score_value < 0.45:
        page_type = "mixed"
    if text_char_count >= 120 and text_coverage_ratio >= 0.02:
        page_type = "native"
    return page_type, text_coverage_ratio, image_coverage_ratio


def decide_ocr_required(
    *,
    page_type: str,
    text_char_count: int,
    text_span_count: int,
    text_coverage_ratio: float,
    image_coverage_ratio: float,
    watermark_score_value: float,
    normalized_text: str,
) -> tuple[bool, str]:
    if page_type == "scanned":
        return True, "scanned_page"
    if looks_like_garbled_text(normalized_text):
        return True, "garbled_text"
    if page_type == "mixed":
        if text_char_count < 40 or text_span_count < 10:
            return True, "mixed_with_low_text"
        if text_coverage_ratio < 0.02 and image_coverage_ratio >= 0.18:
            return True, "mixed_image_dominant"
        return False, "mixed_but_text_is_sufficient"
    if watermark_score_value >= 0.55 and text_char_count < 60:
        return True, "watermark_noise_with_low_text"
    return False, "native_text_sufficient"


def stage0_for_page(
    page: fitz.Page,
    repeated_lines: Counter[str],
    watermark_patterns: list[str],
    position_dense_counter: dict[tuple[str, int, int], int],
    visual_template_boxes: list[dict[str, float]],
) -> dict[str, Any]:
    text_dict = page.get_text("dict")
    raw_text = page.get_text("text")
    normalized_text = normalize_whitespace(raw_text)

    filter_regions = [
        *detect_header_footer_regions(page, text_dict, repeated_lines),
        *detect_text_watermarks(page, text_dict, watermark_patterns, position_dense_counter),
        *match_visual_watermark_template(page, visual_template_boxes),
    ]

    text_metrics = filtered_text_metrics(text_dict, filter_regions)
    image_metrics_value = image_metrics(page, text_dict)
    watermark_score_value = watermark_score(filter_regions, page)
    watermark_presence_value = watermark_presence_score(filter_regions, page)
    watermark_interference_value = watermark_interference_score(filter_regions, text_metrics, page)
    page_type, text_coverage_ratio, image_coverage_ratio = detect_page_type(
        page,
        normalized_text,
        text_metrics,
        image_metrics_value,
        watermark_interference_value,
    )
    ocr_required, ocr_reason = decide_ocr_required(
        page_type=page_type,
        text_char_count=int(text_metrics["text_char_count"]),
        text_span_count=int(text_metrics["text_span_count"]),
        text_coverage_ratio=text_coverage_ratio,
        image_coverage_ratio=image_coverage_ratio,
        watermark_score_value=watermark_interference_value,
        normalized_text=normalized_text,
    )

    return {
        "page_index": page.number + 1,
        "page_type": page_type,
        "filter_regions": filter_regions,
        "ocr_required": ocr_required,
        "text_coverage_ratio": text_coverage_ratio,
        "text_span_count": int(text_metrics["text_span_count"]),
        "text_line_count": int(text_metrics["text_line_count"]),
        "text_block_count": int(text_metrics["text_block_count"]),
        "text_char_count": int(text_metrics["text_char_count"]),
        "image_count": int(image_metrics_value["image_count"]),
        "image_coverage_ratio": image_coverage_ratio,
        "watermark_score": watermark_score_value,
        "watermark_presence_score": watermark_presence_value,
        "watermark_interference_score": watermark_interference_value,
        "ocr_decision_reason": ocr_reason,
        "raw_text_preview": raw_text[:1000],
        "filtered_text_preview": filtered_text_preview(text_dict, filter_regions),
    }


def main() -> None:
    pdf_path = PROJECT_ROOT / "data" / "招股说明书1.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")

    output_dir = PROJECT_ROOT / "artifacts" / "stage0_smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    watermark_patterns = load_watermark_patterns()

    results: list[dict[str, Any]] = []
    with fitz.open(str(pdf_path)) as doc:
        target_page_indices = [index for index in TEST_PAGE_INDICES if 0 <= index < len(doc)]
        if not target_page_indices:
            target_page_indices = list(range(len(doc)))
        preview_pages = [doc.load_page(index) for index in target_page_indices]
        repeated_lines = collect_repeated_lines(preview_pages)
        position_dense_counter = collect_position_dense_candidates(preview_pages)
        visual_template_boxes, visual_template_debug = learn_visual_watermark_template(
            preview_pages,
            learn_pages=min(5, len(preview_pages)),
        )

        for index in target_page_indices:
            page = doc.load_page(index)
            stage0_result = stage0_for_page(
                page,
                repeated_lines,
                watermark_patterns,
                position_dense_counter,
                visual_template_boxes,
            )
            results.append(
                {
                    "page_index": index + 1,
                    "page_size": {
                        "width": float(page.rect.width),
                        "height": float(page.rect.height),
                    },
                    "page_text_preview": stage0_result["filtered_text_preview"][:500],
                    "stage0_result": stage0_result,
                }
            )

    payload = {
        "pdf": str(pdf_path),
        "tested_pages": [index + 1 for index in target_page_indices],
        "visual_watermark_template_boxes": visual_template_boxes,
        "visual_watermark_template_debug": visual_template_debug,
        "technology_notes": [
            {
                "step": "page_object_read",
                "technology": "PyMuPDF / fitz",
                "description": "读取 PDF 页面对象、文本字典、页面图片对象和页面尺寸。",
            },
            {
                "step": "header_footer_detection",
                "technology": "跨页重复行统计 + 页边规则",
                "description": "结合跨页重复文本和页边位置，检测页眉、页脚和逻辑页码。",
            },
            {
                "step": "text_watermark_detection",
                "technology": "文本 span 规则 + 位置密度学习",
                "description": "按关键词、大字号、浅色、中部覆盖和跨页高频重复位置，检测文字型水印。",
            },
            {
                "step": "visual_watermark_template_learning",
                "technology": "整页灰度渲染 + 前几页模板学习 + 后续页复用",
                "description": "先对前几页整页缩略渲染，学习稳定出现的浅色覆盖区域，再在后续页面复用该模板检测视觉型水印。",
            },
            {
                "step": "filtered_text_metrics",
                "technology": "过滤后文本统计",
                "description": "对去掉页眉、页脚、文字型水印候选后的正文重新统计 line/span/block/char。",
            },
            {
                "step": "coverage_estimation",
                "technology": "bbox 面积估算",
                "description": "分别估算正文文字覆盖率和图片覆盖率。",
            },
            {
                "step": "page_type_decision",
                "technology": "规则判定",
                "description": "根据正文文字量、图片覆盖率和水印噪声，判定 native / mixed / scanned。",
            },
            {
                "step": "ocr_decision",
                "technology": "统一 OCR 决策规则",
                "description": "输出 ocr_required 和 ocr_decision_reason，避免 mixed 页面一律走 OCR。",
            },
        ],
        "results": results,
    }

    output_path = output_dir / "stage0_selected_pages.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
