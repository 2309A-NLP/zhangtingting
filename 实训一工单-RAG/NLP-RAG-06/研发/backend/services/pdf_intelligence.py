from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import fitz
import pdfplumber

try:
    import imagehash
    from PIL import Image
except Exception:  # pragma: no cover
    imagehash = None
    Image = None

from backend.config import settings
from backend.services.pdf_intelligence_models import (
    ContextLink,
    ExtractedFigure,
    ExtractedTable,
    FilterRegion,
    InlineImage,
    LayoutElement,
    Stage0PageResult,
    Stage1PageLayout,
    StructuredFact,
)
from backend.services.text_utils import dedupe_preserve_order, normalize_whitespace


def _levenshtein_ratio(a: str, b: str) -> float:
    left = a or ""
    right = b or ""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    rows = len(left) + 1
    cols = len(right) + 1
    distance = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + cost,
            )
    edit_distance = distance[-1][-1]
    return 1.0 - edit_distance / max(len(left), len(right), 1)


class PDFIntelligencePipeline:
    """Prompt-driven staged PDF intelligence pipeline."""

    def __init__(self) -> None:
        self.output_root = settings.pdf_intelligence_output_dir
        self.stage_dirs = {
            "stage_0_preprocess": self.output_root / "stage_0_preprocess",
            "stage_1_layout": self.output_root / "stage_1_layout",
            "stage_2_extraction": self.output_root / "stage_2_extraction",
            "stage_3_context": self.output_root / "stage_3_context",
            "stage_4_multimodal": self.output_root / "stage_4_multimodal",
            "stage_5_index": self.output_root / "stage_5_index",
            "logs": self.output_root / "logs",
        }
        for directory in self.stage_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_root / "checkpoint.json"
        self.watermark_patterns = self._load_watermark_patterns()

    def _safe_pattern(self, value: str) -> str:
        try:
            return value.encode("latin1").decode("utf-8")
        except Exception:
            return value

    def _load_watermark_patterns(self) -> List[str]:
        path = settings.watermark_patterns_path
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(payload, list):
            return [self._safe_pattern(str(item).strip()) for item in payload if str(item).strip()]
        return []

    def _save_json(self, target: Path, payload: Any) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_checkpoint(self) -> Dict[str, Any]:
        if not self.checkpoint_path.exists():
            return {}
        try:
            return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_checkpoint(self, payload: Dict[str, Any]) -> None:
        self._save_json(self.checkpoint_path, payload)

    def _page_pixel_area(self, page: fitz.Page) -> float:
        return max(1.0, float(page.rect.width or 1) * float(page.rect.height or 1))

    def _text_metrics(self, text_dict: Dict[str, Any]) -> Dict[str, float]:
        blocks = list(text_dict.get("blocks") or [])
        text_blocks = [block for block in blocks if block.get("type") == 0]
        image_blocks = [block for block in blocks if block.get("type") == 1]
        text_span_count = 0
        text_line_count = 0
        text_char_count = 0
        text_area = 0.0
        for block in text_blocks:
            for line in block.get("lines") or []:
                spans = line.get("spans") or []
                if spans:
                    text_line_count += 1
                text_span_count += len(spans)
                text_char_count += sum(len(str(span.get("text") or "")) for span in spans)
                bbox = line.get("bbox") or block.get("bbox") or [0, 0, 0, 0]
                if len(bbox) == 4:
                    text_area += max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
        return {
            "text_block_count": len(text_blocks),
            "image_block_count": len(image_blocks),
            "text_span_count": text_span_count,
            "text_line_count": text_line_count,
            "text_char_count": text_char_count,
            "text_area": text_area,
        }

    def _line_text_and_bbox(self, block: Dict[str, Any], line: Dict[str, Any]) -> Tuple[str, List[float]]:
        text = normalize_whitespace(
            "".join(str(span.get("text") or "") for span in line.get("spans") or [])
        )
        bbox = list(line.get("bbox") or block.get("bbox") or [0, 0, 0, 0])
        return text, bbox

    def _bbox_overlap_ratio(self, left: List[float], right: List[float]) -> float:
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

    def _is_filtered_line(self, bbox: List[float], filter_regions: List[FilterRegion]) -> bool:
        for region in filter_regions:
            if region.type not in {"header", "footer", "watermark"}:
                continue
            if self._bbox_overlap_ratio(bbox, region.bbox) >= 0.55:
                return True
        return False

    def _filtered_text_metrics(self, text_dict: Dict[str, Any], filter_regions: List[FilterRegion]) -> Dict[str, float]:
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
                text, bbox = self._line_text_and_bbox(block, line)
                if not text or self._is_filtered_line(bbox, filter_regions):
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
            "text_block_count": text_block_count,
            "text_span_count": text_span_count,
            "text_line_count": text_line_count,
            "text_char_count": text_char_count,
            "text_area": text_area,
        }

    def _build_filtered_text_preview(self, text_dict: Dict[str, Any], filter_regions: List[FilterRegion]) -> str:
        kept_lines: List[str] = []
        for block in text_dict.get("blocks") or []:
            if block.get("type") != 0:
                continue
            for line in block.get("lines") or []:
                text, bbox = self._line_text_and_bbox(block, line)
                if not text or self._is_filtered_line(bbox, filter_regions):
                    continue
                kept_lines.append(text)
        return "\n".join(kept_lines)[:1000]

    def _image_metrics(self, page: fitz.Page, text_dict: Dict[str, Any]) -> Dict[str, float]:
        blocks = list(text_dict.get("blocks") or [])
        image_blocks = [block for block in blocks if block.get("type") == 1]
        image_area = 0.0
        for block in image_blocks:
            bbox = block.get("bbox") or [0, 0, 0, 0]
            if len(bbox) == 4:
                image_area += max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
        return {
            "image_count": len(image_blocks) + len(page.get_images(full=True)),
            "image_area": image_area,
        }

    def _detect_page_type(
        self,
        page: fitz.Page,
        text_dict: Dict[str, Any],
        normalized_text: str,
        text_metrics: Dict[str, float],
        image_metrics: Dict[str, float],
        watermark_score: float,
    ) -> Tuple[str, float, float]:
        text_span_count = int(text_metrics["text_span_count"])
        text_char_count = int(text_metrics["text_char_count"])
        text_coverage_ratio = float(text_metrics["text_area"]) / self._page_pixel_area(page)
        image_coverage_ratio = float(image_metrics["image_area"]) / self._page_pixel_area(page)

        page_type = "native"
        if text_char_count < 20 or text_span_count < 6 or self._looks_like_garbled_text(normalized_text):
            page_type = "scanned"
        elif (
            text_char_count >= 20
            and image_coverage_ratio >= 0.16
            and watermark_score < 0.45
        ):
            if image_coverage_ratio >= 0.16:
                page_type = "mixed"
        if text_char_count >= 120 and text_coverage_ratio >= 0.02:
            page_type = "native"
        return page_type, text_coverage_ratio, image_coverage_ratio

    def _looks_like_garbled_text(self, text: str) -> bool:
        sample = (text or "").strip()
        if not sample:
            return True
        junk_patterns = ["���", "�", "\ufffd"]
        if any(pattern in sample for pattern in junk_patterns):
            return True
        non_printable = sum(1 for char in sample if ord(char) < 32 and char not in "\n\r\t")
        return non_printable > 10

    def _detect_watermark_regions(self, text_dict: Dict[str, Any], page: fitz.Page) -> List[FilterRegion]:
        matches: List[FilterRegion] = []
        for block in text_dict.get("blocks") or []:
            if block.get("type") != 0:
                continue
            for line in block.get("lines") or []:
                spans = line.get("spans") or []
                text, bbox = self._line_text_and_bbox(block, line)
                if not text:
                    continue
                for span in spans:
                    span_text = str(span.get("text") or "").strip()
                    if not span_text:
                        continue
                    span_bbox = span.get("bbox") or [0, 0, 0, 0]
                    if len(span_bbox) != 4:
                        continue
                    width = max(0.0, float(span_bbox[2]) - float(span_bbox[0]))
                    height = max(0.0, float(span_bbox[3]) - float(span_bbox[1]))
                    font_size = float(span.get("size") or 0.0)
                    color = int(span.get("color") or 0)
                    is_pattern_match = any(pattern and pattern in span_text for pattern in self.watermark_patterns)
                    is_large_overlay = len(span_text) >= 4 and (font_size >= 18 or width >= 180) and height >= 12
                    is_light_color = color >= 0x888888
                    is_center_overlay = (
                        bbox[1] > float(page.rect.height) * 0.18
                        and bbox[3] < float(page.rect.height) * 0.82
                    )
                    if is_pattern_match or (is_large_overlay and is_light_color and is_center_overlay):
                        matches.append(FilterRegion(type="watermark", bbox=bbox, content=span_text))
        return matches

    def _detect_image_watermarks(self, page: fitz.Page) -> List[FilterRegion]:
        if imagehash is None or Image is None:
            return []
        images = page.get_images(full=True)
        if not images:
            return []
        matches: List[FilterRegion] = []
        for image_index, image in enumerate(images[:4]):
            xref = image[0]
            try:
                pix = fitz.Pixmap(page.parent, xref)
                if pix.alpha:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                _ = imagehash.phash(pil_image)
                page_area = self._page_pixel_area(page)
                image_area = float(pix.width * pix.height)
                if image_area / max(page_area, 1.0) < 0.12:
                    continue
            except Exception:
                continue
            bbox = [0.0, 0.0, float(page.rect.width), float(page.rect.height)]
            matches.append(FilterRegion(type="watermark", bbox=bbox, content=f"image_watermark_{image_index}"))
        return matches[:1]

    def _watermark_score(self, filter_regions: List[FilterRegion], page: fitz.Page) -> float:
        if not filter_regions:
            return 0.0
        page_area = self._page_pixel_area(page)
        score = 0.0
        for region in filter_regions:
            bbox = list(region.bbox or [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
            area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
            score += min(0.5, area / max(page_area, 1.0))
            if region.type == "watermark":
                score += 0.18
        return min(1.0, score)

    def _decide_ocr_required(
        self,
        *,
        page_type: str,
        text_char_count: int,
        text_span_count: int,
        text_coverage_ratio: float,
        image_coverage_ratio: float,
        watermark_score: float,
        normalized_text: str,
    ) -> Tuple[bool, str]:
        if page_type == "scanned":
            return True, "scanned_page"
        if self._looks_like_garbled_text(normalized_text):
            return True, "garbled_text"
        if page_type == "mixed":
            if text_char_count < 40 or text_span_count < 10:
                return True, "mixed_with_low_text"
            if text_coverage_ratio < 0.02 and image_coverage_ratio >= 0.18:
                return True, "mixed_image_dominant"
            return False, "mixed_but_text_is_sufficient"
        if watermark_score >= 0.55 and text_char_count < 60:
            return True, "watermark_noise_with_low_text"
        return False, "native_text_sufficient"

    def _detect_header_footer_regions(self, page: fitz.Page, text_dict: Dict[str, Any], repeated_lines: Counter[str]) -> List[FilterRegion]:
        width = float(page.rect.width or 0.0)
        height = float(page.rect.height or 0.0)
        matches: List[FilterRegion] = []
        for block in text_dict.get("blocks") or []:
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox") or [0, 0, 0, 0]
            if len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in bbox]
            text = normalize_whitespace(" ".join(
                "".join(str(span.get("text") or "") for span in line.get("spans") or [])
                for line in block.get("lines") or []
            ))
            if not text:
                continue
            repeated = repeated_lines.get(text, 0) >= 3
            if repeated and y0 < min(50.0, height * 0.08):
                matches.append(FilterRegion(type="header", bbox=[x0, y0, x1, y1], content=text))
            elif repeated and y1 > max(height - 50.0, height * 0.92):
                matches.append(FilterRegion(type="footer", bbox=[x0, y0, x1, y1], content=text))
            elif re.fullmatch(r"(?:-?\d+-?|第\s*\d+\s*页|\d+)", text) and (y0 < 50.0 or y1 > height - 50.0 or x1 > width * 0.8):
                matches.append(FilterRegion(type="footer", bbox=[x0, y0, x1, y1], content=text))
        return matches

    def _collect_repeated_lines(self, pages: Iterable[fitz.Page]) -> Counter[str]:
        counter: Counter[str] = Counter()
        for page in pages:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks") or []:
                if block.get("type") != 0:
                    continue
                bbox = block.get("bbox") or [0, 0, 0, 0]
                if len(bbox) != 4:
                    continue
                y0 = float(bbox[1])
                y1 = float(bbox[3])
                if y0 < 60.0 or y1 > float(page.rect.height or 0) - 60.0:
                    text = normalize_whitespace(" ".join(
                        "".join(str(span.get("text") or "") for span in line.get("spans") or [])
                        for line in block.get("lines") or []
                    ))
                    if text:
                        counter[text] += 1
        return counter

    def stage0_preprocess(self, pdf_path: Path) -> List[Stage0PageResult]:
        with fitz.open(str(pdf_path)) as doc:
            preview_pages = [doc.load_page(index) for index in range(min(20, len(doc)))]
            repeated_lines = self._collect_repeated_lines(preview_pages)
            results: List[Stage0PageResult] = []
            for index in range(len(doc)):
                page = doc.load_page(index)
                text_dict = page.get_text("dict")
                normalized_text = normalize_whitespace(page.get_text("text"))
                filter_regions = [
                    *self._detect_header_footer_regions(page, text_dict, repeated_lines),
                    *self._detect_watermark_regions(text_dict),
                    *self._detect_image_watermarks(page),
                ]
                text_metrics = self._text_metrics(text_dict)
                image_metrics = self._image_metrics(page, text_dict)
                watermark_score = self._watermark_score(filter_regions, page)
                page_type, text_coverage_ratio, image_coverage_ratio = self._detect_page_type(
                    page,
                    text_dict,
                    normalized_text,
                    text_metrics,
                    image_metrics,
                    watermark_score,
                )
                ocr_required, ocr_reason = self._decide_ocr_required(
                    page_type=page_type,
                    text_char_count=int(text_metrics["text_char_count"]),
                    text_span_count=int(text_metrics["text_span_count"]),
                    text_coverage_ratio=text_coverage_ratio,
                    image_coverage_ratio=image_coverage_ratio,
                    watermark_score=watermark_score,
                    normalized_text=normalized_text,
                )
                result = Stage0PageResult(
                    page_index=index + 1,
                    page_type=page_type,
                    filter_regions=filter_regions,
                    ocr_required=ocr_required,
                    text_coverage_ratio=text_coverage_ratio,
                    text_span_count=int(text_metrics["text_span_count"]),
                    text_line_count=int(text_metrics["text_line_count"]),
                    text_block_count=int(text_metrics["text_block_count"]),
                    text_char_count=int(text_metrics["text_char_count"]),
                    image_count=int(image_metrics["image_count"]),
                    image_coverage_ratio=image_coverage_ratio,
                    watermark_score=watermark_score,
                    ocr_decision_reason=ocr_reason,
                )
                results.append(result)
                self._save_json(
                    self.stage_dirs["stage_0_preprocess"] / f"page_{index + 1}.json",
                    result.to_dict(),
                )
            checkpoint = self._load_checkpoint()
            checkpoint["stage_0_preprocess"] = {
                "status": "completed",
                "pdf": str(pdf_path),
                "pages": len(results),
            }
            self._save_checkpoint(checkpoint)
            return results

    def _build_layout_elements(self, page: fitz.Page, page_index: int, text: str, tables_markdown: str, stage0: Stage0PageResult) -> List[LayoutElement]:
        elements: List[LayoutElement] = []
        text_blocks = page.get_text("blocks")
        reading_order = 1
        for block_index, block in enumerate(text_blocks):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, block_text = block[:5]
            normalized = normalize_whitespace(str(block_text or ""))
            if not normalized:
                continue
            subtype = "paragraph"
            if len(normalized) <= 40 and re.search(r"(第.+[章节])|^[一二三四五六七八九十]+[、.]", normalized):
                subtype = "heading"
            elements.append(
                LayoutElement(
                    id=f"elem_{page_index:03d}_{block_index:02d}",
                    type="text",
                    subtype=subtype,
                    bbox=[float(x0), float(y0), float(x1), float(y1)],
                    content=normalized,
                    reading_order=reading_order,
                    confidence=0.78,
                )
            )
            reading_order += 1
        if tables_markdown:
            elements.append(
                LayoutElement(
                    id=f"tbl_{page_index:03d}_00",
                    type="table",
                    subtype="data_table",
                    bbox=[0.0, 0.0, float(page.rect.width), float(page.rect.height)],
                    content=tables_markdown[:1200],
                    reading_order=reading_order,
                    confidence=0.82,
                )
            )
            reading_order += 1
        if stage0.image_count > 0 and stage0.page_type in {"mixed", "native"}:
            image_subtype = "data_viz"
            if any(token in text for token in ["组织结构图", "组织架构图", "组织机构图"]):
                image_subtype = "flowchart"
            elif any(token in text for token in ["增长图", "柱状图", "折线图", "饼图", "应用结构"]):
                image_subtype = "data_viz"
            elements.append(
                LayoutElement(
                    id=f"fig_{page_index:03d}_00",
                    type="image",
                    subtype=image_subtype,
                    bbox=[0.0, 0.0, float(page.rect.width), float(page.rect.height)],
                    content=text[:400],
                    reading_order=reading_order,
                    confidence=0.76,
                )
            )
        return elements

    def _is_cross_page_table(self, current_markdown: str, next_markdown: str) -> Tuple[bool, float]:
        if not current_markdown or not next_markdown:
            return False, 0.0
        current_lines = [line.strip() for line in current_markdown.splitlines() if line.strip()]
        next_lines = [line.strip() for line in next_markdown.splitlines() if line.strip()]
        if len(current_lines) < 2 or len(next_lines) < 2:
            return False, 0.0
        current_header = current_lines[0]
        next_header = next_lines[0]
        current_cols = current_header.count("|")
        next_cols = next_header.count("|")
        if current_cols <= 2 or next_cols <= 2:
            return False, 0.0
        col_align = 1.0 - abs(current_cols - next_cols) / max(current_cols, next_cols, 1)
        current_types = self._infer_table_column_types(current_lines[:5])
        next_types = self._infer_table_column_types(next_lines[:5])
        type_score = self._list_overlap_ratio(current_types, next_types)
        header_score = _levenshtein_ratio(current_header, next_header)
        score = col_align * 0.4 + type_score * 0.3 + header_score * 0.3
        return score >= settings.cross_page_merge_low, score

    def _list_overlap_ratio(self, left: List[str], right: List[str]) -> float:
        if not left or not right:
            return 0.0
        left_counter = Counter(left)
        right_counter = Counter(right)
        overlap = sum((left_counter & right_counter).values())
        return overlap / max(len(left), len(right), 1)

    def _infer_table_column_types(self, lines: List[str]) -> List[str]:
        rows: List[List[str]] = []
        for line in lines:
            if "|" not in line:
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells and not all(cell.replace("-", "") == "" for cell in cells):
                rows.append(cells)
        if not rows:
            return []
        width = max(len(row) for row in rows)
        types: List[str] = []
        for col_index in range(width):
            column_values = [row[col_index] for row in rows if col_index < len(row)]
            joined = " ".join(column_values)
            if "%" in joined or any(token in joined for token in ["比例", "比率", "占比"]):
                types.append("percent")
            elif re.search(r"\d{4}[-/年]\d{1,2}", joined):
                types.append("date")
            elif any(token in joined for token in ["万元", "亿元", "元", "金额", "收入", "成本"]):
                types.append("amount")
            elif re.fullmatch(r"[\d,.\- ]+", joined.replace("%", "").strip()):
                types.append("number")
            else:
                types.append("text")
        return types

    def stage1_layout(self, pdf_path: Path, stage0_results: List[Stage0PageResult], pages: List[Dict[str, Any]]) -> List[Stage1PageLayout]:
        stage0_map = {item.page_index: item for item in stage0_results}
        with fitz.open(str(pdf_path)) as doc:
            results: List[Stage1PageLayout] = []
            for page_payload in pages:
                page_index = int(page_payload["page_number"])
                page = doc.load_page(page_index - 1)
                stage0 = stage0_map[page_index]
                elements = self._build_layout_elements(
                    page=page,
                    page_index=page_index,
                    text=str(page_payload.get("text") or ""),
                    tables_markdown=str(page_payload.get("tables_markdown") or ""),
                    stage0=stage0,
                )
                layout = Stage1PageLayout(page_index=page_index, elements=elements)
                results.append(layout)
                self._save_json(
                    self.stage_dirs["stage_1_layout"] / f"page_{page_index}.json",
                    layout.to_dict(),
                )
            for index in range(len(pages) - 1):
                current = pages[index]
                nxt = pages[index + 1]
                matched, score = self._is_cross_page_table(
                    str(current.get("tables_markdown") or ""),
                    str(nxt.get("tables_markdown") or ""),
                )
                if matched:
                    for layout in results:
                        if layout.page_index == int(current["page_number"]):
                            for element in layout.elements:
                                if element.type == "table":
                                    element.span_pages = [int(current["page_number"]), int(nxt["page_number"])]
                                    element.metadata["merge_confidence"] = score
                                    element.metadata["needs_review"] = score < settings.cross_page_merge_high
            checkpoint = self._load_checkpoint()
            checkpoint["stage_1_layout"] = {
                "status": "completed",
                "pdf": str(pdf_path),
                "pages": len(results),
            }
            self._save_checkpoint(checkpoint)
            return results

    def _extract_markdown_headers(self, markdown: str) -> List[Dict[str, Any]]:
        lines = [line.strip() for line in markdown.splitlines() if line.strip()]
        if not lines:
            return []
        header_cells = [cell.strip() for cell in lines[0].strip("|").split("|")]
        headers: List[Dict[str, Any]] = []
        for col_index, header in enumerate(header_cells):
            if header:
                headers.append({"level": 1, "text": header, "col_index": col_index, "span": [col_index, col_index]})
        return headers

    def _extract_markdown_rows(self, markdown: str) -> List[List[str]]:
        rows: List[List[str]] = []
        for line in markdown.splitlines():
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if all(cell.replace("-", "") == "" for cell in cells):
                continue
            rows.append(cells)
        return rows[1:] if len(rows) > 1 else []

    def stage2_extract(
        self,
        pdf_path: Path,
        pages: List[Dict[str, Any]],
        layouts: List[Stage1PageLayout],
    ) -> Dict[str, List[Dict[str, Any]]]:
        tables: List[ExtractedTable] = []
        figures: List[ExtractedFigure] = []
        texts: List[Dict[str, Any]] = []
        images_dir = self.stage_dirs["stage_2_extraction"] / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        with fitz.open(str(pdf_path)) as doc:
            for page_payload, layout in zip(pages, layouts):
                page_index = int(page_payload["page_number"])
                text = str(page_payload.get("text") or "")
                texts.append(
                    {
                        "page_index": page_index,
                        "text": text,
                        "section_title": str(page_payload.get("section_title") or ""),
                        "page_type": str(page_payload.get("page_type") or "text"),
                    }
                )
                table_markdown = str(page_payload.get("tables_markdown") or "")
                if table_markdown:
                    table = ExtractedTable(
                        table_id=f"tbl_{page_index:03d}",
                        page=[page_index],
                        headers=self._extract_markdown_headers(table_markdown),
                        data=self._extract_markdown_rows(table_markdown),
                        merged_cells=[],
                        inline_images=[],
                        html="",
                        markdown=table_markdown,
                        extraction_backend="pdfplumber|camelot",
                        confidence=float(page_payload.get("type_confidence") or 0.8),
                    )
                    tables.append(table)
                    self._save_json(
                        self.stage_dirs["stage_2_extraction"] / f"{table.table_id}.json",
                        table.to_dict(),
                    )

                if any(element.type == "image" for element in layout.elements):
                    page = doc.load_page(page_index - 1)
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                    image_path = images_dir / f"page_{page_index}.png"
                    pix.save(str(image_path))
                    figure_type = "mixed_image"
                    if any(token in text for token in ["组织结构图", "组织架构图", "组织机构图"]):
                        figure_type = "flowchart"
                    elif any(token in text for token in ["增长图", "柱状图", "折线图", "饼图", "应用结构"]):
                        figure_type = "data_viz"
                    figure = ExtractedFigure(
                        figure_id=f"fig_{page_index:03d}",
                        page=page_index,
                        figure_type=figure_type,
                        file=str(image_path),
                        bbox=[0.0, 0.0, float(page.rect.width), float(page.rect.height)],
                        ocr_text=str(page_payload.get("handwriting") or ""),
                        caption=str(page_payload.get("section_title") or ""),
                        confidence=float(page_payload.get("type_confidence") or 0.75),
                    )
                    figures.append(figure)
                    self._save_json(
                        self.stage_dirs["stage_2_extraction"] / f"{figure.figure_id}.json",
                        figure.to_dict(),
                    )
        payload = {
            "texts": texts,
            "tables": [item.to_dict() for item in tables],
            "figures": [item.to_dict() for item in figures],
        }
        checkpoint = self._load_checkpoint()
        checkpoint["stage_2_extraction"] = {
            "status": "completed",
            "pdf": str(pdf_path),
            "text_pages": len(texts),
            "tables": len(tables),
            "figures": len(figures),
        }
        self._save_checkpoint(checkpoint)
        return payload

    def stage3_context(
        self,
        pages: List[Dict[str, Any]],
        layouts: List[Stage1PageLayout],
        extraction: Dict[str, List[Dict[str, Any]]],
    ) -> List[ContextLink]:
        links: List[ContextLink] = []
        page_text_map = {int(page["page_number"]): str(page.get("text") or "") for page in pages}
        page_section_map = {int(page["page_number"]): str(page.get("section_title") or "") for page in pages}
        for table_payload in extraction.get("tables", []):
            page_numbers = list(table_payload.get("page") or [])
            page_number = int(page_numbers[0]) if page_numbers else 0
            pre_context = page_text_map.get(page_number, "")[:200]
            post_context = page_text_map.get(page_number, "")[-200:]
            marker = f"[REF:{table_payload['table_id']}|type=data_table|page={page_number}|caption={page_section_map.get(page_number, '')}]"
            link = ContextLink(
                element_id=str(table_payload["table_id"]),
                type="table",
                pre_context=pre_context,
                post_context=post_context,
                chapter_title=page_section_map.get(page_number, ""),
                section_title=page_section_map.get(page_number, ""),
                marker_in_text=marker,
                is_continuation=len(page_numbers) > 1,
                page_mapping=[{"row_start": 0, "row_end": len(table_payload.get("data") or []), "page": page_number, "is_header": True}],
                merge_confidence=0.92 if len(page_numbers) > 1 else 0.0,
                needs_review=False,
            )
            links.append(link)
            self._save_json(
                self.stage_dirs["stage_3_context"] / f"{link.element_id}.json",
                link.to_dict(),
            )

        for figure_payload in extraction.get("figures", []):
            page_number = int(figure_payload["page"])
            marker = f"[REF:{figure_payload['figure_id']}|type={figure_payload['figure_type']}|page={page_number}|caption={page_section_map.get(page_number, '')}]"
            link = ContextLink(
                element_id=str(figure_payload["figure_id"]),
                type="figure",
                pre_context=page_text_map.get(page_number, "")[:200],
                post_context=page_text_map.get(page_number, "")[-200:],
                chapter_title=page_section_map.get(page_number, ""),
                section_title=page_section_map.get(page_number, ""),
                marker_in_text=marker,
                is_continuation=False,
                page_mapping=[],
                merge_confidence=0.0,
                needs_review=False,
            )
            links.append(link)
            self._save_json(
                self.stage_dirs["stage_3_context"] / f"{link.element_id}.json",
                link.to_dict(),
            )
        checkpoint = self._load_checkpoint()
        checkpoint["stage_3_context"] = {
            "status": "completed",
            "links": len(links),
        }
        self._save_checkpoint(checkpoint)
        return links

    def _table_facts(self, table: Dict[str, Any], page_section: str, source_pdf: str, page_number: int) -> List[StructuredFact]:
        facts: List[StructuredFact] = []
        headers = [str(item.get("text") or "") for item in table.get("headers") or []]
        rows = list(table.get("data") or [])
        for row_index, row in enumerate(rows[: settings.table_partition_max_rows]):
            if not row:
                continue
            row_header = str(row[0]).strip() if row else ""
            values = [str(cell).strip() for cell in row[1:] if str(cell).strip()]
            if not row_header or not values:
                continue
            facts.append(
                StructuredFact(
                    fact_id=f"{table['table_id']}_row_{row_index}",
                    title=headers[0] if headers else row_header,
                    value=" | ".join(values),
                    evidence=" | ".join(row),
                    fact_type="table_row",
                    page_number=page_number,
                    source_pdf=source_pdf,
                    source_element_id=str(table["table_id"]),
                    primary_type="table",
                    sub_type="data_table",
                    row_header=row_header,
                    col_header=" | ".join(headers[1:]) if len(headers) > 1 else "",
                    section_title=page_section,
                    confidence=float(table.get("confidence") or 0.8),
                )
            )
        if rows:
            facts.append(
                StructuredFact(
                    fact_id=f"{table['table_id']}_summary",
                    title=headers[0] if headers else "表格摘要",
                    value=f"本表共{len(rows)}行，主要字段包括：{'、'.join(headers[:6])}" if headers else f"本表共{len(rows)}行",
                    evidence=str(table.get("markdown") or "")[:800],
                    fact_type="table_summary",
                    page_number=page_number,
                    source_pdf=source_pdf,
                    source_element_id=str(table["table_id"]),
                    primary_type="table",
                    sub_type="table_summary",
                    section_title=page_section,
                    confidence=float(table.get("confidence") or 0.78),
                )
            )
        return facts

    def _figure_facts(self, figure: Dict[str, Any], page_text: str, source_pdf: str) -> List[StructuredFact]:
        page_number = int(figure["page"])
        figure_type = str(figure.get("figure_type") or "")
        title = str(figure.get("caption") or figure.get("figure_id") or "图表")
        facts: List[StructuredFact] = [
            StructuredFact(
                fact_id=f"{figure['figure_id']}_summary",
                title=title,
                value=f"{figure_type} 页面对象，已保留图片路径和 OCR 文本",
                evidence=str(figure.get("ocr_text") or page_text[:200]),
                fact_type=figure_type or "figure_summary",
                page_number=page_number,
                source_pdf=source_pdf,
                source_element_id=str(figure["figure_id"]),
                primary_type="figure",
                sub_type=figure_type or "figure",
                section_title=title,
                confidence=float(figure.get("confidence") or 0.75),
            )
        ]
        if figure_type == "flowchart":
            for match in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,20}(?:部|处|中心|委员会|大会|董事会|监事会|总经理)", page_text):
                facts.append(
                    StructuredFact(
                        fact_id=f"{figure['figure_id']}_{len(facts)}",
                        title="组织节点",
                        value=match,
                        evidence=page_text[:400],
                        fact_type="org_chart_node",
                        page_number=page_number,
                        source_pdf=source_pdf,
                        source_element_id=str(figure["figure_id"]),
                        primary_type="figure",
                        sub_type="org_chart",
                        entities=[match],
                        confidence=float(figure.get("confidence") or 0.74),
                    )
                )
        if figure_type == "data_viz":
            for match in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,20}(?:行业|领域).{0,10}?\d[\d,]*(?:\.\d+)?%?", page_text):
                facts.append(
                    StructuredFact(
                        fact_id=f"{figure['figure_id']}_{len(facts)}",
                        title="图表数据点",
                        value=match,
                        evidence=page_text[:400],
                        fact_type="chart_fact",
                        page_number=page_number,
                        source_pdf=source_pdf,
                        source_element_id=str(figure["figure_id"]),
                        primary_type="figure",
                        sub_type="chart_summary",
                        confidence=float(figure.get("confidence") or 0.72),
                    )
                )
        return facts

    def stage4_multimodal(
        self,
        pages: List[Dict[str, Any]],
        extraction: Dict[str, List[Dict[str, Any]]],
        links: List[ContextLink],
    ) -> List[StructuredFact]:
        link_map = {item.element_id: item for item in links}
        source_pdf = str(pages[0].get("source_pdf") or "") if pages else ""
        page_text_map = {int(page["page_number"]): str(page.get("text") or "") for page in pages}
        page_section_map = {int(page["page_number"]): str(page.get("section_title") or "") for page in pages}
        facts: List[StructuredFact] = []

        for table in extraction.get("tables", []):
            page_number = int((table.get("page") or [0])[0])
            table_facts = self._table_facts(table, page_section_map.get(page_number, ""), source_pdf, page_number)
            for fact in table_facts:
                if fact.source_element_id in link_map:
                    fact.marker_in_text = link_map[fact.source_element_id].marker_in_text
                facts.append(fact)

        for figure in extraction.get("figures", []):
            page_number = int(figure["page"])
            figure_facts = self._figure_facts(figure, page_text_map.get(page_number, ""), source_pdf)
            for fact in figure_facts:
                if fact.source_element_id in link_map:
                    fact.marker_in_text = link_map[fact.source_element_id].marker_in_text
                facts.append(fact)

        stage4_dir = self.stage_dirs["stage_4_multimodal"]
        for fact in facts:
            self._save_json(stage4_dir / f"{fact.fact_id}.json", fact.to_dict())
        checkpoint = self._load_checkpoint()
        checkpoint["stage_4_multimodal"] = {
            "status": "completed",
            "facts": len(facts),
        }
        self._save_checkpoint(checkpoint)
        return facts

    def run(
        self,
        *,
        pdf_path: Path,
        base_pages: List[Dict[str, Any]],
        force: bool = False,
    ) -> Dict[str, Any]:
        del force
        stage0 = self.stage0_preprocess(pdf_path)
        stage1 = self.stage1_layout(pdf_path, stage0, base_pages)
        stage2 = self.stage2_extract(pdf_path, base_pages, stage1)
        stage3 = self.stage3_context(base_pages, stage1, stage2)
        stage4 = self.stage4_multimodal(base_pages, stage2, stage3)
        return {
            "stage0": [item.to_dict() for item in stage0],
            "stage1": [item.to_dict() for item in stage1],
            "stage2": stage2,
            "stage3": [item.to_dict() for item in stage3],
            "stage4": [item.to_dict() for item in stage4],
        }
