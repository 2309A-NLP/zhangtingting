# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations

import json
import time
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import fitz

from backend.pipeline.stages.stage1_layout_analysis._bbox import bbox_area, is_inside, overlap_ratio
from backend.pipeline.stages.stage1_layout_analysis._config import (
    OUTPUT_DIR,
    DEFAULT_PAGE_RANGE,
    load_config,
    resolve_pdf_path,
    log,
    extract_text_blocks,
    extract_text_lines,
)
from backend.pipeline.stages.stage1_layout_analysis._table import (
    collect_repeated_lines,
    collect_position_dense_candidates,
    detect_header_footer_regions,
    detect_text_watermarks,
    detect_pymupdf_tables,
    detect_pipe_tables,
    detect_multiline_row_tables,
    detect_top_continuation_table,
    detect_table_by_lines,
    detect_table_by_text_alignment,
    dedupe_table_candidates,
    merge_same_page_continuation_tables,
    collect_texts_in_bbox,
    table_looks_like_flowchart,
    infer_table_sub_type,
    is_filtered_line,
    detect_cross_page_tables,
)
from backend.pipeline.stages.stage1_layout_analysis._visual import (
    extract_image_blocks,
    looks_like_flowchart_text,
    extract_nearby_text,
    detect_figure_regions,
    cluster_visual_candidates,
)


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
    reading_order: int = 0

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


KEY_VALUE_PATTERN = re.compile(r"[:：]")


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
        region.reading_order = i
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
            sub_type = "key-value"
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
    page_result = LayoutPage(
        page_index=page_num + 1,
        page_bbox=page_bbox,
        page_type=page_type,
        sub_type=sub_type,
        regions=regions,
    )
    page_result.text_flow = build_text_flow(page_result)
    return page_result


def main() -> None:
    started_at = time.time()
    config = load_config()
    pdf_path = resolve_pdf_path(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"PDF={pdf_path}")
    log(f"页面范围={DEFAULT_PAGE_RANGE}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    if DEFAULT_PAGE_RANGE:
        target_pages = [page_num for page_num in DEFAULT_PAGE_RANGE if page_num < total_pages]
    else:
        target_pages = list(range(total_pages))
    preview_pages = [doc.load_page(index) for index in target_pages]
    repeated_lines = collect_repeated_lines(preview_pages)
    watermark_patterns = load_watermark_patterns_from_config()
    position_dense_counter = collect_position_dense_candidates(preview_pages)

    pages: list[LayoutPage] = []
    for page_num in target_pages:
        log(f"正在处理 {page_num + 1} 页...")
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
            f"第 {page_num + 1} 页完成，区域={len(layout_page.regions)}个，"
            f"类型={layout_page.page_type}/{layout_page.sub_type}，耗时={time.time() - page_start:.2f}s"
        )

    log("开始跨页表格合并...")
    cross_page_tables, cross_page_debug = detect_cross_page_tables(pages, config)
    log(f"跨页表格分析完成，待选={len(cross_page_debug)}个，合并={len(cross_page_tables)}")

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
                "阶段1全程使用 PyMuPDF 进行页面解析",
                "执行顺序为文本/图片/图表提取 -> 表格检测 -> 图表聚类 -> 阅读顺序切分 -> 跨页表格合并",
                "本版本为无边框纯文本表格检测路径",
                "图表分类基于绘图的容器框、占比和关键词分类",
            ],
        },
        "technology_notes": [
            {
                "step": "layout_segmentation",
                "technology": "PyMuPDF",
                "description": "提取文本块、图片块、绘图段落的 bbox，进行分类和阅读顺序",
            },
            {
                "step": "table_detection",
                "technology": "边框检测 + 文本对齐",
                "description": "同时使用边框线和纯文本对齐两种策略检测表格",
            },
            {
                "step": "figure_classification",
                "technology": "启发式规则",
                "description": "基于绘图的包围框、箭头、容器框、关键词判断 data_viz / flowchart / mixed_image / inline_image",
            },
            {
                "step": "cross_page_table_detection",
                "technology": "多级匹配 + 启发式合并",
                "description": "扫描相邻页面配对，基于置信度合并为跨页表格",
            },
            {
                "step": "reading_order_recovery",
                "technology": "y-x 排序",
                "description": "基于相邻关系切分，同一列向下顺序排列",
            },
        ],
        "text_flow": text_flow_output,
        "structure_objects": structure_objects,
        "cross_page_table_debug": cross_page_debug,
        "repeated_header_footer_lines": dict(repeated_lines),
        "page_details": [page.to_dict() for page in pages],
    }

    output_path = OUTPUT_DIR / "stage1_layout_analysis.json"
    output_path.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"结果写入文件: {output_path}")
    log(f"总耗时: {time.time() - started_at:.2f}s")
    log(f"分析页数: {len(pages)}")
    log(f"表格数: {len(structure_objects['tables'])}")
    log(f"图表数: {len(structure_objects['figures'])}")
    log(f"图片数: {len(structure_objects['images'])}")


def load_watermark_patterns_from_config() -> list[str]:
    from backend.pipeline.stages.stage1_layout_analysis._config import load_watermark_patterns
    return load_watermark_patterns()


if __name__ == "__main__":
    main()
