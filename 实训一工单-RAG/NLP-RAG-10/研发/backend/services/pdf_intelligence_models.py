from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class FilterRegion:
    type: str
    bbox: List[float]
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Stage0PageResult:
    page_index: int
    page_type: str
    filter_regions: List[FilterRegion] = field(default_factory=list)
    ocr_required: bool = False
    text_coverage_ratio: float = 0.0
    text_span_count: int = 0
    text_line_count: int = 0
    text_block_count: int = 0
    text_char_count: int = 0
    image_count: int = 0
    image_coverage_ratio: float = 0.0
    watermark_score: float = 0.0
    ocr_decision_reason: str = ""
    raw_text_preview: str = ""
    filtered_text_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["filter_regions"] = [item.to_dict() for item in self.filter_regions]
        return payload


@dataclass
class LayoutElement:
    id: str
    type: str
    subtype: str
    bbox: List[float]
    content: str
    reading_order: int
    confidence: float = 0.0
    span_pages: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Stage1PageLayout:
    page_index: int
    elements: List[LayoutElement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_index": self.page_index,
            "elements": [item.to_dict() for item in self.elements],
        }


@dataclass
class InlineImage:
    cell: List[int]
    file: str
    image_type: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedTable:
    table_id: str
    page: List[int]
    headers: List[Dict[str, Any]] = field(default_factory=list)
    data: List[List[str]] = field(default_factory=list)
    merged_cells: List[Dict[str, Any]] = field(default_factory=list)
    inline_images: List[InlineImage] = field(default_factory=list)
    html: str = ""
    markdown: str = ""
    extraction_backend: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["inline_images"] = [item.to_dict() for item in self.inline_images]
        return payload


@dataclass
class ExtractedFigure:
    figure_id: str
    page: int
    figure_type: str
    file: str
    bbox: List[float]
    ocr_text: str = ""
    caption: str = ""
    structured: Dict[str, Any] = field(default_factory=dict)
    multimodal_description: str = ""
    model_used: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextLink:
    element_id: str
    type: str
    pre_context: str = ""
    post_context: str = ""
    chapter_title: str = ""
    section_title: str = ""
    marker_in_text: str = ""
    is_continuation: bool = False
    page_mapping: List[Dict[str, Any]] = field(default_factory=list)
    merge_confidence: float = 0.0
    needs_review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredFact:
    fact_id: str
    title: str
    value: str
    evidence: str
    fact_type: str
    page_number: int
    logical_page: str | None = None
    source_pdf: str = ""
    source_element_id: str = ""
    primary_type: str = "text"
    sub_type: str = "paragraph"
    entities: List[str] = field(default_factory=list)
    unit: str = ""
    row_header: str = ""
    col_header: str = ""
    chapter_title: str = ""
    section_title: str = ""
    marker_in_text: str = ""
    needs_review: bool = False
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
