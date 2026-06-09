# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    logical_page: str | None
    metadata: Dict[str, str]
    raw_text: str = ""
    normalized_text: str = ""
    search_text: str = ""
    primary_type: str = "text"
    sub_type: str = "paragraph"
    layout_tags: List[str] = field(default_factory=list)
    content_tags: List[str] = field(default_factory=list)
    structured_facts: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.raw_text = self.raw_text or self.text
        self.normalized_text = normalize_whitespace(self.normalized_text or self.text)
        self.search_text = normalize_whitespace(self.search_text or self.normalized_text or self.text)
        self.text = self.search_text
        self.layout_tags = dedupe_preserve_order([tag for tag in self.layout_tags if tag])
        self.content_tags = dedupe_preserve_order([tag for tag in self.content_tags if tag])
        self.metadata = enrich_chunk_metadata(
            metadata=self.metadata,
            page_number=self.page_number,
            logical_page=self.logical_page,
            primary_type=self.primary_type,
            sub_type=self.sub_type,
            raw_text=self.raw_text,
            normalized_text=self.normalized_text,
            search_text=self.search_text,
            layout_tags=self.layout_tags,
            content_tags=self.content_tags,
            structured_facts=self.structured_facts,
            confidence=self.confidence,
        )


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def serialize_tag_list(tags: Sequence[str]) -> str:
    return "|".join(dedupe_preserve_order([tag.strip() for tag in tags if str(tag).strip()]))


def infer_chunk_types(page_type: str, field_type: str = "") -> tuple[str, str]:
    normalized_page_type = (page_type or "").strip().lower()
    normalized_field_type = (field_type or "").strip().lower()

    if normalized_page_type in {"table", "table_markdown"}:
        return "table", "simple_table"
    if normalized_page_type == "ocr":
        return "text", "ocr_text"
    if normalized_page_type == "structured":
        return "form", "field_summary"
    if normalized_page_type == "table_analysis":
        return "table", "table_summary"
    if normalized_page_type == "org_chart_summary":
        return "figure", "org_chart"
    if normalized_page_type == "chart_summary":
        return "figure", "chart_summary"
    if normalized_page_type == "vlm_structured":
        if normalized_field_type.startswith("org_chart"):
            return "figure", "org_chart"
        if normalized_field_type.startswith("chart_"):
            return "figure", "chart_summary"
        return "mixed", "visual_summary"
    return "text", "paragraph"


def derive_layout_tags(
    *,
    page_type: str,
    has_table: bool,
    has_ocr: bool,
    parse_metadata: Dict[str, object] | None = None,
) -> List[str]:
    tags: List[str] = []
    if has_table:
        tags.append("table_present")
    if has_ocr:
        tags.append("ocr_used")
    normalized_page_type = (page_type or "").strip().lower()
    if normalized_page_type == "ocr":
        tags.append("scanned_like")
    if normalized_page_type in {"table_analysis", "table_markdown"}:
        tags.append("table_layout")
    if normalized_page_type in {"org_chart_summary", "chart_summary", "vlm_structured"}:
        tags.append("visual_enhanced")

    parse_metadata = parse_metadata or {}
    if parse_metadata.get("has_table"):
        tags.append("table_present")
    if parse_metadata.get("used_rapidocr"):
        tags.append("rapidocr_used")
    if int(parse_metadata.get("image_count") or 0) >= 6:
        tags.append("image_heavy")
    return dedupe_preserve_order(tags)


def derive_content_tags(section_title: str, field_title: str, text: str) -> List[str]:
    section = section_title or ""
    field = field_title or ""
    payload = f"{section}\n{field}\n{text}"
    rules = {
        "fundraising": ["募集资金", "募投项目", "补充流动资金"],
        "revenue": ["营业收入", "主营业务收入", "收入构成"],
        "military_revenue": ["军用领域收入", "军用收入", "国防客户"],
        "shareholding": ["股本", "持股", "发行股数", "总股本"],
        "legal_representative": ["法定代表人"],
        "organization_structure": ["组织结构", "销售部", "销售处", "下设部门"],
        "chart_analysis": ["增长率", "增长图", "应用结构", "负增长"],
        "supplier": ["供应商", "上游", "下游"],
    }
    tags = [tag for tag, keywords in rules.items() if any(keyword in payload for keyword in keywords)]
    return dedupe_preserve_order(tags)


def enrich_chunk_metadata(
    *,
    metadata: Dict[str, str] | None,
    page_number: int,
    logical_page: str | None,
    primary_type: str,
    sub_type: str,
    raw_text: str,
    normalized_text: str,
    search_text: str,
    layout_tags: Sequence[str],
    content_tags: Sequence[str],
    structured_facts: Sequence[Dict[str, str]],
    confidence: float,
) -> Dict[str, str]:
    merged = {str(key): str(value) for key, value in dict(metadata or {}).items() if value is not None}
    merged["page_number"] = str(page_number)
    merged["logical_page"] = str(logical_page or "")
    merged["primary_type"] = primary_type
    merged["sub_type"] = sub_type
    merged["raw_text"] = raw_text
    merged["normalized_text"] = normalized_text
    merged["search_text"] = search_text
    merged["layout_tags"] = serialize_tag_list(layout_tags)
    merged["content_tags"] = serialize_tag_list(content_tags)
    merged["structured_facts"] = json.dumps(list(structured_facts), ensure_ascii=False) if structured_facts else ""
    merged["confidence"] = f"{confidence:.4f}"
    return merged


def make_chunk(
    *,
    chunk_id: str,
    text: str,
    page_number: int,
    logical_page: str | None,
    metadata: Dict[str, str] | None = None,
    raw_text: str = "",
    normalized_text: str = "",
    search_text: str = "",
    primary_type: str = "",
    sub_type: str = "",
    layout_tags: Sequence[str] | None = None,
    content_tags: Sequence[str] | None = None,
    structured_facts: Sequence[Dict[str, str]] | None = None,
    confidence: float = 1.0,
) -> Chunk:
    metadata_dict = dict(metadata or {})
    inferred_primary, inferred_sub = infer_chunk_types(
        metadata_dict.get("page_type", ""),
        metadata_dict.get("field_type", ""),
    )
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        page_number=page_number,
        logical_page=logical_page,
        metadata=metadata_dict,
        raw_text=raw_text or text,
        normalized_text=normalized_text or text,
        search_text=search_text or normalized_text or text,
        primary_type=primary_type or inferred_primary,
        sub_type=sub_type or inferred_sub,
        layout_tags=list(layout_tags or []),
        content_tags=list(content_tags or []),
        structured_facts=list(structured_facts or []),
        confidence=confidence,
    )


def stable_chunk_id(page_number: int, offset: int, text: str, namespace: str = "") -> str:
    digest = hashlib.md5(f"{namespace}:{page_number}:{offset}:{text}".encode("utf-8")).hexdigest()
    return f"chunk-{digest[:16]}"


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [segment.strip() for segment in re.split(r"\n{2,}", text) if segment.strip()]
    if not paragraphs:
        return _sliding_window(text, chunk_size, overlap)

    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        sentence_parts = _split_long_paragraph(paragraph, chunk_size, overlap)
        chunks.extend(sentence_parts[:-1])
        current = sentence_parts[-1]

    if current:
        chunks.append(current)

    return _merge_small_chunks(chunks, chunk_size, overlap)


def _sliding_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _split_long_paragraph(text: str, chunk_size: int, overlap: int) -> List[str]:
    sentences = [segment.strip() for segment in re.split(r"(?<=[。！？；;.!?])", text) if segment.strip()]
    if len(sentences) <= 1:
        return _sliding_window(text, chunk_size, overlap)

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current}{sentence}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) > chunk_size:
            chunks.extend(_sliding_window(sentence, chunk_size, overlap))
            current = ""
        else:
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _merge_small_chunks(chunks: List[str], chunk_size: int, overlap: int) -> List[str]:
    if not chunks:
        return []
    merged: List[str] = []
    current = chunks[0]
    for chunk in chunks[1:]:
        candidate = f"{current}\n\n{chunk}"
        if len(current) < max(120, overlap * 2) and len(candidate) <= chunk_size:
            current = candidate
        else:
            merged.append(current)
            current = chunk
    merged.append(current)
    return merged


def build_chunks(
    pages: Sequence[Dict[str, object]],
    chunk_size: int,
    overlap: int,
) -> List[Chunk]:
    results: List[Chunk] = []
    for page in pages:
        page_text = str(page.get("text") or "").strip()
        if not page_text:
            continue

        page_type = str(page.get("page_type") or "text")
        section_title = str(page.get("section_title") or "")
        source_pdf = str(page.get("source_pdf") or "")
        source_pdf_path = str(page.get("source_pdf_path") or "")
        parse_metadata = dict(page.get("parse_metadata") or {})
        has_table = bool(page.get("tables_markdown"))
        has_ocr = bool(page.get("handwriting"))
        primary_type = str(page.get("primary_type") or "")
        sub_type = str(page.get("sub_type") or "")
        if not primary_type or not sub_type:
            inferred_primary, inferred_sub = infer_chunk_types(page_type)
            primary_type = primary_type or inferred_primary
            sub_type = sub_type or inferred_sub
        layout_tags = derive_layout_tags(
            page_type=page_type,
            has_table=has_table,
            has_ocr=has_ocr,
            parse_metadata=parse_metadata,
        )
        layout_tags = dedupe_preserve_order([*list(page.get("layout_tags") or []), *layout_tags])
        segments = split_text(page_text, chunk_size, overlap)
        offset = 0

        for segment in segments:
            normalized_segment = normalize_whitespace(segment)
            if not normalized_segment:
                continue
            chunk_id = stable_chunk_id(int(page["page_number"]), offset, normalized_segment, namespace=source_pdf)
            metadata = {
                "page_number": str(page["page_number"]),
                "logical_page": str(page.get("logical_page") or ""),
                "page_type": page_type,
                "section_title": section_title,
                "has_table": "1" if has_table else "0",
                "has_ocr": "1" if has_ocr else "0",
                "source": str(page.get("source") or "builtin"),
                "source_pdf": source_pdf,
                "source_pdf_path": source_pdf_path,
            }
            content_tags = dedupe_preserve_order(
                [*list(page.get("content_tags") or []), *derive_content_tags(section_title, "", normalized_segment)]
            )
            results.append(
                make_chunk(
                    chunk_id=chunk_id,
                    text=normalized_segment,
                    page_number=int(page["page_number"]),
                    logical_page=page.get("logical_page"),
                    metadata=metadata,
                    raw_text=segment,
                    normalized_text=normalized_segment,
                    search_text=normalized_segment,
                    primary_type=primary_type,
                    sub_type=sub_type,
                    layout_tags=layout_tags,
                    content_tags=content_tags,
                )
            )
            offset += max(1, len(normalized_segment) - overlap)

        table_text = normalize_whitespace(str(page.get("tables_markdown") or ""))
        if table_text:
            table_segments = split_text(table_text, chunk_size, overlap)
            table_offset = 500000
            for segment in table_segments:
                normalized_segment = normalize_whitespace(segment)
                if not normalized_segment:
                    continue
                chunk_id = stable_chunk_id(int(page["page_number"]), table_offset, normalized_segment, namespace=source_pdf)
                metadata = {
                    "page_number": str(page["page_number"]),
                    "logical_page": str(page.get("logical_page") or ""),
                    "page_type": "table_markdown",
                    "section_title": section_title,
                    "has_table": "1",
                    "has_ocr": "1" if page.get("handwriting") else "0",
                    "source": str(page.get("source") or "builtin"),
                    "source_pdf": source_pdf,
                    "source_pdf_path": source_pdf_path,
                }
                content_tags = dedupe_preserve_order(
                    [
                        *list(page.get("content_tags") or []),
                        *derive_content_tags(section_title, "", normalized_segment),
                        "table_content",
                    ]
                )
                results.append(
                    make_chunk(
                        chunk_id=chunk_id,
                        text=normalized_segment,
                        page_number=int(page["page_number"]),
                        logical_page=page.get("logical_page"),
                        metadata=metadata,
                        raw_text=segment,
                        normalized_text=normalized_segment,
                        search_text=normalized_segment,
                        primary_type=primary_type if primary_type == "table" else "table",
                        sub_type=sub_type if primary_type == "table" else "simple_table",
                        layout_tags=dedupe_preserve_order([*layout_tags, "table_layout"]),
                        content_tags=content_tags,
                    )
                )
                table_offset += max(1, len(normalized_segment) - overlap)
    return results


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def keyword_overlap_score(query: str, text: str) -> float:
    query_terms = {term for term in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{1,}", query) if term.strip()}
    text_terms = {term for term in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{1,}", text) if term.strip()}
    if not query_terms or not text_terms:
        return 0.0
    hits = len(query_terms & text_terms)
    return hits / max(1, len(query_terms))
