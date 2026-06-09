# ������ţ��˹����� NLP-RAG-ͼ�����ݽ����������Ż�
from __future__ import annotations

"""RAG chunk building helpers."""

import json
from pathlib import Path
from typing import Any

import fitz

from backend.config import settings
from backend.schemas import SourceChunk
from backend.services.text_utils import (
    Chunk,
    build_chunks,
    dedupe_preserve_order,
    derive_content_tags,
    derive_layout_tags,
    make_chunk,
    stable_chunk_id,
)


def build_main_chunks(pages: list[dict[str, object]]) -> list[Chunk]:
    return build_chunks(pages, settings.chunk_size, settings.chunk_overlap)


def build_structured_chunk(
    page: dict[str, object],
    text: str,
    chunk_id: str,
    page_type: str,
    field_title: str,
    field_type: str,
    source: str,
    primary_type: str,
    sub_type: str,
    extra_content_tags: list[str] | None = None,
    structured_facts: list[dict[str, str]] | None = None,
    confidence: float = 0.9,
) -> Chunk:
    metadata = {
        "page_number": str(page["page_number"]),
        "logical_page": str(page.get("logical_page") or ""),
        "page_type": page_type,
        "section_title": str(page.get("section_title") or ""),
        "has_table": "1" if page.get("tables_markdown") else "0",
        "has_ocr": "1" if page.get("handwriting") else "0",
        "source": source,
        "source_pdf": str(page.get("source_pdf") or ""),
        "source_pdf_path": str(page.get("source_pdf_path") or ""),
        "field_title": field_title,
        "field_type": field_type,
    }
    normalized_text = text.strip()
    layout_tags = derive_layout_tags(
        page_type=page_type,
        has_table=bool(page.get("tables_markdown")),
        has_ocr=bool(page.get("handwriting")),
        parse_metadata=dict(page.get("parse_metadata") or {}),
    )
    content_tags = dedupe_preserve_order(
        [
            *derive_content_tags(str(page.get("section_title") or ""), field_title, normalized_text),
            *(extra_content_tags or []),
        ]
    )
    return make_chunk(
        chunk_id=chunk_id,
        text=normalized_text,
        page_number=int(page["page_number"]),
        logical_page=page.get("logical_page"),
        metadata=metadata,
        raw_text=normalized_text,
        normalized_text=normalized_text,
        search_text=normalized_text,
        primary_type=primary_type,
        sub_type=sub_type,
        layout_tags=layout_tags,
        content_tags=content_tags,
        structured_facts=structured_facts or [],
        confidence=confidence,
    )


def _make_structured_fact(title: str, value: str, evidence: str, ftype: str, **extra: str) -> dict[str, str]:
    result = {"title": title, "value": value, "evidence": evidence, "type": ftype}
    result.update({k: v for k, v in extra.items() if v})
    return result


def build_pdf_intelligence_chunks(pages: list[dict[str, object]]) -> list[Chunk]:
    intelligence_chunks: list[Chunk] = []
    for page in pages:
        structured_items = list(page.get("structured_facts") or [])
        if not structured_items:
            continue
        for index, item in enumerate(structured_items):
            title = str(item.get("title") or item.get("fact_type") or "structured_fact").strip()
            value = str(item.get("value") or "").strip()
            evidence = str(item.get("evidence") or "").strip()
            fact_type = str(item.get("fact_type") or "structured_fact").strip()
            primary_type = str(item.get("primary_type") or page.get("primary_type") or "text").strip()
            sub_type = str(item.get("sub_type") or page.get("sub_type") or "paragraph").strip()
            if not value:
                continue

            text = f"\u5b57\u6bb5\uff1a{title}\n\u503c\uff1a{value}\n\u8bc1\u636e\uff1a{evidence}\n\u9875\u7801\uff1a{page['page_number']}".strip()
            chunk_id = stable_chunk_id(
                int(page["page_number"]),
                400000 + index,
                text,
                namespace=str(page.get("source_pdf") or ""),
            )
            extra_tags = [
                "pdf_intelligence",
                fact_type,
                str(item.get("section_title") or "").strip(),
            ]
            intelligence_chunks.append(
                build_structured_chunk(
                    page,
                    text,
                    chunk_id,
                    "structured",
                    title,
                    fact_type,
                    "pdf_intelligence",
                    primary_type,
                    sub_type,
                    [t for t in extra_tags if t],
                    [_make_structured_fact(
                        title, value, evidence, fact_type,
                        source_element_id=str(item.get("source_element_id") or ""),
                        marker_in_text=str(item.get("marker_in_text") or ""),
                    )],
                    float(item.get("confidence") or 0.86),
                )
            )
    return intelligence_chunks


def _is_valid_enhanced_item(item: dict[str, object]) -> bool:
    title = str(item.get("title") or "").strip()
    value = str(item.get("value") or "").strip()
    evidence = str(item.get("evidence") or "").strip()
    item_type = str(item.get("type") or "").strip()

    if not title or not value:
        return False

    for marker in ("\u65e0\u5177\u4f53", "\u672a\u62ab\u9732", "\u672a\u8bf4\u660e", "\u65e0\u6cd5\u5224\u65ad",
                   "\u65e0\u6cd5\u786e\u5b9a", "\u672a\u68c0\u7d22\u5230", "\u6ca1\u6709\u63d0\u53ca", "\u4e0d\u5984"):
        if marker in value:
            return False

    if item_type in {"field", "fact", "table_fact", "table_summary"} and len(evidence) < 6:
        return False

    return True


def build_enhanced_chunks(pages: list[dict[str, object]], llm: Any) -> list[Chunk]:
    if not settings.llm_enhancement_enabled:
        return []

    from backend.services.rag_pipeline.rag_utils import is_valid_enhanced_item
    enhanced_chunks: list[Chunk] = []
    for page in pages:
        structured_items = [
            item for item in llm.structure_page(page) if is_valid_enhanced_item(item)
        ]
        for index, item in enumerate(structured_items):
            title = str(item["title"])
            value = str(item["value"])
            evidence = str(item.get("evidence") or "")
            ftype = str(item.get("type") or "field")
            text = f"\u5b57\u6bb5\uff1a{title}\n\u503c\uff1a{value}\n\u8bc1\u636e\uff1a{evidence}\n\u9875\u7801\uff1a{page['page_number']}".strip()
            chunk_id = stable_chunk_id(
                int(page["page_number"]), 100000 + index, text,
                namespace=str(page.get("source_pdf") or "")
            )
            enhanced_chunks.append(
                build_structured_chunk(
                    page, text, chunk_id, "structured",
                    title, ftype, "llm_enhanced",
                    "form", "field_summary",
                    ["llm_enhanced"],
                    [{"title": title, "value": value, "evidence": evidence, "type": ftype}],
                    0.88,
                )
            )

        if settings.llm_table_analysis_enabled and page.get("tables_markdown"):
            table_items = [
                item for item in llm.analyze_table(page) if is_valid_enhanced_item(item)
            ]
            for table_index, item in enumerate(table_items, start=len(structured_items)):
                title = str(item["title"])
                value = str(item["value"])
                evidence = str(item.get("evidence") or "")
                ftype = str(item.get("type") or "table_trend")
                text = f"\u8868\u683c\u5206\u6790\uff1a{title}\n\u7ed3\u8bba\uff1a{value}\n\u8bc1\u636e\uff1a{evidence}\n\u9875\u7801\uff1a{page['page_number']}".strip()
                chunk_id = stable_chunk_id(
                    int(page["page_number"]), 200000 + table_index, text,
                    namespace=str(page.get("source_pdf") or "")
                )
                enhanced_chunks.append(
                    build_structured_chunk(
                        page, text, chunk_id, "table_analysis",
                        title, ftype, "llm_table_analysis",
                        "table", "table_summary",
                        ["table_analysis", "llm_enhanced"],
                        [{"title": title, "value": value, "evidence": evidence, "type": ftype}],
                        0.84,
                    )
                )
    return enhanced_chunks


def build_candidate_from_page(
    page: dict[str, object],
    *,
    text: str | None = None,
    score: float = 1.32,
    field_title: str = "",
    page_type: str | None = None,
    primary_type: str | None = None,
    sub_type: str | None = None,
    source: str = "page_fallback",
    structured_facts: list[dict[str, str]] | None = None,
    content_tags: list[str] | None = None,
) -> dict[str, object]:
    page_text = str(text or page.get("text") or page.get("tables_markdown") or "")
    derived_content_tags = dedupe_preserve_order(
        [
            *(content_tags or []),
            *derive_content_tags(
                str(page.get("section_title") or ""),
                field_title,
                page_text,
            ),
        ]
    )
    metadata = {
        "source_pdf": str(page.get("source_pdf") or ""),
        "source_pdf_path": str(page.get("source_pdf_path") or ""),
        "page_type": page_type or str(page.get("page_type") or "text"),
        "primary_type": primary_type or str(page.get("primary_type") or "text"),
        "sub_type": sub_type or str(page.get("sub_type") or "paragraph"),
        "section_title": str(page.get("section_title") or ""),
        "field_title": field_title,
        "content_tags": "|".join(derived_content_tags),
        "source": source,
        "structured_facts": json.dumps(structured_facts or [], ensure_ascii=False) if structured_facts else "",
    }
    chunk_id = stable_chunk_id(
        int(page.get("page_number") or 0),
        900000 + abs(hash(f"{metadata['source_pdf']}::{field_title}::{metadata['page_type']}")) % 10000,
        page_text,
        namespace=str(metadata["source_pdf"]),
    )
    return {
        "chunk_id": chunk_id,
        "page_number": int(page.get("page_number") or 0),
        "logical_page": page.get("logical_page"),
        "text": page_text,
        "score": min(1.0, score),
        "raw_score": score,
        "specialized_score": score,
        "metadata": metadata,
    }
