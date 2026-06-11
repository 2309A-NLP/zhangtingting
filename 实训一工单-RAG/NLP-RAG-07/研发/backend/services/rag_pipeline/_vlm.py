# ������ţ��˹����� NLP-RAG-ͼ�����ݽ����������Ż�
from __future__ import annotations

"""PDF VLM enhancement helpers."""

import json
import re
from pathlib import Path
from typing import Any

import fitz

from backend.config import settings
from backend.services.rag_pipeline.rag_utils import (
    is_valid_enhanced_item,
    write_pdf_vlm_failure,
    sanitize_vlm_context,
)


def build_vlm_chunks(
    pages: list[dict[str, object]],
    pdf_path: Path,
    pdf_vlm: Any,
    cache_root: Path | None = None,
    failure_path: Path | None = None,
) -> tuple[list[Any], list[int], list[int], list[int], list[int]]:
    if not pdf_vlm.is_enabled():
        return [], [], [], [], []

    selected_indexes = list(range(len(pages)))
    if not selected_indexes:
        return [], [], [], [], []

    doc = fitz.open(str(pdf_path))
    enhanced_chunks: list[Any] = []
    failed_pages: list[int] = []
    cache_hit_pages: list[int] = []
    api_success_pages: list[int] = []
    cache_dir = (cache_root or (settings.artifact_dir / "pdf_vlm_cache")) / pdf_path.stem

    try:
        for page_index in selected_indexes:
            page = pages[page_index]
            page_number = int(page["page_number"])
            pdf_page = doc.load_page(page_index)
            pix = pdf_page.get_pixmap(
                matrix=fitz.Matrix(settings.pdf_vlm_render_scale, settings.pdf_vlm_render_scale),
                alpha=False,
            )
            try:
                image_first_result = pdf_vlm.enhance_page(
                    page_number=page_number,
                    logical_page=page.get("logical_page"),
                    local_text="",
                    table_markdown="",
                    image_bytes=pix.tobytes("png"),
                    cache_dir=cache_dir,
                    mode="image_only",
                    force_items=True,
                    cache_variant="image_first",
                )
                result = image_first_result

                if not list(result.get("items") or []):
                    safe_local_text, safe_table_markdown = sanitize_vlm_context(page)
                    if safe_local_text or safe_table_markdown:
                        result = pdf_vlm.enhance_page(
                            page_number=page_number,
                            logical_page=page.get("logical_page"),
                            local_text=safe_local_text,
                            table_markdown=safe_table_markdown,
                            image_bytes=pix.tobytes("png"),
                            cache_dir=cache_dir,
                            mode="full",
                            force_items=True,
                            cache_variant="assist",
                        )
            except Exception as exc:
                failed_pages.append(page_number)
                write_pdf_vlm_failure(
                    pdf_path,
                    page_number,
                    str(exc),
                    failed_pages,
                    output_path=failure_path,
                )
                raise RuntimeError(
                    f"PDF VLM enhancement failed on page {page_number}. "
                    f"failed_pages={failed_pages}. detail={exc}"
                ) from exc

            status = result.get("status")
            if status == "cache_hit":
                cache_hit_pages.append(page_number)
            elif status == "api_success":
                api_success_pages.append(page_number)
            elif status == "failed":
                failed_pages.append(page_number)

            items = [item for item in list(result.get("items") or []) if is_valid_enhanced_item(item)]
            for item_index, item in enumerate(items):
                item_type = str(item.get("type") or "field")
                page_type_hint = "vlm_structured"
                if item_type.startswith("org_chart"):
                    page_type_hint = "org_chart_summary"
                elif item_type.startswith("chart_"):
                    page_type_hint = "chart_summary"

                text = (
                    f"�ֶΣ�{item['title']}\n"
                    f"ֵ��{item['value']}\n"
                    f"֤�ݣ�{item.get('evidence') or ''}\n"
                    f"ҳ�룺{page_number}"
                ).strip()
                from backend.services.text_utils import stable_chunk_id
                chunk_id = stable_chunk_id(
                    page_number, 300000 + item_index, text,
                    namespace=str(page.get("source_pdf") or "")
                )

                extra_tags = ["pdf_vlm_enhanced"]
                primary_type = "mixed"
                sub_type = "visual_summary"

                from backend.services.rag_pipeline._chunks import build_structured_chunk
                chunk = build_structured_chunk(
                    page=page,
                    text=text,
                    chunk_id=chunk_id,
                    page_type=page_type_hint,
                    field_title=str(item["title"]),
                    field_type=item_type,
                    source="pdf_vlm_enhanced",
                    primary_type=primary_type,
                    sub_type=sub_type,
                    extra_content_tags=extra_tags,
                    structured_facts=[{
                        "title": str(item["title"]),
                        "value": str(item["value"]),
                        "evidence": str(item.get("evidence") or ""),
                        "type": item_type,
                    }],
                    confidence=0.82,
                )

                if item_type.startswith("org_chart"):
                    chunk.primary_type = "figure"
                    chunk.sub_type = "org_chart"
                    from backend.services.text_utils import dedupe_preserve_order
                    chunk.content_tags = dedupe_preserve_order([*chunk.content_tags, "organization_structure"])
                    chunk.metadata["primary_type"] = "figure"
                    chunk.metadata["sub_type"] = "org_chart"
                    chunk.metadata["content_tags"] = "|".join(chunk.content_tags)
                elif item_type.startswith("chart_"):
                    chunk.primary_type = "figure"
                    chunk.sub_type = "chart_summary"
                    from backend.services.text_utils import dedupe_preserve_order
                    chunk.content_tags = dedupe_preserve_order([*chunk.content_tags, "chart_analysis"])
                    chunk.metadata["primary_type"] = "figure"
                    chunk.metadata["sub_type"] = "chart_summary"
                    chunk.metadata["content_tags"] = "|".join(chunk.content_tags)

                enhanced_chunks.append(chunk)

    finally:
        doc.close()

    selected_pages = [int(pages[i]["page_number"]) for i in selected_indexes]
    return enhanced_chunks, selected_pages, failed_pages, cache_hit_pages, api_success_pages


def load_pdf_vlm_items(source_pdf: str, page_numbers: list[int] | None = None) -> list[dict[str, object]]:
    cache_dir = settings.artifact_dir / "pdf_vlm_cache" / Path(source_pdf).stem
    if not cache_dir.exists():
        return []

    allowed_pages = set(page_numbers or [])
    items: list[dict[str, object]] = []
    for cache_file in sorted(cache_dir.glob("page_*.json")):
        name = cache_file.name
        if ".raw." in name:
            continue
        match = re.match(r"page_(\d+)", name)
        if not match:
            continue
        page_number = int(match.group(1))
        if allowed_pages and page_number not in allowed_pages:
            continue
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            value = str(item.get("value") or "").strip()
            evidence = str(item.get("evidence") or "").strip()
            item_type = str(item.get("type") or "field").strip()
            if not title or not value:
                continue
            items.append({
                "source_pdf": source_pdf,
                "page_number": page_number,
                "title": title,
                "value": value,
                "evidence": evidence,
                "item_type": item_type,
            })
    return items
