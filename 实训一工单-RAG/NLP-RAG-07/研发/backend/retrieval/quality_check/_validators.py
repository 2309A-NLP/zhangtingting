# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from backend.config import settings
from backend.services.embedding import EmbeddingService
from backend.utils.retrieval import (
    MARKER_RE,
    normalize_text,
)
from ._schema import (
    load_jsonl,
    resolve_artifact_dir,
    infer_total_pages_from_chunks,
)
from ._metrics import (
    preview,
    tokenize_for_bm25,
    compute_marker_boost,
    compute_answer_boost,
    compute_focus_signal,
    compute_company_signal,
    compute_page_position_penalty,
    compute_table_relevance_score_v2,
    score_table_row_for_query,
    build_table_query_text,
    build_visual_query_text,
)


MOJIBAKE_HINT_RE = re.compile(r"[\u4e00-\u9fff]")


def looks_like_mojibake(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    hint_count = len(MOJIBAKE_HINT_RE.findall(normalized))
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", normalized))
    if hint_count >= 6:
        return True
    if hint_count >= 3 and hint_count * 4 >= max(1, cjk_count):
        return True
    return False


def stringify_text_list(values: list[Any]) -> str:
    cleaned = [normalize_text(str(value or "")) for value in values]
    cleaned = [value for value in cleaned if value]
    return "\uff1b".join(cleaned)


def build_visual_search_text_from_vlm(item: dict[str, Any]) -> str:
    structured = item.get("structured_content") or {}
    if not isinstance(structured, dict):
        structured = {}

    parts: list[str] = []
    visual_type = normalize_text(str(structured.get("visual_type") or ""))
    if visual_type:
        parts.append(f"visual_type: {visual_type}")

    source_pages = [str(page).strip() for page in list(item.get("source_pages") or []) if str(page).strip()]
    if source_pages:
        parts.append(f"pages: {', '.join(source_pages)}")

    final_object_id = normalize_text(str(item.get("final_object_id") or ""))
    if final_object_id:
        parts.append(f"visual_id: {final_object_id}")

    for field in [
        normalize_text(str(item.get("content") or "")),
        normalize_text(str(structured.get("summary") or "")),
        normalize_text(str(structured.get("detailed_description") or "")),
    ]:
        if field:
            parts.append(field)

    group_summaries = []
    for group in list(structured.get("groups") or []):
        if not isinstance(group, dict):
            continue
        summary = normalize_text(str(group.get("summary") or ""))
        if summary:
            group_summaries.append(summary)
    if group_summaries:
        parts.append("groups: " + stringify_text_list(group_summaries))

    flow = structured.get("flow_description") or {}
    if isinstance(flow, dict):
        flow_parts = [
            normalize_text(str(flow.get("start") or "")),
            stringify_text_list(list(flow.get("steps") or [])),
            stringify_text_list(list(flow.get("decision_points") or [])),
            normalize_text(str(flow.get("end") or "")),
        ]
        flow_parts = [value for value in flow_parts if value]
        if flow_parts:
            parts.append("flow: " + " -> ".join(flow_parts))

    chart = structured.get("chart_analysis") or {}
    if isinstance(chart, dict):
        chart_parts: list[str] = []
        for key in ["chart_type", "x_axis", "y_axis", "trend_summary"]:
            value = normalize_text(str(chart.get(key) or ""))
            if value:
                chart_parts.append(value)
        for key in ["series", "max_points", "min_points", "turning_points", "comparison_points"]:
            values = stringify_text_list(list(chart.get(key) or []))
            if values:
                chart_parts.append(values)
        if chart_parts:
            parts.append("chart: " + "\uff1b".join(chart_parts))

    for key in ["labels", "numbers", "relations", "key_observations", "notes"]:
        values = stringify_text_list(list(structured.get(key) or []))
        if values:
            parts.append(f"{key}: {values}")

    return "\n".join(part for part in parts if part).strip()


def load_visual_vlm_results(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    result_file = artifact_dir / "vlm_results.jsonl"
    if not result_file.exists():
        return {}
    rows = load_jsonl(result_file)
    result_map: dict[str, dict[str, Any]] = {}
    for item in rows:
        if item.get("task_type") != "figure_or_image_understanding":
            continue
        if item.get("status") != "success":
            continue
        visual_id = str(item.get("final_object_id") or "").strip()
        if visual_id:
            result_map[visual_id] = item
    return result_map


def hydrate_visual_row_from_vlm(row: dict[str, Any], vlm_item: dict[str, Any] | None) -> dict[str, Any]:
    if not vlm_item:
        return dict(row)
    hydrated = dict(row)
    better_summary = normalize_text(str(vlm_item.get("content") or ""))
    better_search = build_visual_search_text_from_vlm(vlm_item)
    current_summary = str(hydrated.get("summary_text") or "")
    current_search = str(hydrated.get("search_text") or "")
    if better_summary and (not current_summary.strip() or looks_like_mojibake(current_summary)):
        hydrated["summary_text"] = better_summary
    if better_search and (not current_search.strip() or looks_like_mojibake(current_search)):
        hydrated["search_text"] = better_search
    if not str(hydrated.get("visual_type") or "").strip():
        structured = vlm_item.get("structured_content") or {}
        if isinstance(structured, dict):
            hydrated["visual_type"] = str(structured.get("visual_type") or "").strip()
    if not hydrated.get("page_number"):
        source_pages = list(vlm_item.get("source_pages") or [])
        if source_pages:
            hydrated["page_number"] = int(source_pages[0] or 0)
    hydrated["vlm_result"] = vlm_item
    return hydrated


def load_text_chunk_lookup(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    chunk_file = artifact_dir / "stage3_text_chunking" / "text_chunks.jsonl"
    if not chunk_file.exists():
        return {}
    return {
        str(item.get("chunk_id") or ""): item
        for item in load_jsonl(chunk_file)
        if str(item.get("chunk_id") or "")
    }


def load_text_chunk_sequence(artifact_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    chunk_file = artifact_dir / "stage3_text_chunking" / "text_chunks.jsonl"
    if not chunk_file.exists():
        return [], {}
    rows = load_jsonl(chunk_file)
    ordered = sorted(
        rows,
        key=lambda item: (
            int(item.get("chunk_index") or 0),
            str(item.get("chunk_id") or ""),
        ),
    )
    position_by_id = {
        str(item.get("chunk_id") or ""): index
        for index, item in enumerate(ordered)
        if str(item.get("chunk_id") or "")
    }
    return ordered, position_by_id


def get_marker_ids_from_text(markers: list[str], text: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    marker_values = list(markers or [])
    if text:
        marker_values.extend(match.group(0) for match in MARKER_RE.finditer(text))
    for marker in marker_values:
        marker_text = str(marker)
        match = MARKER_RE.search(marker_text)
        marker_id = match.group(2) if match else marker_text
        if marker_id and marker_id not in seen:
            seen.add(marker_id)
            ordered.append(marker_id)
    return ordered


def build_text_windows(
    row: dict[str, Any],
    ordered_chunks: list[dict[str, Any]],
    position_by_id: dict[str, int],
) -> list[dict[str, Any]]:
    chunk_id = str(row.get("chunk_id") or "")
    if chunk_id not in position_by_id:
        return []
    pos = position_by_id[chunk_id]
    candidate_positions = [
        [pos],
        [pos - 1, pos],
        [pos, pos + 1],
        [pos - 1, pos, pos + 1],
    ]
    windows: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, ...]] = set()
    for positions in candidate_positions:
        valid_positions = [index for index in positions if 0 <= index < len(ordered_chunks)]
        if not valid_positions:
            continue
        key = tuple(valid_positions)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        chunks = [ordered_chunks[index] for index in valid_positions]
        text = "\n\n".join(str(item.get("text") or "") for item in chunks if str(item.get("text") or "").strip())
        markers: list[str] = []
        source_pages: list[int] = []
        for item in chunks:
            markers.extend(list(item.get("markers") or []))
            source_pages.extend(list(item.get("source_pages") or []))
        window_marker_ids = get_marker_ids_from_text(markers, text)
        windows.append(
            {
                **row,
                "window_chunk_ids": [str(item.get("chunk_id") or "") for item in chunks],
                "window_text": text,
                "window_source_pages": sorted({int(page) for page in source_pages if int(page) > 0}),
                "markers": list(markers),
                "marker_ids": window_marker_ids,
            }
        )
    return windows


def collect_neighbor_marker_ids(
    row: dict[str, Any],
    ordered_chunks: list[dict[str, Any]],
    position_by_id: dict[str, int],
    radius: int = 1,
) -> list[str]:
    chunk_id = str(row.get("chunk_id") or "")
    pos = position_by_id.get(chunk_id)
    if pos is None:
        return list(row.get("marker_ids") or get_marker_ids_from_text(list(row.get("markers") or []), str(row.get("text") or "")))
    markers: list[str] = []
    text_parts: list[str] = []
    for neighbor_pos in range(pos - radius, pos + radius + 1):
        if not (0 <= neighbor_pos < len(ordered_chunks)):
            continue
        item = ordered_chunks[neighbor_pos]
        markers.extend(list(item.get("markers") or []))
        text_parts.append(str(item.get("text") or ""))
    return get_marker_ids_from_text(markers, "\n\n".join(text_parts))


def collect_neighbor_chunks(
    row: dict[str, Any],
    ordered_chunks: list[dict[str, Any]],
    position_by_id: dict[str, int],
    radius: int = 1,
) -> list[dict[str, Any]]:
    chunk_id = str(row.get("chunk_id") or "")
    pos = position_by_id.get(chunk_id)
    if pos is None:
        return [row]
    chunks: list[dict[str, Any]] = []
    for neighbor_pos in range(pos - radius, pos + radius + 1):
        if 0 <= neighbor_pos < len(ordered_chunks):
            chunks.append(ordered_chunks[neighbor_pos])
    return chunks


def build_neighbor_window_payload(
    row: dict[str, Any],
    ordered_chunks: list[dict[str, Any]],
    position_by_id: dict[str, int],
    radius: int = 1,
) -> dict[str, Any]:
    chunks = collect_neighbor_chunks(row, ordered_chunks, position_by_id, radius=radius)
    text = "\n\n".join(str(item.get("text") or "") for item in chunks if str(item.get("text") or "").strip())
    markers: list[str] = []
    source_pages: list[int] = []
    for item in chunks:
        markers.extend(list(item.get("markers") or []))
        source_pages.extend(list(item.get("source_pages") or []))
    return {
        "neighbor_window_text": text,
        "neighbor_window_chunk_ids": [str(item.get("chunk_id") or "") for item in chunks],
        "neighbor_window_source_pages": sorted({int(page) for page in source_pages if int(page) > 0}),
        "neighbor_marker_ids": get_marker_ids_from_text(markers, text),
    }


def expand_neighbor_candidates(
    base_rows: list[dict[str, Any]],
    ordered_chunks: list[dict[str, Any]],
    position_by_id: dict[str, int],
    chunk_lookup: dict[str, dict[str, Any]],
    *,
    seed_limit: int = 5,
) -> list[dict[str, Any]]:
    expanded_ids: list[str] = []
    seen: set[str] = set()
    seed_rows = base_rows[: max(seed_limit * 3, 12)]
    for row in seed_rows:
        chunk_id = str(row.get("chunk_id") or "")
        pos = position_by_id.get(chunk_id)
        if pos is None:
            continue
        for neighbor_pos in range(pos - 2, pos + 3):
            if not (0 <= neighbor_pos < len(ordered_chunks)):
                continue
            neighbor_id = str(ordered_chunks[neighbor_pos].get("chunk_id") or "")
            if neighbor_id and neighbor_id not in seen:
                seen.add(neighbor_id)
                expanded_ids.append(neighbor_id)
    score_map = {str(item.get("chunk_id") or ""): item for item in base_rows}
    expanded_rows: list[dict[str, Any]] = []
    for chunk_id in expanded_ids:
        if chunk_id in score_map:
            expanded_rows.append(score_map[chunk_id])
            continue
        payload = chunk_lookup.get(chunk_id, {})
        expanded_rows.append(
            {
                "chunk_id": chunk_id,
                "dense_score": 0.0,
                "bm25_score": 0.0,
                "overlap_score": 0.0,
                "hybrid_score": 0.0,
                "marker_boost": 0.0,
                "answer_boost": 0.0,
                "score": 0.0,
                "doc_name": "",
                "page_start": int(min(list(payload.get("source_pages") or [0]))),
                "page_end": int(max(list(payload.get("source_pages") or [0]))),
                "source_pages": list(payload.get("source_pages") or []),
                "markers": list(payload.get("markers") or []),
                "text": str(payload.get("text") or ""),
            }
        )
    return expanded_rows


def extract_linked_markers(text_rows: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in text_rows:
        for marker_id in get_marker_ids_from_text(list(row.get("markers") or []), str(row.get("preview_text") or row.get("text") or "")):
            if marker_id not in seen:
                seen.add(marker_id)
                ordered.append(marker_id)
    return ordered


def extract_linked_marker_candidates_v2(text_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for text_rank, row in enumerate(text_rows, start=1):
        text = str(row.get("window_text") or row.get("preview_text") or row.get("text") or "")
        marker_ids = get_marker_ids_from_text(list(row.get("markers") or []), text)
        for marker_rank, marker_id in enumerate(marker_ids, start=1):
            candidates.append(
                {
                    "marker_id": marker_id,
                    "text_rank": text_rank,
                    "marker_rank": marker_rank,
                    "source_text_score": float(row.get("score") or 0.0),
                    "marker_context": extract_marker_context_snippet_v2(text, marker_id),
                }
            )
    return candidates


def extract_marker_context_snippet_v2(text: str, marker_id: str, radius: int = 120) -> str:
    source = str(text or "")
    if not source or not marker_id:
        return ""
    match = re.search(rf"\[(?:TABLE|IMAGE):{re.escape(marker_id)}\]", source)
    if not match:
        match = re.search(re.escape(marker_id), source)
    if not match:
        return ""
    start = max(0, match.start() - radius)
    end = min(len(source), match.end() + radius)
    return source[start:end]


def extract_target_company(query: str) -> str:
    patterns = [
        r"([\u4e00-\u9fffA-Za-z0-9\uff08\uff09]{4,}?(?:\u80a1\u4efd\u6709\u9650\u516c\u53f8|\u6709\u9650\u8d23\u4efb\u516c\u53f8|\u96c6\u56e2\u80a1\u4efd\u6709\u9650\u516c\u53f8|\u96c6\u56e2\u6709\u9650\u516c\u53f8))",
    ]
    from backend.utils.retrieval import strip_company_query_prefixes
    sanitized_query = strip_company_query_prefixes(query)
    for pattern in patterns:
        match = re.search(pattern, sanitized_query)
        if match:
            return match.group(1).strip()
    return ""


def stringify_table_row(row: list[Any]) -> str:
    return " | ".join(str(cell).strip() for cell in row if str(cell).strip())


def select_table_rows_for_query(query: str, headers: list[str], rows: list[list[Any]], *, max_rows: int) -> list[str]:
    header_text = stringify_table_row(headers)
    scored: list[tuple[int, float, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        row_text = stringify_table_row(row)
        if not row_text:
            continue
        combined_text = f"{header_text} | {row_text}" if header_text else row_text
        scored.append((index, score_table_row_for_query(query, combined_text), row_text))

    positive = [item for item in scored if item[1] > 0]
    selected = positive if positive else scored[:max_rows]
    selected.sort(key=lambda item: (-item[1], item[0]))
    top_items = sorted(selected[:max_rows], key=lambda item: item[0])
    return [item[2] for item in top_items]


def build_table_context_text(table_id: str, table_lookup: dict[str, dict[str, Any]], query: str = "") -> str:
    item = table_lookup.get(table_id) or {}
    local_result = item.get("local_table_result") or {}
    vlm_result = item.get("vlm_result") or {}
    lines: list[str] = []

    local_segments = list(local_result.get("segments") or [])
    if local_segments:
        headers: list[str] = []
        merged_rows: list[list[Any]] = []
        for segment in local_segments:
            if not isinstance(segment, dict):
                continue
            if not headers:
                headers = list(segment.get("headers") or [])
            merged_rows.extend(list(segment.get("rows") or []))
        if headers:
            lines.append("HEADER: " + " | ".join(str(cell) for cell in headers))
        for row_text in select_table_rows_for_query(query, headers, merged_rows, max_rows=18):
            lines.append("ROW: " + row_text)

    structured = (vlm_result.get("structured_content") or {}) if isinstance(vlm_result, dict) else {}
    if structured:
        headers = list(structured.get("headers") or [])
        rows = list(structured.get("rows") or [])
        if headers:
            lines.append("STRUCTURED_HEADER: " + " | ".join(str(cell) for cell in headers))
        for row_text in select_table_rows_for_query(query, headers, rows, max_rows=18):
            lines.append("STRUCTURED_ROW: " + row_text)

    return "\n".join(line for line in lines if line.strip())


def build_table_full_row_texts_v2(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines: list[str] = []
    if headers:
        lines.append("HEADER: " + " | ".join(str(cell) for cell in headers))
    for row in rows:
        if not isinstance(row, list):
            continue
        row_text = stringify_table_row(row)
        if row_text:
            lines.append("ROW: " + row_text)
    return lines


def build_table_context_text_v2(
    table_id: str,
    table_lookup: dict[str, dict[str, Any]],
    query: str = "",
    *,
    include_all_rows: bool = False,
) -> str:
    item = table_lookup.get(table_id) or {}
    local_result = item.get("local_table_result") or {}
    vlm_result = item.get("vlm_result") or {}
    lines: list[str] = []

    local_segments = list(local_result.get("segments") or [])
    if local_segments:
        headers: list[str] = []
        merged_rows: list[list[Any]] = []
        for segment in local_segments:
            if not isinstance(segment, dict):
                continue
            if not headers:
                headers = list(segment.get("headers") or [])
            merged_rows.extend(list(segment.get("rows") or []))
        if include_all_rows:
            lines.extend(build_table_full_row_texts_v2(headers, merged_rows))
        else:
            if headers:
                lines.append("HEADER: " + " | ".join(str(cell) for cell in headers))
            for row_text in select_table_rows_for_query(query, headers, merged_rows, max_rows=18):
                lines.append("ROW: " + row_text)

    structured = (vlm_result.get("structured_content") or {}) if isinstance(vlm_result, dict) else {}
    if structured:
        headers = list(structured.get("headers") or [])
        rows = list(structured.get("rows") or [])
        if include_all_rows:
            if headers:
                lines.append("STRUCTURED_HEADER: " + " | ".join(str(cell) for cell in headers))
            for row in rows:
                if not isinstance(row, list):
                    continue
                row_text = stringify_table_row(row)
                if row_text:
                    lines.append("STRUCTURED_ROW: " + row_text)
        else:
            if headers:
                lines.append("STRUCTURED_HEADER: " + " | ".join(str(cell) for cell in headers))
            for row_text in select_table_rows_for_query(query, headers, rows, max_rows=18):
                lines.append("STRUCTURED_ROW: " + row_text)

    return "\n".join(line for line in lines if line.strip())


def is_industry_chain_query(query: str) -> bool:
    return False


def is_strong_industry_chain_context(query: str, text: str) -> bool:
    return True


def build_visual_context_text(item: dict[str, Any]) -> str:
    lines = [
        f"VISUAL_TYPE: {str(item.get('visual_type') or '').strip()}",
        f"PAGE: {int(item.get('page_number') or 0)}",
    ]
    summary_text = normalize_text(str(item.get("summary_text") or ""))
    search_text = normalize_text(str(item.get("search_text") or ""))
    if summary_text:
        lines.append(summary_text)
    if search_text and search_text != summary_text:
        lines.append(search_text)
    return "\n".join(line for line in lines if line.strip())


def build_complete_context_text(row: dict[str, Any]) -> str:
    for key in ["window_text", "preview_text", "text"]:
        value = normalize_text(str(row.get(key) or ""))
        if value:
            return value
    return ""


def should_keep_context(query: str, text: str, doc_name: str = "") -> bool:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return False
    overlap = keyword_overlap_score(query, normalized_text)
    focus_signal, focus_details = compute_focus_signal(query, normalized_text)
    company_signal, _ = compute_company_signal(query, normalized_text, doc_name=doc_name)
    if list(focus_details.get("hits") or []):
        return True
    if overlap >= 0.18:
        return True
    if overlap >= 0.08 and company_signal >= -0.03:
        return True
    if focus_signal >= 0.02 and company_signal >= -0.03:
        return True
    return False


def infer_question_type(query: str) -> str:
    from backend.services.query_understanding import analyze_query
    intent = analyze_query(query)
    return str(intent.question_type or "")


def infer_query_tags(query: str) -> list[str]:
    from backend.services.query_understanding import analyze_query
    intent = analyze_query(query)
    return list(intent.query_tags or [])


def load_table_documents(artifact_dir: Path) -> list[dict[str, Any]]:
    preview_path = artifact_dir / "stage5_table_documents_preview.jsonl"
    if preview_path.exists():
        return load_jsonl(preview_path)
    return []


def search_tables_keyword(query: str, artifact_dir: Path, *, table_top_k: int = 5) -> list[dict[str, Any]]:
    docs = load_table_documents(artifact_dir)
    scored: list[dict[str, Any]] = []
    for doc in docs:
        local_text = json.dumps(doc.get("local_table_result") or {}, ensure_ascii=False)
        vlm_text = json.dumps(doc.get("vlm_result") or {}, ensure_ascii=False)
        merged_text = local_text + "\n" + vlm_text
        score = keyword_score(query, merged_text)
        if score <= 0:
            continue
        scored.append(
            {
                "table_id": str(doc.get("table_id") or ""),
                "page_start": int(doc.get("page_start") or 0),
                "page_end": int(doc.get("page_end") or 0),
                "score": score,
                "local_headers": (((doc.get("local_table_result") or {}).get("segments") or [{}])[0] or {}).get("headers") or [],
                "local_rows_preview": ((((doc.get("local_table_result") or {}).get("segments") or [{}])[0] or {}).get("rows") or [])[:3],
                "vlm_headers": (((doc.get("vlm_result") or {}).get("structured_content") or {}).get("headers") or []),
                "vlm_rows_preview": ((((doc.get("vlm_result") or {}).get("structured_content") or {}).get("rows") or [])[:3]),
            }
        )
    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    return scored[: max(1, table_top_k)]


def build_table_lookup(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    return {str(item.get("marker_id") or item.get("final_object_id") or item.get("table_id") or ""): item for item in load_table_documents(artifact_dir)}


def build_visual_lookup(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    path = artifact_dir / "stage4_vectorized_visuals" / "visual_vector_index.persisted.jsonl"
    if not path.exists():
        path = artifact_dir / "stage4_vectorized_visuals" / "visual_vector_index.jsonl"
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    vlm_map = load_visual_vlm_results(artifact_dir)
    lookup: dict[str, dict[str, Any]] = {}
    for item in rows:
        visual_id = str(item.get("marker_id") or item.get("visual_id") or "").strip()
        if not visual_id:
            continue
        lookup[visual_id] = hydrate_visual_row_from_vlm(item, vlm_map.get(visual_id))
    for visual_id, vlm_item in vlm_map.items():
        if visual_id in lookup:
            continue
        lookup[visual_id] = hydrate_visual_row_from_vlm(
            {
                "visual_id": visual_id,
                "marker_id": visual_id,
                "page_number": int(((vlm_item.get("source_pages") or [0]) or [0])[0] or 0),
                "visual_type": str(((vlm_item.get("structured_content") or {}) or {}).get("visual_type") or ""),
                "summary_text": "",
                "search_text": "",
                "minio_path": "",
            },
            vlm_item,
        )
    return lookup


def build_linked_table_results(
    marker_ids: list[str],
    table_lookup: dict[str, dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for marker_id in marker_ids:
        item = table_lookup.get(marker_id)
        if not item:
            continue
        local_text = json.dumps(item.get("local_table_result") or {}, ensure_ascii=False)
        vlm_text = json.dumps(item.get("vlm_result") or {}, ensure_ascii=False)
        score = keyword_score(query, local_text + "\n" + vlm_text)
        if score <= 0:
            continue
        rows.append(
            {
                "table_id": str(item.get("table_id") or marker_id),
                "page_start": int(item.get("page_start") or 0),
                "page_end": int(item.get("page_end") or 0),
                "score": score,
                "local_headers": (((item.get("local_table_result") or {}).get("segments") or [{}])[0] or {}).get("headers") or [],
                "local_rows_preview": ((((item.get("local_table_result") or {}).get("segments") or [{}])[0] or {}).get("rows") or [])[:3],
                "vlm_headers": (((item.get("vlm_result") or {}).get("structured_content") or {}).get("headers") or []),
                "vlm_rows_preview": ((((item.get("vlm_result") or {}).get("structured_content") or {}).get("rows") or [])[:3]),
                "from_text_marker": True,
            }
        )
    rows.sort(key=lambda item: (float(item["score"]), -int(item["page_start"])), reverse=True)
    return rows


def build_linked_table_results_v2(
    marker_candidates: list[dict[str, Any]],
    table_lookup: dict[str, dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    rows_by_table_id: dict[str, dict[str, Any]] = {}
    for candidate in marker_candidates:
        marker_id = str(candidate.get("marker_id") or "")
        item = table_lookup.get(marker_id)
        if not item:
            continue
        table_text = build_table_context_text_v2(marker_id, table_lookup, query=query, include_all_rows=True)
        score = compute_table_relevance_score_v2(
            query,
            table_text,
            marker_context=str(candidate.get("marker_context") or ""),
            source_text_rank=int(candidate.get("text_rank") or 999),
            marker_rank=int(candidate.get("marker_rank") or 999),
            source_text_score=float(candidate.get("source_text_score") or 0.0),
        )
        if score <= 0:
            continue
        table_id = str(item.get("table_id") or marker_id)
        payload = {
            "table_id": table_id,
            "page_start": int(item.get("page_start") or 0),
            "page_end": int(item.get("page_end") or 0),
            "score": score,
            "local_headers": (((item.get("local_table_result") or {}).get("segments") or [{}])[0] or {}).get("headers") or [],
            "local_rows_preview": ((((item.get("local_table_result") or {}).get("segments") or [{}])[0] or {}).get("rows") or [])[:3],
            "vlm_headers": (((item.get("vlm_result") or {}).get("structured_content") or {}).get("headers") or []),
            "vlm_rows_preview": ((((item.get("vlm_result") or {}).get("structured_content") or {}).get("rows") or [])[:3]),
            "from_text_marker": True,
            "marker_id": marker_id,
            "source_text_rank": int(candidate.get("text_rank") or 999),
            "marker_rank": int(candidate.get("marker_rank") or 999),
        }
        previous = rows_by_table_id.get(table_id)
        if previous is None or float(payload["score"]) > float(previous.get("score") or 0.0):
            rows_by_table_id[table_id] = payload
    rows = list(rows_by_table_id.values())
    rows.sort(
        key=lambda item: (
            float(item["score"]),
            -int(item.get("source_text_rank") or 999),
            -int(item.get("marker_rank") or 999),
            -int(item["page_start"]),
        ),
        reverse=True,
    )
    return rows
