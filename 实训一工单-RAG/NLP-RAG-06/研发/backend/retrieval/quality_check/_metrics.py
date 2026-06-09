# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations

import json
import re
from typing import Any

from backend.utils.retrieval import (
    keyword_score,
    keyword_overlap_score,
    normalize_text,
    extract_company_aliases,
    extract_focus_terms,
)


def preview(text: str, limit: int = 180) -> str:
    cleaned = normalize_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = normalize_text(text).lower()
    if not normalized:
        return []
    tokens: list[str] = []
    segments = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", normalized)
    for segment in segments:
        if re.fullmatch(r"[a-z0-9]+", segment):
            tokens.append(segment)
            continue
        if len(segment) == 1:
            tokens.append(segment)
            continue
        for n in (2, 3):
            if len(segment) >= n:
                tokens.extend(segment[index : index + n] for index in range(len(segment) - n + 1))
        if len(segment) <= 8:
            tokens.append(segment)
    return tokens


def compute_marker_boost(
    query: str,
    marker_ids: list[str],
    table_lookup: dict[str, dict[str, Any]],
    visual_lookup: dict[str, dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    best_boost = 0.0
    details: list[dict[str, Any]] = []
    for marker_id in marker_ids:
        if marker_id in table_lookup:
            payload = table_lookup[marker_id]
            raw_score = keyword_score(query, build_table_query_text(payload))
            boost = min(0.22, raw_score / 40.0)
            best_boost = max(best_boost, boost)
            details.append({"marker_id": marker_id, "type": "table", "raw_score": raw_score, "boost": boost})
        elif marker_id in visual_lookup:
            payload = visual_lookup[marker_id]
            raw_score = keyword_score(query, build_visual_query_text(payload))
            boost = min(0.16, raw_score / 45.0)
            best_boost = max(best_boost, boost)
            details.append({"marker_id": marker_id, "type": "visual", "raw_score": raw_score, "boost": boost})
    return best_boost, details


def compute_answer_boost(query: str, text: str) -> float:
    score = keyword_score(query, text)
    overlap = keyword_overlap_score(query, text)
    return min(0.15, score / 35.0) + min(0.08, overlap * 0.15)


def compute_focus_signal(query: str, text: str) -> tuple[float, dict[str, Any]]:
    focus_terms = extract_focus_terms(query)
    if not focus_terms:
        return 0.0, {"hits": []}
    hits = [term for term in focus_terms if term in text]
    signal = len(hits) * 0.05
    return min(0.2, signal), {"hits": hits, "count": len(hits)}


def compute_company_signal(query: str, text: str, doc_name: str = "") -> float:
    signal = 0.0
    company_in_doc = extract_company_aliases(doc_name)
    query_terms = extract_focus_terms(query)
    for term in company_in_doc:
        if term in text:
            signal += 0.06
        if any(alias in term or term in alias for alias in query_terms):
            signal += 0.08
    return signal


def compute_page_position_penalty(
    source_pages: list[int],
    total_pages: int,
    is_visual: bool = False,
) -> tuple[float, dict[str, Any]]:
    if not source_pages:
        return 0.0, {"penalty": 0.0, "reason": "no_pages"}
    if total_pages <= 0:
        return 0.0, {"penalty": 0.0, "reason": "total_pages_unknown"}
    first_page = min(page for page in source_pages if page > 0)
    last_page = max(source_pages)
    page_range = max(1, last_page - first_page)
    depth_score = first_page / max(1, total_pages)
    range_penalty = min(0.12, page_range * 0.015) if not is_visual else min(0.08, page_range * 0.01)
    position_penalty = min(0.15, depth_score * 0.2) + range_penalty
    return -position_penalty, {
        "penalty": -position_penalty,
        "first_page": first_page,
        "last_page": last_page,
        "total_pages": total_pages,
        "depth_score": depth_score,
    }


def compute_table_relevance_score_v2(
    query: str,
    table_text: str,
    *,
    marker_context: str = "",
    source_text_rank: int = 999,
    marker_rank: int = 999,
    source_text_score: float = 0.0,
) -> float:
    score = 0.0
    score += keyword_score(query, table_text)
    score += keyword_overlap_score(query, table_text) * 12.0
    focus_signal_val, focus_details = compute_focus_signal(query, table_text)
    score += max(0.0, focus_signal_val) * 20.0
    score += len(list(focus_details.get("hits") or [])) * 2.0
    if marker_context:
        score += keyword_score(query, marker_context) * 1.6
        score += keyword_overlap_score(query, marker_context) * 8.0
    score += max(0.0, float(source_text_score)) * 1.2
    score += max(0.0, 3.5 - (float(source_text_rank) - 1.0) * 0.75)
    score += max(0.0, 2.0 - (float(marker_rank) - 1.0) * 0.35)
    return score


def score_table_row_for_query(query: str, row_text: str) -> float:
    if not row_text:
        return 0.0
    score = keyword_score(query, row_text)
    score += keyword_overlap_score(query, row_text) * 10.0
    focus_terms = extract_focus_terms(query)
    score += sum(2.5 for term in focus_terms if term in row_text)
    return score


def build_table_query_text(item: dict[str, Any]) -> str:
    return json.dumps(item.get("local_table_result") or {}, ensure_ascii=False) + "\n" + json.dumps(
        item.get("vlm_result") or {},
        ensure_ascii=False,
    )


def build_visual_query_text(item: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(item.get("summary_text") or ""),
            str(item.get("search_text") or ""),
            str(item.get("visual_type") or ""),
        ]
    )
