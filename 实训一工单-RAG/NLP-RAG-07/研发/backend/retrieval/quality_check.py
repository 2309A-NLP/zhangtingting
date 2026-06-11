from __future__ import annotations

# å·¥åç¼å·ï¼äººå·¥æºè½ NLP-RAG-å¾ååå®¹è§£æåæ£ç´¢ä¼å

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.services.embedding import EmbeddingService
from backend.services.llm_client._answers import _clean_answer_text
from backend.utils.retrieval import (
    GENERIC_STOPWORDS,
    MARKER_RE,
    COMPANY_RE,
    PHRASE_HINTS,
    SimpleBM25Index,
    build_query_tokens,
    keyword_score,
    keyword_overlap_score,
    normalize_score_map,
    cosine_similarity,
    normalize_text,
    strip_company_query_prefixes,
    extract_company_aliases,
    extract_focus_terms,
    split_sentences,
    compute_answer_boost,
    compute_focus_signal,
    compute_company_signal,
    compute_page_position_penalty,
    infer_question_type,
    infer_query_tags,
)


MOJIBAKE_HINT_RE = re.compile(r"[ééé¥â¬é¥ééé¿ç¼ççµéºéå¦ç»«æµ æ¶]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple retrieval quality checker for processed PDF outputs.")
    parser.add_argument("--query", type=str, required=True, help="User query to test")
    parser.add_argument("--artifact-dir", type=str, default="", help="Artifact dir used by parse4 processing")
    parser.add_argument("--text-top-k", type=int, default=5, help="Top K text chunks")
    parser.add_argument("--text-candidate-k", type=int, default=20, help="Hybrid retrieval candidate pool size")
    parser.add_argument("--visual-top-k", type=int, default=3, help="Top K visual hits")
    parser.add_argument("--table-top-k", type=int, default=5, help="Top K table docs from keyword fallback")
    parser.add_argument("--disable-rerank", action="store_true", help="Disable reranking even if reranker is available")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM answer synthesis")
    parser.add_argument("--llm-context-k", type=int, default=6, help="Max evidence contexts passed to LLM")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


MOJIBAKE_HINT_RE = re.compile(r"[ééé¥â¬é¥ééé¿ç¼ççµéºéå¦ç»«æµ æ¶]")


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
    return "ï¼".join(cleaned)


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
            parts.append("chart: " + "ï¼".join(chart_parts))

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


def build_test_llm_prompt(query: str, contexts: list[dict[str, Any]], per_context_limit: int = 1600) -> str:
    evidence_blocks: list[str] = []
    for index, item in enumerate(contexts, start=1):
        metadata = dict(item.get("metadata") or {})
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        evidence_blocks.append(
            "\n".join(
                [
                    f"[è¯æ®{index}] é¡µç ={int(item.get('page_number') or 0)} ç±»å={str(metadata.get('page_type') or '')}",
                    text if str(metadata.get("page_type") or "") == "table" or bool(metadata.get("never_truncate")) else text[:per_context_limit],
                ]
            )
        )
    evidence_text = "\n\n".join(evidence_blocks)
    return (
        "ä½ æ¯ä¸ä¸ªä¸¥æ ¼åºäºæè¡ä¹¦è¯æ®åç­é®é¢çå©æã\n"
        "åªåè®¸ä¾æ®ç»å®è¯æ®ä½ç­ï¼ä¸åè®¸ä½¿ç¨å¤é¨ç¥è¯ï¼ä¸åè®¸çæµã\n"
        "å¦æè¯æ®ä¸è½ç´æ¥åç­é®é¢ï¼å°±æç¡®åç­ï¼æªæ£ç´¢å°è¶³å¤è¯æ®ã\n"
        "åç­æ¶è¯·ååè¯æ®å¤æ­ï¼åªæåä¸é®é¢è¯­ä¹ç´æ¥å¯¹åºçä¿¡æ¯ï¼å¿½ç¥åè¯å¼ä¹ãææ¯æè¿°ãè´¢å¡åªå£°åæ å³æ®µè½ã\n"
        "ä¸è¦è¾åºåé¨æ è¯ãè¡¨æ ¼IDãchunk_idãmarkerãsource_pdfãminio_pathç­åé¨å­æ®µã\n"
        "ä¸è¦ç§æå¤§æ®µåæï¼è¦æ´çæèªç¶ä¸­æå¥å­ã\n"
        "ç­æ¡æ«å°¾å¿é¡»éï¼å¼ç¨é¡µç ï¼é¡µç 1ãé¡µç 2ã\n\n"
        f"é®é¢ï¼{query}\n\n"
        f"è¯æ®ï¼\n{evidence_text}\n\n"
        "è¯·ç´æ¥ç»åºæç»ç­æ¡ã"
    )


def build_llm_retry_plan(contexts: list[dict[str, Any]]) -> list[dict[str, int]]:
    context_count = len(contexts)
    plans = [
        {"context_limit": min(context_count, 6), "per_context_limit": 1600, "max_tokens": settings.max_new_tokens},
        {"context_limit": min(context_count, 4), "per_context_limit": 1100, "max_tokens": min(settings.max_new_tokens, 220)},
        {"context_limit": min(context_count, 3), "per_context_limit": 800, "max_tokens": min(settings.max_new_tokens, 180)},
        {"context_limit": min(context_count, 2), "per_context_limit": 560, "max_tokens": min(settings.max_new_tokens, 160)},
    ]
    deduped: list[dict[str, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for item in plans:
        key = (int(item["context_limit"]), int(item["per_context_limit"]), int(item["max_tokens"]))
        if key[0] <= 0 or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_focus_snippet(query: str, text: str, max_sentences: int = 4) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return normalize_text(text)

    focus_terms = extract_focus_terms(query)
    scored: list[tuple[int, float, str]] = []
    for index, sentence in enumerate(sentences):
        score = keyword_score(query, sentence)
        score += keyword_overlap_score(query, sentence) * 10.0
        focus_hits = sum(1 for term in focus_terms if term in sentence)
        if focus_hits:
            score += focus_hits * 3.0
        if any(token in query for token in ["å¤å°", "éé¢", "æ¶å¥", "æ¯ä¾", "å æ¯", "è¡æ°"]) and re.search(
            r"\d[\d,]*(?:\.\d+)?(?:%|å|ä¸å|äº¿å|ä¸è¡|è¡)?",
            sentence,
        ):
            score += 2.0
        scored.append((index, score, sentence))

    positive = [item for item in scored if item[1] > 0]
    selected = positive if positive else scored[:max_sentences]
    selected.sort(key=lambda item: (-item[1], item[0]))
    top_items = sorted(selected[:max_sentences], key=lambda item: item[0])
    return " ".join(item[2] for item in top_items)


def resolve_artifact_dir(raw: str) -> Path:
    if raw:
        return Path(raw).resolve()
    return settings.artifact_dir / "stage2_precise_extraction_rewire_test"


def infer_total_pages_from_chunks(chunk_lookup: dict[str, dict[str, Any]]) -> int:
    total_pages = 0
    for item in chunk_lookup.values():
        for page in list(item.get("source_pages") or []):
            try:
                total_pages = max(total_pages, int(page))
            except Exception:
                continue
        try:
            total_pages = max(total_pages, int(item.get("page_end") or 0))
        except Exception:
            continue
    return total_pages




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


def search_text_vectors(
    query: str,
    artifact_dir: Path,
    table_lookup: dict[str, dict[str, Any]],
    visual_lookup: dict[str, dict[str, Any]],
    *,
    text_top_k: int = 5,
    text_candidate_k: int = 20,
    disable_rerank: bool = False,
) -> list[dict[str, Any]]:
    from pymilvus import Collection, connections

    connections.connect(alias="text_check", uri=settings.milvus_uri)
    collection = Collection(name=settings.text_vector_collection_name, using="text_check")
    collection.load()

    embedder = EmbeddingService(settings.model_dir / "embedding", configured_path=settings.embedding_model_path)
    try:
        from backend.services.rerank_service import RerankService

        reranker = RerankService(settings.reranker_model_path)
    except Exception:
        reranker = None
    chunk_lookup = load_text_chunk_lookup(artifact_dir)
    ordered_chunks, position_by_id = load_text_chunk_sequence(artifact_dir)
    total_pages = infer_total_pages_from_chunks(chunk_lookup)
    query_vector = embedder.embed_query(query)
    candidate_k = max(text_top_k, text_candidate_k)
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {}},
        limit=max(1, candidate_k),
        output_fields=[
            "doc_name",
            "page_start",
            "page_end",
            "source_page_count",
            "marker_count",
            "source_pages_json",
            "markers_json",
            "text",
            "metadata_json",
        ],
    )
    dense_rows: list[dict[str, Any]] = []
    for hit in results[0]:
        entity = hit.entity
        chunk_id = str(hit.id)
        local_chunk = chunk_lookup.get(chunk_id, {})
        text = str(entity.get("text") or local_chunk.get("text") or "")
        markers = json.loads(str(entity.get("markers_json") or "[]"))
        dense_rows.append(
            {
                "chunk_id": chunk_id,
                "dense_score": float(hit.score),
                "doc_name": str(entity.get("doc_name") or ""),
                "page_start": int(entity.get("page_start") or 0),
                "page_end": int(entity.get("page_end") or 0),
                "source_pages": json.loads(str(entity.get("source_pages_json") or "[]")),
                "markers": markers,
                "text": text,
            }
        )

    dense_score_map = {str(item["chunk_id"]): float(item.get("dense_score", 0.0)) for item in dense_rows}
    dense_norm = normalize_score_map(dense_score_map)

    corpus = list(chunk_lookup.values())
    corpus_ids = [str(item.get("chunk_id") or "") for item in corpus]
    corpus_texts = [str(item.get("text") or "") for item in corpus]
    bm25_index = SimpleBM25Index([tokenize_for_bm25(text) for text in corpus_texts], settings.bm25_k1, settings.bm25_b)
    bm25_scores = bm25_index.score(tokenize_for_bm25(query))
    bm25_score_map = {
        chunk_id: float(score)
        for chunk_id, score in zip(corpus_ids, bm25_scores)
        if chunk_id and float(score) > 0.0
    }
    bm25_norm = normalize_score_map(bm25_score_map)

    merged: dict[str, dict[str, Any]] = {str(item["chunk_id"]): dict(item) for item in dense_rows}
    for item in corpus:
        chunk_id = str(item.get("chunk_id") or "")
        if chunk_id not in bm25_score_map:
            continue
        if chunk_id not in merged:
            merged[chunk_id] = {
                "chunk_id": chunk_id,
                "dense_score": 0.0,
                "doc_name": "",
                "page_start": int(min(list(item.get("source_pages") or [0]))),
                "page_end": int(max(list(item.get("source_pages") or [0]))),
                "source_pages": list(item.get("source_pages") or []),
                "markers": list(item.get("markers") or []),
                "text": str(item.get("text") or ""),
            }

    hybrid_rows: list[dict[str, Any]] = []
    for chunk_id, item in merged.items():
        dense_raw = dense_score_map.get(chunk_id, 0.0)
        bm25_raw = bm25_score_map.get(chunk_id, 0.0)
        text = str(item.get("text") or "")
        doc_name = str(item.get("doc_name") or "")
        marker_ids = get_marker_ids_from_text(list(item.get("markers") or []), text)
        overlap = keyword_overlap_score(query, text)
        marker_boost, marker_details = compute_marker_boost(query, marker_ids, table_lookup, visual_lookup)
        answer_boost = compute_answer_boost(query, text)
        focus_signal, focus_details = compute_focus_signal(query, text)
        company_signal, company_details = compute_company_signal(query, text, doc_name=doc_name)
        page_position_penalty, page_position_details = compute_page_position_penalty(
            list(item.get("source_pages") or []),
            total_pages,
            is_visual=False,
        )
        hybrid_raw = (
            dense_norm.get(chunk_id, 0.0) * settings.hybrid_dense_weight
            + bm25_norm.get(chunk_id, 0.0) * settings.hybrid_lexical_weight
            + overlap * settings.hybrid_overlap_weight
            + marker_boost
            + answer_boost
            + focus_signal
            + company_signal
            + page_position_penalty
        )
        enriched = dict(item)
        enriched["dense_score"] = dense_raw
        enriched["bm25_score"] = bm25_raw
        enriched["overlap_score"] = overlap
        enriched["marker_ids"] = marker_ids
        enriched["marker_boost"] = marker_boost
        enriched["marker_details"] = marker_details
        enriched["answer_boost"] = answer_boost
        enriched["focus_signal"] = focus_signal
        enriched["focus_details"] = focus_details
        enriched["company_signal"] = company_signal
        enriched["company_details"] = company_details
        enriched["page_position_penalty"] = page_position_penalty
        enriched["page_position_details"] = page_position_details
        enriched["hybrid_score"] = hybrid_raw
        enriched["score"] = hybrid_raw
        hybrid_rows.append(enriched)

    hybrid_rows.sort(key=lambda item: float(item.get("hybrid_score", 0.0)), reverse=True)
    candidate_rows = expand_neighbor_candidates(
        hybrid_rows,
        ordered_chunks,
        position_by_id,
        chunk_lookup,
        seed_limit=text_top_k,
    )

    window_candidates: list[dict[str, Any]] = []
    for row in candidate_rows:
        windows = build_text_windows(row, ordered_chunks, position_by_id)
        for window in windows:
            window_text = str(window.get("window_text") or "")
            window_marker_ids = list(window.get("marker_ids") or [])
            window_marker_boost, window_marker_details = compute_marker_boost(
                query,
                window_marker_ids,
                table_lookup,
                visual_lookup,
            )
            window_answer_boost = compute_answer_boost(query, window_text)
            window_overlap = keyword_overlap_score(query, window_text)
            window["marker_boost"] = max(float(window.get("marker_boost") or 0.0), window_marker_boost)
            window["marker_details"] = window_marker_details
            window["answer_boost"] = max(float(window.get("answer_boost") or 0.0), window_answer_boost)
            window["overlap_score"] = max(float(window.get("overlap_score") or 0.0), window_overlap)
            window_focus_signal, window_focus_details = compute_focus_signal(query, window_text)
            window_company_signal, window_company_details = compute_company_signal(
                query,
                window_text,
                doc_name=str(window.get("doc_name") or ""),
            )
            window["focus_signal"] = max(float(window.get("focus_signal") or 0.0), window_focus_signal)
            window["focus_details"] = window_focus_details
            window["company_signal"] = max(float(window.get("company_signal") or 0.0), window_company_signal)
            window["company_details"] = window_company_details
            window_page_penalty, window_page_details = compute_page_position_penalty(
                list(window.get("window_source_pages") or window.get("source_pages") or []),
                total_pages,
                is_visual=False,
            )
            window["page_position_penalty"] = window_page_penalty
            window["page_position_details"] = window_page_details
            window["window_score"] = (
                float(window.get("hybrid_score") or 0.0)
                + window_marker_boost
                + window_answer_boost
                + window_overlap * 0.08
                + window_focus_signal
                + window_company_signal
                + window_page_penalty
            )
            window_candidates.append(window)

    if not window_candidates:
        final_rows = hybrid_rows[: max(1, text_top_k)]
        for item in final_rows:
            item.update(build_neighbor_window_payload(item, ordered_chunks, position_by_id, radius=1))
        return final_rows

    if reranker is not None and reranker.is_enabled() and not disable_rerank and window_candidates:
        rerank_candidates = [{"text": str(item.get("window_text") or ""), **item} for item in window_candidates]
        reranked = reranker.rerank(query, rerank_candidates, len(rerank_candidates))
        best_by_chunk: dict[str, dict[str, Any]] = {}
        for item in reranked:
            chunk_id = str(item.get("chunk_id") or "")
            existing = best_by_chunk.get(chunk_id)
            if existing is None or float(item.get("rerank_score", 0.0)) > float(existing.get("rerank_score", 0.0)):
                enriched = dict(item)
                enriched["score"] = float(item.get("rerank_score", 0.0))
                enriched["preview_text"] = str(item.get("window_text") or item.get("text") or "")
                best_by_chunk[chunk_id] = enriched
        final_rows = list(best_by_chunk.values())
        final_rows.sort(
            key=lambda item: (
                float(item.get("rerank_score", 0.0)),
                float(item.get("window_score", 0.0)),
                float(item.get("hybrid_score", 0.0)),
            ),
            reverse=True,
        )
        final_rows = final_rows[: max(1, text_top_k)]
        for item in final_rows:
            item.update(build_neighbor_window_payload(item, ordered_chunks, position_by_id, radius=1))
        return final_rows

    best_by_chunk: dict[str, dict[str, Any]] = {}
    for item in window_candidates:
        chunk_id = str(item.get("chunk_id") or "")
        existing = best_by_chunk.get(chunk_id)
        if existing is None or float(item.get("window_score", 0.0)) > float(existing.get("window_score", 0.0)):
            enriched = dict(item)
            enriched["score"] = float(item.get("window_score", 0.0))
            enriched["preview_text"] = str(item.get("window_text") or item.get("text") or "")
            best_by_chunk[chunk_id] = enriched
    final_rows = list(best_by_chunk.values())
    final_rows.sort(
        key=lambda item: (
            float(item.get("window_score", 0.0)),
            float(item.get("hybrid_score", 0.0)),
        ),
        reverse=True,
    )
    final_rows = final_rows[: max(1, text_top_k)]
    for item in final_rows:
        item.update(build_neighbor_window_payload(item, ordered_chunks, position_by_id, radius=1))
    return final_rows


def search_visual_vectors(query: str, *, visual_top_k: int = 3) -> list[dict[str, Any]]:
    from pymilvus import Collection, connections

    connections.connect(alias="visual_check", uri=settings.milvus_uri)
    collection = Collection(name=settings.visual_vector_collection_name, using="visual_check")
    collection.load()

    embedder = EmbeddingService(settings.model_dir / "embedding", configured_path=settings.embedding_model_path)
    query_vector = embedder.embed_query(query)
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {}},
        limit=max(1, visual_top_k),
        output_fields=[
            "doc_name",
            "page_number",
            "visual_type",
            "marker_id",
            "summary_text",
            "search_text",
            "minio_path",
            "metadata_json",
        ],
    )
    rows: list[dict[str, Any]] = []
    for hit in results[0]:
        entity = hit.entity
        rows.append(
            {
                "visual_id": str(hit.id),
                "score": float(hit.score),
                "doc_name": str(entity.get("doc_name") or ""),
                "page_number": int(entity.get("page_number") or 0),
                "visual_type": str(entity.get("visual_type") or ""),
                "marker_id": str(entity.get("marker_id") or ""),
                "summary_text": str(entity.get("summary_text") or ""),
                "search_text": str(entity.get("search_text") or ""),
                "minio_path": str(entity.get("minio_path") or ""),
            }
        )
    return rows


def rerank_visual_rows_by_page_position(rows: list[dict[str, Any]], total_pages: int) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for row in rows:
        penalty, details = compute_page_position_penalty(
            [int(row.get("page_number") or 0)],
            total_pages,
            is_visual=True,
        )
        enriched = dict(row)
        enriched["page_position_penalty"] = penalty
        enriched["page_position_details"] = details
        enriched["score"] = float(row.get("score") or 0.0) + penalty
        adjusted.append(enriched)
    adjusted.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return adjusted


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
    lookup: dict[str, dict[str, Any]] = {}
    for item in load_table_documents(artifact_dir):
        aliases: set[str] = set()
        aliases.add(str(item.get("marker_id") or item.get("final_object_id") or item.get("table_id") or "").strip())
        aliases.add(str(item.get("table_id") or "").strip())
        aliases.add(str(item.get("final_object_id") or "").strip())
        for region_id in item.get("source_region_ids") or []:
            aliases.add(str(region_id or "").strip())
        local_result = item.get("local_table_result") or {}
        for region_id in local_result.get("source_region_ids") or []:
            aliases.add(str(region_id or "").strip())
        for segment in local_result.get("segments") or []:
            aliases.add(str(segment.get("table_id") or "").strip())
            aliases.add(str(segment.get("source_region_id") or "").strip())
        for alias in aliases:
            if alias:
                lookup[alias] = item
    return lookup


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
        hydrated = hydrate_visual_row_from_vlm(item, vlm_map.get(visual_id))
        aliases: set[str] = {visual_id, str(item.get("visual_id") or "").strip()}
        try:
            aliases.update(str(region_id or "").strip() for region_id in json.loads(str(item.get("source_region_ids_json") or "[]")))
        except Exception:
            pass
        for alias in aliases:
            if alias:
                lookup[alias] = hydrated
    for visual_id, vlm_item in vlm_map.items():
        hydrated = hydrate_visual_row_from_vlm(
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
        aliases: set[str] = {str(visual_id or "").strip()}
        for region_id in vlm_item.get("source_region_ids") or []:
            aliases.add(str(region_id or "").strip())
        for alias in aliases:
            if alias and alias not in lookup:
                lookup[alias] = hydrated
    return lookup


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


def print_text_results(rows: list[dict[str, Any]]) -> None:
    print("\n=== Text Hits ===")
    if not rows:
        print("No text hits.")
        return
    for index, row in enumerate(rows, start=1):
        print(
            f"[{index}] score={row['score']:.4f} hybrid={float(row.get('hybrid_score', 0.0)):.4f} "
            f"dense={float(row.get('dense_score', 0.0)):.4f} bm25={float(row.get('bm25_score', 0.0)):.4f} "
            f"overlap={float(row.get('overlap_score', 0.0)):.4f} "
            f"marker_boost={float(row.get('marker_boost', 0.0)):.4f} "
            f"answer_boost={float(row.get('answer_boost', 0.0)):.4f} "
            f"focus={float(row.get('focus_signal', 0.0)):.4f} "
            f"company={float(row.get('company_signal', 0.0)):.4f} "
            f"page_bias={float(row.get('page_position_penalty', 0.0)):.4f} "
            f"rerank={float(row.get('rerank_score', 0.0)):.4f} "
            f"pages={row['page_start']}-{row['page_end']} chunk={row['chunk_id']}"
        )
        print(
            f"    markers={len(row.get('markers') or [])} source_pages={row.get('source_pages')} "
            f"window_pages={row.get('window_source_pages') or row.get('source_pages')} "
            f"window_chunks={row.get('window_chunk_ids') or [row.get('chunk_id')]}"
        )
        focus_hits = ((row.get("focus_details") or {}).get("hits") or [])
        print(f"    focus_hits={focus_hits}")
        print(f"    preview={preview(str(row.get('preview_text') or row.get('window_text') or row.get('text') or ''))}")


def print_visual_results(rows: list[dict[str, Any]]) -> None:
    print("\n=== Visual Hits ===")
    if not rows:
        print("No visual hits.")
        return
    for index, row in enumerate(rows, start=1):
        print(
            f"[{index}] score={row['score']:.4f} page_bias={float(row.get('page_position_penalty', 0.0)):.4f} "
            f"page={row['page_number']} visual_type={row['visual_type']} marker={row['marker_id']}"
        )
        print(f"    summary={preview(str(row.get('summary_text') or ''))}")
        print(f"    minio={row.get('minio_path') or ''}")


def print_linked_objects(marker_ids: list[str], table_lookup: dict[str, dict[str, Any]], visual_lookup: dict[str, dict[str, Any]]) -> None:
    print("\n=== Linked Objects From Text Markers ===")
    if not marker_ids:
        print("No linked markers in top text chunks.")
        return
    for marker_id in marker_ids:
        if marker_id in table_lookup:
            item = table_lookup[marker_id]
            print(
                f"[TABLE] marker={marker_id} pages={item.get('page_start')}-{item.get('page_end')} "
                f"type={item.get('table_type')}"
            )
            vlm_headers = (((item.get("vlm_result") or {}).get("structured_content") or {}).get("headers") or [])
            local_segments = ((item.get("local_table_result") or {}).get("segments") or [])
            local_headers = (local_segments[0].get("headers") if local_segments else []) or []
            print(f"    vlm_headers={vlm_headers[:8]}")
            print(f"    local_headers={local_headers[:8]}")
            continue
        if marker_id in visual_lookup:
            item = visual_lookup[marker_id]
            print(
                f"[VISUAL] marker={marker_id} page={item.get('page_number')} type={item.get('visual_type')} "
                f"minio={item.get('minio_path') or ''}"
            )
            print(f"    summary={preview(str(item.get('summary_text') or ''))}")
            continue
        print(f"[UNKNOWN] marker={marker_id}")


def print_table_keyword_results(rows: list[dict[str, Any]]) -> None:
    print("\n=== Table Keyword Fallback Hits ===")
    if not rows:
        print("No table keyword hits.")
        return
    for index, row in enumerate(rows, start=1):
        print(f"[{index}] score={row['score']:.2f} pages={row['page_start']}-{row['page_end']} table={row['table_id']}")
        print(f"    vlm_headers={row.get('vlm_headers') or []}")
        print(f"    local_headers={row.get('local_headers') or []}")
        print(f"    vlm_rows_preview={row.get('vlm_rows_preview') or []}")
        print(f"    local_rows_preview={row.get('local_rows_preview') or []}")


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


def extract_target_company(query: str) -> str:
    patterns = [
        r"([\u4e00-\u9fffA-Za-z0-9()ï¼ï¼]{4,}?(?:è¡ä»½æéå¬å¸|æéè´£ä»»å¬å¸|éå¢è¡ä»½æéå¬å¸|éå¢æéå¬å¸))",
    ]
    sanitized_query = strip_company_query_prefixes(query)
    for pattern in patterns:
        match = re.search(pattern, sanitized_query)
        if match:
            return match.group(1).strip()
    return ""


def stringify_table_row(row: list[Any]) -> str:
    return " | ".join(str(cell).strip() for cell in row if str(cell).strip())


def score_table_row_for_query(query: str, row_text: str) -> float:
    if not row_text:
        return 0.0
    score = keyword_score(query, row_text)
    score += keyword_overlap_score(query, row_text) * 10.0
    focus_terms = extract_focus_terms(query)
    score += sum(2.5 for term in focus_terms if term in row_text)
    normalized_query = normalize_text(query)
    if any(token in normalized_query for token in ["??", "??", "???", "??", "??"]):
        if any(token in row_text for token in ["????", "??????", "??", "??"]):
            score += 2.0
    return score


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
    focus_signal, focus_details = compute_focus_signal(query, table_text)
    score += max(0.0, focus_signal) * 20.0
    score += len(list(focus_details.get("hits") or [])) * 2.0
    if marker_context:
        score += keyword_score(query, marker_context) * 1.6
        score += keyword_overlap_score(query, marker_context) * 8.0
    score += max(0.0, float(source_text_score)) * 1.2
    score += max(0.0, 3.5 - (float(source_text_rank) - 1.0) * 0.75)
    score += max(0.0, 2.0 - (float(marker_rank) - 1.0) * 0.35)
    return score


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


def build_llm_contexts(
    query: str,
    text_rows: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
    table_lookup: dict[str, dict[str, Any]],
    visual_lookup: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    fallback_contexts: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, str]] = set()
    question_type = infer_question_type(query)
    query_tags = set(infer_query_tags(query))
    table_row_by_id: dict[str, dict[str, Any]] = {
        str(row.get("table_id") or ""): row
        for row in table_rows
        if str(row.get("table_id") or "")
    }
    visual_row_by_id: dict[str, dict[str, Any]] = {}
    for row in visual_rows:
        visual_id = str(row.get("marker_id") or row.get("visual_id") or "").strip()
        if not visual_id:
            continue
        visual_row_by_id[visual_id] = hydrate_visual_row_from_vlm(
            row,
            (visual_lookup.get(visual_id) or {}).get("vlm_result") if visual_lookup.get(visual_id) else None,
        )

    def ensure_visual_row(visual_id: str) -> dict[str, Any] | None:
        if not visual_id:
            return None
        existing = visual_row_by_id.get(visual_id)
        if existing is not None:
            return existing
        source = visual_lookup.get(visual_id)
        if not source:
            return None
        row = hydrate_visual_row_from_vlm({
            "visual_id": str(source.get("visual_id") or visual_id),
            "score": float(source.get("score") or 0.0),
            "doc_name": str(source.get("doc_name") or ""),
            "page_number": int(source.get("page_number") or 0),
            "visual_type": str(source.get("visual_type") or ""),
            "marker_id": str(source.get("marker_id") or visual_id),
            "summary_text": str(source.get("summary_text") or ""),
            "search_text": str(source.get("search_text") or ""),
            "minio_path": str(source.get("minio_path") or ""),
        }, source.get("vlm_result") if isinstance(source, dict) else None)
        visual_row_by_id[visual_id] = row
        return row

    def build_first_hit_expanded_payload() -> tuple[dict[str, Any] | None, set[str], set[str]]:
        if not text_rows:
            return None, set(), set()
        first_row = text_rows[0]
        source_text = str(
            first_row.get("neighbor_window_text")
            or first_row.get("window_text")
            or first_row.get("preview_text")
            or first_row.get("text")
            or ""
        )
        if not source_text.strip():
            return None, set(), set()

        used_table_ids: set[str] = set()
        used_visual_ids: set[str] = set()
        parts: list[str] = []
        cursor = 0
        matches = list(MARKER_RE.finditer(source_text))
        if not matches:
            payload = {
                "page_number": int(first_row.get("page_start") or 0),
                "logical_page": "",
                "text": normalize_text(source_text),
                "metadata": {
                    "page_type": "text_expanded",
                    "chunk_id": str(first_row.get("chunk_id") or ""),
                    "source_pdf": str(first_row.get("doc_name") or ""),
                    "source_pages": list(first_row.get("neighbor_window_source_pages") or first_row.get("window_source_pages") or first_row.get("source_pages") or []),
                    "never_truncate": True,
                },
            }
            return payload, used_table_ids, used_visual_ids

        for match in matches:
            if match.start() > cursor:
                parts.append(source_text[cursor:match.start()])
            marker_id = str(match.group(2) or "")
            replacement = match.group(0)
            if marker_id in table_lookup:
                table_text = build_table_context_text_v2(marker_id, table_lookup, query=query, include_all_rows=True)
                if table_text:
                    replacement = f"\nè¡¨æ ¼åå®¹ï¼\n{table_text}\n"
                    used_table_ids.add(marker_id)
            else:
                visual_row = ensure_visual_row(marker_id)
                if visual_row is not None:
                    visual_text = build_visual_context_text(visual_row)
                    if visual_text:
                        replacement = f"\nå¾è¡¨åå®¹ï¼\n{visual_text}\n"
                        used_visual_ids.add(marker_id)
            parts.append(replacement)
            cursor = match.end()
        if cursor < len(source_text):
            parts.append(source_text[cursor:])

        payload = {
            "page_number": int(first_row.get("page_start") or 0),
            "logical_page": "",
            "text": normalize_text("".join(parts)),
            "metadata": {
                "page_type": "text_expanded",
                "chunk_id": str(first_row.get("chunk_id") or ""),
                "source_pdf": str(first_row.get("doc_name") or ""),
                "source_pages": list(first_row.get("neighbor_window_source_pages") or first_row.get("window_source_pages") or first_row.get("source_pages") or []),
                "never_truncate": True,
            },
        }
        payload["text"] = (
            str(payload.get("text") or "")
            .replace("猫隆篓忙聽录氓聠聟氓庐鹿茂录職", "TABLE CONTENT:")
            .replace("氓聸戮猫隆篓氓聠聟氓庐鹿茂录職", "VISUAL CONTENT:")
            .replace("è¡¨æ ¼å å®¹ï¼", "TABLE CONTENT:")
            .replace("å¾è¡¨å å®¹ï¼", "VISUAL CONTENT:")
        )
        return payload, used_table_ids, used_visual_ids
    prioritized_table_ids: list[str] = []
    prioritized_seen: set[str] = set()
    prioritized_visual_ids: list[str] = []
    prioritized_visual_seen: set[str] = set()
    first_hit_marker_ids: list[str] = []
    first_hit_chunk_id = str(text_rows[0].get("chunk_id") or "") if text_rows else ""
    for row in text_rows[: max(1, limit * 2)]:
        marker_ids = list(row.get("marker_ids") or [])
        if not marker_ids:
            marker_ids = get_marker_ids_from_text(
                list(row.get("markers") or []),
                str(row.get("window_text") or row.get("preview_text") or row.get("text") or ""),
            )
        if not first_hit_marker_ids:
            neighbor_marker_ids = list(row.get("neighbor_marker_ids") or [])
            first_hit_marker_ids = [
                marker_id
                for marker_id in (neighbor_marker_ids or marker_ids)
                if marker_id
            ]
        for marker_id in marker_ids:
            if marker_id in table_lookup and marker_id not in prioritized_seen:
                prioritized_seen.add(marker_id)
                prioritized_table_ids.append(marker_id)
            if ensure_visual_row(marker_id) is not None and marker_id not in prioritized_visual_seen:
                prioritized_visual_seen.add(marker_id)
                prioritized_visual_ids.append(marker_id)
    first_hit_table_ids = [marker_id for marker_id in first_hit_marker_ids if marker_id in table_lookup]
    first_hit_visual_ids = [marker_id for marker_id in first_hit_marker_ids if ensure_visual_row(marker_id) is not None]
    org_structure_prioritized_visual_ids = list(prioritized_visual_ids) if question_type == "org_structure" else []
    first_hit_org_chart_visual_ids = [
        marker_id
        for marker_id in first_hit_visual_ids
        if str((ensure_visual_row(marker_id) or {}).get("visual_type") or "").strip() == "org_chart"
    ]
    context_limit = max(1, limit, 1 if text_rows else 0)

    def push_context(payload: dict[str, Any], *, fallback_only: bool = False) -> None:
        page_number = int(payload.get("page_number") or 0)
        metadata = dict(payload.get("metadata") or {})
        unique_id = str(
            metadata.get("chunk_id")
            or metadata.get("table_id")
            or metadata.get("visual_id")
            or metadata.get("source_pdf")
            or f"page_{page_number}"
        )
        page_type = str(metadata.get("page_type") or "")
        key = (page_type, page_number, unique_id)
        if key in seen_keys:
            return
        seen_keys.add(key)
        if fallback_only:
            fallback_contexts.append(payload)
        else:
            contexts.append(payload)

    def append_table_contexts(
        max_count: int,
        preferred_ids: list[str] | None = None,
        force_ids: list[str] | None = None,
    ) -> None:
        if max_count <= 0:
            return
        added = 0
        preferred_id_set = {table_id for table_id in (preferred_ids or []) if table_id}
        force_id_set = {table_id for table_id in (force_ids or []) if table_id}
        ordered_rows: list[dict[str, Any]] = []
        local_seen: set[str] = set()
        for table_id in preferred_ids or []:
            row = table_row_by_id.get(table_id)
            if not row:
                linked_item = table_lookup.get(table_id) or {}
                row = {
                    "table_id": table_id,
                    "page_start": int(linked_item.get("page_start") or 0),
                    "page_end": int(linked_item.get("page_end") or 0),
                }
            table_key = str(row.get("table_id") or "")
            if table_key and table_key not in local_seen:
                local_seen.add(table_key)
                ordered_rows.append(row)
        for row in table_rows:
            table_key = str(row.get("table_id") or "")
            if table_key and table_key not in local_seen:
                local_seen.add(table_key)
                ordered_rows.append(row)

        for row in ordered_rows:
            table_id = str(row.get("table_id") or "")
            if table_id in first_hit_consumed_table_ids:
                continue
            page_number = int(row.get("page_start") or 0)
            text = build_table_context_text_v2(table_id, table_lookup, query=query, include_all_rows=True)
            if not text:
                continue
            if table_id not in force_id_set and keyword_score(query, text) <= 0:
                continue
            payload = {
                "page_number": page_number,
                "logical_page": "",
                "text": text,
                "metadata": {
                    "page_type": "table",
                    "table_id": table_id,
                    "page_end": int(row.get("page_end") or page_number),
                },
            }
            if table_id in force_id_set:
                push_context(payload)
            elif table_id in preferred_id_set:
                push_context(payload)
            elif should_keep_context(query, text):
                push_context(payload)
            else:
                push_context(payload, fallback_only=True)
            if not any(
                str(existing.get("metadata", {}).get("table_id") or "") == table_id
                for existing in contexts
            ):
                continue
            added += 1
            if len(contexts) >= context_limit or added >= max_count:
                break

    def append_text_contexts(max_count: int) -> None:
        if max_count <= 0:
            return
        added = 0
        for row in text_rows[: max(1, limit * 2)]:
            page_number = int(row.get("page_start") or 0)
            chunk_id = str(row.get("chunk_id") or "")
            if first_hit_chunk_id and chunk_id == first_hit_chunk_id:
                continue
            text = build_complete_context_text(row)
            if not text:
                continue
            payload = {
                "page_number": page_number,
                "logical_page": "",
                "text": text,
                "metadata": {
                    "page_type": "text_chunk",
                    "source_pdf": str(row.get("doc_name") or ""),
                    "chunk_id": chunk_id,
                    "source_pages": list(row.get("window_source_pages") or row.get("source_pages") or []),
                },
            }
            if should_keep_context(query, text, doc_name=str(row.get("doc_name") or "")):
                push_context(payload)
            else:
                push_context(payload, fallback_only=True)
            if not any(
                str(existing.get("metadata", {}).get("chunk_id") or "") == chunk_id
                for existing in contexts
            ):
                continue
            added += 1
            if len(contexts) >= context_limit or added >= max_count:
                break

    def append_visual_contexts(
        max_count: int,
        preferred_ids: list[str] | None = None,
        force_ids: list[str] | None = None,
    ) -> None:
        if max_count <= 0:
            return
        added = 0
        preferred_id_set = {visual_id for visual_id in (preferred_ids or []) if visual_id}
        force_id_set = {visual_id for visual_id in (force_ids or []) if visual_id}
        ordered_rows: list[dict[str, Any]] = []
        local_seen: set[str] = set()
        for visual_id in preferred_ids or []:
            row = visual_row_by_id.get(visual_id)
            if not row:
                continue
            visual_key = str(row.get("marker_id") or row.get("visual_id") or "")
            if visual_key and visual_key not in local_seen:
                local_seen.add(visual_key)
                ordered_rows.append(row)
        for row in visual_rows:
            visual_key = str(row.get("marker_id") or row.get("visual_id") or "")
            if visual_key and visual_key not in local_seen:
                local_seen.add(visual_key)
                ordered_rows.append(row)

        for row in ordered_rows:
            visual_id = str(row.get("marker_id") or row.get("visual_id") or "")
            if visual_id in first_hit_consumed_visual_ids and visual_id not in force_id_set:
                continue
            page_number = int(row.get("page_number") or 0)
            text = build_visual_context_text(row)
            if not text:
                continue
            if visual_id not in force_id_set and keyword_score(query, text) <= 0:
                continue
            payload = {
                "page_number": page_number,
                "logical_page": "",
                "text": text,
                "metadata": {
                    "page_type": "visual",
                    "visual_id": visual_id,
                    "visual_type": str(row.get("visual_type") or ""),
                    "minio_path": str(row.get("minio_path") or ""),
                },
            }
            if visual_id in force_id_set:
                push_context(payload)
            elif visual_id in preferred_id_set:
                push_context(payload)
            elif should_keep_context(query, text, doc_name=str(row.get("doc_name") or "")):
                push_context(payload)
            else:
                push_context(payload, fallback_only=True)
            if not any(
                str(existing.get("metadata", {}).get("visual_id") or "") == visual_id
                for existing in contexts
            ):
                continue
            added += 1
            if len(contexts) >= context_limit or added >= max_count:
                break

    reserve_table_slots = 0
    reserve_visual_slots = 0
    if table_rows and question_type in {"table_list", "table_numeric"}:
        reserve_table_slots = min(max(2, limit // 2), limit)
    elif table_rows and "related_party" in query_tags:
        reserve_table_slots = min(2, limit)
    if visual_rows and question_type == "chart_trend":
        reserve_visual_slots = min(max(2, limit // 2), limit)
    elif org_structure_prioritized_visual_ids:
        reserve_visual_slots = max(
            reserve_visual_slots,
            min(len(org_structure_prioritized_visual_ids), limit),
        )
    elif first_hit_org_chart_visual_ids and question_type == "org_structure":
        reserve_visual_slots = max(reserve_visual_slots, min(len(first_hit_org_chart_visual_ids), limit))

    first_hit_payload, first_hit_consumed_table_ids, first_hit_consumed_visual_ids = build_first_hit_expanded_payload()
    if first_hit_payload is not None:
        push_context(first_hit_payload)

    if org_structure_prioritized_visual_ids:
        append_visual_contexts(
            max_count=max(1, len(org_structure_prioritized_visual_ids)),
            preferred_ids=org_structure_prioritized_visual_ids,
            force_ids=org_structure_prioritized_visual_ids,
        )

    if first_hit_org_chart_visual_ids:
        append_visual_contexts(
            max_count=max(1, len(first_hit_org_chart_visual_ids)),
            preferred_ids=first_hit_org_chart_visual_ids,
            force_ids=first_hit_org_chart_visual_ids,
        )

    if reserve_visual_slots > 0:
        append_visual_contexts(reserve_visual_slots, prioritized_visual_ids)
        append_text_contexts(max(0, context_limit - len(contexts)))
        if len(contexts) < context_limit:
            append_table_contexts(max(0, context_limit - len(contexts)), prioritized_table_ids)
        if len(contexts) < context_limit:
            append_visual_contexts(max(0, context_limit - len(contexts)), prioritized_visual_ids)
    else:
        append_table_contexts(reserve_table_slots, prioritized_table_ids)
        append_text_contexts(max(0, context_limit - len(contexts)))
        if len(contexts) < context_limit:
            append_table_contexts(max(0, context_limit - len(contexts)), prioritized_table_ids)
        if len(contexts) < context_limit:
            append_visual_contexts(max(0, context_limit - len(contexts)), prioritized_visual_ids)

    if not contexts and fallback_contexts:
        contexts.extend(fallback_contexts[: max(1, context_limit)])
    elif len(contexts) < context_limit and fallback_contexts:
        contexts.extend(fallback_contexts[: max(0, context_limit - len(contexts))])

    return contexts[: max(1, context_limit)]


def build_llm_diagnostic_contexts(query: str, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    focus_terms = extract_focus_terms(query)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(contexts, start=1):
        text = str(item.get("text") or "")
        metadata = dict(item.get("metadata") or {})
        rows.append(
            {
                "rank": index,
                "page_number": int(item.get("page_number") or 0),
                "page_type": str(metadata.get("page_type") or ""),
                "chunk_id": str(metadata.get("chunk_id") or ""),
                "table_id": str(metadata.get("table_id") or ""),
                "visual_id": str(metadata.get("visual_id") or ""),
                "source_pages": list(metadata.get("source_pages") or []),
                "focus_hits": [term for term in focus_terms if term in text],
                "preview": preview(text, limit=260),
            }
        )
    return rows


def generate_llm_answer(
    query: str,
    text_rows: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
    table_lookup: dict[str, dict[str, Any]],
    visual_lookup: dict[str, dict[str, Any]],
    limit: int,
) -> tuple[str | None, str | None]:
    try:
        from backend.services.llm_client import LLMClient
    except Exception as exc:
        return None, f"LLMClient import failed: {exc}"

    contexts = build_llm_contexts(
        query=query,
        text_rows=text_rows,
        table_rows=table_rows,
        visual_rows=visual_rows,
        table_lookup=table_lookup,
        visual_lookup=visual_lookup,
        limit=limit,
    )
    if not contexts:
        return None, "No contexts available for LLM."

    aggregate_text = "\n".join(str(item.get("text") or "") for item in contexts)
    focus_signal, focus_details = compute_focus_signal(query, aggregate_text)
    intent = SimpleNamespace(
        target_company=extract_target_company(query),
        question_type=infer_question_type(query),
        query_tags=infer_query_tags(query),
    )
    client = LLMClient(
        provider=settings.llm_provider,
        api_url=settings.llm_api_url,
        api_key=settings.llm_api_key,
        model_name=settings.llm_model_name,
        fallback_api_url=settings.llm_fallback_api_url,
        fallback_api_key=settings.llm_fallback_api_key,
        fallback_model_name=settings.llm_fallback_model_name,
    )
    try:
        answer = client.answer(query, contexts, intent=intent)
    except Exception as exc:
        return None, f"LLM request failed: {exc}"
    cleaned = normalize_text(answer)
    return cleaned or None, None


def print_llm_answer(answer: str | None, error: str | None) -> None:
    print("\n=== LLM Answer ===")
    if error:
        print(f"Unavailable: {error}")
        return
    if not answer:
        print("No LLM answer.")
        return
    print(answer)


def generate_llm_answer_with_diagnostics(
    query: str,
    text_rows: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
    table_lookup: dict[str, dict[str, Any]],
    visual_lookup: dict[str, dict[str, Any]],
    limit: int,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    diagnostics: dict[str, Any] = {
        "query": query,
        "target_company": extract_target_company(query),
        "question_type": infer_question_type(query),
        "query_tags": infer_query_tags(query),
    }
    try:
        from backend.services.llm_client import LLMClient
    except Exception as exc:
        diagnostics["answer_source"] = "error"
        diagnostics["error"] = f"LLMClient import failed: {exc}"
        return None, f"LLMClient import failed: {exc}", diagnostics

    contexts = build_llm_contexts(
        query=query,
        text_rows=text_rows,
        table_rows=table_rows,
        visual_rows=visual_rows,
        table_lookup=table_lookup,
        visual_lookup=visual_lookup,
        limit=limit,
    )
    diagnostics["context_count"] = len(contexts)
    diagnostics["contexts"] = build_llm_diagnostic_contexts(query, contexts)
    if not contexts:
        diagnostics["answer_source"] = "no_context"
        return None, "No contexts available for LLM.", diagnostics

    aggregate_text = "\n".join(str(item.get("text") or "") for item in contexts)
    focus_signal, focus_details = compute_focus_signal(query, aggregate_text)
    diagnostics["aggregate_focus_signal"] = focus_signal
    diagnostics["aggregate_focus_hits"] = list(focus_details.get("hits") or [])
    diagnostics["precheck_bypassed"] = True

    intent = SimpleNamespace(
        target_company=diagnostics["target_company"],
        question_type=diagnostics["question_type"],
        query_tags=diagnostics["query_tags"],
    )
    client = LLMClient(
        provider=settings.llm_provider,
        api_url=settings.llm_api_url,
        api_key=settings.llm_api_key,
        model_name=settings.llm_model_name,
        fallback_api_url=settings.llm_fallback_api_url,
        fallback_api_key=settings.llm_fallback_api_key,
        fallback_model_name=settings.llm_fallback_model_name,
    )
    diagnostics["provider"] = str(client.provider or "")
    diagnostics["model_name"] = str(client.model_name or client.fallback_model_name or "")

    try:
        retry_plan = build_llm_retry_plan(contexts)
        diagnostics["llm_attempts"] = []
        final_error = "LLM returned no response."
        for attempt_index, plan in enumerate(retry_plan, start=1):
            attempt_contexts = contexts[: int(plan["context_limit"])]
            prompt = build_test_llm_prompt(
                query,
                attempt_contexts,
                per_context_limit=int(plan["per_context_limit"]),
            )
            content = client._call_primary_then_fallback(
                prompt=prompt,
                system_prompt="ä½ æ¯ä¸ä¸ªä¸¥æ ¼åºäºè¯æ®åç­é®é¢ç RAG å©æã",
                max_tokens=int(plan["max_tokens"]),
            )
            cleaned = normalize_text(_clean_answer_text(content or "")) if content is not None else ""
            call_details = dict(getattr(client, "last_call_details", {}) or {})
            attempt_record = {
                "attempt_index": attempt_index,
                "context_count": len(attempt_contexts),
                "per_context_limit": int(plan["per_context_limit"]),
                "max_tokens": int(plan["max_tokens"]),
                "prompt_chars": len(prompt),
                "call_status": str(call_details.get("status") or ""),
                "selected_route": str(call_details.get("selected_route") or ""),
                "raw_output_preview": preview(str(content or ""), limit=180) if content is not None else "",
                "cleaned_output_preview": preview(cleaned, limit=180) if cleaned else "",
                "call_attempts": list(call_details.get("attempts") or []),
            }
            diagnostics["llm_attempts"].append(attempt_record)

            if attempt_index == 1:
                diagnostics["prompt_preview"] = preview(prompt, limit=520)
                diagnostics["raw_llm_output_preview"] = preview(str(content or ""), limit=220) if content is not None else ""

            if cleaned:
                diagnostics["answer_source"] = "llm"
                diagnostics["llm_attempt_index"] = attempt_index
                diagnostics["llm_attempt_context_count"] = len(attempt_contexts)
                return cleaned, None, diagnostics

            last_error = str(call_details.get("last_error") or "").strip()
            if last_error:
                final_error = last_error
            elif content is not None:
                final_error = "LLM returned empty content after cleaning."

        diagnostics["answer_source"] = "llm_no_response"
        diagnostics["error"] = final_error
        return None, final_error, diagnostics
    except Exception as exc:
        diagnostics["answer_source"] = "error"
        diagnostics["error"] = f"LLM request failed: {exc}"
        return None, f"LLM request failed: {exc}", diagnostics


def print_llm_diagnostics(diagnostics: dict[str, Any] | None) -> None:
    print("\n=== LLM Diagnostics ===")
    if not diagnostics:
        print("No diagnostics.")
        return
    print(f"answer_source={diagnostics.get('answer_source') or ''}")
    print(f"provider={diagnostics.get('provider') or ''}")
    print(f"model_name={diagnostics.get('model_name') or ''}")
    print(f"question_type={diagnostics.get('question_type') or ''}")
    print(f"target_company={diagnostics.get('target_company') or ''}")
    print(f"query_tags={diagnostics.get('query_tags') or []}")
    print(f"context_count={diagnostics.get('context_count') or 0}")
    print(f"aggregate_focus_hits={diagnostics.get('aggregate_focus_hits') or []}")
    if diagnostics.get("error"):
        print(f"error={diagnostics.get('error')}")
    if diagnostics.get("raw_llm_output_preview"):
        print(f"raw_llm_output_preview={diagnostics.get('raw_llm_output_preview')}")
    if diagnostics.get("prompt_preview"):
        print(f"prompt_preview={diagnostics.get('prompt_preview')}")
    if diagnostics.get("llm_attempt_index"):
        print(f"llm_attempt_index={diagnostics.get('llm_attempt_index')}")
        print(f"llm_attempt_context_count={diagnostics.get('llm_attempt_context_count')}")
    attempts = list(diagnostics.get("llm_attempts") or [])
    if attempts:
        print("\n=== LLM Attempt Log ===")
        for item in attempts:
            print(
                f"[{int(item.get('attempt_index') or 0)}] contexts={int(item.get('context_count') or 0)} "
                f"per_context_limit={int(item.get('per_context_limit') or 0)} "
                f"max_tokens={int(item.get('max_tokens') or 0)} "
                f"prompt_chars={int(item.get('prompt_chars') or 0)} "
                f"status={item.get('call_status') or ''} route={item.get('selected_route') or ''}"
            )
            if item.get("cleaned_output_preview"):
                print(f"    cleaned={item.get('cleaned_output_preview')}")
            elif item.get("raw_output_preview"):
                print(f"    raw={item.get('raw_output_preview')}")
            for call_attempt in list(item.get("call_attempts") or []):
                error = str(call_attempt.get("error") or "")
                route = str(call_attempt.get("route") or "")
                model_name = str(call_attempt.get("model_name") or "")
                result = str(call_attempt.get("result") or "")
                print(f"    call route={route} model={model_name} result={result}")
                if error:
                    print(f"    error={error}")

    print("\n=== Final Contexts To LLM ===")
    contexts = list(diagnostics.get("contexts") or [])
    if not contexts:
        print("No contexts.")
        return
    for item in contexts:
        print(
            f"[{int(item.get('rank') or 0)}] page={int(item.get('page_number') or 0)} "
            f"type={item.get('page_type') or ''} "
            f"chunk={item.get('chunk_id') or ''} table={item.get('table_id') or ''} visual={item.get('visual_id') or ''}"
        )
        print(f"    source_pages={item.get('source_pages') or []} focus_hits={item.get('focus_hits') or []}")
        print(f"    preview={item.get('preview') or ''}")


if __name__ == "__main__":
    args = parse_args()
    artifact_dir = resolve_artifact_dir(args.artifact_dir)

    print(f"query={args.query}")
    print(f"artifact_dir={artifact_dir}")
    print(f"text_collection={settings.text_vector_collection_name}")
    print(f"visual_collection={settings.visual_vector_collection_name}")
    print(f"mongo_collection={settings.mongodb_table_collection}")
    print(
        "text_retrieval_mode="
        f"dense+bm25+overlap{' + rerank' if (not args.disable_rerank) else ''}"
    )

    table_lookup = build_table_lookup(artifact_dir)
    visual_lookup = build_visual_lookup(artifact_dir)
    total_pages = infer_total_pages_from_chunks(load_text_chunk_lookup(artifact_dir))
    text_rows = search_text_vectors(args.query, artifact_dir, table_lookup, visual_lookup)
    visual_rows = rerank_visual_rows_by_page_position(search_visual_vectors(args.query), total_pages)
    linked_marker_ids = extract_linked_markers(text_rows)
    linked_marker_candidates = extract_linked_marker_candidates_v2(text_rows)
    linked_table_rows = build_linked_table_results_v2(linked_marker_candidates, table_lookup, args.query)
    keyword_table_rows = search_tables_keyword(args.query, artifact_dir)

    merged_table_rows: list[dict[str, Any]] = []
    seen_table_ids: set[str] = set()
    for row in [*linked_table_rows, *keyword_table_rows]:
        table_id = str(row.get("table_id") or "")
        if table_id in seen_table_ids:
            continue
        seen_table_ids.add(table_id)
        merged_table_rows.append(row)
    merged_table_rows.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    table_rows = merged_table_rows[: max(1, args.table_top_k)]
    llm_answer: str | None = None
    llm_error: str | None = None
    llm_diagnostics: dict[str, Any] | None = None
    if not args.disable_llm:
        llm_answer, llm_error, llm_diagnostics = generate_llm_answer_with_diagnostics(
            query=args.query,
            text_rows=text_rows,
            table_rows=table_rows,
            visual_rows=visual_rows,
            table_lookup=table_lookup,
            visual_lookup=visual_lookup,
            limit=args.llm_context_k,
        )

    print_text_results(text_rows)
    print_visual_results(visual_rows)
    print_linked_objects(linked_marker_ids, table_lookup, visual_lookup)
    print_table_keyword_results(table_rows)
    if not args.disable_llm:
        print_llm_answer(llm_answer, llm_error)
        print_llm_diagnostics(llm_diagnostics)
