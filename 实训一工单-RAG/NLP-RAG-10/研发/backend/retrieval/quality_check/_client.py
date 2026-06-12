# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from backend.config import settings
from backend.services.embedding import EmbeddingService
from backend.services.llm_client._answers import _clean_answer_text
from backend.utils.retrieval import (
    SimpleBM25Index,
    normalize_score_map,
    keyword_overlap_score,
    extract_focus_terms,
    split_sentences,
)
from ._schema import (
    parse_args,
    resolve_artifact_dir,
    infer_total_pages_from_chunks,
    load_jsonl,
)
from ._metrics import (
    preview,
    tokenize_for_bm25,
    compute_marker_boost,
    compute_answer_boost,
    compute_focus_signal,
    compute_company_signal,
    compute_page_position_penalty,
    score_table_row_for_query,
    build_table_query_text,
    build_visual_query_text,
)
from ._validators import (
    looks_like_mojibake,
    load_text_chunk_lookup,
    load_text_chunk_sequence,
    get_marker_ids_from_text,
    build_text_windows,
    collect_neighbor_chunks,
    build_neighbor_window_payload,
    expand_neighbor_candidates,
    extract_linked_markers,
    extract_linked_marker_candidates_v2,
    extract_marker_context_snippet_v2,
    extract_target_company,
    stringify_table_row,
    select_table_rows_for_query,
    build_table_context_text,
    build_table_context_text_v2,
    build_table_full_row_texts_v2,
    is_industry_chain_query,
    is_strong_industry_chain_context,
    build_visual_context_text,
    build_complete_context_text,
    should_keep_context,
    infer_question_type,
    infer_query_tags,
    load_table_documents,
    search_tables_keyword,
    build_table_lookup,
    build_visual_lookup,
    build_linked_table_results,
    build_linked_table_results_v2,
    load_visual_vlm_results,
    hydrate_visual_row_from_vlm,
)
from ._report import (
    print_text_results,
    print_visual_results,
    print_linked_objects,
    print_table_keyword_results,
    print_llm_answer,
    print_llm_diagnostics,
)


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
                    f"[\u8bc1\u636e{index}] \u9875\u7801={int(item.get('page_number') or 0)} \u7c7b\u578b={str(metadata.get('page_type') or '')}",
                    text if str(metadata.get("page_type") or "") == "table" or bool(metadata.get("never_truncate")) else text[:per_context_limit],
                ]
            )
        )
    evidence_text = "\n\n".join(evidence_blocks)
    return (
        "\u4f60\u662f\u4e00\u4e2a\u4e25\u683c\u57fa\u4e8e\u62db\u80a1\u4e66\u8bc1\u636e\u56de\u7b54\u95ee\u9898\u7684\u52a9\u624b\u3002\n"
        "\u53ea\u5141\u8bb8\u4f9d\u636e\u7ed9\u5b9a\u8bc1\u636e\u4f5c\u7b54\uff0c\u4e0d\u5141\u8bb8\u4f7f\u7528\u5916\u90e8\u77e5\u8bc6\uff0c\u4e0d\u5141\u8bb8\u731c\u6d4b\u3002\n"
        "\u5982\u679c\u8bc1\u636e\u4e0d\u80fd\u76f4\u63a5\u56de\u7b54\u95ee\u9898\uff0c\u5c31\u660e\u786e\u56de\u7b54\uff1a\u672a\u68c0\u7d22\u5230\u8db3\u591f\u8bc1\u636e\u3002\n"
        "\u56de\u7b54\u65f6\u8bf7\u5148\u505a\u8bc1\u636e\u5224\u65ad\uff0c\u53ea\u63d0\u53d6\u4e0e\u95ee\u9898\u8bed\u4e49\u76f4\u63a5\u5bf9\u5e94\u7684\u4fe1\u606f\uff0c\u5ffd\u7565\u540c\u8bcd\u5f02\u4e49\u3001\u65c1\u652f\u63cf\u8ff0\u3001\u8d22\u52a1\u566a\u58f0\u548c\u65e0\u5173\u6bb5\u843d\u3002\n"
        "\u4e0d\u8981\u8f93\u51fa\u5185\u90e8\u6807\u8bb0\u3001\u8868\u683cID\u3001chunk_id\u3001marker\u3001source_pdf\u3001minio_path\u7b49\u5185\u90e8\u5b57\u6bb5\u3002\n"
        "\u4e0d\u8981\u7167\u6458\u5927\u6bb5\u539f\u6587\uff0c\u8981\u6574\u7406\u6210\u81ea\u7136\u4e2d\u6587\u53e5\u5b50\u3002\n"
        "\u7b54\u6848\u672b\u5c3e\u5fc5\u987b\u9644\uff1a\u5f15\u7528\u9875\u7801\uff1a\u9875\u78011\u3001\u9875\u78012\u3002\n\n"
        f"\u95ee\u9898\uff1a{query}\n\n"
        f"\u8bc1\u636e\uff1a\n{evidence_text}\n\n"
        "\u8bf7\u76f4\u63a5\u7ed9\u51fa\u6700\u7ec8\u7b54\u6848\u3002"
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
        score = keyword_overlap_score(query, sentence) * 10.0
        focus_hits = sum(1 for term in focus_terms if term in sentence)
        if focus_hits:
            score += focus_hits * 3.0
        scored.append((index, score, sentence))

    positive = [item for item in scored if item[1] > 0]
    selected = positive if positive else scored[:max_sentences]
    selected.sort(key=lambda item: (-item[1], item[0]))
    top_items = sorted(selected[:max_sentences], key=lambda item: item[0])
    return " ".join(item[2] for item in top_items)


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
                    replacement = f"\n\u8868\u683c\u5185\u5bb9\uff1a\n{table_text}\n"
                    used_table_ids.add(marker_id)
            else:
                visual_row = ensure_visual_row(marker_id)
                if visual_row is not None:
                    visual_text = build_visual_context_text(visual_row)
                    if visual_text:
                        replacement = f"\n\u56fe\u8868\u5185\u5bb9\uff1a\n{visual_text}\n"
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

    from backend.utils.retrieval import normalize_text as norm_text
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
    cleaned = norm_text(answer)
    return cleaned or None, None


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

    from backend.utils.retrieval import normalize_text as norm_text
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
                system_prompt="\u4f60\u662f\u4e00\u4e2a\u4e25\u683c\u57fa\u4e8e\u8bc1\u636e\u56de\u7b54\u95ee\u9898\u7684 RAG \u52a9\u624b\u3002",
                max_tokens=int(plan["max_tokens"]),
            )
            cleaned = norm_text(_clean_answer_text(content or "")) if content is not None else ""
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


class QualityChecker:
    """Quality checker class delegating to extracted functions."""
    
    def __init__(self) -> None:
        pass

    def check_text_results(self, rows: list[dict[str, Any]]) -> None:
        print_text_results(rows)

    def check_visual_results(self, rows: list[dict[str, Any]]) -> None:
        print_visual_results(rows)

    def check_linked_objects(self, marker_ids: list[str], table_lookup: dict[str, dict[str, Any]], visual_lookup: dict[str, dict[str, Any]]) -> None:
        print_linked_objects(marker_ids, table_lookup, visual_lookup)

    def check_table_results(self, rows: list[dict[str, Any]]) -> None:
        print_table_keyword_results(rows)

    def check_llm_answer(self, answer: str | None, error: str | None) -> None:
        print_llm_answer(answer, error)

    def check_llm_diagnostics(self, diagnostics: dict[str, Any] | None) -> None:
        print_llm_diagnostics(diagnostics)

    def search_text(self, query: str, artifact_dir: Path, table_lookup: dict[str, dict[str, Any]], visual_lookup: dict[str, dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        return search_text_vectors(query, artifact_dir, table_lookup, visual_lookup, **kwargs)

    def search_visual(self, query: str, **kwargs) -> list[dict[str, Any]]:
        rows = search_visual_vectors(query, **kwargs)
        return rows

    def generate_answer(self, query: str, text_rows: list[dict[str, Any]], table_rows: list[dict[str, Any]], visual_rows: list[dict[str, Any]], table_lookup: dict[str, dict[str, Any]], visual_lookup: dict[str, dict[str, Any]], limit: int) -> tuple[str | None, str | None]:
        return generate_llm_answer(query, text_rows, table_rows, visual_rows, table_lookup, visual_lookup, limit)

    def generate_answer_with_diagnostics(self, query: str, text_rows: list[dict[str, Any]], table_rows: list[dict[str, Any]], visual_rows: list[dict[str, Any]], table_lookup: dict[str, dict[str, Any]], visual_lookup: dict[str, dict[str, Any]], limit: int) -> tuple[str | None, str | None, dict[str, Any] | None]:
        return generate_llm_answer_with_diagnostics(query, text_rows, table_rows, visual_rows, table_lookup, visual_lookup, limit)


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
