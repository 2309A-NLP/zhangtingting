# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations

from typing import Any

from ._metrics import preview


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


def print_llm_answer(answer: str | None, error: str | None) -> None:
    print("\n=== LLM Answer ===")
    if error:
        print(f"Unavailable: {error}")
        return
    if not answer:
        print("No LLM answer.")
        return
    print(answer)


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
