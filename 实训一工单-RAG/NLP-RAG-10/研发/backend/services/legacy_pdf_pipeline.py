from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from backend.config import settings
from backend.services.embedding import EmbeddingService
from backend.services.rag_pipeline.upload_registry import build_uploaded_collection_names, ensure_uploads_root
from backend.utils.logging import get_logger

from backend.data_process import stage0_cleanup, stage1_layout_analysis, stage2_multimodal_runner, stage2_precise_extraction
from backend.data_process import stage3_text_chunking, stage4_vectorize_text, stage4_vectorize_visual, stage5_persist_outputs

logger = get_logger(__name__)


@dataclass
class LegacyPdfProcessResult:
    artifact_dir: Path
    text_collection_name: str
    visual_collection_name: str
    mongo_collection_name: str
    stage0_path: Path
    stage1_path: Path
    stage2_dir: Path
    stage3_dir: Path
    stage4_text_dir: Path
    stage4_visual_dir: Path
    stage5_manifest_path: Path
    page_text_flow_resolved_path: Path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _artifact_ready(path: Path, *, min_size: int = 16) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size >= min_size


def _dir_has_files(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _stage5_result_path(artifact_dir: Path) -> Path:
    return artifact_dir / stage5_persist_outputs.DEFAULT_OUTPUT_NAME


def _slugify_collection_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "uploaded_doc"


def build_upload_storage_names(upload_id: str, filename: str) -> tuple[str, str, str]:
    base_name = Path(filename).stem
    base_component = _slugify_collection_component(base_name)[:48]
    upload_component = _slugify_collection_component(upload_id)[:48]
    text_collection = f"pdf_text_{base_component}_{upload_component}"
    visual_collection = f"pdf_visual_{base_component}_{upload_component}"
    mongo_collection = f"pdf_tables_{base_component}_{upload_component}"
    return text_collection, visual_collection, mongo_collection


def _run_stage0(pdf_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    watermark_patterns = stage0_cleanup.load_watermark_patterns()
    with fitz.open(str(pdf_path)) as doc:
        preview_pages = [doc.load_page(index) for index in range(min(20, len(doc)))]
        repeated_lines = stage0_cleanup.collect_repeated_lines(preview_pages)
        position_dense_counter = stage0_cleanup.collect_position_dense_candidates(preview_pages)
        visual_template_boxes, visual_template_debug = stage0_cleanup.learn_visual_watermark_template(
            preview_pages,
            learn_pages=min(5, len(preview_pages)),
        )
        results: list[dict[str, Any]] = []
        for index in range(len(doc)):
            page = doc.load_page(index)
            results.append(
                stage0_cleanup.stage0_for_page(
                    page,
                    repeated_lines,
                    watermark_patterns,
                    position_dense_counter,
                    visual_template_boxes,
                )
            )
    payload = {
        "pdf_path": str(pdf_path),
        "page_count": len(results),
        "visual_watermark_template_boxes": visual_template_boxes,
        "visual_watermark_template_debug": visual_template_debug,
        "results": results,
    }
    output_path = output_dir / "stage0_selected_pages.json"
    _write_json(output_path, payload)
    return output_path


def _run_stage1(pdf_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = stage1_layout_analysis.load_config()
    started_at = stage1_layout_analysis.time.time()
    with fitz.open(str(pdf_path)) as doc:
        total_pages = len(doc)
        target_pages = list(range(total_pages))
        preview_pages = [doc.load_page(index) for index in target_pages]
        repeated_lines = stage1_layout_analysis.collect_repeated_lines(preview_pages)
        watermark_patterns = stage1_layout_analysis.load_watermark_patterns()
        position_dense_counter = stage1_layout_analysis.collect_position_dense_candidates(preview_pages)

        pages: list[Any] = []
        for page_num in target_pages:
            layout_page = stage1_layout_analysis.analyze_page_layout(
                doc,
                page_num,
                repeated_lines,
                watermark_patterns,
                position_dense_counter,
            )
            pages.append(layout_page)

    cross_page_tables, cross_page_debug = stage1_layout_analysis.detect_cross_page_tables(pages, config)
    for page in pages:
        page.text_flow = stage1_layout_analysis.build_text_flow(page)

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
    for merged_table in cross_page_tables:
        structure_objects["tables"].append(merged_table.to_dict())

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
        "text_flow": {
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
        },
        "structure_objects": structure_objects,
        "cross_page_table_debug": cross_page_debug,
        "repeated_header_footer_lines": repeated_lines,
        "page_details": [page.to_dict() for page in pages],
        "elapsed_seconds": round(stage1_layout_analysis.time.time() - started_at, 3),
    }
    output_path = output_dir / "stage1_layout_analysis.json"
    _write_json(output_path, final_output)
    return output_path


def _run_stage2(pdf_path: Path, stage1_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = stage2_precise_extraction.load_config()
    stage1 = _read_json(stage1_path)
    layout_by_page = stage2_precise_extraction.build_layout_lookup(stage1)
    start_time = stage2_precise_extraction.time.time()
    with fitz.open(str(pdf_path)) as doc:
        text_regions, page_text_flow = stage2_precise_extraction.build_text_regions(doc, stage1.get("page_details", []))
        figure_tasks = stage2_precise_extraction.build_figure_tasks(doc, layout_by_page, output_dir)
    raw_tables, raw_index = stage2_precise_extraction.build_tables(pdf_path, layout_by_page, stage1, config)
    table_chains, _ = stage2_precise_extraction.build_table_chains(stage1, raw_index)
    with fitz.open(str(pdf_path)) as doc:
        multimodal_tasks, object_registry, resolved_page_text_flow, _ = stage2_precise_extraction.build_multimodal_objects(
            doc,
            page_text_flow,
            raw_tables,
            table_chains,
            figure_tasks,
            output_dir,
            config,
        )
    backend_counts = stage2_precise_extraction.Counter(item.extraction_backend for item in raw_tables)
    stage2_precise_extraction.jsonl_write(output_dir / "text_regions.jsonl", text_regions)
    _write_json(output_dir / "page_text_flow.json", page_text_flow)
    _write_json(output_dir / "page_text_flow_resolved.json", resolved_page_text_flow)
    stage2_precise_extraction.jsonl_write(output_dir / "tables_raw.jsonl", [item.to_dict() for item in raw_tables])
    stage2_precise_extraction.jsonl_write(output_dir / "table_chains.jsonl", table_chains)
    stage2_precise_extraction.jsonl_write(output_dir / "figure_tasks.jsonl", figure_tasks)
    stage2_precise_extraction.jsonl_write(output_dir / "multimodal_tasks.jsonl", multimodal_tasks)
    stage2_precise_extraction.jsonl_write(output_dir / "object_registry.jsonl", object_registry)
    summary = {
        "pdf_path": str(pdf_path),
        "page_count": len(stage1.get("page_details", [])),
        "text_region_count": len(text_regions),
        "table_count": len(raw_tables),
        "cross_page_table_count": len(table_chains),
        "figure_task_count": len(figure_tasks),
        "multimodal_task_count": len(multimodal_tasks),
        "object_registry_count": len(object_registry),
        "table_backend_counts": dict(backend_counts),
        "generated_files": [
            "text_regions.jsonl",
            "page_text_flow.json",
            "page_text_flow_resolved.json",
            "tables_raw.jsonl",
            "table_chains.jsonl",
            "figure_tasks.jsonl",
            "multimodal_tasks.jsonl",
            "object_registry.jsonl",
        ],
        "elapsed_seconds": round(stage2_precise_extraction.time.time() - start_time, 3),
    }
    _write_json(output_dir / "stage2_manifest.json", summary)
    return output_dir


def _run_stage2_vlm(stage2_dir: Path) -> Path:
    stage2_multimodal_runner.ensure_vlm_enabled()
    task_path = stage2_dir / "multimodal_tasks.jsonl"
    result_path = stage2_dir / stage2_multimodal_runner.RESULT_FILE_NAME
    raw_dir = stage2_dir / stage2_multimodal_runner.RAW_DIR_NAME
    raw_dir.mkdir(parents=True, exist_ok=True)

    tasks = stage2_multimodal_runner.load_jsonl(task_path)
    if not tasks:
        raise FileNotFoundError(f"No multimodal tasks found: {task_path}")

    existing_rows = stage2_multimodal_runner.load_jsonl(result_path)
    existing_map = {str(item.get("final_object_id") or ""): item for item in existing_rows}
    new_results: dict[str, dict[str, Any]] = {}
    pending_tasks = [
        task
        for task in tasks
        if not (str(task.get("final_object_id") or "") in existing_map and existing_map[str(task.get("final_object_id") or "")].get("status") == "success")
    ]
    logger.info(
        "[upload-pdf][stage2_vlm] start task_total=%s pending=%s artifact_dir=%s",
        len(tasks),
        len(pending_tasks),
        stage2_dir,
    )

    for index, task in enumerate(pending_tasks, start=1):
        task_id = str(task.get("task_id") or "")
        final_object_id = str(task.get("final_object_id") or "")
        task_type = str(task.get("task_type") or "")
        logger.info(
            "[upload-pdf][stage2_vlm] task %s/%s task_id=%s final_object_id=%s type=%s",
            index,
            len(pending_tasks),
            task_id,
            final_object_id,
            task_type,
        )
        try:
            raw_response, meta = stage2_multimodal_runner.request_vlm(task)
            raw_path = raw_dir / f"{task_id}.raw.json"
            raw_path.write_text(json.dumps(raw_response, ensure_ascii=False, indent=2), encoding="utf-8")
            content, structured_content = stage2_multimodal_runner.parse_vlm_payload(task, meta["message_content"])
            result = stage2_multimodal_runner.build_result_record(
                task,
                status="success",
                content=content,
                structured_content=structured_content,
                model=settings.pdf_vlm_model_name,
                provider=settings.pdf_vlm_provider,
                raw_response_path=str(raw_path),
                error=None,
                prompt_digest_value=meta["prompt_digest"],
            )
        except Exception as exc:
            logger.exception(
                "[upload-pdf][stage2_vlm] task failed task_id=%s final_object_id=%s",
                task_id,
                final_object_id,
            )
            result = stage2_multimodal_runner.build_result_record(
                task,
                status="failed",
                content="",
                structured_content={},
                model=settings.pdf_vlm_model_name,
                provider=settings.pdf_vlm_provider,
                raw_response_path="",
                error=f"{type(exc).__name__}: {exc}",
                prompt_digest_value="",
            )
        existing_map[final_object_id] = result
        new_results[final_object_id] = result

    final_rows = list(existing_map.values())
    final_rows.sort(key=lambda item: (str(item.get("final_object_id") or ""), str(item.get("created_at") or "")))
    stage2_multimodal_runner.write_jsonl(result_path, final_rows)
    stage2_multimodal_runner.update_registry_and_resolved_text(stage2_dir, new_results)
    logger.info(
        "[upload-pdf][stage2_vlm] done processed=%s result_path=%s",
        len(new_results),
        result_path,
    )
    return result_path


def _run_stage3(stage2_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages, _ = stage3_text_chunking.load_page_text_flow(stage2_dir, "auto")
    normalized_pages = [stage3_text_chunking.normalize_page_entry(item) for item in pages]
    blocks, flow_debug = stage3_text_chunking.build_flow_blocks(normalized_pages)
    chunks = stage3_text_chunking.chunk_blocks(
        blocks,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    stage3_text_chunking.attach_heading_links(chunks)
    heading_outline = stage3_text_chunking.build_heading_outline(blocks, chunks)
    stage3_text_chunking.write_jsonl(output_dir / "text_chunks.jsonl", chunks)
    stage3_text_chunking.write_json(
        output_dir / "chunk_manifest.json",
        stage3_text_chunking.build_summary(
            source_path=stage2_dir,
            output_dir=output_dir,
            pages=normalized_pages,
            blocks=blocks,
            chunks=chunks,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            seed_pages=stage3_text_chunking.DEFAULT_SEED_PAGES,
            append_threshold=stage3_text_chunking.DEFAULT_APPEND_THRESHOLD,
            heading_outline=heading_outline,
            flow_debug=flow_debug,
        ),
    )
    return output_dir


def _run_stage4_text(stage3_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = stage4_vectorize_text.load_chunks(stage3_dir / "text_chunks.jsonl")
    embedder = EmbeddingService(settings.model_dir / "embedding", configured_path=settings.embedding_model_path)
    vectors = embedder.embed_texts(str(item.get("text") or "") for item in chunks)
    vector_index_rows: list[dict[str, Any]] = []
    for offset, (chunk_item, embedding) in enumerate(zip(chunks, vectors)):
        vector_index_rows.append(
            {
                "chunk_id": str(chunk_item.get("chunk_id") or ""),
                "chunk_index": int(chunk_item.get("chunk_index") or 0),
                "source_pages": list(chunk_item.get("source_pages") or []),
                "source_page_count": int(chunk_item.get("source_page_count") or 0),
                "char_count": int(chunk_item.get("char_count") or 0),
                "marker_count": int(chunk_item.get("marker_count") or 0),
                "markers": list(chunk_item.get("markers") or []),
                "embedding_offset": offset,
                "embedding_dim": len(embedding),
                "text_preview": str(chunk_item.get("text") or "")[:240],
            }
        )
    stage4_vectorize_text.np.save(output_dir / "chunk_embeddings.npy", stage4_vectorize_text.np.asarray(vectors, dtype=stage4_vectorize_text.np.float32))
    stage4_vectorize_text.write_jsonl(output_dir / "chunk_vector_index.jsonl", vector_index_rows)
    stage4_vectorize_text.write_json(
        output_dir / "vector_manifest.json",
        {
            "source_chunk_dir": str(stage3_dir),
            "output_dir": str(output_dir),
            "chunk_count": len(chunks),
            "vector_count": len(vectors),
            "embedding_dimension": embedder.dimension,
            "embedding_backend": embedder.backend,
            "embedding_model_path": settings.embedding_model_path,
        },
    )
    return output_dir


def _run_stage4_visual(stage2_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = stage4_vectorize_visual.load_visual_results(stage2_dir / stage4_vectorize_visual.DEFAULT_RESULT_FILE)
    embedder = EmbeddingService(settings.model_dir / "embedding", configured_path=settings.embedding_model_path)
    search_text_rows = []
    vector_index_rows = []
    vectors = []
    for offset, item in enumerate(results):
        search_text = stage4_vectorize_visual.build_visual_search_text(item)
        embedding = embedder.embed_query(search_text)
        search_text_rows.append({"visual_id": str(item.get("final_object_id") or ""), "search_text": search_text})
        vector_index_rows.append(stage4_vectorize_visual.build_visual_vector_record(item, embedding, offset))
        vectors.append(embedding)
    stage4_vectorize_visual.np.save(output_dir / "visual_embeddings.npy", stage4_vectorize_visual.np.asarray(vectors, dtype=stage4_vectorize_visual.np.float32))
    stage4_vectorize_visual.write_jsonl(output_dir / "visual_vector_index.jsonl", vector_index_rows)
    stage4_vectorize_visual.write_jsonl(output_dir / "visual_search_texts.jsonl", search_text_rows)
    stage4_vectorize_visual.write_json(
        output_dir / "visual_vector_manifest.json",
        {
            "source_result_file": str(stage2_dir / stage4_vectorize_visual.DEFAULT_RESULT_FILE),
            "output_dir": str(output_dir),
            "visual_count": len(results),
            "vector_count": len(vectors),
            "embedding_dimension": embedder.dimension,
            "embedding_backend": embedder.backend,
            "embedding_model_path": settings.embedding_model_path,
        },
    )
    return output_dir


def _run_stage5(
    artifact_dir: Path,
    *,
    doc_name: str,
    text_collection_name: str,
    visual_collection_name: str,
    mongo_collection_name: str,
) -> Path:
    output_manifest = artifact_dir / "stage5_persist_manifest.json"
    argv = [
        "stage5_persist_outputs.py",
        "--artifact-dir",
        str(artifact_dir),
        "--output-manifest",
        str(output_manifest),
        "--doc-name",
        doc_name,
        "--text-collection",
        text_collection_name,
        "--visual-collection",
        visual_collection_name,
        "--mongo-collection",
        mongo_collection_name,
    ]
    previous_argv = os.sys.argv[:]
    try:
        os.sys.argv = argv
        stage5_persist_outputs.main()
    finally:
        os.sys.argv = previous_argv
    return output_manifest


def run_legacy_pdf_pipeline(pdf_path: Path, artifact_dir: Path, original_filename: str, upload_id: str) -> LegacyPdfProcessResult:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    doc_name = Path(original_filename).stem
    text_collection_name, visual_collection_name = build_uploaded_collection_names(upload_id)
    mongo_collection_name = build_upload_storage_names(upload_id, original_filename)[2]

    stage0_dir = artifact_dir / "stage0_cleanup"
    stage1_dir = artifact_dir / "stage1_layout_analysis"
    stage2_dir = artifact_dir / "stage2_precise_extraction"
    stage3_dir = artifact_dir / "stage3_text_chunking"
    stage4_text_dir = artifact_dir / "stage4_vectorized_chunks"
    stage4_visual_dir = artifact_dir / "stage4_vectorized_visuals"

    stage0_path = stage0_dir / "stage0_selected_pages.json"
    if _artifact_ready(stage0_path):
        logger.info("[upload-pdf][resume] reuse stage0 path=%s", stage0_path)
    else:
        stage0_path = _run_stage0(pdf_path, stage0_dir)

    stage1_path = stage1_dir / "stage1_layout_analysis.json"
    if _artifact_ready(stage1_path):
        logger.info("[upload-pdf][resume] reuse stage1 path=%s", stage1_path)
    else:
        stage1_path = _run_stage1(pdf_path, stage1_dir)

    stage2_manifest_path = stage2_dir / "stage2_manifest.json"
    if _artifact_ready(stage2_manifest_path) and _artifact_ready(stage2_dir / "page_text_flow_resolved.json"):
        logger.info("[upload-pdf][resume] reuse stage2 dir=%s", stage2_dir)
    else:
        _run_stage2(pdf_path, stage1_path, stage2_dir)

    stage2_vlm_result_path = stage2_dir / stage2_multimodal_runner.RESULT_FILE_NAME
    if _artifact_ready(stage2_dir / "multimodal_tasks.jsonl"):
        _run_stage2_vlm(stage2_dir)
    else:
        logger.info("[upload-pdf][resume] skip stage2_vlm no tasks file dir=%s", stage2_dir)

    stage3_manifest_path = stage3_dir / "chunk_manifest.json"
    if _artifact_ready(stage3_manifest_path) and _artifact_ready(stage3_dir / "text_chunks.jsonl"):
        logger.info("[upload-pdf][resume] reuse stage3 dir=%s", stage3_dir)
    else:
        _run_stage3(stage2_dir, stage3_dir)

    stage4_text_manifest_path = stage4_text_dir / "vector_manifest.json"
    if _artifact_ready(stage4_text_manifest_path) and _artifact_ready(stage4_text_dir / "chunk_vector_index.jsonl"):
        logger.info("[upload-pdf][resume] reuse stage4_text dir=%s", stage4_text_dir)
    else:
        _run_stage4_text(stage3_dir, stage4_text_dir)

    stage4_visual_manifest_path = stage4_visual_dir / "visual_vector_manifest.json"
    if _artifact_ready(stage2_vlm_result_path) and _artifact_ready(stage4_visual_manifest_path):
        logger.info("[upload-pdf][resume] reuse stage4_visual dir=%s", stage4_visual_dir)
    else:
        _run_stage4_visual(stage2_dir, stage4_visual_dir)

    stage5_manifest_path = _stage5_result_path(artifact_dir)
    if _artifact_ready(stage5_manifest_path):
        logger.info("[upload-pdf][resume] reuse stage5 path=%s", stage5_manifest_path)
    else:
        stage5_manifest_path = _run_stage5(
            artifact_dir,
            doc_name=doc_name,
            text_collection_name=text_collection_name,
            visual_collection_name=visual_collection_name,
            mongo_collection_name=mongo_collection_name,
        )

    manifest_payload = {
        "upload_id": upload_id,
        "filename": original_filename,
        "pdf_path": str(pdf_path),
        "artifact_dir": str(artifact_dir),
        "doc_name": doc_name,
        "text_collection_name": text_collection_name,
        "visual_collection_name": visual_collection_name,
        "mongo_collection_name": mongo_collection_name,
        "stages": {
            "stage0": str(stage0_path),
            "stage1": str(stage1_path),
            "stage2": str(stage2_dir),
            "stage3": str(stage3_dir),
            "stage4_text": str(stage4_text_dir),
            "stage4_visual": str(stage4_visual_dir),
            "stage5": str(stage5_manifest_path),
        },
    }
    _write_json(artifact_dir / "manifest.json", manifest_payload)

    return LegacyPdfProcessResult(
        artifact_dir=artifact_dir,
        text_collection_name=text_collection_name,
        visual_collection_name=visual_collection_name,
        mongo_collection_name=mongo_collection_name,
        stage0_path=stage0_path,
        stage1_path=stage1_path,
        stage2_dir=stage2_dir,
        stage3_dir=stage3_dir,
        stage4_text_dir=stage4_text_dir,
        stage4_visual_dir=stage4_visual_dir,
        stage5_manifest_path=stage5_manifest_path,
        page_text_flow_resolved_path=stage2_dir / "page_text_flow_resolved.json",
    )


def load_resolved_pages_as_parser_output(result: LegacyPdfProcessResult, pdf_path: Path) -> list[dict[str, Any]]:
    resolved_pages = _read_json(result.page_text_flow_resolved_path)
    object_registry_rows = _read_jsonl(result.stage2_dir / "object_registry.jsonl")
    registry_by_id = {str(item.get("final_object_id") or ""): item for item in object_registry_rows}
    pages: list[dict[str, Any]] = []
    for page in resolved_pages:
        page_number = int(page.get("page_index") or 0)
        page_text = str(page.get("page_text_flow") or "").strip()
        object_flow = list(page.get("object_flow") or [])
        structured_facts: list[dict[str, Any]] = []
        context_links: list[dict[str, Any]] = []
        for entry in object_flow:
            final_object_id = str(entry.get("final_object_id") or "")
            registry_entry = registry_by_id.get(final_object_id, {})
            if not final_object_id:
                continue
            context_links.append(
                {
                    "element_id": final_object_id,
                    "type": str(entry.get("region_type") or "object"),
                    "marker_in_text": str(entry.get("marker") or ""),
                    "page_number": page_number,
                    "status": str(entry.get("resolved_status") or registry_entry.get("status") or ""),
                }
            )
            latest_content = str(registry_entry.get("latest_content") or "").strip()
            latest_structured = registry_entry.get("latest_structured_content")
            if latest_content:
                structured_facts.append(
                    {
                        "page_number": page_number,
                        "source_element_id": final_object_id,
                        "primary_type": str(registry_entry.get("object_type") or "mixed"),
                        "fact_type": str(registry_entry.get("object_type") or "object_summary"),
                        "title": final_object_id,
                        "value": latest_content,
                        "structured_content": latest_structured,
                    }
                )
        pages.append(
            {
                "page_number": page_number,
                "logical_page": None,
                "text": page_text,
                "redacted_text": page_text,
                "raw_text": page_text,
                "tables_markdown": "",
                "handwriting": "",
                "page_type": str(page.get("page_type") or "text"),
                "primary_type": "text",
                "sub_type": str(page.get("sub_type") or "paragraph"),
                "type_confidence": 0.9,
                "candidate_types": [str(page.get("page_type") or "text")],
                "layout_tags": ["legacy_stage_pipeline"],
                "content_tags": ["legacy_stage_pipeline"],
                "section_title": "",
                "source": "legacy_stage_pipeline",
                "source_pdf": pdf_path.name,
                "source_pdf_path": str(pdf_path),
                "parse_metadata": {
                    "legacy_stage_pipeline": True,
                    "object_flow_count": len(object_flow),
                    "structured_fact_count": len(structured_facts),
                },
                "context_links": context_links,
                "structured_facts": structured_facts,
                "pdf_intelligence": {
                    "stage2_dir": str(result.stage2_dir),
                    "stage5_manifest_path": str(result.stage5_manifest_path),
                },
            }
        )
    return pages
