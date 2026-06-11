from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.services.embedding import EmbeddingService


DEFAULT_RESULT_FILE = "vlm_results.jsonl"
DEFAULT_OUTPUT_DIRNAME = "stage4_vectorized_visuals"
DEFAULT_BATCH_SIZE = 32


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[stage4-visual {timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize visual/chart explanation results without inserting into Milvus.")
    parser.add_argument("--artifact-dir", type=str, default="", help="Artifact dir containing vlm_results.jsonl")
    parser.add_argument("--result-file", type=str, default="", help="Direct path to vlm_results.jsonl")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for vector outputs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    return parser.parse_args()


def resolve_result_file(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.result_file:
        result_file = Path(args.result_file).resolve()
        return result_file.parent, result_file
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else settings.artifact_dir
    result_file = artifact_dir / DEFAULT_RESULT_FILE
    if not result_file.exists():
        raise FileNotFoundError(f"Missing result file: {result_file}")
    return artifact_dir, result_file


def load_visual_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            if payload.get("task_type") != "figure_or_image_understanding":
                continue
            if payload.get("status") != "success":
                continue
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_batch(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def stringify_list(values: list[Any]) -> str:
    cleaned = [str(item).strip() for item in values if str(item).strip()]
    return "；".join(cleaned)


def build_flow_text(flow_description: dict[str, Any]) -> str:
    if not isinstance(flow_description, dict):
        return ""
    parts: list[str] = []
    start = str(flow_description.get("start") or "").strip()
    steps = stringify_list(list(flow_description.get("steps") or []))
    decision_points = stringify_list(list(flow_description.get("decision_points") or []))
    end = str(flow_description.get("end") or "").strip()
    if start:
        parts.append(f"结构起点：{start}")
    if steps:
        parts.append(f"结构步骤：{steps}")
    if decision_points:
        parts.append(f"判断节点：{decision_points}")
    if end:
        parts.append(f"结构终点：{end}")
    return "\n".join(parts)


def build_chart_text(chart_analysis: dict[str, Any]) -> str:
    if not isinstance(chart_analysis, dict):
        return ""
    parts: list[str] = []
    chart_type = str(chart_analysis.get("chart_type") or "").strip()
    x_axis = str(chart_analysis.get("x_axis") or "").strip()
    y_axis = str(chart_analysis.get("y_axis") or "").strip()
    series = stringify_list(list(chart_analysis.get("series") or []))
    trend_summary = str(chart_analysis.get("trend_summary") or "").strip()
    max_points = stringify_list(list(chart_analysis.get("max_points") or []))
    min_points = stringify_list(list(chart_analysis.get("min_points") or []))
    turning_points = stringify_list(list(chart_analysis.get("turning_points") or []))
    comparison_points = stringify_list(list(chart_analysis.get("comparison_points") or []))
    if chart_type:
        parts.append(f"图表类型：{chart_type}")
    if x_axis:
        parts.append(f"横轴：{x_axis}")
    if y_axis:
        parts.append(f"纵轴：{y_axis}")
    if series:
        parts.append(f"数据系列：{series}")
    if trend_summary:
        parts.append(f"趋势总结：{trend_summary}")
    if max_points:
        parts.append(f"最大值：{max_points}")
    if min_points:
        parts.append(f"最小值：{min_points}")
    if turning_points:
        parts.append(f"拐点：{turning_points}")
    if comparison_points:
        parts.append(f"对比结论：{comparison_points}")
    return "\n".join(parts)


def build_group_text(groups: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            continue
        group_name = str(group.get("group_name") or "").strip() or f"组{index}"
        region_ids = stringify_list(list(group.get("region_ids") or []))
        summary = str(group.get("summary") or "").strip()
        same_object = "是" if bool(group.get("is_same_object")) else "否"
        parts = [f"{group_name}"]
        if region_ids:
            parts.append(f"区域ID：{region_ids}")
        parts.append(f"同一对象：{same_object}")
        if summary:
            parts.append(f"摘要：{summary}")
        lines.append("；".join(parts))
    return "\n".join(lines)


def build_visual_search_text(item: dict[str, Any]) -> str:
    structured = item.get("structured_content") or {}
    if not isinstance(structured, dict):
        structured = {}

    references = item.get("references") or []
    first_reference = references[0] if isinstance(references, list) and references else {}
    doc_name = str((first_reference or {}).get("source_pdf") or "").strip()
    source_pages = item.get("source_pages") or []
    page_number = source_pages[0] if source_pages else (first_reference or {}).get("page_index") or ""

    parts: list[str] = [
        f"对象类型：{str(structured.get('visual_type') or '').strip()}",
        f"文档名称：{doc_name}",
        f"页码：第 {page_number} 页",
        f"对象ID：{str(item.get('final_object_id') or '').strip()}",
        f"摘要：{str(structured.get('summary') or '').strip()}",
        f"结论：{str(item.get('content') or '').strip()}",
        f"详细描述：{str(structured.get('detailed_description') or '').strip()}",
    ]

    group_text = build_group_text(list(structured.get("groups") or []))
    if group_text:
        parts.append(f"对象分组：\n{group_text}")

    flow_text = build_flow_text(structured.get("flow_description") or {})
    if flow_text:
        parts.append(flow_text)

    chart_text = build_chart_text(structured.get("chart_analysis") or {})
    if chart_text:
        parts.append(chart_text)

    labels = stringify_list(list(structured.get("labels") or []))
    if labels:
        parts.append(f"标签：{labels}")
    numbers = stringify_list(list(structured.get("numbers") or []))
    if numbers:
        parts.append(f"数字信息：{numbers}")
    relations = stringify_list(list(structured.get("relations") or []))
    if relations:
        parts.append(f"关系信息：{relations}")
    observations = stringify_list(list(structured.get("key_observations") or []))
    if observations:
        parts.append(f"关键观察：{observations}")
    notes = stringify_list(list(structured.get("notes") or []))
    if notes:
        parts.append(f"备注：{notes}")

    return "\n".join(part for part in parts if part and not part.endswith("：")).strip()


def build_visual_vector_record(item: dict[str, Any], embedding: list[float], offset: int) -> dict[str, Any]:
    structured = item.get("structured_content") or {}
    if not isinstance(structured, dict):
        structured = {}
    references = item.get("references") or []
    first_reference = references[0] if isinstance(references, list) and references else {}
    source_pages = item.get("source_pages") or []
    source_region_ids = item.get("source_region_ids") or []
    return {
        "visual_id": str(item.get("final_object_id") or ""),
        "task_id": str(item.get("task_id") or ""),
        "page_number": int(source_pages[0]) if source_pages else int((first_reference or {}).get("page_index") or 0),
        "source_pages": list(source_pages),
        "source_region_ids": list(source_region_ids),
        "visual_type": str(structured.get("visual_type") or "").strip(),
        "summary_text": str(item.get("content") or "").strip(),
        "search_text": build_visual_search_text(item),
        "marker_id": str(item.get("final_object_id") or "").strip(),
        "crop_path": str((first_reference or {}).get("crop_path") or "").strip(),
        "raw_response_path": str(item.get("raw_response_path") or "").strip(),
        "minio_path": "",
        "embedding_offset": offset,
        "embedding_dim": len(embedding),
        "prompt_digest": str(item.get("prompt_digest") or ""),
        "created_at": str(item.get("created_at") or ""),
    }


def main() -> None:
    args = parse_args()
    artifact_dir, result_file = resolve_result_file(args)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (artifact_dir / DEFAULT_OUTPUT_DIRNAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_visual_results(result_file)
    if not results:
        raise ValueError(f"No successful visual results found in {result_file}")

    embedder = EmbeddingService(settings.model_dir / "embedding", configured_path=settings.embedding_model_path)
    batch_size = max(1, int(args.batch_size))

    log(f"result_file={result_file}")
    log(f"output_dir={output_dir}")
    log(f"visual_count={len(results)} batch_size={batch_size}")
    log(f"embedding_backend={embedder.backend} dimension={embedder.dimension}")

    search_text_rows: list[dict[str, Any]] = []
    for item in results:
        search_text_rows.append(
            {
                "visual_id": str(item.get("final_object_id") or ""),
                "task_id": str(item.get("task_id") or ""),
                "search_text": build_visual_search_text(item),
            }
        )

    vectors: list[list[float]] = []
    vector_index_rows: list[dict[str, Any]] = []
    batches = chunk_batch(results, batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        texts = [build_visual_search_text(item) for item in batch]
        embeddings = embedder.embed_texts(texts)
        vectors.extend(embeddings)
        for item, embedding in zip(batch, embeddings):
            vector_index_rows.append(build_visual_vector_record(item, embedding, len(vector_index_rows)))
        log(f"embedded_batch {batch_index}/{len(batches)} size={len(batch)} total_vectors={len(vectors)}")

    matrix = np.asarray(vectors, dtype=np.float32)
    np.save(output_dir / "visual_embeddings.npy", matrix)
    write_jsonl(output_dir / "visual_vector_index.jsonl", vector_index_rows)
    write_jsonl(output_dir / "visual_search_texts.jsonl", search_text_rows)
    write_json(
        output_dir / "visual_vector_manifest.json",
        {
            "source_result_file": str(result_file),
            "output_dir": str(output_dir),
            "visual_count": len(results),
            "vector_count": int(matrix.shape[0]),
            "embedding_dimension": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            "embedding_backend": embedder.backend,
            "embedding_model_path": settings.embedding_model_path,
            "batch_size": batch_size,
            "generated_files": [
                "visual_embeddings.npy",
                "visual_vector_index.jsonl",
                "visual_search_texts.jsonl",
                "visual_vector_manifest.json",
            ],
        },
    )

    log(f"vector_count={matrix.shape[0]}")
    log(f"embedding_dimension={matrix.shape[1] if matrix.ndim == 2 else 0}")
    log(f"manifest={output_dir / 'visual_vector_manifest.json'}")


if __name__ == "__main__":
    main()
