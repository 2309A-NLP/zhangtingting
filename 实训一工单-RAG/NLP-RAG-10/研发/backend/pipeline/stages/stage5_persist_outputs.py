from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("NLP_RAG_PROJECT_ROOT") or Path(__file__).resolve().parents[3])
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.services.rag_pipeline.upload_registry import sanitize_collection_component

from backend.pipeline.stages.stage5_persist_outputs._writers import (
    MilvusCollectionWriter,
    MongoTableWriter,
    MinioUploader,
)


TEXT_VECTOR_DIRNAME = "stage4_vectorized_chunks"
VISUAL_VECTOR_DIRNAME = "stage4_vectorized_visuals"
STAGE3_DIRNAME = "stage3_text_chunking"
DEFAULT_OUTPUT_NAME = "stage5_persist_manifest.json"
TABLE_TASK_TYPES = {"single_table_understanding", "cross_page_table_merge"}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[stage5 {timestamp}] {message}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist stage2/3/4 outputs into Milvus, MongoDB and MinIO.")
    parser.add_argument("--artifact-dir", type=str, default="", help="Artifact dir containing stage2/3/4 outputs")
    parser.add_argument("--output-manifest", type=str, default="", help="Path to write stage5 persistence manifest")
    parser.add_argument("--doc-name", type=str, default="", help="Logical document name for persisted records")
    parser.add_argument("--parse-version", type=str, default="parse4", help="Parse version tag")
    parser.add_argument("--ingest-run-id", type=str, default="", help="Ingest run id. Default: UTC timestamp")
    parser.add_argument("--text-collection", type=str, default="", help="Milvus text collection name")
    parser.add_argument("--visual-collection", type=str, default="", help="Milvus visual collection name")
    parser.add_argument("--mongo-collection", type=str, default="", help="MongoDB table collection name")
    parser.add_argument("--visual-bucket", type=str, default="", help="MinIO bucket for visual crops")
    parser.add_argument("--artifact-bucket", type=str, default="", help="MinIO bucket for artifacts")
    parser.add_argument("--skip-text", action="store_true", help="Skip persisting text vectors to Milvus")
    parser.add_argument("--skip-visual", action="store_true", help="Skip persisting visual vectors to Milvus")
    parser.add_argument("--skip-tables", action="store_true", help="Skip persisting tables to MongoDB")
    parser.add_argument("--skip-visual-upload", action="store_true", help="Skip uploading visual crops to MinIO")
    parser.add_argument("--skip-artifact-upload", action="store_true", help="Skip uploading artifacts to MinIO")
    parser.add_argument("--recreate-text-collection", action="store_true", help="Drop and recreate text collection")
    parser.add_argument("--recreate-visual-collection", action="store_true", help="Drop and recreate visual collection")
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_artifact_dir(raw: str) -> Path:
    if raw:
        path = Path(raw).resolve()
    else:
        path = settings.artifact_dir
    if not path.exists():
        raise FileNotFoundError(f"Artifact dir not found: {path}")
    return path


def resolve_doc_name(args: argparse.Namespace, artifact_dir: Path) -> str:
    if args.doc_name:
        return args.doc_name
    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        pdf_path = str(manifest.get("pdf_path") or "").strip()
        if pdf_path:
            return Path(pdf_path).stem
    if settings.pdf_path.exists():
        return settings.pdf_path.stem
    return artifact_dir.name


def safe_json_dumps(payload: Any, limit: int = 32000) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def normalize_key_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"
def build_text_rows(
    artifact_dir: Path,
    doc_name: str,
    parse_version: str,
    ingest_run_id: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    chunk_file = artifact_dir / STAGE3_DIRNAME / "text_chunks.jsonl"
    vector_index_file = artifact_dir / TEXT_VECTOR_DIRNAME / "chunk_vector_index.jsonl"
    vector_file = artifact_dir / TEXT_VECTOR_DIRNAME / "chunk_embeddings.npy"
    if not chunk_file.exists() or not vector_index_file.exists() or not vector_file.exists():
        raise FileNotFoundError("Missing stage3/stage4 text vector outputs")

    chunk_rows = load_jsonl(chunk_file)
    vector_index_rows = load_jsonl(vector_index_file)
    vector_matrix = np.load(vector_file)
    chunk_map = {str(item.get("chunk_id") or ""): item for item in chunk_rows}

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for item in vector_index_rows:
        chunk_id = str(item.get("chunk_id") or "")
        chunk_payload = chunk_map.get(chunk_id, {})
        offset = int(item.get("embedding_offset") or 0)
        source_pages = list(chunk_payload.get("source_pages") or item.get("source_pages") or [])
        page_start = int(min(source_pages)) if source_pages else 0
        page_end = int(max(source_pages)) if source_pages else 0
        row = {
            "chunk_id": chunk_id,
            "doc_name": doc_name,
            "parse_version": parse_version,
            "ingest_run_id": ingest_run_id,
            "page_start": page_start,
            "page_end": page_end,
            "source_page_count": int(chunk_payload.get("source_page_count") or item.get("source_page_count") or 0),
            "marker_count": int(chunk_payload.get("marker_count") or item.get("marker_count") or 0),
            "source_pages_json": safe_json_dumps(source_pages, limit=2048),
            "markers_json": safe_json_dumps(list(chunk_payload.get("markers") or item.get("markers") or []), limit=8192),
            "text": str(chunk_payload.get("text") or ""),
            "metadata_json": safe_json_dumps(chunk_payload or item),
        }
        rows.append(row)
        vectors.append(np.asarray(vector_matrix[offset], dtype=np.float32))
    return rows, np.asarray(vectors, dtype=np.float32)


def build_visual_upload_map(
    artifact_dir: Path,
    doc_name: str,
    parse_version: str,
    ingest_run_id: str,
    uploader: MinioUploader | None,
    bucket: str,
    skip_upload: bool,
) -> dict[str, str]:
    vector_index_file = artifact_dir / VISUAL_VECTOR_DIRNAME / "visual_vector_index.jsonl"
    if not vector_index_file.exists():
        raise FileNotFoundError("Missing stage4 visual vector index output")
    rows = load_jsonl(vector_index_file)
    upload_map: dict[str, str] = {}
    if uploader is None or skip_upload:
        for row in rows:
            visual_id = str(row.get("visual_id") or "")
            crop_path = Path(str(row.get("crop_path") or ""))
            if crop_path.exists():
                relative = crop_path.relative_to(artifact_dir) if artifact_dir in crop_path.parents else Path("visuals") / crop_path.name
                key = "/".join(
                    [
                        normalize_key_component(parse_version),
                        normalize_key_component(ingest_run_id),
                        "visuals",
                        normalize_key_component(doc_name),
                        relative.as_posix(),
                    ]
                )
                upload_map[visual_id] = f"s3://{bucket}/{key}"
        return upload_map

    uploader.ensure_bucket(bucket)
    for row in rows:
        visual_id = str(row.get("visual_id") or "")
        crop_path = Path(str(row.get("crop_path") or ""))
        if not crop_path.exists():
            upload_map[visual_id] = ""
            continue
        relative = crop_path.relative_to(artifact_dir) if artifact_dir in crop_path.parents else Path("visuals") / crop_path.name
        key = "/".join(
            [
                normalize_key_component(parse_version),
                normalize_key_component(ingest_run_id),
                "visuals",
                normalize_key_component(doc_name),
                relative.as_posix(),
            ]
        )
        upload_map[visual_id] = uploader.upload_file(bucket, key, crop_path)
    return upload_map


def build_visual_rows(
    artifact_dir: Path,
    doc_name: str,
    parse_version: str,
    ingest_run_id: str,
    minio_map: dict[str, str],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    vector_index_file = artifact_dir / VISUAL_VECTOR_DIRNAME / "visual_vector_index.jsonl"
    vector_file = artifact_dir / VISUAL_VECTOR_DIRNAME / "visual_embeddings.npy"
    if not vector_index_file.exists() or not vector_file.exists():
        raise FileNotFoundError("Missing stage4 visual vector outputs")

    vector_index_rows = load_jsonl(vector_index_file)
    vector_matrix = np.load(vector_file)
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for item in vector_index_rows:
        offset = int(item.get("embedding_offset") or 0)
        visual_id = str(item.get("visual_id") or "")
        row = {
            "visual_id": visual_id,
            "task_id": str(item.get("task_id") or ""),
            "doc_name": doc_name,
            "parse_version": parse_version,
            "ingest_run_id": ingest_run_id,
            "page_number": int(item.get("page_number") or 0),
            "visual_type": str(item.get("visual_type") or ""),
            "marker_id": str(item.get("marker_id") or visual_id),
            "source_pages_json": safe_json_dumps(list(item.get("source_pages") or []), limit=2048),
            "source_region_ids_json": safe_json_dumps(list(item.get("source_region_ids") or []), limit=8192),
            "summary_text": str(item.get("summary_text") or ""),
            "search_text": str(item.get("search_text") or ""),
            "minio_path": minio_map.get(visual_id, str(item.get("minio_path") or "")),
            "metadata_json": safe_json_dumps(item),
        }
        rows.append(row)
        vectors.append(np.asarray(vector_matrix[offset], dtype=np.float32))
    return rows, np.asarray(vectors, dtype=np.float32)


def load_table_vlm_results(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    result_candidates = [
        artifact_dir / "vlm_results.jsonl",
        artifact_dir / "stage2_precise_extraction" / "vlm_results.jsonl",
    ]
    result_file = next((candidate for candidate in result_candidates if candidate.exists()), result_candidates[0])
    if not result_file.exists():
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for item in load_jsonl(result_file):
        task_type = str(item.get("task_type") or "")
        if task_type not in TABLE_TASK_TYPES:
            continue
        final_object_id = str(item.get("final_object_id") or "")
        if final_object_id:
            mapping[final_object_id] = item
    return mapping


def resolve_table_file(artifact_dir: Path) -> Path:
    candidates = [
        artifact_dir / "tables_raw.jsonl",
        artifact_dir / "stage2_precise_extraction" / "tables_raw.jsonl",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def build_table_documents(
    artifact_dir: Path,
    doc_name: str,
    parse_version: str,
    ingest_run_id: str,
) -> list[dict[str, Any]]:
    table_file = resolve_table_file(artifact_dir)
    if not table_file.exists():
        raise FileNotFoundError(f"Missing table file: {table_file}")
    table_rows = load_jsonl(table_file)
    vlm_by_object = load_table_vlm_results(artifact_dir)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in table_rows:
        key = str(row.get("final_object_id") or row.get("table_id") or "")
        grouped.setdefault(key, []).append(row)

    documents: list[dict[str, Any]] = []
    for final_object_id, segments in grouped.items():
        ordered_segments = sorted(
            segments,
            key=lambda item: (
                int(item.get("page_index") or 0),
                str(item.get("source_region_id") or ""),
            ),
        )
        source_pages = sorted({int(item.get("page_index") or 0) for item in ordered_segments if int(item.get("page_index") or 0) > 0})
        source_region_ids = [str(item.get("source_region_id") or "") for item in ordered_segments]
        local_table_result = {
            "final_object_id": final_object_id,
            "segment_count": len(ordered_segments),
            "is_cross_page": any(bool(item.get("is_cross_page_member")) for item in ordered_segments) or len(source_pages) > 1,
            "source_pages": source_pages,
            "source_region_ids": source_region_ids,
            "segments": ordered_segments,
        }
        vlm_result = vlm_by_object.get(final_object_id)
        content_hash = hashlib.sha256(
            json.dumps(
                {
                    "local_table_result": local_table_result,
                    "vlm_result": vlm_result,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        table_id = final_object_id or str(ordered_segments[0].get("table_id") or "")
        documents.append(
            {
                "table_id": table_id,
                "final_object_id": final_object_id,
                "doc_name": doc_name,
                "page_start": min(source_pages) if source_pages else 0,
                "page_end": max(source_pages) if source_pages else 0,
                "source_pages": source_pages,
                "source_region_ids": source_region_ids,
                "marker_id": final_object_id or table_id,
                "table_type": str(ordered_segments[0].get("sub_type") or ""),
                "extraction_backend": "|".join(
                    sorted({str(item.get("extraction_backend") or "") for item in ordered_segments if str(item.get("extraction_backend") or "")})
                ),
                "local_table_result": local_table_result,
                "vlm_result": vlm_result,
                "content_hash": content_hash,
                "parse_version": parse_version,
                "ingest_run_id": ingest_run_id,
                "created_at": utc_now(),
            }
        )
    documents.sort(key=lambda item: (int(item.get("page_start") or 0), str(item.get("table_id") or "")))
    return documents


def upload_artifacts(
    artifact_dir: Path,
    uploader: MinioUploader | None,
    bucket: str,
    parse_version: str,
    ingest_run_id: str,
    doc_name: str,
    skip_upload: bool,
) -> list[dict[str, str]]:
    uploaded: list[dict[str, str]] = []
    if uploader is None:
        return uploaded
    uploader.ensure_bucket(bucket)
    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(artifact_dir)
        key = "/".join(
            [
                normalize_key_component(parse_version),
                normalize_key_component(ingest_run_id),
                "artifacts",
                normalize_key_component(doc_name),
                relative.as_posix(),
            ]
        )
        if skip_upload:
            uri = f"s3://{bucket}/{key}"
        else:
            uri = uploader.upload_file(bucket, key, path)
        uploaded.append({"path": str(relative), "minio_uri": uri})
    return uploaded


def main() -> None:
    args = parse_args()
    artifact_dir = resolve_artifact_dir(args.artifact_dir)
    doc_name = resolve_doc_name(args, artifact_dir)
    parse_version = str(args.parse_version or "parse4")
    ingest_run_id = str(args.ingest_run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    text_collection = sanitize_collection_component(
        str(args.text_collection or settings.text_vector_collection_name),
        fallback=settings.text_vector_collection_name,
    )
    visual_collection = sanitize_collection_component(
        str(args.visual_collection or settings.visual_vector_collection_name),
        fallback=settings.visual_vector_collection_name,
    )
    mongo_collection = str(args.mongo_collection or settings.mongodb_table_collection)
    visual_bucket = str(args.visual_bucket or settings.minio_bucket_visuals)
    artifact_bucket = str(args.artifact_bucket or settings.minio_bucket_artifacts)
    output_manifest = Path(args.output_manifest).resolve() if args.output_manifest else (artifact_dir / DEFAULT_OUTPUT_NAME)

    log(f"artifact_dir={artifact_dir}")
    log(f"doc_name={doc_name}")
    log(f"parse_version={parse_version}")
    log(f"ingest_run_id={ingest_run_id}")

    summary: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "doc_name": doc_name,
        "parse_version": parse_version,
        "ingest_run_id": ingest_run_id,
        "started_at": utc_now(),
        "text_collection": text_collection,
        "visual_collection": visual_collection,
        "mongo_collection": mongo_collection,
        "visual_bucket": visual_bucket,
        "artifact_bucket": artifact_bucket,
        "steps": {},
    }

    uploader: MinioUploader | None = None
    if not args.skip_visual_upload or not args.skip_artifact_upload:
        uploader = MinioUploader(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

    visual_minio_map: dict[str, str] = {}
    if not args.skip_visual_upload:
        started = time.perf_counter()
        visual_minio_map = build_visual_upload_map(
            artifact_dir=artifact_dir,
            doc_name=doc_name,
            parse_version=parse_version,
            ingest_run_id=ingest_run_id,
            uploader=uploader,
            bucket=visual_bucket,
            skip_upload=False,
        )
        summary["steps"]["visual_upload"] = {
            "count": len([value for value in visual_minio_map.values() if value]),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        log(f"visual_upload_count={summary['steps']['visual_upload']['count']}")
    elif uploader is not None:
        visual_minio_map = build_visual_upload_map(
            artifact_dir=artifact_dir,
            doc_name=doc_name,
            parse_version=parse_version,
            ingest_run_id=ingest_run_id,
            uploader=None,
            bucket=visual_bucket,
            skip_upload=True,
        )

    milvus_writer: MilvusCollectionWriter | None = None
    try:
        if not args.skip_text or not args.skip_visual:
            milvus_writer = MilvusCollectionWriter(settings.milvus_uri)

        if not args.skip_text:
            started = time.perf_counter()
            text_rows, text_vectors = build_text_rows(artifact_dir, doc_name, parse_version, ingest_run_id)
            text_collection_ref = milvus_writer.ensure_text_collection(
                text_collection,
                dim=int(text_vectors.shape[1]) if text_vectors.ndim == 2 else 0,
                recreate=bool(args.recreate_text_collection),
            )
            text_count = milvus_writer.insert_text_rows(text_collection_ref, text_rows, text_vectors)
            summary["steps"]["text_milvus"] = {
                "count": text_count,
                "dimension": int(text_vectors.shape[1]) if text_vectors.ndim == 2 else 0,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            log(f"text_milvus_count={text_count}")

        if not args.skip_visual:
            started = time.perf_counter()
            visual_rows, visual_vectors = build_visual_rows(
                artifact_dir,
                doc_name,
                parse_version,
                ingest_run_id,
                visual_minio_map,
            )
            visual_collection_ref = milvus_writer.ensure_visual_collection(
                visual_collection,
                dim=int(visual_vectors.shape[1]) if visual_vectors.ndim == 2 else 0,
                recreate=bool(args.recreate_visual_collection),
            )
            visual_count = milvus_writer.insert_visual_rows(visual_collection_ref, visual_rows, visual_vectors)
            updated_visual_index_path = artifact_dir / VISUAL_VECTOR_DIRNAME / "visual_vector_index.persisted.jsonl"
            write_jsonl(updated_visual_index_path, visual_rows)
            summary["steps"]["visual_milvus"] = {
                "count": visual_count,
                "dimension": int(visual_vectors.shape[1]) if visual_vectors.ndim == 2 else 0,
                "updated_index": str(updated_visual_index_path),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            log(f"visual_milvus_count={visual_count}")

        if not args.skip_tables:
            started = time.perf_counter()
            mongo_writer = MongoTableWriter(
                uri=settings.mongodb_uri,
                db_name=settings.mongodb_db_name,
                collection_name=mongo_collection,
            )
            try:
                table_documents = build_table_documents(artifact_dir, doc_name, parse_version, ingest_run_id)
                table_count = mongo_writer.replace_documents(table_documents)
            finally:
                mongo_writer.close()
            table_preview_path = artifact_dir / "stage5_table_documents_preview.jsonl"
            write_jsonl(table_preview_path, table_documents)
            summary["steps"]["tables_mongodb"] = {
                "count": table_count,
                "preview_path": str(table_preview_path),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            log(f"tables_mongodb_count={table_count}")

        if not args.skip_artifact_upload:
            started = time.perf_counter()
            uploaded_artifacts = upload_artifacts(
                artifact_dir=artifact_dir,
                uploader=uploader,
                bucket=artifact_bucket,
                parse_version=parse_version,
                ingest_run_id=ingest_run_id,
                doc_name=doc_name,
                skip_upload=False,
            )
            artifact_index_path = artifact_dir / "stage5_artifact_minio_index.jsonl"
            write_jsonl(artifact_index_path, uploaded_artifacts)
            summary["steps"]["artifact_upload"] = {
                "count": len(uploaded_artifacts),
                "index_path": str(artifact_index_path),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            log(f"artifact_upload_count={len(uploaded_artifacts)}")

    finally:
        if milvus_writer is not None:
            milvus_writer.close()

    summary["finished_at"] = utc_now()
    summary["elapsed_seconds"] = round(
        max(
            0.0,
            datetime.fromisoformat(summary["finished_at"]).timestamp()
            - datetime.fromisoformat(summary["started_at"]).timestamp(),
        ),
        3,
    )
    write_json(output_manifest, summary)
    if uploader is not None and not args.skip_artifact_upload:
        uploader.ensure_bucket(artifact_bucket)
        manifest_key = "/".join(
            [
                normalize_key_component(parse_version),
                normalize_key_component(ingest_run_id),
                "artifacts",
                normalize_key_component(doc_name),
                output_manifest.name,
            ]
        )
        manifest_uri = uploader.upload_file(artifact_bucket, manifest_key, output_manifest)
        summary["steps"].setdefault("artifact_upload", {})
        summary["steps"]["artifact_upload"]["stage5_manifest_uri"] = manifest_uri
        write_json(output_manifest, summary)
    log(f"manifest={output_manifest}")


if __name__ == "__main__":
    main()
