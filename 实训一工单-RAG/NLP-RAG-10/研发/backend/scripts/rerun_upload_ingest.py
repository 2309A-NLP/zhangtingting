from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.pipeline.stages import stage5_persist_outputs as stage5
from backend.services.rag_pipeline.upload_registry import (
    build_uploaded_collection_names,
    file_sha1,
    sanitize_collection_component,
    upsert_upload_registry_entry,
    upload_manifest_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit upload artifacts and persist them into storage backends.")
    parser.add_argument("--artifact-dir", required=True, help="Artifact dir containing stage2/3/4 outputs")
    parser.add_argument("--filename", default="", help="Original uploaded PDF filename")
    parser.add_argument("--upload-id", default="", help="Existing upload id; inferred from artifact dir when omitted")
    parser.add_argument("--doc-name", default="", help="Logical document name override")
    parser.add_argument("--parse-version", default="parse4", help="Parse version tag")
    parser.add_argument("--ingest-run-id", default="", help="Ingest run id override")
    parser.add_argument("--text-collection", default="", help="Milvus text collection")
    parser.add_argument("--visual-collection", default="", help="Milvus visual collection")
    parser.add_argument("--mongo-collection", default="", help="Mongo collection for tables")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--skip-visual-upload", action="store_true")
    parser.add_argument("--skip-artifact-upload", action="store_true")
    parser.add_argument("--recreate-text-collection", action="store_true")
    parser.add_argument("--recreate-visual-collection", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def infer_upload_id(artifact_dir: Path, explicit: str) -> str:
    if explicit:
        return explicit
    if artifact_dir.parent.name == "artifacts" and artifact_dir.parent.parent.name:
        return artifact_dir.parent.parent.name
    if artifact_dir.name == "artifacts" and artifact_dir.parent.name:
        return artifact_dir.parent.name
    return sanitize_collection_component(artifact_dir.parent.name or artifact_dir.name, fallback="uploaded")


def infer_filename(args: argparse.Namespace, artifact_dir: Path, upload_id: str) -> str:
    if args.filename:
        return Path(args.filename).name
    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        filename = str(manifest.get("filename") or "").strip()
        if filename:
            return filename
        pdf_path = str(manifest.get("pdf_path") or "").strip()
        if pdf_path:
            return Path(pdf_path).name
    upload_manifest = upload_manifest_path(upload_id)
    if upload_manifest.exists():
        manifest = load_json(upload_manifest)
        filename = str(manifest.get("filename") or "").strip()
        if filename:
            return filename
    source_dir = artifact_dir.parent.parent / "source"
    if source_dir.exists():
        pdfs = sorted(source_dir.glob("*.pdf"), key=lambda path: path.stat().st_mtime, reverse=True)
        if pdfs:
            return pdfs[0].name
    return f"{upload_id}.pdf"


def summarize_presence(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, int]]:
    total = len(rows)
    summary: dict[str, dict[str, int]] = {}
    for key in keys:
        present = 0
        non_empty = 0
        for row in rows:
            if key in row:
                present += 1
                value = row.get(key)
                if value not in (None, "", [], {}, ()):  # noqa: PLC1901
                    non_empty += 1
        summary[key] = {"present": present, "non_empty": non_empty, "total": total}
    return summary


def build_audit_report(artifact_dir: Path, doc_name: str, parse_version: str, ingest_run_id: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "doc_name": doc_name,
        "parse_version": parse_version,
        "ingest_run_id": ingest_run_id,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "files": {},
        "metadata_coverage": {},
        "warnings": [],
    }

    chunk_file = artifact_dir / stage5.STAGE3_DIRNAME / "text_chunks.jsonl"
    text_index_file = artifact_dir / stage5.TEXT_VECTOR_DIRNAME / "chunk_vector_index.jsonl"
    visual_index_file = artifact_dir / stage5.VISUAL_VECTOR_DIRNAME / "visual_vector_index.jsonl"
    stage5_manifest = artifact_dir / stage5.DEFAULT_OUTPUT_NAME
    table_file = stage5.resolve_table_file(artifact_dir)

    required_files = {
        "text_chunks": chunk_file,
        "text_vector_index": text_index_file,
        "text_embeddings": artifact_dir / stage5.TEXT_VECTOR_DIRNAME / "chunk_embeddings.npy",
        "visual_vector_index": visual_index_file,
        "visual_embeddings": artifact_dir / stage5.VISUAL_VECTOR_DIRNAME / "visual_embeddings.npy",
        "table_results": table_file,
        "stage5_manifest": stage5_manifest,
    }
    for label, path in required_files.items():
        report["files"][label] = {"path": str(path), "exists": path.exists()}

    if chunk_file.exists():
        chunk_rows = load_jsonl(chunk_file)
        report["metadata_coverage"]["text_chunks"] = summarize_presence(
            chunk_rows,
            [
                "chunk_id",
                "text",
                "source_pages",
                "markers",
                "section_title",
                "content_tags",
                "page_type",
                "sub_type",
                "context_links",
                "structured_facts",
            ],
        )
        if any(not row.get("text") for row in chunk_rows):
            report["warnings"].append("Some text chunks have empty text.")

    if text_index_file.exists():
        text_index_rows = load_jsonl(text_index_file)
        report["metadata_coverage"]["text_vector_index"] = summarize_presence(
            text_index_rows,
            ["chunk_id", "embedding_offset", "source_pages", "markers", "source_page_count", "marker_count"],
        )

    if visual_index_file.exists():
        visual_rows = load_jsonl(visual_index_file)
        report["metadata_coverage"]["visual_vector_index"] = summarize_presence(
            visual_rows,
            [
                "visual_id",
                "task_id",
                "page_number",
                "visual_type",
                "marker_id",
                "source_pages",
                "source_region_ids",
                "summary_text",
                "search_text",
                "crop_path",
                "minio_path",
            ],
        )
        if any(not row.get("search_text") for row in visual_rows):
            report["warnings"].append("Some visual rows have empty search_text.")

    if table_file.exists():
        table_rows = load_jsonl(table_file)
        report["metadata_coverage"]["table_results"] = summarize_presence(
            table_rows,
            [
                "final_object_id",
                "table_id",
                "page_number",
                "source_pages",
                "source_region_ids",
                "sub_type",
                "local_table_result",
                "vlm_result",
                "extraction_backend",
            ],
        )
        if not table_rows:
            report["warnings"].append("Table result file exists but has no rows.")

    return report


def persist_outputs(args: argparse.Namespace, artifact_dir: Path, doc_name: str, parse_version: str, ingest_run_id: str) -> Path:
    output_manifest = artifact_dir / stage5.DEFAULT_OUTPUT_NAME
    text_collection = sanitize_collection_component(
        args.text_collection or build_uploaded_collection_names(args.upload_id, args.filename)[0],
        fallback=settings.text_vector_collection_name,
    )
    visual_collection = sanitize_collection_component(
        args.visual_collection or build_uploaded_collection_names(args.upload_id, args.filename)[1],
        fallback=settings.visual_vector_collection_name,
    )
    mongo_collection = sanitize_collection_component(
        args.mongo_collection or f"pdf_tables_{args.upload_id}",
        fallback=settings.mongodb_table_collection,
    )

    argv = [
        "stage5_persist_outputs.py",
        "--artifact-dir",
        str(artifact_dir),
        "--output-manifest",
        str(output_manifest),
        "--doc-name",
        doc_name,
        "--parse-version",
        parse_version,
        "--ingest-run-id",
        ingest_run_id,
        "--text-collection",
        text_collection,
        "--visual-collection",
        visual_collection,
        "--mongo-collection",
        mongo_collection,
    ]
    if args.skip_text:
        argv.append("--skip-text")
    if args.skip_visual:
        argv.append("--skip-visual")
    if args.skip_tables:
        argv.append("--skip-tables")
    if args.skip_visual_upload:
        argv.append("--skip-visual-upload")
    if args.skip_artifact_upload:
        argv.append("--skip-artifact-upload")
    if args.recreate_text_collection:
        argv.append("--recreate-text-collection")
    if args.recreate_visual_collection:
        argv.append("--recreate-visual-collection")

    previous_argv = os.sys.argv[:]
    try:
        os.sys.argv = argv
        stage5.main()
    finally:
        os.sys.argv = previous_argv
    return output_manifest


def update_upload_records(
    upload_id: str,
    filename: str,
    artifact_dir: Path,
    output_manifest: Path,
    doc_name: str,
    parse_version: str,
    ingest_run_id: str,
) -> None:
    stage5_manifest = load_json(output_manifest)
    text_collection_name = str(stage5_manifest.get("text_collection") or "")
    visual_collection_name = str(stage5_manifest.get("visual_collection") or "")
    mongo_collection_name = str(stage5_manifest.get("mongo_collection") or "")
    total_inserted = sum(
        step.get("count", 0)
        for step in (stage5_manifest.get("steps") or {}).values()
        if isinstance(step, dict) and "count" in step
    )

    upload_manifest_payload = {
        "upload_id": upload_id,
        "filename": filename,
        "pdf_path": str((artifact_dir.parent.parent / "source" / filename).resolve()),
        "artifact_dir": str(artifact_dir),
        "doc_name": doc_name,
        "parse_version": parse_version,
        "ingest_run_id": ingest_run_id,
        "text_collection_name": text_collection_name,
        "visual_collection_name": visual_collection_name,
        "mongo_collection_name": mongo_collection_name,
        "stages": {
            "stage5": str(output_manifest),
        },
    }
    upload_manifest_path(upload_id).write_text(
        json.dumps(upload_manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    source_pdf = artifact_dir.parent.parent / "source" / filename
    registry_entry = {
        "upload_id": upload_id,
        "filename": filename,
        "stored_pdf_path": str(source_pdf),
        "artifact_dir": str(artifact_dir),
        "text_collection_name": text_collection_name,
        "visual_collection_name": visual_collection_name,
        "mongo_collection_name": mongo_collection_name,
        "chunks": int(total_inserted),
        "file_sha1": file_sha1(source_pdf) if source_pdf.exists() else "",
        "uploaded_at": int(time.time()),
        "status": "ready",
    }
    upsert_upload_registry_entry(registry_entry)


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact dir not found: {artifact_dir}")

    upload_id = infer_upload_id(artifact_dir, args.upload_id)
    args.upload_id = upload_id
    filename = infer_filename(args, artifact_dir, upload_id)
    doc_name = args.doc_name or Path(filename).stem or artifact_dir.name
    ingest_run_id = args.ingest_run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parse_version = str(args.parse_version or "parse4")

    audit_report = build_audit_report(artifact_dir, doc_name, parse_version, ingest_run_id)
    audit_path = artifact_dir / "stage5_ingest_audit.json"
    audit_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")

    output_manifest = persist_outputs(args, artifact_dir, doc_name, parse_version, ingest_run_id)
    update_upload_records(upload_id, filename, artifact_dir, output_manifest, doc_name, parse_version, ingest_run_id)

    print(json.dumps({
        "ok": True,
        "upload_id": upload_id,
        "filename": filename,
        "artifact_dir": str(artifact_dir),
        "audit_report": str(audit_path),
        "stage5_manifest": str(output_manifest),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
