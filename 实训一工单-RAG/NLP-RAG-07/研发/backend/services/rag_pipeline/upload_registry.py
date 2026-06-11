from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from pymilvus import Collection, connections, utility

from backend.config import settings


UPLOADS_ROOT = settings.artifact_dir / "uploaded_documents"
UPLOAD_REGISTRY_PATH = UPLOADS_ROOT / "registry.json"


def ensure_uploads_root() -> Path:
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    return UPLOADS_ROOT


def slugify_filename(filename: str) -> str:
    stem = Path(filename or "uploaded").stem.strip() or "uploaded"
    normalized = re.sub(r"\s+", "_", stem)
    normalized = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return (normalized or "uploaded")[:48]


def sanitize_collection_component(value: str, fallback: str = "uploaded") -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    return (normalized or fallback)[:48]


def build_upload_id(filename: str) -> str:
    slug = slugify_filename(filename)
    token = uuid.uuid4().hex[:8]
    return f"{slug}_{token}"


def build_uploaded_collection_names(upload_id: str, filename: str = "") -> tuple[str, str]:
    base = sanitize_collection_component(slugify_filename(filename), fallback="uploaded") if filename else "uploaded"
    suffix = sanitize_collection_component(upload_id, fallback=uuid.uuid4().hex[:8])
    return f"pdf_text_{base}_{suffix}", f"pdf_visual_{base}_{suffix}"


def upload_root(upload_id: str) -> Path:
    return ensure_uploads_root() / upload_id


def upload_source_dir(upload_id: str) -> Path:
    return upload_root(upload_id) / "source"


def upload_artifact_dir(upload_id: str) -> Path:
    return upload_root(upload_id) / "artifacts"


def upload_pipeline_dir(upload_id: str) -> Path:
    return upload_artifact_dir(upload_id) / "pipeline"


def upload_stage_dir(upload_id: str, stage_name: str) -> Path:
    return upload_pipeline_dir(upload_id) / stage_name


def upload_pdf_intelligence_dir(upload_id: str) -> Path:
    return upload_artifact_dir(upload_id) / "pdf_intelligence"


def upload_manifest_path(upload_id: str) -> Path:
    return upload_root(upload_id) / "manifest.json"


def upload_parsed_pages_path(upload_id: str) -> Path:
    return upload_stage_dir(upload_id, "stage_0_parse") / "parsed_pages.json"


def upload_redacted_pages_path(upload_id: str) -> Path:
    return upload_stage_dir(upload_id, "stage_0_parse") / "parsed_pages_redacted.json"


def upload_enhanced_pages_path(upload_id: str) -> Path:
    return upload_stage_dir(upload_id, "stage_1_enhanced") / "enhanced_pages.json"


def upload_chunk_manifest_path(upload_id: str) -> Path:
    return upload_stage_dir(upload_id, "stage_2_chunking") / "chunk_manifest.json"


def upload_vlm_failure_path(upload_id: str) -> Path:
    return upload_stage_dir(upload_id, "stage_1_vlm") / "pdf_vlm_last_failure.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_upload_registry() -> List[Dict[str, Any]]:
    ensure_uploads_root()
    payload = _read_json(UPLOAD_REGISTRY_PATH, {"items": []})
    if isinstance(payload, dict):
        items = payload.get("items") or []
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def write_upload_registry(items: List[Dict[str, Any]]) -> None:
    ensure_uploads_root()
    UPLOAD_REGISTRY_PATH.write_text(
        json.dumps({"items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upsert_upload_registry_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = read_upload_registry()
    replaced = False
    for index, item in enumerate(items):
        if str(item.get("upload_id") or "") == str(entry.get("upload_id") or ""):
            items[index] = entry
            replaced = True
            break
    if not replaced:
        items.insert(0, entry)
    write_upload_registry(items)
    return items


def _infer_filename_from_upload_dir(upload_id: str) -> str:
    source_dir = upload_source_dir(upload_id)
    if source_dir.exists():
        pdf_files = sorted(source_dir.glob("*.pdf"), key=lambda path: path.stat().st_mtime, reverse=True)
        if pdf_files:
            return pdf_files[0].name
    manifest = _read_json(upload_manifest_path(upload_id), {})
    if isinstance(manifest, dict):
        filename = str(manifest.get("filename") or "").strip()
        if filename:
            return filename
    return upload_id


def discover_uploaded_documents_from_milvus() -> List[Dict[str, Any]]:
    alias = f"upload_registry_{uuid.uuid4().hex[:8]}"
    discovered: List[Dict[str, Any]] = []
    known_roots = [path for path in ensure_uploads_root().iterdir()] if ensure_uploads_root().exists() else []
    known_upload_ids = {path.name for path in known_roots if path.is_dir()}
    try:
        connections.connect(alias=alias, uri=settings.milvus_uri)
        collection_names = utility.list_collections(using=alias) or []
        text_collections = [name for name in collection_names if isinstance(name, str) and name.startswith("pdf_text_")]
        for collection_name in text_collections:
            matched_upload_id = ""
            for upload_id in known_upload_ids:
                if collection_name.endswith(sanitize_collection_component(upload_id, fallback=upload_id)):
                    matched_upload_id = upload_id
                    break
            if not matched_upload_id:
                continue
            chunks = 0
            try:
                collection = Collection(name=collection_name, using=alias)
                collection.load()
                chunks = int(collection.num_entities)
            except Exception:
                chunks = 0
            discovered.append(
                {
                    "upload_id": matched_upload_id,
                    "filename": _infer_filename_from_upload_dir(matched_upload_id),
                    "stored_pdf_path": "",
                    "artifact_dir": str(upload_artifact_dir(matched_upload_id)),
                    "text_collection_name": collection_name,
                    "visual_collection_name": collection_name.replace("pdf_text_", "pdf_visual_", 1),
                    "mongo_collection_name": "",
                    "chunks": chunks,
                    "file_sha1": "",
                    "uploaded_at": int(upload_root(matched_upload_id).stat().st_mtime) if upload_root(matched_upload_id).exists() else 0,
                    "status": "ready" if chunks > 0 else "processing",
                    "source": "milvus_fallback",
                }
            )
    except Exception:
        return []
    finally:
        try:
            connections.disconnect(alias=alias)
        except Exception:
            pass
    discovered.sort(key=lambda item: (int(item.get("uploaded_at") or 0), str(item.get("upload_id") or "")), reverse=True)
    return discovered


def merge_uploaded_documents(registry_items: List[Dict[str, Any]], discovered_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in discovered_items:
        upload_id = str(item.get("upload_id") or "")
        if upload_id:
            merged[upload_id] = dict(item)
    for item in registry_items:
        upload_id = str(item.get("upload_id") or "")
        if not upload_id:
            continue
        merged[upload_id] = {**merged.get(upload_id, {}), **item}
    items = list(merged.values())
    items.sort(key=lambda item: (int(item.get("uploaded_at") or 0), str(item.get("upload_id") or "")), reverse=True)
    return items


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
