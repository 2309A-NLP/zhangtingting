from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_SOURCES_DIR = ROOT_DIR / "data" / "raw_sources"
COLLECTED_DIR = ROOT_DIR / "data" / "collected"
REGISTRY_PATH = ROOT_DIR / "scripts" / "source_registry.json"

NON_WORD_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_doc_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def normalize_text(value: str) -> str:
    return MULTI_SPACE_PATTERN.sub(" ", value).strip()


def safe_slug(value: str, max_length: int = 80) -> str:
    normalized = normalize_text(value).replace(" ", "_")
    cleaned = NON_WORD_PATTERN.sub("_", normalized).strip("._-")
    return cleaned[:max_length] or "document"


def source_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def write_raw_payload(path: Path, payload: str | bytes) -> None:
    ensure_dir(path.parent)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")


def build_text_document(doc: dict[str, Any]) -> str:
    lines = [
        f"Title: {doc['title']}",
        f"Source Name: {doc['source_name']}",
        f"Source URL: {doc['source_url']}",
    ]
    if doc.get("published_at"):
        lines.append(f"Published At: {doc['published_at']}")
    if doc.get("tags"):
        lines.append(f"Tags: {', '.join(doc['tags'])}")
    lines.extend(["", "Content:", doc["content"].strip()])
    return "\n".join(lines).strip() + "\n"


def write_collected_document(role_id: str, doc: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = ensure_dir(COLLECTED_DIR / role_id)
    doc_id = doc["doc_id"]
    text_path = output_dir / f"{doc_id}.txt"
    meta_path = output_dir / f"{doc_id}.meta.json"
    text_path.write_text(build_text_document(doc), encoding="utf-8")
    meta_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return text_path, meta_path


def seen_source_url(role_id: str, url: str) -> bool:
    output_dir = COLLECTED_DIR / role_id
    if not output_dir.exists():
        return False
    for meta_path in output_dir.glob("*.meta.json"):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("source_url") == url:
            return True
    return False


def user_agent(contact_email: str) -> str:
    return f"rag-app-bot/1.0 (contact: {contact_email})"
