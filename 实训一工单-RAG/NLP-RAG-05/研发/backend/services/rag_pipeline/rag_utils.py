# ������ţ��˹����� NLP-RAG-ͼ�����ݽ����������Ż�
from __future__ import annotations

"""RAG Pipeline utilities."""

import json
import re
from pathlib import Path
from typing import Any

from backend.config import settings


_MOJIBAKE_MARKERS = frozenset([
    "?", "?", "?", "??", "?", "???", "?",
    "???", "?", "???", "?", "?",
])


def looks_like_mojibake(text: str) -> bool:
    sample = (text or "").strip()
    if not sample:
        return False
    return sum(1 for m in _MOJIBAKE_MARKERS if m in sample) >= 4


def sanitize_vlm_context(page: dict[str, object]) -> tuple[str, str]:
    searchable_parts: list[str] = []
    raw_parts: list[str] = []
    blocks: list[dict[str, Any]] = page.get("text_dict", {}).get("blocks", [])
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = ""
            for span in line.get("spans", []):
                if not (span.get("flags", 0) & 2):
                    text += span.get("text", "")
            if text.strip():
                searchable_parts.append(text.strip())
                raw_parts.append(text)
    return "\n".join(searchable_parts), "\n".join(raw_parts)


def normalize_company_name(name: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9����()]", "", name).strip()


def get_company_aliases(name: str) -> list[str]:
    normalized = normalize_company_name(name)
    if not normalized:
        return []
    aliases = [normalized]
    for suffix in ("\u80a1\u4efd\u6709\u9650\u516c\u53f8", "\u6709\u9650\u8d23\u4efb\u516c\u53f8",
                  "\u96c6\u56e2\u80a1\u4efd\u6709\u9650\u516c\u53f8", "\u96c6\u56e2\u6709\u9650\u516c\u53f8"):
        if normalized.endswith(suffix):
            short = normalized[:-len(suffix)].strip()
            if short:
                aliases.append(short)
            break
    return aliases


def is_valid_enhanced_item(item: dict[str, object]) -> bool:
    title = str(item.get("title") or "").strip()
    value = str(item.get("value") or "").strip()
    evidence = str(item.get("evidence") or "").strip()
    item_type = str(item.get("type") or "").strip()

    if not title or not value:
        return False

    for marker in ("\u65e0\u5177\u4f53", "\u672a\u62ab\u9732", "\u672a\u8bf4\u660e", "\u65e0\u6cd5\u5224\u65ad",
                   "\u65e0\u6cd5\u786e\u5b9a", "\u672a\u68c0\u7d22\u5230", "\u6ca1\u6709\u63d0\u53ca", "\u4e0d\u5984"):
        if marker in value:
            return False

    if item_type in {"field", "fact", "table_fact", "table_summary"} and len(evidence) < 6:
        return False

    return True


def write_redacted_export(pages: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "page_number": page["page_number"],
            "logical_page": page["logical_page"],
            "page_type": page.get("page_type", "text"),
            "section_title": page.get("section_title", ""),
            "source": page.get("source", "builtin"),
            "source_pdf": page.get("source_pdf", ""),
            "source_pdf_path": page.get("source_pdf_path", ""),
            "text": page.get("redacted_text", page["text"]),
            "redaction_stats": page["redaction_stats"],
        }
        for page in pages
    ]
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_pdf_vlm_failure(
    pdf_path: Path, page_number: int, error: str, failed_pages: list[int]
) -> None:
    failure_path = settings.artifact_dir / "pdf_vlm_last_failure.json"
    failure_path.write_text(
        json.dumps(
            {
                "pdf": str(pdf_path),
                "page_number": page_number,
                "failed_pages": failed_pages,
                "error": error,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


_LOW_VALUE_KEYWORDS = frozenset([
    "\u4fdd\u4ecb\u673a\u6784(\u4e3b\u627f\u9500\u5546)\u58f0\u660e",
    "\u53d1\u884c\u4eba\u5168\u4f53\u8463\u4e8b\u3001\u76d1\u4e8b\u53ca\u9ad8\u7ea7\u7ba1\u7406\u4eba\u5458\u58f0\u660e",
    "\u62db\u80a1\u610f\u5411\u4e66",
    "\u5907\u67e5\u6587\u4ef6",
    "\u6709\u5173\u58f0\u660e",
])


def is_low_value_context(item: dict[str, object], intent: Any) -> bool:
    metadata = dict(item.get("metadata") or {})
    text = str(item.get("text") or "")
    normalized = re.sub(r"\s+", " ", text).strip()
    field_title = str(metadata.get("field_title") or "")
    section_title = str(metadata.get("section_title") or "")
    page_type = str(metadata.get("page_type") or "")

    if not normalized:
        return True

    qt = getattr(intent, "question_type", None)
    if field_title in {"\u516c\u53f8\u540d\u79f0", "\u672c\u516c\u53f8"} and qt not in {"field_lookup"}:
        return True

    for marker in ("\u4fdd\u4ecb\u673a\u6784(\u4e3b\u627f\u9500\u5546)\u58f0\u660e",
                   "\u53d1\u884c\u4eba\u5168\u4f53\u8463\u4e8b\u3001\u76d1\u4e8b\u53ca\u9ad8\u7ea7\u7ba1\u7406\u4eba\u5458\u58f0\u660e",
                   "\u62db\u80a1\u610f\u5411\u4e66", "\u5907\u67e5\u6587\u4ef6", "\u6709\u5173\u58f0\u660e"):
        if marker in normalized:
            return True

    if page_type in {"structured", "vlm_structured"} and any(
        m in normalized for m in ["\u5b57\u6bb5\uff1a\u516c\u53f8\u540d\u79f0", "\u5b57\u6bb5\uff1a\u672c\u516c\u53f8"]
    ):
        return True

    if len(normalized) < 18 and not any(c.isdigit() for c in normalized):
        return True

    payload = f"{field_title}\n{section_title}\n{normalized}"
    qtags = getattr(intent, "query_tags", []) or []
    rwq = getattr(intent, "rewritten_query", "") or ""

    if "fundraising" in qtags and not any(t in payload for t in ["\u52df\u96c6\u8d44\u91d1", "\u52df\u6295\u9879\u76ee", "\u8865\u5145\u6d41\u52a8\u8d44\u91d1", "\u9879\u76ee"]):
        return True
    if "related_party" in qtags and not any(t in payload for t in ["\u5173\u8054\u65b9", "\u5173\u8054\u5173\u7cfb", "\u6301\u80a1\u6bd4\u4f8b", "\u63a7\u80a1\u5468", "\u5b9e\u9645\u63a7\u5236\u4eba"]):
        return True
    if "military_revenue" in qtags and not any(t in payload for t in ["\u519b\u7528\u9886\u57df", "\u56fd\u9632\u5ba2\u6237", "\u519b\u65b9\u5e02\u573a", "\u9500\u552e\u989d", "\u6bd4\u91cd"]):
        return True
    if "technical_standard" in qtags and not any(t in payload for t in ["\u53c2\u4e0e\u5236\u5b9a", "\u6280\u672f\u6807\u51c6", "\u89c4\u8303", "\u6807\u51c6"]):
        return True
    if any(t in rwq for t in ["\u4e0a\u6e38", "\u4e0b\u6e38"]) and not any(t in payload for t in ["\u4e0a\u6e38", "\u4e0b\u6e38", "\u884c\u4e1a", "\u4f01\u4e1a", "\u5e94\u7528"]):
        return True
    if any(t in rwq for t in ["\u4e00\u7b49\u5956", "\u56fd\u5bb6\u79d1\u5b66\u8fdb\u6b65\u4e00\u7b49\u5956", "\u5de5\u7a0b"]) and not any(t in payload for t in ["\u56fd\u5bb6\u79d1\u5b66\u8fdb\u6b65\u4e00\u7b49\u5956", "\u5de5\u7a0b", "\u8363\u8001"]):
        return True
    if qt == "org_structure" and not any(t in payload for t in ["\u7ec4\u7ec7\u7ed3\u6784", "\u9500\u552e\u90e8", "\u9500\u552e\u5904", "\u6784\u6210", "\u4e0b\u8bbe"]):
        return True
    if qt == "chart_trend" and not any(t in payload for t in ["\u589e\u957f\u7387", "\u8d1f\u589e\u957f", "\u5e94\u7528\u7ed3\u6784", "\u56fe"]):
        return True

    return False


def prune_low_value_contexts(
    matches: list[dict[str, object]], intent: Any
) -> list[dict[str, object]]:
    return [m for m in matches if not is_low_value_context(m, intent)]


def main_manifest_path() -> Path:
    return settings.artifact_dir / "ingest_manifest.json"


def enhance_manifest_path() -> Path:
    return settings.artifact_dir / "enhance_manifest.json"


def parsed_cache_path() -> Path:
    return settings.artifact_dir / "parsed_pages.json"


def redacted_cache_path() -> Path:
    return settings.artifact_dir / "parsed_pages_redacted.json"


def default_pdf_paths() -> list[Path]:
    return [p for p in settings.pdf_paths if p.exists()]


def resolve_target_pdfs(target_company: str, company_map: dict[str, list[str]]) -> list[str]:
    if not target_company:
        return []
    for alias in get_company_aliases(target_company):
        if alias in company_map:
            return company_map[alias]
    for pdf, companies in company_map.items():
        for alias in get_company_aliases(target_company):
            if alias in companies:
                return [pdf]
    return []
