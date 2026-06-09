# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MOJIBAKE_HINT_RE = re.compile(r"[\u4e00-\u9fff]")


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


def resolve_artifact_dir(raw: str) -> Path:
    from backend.config import settings
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


@dataclass
class ThresholdConfig:
    min_keyword_overlap: float = 0.18
    min_focus_signal: float = 0.02
    min_context_limit: int = 6
    max_context_limit: int = 10
    table_top_k: int = 5
    visual_top_k: int = 3
    text_top_k: int = 5


@dataclass
class QualityIssue:
    issue_type: str
    severity: str
    description: str
    location: str = ""
    suggestion: str = ""


@dataclass
class CheckResult:
    passed: bool
    score: float
    issues: list[QualityIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
